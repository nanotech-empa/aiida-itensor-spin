# Isolated, full-space DMRG for compiled Hermitian local-product Hamiltonians.
# No AiiDA, scheduler, fixed-Sz restriction, or total-spin symmetrization.
using ITensors, ITensorMPS, LinearAlgebra, Random, TOML, JLD2
include(joinpath(@__DIR__, "GenericSpinSites.jl"))
using .GenericSpinSites

const USER_OPERATORS = Dict{String,Matrix{ComplexF64}}()

function ITensors.space(::SiteType"GraphPreview"; dimension)
    return dimension
end

function ITensors.op(::OpName{Name}, ::SiteType"GraphPreview", site::Index) where {Name}
    name = String(Name)
    spin = GenericSpinSites.spin_matrices(dim(site)-1)
    matrix = if name == "Id"
        spin.Id
    elseif name == "Sz"
        spin.Sz
    elseif name == "S+"
        spin.Sp
    elseif name == "S-"
        spin.Sm
    elseif name == "Sx"
        (spin.Sp + spin.Sm)/2
    elseif name == "Sy"
        (spin.Sp - spin.Sm)/(2im)
    else
        USER_OPERATORS[name]
    end
    size(matrix) == (dim(site),dim(site)) || error("Local operator dimension mismatch")
    return itensor(matrix, prime(site), dag(site))
end

function write_toml(path, value)
    open(path*".tmp", "w") do io
        TOML.print(io, value)
    end
    mv(path*".tmp", path; force=true)
end

mutable struct PreviewObserver <: AbstractObserver
    rows::Vector{Dict{String,Any}}
    maximum_error::Float64
    started::Float64
    deadline::Float64
    progress_path::String
    offset::Float64
end

function ITensorMPS.measure!(observer::PreviewObserver; kwargs...)
    observer.maximum_error = max(observer.maximum_error, kwargs[:spec].truncerr)
    time() < observer.deadline || error("DMRG preview time budget reached")
end

function ITensorMPS.checkdone!(observer::PreviewObserver; energy, psi, sweep, kwargs...)
    energy = real(energy) + observer.offset
    delta = isempty(observer.rows) ? 0.0 : abs(energy-observer.rows[end]["energy_meV"])
    row = Dict{String,Any}("sweep"=>sweep, "energy_meV"=>energy,
        "delta_energy_meV"=>delta, "max_bond_dimension"=>maxlinkdim(psi),
        "maximum_truncation_error"=>observer.maximum_error, "elapsed_seconds"=>time()-observer.started)
    push!(observer.rows,row)
    write_toml(observer.progress_path, Dict("sweeps"=>observer.rows))
    observer.maximum_error = 0.0
    return false
end

function compiled_mpo(data, sites, positions)
    ops = OpSum()
    for term in get(data,"terms",[])
        entries = Any[term["coefficient"]]
        for (site,name) in zip(term["sites"],term["names"])
            push!(entries,name,positions[site])
        end
        add!(ops, entries...)
    end
    return isempty(get(data,"terms",[])) ? 0.0*MPO(sites,"Id") : MPO(ComplexF64,ops,sites)
end

function run_preview(input_path, output_dir)
    started = time()
    data = TOML.parsefile(input_path)
    settings = data["settings"]
    BLAS.set_num_threads(1)
    order, two_spins = data["order"], data["two_spins"]
    sort(order) == collect(1:length(two_spins)) || error("Invalid original-site permutation")
    sites = [siteind("GraphPreview"; dimension=two_spins[label]+1) for label in order]
    positions = Dict(label=>position for (position,label) in enumerate(order))
    for entry in get(data,"operators",[])
        real_part = reduce(vcat,permutedims.(entry["real"]))
        imag_part = reduce(vcat,permutedims.(entry["imag"]))
        matrix = complex.(real_part,imag_part)
        ishermitian(matrix) || error("Compiled local basis operator is not Hermitian")
        USER_OPERATORS[entry["name"]] = matrix
    end
    H = compiled_mpo(data,sites,positions)
    Random.seed!(settings["seed"])
    psi0 = random_mps(ComplexF64,sites; linkdims=settings["initial_bond"])
    observer = PreviewObserver(Dict{String,Any}[],0.0,started,started+settings["timeout_seconds"],
        joinpath(output_dir,"progress.toml"),data["offset_meV"])
    constant_hamiltonian = isempty(get(data,"terms",[]))
    if constant_hamiltonian
        psi = psi0
    else
        _, psi = dmrg(H,psi0; nsweeps=settings["sweeps"],maxdim=settings["maxdim_schedule"],
            cutoff=settings["cutoff"],noise=[1e-6,1e-7,1e-8,0.0],
            eigsolve_krylovdim=settings["krylov_dimension"],observer,outputlevel=1)
    end
    normalize!(psi)
    energy = real(inner(psi',H,psi))
    h2 = real(inner(H,psi,H,psi))
    variance = h2-energy^2
    roundoff = 1e-12*max(1.0,abs(h2),energy^2)
    residual = sqrt(max(0,variance))
    energy_stable = length(observer.rows)>=3 && all(
        row["delta_energy_meV"]<=settings["energy_tolerance_meV"] for row in observer.rows[end-1:end])
    residual_ok = variance>=-roundoff && residual<=settings["residual_tolerance_meV"]
    final_error = isempty(observer.rows) ? 0.0 : observer.rows[end]["maximum_truncation_error"]
    cap_hit = maxlinkdim(psi)>=settings["max_bond"]
    diagnostics_passed = (energy_stable || constant_hamiltonian) && residual_ok && final_error<=settings["cutoff"]*10
    magnetization = Dict{String,Any}[]
    mx,my,mz = expect(psi,["Sx","Sy","Sz"])
    for label in 1:length(order)
        position = positions[label]
        push!(magnetization,Dict("site"=>label,"Sx"=>real(mx[position]),"Sy"=>real(my[position]),"Sz"=>real(mz[position])))
    end
    correlations = Dict{String,Any}[]
    for endpoints in get(data,"edges",[])
        i,j = endpoints
        a,b = positions[i],positions[j]
        ops = OpSum()
        ops += 1.0,"Sz",a,"Sz",b
        ops += 0.5,"S+",a,"S-",b
        ops += 0.5,"S-",a,"S+",b
        value = real(inner(psi',MPO(ComplexF64,ops,sites),psi))
        push!(correlations,Dict("i"=>i,"j"=>j,"dot"=>value))
    end
    entropies = Float64[]
    work = copy(psi)
    for bond in 1:length(order)-1
        orthogonalize!(work,bond)
        left = bond==1 ? (siteind(work,bond),) : (linkind(work,bond-1),siteind(work,bond))
        _,S,_ = svd(work[bond],left)
        probabilities = abs2.(diag(S))
        push!(entropies,-sum(p*log(p) for p in probabilities if p>0))
    end
    jldsave(joinpath(output_dir,"state.jld2"); psi, order, input=data)
    result = Dict{String,Any}(
        "solver"=>"DMRG", "energy_meV"=>energy+data["offset_meV"],
        "energy_variance_meV2"=>variance,"residual_meV"=>residual,
        "variance_roundoff_estimate_meV2"=>roundoff,"energy_stable"=>energy_stable,
        "residual_ok"=>residual_ok,"diagnostics_passed"=>diagnostics_passed,
        "bond_cap_hit"=>cap_hit,"max_bond_dimension"=>maxlinkdim(psi),
        "mpo_bond_dimension"=>maxlinkdim(H),"constant_hamiltonian"=>constant_hamiltonian,
        "elapsed_seconds"=>time()-started,"order"=>order,"settings"=>settings,
        "entanglement_entropy_nats"=>entropies,"magnetization"=>magnetization,
        "correlations"=>correlations,"sweeps"=>observer.rows,
        "julia_version"=>string(VERSION),"itensors_version"=>string(pkgversion(ITensors)),
        "itensormps_version"=>string(pkgversion(ITensorMPS)),"blas_threads"=>BLAS.get_num_threads(),
        "julia_threads"=>Threads.nthreads(),"conserve_sz"=>false)
    write_toml(joinpath(output_dir,"result.toml"),result)
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==2 || error("Usage: general_graph_dmrg.jl INPUT.toml OUTPUT_DIR")
    run_preview(ARGS[1],ARGS[2])
end
