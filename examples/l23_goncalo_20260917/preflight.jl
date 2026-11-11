using ITensors, ITensorMPS, JLD2, TOML, LinearAlgebra, Random
root = @__DIR__
for name in ("prepare_gs.jl", "chi_single_task_checkpoint.jl", "SpectralFunctionCheckpointed.jl", "collect_tasks.jl")
    parsed = Meta.parseall(read(joinpath(root, "engine", name), String))
    function check_parse(x)
        x isa Expr || return
        x.head in (:error, :incomplete) && error("Parse error in $name: $x")
        foreach(check_parse, x.args)
    end
    check_parse(parsed)
end
Meta.parseall(read(joinpath(root, "compare_reference.jl"), String))
reference = load(joinpath(root, "reference", "chi_L23.jld2"), "χ_n_i")
@assert length(reference) == 1 && length(reference[1]) == 23
@assert all(length(spectrum) == 333 && all(isfinite, spectrum) for spectrum in reference[1])
physics = TOML.parsefile(joinpath(root, "physics.toml"))
@assert physics["JR_meV"] == 51.1 && physics["spectral_cutoff"] == 1e-6
include(joinpath(root, "engine", "ITensor_libs", "Imports.jl"))
include(joinpath(root, "engine", "SpectralFunctionCheckpointed.jl"))
module Golden
    include(joinpath(@__DIR__, "engine", "ITensor_libs", "Operators.jl"))
    include(joinpath(@__DIR__, "reference", "SpectralFunction.jl"))
end
BLAS.set_num_threads(1)
Random.seed!(23)
sites = siteinds("S=1/2", 3; conserve_sz=true)
psi = randomMPS(sites, ["Up", "Dn", "Up"], 4)
H = Hamiltonians.H_Heis_impurity_1D_OBC(1.0, 1.0, 1.0, 2, sites)
for channel in ("zz", "pm", "mp")
    B, C = if channel == "zz"
        Operators.Szi_op(2, sites), Operators.Szi_op(2, sites)
    elseif channel == "pm"
        Operators.Spi_op(2, sites), Operators.Smi_op(2, sites)
    else
        Operators.Smi_op(2, sites), Operators.Spi_op(2, sites)
    end
    args = (psi, H, B, C, -2.0, 2.0, sites)
    kwargs = (; cutoff=1e-6, Emax=4.0, N=14, Nω=31, try_doubleN=true)
    golden = Golden.SpectralFunction.spectral_function(args...; kwargs...)
    ours = SpectralFunctionCheckpointed.spectral_function(args...; kwargs...)
    difference = maximum(abs.(ours .- golden))
    println("PREFLIGHT channel=$channel max_abs_difference=$difference")
    @assert isapprox(ours, golden; atol=1e-12, rtol=1e-12)
end
println("PREFLIGHT PASSED: reference dimensions, syntax and three-channel algorithm equivalence")
