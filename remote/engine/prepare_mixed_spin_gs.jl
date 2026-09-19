###############################################################################
# Ground-state preparation for a nearest-neighbour mixed-spin Heisenberg chain.
#
# H = sum_i J_i S_i . S_(i+1)
###############################################################################
using Dates
using LinearAlgebra
using Random
using ITensors
using ITensorMPS
using JLD2

const CASE_ROOT = @__DIR__
include(joinpath(CASE_ROOT, "ITensor_libs", "Imports.jl"))

function parse_options(args)
    options = Dict{String,String}()
    index = 1
    while index <= length(args)
        flag = args[index]
        startswith(flag, "--") || error("Unexpected positional argument: $flag")
        index == length(args) && error("Missing value for $flag")
        options[flag] = args[index + 1]
        index += 2
    end
    return options
end

function parse_bool(value, name)
    normalized = lowercase(value)
    normalized in ("true", "false") || error("$name must be true or false")
    return normalized == "true"
end

function parse_csv(::Type{T}, value, name) where {T}
    fields = split(value, ",")
    isempty(fields) && error("$name cannot be empty")
    try
        return parse.(T, fields)
    catch
        error("$name must be a comma-separated list of $(T)")
    end
end

function mixed_heisenberg_hamiltonian(couplings, sites)
    length(couplings) == length(sites) - 1 ||
        error("Require exactly one coupling per nearest-neighbour bond")
    ampo = OpSum()
    for (site, coupling) in enumerate(couplings)
        ampo += coupling, "Sz", site, "Sz", site + 1
        ampo += coupling / 2, "S+", site, "S-", site + 1
        ampo += coupling / 2, "S-", site, "S+", site + 1
    end
    return MPO(ampo, sites)
end

options = parse_options(ARGS)
output_dir = get(options, "--output", "")
isempty(output_dir) && error("Required option: --output DIR")
mkpath(output_dir)

model = get(options, "--model", "")
model == "mixed_spin_heisenberg_chain_v1" || error("Unsupported model: $model")
seed = parse(Int, get(options, "--seed", "20260724"))
two_spins = parse_csv(Int, options["--two-spins"], "--two-spins")
all(two_s -> two_s >= 1, two_spins) || error("--two-spins values must be positive")
L = length(two_spins)
L >= 3 || error("Require at least three sites")
site_spins = two_spins ./ 2
couplings_meV = parse_csv(Float64, options["--couplings-meV"], "--couplings-meV")
length(couplings_meV) == L - 1 ||
    error("--couplings-meV must contain exactly L-1 entries")
target_twice_sz = parse(Int, options["--target-twice-sz"])
abs(target_twice_sz) <= sum(two_spins) ||
    error("--target-twice-sz lies outside the physical range")
iszero(mod(target_twice_sz - sum(two_spins), 2)) ||
    error("--target-twice-sz has incompatible parity")

threading_mode = lowercase(get(options, "--threading-mode", "outer"))
threading_mode in ("outer", "blocksparse") ||
    error("Unsupported --threading-mode $threading_mode")
cutoff = parse(Float64, options["--dmrg-cutoff"])
spectral_cutoff = parse(Float64, options["--spectral-cutoff"])
precE = parse(Float64, options["--energy-precision-meV"])
precEmax = parse(Float64, options["--highest-energy-precision-meV"])
linkdim = parse(Int, options["--initial-linkdim"])
nexc = parse(Int, options["--excited-states"])
w = parse(Float64, options["--excited-state-penalty-meV"])
T = parse(Float64, options["--temperature-K"])
tol_sym = parse(Float64, options["--symmetry-tolerance"])
ωmax = parse(Float64, options["--omega-max-meV"])
ω_smearing = parse(Float64, options["--omega-smearing-meV"])
try_doubleN = parse_bool(options["--try-doubleN"], "--try-doubleN")

start_time = DateTime(now())
start_wall = time()
BLAS.set_num_threads(1)
if threading_mode == "blocksparse"
    ITensors.Strided.disable_threads()
    ITensors.enable_threaded_blocksparse()
end
Random.seed!(seed)

summary_filename = joinpath(output_dir, "gs_summary.txt")
fw = open(summary_filename, "w")
write(fw, "# Model and physical parameters:\n")
write(fw, "model = $model\n")
write(fw, "L = $L\nsite spins = $site_spins\n")
write(fw, "couplings = $couplings_meV meV\n")
write(fw, "target twice Sz = $target_twice_sz\n")
write(fw, "temperature = $T K\nomega max = $ωmax meV\n")
write(fw, "omega smearing = $ω_smearing meV\n\n")
write(fw, "# Numerical parameters:\n")
write(fw, "DMRG cutoff = $cutoff\nChebyshev cutoff = $spectral_cutoff\n")
write(fw, "excited states = $nexc\n\n")
write(fw, "# Random seed:\n$seed\n\n")

sites = [
    siteind("GenericSpin"; two_s=two_s, conserve_sz=true)
    for two_s in two_spins
]
H = mixed_heisenberg_hamiltonian(couplings_meV, sites)
S2 = Operators.S2_op(sites)
Sz = Operators.Sz_op(sites)

statei = product_state_for_twice_sz(two_spins, target_twice_sz)
ψi = randomMPS(sites, statei, linkdim)

E0, ψ0 = DMRGsweeps.DMRGmaxdim_convE(H, ψi, precE; cutoff=cutoff)
println()
En = [E0]
ψn = [ψ0]
for _ in 1:nexc
    Eaux, ψaux = DMRGsweeps.DMRGmaxdim_convE(
        H, ψi, precE; ψn=ψn, w=w, cutoff=cutoff
    )
    println()
    push!(En, Eaux)
    push!(ψn, ψaux)
end
idx = sortperm(En)
En = En[idx]
ψn = ψn[idx]
write(fw, "# List of energies:\n$(En)\n\n")

E0_mH, _ = DMRGsweeps.DMRGmaxdim_convE(-H, ψi, precEmax; cutoff=cutoff)
Emax = -E0_mH
println()
write(fw, "# Energy of highest-energy state:\n$Emax\n\n")

S2n = [inner(ψn[n]', S2, ψn[n]) for n in eachindex(En)]
Sn = (-1 .+ sqrt.(max.(0, 1 .+ 4 .* S2n))) ./ 2
Szn = [inner(ψn[n]', Sz, ψn[n]) for n in eachindex(En)]
write(fw, "# List of S:\n$(Sn)\n\n")
write(fw, "# List of Sz:\n$(Szn)\n\n")

kB = 8.617333262e-2
T_meV = kB * T
nGS = count(E -> E - En[1] <= T_meV / 5, En)
nGS >= 1 || error("Ground-state manifold is empty")

write(fw, "========== Symmetrization (GS manifold) ==========\n\n")
ψn_GS = ψn[1:nGS]
S2n_sym, ψn_sym = AuxFunctions.symmetrize_eigenstates(ψn_GS, S2)
En_sym = [inner(ψn_sym[n]', H, ψn_sym[n]) for n in eachindex(ψn_sym)]
idx = sortperm(En_sym)
En_sym = En_sym[idx]
ψn_sym = ψn_sym[idx]
S2n_sym = S2n_sym[idx]
Sn_sym = (-1 .+ sqrt.(max.(0, 1 .+ 4 .* S2n_sym))) ./ 2
Szn_sym = [inner(ψn_sym[n]', Sz, ψn_sym[n]) for n in eachindex(ψn_sym)]

write(fw, "# List of energies after symmetrization:\n$(En_sym)\n\n")
write(fw, "# List of S after symmetrization:\n$(Sn_sym)\n\n")
write(fw, "# List of Sz after symmetrization:\n$(Szn_sym)\n\n")

Sn_sym_round = round.(2 .* Sn_sym) ./ 2
Szn_sym_round = round.(2 .* Szn_sym) ./ 2
@assert all(isapprox.(Sn_sym, Sn_sym_round; atol=tol_sym, rtol=0))
@assert all(isapprox.(Szn_sym, Szn_sym_round; atol=tol_sym, rtol=0))

prep_seconds = time() - start_wall
julia_version = string(VERSION)
itensors_version = string(pkgversion(ITensors))
itensormps_version = string(pkgversion(ITensorMPS))
checkpoint_filename = joinpath(output_dir, "gs_checkpoint.jld2")
jldsave(
    checkpoint_filename;
    model,
    L,
    site_spins,
    two_spins,
    couplings_meV,
    target_twice_sz,
    linkdim,
    cutoff,
    spectral_cutoff,
    precE,
    precEmax,
    nexc,
    w,
    T,
    tol_sym,
    ωmax,
    try_doubleN,
    ω_smearing,
    seed,
    threading_mode,
    sites,
    H,
    Emax,
    En,
    Sn,
    Szn,
    nGS,
    En_sym,
    ψn_sym,
    Sn_sym,
    Szn_sym,
    prep_seconds,
    julia_version,
    itensors_version,
    itensormps_version,
)

write(fw, "========== GS checkpoint ==========\n\n")
write(fw, "threading mode = $threading_mode\n")
write(fw, "Julia threads = $(Threads.nthreads())\n")
write(fw, "checkpoint = $checkpoint_filename\n")
write(fw, "preparation seconds = $prep_seconds\n")
write(fw, "Julia = $julia_version\n")
write(fw, "ITensors = $itensors_version\n")
write(fw, "ITensorMPS = $itensormps_version\n\n")
write(fw, "----------\n")
write(
    fw,
    "total time = " *
    string(
        Dates.canonicalize(
            Dates.CompoundPeriod(Dates.DateTime(now()) - Dates.DateTime(start_time))
        )
    ) *
    "\n",
)
close(fw)

dimensions_filename = joinpath(output_dir, "gs_dimensions.txt")
open(dimensions_filename, "w") do dimensions
    println(dimensions, "$L\t$nGS\t$(3 * L * nGS)")
end

ground_energy = minimum(En_sym)
bandwidth = Emax - ground_energy
energy_window_filename = joinpath(output_dir, "gs_energy_window.txt")
open(energy_window_filename, "w") do energy_window
    println(energy_window, "$ground_energy\t$Emax\t$bandwidth")
end

println("GS checkpoint written to $checkpoint_filename")
