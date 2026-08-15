###############################################################################
# Compute the common DMRG/GS data once and save it for independent site jobs.
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

options = parse_options(ARGS)
output_dir = get(options, "--output", "")
isempty(output_dir) && error("Required option: --output DIR")
seed = parse(Int, get(options, "--seed", "20260724"))
L = parse(Int, get(options, "--L", "0"))
L >= 3 || error("Required option: --L INTEGER >= 3")
threading_mode = lowercase(get(options, "--threading-mode", "outer"))
threading_mode in ("outer", "blocksparse") ||
    error("Unsupported --threading-mode $threading_mode")
i_imp = parse(Int, get(options, "--impurity-site", string(cld(L, 2))))
1 <= i_imp <= L || error("Require 1 <= impurity-site <= L")
mkpath(output_dir)

# Physical parameters: same model as Goncalo's working case.
J = 18.4
Jp = 16.7

# Numerical parameters: unchanged from Goncalo's working case.
linkdim = 100
cutoff = 1e-8
precE = 1e-5
precEmax = 1e-3
nexc = 2
w = 1000
T = 4
tol_sym = 1e-2
ωmax = 100.0
try_doubleN = true
ω_smearing = 2.0

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
write(fw, "# Random seed:\n$seed\n\n")

# Sites and common operators.
sites = siteinds("S=1", L; conserve_sz=true)
sites[i_imp] = siteind("S=1/2"; conserve_sz=true)
H = Hamiltonians.H_Heis_impurity_1D_OBC(J, Jp, Jp, i_imp, sites)
S2 = Operators.S2_op(sites)
Sz = Operators.Sz_op(sites)

# Initial state and DMRG, kept identical to the working case.
statei = AuxFunctions.spinstate_sNSz(1, L - 1, 0)
insert!(statei, i_imp, "Up")
ψi = randomMPS(sites, statei, linkdim)

E0, ψ0 = DMRGsweeps.DMRGmaxdim_convE(H, ψi, precE; cutoff=cutoff)
println()
En = [E0]
ψn = [ψ0]
for n in 1:nexc
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
Sn = (-1 .+ sqrt.(1 .+ 4 .* S2n)) ./ 2
Szn = [inner(ψn[n]', Sz, ψn[n]) for n in eachindex(En)]
write(fw, "# List of S:\n$(Sn)\n\n")
write(fw, "# List of Sz:\n$(Szn)\n\n")

kB = 8.617333262e-2
T_meV = kB * T
nGS = count(E -> E - En[1] <= T_meV / 5, En)

write(fw, "========== Symmetrization (GS manifold) ==========\n\n")
ψn_GS = ψn[1:nGS]
S2n_sym, ψn_sym = AuxFunctions.symmetrize_eigenstates(ψn_GS, S2)
En_sym = [inner(ψn_sym[n]', H, ψn_sym[n]) for n in eachindex(ψn_sym)]
idx = sortperm(En_sym)
En_sym = En_sym[idx]
ψn_sym = ψn_sym[idx]
S2n_sym = S2n_sym[idx]
Sn_sym = (-1 .+ sqrt.(1 .+ 4 .* S2n_sym)) ./ 2
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
    L,
    i_imp,
    J,
    Jp,
    linkdim,
    cutoff,
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

println("GS checkpoint written to $checkpoint_filename")
