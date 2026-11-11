# Read-only symmetry and saved-moment checks; does not rerun a spectrum.
using ITensors, ITensorMPS, JLD2, LinearAlgebra
include("../remote/engine/GenericSpinSites.jl")

checkpoint = load(ARGS[1])
sites = checkpoint["sites"]
L = length(sites)
all(j -> dim(sites[j]) == dim(sites[L+1-j]), 1:L) ||
    error("Reflection comparison requires a mirror-symmetric spin pattern")
for (index, psi) in enumerate(checkpoint["ψn_sym"])
    reflected = MPS([replaceind(psi[j], sites[j]=>sites[L+1-j]) for j in L:-1:1])
    overlap = abs(inner(psi, reflected)) / sqrt(real(inner(psi, psi)*inner(reflected, reflected)))
    sz = expect(psi, "Sz")
    println("state=$index reflection_overlap=$overlap max_Sz_mirror_difference=$(maximum(abs.(sz-reverse(sz))))")
end
for (site, filename) in zip((1,L), ARGS[2:end])
    progress = load(filename)
    os = OpSum()
    os += 1.0, "Sz", site
    B = MPO(os, sites)
    direct = inner(checkpoint["ψn_sym"][1]', B, progress["tnm1"])
    moments = progress["moments"]
    println("site=$site mu0=$(moments[1]) max_abs_moment_over_mu0=$(maximum(abs.(moments))/moments[1])")
    println("site=$site order=$(progress["completed_n"]-1) direct_moment=$direct reconstructed_moment=$(moments[progress["completed_n"]])")
end
