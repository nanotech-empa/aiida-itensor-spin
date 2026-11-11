using LinearAlgebra
using ITensors
using ITensorMPS

include(joinpath(@__DIR__, "..", "remote", "engine", "GenericSpinSites.jl"))
using .GenericSpinSites

for two_s in (1, 2, 3, 4)
    matrices = spin_matrices(two_s)
    @assert size(matrices.Sz) == (two_s + 1, two_s + 1)
    @assert isapprox(
        matrices.Sp * matrices.Sm - matrices.Sm * matrices.Sp,
        2 * matrices.Sz;
        atol=1e-13,
        rtol=0,
    )
    site = siteind("GenericSpin"; two_s=two_s, conserve_sz=true)
    @assert dim(site) == two_s + 1
    @assert size(op("Sz", site)) == (two_s + 1, two_s + 1)
    @assert size(op("S+", site)) == (two_s + 1, two_s + 1)
    @assert size(op("S-", site)) == (two_s + 1, two_s + 1)
end

state = product_state_for_twice_sz([1, 2, 4], 1)
sites = [
    siteind("GenericSpin"; two_s=two_s, conserve_sz=true)
    for two_s in (1, 2, 4)
]
psi = randomMPS(sites, state, 4)
@assert length(psi) == 3
println("generic spin site validation passed; product state = $(state)")
