module GenericSpinSites

using ITensors
using ITensorMPS
using LinearAlgebra

export product_state_for_twice_sz, spin_matrices

function _twice_spin_from_index(site::Index)
    dimension = dim(site)
    dimension >= 2 || error("a spin site must have dimension >= 2")
    return dimension - 1
end

function spin_matrices(two_s::Int)
    two_s >= 1 || error("two_s must be a positive integer")
    spin = two_s / 2
    dimension = two_s + 1
    m_values = [spin - (index - 1) for index in 1:dimension]
    sz = Matrix(Diagonal(m_values))
    raising = zeros(Float64, dimension, dimension)
    for column in 2:dimension
        m_value = m_values[column]
        raising[column - 1, column] =
            sqrt(spin * (spin + 1) - m_value * (m_value + 1))
    end
    lowering = transpose(raising)
    return (Id=Matrix{Float64}(I, dimension, dimension), Sz=sz, Sp=raising, Sm=lowering)
end

function ITensors.space(
    ::SiteType"GenericSpin";
    two_s::Int,
    conserve_qns=false,
    conserve_sz=conserve_qns,
    qnname_sz="Sz",
)
    two_s >= 1 || error("two_s must be a positive integer")
    if conserve_sz
        return [QN(qnname_sz, twice_m) => 1 for twice_m in two_s:-2:-two_s]
    end
    return two_s + 1
end

function ITensors.state(::StateName{N}, ::SiteType"GenericSpin", site::Index) where {N}
    basis_index = parse(Int, String(N))
    0 <= basis_index < dim(site) ||
        error("basis state $basis_index outside 0:$(dim(site) - 1)")
    values = zeros(Float64, dim(site))
    values[basis_index + 1] = 1.0
    return itensor(values, site)
end

function _operator_tensor(matrix, site)
    return itensor(matrix, prime(site), dag(site))
end

function ITensors.op(::OpName"Id", ::SiteType"GenericSpin", site::Index)
    return _operator_tensor(spin_matrices(_twice_spin_from_index(site)).Id, site)
end

function ITensors.op(::OpName"Sz", ::SiteType"GenericSpin", site::Index)
    return _operator_tensor(spin_matrices(_twice_spin_from_index(site)).Sz, site)
end

function ITensors.op(::OpName"S+", ::SiteType"GenericSpin", site::Index)
    return _operator_tensor(spin_matrices(_twice_spin_from_index(site)).Sp, site)
end

function ITensors.op(::OpName"S-", ::SiteType"GenericSpin", site::Index)
    return _operator_tensor(spin_matrices(_twice_spin_from_index(site)).Sm, site)
end

function product_state_for_twice_sz(two_spins::Vector{Int}, target_twice_sz::Int)
    isempty(two_spins) && error("at least one spin is required")
    all(two_s -> two_s >= 1, two_spins) || error("all twice-spin values must be positive")

    paths = Dict(0 => Int[])
    for two_s in two_spins
        next_paths = Dict{Int,Vector{Int}}()
        for (partial_sum, path) in paths
            for (basis_index, twice_m) in enumerate(two_s:-2:-two_s)
                candidate = partial_sum + twice_m
                if !haskey(next_paths, candidate)
                    next_paths[candidate] = [path; basis_index - 1]
                end
            end
        end
        paths = next_paths
    end
    haskey(paths, target_twice_sz) ||
        error("no product state exists for twice-Sz=$target_twice_sz")
    return string.(paths[target_twice_sz])
end

end
