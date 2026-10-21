module SpinGraphHamiltonian
using ITensors, ITensorMPS
export graph_heisenberg_hamiltonian, parse_exchange_edges

function parse_exchange_edges(value, nsites)
    edges = Tuple{Int,Int,Float64}[]
    seen = Set{Tuple{Int,Int}}()
    for entry in split(value, ';')
        fields = split(entry, ',')
        length(fields) == 3 || error("Each exchange edge requires i,j,J_meV")
        i, j = parse.(Int, fields[1:2])
        J = parse(Float64, fields[3])
        1 <= i <= nsites && 1 <= j <= nsites && i != j || error("Invalid exchange edge endpoints")
        isfinite(J) || error("Exchange coupling must be finite")
        key = minmax(i, j)
        key in seen && error("Duplicate undirected exchange edge")
        push!(seen, key)
        push!(edges, (key[1], key[2], J))
    end
    isempty(edges) && error("Provide at least one exchange edge")
    return edges
end

function graph_heisenberg_hamiltonian(edges, sites)
    # OpSum produces the MPO in the explicitly recorded site order, independently
    # of drawing coordinates. Isotropic exchange preserves total Sz and S².
    ampo = OpSum()
    for (i, j, J) in edges
        1 <= i <= length(sites) && 1 <= j <= length(sites) && i != j || error("Invalid exchange edge")
        isfinite(J) || error("Exchange coupling must be finite")
        ampo += J, "Sz", i, "Sz", j
        ampo += J / 2, "S+", i, "S-", j
        ampo += J / 2, "S-", i, "S+", j
    end
    return MPO(ampo, sites)
end
end
