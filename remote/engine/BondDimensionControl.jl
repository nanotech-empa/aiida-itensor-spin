module BondDimensionControl

using ITensors, ITensorMPS

export compression_report, controlled_apply, controlled_add, compress_resume!

"""Per-task measured discarded weights, not a final-spectrum error estimate."""
function compression_report(maxdim::Int, cutoff::Float64)
    maxdim >= 0 || error("max_bond_dimension must be nonnegative (0 means unlimited)")
    return Dict{String,Any}(
        "max_bond_dimension" => maxdim,
        "requested_cutoff" => cutoff,
        "measurement" => maxdim > 0 ? "density-matrix-discarded-weight" : "not_measured",
        "measured_bonds" => 0,
        "cap_reached_bonds" => 0,
        "cutoff_exceeded_bonds" => 0,
        "max_discarded_weight" => 0.0,
        "max_observed_bond_dimension" => 0,
        "worst_event" => nothing,
        "policy_history" => Any[],
    )
end

function record!(report, spec, dimension, iteration, operation, bond)
    discarded = Float64(truncerror(spec))
    report["measured_bonds"] += 1
    report["max_observed_bond_dimension"] = max(report["max_observed_bond_dimension"], dimension)
    reached = dimension >= report["max_bond_dimension"]
    reached && (report["cap_reached_bonds"] += 1)
    exceeded = discarded > report["requested_cutoff"]
    exceeded && (report["cutoff_exceeded_bonds"] += 1)
    if discarded > report["max_discarded_weight"]
        report["max_discarded_weight"] = discarded
        report["worst_event"] = Dict(
            "iteration" => iteration, "operation" => operation, "bond" => bond,
            "bond_dimension" => dimension, "discarded_weight" => discarded,
        )
    end
    if reached && report["cap_reached_bonds"] == 1
        println("WARNING: max bond dimension reached at iteration=$iteration operation=$operation bond=$bond maxdim=$(report["max_bond_dimension"]) discarded_weight=$discarded requested_cutoff=$(report["requested_cutoff"])")
        flush(stdout)
    end
end

# Instrumented forms of ITensorMPS 0.4.1 density-matrix apply and addition.
# The contraction/diagonalization order is preserved; only the supplied maxdim
# and observation of each returned Spectrum are added. Adapted from ITensorMPS:
# https://github.com/ITensor/ITensorMPS.jl/tree/v0.4.1/src (Apache-2.0).
# See THIRD_PARTY_LICENSE_ITensorMPS.txt. No upstream methods are monkeypatched.
function controlled_apply(A::MPO, psi::MPS, report, iteration, operation)
    cutoff = report["requested_cutoff"]
    maxdim = report["max_bond_dimension"]
    maxdim == 0 && return apply(A, psi; cutoff=cutoff)
    n = length(A)
    n == length(psi) || error("MPO/MPS lengths differ")
    n == 1 && return replaceprime(MPS([A[1] * psi[1]]), 1 => 0)
    result = similar(psi)
    A = sim(linkinds, A)
    psi_c, A_c = dag(psi), dag(A)
    sim!(linkinds, A_c)
    sim!(linkinds, psi_c)
    sim!(siteinds, commoninds, A_c, psi_c)
    simA_c = sim(siteinds, uniqueinds, A_c, psi_c)
    E = Vector{ITensor}(undef, n - 1)
    E[1] = psi[1] * A[1] * A_c[1] * psi_c[1]
    for j in 2:(n - 1)
        E[j] = E[j - 1] * psi[j] * A[j] * A_c[j] * psi_c[j]
    end
    R = psi[n] * A[n]
    simR_c = psi_c[n] * simA_c[n]
    rho = E[n - 1] * R * simR_c
    l = linkind(psi, n - 1)
    ts = isnothing(l) ? "" : tags(l)
    Lis = siteinds(uniqueinds, A, psi, n)
    Ris = siteinds(uniqueinds, simA_c, psi_c, n)
    F = eigen(rho, Lis, Ris; ishermitian=true, tags=ts, cutoff, maxdim, mindim=1)
    U, Ut = F.V, F.Vt
    left, right = F.l, F.r
    record!(report, F.spec, dim(right), iteration, operation, n - 1)
    result[n] = Ut
    R = R * dag(Ut) * psi[n - 1] * A[n - 1]
    simR_c = simR_c * U * psi_c[n - 1] * simA_c[n - 1]
    for j in reverse(2:(n - 1))
        requested = min(dim(commoninds(psi[j], E[j - 1])) * dim(commoninds(A[j], E[j - 1])), maxdim)
        s = siteinds(uniqueinds, A, psi, j)
        sc = siteinds(uniqueinds, simA_c, psi_c, j)
        rho = E[j - 1] * R * simR_c
        l = linkind(psi, j - 1)
        ts = isnothing(l) ? "" : tags(l)
        F = eigen(rho, (s..., left), (sc..., right); ishermitian=true, tags=ts,
            cutoff, maxdim=requested, mindim=1)
        U, Ut = F.V, F.Vt
        left, right = F.l, F.r
        record!(report, F.spec, dim(right), iteration, operation, j - 1)
        result[j] = Ut
        R = R * dag(Ut) * psi[j - 1] * A[j - 1]
        simR_c = simR_c * U * psi_c[j - 1] * simA_c[j - 1]
    end
    result[1] = R
    ITensorMPS.setleftlim!(result, 0)
    ITensorMPS.setrightlim!(result, 2)
    # contract returns physical prime level 1; apply returns level 0.
    return replaceprime(result, 1 => 0)
end

function controlled_add(a::MPS, b::MPS, report, iteration, operation)
    cutoff = report["requested_cutoff"]
    maxdim = report["max_bond_dimension"]
    maxdim == 0 && return add(a, b; cutoff=cutoff)
    n = length(a)
    n == length(b) || error("MPS lengths differ")
    # Spin-chain MPS have exactly one physical index at each site.
    all(j -> length(siteinds(a, j)) == 1 && siteind(a, j) == siteind(b, j), 1:n) ||
        error("controlled_add requires matching single physical spin indices")
    n == 1 && return MPS([a[1] + b[1]])
    inputs = [orthogonalize(copy(a), n), orthogonalize(copy(b), n)]
    s = siteinds(inputs[1])
    C = last.(inputs)
    rho = sum([prime(p[n], s[n]) * dag(p[n]) for p in inputs])
    result = MPS(n)
    for j in reverse(2:n)
        D, V, spec = eigen(rho; ishermitian=true, tags=tags(linkind(inputs[1], j - 1)),
            cutoff, maxdim=min(maxdim, sum(linkdim(p, j - 1) for p in inputs)))
        link = commonind(D, V)
        record!(report, spec, dim(link), iteration, operation, j - 1)
        result[j] = V
        C = [inputs[i][j - 1] * C[i] * dag(V) for i in eachindex(inputs)]
        rho = sum([prime(c, (s[j - 1], link)) * dag(c) for c in C])
    end
    result[1] = sum(C)
    ITensorMPS.setleftlim!(result, 0)
    ITensorMPS.setrightlim!(result, 2)
    return result
end

function compress_resume!(psi::MPS, report, iteration, operation)
    maxdim = report["max_bond_dimension"]
    (maxdim == 0 || maxlinkdim(psi) <= maxdim) && return psi
    function callback(; link, truncation_error)
        # truncate! gives a discarded weight rather than a Spectrum.
        bond = last(link)
        discarded = Float64(truncation_error)
        report["measured_bonds"] += 1
        dimension = linkdim(psi, bond)
        report["max_observed_bond_dimension"] = max(report["max_observed_bond_dimension"], dimension)
        dimension >= maxdim && (report["cap_reached_bonds"] += 1)
        discarded > report["requested_cutoff"] && (report["cutoff_exceeded_bonds"] += 1)
        if discarded > report["max_discarded_weight"]
            report["max_discarded_weight"] = discarded
            report["worst_event"] = Dict("iteration" => iteration, "operation" => operation,
                "bond" => bond, "bond_dimension" => dimension, "discarded_weight" => discarded)
        end
    end
    truncate!(psi; cutoff=report["requested_cutoff"], maxdim, callback)
    return psi
end

end
