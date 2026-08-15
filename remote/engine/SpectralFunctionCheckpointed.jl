module SpectralFunctionCheckpointed

using ITensorMPS, ITensors
using LinearAlgebra
using Dates
using JLD2
using ..Operators
using ..SpectralFunction

const MOMENT_CONVENTION = "chebyshev-2n-minus-1-v2"
const CHECKPOINT_FORMAT_VERSION = 2

struct IntentionalCheckpointStop <: Exception
    completed_n::Int
end

function Base.showerror(io::IO, error::IntentionalCheckpointStop)
    print(io, "intentional stop after checkpointing iteration ", error.completed_n)
end

"""
    Chebyshev_moments_streaming(C, H, ψ0, B, N; ...)

Compute the Chebyshev recurrence while retaining only adjacent MPS vectors.
With `doubleN=true`, `N` vectors determine exactly `2N-1` moments,
`mu_0, ..., mu_(2N-2)`.
"""
function Chebyshev_moments_streaming(
    C::MPO,
    H::MPO,
    ψ0::MPS,
    B::MPO,
    N::Int;
    cutoff::Float64=1e-8,
    doubleN::Bool=false,
    dK::Union{Int,Nothing}=nothing,
    nsweeps::Union{Int,Nothing}=nothing,
    ϵP::Union{Float64,Nothing}=nothing,
    checkpoint_filename::Union{String,Nothing}=nothing,
    checkpoint_key::Union{String,Nothing}=nothing,
    checkpoint_every::Int=1,
    heartbeat_every::Int=1,
    stop_after_n::Union{Int,Nothing}=nothing,
    stop_before_epoch::Union{Float64,Nothing}=nothing,
)
    energy_truncation =
        dK !== nothing && nsweeps !== nothing && ϵP !== nothing
    checkpoint_every >= 1 || error("checkpoint_every must be >= 1")
    heartbeat_every >= 1 || error("heartbeat_every must be >= 1")
    stop_before_epoch === nothing || stop_before_epoch > 0 ||
        error("stop_before_epoch must be positive")
    checkpoint_filename === nothing || checkpoint_key !== nothing ||
        error("checkpoint_key is required when checkpointing is enabled")

    function save_progress(completed_n, moments, tnm2, tnm1)
        checkpoint_filename === nothing && return
        temporary_filename = checkpoint_filename * ".tmp.$(getpid())"
        jldsave(
            temporary_filename;
            checkpoint_format_version=CHECKPOINT_FORMAT_VERSION,
            moment_convention=MOMENT_CONVENTION,
            checkpoint_key,
            completed_n,
            N,
            cutoff,
            doubleN,
            energy_truncation,
            dK,
            nsweeps,
            epsilonP=ϵP,
            moments,
            tnm2,
            tnm1,
            saved_at=DateTime(now()),
        )
        mv(temporary_filename, checkpoint_filename; force=true)
        println(
            "checkpointed recurrence completed_n=$completed_n/$N " *
            "checkpoint=$checkpoint_filename",
        )
        flush(stdout)
        return
    end

    recurrence_start = time()
    if checkpoint_filename !== nothing && isfile(checkpoint_filename)
        progress = load(checkpoint_filename)
        expected = (
            "checkpoint_format_version" => CHECKPOINT_FORMAT_VERSION,
            "moment_convention" => MOMENT_CONVENTION,
            "checkpoint_key" => checkpoint_key,
            "N" => N,
            "cutoff" => cutoff,
            "doubleN" => doubleN,
            "energy_truncation" => energy_truncation,
            "dK" => dK,
            "nsweeps" => nsweeps,
            "epsilonP" => ϵP,
        )
        for (key, expected_value) in expected
            haskey(progress, key) ||
                error("Recurrence checkpoint is missing key: $key")
            progress[key] == expected_value ||
                error("Recurrence checkpoint mismatch for $key")
        end
        completed_n = progress["completed_n"]
        2 <= completed_n <= N ||
            error("Invalid checkpoint iteration completed_n=$completed_n")
        moments = progress["moments"]
        length(moments) == (doubleN ? 2 * N - 1 : N) ||
            error("Recurrence checkpoint has invalid moments length")
        tnm2 = progress["tnm2"]
        tnm1 = progress["tnm1"]
        println(
            "resumed recurrence completed_n=$completed_n next_n=$(completed_n + 1) " *
            "checkpoint=$checkpoint_filename",
        )
        flush(stdout)
    else
        t0 = apply(C, ψ0; cutoff=cutoff)
        if energy_truncation
            println("energy truncation of |t_1>")
            t0 = SpectralFunction.proj_highE(
                t0,
                H,
                dK,
                nsweeps,
                ϵP;
                cutoff=cutoff,
            )
        end

        t1 = apply(H, t0; cutoff=cutoff)
        if energy_truncation
            println("\nenergy truncation of |t_2>")
            t1 = SpectralFunction.proj_highE(
                t1,
                H,
                dK,
                nsweeps,
                ϵP;
                cutoff=cutoff,
            )
        end

        if doubleN
            # N vectors |t_0>, ..., |t_(N-1)> determine moments
            # mu_0, ..., mu_(2N-2): exactly 2N-1 values.
            moments = zeros(2 * N - 1)
            moments[1] = inner(t0, t0)
            moments[2] = inner(t1, t0)

            # Seed mu_1 and then apply the same recurrence identity at n=1.
            moments[2] = 2 * inner(t1, t0) - moments[2]
            moments[3] = 2 * inner(t1, t1) - moments[1]
        else
            moments = zeros(N)
            moments[1] = inner(adjoint(ψ0), B, t0)
            moments[2] = inner(adjoint(ψ0), B, t1)
        end

        tnm2 = t0
        tnm1 = t1
        completed_n = 2
        save_progress(completed_n, moments, tnm2, tnm1)
    end

    for n in (completed_n + 1):N
        iteration_start = time()
        tmp = 2 * apply(H, tnm1; cutoff=cutoff)
        tn = add(tmp, -tnm2; cutoff=cutoff)
        if energy_truncation
            println("\nenergy truncation of |t_", n, ">")
            tn = SpectralFunction.proj_highE(
                tn,
                H,
                dK,
                nsweeps,
                ϵP;
                cutoff=cutoff,
            )
            n == N && println("\n")
        end

        if doubleN
            moments[2*n-2] = 2 * inner(tn, tnm1) - moments[2]
            moments[2*n-1] = 2 * inner(tn, tn) - moments[1]
        else
            moments[n] = inner(adjoint(ψ0), B, tn)
        end
        tnm2, tnm1 = tnm1, tn
        completed_n = n

        if n % heartbeat_every == 0 || n == N
            println(
                "progress completed_n=$n/$N " *
                "iteration_seconds=$(time() - iteration_start) " *
                "elapsed_seconds=$(time() - recurrence_start) " *
                "maxlinkdim=$(maxlinkdim(tnm1))",
            )
            flush(stdout)
        end
        deadline_reached = stop_before_epoch !== nothing && time() >= stop_before_epoch
        should_stop = n < N && (n == stop_after_n || deadline_reached)
        if checkpoint_filename !== nothing &&
           (n % checkpoint_every == 0 || n == N || should_stop)
            save_progress(completed_n, moments, tnm2, tnm1)
        end
        should_stop && throw(IntentionalCheckpointStop(n))
    end

    return moments
end

function spectral_function(
    ψ0::MPS,
    H::MPO,
    B::MPO,
    C::MPO,
    E0::Float64,
    ωmax::Float64,
    sites;
    cutoff::Float64=1e-8,
    ϵt::Float64=0.025,
    Nω::Union{Int,Nothing}=nothing,
    Emax::Union{Float64,Nothing}=nothing,
    Wstar::Union{Float64,Nothing}=nothing,
    try_doubleN::Bool=false,
    ω_smearing::Union{Float64,Nothing}=nothing,
    N::Union{Int,Nothing}=nothing,
    dK::Int=30,
    nsweeps::Int=10,
    ϵP::Float64=1.0,
    checkpoint_filename::Union{String,Nothing}=nothing,
    checkpoint_key::Union{String,Nothing}=nothing,
    checkpoint_every::Int=1,
    heartbeat_every::Int=1,
    stop_after_n::Union{Int,Nothing}=nothing,
    stop_before_epoch::Union{Float64,Nothing}=nothing,
)
    if ωmax <= 0
        error("ERROR: ωmax must be > 0")
    elseif cutoff <= 0
        error("ERROR: cutoff must be > 0")
    elseif ϵt <= 0
        error("ERROR: ϵt must be > 0")
    elseif dK <= 0
        error("ERROR: dK must be > 0")
    elseif nsweeps <= 0
        error("ERROR: nsweeps must be > 0")
    elseif ϵP <= 0
        error("ERROR: ϵP must be > 0")
    end

    println("# Bandwidth info")
    if Emax !== nothing && Wstar === nothing
        println("Emax provided -> calculation with full bandwidth")
        Wstar = Emax - E0
        energy_truncation = false
    elseif Wstar !== nothing && Emax === nothing
        println(
            "Wstar provided -> calculation with effective bandwidth " *
            "(energy truncation triggered)",
        )
        energy_truncation = true
    elseif Emax === nothing && Wstar === nothing
        error("ERROR: provide either Emax or Wstar")
    else
        error("ERROR: provide only Emax or Wstar, not both")
    end
    println()

    if Wstar <= 0
        error("ERROR: Wstar must be > 0, check Emax or Wstar")
    elseif ωmax > Wstar
        error("ERROR: ωmax must be <= Wstar")
    end

    println("# Info about Chebyshev polynomials")
    doubleN = try_doubleN && isapprox(B, swapprime(dag(C), 0 => 1))
    if doubleN
        println(
            "Routine that doubles number of Chebyshev polynomials " *
            "triggered (B = C^†)",
        )
    end

    if ω_smearing !== nothing && N === nothing
        N = ceil(Int, Wstar / ω_smearing)
        if doubleN
            N = round(Int, N / 2)
            Neff = 2 * N - 1
        else
            Neff = N
        end
        println(
            "ω_smearing provided -> effective number of Chebyshev " *
            "polynomials = $Neff",
        )
        Nω === nothing && (Nω = round(Int, 5 * ωmax / ω_smearing))
    elseif N !== nothing && ω_smearing === nothing
        if doubleN
            N = round(Int, N / 2)
            Neff = 2 * N - 1
        else
            Neff = N
        end
        println("N provided -> effective number of Chebyshev polynomials = $Neff")
        Nω === nothing && (Nω = round(Int, 5 * ωmax * N / Wstar))
    elseif ω_smearing === nothing && N === nothing
        error("ERROR: provide either ω_smearing or N")
    else
        error("ERROR: provide only ω_smearing or N, not both")
    end
    println()

    N < 2 && error("ERROR: N must be >= 2, check ω_smearing or N")
    Nω < 2 && error("ERROR: Nω must be >= 2")

    ω = collect(range(0, ωmax, length=Nω))
    Wp = 1 - ϵt / 2
    a = Wstar / (2 * Wp)
    ωp = ω / a .- Wp
    I = Operators.I_op(sites)
    Hp = add(1/a * add(H, -E0*I; cutoff=cutoff), -Wp*I; cutoff=cutoff)

    if energy_truncation
        moments = Chebyshev_moments_streaming(
            C,
            Hp,
            ψ0,
            B,
            N;
            cutoff=cutoff,
            doubleN=doubleN,
            dK=dK,
            nsweeps=nsweeps,
            ϵP=ϵP,
            checkpoint_filename=checkpoint_filename,
            checkpoint_key=checkpoint_key,
            checkpoint_every=checkpoint_every,
            heartbeat_every=heartbeat_every,
            stop_after_n=stop_after_n,
            stop_before_epoch=stop_before_epoch,
        )
    else
        moments = Chebyshev_moments_streaming(
            C,
            Hp,
            ψ0,
            B,
            N;
            cutoff=cutoff,
            doubleN=doubleN,
            checkpoint_filename=checkpoint_filename,
            checkpoint_key=checkpoint_key,
            checkpoint_every=checkpoint_every,
            heartbeat_every=heartbeat_every,
            stop_after_n=stop_after_n,
            stop_before_epoch=stop_before_epoch,
        )
    end
    @assert Neff == length(moments)

    damping = SpectralFunction.Jackson_damping(Neff)
    polynomials = SpectralFunction.Chebyshev_polynomials(ωp, Neff)
    prefactor = 1/a * 1 ./ (pi * sqrt.(1 .- ωp.^2))
    sumval = sum(
        [damping[n] * moments[n] * polynomials[:, n] for n in 2:Neff],
    )
    return prefactor .* (damping[1] * moments[1] .+ 2 * sumval)
end

end
