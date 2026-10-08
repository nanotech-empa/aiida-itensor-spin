pushfirst!(DEPOT_PATH, "/home/jovyan/.julia-itensor-spin-localhost")
ENV["GKSwstype"] = "100"
using ITensors, ITensorMPS, LinearAlgebra, JLD2, Printf
BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "production_engine", "Operators.jl"))
include(joinpath(@__DIR__, "production_engine", "SpectralFunction.jl"))
include(joinpath(@__DIR__, "production_engine", "SpectralFunctionCheckpointed.jl"))

function run_validation()
    # Isolated numerical check, not a replacement L101 calculation.
    # Both runs use the same exact ground state and Hamiltonian.
    L, J, site = 6, 100.0, 3
    cutoff, smearing, omega_max = 1e-10, 5.0, 100.0
    Wp = 1 - 0.025 / 2
    sites = siteinds("S=1/2", L; conserve_qns=false)
    os = OpSum()
    for j in 1:(L-1)
        os += J, "Sz", j, "Sz", j+1
        os += J/2, "S+", j, "S-", j+1
        os += J/2, "S-", j, "S+", j+1
    end
    H = MPO(os, sites)
    C = Operators.Szi_op(site, sites)
    dims = Tuple(dim.(sites))
    Hd = reshape(Array(contract(H), prime.(sites)..., sites...), prod(dims), prod(dims))
    Cd = reshape(Array(contract(C), prime.(sites)..., sites...), prod(dims), prod(dims))
    eig = eigen(Hermitian(Hd))
    E0, Emax = first(eig.values), last(eig.values)
    ground = eig.vectors[:, 1]
    psi = MPS(ITensor(reshape(ground, dims), sites...), sites; cutoff=1e-14)
    weights = abs2.(eig.vectors' * (Cd * ground))
    omega = collect(range(0, omega_max; length=100))
    results = Dict{String,Any}(
        "julia_version" => string(VERSION),
        "ITensors_version" => string(pkgversion(ITensors)),
        "ITensorMPS_version" => string(pkgversion(ITensorMPS)),
        "length" => L, "J_meV" => J, "site" => site,
        "cutoff" => cutoff, "omega_smearing_meV" => smearing,
        "omega_max_meV" => omega_max, "ground_energy_meV" => E0,
        "highest_energy_meV" => Emax, "full_bandwidth_meV" => Emax-E0,
        "scope" => "Small spin-half-chain projection validation; not L101 acceptance",
    )

    function reference(W, N; filtered)
        a = W / (2Wp)
        x = (eig.values .- E0) ./ a .- Wp
        keep = filtered ? findall(x .< 1.0) : collect(eachindex(x))
        mu = [sum(weights[k] * cos(n * acos(clamp(x[k], -1.0, 1.0))) for k in keep)
              for n in 0:(2N-2)]
        g = SpectralFunction.Jackson_damping(length(mu))
        xp = omega ./ a .- Wp
        polynomials = SpectralFunction.Chebyshev_polynomials(xp, length(mu))
        spectrum = (g[1]*mu[1] .+ 2 .* sum(
            [g[n]*mu[n] .* polynomials[:,n] for n in 2:length(mu)])) ./
            (a .* pi .* sqrt.(1 .- xp.^2))
        return mu, spectrum
    end

    spectra = Dict{String,Vector{Float64}}()
    for (name, W, filtered) in [("full", Emax-E0, false), ("filtered160", 160.0, true)]
        N = round(Int, ceil(Int, W / smearing) / 2)
        checkpoint = joinpath(@__DIR__, name * "_recurrence.jld2")
        # Fresh isolated paths make a repeated launch explicit rather than
        # silently reusing an earlier checkpoint or changing its convention.
        isfile(checkpoint) && error("Existing validation checkpoint: $checkpoint")
        println("VALIDATION_START name=$name W=$W N=$N filtered=$filtered")
        flush(stdout)
        timer = time()
        common = (; cutoff=cutoff, try_doubleN=true, ω_smearing=smearing,
                  Nω=length(omega), checkpoint_filename=checkpoint,
                  checkpoint_key=name, checkpoint_every=10, heartbeat_every=1)
        spectrum = if filtered
            SpectralFunctionCheckpointed.spectral_function(
                copy(psi), H, C, C, E0, omega_max, sites;
                common..., Wstar=W, dK=12, nsweeps=1, ϵP=1.0)
        else
            SpectralFunctionCheckpointed.spectral_function(
                copy(psi), H, C, C, E0, omega_max, sites;
                common..., Emax=Emax)
        end
        elapsed = time() - timer
        saved = load(checkpoint)
        exact_mu, exact_spectrum = reference(W, N; filtered=filtered)
        measured_mu = saved["moments"]
        metrics = Dict(
            "vectors" => N, "moments" => length(measured_mu),
            "elapsed_seconds_including_first_call_compilation" => elapsed,
            "max_bond_dimension_last_vector" => maxlinkdim(saved["tnm1"]),
            "moment_max_abs_difference_from_ED" => maximum(abs.(measured_mu-exact_mu)),
            "spectrum_relative_L2_difference_from_ED" => norm(spectrum-exact_spectrum)/norm(exact_spectrum),
            "spectrum_max_abs_difference_from_ED" => maximum(abs.(spectrum-exact_spectrum)),
            "all_finite" => all(isfinite, spectrum) && all(isfinite, measured_mu),
            "spectrum" => spectrum, "ED_spectrum" => exact_spectrum,
        )
        results[name] = metrics
        spectra[name] = spectrum
        println("VALIDATION_RESULT $name moment_error=$(metrics["moment_max_abs_difference_from_ED"]) spectrum_relative_L2=$(metrics["spectrum_relative_L2_difference_from_ED"]) elapsed=$elapsed")
        flush(stdout)
        jldsave(joinpath(@__DIR__, "validation_results.jld2"); results=results, omega=omega)
    end
    results["full_vs_filtered_spectrum_relative_L2"] = norm(spectra["full"]-spectra["filtered160"])/norm(spectra["full"])
    jldsave(joinpath(@__DIR__, "validation_results.jld2"); results=results, omega=omega)
    println("VALIDATION_COMPLETE full_vs_filtered_relative_L2=$(results["full_vs_filtered_spectrum_relative_L2"])")
end

open(joinpath(@__DIR__, "validation.log"), "w") do log
    redirect_stdout(log) do
        redirect_stderr(log) do
            run_validation()
        end
    end
end
println("Results and log saved to ", @__DIR__)
