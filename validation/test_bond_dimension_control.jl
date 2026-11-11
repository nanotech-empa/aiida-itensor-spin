using Test, Random, ITensors, ITensorMPS, JLD2
include("../remote/engine/ITensor_libs/Imports.jl")
include("../remote/engine/SpectralFunctionCheckpointed.jl")
using .SpectralFunctionCheckpointed
using .SpectralFunctionCheckpointed: Chebyshev_moments_streaming
using .SpectralFunctionCheckpointed.BondDimensionControl

Random.seed!(20260724)
@testset "Measured bond cap" begin
    for conserve in (false, true)
        sites = siteinds("S=1/2", 6; conserve_qns=conserve)
        psi = random_mps(sites, ["Up", "Dn", "Up", "Dn", "Up", "Dn"]; linkdims=8)
        os = OpSum()
        for j in 1:5
            os += 0.7, "Sz", j, "Sz", j+1
            os += 0.35, "S+", j, "S-", j+1
            os += 0.35, "S-", j, "S+", j+1
        end
        H = MPO(os, sites)
        for cap in (2, 100)
            report = compression_report(cap, 1e-12)
            a = controlled_apply(H, psi, report, 1, "apply")
            native = apply(H, psi; cutoff=1e-12, maxdim=cap)
            @test maxlinkdim(a) <= cap
            @test norm(a - native) < 1e-7
            b = controlled_add(2a, -psi, report, 2, "add")
            native_sum = add(2a, -psi; cutoff=1e-12, maxdim=cap)
            @test maxlinkdim(b) <= cap
            @test norm(b - native_sum) < 1e-7
            @test report["measured_bonds"] == 10
            conserve && @test flux(b) == flux(psi)
            if cap == 2
                @test report["cap_reached_bonds"] > 0
                @test report["cutoff_exceeded_bonds"] > 0
                @test report["max_discarded_weight"] > 1e-12
            else
                @test report["cap_reached_bonds"] == 0
            end
        end
        B = Operators.Szi_op(1, sites)
        mktempdir() do directory
            file = joinpath(directory, "recurrence.jld2")
            common = (; cutoff=1e-12, doubleN=true, max_bond_dimension=2,
                checkpoint_filename=file, checkpoint_key="bond-cap-test")
            @test_throws SpectralFunctionCheckpointed.IntentionalCheckpointStop Chebyshev_moments_streaming(B, 0.1H, psi, B, 7; common..., stop_after_n=3)
            checkpoint = load(file)
            @test maxlinkdim(checkpoint["tnm1"]) <= 2
            report = Dict{String,Any}()
            resumed = Chebyshev_moments_streaming(B, 0.1H, psi, B, 7;
                common..., compression_diagnostics=report)
            direct = Chebyshev_moments_streaming(B, 0.1H, psi, B, 7;
                cutoff=1e-12, doubleN=true, max_bond_dimension=2)
            @test resumed ≈ direct atol=1e-10
            @test report["cap_reached_bonds"] > 0
            @test report["max_observed_bond_dimension"] <= 2
        end
        # A legacy uncapped checkpoint can adopt the explicitly requested cap.
        # Earlier moments and their uncapped policy must remain identifiable.
        mktempdir() do directory
            file = joinpath(directory, "legacy.jld2")
            common = (; cutoff=1e-12, doubleN=true,
                checkpoint_filename=file, checkpoint_key="legacy-test")
            @test_throws SpectralFunctionCheckpointed.IntentionalCheckpointStop Chebyshev_moments_streaming(B, 0.1H, psi, B, 7; common..., stop_after_n=3)
            old = load(file)
            # Remove new optional fields to exercise the genuine old format.
            delete!(old, "max_bond_dimension")
            delete!(old, "compression_diagnostics")
            jldsave(file; (Symbol(k)=>v for (k,v) in old)...)
            report = Dict{String,Any}()
            moments = Chebyshev_moments_streaming(B, 0.1H, psi, B, 7;
                common..., max_bond_dimension=2, compression_diagnostics=report)
            @test moments[1:5] == old["moments"][1:5]
            @test report["policy_history"][end]["previous_max_bond_dimension"] == 0
            @test report["policy_history"][end]["from_iteration"] == 4
            @test maxlinkdim(load(file)["tnm1"]) <= 2
        end
    end
end
