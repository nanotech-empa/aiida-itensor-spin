# Small physical restart test, run on a compute node before the production driver.
using Test, Random, ITensors, ITensorMPS, JLD2, LinearAlgebra
root = ENV["ITENSOR_SPIN_CASE_ROOT"]
include(joinpath(root, "ITensor_libs", "Imports.jl"))
include(joinpath(root, "SpectralFunctionCheckpointed.jl"))
include(joinpath(root, "KernelThreading.jl"))
using .SpectralFunctionCheckpointed: Chebyshev_moments_streaming, IntentionalCheckpointStop
Random.seed!(20261016)
@testset "Legacy checkpoint advances with BLAS8" begin
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
        B = Operators.Szi_op(1, sites)
        mktempdir() do directory
            file = joinpath(directory, "recurrence.jld2")
            common = (; cutoff=1e-8, doubleN=true, max_bond_dimension=8000,
                      checkpoint_filename=file, checkpoint_key="blas-switch-test")
            KernelThreading.configure_kernel_threading("blocksparse"; environment=Dict("ITENSOR_SPIN_KERNEL_THREADING_MODE"=>"legacy"))
            ENV["ITENSOR_SPIN_KERNEL_THREADING_MODE"] = "legacy"
            @test_throws IntentionalCheckpointStop Chebyshev_moments_streaming(B, 0.1H, psi, B, 9; common..., stop_after_n=3)
            old = load(file)
            # Represent the original checkpoint format, without new execution annotations.
            for key in ("kernel_threading_mode", "julia_threads", "blas_threads")
                delete!(old, key)
            end
            jldsave(file; (Symbol(k)=>v for (k,v) in old)...)
            baseline = Chebyshev_moments_streaming(B, 0.1H, psi, B, 9; cutoff=1e-8, doubleN=true, max_bond_dimension=8000)
            ENV["ITENSOR_SPIN_KERNEL_THREADING_MODE"] = "blas"
            ENV["ITENSOR_SPIN_BLAS_THREADS"] = "8"
            KernelThreading.configure_kernel_threading("blocksparse")
            @test Threads.nthreads() == 1
            @test BLAS.get_num_threads() == 8
            @test !ITensors.using_threaded_blocksparse()
            @test ITensors.Strided.get_num_threads() == 1
            @test_throws IntentionalCheckpointStop Chebyshev_moments_streaming(B, 0.1H, psi, B, 9; common..., stop_after_n=5)
            advanced = load(file)
            @test advanced["completed_n"] == 5
            @test advanced["moments"][1:5] == old["moments"][1:5]
            @test advanced["kernel_threading_mode"] == "blas"
            @test advanced["blas_threads"] == 8
            @test advanced["julia_threads"] == 1
            result = Chebyshev_moments_streaming(B, 0.1H, psi, B, 9; common...)
            @test result ≈ baseline atol=1e-10 rtol=1e-10
        end
    end
end
println("BLAS_CONTINUATION_PREFLIGHT_PASSED")
