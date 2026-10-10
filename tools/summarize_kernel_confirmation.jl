using TOML

function main(args)
    iseven(length(args)) || error("Options require --name value pairs")
    o = Dict(args[i]=>args[i+1] for i in 1:2:length(args))
    root = o["--results"]
    baseline = TOML.parsefile(joinpath(root,"isolated","blocks2.toml"))
    candidate = TOML.parsefile(joinpath(root,"isolated","blas8.toml"))
    for key in ("fixture_sha256","benchmark_script_sha256","julia_version","ITensors_version","ITensorMPS_version")
        baseline[key] == candidate[key] || error("Reference/candidate mismatch: $key")
    end
    all(r["warmup_uses_measured_fixture"] && r["repeats"]==3 && length(r["samples"])==3
        for r in (baseline,candidate)) || error("Expected full physical warmup and three repeats")
    result = Dict(
        "status"=>"complete",
        "fixture_metadata"=>candidate["fixture_metadata"],
        "fixture_sha256"=>candidate["fixture_sha256"],
        "baseline_kernel_median_seconds"=>baseline["median"]["kernel_seconds"],
        "candidate_kernel_median_seconds"=>candidate["median"]["kernel_seconds"],
        "candidate_speedup_vs_repeated_baseline"=>baseline["median"]["kernel_seconds"]/candidate["median"]["kernel_seconds"],
        "baseline_kernel_sample_seconds"=>[r["kernel_seconds"] for r in baseline["samples"]],
        "candidate_kernel_sample_seconds"=>[r["kernel_seconds"] for r in candidate["samples"]],
        "candidate_vs_previous_reference"=>TOML.parsefile(joinpath(root,"isolated","blas8.vs_previous.comparison.toml")),
        "repeated_baseline_vs_candidate"=>TOML.parsefile(joinpath(root,"isolated","blocks2.vs_blas8.comparison.toml")),
        "concurrent"=>TOML.parsefile(joinpath(root,"concurrent","summary.toml")),
        "concurrent_worker_1_vector_check"=>TOML.parsefile(joinpath(root,"concurrent","worker_1.comparison.toml")),
        "concurrent_worker_6_vector_check"=>TOML.parsefile(joinpath(root,"concurrent","worker_6.comparison.toml")),
        "note"=>"Same saved recurrence step repeated, not consecutive recurrence advancement. Six identical replicas are a contention proxy; heterogeneous production and final spectra still need validation. Comparisons use explicit MPS subtraction cutoff 1e-14.")
    open(joinpath(root,"summary.toml"),"w") do io
        TOML.print(io,result;sorted=true)
    end
    println(result)
end

main(ARGS)
