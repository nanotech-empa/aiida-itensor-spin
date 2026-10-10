using TOML, Statistics

function logical_cpus(value)
    result = Set{Int}()
    for part in split(value, ',')
        endpoints = parse.(Int, split(part, '-'))
        union!(result, length(endpoints) == 1 ? endpoints : (endpoints[1]:endpoints[2]))
    end
    return result
end

function main(args)
    iseven(length(args)) || error("Options require --name value pairs")
    o = Dict(args[i]=>args[i+1] for i in 1:2:length(args))
    directory = o["--directory"]
    workers = parse(Int, get(o, "--workers", "6"))
    workers > 0 || error("Worker count must be positive")
    reference = TOML.parsefile(o["--reference"])
    records = [TOML.parsefile(joinpath(directory,"worker_$i.toml")) for i in 1:workers]
    cpu_sets = Set{Int}[]
    for r in records
        for key in ("fixture_sha256","benchmark_script_sha256","mode","julia_threads","blas_threads")
            r[key] == reference[key] || error("Worker/reference mismatch: $key")
        end
        r["barrier_enabled"] && r["warmup_uses_measured_fixture"] || error("Missing full warmup/barrier")
        length(r["samples"]) == r["repeats"] || error("Incomplete measurements")
        affinity = logical_cpus(r["cpu_affinity_logical_ids"])
        !isempty(affinity) || error("Empty CPU affinity")
        all(isempty(intersect(affinity, previous)) for previous in cpu_sets) || error("Worker CPU affinities overlap")
        push!(cpu_sets, affinity)
    end
    started = minimum(r["measurements_started_at_unix"] for r in records)
    finished = maximum(r["measurements_finished_at_unix"] for r in records)
    elapsed = finished - started
    elapsed > 0 || error("Invalid measurement window")
    measured_steps = sum(r["repeats"] for r in records)
    reference_rate = reference["repeats"] / reference["measurements_elapsed_seconds"]
    result = Dict(
        "workers"=>workers,
        "measured_steps"=>measured_steps,
        "measurement_window_seconds"=>elapsed,
        "completed_steps_per_hour_including_gc_and_representative_writes"=>3600*measured_steps/elapsed,
        "throughput_vs_ideal_independent_workers"=>(measured_steps/elapsed)/(workers*reference_rate),
        "kernel_median_seconds_by_worker"=>[r["median"]["kernel_seconds"] for r in records],
        "kernel_slowdown_vs_isolated_by_worker"=>[r["median"]["kernel_seconds"]/reference["median"]["kernel_seconds"] for r in records],
        "rss_high_water_kib_by_worker"=>[r["rss_high_water_kib_before_final_save"] for r in records],
        "cpu_affinity_logical_ids_by_worker"=>[r["cpu_affinity_logical_ids"] for r in records],
        "max_abs_moment_difference_vs_isolated"=>maximum(maximum(abs.(r["moments"] .- reference["moments"])) for r in records),
        "fixture_sha256"=>reference["fixture_sha256"],
        "note"=>"Identical input replicated across workers: a contention diagnostic, not the heterogeneous production workload. Warmup and load excluded; measurement window includes GC and representative writes, not final result export. Summed individual peak RSS is not a simultaneous node-memory measurement.")
    open(o["--output"],"w") do io
        TOML.print(io, result; sorted=true)
    end
    println(result)
end

main(ARGS)
