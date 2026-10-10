using ITensors, ITensorMPS, JLD2, LinearAlgebra, Random, Statistics, TOML, SHA
include("../remote/engine/GenericSpinSites.jl")
include("../remote/engine/ITensor_libs/Operators.jl")
include("../remote/engine/BondDimensionControl.jl")
using .GenericSpinSites, .BondDimensionControl

function options(args)
    iseven(length(args)) || error("Options require --name value pairs")
    Dict(args[i] => args[i+1] for i in 1:2:length(args))
end

function configure(mode, blas_threads)
    ITensors.Strided.set_num_threads(1)
    BLAS.set_num_threads(blas_threads)
    ITensors.enable_threaded_blocksparse(mode == "blocks")
    mode in ("serial", "blocks", "blas") || error("Unknown mode: $mode")
    mode == "blocks" && blas_threads != 1 && error("Avoid nested BLAS/block threading")
    mode == "blas" && Threads.nthreads() != 1 && error("BLAS mode uses one Julia thread")
end

function linux_process_info()
    Sys.islinux() || return Dict{String,Any}()
    status = read("/proc/self/status", String)
    rss = match(r"(?m)^VmHWM:\s+(\d+)\s+kB", status)
    cpus = match(r"(?m)^Cpus_allowed_list:\s+(\S+)", status)
    info = Dict{String,Any}()
    isnothing(rss) || (info["rss_high_water_kib_before_final_save"] = parse(Int, rss[1]))
    isnothing(cpus) || (info["cpu_affinity_logical_ids"] = cpus[1])
    return info
end

function wait_for_barrier(o)
    haskey(o, "--barrier-directory") || return 0.0
    directory = o["--barrier-directory"]
    id = parse(Int, o["--barrier-id"])
    workers = parse(Int, o["--barrier-workers"])
    1 <= id <= workers || error("Invalid barrier worker id/count")
    isdir(directory) || error("Barrier directory must be created by the coordinator")
    ready = joinpath(directory, "ready-$id")
    isfile(ready) && error("Refusing to reuse barrier readiness marker")
    write(ready, string(time()))
    started = time()
    while !isfile(joinpath(directory, "release"))
        time() - started <= 1800 || error("Barrier release timed out")
        sleep(0.25)
    end
    return time() - started
end

function prepare(o)
    Random.seed!(20260724)
    configure("serial", 1)
    output = o["--fixture"]
    isfile(output) && error("Refusing to overwrite fixture: $output")
    if haskey(o, "--ground-state")
        gs_path, progress_path = o["--ground-state"], o["--progress"]
        gs, progress = load(gs_path), load(progress_path)
        gs_hash = bytes2hex(open(sha256, gs_path))
        split(progress["checkpoint_key"], '|')[2] == gs_hash || error("GS/checkpoint identity mismatch")
        progress["moment_convention"] == "chebyshev-2n-minus-1-v2" || error("Moment convention mismatch")
        progress["energy_truncation"] == false || error("Filtered checkpoints require a separate benchmark")
        cutoff = Float64(progress["cutoff"])
        cap = Int(get(progress, "max_bond_dimension", 0))
        Wp = 1 - 0.025/2
        E0 = minimum(gs["En_sym"])
        a = (gs["Emax"] - E0)/(2Wp)
        Iop = Operators.I_op(gs["sites"])
        H = add((1/a)*add(gs["H"], -E0*Iop; cutoff=cutoff), -Wp*Iop; cutoff=cutoff)
        previous, current = progress["tnm2"], progress["tnm1"]
        moment0, moment1 = Float64(progress["moments"][1]), Float64(progress["moments"][2])
        initial_moments = Float64.(progress["moments"])
        metadata = Dict(
            "case" => "saved physical recurrence checkpoint",
            "ground_state_sha256" => gs_hash,
            "progress_sha256" => bytes2hex(open(sha256, progress_path)),
            "completed_vectors" => Int(progress["completed_n"]),
            "target_vectors" => Int(progress["N"]),
            "length" => length(gs["sites"]),
            "source_segment_uuid" => get(o, "--source-uuid", "unspecified"),
        )
    else
        L = parse(Int, get(o, "--length", "24"))
        bond = parse(Int, get(o, "--bond", "256"))
        iseven(L) && L >= 6 || error("Synthetic mixed-spin case requires even length >=6")
        spins = fill(2, L)
        spins[L÷2:L÷2+1] .= 1
        sites = [siteind("GenericSpin"; two_s=s, conserve_sz=true) for s in spins]
        state = GenericSpinSites.product_state_for_twice_sz(spins, 0)
        current = random_mps(sites, state; linkdims=bond)
        previous = copy(current)
        os = OpSum()
        for j in 1:(L-1)
            J = j == L÷2 ? 96.8 : j in (L÷2-1,L÷2+1) ? 16.7 : 18.4
            os += 0.01J, "Sz", j, "Sz", j+1
            os += 0.005J, "S+", j, "S-", j+1
            os += 0.005J, "S-", j, "S+", j+1
        end
        H = MPO(os, sites)
        cutoff, cap = 1e-8, bond
        moment0, moment1 = inner(current,current), inner(current,previous)
        initial_moments = zeros(31)
        initial_moments[1:2] .= (moment0,moment1)
        metadata = Dict("case"=>"synthetic QN kernel workload, not a ground state or physical spectrum",
            "length"=>L, "requested_bond_dimension"=>bond, "seed"=>20260724)
    end
    metadata["max_current_bond"] = maxlinkdim(current)
    metadata["cutoff"] = cutoff
    metadata["max_bond_dimension"] = cap
    mkpath(dirname(abspath(output)))
    jldsave(output; H, previous, current, moment0, moment1, initial_moments,
        cutoff, cap, metadata)
    println("FIXTURE ", output, " bond=", maxlinkdim(current), " case=", metadata["case"])
end

function step(data, checkpoint_file; record_time=true)
    report = compression_report(data["cap"], data["cutoff"])
    t0 = time_ns()
    tmp = 2*controlled_apply(data["H"], data["current"], report, 1, "hamiltonian")
    t1 = time_ns()
    next = controlled_add(tmp, -data["previous"], report, 1, "recurrence_sum")
    t2 = time_ns()
    measured_moments = [2*inner(next,data["current"])-data["moment1"],
                        2*inner(next,next)-data["moment0"]]
    t3 = time_ns()
    jldsave(checkpoint_file; tnm2=data["current"], tnm1=next,
        moments=data["initial_moments"], compression_diagnostics=report, benchmark_only=true)
    t4 = time_ns()
    timings = Dict("apply_seconds"=>(t1-t0)/1e9, "add_seconds"=>(t2-t1)/1e9,
        "moments_seconds"=>(t3-t2)/1e9, "checkpoint_seconds"=>(t4-t3)/1e9,
        "kernel_seconds"=>(t3-t0)/1e9, "total_seconds"=>(t4-t0)/1e9)
    return next, measured_moments, timings, report
end

function run(o)
    mode = get(o,"--mode","serial")
    configure(mode, parse(Int,get(o,"--blas-threads","1")))
    output = o["--output"]
    isfile(output*".toml") && error("Refusing to overwrite measured result")
    tload = time()
    data = load(o["--fixture"])
    load_seconds = time()-tload
    repeats = parse(Int,get(o,"--repeats","3"))
    repeats >= 1 || error("Repeats must be >=1")
    mkpath(dirname(abspath(output)))
    println("WARMUP mode=$mode Julia=$(Threads.nthreads()) BLAS=$(BLAS.get_num_threads())")
    flush(stdout)
    twarm = time()
    warm_data = haskey(o,"--warmup-fixture") ? load(o["--warmup-fixture"]) : data
    (warm_data["cap"] > 0) == (data["cap"] > 0) || error("Warmup must exercise the same capped/uncapped implementation")
    warm, warm_moments, _, _ = step(warm_data, output*".scratch.jld2")
    warm_seconds = time()-twarm
    barrier_wait_seconds = wait_for_barrier(o)
    samples = Dict{String,Float64}[]
    next, measured_moments, report = warm, warm_moments, Dict{String,Any}()
    measurements_started_at_unix = time()
    for sample in 1:repeats
        GC.gc()
        next, measured_moments, timings, report = step(data, output*".scratch.jld2")
        push!(samples,timings)
        println("SAMPLE $sample kernel=$(timings["kernel_seconds"]) apply=$(timings["apply_seconds"]) add=$(timings["add_seconds"])")
        flush(stdout)
    end
    measurements_finished_at_unix = time()
    summary = Dict(key=>median([s[key] for s in samples]) for key in keys(samples[1]))
    result = Dict{String,Any}(
        "mode"=>mode, "julia_threads"=>Threads.nthreads(), "blas_threads"=>BLAS.get_num_threads(),
        "strided_threads"=>ITensors.Strided.get_num_threads(),
        "threaded_blocksparse"=>ITensors.using_threaded_blocksparse(),
        "julia_version"=>string(VERSION), "ITensors_version"=>string(pkgversion(ITensors)),
        "ITensorMPS_version"=>string(pkgversion(ITensorMPS)), "blas_config"=>string(BLAS.get_config()),
        "os"=>string(Sys.KERNEL), "architecture"=>string(Sys.ARCH),
        "cpu_threads_visible"=>Sys.CPU_THREADS, "memory_total_bytes"=>Sys.total_memory(),
        "fixture_sha256"=>bytes2hex(open(sha256,o["--fixture"])),
        "benchmark_script_sha256"=>bytes2hex(open(sha256,@__FILE__)),
        "fixture_metadata"=>data["metadata"], "load_seconds"=>load_seconds,
        "warmup_fixture_sha256"=>haskey(o,"--warmup-fixture") ? bytes2hex(open(sha256,o["--warmup-fixture"])) : bytes2hex(open(sha256,o["--fixture"])),
        "warmup_uses_measured_fixture"=>!haskey(o,"--warmup-fixture"),
        "warmup_seconds"=>warm_seconds, "median"=>summary, "samples"=>samples,
        "barrier_wait_seconds"=>barrier_wait_seconds,
        "barrier_enabled"=>haskey(o,"--barrier-directory"),
        "measurements_started_at_unix"=>measurements_started_at_unix,
        "measurements_finished_at_unix"=>measurements_finished_at_unix,
        "measurements_elapsed_seconds"=>measurements_finished_at_unix-measurements_started_at_unix,
        "moments"=>measured_moments, "next_max_bond"=>maxlinkdim(next),
        "max_discarded_weight"=>get(report,"max_discarded_weight",0.0),
        "repeats"=>repeats,
        "note"=>"Each repeat starts from identical saved vectors. Warmup and load are excluded from medians. Checkpoint timer is a representative write, not a production I/O benchmark.")
    merge!(result, linux_process_info())
    jldsave(output*".jld2"; next, measured_moments)
    open(output*".toml","w") do io
        TOML.print(io,result; sorted=true)
    end
    println("RESULT ",output*".toml")
end

function compare(o)
    configure(get(o,"--mode","serial"), parse(Int,get(o,"--blas-threads","1")))
    a,b = load(o["--reference"]),load(o["--candidate"])
    # Stable explicit subtraction avoids subtracting nearly equal scalar norms.
    relative_vector_error = norm(add(a["next"],-b["next"]; cutoff=1e-14))/norm(a["next"])
    moment_error = maximum(abs.(a["measured_moments"]-b["measured_moments"]))
    result=Dict("relative_vector_error"=>relative_vector_error,"moment_max_abs_error"=>moment_error,
        "comparison_blas_threads"=>BLAS.get_num_threads(),"comparison_julia_threads"=>Threads.nthreads())
    open(o["--output"],"w") do io; TOML.print(io,result); end
    println(result)
end

o=options(ARGS)
action=get(o,"--action","run")
action == "prepare" ? prepare(o) : action == "compare" ? compare(o) : run(o)
