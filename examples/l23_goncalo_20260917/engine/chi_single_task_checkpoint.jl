###############################################################################
# Generic spectral worker.
#
# The complete workload is derived from the GS checkpoint:
#   (symmetrized ground state, site, channel), channel ∈ (zz, pm, mp).
# Each rank receives an explicit manifest of unfinished task IDs. No system
# size is hardcoded, and the ground state is never recomputed here.
###############################################################################
using Dates
using JLD2
using LinearAlgebra
using Printf
using SHA
using Base.Threads
using ITensors
using ITensorMPS

const CASE_ROOT = @__DIR__
include(joinpath(CASE_ROOT, "ITensor_libs", "Imports.jl"))
include(joinpath(CASE_ROOT, "SpectralFunctionCheckpointed.jl"))

const CHANNELS = ("zz", "pm", "mp")
const CHANNEL_WEIGHTS = Dict("zz" => 1.0, "pm" => 0.5, "mp" => 0.5)
const FORMAT_VERSION = 3
const MOMENT_CONVENTION = SpectralFunctionCheckpointed.MOMENT_CONVENTION

function parse_options(args)
    options = Dict{String,String}()
    index = 1
    while index <= length(args)
        flag = args[index]
        startswith(flag, "--") || error("Unexpected positional argument: $flag")
        index == length(args) && error("Missing value for $flag")
        options[flag] = args[index + 1]
        index += 2
    end
    return options
end

function required_option(options, name)
    value = get(options, name, "")
    isempty(value) && error("Required option: $name VALUE")
    return value
end

function task_table(nstates, L)
    tasks = NamedTuple[]
    task_id = 0
    for state_index in 1:nstates
        for site_index in 1:L
            for channel in CHANNELS
                task_id += 1
                push!(
                    tasks,
                    (
                        task_id=task_id,
                        state_index=state_index,
                        site_index=site_index,
                        channel=channel,
                    ),
                )
            end
        end
    end
    return tasks
end

function task_filename(output_dir, task)
    return joinpath(
        output_dir,
        @sprintf(
            "task_%06d_s%04d_i%05d_c%s.jld2",
            task.task_id,
            task.state_index,
            task.site_index,
            task.channel,
        ),
    )
end

function read_task_ids(filename)
    isfile(filename) || error("Task manifest is missing: $filename")
    task_ids = Int[]
    for line in eachline(filename)
        value = strip(line)
        isempty(value) && continue
        task_id = tryparse(Int, value)
        task_id === nothing && error("Invalid task ID in manifest $filename: $value")
        push!(task_ids, task_id)
    end
    isempty(task_ids) && error("Task manifest is empty: $filename")
    length(unique(task_ids)) == length(task_ids) ||
        error("Task manifest contains duplicate IDs: $filename")
    return task_ids
end

function main(args)
options = parse_options(args)
checkpoint_filename = abspath(required_option(options, "--checkpoint"))
output_dir = abspath(required_option(options, "--output"))
worker_rank = parse(Int, required_option(options, "--worker-rank"))
worker_count = parse(Int, required_option(options, "--worker-count"))
task_ids_filename = abspath(required_option(options, "--task-ids-file"))
resume = lowercase(get(options, "--resume", "false")) in ("1", "true", "yes")
backend = lowercase(get(options, "--backend", "full"))
backend in ("full", "streaming") ||
    error("Unsupported --backend $backend; choose full or streaming")
threading_mode = lowercase(get(options, "--threading-mode", "outer"))
threading_mode in ("outer", "blocksparse") ||
    error("Unsupported --threading-mode $threading_mode; choose outer or blocksparse")
backend == "streaming" ||
    error("The checkpointed single-task worker requires --backend streaming")
checkpoint_every = parse(Int, get(options, "--checkpoint-every", "1"))
heartbeat_every = parse(Int, get(options, "--heartbeat-every", "1"))
stop_after_value = parse(Int, get(options, "--stop-after-n", "0"))
stop_after_n = stop_after_value > 0 ? stop_after_value : nothing
stop_before_value = parse(Float64, get(options, "--stop-before-epoch", "0"))
stop_before_epoch = stop_before_value > 0 ? stop_before_value : nothing
checkpoint_every >= 1 || error("--checkpoint-every must be >= 1")
heartbeat_every >= 1 || error("--heartbeat-every must be >= 1")

0 <= worker_rank < worker_count ||
    error("Require 0 <= worker-rank < worker-count")
Threads.nthreads(:interactive) == 0 ||
    error("Use JULIA_NUM_THREADS=<threads>,0: interactive thread pool must be zero")
BLAS.set_num_threads(1)
if threading_mode == "blocksparse"
    ITensors.Strided.disable_threads()
    ITensors.enable_threaded_blocksparse()
end
strided_threads = ITensors.Strided.get_num_threads()
threaded_blocksparse = ITensors.using_threaded_blocksparse()
mkpath(output_dir)

start_time = DateTime(now())
start_wall = time()
checkpoint_hash = bytes2hex(open(sha256, checkpoint_filename))
checkpoint = load(checkpoint_filename)

required_keys = (
    "L",
    "i_imp",
    "J",
    "Jp",
    "cutoff",
    "ωmax",
    "try_doubleN",
    "ω_smearing",
    "seed",
    "sites",
    "H",
    "Emax",
    "En",
    "Sn",
    "Szn",
    "En_sym",
    "ψn_sym",
    "Sn_sym",
    "Szn_sym",
)
for key in required_keys
    haskey(checkpoint, key) || error("Checkpoint is missing required key: $key")
end

L = checkpoint["L"]
sites = checkpoint["sites"]
length(sites) == L ||
    error("Checkpoint inconsistency: L=$L but length(sites)=$(length(sites))")
ψn_sym = checkpoint["ψn_sym"]
nstates = length(ψn_sym)
nstates >= 1 || error("Checkpoint contains no symmetrized ground states")

all_tasks = task_table(nstates, L)
task_ids = read_task_ids(task_ids_filename)
all(1 <= task_id <= length(all_tasks) for task_id in task_ids) ||
    error("Task manifest contains an out-of-range task ID")
assigned_tasks = [all_tasks[task_id] for task_id in task_ids]

H = checkpoint["H"]
E0_sym = minimum(checkpoint["En_sym"])
cutoff = get(checkpoint, "spectral_cutoff", checkpoint["cutoff"])
ωmax = checkpoint["ωmax"]
Emax = checkpoint["Emax"]
try_doubleN = checkpoint["try_doubleN"]
ω_smearing = checkpoint["ω_smearing"]

# Build each operator once per site represented in this worker's batch.
assigned_sites = sort(unique(task.site_index for task in assigned_tasks))
Szi = Dict(site => Operators.Szi_op(site, sites) for site in assigned_sites)
Spi = Dict(site => Operators.Spi_op(site, sites) for site in assigned_sites)
Smi = Dict(site => Operators.Smi_op(site, sites) for site in assigned_sites)
setup_seconds = time() - start_wall

println(
    "rank=$worker_rank/$worker_count host=$(gethostname()) " *
    "threads=$(Threads.nthreads()) tasks=$(length(assigned_tasks)) " *
    "L=$L states=$nstates backend=$backend threading_mode=$threading_mode " *
    "blocksparse=$threaded_blocksparse strided_threads=$strided_threads",
)
println("task IDs = $([task.task_id for task in assigned_tasks])")
println("GS checkpoint = $checkpoint_filename")
flush(stdout)

write_lock = ReentrantLock()
completed = Atomic{Int}(0)
skipped = Atomic{Int}(0)
spectral_start = time()

function process_task(local_index)
    task = assigned_tasks[local_index]
    final_filename = task_filename(output_dir, task)
    progress_filename = joinpath(output_dir, "progress_" * basename(final_filename))
    recurrence_key = join(
        (
            "single-task-checkpoint-v2",
            checkpoint_hash,
            string(task.task_id),
            string(task.state_index),
            string(task.site_index),
            task.channel,
            backend,
            threading_mode,
            MOMENT_CONVENTION,
            string(cutoff),
            string(Emax),
            string(ωmax),
            string(try_doubleN),
            string(ω_smearing),
        ),
        "|",
    )

    if isfile(final_filename)
        resume || error("Task output already exists: $final_filename")
        existing = load(final_filename)
        get(existing, "format_version", 0) == FORMAT_VERSION ||
            error("Cannot resume incompatible task output: $final_filename")
        existing["checkpoint_hash"] == checkpoint_hash ||
            error("Cannot resume task from a different checkpoint: $final_filename")
        existing["task_id"] == task.task_id ||
            error("Cannot resume mismatched task output: $final_filename")
        existing["backend"] == backend ||
            error("Cannot resume task from a different backend: $final_filename")
        existing["threading_mode"] == threading_mode ||
            error("Cannot resume task from a different threading mode: $final_filename")
        get(existing, "moment_convention", "") == MOMENT_CONVENTION ||
            error("Cannot resume task from a different moment convention: $final_filename")
        atomic_add!(skipped, 1)
        return
    end

    if stop_before_epoch !== nothing && time() >= stop_before_epoch
        println("deadline reached before task=" * string(task.task_id) * "; leaving it checkpointed")
        flush(stdout)
        return
    end

    println(
        "starting task=$(task.task_id) state=$(task.state_index) " *
        "site=$(task.site_index) channel=$(task.channel)",
    )
    flush(stdout)
    task_start = time()

    site = task.site_index
    if task.channel == "zz"
        B, C = Szi[site], Szi[site]
    elseif task.channel == "pm"
        B, C = Spi[site], Smi[site]
    elseif task.channel == "mp"
        B, C = Smi[site], Spi[site]
    else
        error("Unsupported channel: $(task.channel)")
    end

    spectral_function = SpectralFunctionCheckpointed.spectral_function
    spectrum = try
        spectral_function(
        ψn_sym[task.state_index],
        H,
        B,
        C,
        E0_sym,
        ωmax,
        sites;
        cutoff=cutoff,
        Emax=Emax,
        try_doubleN=try_doubleN,
        ω_smearing=ω_smearing,
        checkpoint_filename=progress_filename,
        checkpoint_key=recurrence_key,
        checkpoint_every=checkpoint_every,
        heartbeat_every=heartbeat_every,
        stop_after_n=stop_after_n,
        stop_before_epoch=stop_before_epoch,
        )
    catch error
        if error isa SpectralFunctionCheckpointed.IntentionalCheckpointStop
            println("intentional validation stop completed_n=$(error.completed_n)")
            flush(stdout)
            return
        end
        rethrow()
    end
    all(isfinite, spectrum) ||
        error("Non-finite spectrum in task $(task.task_id)")
    omega = collect(range(0.0, ωmax, length=length(spectrum)))
    task_seconds = time() - task_start
    channel_weight = CHANNEL_WEIGHTS[task.channel]
    worker_threads = Threads.nthreads()
    finish_time = DateTime(now())
    temporary_filename = final_filename * ".tmp.$worker_rank.$(Threads.threadid())"

    lock(write_lock) do
        jldsave(
            temporary_filename;
            format_version=FORMAT_VERSION,
            backend,
            threading_mode,
            moment_convention=MOMENT_CONVENTION,
            threaded_blocksparse,
            strided_threads,
            checkpoint_filename,
            checkpoint_hash,
            L,
            nstates,
            task_id=task.task_id,
            state_index=task.state_index,
            site_index=task.site_index,
            channel=task.channel,
            channel_weight,
            omega,
            spectrum,
            worker_rank,
            worker_count,
            worker_threads,
            task_seconds,
            start_time,
            finish_time,
        )
        mv(temporary_filename, final_filename; force=false)
    end
    atomic_add!(completed, 1)
    println("completed task=$(task.task_id) seconds=$task_seconds")
    flush(stdout)
end

if threading_mode == "blocksparse"
    # Run outer tasks sequentially; each task uses the full Julia pool for
    # ITensor BlockSparse contractions. This avoids nested oversubscription.
    for local_index in eachindex(assigned_tasks)
        process_task(local_index)
    end
else
    Threads.@threads :static for local_index in eachindex(assigned_tasks)
        process_task(local_index)
    end
end

spectral_seconds = time() - spectral_start
worker_seconds = time() - start_wall
summary_filename = joinpath(output_dir, "worker_summary.txt")
open(summary_filename, "w") do io
    write(io, "format version = $FORMAT_VERSION\n")
    write(io, "backend = $backend\n")
    write(io, "threading mode = $threading_mode\n")
    write(io, "moment convention = $MOMENT_CONVENTION\n")
    write(io, "threaded blocksparse = $threaded_blocksparse\n")
    write(io, "strided threads = $strided_threads\n")
    write(io, "worker rank = $worker_rank\n")
    write(io, "worker count = $worker_count\n")
    write(io, "L = $L\n")
    write(io, "states = $nstates\n")
    write(io, "threads = $(Threads.nthreads())\n")
    write(io, "assigned task IDs = $([task.task_id for task in assigned_tasks])\n")
    write(io, "completed tasks = $(completed[])\n")
    write(io, "resumed tasks = $(skipped[])\n")
    write(io, "setup seconds = $setup_seconds\n")
    write(io, "spectral seconds = $spectral_seconds\n")
    write(io, "worker seconds = $worker_seconds\n")
    write(io, "checkpoint = $checkpoint_filename\n")
    write(io, "checkpoint sha256 = $checkpoint_hash\n")
    write(io, "start = $start_time\n")
    write(io, "finish = $(DateTime(now()))\n")
end

println(
    "worker $worker_rank complete: computed=$(completed[]) resumed=$(skipped[]) " *
    "seconds=$worker_seconds",
)
end

main(ARGS)
