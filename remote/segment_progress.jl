using JLD2
using ITensors
using ITensorMPS

const TASK_FORMAT_VERSION = 3
const CHECKPOINT_FORMAT_VERSION = 2
const MOMENT_CONVENTION = "chebyshev-2n-minus-1-v2"

function main(args)
length(args) == 2 || error("Usage: segment_progress.jl GS_CHECKPOINT WORKER_ROOT")
checkpoint_filename = abspath(args[1])
worker_root = abspath(args[2])
checkpoint = load(checkpoint_filename)
L = checkpoint["L"]
nstates = length(checkpoint["ψn_sym"])
total_tasks = 3 * L * nstates

Wstar = checkpoint["Emax"] - minimum(checkpoint["En_sym"])
raw_N = ceil(Int, Wstar / checkpoint["ω_smearing"])
recurrence_target = checkpoint["try_doubleN"] ? round(Int, raw_N / 2) : raw_N

completed = Set{Int}()
progress = Dict{Int,Int}()
task_pattern = r"^task_(\d{6})_.*\.jld2$"
progress_pattern = r"^progress_task_(\d{6})_.*\.jld2$"
for (root, _, files) in walkdir(worker_root)
    for filename in files
        if (matched = match(task_pattern, filename)) !== nothing
            task_id = parse(Int, matched.captures[1])
            1 <= task_id <= total_tasks || error("Invalid completed task ID: $task_id")
            data = load(joinpath(root, filename))
            data["format_version"] == TASK_FORMAT_VERSION ||
                error("Task format mismatch for task $task_id")
            get(data, "moment_convention", "") == MOMENT_CONVENTION ||
                error("Moment convention mismatch for task $task_id")
            task_id in completed && error("Duplicate completed task ID: $task_id")
            push!(completed, task_id)
        elseif (matched = match(progress_pattern, filename)) !== nothing
            task_id = parse(Int, matched.captures[1])
            1 <= task_id <= total_tasks || error("Invalid progress task ID: $task_id")
            data = load(joinpath(root, filename))
            get(data, "checkpoint_format_version", 0) == CHECKPOINT_FORMAT_VERSION ||
                error("Checkpoint format mismatch for task $task_id")
            get(data, "moment_convention", "") == MOMENT_CONVENTION ||
                error("Moment convention mismatch for task $task_id")
            data["N"] == recurrence_target ||
                error("Recurrence target mismatch for task $task_id")
            completed_n = data["completed_n"]
            2 <= completed_n <= recurrence_target || error("Invalid progress for task $task_id")
            haskey(progress, task_id) && error("Duplicate progress task ID: $task_id")
            progress[task_id] = completed_n
        end
    end
end

progress_units = length(completed) * recurrence_target
for (task_id, completed_n) in progress
    task_id in completed || (progress_units += completed_n)
end
progress_units_total = total_tasks * recurrence_target
println(
    "$total_tasks $(length(completed)) $progress_units " *
    "$progress_units_total $recurrence_target",
)

end

main(ARGS)
