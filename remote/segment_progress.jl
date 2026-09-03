using JLD2
using ITensors
using ITensorMPS

const CHECKPOINT_FORMAT_VERSION = 2
const MOMENT_CONVENTION = "chebyshev-2n-minus-1-v2"

function progress_iteration(filename, recurrence_target)
    status_filename = filename * ".status"
    if !isfile(status_filename)
        # v4 checkpoints predate the lightweight sidecar. completed_n=2 is a
        # conservative lower bound; workers validate the full file on resume.
        return 2
    end
    values = Dict{String,String}()
    for line in eachline(status_filename)
        fields = split(chomp(line), "="; limit=2)
        length(fields) == 2 || error("Invalid progress sidecar: $status_filename")
        values[fields[1]] = fields[2]
    end
    get(values, "checkpoint_format_version", "") == string(CHECKPOINT_FORMAT_VERSION) ||
        error("Checkpoint format mismatch: $status_filename")
    get(values, "moment_convention", "") == MOMENT_CONVENTION ||
        error("Moment convention mismatch: $status_filename")
    parse(Int, get(values, "N", "0")) == recurrence_target ||
        error("Recurrence target mismatch: $status_filename")
    completed_n = parse(Int, get(values, "completed_n", "0"))
    2 <= completed_n <= recurrence_target ||
        error("Invalid recurrence progress: $status_filename")
    return completed_n
end

function main(args)
length(args) == 2 || error("Usage: segment_progress.jl GS_CHECKPOINT WORKER_ROOT")
checkpoint_filename = abspath(args[1])
worker_root = abspath(args[2])
L, nstates, Emax, minimum_energy, omega_smearing, try_doubleN =
    jldopen(checkpoint_filename, "r") do checkpoint
        (
            checkpoint["L"],
            length(checkpoint["En_sym"]),
            checkpoint["Emax"],
            minimum(checkpoint["En_sym"]),
            checkpoint["ω_smearing"],
            checkpoint["try_doubleN"],
        )
    end
total_tasks = 3 * L * nstates

Wstar = Emax - minimum_energy
raw_N = ceil(Int, Wstar / omega_smearing)
recurrence_target = try_doubleN ? round(Int, raw_N / 2) : raw_N

completed = Set{Int}()
progress = Dict{Int,Int}()
task_pattern = r"^task_(\d{6})_.*\.jld2$"
progress_pattern = r"^progress_task_(\d{6})_.*\.jld2$"
for (root, _, files) in walkdir(worker_root)
    for filename in files
        if (matched = match(task_pattern, filename)) !== nothing
            task_id = parse(Int, matched.captures[1])
            1 <= task_id <= total_tasks || error("Invalid completed task ID: $task_id")
            filesize(joinpath(root, filename)) > 0 || error("Empty task result for task $task_id")
            task_id in completed && error("Duplicate completed task ID: $task_id")
            push!(completed, task_id)
        elseif (matched = match(progress_pattern, filename)) !== nothing
            task_id = parse(Int, matched.captures[1])
            1 <= task_id <= total_tasks || error("Invalid progress task ID: $task_id")
            path = joinpath(root, filename)
            filesize(path) > 0 || error("Empty recurrence checkpoint for task $task_id")
            completed_n = progress_iteration(path, recurrence_target)
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
