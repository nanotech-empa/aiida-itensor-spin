using JLD2, Printf

length(ARGS) == 2 || error("Usage: export_all_moments.jl CHECKPOINT_ROOT OUTPUT")
checkpoint_root, output = ARGS
const NTASKS = 909
const TARGET = 1474
const NMOMENTS = 2TARGET - 1
moments = zeros(Float64, NMOMENTS, NTASKS)
completed_vectors = zeros(Int, NTASKS)
state_indices = zeros(Int, NTASKS)
site_indices = zeros(Int, NTASKS)
channel_indices = zeros(Int, NTASKS)
final_task_exists = zeros(Int, NTASKS)
source_files = fill("", NTASKS)
seen = Set{Int}()
channel_ids = Dict("zz" => 1, "pm" => 2, "mp" => 3)

# Access only scalar/array datasets, never any saved MPS vector.
for (root, dirs, files) in walkdir(joinpath(checkpoint_root, "state", "workers"))
    for filename in sort(files)
        m = match(r"^progress_task_(\d{6})_s(\d{4})_i(\d{5})_c(zz|pm|mp)\.jld2$", filename)
        m === nothing && continue
        task, state, site = parse.(Int, m.captures[1:3])
        channel = channel_ids[m[4]]
        task == ((state-1)*101 + site-1)*3 + channel || error("Task identity mismatch")
        1 <= task <= NTASKS || error("Unexpected task")
        task in seen && error("Duplicate task $task")
        path = joinpath(root, filename)
        jldopen(path, "r") do f
            n = Int(f["completed_n"])
            Int(f["N"]) == TARGET || error("Unexpected recurrence target in $path")
            f["doubleN"] == true || error("Expected doubled-moment convention")
            f["energy_truncation"] == false || error("Expected full-bandwidth state")
            f["moment_convention"] == "chebyshev-2n-minus-1-v2" || error("Convention mismatch")
            raw = Float64.(f["moments"])
            length(raw) == NMOMENTS || error("Moment allocation mismatch")
            used = 2n - 1
            2 <= n <= TARGET || error("Invalid vector count")
            all(isfinite, raw[1:used]) || error("Nonfinite moment")
            moments[1:used, task] = raw[1:used]
            completed_vectors[task] = n
        end
        state_indices[task], site_indices[task], channel_indices[task] = state, site, channel
        final_task_exists[task] = isfile(joinpath(root, replace(filename, "progress_"=>""; count=1))) ? 1 : 0
        final_task_exists[task] == 1 && completed_vectors[task] != TARGET && error("Final task has partial checkpoint")
        source_files[task] = path
        push!(seen, task)
        length(seen) % 100 == 0 && (println("Read $(length(seen))/$NTASKS moments datasets"); flush(stdout))
    end
end
length(seen) == NTASKS || error("Incomplete coverage: $(length(seen))/$NTASKS")
sum(final_task_exists) == 891 || error("Expected 891 final tasks")
sum(completed_vectors) == 1322972 || error("Progress inventory mismatch")
metadata = jldopen(joinpath(checkpoint_root, "gs_checkpoint.jld2"), "r") do f
    (; state_energies_meV=Float64.(f["En_sym"]), highest_energy_meV=Float64(f["Emax"]),
       temperature_K=Float64(f["T"]), original_smearing_parameter_meV=Float64(f["ω_smearing"]))
end
jldsave(output; moments, completed_vectors, state_indices, site_indices, channel_indices,
    final_task_exists, source_files, metadata...,
    source_segment_uuid="4b4117de-7c29-4a5b-a1fe-78f438c854eb",
    convention="chebyshev-2n-minus-1-v2")
println("Exported $NTASKS tasks; 891 complete and 18 partial; all preserved inputs read-only.")
