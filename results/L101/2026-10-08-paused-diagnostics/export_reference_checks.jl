using JLD2, SHA
length(ARGS) == 3 || error("Usage: export_reference_checks.jl EXPORT CHECKPOINT_ROOT OUTPUT")
data = load(ARGS[1])
source_files = data["source_files"]
expected_hash = bytes2hex(open(sha256, joinpath(ARGS[2], "gs_checkpoint.jld2")))
cutoffs = zeros(Float64, 909)
reference_task_ids = [1, 148, 150, 304, 607, 909]
reference_omega_meV = Float64[]
reference_spectra = Matrix{Float64}(undef, 0, 0)
for task in 1:909
    path = source_files[task]
    jldopen(path, "r") do f
        key = split(f["checkpoint_key"], '|')
        key[2] == expected_hash || error("Ground-state identity mismatch in task $task")
        parse(Int, key[3]) == task || error("Checkpoint task mismatch")
        cutoffs[task] = Float64(f["cutoff"])
    end
    column = findfirst(==(task), reference_task_ids)
    if column !== nothing
        final_path = joinpath(dirname(path), replace(basename(path), "progress_"=>""; count=1))
        jldopen(final_path, "r") do f
            omega = Float64.(f["omega"])
            f["checkpoint_hash"] == expected_hash || error("Final task ground-state mismatch")
            if isempty(reference_omega_meV)
                global reference_omega_meV = omega
                global reference_spectra = zeros(length(omega), length(reference_task_ids))
            end
            omega == reference_omega_meV || error("Grid mismatch")
            reference_spectra[:, column] = Float64.(f["spectrum"])
        end
    end
end
jldsave(ARGS[3]; cutoffs, reference_task_ids, reference_omega_meV, reference_spectra,
    ground_state_sha256=expected_hash)
println("All 909 ground-state identities verified; cutoffs=", sort(unique(cutoffs)))
