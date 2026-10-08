using JLD2, Printf

# Read scalar metadata and the tiny moments dataset only. Never deserialize
# the MPS vectors or write to the preserved calculation directory.
length(ARGS) == 2 || error("Usage: export_partial_moments.jl CHECKPOINT_ROOT OUTPUT")
checkpoint_root, output = ARGS
open(output, "w") do io
    println(io, "task\tstate\tsite\tchannel\tcompleted_vectors\ttarget_vectors\torder\tmoment")
    count = 0
    for (root, dirs, files) in walkdir(joinpath(checkpoint_root, "state", "workers"))
        for filename in sort(files)
            m = match(r"^progress_task_(\d{6})_s(\d{4})_i(\d{5})_c(zz|pm|mp)\.jld2$", filename)
            m === nothing && continue
            final_name = replace(filename, "progress_" => ""; count=1)
            isfile(joinpath(root, final_name)) && continue
            jldopen(joinpath(root, filename), "r") do f
                n, target = Int(f["completed_n"]), Int(f["N"])
                f["doubleN"] == true || error("Expected doubled-moment convention")
                f["energy_truncation"] == false || error("Expected full-bandwidth state")
                f["moment_convention"] == "chebyshev-2n-minus-1-v2" || error("Unexpected convention")
                moments = f["moments"]
                used = 2n - 1
                for k in 1:used
                    value = Float64(moments[k])
                    isfinite(value) || error("Nonfinite moment in $filename")
                    @printf(io, "%d\t%d\t%d\t%s\t%d\t%d\t%d\t%.17g\n",
                        parse(Int,m[1]), parse(Int,m[2]), parse(Int,m[3]), m[4], n, target, k-1, value)
                end
                println("EXPORTED task=$(parse(Int,m[1])) vectors=$n/$target moments=$used")
                flush(stdout)
            end
            count += 1
        end
    end
    count == 18 || error("Expected 18 unfinished tasks, got $count")
end
