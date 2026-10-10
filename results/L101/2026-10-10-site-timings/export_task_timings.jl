using JLD2, TOML, Dates
length(ARGS)==2 || error("Usage: export_task_timings.jl TASK_DIRECTORY OUTPUT")
rows=Dict{String,Any}[]
for filename in sort(readdir(ARGS[1]))
    startswith(filename,"task_") && endswith(filename,".jld2") || continue
    path=joinpath(ARGS[1],filename)
    jldopen(path,"r") do f
        row=Dict{String,Any}("filename"=>filename)
        for key in ("task_id","state_index","site_index","channel","task_seconds","worker_threads","threading_mode","worker_rank","checkpoint_hash","L")
            row[key]=f[key]
        end
        row["start_time"]=string(f["start_time"])
        row["finish_time"]=string(f["finish_time"])
        push!(rows,row)
    end
end
length(rows)==36 || error("Expected 4 sites x 3 states x 3 channels")
length(Set(row["checkpoint_hash"] for row in rows))==1 || error("Mixed ground states")
open(ARGS[2],"w") do io; TOML.print(io,Dict("tasks"=>rows);sorted=true); end
for row in rows
    println("site=",row["site_index"]," state=",row["state_index"]," channel=",row["channel"]," seconds=",row["task_seconds"]," start=",row["start_time"]," finish=",row["finish_time"])
end
