# Julia-only driver for cluster login environments without recent Python.
using TOML
function main(args)
    iseven(length(args)) || error("Options require --name value pairs")
    o=Dict(args[i]=>args[i+1] for i in 1:2:length(args))
    root=dirname(@__DIR__)
    script=joinpath(root,"validation","benchmark_kernels.jl")
    julia=get(o,"--julia",first(Base.julia_cmd().exec))
    project=get(o,"--project",joinpath(root,"remote","engine","ITensor_env"))
    fixture=abspath(o["--fixture"])
    output=abspath(o["--output"])
    counts=parse.(Int,split(get(o,"--threads","1,2,4,8"),','))
    !isempty(counts) && all(n->n>=1,counts) || error("Thread counts must be positive")
    repeats=get(o,"--repeats","3")
    mkpath(output)
    configs=[("serial",1,1)]
    append!(configs,[("blocks",n,1) for n in counts if n>1])
    append!(configs,[("blas",1,n) for n in counts if n>1])
    records=Dict{String,Any}[]
    reference=""
    baseline=0.0
    for (mode,jthreads,bthreads) in configs
        label="$(mode)_j$(jthreads)_b$(bthreads)"
        target=joinpath(output,label)
        println("RUN $label"); flush(stdout)
        command=Cmd([julia,"--startup-file=no","--threads=$(jthreads),0","--project=$project",
            script,"--action","run","--fixture",fixture,"--mode",mode,
            "--blas-threads",string(bthreads),"--repeats",repeats,"--output",target])
        open(target*".log","w") do io; run(pipeline(command,stdout=io,stderr=io)); end
        record=TOML.parsefile(target*".toml")
        if isempty(records)
            reference=target*".jld2"
            baseline=record["median"]["kernel_seconds"]
            record["reference_check"]=Dict("relative_vector_error"=>0.0,"moment_max_abs_error"=>0.0)
        else
            command=Cmd([julia,"--startup-file=no","--threads=1,0","--project=$project",script,
                "--action","compare","--reference",reference,"--candidate",target*".jld2",
                "--output",target*".comparison.toml"])
            open(target*".comparison.log","w") do io; run(pipeline(command,stdout=io,stderr=io)); end
            record["reference_check"]=TOML.parsefile(target*".comparison.toml")
        end
        record["speedup_vs_serial"]=baseline/record["median"]["kernel_seconds"]
        push!(records,record)
        open(joinpath(output,"summary.toml"),"w") do io
            TOML.print(io,Dict("configurations"=>records);sorted=true)
        end
        println("DONE $label kernel=$(record["median"]["kernel_seconds"]) speedup=$(record["speedup_vs_serial"]) vector_difference=$(record["reference_check"]["relative_vector_error"])")
        flush(stdout)
    end
end
main(ARGS)
