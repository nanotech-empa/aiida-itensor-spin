module KernelThreading
using ITensors, LinearAlgebra
function configure_kernel_threading(threading_mode; environment=ENV)
kernel_threading_mode = get(environment, "ITENSOR_SPIN_KERNEL_THREADING_MODE", "legacy")
blas_threads = parse(Int, get(environment, "ITENSOR_SPIN_BLAS_THREADS", "1"))
if kernel_threading_mode == "blas"
    Threads.nthreads() == 1 || error("BLAS execution requires one Julia thread")
    blas_threads == parse(Int, environment["SLURM_CPUS_PER_TASK"]) || error("BLAS threads must match allocated cores")
    blas_threads >= 1 || error("BLAS threads must be positive")
    ITensors.Strided.set_num_threads(1)
    BLAS.set_num_threads(blas_threads)
    ITensors.enable_threaded_blocksparse(false)
elseif kernel_threading_mode == "legacy"
    BLAS.set_num_threads(1)
    if threading_mode == "blocksparse"
        ITensors.Strided.disable_threads()
        ITensors.enable_threaded_blocksparse()
    end
else
    error("Unsupported kernel_threading_mode=$kernel_threading_mode")
end
return kernel_threading_mode
end
end
