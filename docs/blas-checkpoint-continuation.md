# Explicit BLAS checkpoint continuation

The legacy default remains at two allocated CPU cores and block-sparse Julia
threading. For a separately registered runtime declaring
`supports_blas_execution_policy=True`, spectral parameters can request
`kernel_threading_mode="blas"` and `blas_threads=8`. Resource limits must also
specify `threads_per_process=8` and `max_threads_per_process >= 8`.

The explicit resource value survives normal replanning and OOM density backoff.
The worker uses one Julia thread, eight BLAS threads, one Strided thread and
unthreaded BlockSparse contractions, as in the repeated Daint benchmark.
The CalcJob rejects a runtime lacking the capability flag and rejects a mismatch
between BLAS threads and allocated cores. Invalid or oversubscribed explicit
resource limits are rejected.

For continuation, `threading_mode="blocksparse"` retains its historical
checkpoint-key role. It does not describe the overridden kernel execution.
`kernel_threading_mode`, actual Julia/BLAS thread counts, runtime source checksum,
and the new Code identify execution explicitly. New recurrence checkpoints and
final task outputs record the actual execution policy; inherited completed task
files retain their historical metadata. Scientific inputs, saved moments,
recurrence keys, cutoff and bond cap are preserved. No old node or checkpoint is
rewritten. Floating-point roundoff can differ between thread policies.

`validation/test_blas_continuation.jl` advances a checkpoint with the legacy
metadata format under BLAS-8 for both dense and quantum-number-conserving MPS,
checks that saved moments are preserved, and compares the final moments against
a direct legacy calculation. Run it on a compute node with one Julia thread,
eight allocated cores, and `ITENSOR_SPIN_CASE_ROOT` pointing at the engine.
The test is small and does not establish production throughput or convergence.

For the October 16 L50 continuation, twelve remaining tasks are assigned to
four nodes, three workers/node, eight CPU cores/worker and 256 GiB/node.
This is deliberately less dense than the historical six-worker benchmark.
The existing 288 completed tasks and 212239/216000 saved recurrence units are
reused. The new workflow retains the original remaining 1160-node-hour ceiling;
the original full workflow is superseded, not reported as scientifically complete.
