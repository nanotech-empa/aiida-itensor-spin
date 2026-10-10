# BLAS confirmation launch and preliminary checks

Confirmation job 5019606 was submitted on October 10, 2026 and verified RUNNING on one dedicated Daint node. Source code commit: 558e1df497605d1a8fe8d15695818ef18dfc5955. Isolated deployment: /ritom/scratch/cscs/cpi/goncalo/kernel_confirm_20261010_558e1df.

The code checksum manifest and hashes of both existing fixtures passed. The concurrent synthetic smoke passed: six workers executed two measured repetitions each; all six CPU-affinity sets were disjoint, moments matched the isolated smoke, and the sampled full-vector comparison reported zero residual. These subsecond smoke timings are not production-performance evidence. The archived TOML files document the successful coordinator checks.

The job then started the full physical BLAS8 warmup. Physical stages: isolated BLAS8 with full warmup and three identical-input repetitions; six simultaneous BLAS8 replicas with full warmup and two repetitions each; existing block2 reference with full warmup and three repetitions. Full-vector comparisons use BLAS8, as recorded in their metadata, with explicit MPS subtraction cutoff 1e-14. All six workers receive moment comparisons; workers 1 and 6 receive full-vector checks.

Resource request: one normal node, six tasks x eight CPU cores, 384 GB job memory, 64 GB per step, eight-hour maximum. Six identical replicas measure contention and are not the production channel/state mixture. Current production L50 host headers independently show six active ranks on each of its two nodes, despite the allocation ceiling of ten processes per node.

No physical confirmation result was available when this snapshot was created. Retrieve isolated and concurrent timing files, comparison residuals and Slurm accounting before drawing a production conclusion. Deployed engines and existing AiiDA workchains were not modified.
