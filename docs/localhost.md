# Localhost setup (Linux ARM64 AiiDAlab container)

This setup runs the same AiiDA CalcJobs and WorkChains as Daint through the
existing `localhost` computer. Only the executable launchers differ: the local
segment launcher starts independent Julia processes directly instead of using
Slurm `srun`.

## Validated environment

- Linux `aarch64` container on Apple Silicon
- Python 3.12.11
- AiiDA 2.8.0
- Julia 1.12.6
- ITensors 0.9.30
- ITensorMPS 0.4.1
- JLD2 0.6.5

The pinned Julia manifest was created with Julia 1.12.6. The official Linux
AArch64 archive is:

```text
https://julialang-s3.julialang.org/bin/linux/aarch64/1.12/julia-1.12.6-linux-aarch64.tar.gz
SHA-256 029b93b857bd0ffd627f9a8580d3bbaa63daf008d7b7aed02fbceb8fd57c4899
```

Install it under a stable user-owned prefix such as
`/home/jovyan/opt/julia-1.12.6`. Do not reuse an x86_64 Julia binary in an
ARM64 container.

The local launcher uses `ITENSOR_SPIN_JULIA` when configured, otherwise the
prefix above. It forwards the same executable to the spectral rank workers,
so checkpoint inspection and computation use one Julia installation. Set this
variable in the Code environment when installing Julia in another prefix;
no Daint executable is needed for localhost execution.

The same local Code supports `--inspect inventory.json`, used by the official
inventory-first restart workflow. This mode reads the ground-state checkpoint
and preserved worker state; it does not launch spectral workers or edit the
source folder. New spectral work belongs to the following segment.

## Prepare the Python and Julia environments

From the repository root:

```console
python -m pip install --user -e .
JULIA_DEPOT_PATH=/home/jovyan/.julia-itensor-spin-localhost \
  /home/jovyan/opt/julia-1.12.6/bin/julia \
  --startup-file=no --project=remote/engine/ITensor_env \
  -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
python -m pip check
```

The Julia depot is deliberately separate from other projects.

## Register the AiiDA Codes

The AiiDA profile must already contain a `localhost` computer using
`core.local` and `core.direct`. Then run:

```console
verdi run scripts/setup_localhost.py
verdi daemon restart
verdi code test itensor-spin-gs-0.4.0-local@localhost
verdi code test itensor-spin-segment-0.4.0-local@localhost
```

The setup is idempotent. If a code label already exists with different
executable or environment settings, it stops instead of modifying a Code that
may already have provenance.

## Submit tests

A very small end-to-end workflow:

```console
verdi run examples/submit_localhost.py -- \
  --length 3 --omega-max-meV 80 --max-processes 2 --max-threads 2 \
  --wallclock-seconds 1800 --label "ITensor localhost L3 smoke"
```

The L=23 comparison case:

```console
verdi run examples/submit_localhost.py -- \
  --length 23 --impurity-site 12 \
  --max-processes 4 --max-threads 2 \
  --wallclock-seconds 10800 --label "ITensor localhost L23 gold parameters"
```

The L=23 default physics and numerical parameters are J=18.4, JL=16.7,
JR=51.1 meV, DMRG cutoff 1e-8, Chebyshev cutoff 1e-6, maximum frequency
100 meV, smearing 1.5 meV, and seed 20260724. They are explicit AiiDA inputs,
not hidden launcher settings.

A direct-scheduler container represents one node. Keep `max_nodes=1` and do
not request more than the CPU affinity visible inside the container. The
submission script checks both GS threads and
`max_processes * max_threads` before submitting.

Use `verdi process show UUID` and `verdi process report UUID` to inspect a
run. The completed WorkChain returns the GS checkpoint and summary, final
status, resource plan, spectrum, collection report, and task timings.

## Validated L=23 result

The command above completed through the standard AiiDA WorkChain on
18 September 2026. The GS took 1 min 22 s and the single spectral segment took
40 min 23 s; total WorkChain elapsed time was 41 min 48 s. The segment used four
processes with two threads each and completed 69/69 tasks without continuation.

Against Gonçalo Catarina's independently generated reference, the relative
spectral L2 difference was 0.1005701%, the largest per-site relative integrated-
weight difference was 0.0075104%, and all 23 sampled peak positions matched.
No acceptance tolerance was specified in advance. The runs did not share a GS
checkpoint and the reference seed is unknown, so this is an independent
physics reproduction rather than an exact implementation-equivalence test.
The retained data and durable AiiDA UUIDs are documented in
[`results/L23/README.md`](../results/L23/README.md).

## Deployment on another AiiDAlab instance

The portable pieces are the Python package, pinned Julia project, local
launchers, and idempotent setup script. Instance deployment therefore consists
of installing the package and native Julia runtime, instantiating the manifest,
ensuring the standard localhost computer exists, and running
`setup_localhost.py`.

A later `aiidalab-empa-setup` integration should call these same checks and
registration logic. It should not duplicate the scientific engine or create
unversioned AiiDA Codes.
