"""Exercise the real local driver and rank shell without running a scientific solver."""

import json
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path

import pytest


@pytest.mark.parametrize("configured", [False, True], ids=["default", "explicit"])
def test_selected_local_julia_reaches_the_actual_rank_wrapper(tmp_path, monkeypatch, configured):
    root = Path(__file__).resolve().parents[1]
    loader = SourceFileLoader("local_spectral_driver", str(root / "remote/itensor-spin-segment-local"))
    spec = spec_from_loader(loader.name, loader)
    driver = module_from_spec(spec)
    loader.exec_module(driver)

    # A path containing spaces also exercises actual shell argument quoting.
    julia = tmp_path / "local runtime" / "julia"
    julia.parent.mkdir()
    julia.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if any(x.endswith("checkpoint_info.jl") for x in args):
    print("3 1 9")
elif any(x.endswith("segment_progress.jl") for x in args):
    print("9 9 117 117 13" if Path("launched.json").exists() else "9 0 0 117 13")
elif any(x.endswith("chi_single_task_checkpoint.jl") for x in args):
    Path("launched.json").write_text(json.dumps({
        "executable": sys.argv[0], "args": args,
        "environment": {k: os.environ.get(k) for k in [
            "ITENSOR_SPIN_JULIA", "JULIA_NUM_THREADS", "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "SLURM_PROCID",
            "SLURM_NTASKS", "SLURM_CPUS_PER_TASK"]}}))
elif any(x.endswith("collect_tasks.jl") for x in args):
    output = Path(args[args.index("--output") + 1])
    output.mkdir(parents=True, exist_ok=True)
    for name in ["spectrum.tsv", "collection_report.txt", "task_timings.tsv"]:
        (output / name).write_text("fixture result\\n")
else:
    raise SystemExit("Unexpected fixture command")
''')
    julia.chmod(0o755)
    monkeypatch.setattr(driver, "DEFAULT_JULIA", julia)
    monkeypatch.setenv("ITENSOR_SPIN_CASE_ROOT", str(root / "remote/engine"))
    if configured:
        monkeypatch.setenv("ITENSOR_SPIN_JULIA", str(julia))
    else:
        monkeypatch.delenv("ITENSOR_SPIN_JULIA", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "checkpoint.jld2").write_bytes(b"fixture; helpers use the fake executable")
    config = {
        "format_version": 1,
        "ground_state": "checkpoint.jld2",
        "layout": {"nodes": 1, "processes_per_node": 1, "threads_per_process": 1,
                   "wallclock_seconds": 1200, "internal_stop_seconds": 1128},
        "parameters": {"backend": "streaming", "threading_mode": "blocksparse",
                       "chebyshev_convention": "chebyshev-2n-minus-1-v2",
                       "checkpoint_every": 10, "heartbeat_every": 1},
        "workers": 1, "segment_index": 0, "output_state": "state",
        "status_file": "segment_status.json", "summary_file": "segment_summary.json",
    }
    (tmp_path / "segment.json").write_text(json.dumps(config))
    monkeypatch.setattr(driver.sys, "argv", [loader.path, "segment.json"])

    assert driver.main() == 0

    launched = json.loads((tmp_path / "launched.json").read_text())
    assert launched["executable"] == str(julia)
    assert launched["environment"] == {
        "ITENSOR_SPIN_JULIA": str(julia), "JULIA_NUM_THREADS": "1,0",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "SLURM_PROCID": "0", "SLURM_NTASKS": "1", "SLURM_CPUS_PER_TASK": "1",
    }
    assert launched["args"][launched["args"].index("--threading-mode") + 1] == "blocksparse"
    assert json.loads((tmp_path / "segment_summary.json").read_text())["phase"] == "finalized"
    assert json.loads((tmp_path / "segment_status.json").read_text())["completed_tasks"] == 9
    assert ("ITENSOR_SPIN_JULIA" in driver.os.environ) is configured
