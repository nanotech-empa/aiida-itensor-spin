"""The official --inspect protocol must work on the local Code without workers."""
import hashlib
import json
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path

import pytest


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    loader = SourceFileLoader("local_inventory_driver", str(root / "remote/itensor-spin-segment-local"))
    spec = spec_from_loader(loader.name, loader)
    driver = module_from_spec(spec)
    loader.exec_module(driver)
    julia = tmp_path / "local Julia"
    julia.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
with Path("helper-calls.jsonl").open("a") as stream:
    stream.write(json.dumps(sys.argv[1:])+"\\n")
if any(x.endswith("checkpoint_info.jl") for x in sys.argv):
    print("3 1 9")
elif any(x.endswith("segment_progress.jl") for x in sys.argv):
    done = int(os.environ["ITENSOR_FIXTURE_COMPLETED"])
    print(f"9 {done} {13*done} 117 13")
else:
    raise SystemExit("Inventory must not launch a worker or collector")
''')
    julia.chmod(0o755)
    monkeypatch.setattr(driver, "DEFAULT_JULIA", julia)
    monkeypatch.delenv("ITENSOR_SPIN_JULIA", raising=False)
    monkeypatch.setenv("ITENSOR_SPIN_CASE_ROOT", str(root / "remote/engine"))
    monkeypatch.setenv("ITENSOR_FIXTURE_COMPLETED", "0")
    monkeypatch.setattr(driver, "run_local_workers", lambda *a: pytest.fail("Inventory launched workers"))
    source = tmp_path / "preserved"
    workers = source / "workers"
    workers.mkdir(parents=True)
    (workers / "preserved-task.jld2").write_bytes(b"unchanged fixture checkpoint")
    job = tmp_path / "inventory-job"
    job.mkdir()
    (job / "resume_state").symlink_to(source, target_is_directory=True)
    (job / "checkpoint.jld2").write_bytes(b"fake checkpoint; helpers use the fake executable")
    monkeypatch.chdir(job)
    config = {"format_version": 1, "segment_index": 1,
              "parameters": {"backend": "streaming", "threading_mode": "blocksparse",
                             "chebyshev_convention": "chebyshev-2n-minus-1-v2"},
              "ground_state": "checkpoint.jld2", "resume_state": "resume_state",
              "status_file": "inventory_status.json", "summary_file": "inventory_summary.json"}
    (job / "inventory.json").write_text(json.dumps(config))
    monkeypatch.setattr(driver.sys, "argv", [loader.path, "--inspect", "inventory.json"])
    return driver, job, source, config


@pytest.mark.parametrize("completed", [0, 3])
def test_actual_local_inspect_preserves_source_and_only_reads_inventory(inventory, monkeypatch, completed):
    driver, job, source, config = inventory
    monkeypatch.setenv("ITENSOR_FIXTURE_COMPLETED", str(completed))
    def state():
        return {str(p.relative_to(source)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                for p in source.rglob("*") if p.is_file()}
    before = state()
    assert driver.main() == 0
    assert state() == before
    status = json.loads((job / "inventory_status.json").read_text())
    assert status["completed_tasks"] == completed
    assert status["total_tasks"] == 9 and status["progress_units_total"] == 117
    assert status["progress_units_completed"] == 13*completed
    assert status["segment_index"] == 1
    summary = json.loads((job / "inventory_summary.json").read_text())
    assert summary["phase"] == "inventory"
    assert summary["execution_environment"] == "localhost-direct"
    calls = [json.loads(x) for x in (job / "helper-calls.jsonl").read_text().splitlines()]
    assert len(calls) == 2
    assert calls[1][-1] == "resume_state/workers"
    assert not (job / "task_manifest").exists()
    assert not (job / "state").exists()


def test_local_inspect_rejects_invalid_format(inventory):
    driver, job, source, config = inventory
    config["format_version"] = 99
    (job / "inventory.json").write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="inventory input format"):
        driver.main()
    assert not (job / "helper-calls.jsonl").exists()


def test_local_inspect_rejects_missing_source_workers(inventory):
    driver, job, source, config = inventory
    config["resume_state"] = "absent"
    (job / "inventory.json").write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="worker state is unreadable"):
        driver.main()
    assert not (job / "helper-calls.jsonl").exists()
