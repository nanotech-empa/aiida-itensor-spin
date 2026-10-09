"""Run independent fresh Julia processes sequentially, with identical input.

No AiiDA submission, deployment changes, package installation or production
checkpoint writes. Default grid covers serial, block and BLAS threading.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tomllib

parser=argparse.ArgumentParser()
parser.add_argument("--julia",default="julia")
parser.add_argument("--project",type=Path)
parser.add_argument("--depot",type=Path)
parser.add_argument("--fixture",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--threads",default="1,2,4,8")
parser.add_argument("--repeats",type=int,default=3)
parser.add_argument("--resume",action="store_true",help="Reuse completed measurements after validating input and thread settings")
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
script=root/"validation/benchmark_kernels.jl"
project=(args.project or root/"remote/engine/ITensor_env").resolve()
args.output.mkdir(parents=True,exist_ok=True)
env=os.environ.copy()
if args.depot:
    env["JULIA_DEPOT_PATH"]=str(args.depot.resolve())
counts=[int(n) for n in args.threads.split(",")]
if not counts or any(n<1 for n in counts):
    parser.error("Thread counts must be positive")
configs=[("serial",1,1)]+[("blocks",n,1) for n in counts if n>1]+[("blas",1,n) for n in counts if n>1]
records=[]
fixture_hash=hashlib.sha256(args.fixture.read_bytes()).hexdigest()
script_hash=hashlib.sha256(script.read_bytes()).hexdigest()
for mode,jthreads,bthreads in configs:
    label=f"{mode}_j{jthreads}_b{bthreads}"
    target=args.output/label
    command=[args.julia,"--startup-file=no",f"--threads={jthreads},0",f"--project={project}",str(script),
        "--action","run","--fixture",str(args.fixture.resolve()),"--mode",mode,
        "--blas-threads",str(bthreads),"--repeats",str(args.repeats),"--output",str(target.resolve())]
    if args.resume and target.with_suffix(".toml").exists():
        print(f"REUSE {label}",flush=True)
    else:
        print(f"RUN {label}",flush=True)
        with target.with_suffix(".log").open("w") as handle:
            subprocess.run(command,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
    record=tomllib.loads(target.with_suffix(".toml").read_text())
    expected={"fixture_sha256":fixture_hash,"mode":mode,"julia_threads":jthreads,
              "blas_threads":bthreads,"repeats":args.repeats}
    if any(record.get(key)!=value for key,value in expected.items()):
        raise RuntimeError(f"Saved result does not match requested benchmark: {target}")
    saved_script_hash=record.get("benchmark_script_sha256")
    if saved_script_hash and saved_script_hash!=script_hash:
        raise RuntimeError(f"Saved result used a different benchmark script: {target}")
    if not saved_script_hash:
        print(f"NOTE {label}: legacy result predates script-hash metadata; preserve this limitation",flush=True)
    if not target.with_suffix(".jld2").exists():
        raise RuntimeError(f"Missing numerical result: {target}")
    if not records:
        reference=target.with_suffix(".jld2")
        record["reference_check"]={"relative_vector_error":0.0,"moment_max_abs_error":0.0}
        baseline=record["median"]["kernel_seconds"]
    else:
        comparison=target.with_suffix(".comparison.toml")
        command=[args.julia,"--startup-file=no","--threads=1,0",f"--project={project}",str(script),
            "--action","compare","--reference",str(reference.resolve()),
            "--candidate",str(target.with_suffix(".jld2").resolve()),"--output",str(comparison.resolve())]
        if not (args.resume and comparison.exists()):
            with target.with_suffix(".comparison.log").open("w") as handle:
                subprocess.run(command,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
        record["reference_check"]=tomllib.loads(comparison.read_text())
    record["speedup_vs_serial"]=baseline/record["median"]["kernel_seconds"]
    records.append(record)
    (args.output/"summary.json").write_text(json.dumps(records,indent=2)+"\n")
    print(f"DONE {label}: kernel={record['median']['kernel_seconds']:.3f}s speedup={record['speedup_vs_serial']:.2f} "
          f"vector_difference={record['reference_check']['relative_vector_error']:.3g}",flush=True)
print("Summary:",args.output/"summary.json")
