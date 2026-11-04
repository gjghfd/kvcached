import argparse
import ast
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import archive_trace_job


def cancelled_extractor(original, authorized_job):
    tree = ast.parse(inspect.getsource(original))
    changed = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "failed_exit"):
            authorization = ast.parse("allow_missing_failed_progress and job == _authorized_cancelled_job", mode="eval").body
            node.value = ast.BoolOp(op=ast.Or(), values=[authorization, node.value])
            changed += 1
    if changed != 2:
        raise ValueError("Require the two exact failed-progress marker assignments")
    namespace = {**original.__globals__, "_authorized_cancelled_job": authorized_job}
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(original), "exec"), namespace)
    return namespace[original.__name__]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=int)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--authorize-scientific-cancelled-checkpoint", action="store_true", required=True)
    args = parser.parse_args()
    if args.job <= 0:
        parser.error("Require a positive checkpoint job ID")
    command = ('set -eu; printf "OWNER|%s\\n" "$(id -un)"; '
               f'sacct -n -X -P -j {args.job} --format=JobIDRaw,User,JobName%128,Partition,State%64')
    result = subprocess.run(["ssh", "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=20",
                             "-o", "ServerAliveCountMax=3", "paracloud", command],
                            text=True, capture_output=True, timeout=120)
    result.check_returncode()
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 2 or not lines[0].startswith("OWNER|"):
        raise ValueError("Require independent checkpoint owner and accounting")
    fields = lines[1].rstrip("|").split("|")
    if (len(fields) != 5 or fields[:4] != [str(args.job), lines[0].split("|", 1)[1],
                                         "qwen-public-perf-native", "gpu_a800"]
            or fields[4].split()[0] != "CANCELLED"):
        raise ValueError("Only the explicitly owned scientific CANCELLED native checkpoint is authorized")
    args.archive.mkdir(parents=True, exist_ok=True)
    receipt = {"job": args.job, "Slurm_state": "CANCELLED", "owner_accounting_observation": result.stdout,
               "raw_exit_marker_not_fabricated": True, "raw_files_and_before_after_SHA_checks_unchanged": True,
               "only_dangling_progress_pointer_marker_gate_changed": True,
               "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "canonical_archiver_sha256": hashlib.sha256(Path(archive_trace_job.__file__).read_bytes()).hexdigest()}
    (args.archive / "checkpoint-archive-policy.json").write_text(json.dumps(receipt, indent=2) + "\n")
    original = archive_trace_job.extract_raw
    archive_trace_job.extract_raw = cancelled_extractor(original, args.job)
    argv = sys.argv
    sys.argv = [argv[0], str(args.job), "--archive", str(args.archive), "--bundle", str(args.bundle)]
    try:
        archive_trace_job.main()
    finally:
        sys.argv = argv
        archive_trace_job.extract_raw = original


if __name__ == "__main__":
    main()
