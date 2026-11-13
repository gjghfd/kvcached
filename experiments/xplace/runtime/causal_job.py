import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from causal_deployment import read_manifest, sha256


BLOCKS = (("off", "collect"), ("collect", "off"))


def save(filename, value):
    filename.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def remaining_seconds(started_at, now=None):
    age = (time.time() if now is None else now) - float(started_at)
    if not math.isfinite(age) or age < 0 or age >= 3300:
        raise ValueError("Invalid or exhausted allocation budget")
    return 3300 - age


def process_group_usage(group, proc_root=Path("/proc")):
    records, unavailable = [], []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            raw = (directory / "stat").read_text()
            fields = raw[raw.rfind(")") + 2:].split()
            if int(fields[2]) != group:
                continue
            records.append(dict(pid=int(directory.name), parent=int(fields[1]), state=fields[0],
                                start_ticks=int(fields[19]), cpu_ticks=int(fields[11]) + int(fields[12]),
                                rss_pages=int(fields[21])))
        except (OSError, ValueError, IndexError) as error:
            unavailable.append(dict(pid=int(directory.name), reason=type(error).__name__))
    return dict(at=time.monotonic(), group=group, processes=records, read_failures=unavailable,
                clock_ticks_per_s=os.sysconf("SC_CLK_TCK"), page_bytes=os.sysconf("SC_PAGE_SIZE"),
                scope="sampled_process_group_not_exact_cumulative_cpu_or_peak_memory")


def signal_group(group, action):
    try:
        os.killpg(group, action)
        return True
    except ProcessLookupError:
        return False


def run_child(command, environment, root, deadline, sample=process_group_usage):
    root.mkdir(parents=True, exist_ok=False)
    record = dict(command=command, started_at=time.time(), status="starting", returncode=None,
                  forced_cleanup=False, timeout=False)
    save(root / "process.json", record)
    child = None
    try:
        with (root / "stdout.log").open("x") as stdout, (root / "stderr.log").open("x") as stderr, \
                (root / "process-usage.jsonl").open("x") as usage:
            child = subprocess.Popen(command, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
            record.update(pid=child.pid, status="running")
            save(root / "process.json", record)
            while child.poll() is None:
                usage.write(json.dumps(sample(child.pid), allow_nan=False) + "\n")
                usage.flush()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    record["timeout"] = True
                    raise TimeoutError("Shared allocation soft deadline exhausted")
                try:
                    child.wait(timeout=min(1.0, remaining))
                except subprocess.TimeoutExpired:
                    pass
            record["returncode"] = child.returncode
            record["status"] = "passed" if child.returncode == 0 else "failed"
    except BaseException as error:
        record.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        if child is not None:
            if record["status"] == "passed":
                cleanup_started = time.monotonic()
                while signal_group(child.pid, 0) and time.monotonic() - cleanup_started < 5:
                    time.sleep(0.1)
                record["natural_group_exit_wait_s"] = time.monotonic() - cleanup_started
            record["forced_cleanup"] = signal_group(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                signal_group(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
            if record["forced_cleanup"]:
                limit = time.monotonic() + 5
                while signal_group(child.pid, 0) and time.monotonic() < limit:
                    time.sleep(0.1)
                record["kill_remaining_group"] = signal_group(child.pid, signal.SIGKILL)
                if record["status"] == "passed":
                    record["status"] = "failed_lingering_process_group"
            record["returncode"] = child.returncode
        record["finished_at"] = time.time()
        save(root / "process.json", record)
    return record


def run_block(*, block, root, common_args, environment, runner=run_child):
    if type(block) is not int or block not in range(len(BLOCKS)):
        raise ValueError("Unknown fixed overhead block")
    root.mkdir(parents=True, exist_ok=False)
    remaining = remaining_seconds(environment["XPLACE_JOB_STARTED_AT"])
    deadline = time.monotonic() + remaining
    result = dict(block=block, planned=list(BLOCKS[block]), started=[], completed=[], status="running",
                  prediction_gate_passed=False, overhead_accepted=False)
    save(root / "status.json", result)
    try:
        for mode in BLOCKS[block]:
            if deadline - time.monotonic() < 1500:
                raise TimeoutError("Insufficient time to start a fixed four-case condition")
            transfer = Path(environment["DP_OPT_TRANSFER_ROOT"]) / f"block{block}-{mode}"
            transfer.mkdir(exist_ok=False)
            output = root / mode
            command = [sys.executable, str(Path(__file__).with_name("main.py")), *common_args,
                       "--output", str(output / "out"), "--causal-mode", mode]
            result["started"].append(mode)
            save(root / "status.json", result)
            receipt = runner(command, dict(environment, DP_OPT_TRANSFER_ROOT=str(transfer)), output, deadline)
            result["completed"].append(dict(mode=mode, **receipt))
            save(root / "status.json", result)
            if receipt["status"] != "passed":
                raise RuntimeError(f"Calibration condition {mode} failed; no automatic retry")
        result["status"] = "collected"
    except BaseException as error:
        result.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        result["not_executed"] = [mode for mode in BLOCKS[block] if mode not in result["started"]]
        save(root / "status.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", type=int, choices=range(2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, common = parser.parse_known_args()
    if "--causal-mode" in common:
        parser.error("Mode order is fixed by the job block")
    manifest, identity = read_manifest()
    protocol_path = Path(os.environ["XPLACE_CAUSAL_PROTOCOL"])
    if sha256(protocol_path) != manifest["protocol_sha256"]:
        raise ValueError("Frozen deployment protocol differs")
    protocol = json.loads(protocol_path.read_text())
    if protocol["blocks"] != [list(block) for block in BLOCKS]:
        raise ValueError("Frozen overhead order differs")
    environment = dict(os.environ)
    for option, digest_key in (("--causal-config", "training_plan_sha256"), ("--prompt-bank", "prompt_bank_sha256")):
        if common.count(option) != 1 or sha256(common[common.index(option) + 1]) != manifest[digest_key]:
            raise ValueError(f"Frozen {option} differs")
    def interrupted(signum, frame):
        raise InterruptedError(f"Job interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    print(json.dumps(dict(manifest_sha256=identity, block=args.block)), flush=True)
    run_block(block=args.block, root=args.output / "causal-block", common_args=common, environment=environment)


if __name__ == "__main__":
    main()
