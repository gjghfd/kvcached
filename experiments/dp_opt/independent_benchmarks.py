import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from benchmark import read_memory, verify_workers
from single_model_benchmark import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", nargs=2, type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--rank-requests", nargs=2, type=int, default=[8, 8])
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=256)
    args = parser.parse_args()
    workers = verify_workers(args.evidence_root)
    if any(worker["checkpoint"] != args.checkpoint for worker in workers):
        raise RuntimeError("Workers do not use the requested checkpoint")
    names = [worker["ipc_name"] for worker in workers]
    before = read_memory(names)
    root = args.evidence_root / "independent"
    root.mkdir(parents=True, exist_ok=True)
    samples = []
    stop = threading.Event()
    clients = []
    logs = []
    releases = []

    def monitor():
        while not stop.is_set():
            try:
                samples.append({"time_ns": time.time_ns(), "pools": read_memory(names)})
            except Exception as error:
                samples.append({"error": str(error)})
            stop.wait(0.5)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        for service_index, port in enumerate(args.ports, start=1):
            service = f"instance-{service_index}"
            log = (root / f"{service}.log").open("w")
            logs.append(log)
            command = [
                sys.executable, str(Path(__file__).with_name("single_model_benchmark.py")),
                "--url", f"http://127.0.0.1:{port}",
                "--model", f"Qwen/Qwen3-30B-A3B-Instruct-2507-{service}",
                "--service", service, "--checkpoint", args.checkpoint,
                "--output", str(root / f"{service}.json"), "--sync-root", str(root),
                "--rank-requests", *(str(count) for count in args.rank_requests),
                "--concurrency", str(args.concurrency), "--rounds", str(args.rounds),
                "--input-tokens", str(args.input_tokens),
                "--output-tokens", str(args.output_tokens),
            ]
            clients.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        for round_index in range(args.rounds):
            deadline = time.monotonic() + 1200
            markers = [root / f"instance-{service_index}-round-{round_index}.ready"
                       for service_index in (1, 2)]
            while not all(marker.exists() for marker in markers):
                if any(client.poll() is not None for client in clients):
                    raise RuntimeError("Benchmark client exited before synchronization; see client logs")
                if time.monotonic() > deadline:
                    raise RuntimeError("Benchmark clients did not become ready")
                time.sleep(0.1)
            releases.append({"round": round_index, "time_ns": time.time_ns()})
            (root / f"round-{round_index}.go").write_text(str(releases[-1]["time_ns"]))
        for client in clients:
            if client.wait(timeout=1200) != 0:
                raise RuntimeError("Independent benchmark client failed; see client report/log")
    finally:
        for client in clients:
            if client.poll() is None:
                client.terminate()
        for client in clients:
            try:
                client.wait(timeout=10)
            except subprocess.TimeoutExpired:
                client.kill()
                client.wait(timeout=10)
        stop.set()
        monitor_thread.join(timeout=10)
        for log in logs:
            log.close()
        save_json(root / "kv-memory.json", {"before": before, "samples": samples,
                                            "after": read_memory(names)})
    reports = [json.loads((root / f"instance-{service_index}.json").read_text())
               for service_index in (1, 2)]
    if any(report["status"] != "passed" for report in reports):
        raise RuntimeError("Independent client verification did not pass")
    if len({report["client_pid"] for report in reports}) != 2:
        raise RuntimeError("Expected two separate benchmark processes")
    if any("error" in sample for sample in samples):
        raise RuntimeError("KV memory monitoring failed")
    rounds = []
    for round_index in range(args.rounds):
        records = [report["rounds"][round_index]["requests"] for report in reports]
        starts = [min(record["start_ns"] for record in model_records) for model_records in records]
        ends = [max(record["end_ns"] for record in model_records) for model_records in records]
        span_s = (max(ends) - min(starts)) / 1e9
        overlap_s = (min(ends) - max(starts)) / 1e9
        tokens = sum(record["usage"]["completion_tokens"]
                     for model_records in records for record in model_records)
        if overlap_s <= 0:
            raise RuntimeError("Independent model benchmarks did not overlap")
        resident_samples = sum(
            min(starts) <= sample["time_ns"] <= max(ends)
            and all(pool["mapped_live_bytes"] > before[name]["mapped_live_bytes"]
                    for name, pool in sample["pools"].items())
            for sample in samples
        )
        if resident_samples == 0:
            raise RuntimeError("No simultaneous above-idle KV residency in a measured round")
        rounds.append({"round": round_index, "request_span_s": span_s,
                       "overlap_s": overlap_s, "completion_tokens": tokens,
                       "output_tokens_per_second": tokens / span_s,
                       "four_pools_above_idle_samples": resident_samples})
    report = {
        "status": "passed", "backend": "native_vllm+kvcached", "workers": workers,
        "client_pids": [client["client_pid"] for client in reports],
        "models": {client["service"]: client["summary"] for client in reports},
        "rounds": rounds, "round_releases": releases,
        "completion_tokens": sum(result["completion_tokens"] for result in rounds),
        "active_output_tokens_per_second": sum(result["completion_tokens"] for result in rounds)
        / sum(result["request_span_s"] for result in rounds),
        "scope": "Two independent per-model clients, simultaneous closed-loop bursts; no migration claim",
    }
    save_json(args.evidence_root / "independent-benchmark.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "workers"}, indent=2))


if __name__ == "__main__":
    main()
