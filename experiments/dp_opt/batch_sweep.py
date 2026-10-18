import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from motivation import validate_capacity
from single_model_benchmark import save_json


def workload_cases():
    cases = [{"name": f"batch-{requests}-ctx1024", "requests": requests,
              "input_tokens": 1024, "output_tokens": 512, "family": "batch"}
             for requests in (16, 32, 64, 128, 256)]
    cases.extend({"name": f"kv-{requests}-ctx{context}", "requests": requests,
                  "input_tokens": context, "output_tokens": 512, "family": "kv"}
                 for requests, context in ((8, 32768), (16, 16384), (32, 8192),
                                           (64, 4096), (128, 2048)))
    return cases


def case_metrics(report):
    balanced = next(row for row in report["summary"] if row["imbalance"] == 0)
    extremes = [row for row in report["summary"] if row["imbalance"] == 1]
    if len(extremes) != 2:
        raise ValueError("Both hot-rank orientations are required")
    rows = []
    for row in extremes:
        interval = row["paired_slowdown_t95_ci"]
        rows.append({
            "layout": row["layout"], "paired_slowdown": row["paired_slowdown"],
            "paired_slowdown_t95_ci": interval,
            "steady_decode_loss": 1 - 1 / row["paired_slowdown"],
            "steady_decode_loss_t95_ci": None if interval is None else [
                1 - 1 / bound for bound in interval
            ],
            "burst_loss": 1 - row["output_tokens_per_second"] / balanced["output_tokens_per_second"],
            "cadence_ms": row["cadence_s"] * 1000,
        })
    return {"orientations": rows,
            "minimum_orientation_decode_loss": min(row["steady_decode_loss"] for row in rows),
            "maximum_orientation_decode_loss": max(row["steady_decode_loss"] for row in rows),
            "maximum_orientation_burst_loss": max(row["burst_loss"] for row in rows)}


def confirmed_target(metrics, target=0.30):
    return all(row["steady_decode_loss_t95_ci"] is not None
               and row["steady_decode_loss_t95_ci"][0] >= target
               for row in metrics["orientations"])


def resumed_screen(case, previous_root, workers):
    case_name = "screen-" + case["name"]
    path = previous_root / "motivation/cases" / case_name / "results.json"
    if not path.is_file():
        return None
    result = json.loads(path.read_text())
    if result["status"] != "passed":
        return None
    protocol = result["protocol"]
    expected = {"requests": case["requests"], "input_tokens": case["input_tokens"],
                "output_tokens": case["output_tokens"], "repeats": 1,
                "screening": True, "extremes_only": True,
                "trim_steps_each_end": 32, "minimum_stable_steps": 128}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("Resumed screening protocol does not match the planned workload")
    if protocol.get("expected_cudagraph_mode", "NONE") != "NONE":
        raise ValueError("Only eager screening can be resumed by the eager sweep")
    fields = ("checkpoint", "gpu_name", "parameter_bytes", "max_model_len",
              "max_num_seqs", "max_num_batched_tokens", "kv_num_blocks", "kv_block_size")
    old_workers = sorted(result["workers"], key=lambda worker: worker["dp_rank"])
    new_workers = sorted(workers, key=lambda worker: worker["dp_rank"])
    if len(old_workers) != 2 or len(new_workers) != 2:
        raise ValueError("Resumed screening requires two worker records")
    if any(old.get(field) != new.get(field)
           for old, new in zip(old_workers, new_workers) for field in fields):
        raise ValueError("Resumed screening hardware/configuration differs from this allocation")
    return {**case, "case_name": case_name, **case_metrics(result),
            "resumed_from": str(path), "selection_only": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--resume-root", type=Path)
    args = parser.parse_args()
    root = args.evidence_root / "motivation/batch-sweep"
    root.mkdir(parents=True, exist_ok=True)
    workers = [json.loads(path.read_text()) for path in sorted(args.evidence_root.glob("worker-*.json"))]
    if len(workers) != 2:
        raise RuntimeError("Two worker capacity records are required")
    report = {"status": "running", "target_decode_loss": 0.30,
              "primary_metric": "paired stable decode throughput loss, not latency increase or burst loss",
              "selection": "Exploratory screening, then fresh n=3 paired confirmation. "
                           "Target requires lower 95% ratio-derived loss bound >=30% in both orientations.",
              "screening": [], "confirmations": [], "skipped": []}
    candidates = []
    cases = workload_cases()
    random.Random(20261018).shuffle(cases)
    for case in cases:
        try:
            capacity = validate_capacity(workers, case["input_tokens"], case["output_tokens"],
                                         total=case["requests"])
            if max(capacity["required_hot_rank_blocks"] / blocks
                   for blocks in capacity["available_blocks_per_rank"]) >= 0.85:
                raise ValueError("Analytical virtual KV utilization must stay below 85%")
        except ValueError as error:
            report["skipped"].append({**case, "reason": str(error)})
            continue
        candidates.append(case)
    save_json(root / "plan.json", {"screening": candidates, "skipped": report["skipped"]})

    def run(case, confirmation=False):
        case_name = ("confirm-" if confirmation else "screen-") + case["name"]
        command = [sys.executable, str(Path(__file__).with_name("motivation.py")),
                   "--url", args.url, "--checkpoint", args.checkpoint,
                   "--evidence-root", str(args.evidence_root), "--case-name", case_name,
                   "--requests", str(case["requests"]), "--input-tokens", str(case["input_tokens"]),
                   "--output-tokens", str(case["output_tokens"]), "--extremes-only"]
        if not confirmation:
            command.extend(("--screening", "--repeats", "1", "--skip-overhead-controls"))
        print(json.dumps({"starting_case": case, "confirmation": confirmation}), flush=True)
        subprocess.run(command, check=True)
        result = json.loads((args.evidence_root / "motivation/cases" / case_name / "results.json").read_text())
        if result["status"] != "passed":
            raise RuntimeError("Only verified trials can enter the sweep summary")
        return {**case, "case_name": case_name, **case_metrics(result)}

    try:
        for case in candidates:
            resumed = resumed_screen(case, args.resume_root, workers) if args.resume_root else None
            report["screening"].append(resumed if resumed is not None else run(case))
            save_json(root / "results.json", report)
        best = sorted(report["screening"],
                      key=lambda case: case["minimum_orientation_decode_loss"], reverse=True)
        for case in best[:2]:
            confirmation = run(case, True)
            confirmation["target_confirmed"] = confirmed_target(confirmation)
            report["confirmations"].append(confirmation)
            save_json(root / "results.json", report)
        report["status"] = "passed"
        report["target_confirmed"] = any(case["target_confirmed"] for case in report["confirmations"])
        report["completed_ns"] = time.time_ns()
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        save_json(root / "results.json", report)
        raise
    save_json(root / "results.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
