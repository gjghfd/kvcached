"""Verify dense decode calibration against per-request cadence and summarize repeats."""
import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
sys.path.insert(0, str(RUNTIME))
from calibrate import steady_step
from imbalance_calibration import scan_cases


def require(condition, message):
    if not condition:
        raise ValueError(message)


def analyze(root):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    config = plan["config"]
    num_ranks = config.get("num_ranks", 4)
    expected = {case["name"]: case for case in scan_cases(config, num_ranks)}
    require(plan["cases"] == list(expected.values()), "Stored plan differs from the scan configuration")
    records = json.loads((root / "results.json").read_text())
    require(len(records) == len(expected) and {record["name"] for record in records} == set(expected),
            "Scan is incomplete; do not replace the paper plot")
    grouped = defaultdict(list)
    paths = [root / "plan.json", root / "results.json", root / "summary.json"]
    windows = []
    total_requests = 0
    for record in records:
        case = expected[record["name"]]
        require(all(record[key] == value for key, value in case.items()), f"Case configuration differs: {case['name']}")
        require(record["status"] == "measured" and record["preemptions"] == 0,
                f"Invalid measurement: {case['name']}")
        require(record["requests_file"] == f"{case['name']}-requests.json", "Unexpected request-file path")
        path = root / record["requests_file"]
        paths.append(path)
        requests = json.loads(path.read_text())
        require(len(requests) == case["batch_size"] and len({request["request_id"] for request in requests}) == len(requests),
                "Request count or identifiers differ from the fixed batch")
        ranks = Counter(request["rank_initial"] for request in requests)
        require(set(ranks) <= set(range(num_ranks)) and [ranks[rank] for rank in range(num_ranks)] == case["layout"],
                "Request placement differs from the registered layout")
        for request in requests:
            require(request["prompt_tokens"] == config["context_tokens"]
                    and request["output_tokens"] == config["output_tokens"] and request["model"] == 1,
                    "Request length or model differs from the calibration setting")
            cadence = request["cadence"]
            require(cadence and cadence[-1][1] == config["output_tokens"]
                    and all(later[0] >= earlier[0] and later[1] > earlier[1]
                            for earlier, later in zip(cadence, cadence[1:])), "Invalid token cadence")
        step, window = steady_step(requests)
        require(step is not None and math.isclose(step, record["step_s"], rel_tol=1e-12)
                and window == record["window"] and window["samples"] == case["batch_size"],
                "Reported decode step differs from the common all-decode window")
        require(record["common_start"] == max(request["first_token_at"] for request in requests) + 1.0
                and record["common_end"] == min(request["completed_at"] for request in requests) - 1.0,
                "Window must exclude prefill and the completion tail")
        windows.append(window["window_s"])
        total_requests += len(requests)
        grouped[(case["batch_size"], case["imbalance"])].append(record)
    points = []
    for (batch, imbalance), trials in sorted(grouped.items()):
        require(len(trials) == config["repeats"] and {trial["repeat"] for trial in trials} == set(range(config["repeats"])),
                "Repeated measurement missing or duplicated")
        samples = [trial["step_s"] * 1000 for trial in sorted(trials, key=lambda trial: trial["repeat"])]
        points.append({"batch_size": batch, "imbalance": imbalance, "mean_ms": statistics.mean(samples),
                       "normalized_imbalance": imbalance / (batch / num_ranks),
                       "min_ms": min(samples), "max_ms": max(samples), "samples_ms": samples,
                       "capture_bucket": trials[0]["capture_bucket"], "trials": [trial["name"] for trial in trials]})
    baselines = {point["batch_size"]: point["mean_ms"] for point in points if point["imbalance"] == 0}
    for point in points:
        point["change_pct"] = (point["mean_ms"] / baselines[point["batch_size"]] - 1) * 100
    summary = json.loads((root / "summary.json").read_text())
    require(summary["status"] == "passed" and summary["planned"] == summary["completed"] == summary["measured"] == len(records),
            "Scan summary is incomplete")
    return {"status": "passed", "trials": len(records), "requests": total_requests,
            "num_ranks": num_ranks, "tensor_parallel_size": config.get("tensor_parallel_size", 1),
            "expert_parallel_size": num_ranks * config.get("tensor_parallel_size", 1),
            "points": points, "repeats": config["repeats"], "context_tokens": config["context_tokens"],
            "output_tokens": config["output_tokens"], "minimum_common_window_s": min(windows),
            "uncertainty": "min-max of repeated trial step times, not a confidence interval",
            "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="job/out/calibration/imbalance directory")
    parser.add_argument("--output", type=Path, required=True, help="Analysis output outside the raw evidence directory")
    args = parser.parse_args()
    require(args.root.resolve() not in args.output.resolve().parents, "Do not write analysis into raw evidence")
    report = analyze(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"PASS: {report['trials']} trials, {len(report['points'])} points, {report['requests']} requests")
    for batch in sorted({point["batch_size"] for point in report["points"]}):
        points = [point for point in report["points"] if point["batch_size"] == batch]
        print(f"BS{batch}: {len(points)} points, {points[0]['mean_ms']:.3f}–{points[-1]['mean_ms']:.3f} ms")


if __name__ == "__main__":
    main()
