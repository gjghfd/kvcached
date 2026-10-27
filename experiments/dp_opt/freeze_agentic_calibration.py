import argparse
import hashlib
import json
import statistics
from pathlib import Path

from agentic_trace import generate_mixed_trace


def choose_rate(calibration_root, summary):
    if summary["status"] != "passed":
        raise ValueError("Cannot freeze from incomplete calibration")
    candidates = []
    for index, trial in enumerate(summary["trials"]):
        directory = calibration_root / f"calibration-{index}"
        requests = json.loads((directory / "requests.json").read_text())
        samples = json.loads((directory / "population.json").read_text())
        duration = trial["duration_s"]
        early = [sample for sample in samples if duration / 3 <= sample["elapsed_s"] < duration / 2]
        late = [sample for sample in samples if duration * 5 / 6 <= sample["elapsed_s"] < duration]
        queue = lambda selection: statistics.mean(max(0, sample["unfinished"] - sample["active"]
                                                       + sample.get("capacity_pending", 0))
                                                  for sample in selection)
        growth = queue(late) - queue(early)
        conv = [request for request in requests if request["model"] == 1]
        conv_queue = statistics.mean(request.get("reserved_at", request["submitted_at"])
                                     - request["arrived_at"] for request in conv)
        candidates.append({"index": index, "conv_rate_rps": trial["conv_rate_rps"],
                           "code_rate_rps": trial["code_rate_rps"],
                           "mean_conv_reserved_mib": trial["mean_sampled_conv_reserved_mib"],
                           "external_queue_growth_requests": growth, "mean_conv_queue_s": conv_queue,
                           "drain_s": trial["summary"]["drain_s"],
                           "eligible": growth <= 10 and conv_queue <= 5 and trial["summary"]["drain_s"] <= 90})
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise ValueError("No stable candidate under predeclared queue/drain guardrails")
    winner = min(eligible, key=lambda candidate: (abs(candidate["mean_conv_reserved_mib"]
                                                     - summary["target_conv_reserved_mib"]),
                                                 candidate["conv_rate_rps"]))
    target = summary["target_conv_reserved_mib"]
    if not .95 * target <= winner["mean_conv_reserved_mib"] <= 1.25 * target:
        raise ValueError("Stable calibration rates do not reach the declared KV pressure window")
    return {"winner": winner, "candidates": candidates, "target_conv_reserved_mib": target,
            "scope": "rate_chosen_before_policy_outcomes_not_steady_state_certificate",
            "guardrails": {"max_external_queue_growth_requests": 10, "max_mean_conv_queue_s": 5,
                           "max_drain_s": 90, "target_range_fractions": [.95, 1.25]}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        summary = json.loads((args.calibration / "calibration-summary.json").read_text())
        decision = choose_rate(args.calibration, summary)
        code_marks = json.loads((args.inputs / "code-marks.json").read_text())
        conv_marks = json.loads((args.inputs / "conv-marks.json").read_text())
        winner = decision["winner"]
        trace = generate_mixed_trace(code_marks, conv_marks, winner["code_rate_rps"], winner["conv_rate_rps"],
                                     1200, 20261021, 131072)
        trace["prompt_bank_sha256"] = hashlib.sha256((args.inputs / "prompt-bank.json").read_bytes()).hexdigest()
        trace["slice"] = {"code": "ten_verified_SWE_smith_history_episodes_100K..105K_output64",
                          "conv": "full_Azure_Conv_Pge2048_O512to1024_paired_lengths"}
        sources = [args.calibration / "calibration-summary.json", args.inputs / "code-marks.json",
                   args.inputs / "conv-marks.json", args.inputs / "prompt-bank.json"]
        decision["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        decision["request_counts"] = [sum(mark["model"] == model for mark in trace["requests"]) for model in (0, 1)]
        trace_path = args.output / "trace.json"
        trace_path.write_text(json.dumps(trace, indent=2) + "\n")
        decision["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        config = json.loads((args.calibration / "config.json").read_text())
        decision["kv_budget_mib_per_gpu"] = config["kv_budget_mib"]
        decision["max_model_len"] = config["max_model_len"]
        (args.output / "rate-decision.json").write_text(json.dumps(decision, indent=2) + "\n")
        print(json.dumps(decision, indent=2))
    except BaseException as error:
        (args.output / "error.json").write_text(json.dumps({"type": type(error).__name__, "message": str(error)}) + "\n")
        raise


if __name__ == "__main__":
    main()
