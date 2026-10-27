import argparse
import json
import statistics
from pathlib import Path

from live_benefit import percentile


def summarize(values):
    if not values:
        return None
    return {"count": len(values), "mean": statistics.mean(values),
            **{name: percentile(values, quantile) for name, quantile in (("p90", .90), ("p95", .95), ("p99", .99))},
            "max": max(values)}


def analyze_phases(root, start_s=600, end_s=2400):
    config = json.loads((root / "config.json").read_text())
    trace = json.loads((root / "trace.json").read_text())
    marks = {mark["request_id"]: mark for mark in trace["requests"]}
    result = {"evaluation_window_s": [start_s, end_s], "policies": {},
              "scope": "observed_phase_attribution_not_a_counterfactual_or_kernel_profile"}
    for item in json.loads((root / "order.json").read_text()):
        directory = root / f"r{item['repeat']}-{item['policy']}"
        records = json.loads((directory / "requests.json").read_text())
        origin = json.loads((directory / "workload.json").read_text())["origin"]
        decisions = json.loads((directory / "decisions.json").read_text())
        selected = [record for record in records if start_s <= marks[record["request_id"]]["arrival_s"] < end_s]
        if len(records) != len(marks) or {record["request_id"] for record in records} != set(marks):
            raise ValueError("Cannot attribute tails with missing or duplicate completed requests")
        phases = []
        for record in selected:
            reserved = record["reserved_at"]
            if not record["arrived_at"] <= reserved <= record["first_token_at"] <= record["completed_at"]:
                raise ValueError("TTFT phases have unordered timestamps")
            overlaps = []
            for decision in decisions:
                intervals = [(receipt["reached_at"], decision["resume_completed_at"])
                             for ranks in decision.get("pause_receipts", []) for receipt in ranks
                             if "resume_completed_at" in decision]
                if intervals:
                    duration = max(0, min(record["first_token_at"], max(end for _, end in intervals))
                                   - max(record["arrived_at"], min(begin for begin, _ in intervals)))
                    if duration:
                        overlaps.append({"path": decision["path"], "overlap_upper_s": duration,
                                         "moves": len(decision["moves"])})
            phases.append({"request_id": record["request_id"], "model": record["model"],
                           "arrival_s": record["arrived_at"] - origin, "prompt_tokens": record["prompt_tokens"],
                           "ttft_s": record["first_token_at"] - record["arrived_at"],
                           "reservation_queue_s": reserved - record["arrived_at"],
                           "reserved_to_first_token_s": record["first_token_at"] - reserved,
                           "decode_s": record["completed_at"] - record["first_token_at"],
                           "measured_pause_overlaps": overlaps})
        groups = {"all": phases, **{f"model_{model}": [record for record in phases if record["model"] == model]
                                   for model in (0, 1)}}
        result["policies"][directory.name] = {
            "groups": {name: {"phase_statistics": {field: summarize([record[field] for record in group])
                for field in ("ttft_s", "reservation_queue_s", "reserved_to_first_token_s", "decode_s")},
                "worst_requests": sorted(group, key=lambda record: record["ttft_s"], reverse=True)[:10]}
                for name, group in groups.items()},
            "control_totals_s": {field: sum(decision.get(field, 0) for decision in decisions)
                                  for field in ("pause_call_s", "preplan_s", "solve_s")}}
    result["deadline_s"] = config["deadline_s"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze_phases(args.root), indent=2) + "\n")


if __name__ == "__main__":
    main()
