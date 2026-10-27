import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from verify_benefit import percentile
from verify_trace import request_metrics


QUANTILES = {"p90": .90, "p95": .95, "p99": .99}
METRICS = ("queue_s", "ttft_s", "latency_s", "tpot_s")


def metrics(request):
    output = len(request["token_ids"])
    return {"queue_s": request["reserved_at"] - request["arrived_at"],
            "ttft_s": request["first_token_at"] - request["arrived_at"],
            "latency_s": request["completed_at"] - request["arrived_at"],
            "tpot_s": ((request["completed_at"] - request["first_token_at"]) / (output - 1)
                       if output > 1 else None)}


def distribution(values):
    if not values:
        return None
    return {"samples": len(values), "mean_s": statistics.mean(values), "max_s": max(values),
            **{name + "_s": percentile(values, fraction) for name, fraction in QUANTILES.items()}}


def summarize_group(records, deadline):
    measurements = [metrics(record) for record in records]
    met = sum(item["latency_s"] <= deadline for item in measurements)
    return {"requests": len(records), "deadline_s": deadline, "deadline_met": met,
            "deadline_missed": len(records) - met,
            "deadline_met_fraction": met / len(records) if records else None,
            "expected_top_one_percent_samples": len(records) * .01,
            "metrics": {name: distribution([item[name] for item in measurements if item[name] is not None])
                        for name in METRICS}}


def compare_group(baseline, treatment, identifiers, deadline):
    selected = sorted(identifiers)
    before = summarize_group([baseline[identifier] for identifier in selected], deadline)
    after = summarize_group([treatment[identifier] for identifier in selected], deadline)
    changes = {}
    for name in METRICS:
        changes[name] = {}
        if before["metrics"][name] is None:
            continue
        for statistic in ("mean_s", "max_s", "p90_s", "p95_s", "p99_s"):
            baseline_value = before["metrics"][name][statistic]
            treatment_value = after["metrics"][name][statistic]
            changes[name][statistic] = {"delta_s": treatment_value - baseline_value,
                                      "reduction_fraction": 1 - treatment_value / baseline_value
                                      if baseline_value > 0 else None}
    pairs = [(identifier, metrics(baseline[identifier]), metrics(treatment[identifier])) for identifier in selected]
    rescued = [identifier for identifier, original, updated in pairs
               if original["latency_s"] > deadline >= updated["latency_s"]]
    regressed = [identifier for identifier, original, updated in pairs
                 if updated["latency_s"] > deadline >= original["latency_s"]]
    return {"requests": len(selected), "request_ids": selected, "baseline": before, "treatment": after,
            "changes": changes, "deadline_met_difference": after["deadline_met"] - before["deadline_met"],
            "rescued_request_ids": rescued, "regressed_request_ids": regressed,
            "paired_latency_sum_delta_s": sum(updated["latency_s"] - original["latency_s"]
                                              for identifier, original, updated in pairs),
            "paired_slower_requests": sum(updated["latency_s"] > original["latency_s"]
                                          for identifier, original, updated in pairs)}


def validate_records(records, marks):
    by_id = {record["request_id"]: record for record in records}
    if not records or len(by_id) != len(records) or set(by_id) != set(marks):
        raise ValueError("Requests must be unique and cover the complete frozen trace")
    for identifier, record in by_id.items():
        mark = marks[identifier]
        if (record["model"] != mark["model"] or record["prompt_tokens"] != mark["prompt_tokens"]
                or len(record["token_ids"]) != mark["output_tokens"]):
            raise ValueError("Request model or token lengths differ from frozen trace")
        times = [record[name] for name in ("arrived_at", "submitted_at", "reserved_at",
                                          "first_token_at", "completed_at")]
        if not all(math.isfinite(value) for value in times) or times != sorted(times):
            raise ValueError("Request timestamps must be finite and ordered")
    return by_id


def analyze(root, baseline="no_move", treatment="kv_only", repeat=0):
    root = Path(root)
    names = ["config.json", "trace.json", "exit-status", "summary.json"]
    config = json.loads((root / "config.json").read_text())
    trace = json.loads((root / "trace.json").read_text())
    if ((root / "exit-status").read_text().strip() != "0"
            or json.loads((root / "summary.json").read_text())["status"] != "passed"):
        raise ValueError("Tail comparison requires a successfully completed allocation")
    deadline = config["deadline_s"]
    if not math.isfinite(deadline) or deadline <= 0:
        raise ValueError("SLO deadline must be the positive frozen configured deadline")
    marks = {mark["request_id"]: mark for mark in trace["requests"]}
    if len(marks) != len(trace["requests"]):
        raise ValueError("Frozen trace contains duplicate request IDs")
    policies = {}
    origins = {}
    trace_sha = hashlib.sha256((root / "trace.json").read_bytes()).hexdigest()
    for policy in (baseline, treatment):
        if policy not in config["policies"]:
            raise ValueError("Compared policy is absent from frozen configuration")
        prefix = f"r{repeat}-{policy}"
        names.extend(f"{prefix}/{name}.json" for name in ("requests", "summary", "workload"))
        records = json.loads((root / prefix / "requests.json").read_text())
        summary = json.loads((root / prefix / "summary.json").read_text())
        workload = json.loads((root / prefix / "workload.json").read_text())
        if (workload["requests"] != trace["requests"] or workload["trace_sha256"] != trace_sha
                or workload["duration_s"] != trace["duration_s"]):
            raise ValueError("Policy workload differs from the frozen trace")
        policies[policy] = validate_records(records, marks)
        origins[policy] = workload["origin"]
        calculated = request_metrics(records, workload["origin"], trace["duration_s"], deadline)
        if any(not math.isclose(summary[name], value, rel_tol=1e-9, abs_tol=1e-6)
               for name, value in calculated.items() if value is not None):
            raise ValueError("Request metrics disagree with archived policy summary")
    before, after = policies[baseline], policies[treatment]
    if any(before[identifier]["token_ids"] != after[identifier]["token_ids"] for identifier in marks):
        raise ValueError("Paired policies generated different output token IDs")
    arrivals = [origins[baseline] + mark["arrival_s"] for mark in marks.values() if mark["model"] == 0]
    old_conv = {identifier for identifier, record in before.items() if record["model"] == 1
                and any(record["reserved_at"] <= arrival < record["completed_at"] for arrival in arrivals)}
    groups = {"all_requests": set(marks), "Code": {identifier for identifier in marks if marks[identifier]["model"] == 0},
              "Conv": {identifier for identifier in marks if marks[identifier]["model"] == 1},
              "baseline_resident_Conv_at_Code_arrival": old_conv}
    code_records = [{"request_id": identifier, "arrival_s": marks[identifier]["arrival_s"],
                     "baseline": metrics(before[identifier]), "treatment": metrics(after[identifier])}
                    for identifier in sorted(groups["Code"], key=lambda identifier: marks[identifier]["arrival_s"])]
    return {"status": "paired_tail_slo_raw_metrics_verified", "baseline": baseline, "treatment": treatment,
            "repeat": repeat, "deadline_s": deadline, "duration_s": trace["duration_s"],
            "percentile_method": "linear_interpolation_position_(n-1)*q_type7_inclusive",
            "scope": "posthoc_descriptive_percentiles_existing_predeclared_SLO_one_trial_not_multiseed_significance",
            "groups": {name: compare_group(before, after, identifiers, deadline) for name, identifiers in groups.items()},
            "Code_paired_requests": code_records,
            "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--baseline", default="no_move")
    parser.add_argument("--treatment", default="kv_only")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.baseline, args.treatment, args.repeat)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "baseline": result["baseline"], "treatment": result["treatment"],
                      "requests": result["groups"]["all_requests"]["requests"], "deadline_s": result["deadline_s"]}))


if __name__ == "__main__":
    main()
