import argparse
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path

from amortized_campaign import paired_window
from analyze_tail_slo import compare_group


def finite_time(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Require finite workload-relative timestamps")
    return value


def stable_counts(sample):
    if not sample.get("admission_stable_across_observation", False):
        return None
    counts = sample["counts"]
    if (len(counts) != 2 or any(len(ranks) != 2 for ranks in counts)
            or any(type(count) is not int or count < 0 for ranks in counts for count in ranks)):
        raise ValueError("Stable population samples require two nonnegative integer DP counts per tenant")
    finite_time(sample["elapsed_s"])
    return counts


def committed_returns(decisions, origin):
    returns = []
    for decision in decisions:
        if (decision["path"] != "release_rebalance_ilp" or decision.get("reason") != "rebalanced"
                or not any(move["model"] == 1 for move in decision.get("moves", []))):
            continue
        started = finite_time(decision["started_at"]) - origin
        resumed = finite_time(decision["resume_completed_at"]) - origin
        if resumed < started:
            raise ValueError("Committed return must resume after its start")
        returns.append({"start_s": started, "resume_s": resumed,
                        "reported_pause_call_s": decision.get("pause_call_s"),
                        "reported_active_blocking_upper_request_s": decision.get("active_blocking_upper_request_s"),
                        "moves": decision["moves"]})
    return sorted(returns, key=lambda event: event["start_s"])


def release_windows(records, population, returns, origin, duration):
    code = sorted((record for record in records if record["model"] == 0),
                  key=lambda record: record["completed_at"])
    samples = [(sample, stable_counts(sample)) for sample in population]
    result = []
    for record in code:
        released = finite_time(record["completed_at"]) - origin
        future = [other["arrived_at"] - origin for other in code if other["arrived_at"] - origin > released]
        end = min([duration, *future])
        overlapped = any(other["request_id"] != record["request_id"]
                         and other["arrived_at"] <= record["completed_at"] < other["completed_at"] for other in code)
        observed = [(sample, counts) for sample, counts in samples if counts is not None
                    and released <= sample["elapsed_s"] < end and sum(counts[0]) == 0 and sum(counts[1]) >= 2]
        observed.sort(key=lambda entry: entry[0]["elapsed_s"])
        balanced = [sample["elapsed_s"] for sample, counts in observed if abs(counts[1][0] - counts[1][1]) <= 1]
        events = [event for event in returns if released <= event["start_s"] < end]
        result.append({"Code_request_id": record["request_id"], "client_completion_s": released,
                       "window_end_s": end, "overlapping_Code_at_completion": overlapped,
                       "stable_Conv_only_population_samples": len(observed),
                       "single_attention_rank_samples": sum(min(counts[1]) == 0 for _, counts in observed),
                       "first_balanced_observation_delay_s": balanced[0] - released if balanced else None,
                       "first_balanced_observation_counts": next((counts[1] for sample, counts in observed
                           if balanced and sample["elapsed_s"] == balanced[0]), None),
                       "first_return_delay_s": events[0]["start_s"] - released if events else None,
                       "returns": events,
                       "maximum_population_sample_gap_s": max((later[0]["elapsed_s"] - earlier[0]["elapsed_s"]
                           for earlier, later in zip(observed, observed[1:])), default=None)})
    return result


def analyze_recovery(root, baseline="random_no_return", treatment="kv_only_workaware", start_s=600, end_s=None):
    trace = json.loads((root / "trace.json").read_text())
    end_s = trace["duration_s"] if end_s is None else end_s
    comparison = paired_window(root, baseline, treatment, start_s, end_s)
    inputs, policies, records_by_policy = {}, {}, {}
    for policy in (baseline, treatment):
        directory = root / ("r0-" + policy)
        payload = {}
        for name in ("requests", "workload", "population", "decisions"):
            path = directory / (name + ".json")
            data = path.read_bytes()
            inputs[str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
            payload[name] = json.loads(data)
        origin = finite_time(payload["workload"]["origin"])
        returns = committed_returns(payload["decisions"], origin)
        records = {record["request_id"]: record for record in payload["requests"]}
        records_by_policy[policy] = records
        worst = sorted((record for record in records.values() if record["model"] == 1
                        and start_s <= record["arrived_at"] - origin < end_s),
                       key=lambda record: record["first_token_at"] - record["arrived_at"], reverse=True)[:10]
        policies[policy] = {"returns": returns, "Code_release_windows": release_windows(
            payload["requests"], payload["population"], returns, origin, trace["duration_s"]),
            "unstable_population_samples_excluded": sum(stable_counts(sample) is None for sample in payload["population"]),
            "worst_Conv_TTFT": [{"request_id": record["request_id"],
                "arrival_s": record["arrived_at"] - origin,
                "first_token_s": record["first_token_at"] - origin,
                "TTFT_s": record["first_token_at"] - record["arrived_at"],
                "first_token_precedes_all_returns": (all(record["first_token_at"] - origin < event["start_s"]
                                                         for event in returns) if returns else None)} for record in worst]}
    marks = {mark["request_id"]: mark for mark in trace["requests"]}
    cohorts = {}
    for window in policies[treatment]["Code_release_windows"]:
        if not window["returns"] or window["overlapping_Code_at_completion"]:
            continue
        resumed = window["returns"][0]["resume_s"]
        identifiers = {identifier for identifier, mark in marks.items() if mark["model"] == 1
                       and max(start_s, resumed) <= mark["arrival_s"] < min(end_s, window["window_end_s"])}
        cohorts[window["Code_request_id"]] = {"arrival_window_s": [max(start_s, resumed), min(end_s, window["window_end_s"])],
            "comparison": compare_group(records_by_policy[baseline], records_by_policy[treatment],
                                        identifiers, comparison["deadline_s"])}
    return {"status": "complete_paired_raw_return_recovery_diagnostics", "source_sha256": inputs,
            "baseline": baseline, "treatment": treatment, "primary_window_comparison": comparison,
            "policies": policies, "matched_post_return_arrival_cohorts": cohorts,
            "scope": "descriptive_event_alignment_not_a_causal_return_counterfactual",
            "caveats": ["Client completion is not the exact allocator release timestamp.",
                "First balanced observation is not proof of sustained balance; sample gaps are reported.",
                "Post-return cohorts are posthoc and share request IDs, not policy-specific populations.",
                "TTFT endpoints preceding returns cannot be improved by those returns.",
                "Few Code requests do not statistically confirm Code P95/P99 or unique return benefit.",
                "Ownership, physical-budget and graph verification remain the campaign verifier's responsibility."]}


def campaign_result(root, state):
    if state.get("phase") != "early_return_lower_rate_SHA_verified_analyzed" or not state.get("GPU_experiment_complete"):
        raise ValueError("Wait for successful full SHA archive and independent campaign verification")
    raw = root / ("live-trace-" + str(state["job"]))
    return {"job": state["job"], "recorded_at": datetime.now().astimezone().isoformat(),
            "phases": {phase: analyze_recovery(raw / phase) for phase in ("cache-fixed-original", "lower-code-rate")}}


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", type=Path)
    source.add_argument("--campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    if args.wait and args.campaign is None:
        parser.error("Only the local campaign observer may wait")
    if args.output.exists():
        parser.error("Preserve prior diagnostic output")
    if args.campaign:
        while True:
            state = json.loads((args.campaign / "campaign-state.json").read_text())
            print(json.dumps({"at": datetime.now().astimezone().isoformat(), "phase": state["phase"]}), flush=True)
            if state.get("phase") == "diagnosis_required_no_automatic_duplicate_submission":
                raise RuntimeError("Campaign failed; preserve raw and diagnose before return analysis")
            if state.get("GPU_experiment_complete") or not args.wait:
                break
            time.sleep(300)
        result = campaign_result(args.campaign, state)
    else:
        result = analyze_recovery(args.root)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "status": "analysis_complete_not_research_goal_complete"}), flush=True)


if __name__ == "__main__":
    main()
