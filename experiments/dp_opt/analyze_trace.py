import argparse
import collections
import copy
import json
import math
import statistics
from pathlib import Path

from ilp_reference import enumerate_targets
from poisson_trace import generate_trace, source_rows
from verify_trace import capture_entries, verify


def peak_intervals(requests, end_field):
    events = sorted([(request["arrived_at"], 1) for request in requests]
                    + [(request.get("reserved_at", request["submitted_at"]) if end_field == "reserved_at"
                        else request[end_field], -1) for request in requests])
    active, peak = 0, 0
    for _, change in events:
        active += change
        peak = max(peak, active)
    return peak


def plan_signature(plan):
    winner = plan.get("winner")
    return (winner["target"], tuple(winner["moves"])) if winner else None


def paired_request_outcomes(baseline, treatment, identifiers, deadline):
    if not identifiers:
        return {"requests": 0}
    selected = sorted(identifiers)

    def metric(request, name):
        if name == "queue_s":
            return request.get("reserved_at", request["submitted_at"]) - request["arrived_at"]
        if name == "ttft_s":
            return request["first_token_at"] - request["arrived_at"]
        if name == "latency_s":
            return request["completed_at"] - request["arrived_at"]
        return ((request["completed_at"] - request["first_token_at"]) / (len(request["token_ids"]) - 1)
                if len(request["token_ids"]) > 1 else None)

    metrics = {}
    for name in ("queue_s", "ttft_s", "latency_s", "tpot_s"):
        pairs = [(metric(baseline[identifier], name), metric(treatment[identifier], name))
                 for identifier in selected]
        pairs = [pair for pair in pairs if None not in pair]
        if not pairs:
            metrics[name] = None
            continue
        before = statistics.mean(pair[0] for pair in pairs)
        after = statistics.mean(pair[1] for pair in pairs)
        metrics[name] = {"baseline_mean_s": before, "ilp_mean_s": after,
                         "paired_mean_delta_s": after - before,
                         "paired_sum_delta_s": sum(pair[1] - pair[0] for pair in pairs),
                         "reduction_fraction": 1 - after / before if before > 0 else None,
                         "slower_requests": sum(pair[1] > pair[0] + 1e-9 for pair in pairs)}
    baseline_met = sum(metric(baseline[identifier], "latency_s") <= deadline for identifier in selected)
    ilp_met = sum(metric(treatment[identifier], "latency_s") <= deadline for identifier in selected)
    return {"requests": len(selected), "request_ids": selected, "metrics": metrics,
            "baseline_deadline_met": baseline_met, "ilp_deadline_met": ilp_met,
            "deadline_met_difference": ilp_met - baseline_met}


def paired_tenant_outcomes(root, verified, trace, deadline):
    comparisons = []
    code_arrivals = [mark["arrival_s"] for mark in trace["requests"] if mark["model"] == 0]
    for comparison in verified["paired_comparisons"]:
        repeat, policy = comparison["repeat"], comparison["baseline"]
        directory = root / f"r{repeat}-{policy}"
        before = {request["request_id"]: request for request in json.loads((directory / "requests.json").read_text())}
        after = {request["request_id"]: request for request in json.loads(
            (root / f"r{repeat}-ilp/requests.json").read_text())}
        origin = json.loads((directory / "workload.json").read_text())["origin"]
        old_conv = {identifier for identifier, request in before.items() if request["model"] == 1
                    and any(request.get("reserved_at", request["submitted_at"]) <= origin + arrival
                            < request["completed_at"] for arrival in code_arrivals)}
        groups = {"all_requests": set(before),
                  "all_Code": {identifier for identifier, request in before.items() if request["model"] == 0},
                  "all_Conv": {identifier for identifier, request in before.items() if request["model"] == 1},
                  "baseline_resident_Conv_at_Code_arrival": old_conv}
        comparisons.append({"repeat": repeat, "baseline": policy, "treatment": "ilp",
                            "groups": {name: paired_request_outcomes(before, after, identifiers, deadline)
                                       for name, identifiers in groups.items()},
                            "scope": "same_request_ids_full_cost_deltas_baseline_defined_resident_group_not_causal_kernel_attribution"})
    return comparisons


def model_residency(requests, model_rates, origin, duration, block_size, block_bytes):
    results = []
    for model, rate in enumerate(model_rates):
        selected = [request for request in requests if request["model"] == model]
        reservations = [math.ceil((request["prompt_tokens"] + len(request["token_ids"])) / block_size)
                        * block_bytes / 1024**2 for request in selected]
        services = [request["completed_at"] - request.get("reserved_at", request["submitted_at"]) for request in selected]
        intervals = [max(0, min(request["completed_at"], origin + duration)
                           - max(request.get("reserved_at", request["submitted_at"]), origin)) for request in selected]
        mean_kv_seconds = statistics.mean(memory * service for memory, service in zip(reservations, services))
        results.append({"model": model, "nominal_rate_rps": rate, "realized_rate_rps": len(selected) / duration,
                        "requests": len(selected), "mean_resident_s": statistics.mean(services),
                        "mean_reserved_mib_seconds": mean_kv_seconds,
                        "nominal_rate_times_measured_KV_seconds_mib": rate * mean_kv_seconds,
                        "arrival_window_time_average_reserved_mib": sum(memory * interval for memory, interval
                                                                        in zip(reservations, intervals)) / duration,
                        "arrival_window_time_average_resident_count": sum(intervals) / duration,
                        "scope": "logical_full_reservation_not_physical_mapped_pages_or_stability_guarantee"})
    return results


def analyze(root, sample=None, prompt_bank=None):
    verified = verify(root, prompt_bank)
    read = lambda name: json.loads((root / name).read_text())
    trace, config = read("trace.json"), read("config.json")
    source_verified = False
    if sample is not None:
        selection = trace["slice"]
        rows = source_rows(sample, selection["dataset"], trace["max_model_len"], selection["minimum_context"],
                           selection["minimum_output"], selection["maximum_output"])
        regenerated = generate_trace(rows, trace["total_rate_rps"], trace["duration_s"], trace["seed"], trace["max_model_len"])
        if regenerated["requests"] != trace["requests"] or len(rows) != selection["eligible_source_rows"]:
            raise ValueError("Fixed trace does not reproduce from the archived Azure sample")
        source_verified = True
    marks = {mark["request_id"]: mark for mark in trace["requests"]}
    diagnostics = []
    for trial in verified["trials"]:
        directory = root / f"r{trial['repeat']}-{trial['policy']}"
        load = lambda name: json.loads((directory / name).read_text())
        requests, decisions, population = load("requests.json"), load("decisions.json"), load("population.json")
        workload = load("workload.json")
        boundary = workload["origin"] + trace["duration_s"]
        staged = [decision for decision in decisions if decision["path"] == "blocked_staged_ilp"]
        failed = [decision for decision in staged if not decision["admitted"]]
        geometry = decisions[0]["observations"][0][0]
        reservations = [math.ceil((request["prompt_tokens"] + len(request["token_ids"])) / geometry["block_size"])
                        * geometry["block_bytes"] / 1024**2 for request in requests]
        resident = [request["completed_at"] - request.get("reserved_at", request["submitted_at"]) for request in requests]
        mean_kv_seconds = statistics.mean(size * seconds for size, seconds in zip(reservations, resident))
        movement = []
        for decision in staged:
            if decision["moves"]:
                mark = marks[decision["request_id"]]
                observed = decision["observations"]
                used_mib = [sum(observed[model][rank]["mapped_bytes"] for model in (0, 1)) / 1024**2 for rank in (0, 1)]
                free_mib = [budget * 1024 - used for budget, used in zip(workload["capacity_gib"], used_mib)]
                required_blocks = math.ceil((mark["prompt_tokens"] + mark["output_tokens"]) / geometry["block_size"])
                logical_mib = required_blocks * geometry["block_bytes"] / 1024**2
                incremental_mib = [math.ceil(max(0, required_blocks - rank["mapped_free_block_slots"])
                                             / rank["page_block_capacity"]) * rank["page_bytes"] / 1024**2
                                   for rank in observed[mark["model"]]]
                movement.append({"arrival_id": mark["request_id"], "arrival_model": mark["model"],
                                 "prompt_tokens": mark["prompt_tokens"], "output_tokens": mark["output_tokens"],
                                 "control_s": decision["end_to_end_control_s"],
                                 "queue_until_decision_s": decision["elapsed_from_arrival_s"],
                                 "counts_model_order_arrival_first": decision["snapshot"]["counts"],
                                 "initial_mapped_gib": decision["snapshot"]["initial_gib"],
                                 "reserved_new_gib": decision["snapshot"]["new_gib"],
                                 "free_mib": free_mib, "incoming_logical_mib": logical_mib,
                                 "incoming_actual_incremental_mib": incremental_mib,
                                 "global_enough_local_blocked": sum(free_mib) >= logical_mib and
                                 all(increment > free + 1e-6 for increment, free in zip(incremental_mib, free_mib)),
                                 "physical_copy_peak_gib": [move["copy_peak_gib"] for move in decision["moves"]],
                                 "moves": decision["moves"]})
        replays = []
        if trial["policy"] == "ilp":
            for decision in staged:
                if decision.get("snapshot") is None or decision.get("plan") is None:
                    continue
                snapshot = copy.deepcopy(decision["snapshot"])
                snapshot["performance_cost_tables_s"] = [[0] * len(table) for table in snapshot["performance_cost_tables_s"]]
                replay = enumerate_targets(snapshot, max_moves=config["max_moves"], time_limit_s=config["solver_limit_s"])
                original_signature, replay_signature = plan_signature(decision["plan"]), plan_signature(replay)
                replays.append({"arrival_id": decision["request_id"], "original_status": decision["plan"]["status"],
                                "original_action_scope": ("revalidated_original_candidates_not_current_ILP_optimum"
                                    if decision.get("preplan_candidate_revalidated") else "snapshot_bounded_ILP"),
                                "kv_only_status": replay["status"], "original_signature": original_signature,
                                "kv_only_signature": replay_signature, "same_action": original_signature == replay_signature})
        graph = []
        for model, ranks in enumerate(load("captures.json")):
            for rank, capture in enumerate(ranks):
                padding = collections.Counter()
                for entry in capture_entries(capture):
                    if "FULL" in entry["mode"]:
                        padding[entry["padded_tokens"]] += entry["steps"]
                graph.append({"model": model, "rank": rank, "FULL_padded_size_steps": dict(padding)})
        budget_mib = sum(workload["capacity_gib"]) * 1024
        segments = []
        for begin in range(0, math.ceil(trace["duration_s"]), 200):
            end = min(begin + 200, trace["duration_s"])
            selected = [request for request in requests if begin <= request["arrived_at"] - workload["origin"] < end]
            segments.append({"begin_s": begin, "end_s": end, "requests": len(selected),
                             "mean_queue_s": statistics.mean(request.get("reserved_at", request["submitted_at"])
                                                              - request["arrived_at"] for request in selected) if selected else None,
                             "mean_resident_s": statistics.mean(request["completed_at"]
                                                                 - request.get("reserved_at", request["submitted_at"])
                                                                 for request in selected) if selected else None})
        model_rates = trace.get("model_rates_rps", [trace["total_rate_rps"] / 2] * 2)
        residency = model_residency(requests, model_rates, workload["origin"], trace["duration_s"],
                                   geometry["block_size"], geometry["block_bytes"])
        diagnostics.append({"policy": trial["policy"], "repeat": trial["repeat"],
                            "peak_unfinished_exact": peak_intervals(requests, "completed_at"),
                            "peak_waiting_exact": peak_intervals(requests, "reserved_at"),
                            "unfinished_at_arrival_window_end": sum(request["completed_at"] > boundary for request in requests),
                            "waiting_at_arrival_window_end": sum(request.get("reserved_at", request["submitted_at"])
                                                                  > boundary for request in requests),
                            "mean_resident_s": statistics.mean(resident), "mean_reserved_mib": statistics.mean(reservations),
                            "mean_reserved_mib_seconds": mean_kv_seconds,
                            "nominal_rate_times_measured_reserved_seconds_over_budget": sum(item["nominal_rate_times_measured_KV_seconds_mib"]
                                                                                            for item in residency) / budget_mib,
                            "realized_rate_times_measured_reserved_seconds_over_budget": len(requests) / trace["duration_s"] * mean_kv_seconds / budget_mib,
                            "load_scope": "conditional_measured_residence_not_independent_sustainable_rate_or_mapped_page_usage",
                            "arrival_dispatch_lag_max_s": max(request["arrival_dispatch_lag_s"] for request in requests),
                            "population_sample_gap_max_s": max(right["elapsed_s"] - left["elapsed_s"]
                                                                 for left, right in zip(population, population[1:])),
                            "max_sampled_resident_count": max(sum(sum(counts) for counts in item["counts"]) for item in population),
                            "failed_staged_attempts": len(failed),
                            "failed_staged_control_s": sum(decision["end_to_end_control_s"] for decision in failed),
                            "failed_staged_control_s_including_preplan": sum(decision["end_to_end_control_s"]
                                                                             + decision.get("preplan_s", 0) for decision in failed),
                            "cross_tenant_migrations": sum(move["model"] != item["arrival_model"] for item in movement for move in item["moves"]),
                            "migrated_admissions": movement, "same_snapshot_KV_only_replay": replays,
                            "replay_scope": "offline_action_ablation_not_real_counterfactual_latency",
                            "graph": graph, "arrival_segments": segments, "model_residency": residency})
    return {"status": "verified_trace_diagnostics", "source_sample_reproduced": source_verified,
            "paired_tenant_outcomes": paired_tenant_outcomes(root, verified, trace, config["deadline_s"]),
            "nominal_rate_rps": trace["total_rate_rps"], "realized_arrivals_per_window_s": len(marks) / trace["duration_s"],
            "diagnostics": diagnostics, "scope": "single_seed_diagnostics_do_not_attribute_all_delay_to_pause"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--prompt-bank", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.sample, args.prompt_bank)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
