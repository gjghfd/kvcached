import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from verify_benefit import percentile, verify_ownership


def capture_entries(value):
    if isinstance(value, list):
        return [entry for child in value for entry in capture_entries(child)]
    if isinstance(value, dict) and "histogram" in value:
        return value["histogram"]
    raise ValueError("Unexpected collective capture evidence shape")


def request_metrics(requests, origin, duration, deadline):
    queue = [request.get("reserved_at", request["submitted_at"]) - request["arrived_at"] for request in requests]
    ttft = [request["first_token_at"] - request["arrived_at"] for request in requests]
    latency = [request["completed_at"] - request["arrived_at"] for request in requests]
    tpot = [(request["completed_at"] - request["first_token_at"]) / (len(request["token_ids"]) - 1)
            for request in requests if len(request["token_ids"]) > 1]
    makespan = max(request["completed_at"] for request in requests) - origin
    output_tokens = sum(len(request["token_ids"]) for request in requests)
    deadline_met = sum(value <= deadline for value in latency)
    return {"mean_queue_s": statistics.mean(queue), "p95_queue_s": percentile(queue, .95),
            "p99_queue_s": percentile(queue, .99), "mean_ttft_s": statistics.mean(ttft),
            "p95_ttft_s": percentile(ttft, .95), "p99_ttft_s": percentile(ttft, .99),
            "mean_latency_s": statistics.mean(latency), "p99_latency_s": percentile(latency, .99),
            "mean_tpot_s": statistics.mean(tpot) if tpot else None, "makespan_s": makespan,
            "drain_s": max(0, makespan - duration), "arrival_window_s": duration,
            "deadline_s": deadline, "deadline_met_requests": deadline_met,
            "deadline_goodput_req_s": deadline_met / duration, "completed_requests": len(requests),
            "completed_within_arrival_window": sum(request["completed_at"] <= origin + duration for request in requests),
            "output_tokens": output_tokens, "output_tokens_per_complete_drain_s": output_tokens / makespan}


def control_metrics(decisions):
    direct = [decision for decision in decisions if decision["path"] == "direct_fastpath"]
    staged = [decision for decision in decisions if decision["path"] == "blocked_staged_ilp"]
    rebalances = [decision for decision in decisions if decision["path"] == "release_rebalance_ilp"]
    controlled = staged + rebalances
    return {"direct_admissions": len(direct),
            "direct_control_s": sum(decision["end_to_end_control_s"] for decision in direct),
            "staged_attempts": len(staged), "staged_admissions": sum(decision["admitted"] for decision in staged),
            "staged_control_s": sum(decision["end_to_end_control_s"] for decision in staged),
            "pause_call_s": sum(decision["pause_call_s"] for decision in controlled),
            "solve_s": sum(decision["solve_s"] for decision in controlled),
            "preplan_s": sum(decision.get("preplan_s", 0) for decision in controlled),
            "release_rebalance_attempts": len(rebalances),
            "release_rebalance_control_s": sum(decision["end_to_end_control_s"] for decision in rebalances),
            "total_control_s_including_preplan": sum(decision["end_to_end_control_s"]
                                                     + decision.get("preplan_s", 0) for decision in decisions),
            "active_blocking_upper_request_s": sum(decision.get("active_blocking_upper_request_s") or 0
                                                   for decision in controlled),
            "resume_s": sum(decision.get("resume_s", 0) for decision in controlled),
            "migration_elapsed_s": sum(move["elapsed_s"] for decision in controlled for move in decision["moves"]),
            "scope": "direct_control_includes_RPC_and_reservation_ack_not_GPU_pause"}


def paired_summary_comparisons(trials, baselines, treatment):
    comparisons = []
    for repeat in sorted({trial["repeat"] for trial in trials}):
        by_policy = {trial["policy"]: trial for trial in trials if trial["repeat"] == repeat}
        if treatment not in by_policy:
            continue
        for policy in baselines:
            if policy not in by_policy or policy == treatment:
                continue
            before, after = by_policy[policy], by_policy[treatment]
            comparison = {name + "_reduction_fraction": 1 - after[name] / before[name]
                          for name in ("mean_queue_s", "p99_queue_s", "mean_ttft_s", "mean_latency_s", "drain_s")
                          if before[name] > 0}
            comparison.update({"repeat": repeat, "baseline": policy, "treatment": treatment,
                               "deadline_met_request_difference": after["deadline_met_requests"] - before["deadline_met_requests"]})
            comparisons.append(comparison)
    return comparisons


def verify_preplan_decision(decision):
    elapsed, preplan = decision["end_to_end_control_s"], decision.get("preplan_s", 0)
    if any(not math.isfinite(value) or value < 0 for value in (elapsed, preplan)):
        raise ValueError("Invalid measured control/preplan elapsed time")
    including = decision.get("end_to_end_control_s_including_preplan")
    if including is not None and not math.isclose(including, elapsed + preplan, abs_tol=1e-8):
        raise ValueError("Control elapsed omits or duplicates preplan cost")
    if preplan == 0 or decision.get("pause_call_s", 0) == 0 or decision["path"] == "release_rebalance_ilp":
        return
    receipts = decision.get("pause_receipts")
    if (decision["solve_s"] != 0 or receipts is None or len(receipts) != 2
            or any(len(ranks) != 2 for ranks in receipts)):
        raise ValueError("Prepared trace decision lacks default boundary receipts or solved inside pause")
    for ranks in receipts:
        for rank, receipt in enumerate(ranks):
            times = [receipt[field] for field in ("armed_at", "reached_at", "callback_at")]
            step = receipt["boundary_step_counter"]
            if (receipt["rank"] != rank or type(step) is not int or step <= 0 or step % 32
                    or any(not math.isfinite(value) for value in times) or times != sorted(times)):
                raise ValueError("Prepared trace pause did not use default32 consensus boundary")
    if decision["moves"] and decision.get("active_blocking_upper_request_s") is None:
        raise ValueError("Prepared migration lacks measured rank-weighted old blocking cost")


def verify_prompt_bank(trace, prompt_bank):
    real_marks = [mark for mark in trace["requests"] if "prompt_key" in mark]
    if not real_marks:
        return
    if prompt_bank is None:
        raise ValueError("Real agentic trace requires its frozen prompt bank")
    payload = prompt_bank.read_bytes()
    if hashlib.sha256(payload).hexdigest() != trace.get("prompt_bank_sha256"):
        raise ValueError("Prompt bank checksum differs from frozen trace")
    prompts = json.loads(payload)["prompts"]
    for mark in real_marks:
        entry = prompts.get(mark["prompt_key"])
        if (entry is None or len(entry["token_ids"]) != mark["prompt_tokens"]
                or entry["prompt_tokens"] != mark["prompt_tokens"]
                or entry["traj_id"] != mark["source_traj_id"]):
            raise ValueError("Real prompt length or trajectory source differs")


def verify_journals(directory):
    for name in ("requests", "decisions", "population", "headroom-arrivals"):
        journal = directory / f"{name}.jsonl"
        if not journal.exists():
            continue
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        final = json.loads((directory / f"{name}.json").read_text())
        canonical = lambda records: sorted(json.dumps(record, sort_keys=True, separators=(",", ":"))
                                           for record in records)
        if canonical(events) != canonical(final):
            raise ValueError("Incremental journal differs from final evidence: " + name)


def verify_paired_tokens(config, token_maps, references):
    policies = config["policies"]
    if not policies or len(policies) != len(set(policies)):
        raise ValueError("Require unique configured reference policies")
    reference_policy = config.get("reference_policy", "no_move" if "no_move" in policies else policies[0])
    if reference_policy not in policies:
        raise ValueError("Token reference policy is not a configured trial")
    for repeat in range(config["repeats"]):
        baseline = token_maps[(repeat, reference_policy)]
        for policy in policies:
            if token_maps[(repeat, policy)] != baseline:
                raise ValueError("Paired policy outputs differ")
        for reference in references:
            if baseline[reference["trace_request_id"]] != reference["token_ids"]:
                raise ValueError("Full serial reference differs")
    return reference_policy


def verify(root, prompt_bank=None):
    read = lambda name: json.loads((root / name).read_text())
    if (root / "exit-status").read_text().strip() != "0" or read("summary.json")["status"] != "passed":
        raise ValueError("Trace allocation did not complete successfully")
    config, trace = read("config.json"), read("trace.json")
    lookup = None
    placement_fit = None
    if config.get("revalidate_prepared") or any(policy in config["policies"] for policy in ("kv_only_rebalance", "kv_only_amortized", "kv_only_headroom", "kv_only_workaware")):
        placement_fit = read("placement-fit.json")
        fit_digest = hashlib.sha256((root / "placement-fit.json").read_bytes()).hexdigest()
        if not any(line.split(maxsplit=1)[0] == fit_digest and line.split(maxsplit=1)[1] == config["fit"]
                   for line in (root / "source.sha256").read_text().splitlines()):
            raise ValueError("Archived candidate placement fit differs from frozen source SHA")
    if config.get("revalidate_prepared"):
        if not config.get("preplan") or read("runtime-control.json").get("candidate_revalidation_enabled") is not True:
            raise ValueError("Candidate revalidation configuration/runtime control differs")
    if "kv_only_rebalance" in config["policies"] and read("runtime-control.json").get("release_rebalance_policy") != "kv_only_rebalance":
        raise ValueError("Release rebalance configuration/runtime control differs")
    if "kv_only_amortized" in config["policies"] and read("runtime-control.json").get("amortized_rebalance_policy") != "kv_only_amortized":
        raise ValueError("Amortized configuration/runtime control differs")
    if "kv_only_headroom" in config["policies"] and read("runtime-control.json").get("headroom_rebalance_policy") != "kv_only_headroom":
        raise ValueError("Headroom configuration/runtime control differs")
    if any(policy in config["policies"] for policy in ("kv_only_amortized", "kv_only_headroom")):
        verify_amortized_runtime(root, headroom="kv_only_headroom" in config["policies"])
    if config.get("layout_cost_lookup") is not None:
        from layout_cost_pilot import load_verified_lookup

        lookup = load_verified_lookup(root / "layout-cost-lookup.json", config["layout_cost_lookup_sha256"],
                                      root / "layout-cost-source")
    if config.get("measured_control_config") is not None:
        from freeze_measured_control import load_control

        control = load_control(root / "control-config.json", config["measured_control_config_sha256"])
        if (any(config.get(name) != value for name, value in control["parameters"].items())
                or control["layout_lookup_sha256"] != config.get("layout_cost_lookup_sha256")
                or lookup is None or control["layout_source_files_sha256"] != lookup["source_files_sha256"]
                or not config.get("preplan") or not config.get("wait_aware")
                or read("runtime-control.json")["kv_transport"] != "dp_echo"):
            raise ValueError("Frozen measured control differs from serving settings/lookup")
    verify_prompt_bank(trace, prompt_bank)
    if "kv_only_workaware" in config["policies"]:
        from workaware_pilot import load_verified_fit
        from workaware_evidence import verify_runtime

        calibrated = load_verified_fit(root / "workaware-fit.json", config["workaware_fit_sha256"],
                                       root / "workaware-cost-source")
        if (config["workaware_fit_data"] != calibrated
                or read("runtime-control.json").get("workaware_rebalance_policy") != "kv_only_workaware"):
            raise ValueError("Workaware serving differs from independently verified frozen calibration")
        verify_runtime(root)
    order = read("order.json")
    expected_trials = {(repeat, policy) for repeat in range(config["repeats"]) for policy in config["policies"]}
    if len(order) != len(expected_trials) or {(item["repeat"], item["policy"]) for item in order} != expected_trials:
        raise ValueError("Missing, duplicate or unknown policy trial")
    marks = {request["request_id"]: request for request in trace["requests"]}
    if (len(marks) != len(trace["requests"]) or config["max_model_len"] <= 0
            or trace.get("max_model_len", config["max_model_len"]) > config["max_model_len"]
            or any(mark["prompt_tokens"] + mark["output_tokens"] > config["max_model_len"]
                   for mark in marks.values())):
        raise ValueError("Duplicate trace IDs or unexpected model limit")
    workers = [[read(f"worker-instance-{model + 1}-dp{rank}.json") for rank in (0, 1)] for model in (0, 1)]
    if len({worker["ipc_name"] for ranks in workers for worker in ranks}) != 4:
        raise ValueError("KV namespaces overlap")
    for ranks in workers:
        if ranks[0]["gpu_uuid"] == ranks[1]["gpu_uuid"]:
            raise ValueError("Attention DP ranks share a physical GPU")
        for worker in ranks:
            if worker["max_model_len"] != config["max_model_len"] or worker["dp_pause_sync_interval"] != 32:
                raise ValueError("Worker length or pause config differs")
            if any(worker["groups"][name]["world_size"] != size for name, size in (("tp", 1), ("dp", 2), ("ep", 2))):
                raise ValueError("Worker topology differs")
        for left, right in zip(ranks[0]["expert_ids"], ranks[1]["expert_ids"]):
            if set(left) & set(right) or set(left) | set(right) != set(range(128)):
                raise ValueError("Experts not complementary")
    if any(workers[0][rank]["gpu_uuid"] != workers[1][rank]["gpu_uuid"] for rank in (0, 1)):
        raise ValueError("Tenant ranks not colocated")
    trials, token_maps = [], {}
    for item in order:
        directory = root / f"r{item['repeat']}-{item['policy']}"
        verify_journals(directory)
        requests = json.loads((directory / "requests.json").read_text())
        workload = json.loads((directory / "workload.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        by_id = {request["request_id"]: request for request in requests}
        if set(by_id) != set(marks) or len(requests) != len(marks) or workload["requests"] != trace["requests"]:
            raise ValueError("Lost, duplicate, replaced or delayed trace marks")
        origin = workload["origin"]
        for identifier, request in by_id.items():
            mark = marks[identifier]
            if (len(request["token_ids"]) != mark["output_tokens"] or request["prompt_tokens"] != mark["prompt_tokens"]
                    or request["model"] != mark["model"]
                    or request.get("prompt_key") != mark.get("prompt_key")
                    or not math.isclose(request["arrived_at"], origin + mark["arrival_s"], abs_tol=1e-6)):
                raise ValueError("Request length, tenant or frozen arrival differs")
            if not request["arrived_at"] <= request["submitted_at"] <= request["first_token_at"] <= request["completed_at"]:
                raise ValueError("Invalid request timing")
            if not request["submitted_at"] <= request.get("reserved_at", request["submitted_at"]) <= request["first_token_at"]:
                raise ValueError("Invalid full KV reservation timing")
        token_maps[(item["repeat"], item["policy"])] = {identifier: request["token_ids"] for identifier, request in by_id.items()}
        if summary["policy"] != item["policy"] or summary["repeat"] != item["repeat"]:
            raise ValueError("Trial summary identity differs")
        recalculated = request_metrics(requests, origin, trace["duration_s"], config["deadline_s"])
        for name, value in recalculated.items():
            if (summary[name] != value if value is None else
                    not math.isclose(summary[name], value, rel_tol=1e-9, abs_tol=1e-9)):
                raise ValueError("Independent metric mismatch: " + name)
        budgets = workload["capacity_gib"]
        for sample in json.loads((directory / "population.json").read_text()):
            if ("admission_stable_across_observation" in sample
                    and (sample["admission_stable_across_observation"] is not True
                         or sample["admission_generation_before"] != sample["admission_generation_after"])):
                raise ValueError("Trace population spans an admission epoch transition")
            if any(value > budget + 1e-6 for value, budget in zip(sample["mapped_gib"], budgets)):
                raise ValueError("Sampled mapped budget exceeded")
        decisions = json.loads((directory / "decisions.json").read_text())
        migrations = json.loads((directory / "migrations.json").read_text())
        if item["policy"] == "no_move_fair":
            from verify_queue_fairness import verify as verify_fairness

            summary["fairness_verification"] = verify_fairness(directory, config["fairness_queue_threshold_s"])
        elif item["policy"] == "no_move" and (migrations or summary.get("fairness", {}).get("enabled", False)
                or any(decision.get("pause_call_s", 0) != 0 or decision["moves"] for decision in decisions)):
            raise ValueError("Independent queue baseline must not migrate, pause or apply cross-model fairness")
        if migrations != [move for decision in decisions for move in decision["moves"]]:
            raise ValueError("Migration inventory differs from decisions")
        totals = {"moves": len(migrations), "migration_kv_bytes": sum(move["kv_bytes"] for move in migrations),
                  "control_s": sum(decision["end_to_end_control_s"] for decision in decisions),
                  "ilp_attempts": sum(decision["path"] == "blocked_staged_ilp" for decision in decisions)}
        if "control_s_including_preplan" in summary:
            totals["control_s_including_preplan"] = sum(decision["end_to_end_control_s"]
                                                       + decision.get("preplan_s", 0) for decision in decisions)
        for name, value in totals.items():
            if not math.isclose(summary[name], value, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Independent control mismatch: " + name)
        if item["policy"] in ("kv_only_rebalance", "kv_only_amortized", "kv_only_headroom", "kv_only_workaware"):
            rebalances = [decision for decision in decisions if decision["path"] == "release_rebalance_ilp"]
            expected = {"attempts": len(rebalances), "transactions": sum(bool(decision["moves"]) for decision in rebalances),
                        "moves": sum(len(decision["moves"]) for decision in rebalances),
                        "control_s_including_preplan": sum(decision["end_to_end_control_s"] + decision.get("preplan_s", 0)
                                                            for decision in rebalances)}
            if summary.get("release_rebalance") != expected:
                raise ValueError("Release rebalance summary omits control or migration cost")
        if item["policy"] == "kv_only_amortized":
            from amortized_rebalance import verify_amortized_history

            verify_amortized_history(decisions, config)
        if item["policy"] == "kv_only_headroom":
            from headroom_rebalance import verify_headroom_history

            verify_headroom_history(decisions, by_id, config, origin, marks,
                                    json.loads((directory / "headroom-arrivals.json").read_text()))
        if item["policy"] == "kv_only_workaware":
            from verify_workaware_rebalance import verify_history

            verify_history(decisions, by_id, json.loads((directory / "headroom-arrivals.json").read_text()),
                           config, origin, marks)
        for decision in decisions:
            verify_preplan_decision(decision)
            if decision["path"] == "release_rebalance_ilp":
                from verify_rebalance import verify_rebalance

                if item["policy"] not in ("kv_only_rebalance", "kv_only_amortized", "kv_only_headroom", "kv_only_workaware"):
                    raise ValueError("Release rebalance changed a baseline policy")
                verify_rebalance(decision, budgets, placement_fit, lookup, config, by_id)
            if decision.get("preplan_candidate_revalidated"):
                from verify_candidate_revalidation import verify_revalidation

                if placement_fit is None:
                    raise ValueError("Candidate revalidation lacks archived placement calibration")
                verify_revalidation(decision, placement_fit, lookup)
                for check in decision["candidate_revalidations"]:
                    if check["wait_gate"] is not None:
                        options = check["wait_recheck_options"]
                        if (options["copy_seconds_per_gib"] != config["wait_copy_seconds_per_gib"]
                                or options["safety_fraction"] != config["wait_safety_fraction"]
                                or options["boundary_cost"] != {
                                    "incoming_handshake_s": config["wait_incoming_handshake_s"],
                                    "old_transaction_fixed_s": config["wait_old_transaction_fixed_s"],
                                    "preplan_s": config["wait_solve_floor_s"]}):
                            raise ValueError("Candidate wait revalidation differs from frozen measured costs")
            for field in ("observations", "after_admission", "before_admission"):
                if decision.get(field) is not None:
                    verify_ownership(decision[field], budgets)
            for move in decision["moves"]:
                if any(value > budget + 1e-6 for value, budget in zip(move["copy_peak_gib"], budgets)):
                    raise ValueError("Copy-before-release peak exceeded")
                if len(move["sha256"]) != 64 or move["kv_bytes"] <= 0:
                    raise ValueError("Missing migration KV hash")
        captures = json.loads((directory / "captures.json").read_text())
        if len(captures) != 2 or any(len(ranks) != 2 for ranks in captures):
            raise ValueError("Missing tenant/rank capture evidence")
        for ranks in captures:
            for capture in ranks:
                if not any("FULL" in entry["mode"] and entry["steps"] > 0 for entry in capture_entries(capture)):
                    raise ValueError("No real CUDA graph decode evidence on a tenant/rank")
        trials.append({**summary, "control_breakdown": control_metrics(decisions),
                       "tenants": {str(model): request_metrics([request for request in requests if request["model"] == model],
                                                               origin, trace["duration_s"], config["deadline_s"])
                                   for model in (0, 1)}})
    reference_policy = verify_paired_tokens(config, token_maps, read("references.json"))
    comparisons = paired_summary_comparisons(trials, ("no_move", "no_move_fair", "own_model", "kv_only"), "ilp")
    capacity_comparisons = paired_summary_comparisons(trials, ("no_move", "no_move_fair", "own_model"), "kv_only")
    rebalance_comparisons = paired_summary_comparisons(trials, ("no_move", "no_move_fair", "kv_only"), "kv_only_rebalance")
    amortized_comparisons = paired_summary_comparisons(
        trials, ("no_move", "no_move_fair", "kv_only", "kv_only_rebalance"), "kv_only_amortized")
    return {"status": "raw_trace_verified", "requests_per_policy": len(marks), "trials": trials,
            "token_reference_policy": reference_policy,
            "ilp_vs_no_move": next((comparison for comparison in comparisons if comparison["baseline"] == "no_move"), None),
            "paired_comparisons": comparisons, "capacity_first_paired_comparisons": capacity_comparisons,
            "release_rebalance_paired_comparisons": rebalance_comparisons,
            "amortized_rebalance_paired_comparisons": amortized_comparisons,
            "trace_sha256": hashlib.sha256((root / "trace.json").read_bytes()).hexdigest(),
            "scope": "one_trace_initial_comparison_not_multiseed_significance"}


def verify_amortized_runtime(root, headroom=False):
    required = {"amortized_rebalance.py", "ilp_reference.py", "live_controller.py", "live_scheduler.py",
                "live_trace.py", "live_rebalance.py", "verify_trace.py", "verify_rebalance.py",
                "trace-benefit.sbatch", "rebalance-trace.sbatch"}
    manifest = json.loads((root / "runtime-source.json").read_text())
    hashes = {line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
              for line in (root / "source.sha256").read_text().splitlines()}
    if headroom:
        required.add("headroom_rebalance.py")
    if manifest.get("policy") != ("kv_only_headroom" if headroom else "kv_only_amortized") or set(manifest["sources"]) != required:
        raise ValueError("Amortized runtime source manifest is incomplete")
    for name, source in manifest["sources"].items():
        actual = hashlib.sha256((root / "runtime-source" / name).read_bytes()).hexdigest()
        if actual != source["sha256"] or hashes.get(source["path"]) != actual:
            raise ValueError("Amortized runtime archive differs from launch source SHA")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-bank", type=Path)
    args = parser.parse_args()
    result = verify(args.root, args.prompt_bank)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
