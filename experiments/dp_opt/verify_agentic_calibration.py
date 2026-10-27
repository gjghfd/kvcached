import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from agentic_trace import generate_mixed_trace
from analyze_trace import model_residency
from freeze_agentic_calibration import choose_rate
from verify_benefit import verify_ownership
from verify_trace import capture_entries, control_metrics, request_metrics, verify_journals, verify_prompt_bank


def verify(calibration, inputs, allow_partial=False):
    read = lambda name: json.loads((calibration / name).read_text())
    summary, config = read("calibration-summary.json"), read("config.json")
    exit_status = calibration / "exit-status"
    complete = (exit_status.exists() and exit_status.read_text().strip() == "0"
                and summary["status"] == "passed" and read("summary.json")["status"] == "calibration_passed")
    if not complete and not allow_partial:
        raise ValueError("Open-loop calibration did not completely drain")
    rates = config["calibration_conv_rates"]
    if (len(summary["trials"]) != len(rates) if complete else
            not 0 < len(summary["trials"]) <= len(rates)):
        raise ValueError("Calibration rate coverage differs from declared configuration")
    code_marks = json.loads((inputs / "code-marks.json").read_text())
    conv_marks = json.loads((inputs / "conv-marks.json").read_text())
    bank = inputs / "prompt-bank.json"
    bank_hash = hashlib.sha256(bank.read_bytes()).hexdigest()
    workers = [[read(f"worker-instance-{model + 1}-dp{rank}.json") for rank in (0, 1)] for model in (0, 1)]
    if len({worker["ipc_name"] for ranks in workers for worker in ranks}) != 4:
        raise ValueError("Calibration KV namespaces overlap")
    for ranks in workers:
        if ranks[0]["gpu_uuid"] == ranks[1]["gpu_uuid"]:
            raise ValueError("Calibration DP ranks share a GPU")
        for worker in ranks:
            if worker["max_model_len"] != config["max_model_len"] or worker["dp_pause_sync_interval"] != 32:
                raise ValueError("Calibration serving length or pause config differs")
            if any(worker["groups"][name]["world_size"] != size for name, size in (("tp", 1), ("dp", 2), ("ep", 2))):
                raise ValueError("Calibration topology differs from DP2/EP2/TP1")
        if len(ranks[0]["expert_ids"]) != len(ranks[1]["expert_ids"]) or any(
                set(left) & set(right) or set(left) | set(right) != set(range(128))
                for left, right in zip(ranks[0]["expert_ids"], ranks[1]["expert_ids"])):
            raise ValueError("Calibration experts are not complementary")
    if any(workers[0][rank]["gpu_uuid"] != workers[1][rank]["gpu_uuid"] for rank in (0, 1)):
        raise ValueError("Calibration tenants are not colocated")
    trials = []
    for index, (rate, declared) in enumerate(zip(rates, summary["trials"])):
        directory = calibration / f"calibration-{index}"
        load = lambda name: json.loads((directory / name).read_text())
        trace, workload = load("trace.json"), load("workload.json")
        generated = generate_mixed_trace(code_marks, conv_marks, config["calibration_code_rate"], rate,
                                         config["calibration_duration_s"], config["seed"], config["max_model_len"])
        generated["prompt_bank_sha256"] = bank_hash
        if trace != generated or workload["requests"] != trace["requests"]:
            raise ValueError("Calibration trace differs from declared rates, seed or source marks")
        if (workload["policy"] != "no_move" or workload["repeat"] != index
                or workload["duration_s"] != trace["duration_s"]
                or workload["capacity_gib"] != [config["kv_budget_mib"] / 1024] * 2):
            raise ValueError("Calibration policy, window or budget differs")
        trace_hash = hashlib.sha256((json.dumps(trace, indent=2) + "\n").encode()).hexdigest()
        if workload["trace_sha256"] != trace_hash:
            raise ValueError("Calibration workload hash differs from trace")
        verify_prompt_bank(trace, bank)
        verify_journals(directory)
        marks = {mark["request_id"]: mark for mark in trace["requests"]}
        requests, decisions, population = load("requests.json"), load("decisions.json"), load("population.json")
        if (len(marks) != len(trace["requests"]) or len(requests) != len(marks)
                or {request["request_id"] for request in requests} != set(marks)):
            raise ValueError("Calibration lost, duplicate or replaced requests")
        for request in requests:
            mark = marks[request["request_id"]]
            if (request["model"] != mark["model"] or request["prompt_tokens"] != mark["prompt_tokens"]
                    or len(request["token_ids"]) != mark["output_tokens"]
                    or request.get("prompt_key") != mark.get("prompt_key")
                    or request["source_fragment"] != mark["source_fragment"]
                    or request["source_internal_row"] != mark["source_internal_row"]
                    or not math.isclose(request["arrived_at"], workload["origin"] + mark["arrival_s"], abs_tol=1e-6)):
                raise ValueError("Calibration request differs from frozen mark")
            if not request["arrived_at"] <= request["submitted_at"] <= request["first_token_at"] <= request["completed_at"]:
                raise ValueError("Calibration request timing is invalid")
            if not request["submitted_at"] <= request.get("reserved_at", request["submitted_at"]) <= request["first_token_at"]:
                raise ValueError("Calibration full KV reservation timing is invalid")
        if (len(decisions) != len(marks) or {decision["request_id"] for decision in decisions} != set(marks)
                or load("migrations.json") or any(not (decision["admitted"] or decision.get("submitted", False))
                                                 or decision["moves"] or decision["path"] not in
                                                 ("direct_fastpath", "direct_allocator_wait") for decision in decisions)):
            raise ValueError("Calibration must use only complete no-move admissions")
        for decision in decisions:
            if decision["arriving_model"] != marks[decision["request_id"]]["model"]:
                raise ValueError("Calibration admission tenant differs")
            for field in ("observations", "after_admission"):
                verify_ownership(decision[field], workload["capacity_gib"])
        geometry = decisions[0]["observations"][0][0]
        metrics = request_metrics(requests, workload["origin"], trace["duration_s"], config["deadline_s"])
        actual_summary = load("summary.json")
        if actual_summary != declared["summary"]:
            raise ValueError("Calibration trial summary differs from collected summary")
        for name, value in metrics.items():
            if (actual_summary[name] != value if value is None else
                    not math.isclose(actual_summary[name], value, rel_tol=1e-9, abs_tol=1e-9)):
                raise ValueError("Independent calibration metric mismatch: " + name)
        totals = {"moves": 0, "migration_kv_bytes": 0, "ilp_attempts": 0,
                  "control_s": sum(decision["end_to_end_control_s"] for decision in decisions)}
        if (actual_summary["policy"] != "no_move" or actual_summary["repeat"] != index
                or any(not math.isclose(actual_summary[name], value, rel_tol=1e-9, abs_tol=1e-9)
                       for name, value in totals.items())):
            raise ValueError("Independent calibration control summary differs")
        captures = load("captures.json")
        if len(captures) != 2 or any(len(ranks) != 2 for ranks in captures) or any(
                not any("FULL" in entry["mode"] and entry["steps"] > 0 for entry in capture_entries(capture))
                for ranks in captures for capture in ranks):
            raise ValueError("Calibration lacks actual FULL decode graph evidence")
        if not population or any(right["elapsed_s"] <= left["elapsed_s"]
                                 for left, right in zip(population, population[1:])):
            raise ValueError("Calibration population samples are missing or unordered")
        for sample in population:
            if ("admission_stable_across_observation" in sample
                    and (sample["admission_stable_across_observation"] is not True
                         or sample["admission_generation_before"] != sample["admission_generation_after"])):
                raise ValueError("Calibration population spans an admission epoch transition")
            if (len(sample["mapped_gib"]) != 2 or any(not 0 <= used <= budget + 1e-6
                    for used, budget in zip(sample["mapped_gib"], workload["capacity_gib"]))):
                raise ValueError("Calibration sampled mapped budget exceeded")
            for head in sample["pending_heads"]:
                mark = marks[head["request_id"]]
                logical = math.ceil((mark["prompt_tokens"] + mark["output_tokens"]) / geometry["block_size"]) * geometry["block_bytes"]
                free = (sum(workload["capacity_gib"]) - sum(sample["mapped_gib"])) * 1024**3
                if (head["model"] != mark["model"] or head["incoming_logical_bytes"] != logical
                        or not math.isclose(head["global_free_bytes"], free, abs_tol=1e-3)
                        or head["global_enough_local_blocked"] != (head["local_blocked"] and free >= logical)):
                    raise ValueError("Calibration sampled capacity flag arithmetic differs")
        samples = [sample for sample in population if trace["duration_s"] / 3 <= sample["elapsed_s"] < trace["duration_s"]]
        conv_mib = statistics.mean(sum(sample["reserved_mib"][1]) for sample in samples)
        queue_mean = statistics.mean(max(0, sample["unfinished"] - sample["active"]
                                         + sample.get("capacity_pending", 0)) for sample in samples)
        heads = [head for sample in samples for head in sample["pending_heads"] if head["model"] == 0]
        expected = {"conv_rate_rps": rate, "code_rate_rps": config["calibration_code_rate"],
                    "duration_s": trace["duration_s"], "mean_sampled_conv_reserved_mib": conv_mib,
                    "mean_sampled_external_queue_count": queue_mean, "code_head_samples": len(heads),
                    "global_enough_local_blocked_code_samples": sum(head["global_enough_local_blocked"] for head in heads)}
        if any(not math.isclose(declared[name], value, rel_tol=1e-9, abs_tol=1e-9) for name, value in expected.items()):
            raise ValueError("Independent calibration sample summary differs")
        trials.append({**expected, "trace_sha256": trace_hash, "metrics": metrics,
                       "tenants": {str(model): request_metrics([request for request in requests if request["model"] == model],
                                                               workload["origin"], trace["duration_s"], config["deadline_s"])
                                   for model in (0, 1)},
                       "control": control_metrics(decisions),
                       "residency": model_residency(requests, trace["model_rates_rps"], workload["origin"],
                                                     trace["duration_s"], geometry["block_size"], geometry["block_bytes"]),
                       "capacity_evidence_scope": "sampled_controller_flags_not_actual_allocation_or_copy_feasibility_probes"})
    if complete:
        try:
            decision = choose_rate(calibration, summary)
            guardrails = {"status": "eligible", "decision": decision}
        except ValueError as error:
            guardrails = {"status": "rejected", "reason": str(error)}
    else:
        guardrails = {"status": "not_evaluated_incomplete", "reason": "Cannot freeze from completed prefix of failed calibration"}
    return {"status": "raw_calibration_verified" if complete else "raw_completed_calibration_prefix_verified",
            "calibration_complete": complete, "declared_trials": len(rates), "completed_trials_verified": len(trials),
            "trials": trials, "guardrails": guardrails,
            "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (inputs / "code-marks.json", inputs / "conv-marks.json", bank)},
            "scope": "finite_open_loop_rate_calibration_not_four_policy_benefit_or_stability_certificate"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = verify(args.calibration, args.inputs, allow_partial=args.allow_partial)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
