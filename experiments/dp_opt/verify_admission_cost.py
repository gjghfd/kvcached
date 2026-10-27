import argparse
import json
import math
from pathlib import Path

from admission_cost_pilot import pilot_cases
from live_controller import admission_candidates, allocation_increment, stable_ownership
from verify_benefit import verify_ownership, verify_runtime_control
from verify_trace import capture_entries, verify_prompt_bank


def verify(root, inputs):
    read = lambda name: json.loads((root / name).read_text())
    config, summary, trace = read("config.json"), read("summary.json"), read("trace.json")
    if (summary["status"] != "admission_cost_pilot_passed" or (root / "exit-status").read_text().strip() != "0"
            or not config["admission_cost_pilot"] or config["transfer_pilot"]
            or config["max_model_len"] != 131072 or config["cache_bytes"] != 24 * 1024**3
            or config["kv_budget_mib"] != 11328 or config["allocator_gpu_utilization"] != .98):
        raise ValueError("Admission cost pilot did not complete with its frozen serving settings")
    verify_prompt_bank(trace, inputs / "prompt-bank.json")
    marks = json.loads((inputs / "conv-marks.json").read_text())
    cases = pilot_cases(marks)
    if cases != read("admission-cases.json") or summary["cases"] != len(cases):
        raise ValueError("Admission cost pilot changed predeclared cases")
    code = max((mark for mark in trace["requests"] if mark["model"] == 0),
               key=lambda mark: mark["prompt_tokens"])
    if read("admission-code-mark.json") != code or code["prompt_tokens"] < 100000 or code["output_tokens"] != 64:
        raise ValueError("Admission pilot truncated or replaced longest intact trace Code mark")
    workers = [[read(f"worker-instance-{model + 1}-dp{rank}.json") for rank in (0, 1)] for model in (0, 1)]
    control = verify_runtime_control(root, workers)
    if (control["dp_pause_sync_interval"] != 32 or control["kv_transport"] != "dp_echo"
            or len({worker["ipc_name"] for ranks in workers for worker in ranks}) != 4):
        raise ValueError("Admission pilot changed pause cadence, exact transport or KV namespace")
    for model, ranks in enumerate(workers):
        for rank, worker in enumerate(ranks):
            if ("A800" not in worker["gpu_name"] or worker["gpu_uuid"] != workers[0][rank]["gpu_uuid"]
                    or ranks[0]["gpu_uuid"] == ranks[1]["gpu_uuid"]
                    or worker["checkpoint"] != workers[0][0]["checkpoint"]
                    or worker["max_model_len"] != 131072 or worker["kv_cache_memory_bytes"] != 24 * 1024**3
                    or any(worker["groups"][name]["world_size"] != size
                           for name, size in (("dp", 2), ("ep", 2), ("tp", 1)))):
                raise ValueError("Admission pilot worker topology differs")
    references = {request["model"]: request for request in read("admission-references.json")}
    if set(references) != {0, 1}:
        raise ValueError("Admission pilot lacks both full serial references")
    trials = read("admission-trials.json")
    if len(trials) != len(cases):
        raise ValueError("Admission pilot omitted a cost case")
    budgets, results = [11328 / 1024] * 2, []
    for index, (trial, case) in enumerate(zip(trials, cases)):
        if trial["case"] != case:
            raise ValueError("Admission pilot changed case order")
        before, decision = trial["before"], trial["decision"]
        if [[len(rank["requests"]) for rank in ranks] for ranks in before] != [[0, 0], [case["population"] // 2] * 2]:
            raise ValueError("Admission pilot lacks balanced declared mature Conv population")
        if any(not request["movable"] or request["output_tokens"] < 32
               for ranks in before for rank in ranks for request in rank["requests"]):
            raise ValueError("Admission pilot did not reach mature decode")
        verify_ownership(before, budgets)
        used, direct = admission_candidates(before, 0, code["prompt_tokens"] + 64, budgets)
        geometry = before[0][0]
        logical = math.ceil((code["prompt_tokens"] + 64) / geometry["block_size"]) * geometry["block_bytes"]
        expected = {"initial_gib": used, "global_free_bytes": round((sum(budgets) - sum(used)) * 1024**3),
                    "incoming_logical_bytes": logical, "direct_ranks": direct,
                    "allocation_increment_gib": [allocation_increment(before[0][rank], code["prompt_tokens"] + 64)
                                                 for rank in (0, 1)]}
        if trial["witness"] != expected or direct or expected["global_free_bytes"] < logical:
            raise ValueError("Admission pilot lacks global-enough/local-blocked actual mapped-slot witness")
        for field in ("observations", "before_admission", "after_admission"):
            if decision.get(field) is None:
                raise ValueError("Admission pilot lacks transaction ownership snapshots")
            verify_ownership(decision[field], budgets)
        moves = decision["moves"]
        if (not decision["admitted"] or not decision["submitted"] or not moves
                or decision["actual_allocator_blocked_after_plan"] or decision["reservation_pending"]):
            raise ValueError("Admission pilot did not really reserve Code after migration")
        initial = {request["id"]: (model, rank, request) for model, ranks in enumerate(decision["observations"])
                   for rank, observation in enumerate(ranks) for request in observation["requests"]}
        final = {request["id"]: (model, rank, request) for model, ranks in enumerate(decision["before_admission"])
                 for rank, observation in enumerate(ranks) for request in observation["requests"]}
        if set(initial) != set(final) or len({move["request_id"] for move in moves}) != len(moves):
            raise ValueError("Admission pilot lost or duplicated old request ownership")
        for move in moves:
            model, source, old = initial[move["request_id"]]
            destination_model, destination, moved = final[move["request_id"]]
            if (model != 1 or destination_model != model or move["model"] != model or source != move["source"]
                    or destination != move["destination"] or destination != 1 - source
                    or moved["output_tokens"] != old["output_tokens"] or len(moved["blocks"]) != len(old["blocks"])
                    or move["kv_bytes"] != len(old["blocks"]) * decision["observations"][model][source]["block_bytes"]
                    or move["integrity"] != "device_echo_full_bytewise" or move["bitwise_verified"] is not True
                    or move["sha256_scope"] != "verified_integrity_receipt_not_KV_content"
                    or len(move["sha256"]) != 64 or any(peak > budget + 1e-6
                        for peak, budget in zip(move["copy_peak_gib"], budgets))):
                raise ValueError("Admission pilot lacks full-byte integrity, unique migrated owner or bounded copy peak")
        for identifier, (model, rank, request) in initial.items():
            if identifier not in {move["request_id"] for move in moves} and final[identifier] != (model, rank, request):
                raise ValueError("Admission pilot changed an unmoved old request")
        target = decision["winner"]["target"]
        admitted = decision["after_admission"][0][target]["requests"]
        if (len(admitted) != 1 or admitted[0]["reservation_pending"]
                or len(admitted[0]["blocks"]) * geometry["block_bytes"] != logical):
            raise ValueError("Admission pilot lacks intact physical Code KV reservation on planned rank")
        requests = trial["requests"]
        if (len(requests) != case["population"] + 1 or len({request["request_id"] for request in requests}) != len(requests)
                or sum(request["model"] == 0 for request in requests) != 1):
            raise ValueError("Admission pilot omitted request results")
        for request in requests:
            mark = code if request["model"] == 0 else case["conv_mark"]
            times = [request[field] for field in ("arrived_at", "submitted_at", "first_token_at", "completed_at")]
            if (request["prompt_tokens"] != mark["prompt_tokens"] or len(request["token_ids"]) != mark["output_tokens"]
                    or request["token_ids"] != references[request["model"]]["token_ids"]
                    or any(not math.isfinite(value) for value in times) or times != sorted(times)):
                raise ValueError("Admission pilot request lengths, serial tokens or timing differ")
        receipts = decision["pause_receipts"]
        if case["mode"] == "preplan_boundary":
            if (decision["solve_s"] != 0 or decision["preplan_s"] <= 0 or receipts is None
                    or len(receipts) != 2 or any(len(ranks) != 2 for ranks in receipts)
                    or stable_ownership(before) != stable_ownership(decision["observations"])):
                raise ValueError("Admission pilot failed stable preplan reuse outside pause")
            for ranks in receipts:
                for rank, receipt in enumerate(ranks):
                    times = [receipt[field] for field in ("armed_at", "reached_at", "callback_at")]
                    step = receipt["boundary_step_counter"]
                    if (receipt["rank"] != rank or type(step) is not int or step <= 0 or step % 32
                            or any(not math.isfinite(value) for value in times) or times != sorted(times)):
                        raise ValueError("Admission pilot boundary receipt is not default32 consensus")
        elif receipts is not None or decision["preplan_s"] != 0:
            raise ValueError("Native inside-pause pilot cannot claim preplan boundary receipts")
        code_result = next(request for request in requests if request["model"] == 0)
        gaps = [later["time"] - earlier["time"] for request in requests if request["model"] == 1
                for earlier, later in zip(request["cadence"], request["cadence"][1:])]
        if not gaps or any(not math.isfinite(value) or value < 0 for value in gaps):
            raise ValueError("Admission pilot lacks actual old-client cadence")
        results.append({"case": case, "moves": len(moves), "copy_bytes": sum(move["kv_bytes"] for move in moves),
                        "incoming_control_s": decision["pilot_arrival_to_control_finish_s"],
                        "preplan_s": decision["preplan_s"], "pause_call_s": decision["pause_call_s"],
                        "solve_inside_pause_s": decision["solve_s"],
                        "active_blocking_upper_request_s": decision.get("active_blocking_upper_request_s"),
                        "code_ttft_s": code_result["first_token_at"] - trial["arriving_at"],
                        "old_conv_max_client_gap_s": max(gaps),
                        "old_conv_latency_sum_s": sum(request["completed_at"] - request["arrived_at"]
                                                       for request in requests if request["model"] == 1)})
    captures = read("admission-captures.json")
    if len(captures) != 2 or any(len(ranks) != 2 for ranks in captures) or any(
            not any("FULL" in entry["mode"] and entry["steps"] > 0 for entry in capture_entries(capture))
            for ranks in captures for capture in ranks):
        raise ValueError("Admission pilot lacks real decode graph execution on all four ranks")
    return {"status": "admission_cost_raw_verified", "cases": len(cases),
            "requests": sum(case["population"] + 1 for case in cases), "costs": results,
            "scope": "joint_population_cost_and_real_100K_admission_not_final_Poisson_net_benefit"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root, args.inputs)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
