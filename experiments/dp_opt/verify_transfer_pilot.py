import argparse
import json
import math
import statistics
from pathlib import Path

from transfer_pilot import pilot_cases
from verify_benefit import verify_ownership, verify_runtime_control
from verify_trace import capture_entries


def verify(root, marks_path):
    read = lambda name: json.loads((root / name).read_text())
    config, summary = read("config.json"), read("transfer-summary.json")
    if summary["status"] != "transfer_pilot_passed" or (root / "exit-status").read_text().strip() != "0":
        raise ValueError("Transfer pilot did not complete")
    marks = json.loads(marks_path.read_text())
    cases = read("transfer-cases.json")
    transports = tuple(config.get("transfer_transports", ("host_file", "dp_allgatherv")))
    pause_modes = config.get("transfer_pause_modes")
    variants = tuple(pause_modes) if pause_modes else transports
    variant_key = "pause_mode" if pause_modes else "transport"
    if (cases != pilot_cases(marks, transports=transports, pause_modes=pause_modes)
            or summary["cases"] != len(cases) or summary["chunk_mib"] != 32):
        raise ValueError("Transfer pilot changed its predeclared cases")
    if (config["max_model_len"] != 131072 or config["cache_bytes"] != 24 * 1024**3
            or config["kv_budget_mib"] != 11328 or config["allocator_gpu_utilization"] != .98):
        raise ValueError("Transfer pilot serving configuration changed")
    workers = [[read(f"worker-instance-{model + 1}-dp{rank}.json") for rank in (0, 1)] for model in (0, 1)]
    control = verify_runtime_control(root, workers)
    if control["dp_pause_sync_interval"] != 32 or len({worker["ipc_name"] for ranks in workers for worker in ranks}) != 4:
        raise ValueError("Transfer pilot changed default pause or tenant KV independence")
    for ranks in workers:
        for rank, worker in enumerate(ranks):
            if ("A800" not in worker["gpu_name"] or worker["gpu_uuid"] != workers[0][rank]["gpu_uuid"]
                    or worker["max_model_len"] != 131072 or worker["kv_cache_memory_bytes"] != 24 * 1024**3
                    or any(worker["groups"][name]["world_size"] != size for name, size in (("dp", 2), ("ep", 2), ("tp", 1)))):
                raise ValueError("Transfer pilot worker topology differs")
    references = {(item["model"], item["prompt_tokens"], len(item["token_ids"])): item
                  for item in read("transfer-references.json")}
    trials = read("transfer-trials.json")
    if len(trials) != len(cases):
        raise ValueError("Transfer pilot omitted cases")
    budgets = [11328 / 1024] * 2
    for index, (trial, case) in enumerate(zip(trials, cases)):
        if trial["case"] != case:
            raise ValueError("Transfer pilot case order differs")
        before, after, event, request = (trial[name] for name in ("before", "after", "event", "request"))
        verify_ownership(before, budgets)
        verify_ownership(after, budgets)
        model, source = case["model"], case["source"]
        old = next(item for item in before[model][source]["requests"] if item["id"] == event["request_id"])
        moved = next(item for item in after[model][1 - source]["requests"] if item["id"] == event["request_id"])
        if (not old["movable"] or old["output_tokens"] < 32 or moved["output_tokens"] != old["output_tokens"]
                or any(item["id"] == old["id"] for item in after[model][source]["requests"])
                or len(moved["blocks"]) != len(old["blocks"])):
            raise ValueError("Transfer pilot did not move mature unique request ownership")
        for rank in (0, 1):
            if any(before[1 - model][rank][name] != after[1 - model][rank][name]
                   for name in ("mapped_bytes", "pages", "requests")):
                raise ValueError("Transfer pilot changed another tenant's ownership")
        if (event["kv_bytes"] != len(old["blocks"]) * before[model][source]["block_bytes"]
                or len(event["sha256"]) != 64 or event["source"] != source
                or event["destination"] != 1 - source or event["model"] != model
                or any(peak > cap + 1e-6 for peak, cap in zip(event["copy_peak_gib"], budgets))):
            raise ValueError("Transfer pilot KV bytes, integrity or copy peak differs")
        mark = case["mark"]
        key = (model, mark["prompt_tokens"], mark["output_tokens"])
        if (request["token_ids"] != references[key]["token_ids"] or request["prompt_tokens"] != key[1]
                or len(request["token_ids"]) != key[2] or request["model"] != model
                or request["request_id"] != f"transfer-{index}"
                or len(old["blocks"]) != math.ceil(sum(key[1:]) / before[model][source]["block_size"])):
            raise ValueError("Transfer pilot output differs from serial reference")
        times = [trial["pause_s"], trial["resume_s"], trial["paused_transaction_s"], event["elapsed_s"],
                 *event["stage_s"].values()]
        times.extend(value for stages in event["worker_transfer_stages_s"].values() for value in stages.values())
        if any(not math.isfinite(value) or value < 0 for value in times):
            raise ValueError("Transfer pilot timings are not finite and nonnegative")
        expected = {"dp_allgatherv": "existing_dp_allgatherv_device_payload_host_integrity_readback",
                    "dp_echo": "existing_dp_allgatherv_full_device_bytewise_echo",
                    "host_file": "host_staged_file_not_nvlink_optimized"}[case["transport"]]
        if event["transport"] != expected:
            raise ValueError("Transfer pilot mislabeled transport")
        if case["transport"] == "dp_echo" and (event.get("integrity") != "device_echo_full_bytewise"
                or event.get("bitwise_verified") is not True
                or event.get("sha256_scope") != "verified_integrity_receipt_not_KV_content"):
            raise ValueError("Transfer pilot echo lacks exact bytewise proof or mislabels receipt as content SHA")
        if pause_modes:
            timestamps = [trial.get(name) for name in
                          ("pause_started_at", "resume_started_at", "resume_completed_at")]
            if any(value is None or not math.isfinite(value) or value < 0 for value in timestamps):
                raise ValueError("Pause pilot lacks finite control timestamps")
            start, resume, finish = timestamps
            if (not start <= resume <= finish
                    or not math.isclose(resume - start, trial["pause_s"] + trial["paused_transaction_s"], abs_tol=1e-5)
                    or not math.isclose(finish - resume, trial["resume_s"], abs_tol=1e-5)):
                raise ValueError("Pause pilot timestamps disagree with control timings")
            receipts = trial.get("pause_receipts")
            if case["pause_mode"] == "default32_boundary":
                if receipts is None or len(receipts) != 2 or any(len(ranks) != 2 for ranks in receipts):
                    raise ValueError("Boundary pause requires receipts from all four ranks")
                for ranks in receipts:
                    for rank, receipt in enumerate(ranks):
                        times = [receipt.get(name) for name in ("armed_at", "reached_at", "callback_at")]
                        step = receipt.get("boundary_step_counter")
                        if (any(value is None or not math.isfinite(value) for value in times)
                                or receipt.get("rank") != rank or not isinstance(step, int)
                                or isinstance(step, bool) or step <= 0 or step % 32
                                or not start <= times[0] <= times[1] <= times[2] <= resume):
                            raise ValueError("Boundary pause receipt lacks default32 consensus or valid timestamps")
            elif receipts is not None:
                raise ValueError("Native pause cannot claim boundary receipts")
    captures = read("transfer-captures.json")
    if len(captures) != 2 or any(len(ranks) != 2 for ranks in captures):
        raise ValueError("Transfer pilot requires all four rank capture records")
    for ranks in captures:
        for rank in ranks:
            if not any("FULL" in entry["mode"] and entry["steps"] > 0 for entry in capture_entries(rank)):
                raise ValueError("Transfer pilot lacks each rank's real decode graph")
    pairs = []
    for target in (2560, 4608, 8192):
        for repeat in range(3):
            for model in (0, 1):
                pair = {trial["case"][variant_key]: trial for trial in trials
                        if (trial["case"]["target_tokens"], trial["case"]["repeat"], trial["case"]["model"])
                        == (target, repeat, model)}
                host, peer = (pair[name]["event"]["elapsed_s"] for name in variants)
                pairs.append({"target_tokens": target, "repeat": repeat, "model": model,
                              "baseline_transport": variants[0], "treatment_transport": variants[1],
                              "kv_bytes": pair[variants[0]]["event"]["kv_bytes"],
                              "baseline_s": host, "treatment_s": peer,
                              "peer_reduction_fraction": 1 - peer / host if host > 0 else None})
    pause_costs = []
    if pause_modes:
        for trial in trials:
            model = trial["case"]["model"]
            reached = (min(receipt["reached_at"] for receipt in trial["pause_receipts"][model])
                       if trial["case"]["pause_mode"] == "default32_boundary" else trial["pause_started_at"])
            cadence = trial["request"].get("cadence", [])
            gaps = [later["time"] - earlier["time"] for earlier, later in zip(cadence, cadence[1:])]
            if not gaps or any(not math.isfinite(value) or value < 0 for value in gaps):
                raise ValueError("Pause pilot lacks complete client token gap evidence")
            before_tokens = max((entry["tokens"] for entry in cadence if entry["time"] <= trial["pause_started_at"]), default=0)
            reached_tokens = max((entry["tokens"] for entry in cadence if entry["time"] <= reached), default=0)
            pause_costs.append({"case": trial["case"], "incoming_control_delay_s":
                                trial["resume_completed_at"] - trial["pause_started_at"],
                                "active_tenant_blocking_upper_s": trial["resume_completed_at"] - reached,
                                "client_progress_during_handshake_tokens": max(0, reached_tokens - before_tokens),
                                "max_client_token_gap_s": max(gaps)})
    return {"status": "transfer_pilot_raw_verified", "cases": len(trials), "paired_costs": pairs,
            "baseline_transport": variants[0], "treatment_transport": variants[1],
            "baseline_mean_s": statistics.mean(pair["baseline_s"] for pair in pairs),
            "treatment_mean_s": statistics.mean(pair["treatment_s"] for pair in pairs),
            "pause_costs": pause_costs,
            "scope": "bounded_payload_full_integrity_single_request_cost_not_net_trace_benefit"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--marks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root, args.marks)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
