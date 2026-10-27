import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from agentic_capacity import calibration_rates, capacity_setting
from verify_benefit import verify_ownership, verify_runtime_control
from verify_trace import capture_entries, verify_prompt_bank


def verify(root, inputs):
    read = lambda name: json.loads((root / name).read_text())
    summary, config = read("summary.json"), read("config.json")
    if (root / "exit-status").read_text().strip() != "0" or summary["status"] != "capacity_passed":
        raise ValueError("A800 capacity pilot did not complete successfully")
    if (config["capacity_pilot"] is not True or config["max_model_len"] != 131072
            or config["cache_bytes"] != 25769803776):
        raise ValueError("Capacity pilot length or virtual KV configuration differs")
    trace = read("trace.json")
    verify_prompt_bank(trace, inputs / "prompt-bank.json")
    code_marks = json.loads((inputs / "code-marks.json").read_text())
    conv_marks = json.loads((inputs / "conv-marks.json").read_text())
    verify_prompt_bank({"requests": code_marks, "prompt_bank_sha256": trace["prompt_bank_sha256"]},
                       inputs / "prompt-bank.json")
    code = max(code_marks, key=lambda mark: mark["prompt_tokens"] + mark["output_tokens"])
    conv = min(conv_marks, key=lambda mark: abs(mark["prompt_tokens"] - 3949)
               + abs(mark["output_tokens"] - 659))
    if code["prompt_tokens"] < 100000 or code["prompt_tokens"] + code["output_tokens"] > config["max_model_len"]:
        raise ValueError("Longest Code is not an intact supported 100K request")
    workers = [[read(f"worker-instance-{model + 1}-dp{rank}.json") for rank in (0, 1)] for model in (0, 1)]
    if len({worker["ipc_name"] for ranks in workers for worker in ranks}) != 4:
        raise ValueError("Capacity pilot KV namespaces overlap")
    verify_runtime_control(root, workers)
    for model, ranks in enumerate(workers):
        if ranks[0]["gpu_uuid"] == ranks[1]["gpu_uuid"]:
            raise ValueError("Capacity pilot DP ranks share one physical GPU")
        for rank, worker in enumerate(ranks):
            if ("A800" not in worker["gpu_name"] or worker["max_model_len"] != config["max_model_len"]
                    or worker["kv_cache_memory_bytes"] != config["cache_bytes"]
                    or worker["dp_pause_sync_interval"] != 32
                    or worker["gpu_uuid"] != workers[0][rank]["gpu_uuid"]
                    or worker["checkpoint"] != workers[0][0]["checkpoint"]
                    or any(worker["groups"][name]["world_size"] != size
                           for name, size in (("tp", 1), ("dp", 2), ("ep", 2)))):
                raise ValueError("Capacity pilot A800 topology or runtime differs")
        if (len(ranks[0]["expert_ids"]) != len(ranks[1]["expert_ids"])
                or any(set(left) & set(right) or set(left) | set(right) != set(range(128))
                       for left, right in zip(ranks[0]["expert_ids"], ranks[1]["expert_ids"]))):
            raise ValueError("Capacity pilot experts are not complementary EP shards")
    samples, probes = read("memory-samples.json"), read("long-context-probes.json")
    if len(samples) < 2 or any(right["sampled_at"] <= left["sampled_at"]
                               for left, right in zip(samples, samples[1:])):
        raise ValueError("Capacity memory observations missing or unordered")
    for sample in samples:
        if "ownership_stable_across_memory_query" in sample:
            before = sample["ownership_before_memory_query"]
            after = sample["ownership_after_memory_query"]
            if (sample["ownership_stable_across_memory_query"] is not True or before != after
                    or len(before) != 2 or any(len(ranks) != 2 for ranks in before)
                    or any(sample["mapped_mib"][rank] != sum(before[model][rank]["mapped_bytes"]
                           for model in (0, 1)) / 1024**2 for rank in (0, 1))):
                raise ValueError("Capacity sample crosses ownership transition or mapped geometry differs")
        for rank, memory in enumerate(sample["gpu_memory"]):
            if (memory["gpu_uuid"] != workers[0][rank]["gpu_uuid"] or "A800" not in memory["gpu_name"]
                    or any(not math.isfinite(value) for value in
                           (memory["free_mib"], memory["total_mib"], sample["mapped_mib"][rank]))):
                raise ValueError("Capacity memory observation hardware or values differ")
    if len(probes) != 4 or {(probe["model"], probe["rank"]) for probe in probes} != {
            (model, rank) for model in (0, 1) for rank in (0, 1)}:
        raise ValueError("Capacity pilot requires four distinct allocation probes")
    page_mib = probes[0]["allocator"]["before"]["page_bytes"] / 1024**2
    initial = capacity_setting(samples[:1], code_marks, page_mib, config["capacity_guard_mib"],
                               allocator_gpu_utilization=config["allocator_gpu_utilization"])
    if initial != read("capacity-setting-initial.json"):
        raise ValueError("Initial capacity setting differs from measured memory")
    for probe in probes:
        allocator = probe["allocator"]
        before, during, after = (allocator[name] for name in ("before", "during", "after"))
        for observation in (before, during, after):
            verify_ownership([[observation]])
            if (observation["page_bytes"] != 96 * 1024**2 or observation["block_bytes"] != 1.5 * 1024**2
                    or observation["block_size"] != 16 or observation["rank"] != probe["rank"]):
                raise ValueError("Capacity probe KV geometry differs")
        inventory = lambda observation: (observation["mapped_bytes"], observation["pages"], observation["requests"])
        if inventory(before) != inventory(after) or before["requests"] or probe["rollback_verified"] is not True:
            raise ValueError("Capacity allocation rollback changed page/request ownership")
        if len(during["requests"]) != 1:
            raise ValueError("Capacity probe lacks a unique allocated Code request")
        request = during["requests"][0]
        blocks = math.ceil((code["prompt_tokens"] + code["output_tokens"]) / during["block_size"])
        if (request["id"] != allocator["probe_request_id"] or request["prompt_tokens"] != code["prompt_tokens"]
                or request["max_tokens"] != code["output_tokens"] or len(request["blocks"]) != blocks
                or allocator["blocks"] != request["blocks"]):
            raise ValueError("Capacity allocation probe truncated or changed the longest Code")
        increment = during["mapped_bytes"] - before["mapped_bytes"]
        peak = [value / 1024 for value in samples[0]["mapped_mib"]]
        peak[probe["rank"]] += increment / 1024**3
        if (probe["allocated"] is not True or allocator["allocated"] is not True
                or probe["fits_budget"] is not True or probe["additional_mapped_bytes"] != increment
                or probe["initial_gib"] != [value / 1024 for value in samples[0]["mapped_mib"]]
                or probe["allocation_peak_gib"] != peak
                or max(peak) > initial["kv_budget_mib_per_gpu"] / 1024 + 1e-6):
            raise ValueError("Capacity allocation probe does not fit its declared measured budget")
    requests = read("requests.json")
    if len(requests) != 5 or len({request["request_id"] for request in requests}) != 5:
        raise ValueError("Capacity pilot requests missing or duplicated")
    if sum(request["model"] == 0 for request in requests) != 1 or sum(request["model"] == 1 for request in requests) != 4:
        raise ValueError("Capacity pilot must execute one Code and four Conv requests")
    for request in requests:
        mark = code if request["model"] == 0 else conv
        times = [request[name] for name in ("arrived_at", "submitted_at", "first_token_at", "completed_at")]
        if (request["prompt_tokens"] != mark["prompt_tokens"] or len(request["token_ids"]) != mark["output_tokens"]
                or request["rank_initial"] != request["model"] or request["role"] != "capacity"
                or any(not math.isfinite(value) for value in times) or times != sorted(times)):
            raise ValueError("Capacity pilot request lengths, ranks or timing differ")
    graph = summary["graph"]
    if (len(graph) != 2 or any(len(ranks) != 2 for ranks in graph)
            or any(not any("FULL" in entry["mode"] and entry["steps"] > 0 for entry in capture_entries(capture))
                   for ranks in graph for capture in ranks)):
        raise ValueError("Capacity pilot lacks real FULL decode graph evidence on four ranks")
    setting = capacity_setting(samples, code_marks, page_mib, config["capacity_guard_mib"],
                               allocator_gpu_utilization=config["allocator_gpu_utilization"])
    service = statistics.mean(request["completed_at"] - request["submitted_at"]
                              for request in requests if request["model"] == 1)
    setting["calibration"] = calibration_rates(setting, conv_marks, service, conv["output_tokens"])
    if setting != read("capacity-setting.json") or setting != summary["setting"]:
        raise ValueError("Final capacity budget or calibration rate bracket differs")
    expected_env = (f'export DP_OPT_TRACE_BUDGET_MIB={setting["kv_budget_mib_per_gpu"]}\n'
                    f'export DP_OPT_TARGET_CONV_MIB={setting["target_conv_reserved_mib"]}\n'
                    'export DP_OPT_CALIBRATION_CONV_RATES="'
                    + ' '.join(map(str, setting["calibration"]["conv_rates_rps"])) + '"\n')
    if (root / "capacity-config.env").read_text() != expected_env:
        raise ValueError("Capacity launch environment differs from verified setting")
    return {"status": "raw_a800_capacity_verified", "setting": setting, "memory_samples": len(samples),
            "allocation_probes": 4, "requests": 5,
            "source_sha256": {name: hashlib.sha256((inputs / name).read_bytes()).hexdigest()
                              for name in ("code-marks.json", "conv-marks.json", "prompt-bank.json")},
            "scope": "finite_sample_capacity_and_actual_100K_execution_not_peak_certificate_or_policy_benefit"}


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
