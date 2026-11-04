import argparse
import hashlib
import json
import math
from pathlib import Path

from verify_benefit import verify_runtime_control
from verify_roundtrip_transaction import verify as verify_transaction
from verify_trace import capture_entries, verify_prompt_bank


def validate_records(trace, records):
    marks = sorted(trace["requests"], key=lambda mark: (mark["model"], mark["request_id"]))
    if len(marks) != 3 or sum(mark["model"] == 1 for mark in marks) != 2:
        raise ValueError("Require the unchanged two-Conv/one-Code control trace")
    expected = [(mark, mode, source) for mark in marks
                for mode in (("serial", "boundary_noop", "paused_roundtrip")
                             if mark["model"] == 1 else ("serial",)) for source in (0, 1)]
    if len(records) != len(expected) or len({record["request_id"] for record in records}) != len(expected):
        raise ValueError("Missing or duplicate full control requests")
    references, trials = {}, []
    for (mark, mode, source), record in zip(expected, records):
        times = [record[name] for name in ("arrived_at", "submitted_at", "first_token_at", "completed_at")]
        if (record["model"] != mark["model"] or record["rank_initial"] != source
                or record["role"] != "state-control" or record["prompt_tokens"] != mark["prompt_tokens"]
                or len(record["token_ids"]) != mark["output_tokens"]
                or any(type(token) is not int or token < 0 for token in record["token_ids"])
                or any(isinstance(value, bool) or not math.isfinite(value) for value in times)
                or times != sorted(times)):
            raise ValueError("Control changed original lengths, rank, tenant, tokens or timing")
        key = (mark["request_id"], source)
        if mode == "serial":
            references[key] = record["token_ids"]
        if record["token_ids"] != references[key]:
            raise ValueError("Same-source full serial tokens differ; no tolerance is allowed")
        trials.append({"request_id": mark["request_id"], "mode": mode, "source": source,
                       "original_output_tokens": mark["output_tokens"],
                       "same_source_full_serial_tokens_equal": True})
    return {"status": "passed_same_shape_roundtrip_control_not_benefit", "trials": trials,
            "cross_rank_serial_comparisons": [{"request_id": mark["request_id"],
                "rank_serial_tokens_equal": references[(mark["request_id"], 0)] ==
                                             references[(mark["request_id"], 1)]} for mark in marks],
            "is_algorithm_benefit": False, "mixed_batch_migration_correctness_proven": False,
            "lengths_extended_or_tokens_truncated": False, "greedy_tolerance_acceptance": False}


def verify(root, prompt_bank):
    root = Path(root)
    read = lambda name: json.loads((root / name).read_text())
    if (root / "exit-status").read_text().strip() != "0":
        raise ValueError("State controls did not finish successfully")
    config, trace = read("config.json"), read("trace.json")
    if (not config["pilot"] or config["max_model_len"] != 131072
            or config["cache_bytes"] != 25769803776 or config["kv_budget_mib"] != 10240):
        raise ValueError("State-control configuration differs from the frozen A800 protocol")
    verify_prompt_bank(trace, prompt_bank)
    workers = [[read(f"worker-instance-{model + 1}-dp{rank}.json") for rank in (0, 1)]
               for model in (0, 1)]
    if len({worker["ipc_name"] for ranks in workers for worker in ranks}) != 4:
        raise ValueError("Independent tenant/rank KV namespaces overlap")
    for ranks in workers:
        if ranks[0]["gpu_uuid"] == ranks[1]["gpu_uuid"]:
            raise ValueError("Attention ranks must occupy different GPUs")
        for worker in ranks:
            if ("A800" not in worker["gpu_name"] or worker["checkpoint"] != config["checkpoint"]
                    or worker["max_model_len"] != config["max_model_len"]
                    or worker["kv_cache_memory_bytes"] != config["cache_bytes"]
                    or worker["max_num_batched_tokens"] != 4096
                    or any(worker["groups"][name]["world_size"] != size
                           for name, size in (("tp", 1), ("dp", 2), ("ep", 2)))):
                raise ValueError("Worker checkpoint, A800 geometry or topology differs")
        if len(ranks[0]["expert_ids"]) != len(ranks[1]["expert_ids"]) or not ranks[0]["expert_ids"]:
            raise ValueError("Missing complementary expert layer evidence")
        for left, right in zip(ranks[0]["expert_ids"], ranks[1]["expert_ids"]):
            if set(left) & set(right) or set(left) | set(right) != set(range(128)):
                raise ValueError("Experts are not complementary")
    if any(workers[0][rank]["gpu_uuid"] != workers[1][rank]["gpu_uuid"] for rank in (0, 1)):
        raise ValueError("The two tenants are not colocated")
    runtime = verify_runtime_control(root, workers)
    if runtime is None or runtime["dp_pause_sync_interval"] != 32 or runtime["kv_transport"] != "dp_echo":
        raise ValueError("Require unchanged default32 pause and exact device transport")
    captures = read("state-control-captures.json")
    if len(captures) != 2 or any(len(ranks) != 2 for ranks in captures):
        raise ValueError("Missing four-rank graph evidence")
    if any(not any("FULL" in entry["mode"] and entry["steps"] > 0
                   for entry in capture_entries(capture)) for ranks in captures for capture in ranks):
        raise ValueError("No real CUDA graph decode on a tenant/rank")
    hashes = read("roundtrip-source.json")
    if set(hashes) != {"roundtrip_state_probe.py", "migration_roundtrip_control.py"} or any(
            hashlib.sha256((root / (name + ".executed")).read_bytes()).hexdigest() != expected
            for name, expected in hashes.items()):
        raise ValueError("Executed state-control sources differ from their SHA manifest")
    records, transactions = [], []
    for mark in sorted(trace["requests"], key=lambda item: (item["model"], item["request_id"])):
        for mode in (("serial", "boundary_noop", "paused_roundtrip") if mark["model"] == 1 else ("serial",)):
            for source in (0, 1):
                directory = root / f"state-{mark['request_id']}-{mode}-rank{source}"
                records.append(json.loads((directory / "request.json").read_text()))
                if mode != "serial":
                    control = json.loads((directory / "control.json").read_text())
                    if (control["model"] != mark["model"] or control["source"] != source
                            or control["mode"] != mode or control["budgets_gib"] != [10, 10]
                            or control["inference_between_roundtrip_hops"] is not False):
                        raise ValueError("Paused transaction identity or controlled budget differs")
                    transactions.append(verify_transaction(directory))
    rebuilt = validate_records(trace, records)
    if rebuilt != read("state-control-summary.json"):
        raise ValueError("Independent full-output state-control summary differs")
    return {"status": "full_same_shape_controls_independently_verified_not_benefit",
            "controls": rebuilt, "transactions": transactions, "graph_ranks": 4,
            "trace_sha256": hashlib.sha256((root / "trace.json").read_bytes()).hexdigest(),
            "is_algorithm_benefit": False, "mixed_batch_migration_correctness_proven": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--prompt-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root, args.prompt_bank)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
