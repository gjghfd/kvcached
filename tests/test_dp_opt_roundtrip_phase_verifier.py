import copy
import hashlib
import json

import pytest

from verify_roundtrip_phase import validate_records, verify


def fixture():
    marks = [{"request_id": "code", "model": 0, "prompt_tokens": 100001, "output_tokens": 64},
             {"request_id": "conv-0", "model": 1, "prompt_tokens": 74, "output_tokens": 2048},
             {"request_id": "conv-1", "model": 1, "prompt_tokens": 149, "output_tokens": 2049}]
    records = [{"request_id": f"control-{index}", "model": mark["model"], "rank_initial": source,
                "prompt_tokens": mark["prompt_tokens"], "role": "state-control",
                "token_ids": [source + 1] * mark["output_tokens"], "arrived_at": 0.,
                "submitted_at": .1, "first_token_at": .2, "completed_at": 10.}
               for index, (mark, mode, source) in enumerate(
                   (mark, mode, source) for mark in marks for mode in (
                       ("serial", "boundary_noop", "paused_roundtrip") if mark["model"] == 1 else ("serial",))
                   for source in (0, 1))]
    return {"requests": marks}, records


def test_all_fourteen_full_length_controls_with_strict_same_source_references():
    trace, records = fixture()
    result = validate_records(trace, records)
    assert len(result["trials"]) == 14
    assert result["is_algorithm_benefit"] is False
    assert all(not item["rank_serial_tokens_equal"] for item in result["cross_rank_serial_comparisons"])


@pytest.mark.parametrize("case", ["lost", "duplicate", "truncated", "extended", "token", "rank",
                                 "model", "prompt", "nan", "time", "bool_token"])
def test_reject_missing_or_changed_control_records(case):
    trace, records = fixture()
    records = copy.deepcopy(records)
    if case == "lost":
        records.pop()
    elif case == "duplicate":
        records[-1]["request_id"] = records[0]["request_id"]
    elif case == "truncated":
        records[-1]["token_ids"].pop()
    elif case == "extended":
        records[-1]["token_ids"].append(1)
    elif case == "token":
        records[-1]["token_ids"][100] = 123
    elif case == "rank":
        records[-1]["rank_initial"] = 0
    elif case == "model":
        records[-1]["model"] = 0
    elif case == "prompt":
        records[-1]["prompt_tokens"] += 1
    elif case == "nan":
        records[-1]["first_token_at"] = float("nan")
    elif case == "time":
        records[-1]["submitted_at"] = 20.
    else:
        records[-1]["token_ids"][0] = True
    with pytest.raises(ValueError):
        validate_records(trace, records)


def phase_fixture(root, monkeypatch):
    import verify_roundtrip_phase

    trace, records = fixture()
    config = {"pilot": True, "max_model_len": 131072, "cache_bytes": 25769803776,
              "kv_budget_mib": 10240, "checkpoint": "unchanged-checkpoint"}
    runtime = {"dp_pause_sync_interval": 32, "same_interval_for_all_policies": True,
               "idle_rank_expert_execution_not_disabled": True, "kv_transport": "dp_echo"}
    files = {"config.json": config, "trace.json": trace, "runtime-control.json": runtime,
             "state-control-summary.json": validate_records(trace, records),
             "state-control-captures.json": [[{"histogram": [{"mode": "FULL", "steps": 1}]}]
                                               * 2 for model in (0, 1)]}
    for model in (0, 1):
        for source in (0, 1):
            files[f"worker-instance-{model + 1}-dp{source}.json"] = {
                "ipc_name": f"tenant-{model}-rank-{source}", "gpu_uuid": f"gpu-{source}",
                "gpu_name": "NVIDIA A800-SXM4-80GB", "checkpoint": config["checkpoint"],
                "max_model_len": 131072, "kv_cache_memory_bytes": 25769803776,
                "max_num_batched_tokens": 4096, "dp_pause_sync_interval": 32,
                "groups": {name: {"world_size": size} for name, size in (("tp", 1), ("dp", 2), ("ep", 2))},
                "expert_ids": [list(range(source * 64, (source + 1) * 64))]}
    hashes = {}
    for name in ("roundtrip_state_probe.py", "migration_roundtrip_control.py"):
        payload = name.encode()
        (root / (name + ".executed")).write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    files["roundtrip-source.json"] = hashes
    for name, value in files.items():
        (root / name).write_text(json.dumps(value))
    (root / "exit-status").write_text("0\n")
    index = 0
    for mark in trace["requests"]:
        for mode in (("serial", "boundary_noop", "paused_roundtrip") if mark["model"] == 1 else ("serial",)):
            for source in (0, 1):
                directory = root / f"state-{mark['request_id']}-{mode}-rank{source}"
                directory.mkdir()
                (directory / "request.json").write_text(json.dumps(records[index]))
                if mode != "serial":
                    (directory / "control.json").write_text(json.dumps({"model": mark["model"],
                        "source": source, "mode": mode, "budgets_gib": [10, 10],
                        "inference_between_roundtrip_hops": False}))
                index += 1
    monkeypatch.setattr(verify_roundtrip_phase, "verify_transaction",
                        lambda directory: {"status": "transaction-test-double"})


def test_full_phase_reconstruction_routes_all_eight_paused_transactions(tmp_path, monkeypatch):
    phase_fixture(tmp_path, monkeypatch)
    result = verify(tmp_path, tmp_path / "unused-bank")
    assert len(result["transactions"]) == 8
    assert result["graph_ranks"] == 4
    assert result["mixed_batch_migration_correctness_proven"] is False


@pytest.mark.parametrize("case", ["graph", "source", "namespace", "expert", "topology", "budget", "exit"])
def test_full_phase_rejects_invalid_runtime_evidence(tmp_path, monkeypatch, case):
    phase_fixture(tmp_path, monkeypatch)
    if case == "source":
        (tmp_path / "roundtrip_state_probe.py.executed").write_bytes(b"changed")
    elif case == "exit":
        (tmp_path / "exit-status").write_text("1\n")
    else:
        filename = "state-control-captures.json" if case == "graph" else "worker-instance-1-dp0.json"
        if case == "budget":
            filename = "state-conv-0-boundary_noop-rank0/control.json"
        path = tmp_path / filename
        value = json.loads(path.read_text())
        if case == "graph":
            value[0][0]["histogram"][0]["steps"] = 0
        elif case == "namespace":
            value["ipc_name"] = "tenant-1-rank-0"
        elif case == "expert":
            value["expert_ids"] = []
        elif case == "topology":
            value["groups"]["tp"]["world_size"] = 2
        else:
            value["budgets_gib"] = [11, 11]
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        verify(tmp_path, tmp_path / "unused-bank")
