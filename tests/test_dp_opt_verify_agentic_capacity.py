import copy
import importlib
import json
import math
from pathlib import Path

import pytest


@pytest.fixture
def capacity(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    verifier = importlib.import_module("verify_agentic_capacity")
    root, inputs = tmp_path / "capacity", tmp_path / "inputs"
    root.mkdir()
    inputs.mkdir()

    def save(path, value):
        path.write_text(json.dumps(value) + "\n")

    code = {"model": 0, "prompt_tokens": 101675, "output_tokens": 64,
            "prompt_key": "swe", "source_traj_id": "trajectory"}
    conv = {"prompt_tokens": 3949, "output_tokens": 659}
    save(inputs / "code-marks.json", [code])
    save(inputs / "conv-marks.json", [conv])
    save(inputs / "prompt-bank.json", {"prompts": {"swe": {
        "token_ids": [1] * code["prompt_tokens"], "prompt_tokens": code["prompt_tokens"],
        "traj_id": "trajectory"}}})
    save(root / "trace.json", {"requests": [code], "prompt_bank_sha256":
                              verifier.hashlib.sha256((inputs / "prompt-bank.json").read_bytes()).hexdigest()})
    save(root / "config.json", {"capacity_pilot": True, "max_model_len": 131072,
                                "cache_bytes": 25769803776, "capacity_guard_mib": 1024,
                                "allocator_gpu_utilization": .98})
    save(root / "runtime-control.json", {"dp_pause_sync_interval": 32,
                                         "same_interval_for_all_policies": True,
                                         "idle_rank_expert_execution_not_disabled": True})
    for model in (0, 1):
        for rank in (0, 1):
            save(root / f"worker-instance-{model + 1}-dp{rank}.json", {
                "gpu_uuid": f"gpu-{rank}", "gpu_name": "NVIDIA A800", "checkpoint": "qwen",
                "ipc_name": f"model-{model}-rank-{rank}", "max_model_len": 131072,
                "kv_cache_memory_bytes": 25769803776, "dp_pause_sync_interval": 32,
                "groups": {name: {"world_size": size} for name, size in (("tp", 1), ("dp", 2), ("ep", 2))},
                "expert_ids": [list(range(rank * 64, (rank + 1) * 64))]})
    samples = []
    for stamp, free, mapped in ((10, [12000, 12000], [192, 192]),
                                (20, [2500, 10400], [9696, 1920]),
                                (41, [12000, 12000], [192, 192])):
        samples.append({"sampled_at": stamp, "mapped_mib": mapped,
                        "gpu_memory": [{"gpu_uuid": f"gpu-{rank}", "gpu_name": "NVIDIA A800",
                                        "free_mib": free[rank], "total_mib": 81920} for rank in (0, 1)]})
    save(root / "memory-samples.json", samples)
    initial = verifier.capacity_setting(samples[:1], [code], allocator_gpu_utilization=.98)
    save(root / "capacity-setting-initial.json", initial)
    probes = []
    blocks = list(range(1, math.ceil((code["prompt_tokens"] + code["output_tokens"]) / 16) + 1))
    page_count = math.ceil((len(blocks) + 1) / 64)
    for model in (0, 1):
        for rank in (0, 1):
            identifier = f"probe-{model}-{rank}"
            before = {"rank": rank, "requests": [], "mapped_bytes": 96 * 1024**2,
                      "page_bytes": 96 * 1024**2, "block_bytes": 1.5 * 1024**2,
                      "block_size": 16, "page_block_capacity": 64,
                      "pages": [{"id": 0, "owners": []}]}
            during = copy.deepcopy(before)
            during.update({"mapped_bytes": page_count * 96 * 1024**2,
                           "pages": [{"id": page, "owners": [identifier]} for page in range(page_count)],
                           "requests": [{"id": identifier, "prompt_tokens": code["prompt_tokens"],
                                         "max_tokens": code["output_tokens"], "blocks": blocks}]})
            increment = during["mapped_bytes"] - before["mapped_bytes"]
            peak = [192 / 1024] * 2
            peak[rank] += increment / 1024**3
            probes.append({"model": model, "rank": rank, "allocated": True, "fits_budget": True,
                           "rollback_verified": True, "initial_gib": [192 / 1024] * 2,
                           "allocation_peak_gib": peak, "additional_mapped_bytes": increment,
                           "allocator": {"allocated": True, "probe_request_id": identifier,
                                         "before": before, "during": during, "after": copy.deepcopy(before),
                                         "blocks": blocks}})
    save(root / "long-context-probes.json", probes)
    requests = []
    for index in range(5):
        model = int(index > 0)
        mark = conv if model else code
        requests.append({"request_id": f"capacity-{index}", "model": model, "rank_initial": model,
                         "role": "capacity", "prompt_tokens": mark["prompt_tokens"],
                         "token_ids": [42] * mark["output_tokens"], "arrived_at": 10, "submitted_at": 10,
                         "first_token_at": 12, "completed_at": 40 if model else 30})
    save(root / "requests.json", requests)
    setting = verifier.capacity_setting(samples, [code], allocator_gpu_utilization=.98)
    setting["calibration"] = verifier.calibration_rates(setting, [conv], 30, conv["output_tokens"])
    save(root / "capacity-setting.json", setting)
    graph = [[[{"histogram": [{"mode": "FULL", "steps": 1}]}] for rank in (0, 1)] for model in (0, 1)]
    save(root / "summary.json", {"status": "capacity_passed", "setting": setting, "graph": graph})
    (root / "exit-status").write_text("0\n")
    (root / "capacity-config.env").write_text(
        f'export DP_OPT_TRACE_BUDGET_MIB={setting["kv_budget_mib_per_gpu"]}\n'
        f'export DP_OPT_TARGET_CONV_MIB={setting["target_conv_reserved_mib"]}\n'
        'export DP_OPT_CALIBRATION_CONV_RATES="'
        + ' '.join(map(str, setting["calibration"]["conv_rates_rps"])) + '"\n')
    return verifier, root, inputs, save


def test_capacity_raw_evidence_is_independently_reproduced(capacity):
    verifier, root, inputs, save = capacity
    result = verifier.verify(root, inputs)
    assert result["status"] == "raw_a800_capacity_verified"
    assert result["allocation_probes"] == 4 and result["requests"] == 5
    assert result["setting"]["kv_budget_mib_per_gpu"] == 10464
    assert len(result["source_sha256"]) == 3


@pytest.mark.parametrize("mutation,match", [
    ("budget", "Final capacity"), ("rate", "Final capacity"), ("launch", "launch environment"),
    ("probe", "truncated"), ("rollback", "rollback"), ("graph", "FULL decode"),
    ("tokens", "request lengths"), ("hardware", "hardware"), ("topology", "topology"),
    ("duplicate_probe", "four distinct"), ("memory_order", "unordered"),
    ("initial", "Initial capacity"), ("prompt_bank", "checksum"),
])
def test_capacity_verifier_rejects_corrupt_raw_evidence(capacity, mutation, match):
    verifier, root, inputs, save = capacity
    if mutation in ("budget", "rate", "initial"):
        path = root / ("capacity-setting-initial.json" if mutation == "initial" else "capacity-setting.json")
        setting = json.loads(path.read_text())
        if mutation == "rate":
            setting["calibration"]["conv_rates_rps"][0] += .1
        else:
            setting["kv_budget_mib_per_gpu"] += 96
        save(path, setting)
    elif mutation in ("probe", "rollback", "duplicate_probe"):
        path = root / "long-context-probes.json"
        probes = json.loads(path.read_text())
        if mutation == "probe":
            probes[0]["allocator"]["during"]["requests"][0]["prompt_tokens"] -= 1
        elif mutation == "rollback":
            probes[0]["rollback_verified"] = False
        else:
            probes[1] = probes[0]
        save(path, probes)
    elif mutation == "graph":
        path = root / "summary.json"
        summary = json.loads(path.read_text())
        summary["graph"][0][0][0]["histogram"][0]["mode"] = "NONE"
        save(path, summary)
    elif mutation == "tokens":
        path = root / "requests.json"
        requests = json.loads(path.read_text())
        requests[0]["token_ids"].pop()
        save(path, requests)
    elif mutation in ("hardware", "memory_order"):
        path = root / "memory-samples.json"
        samples = json.loads(path.read_text())
        if mutation == "hardware":
            samples[0]["gpu_memory"][0]["gpu_name"] = "NVIDIA H200"
        else:
            samples.reverse()
        save(path, samples)
    elif mutation == "topology":
        path = root / "worker-instance-1-dp0.json"
        worker = json.loads(path.read_text())
        worker["groups"]["ep"]["world_size"] = 1
        save(path, worker)
    elif mutation == "prompt_bank":
        with (inputs / "prompt-bank.json").open("a") as output:
            output.write("\n")
    else:
        (root / "capacity-config.env").write_text("export DP_OPT_TRACE_BUDGET_MIB=20480\n")
    with pytest.raises(ValueError, match=match):
        verifier.verify(root, inputs)


def test_capacity_verifier_rejects_mixed_time_ownership_sample(capacity):
    verifier, root, inputs, save = capacity
    samples = json.loads((root / "memory-samples.json").read_text())
    sample = samples[0]
    sample.update({"ownership_stable_across_memory_query": True,
                   "ownership_before_memory_query": [[{"mapped_bytes": 96 * 1024**2}] * 2] * 2,
                   "ownership_after_memory_query": [[{"mapped_bytes": 192 * 1024**2}] * 2] * 2})
    save(root / "memory-samples.json", samples)
    with pytest.raises(ValueError, match="ownership transition"):
        verifier.verify(root, inputs)
