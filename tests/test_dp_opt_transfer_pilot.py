import copy
import importlib
import json
import math
from pathlib import Path

import pytest


@pytest.fixture
def pilot(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("transfer_pilot")


def paired_marks():
    return [{"prompt_tokens": prompt, "output_tokens": output, "source_internal_row": index}
            for index, (prompt, output) in enumerate(((2048, 512), (3584, 1024), (7167, 1024)))]


def test_cases_keep_real_pairs_and_rotate_transport_and_direction(pilot):
    marks = paired_marks()
    cases = pilot.pilot_cases(marks)
    assert len(cases) == 36
    assert all(case["mark"] in marks for case in cases)
    assert cases[0]["transport"] == "host_file"
    assert cases[4]["transport"] == "dp_allgatherv"
    assert {case["source"] for case in cases} == {0, 1}
    assert {case["model"] for case in cases} == {0, 1}


def test_boundary_cases_keep_exact_echo_and_alternate_pause_modes(pilot):
    cases = pilot.pilot_cases(paired_marks(), transports=("dp_echo",),
                             pause_modes=("native_keep", "default32_boundary"))
    assert len(cases) == 36
    assert {case["transport"] for case in cases} == {"dp_echo"}
    assert cases[0]["pause_mode"] == "native_keep"
    assert cases[4]["pause_mode"] == "default32_boundary"


@pytest.mark.parametrize("fault", [None, "receipt", "step", "timestamp", "cadence"])
def test_boundary_verifier_requires_four_default32_receipts(pilot, tmp_path, fault):
    verifier = importlib.import_module("verify_transfer_pilot")
    trials = fixture(tmp_path, pilot, transports=("dp_echo",),
                     pause_modes=("native_keep", "default32_boundary"))
    if fault == "receipt":
        trials[2]["pause_receipts"] = None
    elif fault == "step":
        trials[2]["pause_receipts"][0][0]["boundary_step_counter"] = 31
    elif fault == "timestamp":
        trials[2]["pause_receipts"][0][0]["reached_at"] = 9
    elif fault == "cadence":
        trials[0]["request"]["cadence"] = []
    (tmp_path / "transfer-trials.json").write_text(json.dumps(trials))
    if fault:
        with pytest.raises(ValueError):
            verifier.verify(tmp_path, tmp_path / "marks.json")
    else:
        result = verifier.verify(tmp_path, tmp_path / "marks.json")
        assert len(result["pause_costs"]) == 36
        assert result["baseline_transport"] == "native_keep"
        assert result["treatment_transport"] == "default32_boundary"


def rank_snapshot(identifier=None, blocks=None, output_tokens=32):
    blocks = blocks or []
    page_ids = sorted({0, *(block // 64 for block in blocks)})
    return {"requests": [{"id": identifier, "blocks": blocks, "movable": True,
                          "output_tokens": output_tokens}] if identifier else [],
            "pages": [{"id": page, "owners": [identifier] if any(block // 64 == page for block in blocks) else [],
                       "pinned": page == 0} for page in page_ids],
            "mapped_bytes": len(page_ids) * 96 * 1024**2, "page_bytes": 96 * 1024**2,
            "block_bytes": int(1.5 * 1024**2), "block_size": 16, "page_block_capacity": 64}


def fixture(root, pilot, transports=("host_file", "dp_allgatherv"), pause_modes=None):
    def save(name, value):
        (root / name).write_text(json.dumps(value))

    marks = paired_marks()
    cases = pilot.pilot_cases(marks, transports=transports, pause_modes=pause_modes)
    save("marks.json", marks)
    save("config.json", {"max_model_len": 131072, "cache_bytes": 24 * 1024**3,
                         "kv_budget_mib": 11328, "allocator_gpu_utilization": .98,
                         "transfer_transports": list(transports), "transfer_pause_modes": pause_modes})
    save("transfer-summary.json", {"status": "transfer_pilot_passed", "cases": 36, "chunk_mib": 32})
    save("runtime-control.json", {"dp_pause_sync_interval": 32, "same_interval_for_all_policies": True,
                                  "idle_rank_expert_execution_not_disabled": True})
    for model in (0, 1):
        for rank in (0, 1):
            save(f"worker-instance-{model + 1}-dp{rank}.json", {
                "dp_pause_sync_interval": 32, "ipc_name": f"tenant{model}-rank{rank}",
                "gpu_uuid": f"GPU{rank}", "gpu_name": "A800", "max_model_len": 131072,
                "kv_cache_memory_bytes": 24 * 1024**3,
                "groups": {name: {"world_size": size} for name, size in (("dp", 2), ("ep", 2), ("tp", 1))}})
    references = [{"model": model, "prompt_tokens": mark["prompt_tokens"], "token_ids": [1] * mark["output_tokens"]}
                  for model in (0, 1) for mark in marks]
    save("transfer-references.json", references)
    trials = []
    for index, case in enumerate(cases):
        model, source, mark = case["model"], case["source"], case["mark"]
        identifier = f"transfer-{index}"
        blocks = list(range(1, 1 + math.ceil((mark["prompt_tokens"] + mark["output_tokens"]) / 16)))
        before = [[rank_snapshot() for rank in (0, 1)] for tenant in (0, 1)]
        before[model][source] = rank_snapshot(identifier, blocks)
        after = copy.deepcopy(before)
        after[model][source] = rank_snapshot()
        after[model][1 - source] = rank_snapshot(identifier, blocks)
        elapsed = 2 if case["transport"] == "host_file" else 1
        event = {"request_id": identifier, "model": model, "source": source, "destination": 1 - source,
                 "kv_bytes": len(blocks) * int(1.5 * 1024**2), "sha256": "a" * 64,
                 "copy_peak_gib": [1, 1], "elapsed_s": elapsed, "stage_s": {"rpc": elapsed},
                 "worker_transfer_stages_s": {"export": {"data": .1}, "import": {"data": .1}},
                 "transport": {"dp_allgatherv": "existing_dp_allgatherv_device_payload_host_integrity_readback",
                               "dp_echo": "existing_dp_allgatherv_full_device_bytewise_echo",
                               "host_file": "host_staged_file_not_nvlink_optimized"}[case["transport"]]}
        if case["transport"] == "dp_echo":
            event.update({"integrity": "device_echo_full_bytewise", "bitwise_verified": True,
                          "sha256_scope": "verified_integrity_receipt_not_KV_content"})
        request = {"request_id": identifier, "model": model, "prompt_tokens": mark["prompt_tokens"],
                   "token_ids": [1] * mark["output_tokens"],
                   "cadence": [{"time": 10 + position * .01, "tokens": position + 1}
                               for position in range(mark["output_tokens"])]}
        trials.append({"case": case, "before": before, "after": after, "event": event, "request": request,
                       "pause_s": .2, "resume_s": .01, "paused_transaction_s": elapsed})
        if pause_modes:
            trials[-1].update({"pause_started_at": 10, "resume_started_at": 10.2 + elapsed,
                               "resume_completed_at": 10.21 + elapsed,
                               "pause_receipts": [[{"rank": rank, "armed_at": 10.01,
                                   "reached_at": 10.1, "callback_at": 10.19, "boundary_step_counter": 32}
                                   for rank in (0, 1)] for tenant in (0, 1)]
                               if case["pause_mode"] == "default32_boundary" else None})
    save("transfer-cases.json", cases)
    save("transfer-trials.json", trials)
    save("transfer-captures.json", [[[{"histogram": [{"mode": "FULL", "steps": 100}]}] for rank in (0, 1)]
                                    for model in (0, 1)])
    (root / "exit-status").write_text("0")
    return trials


def test_verifier_rebuilds_all_pairs_without_claiming_trace_benefit(pilot, tmp_path):
    verifier = importlib.import_module("verify_transfer_pilot")
    fixture(tmp_path, pilot)
    result = verifier.verify(tmp_path, tmp_path / "marks.json")
    assert result["status"] == "transfer_pilot_raw_verified"
    assert result["cases"] == 36 and len(result["paired_costs"]) == 18
    assert result["baseline_mean_s"] == 2 and result["treatment_mean_s"] == 1
    assert "not_net_trace_benefit" in result["scope"]


@pytest.mark.parametrize("fault", [None, "missing_proof", "mislabel_content_sha"])
def test_echo_verifier_requires_exact_bytewise_proof_and_labels_receipt(pilot, tmp_path, fault):
    verifier = importlib.import_module("verify_transfer_pilot")
    trials = fixture(tmp_path, pilot, transports=("dp_allgatherv", "dp_echo"))
    echo = next(trial for trial in trials if trial["case"]["transport"] == "dp_echo")
    if fault == "missing_proof":
        echo["event"]["bitwise_verified"] = False
    elif fault == "mislabel_content_sha":
        echo["event"]["sha256_scope"] = "KV_content"
    (tmp_path / "transfer-trials.json").write_text(json.dumps(trials))
    if fault is not None:
        with pytest.raises(ValueError, match="bytewise"):
            verifier.verify(tmp_path, tmp_path / "marks.json")
    else:
        result = verifier.verify(tmp_path, tmp_path / "marks.json")
        assert result["baseline_transport"] == "dp_allgatherv" and result["treatment_transport"] == "dp_echo"


@pytest.mark.parametrize("fault", ["tokens", "owner", "budget", "timing", "transport", "cases", "graph"])
def test_verifier_rejects_independent_corruption(pilot, tmp_path, fault):
    verifier = importlib.import_module("verify_transfer_pilot")
    trials = fixture(tmp_path, pilot)
    if fault == "tokens":
        trials[0]["request"]["token_ids"][0] = 2
    elif fault == "owner":
        trials[0]["after"][0][0] = copy.deepcopy(trials[0]["before"][0][0])
    elif fault == "budget":
        trials[0]["event"]["copy_peak_gib"][0] = 100
    elif fault == "timing":
        trials[0]["event"]["stage_s"]["rpc"] = float("nan")
    elif fault == "transport":
        trials[0]["event"]["transport"] = "pretend_nvlink"
    elif fault == "cases":
        trials.pop()
    elif fault == "graph":
        (tmp_path / "transfer-captures.json").write_text(json.dumps([[{"histogram": []}]]))
    (tmp_path / "transfer-trials.json").write_text(json.dumps(trials))
    with pytest.raises((ValueError, StopIteration)):
        verifier.verify(tmp_path, tmp_path / "marks.json")
