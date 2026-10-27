import copy
import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def calibration(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    verifier = importlib.import_module("verify_agentic_calibration")
    generator = importlib.import_module("agentic_trace")
    inputs, root = tmp_path / "inputs", tmp_path / "calibration"
    inputs.mkdir()
    root.mkdir()

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n")

    code = [{"prompt_tokens": 16, "output_tokens": 2, "prompt_key": "swe",
             "source_fragment": "code", "source_internal_row": 0, "source_traj_id": "trajectory"}]
    conv = [{"prompt_tokens": 32, "output_tokens": 2, "count": 3,
             "source_fragment": "conv", "source_internal_row": 1}]
    save(inputs / "code-marks.json", code)
    save(inputs / "conv-marks.json", conv)
    save(inputs / "prompt-bank.json", {"prompts": {"swe": {"prompt_tokens": 16,
                                                             "token_ids": [1] * 16, "traj_id": "trajectory"}}})
    config = {"calibration_conv_rates": [.03, .04], "calibration_code_rate": .03,
              "calibration_duration_s": 300, "seed": 20261021, "max_model_len": 131072,
              "deadline_s": 120, "kv_budget_mib": 20480}
    save(root / "config.json", config)
    for model in (0, 1):
        for rank in (0, 1):
            save(root / f"worker-instance-{model + 1}-dp{rank}.json", {
                "ipc_name": f"tenant{model}-rank{rank}", "gpu_uuid": f"gpu{rank}",
                "max_model_len": 131072, "dp_pause_sync_interval": 32,
                "groups": {name: {"world_size": size} for name, size in (("tp", 1), ("dp", 2), ("ep", 2))},
                "expert_ids": [list(range(rank * 64, (rank + 1) * 64))]})
    observation = {"mapped_bytes": 0, "page_bytes": 96 * 1024**2, "pages": [], "requests": [],
                   "block_size": 16, "block_bytes": 16 * 96 * 1024, "page_block_capacity": 64}
    snapshots = [[copy.deepcopy(observation) for rank in (0, 1)] for model in (0, 1)]
    trials = []
    for index, rate in enumerate(config["calibration_conv_rates"]):
        directory = root / f"calibration-{index}"
        trace = generator.generate_mixed_trace(code, conv, .03, rate, 300, config["seed"])
        trace["prompt_bank_sha256"] = hashlib.sha256((inputs / "prompt-bank.json").read_bytes()).hexdigest()
        save(directory / "trace.json", trace)
        save(directory / "workload.json", {"origin": 1000, "policy": "no_move", "repeat": index,
                                            "requests": trace["requests"], "duration_s": 300,
                                            "capacity_gib": [20, 20], "trace_sha256": hashlib.sha256(
                                                (json.dumps(trace, indent=2) + "\n").encode()).hexdigest()})
        requests = [{**{key: value for key, value in mark.items() if key in (
                         "request_id", "model", "prompt_tokens", "prompt_key", "source_fragment", "source_internal_row")},
                     "arrived_at": 1000 + mark["arrival_s"], "submitted_at": 1000.1 + mark["arrival_s"],
                     "first_token_at": 1000.2 + mark["arrival_s"], "completed_at": 1000.3 + mark["arrival_s"],
                     "token_ids": [2, 3]} for mark in trace["requests"]]
        decisions = [{"request_id": mark["request_id"], "arriving_model": mark["model"], "admitted": True,
                      "path": "direct_fastpath", "moves": [], "end_to_end_control_s": .01,
                      "observations": snapshots, "after_admission": snapshots} for mark in trace["requests"]]
        population = [{"elapsed_s": elapsed, "mapped_gib": [0, 0], "reserved_mib": [[0, 0], [1, 1]],
                       "active": 2, "unfinished": 3, "pending_heads": []} for elapsed in (110, 200, 280)]
        for name, records in (("requests", requests), ("decisions", decisions), ("population", population)):
            save(directory / f"{name}.json", records)
            (directory / f"{name}.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
        save(directory / "migrations.json", [])
        save(directory / "captures.json", [[{"histogram": [{"mode": "FULL", "steps": 1}]}] * 2] * 2)
        metrics = verifier.request_metrics(requests, 1000, 300, 120)
        actual = {**metrics, "policy": "no_move", "repeat": index, "moves": 0, "migration_kv_bytes": 0,
                  "ilp_attempts": 0, "control_s": sum(decision["end_to_end_control_s"] for decision in decisions)}
        save(directory / "summary.json", actual)
        trials.append({"conv_rate_rps": rate, "code_rate_rps": .03, "duration_s": 300,
                       "summary": actual, "mean_sampled_conv_reserved_mib": 2,
                       "mean_sampled_external_queue_count": 1, "code_head_samples": 0,
                       "global_enough_local_blocked_code_samples": 0})
    save(root / "calibration-summary.json", {"status": "passed", "trials": trials, "target_conv_reserved_mib": 23552})
    save(root / "summary.json", {"status": "calibration_passed"})
    (root / "exit-status").write_text("0\n")
    return verifier, root, inputs, save


def test_verifies_all_raw_trials_but_reports_underfilled_guardrail(calibration):
    verifier, root, inputs, save = calibration
    result = verifier.verify(root, inputs)
    assert result["status"] == "raw_calibration_verified"
    assert len(result["trials"]) == 2
    assert result["guardrails"]["status"] == "rejected"
    assert "pressure window" in result["guardrails"]["reason"]
    assert len(result["source_sha256"]) == 3


def test_completed_prefix_verification_never_allows_freeze(calibration):
    verifier, root, inputs, save = calibration
    summary = json.loads((root / "calibration-summary.json").read_text())
    summary.update({"status": "running", "trials": summary["trials"][:1]})
    save(root / "calibration-summary.json", summary)
    (root / "exit-status").write_text("1\n")
    with pytest.raises(ValueError, match="completely drain"):
        verifier.verify(root, inputs)
    result = verifier.verify(root, inputs, allow_partial=True)
    assert result["status"] == "raw_completed_calibration_prefix_verified"
    assert result["completed_trials_verified"] == 1 and result["declared_trials"] == 2
    assert not result["calibration_complete"]
    assert result["guardrails"]["status"] == "not_evaluated_incomplete"


def test_closed_prefix_of_active_job_needs_no_fake_exit_marker(calibration):
    verifier, root, inputs, save = calibration
    summary = json.loads((root / "calibration-summary.json").read_text())
    summary.update({"status": "running", "trials": summary["trials"][:1]})
    save(root / "calibration-summary.json", summary)
    (root / "exit-status").unlink()
    (root / "summary.json").unlink()
    with pytest.raises(ValueError, match="completely drain"):
        verifier.verify(root, inputs)
    result = verifier.verify(root, inputs, allow_partial=True)
    assert result["completed_trials_verified"] == 1
    assert not result["calibration_complete"]
    assert result["guardrails"]["status"] == "not_evaluated_incomplete"


@pytest.mark.parametrize("changed", [False, True])
def test_population_epoch_coherence_is_independently_checked(calibration, changed):
    verifier, root, inputs, save = calibration
    directory = root / "calibration-0"
    samples = json.loads((directory / "population.json").read_text())
    for sample in samples:
        sample.update({"admission_stable_across_observation": True,
                       "admission_generation_before": 1, "admission_generation_after": 1 + int(changed)})
    save(directory / "population.json", samples)
    (directory / "population.jsonl").write_text("".join(json.dumps(sample) + "\n" for sample in samples))
    if changed:
        with pytest.raises(ValueError, match="epoch transition"):
            verifier.verify(root, inputs)
    else:
        assert verifier.verify(root, inputs)["status"] == "raw_calibration_verified"


@pytest.mark.parametrize("mutation,match", [
    ("trace", "trace differs"),
    ("journal", "Incremental journal"),
    ("request", "request differs"),
    ("decision", "complete no-move"),
    ("graph", "FULL decode"),
    ("summary", "metric mismatch"),
    ("budget", "mapped budget exceeded"),
    ("topology", "topology differs"),
])
def test_rejects_corrupted_calibration_evidence(calibration, mutation, match):
    verifier, root, inputs, save = calibration
    directory = root / "calibration-0"
    if mutation == "trace":
        trace = json.loads((directory / "trace.json").read_text())
        trace["requests"][0]["arrival_s"] += 1
        save(directory / "trace.json", trace)
    elif mutation == "journal":
        (directory / "requests.jsonl").write_text("")
    elif mutation in ("request", "decision", "budget"):
        name = {"request": "requests", "decision": "decisions", "budget": "population"}[mutation]
        records = json.loads((directory / f"{name}.json").read_text())
        if mutation == "request":
            records[0]["prompt_tokens"] += 1
        elif mutation == "decision":
            records[0]["moves"] = [{"model": 1}]
        else:
            records[0]["mapped_gib"] = [21, 0]
        save(directory / f"{name}.json", records)
        (directory / f"{name}.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    elif mutation == "graph":
        save(directory / "captures.json", [[{"histogram": [{"mode": "NONE", "steps": 1}]}] * 2] * 2)
    elif mutation == "summary":
        actual = json.loads((directory / "summary.json").read_text())
        actual["mean_queue_s"] += 1
        save(directory / "summary.json", actual)
        collected = json.loads((root / "calibration-summary.json").read_text())
        collected["trials"][0]["summary"] = actual
        save(root / "calibration-summary.json", collected)
    else:
        path = root / "worker-instance-1-dp0.json"
        worker = json.loads(path.read_text())
        worker["groups"]["tp"]["world_size"] = 2
        save(path, worker)
    with pytest.raises(ValueError, match=match):
        verifier.verify(root, inputs)
