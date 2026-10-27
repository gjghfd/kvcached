import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("freeze_agentic_calibration")


def calibration(tmp_path, reservations=(10000, 24000), late_queue=1):
    trials = []
    for index, memory in enumerate(reservations):
        directory = tmp_path / f"calibration-{index}"
        directory.mkdir()
        (directory / "requests.json").write_text(json.dumps([
            {"model": 0, "submitted_at": 1, "arrived_at": 1},
            {"model": 1, "submitted_at": 1.1, "arrived_at": 1}]))
        (directory / "population.json").write_text(json.dumps([
            {"elapsed_s": 110, "unfinished": 51, "active": 50},
            {"elapsed_s": 280, "unfinished": 50 + late_queue, "active": 50}]))
        trials.append({"conv_rate_rps": 2 + index, "code_rate_rps": .013,
                       "duration_s": 300, "mean_sampled_conv_reserved_mib": memory,
                       "summary": {"drain_s": 20}})
    return {"status": "passed", "target_conv_reserved_mib": 23552, "trials": trials}


def test_freeze_chooses_closest_stable_population_not_policy_gain(module, tmp_path):
    summary = calibration(tmp_path)
    decision = module.choose_rate(tmp_path, summary)
    assert decision["winner"]["conv_rate_rps"] == 3
    assert decision["guardrails"]["target_range_fractions"] == [.95, 1.25]


def test_freeze_rejects_queue_growth(module, tmp_path):
    summary = calibration(tmp_path, late_queue=25)
    with pytest.raises(ValueError, match="No stable"):
        module.choose_rate(tmp_path, summary)


def test_freeze_rejects_underfilled_capacity(module, tmp_path):
    summary = calibration(tmp_path, reservations=(8000, 10000))
    with pytest.raises(ValueError, match="pressure window"):
        module.choose_rate(tmp_path, summary)


def test_freeze_rejects_incomplete_calibration(module, tmp_path):
    summary = calibration(tmp_path)
    summary["status"] = "running"
    with pytest.raises(ValueError, match="incomplete"):
        module.choose_rate(tmp_path, summary)


def test_freeze_counts_internal_allocator_wait_as_queue(module, tmp_path):
    summary = calibration(tmp_path)
    for index in range(len(summary["trials"])):
        directory = tmp_path / f"calibration-{index}"
        records = json.loads((directory / "requests.json").read_text())
        records[1]["reserved_at"] = 20
        (directory / "requests.json").write_text(json.dumps(records))
    with pytest.raises(ValueError, match="No stable"):
        module.choose_rate(tmp_path, summary)
