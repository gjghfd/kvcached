import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("diagnose_return_gates")


def plan(gain=3., uncertainty=4., accepted=False):
    return {"active_population": 2, "forecast_horizon_s": 10.,
            "reference": {"objective_s": 10.},
            "winner": {"objective_s": 10. - gain, "moves": ["conv"], "migration_cost_s": .5,
                       "migration_activation_cost_s": 1., "status": "optimal"},
            "gate": {"predicted_net_gain_request_s": gain, "uncertainty_request_s": uncertainty,
                     "min_gain_request_s": 1., "accepted": accepted}}


def test_uncertainty_and_cost_units_are_separated(module):
    summary = module.summarize_plan(plan())
    assert summary["gate_blocker"] == "uncertainty_exceeds_point_gain"
    assert summary["guarded_margin_request_s"] == -2.
    assert summary["migration_cost_request_s"] == .5
    assert summary["activation_cost_request_s"] == 1.


@pytest.mark.parametrize("fault", ["gain", "accepted", "negative", "nan"])
def test_inconsistent_or_nonfinite_raw_gate_is_rejected(module, fault):
    value = plan()
    if fault == "gain":
        value["gate"]["predicted_net_gain_request_s"] = 99.
    elif fault == "accepted":
        value["gate"]["accepted"] = True
    elif fault == "negative":
        value["gate"]["uncertainty_request_s"] = -1.
    else:
        value["winner"]["objective_s"] = float("nan")
    with pytest.raises(ValueError):
        module.summarize_plan(value)


def test_streaming_diagnosis_records_release_window_and_SHA_without_mutating_inputs(module, tmp_path):
    (tmp_path / "workload.json").write_text(json.dumps({"origin": 100.}))
    (tmp_path / "requests.json").write_text(json.dumps([
        {"request_id": "code", "model": 0, "completed_at": 105.},
        {"request_id": "conv", "model": 1, "completed_at": 200.}]))
    rows = [{"path": "admission", "started_at": 104., "reason": "admitted"},
            {"path": "release_rebalance_ilp", "started_at": 105.1, "reason": "no_profitable_rebalance",
             "before_pause": plan(), "preplan_s": .1},
            {"path": "release_rebalance_ilp", "started_at": 110., "reason": "outside_verified_workaware_calibration",
             "calibration_deferral": "outside"}]
    source = tmp_path / "decisions.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    before = source.read_bytes()
    result = module.analyze(tmp_path, release_window_s=2)
    assert result["release_decision_counts"] == {"no_profitable_rebalance": 1, "outside_verified_workaware_calibration": 1}
    assert result["before_pause_gate_blockers"] == {"uncertainty_exceeds_point_gain": 1}
    events = result["Code_release_windows"][0]["events"]
    assert len(events) == 1
    assert events[0]["delay_from_release_s"] == pytest.approx(.1)
    assert result["source_sha256"]["decisions.jsonl"] == module.digest(source)
    assert source.read_bytes() == before
    assert result["runtime_changed"] is False
