import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def rates(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("recalibrate_agentic_rates")


def requests(services):
    return [{"model": 0, "reserved_at": 10, "completed_at": 10 + service,
             "reservation_time_source": "scheduler_monotonic"} for service in services]


def test_a800_code_rate_uses_original_occupancy_rule(rates):
    result = rates.code_rate_from_residency(requests([55, 57, 56, 58]))
    assert result["code_rate_rps"] == .004
    assert result["code_rate_raw_rps"] == pytest.approx(.25 / 56.5)
    assert result["predicted_code_resident_count"] == pytest.approx(.226)
    assert result["code_resident_samples"] == 4


def test_rate_cap_preserves_fast_code(rates):
    result = rates.code_rate_from_residency(requests([1, 2]))
    assert result["code_rate_rps"] == .02


@pytest.mark.parametrize("source", ["snapshot_upper_bound", "first_token_upper_bound", None])
def test_rate_refuses_nonexact_reservation(rates, source):
    records = requests([56])
    records[0]["reservation_time_source"] = source
    with pytest.raises(ValueError, match="exact scheduler"):
        rates.code_rate_from_residency(records)


@pytest.mark.parametrize("service", [0, -1, float("nan"), float("inf"), 1000])
def test_rate_refuses_invalid_service_or_zero_rounding(rates, service):
    with pytest.raises(ValueError, match="resident time|precision"):
        rates.code_rate_from_residency(requests([service]))


def test_rate_refuses_other_model_and_empty_samples(rates):
    records = requests([56])
    records[0]["model"] = 1
    with pytest.raises(ValueError, match="exact scheduler"):
        rates.code_rate_from_residency(records)
    with pytest.raises(ValueError, match="Require Code"):
        rates.code_rate_from_residency([])


def pressure_candidates():
    return [{"conv_rate_rps": .30, "mean_conv_reserved_mib": 2765.714285714286, "eligible": True},
            {"conv_rate_rps": .45, "mean_conv_reserved_mib": 7757.06, "eligible": True},
            {"conv_rate_rps": .60, "mean_conv_reserved_mib": 7712.78, "eligible": False},
            {"conv_rate_rps": .80, "mean_conv_reserved_mib": 7241.88, "eligible": False}]


def test_conv_refinement_interpolates_only_stable_observed_bracket(rates):
    result = rates.conv_bracket_from_pressure(pressure_candidates(), 4347)
    assert result["conv_rates_rps"] == [.33, .35, .37, .39]
    assert result["central_conv_rate_raw_rps"] == pytest.approx(.34752, abs=.00001)
    assert len(result["observed_candidates"]) == 4
    assert [candidate["conv_rate_rps"] for candidate in result["observed_stable_bracket"]] == [.30, .45]


def test_conv_refinement_rejects_unstable_target_boundary(rates):
    candidates = pressure_candidates()
    candidates[1]["eligible"] = False
    with pytest.raises(ValueError, match="queue-stable"):
        rates.conv_bracket_from_pressure(candidates, 4347)


@pytest.mark.parametrize("failure", ["nan", "zero", "duplicate", "outside", "precision"])
def test_conv_refinement_rejects_invalid_or_unsupported_bracket(rates, failure):
    candidates, target = pressure_candidates(), 4347
    if failure == "nan":
        candidates[0]["mean_conv_reserved_mib"] = float("nan")
    elif failure == "zero":
        target = 0
    elif failure == "duplicate":
        candidates[1]["conv_rate_rps"] = .30
    elif failure == "outside":
        target = 10000
    else:
        candidates[1]["conv_rate_rps"] = .31
    with pytest.raises(ValueError, match="Pressure|distinct|queue-stable|precision"):
        rates.conv_bracket_from_pressure(candidates, target)


def test_long_window_rate_preserves_all_fixed_arrival_subset_requests(rates, monkeypatch, tmp_path):
    inputs = tmp_path / "input"
    inputs.mkdir()
    for filename in ("code-marks.json", "conv-marks.json", "prompt-bank.json"):
        (inputs / filename).write_text("{}")
    (tmp_path / "config.json").write_text('{"calibration_code_rate":0.004,"seed":20261021,"calibration_duration_s":300}')
    (tmp_path / "calibration-summary.json").write_text('{"target_conv_reserved_mib":100}')
    directory = tmp_path / "calibration-0"
    directory.mkdir()
    (directory / "workload.json").write_text('{"origin":100}')
    (directory / "decisions.json").write_text('[{"observations":[[{"block_size":16,"block_bytes":1048576}]]}]')
    records = [{"model":0,"arrived_at":271}]
    records.extend({"model":1,"arrived_at":start,"submitted_at":start,"reserved_at":start,
                    "completed_at":start+10,"prompt_tokens":16,"token_ids":[],
                    "reservation_time_source":"scheduler_monotonic"} for start in (130,140,219))
    (directory / "requests.json").write_text(json.dumps(records))
    monkeypatch.setattr(rates, "verify", lambda *args: {"trials":[{"conv_rate_rps":.8,"duration_s":300}],"guardrails":{}})
    result = rates.long_window_bracket(tmp_path, inputs)
    assert result["reference"]["selected_requests"] == 3
    assert result["reference"]["latest_selected_completion_s"] == 129
    assert result["conv_rates_rps"] == [7,8.5,10,11.5]
    assert result["calibration_duration_s"] == 1200 and result["code_rate_rps_unchanged"] == .004

    records[1]["completed_at"] = 272
    (directory / "requests.json").write_text(json.dumps(records))
    with pytest.raises(ValueError, match="No complete exact-timestamp"):
        rates.long_window_bracket(tmp_path, inputs)
    records[1]["completed_at"] = 140
    records[1]["reservation_time_source"] = "snapshot_upper_bound"
    (directory / "requests.json").write_text(json.dumps(records))
    with pytest.raises(ValueError, match="No complete exact-timestamp"):
        rates.long_window_bracket(tmp_path, inputs)
