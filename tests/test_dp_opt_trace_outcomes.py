import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def analyzer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("analyze_trace")


def request(identifier, model, arrival, reserved, first, completed):
    return {"request_id": identifier, "model": model, "arrived_at": arrival, "submitted_at": arrival,
            "reserved_at": reserved, "first_token_at": first, "completed_at": completed, "token_ids": [1, 2]}


def test_paired_metrics_keep_old_conv_cost_and_negative_net_result(analyzer):
    baseline = {"code": request("code", 0, 10, 30, 35, 40),
                "conv": request("conv", 1, 0, 0, 1, 10)}
    ilp = {"code": request("code", 0, 100, 105, 110, 115),
           "conv": request("conv", 1, 90, 90, 91, 120)}
    code = analyzer.paired_request_outcomes(baseline, ilp, {"code"}, 25)
    conv = analyzer.paired_request_outcomes(baseline, ilp, {"conv"}, 25)
    total = analyzer.paired_request_outcomes(baseline, ilp, set(baseline), 25)
    assert code["metrics"]["queue_s"]["reduction_fraction"] == .75
    assert code["metrics"]["latency_s"]["paired_sum_delta_s"] == -15
    assert conv["metrics"]["latency_s"]["paired_sum_delta_s"] == 20
    assert conv["deadline_met_difference"] == -1
    assert total["metrics"]["latency_s"]["paired_sum_delta_s"] == 5
    assert total["metrics"]["latency_s"]["reduction_fraction"] < 0


def test_baseline_resident_conv_group_is_fixed_across_treatment(analyzer, tmp_path):
    baseline = [request("code", 0, 110, 130, 135, 140),
                request("old", 1, 100, 105, 106, 120),
                request("finished", 1, 100, 101, 102, 109),
                request("waiting", 1, 100, 111, 112, 120)]
    ilp = [request("code", 0, 1010, 1011, 1012, 1015),
           request("old", 1, 1000, 1001, 1002, 1005),
           request("finished", 1, 1000, 1001, 1002, 1020),
           request("waiting", 1, 1000, 1001, 1002, 1030)]
    for policy, records, origin in (("no_move", baseline, 100), ("ilp", ilp, 1000)):
        directory = tmp_path / f"r0-{policy}"
        directory.mkdir()
        (directory / "requests.json").write_text(json.dumps(records))
        (directory / "workload.json").write_text(json.dumps({"origin": origin}))
    verified = {"paired_comparisons": [{"repeat": 0, "baseline": "no_move"}]}
    trace = {"requests": [{"model": 0, "arrival_s": 10}]}
    result = analyzer.paired_tenant_outcomes(tmp_path, verified, trace, 120)[0]
    assert result["groups"]["baseline_resident_Conv_at_Code_arrival"]["request_ids"] == ["old"]
    assert result["groups"]["all_Conv"]["requests"] == 3
    assert result["groups"]["all_Code"]["requests"] == 1
    assert result["groups"]["baseline_resident_Conv_at_Code_arrival"]["metrics"]["latency_s"]["paired_sum_delta_s"] == -15


def test_empty_groups_and_single_output_tpot_are_explicit(analyzer):
    assert analyzer.paired_request_outcomes({}, {}, set(), 120) == {"requests": 0}
    record = request("conv", 1, 0, 0, 1, 2)
    record["token_ids"] = [1]
    result = analyzer.paired_request_outcomes({"conv": record}, {"conv": record}, {"conv"}, 120)
    assert result["metrics"]["tpot_s"] is None
    assert result["metrics"]["queue_s"]["reduction_fraction"] is None
