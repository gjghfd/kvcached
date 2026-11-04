import copy
import importlib
from pathlib import Path

import pytest

from test_dp_opt_workaware_robust_rebalance import legacy_fit, observations, settings


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("workaware_robust_adapter")


def raw_plan(phase="after_pause"):
    planning_s = .25
    return {"scope": "independent_robust_release_planner_not_live_runtime_or_GPU_benefit", "planning_s": planning_s,
            "active_population": 2, "gate": {"phase": phase, "accepted": True,
                "predicted_net_gain_request_s": 20., "uncertainty_request_s": 1.,
                "active_planning_blocking_request_s": .5 if phase == "after_pause" else 0.,
                "measured_blocking_so_far_request_s": 3.5 if phase == "after_pause" else 0.}}


def test_paid_controller_time_is_recorded_without_changing_profit(adapter):
    raw = raw_plan()
    original = copy.deepcopy(raw)
    check = adapter.wrap_plan(raw)
    adapter.account_paused_controller_planning(check, 1000., 3.)
    assert check["gate"]["accepted"] is True
    assert check["gate"]["predicted_net_gain_request_s"] == 20.
    assert check["gate"]["uncertainty_request_s"] == 1.
    assert check["gate"]["active_planning_blocking_request_s"] == 2000.
    assert check["gate"]["measured_blocking_so_far_request_s"] == 2003.
    assert check["controller_adapter"]["raw_planner_accounting"]["planning_s"] == .25
    assert raw == original


@pytest.mark.parametrize("elapsed", [0., .1, -1., True, float("nan"), float("inf"), "1"])
def test_invalid_or_shorter_controller_time_does_not_mutate_plan(adapter, elapsed):
    check = adapter.wrap_plan(raw_plan())
    original = copy.deepcopy(check)
    with pytest.raises(ValueError, match="walltime"):
        adapter.account_paused_controller_planning(check, elapsed, 3.)
    assert check == original


@pytest.mark.parametrize("blocking", [-1., True, float("nan"), float("inf"), "1"])
def test_invalid_original_blocking_is_rejected(adapter, blocking):
    check = adapter.wrap_plan(raw_plan())
    with pytest.raises(ValueError):
        adapter.account_paused_controller_planning(check, 1., blocking)


def test_before_pause_time_cannot_be_reclassified_as_frozen(adapter):
    check = adapter.wrap_plan(raw_plan("before_pause"))
    with pytest.raises(ValueError, match="after-pause"):
        adapter.account_paused_controller_planning(check, 1., 0.)


def test_already_wrapped_plan_is_not_nested_again(adapter):
    with pytest.raises(ValueError, match="unwrapped"):
        adapter.wrap_plan(adapter.wrap_plan(raw_plan()))


def test_empty_population_has_zero_inference_blocking_cost(adapter):
    raw = raw_plan()
    raw["active_population"] = 0
    raw["gate"]["accepted"] = False
    check = adapter.wrap_plan(raw)
    adapter.account_paused_controller_planning(check, 2., 0.)
    assert check["gate"]["active_planning_blocking_request_s"] == 0.
    assert check["gate"]["measured_blocking_so_far_request_s"] == 0.
    assert check["gate"]["accepted"] is False


def test_boolean_adapter_version_cannot_mutate_accounting(adapter):
    check = adapter.wrap_plan(raw_plan())
    check["controller_adapter"]["schema_version"] = True
    original = copy.deepcopy(check)
    with pytest.raises(ValueError):
        adapter.account_paused_controller_planning(check, 1., 3.)
    assert check == original


@pytest.mark.parametrize("phase", ["before_pause", "after_pause"])
def test_real_robust_plan_independent_verification_includes_controller_time(adapter, phase):
    observed, configured = observations(), settings()
    check = adapter.plan(observed, [1., 1.], legacy_fit(), True, configured, None, 3., phase)
    elapsed = check["planning_s"] + .05
    if phase == "after_pause":
        adapter.account_paused_controller_planning(check, elapsed, 3.)
    verified = adapter.verify_controller_plan(check, observed, [1., 1.], legacy_fit(), configured, phase, 3., elapsed)
    assert verified["accepted"] is True
    assert verified["actual_walltime_verified"] is False
    assert verified["paid_controller_planning_excluded_from_continuation_profit"] == (phase == "after_pause")


@pytest.mark.parametrize("field", ["scope", "optimizer", "adapter_version", "paid_planning", "gain", "raw_planning", "actual_clock", "raw_scope"])
def test_tampered_adapter_fails_independent_checks(adapter, field):
    observed, configured = observations(), settings()
    check = adapter.plan(observed, [1., 1.], legacy_fit(), True, configured, None, 3., "after_pause")
    elapsed = check["planning_s"] + .05
    adapter.account_paused_controller_planning(check, elapsed, 3.)
    if field == "scope":
        check["scope"] = "point-policy"
    elif field == "optimizer":
        check["controller_adapter"]["optimizer"] = "point"
    elif field == "adapter_version":
        check["controller_adapter"]["schema_version"] = True
    elif field == "paid_planning":
        check["gate"]["active_planning_blocking_request_s"] = 0.
    elif field == "gain":
        check["gate"]["predicted_net_gain_request_s"] += 1.
    elif field == "raw_planning":
        check["controller_adapter"]["raw_planner_accounting"]["planning_s"] += 10.
    elif field == "actual_clock":
        check["controller_adapter"]["controller_planning_s"] = 0.
    else:
        check["controller_adapter"]["raw_planner_accounting"]["scope"] = "unverified"
    with pytest.raises(ValueError):
        adapter.verify_controller_plan(check, observed, [1., 1.], legacy_fit(), configured, "after_pause", 3., elapsed)
