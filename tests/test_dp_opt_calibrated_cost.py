import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def costs(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "experiments/dp_opt"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("dp_opt_calibrated_cost", root / "calibrated_cost.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fit():
    return {"status": "verified_curve_fit", "raw_verification": {"status": "raw_verified"}, "requests": 128,
            "capture_sizes": [1, 8, 16, 32, 64, 128, 256], "baseline_cadence_s": 0.025,
            "passes_predeclared_0_05_residual": False, "alpha": 0.5,
            "capture_shape_posthoc": {"status": "descriptive_posthoc_fit", "a": 0.3, "b": 0.2}}


def test_shape_cost_has_a_capture_step_and_correct_units(costs, fit):
    result = costs.layout_cost_table(fit, horizon_steps=100, request_weight=2)
    assert result["costs_s"][64] == 0
    assert result["costs_s"][65] == pytest.approx(2 * 100 * 128 * 0.025 * (0.3 / 64 + 0.2))
    assert result["predicted_cadence_ratios"][0] == pytest.approx(1.5)
    assert result["costs_s"] == pytest.approx(list(reversed(result["costs_s"])))
    assert not result["concurrent_models_calibrated"]


def test_rejected_affine_fit_cannot_be_used_silently(costs, fit):
    with pytest.raises(ValueError, match="rejected"):
        costs.layout_cost_table(fit, mode="affine")
    fit["passes_predeclared_0_05_residual"] = True
    result = costs.layout_cost_table(fit, mode="affine")
    assert result["predicted_cadence_ratios"][96] == pytest.approx(1.25)


@pytest.mark.parametrize("change,error", (({"raw_verification": {}}, "raw-verified"),
                                         ({"capture_sizes": [1, 8, 64]}, "cover"),
                                         ({"capture_shape_posthoc": {"status": "not_identifiable"}}, "identifiable")))
def test_invalid_calibration_is_rejected(costs, fit, change, error):
    fit.update(change)
    with pytest.raises(ValueError, match=error):
        costs.layout_cost_table(fit)


def test_nonfinite_coefficient_is_rejected(costs, fit):
    fit["capture_shape_posthoc"]["a"] = float("nan")
    with pytest.raises(ValueError, match="nonnegative"):
        costs.layout_cost_table(fit)


def test_calibrated_table_connects_to_rank_enumeration(costs, fit):
    if not hasattr(pytest.importorskip("scipy.optimize"), "milp"):
        pytest.skip("SciPy MILP is unavailable")
    from ilp_reference import enumerate_targets

    table = costs.layout_cost_table(fit)["costs_s"]
    snapshot = {"counts": [[64, 63], [64, 64]], "capacity_gib": [11, 11], "initial_gib": [8, 8],
                "new_gib": 1, "imbalance_cost_s": [0.01, 0.01], "movable": [],
                "performance_cost_tables_s": [table, table]}
    result = enumerate_targets(snapshot)
    assert result["winner"]["target"] == 1
    assert result["winner"]["objective_s"] == 0


def test_sparse_snapshot_preserves_input_and_labels_concurrency(costs, fit):
    calibration = costs.layout_cost_table(fit)
    snapshot = {"counts": [[64, 63], [64, 64]]}
    result = costs.attach_sparse_costs(snapshot, [calibration, calibration])
    assert result["performance_cost_tables_s"] == [calibration["costs_s"], calibration["costs_s"]]
    assert result["cost_calibration"]["capture_profile"] == "sparse"
    assert not result["cost_calibration"]["concurrent_models_calibrated"]
    result["counts"][0][0] = 0
    result["performance_cost_tables_s"][0][0] = 99
    assert snapshot["counts"][0][0] == 64
    assert calibration["costs_s"][0] != 99


def test_new_arrival_cannot_extrapolate_128_to_129(costs, fit):
    calibration = costs.layout_cost_table(fit)
    with pytest.raises(ValueError, match="post-admission population"):
        costs.attach_sparse_costs({"counts": [[64, 64], [64, 64]]}, [calibration, calibration])


def test_dense_calibration_cannot_enter_sparse_admission(costs, fit):
    fit["capture_sizes"] = [1, 8, 16, 32, 64, 80, 96, 112, 128, 256]
    calibration = costs.layout_cost_table(fit)
    with pytest.raises(ValueError, match="fixed sparse"):
        costs.attach_sparse_costs({"counts": [[64, 63], [64, 64]]}, [calibration, calibration])


@pytest.mark.parametrize("change,error", (({"cost_units": "bytes"}, "request-second"),
                                         ({"costs_s": [float("nan")] * 129}, "finite"),
                                         ({"status": "not_calibrated"}, "labelled")))
def test_invalid_sparse_calibration_metadata_is_rejected(costs, fit, change, error):
    calibration = costs.layout_cost_table(fit)
    calibration.update(change)
    with pytest.raises(ValueError, match=error):
        costs.attach_sparse_costs({"counts": [[64, 63], [64, 64]]}, [calibration, calibration])


def test_sparse_attachment_cannot_overwrite_joint_costs(costs, fit):
    calibration = costs.layout_cost_table(fit)
    snapshot = {"counts": [[64, 63], [64, 64]], "joint_performance_costs_s": [[0]]}
    with pytest.raises(ValueError, match="overwrite joint"):
        costs.attach_sparse_costs(snapshot, [calibration, calibration])


def test_cross_model_move_preserves_a_balanced_graph(costs, fit):
    if not hasattr(pytest.importorskip("scipy.optimize"), "milp"):
        pytest.skip("SciPy MILP is unavailable")
    from ilp_reference import enumerate_targets

    calibration = costs.layout_cost_table(fit, horizon_steps=32)
    snapshot = {"counts": [[64, 63], [80, 48]], "capacity_gib": [24, 24], "initial_gib": [23.90625, 23.90625],
                "new_gib": 0.1875, "imbalance_cost_s": [0.01, 0.01],
                "movable": [{"id": identifier, "model": model, "source": source,
                             "memory_gib": 0.09375, "move_cost_s": 0.01}
                            for identifier, model, source in (("A1", 0, 1), ("B0", 1, 0), ("B1", 1, 1))]}
    snapshot = costs.attach_sparse_costs(snapshot, [calibration, calibration])
    result = enumerate_targets(snapshot)
    assert result["winner"]["target"] == 1
    assert result["winner"]["moves"] == ["B1"]
    assert result["winner"]["final_counts"] == [[64, 64], [81, 47]]
    assert result["winner"]["peak_gib"] == [24, 24]
    assert enumerate_targets(snapshot, max_moves=0)["proven_infeasible"]
    own_model = dict(snapshot, movable=[request for request in snapshot["movable"] if request["model"] == 0])
    assert enumerate_targets(own_model)["winner"]["objective_s"] > result["winner"]["objective_s"]


def test_migration_must_repay_its_cost_over_the_declared_horizon(costs, fit):
    if not hasattr(pytest.importorskip("scipy.optimize"), "milp"):
        pytest.skip("SciPy MILP is unavailable")
    from ilp_reference import enumerate_targets

    snapshot = {"counts": [[66, 61], [64, 64]], "capacity_gib": [2, 2], "initial_gib": [0.5, 0.5],
                "new_gib": 0.1, "imbalance_cost_s": [0.01, 0.01],
                "movable": [{"id": identifier, "model": 0, "source": 0,
                             "memory_gib": 0.09375, "move_cost_s": 10} for identifier in ("A0a", "A0b")]}
    short = costs.layout_cost_table(fit, horizon_steps=1)
    long = costs.layout_cost_table(fit, horizon_steps=100)
    assert enumerate_targets(costs.attach_sparse_costs(snapshot, [short, short]))["winner"]["moves"] == []
    long_winner = enumerate_targets(costs.attach_sparse_costs(snapshot, [long, long]))["winner"]
    assert set(long_winner["moves"]) == {"A0a", "A0b"}
    assert long_winner["final_counts"] == [[64, 64], [64, 64]]
