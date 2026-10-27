import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def fitter():
    path = Path(__file__).resolve().parents[1] / "experiments/dp_opt/fit_imbalance.py"
    spec = importlib.util.spec_from_file_location("dp_opt_curve_fit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_report(alpha, staircase=False):
    layouts = [(64, 64)]
    for count in (80, 96, 112, 128):
        layouts.extend(((count, 128 - count), (128 - count, count)))
    trials, summary = [], []
    for layout in layouts:
        imbalance = abs(layout[0] - layout[1]) / 128
        ratio = 1 + alpha * (int(imbalance > 0) if staircase else imbalance)
        padding = 128 if staircase and imbalance else max(layout)
        summary.append({"layout": list(layout), "imbalance": imbalance, "paired_slowdown": ratio,
                        "paired_ratios": [ratio] * 3, "paired_slowdown_t95_ci": [ratio, ratio],
                        "cadence_s": 0.02 * ratio})
        for repeat in range(3):
            for role in ("baseline_start", "baseline_end") if not imbalance else ("measured",):
                trials.append({"repeat": repeat, "role": role, "layout": list(layout), "imbalance": imbalance,
                               "stable_window": {"critical_cadence_median_s": 0.02 * ratio,
                                                 "ranks": {str(rank): {"padded_tokens": [padding]} for rank in (0, 1)}}})
    return {"status": "passed", "protocol": {"requests": 128}, "summary": summary, "trials": trials}


def test_exact_affine_cadence_curve_is_recovered(fitter):
    result = fitter.fit_curve(synthetic_report(0.5))
    assert result["alpha"] == pytest.approx(0.5)
    assert result["repeat_alphas"] == pytest.approx([0.5] * 3)
    assert result["descriptive_r_squared"] == pytest.approx(1)
    assert result["passes_predeclared_0_05_residual"]
    assert result["points"][0]["loss_from_mean_ratio"] == pytest.approx(1 - 1 / 1.125)
    assert result["capture_shape_posthoc"]["status"].startswith("not_identifiable")


def test_capture_staircase_is_not_declared_linear(fitter):
    result = fitter.fit_curve(synthetic_report(0.5, staircase=True))
    assert not result["passes_predeclared_0_05_residual"]
    assert result["max_absolute_mean_slowdown_residual"] > 0.3
    assert result["endpoint_alpha"] == pytest.approx(0.5)
    assert result["capture_shape_posthoc"]["a"] == pytest.approx(0)
    assert result["capture_shape_posthoc"]["b"] == pytest.approx(0.5)
    assert result["capture_shape_posthoc"]["max_absolute_mean_slowdown_residual"] == pytest.approx(0)


def test_unverified_results_are_rejected(fitter):
    report = synthetic_report(0.5)
    report["status"] = "running"
    with pytest.raises(ValueError, match="verified"):
        fitter.fit_curve(report)
