import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def analyzer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("analyze_workaware_cost")


def calibration():
    fit = {"capture_sizes": [1, 8, 16, 32, 64, 128, 256], "residual_bound_s": .00001,
           "coefficients": {"base_s": .01, "graph_batch_s": .001, "attention_token_s": 1e-7}}
    trials = []
    for population in (2, 3, 8, 9):
        for count in sorted({0, population // 2, population - population // 2, population}):
            counts = [count, population - count]
            contexts = [value * 8000 for value in counts]
            graph = next(size for size in fit["capture_sizes"] if size >= max(counts))
            trials.append({"case": {"model": 1, "population": population, "prompt_tokens": 5953,
                                    "repeat": 0, "rank_zero": count},
                           "sample": {"counts": counts, "context_tokens": contexts,
                                      "cadence_s": .01 + .001 * graph + 1e-7 * max(contexts)}})
    return trials, fit


def test_paired_contrasts_separate_same_graph_attention(analyzer):
    trials, fit = calibration()
    result = analyzer.paired_contrasts(trials, fit)
    assert result["contrast_count"] == 8
    assert result["same_graph_attention"]["contrasts"] == 4
    assert result["same_graph_attention"]["slope_s_per_peak_context_token"] == pytest.approx(1e-7)
    assert result["global_prediction_error_s"]["max_abs"] == pytest.approx(0, abs=1e-15)
    assert result["unmatched_groups"] == []


def test_baseline_averages_both_odd_population_directions(analyzer):
    trials, fit = calibration()
    for trial in trials:
        if trial["case"]["population"] == 3 and trial["case"]["rank_zero"] == 1:
            trial["sample"]["cadence_s"] += .002
    result = analyzer.paired_contrasts(trials, fit)
    rows = [row for row in result["contrasts"] if row["population"] == 3]
    assert all(row["balanced_direction_samples"] == 2 for row in rows)
    assert all(row["global_prediction_error_s"] == pytest.approx(-.001) for row in rows)


def test_missing_balanced_case_is_reported_not_imputed(analyzer):
    trials, fit = calibration()
    trials = [trial for trial in trials if trial["case"]["population"] != 2 or trial["case"]["rank_zero"] != 1]
    result = analyzer.paired_contrasts(trials, fit)
    assert len(result["unmatched_groups"]) == 1
    assert not any(row["population"] == 2 for row in result["contrasts"])


def test_duplicate_cases_are_not_independent_repeats(analyzer):
    trials, fit = calibration()
    with pytest.raises(ValueError, match="Duplicate"):
        analyzer.paired_contrasts([*trials, copy.deepcopy(trials[0])], fit)


def test_negative_measured_slope_is_not_clipped_to_fake_benefit(analyzer):
    trials, fit = calibration()
    for trial in trials:
        if trial["case"]["population"] in (3, 8) and trial["case"]["rank_zero"] in (0, trial["case"]["population"]):
            trial["sample"]["cadence_s"] -= .01
    result = analyzer.paired_contrasts(trials, fit)
    assert result["same_graph_attention"]["slope_s_per_peak_context_token"] < 0
    assert result["same_graph_attention"]["measured_positive_fraction"] == 0


def test_global_uncertainty_can_block_all_gain(analyzer):
    trials, fit = calibration()
    fit["residual_bound_s"] = 1
    result = analyzer.paired_contrasts(trials, fit)
    assert not any(row["global_uncertainty_allows_positive_gain_before_control"] for row in result["contrasts"])


@pytest.mark.parametrize("mutation", ("count", "nan", "capture"))
def test_invalid_features_are_rejected(analyzer, mutation):
    trials, fit = calibration()
    if mutation == "count":
        trials[0]["sample"]["counts"] = [1, 1]
    elif mutation == "nan":
        trials[0]["sample"]["context_tokens"][0] = float("nan")
    else:
        fit["capture_sizes"] = [1]
    with pytest.raises(ValueError):
        analyzer.paired_contrasts(trials, fit)
