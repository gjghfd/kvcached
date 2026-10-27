import importlib
from pathlib import Path

import pytest


@pytest.fixture
def analyzer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("analyze_workaware_partition_cost")


def rows(population=4):
    result = []
    for repeat in (0, 1):
        for context in (4000, 8000, 16000):
            for count in range((population + 1) // 2 + 1, population + 1):
                delta = context * (count - (population + 1) // 2)
                result.append({"model": 1, "population": population, "rank_zero": count,
                    "prompt_tokens": context, "repeat": repeat, "graph_delta": 0,
                    "peak_context_delta_tokens": delta,
                    "measured_cadence_delta_s": count * .001 + 1e-7 * delta})
    return result


def test_actual_counts_are_distinct_inside_the_same_sparse_graph(analyzer):
    result = analyzer.compare_cost_models(rows())
    profile = result["profiles"]["1:4"]
    assert profile["candidate_fit"]["partition_penalties_s"] == pytest.approx({"2": 0, "3": .003, "4": .004})
    assert profile["candidate_fit"]["attention_token_s"] == pytest.approx(1e-7)
    assert profile["candidate"]["bound_ms"] == pytest.approx(0, abs=1e-10)
    assert profile["old"]["bound_ms"] > 0
    assert profile["candidate_fit"]["complete_count_domain"]
    assert result["status"] == "offline_heldout_diagnostic_not_installed_not_GPU_benefit"


def test_unmeasured_count_levels_are_not_interpolated(analyzer):
    selected = [row for row in rows(8) if row["rank_zero"] == 8]
    profile = analyzer.fit_partition_rows(selected)
    assert profile["missing_max_counts"] == [5, 6, 7]
    assert not profile["complete_count_domain"]
    unknown = {**selected[0], "rank_zero": 6}
    with pytest.raises(ValueError, match="unmeasured"):
        analyzer.predict_partition_contrast(unknown, profile)


def test_all_heldout_repeats_and_adverse_residuals_are_retained(analyzer):
    selected = rows()
    for row in selected:
        if row["repeat"] == 1:
            row["measured_cadence_delta_s"] += .001
    result = analyzer.compare_cost_models(selected)
    assert result["contrasts"] == len(selected)
    assert result["profiles"]["1:4"]["heldout_repeats"] == [0, 1]
    assert result["profiles"]["1:4"]["candidate"]["bound_ms"] >= 1 - 1e-10


def test_duplicate_contrasts_cannot_weight_a_favorable_layout_twice(analyzer):
    selected = rows()
    with pytest.raises(ValueError, match="Duplicate"):
        analyzer.compare_cost_models([*selected, selected[0]])


def test_holdout_and_identifiability_are_required(analyzer):
    with pytest.raises(ValueError, match="held-out"):
        analyzer.compare_cost_models([row for row in rows() if row["repeat"] == 0])
    with pytest.raises(ValueError, match="identifiable"):
        analyzer.fit_partition_rows([row for row in rows() if row["prompt_tokens"] == 4000])


@pytest.mark.parametrize("field,value", [("population", True), ("population", 1), ("rank_zero", -1),
    ("rank_zero", 5), ("rank_zero", 2), ("peak_context_delta_tokens", 0),
    ("peak_context_delta_tokens", float("nan")), ("measured_cadence_delta_s", float("inf")),
    ("model", True), ("model", 3), ("repeat", -1)])
def test_invalid_raw_features_are_rejected(analyzer, field, value):
    selected = rows()
    selected[0] = {**selected[0], field: value}
    with pytest.raises(ValueError):
        analyzer.compare_cost_models(selected)
