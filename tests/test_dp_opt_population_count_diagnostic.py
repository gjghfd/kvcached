import copy

import pytest

from diagnose_population_count import count_features, diagnose, fit_count, predict_count


def rows():
    result = []
    for repeat in (0, 1):
        for context in (4096, 8192, 16384):
            for rank_zero in (0, 1, 4, 5):
                peak = max(rank_zero, 5 - rank_zero)
                excess = peak - 3
                delta = excess * (context + 1024)
                result.append({"model": 0, "population": 5, "repeat": repeat,
                               "prompt_tokens": context, "rank_zero": rank_zero,
                               "graph_delta": 0, "peak_context_delta_tokens": delta,
                               "measured_cadence_delta_s": excess * .001 + delta * .00000002})
    return result


@pytest.mark.parametrize("variant", ["linear_count", "monotone_count"])
def test_counts_fit_and_heldout_domains_are_identifiable(variant):
    samples = rows()
    fitted = fit_count(samples, variant)
    assert all(predict_count(row, fitted) == pytest.approx(row["measured_cadence_delta_s"])
               for row in samples)
    report = diagnose(samples)
    profile = report["profiles"]["0:5"]
    assert set(profile["heldout"]) == {"repeat", "prompt_tokens", "rank_zero"}
    assert profile["all_domain_bound_ms"][variant] < 1e-8
    assert profile["all_domain_bound_ms"]["graph_only"] > 0
    assert "not_installed" in report["status"]


def test_count_features_are_rank_symmetric():
    sample = rows()[0]
    reverse = {**sample, "rank_zero": 5 - sample["rank_zero"]}
    for variant in ("linear_count", "monotone_count"):
        assert count_features(sample, variant) == count_features(reverse, variant)


def test_monotone_penalties_cannot_decrease_with_count():
    fitted = fit_count(rows(), "monotone_count")
    costs = [predict_count({**rows()[0], "rank_zero": rank_zero,
                            "peak_context_delta_tokens": 0}, fitted) for rank_zero in (3, 4, 5)]
    assert costs == sorted(costs)


def test_single_context_collinearity_is_rejected():
    samples = [row for row in rows() if row["prompt_tokens"] == 4096]
    with pytest.raises(ValueError, match="not identifiable"):
        fit_count(samples, "linear_count")


@pytest.mark.parametrize("field", ["repeat", "prompt_tokens", "rank_zero"])
def test_missing_heldout_domain_is_rejected(field):
    samples = copy.deepcopy(rows())
    for sample in samples:
        sample[field] = samples[0][field]
    with pytest.raises(ValueError):
        diagnose(samples)


def test_graph_transition_not_silently_fit_as_same_graph():
    samples = rows()
    samples[0]["graph_delta"] = 8
    with pytest.raises(ValueError, match="same-graph"):
        fit_count(samples, "linear_count")


def test_mixed_population_rejected():
    samples = rows()
    samples[0]["population"] = 6
    with pytest.raises(ValueError, match="tenant/population"):
        fit_count(samples, "linear_count")


def test_missing_count_domain_is_not_identifiable():
    samples = [row for row in rows() if row["rank_zero"] in (1, 4)]
    with pytest.raises(ValueError, match="not identifiable"):
        fit_count(samples, "monotone_count")


def test_unknown_variant_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        count_features(rows()[0], "universal")
