import pytest

from analyze_numerics_contrast import candidate_contrast


def record(token, values):
    return {"request_id": "request", "model": 1, "token_ids": [5, token],
            "token_logprobs": [{"5": -.1}, values]}


def test_contrast_cancels_softmax_offset_without_accepting_tolerance():
    reference = record(10, {"10": -1.0, "20": -1.01})
    actual = record(20, {"10": -2.011, "20": -2.0})
    comparison = candidate_contrast(actual, reference)
    contrast = comparison["common_candidate_contrast"]
    assert comparison["shared_prefix_tokens"] == 1
    assert contrast["reference_actual_minus_reference_logprob"] == pytest.approx(-.01)
    assert contrast["actual_actual_minus_reference_logprob"] == pytest.approx(.011)
    assert contrast["contrast_change"] == pytest.approx(.021)
    assert comparison["common_top_logprob_max_abs_delta"] > 1
    assert comparison["numerical_root_cause_proven"] is False
    assert comparison["token_tolerance_acceptance"] is False


def test_missing_selected_candidate_does_not_imply_numerical_proof():
    comparison = candidate_contrast(record(20, {"20": -.5, "30": -.6}),
                                    record(10, {"10": -.5, "40": -.6}))
    assert comparison["common_candidate_contrast"] is None
    assert "not_present" in comparison["contrast_unavailable_reason"]


def test_complete_equality_and_nonfinite_rejection():
    reference = record(10, {"10": -.5, "20": -.6})
    assert candidate_contrast(reference, reference)["common_candidate_contrast"] is None
    with pytest.raises(ValueError, match="finite"):
        candidate_contrast(record(20, {"10": float("nan"), "20": -.5}), reference)
