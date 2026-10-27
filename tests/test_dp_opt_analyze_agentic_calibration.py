import importlib
from pathlib import Path

import pytest


@pytest.fixture
def analyzer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("analyze_agentic_calibration")


def observations():
    requests = [{"request_id": "code", "model": 0, "arrived_at": 8, "submitted_at": 9,
                 "reserved_at": 10, "first_token_at": 20, "completed_at": 22,
                 "reservation_time_source": "scheduler_monotonic"},
                {"request_id": "conv", "model": 1, "arrived_at": 9, "submitted_at": 12,
                 "reserved_at": 12.2, "first_token_at": 13.5, "completed_at": 16,
                 "reservation_time_source": "scheduler_monotonic"}]
    decisions = [{"request_id": "code", "admitted": True, "elapsed_from_arrival_s": 2.5,
                  "end_to_end_control_s": 2.5},
                 {"request_id": "conv", "admitted": True, "elapsed_from_arrival_s": 4,
                  "end_to_end_control_s": 2}]
    return requests, decisions


def test_control_split_counts_rpc_wait_not_cpu(analyzer):
    result = analyzer.summarize_control(*observations())
    conv = result["records"][1]
    assert conv["control_before_submission_s"] == 1
    assert conv["control_after_submission_s"] == 1
    assert conv["queue_s"] == pytest.approx(3.2)
    assert conv["submission_to_reservation_s"] == pytest.approx(.2)
    assert conv["control_overlaps_code_prefill"]
    assert not conv["arrival_during_code_prefill"]
    assert result["conv_by_control_overlap"]["false"]["mean"]["queue_s"] is None
    assert result["reservation_time_sources"] == {"scheduler_monotonic": 2}


def test_pending_reservation_after_ack_is_separate(analyzer):
    requests, decisions = observations()
    requests[1]["reserved_at"] = 15
    decisions[1].update({"admitted": False, "submitted": True, "reservation_pending": True})
    result = analyzer.summarize_control(requests, decisions)
    assert result["reservation_pending_submissions"] == 1
    assert result["records"][1]["submission_to_reservation_s"] == 3
    assert result["records"][1]["control_after_submission_s"] == 1


def test_interval_end_is_not_overlap(analyzer):
    assert analyzer.interval_peak([(10, 20), (12, 18), (20, 22)]) == 2
    assert not analyzer.interval_overlap(20, 21, [(10, 20)])
    assert analyzer.interval_peak([(1, 2), (2, 3), (3, 3)]) == 1


def test_legacy_reservation_remains_explicit(analyzer):
    requests, decisions = observations()
    requests[1].pop("reserved_at")
    requests[1].pop("reservation_time_source")
    result = analyzer.summarize_control(requests, decisions)
    assert result["reservation_time_sources"]["legacy_submitted_at"] == 1
    assert result["records"][1]["submission_to_reservation_s"] == 0


def test_monitor_missing_interval_not_interpreted_as_zero_load(analyzer):
    result = analyzer.summarize_monitoring(
        [{"elapsed_s": 2}, {"elapsed_s": 100}, {"elapsed_s": 102}, {"elapsed_s": 280}],
        [{"elapsed_s": 104}, {"elapsed_s": 106}, {"elapsed_s": 200}], 300)
    assert result["maximum_accepted_sample_gap_s_including_window_boundaries"] == 178
    assert result["transition_samples_preserved_in_arrival_window"] == 3
    assert result["segments"][2]["accepted_samples"] == 0
    assert result["segments"][2]["transition_samples_preserved"] == 1


def test_monitoring_empty_window_is_fully_unobserved(analyzer):
    result = analyzer.summarize_monitoring([], [], 300)
    assert result["maximum_accepted_sample_gap_s_including_window_boundaries"] == 300
    assert result["accepted_samples_in_arrival_window"] == 0


def test_monitoring_rejects_out_of_order_samples(analyzer):
    with pytest.raises(ValueError, match="ordered"):
        analyzer.summarize_monitoring([{"elapsed_s": 200}, {"elapsed_s": 100}], [], 300)


def test_residency_window_clips_lifetimes_and_rounds_blocks(analyzer):
    requests = [{"model": 1, "submitted_at": 99, "reserved_at": 99,
                 "completed_at": 105, "prompt_tokens": 17, "token_ids": [1],
                 "reservation_time_source": "scheduler_monotonic"},
                {"model": 1, "submitted_at": 105, "reserved_at": 105,
                 "completed_at": 115, "prompt_tokens": 16, "token_ids": [],
                 "reservation_time_source": "scheduler_monotonic"}]
    result = analyzer.summarize_residency_window(requests, 100, 0, 10, 16, 1024**2)
    assert result["models"][0]["time_average_logical_reserved_mib"] == 0
    assert result["models"][1]["time_average_logical_reserved_mib"] == 1.5
    assert result["models"][1]["time_average_resident_count"] == 1
    assert result["models"][1]["peak_logical_reserved_mib"] == 2
    assert result["models"][1]["reservation_time_sources"] == {"scheduler_monotonic": 2}


def test_residency_reconstruction_does_not_use_monitor_samples(analyzer):
    requests = [{"model": 0, "submitted_at": 100, "completed_at": 400,
                 "prompt_tokens": 16, "token_ids": []}]
    result = analyzer.summarize_residency_window(requests, 100, 100, 300, 16, 1024**2)
    assert result["models"][0]["time_average_logical_reserved_mib"] == 1
    assert result["models"][0]["reservation_time_sources"] == {"legacy_submitted_at": 1}


@pytest.mark.parametrize("failure", ["window", "geometry", "lifetime"])
def test_residency_reconstruction_rejects_invalid_inputs(analyzer, failure):
    requests = [{"model": 0, "submitted_at": 1, "completed_at": 2,
                 "prompt_tokens": 16, "token_ids": []}]
    if failure == "lifetime":
        requests[0]["completed_at"] = 0
    with pytest.raises(ValueError, match="Residency|lifetime"):
        analyzer.summarize_residency_window(requests, 0, 0, 0 if failure == "window" else 10,
                                           0 if failure == "geometry" else 16, 1024**2)


@pytest.mark.parametrize("failure", ["duplicate", "missing", "early", "negative"])
def test_control_rejects_invalid_submission_intervals(analyzer, failure):
    requests, decisions = observations()
    if failure == "duplicate":
        decisions.append(dict(decisions[0]))
    elif failure == "missing":
        decisions.pop()
    elif failure == "early":
        decisions[1]["elapsed_from_arrival_s"] = 1
    else:
        decisions[1]["end_to_end_control_s"] = -1
    with pytest.raises(ValueError, match="submission|interval"):
        analyzer.summarize_control(requests, decisions)
