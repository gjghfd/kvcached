import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def recovery(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("analyze_return_recovery")


def sample(elapsed, counts, stable=True):
    return {"elapsed_s": elapsed, "counts": counts, "admission_stable_across_observation": stable}


def code(identifier, arrived, completed):
    return {"model": 0, "request_id": identifier, "arrived_at": arrived, "completed_at": completed}


def test_only_committed_Conv_return_is_counted(recovery):
    template = {"path": "release_rebalance_ilp", "reason": "rebalanced", "started_at": 110,
                "resume_completed_at": 112, "moves": [{"model": 1}]}
    decisions = [template, {**template, "reason": "deferred"},
                 {**template, "moves": [{"model": 0}]}, {**template, "path": "blocked_staged_ilp"}]
    assert recovery.committed_returns(decisions, 100) == [{"start_s": 10, "resume_s": 12,
        "reported_pause_call_s": None, "reported_active_blocking_upper_request_s": None, "moves": [{"model": 1}]}]


def test_population_requires_stable_epoch_and_real_multirequest_Conv(recovery):
    population = [sample(12, [[0, 0], [2, 2]], False), sample(13, [[0, 0], [0, 1]]),
                  sample(14, [[1, 0], [2, 2]]), sample(15, [[0, 0], [0, 4]]), sample(20, [[0, 0], [2, 2]])]
    result = recovery.release_windows([code("Code", 100, 110)], population, [], 100, 30)[0]
    assert result["stable_Conv_only_population_samples"] == 2
    assert result["single_attention_rank_samples"] == 1
    assert result["first_balanced_observation_delay_s"] == 10
    assert result["maximum_population_sample_gap_s"] == 5


def test_next_Code_bounds_recovery_and_does_not_count_future_return(recovery):
    records = [code("first", 100, 110), code("second", 120, 125)]
    population = [sample(11, [[0, 0], [0, 4]]), sample(21, [[0, 0], [2, 2]])]
    returns = [{"start_s": 22, "resume_s": 23}]
    windows = recovery.release_windows(records, population, returns, 100, 40)
    assert windows[0]["window_end_s"] == 20
    assert windows[0]["first_balanced_observation_delay_s"] is None
    assert windows[0]["returns"] == []
    assert windows[1]["first_return_delay_s"] is None


def test_overlapping_Code_is_explicit_not_treated_as_idle(recovery):
    records = [code("first", 100, 110), code("second", 105, 120)]
    windows = recovery.release_windows(records, [], [], 100, 30)
    assert windows[0]["overlapping_Code_at_completion"]
    assert not windows[1]["overlapping_Code_at_completion"]


@pytest.mark.parametrize("counts", ([[0, 0]], [[0, 0], [1, -1]], [[0, 0], [True, 0]], [[0, 0], [1.0, 0]]))
def test_invalid_stable_population_is_rejected(recovery, counts):
    with pytest.raises(ValueError, match="DP counts"):
        recovery.stable_counts(sample(1, counts))


@pytest.mark.parametrize("value", (float("nan"), float("inf"), True, "1"))
def test_invalid_timestamp_is_rejected(recovery, value):
    with pytest.raises(ValueError, match="finite"):
        recovery.finite_time(value)


def test_resume_before_return_start_is_rejected(recovery):
    with pytest.raises(ValueError, match="resume after"):
        recovery.committed_returns([{"path": "release_rebalance_ilp", "reason": "rebalanced",
            "started_at": 20, "resume_completed_at": 19, "moves": [{"model": 1}]}], 0)


@pytest.mark.parametrize("state", ({"phase": "submitted_calibration_then_paired_traces"},
    {"phase": "early_return_lower_rate_SHA_verified_analyzed", "GPU_experiment_complete": False},
    {"phase": "diagnosis_required_no_automatic_duplicate_submission", "GPU_experiment_complete": True}))
def test_cannot_analyze_incomplete_or_failed_campaign(recovery, tmp_path, state):
    with pytest.raises(ValueError, match="full SHA archive"):
        recovery.campaign_result(tmp_path, state)


def test_matched_cohorts_and_tail_timing_preserve_same_request_ids(recovery, tmp_path, monkeypatch):
    marks = [{"request_id": "Code", "model": 0, "arrival_s": 0},
             {"request_id": "before", "model": 1, "arrival_s": 0},
             {"request_id": "after", "model": 1, "arrival_s": 15}]
    records = [{"request_id": "Code", "model": 0, "arrived_at": 100, "reserved_at": 100,
                "first_token_at": 102, "completed_at": 110, "token_ids": [1, 2]},
               {"request_id": "before", "model": 1, "arrived_at": 100, "reserved_at": 100,
                "first_token_at": 105, "completed_at": 130, "token_ids": [1, 2]},
               {"request_id": "after", "model": 1, "arrived_at": 115, "reserved_at": 115,
                "first_token_at": 116, "completed_at": 135, "token_ids": [1, 2]}]
    (tmp_path / "trace.json").write_text(json.dumps({"requests": marks, "duration_s": 40}))
    for policy in ("random_no_return", "kv_only_workaware"):
        directory = tmp_path / ("r0-" + policy)
        directory.mkdir()
        decisions = [{"path": "release_rebalance_ilp", "reason": "rebalanced", "started_at": 112,
                      "resume_completed_at": 114, "moves": [{"model": 1}]}] if policy == "kv_only_workaware" else []
        for name, payload in (("requests", records), ("workload", {"origin": 100}),
                              ("population", [sample(15, [[0, 0], [1, 1]])]), ("decisions", decisions)):
            (directory / (name + ".json")).write_text(json.dumps(payload))
    monkeypatch.setattr(recovery, "paired_window", lambda *args: {"deadline_s": 120})
    result = recovery.analyze_recovery(tmp_path, start_s=0)
    cohort = result["matched_post_return_arrival_cohorts"]["Code"]["comparison"]
    assert cohort["request_ids"] == ["after"]
    assert cohort["baseline"]["requests"] == cohort["treatment"]["requests"] == 1
    assert len(result["source_sha256"]) == 8
    assert all(record["first_token_precedes_all_returns"] is None
               for record in result["policies"]["random_no_return"]["worst_Conv_TTFT"])
    worst = {record["request_id"]: record for record in result["policies"]["kv_only_workaware"]["worst_Conv_TTFT"]}
    assert worst["before"]["first_token_precedes_all_returns"]
    assert not worst["after"]["first_token_precedes_all_returns"]
