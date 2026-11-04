import copy
import json

import pytest

from analyze_return_token_windows import analyze, cadence_map, code_absent_end, paired_token_windows, summarize_blocks


def record(origin, step=1., count=16):
    return {"request_id": "conv-1", "prompt_tokens": 4096, "token_ids": [10] * count,
            "model": 1, "first_token_at": origin + step, "completed_at": origin + count * step,
            "cadence": [{"tokens": tokens, "time": origin + tokens * step} for tokens in range(1, count + 1)]}


def test_same_ordinal_intervals_exclude_pre_return_and_preserve_both_clocks():
    result = paired_token_windows(record(100), record(200, .8), 100, 200, 4, 12, 2, 1)
    assert result["first_token_ordinal"] == 6
    assert result["blocks"]
    for block in result["blocks"]:
        assert block["baseline_elapsed_interval_s"][0] >= 4
        assert block["treatment_elapsed_interval_s"][0] >= 4
        assert block["baseline_elapsed_interval_s"][1] < 12
        assert block["treatment_elapsed_interval_s"][1] < 12
        assert block["baseline_TPOT_ms"] == pytest.approx(1000)
        assert block["treatment_TPOT_ms"] == pytest.approx(800)
    assert summarize_blocks(result["blocks"])["reduction_percent"] == pytest.approx(20)


def test_next_code_boundary_excludes_crossing_blocks():
    result = paired_token_windows(record(0), record(0), 0, 0, 4, 7, 2, 0)
    assert [(block["token_start"], block["token_end"]) for block in result["blocks"]] == [(4, 6)]
    assert result["outside_common_window_blocks"] > 0


def test_no_remaining_baseline_tokens_is_explicit_exclusion():
    result = paired_token_windows(record(0, .1), record(0), 0, 0, 2, 15, 2, 0)
    assert not result["blocks"]
    assert result["exclusion"] == "one_policy_has_no_tokens_after_common_start"
    assert summarize_blocks([])["reduction_percent"] is None


def test_no_complete_block_after_settlement_is_explicit_exclusion():
    result = paired_token_windows(record(0), record(0), 0, 0, 1, 15, 128, 32)
    assert not result["blocks"]
    assert result["exclusion"] == "no_full_common_ordinal_block_after_settlement"


@pytest.mark.parametrize("field", ["request_id", "prompt_tokens", "token_ids"])
def test_mismatched_request_or_tokens_rejected(field):
    before, after = record(0), record(0)
    after[field] = {"request_id": "other", "prompt_tokens": 2048, "token_ids": [20] * 16}[field]
    with pytest.raises(ValueError, match="identical"):
        paired_token_windows(before, after, 0, 0, 1, 10, 2, 0)


def test_coalesced_cadence_missing_ordinals_are_not_interpolated():
    before, after = record(0), record(0)
    after["cadence"] = [sample for sample in after["cadence"] if sample["tokens"] % 2 == 0]
    result = paired_token_windows(before, after, 0, 0, 1, 15, 3, 0)
    assert result["missing_ordinal_blocks"] > 0


@pytest.mark.parametrize("change", ["backward_time", "duplicate_ordinal", "incomplete", "nonfinite"])
def test_invalid_cadence_rejected(change):
    sample = copy.deepcopy(record(0))
    if change == "backward_time":
        sample["cadence"][1]["time"] = 0
    elif change == "duplicate_ordinal":
        sample["cadence"][1]["tokens"] = 1
    elif change == "incomplete":
        sample["cadence"].pop()
    else:
        sample["cadence"][1]["time"] = float("nan")
    with pytest.raises(ValueError):
        cadence_map(sample, 0)


def test_overlapping_code_has_no_code_absent_window():
    code = {"request_id": "code-1", "model": 0, "arrived_at": 10, "completed_at": 20}
    other = {"request_id": "code-2", "model": 0, "arrived_at": 19, "completed_at": 30}
    assert code_absent_end([code, other], code, 0, 100) is None
    other["arrived_at"] = 40
    assert code_absent_end([code, other], code, 0, 100) == 40


@pytest.mark.parametrize("window,settle", [(0, 32), (128, -1), (True, 32)])
def test_invalid_window_configuration_rejected(window, settle):
    with pytest.raises(ValueError):
        paired_token_windows(record(0), record(0), 0, 0, 1, 15, window, settle)


def test_negative_reduction_is_not_hidden():
    result = paired_token_windows(record(0, .8), record(0), 0, 0, 2, 12, 2, 0)
    assert summarize_blocks(result["blocks"])["reduction_percent"] == pytest.approx(-25)


def test_complete_closed_phase_analyzer_matches_outputs_and_both_code_releases(tmp_path):
    (tmp_path / "exit-status").write_text("0\n")
    (tmp_path / "trace.json").write_text(json.dumps({"duration_s": 20, "requests": [
        {"request_id": "conv-1"}, {"request_id": "code-1"}]}))
    for policy, origin, step in (("random_no_return", 100, 1.), ("kv_only_workaware", 200, .8)):
        directory = tmp_path / ("r0-" + policy)
        directory.mkdir()
        code = {"request_id": "code-1", "model": 0, "arrived_at": origin + 1,
                "first_token_at": origin + 1.5, "completed_at": origin + 2}
        (directory / "requests.json").write_text(json.dumps([record(origin, step), code]))
        (directory / "workload.json").write_text(json.dumps({"origin": origin}))
        decisions = [] if policy == "random_no_return" else [{
            "path": "release_rebalance_ilp", "reason": "rebalanced", "started_at": origin + 3,
            "resume_completed_at": origin + 3.5, "moves": [{"model": 1, "request_id": "conv-1"}]}]
        (directory / "decisions.json").write_text(json.dumps(decisions))
    result = analyze(tmp_path, window_tokens=2, settle_tokens=1)
    assert result["summary"]["blocks"] > 0
    assert result["summary"]["reduction_percent"] == pytest.approx(20)
    assert result["Code_windows"][0]["common_elapsed_window_s"] == [3.5, 20]
    assert len(result["source_sha256"]) == 8
    assert "not_causal_or_net_benefit" in result["status"]


@pytest.mark.parametrize("marker", [None, "1"])
def test_active_or_failed_phase_is_rejected(tmp_path, marker):
    if marker is not None:
        (tmp_path / "exit-status").write_text(marker)
    with pytest.raises(ValueError, match="closed successful"):
        analyze(tmp_path)


def test_invalid_settings_rejected_even_without_return_events(tmp_path):
    with pytest.raises(ValueError, match="positive token blocks"):
        analyze(tmp_path, window_tokens=0)
