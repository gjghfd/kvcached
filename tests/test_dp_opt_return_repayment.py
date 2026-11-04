import importlib
import json
import sys

import pytest


@pytest.fixture
def repayment(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("analyze_return_repayment")


def record(times, origin=0):
    return {"request_id": "Conv", "model": 1, "prompt_tokens": 2048,
            "token_ids": list(range(len(times))),
            "cadence": [{"tokens": index + 1, "time": origin + value} for index, value in enumerate(times)]}


def span(repayment, times, **options):
    parameters = {"origins": [100, 200], "event": {"start_s": 3.1, "resume_s": 5.9},
                  "release_bounds": [0, 0], "end_bounds": [20, 20], **options}
    return repayment.paired_repayment_span(record(range(1, 9), 100), record(times, 200), **parameters)


def test_pause_is_included_and_repayment_measured(repayment):
    result = span(repayment, [1, 2, 3, 6, 6.2, 6.4, 6.6, 6.8])
    assert result["anchor_token_ordinal"] == 3
    assert result["matched_token_intervals"] == 5
    assert result["maximum_observed_debt_s"] == pytest.approx(2)
    assert result["net_saved_s"] == pytest.approx(1.2)
    assert result["first_repaid_prefix"]["token_ordinal"] == 7
    assert result["sustained_repaid_prefix"]["token_ordinal"] == 7
    assert result["return_control_elapsed_s"] == pytest.approx(2.8)


def test_temporary_positive_prefix_is_not_sustained_repayment(repayment):
    result = span(repayment, [1, 2, 3, 3.5, 4, 8, 8.1, 8.2], event={"start_s": 3.1, "resume_s": 3.3})
    assert result["first_repaid_prefix"]["token_ordinal"] == 4
    assert result["sustained_repaid_prefix"] is None
    assert result["net_saved_s"] == pytest.approx(-.2)


def test_both_Code_absent_clocks_bound_the_span(repayment):
    result = span(repayment, [1, 2, 3, 6, 6.2, 6.4, 6.6, 6.8], end_bounds=[7, 6.5])
    assert result["last_token_ordinal"] == 6
    assert result["net_saved_s"] == pytest.approx(-.4)
    assert result["sustained_repaid_prefix"] is None


def test_anchor_before_baseline_Code_release_is_excluded(repayment):
    result = span(repayment, [1, 2, 3, 6, 6.2, 6.4, 6.6, 6.8], release_bounds=[4, 0])
    assert result["exclusion"] == "no_common_pre_return_ordinal_after_both_Code_releases"


def test_anchor_before_treatment_Code_release_is_excluded(repayment):
    result = span(repayment, [1, 2, 3, 6, 6.2, 6.4, 6.6, 6.8], release_bounds=[0, 4])
    assert result["exclusion"] == "no_common_pre_return_ordinal_after_both_Code_releases"


def test_missing_post_resume_tokens_is_explicit(repayment):
    result = span(repayment, [1, 2, 3, 6, 6.2, 6.4, 6.6, 6.8], end_bounds=[20, 6])
    assert result["exclusion"] == "no_common_post_resume_ordinal_in_both_windows"


@pytest.mark.parametrize("field,value", [("request_id", "other"), ("model", 0), ("prompt_tokens", 2049),
                                        ("token_ids", list(range(7)) + [99])])
def test_mismatched_requests_or_tokens_are_rejected(repayment, field, value):
    before, after = record(range(1, 9)), record(range(1, 9))
    after[field] = value
    with pytest.raises(ValueError, match="identical paired"):
        repayment.paired_repayment_span(before, after, [0, 0], {"start_s": 3, "resume_s": 4}, [0, 0], [20, 20])


@pytest.mark.parametrize("event", [{"start_s": 4, "resume_s": 3}, {"start_s": float("nan"), "resume_s": 4},
                                 {"start_s": 3, "resume_s": True}])
def test_invalid_return_timestamps_are_rejected(repayment, event):
    with pytest.raises(ValueError):
        span(repayment, range(1, 9), event=event)


@pytest.mark.parametrize("field,value", [("origins", [0]), ("release_bounds", [0]), ("end_bounds", [20]),
                                        ("release_bounds", [0, float("inf")]), ("end_bounds", [0, 20])])
def test_invalid_clocks_or_windows_are_rejected(repayment, field, value):
    with pytest.raises(ValueError):
        span(repayment, range(1, 9), **{field: value})


def test_nonmonotone_cadence_is_rejected(repayment):
    with pytest.raises(ValueError, match="monotone"):
        span(repayment, [1, 2, 3, 6, 5, 6.4, 6.6, 6.8])


def test_sparse_cadence_does_not_invent_prefix_samples(repayment):
    before, after = record(range(1, 9)), record([1, 2, 3, 6, 6.2, 6.4, 6.6, 6.8])
    before["cadence"] = [sample for sample in before["cadence"] if sample["tokens"] in (3, 5, 8)]
    result = repayment.paired_repayment_span(before, after, [0, 0], {"start_s": 3.1, "resume_s": 5.9}, [0, 0], [20, 20])
    assert result["matched_prefix_samples"] == 2
    assert result["matched_token_intervals"] == 5
    assert result["sustained_repaid_prefix"]["token_ordinal"] == 8


@pytest.fixture
def phase(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    (root / "exit-status").write_text("0\n")
    (root / "trace.json").write_text(json.dumps({"duration_s": 20,
        "requests": [{"request_id": "Conv"}, {"request_id": "Code"}]}))
    for policy, origin, times in (("random_no_return", 100, list(range(1, 9))),
                                  ("kv_only_workaware", 200, [1, 2, 3, 6, 6.2, 6.4, 6.6, 6.8])):
        directory = root / ("r0-" + policy)
        directory.mkdir()
        requests = [record(times, origin), {"request_id": "Code", "model": 0,
                                           "arrived_at": origin, "completed_at": origin + .5}]
        (directory / "requests.json").write_text(json.dumps(requests))
        (directory / "workload.json").write_text(json.dumps({"origin": origin}))
        decisions = [] if policy == "random_no_return" else [{"path": "release_rebalance_ilp",
            "reason": "rebalanced", "started_at": origin + 3.1, "resume_completed_at": origin + 5.9,
            "moves": [{"model": 1}]}]
        (directory / "decisions.json").write_text(json.dumps(decisions))
    return root


def test_complete_phase_tracks_hashes_and_inclusive_savings(repayment, phase):
    result = repayment.analyze(phase)
    assert result["included_spans"] == 1
    assert result["excluded_spans"] == 0
    assert result["observed_span_saved_request_s"] == pytest.approx(1.2)
    assert len(result["source_files_sha256"]) == 8
    assert result["windows"][0]["release_relative_times_s"] == [.5, .5]


def test_no_return_is_not_reported_as_zero_cost_benefit(repayment, phase):
    (phase / "r0-kv_only_workaware/decisions.json").write_text("[]")
    result = repayment.analyze(phase)
    assert result["included_spans"] == 0
    assert result["windows"][0]["exclusion"] == "no_committed_return"


@pytest.mark.parametrize("marker", ["1\n", ""])
def test_failed_or_active_phase_is_rejected(repayment, phase, marker):
    (phase / "exit-status").write_text(marker)
    with pytest.raises(ValueError, match="closed successful"):
        repayment.analyze(phase)


def test_incomplete_request_set_is_rejected(repayment, phase):
    path = phase / "r0-kv_only_workaware/requests.json"
    path.write_text(json.dumps(json.loads(path.read_text())[:1]))
    with pytest.raises(ValueError, match="all offered"):
        repayment.analyze(phase)


def test_overlapping_Code_is_excluded_under_both_clocks(repayment, phase):
    path = phase / "trace.json"
    trace = json.loads(path.read_text())
    trace["requests"].append({"request_id": "other-Code"})
    path.write_text(json.dumps(trace))
    for policy, origin in (("random_no_return", 100), ("kv_only_workaware", 200)):
        path = phase / ("r0-" + policy) / "requests.json"
        requests = json.loads(path.read_text())
        requests.append({"request_id": "other-Code", "model": 0,
                         "arrived_at": origin + .25, "completed_at": origin + 10})
        path.write_text(json.dumps(requests))
    result = repayment.analyze(phase)
    assert result["windows"][0]["exclusion"] == "overlapping_Code_or_no_positive_Code_absent_window"


def test_cli_checks_SHA_before_analyzing_and_preserves_failed_output(repayment, phase, monkeypatch):
    manifest = phase.parent / "raw.sha256"
    manifest.write_text("bad SHA")
    output = phase.parent / "result.json"
    monkeypatch.setattr(sys, "argv", ["analysis", "--root", str(phase), "--manifest", str(manifest),
        "--prompt-bank", str(phase.parent / "bank.json"), "--output", str(output)])

    def reject_SHA(*arguments):
        raise ValueError("SHA mismatch")

    monkeypatch.setattr(repayment, "verify_raw", reject_SHA)
    with pytest.raises(ValueError, match="SHA mismatch"):
        repayment.main()
    assert not output.exists()


def test_cli_does_not_mutate_archived_raw(repayment, phase, monkeypatch):
    output = phase / "result.json"
    monkeypatch.setattr(sys, "argv", ["analysis", "--root", str(phase), "--manifest", "manifest",
        "--prompt-bank", "bank", "--output", str(output)])
    with pytest.raises(SystemExit):
        repayment.main()
    assert not output.exists()
