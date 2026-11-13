import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_nested_timing import (AuditedInputs, GATES, analyze, callback_gaps, clipped, digest,
                                  duration, heartbeat_windows, merged, prepare_intervals, remainder, window_summary)
from test_audit_observability import (archive_job, complete_archive as archive_fixture,
                                     complete_matrix as matrix_fixture, load, save)


def span(scope_id, label, start, end, parent=None, children=(), thread=1):
    return {"scope_id": scope_id, "label": label, "start_ns": start, "end_ns": end,
            "parent": parent, "children": list(children), "thread": thread}


def sample_spans():
    return [span(1, "worker_sample", 0, 1000, children=(2, 3)),
            span(2, "runner_sample", 100, 200, parent=1),
            span(3, "runner_bookkeeping", 300, 900, parent=1, children=(4,)),
            span(4, "runner_to_list", 400, 800, parent=3)]


@pytest.mark.parametrize("window", [(0, 1000), (350, 850), (-100, 1100), (2000, 3000), (100, 100)])
def test_nested_parent_partitions_and_clipping(window):
    intervals = prepare_intervals(sample_spans())
    result = window_summary(intervals, *window)
    expected = max(0, min(1000, window[1]) - max(0, window[0]))
    assert result["outer_union_ns"] == result["exclusive_all_labels_union_ns"] == expected
    assert result["exclusive_label_sum_ns"] == expected
    assert result["cross_label_overlap_ns"] == 0
    assert result["outside_observed_host_scopes_ns"] == window[1] - window[0] - expected
    for scope in result["scopes"].values():
        assert scope["sum_call_inclusive_ns"] == scope["sum_call_direct_child_union_ns"] + scope["sum_call_exclusive_ns"]
    if window == (0, 1000):
        assert result["scopes"]["worker_sample"]["sum_call_direct_child_union_ns"] == 700
        assert result["scopes"]["worker_sample"]["exclusive_union_ns"] == 300
        assert result["scopes"]["runner_bookkeeping"]["inclusive_union_ns"] == 600
        assert result["scopes"]["runner_bookkeeping"]["exclusive_union_ns"] == 200
        assert result["scopes"]["runner_to_list"]["inclusive_union_ns"] == 400


def test_cross_thread_exclusive_times_are_not_added_as_wall_time():
    spans = sample_spans() + [span(5, "worker_dummy", 0, 1000, children=(6,), thread=2),
                              span(6, "dp_coordinate", 200, 800, parent=5, thread=2)]
    result = window_summary(prepare_intervals(spans), 0, 1000)
    assert result["outer_union_ns"] == result["exclusive_all_labels_union_ns"] == 1000
    assert result["exclusive_label_sum_ns"] == 2000 and result["cross_label_overlap_ns"] == 1000


def test_interval_unions_match_exhaustive_integer_coverage():
    candidates = list(itertools.combinations(range(6), 2))
    for first, second in itertools.product(candidates, repeat=2):
        intervals = [first, second]
        covered = {point for start, end in intervals for point in range(start, end)}
        assert duration(merged(intervals)) == len(covered)
        assert duration(remainder(0, 5, intervals)) + len(covered) == 5
        assert duration(clipped(intervals, 1, 4)) == len(covered & {1, 2, 3})


@pytest.mark.parametrize("damage", ["duplicate", "outside", "backwards", "unknown"])
def test_malformed_interval_trees_rejected(damage):
    spans = sample_spans()
    if damage == "duplicate":
        spans.append(spans[0])
    elif damage == "outside":
        spans[-1]["end_ns"] = 901
    elif damage == "backwards":
        spans[-1]["end_ns"] = 399
    else:
        spans[-1]["label"] = "invented"
    with pytest.raises(ValueError):
        prepare_intervals(spans)


def test_merged_callbacks_use_gap_not_per_token_threshold():
    gaps = callback_gaps({"cadence": [[0.0, 1], [0.1, 33], [0.11, 64]], "output_tokens": 64})
    assert len(gaps) == 2 and gaps[0]["long_gap"] and not gaps[1]["long_gap"]
    assert [gap["visible_token_increment"] for gap in gaps] == [32, 31]
    assert all(gap["merged_callback"] for gap in gaps)
    assert callback_gaps({"cadence": [[1.0, 64]], "output_tokens": 64}) == []
    assert not callback_gaps({"cadence": [[0.0, 1], [0.001, 64]], "output_tokens": 64})[0]["long_gap"]


@pytest.mark.parametrize("cadence", [[], [[float("nan"), 64]], [[1, 1], [0, 64]], [[0, 1], [1, 1]], [[0, 1], [1, 63]]])
def test_incomplete_or_backwards_callbacks_rejected(cadence):
    with pytest.raises(ValueError):
        callback_gaps({"cadence": cadence, "output_tokens": 64})


def test_heartbeat_overlap_is_retrospective_and_keeps_zero_delays():
    frontend = {"events": [{"kind": "async_heartbeat", "expected_ns": 0, "at_ns": 300_000_001, "lateness_ns": 300_000_000},
                           {"kind": "thread_heartbeat", "expected_ns": 12, "at_ns": 11, "lateness_ns": 0}]}
    windows = heartbeat_windows(frontend)
    assert duration(clipped(windows["async_heartbeat"], 0, 100_000_000)) == 100_000_000
    assert windows["thread_heartbeat"] == []
    frontend["events"][0]["expected_ns"] = 400_000_000
    with pytest.raises(ValueError):
        heartbeat_windows(frontend)


@pytest.fixture(scope="module")
def audited_archive(tmp_path_factory):
    matrix = matrix_fixture.__wrapped__(tmp_path_factory, SimpleNamespace(param=True))
    root = archive_fixture.__wrapped__(tmp_path_factory, matrix)
    job = root / "evidence/job-123"
    for path in (job / "out/calibration/observability/timing_on").glob("*/timing-frontend.json"):
        record = load(path)
        for event in record["events"]:
            if event["kind"].endswith("heartbeat"):
                event["expected_ns"] = event["at_ns"] - event["lateness_ns"]
        save(path, record)
    archive_job(root / "evidence")
    staging = root / "staging"
    build = load(staging / "build.json")
    audit_path = root / "audit.json"
    command = [sys.executable, str(Path(__file__).with_name("audit_observability.py")), "--evidence", str(root / "evidence"),
               "--staging", str(staging), "--job-id", "123", "--manifest-sha256", build["manifest_sha256"],
               "--tarball-sha256", build["tarball_sha256"], "--plan-sha256",
               digest(staging / "upload/inputs/observability-plan-nested.json"), "--output", str(audit_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return root


def test_complete_analysis_keeps_all_cases_pairs_and_zero_coverage(audited_archive):
    result = analyze(audited_archive / "evidence", audited_archive / "audit.json", digest(audited_archive / "audit.json"), "123")
    assert result["status"] == "descriptive_only_not_predictive_validation", result["errors"]
    assert len(result["cases"]) == 24 and sum(case["requests"] for case in result["cases"]) == 336
    assert len(result["pairs"]) == 12 and result["pairs"] == load(audited_archive / "audit.json")["observability"]["pairs"]
    for case in result["cases"]:
        assert case["status"] == "described"
        if case["condition"] == "timing_on":
            assert len(case["workers"]) == 8
            assert all(worker["measurement_window"]["outer_union_ns"] == 0 for worker in case["workers"])
        else:
            assert case["workers"] is None
    assert all(result[key] is False for key in (*GATES, "estimator_fitted", "gpu_completion_claim", "performance_claim"))


@pytest.mark.parametrize("damage", ["missing_case", "changed_raw", "missing_worker", "bad_nested_source", "missing_scope", "overflow", "bad_heartbeat", "failed_audit", "missing_pairs", "validator_sha"])
def test_failure_keeps_all_case_slots_without_subset_acceptance(tmp_path, audited_archive, damage):
    root = tmp_path / "run"
    shutil.copytree(audited_archive, root)
    audit_path = root / "audit.json"
    audit = load(audit_path)
    relative = "job-123/out/calibration/observability/timing_on/bal2-r0/"
    def rewrite(name, mutate):
        path = root / "evidence" / (relative + name)
        record = load(path)
        mutate(record)
        save(path, record)
        audit["archive"]["member_sha256"][relative + name] = digest(path)
    if damage == "missing_case":
        (root / "evidence" / (relative + "measurement.json")).unlink()
    elif damage == "changed_raw":
        (root / "evidence" / (relative + "requests.json")).write_text("[]")
    elif damage == "missing_worker":
        rewrite("timing-stop.json", lambda record: record["results"][0]["workers"].pop())
    elif damage == "bad_nested_source":
        rewrite("timing-stop.json", lambda record: record["results"][0]["workers"][0]["nested_timing"].update(source_sha256="0" * 64))
    elif damage == "missing_scope":
        rewrite("timing-stop.json", lambda record: record["results"][0]["workers"][0]["events"].pop(2))
    elif damage == "overflow":
        rewrite("timing-stop.json", lambda record: record["results"][0]["workers"][0].update(dropped=1))
    elif damage == "bad_heartbeat":
        rewrite("timing-frontend.json", lambda record: record["events"][1].pop("expected_ns"))
    elif damage == "failed_audit":
        audit["status"] = "failed"
    elif damage == "missing_pairs":
        audit["observability"] = None
    elif damage == "validator_sha":
        audit["frozen"]["source_sha256"]["runtime/nested_timing.py"] = "0" * 64
    save(audit_path, audit)
    result = analyze(root / "evidence", audit_path, digest(audit_path), "123")
    assert result["status"] == "failed"
    assert len(result["cases"]) == (0 if damage == "validator_sha" else 24)
    assert all(result[key] is False for key in GATES)


@pytest.mark.parametrize("damage", [None, "audit_sha", "job"])
def test_cli_refuses_overwrite_and_never_promotes_gates(tmp_path, audited_archive, damage):
    output = tmp_path / "analysis.json"
    audit_path = audited_archive / "audit.json"
    command = [sys.executable, str(Path(__file__).with_name("analyze_nested_timing.py")), "--evidence", str(audited_archive / "evidence"),
               "--audit", str(audit_path), "--audit-sha256", digest(audit_path) if damage != "audit_sha" else "0" * 64,
               "--job-id", "999" if damage == "job" else "123", "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == int(damage is not None), result.stderr
    report = load(output)
    assert all(report[key] is False for key in GATES)
    original = output.read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0 and output.read_bytes() == original


def test_audited_reader_rejects_symlinks_and_nonfinite_json(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    reader = AuditedInputs(tmp_path, {"link.json": digest(target)})
    with pytest.raises(ValueError):
        reader.read("link.json")
    target.write_text('{"invalid": 1e999}')
    reader = AuditedInputs(tmp_path, {"target.json": digest(target)})
    with pytest.raises(ValueError):
        reader.read("target.json")
