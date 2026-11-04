import importlib
import hashlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def recorder(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("record_robust_native_result")


@pytest.mark.parametrize("state", [
    {"GPU_experiment_complete": False, "state_last_observed": "COMPLETED", "phase": "robust_native_pairs_SHA_verified_analyzed"},
    {"GPU_experiment_complete": True, "state_last_observed": "RUNNING", "phase": "robust_native_pairs_SHA_verified_analyzed"},
    {"GPU_experiment_complete": True, "state_last_observed": "COMPLETED", "phase": "archiving"}])
def test_incomplete_native_state_cannot_produce_a_performance_report(recorder, state):
    assert not recorder.ready(state)


def test_verified_native_state_is_distinct_from_research_goal_completion(recorder):
    assert recorder.ready({"GPU_experiment_complete": True, "state_last_observed": "COMPLETED",
                           "phase": "robust_native_pairs_SHA_verified_analyzed"})


def test_report_labels_counts_prior_TTFT_and_no_final_Poisson_claim(recorder):
    groups = {name: {"completed_requests": 2 if name == "Code" else 4 if name == "Conv" else 6,
        "mean_ttft_s": 1., "p95_ttft_s": 2., "p99_ttft_s": 3., "mean_tpot_s": .01}
        for name in ("Code", "Conv", "combined")}
    text = recorder.markdown({"pooled": {policy: groups for policy in recorder.POLICIES}}, 269567)
    assert "Code=2, Conv=4, combined=6" in text
    assert "not final Poisson" in text and "Initial TTFT precedes return" in text
    assert "Negative reductions are regressions" in text
    assert "Do not reroll seeds" in text and "research goal complete" in text


@pytest.fixture
def native_files(recorder, tmp_path):
    root, repo = tmp_path / "evidence", tmp_path / "repo"
    root.mkdir()
    (repo / "experiments/dp_opt/reports").mkdir(parents=True)
    state = {"GPU_experiment_complete": True, "state_last_observed": "COMPLETED", "job": 269567,
             "phase": "robust_native_pairs_SHA_verified_analyzed", "frozen_runtime_sha256": {}}
    (root / "campaign-state.json").write_text(json.dumps(state))
    raw = root / "live-trace-269567"
    validation = {"phases": {}}
    for ordinal, phase in enumerate(recorder.PHASES):
        path = raw / phase
        path.mkdir(parents=True)
        offered = [{"request_id": name, "model": model, "arrival_s": arrival, "output_tokens": 4}
                   for name, model, arrival in (("conv0", 1, 0), ("conv1", 1, 1), ("code", 0, 120))]
        (path / "trace.json").write_text(json.dumps({"duration_s": 1200, "requests": offered}))
        for policy in recorder.POLICIES:
            case = path / f"r0-{policy}"
            case.mkdir()
            origin = (ordinal + 1) * 10000
            (case / "workload.json").write_text(json.dumps({"duration_s": 1200, "origin": origin}))
            requests = [{**source, "arrived_at": origin + source["arrival_s"],
                "submitted_at": origin + source["arrival_s"] + .1,
                "reserved_at": origin + source["arrival_s"] + .2,
                "first_token_at": origin + source["arrival_s"] + (2 if policy == "random_no_return" else 1),
                "completed_at": origin + source["arrival_s"] + 8, "token_ids": [1, 2, 3, 4]}
                for source in offered]
            (case / "requests.json").write_text(json.dumps(requests))
        validation["phases"][phase] = {"robust_planned_attempts": 2, "committed_returns": 1,
            "repayment": {"excluded_requests": ["missing-anchor"], "scope": "independent_diagnostic"}}
    (root / "independent-verification.json").write_text(json.dumps(validation))
    manifest = "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(raw)}\n"
                       for path in sorted(raw.rglob("*")) if path.is_file())
    (root / "raw-269567.sha256").write_text(manifest)
    return root, repo, raw, validation


def test_complete_record_io_includes_every_request_and_preserves_exclusions(recorder, native_files):
    root, repo, _, _ = native_files
    result = recorder.record(root, repo)
    pooled = json.loads(Path(result["pooled_metrics"]).read_text())
    assert pooled["pooled"]["random_no_return"]["combined"]["completed_requests"] == 6
    assert pooled["pooled"]["kv_only_workaware"]["Code"]["completed_requests"] == 2
    assert pooled["pooled"]["kv_only_workaware"]["Conv"]["completed_requests"] == 4
    assert pooled["pooled"]["random_no_return"]["Conv"]["mean_ttft_s"] == 2
    assert pooled["workaware_reduction_fractions_vs_random"]["Conv"]["mean_ttft_s"] == .5
    assert pooled["phase_return_diagnostics"][recorder.PHASES[0]]["repayment"]["excluded_requests"] == ["missing-anchor"]
    assert "50.000" in Path(result["report"]).read_text()
    assert len(pooled["raw_manifest_sha256"]) == 64
    assert not result["research_goal_complete"]


def test_record_refuses_to_overwrite_prior_results(recorder, native_files):
    root, repo, _, _ = native_files
    recorder.record(root, repo)
    previous = (root / "native-pooled-metrics.json").read_bytes()
    with pytest.raises(ValueError, match="never overwrite"):
        recorder.record(root, repo)
    assert (root / "native-pooled-metrics.json").read_bytes() == previous


@pytest.mark.parametrize("fault", ["duplicate", "wrong_model", "missing_tokens", "arrival", "nonfinite", "tokens_differ"])
def test_pooled_report_rejects_damaged_or_unmatched_requests(recorder, native_files, fault):
    _, _, raw, validation = native_files
    path = raw / recorder.PHASES[0] / "r0-kv_only_workaware/requests.json"
    requests = json.loads(path.read_text())
    if fault == "duplicate":
        requests[1] = requests[0]
    elif fault == "wrong_model":
        requests[0]["model"] = 0
    elif fault == "missing_tokens":
        requests[0]["token_ids"].pop()
    elif fault == "arrival":
        requests[0]["arrived_at"] += .01
    elif fault == "nonfinite":
        requests[0]["completed_at"] = float("nan")
    else:
        requests[0]["token_ids"][0] = 100
    path.write_text(json.dumps(requests))
    with pytest.raises(ValueError):
        recorder.summarize(raw, validation)


def test_raw_tampering_stops_record_before_performance_output(recorder, native_files):
    root, repo, raw, _ = native_files
    (raw / recorder.PHASES[0] / "trace.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        recorder.record(root, repo)
    assert not (root / "native-pooled-metrics.json").exists()


def test_native_failure_records_diagnosis_without_reporting_or_submitting(recorder, native_files):
    root, repo, _, _ = native_files
    (root / "campaign-state.json").write_text(json.dumps({"phase": "diagnosis_required"}))
    with pytest.raises(RuntimeError, match="Native failure"):
        recorder.run(root, repo)
    status = json.loads((root / "report-followup-state.json").read_text())
    assert status["status"].startswith("diagnosis_required")
    assert not status["GPU_submissions"] and not status["SSH_queries"]
    assert not (root / "native-pooled-metrics.json").exists()


def test_duplicate_consumer_preserves_existing_status(recorder, native_files):
    root, repo, _, _ = native_files
    path = root / "report-followup-state.json"
    path.write_text("original")
    with pytest.raises(RuntimeError, match="duplicate"):
        recorder.run(root, repo)
    assert path.read_text() == "original"
