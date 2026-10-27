import copy
import hashlib
import importlib
import json
import statistics
from pathlib import Path

import pytest


@pytest.fixture
def analyzer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("analyze_tail_slo")


def record(identifier, model, origin, latency):
    return {"request_id": identifier, "model": model, "prompt_tokens": 16,
            "arrived_at": origin, "submitted_at": origin, "reserved_at": origin,
            "first_token_at": origin + 1, "completed_at": origin + latency, "token_ids": [1, 2]}


def write_trial(root, analyzer):
    marks = [{"request_id": identifier, "model": model, "prompt_tokens": 16,
              "output_tokens": 2, "arrival_s": 0} for identifier, model in (("code", 0), ("conv", 1))]
    trace = {"duration_s": 100, "requests": marks}
    (root / "trace.json").write_text(json.dumps(trace))
    digest = hashlib.sha256((root / "trace.json").read_bytes()).hexdigest()
    (root / "config.json").write_text(json.dumps({"deadline_s": 120, "policies": ["no_move", "kv_only"]}))
    (root / "summary.json").write_text(json.dumps({"status": "passed"}))
    (root / "exit-status").write_text("0\n")
    for policy, origin, latencies in (("no_move", 100, (140, 20)), ("kv_only", 1000, (90, 30))):
        directory = root / f"r0-{policy}"
        directory.mkdir()
        records = [record(mark["request_id"], mark["model"], origin, latency)
                   for mark, latency in zip(marks, latencies)]
        (directory / "requests.json").write_text(json.dumps(records))
        (directory / "workload.json").write_text(json.dumps({"origin": origin, "requests": marks,
                                                            "trace_sha256": digest, "duration_s": 100}))
        (directory / "summary.json").write_text(json.dumps(analyzer.request_metrics(records, origin, 100, 120)))


def test_quantiles_match_inclusive_small_sample_interpolation(analyzer):
    result = analyzer.distribution([40, 10, 30, 20])
    assert result["p90_s"] == pytest.approx(37)
    assert result["p95_s"] == pytest.approx(38.5)
    assert result["p99_s"] == pytest.approx(39.7)
    assert result["max_s"] == 40
    assert analyzer.distribution([4])["p99_s"] == 4
    assert analyzer.distribution([]) is None


@pytest.mark.parametrize("samples", [[5, 5, 5, 5], [40, 10, 30, 20], list(range(749))])
def test_quantiles_match_independent_stdlib_estimator(analyzer, samples):
    expected = statistics.quantiles(samples, n=100, method="inclusive")
    actual = analyzer.distribution(samples)
    assert actual["p90_s"] == pytest.approx(expected[89])
    assert actual["p95_s"] == pytest.approx(expected[94])
    assert actual["p99_s"] == pytest.approx(expected[98])


def test_empty_group_and_single_output_tpot_remain_explicit(analyzer):
    empty = analyzer.summarize_group([], 120)
    assert empty["deadline_met_fraction"] is None
    assert empty["metrics"]["latency_s"] is None
    single = record("one", 0, 100, 120)
    single["token_ids"] = [1]
    result = analyzer.summarize_group([single], 120)
    assert result["metrics"]["tpot_s"] is None
    assert result["deadline_met"] == 1


def test_slo_rescues_and_regressions_use_fixed_inclusive_threshold(analyzer):
    before = {"rescued": record("rescued", 0, 0, 121), "regressed": record("regressed", 1, 0, 120)}
    after = {"rescued": record("rescued", 0, 1000, 120), "regressed": record("regressed", 1, 1000, 130)}
    result = analyzer.compare_group(before, after, set(before), 120)
    assert result["rescued_request_ids"] == ["rescued"]
    assert result["regressed_request_ids"] == ["regressed"]
    assert result["deadline_met_difference"] == 0
    assert result["paired_latency_sum_delta_s"] == 9


def test_complete_trace_analysis_preserves_old_conv_and_source_hashes(analyzer, tmp_path):
    write_trial(tmp_path, analyzer)
    result = analyzer.analyze(tmp_path)
    assert result["groups"]["all_requests"]["deadline_met_difference"] == 1
    assert result["groups"]["Code"]["treatment"]["expected_top_one_percent_samples"] == .01
    assert result["groups"]["baseline_resident_Conv_at_Code_arrival"]["request_ids"] == ["conv"]
    assert result["groups"]["Conv"]["paired_latency_sum_delta_s"] == 10
    assert result["Code_paired_requests"][0]["baseline"]["latency_s"] == 140
    assert len(result["source_sha256"]) == 10


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "model", "length", "time", "nonfinite"])
def test_record_validation_rejects_incomplete_or_inconsistent_data(analyzer, mutation):
    original = record("code", 0, 100, 120)
    records = [copy.deepcopy(original)]
    marks = {"code": {"model": 0, "prompt_tokens": 16, "output_tokens": 2}}
    if mutation == "missing":
        records = []
    elif mutation == "duplicate":
        records.append(copy.deepcopy(original))
    elif mutation == "model":
        records[0]["model"] = 1
    elif mutation == "length":
        records[0]["token_ids"] = [1]
    elif mutation == "time":
        records[0]["reserved_at"] = 300
    elif mutation == "nonfinite":
        records[0]["completed_at"] = float("inf")
    with pytest.raises(ValueError):
        analyzer.validate_records(records, marks)


@pytest.mark.parametrize("mutation", ["summary", "workload", "tokens", "exit"])
def test_trial_validation_rejects_summary_trace_token_or_exit_mismatch(analyzer, tmp_path, mutation):
    write_trial(tmp_path, analyzer)
    directory = tmp_path / "r0-kv_only"
    if mutation == "summary":
        path = directory / "summary.json"
        value = json.loads(path.read_text())
        value["p99_latency_s"] += 1
    elif mutation == "workload":
        path = directory / "workload.json"
        value = json.loads(path.read_text())
        value["trace_sha256"] = "wrong"
    elif mutation == "tokens":
        path = directory / "requests.json"
        value = json.loads(path.read_text())
        value[0]["token_ids"] = [3, 4]
    else:
        (tmp_path / "exit-status").write_text("1\n")
        path = None
    if path is not None:
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        analyzer.analyze(tmp_path)
