import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as element_tree

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tests"))

import analyze_causal_calibration as analysis
from test_causal_observation import CLOCK, transport_input


def startup_fixture(root, count=3):
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    identities = [dict(classname="tests.test_fixture", name=f"test_identity_{index}") for index in range(count)]
    contract = dict(schema=2, non_timesliced_only=True, nodes=["tests/test_fixture.py"],
                    expected_count=count, expected_testcases=identities)
    filename = root / "inputs/startup-test-contract.json"
    analysis.write(filename, contract)
    document = element_tree.Element("testsuites")
    suite = element_tree.SubElement(document, "testsuite", tests=str(count), failures="0", errors="0", skipped="0")
    for identity in identities:
        element_tree.SubElement(suite, "testcase", **identity)
    element_tree.ElementTree(document).write(root / "startup-tests.xml", encoding="unicode")
    return dict(schema=2, startup_test_contract_sha256=analysis.digest(filename))


def budget_fixture(output):
    output.mkdir(parents=True, exist_ok=True)
    plan = dict(paired_budget_gib=[38.75, 40.25], budget_margin_gib=3.0)
    memory, devices = {}, {}
    for model in (0, 1):
        memory[str(model)] = []
        for rank in (0, 1):
            lanes = []
            for lane in (0, 1):
                identity = f"fixture-{rank}-{lane}"
                devices[(model + 1, rank, lane)] = identity
                lanes.append(dict(gpu_uuid=identity, rank=rank, tp_rank=lane, tp_size=2,
                                  free_mib=[41.75, 43.5][rank] * 1024, total_mib=81152,
                                  released_unused_allocator_cache=True))
            memory[str(model)].append(lanes)
    record = dict(memory=memory, natural_budget_gib=[38.75, 40.5], budget_gib=[38.75, 40.25],
                  required_budget_gib=[38.75, 40.25], paired_budget_status="verified", margin_gib=3.0)
    record.update(capacity_schema=1, num_ranks=2, tensor_parallel_size=2,
                  summary_snapshots=[[dict(mapped_bytes=0) for rank in (0, 1)] for model in (0, 1)],
                  measurement_times=dict(memory_requested_at=1.0, memory_received_at=2.0, summaries_received_at=3.0),
                  rank_calculations=[dict(rank=rank, minimum_free_bytes=[41.75, 43.5][rank] * 1024**3,
                                         mapped_bytes_by_model=[0, 0], mapped_bytes=0,
                                         unrounded_budget_gib=natural, rounded_budget_gib=natural)
                                     for rank, natural in enumerate([38.75, 40.5])])
    analysis.write(output / "budget.json", record)
    return plan, devices


def test_revision_startup_binds_544_exact_identities(tmp_path):
    protocol = startup_fixture(tmp_path, 544)
    result = analysis.audit_startup(tmp_path, protocol)
    assert result["expected_count"] == result["observed_count"] == 544
    assert result["status"] == "passed"


@pytest.mark.parametrize("damage", ["duplicate", "rename", "missing", "extra", "hidden_failure", "hidden_skip",
                                   "missing_summary", "summary_failure", "contract_hash", "contract_duplicate",
                                   "contract_bool_count", "unsafe_selection", "missing_contract", "malformed_xml"])
def test_revision_startup_rejects_false_count_only_proofs(tmp_path, damage):
    protocol = startup_fixture(tmp_path)
    filename = tmp_path / "startup-tests.xml"
    document = element_tree.parse(filename)
    suite = document.getroot().find("testsuite")
    if damage in ("duplicate", "rename"):
        suite[1].set("name", suite[0].get("name") if damage == "duplicate" else "test_unselected")
    elif damage == "missing":
        suite.remove(suite[0])
    elif damage == "extra":
        element_tree.SubElement(suite, "testcase", classname="tests.test_fixture", name="test_extra")
    elif damage in ("hidden_failure", "hidden_skip"):
        element_tree.SubElement(suite[0], "failure" if damage == "hidden_failure" else "skipped")
    elif damage == "missing_summary":
        del suite.attrib["skipped"]
    elif damage == "summary_failure":
        suite.set("failures", "1")
    document.write(filename, encoding="unicode")
    contract_file = tmp_path / "inputs/startup-test-contract.json"
    if damage.startswith("contract") or damage == "unsafe_selection":
        contract = analysis.read(contract_file)
        if damage == "contract_duplicate":
            contract["expected_testcases"][1] = contract["expected_testcases"][0]
        elif damage == "contract_bool_count":
            contract["expected_count"] = True
        elif damage == "unsafe_selection":
            contract["nodes"] = ["../outside.py"]
        else:
            contract["nodes"].append("tests/test_other.py")
        contract_file.write_text(json.dumps(contract))
        if damage != "contract_hash":
            protocol["startup_test_contract_sha256"] = analysis.digest(contract_file)
    if damage == "missing_contract":
        contract_file.unlink()
    if damage == "malformed_xml":
        filename.write_text("<broken")
    with pytest.raises((ValueError, OSError, element_tree.ParseError)):
        analysis.audit_startup(tmp_path, protocol)


def test_revision_startup_order_is_not_an_identity_change(tmp_path):
    protocol = startup_fixture(tmp_path)
    filename = tmp_path / "startup-tests.xml"
    document = element_tree.parse(filename)
    suite = document.getroot().find("testsuite")
    suite[:] = list(reversed(list(suite)))
    document.write(filename, encoding="unicode")
    assert analysis.audit_startup(tmp_path, protocol)["status"] == "passed"


def test_revision_budget_accepts_natural_drift_but_not_budget_drift(tmp_path):
    plan, devices = budget_fixture(tmp_path)
    result = analysis.audit_paired_budget(tmp_path, plan, devices)
    assert result["status"] == "passed" and result["budget_gib"] == [38.75, 40.25]
    assert result["natural_budget_gib"] == [38.75, 40.5]
    assert result["natural_budget_independently_recomputed"] is True


@pytest.mark.parametrize("damage", ["missing_summaries", "mapped", "bool", "rank", "free", "natural", "time", "schema"])
def test_budget_recomputation_rejects_forged_capacity(tmp_path, damage):
    plan, devices = budget_fixture(tmp_path)
    filename = tmp_path / "budget.json"
    record = analysis.read(filename)
    if damage == "missing_summaries":
        del record["summary_snapshots"]
    elif damage in ("mapped", "bool"):
        record["summary_snapshots"][0][0]["mapped_bytes"] = 1024**3 if damage == "mapped" else True
    elif damage == "rank":
        record["rank_calculations"].reverse()
    elif damage == "free":
        record["memory"]["0"][0][0]["free_mib"] -= 512
    elif damage == "natural":
        record["natural_budget_gib"][0] += 0.25
    elif damage == "time":
        record["measurement_times"]["memory_requested_at"] = 4.0
    else:
        record["capacity_schema"] = 0
    filename.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        analysis.audit_paired_budget(tmp_path, plan, devices)


@pytest.mark.parametrize("damage", ["missing", "clipped", "required", "unavailable", "insufficient", "nan", "bool", "huge",
                                   "margin", "memory_missing", "gpu", "lane", "free", "total", "cache"])
def test_revision_budget_fails_closed(tmp_path, damage):
    plan, devices = budget_fixture(tmp_path)
    filename = tmp_path / "budget.json"
    record = analysis.read(filename)
    if damage in ("clipped", "required"):
        record["budget_gib" if damage == "clipped" else "required_budget_gib"][0] -= 0.25
    elif damage == "unavailable":
        record["paired_budget_status"] = "unavailable"
    elif damage in ("insufficient", "nan", "bool", "huge"):
        record["natural_budget_gib"][0] = {"insufficient": 38.5, "nan": float("nan"), "bool": True, "huge": 1e308}[damage]
    elif damage == "margin":
        record["margin_gib"] = 2.0
    elif damage == "memory_missing":
        record["memory"]["0"][0].pop()
    elif damage in ("gpu", "lane", "free", "total", "cache"):
        key, value = {"gpu": ("gpu_uuid", "foreign"), "lane": ("tp_rank", 1), "free": ("free_mib", -1),
                      "total": ("total_mib", 0), "cache": ("released_unused_allocator_cache", False)}[damage]
        record["memory"]["0"][0][0][key] = value
    filename.write_text(json.dumps(record))
    if damage == "missing":
        filename.unlink()
    with pytest.raises((ValueError, OSError)):
        analysis.audit_paired_budget(tmp_path, plan, devices)


@pytest.mark.parametrize("damage", ["missing_device", "duplicate_device", "invalid_device"])
def test_revision_budget_requires_complete_attested_devices(tmp_path, damage):
    plan, devices = budget_fixture(tmp_path)
    if damage == "missing_device":
        del devices[(1, 0, 0)]
    elif damage == "invalid_device":
        devices[(1, 0, 0)] = None
    else:
        devices = {key: "one_gpu" for key in devices}
    with pytest.raises(ValueError, match="devices"):
        analysis.audit_paired_budget(tmp_path, plan, devices)


@pytest.mark.parametrize("damage", [None, "changed", "extra", "missing", "symlink", "unsafe_contract"])
def test_revision_compute_payload_binds_the_exact_set(tmp_path, damage):
    directory = tmp_path / "runtime"
    directory.mkdir()
    filename = directory / "source.py"
    filename.write_text("fixture = 1\n")
    protocol = dict(compute_payload_sha256={"runtime/source.py": analysis.digest(filename)})
    if damage == "changed":
        filename.write_text("fixture = 2\n")
    elif damage == "extra":
        (directory / "extra.py").write_text("fixture = 1\n")
    elif damage == "missing":
        filename.unlink()
    elif damage == "symlink":
        filename.rename(tmp_path / "outside.py")
        filename.symlink_to(tmp_path / "outside.py")
    elif damage == "unsafe_contract":
        protocol["compute_payload_sha256"]["../outside.py"] = "0" * 64
    if damage is None:
        assert analysis.audit_compute_payload(tmp_path, protocol) == dict(status="passed", files=1)
    else:
        with pytest.raises(ValueError):
            analysis.audit_compute_payload(tmp_path, protocol)


def source_packet(as_of):
    packet, _ = transport_input()

    def retime(value):
        if isinstance(value, list):
            return [retime(item) for item in value]
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key in ("observed_at", "captured_at"):
                    result[key] = as_of - 0.05
                elif key == "available_at":
                    result[key] = as_of - 0.04
                elif key == "collected_at":
                    result[key] = as_of - 0.03
                else:
                    result[key] = retime(item)
            return result
        return value

    return retime(packet)


def linked_packet(packet, as_of):
    records = []
    waves = {(row["model"], row["dp_rank"]): row for row in packet["execution"]["rank_waves"]}
    for source in packet["execution"]["rank_waves"] + packet["execution"]["workers"]:
        model, rank = source["model"], source["dp_rank"]
        origin = waves[(model, rank)]
        row = {key: value for key, value in source.items() if key in (
            "model", "dp_rank", "tp_rank", "role", "schema", "clock_id", "producer_pid", "producer_id")}
        features = dict(membership_sha256="a" * 64, requests=1, scheduled_tokens=1,
                        prompt_tokens_sum=8, computed_accounted_sum=10, received_output_tokens_sum=2)
        row.update(kind="causal_batch_link_v1", status="observed", reason=None, history_retained=False,
                   observed_at=as_of - 0.02, available_at=as_of - 0.01, active=True, revision=3, completed=1, failures=0,
                   latest=dict(engine_epoch="a" * 32, engine_pid=origin["producer_pid"], model=model, dp_rank=rank,
                               sequence=2, operation="step", started_at=as_of - 0.1,
                               step_counter=2, current_wave=1, prefill_schedule_interval=32,
                               rpc="sample_tokens", rpc_dispatched_at=as_of - 0.09, features=features))
        records.append(row)
    return dict(kind="live_batch_link_transport_v1", status="observed", clock_id=CLOCK,
                collected_at=as_of - 0.005, records=records, device_activity="unknown")


def state(identity, model, count, received=None, submitted=10.03, final=False, terminal=None):
    return dict(request_id=identity, model=model, prompt_tokens=8, arrived_at=10.01 if identity == "fast" else 10.02,
                submitted_at=submitted, rank_initial=0 if submitted is not None else None, count=count,
                last_received_at=received, final_output_received=final, stream_terminal=terminal,
                receipt_status="observed" if submitted is not None else "unsubmitted")


def recount(report):
    events = report["events"]
    post = report["counts"].get("post_fixture", 0)
    report["counts"].update(attempted=events[-1]["sequence"] + post, retained=len(events),
                             dropped=events[-1]["sequence"] + post - len(events), invalid=0,
                             bytes_retained=sum(len(json.dumps(event, allow_nan=False, separators=(",", ":")).encode()) for event in events),
                             rounds=sum(event["kind"] == "sample" for event in events),
                             incomplete_rounds=sum(event["kind"] == "sample" and not event["transport_complete"] for event in events),
                             missed_ticks=0)
    report["retention_lossless"] = report["counts"]["dropped"] == 0
    return report


def collection_report():
    protocol = dict(case_id="fixture", clock_id=CLOCK, period_s=0.1, deadline_s=0.08,
                    max_events=200000, max_bytes=134217728, max_requests=64)
    events = []

    def emit(kind, timestamp, **payload):
        events.append(dict(kind=kind, recorded_at=timestamp, case_id="fixture", clock_id=CLOCK, **payload))

    emit("collection_start", 10.0, origin=10.0, protocol=protocol)
    emit("arrival", 10.01, **state("fast", 0, 0, submitted=None))
    emit("arrival", 10.02, **state("queued", 1, 0, submitted=None))
    emit("submission", 10.03, **state("fast", 0, 0))
    previous = {"fast": 0, "queued": 0}
    for identity, model, timestamp, count, final in (("fast", 0, 10.1, 2, False), ("fast", 0, 10.4, 20, False),
                                                   ("fast", 0, 10.55, 32, False), ("fast", 0, 10.8, 48, False),
                                                   ("queued", 1, 10.6, 16, False), ("queued", 1, 10.9, 20, True)):
        emit("output_receipt", timestamp + 0.0001, request_id=identity, model=model, received_at=timestamp,
             previous_count=previous[identity], count=count, finished=final,
             finish_reason="length" if final else None, stop_reason=None,
             model_output_observed_at=None, coalescing_possible=True)
        previous[identity] = count
    emit("submission", 10.5, **state("queued", 1, 0, submitted=10.5))
    for tick, as_of, cohort in ((2, 10.25, [state("fast", 0, 2, 10.1), state("queued", 1, 0, submitted=None)]),
                               (5, 10.58, [state("fast", 0, 32, 10.55), state("queued", 1, 0, submitted=10.5)]),
                               (8, 10.85, [state("fast", 0, 48, 10.8), state("queued", 1, 16, 10.6, submitted=10.5)])):
        transport = source_packet(as_of)
        emit("sample", as_of + 0.0001, tick=tick, scheduled_at=10.0 + tick * 0.1,
             started_at=10.0 + tick * 0.1 + 0.001, as_of=as_of,
             cohort_at_start=cohort, cohort_at_completion=cohort, transport=transport,
             batch_links=linked_packet(transport, as_of), transport_complete=True, error_type=None, cancelled=False)
    cutoff = dict(at=11.0, censored_at=11.0, cohort=[state("fast", 0, 48, 10.8),
                                                       state("queued", 1, 20, 10.9, submitted=10.5, final=True)])
    emit("trace_cutoff", 11.0, **cutoff)
    emit("collection_stop", 11.001, open_requests=["fast", "queued"])
    events.sort(key=lambda event: event["recorded_at"])
    for sequence, event in enumerate(events, 1):
        event["sequence"] = sequence
    return recount(dict(kind="causal_collection_v1", protocol=protocol, events=events, counts={},
                        stopped=True, task_ended=True, sampler_error=None, trace_cutoff=cutoff))


def test_full_cohort_k16_labels_coalescing_and_fast_peer_tail():
    report = collection_report()
    rows = list(analysis.label_queries(report))
    assert len(rows) == 11
    first, second, third = (rows[index] for index in (2, 5, 8))
    assert first["status"] == second["status"] == "labeled"
    assert first["joint_target_s"] == pytest.approx(0.35)
    assert first["features"]["queued_counts"] == [0, 1]
    assert first["targets"][0]["needed_count"] == 18 and first["targets"][0]["reached_count"] == 20
    assert first["targets"][0]["receipt_at"] == 10.4 and first["targets"][0]["coalesced"]
    assert second["joint_target_s"] == pytest.approx(0.22)
    assert second["targets"][0]["needed_count"] == 48
    assert third["joint_target_s"] is None and third["status"] == "partial_or_censored_target"
    assert [target["status"] for target in third["targets"]] == ["right_censored", "terminal_before_target"]
    assert third["joint_lower_bound_s"] == pytest.approx(0.15)
    assert sum(row["status"] == "missing_sample" for row in rows) == 8
    assert all(not row["prediction_gate_passed"] for row in rows)
    assert first["feature_available"], first["source_features"]


def test_future_output_fields_cannot_change_features_or_backdate_targets():
    report = collection_report()
    baseline = list(analysis.label_queries(report))[2]
    changed = copy.deepcopy(report)
    for event in changed["events"]:
        if event["kind"] == "sample":
            for member in event["cohort_at_completion"]:
                member.update(max_tokens=999999, remaining_output_tokens=777777, completed_at=100)
            for row in event["batch_links"]["records"]:
                row["latest"]["future_output"] = 888888
    changed["requests"] = [dict(request_id="fast", count=12345678)]
    recount(changed)
    current = list(analysis.label_queries(changed))[2]
    assert current["features"] == baseline["features"]
    assert current["source_features"] == baseline["source_features"]
    assert current["joint_target_s"] == baseline["joint_target_s"]
    assert not any(word in repr(current["features"]) + repr(current["source_features"])
                   for word in ("max_tokens", "remaining_output_tokens", "future_output", "completed_at"))


def test_post_close_rejection_is_separate_but_raw_counts_are_preserved():
    report = collection_report()
    report["counts"]["post_fixture"] = 17
    recount(report)
    result = analysis.retention_status(report)
    assert result["status"] == "complete_collection_stream"
    assert result["collection_dropped"] == 0 and result["post_close_rejected"] == 17
    assert result["raw_counts"]["dropped"] == 17 and result["raw_retention_lossless"] is False
    assert list(analysis.label_queries(report))[2]["target_available"]


@pytest.mark.parametrize("damage", ["gap", "no_stop", "duplicate", "backward", "bytes", "counter", "clock", "invalid", "no_start"])
def test_loss_or_inconsistent_stop_cannot_be_reclassified_as_post_close(damage):
    report = collection_report()
    if damage == "gap":
        del report["events"][3]
    elif damage == "no_stop":
        report["events"].pop()
    elif damage == "duplicate":
        report["events"][2]["sequence"] = report["events"][1]["sequence"]
    elif damage == "backward":
        report["events"][2]["recorded_at"] = 0
    elif damage == "clock":
        report["events"][2]["clock_id"] = "another-clock"
    elif damage == "no_start":
        report["events"].pop(0)
    recount(report)
    if damage == "bytes":
        report["counts"]["bytes_retained"] += 1
    elif damage == "counter":
        report["counts"]["attempted"] += 1
    elif damage == "invalid":
        report["counts"]["invalid"] = 1
    assert analysis.retention_status(report)["status"] == "unavailable"


def test_all_targets_unavailable_on_retention_gap():
    report = collection_report()
    del report["events"][3]
    recount(report)
    rows = list(analysis.label_queries(report))
    assert rows[2]["status"] == "unavailable_event_stream" and not rows[2]["target_available"]
    assert rows[2]["features"] is not None


def test_no_cleanup_receipt_or_terminal_after_cutoff_can_fill_target():
    report = collection_report()
    receipt = copy.deepcopy(next(event for event in report["events"] if event["kind"] == "output_receipt"))
    receipt.update(count=100, previous_count=48, received_at=11.1, recorded_at=11.1001)
    report["events"].insert(-1, receipt)
    report["events"][-1]["recorded_at"] = 11.2
    for sequence, event in enumerate(report["events"], 1):
        event["sequence"] = sequence
    recount(report)
    final = list(analysis.label_queries(report))[8]
    assert final["targets"][0]["status"] == "right_censored" and final["joint_target_s"] is None


@pytest.mark.parametrize("field,value", [("count", 99), ("submitted_at", 100), ("last_received_at", 100), ("arrived_at", 100)])
def test_bad_asof_cohort_is_not_replaced_from_future_receipts(field, value):
    report = collection_report()
    sample = next(event for event in report["events"] if event["kind"] == "sample")
    sample["cohort_at_completion"][0][field] = value
    recount(report)
    row = list(analysis.label_queries(report))[2]
    assert not row["target_available"]


def test_queries_completed_after_cutoff_keep_explicit_missing_label():
    report = collection_report()
    sample = next(event for event in report["events"] if event["kind"] == "sample")
    sample.update(as_of=11.05, recorded_at=11.06)
    report["events"].sort(key=lambda event: event["recorded_at"])
    report["events"][-1]["kind"] = "sample"
    rows = list(analysis.label_queries(report))
    assert rows[2]["status"] == "sample_crosses_cutoff"


@pytest.mark.parametrize("field,value", [("clock_id", "different"), ("producer_pid", 1), ("observed_at", 0),
                                         ("available_at", 100), ("role", "invented"), ("history_retained", True)])
def test_batch_link_identity_clock_or_availability_failure_is_not_a_feature(field, value):
    report = collection_report()
    sample = next(event for event in report["events"] if event["kind"] == "sample")
    sample["batch_links"]["records"][0][field] = value
    recount(report)
    row = list(analysis.label_queries(report))[2]
    assert not row["feature_available"] and row["target_available"]


def test_unmatched_latest_operations_do_not_become_common_rank_barrier():
    report = collection_report()
    sample = next(event for event in report["events"] if event["kind"] == "sample")
    sample["batch_links"]["records"][-1]["latest"]["sequence"] = 1
    recount(report)
    row = list(analysis.label_queries(report))[2]
    links = row["source_features"]["batch_links"]
    assert links["status"] == "available" and not links["common_rank_barrier"]
    assert links["lanes"][-1]["same_latest_engine_operation"] is False


def overhead_rules():
    return dict(primary_metrics=["conv_inverse_total_received_tokens_per_window_second",
                                 "conv_mean_observed_interreceipt_time_per_token", "code_mean_ttft"],
                window_s=240, window_overshoot_limit_s=0.5, per_metric_median_loss_limit=0.05,
                per_metric_worst_case_loss_limit=0.15)


def performance_fixture():
    case = dict(case="fixture", requests=[dict(request_id="conv", model=1, arrival_s=0, prompt_tokens=8, output_tokens=1000),
                                          dict(request_id="code", model=0, arrival_s=30, prompt_tokens=100000, output_tokens=64)])
    workload = dict(policy="no_move", origin=10.0)
    summary = dict(policy="no_move", status="censored", errors=[], expected=2, censored_at_s=240.1)
    completed = [dict(trace_request_id="code", model=0, prompt_tokens=100000, output_tokens=64, arrived_at=40.0,
                       first_token_at=50.0, completed_at=70.1, cadence=[[50.0, 1], [70.0, 64]])]
    incomplete = [dict(trace_request_id="conv", model=1, prompt_tokens=8, output_tokens=321,
                        target_output_tokens=1000, arrived_at=10.0, first_token_at=11.0, first_token_observed=True,
                        completed_at=250.1, censored=True, censored_at=250.1,
                        cadence=[[11.0, 1], [249.9, 320], [250.1, 321]])]
    return case, workload, summary, completed, incomplete


def test_fixed_window_counts_are_bracketed_not_future_interpolated():
    case, workload, summary, completed, incomplete = performance_fixture()
    result = analysis.request_metrics(case, workload, summary, completed, incomplete, overhead_rules())
    assert result["status"] == "available", result
    metric = result["metrics"]["conv_inverse_total_received_tokens_per_window_second"]
    assert metric["lower"] == pytest.approx(240 / 321)
    assert metric["upper"] == pytest.approx(240 / 320) and metric["point"] is None
    assert result["metrics"]["conv_mean_observed_interreceipt_time_per_token"]["point"] == pytest.approx(239.1 / 320)
    assert result["metrics"]["code_mean_ttft"]["point"] == 10


def test_censor_snapshot_wins_over_later_cleanup_completion():
    case, workload, summary, completed, incomplete = performance_fixture()
    completed.append(dict(incomplete[0], censored=False, output_tokens=1000, completed_at=251.0,
                          cadence=[[11.0, 1], [251.0, 1000]]))
    result = analysis.request_metrics(case, workload, summary, completed, incomplete, overhead_rules())
    assert result["status"] == "available" and result["ignored_post_cutoff_completions"] == ["conv"]
    assert result["requests"][0]["received_at_window"] == [320, 321]


@pytest.mark.parametrize("damage", ["missing", "length", "context", "arrival", "failed", "late_cutoff", "imputed_first"])
def test_missing_failed_or_changed_requests_never_get_full_metric_coverage(damage):
    case, workload, summary, completed, incomplete = performance_fixture()
    if damage == "missing":
        completed.clear()
    elif damage == "length":
        incomplete[0]["target_output_tokens"] = 999
    elif damage == "context":
        incomplete[0]["prompt_tokens"] = 1
    elif damage == "arrival":
        incomplete[0]["arrived_at"] = 9
    elif damage == "failed":
        summary["status"] = "failed"
    elif damage == "late_cutoff":
        summary["censored_at_s"] = 241
    elif damage == "imputed_first":
        incomplete[0].update(first_token_observed=False, output_tokens=0, first_token_at=250.1, completed_at=250.1, cadence=[])
    result = analysis.request_metrics(case, workload, summary, completed, incomplete, overhead_rules())
    assert result["status"] != "available"


def paired_blocks(loss=0.04):
    names = overhead_rules()["primary_metrics"]
    return [dict(block=block, cases={mode: {f"case{index}": dict(performance=dict(metrics={
        name: analysis.interval(1 if mode == "off" else 1 + loss) for name in names})) for index in range(4)}
        for mode in ("off", "collect")}) for block in (0, 1)]


def test_two_orders_all_eight_pairs_are_required():
    cases = [f"case{index}" for index in range(4)]
    good = analysis.compare_blocks(paired_blocks(), cases, overhead_rules())
    assert good["passed"] and len(good["pairs"]) == 8 and not good["prediction_gate_passed"]
    partial = analysis.compare_blocks(paired_blocks()[:1], cases, overhead_rules())
    assert not partial["passed"] and partial["missing_blocks"] == [1] and len(partial["pairs"]) == 8


def test_worst_case_and_missing_metrics_cannot_be_hidden_in_median():
    cases = [f"case{index}" for index in range(4)]
    for missing in (False, True):
        blocks = paired_blocks()
        blocks[1]["cases"]["collect"]["case3"]["performance"]["metrics"]["code_mean_ttft"] = None if missing else analysis.interval(1.16)
        result = analysis.compare_blocks(blocks, cases, overhead_rules())
        assert not result["passed"]


def test_conservative_upper_ratio_decides_interval_acceptance():
    blocks = paired_blocks()
    blocks[1]["cases"]["collect"]["case3"]["performance"]["metrics"]["code_mean_ttft"] = analysis.interval(1, 1.16)
    result = analysis.compare_blocks(blocks, [f"case{index}" for index in range(4)], overhead_rules())
    assert not result["passed"] and result["metrics"]["code_mean_ttft"]["worst_loss_upper"] > 0.15


def test_process_usage_keeps_exited_pids_and_pid_reuse_distinct(tmp_path):
    filename = tmp_path / "usage.jsonl"
    records = [dict(read_failures=[], page_bytes=4096, clock_ticks_per_s=100,
                    processes=[dict(pid=10, start_ticks=1, cpu_ticks=200, rss_pages=5)]),
               dict(read_failures=[], page_bytes=4096, clock_ticks_per_s=100,
                    processes=[dict(pid=10, start_ticks=2, cpu_ticks=100, rss_pages=3)])]
    filename.write_text("\n".join(json.dumps(row) for row in records))
    result = analysis.summarize_process_usage(filename)
    assert result["sampled_processes"] == 2 and result["sum_last_observed_process_cpu_s"] == 3
    assert result["maximum_sampled_group_rss_bytes"] == 5 * 4096
    assert not result["exact_cumulative_cpu"] and not result["true_peak_rss"]


@pytest.mark.parametrize("damage", ["omit_waiting", "hide_as_finished", "wrong_model", "wrong_prompt", "wrong_submission"])
def test_full_cohort_is_reconciled_against_causal_arrivals(damage):
    report = collection_report()
    sample = next(event for event in report["events"] if event["kind"] == "sample")
    if damage == "omit_waiting":
        sample["cohort_at_completion"].pop()
    elif damage == "hide_as_finished":
        sample["cohort_at_completion"][1]["final_output_received"] = True
    elif damage == "wrong_model":
        sample["cohort_at_completion"][0]["model"] = 1
    elif damage == "wrong_prompt":
        sample["cohort_at_completion"][0]["prompt_tokens"] += 1
    else:
        sample["cohort_at_completion"][1]["submitted_at"] = 10.05
    recount(report)
    row = list(analysis.label_queries(report))[2]
    assert not row["target_available"] and not row["feature_available"]


@pytest.mark.parametrize("damage", ["schedule", "cancelled", "transport_flag", "foreign_engine", "foreign_epoch"])
def test_query_features_require_causal_sample_and_origin(damage):
    report = collection_report()
    sample = next(event for event in report["events"] if event["kind"] == "sample")
    if damage == "schedule":
        sample["scheduled_at"] = 10.0
    elif damage == "cancelled":
        sample["cancelled"] = True
    elif damage == "transport_flag":
        sample["transport_complete"] = False
    elif damage == "foreign_engine":
        sample["batch_links"]["records"][-1]["latest"]["engine_pid"] = 1
    else:
        sample["batch_links"]["records"][-1]["latest"]["engine_epoch"] = "b" * 32
    recount(report)
    assert not list(analysis.label_queries(report))[2]["feature_available"]


def test_duplicate_or_noninteger_query_tick_rejected():
    for tick in (2, True, -1):
        report = collection_report()
        samples = [event for event in report["events"] if event["kind"] == "sample"]
        samples[1]["tick"] = tick
        recount(report)
        with pytest.raises(ValueError, match="sample_tick"):
            list(analysis.label_queries(report))


def test_counterfeit_cutoff_cannot_extend_labels():
    report = collection_report()
    report["trace_cutoff"]["at"] = 11.5
    report["trace_cutoff"]["censored_at"] = 11.5
    assert analysis.retention_status(report)["status"] == "unavailable"


@pytest.mark.parametrize("damage", ["negative", "inverted", "nonfinite", "zero_upper"])
def test_invalid_intervals_never_pass_overhead(damage):
    blocks = paired_blocks()
    value = {"negative": dict(lower=-1, upper=-1), "inverted": dict(lower=2, upper=1),
             "nonfinite": dict(lower=1, upper=float("nan")), "zero_upper": dict(lower=1, upper=0)}[damage]
    blocks[0]["cases"]["off"]["case0"]["performance"]["metrics"]["code_mean_ttft"] = value
    assert not analysis.compare_blocks(blocks, [f"case{index}" for index in range(4)], overhead_rules())["passed"]


def test_empty_case_grid_cannot_pass_overhead():
    with pytest.raises(ValueError):
        analysis.compare_blocks(paired_blocks(), [], overhead_rules())


def test_censor_timestamps_must_match_immutable_summary():
    case, workload, summary, completed, incomplete = performance_fixture()
    incomplete[0]["censored_at"] = 300.0
    result = analysis.request_metrics(case, workload, summary, completed, incomplete, overhead_rules())
    assert result["status"] == "unavailable"


def naive_states_at(report, timestamp):
    states = {}
    for event in report["events"]:
        if event["recorded_at"] > timestamp:
            continue
        identity = event.get("request_id")
        if event["kind"] == "arrival":
            states[identity] = {key: copy.deepcopy(event[key]) for key in
                                ("request_id", "model", "prompt_tokens", "arrived_at", "submitted_at", "rank_initial",
                                 "count", "last_received_at", "final_output_received", "stream_terminal", "receipt_status")}
        elif event["kind"] == "submission":
            states[identity].update(submitted_at=event["submitted_at"], rank_initial=event["rank_initial"], receipt_status="observed")
        elif event["kind"] == "output_receipt":
            states[identity].update(count=event["count"], last_received_at=event["received_at"], final_output_received=event["finished"])
        elif event["kind"] == "stream_terminal":
            states[identity]["stream_terminal"] = event["terminal"]
    return [states[identity] for identity in sorted(states)]


def complete_collection_report():
    report = collection_report()
    report["events"] = [event for event in report["events"] if event["kind"] != "sample"]
    for tick in range(10):
        scheduled = 10.0 + tick * 0.1
        started, as_of = scheduled + 0.001, scheduled + 0.06
        transport = source_packet(as_of)
        report["events"].append(dict(kind="sample", case_id="fixture", clock_id=CLOCK, recorded_at=as_of + 0.0001,
                                     tick=tick, scheduled_at=scheduled, started_at=started, as_of=as_of,
                                     cohort_at_start=naive_states_at(report, started),
                                     cohort_at_completion=naive_states_at(report, as_of), transport=transport,
                                     batch_links=linked_packet(transport, as_of), transport_complete=True,
                                     error_type=None, cancelled=False))
    report["events"].sort(key=lambda event: event["recorded_at"])
    for sequence, event in enumerate(report["events"], 1):
        event["sequence"] = sequence
    return recount(report)


def coverage_rules():
    return dict(minimum_active_round_transport_complete_fraction=0.9, maximum_missed_tick_fraction=0.1)


def test_actual_source_completeness_not_only_transport_flag():
    report = complete_collection_report()
    good = analysis.collection_coverage(report, coverage_rules())
    assert good["status"] == "passed", good
    assert good["active_rounds"] == 9 and good["active_transport_fraction"] == 1
    for event in report["events"]:
        if event["kind"] == "sample":
            event["batch_links"]["records"].pop()
    recount(report)
    failed = analysis.collection_coverage(report, coverage_rules())
    assert failed["active_transport_fraction"] == 0 and failed["status"] == "not_passed"


@pytest.mark.parametrize("damage", ["rounds", "incomplete", "missed", "hole", "duplicate_tick"])
def test_sampling_counter_or_grid_holes_cannot_pass_coverage(damage):
    report = complete_collection_report()
    if damage in ("rounds", "incomplete", "missed"):
        field = {"rounds": "rounds", "incomplete": "incomplete_rounds", "missed": "missed_ticks"}[damage]
        report["counts"][field] += 1
    elif damage == "hole":
        report["events"] = [event for event in report["events"] if not (event["kind"] == "sample" and event["tick"] == 2)]
        for sequence, event in enumerate(report["events"], 1):
            event["sequence"] = sequence
        recount(report)
    else:
        samples = [event for event in report["events"] if event["kind"] == "sample"]
        samples[1]["tick"] = 0
        recount(report)
    assert analysis.collection_coverage(report, coverage_rules())["status"] == "not_passed"


def test_independent_linear_oracle_checks_every_query_and_joint_maximum():
    report = complete_collection_report()
    rows = list(analysis.label_queries(report))
    for row in rows:
        if row["as_of"] is None:
            assert row["status"] == "missing_sample"
            continue
        current = naive_states_at(report, row["as_of"])
        expected = {}
        for member in current:
            if member["final_output_received"] or member["stream_terminal"] is not None:
                continue
            hits = [event["received_at"] - row["as_of"] for event in report["events"]
                    if event["kind"] == "output_receipt" and event["request_id"] == member["request_id"]
                    and row["as_of"] < event["received_at"] <= report["trace_cutoff"]["censored_at"]
                    and event["count"] >= member["count"] + 16]
            expected[member["request_id"]] = min(hits) if hits else None
        actual = {target["request_id"]: target["residual_s"] for target in row["targets"]}
        assert actual == expected
        joint = max(expected.values()) if expected and all(value is not None for value in expected.values()) else None
        assert row["joint_target_s"] == joint


def test_attested_worker_pid_mismatch_preserves_targets_but_rejects_features():
    report = complete_collection_report()
    row = list(analysis.label_queries(report, attested_workers={}))[2]
    assert row["target_available"] and not row["feature_available"]
    assert analysis.collection_coverage(report, coverage_rules(), attested_workers={})["status"] == "not_passed"


def terminal_job_fixture(tmp_path, block_id):
    root = tmp_path / f"job-{block_id}"
    root.mkdir()

    def save(relative, value):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        analysis.write(target, value)

    case = dict(case="fixture", requests=[dict(request_id="fast", model=1, arrival_s=0.01, prompt_tokens=8, output_tokens=1000),
                                          dict(request_id="queued", model=0, arrival_s=0.02, prompt_tokens=8, output_tokens=20)])
    plan = dict(schema=2, cases=[dict(case, case=f"case{index}") for index in range(4)],
                paired_budget_gib=[38.75, 40.25], budget_margin_gib=3.0, case_window_s=1,
                max_inflight=[1, 16], reserve_extra_tokens=256, kv_mode="prompt")
    save("inputs/causal-training-plan-20261110-r0.json", plan)
    save("inputs/prompt-bank.json", {"purpose": "synthetic CPU fixture only"})
    startup = startup_fixture(root, 533)
    protocol = dict(startup, paired_budget_gib=plan["paired_budget_gib"],
                    blocks=[["off", "collect"], ["collect", "off"]], collection=complete_collection_report()["protocol"].copy(),
                    reference=dict(k=16), coverage=coverage_rules(), overhead=dict(overhead_rules(), window_s=1),
                    planned_cases_per_condition=4, planned_requests_per_condition=8,
                    training_plan_sha256=analysis.digest(root / "inputs/causal-training-plan-20261110-r0.json"),
                    prompt_bank_sha256=analysis.digest(root / "inputs/prompt-bank.json"))
    for key in ("case_id", "clock_id"):
        protocol["collection"].pop(key)
    save("inputs/deployment-protocol.json", protocol)
    (root / "runtime").mkdir()
    (root / "runtime/fixture.py").write_text("fixture_only = True\n")
    manifest = dict(runtime_sha256={"fixture.py": analysis.digest(root / "runtime/fixture.py")},
                    configuration={"fixture": True}, versions={"fixture": "1"},
                    protocol_sha256=analysis.digest(root / "inputs/deployment-protocol.json"),
                    training_plan_sha256=protocol["training_plan_sha256"], prompt_bank_sha256=protocol["prompt_bank_sha256"])
    save("inputs/deployment-manifest.json", manifest)
    manifest_sha = analysis.digest(root / "inputs/deployment-manifest.json")
    (root / "exit-status").write_text("0\n")
    save("out/causal-block/status.json", dict(status="collected", block=block_id, planned=protocol["blocks"][block_id],
                                             started=protocol["blocks"][block_id], not_executed=[],
                                             completed=[dict(mode=mode, status="passed") for mode in protocol["blocks"][block_id]]))
    packet = source_packet(10.25)
    workers = [dict(model=row["model"], dp_rank=row["dp_rank"], tp_rank=row["tp_rank"], pid=row["producer_pid"])
               for row in packet["execution"]["workers"]]
    for position, mode in enumerate(("off", "collect")):
        mode_root = Path("out/causal-block") / mode
        output = mode_root / "out"
        budget_fixture(root / output)
        save(mode_root / "process.json", dict(status="passed", returncode=0, forced_cleanup=False, timeout=False, pid=9000 + position))
        usage = dict(read_failures=[], page_bytes=4096, clock_ticks_per_s=100,
                     processes=[dict(pid=9000 + position, start_ticks=1, cpu_ticks=100, rss_pages=10)])
        (root / mode_root / "process-usage.jsonl").write_text(json.dumps(usage) + "\n")
        front = dict(status="verified", manifest_sha256=manifest_sha, native_checked=True,
                     versions=manifest["versions"], clock_id=CLOCK, pid=9000 + position)
        save(output / "causal-frontend-attestation.json", front)
        save(output / "causal-final-attestation.json", front)
        save(output / "causal-worker-attestations.json", dict(status="verified", clock_id=CLOCK, workers=workers))
        for worker in workers:
            model, rank, lane = worker["model"] + 1, worker["dp_rank"], worker["tp_rank"]
            attestation = dict(status="verified", manifest_sha256=manifest_sha, configuration=manifest["configuration"],
                               clock_id=CLOCK, pid=worker["pid"])
            save(output / f"worker-instance-{model}-dp{rank}-tp{lane}.json",
                 dict(causal_deployment=attestation, gpu_uuid=f"fixture-{rank}-{lane}", gpu_name="fixture-A800",
                      pid=worker["pid"], service=f"instance-{model}", dp_rank=rank,
                      groups=dict(dp=dict(rank=rank, world_size=2), tp=dict(rank=lane, world_size=2))))
        save(output / "calibration/dataplane.json", dict(status="passed"))
        save(output / "phases.json", dict(phases=[dict(phase="startup", status="started"), dict(phase="startup", status="passed",
             natural_budget_gib=[38.75, 40.5], trace_budget_gib=plan["paired_budget_gib"]),
             dict(phase="job", status="passed")]))
        save(output / "calibration/causal/plan.json", plan)
        save(output / "calibration/causal/mode.json", dict(mode=mode))
        for current in plan["cases"]:
            target = output / "calibration/causal" / current["case"]
            save(target / "case.json", current)
            workload = dict(policy="no_move", origin=10.0, budgets_gib=plan["paired_budget_gib"], num_ranks=2,
                            max_inflight=[1, 16], kv_mode="prompt", reserve_extra_tokens=256, trace_sha256="fixture",
                            duration_s=1, requests=2, native=False, policy_switches={})
            save(target / "trace/workload.json", workload)
            save(target / "trace/summary.json", dict(policy="no_move", status="censored", errors=[], expected=2, censored_at_s=1.0))
            save(target / "trace/requests.json", [dict(trace_request_id="queued", request_id="queued", model=0, prompt_tokens=8,
                                                       output_tokens=20, arrived_at=10.02, first_token_at=10.6,
                                                       completed_at=10.9, cadence=[[10.6, 16], [10.9, 20]])])
            save(target / "trace/incomplete.json", [dict(trace_request_id="fast", request_id="fast", model=1, prompt_tokens=8,
                                                         target_output_tokens=1000, output_tokens=48, arrived_at=10.01,
                                                         first_token_at=10.1, first_token_observed=True, completed_at=10.8,
                                                         censored=True, censored_at=11.0,
                                                         cadence=[[10.1, 2], [10.4, 20], [10.55, 32], [10.8, 48]])])
            if mode == "collect":
                report = complete_collection_report()
                report["protocol"]["case_id"] = current["case"]
                for event in report["events"]:
                    event["case_id"] = current["case"]
                    if "model" in event:
                        event["model"] = 1 - event["model"]
                    for key in ("cohort_at_start", "cohort_at_completion", "cohort"):
                        for member in event.get(key, []):
                            member["model"] = 1 - member["model"]
                save(target / "collection.json", recount(report))
    payload = {str(filename.relative_to(root)): analysis.digest(filename)
               for directory in ("runtime", "tests", "inputs") for filename in (root / directory).rglob("*")
               if filename.is_file()}
    return root, dict(deployment_manifest_sha256=manifest_sha, fixture_only=True,
                      measurement_revision=2, compute_payload_sha256=payload)


@pytest.mark.parametrize("damage", ["same_wrong_budget", "startup_duplicate", "budget_record_missing", "memory_uuid"])
def test_revision_previously_equal_or_count_only_cases_reject(tmp_path, damage):
    first, protocol = terminal_job_fixture(tmp_path, 0)
    second, _ = terminal_job_fixture(tmp_path, 1)
    if damage == "same_wrong_budget":
        for mode in ("off", "collect"):
            filename = first / f"out/causal-block/{mode}/out/calibration/causal/case0/trace/workload.json"
            record = analysis.read(filename)
            record["budgets_gib"] = [38.5, 40.25]
            filename.write_text(json.dumps(record))
    elif damage == "startup_duplicate":
        filename = first / "startup-tests.xml"
        document = element_tree.parse(filename)
        suite = document.getroot().find("testsuite")
        suite[1].set("name", suite[0].get("name"))
        document.write(filename, encoding="unicode")
    else:
        filename = first / "out/causal-block/collect/out/budget.json"
        if damage == "budget_record_missing":
            filename.unlink()
        else:
            record = analysis.read(filename)
            record["memory"]["0"][0][0]["gpu_uuid"] = "foreign"
            filename.write_text(json.dumps(record))
    report = analysis.analyze_jobs([(0, first), (1, second)], tmp_path / "analysis", protocol)
    assert not report["complete_overhead_and_collection"]
    assert len(report["overhead"]["pairs"]) == 8


@pytest.mark.parametrize("measurement_revision", [2, 3])
def test_terminal_two_block_integration_writes_all_queries_without_prediction_claim(tmp_path, measurement_revision):
    first, protocol = terminal_job_fixture(tmp_path, 0)
    second, second_protocol = terminal_job_fixture(tmp_path, 1)
    assert protocol == second_protocol
    protocol["measurement_revision"] = measurement_revision
    output = tmp_path / "analysis"
    report = analysis.analyze_jobs([(0, first), (1, second)], output, protocol)
    assert report["complete_overhead_and_collection"], report
    assert not report["prediction_gate_passed"] and not report["independent_validation"]
    assert len(list(output.glob("*-queries.jsonl"))) == 8
    assert len(report["overhead"]["pairs"]) == 8
    assert analysis.read(output / "report.json") == report


@pytest.mark.parametrize("statuses", [["passed"], ["started"], ["started", "failed"],
                                    ["started", "passed", "passed"], ["passed", "started"],
                                    ["started", "started", "passed"]])
def test_startup_lifecycle_rejects_missing_reordered_or_duplicate_events(tmp_path, statuses):
    first, protocol = terminal_job_fixture(tmp_path, 0)
    filename = first / "out/causal-block/off/out/phases.json"
    phases = analysis.read(filename)
    completed = next(row for row in phases["phases"] if row["phase"] == "startup" and row["status"] == "passed")
    phases["phases"] = [dict(completed, status=status) for status in statuses] + [dict(phase="job", status="passed")]
    filename.write_text(json.dumps(phases))
    report = analysis.analyze_jobs([(0, first)], tmp_path / "analysis", protocol)
    assert not report["run_artifacts_passed"]
    assert set(report["blocks"][0]["audit"]["budgets"]) == {"off"}


@pytest.mark.parametrize("measurement_revision", [None, True, 2.0, 3.0, "3", 1, 4])
def test_unsupported_measurement_revision_rejects_before_output(tmp_path, measurement_revision):
    first, protocol = terminal_job_fixture(tmp_path, 0)
    protocol["measurement_revision"] = measurement_revision
    output = tmp_path / "analysis"
    with pytest.raises(ValueError, match="analysis_measurement_revision_differs"):
        analysis.analyze_jobs([(0, first)], output, protocol)
    assert not output.exists()


def test_one_block_cli_keeps_all_missing_pairs_and_reports_not_accepted(tmp_path):
    first, protocol = terminal_job_fixture(tmp_path, 0)
    protocol_file = tmp_path / "analysis-protocol.json"
    analysis.write(protocol_file, protocol)
    command = [sys.executable, str(HERE / "analyze_causal_calibration.py"), "--job", f"0={first}",
               "--analysis-protocol", str(protocol_file), "--output", str(tmp_path / "output")]
    result = subprocess.run(command, capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), timeout=20)
    assert result.returncode == 0, result.stderr
    report = analysis.read(tmp_path / "output/report.json")
    assert report["overhead"]["missing_blocks"] == [1] and len(report["overhead"]["pairs"]) == 8
    assert not report["complete_overhead_and_collection"] and not report["prediction_gate_passed"]


def test_all_missing_cases_are_retained_as_unavailable(tmp_path):
    first, protocol = terminal_job_fixture(tmp_path, 0)
    for mode in ("off", "collect"):
        for index in range(4):
            shutil.rmtree(first / f"out/causal-block/{mode}/out/calibration/causal/case{index}")
    report = analysis.analyze_jobs([(0, first)], tmp_path / "analysis", protocol)
    assert not report["complete_overhead_and_collection"]
    assert sum(case["status"] == "unavailable" for cases in report["blocks"][0]["cases"].values() for case in cases.values()) == 8


@pytest.mark.parametrize("damage", ["startup", "worker", "order", "clock", "cleanup", "workload", "cpu_missing", "case_clock", "case_cutoff", "arrivals"])
def test_terminal_artifact_or_collection_damage_never_accepted(tmp_path, damage):
    first, protocol = terminal_job_fixture(tmp_path, 0)
    second, _ = terminal_job_fixture(tmp_path, 1)
    output = first / "out/causal-block/collect/out"
    paths = {"worker": output / "worker-instance-1-dp0-tp0.json", "order": first / "out/causal-block/status.json",
             "clock": output / "causal-final-attestation.json", "cleanup": output.parent / "process.json",
             "workload": output / "calibration/causal/case0/trace/workload.json",
             "case_clock": output / "calibration/causal/case0/collection.json",
             "case_cutoff": output / "calibration/causal/case0/collection.json",
             "arrivals": output / "calibration/causal/case0/collection.json"}
    if damage == "startup":
        (first / "startup-tests.xml").write_text('<testsuites><testsuite tests="1"/></testsuites>')
    elif damage == "cpu_missing":
        (output.parent / "process-usage.jsonl").unlink()
    else:
        filename = paths[damage]
        value = analysis.read(filename)
        if damage == "worker":
            value["causal_deployment"]["configuration"] = {}
        elif damage == "order":
            value["started"].reverse()
        elif damage == "clock":
            value["clock_id"] = "foreign"
        elif damage == "cleanup":
            value["forced_cleanup"] = True
        elif damage == "workload":
            value["budgets_gib"][0] += 1
        elif damage == "case_clock":
            value["protocol"]["clock_id"] = "foreign"
        elif damage == "case_cutoff":
            value["trace_cutoff"]["censored_at"] -= 0.1
        else:
            arrival = next(event for event in value["events"] if event["kind"] == "arrival")
            arrival["prompt_tokens"] += 1
        filename.write_text(json.dumps(value))
    result = analysis.analyze_jobs([(0, first), (1, second)], tmp_path / "analysis", protocol)
    assert not result["complete_overhead_and_collection"]


def test_manifest_must_match_external_predeclared_digest(tmp_path):
    first, protocol = terminal_job_fixture(tmp_path, 0)
    protocol["deployment_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="predeclared_analysis"):
        analysis.analyze_jobs([(0, first)], tmp_path / "analysis", protocol)


def test_early_empty_cohort_and_finished_request_are_not_reinserted():
    report = complete_collection_report()
    sample = next(event for event in report["events"] if event["kind"] == "sample")
    sample.update(as_of=10.005, recorded_at=10.0051, cohort_at_completion=[])
    sample["transport"] = source_packet(sample["as_of"])
    sample["batch_links"] = linked_packet(sample["transport"], sample["as_of"])
    report["events"].sort(key=lambda event: event["recorded_at"])
    for sequence, event in enumerate(report["events"], 1):
        event["sequence"] = sequence
    recount(report)
    rows = list(analysis.label_queries(report))
    assert rows[0]["status"] == "empty_cohort"
    assert [row["request_id"] for row in rows[9]["features"]["requests"]] == ["fast"]
    assert [row["request_id"] for row in rows[9]["targets"]] == ["fast"]


@pytest.mark.parametrize("terminal", ["cancelled", "error"])
def test_competing_terminal_is_retained_not_imputed_as_k_completion(terminal):
    report = complete_collection_report()
    report["events"] = [event for event in report["events"] if not
                        (event["kind"] == "output_receipt" and event["request_id"] == "queued" and event["finished"])]
    report["events"].append(dict(kind="stream_terminal", request_id="queued", terminal=terminal, received_at=10.7,
                                 recorded_at=10.7001, case_id="fixture", clock_id=CLOCK, count=16,
                                 final_output_received=False, error_type=terminal))
    report["events"].sort(key=lambda event: event["recorded_at"])
    for event in report["events"]:
        if event["kind"] == "sample":
            event["cohort_at_start"] = naive_states_at(report, event["started_at"])
            event["cohort_at_completion"] = naive_states_at(report, event["as_of"])
    report["trace_cutoff"]["cohort"] = naive_states_at(report, 11.0)
    for sequence, event in enumerate(report["events"], 1):
        event["sequence"] = sequence
        if event["kind"] == "trace_cutoff":
            event["cohort"] = report["trace_cutoff"]["cohort"]
    recount(report)
    row = list(analysis.label_queries(report))[6]
    assert row["joint_target_s"] is None
    target = next(target for target in row["targets"] if target["request_id"] == "queued")
    assert target["status"] == "terminal_before_target" and target["terminal"]["kind"] == terminal
    assert target["lower_bound_s"] == pytest.approx(0.04)


@pytest.mark.parametrize("damage", ["flag", "protocol", "cutoff_cohort", "sample_time"])
def test_recorded_contract_mismatch_stays_unavailable(damage):
    report = complete_collection_report()
    if damage == "flag":
        report["retention_lossless"] = False
    elif damage == "protocol":
        report["events"][0]["protocol"] = dict(report["protocol"], deadline_s=0.07)
        recount(report)
    elif damage == "cutoff_cohort":
        report["trace_cutoff"]["cohort"][0]["count"] += 1
        recount(report)
    else:
        for event in report["events"]:
            if event["kind"] == "sample":
                event["scheduled_at"] -= 1
        recount(report)
    assert analysis.collection_coverage(report, coverage_rules())["status"] == "not_passed"


@pytest.mark.parametrize("damage", ["pre_rpc_features", "non_numeric_cadence"])
def test_unvalidated_nested_batch_features_are_not_exposed(damage):
    report = complete_collection_report()
    sample = next(event for event in report["events"] if event["kind"] == "sample" and event["tick"] == 2)
    latest = sample["batch_links"]["records"][0]["latest"]
    if damage == "pre_rpc_features":
        latest["rpc"] = None
        latest["features"] = {"future_output_length": 1000}
    else:
        latest["current_wave"] = {"future_output_length": 1000}
    recount(report)
    row = list(analysis.label_queries(report))[2]
    assert row["target_available"] and not row["feature_available"]
    assert "future_output_length" not in repr(row["source_features"])
