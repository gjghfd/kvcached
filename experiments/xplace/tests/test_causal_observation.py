import asyncio
import copy
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from causal_observation import causal_joint_state
from controller import PlacementController
import live_observation as live
from test_scoring_state import observed_state


CLOCK = "test-single-host-clock"


def transport_input():
    _, rows = observed_state()
    scheduler, waves, workers = [], [], []
    for model in (0, 1):
        for rank in (0, 1):
            pid = 100 + model * 2 + rank
            producer = {"producer_pid": pid, "producer_id": f"{pid:032x}"}
            rows[model][rank].update(clock_id=CLOCK, **producer)
            scheduler.append({"status": "observed", "model": model, "dp_rank": rank,
                              "available_at": 10.2, "snapshot": rows[model][rank]})
            waves.append({"model": model, "dp_rank": rank, "role": "engine", "schema": 1,
                          "status": "observed", "clock_id": CLOCK, "observed_at": 10.05,
                          "available_at": 10.15, "wave_active": False, "device_activity": "unknown",
                          "wave_source": "engine.engines_running", **producer})
            for lane in (0, 1):
                pid = 200 + model * 4 + rank * 2 + lane
                workers.append({"model": model, "dp_rank": rank, "tp_rank": lane, "role": "worker",
                                "schema": 1, "status": "observed", "clock_id": CLOCK,
                                "producer_pid": pid, "producer_id": f"{pid:032x}",
                                "observed_at": 10.1, "available_at": 10.2, "host_call": "sample",
                                "host_call_started_at": 8.0, "host_depth": 1, "failed_calls": 0,
                                "reason": None, "device_activity": "unknown"})
    packet = {"collected_at": 10.4,
              "scheduler": {"kind": "live_scheduler_transport_v1", "status": "observed",
                            "clock_id": CLOCK, "ranks": scheduler},
              "execution": {"kind": "live_execution_transport_v1", "status": "observed",
                            "clock_id": CLOCK, "collected_at": 10.3, "rank_waves": waves, "workers": workers}}
    query = {"as_of": 10.5, "max_age_s": 1.0, "max_skew_s": 0.5, "clock_id": CLOCK,
             "num_ranks": 2, "tensor_parallel_size": 2}
    return packet, query


def error_reasons(result):
    assert result["status"] == "unavailable"
    assert result["device_activity"] == "unknown" and result["forecast_status"] == "unavailable_not_estimated"
    return {error["reason"] for error in result["errors"]}


def test_complete_join_retains_three_distinct_times_not_atomic_or_gpu_idle():
    packet, query = transport_input()
    result = causal_joint_state(packet, **query)
    assert result["status"] == "available" and not result["errors"]
    assert result["scheduler_receipts"][0]["observed_at"] == 10.0
    assert result["rank_waves"][0]["observed_at"] == 10.05
    assert result["workers"][0]["observed_at"] == 10.1
    assert result["workers"][0]["host_call_started_at"] == 8.0
    assert "wave_active" not in result["workers"][0]
    assert result["logical_observed"][0]["phase"] == "idle"
    assert not result["rank_waves"][0]["wave_active"]
    assert result["workers"][0]["host_call"] == "sample"
    assert result["device_activity"] == "unknown" and result["forecast_status"] == "unavailable_not_estimated"
    assert not {"cost", "other_active", "horizon", "horizon_steps"}.intersection(result)


def test_candidate_changes_logical_layout_only_and_never_reads_future_fields():
    packet, query = transport_input()
    request = packet["scheduler"]["ranks"][2]["snapshot"]["requests"][0]
    request.update(max_tokens=99999, remaining=88888, predicted_future_phase="decode", completed_at=20.0)
    original = copy.deepcopy(packet)
    baseline = causal_joint_state(packet, **query)
    result = causal_joint_state(packet, **query, moves=[{"model": 1, "id": "conv-0-0", "source": 0, "destination": 1}],
                                arrival={"model": 0, "prompt_tokens": 101167, "max_tokens": 77777}, target=1)
    assert result["logical_candidate"][1]["counts"]["decode"] == [1, 3]
    assert result["logical_candidate"][0]["counts"]["prefill"] == [0, 1]
    for key in ("workers", "rank_waves", "scheduler_receipts", "logical_observed", "scoring_observations"):
        assert result[key] == baseline[key]
    assert packet == original
    assert "remaining" not in repr(result) and "max_tokens" not in repr(result)
    assert "completed_at" not in repr(result) and "predicted_future_phase" not in repr(result)
    result["scoring_observations"][1][0]["requests"][0]["computed_tokens"] = -1
    result["workers"][0]["host_call"] = "none"
    assert packet == original


def test_own_peer_indices_are_tenant_symmetric():
    packet, query = transport_input()
    baseline = causal_joint_state(packet, **query)
    mirrored = copy.deepcopy(packet)
    for records in (mirrored["scheduler"]["ranks"], mirrored["execution"]["rank_waves"], mirrored["execution"]["workers"]):
        for record in records:
            record["model"] = 1 - record["model"]
    result = causal_joint_state(mirrored, **query)
    assert result["status"] == "available"
    for model in (0, 1):
        for key in ("own_counts", "peer_counts", "own_accounted_contexts", "peer_accounted_contexts"):
            assert result["directional_indices"][model][key] == baseline["directional_indices"][1 - model][key]


@pytest.mark.parametrize("component,field", [("scheduler", "ranks"), ("execution", "rank_waves"), ("execution", "workers")])
@pytest.mark.parametrize("damage", ["missing", "extra", "duplicate", "bad_identity", "malformed"])
def test_incomplete_duplicate_or_malformed_identity_sets_rejected(component, field, damage):
    packet, query = transport_input()
    records = packet[component][field]
    if damage == "missing":
        records.pop()
    elif damage == "extra":
        records.append(copy.deepcopy(records[0]))
    elif damage == "duplicate":
        records[-1] = copy.deepcopy(records[0])
    elif damage == "bad_identity":
        records[0]["dp_rank"] = True
    else:
        records[0] = None
    original = copy.deepcopy(packet)
    error_reasons(causal_joint_state(packet, **query))
    assert packet == original


@pytest.mark.parametrize("component", ["scheduler", "execution"])
@pytest.mark.parametrize("field,value,reason", [
    ("kind", "post_stop_export", "missing_or_wrong_transport_kind"),
    ("clock_id", "other-host", "clock_domain_mismatch"),
    ("status", "incomplete", "incomplete_transport"),
])
def test_envelope_cannot_relabel_archive_or_missing_collection(component, field, value, reason):
    packet, query = transport_input()
    packet[component][field] = value
    assert reason in error_reasons(causal_joint_state(packet, **query))


@pytest.mark.parametrize("kind", ["scheduler", "rank", "worker"])
@pytest.mark.parametrize("observed,available,reason", [
    (8.0, 10.2, "stale_observation"),
    (10.25, 10.2, "availability_before_observation"),
    (10.0, 10.6, "not_available_at_decision"),
    (None, 10.2, "missing_or_invalid_timestamp"),
    (10.0, True, "missing_or_invalid_timestamp"),
])
def test_each_source_uses_own_observation_and_receipt_time(kind, observed, available, reason):
    packet, query = transport_input()
    if kind == "scheduler":
        packet["scheduler"]["ranks"][0]["available_at"] = available
        record = packet["scheduler"]["ranks"][0]["snapshot"]
    else:
        record = packet["execution"]["rank_waves" if kind == "rank" else "workers"][0]
        record["available_at"] = available
    record["observed_at"] = observed
    assert reason in error_reasons(causal_joint_state(packet, **query))


@pytest.mark.parametrize("damage,reason", [
    ("future_query", "collection_not_available_at_decision"),
    ("execution_after_parent", "invalid_collection_time"),
    ("worker_after_collection", "receipt_after_collection"),
    ("scheduler_after_collection", "receipt_after_collection"),
])
def test_internal_receipts_do_not_make_an_unreturned_collection_available(damage, reason):
    packet, query = transport_input()
    if damage == "future_query":
        query["as_of"] = 10.35
    elif damage == "execution_after_parent":
        packet["execution"]["collected_at"] = 10.45
    elif damage == "worker_after_collection":
        packet["execution"]["workers"][0]["available_at"] = 10.35
    else:
        packet["scheduler"]["ranks"][0]["available_at"] = 10.45
    assert reason in error_reasons(causal_joint_state(packet, **query))


@pytest.mark.parametrize("kind", ["scheduler", "rank", "worker"])
def test_individual_clock_domains_are_not_hidden_by_envelope_clock(kind):
    packet, query = transport_input()
    record = (packet["scheduler"]["ranks"][0]["snapshot"] if kind == "scheduler" else
              packet["execution"]["rank_waves" if kind == "rank" else "workers"][0])
    record["clock_id"] = "another-namespace"
    assert "clock_domain_mismatch" in error_reasons(causal_joint_state(packet, **query))


@pytest.mark.parametrize("key,value,reason", [
    ("producer_pid", None, "missing_or_invalid_producer"),
    ("producer_id", "not-a-token", "missing_or_invalid_producer"),
    ("producer_id", "f" * 32, "rank_producer_mismatch"),
    ("producer_pid", 999, "rank_producer_mismatch"),
])
def test_scheduler_must_be_bound_to_same_rank_sampler_incarnation(key, value, reason):
    packet, query = transport_input()
    packet["scheduler"]["ranks"][0]["snapshot"][key] = value
    assert reason in error_reasons(causal_joint_state(packet, **query))


@pytest.mark.parametrize("key", ["producer_pid", "producer_id"])
def test_reused_producer_across_two_workers_is_rejected(key):
    packet, query = transport_input()
    records = packet["execution"]["workers"]
    records[1][key] = records[0][key]
    assert "reused_producer_identity" in error_reasons(causal_joint_state(packet, **query))


@pytest.mark.parametrize("update,reason", [
    ({"wave_active": 0}, "invalid_rank_wave"),
    ({"wave_source": "logical_idle"}, "invalid_rank_wave"),
    ({"schema": True}, "invalid_sample_status_or_schema"),
    ({"device_activity": "idle"}, "invalid_sample_status_or_schema"),
])
def test_rank_wave_is_actual_boolean_host_state_not_logical_gpu_inference(update, reason):
    packet, query = transport_input()
    packet["execution"]["rank_waves"][0].update(update)
    assert reason in error_reasons(causal_joint_state(packet, **query))


@pytest.mark.parametrize("update,reason", [
    ({"host_call": "idle"}, "invalid_host_state"),
    ({"host_depth": 5}, "invalid_host_state"),
    ({"host_depth": True}, "invalid_host_state"),
    ({"failed_calls": -1}, "invalid_host_state"),
    ({"failed_calls": 1}, "host_failures_observed"),
    ({"reason": "host_depth_exceeded"}, "invalid_host_state"),
    ({"host_call": "none"}, "invalid_host_call_start_or_depth"),
    ({"host_depth": 0}, "invalid_host_call_start_or_depth"),
    ({"host_call_started_at": 10.2}, "invalid_host_call_start_or_depth"),
    ({"host_call_started_at": None}, "invalid_host_call_start_or_depth"),
])
def test_worker_bad_or_failed_state_cannot_become_idle_or_available(update, reason):
    packet, query = transport_input()
    packet["execution"]["workers"][0].update(update)
    assert reason in error_reasons(causal_joint_state(packet, **query))


def test_skew_bound_includes_all_three_sources_and_accepts_exact_boundary():
    packet, query = transport_input()
    query["max_skew_s"] = 10.1 - 10.0
    assert causal_joint_state(packet, **query)["status"] == "available"
    packet["execution"]["rank_waves"][0]["observed_at"] = 9.99
    assert "snapshot_skew_exceeded" in error_reasons(causal_joint_state(packet, **query))


def test_freshness_and_decision_availability_exact_boundary():
    packet, query = transport_input()
    query.update(as_of=10.4, max_age_s=0.4)
    query["max_age_s"] = query["as_of"] - 10.0
    assert causal_joint_state(packet, **query)["status"] == "available"


@pytest.mark.parametrize("field,value", [("computed_tokens", -1), ("phase", "unknown"), ("in_flight_tokens", None)])
def test_logical_descriptor_is_derived_only_from_received_valid_snapshot(field, value):
    packet, query = transport_input()
    packet["scheduler"]["ranks"][2]["snapshot"]["requests"][0][field] = value
    error_reasons(causal_joint_state(packet, **query))


def test_source_order_does_not_change_identity_sorted_join():
    packet, query = transport_input()
    expected = causal_joint_state(packet, **query)
    for records in (packet["scheduler"]["ranks"], packet["execution"]["workers"], packet["execution"]["rank_waves"]):
        records.reverse()
    assert causal_joint_state(packet, **query) == expected


@pytest.mark.parametrize("field,value", [("as_of", True), ("max_age_s", -1), ("max_skew_s", float("nan")),
                                        ("clock_id", ""), ("num_ranks", False), ("tensor_parallel_size", 0)])
def test_invalid_query_rejected(field, value):
    packet, query = transport_input()
    query[field] = value
    with pytest.raises(ValueError):
        causal_joint_state(packet, **query)


def test_boolean_candidate_source_does_not_alias_rank_zero():
    packet, query = transport_input()
    with pytest.raises(ValueError, match="candidate move identity"):
        causal_joint_state(packet, **query, moves=[{"model": 1, "id": "conv-0-0", "source": False, "destination": 1}])


def producer_process(root, model, rank, lane, snapshot, ready, stop, connection):
    time.monotonic = lambda: time.clock_gettime(time.CLOCK_MONOTONIC)
    os.environ.update(XPLACE_LIVE_OBSERVATION="1", XPLACE_LIVE_ROOT=str(root), DP_OPT_SERVICE=f"instance-{model + 1}")
    os.environ.pop("XPLACE_TIMING_ALLOWED", None)
    live.host_clock_id = lambda: CLOCK
    if lane is None:
        def sample_scheduler():
            return dict(snapshot, observed_at=time.monotonic())

        engine = SimpleNamespace(dp_rank=rank, engines_running=True, scheduler=SimpleNamespace(
            placement_snapshot=sample_scheduler, placement_summary=sample_scheduler))
        server = live.bind_engine_observer(engine)
        ready.set()
        try:
            while not stop.is_set():
                if connection.poll(0.05):
                    assert connection.recv() == "snapshot"
                    connection.send(engine.scheduler.placement_snapshot())
        finally:
            server.close()
            connection.close()
    else:
        def blocked_sample():
            ready.set()
            if not stop.wait(90):
                raise TimeoutError("Coordinator did not release the sample fixture")

        worker = SimpleNamespace(_placement_timing_identity={"dp_rank": rank, "tp_rank": lane},
                                 execute_model=lambda: None, sample_tokens=blocked_sample,
                                 execute_dummy_batch=lambda: None)
        server = live.start_worker_observer(worker)
        try:
            worker.sample_tokens()
        finally:
            server.close()


def wait_for_producers(processes, ready_events, timeout_s=60):
    if (len(processes) != len(ready_events) or not processes or type(timeout_s) not in (int, float)
            or not 0 < timeout_s <= 60):
        raise ValueError("Invalid bounded producer startup barrier")
    deadline = time.monotonic() + timeout_s
    while True:
        states = [dict(name=process.name, pid=process.pid, exitcode=process.exitcode, ready=ready.is_set())
                  for process, ready in zip(processes, ready_events)]
        if any(row["exitcode"] is not None for row in states):
            raise AssertionError(f"Producer exited before collection: {states}")
        if all(row["ready"] for row in states):
            return states
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(f"Producer startup deadline exceeded: {states}")
        time.sleep(min(0.05, remaining))


def test_real_twelve_process_transport_binds_scheduler_rank_and_all_lanes(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: time.clock_gettime(time.CLOCK_MONOTONIC))
    context = multiprocessing.get_context("spawn")
    stop = context.Event()
    packet, _ = transport_input()
    processes, ready_events, pipes = [], [], {}
    monkeypatch.setattr(live, "host_clock_id", lambda: CLOCK)
    with tempfile.TemporaryDirectory(prefix="xcj-", dir="/tmp") as directory:
        try:
            for model in (0, 1):
                for rank in (0, 1):
                    snapshot = packet["scheduler"]["ranks"][model * 2 + rank]["snapshot"]
                    for lane in (None, 0, 1):
                        parent, child = context.Pipe() if lane is None else (None, None)
                        ready = context.Event()
                        process = context.Process(target=producer_process, name=f"causal-m{model}-dp{rank}-lane{lane}",
                            args=(directory, model, rank, lane, snapshot, ready, stop, child))
                        process.start()
                        if child is not None:
                            child.close()
                            pipes[(model, rank)] = parent
                        processes.append(process)
                        ready_events.append(ready)
            wait_for_producers(processes, ready_events)

            async def scenario():
                controller = PlacementController([None, None], 2, [40, 40])
                controller.tensor_parallel_size = 2

                async def call(model, rank, method):
                    assert method == "placement_snapshot"
                    connection = pipes[(model, rank)]

                    def rpc():
                        connection.send("snapshot")
                        assert connection.poll(3)
                        return connection.recv()

                    return await asyncio.to_thread(rpc)

                controller.call = call
                raw = await controller.prediction_observations(directory, 5)
                joined = causal_joint_state(raw, as_of=time.monotonic(), max_age_s=10, max_skew_s=10,
                                             clock_id=CLOCK, num_ranks=2, tensor_parallel_size=2)
                return raw, joined

            raw, result = asyncio.run(scenario())
            assert result["status"] == "available", result["errors"]
            assert len(result["workers"]) == 8 and len(result["rank_waves"]) == 4
            pids = {record["producer_pid"] for record in result["workers"] + result["rank_waves"]}
            assert pids == {process.pid for process in processes}
            for receipt, wave in zip(result["scheduler_receipts"], result["rank_waves"]):
                assert (receipt["producer_pid"], receipt["producer_id"]) == (wave["producer_pid"], wave["producer_id"])
            assert all(record["host_call"] == "sample" for record in result["workers"])
            assert all(process.is_alive() for process in processes)
            assert result["availability_window"]["collected_at"] == raw["collected_at"]
        finally:
            stop.set()
            for process in processes:
                process.join(4)
                if process.is_alive():
                    process.terminate()
                    process.join(1)
            for connection in pipes.values():
                connection.close()
        assert all(process.exitcode == 0 for process in processes)
        assert not list(Path(directory).iterdir())
