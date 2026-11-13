import asyncio
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

import timing_probe
from timing_probe import (EVENT_LIMIT, FrontendTiming, TimingBuffer, TimingHooks, bind_engine_timing,
                          scheduler_state, start_engine_timing, start_worker_timing, validate_timing,
                          worker_timing_control)


def scheduled():
    return SimpleNamespace(num_scheduled_tokens={"request": 1}, total_num_scheduled_tokens=1)


class Worker:
    def execute_model(self, scheduler_output):
        return scheduler_output

    def sample_tokens(self, grammar_output):
        return grammar_output

    def execute_dummy_batch(self):
        return "dummy"


def engine_fixture():
    future = Future()
    scheduler = SimpleNamespace(running=[SimpleNamespace(num_computed_tokens=4096, num_prompt_tokens=4096)],
                                waiting=[], schedule=lambda *arguments: scheduled(),
                                update_from_output=lambda *arguments: {"finished": True}, has_requests=lambda: False)
    executor = SimpleNamespace(execute_model=lambda *arguments, **keywords: future,
                               sample_tokens=lambda *arguments: "sample", execute_dummy_batch=lambda: "dummy")
    return SimpleNamespace(scheduler=scheduler, model_executor=executor, step_fn=lambda: "step", batch_queue=None,
                           dp_rank=0, collective_rpc=lambda *arguments, **keywords: []), future


def test_default_off_does_not_wrap_or_install_engine_control(monkeypatch):
    monkeypatch.delenv("XPLACE_TIMING_ALLOWED", raising=False)
    engine, _ = engine_fixture()
    original = engine.step_fn
    bind_engine_timing(engine)
    assert engine.step_fn is original and not hasattr(engine, "placement_timing")
    with pytest.raises(RuntimeError, match="disabled"):
        worker_timing_control(Worker(), "start", "case")


@pytest.mark.parametrize("limit", [True, 1, 0, -1, EVENT_LIMIT + 1, 4.5])
def test_invalid_buffer_limits(limit):
    with pytest.raises(ValueError):
        TimingBuffer("case", {}, limit)


def test_buffer_is_bounded_without_replacing_old_events():
    buffer = TimingBuffer("case", {}, limit=4)
    for position in range(10):
        buffer.emit("tick", position=position)
    report = buffer.stop()
    assert report["attempted"] == 12 and report["exported"] == 4 and report["dropped"] == 8
    assert [event.get("position") for event in report["events"]] == [None, 0, 1, 2]
    with pytest.raises(RuntimeError, match="already stopped"):
        buffer.stop()


def test_threaded_writes_have_a_complete_sequence():
    buffer = TimingBuffer("case", {})
    def emit():
        for _ in range(50):
            buffer.emit("tick")
    threads = [threading.Thread(target=emit) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    report = buffer.stop()
    assert report["attempted"] == 202 and not report["dropped"]
    assert [event["sequence"] for event in report["events"]] == list(range(1, 203))


def test_worker_hooks_preserve_returns_and_restore_class_methods():
    worker = Worker()
    hooks = start_worker_timing(worker, "case", {"tp_rank": 1})
    output = scheduled()
    assert worker.execute_model(output) is output
    marker = object()
    assert worker.sample_tokens(marker) is marker
    assert worker.execute_dummy_batch() == "dummy"
    report = hooks.stop()
    assert "execute_model" not in vars(worker) and "sample_tokens" not in vars(worker)
    assert [event["kind"] for event in report["events"]] == ["start", "worker_execute:enter", "worker_execute:return",
                                                             "worker_sample:enter", "worker_sample:return",
                                                             "worker_dummy:enter", "worker_dummy:return", "stop"]


def test_hook_preserves_original_exception():
    failure = RuntimeError("original")
    def fail():
        raise failure
    target = SimpleNamespace(run=fail)
    hooks = TimingHooks(TimingBuffer("case", {}))
    hooks.wrap(target, "run", "run")
    with pytest.raises(RuntimeError) as observed:
        target.run()
    assert observed.value is failure
    assert hooks.stop()["events"][-2]["kind"] == "run:error"
    assert target.run is fail


def test_pending_future_is_not_consumed_or_replaced():
    engine, future = engine_fixture()
    original = engine.model_executor.execute_model
    hooks = start_engine_timing(engine, "case", {})
    returned = engine.model_executor.execute_model(scheduled(), non_block=True)
    assert returned is future and not future.done()
    with pytest.raises(RuntimeError, match="pending"):
        hooks.stop()
    marker = object()
    future.set_result(marker)
    assert returned.result() is marker
    report = hooks.stop()
    assert engine.model_executor.execute_model is original
    assert any(event["kind"] == "executor_execute:ready" for event in report["events"])


def test_schema_errors_do_not_change_method_behavior():
    worker = Worker()
    hooks = start_worker_timing(worker, "case", {})
    marker = object()
    assert worker.execute_model(marker) is marker
    assert hooks.stop()["schema_errors"] == 1


def test_partial_hook_installation_rolls_back():
    worker = SimpleNamespace(execute_model=lambda output: output)
    original = worker.execute_model
    with pytest.raises(AttributeError):
        start_worker_timing(worker, "case", {})
    assert worker.execute_model is original


def test_failed_worker_start_restores_engine_hooks(monkeypatch):
    monkeypatch.setenv("XPLACE_TIMING_ALLOWED", "1")
    monkeypatch.setenv("DP_OPT_SERVICE", "instance-1")
    engine, _ = engine_fixture()
    original = engine.step_fn
    calls = []
    def rpc(name, args):
        calls.append(args[0])
        if args[0] == "start":
            raise RuntimeError("worker failed")
        return []
    engine.collective_rpc = rpc
    bind_engine_timing(engine)
    with pytest.raises(RuntimeError, match="worker failed"):
        engine.placement_timing("start", "case")
    assert engine.step_fn is original and engine._placement_timing_hooks is None
    assert calls == ["start", "abort"]


def test_control_requires_idle_and_disallows_duplicate_start(monkeypatch):
    monkeypatch.setenv("XPLACE_TIMING_ALLOWED", "1")
    worker = Worker()
    worker._placement_timing_identity = {"service": "instance-1", "dp_rank": 0, "tp_rank": 0, "gpu_uuid": "gpu"}
    worker_timing_control(worker, "start", "case")
    with pytest.raises(RuntimeError, match="already active"):
        worker_timing_control(worker, "start", "case")
    worker_timing_control(worker, "stop")
    with pytest.raises(ValueError):
        worker_timing_control(worker, "stop")
    assert worker_timing_control(worker, "abort") == {"inactive": True}
    engine, _ = engine_fixture()
    engine.scheduler.has_requests = lambda: True
    bind_engine_timing(engine)
    with pytest.raises(RuntimeError, match="idle"):
        engine.placement_timing("start", "case")


def test_scheduler_phase_and_accounted_context():
    requests = [SimpleNamespace(num_computed_tokens=64, num_prompt_tokens=100),
                SimpleNamespace(num_computed_tokens=300, num_prompt_tokens=100)]
    state = scheduler_state(SimpleNamespace(running=requests, waiting=["waiting"]))
    assert state == {"running": 2, "waiting": 1, "prefill": 1, "decode": 1, "phase": "mixed",
                     "accounted_context_sum": 364, "prompt_sum": 200}


def test_frontend_heartbeats_stop_without_leaking_threads():
    async def scenario():
        frontend = FrontendTiming("case")
        await frontend.start()
        await asyncio.sleep(0.085)
        report = await frontend.stop()
        assert not frontend.thread.is_alive() and frontend.task.done()
        kinds = {event["kind"] for event in report["events"]}
        assert {"async_heartbeat", "thread_heartbeat"} <= kinds
        assert not report["dropped"] and not report["schema_errors"]
    asyncio.run(scenario())


def valid_reports():
    frontend = TimingBuffer("case", {"role": "frontend"})
    records = []
    for model in range(2):
        for rank in range(2):
            engine = TimingBuffer("case", {"role": "engine", "service": f"instance-{model + 1}", "dp_rank": rank})
            workers = [TimingBuffer("case", {"role": "worker", "service": f"instance-{model + 1}", "dp_rank": rank,
                                             "tp_rank": lane, "gpu_uuid": f"gpu-{rank}-{lane}"}) for lane in range(2)]
            records.append((engine, workers))
    start = time.monotonic_ns()
    frontend.emit("async_heartbeat")
    frontend.emit("thread_heartbeat")
    for engine, workers in records:
        engine.emit("schedule:return", scheduled_tokens=1)
        for worker in workers:
            worker.emit("worker_execute:enter", scheduled_tokens=1)
    end = time.monotonic_ns()
    reports = [{"engine": engine.stop(), "workers": [worker.stop() for worker in workers]} for engine, workers in records]
    front_report = frontend.stop()
    all_records = [front_report] + [record for group in reports for record in [group["engine"]] + group["workers"]]
    for position, record in enumerate(all_records):
        record["identity"]["pid"] = position + 100
    return reports, front_report, start, end


def test_validation_checks_every_engine_worker_and_shared_gpu():
    reports, frontend, start, end = valid_reports()
    assert validate_timing(reports, frontend, "case", start, end)["process_records"] == 13


@pytest.mark.parametrize("problem", ["missing_engine", "duplicate_lane", "wrong_gpu", "overflow", "missing_event",
                                    "wrong_case", "late_start", "missing_heartbeat", "missing_work", "backwards_clock", "duplicate_pid"])
def test_invalid_telemetry_is_not_accepted(problem):
    reports, frontend, start, end = valid_reports()
    if problem == "missing_engine":
        reports.pop()
    elif problem == "duplicate_lane":
        reports[0]["workers"][1]["identity"]["tp_rank"] = 0
    elif problem == "wrong_gpu":
        reports[0]["workers"][0]["identity"]["gpu_uuid"] = "wrong"
    elif problem == "overflow":
        reports[0]["engine"]["dropped"] = 1
    elif problem == "missing_event":
        reports[0]["engine"]["events"].pop()
    elif problem == "wrong_case":
        reports[0]["workers"][0]["case"] = "other"
    elif problem == "late_start":
        reports[0]["workers"][0]["started_ns"] = start + 1
    elif problem == "missing_heartbeat":
        frontend["events"][1]["kind"] = "other"
    elif problem == "missing_work":
        reports[0]["workers"][0]["events"][1]["scheduled_tokens"] = 0
    elif problem == "backwards_clock":
        reports[0]["engine"]["events"][1]["at_ns"] = 0
    elif problem == "duplicate_pid":
        reports[0]["engine"]["identity"]["pid"] = frontend["identity"]["pid"]
    with pytest.raises(ValueError):
        validate_timing(reports, frontend, "case", start, end)


def test_interface_source_guard_rejects_changed_deployment(tmp_path, monkeypatch):
    import hashlib

    source = tmp_path / "interface.py"
    source.write_bytes(b"known interface")
    expected = {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}
    monkeypatch.setattr(timing_probe, "INTERFACE_SHA256", expected)
    assert timing_probe.verify_interface_sources(tmp_path) == expected
    source.write_bytes(b"changed interface")
    with pytest.raises(ValueError, match="Unverified"):
        timing_probe.verify_interface_sources(tmp_path)
