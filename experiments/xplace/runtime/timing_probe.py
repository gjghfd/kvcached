import asyncio
import functools
import hashlib
import os
import threading
import time
from pathlib import Path


EVENT_LIMIT = 100000
HEARTBEAT_S = 0.02
INTERFACE_SHA256 = {
    "v1/engine/core.py": "3ae1381a6af841e21058c825702382dc66faae45c950ac5acb8495d2d3d05aad",
    "v1/worker/gpu_worker.py": "28ac3f85aca3f68f18ac51cb5915e4d7ebe316c4ff3c021f1c2c5d3b90daca56",
    "v1/core/sched/scheduler.py": "2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941",
    "v1/executor/multiproc_executor.py": "fb279ff0e0a08681aa0078037d559351006b5453ecd1b1ea12277995925f3312",
    "v1/executor/abstract.py": "419002195d13cf050c3ea9705d18a12456c890c27c470a1efadce7de27ded0b1",
}


def nested_timing_enabled():
    value = os.environ.get("XPLACE_NESTED_TIMING", "0")
    if value not in ("0", "1"):
        raise ValueError("XPLACE_NESTED_TIMING must be 0 or 1")
    return value == "1"


def interface_sources(nested=False):
    if nested:
        from nested_timing import INTERFACE_SHA256 as nested_sources
        return dict(nested_sources)
    return dict(INTERFACE_SHA256)


def verify_interface_sources(root, nested=False):
    expected = interface_sources(nested)
    actual = {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest() for name in expected}
    if actual != expected:
        raise ValueError(f"Unverified vLLM timing interfaces: {actual}")
    return actual


class TimingBuffer:
    def __init__(self, case, identity, limit=EVENT_LIMIT):
        if not isinstance(case, str) or not case or len(case) > 128:
            raise ValueError("Timing case needs a short nonempty name")
        if type(limit) is not int or not 2 <= limit <= EVENT_LIMIT:
            raise ValueError("Invalid timing event limit")
        self.case = case
        self.identity = dict(identity, pid=os.getpid())
        self.limit = limit
        self.events = []
        self.attempted = 0
        self.dropped = 0
        self.late_events = 0
        self.schema_errors = 0
        self.active = True
        self.lock = threading.Lock()
        self.started_ns = time.monotonic_ns()
        self.wall_start_ns = time.time_ns()
        self.emit("start")

    def emit(self, kind, scope_start=False, **fields):
        with self.lock:
            if not self.active:
                self.late_events += 1
                return
            self.attempted += 1
            if scope_start:
                fields["scope_id"] = self.attempted
            self.schema_errors += int(kind == "schema_error")
            if len(self.events) >= self.limit:
                self.dropped += 1
                return self.attempted
            self.events.append({"sequence": self.attempted, "at_ns": time.monotonic_ns(),
                                "thread": threading.get_ident(), "kind": kind, **fields})
            return self.attempted

    def stop(self):
        if not self.active:
            raise RuntimeError("Timing buffer already stopped")
        self.emit("stop")
        with self.lock:
            self.active = False
            result = {"schema": 1, "case": self.case, "identity": self.identity, "limit": self.limit,
                    "started_ns": self.started_ns, "ended_ns": time.monotonic_ns(),
                    "wall_start_ns": self.wall_start_ns, "wall_end_ns": time.time_ns(),
                    "attempted": self.attempted, "exported": len(self.events), "dropped": self.dropped,
                    "late_events": self.late_events, "schema_errors": self.schema_errors,
                    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "events": list(self.events), "timing_semantics": "host boundaries; Future readiness is observed when RPC responses are drained, not a CUDA completion timestamp"}
            if hasattr(self, "nested_timing"):
                result["nested_timing"] = dict(self.nested_timing)
            return result


def scheduler_state(scheduler):
    running = list(scheduler.running)
    prefill = sum(request.num_computed_tokens < request.num_prompt_tokens for request in running)
    return {"running": len(running), "waiting": len(scheduler.waiting), "prefill": prefill,
            "decode": len(running) - prefill,
            "phase": "mixed" if 0 < prefill < len(running) else "prefill" if prefill else "decode" if running else "idle",
            "accounted_context_sum": sum(request.num_computed_tokens for request in running),
            "prompt_sum": sum(request.num_prompt_tokens for request in running)}


def scheduled_state(output):
    scheduled = output.num_scheduled_tokens
    return {"scheduled_requests": len(scheduled), "scheduled_tokens": output.total_num_scheduled_tokens,
            "scheduled_prefill_sized_requests": sum(count > 1 for count in scheduled.values())}


class TimingHooks:
    def __init__(self, buffer, nested=False):
        self.buffer = buffer
        self.restorations = []
        self.pending = 0
        self.lock = threading.Lock()
        self.nested = nested
        self.scope_threads = threading.local()
        self.active_scopes = 0

    def _fields(self, builder, *arguments):
        if builder is None:
            return {}
        try:
            return builder(*arguments)
        except (AttributeError, TypeError, ValueError) as error:
            self.buffer.emit("schema_error", error_type=type(error).__name__, message=str(error)[:256])
            return {}

    def _watch_future(self, future, label):
        if not callable(getattr(future, "add_done_callback", None)):
            self.buffer.emit("schema_error", message="Executor did not return a callback-capable Future")
            return
        with self.lock:
            self.pending += 1

        def ready(completed):
            try:
                cancelled = completed.cancelled()
                error = None if cancelled else completed.exception()
                self.buffer.emit(label + ":ready", cancelled=cancelled,
                                 error_type=type(error).__name__ if error is not None else None)
            finally:
                with self.lock:
                    self.pending -= 1

        try:
            future.add_done_callback(ready)
        except BaseException:
            with self.lock:
                self.pending -= 1
            raise

    def wrap(self, target, name, label, before=None, after=None, future=False):
        original = getattr(target, name)
        if not callable(original):
            raise ValueError(f"Timing interface {name} is not callable")
        if getattr(original, "_placement_timing_owner", None) is not None:
            raise ValueError(f"Timing interface {name} is already wrapped")
        owned = name in vars(target)

        @functools.wraps(original)
        def wrapped(*arguments, **keywords):
            fields = self._fields(before, arguments, keywords)
            scope_fields = {}
            if self.nested:
                if not hasattr(self.scope_threads, "stack"):
                    self.scope_threads.stack = []
                stack = self.scope_threads.stack
                parent = stack[-1] if stack else None
                with self.lock:
                    self.active_scopes += 1
                scope_id = self.buffer.emit(label + ":enter", scope_start=True, parent_scope_id=parent, **fields)
                stack.append(scope_id)
                scope_fields = {"scope_id": scope_id, "parent_scope_id": parent}
            else:
                self.buffer.emit(label + ":enter", **fields)
            try:
                result = original(*arguments, **keywords)
            except BaseException as error:
                self.buffer.emit(label + ":error", error_type=type(error).__name__, **scope_fields)
                raise
            else:
                self.buffer.emit(label + ":return", **self._fields(after, result), **scope_fields)
            finally:
                if self.nested:
                    stack.pop()
                    with self.lock:
                        self.active_scopes -= 1
            if future:
                self._watch_future(result, label)
            return result

        wrapped._placement_timing_owner = self
        setattr(target, name, wrapped)
        self.restorations.append((target, name, original, owned, wrapped))

    def restore(self):
        conflicts = []
        for target, name, original, owned, wrapped in reversed(self.restorations):
            if getattr(target, name, None) is not wrapped:
                conflicts.append(name)
                continue
            if owned:
                setattr(target, name, original)
            else:
                delattr(target, name)
        self.restorations.clear()
        if conflicts:
            self.buffer.emit("schema_error", message="Timing restoration conflict", names=conflicts)
            raise RuntimeError(f"Timing restoration conflict: {conflicts}")
        if hasattr(self.buffer, "nested_timing"):
            self.buffer.nested_timing["restored"] = True

    def stop(self):
        with self.lock:
            if self.pending:
                raise RuntimeError("Cannot stop timing with pending executor futures")
            if self.active_scopes:
                raise RuntimeError("Cannot stop timing with active scopes")
        self.restore()
        return self.buffer.stop()


def start_worker_timing(worker, case, identity):
    nested = nested_timing_enabled()
    hooks = TimingHooks(TimingBuffer(case, dict(identity, role="worker")), nested=nested)
    try:
        if nested:
            from nested_timing import install_worker_scopes
            hooks.buffer.nested_timing = install_worker_scopes(worker, hooks)
        hooks.wrap(worker, "execute_model", "worker_execute",
                   before=lambda arguments, keywords: scheduled_state(arguments[0] if arguments else keywords["scheduler_output"]))
        hooks.wrap(worker, "sample_tokens", "worker_sample",
                   before=(lambda arguments, keywords: {"runner_state_present": worker.model_runner.execute_model_state is not None})
                   if nested else None)
        hooks.wrap(worker, "execute_dummy_batch", "worker_dummy")
    except BaseException:
        hooks.restore()
        raise
    return hooks


def start_engine_timing(engine, case, identity):
    hooks = TimingHooks(TimingBuffer(case, dict(identity, role="engine")))
    scheduler = engine.scheduler
    try:
        hooks.wrap(engine, "step_fn", "engine_step")
        hooks.wrap(scheduler, "schedule", "schedule", before=lambda arguments, keywords: scheduler_state(scheduler),
                   after=lambda output: {**scheduler_state(scheduler), **scheduled_state(output)})
        hooks.wrap(scheduler, "update_from_output", "update", before=lambda arguments, keywords: scheduler_state(scheduler),
                   after=lambda output: scheduler_state(scheduler))
        hooks.wrap(engine.model_executor, "execute_model", "executor_execute",
                   before=lambda arguments, keywords: scheduled_state(arguments[0] if arguments else keywords["scheduler_output"]),
                   future=True)
        hooks.wrap(engine.model_executor, "sample_tokens", "executor_sample")
        hooks.wrap(engine.model_executor, "execute_dummy_batch", "executor_dummy")
    except BaseException:
        hooks.restore()
        raise
    return hooks


def worker_timing_control(worker, command, case=None):
    if os.environ.get("XPLACE_TIMING_ALLOWED") != "1":
        raise RuntimeError("Timing is disabled for this job")
    current = getattr(worker, "_placement_timing_hooks", None)
    if command == "start":
        if current is not None:
            raise RuntimeError("Worker timing is already active")
        identity = worker._placement_timing_identity
        hooks = start_worker_timing(worker, case, identity)
        worker._placement_timing_hooks = hooks
        return {"case": case, "identity": hooks.buffer.identity, "started_ns": hooks.buffer.started_ns}
    if command == "stop" and current is not None:
        result = current.stop()
        worker._placement_timing_hooks = None
        return result
    if command == "abort":
        if current is None:
            return {"inactive": True}
        result = current.stop()
        worker._placement_timing_hooks = None
        return result
    raise ValueError("Invalid worker timing transition")


def bind_engine_timing(engine):
    if os.environ.get("XPLACE_TIMING_ALLOWED") != "1":
        return

    def control(command, case=None):
        current = getattr(engine, "_placement_timing_hooks", None)
        idle = not engine.scheduler.has_requests() and not engine.batch_queue and not (current and current.pending)
        if command == "ready":
            return {"idle": bool(idle)}
        if not idle:
            raise RuntimeError("Timing transitions require an idle engine")
        if command == "start":
            if current is not None:
                raise RuntimeError("Engine timing is already active")
            identity = {"service": os.environ["DP_OPT_SERVICE"], "dp_rank": engine.dp_rank}
            hooks = start_engine_timing(engine, case, identity)
            engine._placement_timing_hooks = hooks
            try:
                workers = engine.collective_rpc("placement_timing", args=("start", case))
            except BaseException:
                hooks.stop()
                engine._placement_timing_hooks = None
                try:
                    engine.collective_rpc("placement_timing", args=("abort",))
                except Exception:
                    pass
                raise
            return {"engine": {"case": case, "identity": hooks.buffer.identity, "started_ns": hooks.buffer.started_ns},
                    "workers": workers}
        if command == "stop" and current is not None:
            result = current.stop()
            engine._placement_timing_hooks = None
            workers = engine.collective_rpc("placement_timing", args=("stop",))
            return {"engine": result, "workers": workers}
        if command == "abort":
            result = current.stop() if current is not None else {"inactive": True}
            engine._placement_timing_hooks = None
            workers = engine.collective_rpc("placement_timing", args=("abort",))
            return {"engine": result, "workers": workers}
        raise ValueError("Invalid engine timing transition")

    engine.placement_timing = control


class FrontendTiming:
    def __init__(self, case):
        self.buffer = TimingBuffer(case, {"role": "frontend", "cadence_thread": threading.get_ident(),
                                         "cadence_clock": "time.monotonic seconds; original stream cadence reused"})
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._watchdog, name="xplace-timing-watchdog", daemon=True)
        self.task = None

    def _watchdog(self):
        expected = time.monotonic_ns() + int(HEARTBEAT_S * 1e9)
        while not self.stop_event.wait(HEARTBEAT_S):
            now = time.monotonic_ns()
            self.buffer.emit("thread_heartbeat", expected_ns=expected, lateness_ns=max(0, now - expected))
            expected = now + int(HEARTBEAT_S * 1e9)

    async def _heartbeat(self):
        while True:
            expected = time.monotonic_ns() + int(HEARTBEAT_S * 1e9)
            await asyncio.sleep(HEARTBEAT_S)
            self.buffer.emit("async_heartbeat", expected_ns=expected, lateness_ns=max(0, time.monotonic_ns() - expected))

    async def start(self):
        if self.task is not None:
            raise RuntimeError("Frontend timing already started")
        self.thread.start()
        self.task = asyncio.create_task(self._heartbeat())
        await asyncio.sleep(0)

    async def stop(self):
        if self.task is None:
            raise RuntimeError("Frontend timing was not started")
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass
        self.stop_event.set()
        await asyncio.to_thread(self.thread.join)
        return self.buffer.stop()


def validate_timing(reports, frontend, case, start_ns, end_ns, num_ranks=2, tensor_parallel_size=2, nested=False):
    if len(reports) != 2 * num_ranks or start_ns >= end_ns:
        raise ValueError("Incomplete engine timing records or invalid measurement window")
    identities, workers = set(), set()
    physical = {}
    records = [frontend]
    for result in reports:
        engine = result["engine"]
        identity = engine["identity"]
        key = (identity["service"], identity["dp_rank"])
        if key in identities or len(result["workers"]) != tensor_parallel_size:
            raise ValueError("Missing or duplicate engine/worker lane")
        identities.add(key)
        records.append(engine)
        for worker in result["workers"]:
            worker_identity = worker["identity"]
            worker_key = (worker_identity["service"], worker_identity["dp_rank"], worker_identity["tp_rank"])
            if worker_key in workers or worker_key[:2] != key or not worker_identity.get("gpu_uuid"):
                raise ValueError("Invalid worker timing identity")
            workers.add(worker_key)
            physical[worker_key] = worker_identity["gpu_uuid"]
            records.append(worker)
    expected_engines = {(f"instance-{model + 1}", rank) for model in range(2) for rank in range(num_ranks)}
    expected_workers = {(service, rank, lane) for service, rank in expected_engines for lane in range(tensor_parallel_size)}
    if identities != expected_engines or workers != expected_workers or frontend["identity"]["role"] != "frontend":
        raise ValueError("Timing topology does not cover every tenant and TP lane")
    for rank in range(num_ranks):
        for lane in range(tensor_parallel_size):
            if physical[("instance-1", rank, lane)] != physical[("instance-2", rank, lane)]:
                raise ValueError("Timing tenants do not share the expected physical GPU mapping")
    if len(set(physical.values())) != num_ranks * tensor_parallel_size:
        raise ValueError("Timing GPU identities are duplicated")
    for record in records:
        if record["case"] != case or record["schema"] != 1 or record["dropped"] or record["late_events"] or record["schema_errors"]:
            raise ValueError("Timing records have wrong case, overflow, late events or schema errors")
        events = record["events"]
        if record["exported"] != len(events) or record["attempted"] != len(events) or len(events) > record["limit"]:
            raise ValueError("Timing event counts do not match")
        if record["started_ns"] > start_ns or record["ended_ns"] < end_ns or not events:
            raise ValueError("Timing does not cover the complete measured window")
        if [event["sequence"] for event in events] != list(range(1, len(events) + 1)):
            raise ValueError("Timing event sequence is incomplete")
        if events[0]["kind"] != "start" or events[-1]["kind"] != "stop":
            raise ValueError("Timing lifecycle boundaries are missing")
        timestamps = [event["at_ns"] for event in events]
        if (any(type(stamp) is not int for stamp in timestamps) or timestamps != sorted(timestamps)
                or timestamps[0] < record["started_ns"] or timestamps[-1] > record["ended_ns"]):
            raise ValueError("Invalid timing clock boundaries")
    if len({record["identity"]["pid"] for record in records}) != len(records):
        raise ValueError("Timing records do not identify distinct processes")
    heartbeat_kinds = {event["kind"] for event in frontend["events"]}
    if not {"async_heartbeat", "thread_heartbeat"} <= heartbeat_kinds:
        raise ValueError("Frontend heartbeat evidence is missing")
    active_engines = 0
    for result in reports:
        scheduled = sum(event.get("scheduled_tokens", 0) for event in result["engine"]["events"]
                        if event["kind"] == "schedule:return")
        active_engines += int(scheduled > 0)
        for worker in result["workers"]:
            executed = sum(event.get("scheduled_tokens", 0) for event in worker["events"]
                           if event["kind"] == "worker_execute:enter")
            if executed != scheduled:
                raise ValueError("Worker execution evidence does not match scheduled tokens")
            dummy_calls = sum(event["kind"] == "executor_dummy:enter" for event in result["engine"]["events"])
            worker_dummy_calls = sum(event["kind"] == "worker_dummy:enter" for event in worker["events"])
            if dummy_calls != worker_dummy_calls:
                raise ValueError("Worker dummy execution evidence does not match the executor")
    if not active_engines:
        raise ValueError("No scheduler/worker execution evidence")
    result = {"status": "passed", "engines": len(identities), "workers": len(workers), "process_records": len(records),
              "events": sum(len(record["events"]) for record in records)}
    if nested:
        from nested_timing import validate_worker_scopes
        result["nested_timing"] = [validate_worker_scopes(worker) for group in reports for worker in group["workers"]]
    elif any("nested_timing" in record for record in records):
        raise ValueError("Unexpected nested timing without a nested plan")
    return result
