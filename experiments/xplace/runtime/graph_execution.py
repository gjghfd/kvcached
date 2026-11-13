import asyncio
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

from engine import save
from live_observation import host_clock_id


EVENT_LIMIT = 20000
BYTE_LIMIT = 16 * 1024 ** 2
CONTROL_TIMEOUT_S = 10
SOURCE_SHA256 = {
    "vllm/compilation/cuda_graph.py": "cbc474f9098386d2eef2e3ff61364c611fbc1338d9c259ad5a24439aa7f06412",
    "vllm/v1/worker/gpu_model_runner.py": "81b7627fbe81f7aaa2f77b4bf085faa353c69d03662ebfe369536a9773bb70d0",
    "vllm/forward_context.py": "985adacc9e7ed2a7695819ac2e2ad9cb264a7d886806d3091ad75bfda7392242",
    "torch/cuda/graphs.py": "1f4cf2183a1ef1897787c0dc6d13d5afe3ab57de1dd76bb9345323fd64b742b0",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(value):
    require(type(value) is int and 0 <= value < 2 ** 63, "Invalid host token counter")
    return value


def descriptor(mode, batch):
    name = getattr(mode, "name", None)
    require(name in ("NONE", "PIECEWISE", "FULL"), "Unsupported runtime graph mode")
    requests = batch.num_reqs
    require(requests is None or type(requests) is int and requests >= 0, "Invalid batch request count")
    require(type(batch.uniform) is bool, "Invalid batch uniform flag")
    return dict(mode=name, padded_tokens=count(batch.num_tokens), batch_requests=requests, uniform=batch.uniform)


def validate_identity(identity, model, rank, clock):
    require(isinstance(identity, dict), "Missing graph identity")
    for key in ("model", "dp_rank", "tp_rank", "tp_size", "pid"):
        count(identity[key])
    require(identity["model"] == model and model in (0, 1) and identity["dp_rank"] == rank and rank in (0, 1)
            and identity["tp_rank"] in (0, 1) and identity["tp_size"] == 2
            and identity["pid"] > 0 and identity["clock_id"] == clock, "Graph worker lane or clock differs")
    require(isinstance(identity["gpu_uuid"], str) and bool(identity["gpu_uuid"]), "Missing GPU identity")
    require(isinstance(identity["graph_epoch"], str) and len(identity["graph_epoch"]) == 32
            and all(char in "0123456789abcdef" for char in identity["graph_epoch"]), "Invalid graph worker epoch")


class GraphBuffer:
    def __init__(self, case, identity, *, event_limit=EVENT_LIMIT, byte_limit=BYTE_LIMIT):
        require(isinstance(case, str) and 0 < len(case) <= 160, "Invalid graph case")
        require(type(event_limit) is int and 2 <= event_limit <= EVENT_LIMIT, "Invalid event limit")
        require(type(byte_limit) is int and 1024 <= byte_limit <= BYTE_LIMIT, "Invalid byte limit")
        self.case, self.identity = case, copy.deepcopy(identity)
        self.event_limit, self.byte_limit = event_limit, byte_limit
        self.events, self.bytes, self.attempted, self.dropped, self.invalid = [], 0, 0, 0, 0
        self.closed, self.lock = False, threading.RLock()
        self.calls = 0
        self.emit("start")

    def emit(self, kind, **fields):
        with self.lock:
            self.attempted += 1
            record = dict(sequence=self.attempted, kind=kind, at=time.monotonic(), **fields)
            try:
                encoded = json.dumps(record, allow_nan=False, separators=(",", ":"))
                size = len(encoded.encode())
            except (TypeError, ValueError):
                self.invalid += 1
                self.dropped += 1
                return
            if self.closed or len(self.events) >= self.event_limit or self.bytes + size > self.byte_limit:
                self.dropped += 1
                return
            self.events.append(encoded)
            self.bytes += size

    def error(self, error, scope):
        with self.lock:
            self.invalid += 1
            self.emit("observation_error", scope=scope, error_type=type(error).__name__, reason=str(error)[:256])

    def replay_start(self, fields):
        with self.lock:
            self.calls += 1
            self.emit("replay_enter", call=self.calls, **fields)
            return self.calls

    def stop(self):
        with self.lock:
            if not self.closed:
                self.emit("stop")
                self.closed = True
            return self.report()

    def report(self):
        with self.lock:
            return dict(kind="graph_execution_v1", case=self.case, identity=copy.deepcopy(self.identity),
                        events=[json.loads(event) for event in self.events], stopped=self.closed,
                        attempted=self.attempted, retained=len(self.events), dropped=self.dropped, invalid=self.invalid,
                        bytes_retained=self.bytes, event_limit=self.event_limit, byte_limit=self.byte_limit,
                        gpu_completion_observed=False, added_cuda_sync=False, source_sha256=dict(SOURCE_SHA256))


class GraphHooks:
    def __init__(self, runner, graph_class, get_context, identity):
        require(not hasattr(graph_class.replay, "_xplace_graph_owner"), "Graph replay already instrumented")
        self.runner, self.graph_class, self.get_context = runner, graph_class, get_context
        self.identity, self.buffer, self.seen = copy.deepcopy(identity), None, set()
        self.original_selection = runner._determine_batch_execution_and_padding
        self.original_replay = graph_class.replay
        self.installed = False

    def active(self):
        buffer = self.buffer
        return buffer if buffer is not None and not buffer.closed else None

    def install(self):
        require(not self.installed, "Graph hooks already installed")

        def selection(*args, **kwargs):
            try:
                result = self.original_selection(*args, **kwargs)
            except BaseException as error:
                buffer = self.active()
                if buffer is not None:
                    buffer.emit("selection_raised", error_type=type(error).__name__)
                raise
            buffer = self.active()
            if buffer is not None:
                try:
                    mode, batch, ubatch, across_dp, stats = result
                    fields = descriptor(mode, batch)
                    unpadded = kwargs["num_tokens"] if "num_tokens" in kwargs else args[0]
                    require(type(ubatch) is bool, "Invalid microbatch flag")
                    fields.update(unpadded_tokens=count(unpadded), microbatch=ubatch,
                                  dp_padding_metadata_present=across_dp is not None)
                    buffer.emit("selection_return", **fields)
                except Exception as error:
                    buffer.error(error, "selection")
            return result

        def replay(graph, *args, **kwargs):
            buffer = self.active()
            call = None
            if buffer is not None:
                try:
                    context = self.get_context()
                    call = buffer.replay_start(descriptor(context.cudagraph_runtime_mode, context.batch_descriptor))
                except Exception as error:
                    buffer.error(error, "replay_context")
            try:
                result = self.original_replay(graph, *args, **kwargs)
            except BaseException as error:
                if buffer is not None:
                    buffer.emit("replay_raised", call=call, error_type=type(error).__name__)
                raise
            if buffer is not None:
                buffer.emit("replay_return", call=call)
            return result

        replay._xplace_graph_owner = self
        selection._xplace_graph_owner = self
        self.runner._determine_batch_execution_and_padding = selection
        self.graph_class.replay = replay
        self.installed = True

    def control(self, command, case):
        require(self.installed, "Graph hooks are not installed")
        require(getattr(self.graph_class.replay, "_xplace_graph_owner", None) is self
                and getattr(self.runner._determine_batch_execution_and_padding, "_xplace_graph_owner", None) is self,
                "Graph hook ownership changed")
        if command == "start":
            require(self.active() is None and case not in self.seen, "Graph case already active or reused")
            self.buffer = GraphBuffer(case, self.identity)
            self.seen.add(case)
            return dict(kind="graph_start_v1", case=case, identity=copy.deepcopy(self.identity),
                        source_sha256=dict(SOURCE_SHA256), event_limit=EVENT_LIMIT, byte_limit=BYTE_LIMIT,
                        started_at=json.loads(self.buffer.events[0])["at"])
        require(command == "stop" and self.buffer is not None and self.buffer.case == case, "Unknown graph control or case")
        return self.buffer.stop()

    def uninstall(self):
        require(self.active() is None, "Stop active graph case before removing hooks")
        require(getattr(self.graph_class.replay, "_xplace_graph_owner", None) is self, "Graph hook ownership changed")
        require(getattr(self.runner._determine_batch_execution_and_padding, "_xplace_graph_owner", None) is self,
                "Selection hook ownership changed")
        self.graph_class.replay = self.original_replay
        self.runner._determine_batch_execution_and_padding = self.original_selection
        self.installed = False


def start_worker_graph_execution(worker):
    if os.environ.get("XPLACE_GRAPH_EXECUTION_ALLOWED") != "1":
        return
    import torch
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.distributed.parallel_state import get_tp_group
    from vllm.forward_context import get_forward_context

    functions = {"vllm/compilation/cuda_graph.py": CUDAGraphWrapper.__call__,
                 "vllm/v1/worker/gpu_model_runner.py": worker.model_runner._determine_batch_execution_and_padding,
                 "vllm/forward_context.py": get_forward_context, "torch/cuda/graphs.py": torch.cuda.CUDAGraph.replay}
    for name, function in functions.items():
        source = inspect.getsourcefile(function)
        require(source is not None and hashlib.sha256(Path(source).read_bytes()).hexdigest() == SOURCE_SHA256[name],
                "Graph execution source mismatch: " + name)
    service = os.environ["DP_OPT_SERVICE"]
    require(service in ("instance-1", "instance-2"), "Unsupported graph worker identity")
    identity = dict(model=int(service[-1]) - 1, dp_rank=worker.parallel_config.data_parallel_rank,
                    tp_rank=get_tp_group().rank_in_group, tp_size=get_tp_group().world_size,
                    pid=os.getpid(), clock_id=host_clock_id(),
                    graph_epoch=uuid.uuid4().hex,
                    gpu_uuid=str(torch.cuda.get_device_properties(worker.device).uuid))
    hooks = GraphHooks(worker.model_runner, torch.cuda.CUDAGraph, get_forward_context, identity)
    hooks.install()
    worker._xplace_graph_hooks = hooks


def worker_graph_control(worker, command, case):
    hooks = getattr(worker, "_xplace_graph_hooks", None)
    if hooks is None:
        raise RuntimeError("Graph execution evidence was not enabled before worker initialization")
    return hooks.control(command, case)


async def graph_control(controller, model, command, case, root):
    require(controller.num_ranks == 2 and controller.tensor_parallel_size == 2, "Graph scan requires DP2/TP2")
    require(type(model) is int and model in (0, 1) and command in ("start", "stop"), "Invalid graph control")
    require(isinstance(case, str) and 0 < len(case) <= 160 and all(char.isalnum() or char in "_-" for char in case),
            "Invalid graph case")
    clock = host_clock_id()
    tasks = [asyncio.create_task(controller.call(model, rank, "collective_rpc", "placement_graph_execution",
                                                CONTROL_TIMEOUT_S, (command, case), {})) for rank in range(2)]
    interrupted = None
    try:
        done, pending = await asyncio.wait(tasks, timeout=CONTROL_TIMEOUT_S)
    except BaseException as error:
        interrupted = error
        done = {task for task in tasks if task.done()}
        pending = set(tasks) - done
    for task in pending:
        task.cancel()
    cleanup_pending = set(pending)
    if pending:
        try:
            _, cleanup_pending = await asyncio.wait(pending, timeout=0.1)
        except BaseException as error:
            interrupted = interrupted or error
        for task in pending:
            if task.done() and not task.cancelled():
                task.exception()
    results, errors, identities = [], [], []
    if interrupted is not None:
        errors.append(dict(error="control_interrupted", type=type(interrupted).__name__))
    for rank, task in enumerate(tasks):
        if task not in done:
            errors.append(dict(dp_rank=rank, error="control_interrupted" if interrupted else "control_timeout"))
            results.append(None)
        elif task.cancelled() or task.exception() is not None:
            errors.append(dict(dp_rank=rank, error="cancelled" if task.cancelled() else str(task.exception())))
            results.append(None)
        else:
            result = task.result()
            results.append(result)
            try:
                require(isinstance(result, list) and len(result) == 2, "Missing TP graph replies")
                lanes = set()
                for row in result:
                    identity = row["identity"]
                    require(row["case"] == case and row["source_sha256"] == SOURCE_SHA256
                            and row["kind"] == ("graph_start_v1" if command == "start" else "graph_execution_v1"),
                            "Graph reply identity differs")
                    validate_identity(identity, model, rank, clock)
                    lanes.add(identity["tp_rank"])
                    identities.append(identity)
                require(lanes == {0, 1}, "Duplicate graph lane")
            except (KeyError, TypeError, ValueError) as error:
                errors.append(dict(dp_rank=rank, error=str(error)))
    if not errors:
        if len({row["pid"] for row in identities}) != 4 or len({row["gpu_uuid"] for row in identities}) != 4:
            errors.append(dict(error="Graph workers are not four distinct processes and GPUs"))
    record = dict(case=case, command=command, results=results, errors=errors,
                  status="received" if not errors else "unavailable", gpu_completion_observed=False,
                  local_tasks_pending=[rank for rank, task in enumerate(tasks) if task in cleanup_pending],
                  remote_rpc_may_continue=bool(pending))
    save(root / f"{case}-graph-{command}.json", record)
    if interrupted is not None:
        raise interrupted
    if errors:
        raise RuntimeError(f"Graph {command} incomplete: {errors}")
    return record


def validate_graph_pair(start, stop, *, clock_id=None):
    errors, lanes = [], []
    try:
        clock = host_clock_id() if clock_id is None else clock_id
        require(isinstance(clock, str) and clock.startswith("linux-monotonic:") and ":time:[" in clock,
                "Missing attested graph clock")
        require(isinstance(start, dict) and isinstance(stop, dict), "Missing graph control record")
        require(start["command"] == "start" and stop["command"] == "stop" and start["case"] == stop["case"],
                "Graph control pair differs")
        for record in (start, stop):
            require(record["status"] == "received" and record["errors"] == []
                    and record["local_tasks_pending"] == [] and record["remote_rpc_may_continue"] is False
                    and record["gpu_completion_observed"] is False, "Incomplete graph control")
            require(isinstance(record["results"], list) and len(record["results"]) == 2, "Missing DP replies")
        identities = []
        for rank, (initial, final) in enumerate(zip(start["results"], stop["results"])):
            require(isinstance(initial, list) and isinstance(final, list) and len(initial) == len(final) == 2,
                    "Missing TP replies")
            initial_by_lane = {row["identity"]["tp_rank"]: row for row in initial}
            require(set(initial_by_lane) == {0, 1} and {row["identity"]["tp_rank"] for row in final} == {0, 1},
                    "Duplicate graph lane")
            for row in final:
                identity = row["identity"]
                validate_identity(identity, identity["model"], rank, clock)
                first = initial_by_lane[identity["tp_rank"]]
                validate_identity(first["identity"], identity["model"], rank, clock)
                count(first["event_limit"])
                count(first["byte_limit"])
                require(first["identity"] == identity, "Graph identity changed within case")
                require(first["kind"] == "graph_start_v1" and row["kind"] == "graph_execution_v1"
                        and first["case"] == row["case"] == start["case"]
                        and first["source_sha256"] == row["source_sha256"] == SOURCE_SHA256, "Graph attestation differs")
                require(row["stopped"] is True and row["gpu_completion_observed"] is False
                        and row["added_cuda_sync"] is False, "Unsupported graph timing claim")
                for key in ("attempted", "retained", "dropped", "invalid", "bytes_retained", "event_limit", "byte_limit"):
                    count(row[key])
                events = row["events"]
                require(isinstance(events, list) and row["attempted"] == row["retained"] + row["dropped"]
                        and row["retained"] == len(events) and row["dropped"] == row["invalid"] == 0,
                        "Graph event loss or invalid observation")
                require(2 <= first["event_limit"] == row["event_limit"] <= EVENT_LIMIT
                        and 1024 <= first["byte_limit"] == row["byte_limit"] <= BYTE_LIMIT
                        and len(events) <= row["event_limit"], "Graph limits differ")
                size = sum(len(json.dumps(event, allow_nan=False, separators=(",", ":")).encode()) for event in events)
                require(size == row["bytes_retained"] <= row["byte_limit"], "Graph byte accounting differs")
                require(len(events) >= 2 and events[0]["kind"] == "start" and events[-1]["kind"] == "stop",
                        "Missing graph boundaries")
                require(type(first["started_at"]) in (int, float) and first["started_at"] == events[0]["at"],
                        "Graph start timestamp changed")
                calls, finished, selections, replay_modes = set(), set(), {}, {}
                previous_at = -1.0
                for sequence, event in enumerate(events, 1):
                    require(type(event["sequence"]) is int and event["sequence"] == sequence, "Graph sequence gap")
                    stamp = event["at"]
                    require(type(stamp) in (float, int) and math.isfinite(stamp) and stamp >= max(0, previous_at),
                            "Invalid graph timestamp")
                    previous_at = stamp
                    kind = event["kind"]
                    require(kind in ("start", "stop", "selection_return", "replay_enter", "replay_return"),
                            "Failed or unknown graph event")
                    require(kind != "start" or sequence == 1, "Repeated start")
                    require(kind != "stop" or sequence == len(events), "Early stop")
                    if kind in ("selection_return", "replay_enter"):
                        require(event["mode"] in ("NONE", "FULL", "PIECEWISE"), "Unknown graph mode")
                        padded = count(event["padded_tokens"])
                        requests = event["batch_requests"]
                        require(requests is None or type(requests) is int and requests >= 0, "Invalid batch requests")
                        require(type(event["uniform"]) is bool, "Invalid batch flag")
                        bucket = event["mode"] + ":" + str(padded)
                        if kind == "selection_return":
                            require(count(event["unpadded_tokens"]) <= padded and type(event["microbatch"]) is bool
                                    and type(event["dp_padding_metadata_present"]) is bool, "Invalid padding decision")
                            selections[bucket] = selections.get(bucket, 0) + 1
                        else:
                            require(event["mode"] != "NONE", "Replay has eager context")
                            call = count(event["call"])
                            require(call == len(calls) + 1, "Duplicate or missing replay call")
                            calls.add(call)
                            replay_modes[bucket] = replay_modes.get(bucket, 0) + 1
                    if kind == "replay_return":
                        call = count(event["call"])
                        require(call in calls and call not in finished, "Unmatched replay return")
                        finished.add(call)
                require(calls == finished, "Open replay at stop")
                identities.append(identity)
                lanes.append(dict(identity=identity, selections=selections, host_replay_calls=len(calls), replay_modes=replay_modes))
        require(len({identity["pid"] for identity in identities}) == 4
                and len({identity["gpu_uuid"] for identity in identities}) == 4
                and len({identity["model"] for identity in identities}) == 1, "Graph workers are not one model/four GPUs")
    except (KeyError, TypeError, ValueError, IndexError) as error:
        errors.append(dict(type=type(error).__name__, reason=str(error)))
    return dict(valid=not errors, errors=errors, lanes=lanes, gpu_completion_observed=False,
                scope="host_padding_decisions_and_replay_calls_only")
