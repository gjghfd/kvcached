import asyncio
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
import graph_execution as graph


CLOCK = "linux-monotonic:cpu-test:time:[1]"


def identity(rank=0, lane=0):
    return dict(model=1, dp_rank=rank, tp_rank=lane, tp_size=2, pid=100 + rank * 2 + lane,
                gpu_uuid=f"gpu-{rank}-{lane}", graph_epoch=f"{rank * 2 + lane:032x}", clock_id=CLOCK)


def build_hooks(mode="FULL", padded=32, unpadded=17):
    context = SimpleNamespace(cudagraph_runtime_mode=SimpleNamespace(name=mode),
                              batch_descriptor=SimpleNamespace(num_tokens=padded, num_reqs=unpadded, uniform=True))
    result = (context.cudagraph_runtime_mode, context.batch_descriptor, False, None, None)

    class Runner:
        def _determine_batch_execution_and_padding(self, *args, **kwargs):
            return result

    class Graph:
        def replay(self):
            return "native-return"

    runner = Runner()
    hooks = graph.GraphHooks(runner, Graph, lambda: context, identity())
    return hooks, runner, Graph, result, context


def pair(with_replay=True):
    initial, final = [], []
    for rank in range(2):
        starts, stops = [], []
        for lane in range(2):
            hooks, runner, native, result, context = build_hooks()
            hooks.identity = identity(rank, lane)
            hooks.install()
            starts.append(hooks.control("start", "point"))
            runner._determine_batch_execution_and_padding(17)
            if with_replay:
                native().replay()
            stops.append(hooks.control("stop", "point"))
            hooks.uninstall()
        initial.append(starts)
        final.append(stops)
    common = dict(case="point", status="received", errors=[], local_tasks_pending=[],
                  remote_rpc_may_continue=False, gpu_completion_observed=False)
    return dict(common, command="start", results=initial), dict(common, command="stop", results=final)


@pytest.fixture(autouse=True)
def local_clock(monkeypatch):
    monkeypatch.setattr(graph, "host_clock_id", lambda: CLOCK)


@pytest.mark.parametrize("mode,padded,unpadded", [("FULL", 32, 17), ("NONE", 17, 17), ("PIECEWISE", 64, 33)])
@pytest.mark.parametrize("keyword", [False, True])
def test_hooks_preserve_result_and_actual_decision(mode, padded, unpadded, keyword):
    hooks, runner, native, result, context = build_hooks(mode, padded, unpadded)
    original = native.replay
    hooks.install()
    assert runner._determine_batch_execution_and_padding(unpadded) is result
    assert native().replay() == "native-return"
    assert hooks.buffer is None
    hooks.control("start", "point")
    observed = (runner._determine_batch_execution_and_padding(num_tokens=unpadded) if keyword
                else runner._determine_batch_execution_and_padding(unpadded))
    assert observed is result
    if mode != "NONE":
        assert native().replay() == "native-return"
    record = hooks.control("stop", "point")
    event = record["events"][1]
    assert event["mode"] == mode and event["padded_tokens"] == padded and event["unpadded_tokens"] == unpadded
    assert record["invalid"] == record["dropped"] == 0
    assert not record["added_cuda_sync"] and not record["gpu_completion_observed"]
    hooks.uninstall()
    assert native.replay is original


def test_native_replay_exception_preserved():
    hooks, runner, native, result, context = build_hooks()
    failure = RuntimeError("native failure")

    def fail(instance):
        raise failure

    hooks.original_replay = fail
    hooks.install()
    hooks.control("start", "point")
    with pytest.raises(RuntimeError) as caught:
        native().replay()
    assert caught.value is failure
    assert hooks.control("stop", "point")["events"][-2]["kind"] == "replay_raised"
    hooks.uninstall()


def test_native_selection_exception_preserved():
    hooks, runner, native, result, context = build_hooks()
    failure = ValueError("native selection")

    def fail(*args, **kwargs):
        raise failure

    hooks.original_selection = fail
    hooks.install()
    hooks.control("start", "point")
    with pytest.raises(ValueError) as caught:
        runner._determine_batch_execution_and_padding(17)
    assert caught.value is failure
    assert hooks.control("stop", "point")["events"][-2]["kind"] == "selection_raised"
    hooks.uninstall()


@pytest.mark.parametrize("bad_context", [True, False])
def test_observation_error_does_not_change_native_return(bad_context):
    hooks, runner, native, result, context = build_hooks()
    hooks.install()
    hooks.control("start", "point")
    if bad_context:
        hooks.get_context = lambda: None
        assert native().replay() == "native-return"
    else:
        assert runner._determine_batch_execution_and_padding(True) is result
    report = hooks.control("stop", "point")
    assert report["invalid"] == 1 and any(event["kind"] == "observation_error" for event in report["events"])
    hooks.uninstall()


def test_hook_ownership_and_reused_case_rejected():
    hooks, runner, native, result, context = build_hooks()
    hooks.install()
    with pytest.raises(ValueError, match="already instrumented"):
        graph.GraphHooks(runner, native, lambda: context, identity())
    hooks.control("start", "point")
    with pytest.raises(ValueError):
        hooks.control("start", "other")
    hooks.control("stop", "point")
    with pytest.raises(ValueError):
        hooks.control("start", "point")
    owned = runner._determine_batch_execution_and_padding
    runner._determine_batch_execution_and_padding = lambda: None
    with pytest.raises(ValueError, match="ownership"):
        hooks.control("start", "other")
    runner._determine_batch_execution_and_padding = owned
    hooks.uninstall()


@pytest.mark.parametrize("limit", ["events", "bytes", "invalid", "closed"])
def test_bounded_buffer_retains_all_loss_counters(limit):
    buffer = graph.GraphBuffer("point", identity(), event_limit=2 if limit == "events" else 20, byte_limit=1024)
    if limit == "invalid":
        buffer.emit("bad", value=float("nan"))
    elif limit == "closed":
        buffer.stop()
        buffer.emit("late")
    else:
        buffer.emit("large", value="a" * (1024 if limit == "bytes" else 1))
    report = buffer.stop()
    assert report["dropped"] >= 1
    assert report["attempted"] == report["retained"] + report["dropped"]
    assert report["bytes_retained"] <= report["byte_limit"]
    assert report["retained"] <= report["event_limit"]


@pytest.mark.parametrize("with_replay", [False, True])
def test_four_lane_pair_and_no_gpu_claim(with_replay):
    result = graph.validate_graph_pair(*pair(with_replay))
    assert result["valid"], result
    assert len(result["lanes"]) == 4 and not result["gpu_completion_observed"]
    assert all(lane["host_replay_calls"] == int(with_replay) for lane in result["lanes"])


@pytest.mark.parametrize("mutation", ["pid", "epoch", "clock", "source", "bool_rank", "lost_lane", "duplicate_lane", "bytes",
                                      "drop", "invalid", "sequence", "nan", "reverse_time", "open_replay", "extra_return",
                                      "mode", "padding", "early_stop", "start_time", "cancelled_control", "model"])
def test_pair_rejects_invalid_evidence(mutation):
    start, stop = pair()
    row = stop["results"][0][0]
    event = row["events"][1]
    if mutation in ("pid", "epoch", "clock", "bool_rank", "model"):
        key, value = {"pid": ("pid", 999), "epoch": ("graph_epoch", "f" * 32), "clock": ("clock_id", "wrong"),
                      "bool_rank": ("tp_rank", False), "model": ("model", 99)}[mutation]
        row["identity"][key] = value
    elif mutation == "source":
        row["source_sha256"] = {}
    elif mutation == "lost_lane":
        stop["results"][1].pop()
    elif mutation == "duplicate_lane":
        stop["results"][0][1] = copy.deepcopy(row)
    elif mutation in ("bytes", "drop", "invalid"):
        row[{"bytes": "bytes_retained", "drop": "dropped", "invalid": "invalid"}[mutation]] += 1
    elif mutation == "sequence":
        event["sequence"] = 99
    elif mutation == "nan":
        event["at"] = float("nan")
    elif mutation == "reverse_time":
        event["at"] = 0
    elif mutation == "open_replay":
        row["events"].pop(-2)
    elif mutation == "extra_return":
        row["events"][-2]["call"] = 99
    elif mutation == "mode":
        event["mode"] = "UNKNOWN"
    elif mutation == "padding":
        event["unpadded_tokens"] = 999
    elif mutation == "early_stop":
        event["kind"] = "stop"
    elif mutation == "start_time":
        start["results"][0][0]["started_at"] = 0
    elif mutation == "cancelled_control":
        stop["remote_rpc_may_continue"] = True
    if mutation not in ("bytes", "nan"):
        row["bytes_retained"] = sum(len(json.dumps(event, separators=(",", ":")).encode()) for event in row["events"])
    assert not graph.validate_graph_pair(start, stop)["valid"]


@pytest.mark.parametrize("condition", ["success", "missing", "timeout", "cancel", "rpc_error"])
def test_controller_retains_partial_replies_and_cancellation(tmp_path, monkeypatch, condition):
    start, stop = pair()
    entered = None
    finished = []

    class Controller:
        num_ranks, tensor_parallel_size = 2, 2

        async def call(self, model, rank, rpc, method, timeout, args, kwargs):
            assert (model, rpc, method, args, kwargs) == (1, "collective_rpc", "placement_graph_execution", ("start", "point"), {})
            try:
                if rank == 1 and condition in ("timeout", "cancel"):
                    entered.set()
                    await asyncio.Event().wait()
                if rank == 1 and condition == "rpc_error":
                    raise RuntimeError("remote failed")
                return [] if condition == "missing" and rank == 1 else start["results"][rank]
            finally:
                finished.append(rank)

    async def run():
        nonlocal entered
        entered = asyncio.Event()
        task = asyncio.create_task(graph.graph_control(Controller(), 1, "start", "point", tmp_path))
        if condition == "cancel":
            await entered.wait()
            task.cancel()
        if condition == "success":
            assert (await task)["status"] == "received"
        else:
            with pytest.raises(asyncio.CancelledError if condition == "cancel" else RuntimeError):
                await task

    monkeypatch.setattr(graph, "CONTROL_TIMEOUT_S", 0.03)
    asyncio.run(run())
    record = json.loads((tmp_path / "point-graph-start.json").read_text())
    assert record["results"][0] and record["local_tasks_pending"] == [] and sorted(finished) == [0, 1]
    assert (record["status"] == "received") == (condition == "success")
    assert record["remote_rpc_may_continue"] == (condition in ("timeout", "cancel"))


def test_worker_default_off_and_control_unavailable(monkeypatch):
    monkeypatch.delenv("XPLACE_GRAPH_EXECUTION_ALLOWED", raising=False)
    worker = SimpleNamespace()
    graph.start_worker_graph_execution(worker)
    assert not hasattr(worker, "_xplace_graph_hooks")
    with pytest.raises(RuntimeError, match="not enabled"):
        graph.worker_graph_control(worker, "start", "point")


@pytest.mark.parametrize("mismatch", [False, True])
def test_worker_source_pin_and_lifecycle(tmp_path, monkeypatch, mismatch):
    hooks, runner, native, result, context = build_hooks()
    source = tmp_path / "source.py"
    source.write_text("pinned source")
    hashes = {key: hashlib.sha256(source.read_bytes()).hexdigest() for key in graph.SOURCE_SHA256}
    monkeypatch.setattr(graph, "SOURCE_SHA256", hashes)
    monkeypatch.setattr(graph.inspect, "getsourcefile", lambda function: str(source))
    monkeypatch.setenv("XPLACE_GRAPH_EXECUTION_ALLOWED", "1")
    monkeypatch.setenv("DP_OPT_SERVICE", "instance-2")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        CUDAGraph=native, get_device_properties=lambda device: SimpleNamespace(uuid="gpu-0-0"))))
    monkeypatch.setitem(sys.modules, "vllm.compilation.cuda_graph", SimpleNamespace(CUDAGraphWrapper=object))
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", SimpleNamespace(
        get_tp_group=lambda: SimpleNamespace(rank_in_group=0, world_size=2)))
    monkeypatch.setitem(sys.modules, "vllm.forward_context", SimpleNamespace(get_forward_context=lambda: context))
    worker = SimpleNamespace(model_runner=runner, parallel_config=SimpleNamespace(data_parallel_rank=0), device="cuda:0")
    if mismatch:
        source.write_text("changed source")
        with pytest.raises(ValueError, match="source mismatch"):
            graph.start_worker_graph_execution(worker)
        assert not hasattr(worker, "_xplace_graph_hooks")
    else:
        graph.start_worker_graph_execution(worker)
        assert graph.worker_graph_control(worker, "start", "point")["identity"]["tp_size"] == 2
        graph.worker_graph_control(worker, "stop", "point")
        worker._xplace_graph_hooks.uninstall()


@pytest.mark.parametrize("phases,flag,expected", [(["trace"], True, "0"), (["causal"], True, "0"),
                                                 (["dataplane", "imbalance"], True, "1"),
                                                 (["imbalance"], False, "0"), (["imbalance"], 1, "0"),
                                                 (["imbalance"], None, "0")])
def test_actual_main_scope_clears_inherited_flag(tmp_path, monkeypatch, phases, flag, expected):
    source = Path(__file__).resolve().parents[1] / "runtime/main.py"
    run = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.AsyncFunctionDef) and node.name == "run")
    first = next(index for index, node in enumerate(run.body) if isinstance(node, ast.Assign)
                 and "XPLACE_GRAPH_EXECUTION_ALLOWED" in ast.unparse(node))
    statements = ast.Module(body=run.body[first:first + 2], type_ignores=[])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(observe_graph_execution=flag)))
    monkeypatch.setenv("XPLACE_GRAPH_EXECUTION_ALLOWED", "1")
    exec(compile(statements, str(source), "exec"), dict(os=os, json=json, phases=phases,
                                                      args=SimpleNamespace(imbalance_config=path)))
    assert os.environ["XPLACE_GRAPH_EXECUTION_ALLOWED"] == expected


def test_actual_worker_rpc_delegates_without_changing_return(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "runtime/worker.py"
    worker = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "VerificationWorker")
    method = next(node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == "placement_graph_execution")
    init = next(node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == "initialize_from_config")
    assert sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "start_worker_graph_execution"
               for node in ast.walk(init)) == 1
    calls, result = [], object()

    def control(instance, command, case):
        calls.append((instance, command, case))
        return result

    monkeypatch.setattr(graph, "worker_graph_control", control)
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    instance = object()
    assert namespace["placement_graph_execution"](instance, "start", "point") is result
    assert calls == [(instance, "start", "point")]


@pytest.mark.parametrize("key,value", [("tp_rank", False), ("pid", True), ("model", True), ("graph_epoch", "wrong")])
def test_start_identity_types_are_checked(key, value):
    start, stop = pair()
    start["results"][0][0]["identity"][key] = value
    assert not graph.validate_graph_pair(start, stop)["valid"]
