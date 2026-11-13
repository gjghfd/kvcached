import copy
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

import nested_timing
import timing_probe
from timing_probe import TimingBuffer, TimingHooks, start_worker_timing, validate_timing


RUNNER_SOURCE = """
def coordinate_batch_across_dp(num_tokens=1):
    return num_tokens

class Runner:
    def _determine_batch_execution_and_padding(self, num_tokens=1):
        return coordinate_batch_across_dp(num_tokens)

    def _sample(self, marker):
        return marker

    def _bookkeeping_sync(self, marker):
        return self._to_list(marker)

    def _to_list(self, marker):
        return marker
"""


class Worker:
    def __init__(self, runner):
        self.model_runner = runner

    def execute_model(self, output):
        if output.total_num_scheduled_tokens:
            self.model_runner._determine_batch_execution_and_padding(output.total_num_scheduled_tokens)
            self.model_runner.execute_model_state = output
        return output

    def sample_tokens(self, marker):
        if self.model_runner.execute_model_state is not None:
            result = self.model_runner._sample(marker)
            result = self.model_runner._bookkeeping_sync(result)
            self.model_runner.execute_model_state = None
            return result
        return marker

    def execute_dummy_batch(self):
        return self.model_runner._determine_batch_execution_and_padding(1)


def group_config(rank, lane):
    return {"dp": {"name": "dp:0", "ranks": [lane, lane + 2], "rank": rank, "world_size": 2},
            "tp": {"name": "tp:0", "ranks": [rank * 2, rank * 2 + 1], "rank": lane, "world_size": 2},
            "ep": {"name": "ep:0", "ranks": [0, 1, 2, 3], "rank": rank * 2 + lane, "world_size": 4}}


@pytest.fixture
def worker_factory(monkeypatch):
    module = ModuleType("vllm.v1.worker.gpu_model_runner")
    exec(RUNNER_SOURCE, module.__dict__)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("XPLACE_NESTED_TIMING", "1")

    def create(service="instance-1", rank=0, lane=0):
        runner = module.Runner()
        runner.parallel_config = SimpleNamespace(enable_dbo=False, enable_eplb=False, pipeline_parallel_size=1,
                tensor_parallel_size=2, data_parallel_size=2, data_parallel_rank=rank,
                disable_nccl_for_dp_synchronization=False, use_sequence_parallel_moe=True,
                all2all_backend="allgather_reducescatter")
        runner.scheduler_config = SimpleNamespace(async_scheduling=False)
        runner.speculative_config = None
        runner.execute_model_state = None
        monkeypatch.setattr(nested_timing, "parallel_groups", lambda: group_config(rank, lane))
        worker = Worker(runner)
        identity = {"service": service, "dp_rank": rank, "tp_rank": lane, "gpu_uuid": f"gpu-{rank}-{lane}"}
        return worker, identity, module
    return create


def run_worker(worker, identity):
    hooks = start_worker_timing(worker, "case", identity)
    output = SimpleNamespace(num_scheduled_tokens={"request": 1}, total_num_scheduled_tokens=1)
    marker = object()
    assert worker.execute_model(output) is output
    assert worker.sample_tokens(marker) is marker
    assert worker.execute_dummy_batch() == 1
    return hooks.stop()


def test_nested_scopes_target_actual_binding_and_restore(worker_factory):
    worker, identity, module = worker_factory(lane=1)
    original = module.coordinate_batch_across_dp
    original_dispatch = worker.model_runner._determine_batch_execution_and_padding
    worker.model_runner._determine_batch_execution_and_padding = lambda *args: original_dispatch(*args)
    record = run_worker(worker, identity)
    assert module.coordinate_batch_across_dp is original
    assert not any(name in vars(worker.model_runner) for name in ("_sample", "_bookkeeping_sync", "_to_list"))
    checked = nested_timing.validate_worker_scopes(record)
    assert checked["counts"] == {"worker_execute": 1, "worker_sample": 1, "worker_dummy": 1,
                                 "runner_sample": 1, "runner_bookkeeping": 1, "runner_to_list": 1, "dp_coordinate": 2}
    spans = nested_timing.scope_intervals(record)
    by_id = {span["scope_id"]: span for span in spans}
    materialize = next(span for span in spans if span["label"] == "runner_to_list")
    assert by_id[materialize["parent"]]["label"] == "runner_bookkeeping"


def test_empty_sample_and_idle_lane_are_explicit(worker_factory):
    worker, identity, _ = worker_factory()
    hooks = start_worker_timing(worker, "case", identity)
    marker = object()
    assert worker.sample_tokens(marker) is marker
    record = hooks.stop()
    checked = nested_timing.validate_worker_scopes(record)
    assert checked["counts"]["worker_sample"] == 1
    assert all(checked["counts"][name] == 0 for name in nested_timing.SCOPES)
    empty = start_worker_timing(worker, "case", identity).stop()
    assert sum(nested_timing.validate_worker_scopes(empty)["counts"].values()) == 0


def test_nested_default_off_does_not_touch_runner(worker_factory, monkeypatch):
    worker, identity, module = worker_factory()
    monkeypatch.delenv("XPLACE_NESTED_TIMING")
    original = module.coordinate_batch_across_dp
    record = run_worker(worker, identity)
    assert "nested_timing" not in record
    assert module.coordinate_batch_across_dp is original
    assert not any("scope_id" in event for event in record["events"])


@pytest.mark.parametrize("value", ["true", "2", "", "yes"])
def test_invalid_nested_switch_rejected(monkeypatch, value):
    monkeypatch.setenv("XPLACE_NESTED_TIMING", value)
    with pytest.raises(ValueError, match="must be"):
        timing_probe.nested_timing_enabled()


@pytest.mark.parametrize("name,value", [("enable_dbo", True), ("enable_eplb", True),
    ("pipeline_parallel_size", 2), ("disable_nccl_for_dp_synchronization", None),
    ("tensor_parallel_size", True), ("use_sequence_parallel_moe", False)])
def test_configuration_mismatch_fails_before_install(worker_factory, name, value):
    worker, identity, module = worker_factory()
    original = module.coordinate_batch_across_dp
    setattr(worker.model_runner.parallel_config, name, value)
    with pytest.raises(ValueError, match="configuration"):
        start_worker_timing(worker, "case", identity)
    assert module.coordinate_batch_across_dp is original and "_sample" not in vars(worker.model_runner)


def test_partial_nested_install_rolls_back(worker_factory):
    worker, identity, module = worker_factory()
    original = module.coordinate_batch_across_dp
    delattr(type(worker.model_runner), "_to_list")
    with pytest.raises(AttributeError):
        start_worker_timing(worker, "case", identity)
    assert module.coordinate_batch_across_dp is original
    assert not any(name in vars(worker.model_runner) for name in ("_sample", "_bookkeeping_sync"))


def test_module_binding_has_one_owner(worker_factory):
    worker, identity, module = worker_factory()
    hooks = start_worker_timing(worker, "case", identity)
    original_wrapper = module.coordinate_batch_across_dp
    other, other_identity, _ = worker_factory()
    with pytest.raises(ValueError, match="already wrapped"):
        start_worker_timing(other, "case", other_identity)
    assert module.coordinate_batch_across_dp is original_wrapper
    assert "_sample" not in vars(other.model_runner)
    hooks.stop()


def test_nested_exception_preserved_and_rejected_by_validator(worker_factory):
    worker, identity, _ = worker_factory()
    failure = RuntimeError("original sampling failure")
    def fail(marker):
        raise failure
    worker.model_runner._to_list = fail
    hooks = start_worker_timing(worker, "case", identity)
    worker.model_runner.execute_model_state = object()
    with pytest.raises(RuntimeError) as observed:
        worker.sample_tokens(object())
    assert observed.value is failure
    record = hooks.stop()
    assert worker.model_runner._to_list is fail
    assert len([event for event in record["events"] if event["kind"].endswith(":error")]) == 3
    with pytest.raises(ValueError, match="failed nested"):
        nested_timing.validate_worker_scopes(record)


def test_no_file_io_or_tensor_materialization_in_hot_hooks(worker_factory, monkeypatch):
    worker, identity, _ = worker_factory()
    hooks = start_worker_timing(worker, "case", identity)
    worker.model_runner.execute_model_state = object()
    class OpaqueTensor:
        def __getattr__(self, name):
            raise AssertionError("No tensor inspection: " + name)
    marker = OpaqueTensor()
    def forbidden(*args, **kwargs):
        raise AssertionError("No hot-path filesystem access")
    with monkeypatch.context() as context:
        context.setattr(Path, "read_bytes", forbidden)
        context.setattr(Path, "write_text", forbidden)
        assert worker.sample_tokens(marker) is marker
        worker.execute_dummy_batch()
    nested_timing.validate_worker_scopes(hooks.stop())


def test_stop_refuses_active_nested_scope(worker_factory):
    worker, identity, _ = worker_factory()
    entered, release = threading.Event(), threading.Event()
    def wait(marker):
        entered.set()
        assert release.wait(5)
        return marker
    worker.model_runner._sample = wait
    worker.model_runner.execute_model_state = object()
    hooks = start_worker_timing(worker, "case", identity)
    thread = threading.Thread(target=worker.sample_tokens, args=(object(),))
    thread.start()
    try:
        assert entered.wait(5)
        with pytest.raises(RuntimeError, match="active scopes"):
            hooks.stop()
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    nested_timing.validate_worker_scopes(hooks.stop())


def test_restore_does_not_clobber_an_external_replacement():
    target = SimpleNamespace(call=lambda: None)
    hooks = TimingHooks(TimingBuffer("case", {}), nested=True)
    hooks.wrap(target, "call", "worker_dummy")
    replacement = lambda: "external"
    target.call = replacement
    with pytest.raises(RuntimeError, match="restoration conflict"):
        hooks.stop()
    assert target.call is replacement and hooks.buffer.schema_errors == 1


@pytest.mark.parametrize("problem", ["parent", "thread", "missing_child", "duplicate_return", "unclosed",
                                     "clock", "state", "source", "restored", "group", "config"])
def test_corrupt_nested_records_rejected(worker_factory, problem):
    worker, identity, _ = worker_factory()
    record = run_worker(worker, identity)
    events = record["events"]
    entered = next(event for event in events if event["kind"] == "runner_to_list:enter")
    returned = next(event for event in events if event["kind"] == "runner_to_list:return")
    if problem == "parent":
        entered["parent_scope_id"] = None
    elif problem == "thread":
        returned["thread"] += 1
    elif problem == "missing_child":
        record["events"] = [event for event in events if not event["kind"].startswith("runner_to_list:")]
    elif problem == "duplicate_return":
        events.insert(events.index(returned), dict(returned))
    elif problem == "unclosed":
        events.remove(returned)
    elif problem == "clock":
        returned["at_ns"] = entered["at_ns"] - 1
    elif problem == "state":
        next(event for event in events if event["kind"] == "worker_sample:enter")["runner_state_present"] = False
    elif problem == "source":
        record["nested_timing"]["source_sha256"] = "0" * 64
    elif problem == "restored":
        record["nested_timing"]["restored"] = False
    elif problem == "group":
        record["nested_timing"]["configuration"]["groups"]["dp"]["ranks"] = [0, 1]
    elif problem == "config":
        record["nested_timing"]["configuration"]["disable_nccl_for_dp_synchronization"] = True
    with pytest.raises(ValueError):
        nested_timing.validate_worker_scopes(record)


def test_all_24_source_guards_are_used(tmp_path, monkeypatch):
    import hashlib
    paths = {name: hashlib.sha256(name.encode()).hexdigest() for name in nested_timing.INTERFACE_SHA256}
    assert len(paths) == 24
    for name in paths:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(name)
    monkeypatch.setattr(nested_timing, "INTERFACE_SHA256", paths)
    assert timing_probe.verify_interface_sources(tmp_path, nested=True) == paths
    (tmp_path / "v1/worker/gpu_model_runner.py").write_text("changed runner")
    with pytest.raises(ValueError, match="Unverified"):
        timing_probe.verify_interface_sources(tmp_path, nested=True)


def test_all_13_processes_require_nested_evidence_on_every_worker(worker_factory):
    reports, records = [], []
    for model in range(2):
        for rank in range(2):
            identity = {"service": f"instance-{model + 1}", "dp_rank": rank, "role": "engine"}
            buffer = TimingBuffer("case", identity)
            buffer.emit("schedule:return", scheduled_tokens=1)
            buffer.emit("executor_dummy:enter")
            engine = buffer.stop()
            workers = []
            for lane in range(2):
                worker, worker_identity, _ = worker_factory(identity["service"], rank, lane)
                workers.append(run_worker(worker, worker_identity))
            reports.append({"engine": engine, "workers": workers})
            records.extend([engine, *workers])
    buffer = TimingBuffer("case", {"role": "frontend"})
    buffer.emit("async_heartbeat", lateness_ns=0)
    buffer.emit("thread_heartbeat", lateness_ns=0)
    frontend = buffer.stop()
    records.append(frontend)
    started = min(record["started_ns"] for record in records)
    ended = max(record["ended_ns"] for record in records)
    for index, record in enumerate(records):
        record["started_ns"], record["ended_ns"] = started, ended
        record["identity"]["pid"] = index + 1
    result = validate_timing(reports, frontend, "case", started, ended, nested=True)
    assert result["process_records"] == 13 and len(result["nested_timing"]) == 8
    with pytest.raises(ValueError, match="without a nested plan"):
        validate_timing(reports, frontend, "case", started, ended)
    damaged = copy.deepcopy(reports)
    del damaged[3]["workers"][1]["nested_timing"]
    with pytest.raises(ValueError, match="Missing or unrestored"):
        validate_timing(damaged, frontend, "case", started, ended, nested=True)


def test_nested_plan_changes_no_workload_fields():
    from timing_calibration import observation_plan
    old = observation_plan()
    new = observation_plan(nested=True)
    metadata = new.pop("nested_timing")
    assert new == old and metadata["scopes"] == list(nested_timing.SCOPES)
    assert metadata["interface_sha256"] == nested_timing.INTERFACE_SHA256
    assert metadata["source_sha256"] == nested_timing.source_sha256()
    with pytest.raises(ValueError):
        observation_plan(nested=1)
