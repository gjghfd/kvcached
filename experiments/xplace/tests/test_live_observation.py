import asyncio
import importlib.util
import multiprocessing
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))

import live_observation as live
from controller import PlacementController
from prediction_state import execution_delivery_state
from test_scheduler_topology import schedulers


@pytest.fixture
def environment(monkeypatch):
    with tempfile.TemporaryDirectory(prefix='xlo-', dir='/tmp') as directory:
        monkeypatch.setenv('XPLACE_LIVE_OBSERVATION', '1')
        monkeypatch.setenv('XPLACE_LIVE_ROOT', directory)
        monkeypatch.setenv('DP_OPT_SERVICE', 'instance-1')
        monkeypatch.delenv('XPLACE_TIMING_ALLOWED', raising=False)
        monkeypatch.setattr(live, 'host_clock_id', lambda: 'test-local-monotonic')
        yield Path(directory)


class Worker:
    def __init__(self, rank=0, lane=0):
        self._placement_timing_identity = {'dp_rank': rank, 'tp_rank': lane}
        self.entered = threading.Event()
        self.release = threading.Event()

    def execute_model(self, value=None):
        return value

    def sample_tokens(self, value=None):
        self.entered.set()
        if not self.release.wait(3):
            raise RuntimeError('test worker was not released')
        return value

    def execute_dummy_batch(self):
        return self.sample_tokens('dummy')


def engine(rank=0):
    scheduler = SimpleNamespace(placement_snapshot=lambda: {'rank': rank, 'observed_at': time.monotonic()},
                                placement_summary=lambda: {'rank': rank, 'observed_at': time.monotonic()})
    return SimpleNamespace(dp_rank=rank, engines_running=False, scheduler=scheduler)


def receive(root, rank=0, lane=0):
    identity = {'model': 0, 'dp_rank': rank, 'role': 'engine' if lane is None else 'worker'}
    if lane is not None:
        identity['tp_rank'] = lane
    return asyncio.run(live.read_endpoint(live.endpoint_path(root, 0, rank, lane), identity,
                                          live.host_clock_id(), 1))


def test_worker_snapshot_observes_blocked_call_without_utility_rpc(environment):
    worker = Worker()
    server = live.start_worker_observer(worker)
    returned = []
    thread = threading.Thread(target=lambda: returned.append(worker.sample_tokens('token')))
    try:
        thread.start()
        assert worker.entered.wait(1)
        record = receive(environment)
        assert record['status'] == 'observed' and record['host_call'] == 'sample'
        assert record['host_call_started_at'] <= record['observed_at'] <= record['available_at']
        assert record['device_activity'] == 'unknown'
        assert 'wave_active' not in record and thread.is_alive()
        worker.release.set()
        thread.join(1)
        assert returned == ['token']
        assert receive(environment)['host_call'] == 'none'
    finally:
        worker.release.set()
        thread.join(3)
        server.close()
    assert 'sample_tokens' not in vars(worker)
    assert not server.path.exists() and not server.thread.is_alive()


def test_engine_wave_is_independent_and_clock_is_producer_attached(environment):
    target = engine()
    original = target.scheduler.placement_snapshot
    server = live.bind_engine_observer(target)
    try:
        record = receive(environment, lane=None)
        assert record['wave_active'] is False and record['device_activity'] == 'unknown'
        target.engines_running = True
        assert receive(environment, lane=None)['wave_active'] is True
        assert target.scheduler.placement_snapshot()['clock_id'] == live.host_clock_id()
        assert target.scheduler.placement_summary()['clock_id'] == live.host_clock_id()
        assert record['wave_source'] == 'engine.engines_running'
    finally:
        server.close()
    assert target.scheduler.placement_snapshot is original


def test_nested_host_methods_are_bounded_and_restore_outer_state(environment):
    worker = Worker()
    server = live.start_worker_observer(worker)
    thread = threading.Thread(target=worker.execute_dummy_batch)
    try:
        thread.start()
        assert worker.entered.wait(1)
        record = receive(environment)
        assert record['host_call'] == 'sample' and record['host_depth'] == 2
        worker.release.set()
        thread.join(1)
        assert receive(environment)['host_depth'] == 0
    finally:
        worker.release.set()
        thread.join(3)
        server.close()


def test_host_exception_and_argument_return_identity_are_preserved():
    expected = RuntimeError('original failure')
    target = SimpleNamespace(execute=lambda value: value)
    state = live.HostCallState()
    restore = state.wrap(target, 'execute', 'execute')
    value = object()
    assert target.execute(value) is value
    restore()

    def fail():
        raise expected

    target.execute = fail
    restore = state.wrap(target, 'execute', 'execute')
    with pytest.raises(RuntimeError) as error:
        target.execute()
    assert error.value is expected
    assert state.snapshot()['failed_calls'] == 1 and state.snapshot()['host_call'] == 'none'
    restore()
    assert target.execute is fail


def test_depth_overflow_retains_bounded_state_not_solo():
    state = live.HostCallState()
    tokens = [state.enter('execute') for _ in range(20)]
    assert len(state.frames) == live.MAX_HOST_DEPTH
    assert state.snapshot()['reason'] == 'host_depth_exceeded'
    for token in reversed(tokens):
        state.leave(token, False)
    assert state.snapshot()['status'] == 'unavailable'


def test_concurrent_calls_are_unavailable_instead_of_last_writer_wins():
    state = live.HostCallState()
    token = state.enter('execute')
    thread = threading.Thread(target=lambda: state.leave(state.enter('sample'), False))
    thread.start()
    thread.join(1)
    state.leave(token, False)
    assert state.snapshot()['reason'] == 'concurrent_host_calls'


def test_bad_return_order_is_unavailable():
    state = live.HostCallState()
    first = state.enter('dummy')
    state.enter('sample')
    state.leave(first, False)
    assert state.snapshot()['reason'] == 'host_return_order_mismatch'


@pytest.mark.parametrize('setting', [None, '', '0'])
def test_disabled_producer_does_not_read_clock_or_mutate_objects(monkeypatch, setting):
    if setting is None:
        monkeypatch.delenv('XPLACE_LIVE_OBSERVATION', raising=False)
    else:
        monkeypatch.setenv('XPLACE_LIVE_OBSERVATION', setting)
    monkeypatch.setattr(live, 'host_clock_id', lambda: pytest.fail('disabled clock read'))
    target = SimpleNamespace()
    assert live.start_worker_observer(target) is None
    assert live.bind_engine_observer(target) is None
    assert vars(target) == {}


def test_no_hot_call_io_or_new_rpc(environment, monkeypatch):
    worker = Worker()
    server = live.start_worker_observer(worker)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(socket, 'socket', lambda *args, **kwargs: pytest.fail('per-call socket'))
            patch.setattr(Path, 'open', lambda *args, **kwargs: pytest.fail('per-call filesystem'))
            value = object()
            for _ in range(100):
                assert worker.execute_model(value) is value
        assert receive(environment)['host_depth'] == 0
    finally:
        server.close()


def test_conflicting_timing_refused_before_any_change(environment, monkeypatch):
    monkeypatch.setenv('XPLACE_TIMING_ALLOWED', '1')
    worker = Worker()
    with pytest.raises(RuntimeError, match='separate pilots'):
        live.start_worker_observer(worker)
    assert 'execute_model' not in vars(worker)
    assert not list(environment.iterdir())


def test_duplicate_endpoint_does_not_remove_existing_listener(environment):
    worker = Worker()
    server = live.start_worker_observer(worker)
    try:
        other = Worker()
        with pytest.raises(OSError):
            live.start_worker_observer(other)
        assert 'execute_model' not in vars(other)
        assert receive(environment)['status'] == 'observed'
        with pytest.raises(RuntimeError, match='already installed'):
            live.start_worker_observer(worker)
    finally:
        server.close()


def test_failed_install_restores_methods(environment):
    target = SimpleNamespace(_placement_timing_identity={'dp_rank': 0, 'tp_rank': 0},
                             execute_model=lambda: 'original')
    original = target.execute_model
    with pytest.raises(AttributeError):
        live.start_worker_observer(target)
    assert target.execute_model is original and not list(environment.iterdir())


@pytest.mark.parametrize('wave', [None, 0, 'idle'])
def test_missing_or_nonboolean_engine_wave_not_fabricated(environment, wave):
    target = engine()
    target.engines_running = wave
    with pytest.raises(RuntimeError, match='actual DP engine'):
        live.bind_engine_observer(target)


def test_shared_or_symlink_root_rejected(environment):
    os.chmod(environment, 0o755)
    with pytest.raises(ValueError, match='private directory'):
        live.endpoint_path(environment, 0, 0)
    os.chmod(environment, 0o700)
    link = environment / 'link'
    link.symlink_to(environment, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        live.endpoint_path(link, 0, 0)


@pytest.mark.parametrize('model,rank,lane', [(True, 0, 0), (2, 0, 0), (0, -1, 0), (0, 0, False)])
def test_invalid_endpoint_identity_rejected(environment, model, rank, lane):
    with pytest.raises(ValueError, match='identity'):
        live.endpoint_path(environment, model, rank, lane)


@pytest.mark.parametrize('damage', ['clock', 'identity', 'timestamp', 'unavailable', 'oversize'])
def test_invalid_endpoint_is_retained_as_unavailable(environment, damage):
    identity = {'model': 0, 'dp_rank': 0, 'tp_rank': 0, 'role': 'worker'}
    record = {'status': 'observed', 'observed_at': time.monotonic()}
    if damage == 'identity':
        identity['model'] = 1
    elif damage == 'timestamp':
        record['observed_at'] = float('inf')
    elif damage == 'unavailable':
        record['status'] = 'unavailable'
    elif damage == 'oversize':
        record['data'] = 'x' * live.MAX_MESSAGE_BYTES
    server = live.SnapshotServer(live.endpoint_path(environment, 0, 0, 0), identity,
                                 'other-clock' if damage == 'clock' else live.host_clock_id(), lambda: record)
    try:
        result = receive(environment)
        assert result['status'] == 'unavailable' and 'available_at' not in result
    finally:
        server.close()


@pytest.mark.parametrize('timeout', [0, -1, float('inf'), float('nan'), True])
def test_invalid_timeout_refused(timeout):
    with pytest.raises(ValueError, match='timeout'):
        live.valid_timeout(timeout)


def test_slow_snapshot_times_out_without_waiting_for_host_completion(environment):
    release = threading.Event()

    def sample():
        release.wait(2)
        return {'status': 'observed', 'observed_at': time.monotonic()}

    identity = {'model': 0, 'dp_rank': 0, 'tp_rank': 0, 'role': 'worker'}
    server = live.SnapshotServer(live.endpoint_path(environment, 0, 0, 0), identity, live.host_clock_id(), sample)
    try:
        result = asyncio.run(live.read_endpoint(server.path, identity, live.host_clock_id(), 0.02))
        assert result['status'] == 'unavailable' and result['error'] == 'TimeoutError'
        assert not release.is_set()
    finally:
        release.set()
        server.close()


def test_all_lanes_and_independent_rank_waves_survive_partial_collection(environment, monkeypatch):
    servers = []
    try:
        for model in (0, 1):
            monkeypatch.setenv('DP_OPT_SERVICE', f'instance-{model + 1}')
            for rank in (0, 1):
                servers.append(live.bind_engine_observer(engine(rank)))
                for lane in (0, 1):
                    servers.append(live.start_worker_observer(Worker(rank, lane)))
        result = asyncio.run(live.collect_execution(environment, 2, 2, 1))
        assert result['status'] == 'observed' and len(result['workers']) == 8
        assert len(result['rank_waves']) == 4
        assert all('wave_active' not in record for record in result['workers'])
        assert result['kind'] != 'live_execution_snapshot'
        validated = execution_delivery_state(result, 2, 2, time.monotonic(), 1, live.host_clock_id())
        assert validated == {'status': 'unavailable', 'reason': 'not_live_delivery'}
        servers[-1].close()
        partial = asyncio.run(live.collect_execution(environment, 2, 2, 1))
        assert partial['status'] == 'incomplete' and len(partial['workers']) == 8
        assert sum(record['status'] == 'unavailable' for record in partial['workers']) == 1
    finally:
        for server in servers:
            server.close()


def test_scheduler_receipts_are_individual_not_gather_completion(environment):
    async def call(model, rank, method):
        assert method == 'placement_snapshot'
        await asyncio.sleep(0.04 if (model, rank) == (1, 1) else 0)
        return {'rank': rank, 'observed_at': time.monotonic(), 'clock_id': live.host_clock_id()}

    controller = SimpleNamespace(call=call, num_ranks=2)
    result = asyncio.run(live.collect_scheduler(controller, 1))
    assert result['status'] == 'observed'
    ranks = result['ranks']
    assert ranks[-1]['available_at'] - ranks[0]['available_at'] >= 0.02
    assert all(record['snapshot']['observed_at'] <= record['available_at'] for record in ranks)


@pytest.mark.parametrize('failure', ['clock', 'missing', 'rank', 'timeout', 'exception'])
def test_scheduler_failure_keeps_all_rank_records(environment, failure):
    async def call(model, rank, method):
        result = {'rank': rank, 'observed_at': time.monotonic(), 'clock_id': live.host_clock_id()}
        if (model, rank) == (0, 0):
            if failure == 'timeout':
                await asyncio.sleep(1)
            elif failure == 'exception':
                raise RuntimeError('scheduler failure')
            else:
                result[{'clock': 'clock_id', 'missing': 'clock_id', 'rank': 'rank'}[failure]] = None
        return result

    result = asyncio.run(live.collect_scheduler(SimpleNamespace(call=call, num_ranks=2), 0.02))
    assert result['status'] == 'incomplete' and len(result['ranks']) == 4
    assert result['ranks'][0]['status'] == 'unavailable'
    assert all(record['status'] == 'observed' for record in result['ranks'][1:])


def test_controller_opt_in_method_collects_concurrently_without_scoring(environment):
    async def call(model, rank, method):
        return {'rank': rank, 'clock_id': live.host_clock_id(), 'observed_at': time.monotonic()}

    async def scenario():
        controller = PlacementController([None, None], 2, [40, 40])
        controller.tensor_parallel_size = 2
        controller.call = call
        result = await controller.prediction_observations(environment, 0.2)
        assert not controller.events and not controller.transactions
        return result

    result = asyncio.run(scenario())
    assert result['scheduler']['status'] == 'observed'
    assert result['execution']['status'] == 'incomplete'
    assert len(result['execution']['workers']) == 8 and len(result['execution']['rank_waves']) == 4
    assert result['forecast_status'] == 'unavailable_not_estimated'


@pytest.mark.parametrize('timestamp', [None, True, float('nan'), float('inf')])
def test_scheduler_bad_timestamp_is_not_a_valid_receipt(environment, timestamp):
    async def call(model, rank, method):
        return {'rank': rank, 'clock_id': live.host_clock_id(), 'observed_at': timestamp}

    result = asyncio.run(live.collect_scheduler(SimpleNamespace(call=call, num_ranks=2), 0.1))
    assert result['status'] == 'incomplete'
    assert all('available_at' not in record for record in result['ranks'])


def test_observer_error_does_not_silently_kill_listener(environment):
    count = 0

    def sample():
        nonlocal count
        count += 1
        if count == 1:
            raise RuntimeError('sample failure')
        return {'status': 'observed', 'observed_at': time.monotonic()}

    identity = {'model': 0, 'dp_rank': 0, 'tp_rank': 0, 'role': 'worker'}
    server = live.SnapshotServer(live.endpoint_path(environment, 0, 0, 0), identity, live.host_clock_id(), sample)
    try:
        result = receive(environment)
        assert result['status'] == 'unavailable' and 'snapshot_failed' in result['reason']
        assert receive(environment)['status'] == 'observed'
    finally:
        server.close()


def child_blocked_call(root, entered, release):
    state = live.HostCallState()
    identity = {'model': 0, 'dp_rank': 0, 'tp_rank': 0, 'role': 'worker'}
    server = live.SnapshotServer(live.endpoint_path(root, 0, 0, 0), identity,
                                 'test-local-monotonic', state.snapshot)
    token = state.enter('sample')
    try:
        entered.set()
        release.wait(5)
    finally:
        state.leave(token, False)
        server.close()


def test_separate_process_blocked_host_snapshot_and_receipt(environment):
    context = multiprocessing.get_context('spawn')
    entered, release = context.Event(), context.Event()
    process = context.Process(target=child_blocked_call, args=(environment, entered, release))
    process.start()
    try:
        assert entered.wait(3)
        record = receive(environment)
        assert record['status'] == 'observed' and record['host_call'] == 'sample'
        assert record['observed_at'] <= record['available_at']
        assert record['producer_pid'] == process.pid
        assert process.is_alive()
    finally:
        release.set()
        process.join(3)
        if process.is_alive():
            process.terminate()
            process.join(1)
    assert process.exitcode == 0


def test_invalid_root_cancels_sibling_scheduler_calls(environment):
    async def scenario():
        controller = PlacementController([None, None], 2, [40, 40])

        async def call(*args):
            await asyncio.sleep(10)

        controller.call = call
        with pytest.raises(FileNotFoundError):
            await controller.prediction_observations(environment / 'missing', 1)
        await asyncio.sleep(0)
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(scenario())


def test_cancellation_closes_socket_and_does_not_return_late_delivery(environment):
    release = threading.Event()
    entered = threading.Event()

    def sample():
        entered.set()
        release.wait(2)
        return {'status': 'observed', 'observed_at': time.monotonic()}

    identity = {'model': 0, 'dp_rank': 0, 'tp_rank': 0, 'role': 'worker'}
    server = live.SnapshotServer(live.endpoint_path(environment, 0, 0, 0), identity, live.host_clock_id(), sample)

    async def scenario():
        task = asyncio.create_task(live.read_endpoint(server.path, identity, live.host_clock_id(), 2))
        for _ in range(100):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert asyncio.all_tasks() == {asyncio.current_task()}

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        server.close()


def test_native_clock_identity_includes_boot_and_namespace(monkeypatch):
    monkeypatch.setattr(Path, 'read_text', lambda path: 'boot-id\n')
    monkeypatch.setattr(os, 'readlink', lambda path: 'time:[42]')
    assert live.host_clock_id() == 'linux-monotonic:boot-id:time:[42]'
    monkeypatch.setattr(os, 'readlink', lambda path: 'unknown')
    with pytest.raises(RuntimeError, match='clock'):
        live.host_clock_id()


def test_actual_engine_binding_exposes_clock_on_existing_rpc(environment, schedulers):
    scheduler = schedulers.CalibrationScheduler(tp=2)
    scheduler.placement_snapshot = lambda: {'rank': 0, 'observed_at': time.monotonic()}
    scheduler.placement_summary = scheduler.placement_snapshot
    target = SimpleNamespace(dp_rank=0, engines_running=False, scheduler=scheduler)
    schedulers.bind_engine(target)
    try:
        assert target.placement_snapshot()['clock_id'] == live.host_clock_id()
        assert target.placement_summary()['clock_id'] == live.host_clock_id()
        assert receive(environment, lane=None)['wave_active'] is False
    finally:
        target._placement_live_observer.close()


def test_actual_worker_initializer_installs_after_lane_identity(environment, monkeypatch):
    groups = {name: SimpleNamespace(rank_in_group=0, world_size=size)
              for name, size in [('dp', 2), ('tp', 2), ('ep', 4)]}
    monkeypatch.setattr(Worker, 'initialize_from_config', lambda self, config: None, raising=False)
    modules = {
        'torch': SimpleNamespace(cuda=SimpleNamespace(
            get_device_properties=lambda device: SimpleNamespace(uuid='gpu-test'),
            get_device_name=lambda device: 'fake-A800')),
        'vllm.distributed.parallel_state': SimpleNamespace(
            get_dp_group=lambda: groups['dp'], get_tp_group=lambda: groups['tp'], get_ep_group=lambda: groups['ep']),
        'vllm.v1.worker.gpu_worker': SimpleNamespace(Worker=Worker),
        'kvcached': SimpleNamespace(utils=SimpleNamespace(DEFAULT_IPC_NAME='test-ipc')),
        'kvcached.integration.vllm.interfaces': SimpleNamespace(
            observability_snapshot_dict=lambda: {'initialized': True, 'world_size': 2, 'device': 'cuda:0'}),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv('DP_OPT_EVIDENCE_ROOT', str(environment))
    monkeypatch.delenv('DP_OPT_CAPTURE_EVIDENCE', raising=False)
    path = Path(__file__).resolve().parents[1] / 'runtime/worker.py'
    spec = importlib.util.spec_from_file_location('live_worker_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    worker = module.VerificationWorker()
    worker.device = 'cuda:0'
    worker.parallel_config = SimpleNamespace(data_parallel_rank=0)
    worker.model_runner = SimpleNamespace(get_model=lambda: SimpleNamespace(
        model=SimpleNamespace(layers=[]), parameters=lambda: []))
    worker.model_config = SimpleNamespace(model='test-checkpoint', max_model_len=128)
    worker.cache_config = SimpleNamespace(block_size=16, kv_cache_memory_bytes=1024)
    worker.scheduler_config = SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=128)
    worker.initialize_from_config(SimpleNamespace(num_blocks=8))
    try:
        assert receive(environment)['status'] == 'observed'
        assert worker.execute_model('unchanged') == 'unchanged'
    finally:
        worker._placement_live_observer.close()
