import asyncio
from concurrent.futures import Future
from contextlib import nullcontext
import copy
import json
import multiprocessing
from pathlib import Path
import pickle
import sys
import threading
from types import ModuleType, SimpleNamespace

import cloudpickle
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))

import batch_link as link
import live_observation as live
from test_live_observation import environment


class Worker:
    def __init__(self):
        self._placement_timing_identity = {'dp_rank': 0, 'tp_rank': 0}
        self._placement_batch_links = link.WorkerLinks(0, 0)
        self.arguments = []

    def execute_model(self, output):
        self.arguments.append(output)
        return None

    def sample_tokens(self, grammar):
        return {'tokens': [7], 'grammar': grammar}

    def execute_dummy_batch(self):
        return 'dummy'


class NativeExecutor:
    def __init__(self, worker):
        self.worker = worker
        self.options = []
        self.error = None
        self.future = None
        self.envelopes = []

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None, non_block=False,
                       unique_reply_rank=None, kv_output_aggregator=None):
        self.options.append((timeout, non_block, unique_reply_rank, kv_output_aggregator))
        self.envelopes.append(copy.deepcopy(args[0]) if callable(method) else None)
        send_method = cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL) if callable(method) else method
        buffers = []
        payload = pickle.dumps((send_method, args, kwargs or {}, unique_reply_rank),
                               protocol=pickle.HIGHEST_PROTOCOL, buffer_callback=buffers.append)
        received_method, arguments, keywords, output_rank = pickle.loads(payload, buffers=buffers)
        function = getattr(self.worker, received_method) if isinstance(received_method, str) else (
            lambda *values, **named: cloudpickle.loads(received_method)(self.worker, *values, **named))
        if non_block:
            self.future = Future()
            try:
                if self.error:
                    raise self.error
                self.future.set_result(function(*arguments, **keywords))
            except BaseException as error:
                self.future.set_exception(error)
            return self.future
        if self.error:
            raise self.error
        return function(*arguments, **keywords)

    def execute_model(self, output, non_block=False):
        return self.collective_rpc('execute_model', args=(output,), non_block=non_block,
                                   unique_reply_rank=0, timeout=17, kv_output_aggregator='same-aggregator')

    def sample_tokens(self, grammar, non_block=False):
        return self.collective_rpc('sample_tokens', args=(grammar,), non_block=non_block, unique_reply_rank=0)

    def execute_dummy_batch(self):
        return self.collective_rpc('execute_dummy_batch', unique_reply_rank=0)


class NativeScheduler:
    def __init__(self):
        self.requests = {'request': SimpleNamespace(num_prompt_tokens=8, num_computed_tokens=8,
                                                    num_output_tokens=0)}
        self.output = SimpleNamespace(num_scheduled_tokens={'request': 8}, total_num_scheduled_tokens=8)
        self.return_arguments = None

    def schedule(self, throttled=False):
        self.throttled = throttled
        return self.output

    def update_from_output(self, schedule, output):
        self.return_arguments = schedule, output
        return output

    def has_requests(self):
        return bool(self.requests)

    def get_grammar_bitmask(self, output):
        return None


class NativeEngine:
    def __init__(self, worker=None):
        self.dp_rank = 0
        self.step_counter = 9
        self.current_wave = 3
        self.prefill_schedule_interval = 4
        self.batch_queue = None
        self.batch_queue_size = 1
        self.async_scheduling = False
        self.vllm_config = SimpleNamespace(max_concurrent_batches=1,
            parallel_config=SimpleNamespace(pipeline_parallel_size=1))
        self.scheduler = NativeScheduler()
        self.model_executor = NativeExecutor(worker or Worker())
        self.step_fn = self.step
        self.aborts = 0

    def step(self):
        if not self.scheduler.has_requests():
            return {}, False
        output = self.scheduler.schedule(self._should_throttle_prefills())
        future = self.model_executor.execute_model(output, non_block=True)
        returned = future.result()
        if returned is None:
            returned = self.model_executor.sample_tokens(None)
        self._process_aborts_queue()
        result = self.scheduler.update_from_output(output, returned)
        return result, output.total_num_scheduled_tokens > 0

    def _should_throttle_prefills(self):
        return self.prefill_schedule_interval > 1 and self.step_counter % self.prefill_schedule_interval != 0

    def _process_aborts_queue(self):
        self.aborts += 1

    def capture_iteration_details(self, output):
        return nullcontext(None)

    def log_error_detail(self, output):
        return nullcontext(None)

    def _attach_iteration_details(self, output, details):
        return None


@pytest.fixture
def native(monkeypatch):
    for name, attribute, value in (('vllm.v1.engine.core', 'EngineCore', NativeEngine),
                                   ('vllm.v1.executor.multiproc_executor', 'MultiprocExecutor', NativeExecutor)):
        module = ModuleType(name)
        setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)


def envelope(sequence=1, method='execute_model'):
    return {'engine_epoch': 'a' * 32, 'engine_pid': 12, 'model': 0, 'dp_rank': 0,
            'sequence': sequence, 'features': None if method == 'execute_dummy_batch' else {
                'membership_sha256': 'f' * 64, 'requests': 1, 'scheduled_tokens': 8,
                'prompt_tokens_sum': 8, 'computed_accounted_sum': 8, 'received_output_tokens_sum': 0}, 'rpc': method}


def test_explicit_rpc_link_preserves_native_scheduler_future_and_options(native):
    engine = NativeEngine()
    original_accounting = engine.scheduler.update_from_output.__func__
    original_fields = vars(engine.scheduler.output).copy()
    state, cleanup = link.bind_engine_links(engine, 0)
    result = engine.step_fn()
    assert result == ({'tokens': [7], 'grammar': None}, True)
    assert engine.scheduler.return_arguments[0] is engine.scheduler.output
    assert engine.scheduler.return_arguments[1] is result[0]
    assert vars(engine.scheduler.output) == original_fields
    assert engine.scheduler.update_from_output.__func__ is original_accounting
    assert engine.model_executor.options == [(17, True, 0, 'same-aggregator'), (None, False, 0, None)]
    first, second = engine.model_executor.envelopes
    assert first['sequence'] == second['sequence'] == 1
    assert first['features'] == second['features']
    assert first['rpc'] == 'execute_model' and second['rpc'] == 'sample_tokens'
    assert engine.scheduler.throttled and engine.aborts == 1
    assert engine.model_executor.future.result() is None
    assert state.snapshot()['latest']['scheduler_update_return_bound'] == 'native_step_return_upper_bound'
    assert state.snapshot()['latest']['controller_output_receipt_observed'] is False
    assert state.snapshot()['device_activity'] == 'unknown'
    assert engine.model_executor.worker._placement_batch_links.snapshot()['completed'] == 2
    for restore in reversed(cleanup):
        restore()
    assert engine.step_fn.__func__ is NativeEngine.step
    assert engine.model_executor.collective_rpc.__func__ is NativeExecutor.collective_rpc


def test_dummy_has_distinct_token_and_never_inherits_previous_real_batch(native):
    engine = NativeEngine()
    state, cleanup = link.bind_engine_links(engine, 0)
    engine.step_fn()
    assert engine.model_executor.execute_dummy_batch() == 'dummy'
    snapshot = state.snapshot()
    assert snapshot['latest']['sequence'] == 2
    assert snapshot['latest']['features'] is None
    assert snapshot['latest']['scheduler_update_return_bound'] is None
    assert snapshot['latest']['operation'] == 'dummy'
    engine.step_fn()
    assert state.snapshot()['latest']['sequence'] == 3
    assert engine.model_executor.worker._placement_batch_links.snapshot()['status'] == 'observed'


def test_empty_step_is_not_scheduler_or_gpu_completion(native):
    engine = NativeEngine()
    engine.scheduler.requests.clear()
    state, cleanup = link.bind_engine_links(engine, 0)
    assert engine.step_fn() == ({}, False)
    assert state.snapshot()['latest']['scheduler_update_return_bound'] is None
    assert state.snapshot()['device_activity'] == 'unknown'


@pytest.mark.parametrize('mutation', ['queue', 'depth', 'async', 'pipeline', 'step', 'executor'])
def test_unsupported_paths_reject_installation(native, mutation):
    engine = NativeEngine()
    if mutation == 'queue':
        engine.batch_queue = []
    elif mutation == 'depth':
        engine.batch_queue_size = 2
    elif mutation == 'async':
        engine.async_scheduling = True
    elif mutation == 'pipeline':
        engine.vllm_config.parallel_config.pipeline_parallel_size = 2
    elif mutation == 'step':
        engine.step_fn = lambda: ({}, False)
    else:
        engine.model_executor.execute_model = lambda *args, **kwargs: None
    with pytest.raises(ValueError):
        link.bind_engine_links(engine, 0)


def test_future_failure_has_no_false_scheduler_return_and_original_error_survives(native):
    engine = NativeEngine()
    failure = RuntimeError('executor failure')
    engine.model_executor.error = failure
    state, cleanup = link.bind_engine_links(engine, 0)
    with pytest.raises(RuntimeError) as caught:
        engine.step_fn()
    assert caught.value is failure
    assert state.snapshot()['failures'] == 1
    assert state.snapshot()['latest']['scheduler_update_return_bound'] is None
    assert engine.scheduler.return_arguments is None


def test_failed_scheduler_return_keeps_worker_returns_separate(native):
    engine = NativeEngine()
    def fail(*args):
        raise ValueError('scheduler rejected output')
    engine.scheduler.update_from_output = fail
    state, cleanup = link.bind_engine_links(engine, 0)
    with pytest.raises(ValueError, match='scheduler rejected'):
        engine.step_fn()
    assert state.snapshot()['failures'] == 1
    assert engine.model_executor.worker._placement_batch_links.snapshot()['failures'] == 0


def test_changed_configuration_marks_links_unavailable_without_changing_output(native):
    engine = NativeEngine()
    state, cleanup = link.bind_engine_links(engine, 0)
    engine.vllm_config.max_concurrent_batches = 2
    assert engine.step_fn()[0]['tokens'] == [7]
    assert state.snapshot()['reason'] == 'engine_contract_changed'
    assert engine.model_executor.envelopes == [None, None]


@pytest.mark.parametrize('mutation', ['missing_member', 'total', 'bool', 'negative', 'long_id'])
def test_bad_features_unavailable_not_a_fabricated_batch(native, mutation):
    engine = NativeEngine()
    if mutation == 'missing_member':
        engine.scheduler.output.num_scheduled_tokens = {'absent': 8}
    elif mutation == 'total':
        engine.scheduler.output.total_num_scheduled_tokens = 9
    elif mutation == 'bool':
        engine.scheduler.requests['request'].num_output_tokens = True
    elif mutation == 'negative':
        engine.scheduler.requests['request'].num_computed_tokens = -1
    else:
        engine.scheduler.output.num_scheduled_tokens = {'x' * 513: 8}
    state, cleanup = link.bind_engine_links(engine, 0)
    engine.step_fn()
    assert state.snapshot()['reason'] == 'batch_features_unavailable'
    assert state.snapshot()['latest']['scheduler_update_return_bound'] is None


@pytest.mark.parametrize('mutation', ['model', 'rank', 'epoch', 'sequence', 'missing', 'sample_without_execute'])
def test_worker_rejects_bad_or_missing_link_but_preserves_work(mutation):
    worker = Worker()
    record = envelope()
    method = 'execute_model'
    if mutation == 'model':
        record['model'] = 1
    elif mutation == 'rank':
        record['dp_rank'] = 1
    elif mutation == 'epoch':
        record['engine_epoch'] = 'not-an-epoch'
    elif mutation == 'sequence':
        record['sequence'] = True
    elif mutation == 'missing':
        record.pop('features')
    else:
        method = record['rpc'] = 'sample_tokens'
    result = link.linked_worker_call(worker, record, method, None)
    assert result is None or result['tokens'] == [7]
    assert worker._placement_batch_links.snapshot()['status'] == 'unavailable'


@pytest.mark.parametrize('mutation', ['duplicate', 'reverse', 'changed_features', 'new_epoch', 'new_pid'])
def test_worker_rejects_replay_and_mismatched_sample(mutation):
    worker = Worker()
    link.linked_worker_call(worker, envelope(2), 'execute_model', None)
    record = envelope(2, 'sample_tokens')
    if mutation == 'duplicate':
        record['rpc'] = 'execute_model'
    elif mutation == 'reverse':
        record['sequence'] = 1
    elif mutation == 'changed_features':
        record['features'] = None
    elif mutation == 'new_epoch':
        record['engine_epoch'] = 'b' * 32
    else:
        record['engine_pid'] = 13
    link.linked_worker_call(worker, record, record['rpc'], None)
    assert worker._placement_batch_links.snapshot()['status'] == 'unavailable'


def test_failed_worker_and_overlapping_calls_are_not_hidden():
    worker = Worker()
    def fail(output):
        raise RuntimeError('worker failed')
    worker.execute_model = fail
    with pytest.raises(RuntimeError, match='worker failed'):
        link.linked_worker_call(worker, envelope(), 'execute_model', None)
    state = worker._placement_batch_links
    assert state.snapshot()['failures'] == 1
    assert state.accept(envelope(2), 'execute_model')
    assert not state.accept(envelope(3), 'execute_model')
    assert state.snapshot()['reason'] == 'overlapping_linked_calls'
    state.finish(False)


def test_bounded_latest_snapshot_reports_overwrites_without_claiming_history():
    worker = Worker()
    for sequence in range(1, 101):
        link.linked_worker_call(worker, envelope(sequence), 'execute_model', None)
    snapshot = worker._placement_batch_links.snapshot()
    assert snapshot['completed'] == 100 and snapshot['latest']['sequence'] == 100
    assert snapshot['history_retained'] is False
    snapshot['latest']['sequence'] = 0
    assert worker._placement_batch_links.snapshot()['latest']['sequence'] == 100


def test_feature_hash_is_order_independent_but_causal_counter_sensitive():
    scheduler = NativeScheduler()
    scheduler.requests['peer'] = SimpleNamespace(num_prompt_tokens=5, num_computed_tokens=14, num_output_tokens=10)
    output = SimpleNamespace(num_scheduled_tokens={'request': 8, 'peer': 1}, total_num_scheduled_tokens=9)
    first = link.batch_features(scheduler, output)
    output.num_scheduled_tokens = dict(reversed(list(output.num_scheduled_tokens.items())))
    assert link.batch_features(scheduler, output) == first
    scheduler.requests['peer'].num_output_tokens = 11
    assert link.batch_features(scheduler, output)['membership_sha256'] != first['membership_sha256']


def run_spawned_call(connection):
    worker = Worker()
    received = connection.recv_bytes()
    method, arguments, keywords, rank = pickle.loads(received)
    result = cloudpickle.loads(method)(worker, *arguments, **keywords)
    connection.send((result, worker._placement_batch_links.snapshot(), rank))
    connection.close()


def test_rpc_callable_and_explicit_metadata_survive_spawn_and_real_cloudpickle():
    context = multiprocessing.get_context('spawn')
    parent, child = context.Pipe()
    process = context.Process(target=run_spawned_call, args=(child,))
    process.start()
    child.close()
    payload = (cloudpickle.dumps(link.linked_worker_call), (envelope(), 'execute_model', {'batch': 42}), {}, 1)
    parent.send_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    assert parent.poll(10)
    result, snapshot, rank = parent.recv()
    process.join(10)
    parent.close()
    assert process.exitcode == 0 and result is None and rank == 1
    assert snapshot['latest']['engine_epoch'] == envelope()['engine_epoch']
    assert snapshot['status'] == 'observed' and snapshot['completed'] == 1


def test_default_off_does_not_inspect_or_mutate_target(monkeypatch):
    monkeypatch.delenv('XPLACE_BATCH_LINK', raising=False)
    assert link.start_batch_link_observer(object(), None) is None


def test_separate_socket_preserves_existing_observation_payload_and_cleanup(environment, monkeypatch):
    monkeypatch.setenv('XPLACE_BATCH_LINK', '1')
    worker = Worker()
    del worker._placement_batch_links
    server = live.start_worker_observer(worker)
    identity = {'model': 0, 'dp_rank': 0, 'tp_rank': 0, 'role': 'worker'}
    try:
        link.linked_worker_call(worker, envelope(), 'execute_model', None)
        endpoint = live.endpoint_path(environment / 'batch-links', 0, 0, 0)
        snapshot = asyncio.run(live.read_endpoint(endpoint, identity, live.host_clock_id(), 1))
        assert snapshot['kind'] == 'causal_batch_link_v1' and snapshot['latest']['sequence'] == 1
        assert snapshot['observed_at'] <= snapshot['available_at']
        original = asyncio.run(live.read_endpoint(live.endpoint_path(environment, 0, 0, 0),
                                                  identity, live.host_clock_id(), 1))
        assert 'latest' not in original and original['host_depth'] == 0
    finally:
        server.close()
    assert not endpoint.exists() and not hasattr(worker, '_placement_batch_links')


def test_engine_socket_tracks_step_not_device_idle(native, environment, monkeypatch):
    monkeypatch.setenv('XPLACE_BATCH_LINK', '1')
    engine = NativeEngine()
    original = engine.step_fn
    server = link.start_batch_link_observer(engine, (environment, 0, live.host_clock_id()))
    try:
        engine.step_fn()
        record = asyncio.run(live.read_endpoint(live.endpoint_path(environment / 'batch-links', 0, 0),
            {'model': 0, 'dp_rank': 0, 'role': 'engine'}, live.host_clock_id(), 1))
        assert record['completed'] == 1 and record['device_activity'] == 'unknown'
        assert record['latest']['engine_pid'] == record['producer_pid']
    finally:
        server.close()
    assert engine.step_fn == original


def test_socket_startup_failure_restores_original_worker_methods(environment, monkeypatch):
    monkeypatch.setenv('XPLACE_BATCH_LINK', '1')
    (environment / 'batch-links').symlink_to(environment, target_is_directory=True)
    worker = Worker()
    del worker._placement_batch_links
    original = worker.execute_model
    with pytest.raises(ValueError):
        live.start_worker_observer(worker)
    assert worker.execute_model == original
    assert not live.endpoint_path(environment, 0, 0, 0).exists()


def test_socket_can_sample_during_explicitly_linked_host_call(environment, monkeypatch):
    monkeypatch.setenv('XPLACE_BATCH_LINK', '1')
    worker = Worker()
    del worker._placement_batch_links
    entered, release = threading.Event(), threading.Event()
    def execute(output):
        entered.set()
        if not release.wait(3):
            raise RuntimeError('test release missing')
        return output
    worker.execute_model = execute
    server = live.start_worker_observer(worker)
    returned = []
    thread = threading.Thread(target=lambda: returned.append(
        link.linked_worker_call(worker, envelope(), 'execute_model', 'unchanged-output')))
    try:
        thread.start()
        assert entered.wait(1)
        identity = {'model': 0, 'dp_rank': 0, 'tp_rank': 0, 'role': 'worker'}
        record = asyncio.run(live.read_endpoint(live.endpoint_path(environment / 'batch-links', 0, 0, 0),
            identity, live.host_clock_id(), 1))
        host = asyncio.run(live.read_endpoint(live.endpoint_path(environment, 0, 0, 0),
            identity, live.host_clock_id(), 1))
        assert record['active'] and record['latest']['rpc'] == 'execute_model'
        assert host['host_call'] == 'execute' and host['host_depth'] == 1
        assert 'returned_at' not in record['latest']
    finally:
        release.set()
        thread.join(3)
        server.close()
    assert not thread.is_alive() and returned == ['unchanged-output']


def test_snapshots_stay_bounded_even_for_maximum_supported_membership(native):
    engine = NativeEngine()
    request = engine.scheduler.requests['request']
    engine.scheduler.requests = {str(index): copy.copy(request) for index in range(4096)}
    engine.scheduler.output.num_scheduled_tokens = {key: 8 for key in engine.scheduler.requests}
    engine.scheduler.output.total_num_scheduled_tokens = 4096 * 8
    state, cleanup = link.bind_engine_links(engine, 0)
    engine.step_fn()
    for record in (state.snapshot(), engine.model_executor.worker._placement_batch_links.snapshot()):
        assert len(json.dumps(record).encode()) < live.MAX_MESSAGE_BYTES - 500
        assert record['status'] == 'observed'


def test_outside_step_and_missing_worker_observer_never_invent_link(native):
    engine = NativeEngine()
    state, cleanup = link.bind_engine_links(engine, 0)
    assert engine.model_executor.execute_model(engine.scheduler.output) is None
    assert state.snapshot()['status'] == 'unavailable'
    assert engine.model_executor.envelopes == [None]
    worker = Worker()
    del worker._placement_batch_links
    assert link.linked_worker_call(worker, envelope(), 'execute_model', 'still-executed') is None
    assert worker.arguments == ['still-executed']


def test_unrelated_rpc_options_and_keyword_arguments_are_unchanged(native):
    engine = NativeEngine()
    state, cleanup = link.bind_engine_links(engine, 0)
    engine.model_executor.worker.utility = lambda value, *, named: (value, named)
    assert engine.model_executor.collective_rpc('utility', 13, (9,), {'named': 'same'}, False, 1, None) == (9, 'same')
    assert engine.model_executor.options == [(13, False, 1, None)]
    assert state.snapshot()['latest'] is None


@pytest.mark.parametrize('value', [None, [], {'bad': 1}, {'membership_sha256': 'x' * 10000}])
def test_malformed_feature_envelope_is_unavailable_and_bounded(value):
    worker = Worker()
    record = envelope()
    record['features'] = value
    link.linked_worker_call(worker, record, 'execute_model', None)
    snapshot = worker._placement_batch_links.snapshot()
    assert snapshot['status'] == 'unavailable'
    assert len(json.dumps(snapshot)) < 1024
