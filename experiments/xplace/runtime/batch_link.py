import copy
import functools
import inspect
import os
from pathlib import Path
import threading
import time
import uuid

from reference_work import fingerprint


LINK_METHODS = ('execute_model', 'sample_tokens', 'execute_dummy_batch')


class LinkState:
    def __init__(self, model, rank):
        self.model = model
        self.rank = rank
        self.lock = threading.RLock()
        self.latest = None
        self.active = False
        self.error = None
        self.revision = 0
        self.completed = 0
        self.failures = 0

    def reject(self, reason):
        with self.lock:
            self.error = self.error or reason

    def begin(self, record):
        with self.lock:
            if self.active:
                self.reject('overlapping_linked_calls')
                return False
            self.latest = copy.deepcopy(record)
            self.active = True
            self.revision += 1
            return True

    def change(self, **fields):
        with self.lock:
            if self.latest is not None:
                self.latest.update(copy.deepcopy(fields))
                self.revision += 1

    def finish(self, failed, **fields):
        with self.lock:
            self.change(returned_at=time.monotonic(), failed=failed, **fields)
            self.active = False
            self.completed += 1
            self.failures += int(failed)

    def snapshot(self):
        with self.lock:
            return {'kind': 'causal_batch_link_v1', 'observed_at': time.monotonic(),
                    'status': 'unavailable' if self.error else 'observed', 'reason': self.error,
                    'active': self.active, 'revision': self.revision, 'completed': self.completed,
                    'failures': self.failures, 'latest': copy.deepcopy(self.latest),
                    'device_activity': 'unknown', 'history_retained': False}


def batch_features(scheduler, output):
    scheduled = output.num_scheduled_tokens
    if not isinstance(scheduled, dict) or len(scheduled) > 4096:
        raise ValueError('Unsupported batch membership')
    rows = []
    for request_id, tokens in sorted(scheduled.items()):
        if (not isinstance(request_id, str) or not request_id or len(request_id) > 512
                or type(tokens) is not int or tokens <= 0):
            raise ValueError('Invalid scheduled member')
        request = scheduler.requests[request_id]
        counters = [request.num_prompt_tokens, request.num_computed_tokens, request.num_output_tokens]
        if any(type(value) is not int or value < 0 for value in counters):
            raise ValueError('Missing causal request counters')
        rows.append([request_id, tokens, *counters])
    total = sum(row[1] for row in rows)
    if type(output.total_num_scheduled_tokens) is not int or output.total_num_scheduled_tokens != total:
        raise ValueError('Scheduled total does not match membership')
    return {'membership_sha256': fingerprint(rows), 'requests': len(rows), 'scheduled_tokens': total,
            'prompt_tokens_sum': sum(row[2] for row in rows),
            'computed_accounted_sum': sum(row[3] for row in rows),
            'received_output_tokens_sum': sum(row[4] for row in rows)}


def function_of(method):
    return getattr(method, '__func__', method)


def valid_features(features, method):
    if method == 'execute_dummy_batch':
        return features is None
    fields = {'requests', 'scheduled_tokens', 'prompt_tokens_sum', 'computed_accounted_sum',
              'received_output_tokens_sum'}
    return (isinstance(features, dict) and set(features) == fields | {'membership_sha256'}
            and all(type(features[name]) is int and 0 <= features[name] < 2 ** 63 for name in fields)
            and features['requests'] <= 4096
            and isinstance(features['membership_sha256'], str) and len(features['membership_sha256']) == 64
            and all(letter in '0123456789abcdef' for letter in features['membership_sha256']))


class EngineLinks(LinkState):
    def __init__(self, engine, model):
        super().__init__(model, engine.dp_rank)
        self.engine = engine
        self.epoch = uuid.uuid4().hex
        self.sequence = 0
        self.calls = []
        self.envelope = None

    def begin_operation(self, operation):
        with self.lock:
            self.sequence += 1
            record = {'engine_epoch': self.epoch, 'engine_pid': os.getpid(), 'model': self.model,
                      'dp_rank': self.rank, 'sequence': self.sequence, 'operation': operation,
                      'started_at': time.monotonic()}
            for name in ('step_counter', 'current_wave', 'prefill_schedule_interval'):
                value = getattr(self.engine, name, None)
                if type(value) is not int or value < 0:
                    self.reject('missing_dp_cadence_state')
                record[name] = value if type(value) is int and value >= 0 else None
            if not self.begin(record):
                return False
            self.calls = []
            self.envelope = None
            return True

    def dispatch(self, method, arguments):
        with self.lock:
            if not self.active or self.error:
                self.reject('rpc_without_supported_step')
                return None
            if method == 'execute_model' and not self.calls:
                try:
                    features = batch_features(self.engine.scheduler, arguments[0])
                except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                    self.reject('batch_features_unavailable')
                    return None
            elif method == 'sample_tokens' and self.calls == ['execute_model']:
                features = self.envelope['features']
            elif method == 'execute_dummy_batch' and not self.calls and self.latest['operation'] == 'dummy':
                features = None
            else:
                self.reject('unexpected_rpc_order')
                return None
            if not valid_features(features, method):
                self.reject('batch_features_unavailable')
                return None
            self.calls.append(method)
            envelope = {key: self.latest[key] for key in ('engine_epoch', 'engine_pid', 'model', 'dp_rank', 'sequence')}
            envelope.update(features=features, rpc=method)
            self.envelope = envelope
            self.change(rpc=method, rpc_dispatched_at=time.monotonic(), features=features)
            return copy.deepcopy(envelope)


class WorkerLinks(LinkState):
    def __init__(self, model, rank):
        super().__init__(model, rank)
        self.origin = None
        self.previous = None

    def accept(self, envelope, method):
        with self.lock:
            try:
                if (set(envelope) != {'engine_epoch', 'engine_pid', 'model', 'dp_rank', 'sequence', 'features', 'rpc'}
                        or envelope['model'] != self.model or type(envelope['model']) is not int
                        or envelope['dp_rank'] != self.rank or type(envelope['dp_rank']) is not int
                        or method not in LINK_METHODS or envelope['rpc'] != method
                        or type(envelope['sequence']) is not int or envelope['sequence'] <= 0
                        or type(envelope['engine_pid']) is not int or envelope['engine_pid'] <= 0
                        or not valid_features(envelope['features'], method)
                        or not isinstance(envelope['engine_epoch'], str) or len(envelope['engine_epoch']) != 32
                        or any(letter not in '0123456789abcdef' for letter in envelope['engine_epoch'])):
                    raise ValueError('Invalid envelope')
                origin = envelope['engine_pid'], envelope['engine_epoch']
                if self.origin is not None and origin != self.origin:
                    raise ValueError('Changed origin')
                if method == 'sample_tokens':
                    if (self.previous is None or self.previous['rpc'] != 'execute_model'
                            or any(envelope[key] != self.previous[key] for key in envelope if key != 'rpc')):
                        raise ValueError('Sample does not follow its execute')
                elif self.previous is not None and envelope['sequence'] <= self.previous['sequence']:
                    raise ValueError('Repeated or reversed operation')
                if self.error or not self.begin(dict(envelope, started_at=time.monotonic())):
                    return False
                self.origin = origin
                self.previous = copy.deepcopy(envelope)
                return True
            except (KeyError, TypeError, ValueError):
                self.reject('invalid_or_out_of_order_envelope')
                return False


def linked_worker_call(worker, envelope, method, *args, **kwargs):
    state = getattr(worker, '_placement_batch_links', None)
    entered = state is not None and state.accept(envelope, method)
    failed = True
    try:
        result = getattr(worker, method)(*args, **kwargs)
        failed = False
        return result
    finally:
        if entered:
            state.finish(failed)


def replace_method(target, name, method):
    present, previous = name in vars(target), vars(target).get(name)
    setattr(target, name, method)

    def restore():
        if getattr(target, name) is method:
            if present:
                setattr(target, name, previous)
            else:
                delattr(target, name)

    return restore


def engine_contract(engine):
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    if (engine.batch_queue is not None or type(engine.batch_queue_size) is not int or engine.batch_queue_size != 1
            or engine.vllm_config.max_concurrent_batches != 1 or engine.async_scheduling is not False
            or engine.vllm_config.parallel_config.pipeline_parallel_size != 1):
        raise ValueError('Batch linking requires the native synchronous single-batch path')
    bindings = [(engine, 'step_fn', EngineCore.step)]
    bindings.extend((engine.model_executor, name, getattr(MultiprocExecutor, name))
                    for name in ('collective_rpc', *LINK_METHODS))
    if any(function_of(getattr(target, name)) is not expected for target, name, expected in bindings):
        raise ValueError('Unsupported engine or executor override')
    return [(target, name, expected, expected.__code__) for target, name, expected in bindings]


def bind_engine_links(engine, model):
    bindings = engine_contract(engine)
    state = EngineLinks(engine, model)
    original_step = engine.step_fn
    original_rpc = engine.model_executor.collective_rpc
    signature = inspect.signature(original_rpc)

    def guard():
        valid = all(expected.__code__ is code for _, _, expected, code in bindings)
        valid = valid and engine.batch_queue is None and engine.batch_queue_size == 1
        valid = valid and engine.async_scheduling is False and engine.vllm_config.max_concurrent_batches == 1
        valid = valid and engine.vllm_config.parallel_config.pipeline_parallel_size == 1
        valid = valid and all(function_of(getattr(target, name)) is expected for target, name, expected, _ in bindings
                              if name not in ('step_fn', 'collective_rpc'))
        valid = valid and engine.step_fn is step and engine.model_executor.collective_rpc is rpc
        if not valid:
            state.reject('engine_contract_changed')

    @functools.wraps(original_step)
    def step(*args, **kwargs):
        guard()
        entered = state.begin_operation('step')
        failed = True
        try:
            result = original_step(*args, **kwargs)
            failed = False
            return result
        finally:
            if entered:
                state.finish(failed, scheduler_update_return_bound=(
                    'native_step_return_upper_bound' if not failed and state.calls and not state.error else None),
                    controller_output_receipt_observed=False)

    @functools.wraps(original_rpc)
    def rpc(*args, **kwargs):
        call = signature.bind(*args, **kwargs)
        call.apply_defaults()
        method = call.arguments['method']
        if not isinstance(method, str) or method not in LINK_METHODS:
            return original_rpc(*args, **kwargs)
        guard()
        dummy = method == 'execute_dummy_batch'
        entered = state.begin_operation('dummy') if dummy else False
        if dummy and call.arguments['non_block']:
            state.reject('nonblocking_dummy_not_supported')
        envelope = state.dispatch(method, call.arguments['args'])
        if envelope is not None:
            call.arguments['method'] = linked_worker_call
            call.arguments['args'] = (envelope, method, *call.arguments['args'])
        failed = True
        try:
            result = original_rpc(*call.args, **call.kwargs)
            failed = False
            return result
        finally:
            if entered:
                state.finish(failed, scheduler_update_return_bound=None)

    cleanup = [replace_method(engine, 'step_fn', step), replace_method(engine.model_executor, 'collective_rpc', rpc)]
    return state, cleanup


def start_batch_link_observer(target, settings, lane=None):
    if os.environ.get('XPLACE_BATCH_LINK') != '1':
        return None
    from live_observation import SnapshotServer, endpoint_path

    root, model, clock_id = settings
    if getattr(target, '_placement_batch_links', None) is not None:
        raise RuntimeError('Batch linking already installed')
    rank = target.dp_rank if lane is None else target._placement_timing_identity['dp_rank']
    endpoint_path(root, model, rank, lane)
    directory = Path(root) / 'batch-links'
    directory.mkdir(mode=0o700, exist_ok=True)
    endpoint = endpoint_path(directory, model, rank, lane)
    state, cleanup = bind_engine_links(target, model) if lane is None else (WorkerLinks(model, rank), [])
    identity = {'model': model, 'dp_rank': rank, 'role': 'engine' if lane is None else 'worker'}
    if lane is not None:
        identity['tp_rank'] = lane
    try:
        server = SnapshotServer(endpoint, identity, clock_id, state.snapshot)
    except BaseException:
        for restore in reversed(cleanup):
            restore()
        raise
    server.cleanup.extend(cleanup)
    target._placement_batch_links = state
    server.cleanup.append(lambda: delattr(target, '_placement_batch_links'))
    return server
