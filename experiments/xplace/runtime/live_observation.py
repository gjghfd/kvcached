import asyncio
import atexit
import functools
import json
import math
import os
from pathlib import Path
import socket
import stat
import threading
import time
import uuid

from observation_diagnostics import gather_drained, mark_phase, observe_component


MAX_MESSAGE_BYTES = 4096
MAX_HOST_DEPTH = 4


def host_clock_id():
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    namespace = os.readlink('/proc/self/ns/time')
    if not boot or not namespace.startswith('time:['):
        raise RuntimeError('Cannot identify the local monotonic clock')
    return f'linux-monotonic:{boot}:{namespace}'


def endpoint_path(root, model, rank, lane=None):
    if (type(model) is not int or model not in (0, 1) or type(rank) is not int or rank < 0
            or (lane is not None and (type(lane) is not int or lane < 0))):
        raise ValueError('Invalid observation endpoint identity')
    root = Path(root)
    metadata = root.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
        raise ValueError('Observation root must be an owned private directory, not a symlink')
    role = 'engine' if lane is None else f'tp{lane}'
    path = root.resolve() / f'm{model}-dp{rank}-{role}.sock'
    if len(os.fsencode(path)) > 100:
        raise ValueError('Observation socket path exceeds the portable bound')
    return path


class HostCallState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frames = []
        self.error = None
        self.failed_calls = 0

    def enter(self, name):
        with self.lock:
            owner = threading.get_ident()
            if self.error:
                return None
            if len(self.frames) >= MAX_HOST_DEPTH:
                self.error = 'host_depth_exceeded'
                return None
            if self.frames and self.frames[-1][2] != owner:
                self.error = 'concurrent_host_calls'
                return None
            token = (name, time.monotonic(), owner)
            self.frames.append(token)
            return token

    def leave(self, token, failed):
        with self.lock:
            self.failed_calls += int(failed)
            if token is None:
                return
            if not self.frames or self.frames[-1] is not token:
                self.error = 'host_return_order_mismatch'
                return
            self.frames.pop()

    def snapshot(self):
        with self.lock:
            frame = self.frames[-1] if self.frames else None
            return {'observed_at': time.monotonic(), 'status': 'unavailable' if self.error else 'observed',
                    'reason': self.error, 'host_call': frame[0] if frame else 'none',
                    'host_call_started_at': frame[1] if frame else None,
                    'host_depth': len(self.frames), 'failed_calls': self.failed_calls,
                    'device_activity': 'unknown'}

    def wrap(self, target, method, name):
        original = getattr(target, method)
        present = method in vars(target)
        previous = vars(target).get(method)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            token = self.enter(name)
            failed = True
            try:
                result = original(*args, **kwargs)
                failed = False
                return result
            finally:
                self.leave(token, failed)

        setattr(target, method, wrapped)

        def restore():
            if getattr(target, method) is wrapped:
                if present:
                    setattr(target, method, previous)
                else:
                    delattr(target, method)

        return restore


class SnapshotServer:
    def __init__(self, path, identity, clock_id, sample):
        self.path = Path(path)
        self.identity = dict(identity)
        self.clock_id = clock_id
        self.producer_pid = os.getpid()
        self.producer_id = uuid.uuid4().hex
        self.sample = sample
        self.stopped = threading.Event()
        self.cleanup = []
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.listener.bind(str(path))
        except BaseException:
            self.listener.close()
            raise
        try:
            os.chmod(path, 0o600)
            self.inode = self.path.stat().st_ino
            self.listener.listen(1)
            self.listener.settimeout(0.2)
            self.thread = threading.Thread(target=self.serve, name='xplace-live-snapshot', daemon=True)
            self.thread.start()
        except BaseException:
            self.listener.close()
            self.path.unlink()
            raise
        atexit.register(self.close)

    def serve(self):
        while not self.stopped.is_set():
            try:
                connection, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with connection:
                connection.settimeout(0.2)
                try:
                    record = dict(self.sample(), schema=1, clock_id=self.clock_id,
                                  producer_pid=self.producer_pid, producer_id=self.producer_id, **self.identity)
                    payload = json.dumps(record, allow_nan=False, separators=(',', ':')).encode() + b'\n'
                    if len(payload) > MAX_MESSAGE_BYTES:
                        raise ValueError('Observation exceeds message bound')
                    connection.sendall(payload)
                except Exception as error:
                    failure = dict(self.identity, schema=1, clock_id=self.clock_id,
                                   producer_pid=self.producer_pid,
                                   producer_id=self.producer_id,
                                   observed_at=time.monotonic(), status='unavailable',
                                   error=type(error).__name__, reason='snapshot_failed')
                    try:
                        connection.sendall(json.dumps(failure).encode() + b'\n')
                    except OSError:
                        pass

    def close(self):
        if self.stopped.is_set():
            return
        self.stopped.set()
        self.listener.close()
        self.thread.join(timeout=1)
        for restore in reversed(self.cleanup):
            restore()
        try:
            if self.path.lstat().st_ino == self.inode:
                self.path.unlink()
        except FileNotFoundError:
            pass
        atexit.unregister(self.close)


def observer_settings():
    if os.environ.get('XPLACE_LIVE_OBSERVATION') != '1':
        return None
    if os.environ.get('XPLACE_TIMING_ALLOWED') == '1':
        raise RuntimeError('Live observation and stop-export timing must use separate pilots')
    service = os.environ['DP_OPT_SERVICE']
    if service not in ('instance-1', 'instance-2'):
        raise ValueError('Live observation requires explicit two-tenant service identity')
    return os.environ['XPLACE_LIVE_ROOT'], int(service[-1]) - 1, host_clock_id()


def start_worker_observer(worker):
    settings = observer_settings()
    if settings is None:
        return None
    if getattr(worker, '_placement_live_observer', None) is not None:
        raise RuntimeError('Worker observation already installed')
    root, model, clock_id = settings
    identity = worker._placement_timing_identity
    rank, lane = identity['dp_rank'], identity['tp_rank']
    state = HostCallState()
    cleanup = []
    try:
        for method, name in (('execute_model', 'execute'), ('sample_tokens', 'sample'),
                             ('execute_dummy_batch', 'dummy')):
            cleanup.append(state.wrap(worker, method, name))
        server = SnapshotServer(endpoint_path(root, model, rank, lane),
                                {'model': model, 'dp_rank': rank, 'tp_rank': lane, 'role': 'worker'},
                                clock_id, state.snapshot)
        from batch_link import start_batch_link_observer

        batch_server = start_batch_link_observer(worker, settings, lane)
        if batch_server is not None:
            server.cleanup.append(batch_server.close)
    except BaseException:
        if 'server' in locals():
            server.close()
        for restore in reversed(cleanup):
            restore()
        raise
    server.cleanup.extend(cleanup)
    worker._placement_live_observer = server
    return server


def bind_engine_observer(engine):
    settings = observer_settings()
    if settings is None:
        return None
    if getattr(engine, '_placement_live_observer', None) is not None:
        raise RuntimeError('Engine observation already installed')
    if type(getattr(engine, 'engines_running', None)) is not bool:
        raise RuntimeError('Missing actual DP engine wave state')
    root, model, clock_id = settings
    from accounting_capability import AccountingCapability

    capability = AccountingCapability(engine)

    def sample():
        observed_at = time.monotonic()
        wave = engine.engines_running
        return {'observed_at': observed_at, 'status': 'observed' if type(wave) is bool else 'unavailable',
                'wave_active': wave, 'wave_source': 'engine.engines_running', 'device_activity': 'unknown',
                'accounting_capability': capability.snapshot()}

    server = SnapshotServer(endpoint_path(root, model, engine.dp_rank),
                            {'model': model, 'dp_rank': engine.dp_rank, 'role': 'engine'}, clock_id, sample)
    try:
        from batch_link import start_batch_link_observer

        batch_server = start_batch_link_observer(engine, settings)
        if batch_server is not None:
            server.cleanup.append(batch_server.close)
        for method in ('placement_snapshot', 'placement_summary'):
            original = getattr(engine.scheduler, method)

            def with_clock(*args, _original=original, **kwargs):
                return dict(_original(*args, **kwargs), clock_id=clock_id,
                            producer_pid=server.producer_pid, producer_id=server.producer_id,
                            accounting_capability=capability.snapshot())

            setattr(engine.scheduler, method, with_clock)
            server.cleanup.append(functools.partial(setattr, engine.scheduler, method, original))
    except BaseException:
        server.close()
        raise
    engine._placement_live_observer = server
    return server


def valid_timeout(timeout_s):
    if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s) or timeout_s <= 0):
        raise ValueError('Observation timeout must be finite and positive')


async def read_endpoint(path, identity, clock_id, timeout_s):
    valid_timeout(timeout_s)

    async def receive():
        mark_phase('connect_started')
        reader, writer = await asyncio.open_unix_connection(str(path), limit=MAX_MESSAGE_BYTES)
        mark_phase('connected')
        try:
            mark_phase('read_started')
            payload = await reader.readline()
            available_at = time.monotonic()
            mark_phase('read_completed', wire_bytes=len(payload))
            if not payload.endswith(b'\n') or len(payload) > MAX_MESSAGE_BYTES:
                raise ValueError('Incomplete or oversized observation')
            record = json.loads(payload)
            if (not isinstance(record, dict) or record.get('schema') != 1
                    or any(type(record.get(key)) is not type(value) or record[key] != value
                           for key, value in identity.items())):
                raise ValueError('Observation endpoint identity mismatch')
            if record.get('clock_id') != clock_id:
                raise ValueError('Observation clock domain mismatch')
            observed_at = record.get('observed_at')
            if (isinstance(observed_at, bool) or not isinstance(observed_at, (int, float))
                    or not math.isfinite(observed_at) or observed_at > available_at):
                raise ValueError('Invalid observation timestamp')
            if record.get('status') != 'observed':
                raise ValueError(f"Observation producer unavailable: {record.get('reason', 'unknown')}")
            mark_phase('validated')
            return dict(record, available_at=available_at)
        finally:
            mark_phase('close_started')
            writer.close()
            await writer.wait_closed()
            mark_phase('closed')

    try:
        return await asyncio.wait_for(receive(), timeout_s)
    except (OSError, ValueError, asyncio.TimeoutError) as error:
        return dict(identity, status='unavailable', error=type(error).__name__, reason=str(error),
                    failed_at=time.monotonic(), clock_id=clock_id)


async def collect_execution(root, num_ranks, tensor_parallel_size, timeout_s):
    valid_timeout(timeout_s)
    if any(type(value) is not int or value < 1 for value in (num_ranks, tensor_parallel_size)):
        raise ValueError('Observation requires a complete positive topology')
    clock_id = host_clock_id()

    async def receive(identity, lane):
        async def operation():
            mark_phase('path_started')
            path = endpoint_path(root, identity['model'], identity['dp_rank'], lane)
            mark_phase('path_resolved')
            return await read_endpoint(path, identity, clock_id, timeout_s)
        return await observe_component('execution', identity, operation)

    tasks = []
    for model in (0, 1):
        for rank in range(num_ranks):
            for lane in (None, *range(tensor_parallel_size)):
                identity = {'model': model, 'dp_rank': rank, 'role': 'engine' if lane is None else 'worker'}
                if lane is not None:
                    identity['tp_rank'] = lane
                tasks.append(receive(identity, lane))
    records = await gather_drained(*tasks)
    return {'kind': 'live_execution_transport_v1', 'clock_id': clock_id,
            'status': 'observed' if all(record['status'] == 'observed' for record in records) else 'incomplete',
            'collected_at': time.monotonic(), 'device_activity': 'unknown',
            'rank_waves': [record for record in records if record['role'] == 'engine'],
            'workers': [record for record in records if record['role'] == 'worker']}


async def collect_scheduler(controller, timeout_s):
    valid_timeout(timeout_s)
    clock_id = host_clock_id()

    async def operation(model, rank):
        try:
            mark_phase('rpc_started')
            result = await asyncio.wait_for(controller.call(model, rank, 'placement_snapshot'), timeout_s)
            available_at = time.monotonic()
            mark_phase('rpc_returned')
            if (not isinstance(result, dict) or result.get('clock_id') != clock_id
                    or type(result.get('rank')) is not int or result['rank'] != rank):
                raise ValueError('Scheduler clock or rank mismatch')
            observed_at = result.get('observed_at')
            if (isinstance(observed_at, bool) or not isinstance(observed_at, (int, float))
                    or not math.isfinite(observed_at) or observed_at > available_at):
                raise ValueError('Invalid scheduler observation timestamp')
            mark_phase('validated')
            return {'status': 'observed', 'model': model, 'dp_rank': rank,
                    'available_at': available_at, 'snapshot': result}
        except (OSError, ValueError, RuntimeError, asyncio.TimeoutError) as error:
            return {'status': 'unavailable', 'model': model, 'dp_rank': rank,
                    'failed_at': time.monotonic(), 'error': type(error).__name__, 'reason': str(error)}

    async def receive(model, rank):
        return await observe_component('scheduler', dict(model=model, dp_rank=rank),
                                       lambda: operation(model, rank))

    records = await gather_drained(*(receive(model, rank) for model in (0, 1)
                                   for rank in range(controller.num_ranks)))
    return {'kind': 'live_scheduler_transport_v1', 'clock_id': clock_id,
            'status': 'observed' if all(record['status'] == 'observed' for record in records) else 'incomplete',
            'ranks': records}
