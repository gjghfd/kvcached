import asyncio
from enum import Enum
import json
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))

import causal_collection as collection
from engine import stream
import live_observation as live


class OutputKind(Enum):
    CUMULATIVE = 0
    DELTA = 1


class SamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.output_kind = OutputKind.CUMULATIVE


@pytest.fixture(autouse=True)
def sources(monkeypatch):
    monkeypatch.setattr(collection, 'host_clock_id', lambda: 'test-receipt-clock')
    module = ModuleType('vllm')
    module.SamplingParams = SamplingParams
    monkeypatch.setitem(sys.modules, 'vllm', module)


def recorder(**kwargs):
    parameters = dict(case_id='case', clock_id='test-receipt-clock', period_s=0.01, deadline_s=0.02,
                      max_events=200, max_bytes=1000000, max_requests=16)
    parameters.update(kwargs)
    return collection.CausalCollection(**parameters)


def test_revised_retention_capacity_remains_bounded():
    observer = recorder(max_events=1_000_000, max_bytes=512 * 1024 ** 2)
    assert observer.protocol['max_bytes'] == 512 * 1024 ** 2
    assert observer.bytes_retained == 0 and observer.events == []
    with pytest.raises(ValueError, match='bounded'):
        recorder(max_bytes=512 * 1024 ** 2 + 1)
    with pytest.raises(ValueError, match='bounded'):
        recorder(max_events=1_000_001)


def packet(count, *, finished=False, reason=None, request_id='request', index=0):
    return SimpleNamespace(request_id=request_id, finished=finished, outputs=[SimpleNamespace(
        index=index, token_ids=list(range(count)), finish_reason=reason, stop_reason=None)])


class Tenant:
    def __init__(self, outputs, error=None, blocked=False):
        self.outputs = outputs
        self.error = error
        self.blocked = blocked
        self.calls = []
        self.entered = asyncio.Event()
        self.exited = False

    async def generate(self, prompt, params, request_id, data_parallel_rank):
        self.calls.append((prompt, vars(params).copy(), request_id, data_parallel_rank))
        self.entered.set()
        try:
            for output in self.outputs:
                await asyncio.sleep(0)
                yield output
            if self.blocked:
                await asyncio.Event().wait()
            if self.error:
                raise self.error
        finally:
            self.exited = True


def prepare(observer, request_id='request'):
    arrived = time.monotonic()
    assert observer.register_arrival(request_id, 0, 8, arrived)
    return arrived


def events(observer, kind):
    return [event for event in observer.report()['events'] if event['kind'] == kind]


def test_stream_hook_preserves_work_and_records_coalesced_cumulative_receipts():
    async def scenario():
        observer = recorder()
        arrived = prepare(observer)
        outputs = [packet(0), packet(1), packet(1), packet(4, finished=True, reason='length')]
        recorded, plain = Tenant(outputs), Tenant(outputs)
        result = await stream(recorded, 'request', list(range(8)), 4, 0, 0, 'pilot', arrived, receipt_observer=observer)
        baseline = await stream(plain, 'request', list(range(8)), 4, 0, 0, 'pilot', arrived)
        assert recorded.calls == plain.calls
        for name in ('token_sha256', 'first_tokens', 'output_tokens', 'prompt_tokens', 'rank_initial'):
            assert result[name] == baseline[name]
        receipts = events(observer, 'output_receipt')
        assert [event['count'] for event in receipts] == [0, 1, 1, 4]
        assert [event['previous_count'] for event in receipts] == [0, 0, 1, 1]
        assert all(event['model_output_observed_at'] is None for event in receipts)
        assert all('token_ids' not in event and 'max_tokens' not in event for event in receipts)
        assert result['first_token_at'] == receipts[1]['received_at']
        assert events(observer, 'stream_terminal')[0]['terminal'] == 'returned'
        assert observer.report()['requests'][0]['count'] == 4
        return await observer.stop()
    report = asyncio.run(scenario())
    assert report['retention_lossless'] and not report['prediction_gate_passed']


@pytest.mark.parametrize('reason', ['stop', 'length', 'abort'])
def test_early_terminal_output_is_retained_even_when_legacy_length_check_fails(reason):
    async def scenario():
        observer = recorder()
        arrived = prepare(observer)
        tenant = Tenant([packet(2, finished=True, reason=reason)])
        with pytest.raises(RuntimeError, match='exact output-length'):
            await stream(tenant, 'request', [0] * 8, 4, 0, 0, 'pilot', arrived, receipt_observer=observer)
        assert events(observer, 'output_receipt')[0]['finish_reason'] == reason
        assert events(observer, 'stream_terminal')[0]['terminal'] == 'error'
        assert observer.report()['requests'][0]['count'] == 2
    asyncio.run(scenario())


def test_generator_failure_and_empty_output_keep_terminal_rows_and_original_error():
    async def scenario():
        for outputs in ([], [packet(1)]):
            observer = recorder()
            arrived = prepare(observer)
            failure = RuntimeError('engine failure')
            with pytest.raises(RuntimeError) as caught:
                await stream(Tenant(outputs, failure), 'request', [0] * 8, 4, 0, 0, 'pilot', arrived,
                             receipt_observer=observer)
            assert caught.value is failure
            assert events(observer, 'stream_terminal')[0]['terminal'] == 'error'
            assert observer.report()['requests'][0]['count'] == len(outputs)
    asyncio.run(scenario())


def test_cancellation_preserves_partial_receipts_without_cancelling_unrelated_requests():
    async def scenario():
        observer = recorder()
        arrived = prepare(observer)
        tenant = Tenant([packet(1)], blocked=True)
        task = asyncio.create_task(stream(tenant, 'request', [0] * 8, 4, 0, 0, 'pilot', arrived,
                                          receipt_observer=observer))
        await tenant.entered.wait()
        while not events(observer, 'output_receipt'):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert tenant.exited
        assert events(observer, 'stream_terminal')[0]['terminal'] == 'cancelled'
        report = await observer.stop()
        assert report['requests'][0]['count'] == 1 and report['task_ended']
    asyncio.run(scenario())


@pytest.mark.parametrize('mutation', ['identity', 'index', 'count_decreased', 'missing_finish', 'multiple', 'bad_reason'])
def test_invalid_receipt_does_not_fabricate_labels_or_change_stream_result(mutation):
    async def scenario():
        observer = recorder()
        arrived = prepare(observer)
        output = packet(4, finished=True, reason='length')
        outputs = [packet(1), output]
        if mutation == 'identity':
            output.request_id = 'other'
        elif mutation == 'index':
            output.outputs[0].index = 1
        elif mutation == 'count_decreased':
            outputs = [packet(2), packet(1), output]
        elif mutation == 'missing_finish':
            del output.outputs[0].finish_reason
        elif mutation == 'multiple':
            output.outputs.append(output.outputs[0])
        else:
            output.outputs[0].finish_reason = 'unrecognized'
        result = await stream(Tenant(outputs), 'request', [0] * 8, 4, 0, 0, 'pilot', arrived,
                              receipt_observer=observer)
        assert result['output_tokens'] == 4
        assert observer.report()['counts']['invalid'] > 0
        assert observer.report()['requests'][0]['receipt_status'] == 'unavailable'
    asyncio.run(scenario())


def test_unregistered_arrival_and_delta_mode_are_not_silently_assumed_supported():
    observer = recorder()
    now = time.monotonic()
    assert observer.begin_request('missing', 0, 0, 8, now, now, SamplingParams()) is None
    arrived = prepare(observer)
    params = SamplingParams()
    params.output_kind = OutputKind.DELTA
    assert observer.begin_request('request', 0, 0, 8, arrived, time.monotonic(), params) is None
    assert observer.report()['counts']['invalid'] == 2


@pytest.mark.parametrize('field,value', [('period_s', 0), ('deadline_s', float('inf')), ('max_events', True),
    ('max_bytes', 0), ('max_requests', 4097), ('clock_id', 'wrong-clock'), ('case_id', '')])
def test_invalid_protocol_rejected_before_collection(field, value):
    with pytest.raises(ValueError):
        recorder(**{field: value})


def test_memory_bounds_and_sequence_gaps_preserve_loss_counters_and_final_state():
    observer = recorder(max_events=3)
    arrived = prepare(observer)
    request = observer.begin_request('request', 0, 0, 8, arrived, time.monotonic(), SamplingParams())
    observer.receive(request, packet(1), time.monotonic())
    observer.receive(request, packet(4, finished=True, reason='length'), time.monotonic())
    observer.finish_request(request)
    report = asyncio.run(observer.stop())
    assert report['counts']['retained'] == 3 and report['counts']['dropped'] == 3
    assert report['requests'][0]['count'] == 4
    assert not report['retention_lossless'] and report['stopped']
    report['requests'][0]['count'] = 0
    assert observer.report()['requests'][0]['count'] == 4


def test_byte_bound_and_serialization_failures_are_explicit():
    observer = recorder(max_bytes=512)
    assert not observer.emit('too_large', content='x' * 1000)
    assert not observer.emit('invalid', value=float('nan'))
    assert observer.emit('small', value=1)
    report = observer.report()
    assert report['counts']['dropped'] == 2 and report['counts']['invalid'] == 1
    assert report['events'][0]['sequence'] == 3 and report['counts']['bytes_retained'] <= 512


def test_queued_arrivals_remain_in_cohort_without_inventing_submission_or_outputs():
    observer = recorder(max_requests=1)
    prepare(observer)
    assert not observer.register_arrival('overflow', 1, 9, time.monotonic())
    state = observer.cohort()[0]
    assert state['submitted_at'] is None and state['count'] == 0 and state['last_received_at'] is None
    report = asyncio.run(observer.stop())
    assert report['events'][-1]['open_requests'] == ['request']
    assert report['counts']['invalid'] == 1


class Controller:
    num_ranks = tensor_parallel_size = 2

    def __init__(self, delay=0):
        self.delay = delay
        self.active = self.maximum = 0
        self.entered = asyncio.Event()
        self.exited = asyncio.Event()

    async def prediction_observations(self, root, timeout):
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        self.entered.set()
        try:
            await asyncio.sleep(self.delay)
            return {'scheduler': {'status': 'observed'}, 'execution': {'status': 'observed'},
                    'collected_at': time.monotonic()}
        finally:
            self.active -= 1
            self.exited.set()


async def healthy_links(*args):
    await asyncio.sleep(0)
    return {'status': 'observed', 'records': [], 'collected_at': time.monotonic()}


def test_sampler_preserves_partial_collection_on_deadline_and_drains_children(monkeypatch):
    monkeypatch.setattr(collection, 'collect_batch_links', healthy_links)
    async def scenario():
        observer = recorder(deadline_s=0.01)
        controller = Controller(delay=1)
        prepare(observer)
        await observer.collect_round(controller, '/unused', 0, time.monotonic())
        sample = events(observer, 'sample')[0]
        assert sample['error_type'] == 'TimeoutError' and sample['transport'] is None
        assert sample['batch_links']['status'] == 'observed'
        assert sample['cohort_at_completion'][0]['submitted_at'] is None
        assert controller.active == 0 and controller.exited.is_set()
        assert observer.report()['counts']['incomplete_rounds'] == 1
    asyncio.run(scenario())


def test_cancelled_round_retains_already_received_sources(monkeypatch):
    async def stuck(*args):
        await asyncio.Event().wait()
    monkeypatch.setattr(collection, 'collect_batch_links', stuck)
    async def scenario():
        observer, controller = recorder(deadline_s=10), Controller()
        await observer.start(controller, '/unused')
        await controller.exited.wait()
        report = await observer.stop()
        sample = next(event for event in report['events'] if event['kind'] == 'sample')
        assert sample['cancelled'] and sample['error_type'] == 'CancelledError'
        assert sample['transport']['scheduler']['status'] == 'observed'
        assert sample['batch_links'] is None and report['task_ended']
    asyncio.run(scenario())


def test_cadence_skips_overdue_ticks_instead_of_catchup_or_overlapping_requests(monkeypatch):
    monkeypatch.setattr(collection, 'collect_batch_links', healthy_links)
    async def scenario():
        observer, controller = recorder(period_s=0.005, deadline_s=0.1), Controller(delay=0.02)
        await observer.start(controller, '/unused')
        while observer.rounds < 2:
            await asyncio.sleep(0.001)
        report = await observer.stop()
        assert controller.maximum == 1 and controller.active == 0
        assert report['counts']['missed_ticks'] >= 2
        assert any(event['kind'] == 'missed_sample_ticks' for event in report['events'])
        with pytest.raises(ValueError):
            await observer.start(controller, '/unused')
    asyncio.run(scenario())


def test_cohort_at_completion_only_uses_receipts_already_received(monkeypatch):
    monkeypatch.setattr(collection, 'collect_batch_links', healthy_links)
    async def scenario():
        observer, controller = recorder(), Controller(delay=0.005)
        arrived = prepare(observer)
        request = observer.begin_request('request', 0, 0, 8, arrived, time.monotonic(), SamplingParams())
        task = asyncio.create_task(observer.collect_round(controller, '/unused', 0, time.monotonic()))
        await controller.entered.wait()
        observer.receive(request, packet(1), time.monotonic())
        await task
        observer.receive(request, packet(4, finished=True, reason='length'), time.monotonic())
        sample = events(observer, 'sample')[0]
        assert sample['cohort_at_start'][0]['count'] == 0
        assert sample['cohort_at_completion'][0]['count'] == 1
        assert sample['cohort_at_completion'][0]['last_received_at'] <= sample['as_of']
        assert events(observer, 'output_receipt')[-1]['received_at'] > sample['as_of']
    asyncio.run(scenario())


def test_missing_batch_link_directory_retains_every_expected_endpoint(tmp_path):
    records = asyncio.run(collection.collect_batch_links(tmp_path, 2, 2, 0.1))
    assert records['status'] == 'incomplete' and len(records['records']) == 12
    assert all(record['status'] == 'unavailable' for record in records['records'])


def test_receipts_after_cutoff_do_not_rewrite_censored_state():
    observer = recorder()
    arrived = prepare(observer)
    request = observer.begin_request('request', 0, 0, 8, arrived, time.monotonic(), SamplingParams())
    observer.receive(request, packet(1), time.monotonic())
    stopped = asyncio.run(observer.stop())
    observer.receive(request, packet(4, finished=True, reason='length'), time.monotonic())
    observer.finish_request(request)
    assert observer.report()['requests'] == stopped['requests']
    assert observer.report()['counts']['dropped'] == stopped['counts']['dropped'] + 2
    assert observer.report()['requests'][0]['stream_terminal'] is None


def test_malformed_transport_stays_raw_and_incomplete_without_stopping_sampler(monkeypatch):
    monkeypatch.setattr(collection, 'collect_batch_links', healthy_links)
    class MalformedController(Controller):
        async def prediction_observations(self, root, timeout):
            return {'scheduler': [], 'execution': {'status': 'observed'}}
    async def scenario():
        observer = recorder()
        await observer.collect_round(MalformedController(), '/unused', 0, time.monotonic())
        sample = events(observer, 'sample')[0]
        assert sample['transport']['scheduler'] == [] and not sample['transport_complete']
        assert observer.report()['sampler_error'] is None
    asyncio.run(scenario())


@pytest.mark.parametrize('arrival', [-1, float('inf')])
def test_noncausal_arrival_clock_is_not_registered(arrival):
    observer = recorder()
    assert not observer.register_arrival('request', 0, 8, arrival)
    assert not observer.cohort() and observer.report()['counts']['invalid'] == 1


def test_real_socket_collection_preserves_twelve_identities_and_receipt_times():
    temporary = tempfile.TemporaryDirectory(prefix='xrc-', dir='/tmp')
    root = Path(temporary.name)
    directory = root / 'batch-links'
    directory.mkdir(mode=0o700)
    servers = []
    try:
        for model in (0, 1):
            for rank in (0, 1):
                for lane in (None, 0, 1):
                    identity = dict(model=model, dp_rank=rank, role='engine' if lane is None else 'worker')
                    if lane is not None:
                        identity['tp_rank'] = lane
                    servers.append(live.SnapshotServer(live.endpoint_path(directory, model, rank, lane), identity,
                        'test-receipt-clock', lambda: {'kind': 'causal_batch_link_v1', 'observed_at': time.monotonic(),
                                                       'status': 'observed', 'device_activity': 'unknown'}))
        records = asyncio.run(collection.collect_batch_links(root, 2, 2, 1))
        assert records['status'] == 'observed' and len(records['records']) == 12
        assert all(record['observed_at'] <= record['available_at'] <= records['collected_at']
                   for record in records['records'])
    finally:
        for server in servers:
            server.close()
        assert not list(directory.glob('*.sock'))
        temporary.cleanup()
