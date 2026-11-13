import asyncio
import json
from pathlib import Path
import sys
import tempfile
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))

import causal_collection as collection
from controller import PlacementController
import live_observation as live
import observation_diagnostics as diagnostic


def recorder():
    return collection.CausalCollection(case_id='diagnostic', clock_id='test-clock', period_s=0.1,
                                       deadline_s=0.08, max_events=200, max_bytes=1000000, max_requests=16)


@pytest.fixture
def environment(monkeypatch):
    monkeypatch.setattr(live, 'host_clock_id', lambda: 'test-clock')
    monkeypatch.setattr(collection, 'host_clock_id', lambda: 'test-clock')
    with tempfile.TemporaryDirectory(prefix='xcd-', dir='/tmp') as directory:
        root = Path(directory)
        (root / 'batch-links').mkdir(mode=0o700)
        yield root


def install_sources(monkeypatch, stalled_source, entered):
    controller = PlacementController([None, None], 2, [40, 40])
    controller.tensor_parallel_size = 2

    async def wait_if_stalled(source, model, rank, lane=None):
        if source == stalled_source and model == rank == 0 and lane in (None, 0):
            entered.set()
            await asyncio.Event().wait()

    async def call(model, rank, method):
        assert method == 'placement_snapshot'
        await wait_if_stalled('scheduler', model, rank)
        return dict(rank=rank, clock_id='test-clock', observed_at=time.monotonic(), requests=[])

    async def endpoint(path, identity, clock_id, timeout_s):
        source = 'batch_links' if path.parent.name == 'batch-links' else identity['role']
        if source != 'batch_links' or identity['role'] == 'engine':
            await wait_if_stalled(source, identity['model'], identity['dp_rank'], identity.get('tp_rank'))
        diagnostic.mark_phase('test_endpoint_return', wire_bytes=100)
        return dict(identity, kind='causal_batch_link_v1', status='observed', clock_id=clock_id,
                    observed_at=time.monotonic(), available_at=time.monotonic())

    controller.call = call
    monkeypatch.setattr(live, 'read_endpoint', endpoint)
    monkeypatch.setattr(collection, 'read_endpoint', endpoint)
    return controller


@pytest.mark.parametrize('stalled_source', ['scheduler', 'engine', 'worker', 'batch_links'])
def test_failed_round_preserves_each_component_without_recovering_prediction(environment, monkeypatch, stalled_source):
    async def scenario():
        entered = asyncio.Event()
        controller = install_sources(monkeypatch, stalled_source, entered)
        observer = recorder()
        await observer.collect_round(controller, environment, 0, time.monotonic())
        sample = observer.report()['events'][0]
        assert entered.is_set() and sample['error_type'] == 'TimeoutError'
        assert not sample['transport_complete'] and sample['prediction_status'] == 'unavailable_not_estimated'
        detail = sample['component_diagnostics']
        assert detail['diagnostic_only'] and not detail['prediction_input_eligible']
        assert detail['clock_id'] == 'test-clock' and len(detail['components']) == 28
        assert detail['dropped_components'] == detail['partial_results_omitted'] == 0
        cancelled = [item for item in detail['components'] if item['status'] == 'cancelled']
        completed = [item for item in detail['components'] if item['status'] == 'completed']
        assert len(cancelled) == 1 and len(completed) == 27
        assert cancelled[0]['identity']['model'] == cancelled[0]['identity']['dp_rank'] == 0
        assert all(item['observed_by_deadline'] for item in completed)
        assert all(item['result_retention'] == 'retained_diagnostic_only' for item in completed)
        assert all('partial_result' not in item for item in cancelled)
        assert detail['partial_bytes_retained'] <= detail['max_partial_bytes']
        if stalled_source != 'batch_links':
            assert sample['transport'] is None and sample['batch_links']['status'] == 'observed'
        else:
            assert sample['transport']['scheduler']['status'] == 'observed' and sample['batch_links'] is None
        assert diagnostic.ACTIVE_DIAGNOSTICS.get() is None and diagnostic.ACTIVE_COMPONENT.get() is None
        assert asyncio.all_tasks() == {asyncio.current_task()}
    asyncio.run(scenario())


def test_external_cancellation_drains_nested_groups_and_keeps_partial_diagnostics(environment, monkeypatch):
    async def scenario():
        warnings = []
        asyncio.get_running_loop().set_exception_handler(lambda loop, context: warnings.append(context))
        entered = asyncio.Event()
        controller = install_sources(monkeypatch, 'scheduler', entered)
        observer = recorder()
        task = asyncio.create_task(observer.collect_round(controller, environment, 3, time.monotonic()))
        await entered.wait()
        await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        sample = observer.report()['events'][0]
        assert sample['cancelled'] and sample['error_type'] == 'CancelledError'
        assert len(sample['component_diagnostics']['components']) == 28
        assert not sample['transport_complete']
        await asyncio.sleep(0)
        assert not warnings and asyncio.all_tasks() == {asyncio.current_task()}
    asyncio.run(scenario())


def test_successful_round_does_not_duplicate_results_in_diagnostics(environment, monkeypatch):
    async def scenario():
        controller = install_sources(monkeypatch, None, asyncio.Event())
        observer = recorder()
        await observer.collect_round(controller, environment, 0, time.monotonic())
        sample = observer.report()['events'][0]
        assert sample['transport_complete']
        detail = sample['component_diagnostics']
        assert len(detail['components']) == 28 and detail['partial_bytes_retained'] == 0
        assert all(item['result_retention'] == 'not_requested' and 'partial_result' not in item
                   for item in detail['components'])
        assert observer.bytes_retained <= observer.protocol['max_bytes']
    asyncio.run(scenario())


@pytest.mark.parametrize('finished,available,on_time', [(2.0, 1.9, True), (2.001, 1.9, False), (2.001, 2.001, False)])
def test_exact_deadline_and_late_completion_are_distinct_diagnostic_evidence(finished, available, on_time):
    clock = [1.0]
    detail = diagnostic.ComponentDiagnostics(clock_id='clock', deadline_at=2, clock=lambda: clock[0])
    component = detail.begin('scheduler', dict(model=0, dp_rank=0))
    clock[0] = finished
    component.finish(result=dict(status='observed', available_at=available))
    record = detail.snapshot(include_partial_results=True)
    assert record['components'][0]['observed_by_deadline'] is on_time
    assert not record['prediction_input_eligible']


def test_component_phase_and_partial_byte_limits_are_explicit():
    detail = diagnostic.ComponentDiagnostics(clock_id='clock', deadline_at=time.monotonic() + 1,
                                             max_components=2, max_phases=2, max_partial_bytes=20)
    component = detail.begin('execution', dict(model=0, dp_rank=0, role='worker', tp_rank=0))
    for phase in ['connect_started', 'connected', 'read_started']:
        component.mark(phase)
    component.finish(result=dict(status='observed', available_at=time.monotonic(), large='x' * 100))
    detail.begin('scheduler', dict(model=0, dp_rank=0)).finish(result={'status': 'unavailable'})
    assert detail.begin('scheduler', dict(model=1, dp_rank=0)) is None
    record = detail.snapshot(include_partial_results=True)
    assert record['dropped_components'] == 1 and record['partial_results_omitted'] == 2
    assert record['partial_bytes_retained'] == 0
    assert len(record['components'][0]['phases']) == 2 and record['components'][0]['dropped_phases'] == 1
    assert all(item['result_retention'] == 'omitted_byte_limit' for item in record['components'])


def test_invalid_partial_json_is_recorded_not_fabricated():
    detail = diagnostic.ComponentDiagnostics(clock_id='clock', deadline_at=2, clock=lambda: 1)
    detail.begin('scheduler', dict(model=0, dp_rank=0)).finish(result={'value': float('nan')})
    record = detail.snapshot(include_partial_results=True)
    assert record['partial_results_omitted'] == 1
    assert record['components'][0]['result_retention'] == 'omitted_invalid_json'
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize('field,value', [('max_components', 65), ('max_phases', 17), ('max_partial_bytes', 65537),
                                       ('max_components', True), ('clock_id', ''), ('deadline_at', float('inf'))])
def test_invalid_diagnostic_bounds_fail_before_tasks(field, value):
    values = dict(clock_id='clock', deadline_at=2)
    values[field] = value
    with pytest.raises(ValueError, match='bounded'):
        diagnostic.ComponentDiagnostics(**values)


def test_group_failure_drains_sibling_cleanup_and_retrieves_exceptions():
    async def scenario():
        entered, cleaned = asyncio.Event(), asyncio.Event()
        async def blocked():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0.001)
                cleaned.set()
                raise ValueError('cleanup failure retained by drain')
        async def failing():
            await entered.wait()
            raise RuntimeError('primary failure')
        with pytest.raises(RuntimeError, match='primary failure'):
            await diagnostic.gather_drained(blocked(), failing())
        assert cleaned.is_set() and asyncio.all_tasks() == {asyncio.current_task()}
    asyncio.run(scenario())


def test_round_context_is_task_local():
    async def scenario():
        async def collect(label):
            detail = diagnostic.ComponentDiagnostics(clock_id=label, deadline_at=time.monotonic() + 1)
            token = diagnostic.ACTIVE_DIAGNOSTICS.set(detail)
            async def operation():
                await asyncio.sleep(0)
                return dict(status='observed', label=label, available_at=time.monotonic())
            try:
                await diagnostic.observe_component('scheduler', dict(model=0, dp_rank=0), operation)
            finally:
                diagnostic.ACTIVE_DIAGNOSTICS.reset(token)
            return detail.snapshot(include_partial_results=True)
        reports = await asyncio.gather(collect('first'), collect('second'))
        assert [record['components'][0]['partial_result']['label'] for record in reports] == ['first', 'second']
        assert diagnostic.ACTIVE_DIAGNOSTICS.get() is None and diagnostic.ACTIVE_COMPONENT.get() is None
    asyncio.run(scenario())


def test_late_event_loop_result_never_marks_round_complete(environment, monkeypatch):
    async def scenario():
        async def late(root, timeout):
            time.sleep(0.02)
            return dict(scheduler={'status': 'observed'}, execution={'status': 'observed'})
        async def links(*args):
            return {'status': 'observed'}
        controller = PlacementController([None, None], 2, [40, 40])
        controller.prediction_observations = late
        monkeypatch.setattr(collection, 'collect_batch_links', links)
        observer = recorder()
        observer.protocol['deadline_s'] = 0.005
        await observer.collect_round(controller, environment, 0, time.monotonic())
        sample = observer.report()['events'][0]
        assert sample['error_type'] in ('TimeoutError', 'DeadlineExceeded')
        assert sample['completed_at'] > sample['deadline_at'] and not sample['transport_complete']
    asyncio.run(scenario())


@pytest.mark.parametrize('cancel', [False, True])
def test_real_socket_phase_bytes_and_cancellation_cleanup(environment, cancel):
    async def scenario():
        identity = dict(model=0, dp_rank=0, role='worker', tp_rank=0)
        path = live.endpoint_path(environment, 0, 0, 0)
        entered, closed = asyncio.Event(), asyncio.Event()
        payload = json.dumps(dict(identity, schema=1, status='observed', clock_id='test-clock',
                                  observed_at=time.monotonic())).encode() + b'\n'
        async def serve(reader, writer):
            try:
                entered.set()
                if not cancel:
                    writer.write(payload)
                    await writer.drain()
                await reader.read()
            finally:
                writer.close()
                await writer.wait_closed()
                closed.set()
        server = await asyncio.start_unix_server(serve, path=str(path))
        detail = diagnostic.ComponentDiagnostics(clock_id='test-clock', deadline_at=time.monotonic() + 1)
        token = diagnostic.ACTIVE_DIAGNOSTICS.set(detail)
        try:
            task = asyncio.create_task(diagnostic.observe_component('execution', identity,
                lambda: live.read_endpoint(path, identity, 'test-clock', 1)))
            await entered.wait()
            if cancel:
                await asyncio.sleep(0)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                assert (await task)['status'] == 'observed'
            await asyncio.wait_for(closed.wait(), 1)
        finally:
            diagnostic.ACTIVE_DIAGNOSTICS.reset(token)
            server.close()
            await server.wait_closed()
        record = detail.snapshot(include_partial_results=True)['components'][0]
        phases = [item['phase'] for item in record['phases']]
        assert phases[:3] == ['connect_started', 'connected', 'read_started']
        assert phases[-2:] == ['close_started', 'closed']
        assert record['status'] == ('cancelled' if cancel else 'completed')
        assert record['wire_bytes'] == (None if cancel else len(payload))
        assert asyncio.all_tasks() == {asyncio.current_task()}
    asyncio.run(scenario())
