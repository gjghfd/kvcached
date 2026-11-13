import asyncio
import copy
import json
from pathlib import Path
import sys
import tempfile
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_component_diagnostics as diagnostic
import analyze_causal_calibration as causal
from test_analyze_causal_calibration import collection_report, recount, startup_fixture


def protocol_fixture():
    return dict(schema=1, limits=dict(diagnostic.LIMITS), period_s=0.1, deadline_s=0.08,
                component_counts=dict(scheduler=4, execution=12, batch_links=12), expected_components_per_round=28,
                partial_results_prediction_eligible=False, prediction_model_used=False)


def add_diagnostics(report):
    report['protocol']['max_bytes'] = 536870912
    report['protocol']['max_events'] = 1000000
    for sample in report['events']:
        if sample['kind'] != 'sample':
            continue
        sample.update(deadline_at=sample['started_at'] + 0.08, completed_at=sample['as_of'],
                      prediction_status='unavailable_not_estimated')
        components = []
        for source, identity in diagnostic.expected_components().values():
            started, finished = sample['started_at'], sample['as_of'] - 0.0001
            phases = [dict(phase=phase, at=started + (finished - started) * (index + 1) / 11)
                      for index, phase in enumerate(diagnostic.PHASES[source])]
            components.append(dict(source=source, identity=dict(identity), started_at=started, finished_at=finished,
                status='completed', source_status='observed', error_type=None, phases=phases, dropped_phases=0,
                completed_before_deadline=True, received_before_deadline=True, observed_by_deadline=True,
                wire_bytes=None if source == 'scheduler' else 256,
                wire_bytes_status='not_observed' if source == 'scheduler' else 'received_socket_payload',
                result_retention='not_requested'))
        sample['component_diagnostics'] = dict(schema=1, diagnostic_only=True, prediction_input_eligible=False,
            clock_id=report['protocol']['clock_id'], deadline_at=sample['deadline_at'], max_components=64,
            max_phases=16, max_partial_bytes=65536, dropped_components=0, partial_bytes_retained=0,
            partial_results_omitted=0, components=components)
    return recount(report)


def report_fixture():
    return add_diagnostics(collection_report())


def first_sample(report):
    return next(event for event in report['events'] if event['kind'] == 'sample')


def fail_round(sample, source='scheduler', cancelled=False):
    sample.update(transport_complete=False, error_type='CancelledError' if cancelled else 'TimeoutError',
                  cancelled=cancelled)
    component = next(row for row in sample['component_diagnostics']['components'] if row['source'] == source)
    component.update(status='cancelled' if cancelled else 'timed_out', source_status=None,
                     error_type='CancelledError' if cancelled else 'TimeoutError',
                     received_before_deadline=False, observed_by_deadline=False)
    return component


def test_complete_rounds_are_diagnostic_only_and_all28_are_checked():
    report = report_fixture()
    before = copy.deepcopy(report)
    result = diagnostic.audit_collection(report, protocol_fixture())
    assert result['status'] == 'passed' and result['sampled_rounds'] == 3
    assert all(len(row['components']) == 28 for row in result['rounds'])
    assert not result['scientific_gate_passed'] and not result['producer_root_cause_proven']
    assert report == before


@pytest.mark.parametrize('source', ['scheduler', 'execution', 'batch_links'])
@pytest.mark.parametrize('cancelled', [False, True])
def test_failed_rounds_and_round_start_overlap_remain_in_report(source, cancelled):
    report = report_fixture()
    fail_round(first_sample(report), source, cancelled)
    result = diagnostic.audit_collection(recount(report), protocol_fixture())
    assert result['status'] == 'passed' and result['incomplete_rounds'] == 1
    assert len(result['rounds']) == 3 and len(result['rounds'][0]['components']) == 28
    assert any(not row['observed_by_deadline'] for row in result['rounds'][0]['components'])
    assert any(row['request_id'] == 'queued' and row['first_receipt_pending']
               for row in result['rounds'][0]['cohort_at_round_start'])


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'extra', 'identity_bool', 'identity_lane', 'clock',
    'round_deadline', 'component_deadline', 'eligible', 'partial_overflow', 'partial_in_success', 'partial_size',
    'phase_missing', 'phase_order', 'phase_time', 'phase_drop', 'component_drop', 'round_count', 'incomplete_count',
    'omission_count', 'wire', 'late_complete', 'status', 'cohort_duplicate', 'phase_unknown', 'component_null'])
def test_corrupted_diagnostics_never_pass_and_other_rounds_are_retained(damage):
    report = report_fixture()
    sample = first_sample(report)
    detail = sample['component_diagnostics']
    component = detail['components'][0]
    if damage == 'missing':
        detail['components'].pop()
    elif damage == 'duplicate':
        detail['components'][-1] = copy.deepcopy(component)
    elif damage == 'extra':
        detail['components'].append(copy.deepcopy(component))
    elif damage == 'identity_bool':
        component['identity']['model'] = False
    elif damage == 'identity_lane':
        component['identity']['tp_rank'] = 0
    elif damage == 'clock':
        detail['clock_id'] = 'different-clock'
    elif damage == 'round_deadline':
        sample['deadline_at'] += 0.1
    elif damage == 'component_deadline':
        component['observed_by_deadline'] = False
    elif damage == 'eligible':
        detail['prediction_input_eligible'] = True
    elif damage in ('partial_overflow', 'partial_in_success', 'partial_size'):
        if damage != 'partial_in_success':
            fail_round(sample, 'batch_links')
        partial = dict(available_at=component['finished_at'], payload='x' * (65536 if damage == 'partial_overflow' else 10))
        size = len(json.dumps(partial, separators=(',', ':')).encode())
        component.update(partial_result=partial, result_retention='retained_diagnostic_only', result_json_bytes=size)
        detail['partial_bytes_retained'] = size
        if damage == 'partial_size':
            component['result_json_bytes'] += 1
    elif damage == 'phase_missing':
        component['phases'].pop()
    elif damage == 'phase_order':
        component['phases'].reverse()
    elif damage == 'phase_time':
        component['phases'][0]['at'] = component['finished_at'] + 1
    elif damage == 'phase_drop':
        component['dropped_phases'] = 1
    elif damage == 'component_drop':
        detail['dropped_components'] = 1
    elif damage == 'omission_count':
        detail['partial_results_omitted'] = 1
    elif damage == 'wire':
        component['wire_bytes'] = -1
    elif damage == 'late_complete':
        sample['completed_at'] = sample['deadline_at'] + 1
    elif damage == 'status':
        component['status'] = 'invented'
    elif damage == 'cohort_duplicate':
        sample['cohort_at_start'].append(copy.deepcopy(sample['cohort_at_start'][0]))
    elif damage == 'phase_unknown':
        component['phases'][0]['phase'] = 'invented'
    elif damage == 'component_null':
        detail['components'][0] = None
    recount(report)
    if damage == 'round_count':
        report['counts']['rounds'] += 1
    elif damage == 'incomplete_count':
        report['counts']['incomplete_rounds'] += 1
    result = diagnostic.audit_collection(report, protocol_fixture())
    assert result['status'] == 'not_passed'
    assert len(result['rounds']) == 3
    assert not result['scientific_gate_passed']


def test_partial_results_and_omissions_are_counted_without_prediction_recovery():
    report = report_fixture()
    sample = first_sample(report)
    fail_round(sample, 'batch_links')
    component = sample['component_diagnostics']['components'][0]
    partial = dict(component['identity'], available_at=component['finished_at'], status='observed')
    size = len(json.dumps(partial, separators=(',', ':')).encode())
    component.update(partial_result=partial, result_retention='retained_diagnostic_only', result_json_bytes=size)
    sample['component_diagnostics']['components'][1]['result_retention'] = 'omitted_byte_limit'
    sample['component_diagnostics']['components'][1]['result_json_bytes'] = 65537
    sample['component_diagnostics'].update(partial_bytes_retained=size, partial_results_omitted=1)
    result = diagnostic.audit_collection(recount(report), protocol_fixture())
    assert result['status'] == 'passed' and result['incomplete_rounds'] == 1
    assert sum(row['partial_bytes'] for row in result['rounds'][0]['components']) == size
    assert sum(row['partial_omitted'] for row in result['rounds'][0]['components']) == 1
    assert not result['rounds'][0]['transport_complete']


def test_post_close_rejections_differ_from_open_interval_loss():
    report = report_fixture()
    report['counts']['post_fixture'] = 4
    result = diagnostic.audit_collection(recount(report), protocol_fixture())
    assert result['status'] == 'passed' and result['retention']['post_close_rejected'] == 4
    report['events'][-1]['sequence'] += 1
    result = diagnostic.audit_collection(recount(report), protocol_fixture())
    assert result['status'] == 'not_passed' and result['retention']['collection_dropped'] == 1


def binding_fixture(root):
    startup = startup_fixture(root, 571)
    (root / 'runtime').mkdir()
    (root / 'runtime/observation_diagnostics.py').write_text('synthetic_fixture = True\n')
    protocol = dict(protocol_fixture(), startup_expected_count=571,
                    startup_contract_sha256=startup['startup_test_contract_sha256'],
                    source_sha256={'runtime/observation_diagnostics.py': causal.digest(root / 'runtime/observation_diagnostics.py')})
    causal.write(root / 'inputs/component-diagnostic-protocol.json', protocol)
    deployment = dict(startup, component_diagnostic_protocol_sha256=causal.digest(root / 'inputs/component-diagnostic-protocol.json'))
    causal.write(root / 'inputs/deployment-protocol.json', deployment)
    causal.write(root / 'inputs/deployment-manifest.json', dict(protocol_sha256=causal.digest(root / 'inputs/deployment-protocol.json')))
    payload = {str(path.relative_to(root)): causal.digest(path) for path in root.rglob('*')
               if path.is_file() and path.parts[-1] != 'startup-tests.xml'}
    return dict(component_diagnostic_protocol_sha256=deployment['component_diagnostic_protocol_sha256'],
                deployment_manifest_sha256=causal.digest(root / 'inputs/deployment-manifest.json'), compute_payload_sha256=payload)


@pytest.mark.parametrize('damage', [None, 'protocol_hash', 'runtime_source', 'startup_identity', 'deployment_hash', 'missing_module'])
def test_component_protocol_binds_source_and_exact_startup(tmp_path, damage):
    protocol = binding_fixture(tmp_path)
    if damage == 'protocol_hash':
        protocol['component_diagnostic_protocol_sha256'] = '0' * 64
    elif damage == 'runtime_source':
        (tmp_path / 'runtime/observation_diagnostics.py').write_text('changed = True\n')
    elif damage == 'startup_identity':
        filename = tmp_path / 'startup-tests.xml'
        filename.write_text(filename.read_text().replace('test_identity_0', 'test_wrong_identity'))
    elif damage == 'deployment_hash':
        protocol['deployment_manifest_sha256'] = '0' * 64
    elif damage == 'missing_module':
        del protocol['compute_payload_sha256']['runtime/observation_diagnostics.py']
    if damage is None:
        assert diagnostic.bound_protocol(tmp_path, protocol)['startup_expected_count'] == 571
    else:
        with pytest.raises(ValueError):
            diagnostic.bound_protocol(tmp_path, protocol)


def test_job_wrapper_preserves_missing_cases_and_never_promotes_prediction(tmp_path, monkeypatch):
    root = tmp_path / 'job'
    (root / 'inputs').mkdir(parents=True)
    plan = dict(cases=[dict(case=f'case{index}') for index in range(4)])
    causal.write(root / 'inputs/causal-training-plan-20261110-r0.json', plan)
    for case in plan['cases'][:3]:
        directory = root / 'out/causal-block/collect/out/calibration/causal' / case['case']
        directory.mkdir(parents=True)
        causal.write(directory / 'collection.json', report_fixture())
    calls = []

    def baseline(jobs, output, protocol):
        calls.append(jobs)
        output.mkdir()
        result = dict(run_artifacts_passed=True, complete_overhead_and_collection=False)
        causal.write(output / 'report.json', result)
        return result

    monkeypatch.setattr(diagnostic, 'bound_protocol', lambda root, protocol: protocol_fixture())
    monkeypatch.setattr(causal, 'analyze_jobs', baseline)
    output = tmp_path / 'analysis'
    result = diagnostic.analyze_jobs([(0, root)], output, {})
    assert len(calls) == 1 and len(result['cases']) == 4
    assert [row['audit']['status'] for row in result['cases']] == ['passed'] * 3 + ['not_passed']
    assert not result['diagnostic_artifacts_passed'] and not result['prediction_gate_passed']
    assert not result['independent_validation'] and not result['producer_root_cause_proven']
    assert (output / 'report.json').exists() and (output / 'component-report.json').exists()
    with pytest.raises(ValueError, match='already_exists'):
        diagnostic.analyze_jobs([(0, root)], output, {})


@pytest.mark.parametrize('damage', ['schema_bool', 'prediction', 'deadline', 'period', 'limits', 'topology'])
def test_changed_diagnostic_protocol_is_rejected(damage):
    protocol = protocol_fixture()
    if damage == 'schema_bool':
        protocol['schema'] = True
    elif damage == 'prediction':
        protocol['partial_results_prediction_eligible'] = True
    elif damage == 'deadline':
        protocol['deadline_s'] = 0.1
    elif damage == 'period':
        protocol['period_s'] = 0.2
    elif damage == 'limits':
        protocol['limits']['partial_json_bytes'] = 131072
    elif damage == 'topology':
        protocol['component_counts']['execution'] = 8
    assert diagnostic.audit_collection(report_fixture(), protocol)['status'] == 'not_passed'


@pytest.mark.parametrize('offset', [0.0, 0.00001])
def test_component_deadline_boundary_remains_exact(offset):
    report = report_fixture()
    sample = first_sample(report)
    fail_round(sample, 'batch_links')
    component = sample['component_diagnostics']['components'][0]
    component['finished_at'] = sample['deadline_at'] + offset
    component['completed_before_deadline'] = offset == 0
    component['observed_by_deadline'] = offset == 0
    sample['as_of'] = sample['completed_at'] = component['finished_at'] + 0.001
    sample['recorded_at'] = sample['as_of'] + 0.001
    result = diagnostic.audit_collection(recount(report), protocol_fixture())
    assert result['status'] == 'passed'
    assert result['rounds'][0]['components'][0]['observed_by_deadline'] == (offset == 0)
    assert not result['rounds'][0]['transport_complete']


@pytest.mark.parametrize('stalled', [False, True])
def test_actual_collector_and24_unix_endpoints_match_analysis(monkeypatch, stalled):
    import causal_collection as collection
    import live_observation as live
    from controller import PlacementController

    clock = 'component-analysis-cpu-clock'
    monkeypatch.setattr(collection, 'host_clock_id', lambda: clock)
    monkeypatch.setattr(live, 'host_clock_id', lambda: clock)

    async def scenario(root):
        servers, handlers = [], set()
        (root / 'batch-links').mkdir(mode=0o700)
        controller = PlacementController([None, None], 2, [40, 40])
        controller.tensor_parallel_size = 2

        async def call(model, rank, method):
            return dict(rank=rank, clock_id=clock, observed_at=time.monotonic())

        controller.call = call
        try:
            for source, identity in diagnostic.expected_components().values():
                if source == 'scheduler':
                    continue
                directory = root / 'batch-links' if source == 'batch_links' else root
                endpoint = live.endpoint_path(directory, identity['model'], identity['dp_rank'], identity.get('tp_rank'))

                async def serve(reader, writer, identity=identity, source=source):
                    task = asyncio.current_task()
                    handlers.add(task)
                    try:
                        slow = stalled and source == 'execution' and identity == dict(model=0, dp_rank=0, role='engine')
                        if not slow:
                            payload = dict(identity, schema=1, status='observed', clock_id=clock,
                                           observed_at=time.monotonic(), kind='causal_batch_link_v1')
                            writer.write(json.dumps(payload).encode() + b'\n')
                            await writer.drain()
                        await reader.read()
                    finally:
                        writer.close()
                        await writer.wait_closed()
                        handlers.discard(task)

                servers.append(await asyncio.start_unix_server(serve, path=str(endpoint)))
            observer = collection.CausalCollection(case_id='cpu', clock_id=clock, period_s=0.1, deadline_s=0.08,
                max_events=1000000, max_bytes=536870912, max_requests=64)
            origin = time.monotonic()
            observer.emit('collection_start', origin=origin, protocol=observer.protocol)
            await observer.collect_round(controller, root, 0, origin)
            cutoff = time.monotonic()
            observer.mark_trace_cutoff(cutoff, cutoff)
            report = await observer.stop()
            result = diagnostic.audit_collection(report, protocol_fixture())
            assert result['status'] == 'passed', result
            assert len(result['rounds'][0]['components']) == 28
            assert result['rounds'][0]['transport_complete'] is not stalled
            if stalled:
                assert result['incomplete_rounds'] == 1
                assert sum(row['partial_bytes'] for row in result['rounds'][0]['components']) > 0
        finally:
            for server in servers:
                server.close()
                await server.wait_closed()
            if handlers:
                await asyncio.wait_for(asyncio.gather(*handlers, return_exceptions=True), 1)

    with tempfile.TemporaryDirectory(prefix='cadiag-', dir='/tmp') as directory:
        asyncio.run(scenario(Path(directory)))
