import argparse
from collections import Counter
import json
from pathlib import Path

import analyze_causal_calibration as causal


PHASES = {
    'scheduler': ['rpc_started', 'rpc_returned', 'validated'],
    'execution': ['path_started', 'path_resolved', 'connect_started', 'connected', 'read_started',
                  'read_completed', 'validated', 'close_started', 'closed'],
    'batch_links': ['path_started', 'path_resolved', 'connect_started', 'connected', 'read_started',
                    'read_completed', 'validated', 'close_started', 'closed'],
}
LIMITS = dict(components=64, phases_per_component=16, partial_json_bytes=65536, collection_bytes=536870912)
require = causal.require
number = causal.number


def expected_components():
    result = {}
    for model in (0, 1):
        for rank in (0, 1):
            result[f'scheduler:m{model}:dp{rank}'] = ('scheduler', dict(model=model, dp_rank=rank))
            for source in ('execution', 'batch_links'):
                for lane in (None, 0, 1):
                    role = 'engine' if lane is None else 'worker'
                    identity = dict(model=model, dp_rank=rank, role=role)
                    if lane is not None:
                        identity['tp_rank'] = lane
                    result[f'{source}:m{model}:dp{rank}:{role}:{lane}'] = (source, identity)
    return result


def component_key(record):
    identity = record['identity']
    require(isinstance(identity, dict), 'invalid_component_identity')
    require(all(type(identity.get(name)) is int for name in ('model', 'dp_rank')), 'invalid_integer_identity')
    if 'tp_rank' in identity:
        require(type(identity['tp_rank']) is int, 'invalid_integer_identity')
    matches = [key for key, value in expected_components().items()
               if value == (record['source'], identity)]
    require(len(matches) == 1, 'unexpected_component_identity')
    return matches[0]


def check_protocol(protocol):
    require(type(protocol.get('schema')) is int and protocol['schema'] == 1
            and protocol.get('limits') == LIMITS, 'diagnostic_protocol_limits_differ')
    require(protocol.get('period_s') == 0.1 and protocol.get('deadline_s') == 0.08,
            'diagnostic_protocol_schedule_differs')
    require(protocol.get('component_counts') == dict(scheduler=4, execution=12, batch_links=12)
            and protocol.get('expected_components_per_round') == 28, 'diagnostic_protocol_topology_differs')
    require(protocol.get('partial_results_prediction_eligible') is False
            and protocol.get('prediction_model_used') is False, 'diagnostic_protocol_prediction_enabled')


def audit_component(record, sample):
    key = component_key(record)
    deadline = sample['deadline_at']
    started, finished = record['started_at'], record['finished_at']
    require(number(started) and number(finished) and sample['started_at'] <= started <= finished <= sample['as_of'],
            'invalid_component_time')
    require(record['status'] in ('completed', 'unavailable', 'timed_out', 'cancelled'), 'invalid_component_status')
    require(record.get('source_status') is None or isinstance(record['source_status'], str), 'invalid_source_status')
    require(record.get('error_type') is None or isinstance(record['error_type'], str), 'invalid_component_error')
    for name in ('completed_before_deadline', 'received_before_deadline', 'observed_by_deadline'):
        require(type(record.get(name)) is bool, 'missing_component_deadline_flag')
    require(record['completed_before_deadline'] == (finished <= deadline), 'component_finish_deadline_mismatch')
    observed = (record['status'] == 'completed' and record['completed_before_deadline']
                and record['received_before_deadline'])
    require(record['observed_by_deadline'] == observed, 'component_observation_deadline_mismatch')
    if record['status'] == 'completed':
        require(record['source_status'] == 'observed', 'completed_source_not_observed')
    if record['status'] == 'cancelled':
        require(record['error_type'] == 'CancelledError', 'cancellation_error_mismatch')
    if record['status'] == 'timed_out':
        require(record['error_type'] == 'TimeoutError', 'timeout_error_mismatch')
    require(record.get('dropped_phases') == 0 and type(record.get('dropped_phases')) is int, 'phase_metadata_dropped')
    phases = record['phases']
    require(isinstance(phases, list) and len(phases) <= 16, 'invalid_phase_count')
    previous_index, previous_time = -1, started
    for phase in phases:
        require(phase['phase'] in PHASES[record['source']], 'unknown_phase')
        index = PHASES[record['source']].index(phase['phase'])
        require(index > previous_index and number(phase['at']) and previous_time <= phase['at'] <= finished,
                'invalid_phase_order_or_time')
        previous_index, previous_time = index, phase['at']
    if record['status'] == 'completed':
        require([phase['phase'] for phase in phases] == PHASES[record['source']], 'completed_phase_metadata_missing')
    wire = record.get('wire_bytes')
    require((wire is None and record.get('wire_bytes_status') == 'not_observed') or
            (causal.nonnegative_integer(wire) and record.get('wire_bytes_status') == 'received_socket_payload'),
            'invalid_wire_bytes')
    require((record['source'] == 'scheduler' and wire is None) or
            (record['source'] != 'scheduler' and (wire is not None)
             == any(phase['phase'] == 'read_completed' for phase in phases)), 'wire_phase_mismatch')
    partial_bytes, omitted = 0, 0
    retention = record['result_retention']
    require(retention in ('not_requested', 'retained_diagnostic_only', 'omitted_byte_limit', 'omitted_invalid_json'),
            'unknown_partial_retention')
    if 'result_json_bytes' in record:
        require(causal.nonnegative_integer(record['result_json_bytes']), 'invalid_partial_size')
    if retention == 'omitted_byte_limit':
        require(causal.nonnegative_integer(record.get('result_json_bytes')), 'omitted_partial_size_missing')
    if retention == 'retained_diagnostic_only':
        require(sample['transport_complete'] is False and 'partial_result' in record, 'partial_in_complete_round')
        partial = record['partial_result']
        require(isinstance(partial, dict), 'invalid_partial_result')
        require(partial.get('status') == record['source_status']
                and all(type(partial.get(name)) is type(value) and partial[name] == value
                        for name, value in record['identity'].items()), 'partial_identity_or_status_mismatch')
        partial_bytes = len(json.dumps(partial, allow_nan=False, separators=(',', ':')).encode())
        require(type(record.get('result_json_bytes')) is int and record['result_json_bytes'] == partial_bytes,
                'partial_result_bytes_differ')
        available = partial.get('available_at')
        received = number(available) and available <= deadline
        require(not number(available) or started <= available <= finished, 'partial_receipt_time_invalid')
        require(record['received_before_deadline'] == received, 'partial_receipt_deadline_mismatch')
    else:
        require('partial_result' not in record, 'unaccounted_partial_result')
        omitted = int(retention.startswith('omitted_'))
    if sample['transport_complete']:
        require(retention == 'not_requested' and observed, 'complete_round_has_unavailable_component')
    return dict(component=key, status=record['status'], last_phase=phases[-1]['phase'] if phases else None,
                observed_by_deadline=observed, elapsed_s=finished - started, partial_bytes=partial_bytes,
                partial_omitted=omitted, wire_bytes=wire)


def audit_sample(sample, clock_id):
    result = dict(tick=sample.get('tick'), complete=False, errors=[], components=[], cohort_at_round_start=[])
    try:
        require(type(sample.get('transport_complete')) is bool, 'invalid_round_complete_flag')
        result['transport_complete'] = sample['transport_complete']
        require(sample.get('prediction_status') == 'unavailable_not_estimated', 'prediction_input_not_disabled')
        for name in ('started_at', 'deadline_at', 'completed_at', 'as_of'):
            require(number(sample.get(name)), 'missing_round_time:' + name)
        require(sample['started_at'] <= sample['completed_at'] <= sample['as_of'], 'invalid_round_time_order')
        require(sample['deadline_at'] == sample['started_at'] + 0.08, 'round_deadline_changed')
        if sample['transport_complete']:
            require(sample.get('error_type') is None and sample.get('cancelled') is False
                    and sample['completed_at'] <= sample['deadline_at'], 'late_or_failed_round_marked_complete')
        else:
            require(type(sample.get('cancelled')) is bool, 'missing_round_cancellation_flag')
            if sample['cancelled']:
                require(sample.get('error_type') == 'CancelledError', 'round_cancellation_error_mismatch')
        require(sample.get('error_type') is None or isinstance(sample['error_type'], str), 'invalid_round_error')
        require(sample['completed_at'] <= sample['deadline_at'] or sample.get('error_type') is not None,
                'late_round_missing_error')
        require(sample.get('error_type') != 'DeadlineExceeded' or sample['completed_at'] > sample['deadline_at'],
                'early_deadline_exceeded_error')
        detail = sample['component_diagnostics']
        require(type(detail.get('schema')) is int and detail['schema'] == 1 and detail.get('diagnostic_only') is True
                and detail.get('prediction_input_eligible') is False, 'diagnostic_data_prediction_eligible')
        require(detail.get('clock_id') == clock_id and detail.get('deadline_at') == sample['deadline_at'],
                'diagnostic_clock_or_deadline_mismatch')
        require(detail.get('max_components') == 64 and detail.get('max_phases') == 16
                and detail.get('max_partial_bytes') == 65536, 'diagnostic_limits_changed')
        require(type(detail.get('dropped_components')) is int and detail['dropped_components'] == 0,
                'component_metadata_dropped')
        components = detail['components']
        require(isinstance(components, list), 'missing_components')
        seen = set()
        for record in components:
            try:
                row = audit_component(record, sample)
                require(row['component'] not in seen, 'duplicate_component')
                seen.add(row['component'])
                result['components'].append(row)
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                result['errors'].append(str(error))
        require(len(components) == 28 and seen == set(expected_components()), 'component_set_incomplete')
        partial_bytes = sum(row['partial_bytes'] for row in result['components'])
        omitted = sum(row['partial_omitted'] for row in result['components'])
        require(type(detail.get('partial_bytes_retained')) is int
                and partial_bytes == detail['partial_bytes_retained'] <= 65536, 'round_partial_bytes_differ_or_overflow')
        require(type(detail.get('partial_results_omitted')) is int and omitted == detail['partial_results_omitted'],
                'partial_omissions_mismatch')
        cohort = sample['cohort_at_start']
        require(isinstance(cohort, list) and len(cohort) <= 64, 'invalid_round_cohort')
        request_ids = set()
        for request in cohort:
            request_id = request['request_id']
            require(isinstance(request_id, str) and 0 < len(request_id) <= 256 and request_id not in request_ids,
                    'invalid_cohort_identity')
            request_ids.add(request_id)
            require(type(request['model']) is int and request['model'] in (0, 1)
                    and causal.nonnegative_integer(request['count'])
                    and type(request['final_output_received']) is bool, 'invalid_cohort_state')
            if not request['final_output_received']:
                result['cohort_at_round_start'].append(dict(request_id=request_id, model=request['model'],
                    first_receipt_pending=request['count'] == 0, submitted=request.get('submitted_at') is not None))
        result['complete'] = not result['errors']
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as error:
        result['errors'].append(str(error))
    return result


def audit_collection(report, protocol):
    result = dict(status='not_passed', errors=[], rounds=[], component_status_phase_counts={},
                  diagnostic_only=True, scientific_gate_passed=False, producer_root_cause_proven=False)
    try:
        check_protocol(protocol)
        events = report['events']
        require(isinstance(events, list) and all(isinstance(event, dict) for event in events), 'invalid_event_list')
        retention = causal.retention_status(report)
        result['retention'] = retention
        require(report['protocol']['period_s'] == 0.1 and report['protocol']['deadline_s'] == 0.08
                and report['protocol']['max_bytes'] == 536870912 and report['protocol']['max_events'] == 1000000
                and report['protocol']['max_requests'] == 64, 'collection_protocol_changed')
        samples = [event for event in events if event.get('kind') == 'sample']
        for sample in samples:
            result['rounds'].append(audit_sample(sample, report['protocol']['clock_id']))
        counts = report['counts']
        require(causal.nonnegative_integer(counts['bytes_retained']) and counts['bytes_retained'] <= 536870912,
                'retained_bytes_over_limit')
        require(samples and counts['rounds'] == len(samples), 'rounds_missing_or_empty')
        require(counts['incomplete_rounds'] == sum(sample.get('transport_complete') is False for sample in samples),
                'incomplete_round_count_mismatch')
        histogram = Counter()
        for row in result['rounds']:
            for component in row['components']:
                key = (component['component'], component['status'], component['last_phase'],
                       component['observed_by_deadline'], row.get('transport_complete'))
                histogram[json.dumps(key)] += 1
        result['component_status_phase_counts'] = dict(sorted(histogram.items()))
        result['invalid_rounds'] = sum(not row['complete'] for row in result['rounds'])
        result['sampled_rounds'] = len(samples)
        result['incomplete_rounds'] = counts['incomplete_rounds']
        require(retention['status'] == 'complete_collection_stream', 'collection_retention_not_complete')
        require(result['invalid_rounds'] == 0, 'invalid_component_rounds')
        result['status'] = 'passed'
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as error:
        result['errors'].append(str(error))
    return result


def bound_protocol(job_root, analysis_protocol):
    filename = job_root / 'inputs/component-diagnostic-protocol.json'
    identity = causal.digest(filename)
    deployment = causal.read(job_root / 'inputs/deployment-protocol.json')
    manifest = causal.read(job_root / 'inputs/deployment-manifest.json')
    require(identity == analysis_protocol['component_diagnostic_protocol_sha256']
            == deployment['component_diagnostic_protocol_sha256'], 'component_protocol_digest_differs')
    require(causal.digest(job_root / 'inputs/deployment-manifest.json') == analysis_protocol['deployment_manifest_sha256']
            and causal.digest(job_root / 'inputs/deployment-protocol.json') == manifest['protocol_sha256'],
            'deployment_binding_differs')
    protocol = causal.read(filename)
    check_protocol(protocol)
    require(protocol['startup_expected_count'] == 571 and protocol['startup_contract_sha256']
            == deployment['startup_test_contract_sha256']
            == causal.digest(job_root / 'inputs/startup-test-contract.json'), 'diagnostic_startup_binding_differs')
    require(protocol['source_sha256'] and all(analysis_protocol['compute_payload_sha256'].get(name) == identity
            for name, identity in protocol['source_sha256'].items()), 'diagnostic_source_binding_differs')
    causal.audit_compute_payload(job_root, analysis_protocol)
    causal.audit_startup(job_root, deployment)
    return protocol


def analyze_jobs(jobs, output, analysis_protocol):
    require(not output.exists(), 'analysis_output_already_exists')
    protocols = [bound_protocol(root, analysis_protocol) for block, root in jobs]
    baseline = causal.analyze_jobs(jobs, output, analysis_protocol)
    cases = []
    for (block, root), protocol in zip(jobs, protocols):
        plan = causal.read(root / 'inputs/causal-training-plan-20261110-r0.json')
        for case in plan['cases']:
            filename = root / 'out/causal-block/collect/out/calibration/causal' / case['case'] / 'collection.json'
            try:
                audit = audit_collection(causal.read(filename), protocol)
                input_sha256 = causal.digest(filename)
            except (OSError, ValueError) as error:
                audit, input_sha256 = dict(status='not_passed', errors=[str(error)]), None
            cases.append(dict(block=block, case=case['case'], input=str(filename), input_sha256=input_sha256, audit=audit))
    result = dict(schema=1, cases=cases, diagnostic_artifacts_passed=bool(cases)
                  and baseline['run_artifacts_passed'] and all(row['audit']['status'] == 'passed' for row in cases),
                  original_overhead_and_collection_passed=baseline['complete_overhead_and_collection'],
                  prediction_gate_passed=False, independent_validation=False, producer_root_cause_proven=False,
                  note='All rounds included. Request overlap is the round-start cohort,not exact component causal attribution.')
    causal.write(output / 'component-report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--analysis-protocol', type=Path, required=True)
    options = parser.parse_args()
    jobs = [(int(block), Path(directory)) for block, directory in
            (item.split('=', 1) for item in options.job)]
    result = analyze_jobs(jobs, options.output, causal.read(options.analysis_protocol))
    print(json.dumps({key: value for key, value in result.items() if key != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
