import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))

import accounting_capability as capability
import live_observation as live
from reference_work import fingerprint, project_reference_layout, validate_reference
from test_live_observation import environment
from test_reference_work import reference_input


class FakeNativeScheduler:
    def _update_after_schedule(self, output):
        return output

    def update_from_output(self, schedule, output):
        return schedule, output


class FakeRequest:
    pass


class FakePlacementScheduler(FakeNativeScheduler):
    def __init__(self, rank):
        self.parallel_config = SimpleNamespace(data_parallel_rank=rank, data_parallel_size=2,
                                               tensor_parallel_size=2, pipeline_parallel_size=1,
                                               enable_expert_parallel=True)
        self.scheduler_config = SimpleNamespace(async_scheduling=False, enable_chunked_prefill=True)
        self.cache_config = SimpleNamespace(enable_prefix_caching=False, block_size=16)
        self.vllm_config = SimpleNamespace(model_config=SimpleNamespace(is_diffusion=False),
                                           speculative_config=None, kv_transfer_config=None)
        self.lora_config = self.connector = None
        self.is_encoder_decoder = False
        self.num_lookahead_tokens = 0
        self.num_sampled_tokens_per_step = 1
        self.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
        self.max_num_running_reqs = 256
        self.max_num_scheduled_tokens = 4096
        self.max_model_len = 131072

    def placement_snapshot(self):
        return {'rank': self.parallel_config.data_parallel_rank, 'observed_at': time.monotonic()}

    def placement_summary(self):
        return self.placement_snapshot()


class FakeEngine:
    def __init__(self, rank=0):
        self.dp_rank = rank
        self.engines_running = False
        self.scheduler = FakePlacementScheduler(rank)


@pytest.fixture
def native_sources(monkeypatch):
    for name, attribute, value in (
            ('vllm.v1.core.sched.scheduler', 'Scheduler', FakeNativeScheduler),
            ('vllm.v1.request', 'Request', FakeRequest)):
        module = ModuleType(name)
        setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)


def bound_input():
    packet, query = reference_input()
    for record in packet['scheduler']['ranks']:
        model, rank = record['model'], record['dp_rank']
        captured = capability.AccountingCapability(FakeEngine(rank)).snapshot()
        assert captured['status'] == 'supported', captured
        record['snapshot']['accounting_capability'] = captured
        next(wave for wave in packet['execution']['rank_waves']
             if wave['model'] == model and wave['dp_rank'] == rank)['accounting_capability'] = copy.deepcopy(captured)
    example = packet['scheduler']['ranks'][0]['snapshot']['accounting_capability']
    manifest = {'kind': 'reference_accounting_manifest_v1', 'schema': 1,
                'sources': copy.deepcopy(example['sources']),
                'tenants': {str(model): copy.deepcopy(example['configuration']) for model in (0, 1)}}
    query.update(deployment_manifest=manifest, deployment_manifest_sha256=fingerprint(manifest))
    return packet, query


def rehash(record):
    record['capability_id'] = fingerprint({key: value for key, value in record.items() if key != 'capability_id'})


def replace_capability(packet, captured):
    packet['scheduler']['ranks'][0]['snapshot']['accounting_capability'] = copy.deepcopy(captured)
    packet['execution']['rank_waves'][0]['accounting_capability'] = copy.deepcopy(captured)


def rejected(result, expected):
    assert result['status'] == 'unavailable' and 'reference_id' not in result
    assert result['forecast_status'] == 'unavailable_not_estimated' and result['device_activity'] == 'unknown'
    assert expected in {error['reason'] for error in result['errors']}


def test_actual_source_descriptors_and_supported_config_are_recorded(native_sources):
    captured = capability.AccountingCapability(FakeEngine()).snapshot()
    assert captured['status'] == 'supported' and captured['errors'] == []
    expected = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert captured['sources']['vllm_scheduler']['sha256'] == expected
    assert captured['sources']['vllm_request']['sha256'] == expected
    assert captured['sources']['engine']['qualname'] == 'FakeEngine'
    assert captured['configuration']['num_sampled_tokens_per_step'] == 1
    assert captured['configuration']['num_lookahead_tokens'] == 0
    assert captured['capability_id'] == fingerprint({key: value for key, value in captured.items() if key != 'capability_id'})


@pytest.mark.parametrize('field', list(capability.CONFIG_PATHS))
def test_missing_configuration_is_not_defaulted_to_supported(native_sources, field):
    engine = FakeEngine()
    target = engine.scheduler
    attributes = capability.CONFIG_PATHS[field]
    for attribute in attributes[:-1]:
        target = getattr(target, attribute)
    delattr(target, attributes[-1])
    captured = capability.AccountingCapability(engine).snapshot()
    assert captured['status'] == 'unavailable'
    assert 'missing_or_invalid_configuration:' + field in captured['errors']


@pytest.mark.parametrize('field,value', [
    ('data_parallel_size', True), ('data_parallel_size', 1), ('tensor_parallel_size', 0),
    ('pipeline_parallel_size', 2), ('enable_expert_parallel', False), ('async_scheduling', True),
    ('enable_chunked_prefill', 1), ('enable_prefix_caching', True), ('is_encoder_decoder', True),
    ('is_diffusion', True), ('num_lookahead_tokens', 1), ('num_sampled_tokens_per_step', 0),
    ('speculative_config_absent', object()), ('kv_transfer_config_absent', object()),
    ('lora_config_absent', object()), ('connector_absent', object()), ('kv_cache_groups', []),
    ('max_num_scheduled_tokens', 0), ('block_size', 1.5), ('max_model_len', None),
])
def test_unsupported_configuration_is_explicit(native_sources, field, value):
    engine = FakeEngine()
    target = engine.scheduler
    attributes = capability.CONFIG_PATHS[field]
    for attribute in attributes[:-1]:
        target = getattr(target, attribute)
    setattr(target, attributes[-1], value)
    captured = capability.AccountingCapability(engine).snapshot()
    assert captured['status'] == 'unavailable'
    assert 'unsupported_configuration:' + field in captured['errors']


def test_sources_only_read_at_startup_and_configuration_rechecked(native_sources, monkeypatch):
    engine = FakeEngine()
    observed = capability.AccountingCapability(engine)
    def forbidden_read(*args, **kwargs):
        raise AssertionError('Per-sample source IO is prohibited')
    monkeypatch.setattr(Path, 'read_bytes', forbidden_read)
    assert observed.snapshot()['status'] == 'supported'
    engine.scheduler.max_num_scheduled_tokens = 2048
    changed = observed.snapshot()
    assert changed['status'] == 'unavailable'
    assert 'configuration_changed_since_startup' in changed['errors']


@pytest.mark.parametrize('mutation', ['engine_rank', 'scheduler_rank', 'both_rank', 'method', 'missing_method'])
def test_identity_or_accounting_method_changes_rejected(native_sources, mutation, monkeypatch):
    engine = FakeEngine()
    observed = capability.AccountingCapability(engine)
    if mutation in ('engine_rank', 'both_rank'):
        engine.dp_rank = 1
    if mutation in ('scheduler_rank', 'both_rank'):
        engine.scheduler.parallel_config.data_parallel_rank = 1
    if mutation == 'method':
        engine.scheduler.update_from_output = lambda *args: None
    if mutation == 'missing_method':
        monkeypatch.delattr(FakeNativeScheduler, 'update_from_output')
    assert observed.snapshot()['status'] == 'unavailable'


def test_override_and_missing_request_source_do_not_claim_support(native_sources, monkeypatch):
    engine = FakeEngine()
    engine.scheduler.update_from_output = lambda *args: None
    captured = capability.AccountingCapability(engine).snapshot()
    assert captured['status'] == 'unavailable' and not captured['sources']
    monkeypatch.delattr(sys.modules['vllm.v1.request'], 'Request')
    assert capability.AccountingCapability(FakeEngine()).snapshot()['status'] == 'unavailable'


def test_class_level_method_replacement_cannot_hide_behind_class_file_sha(native_sources, monkeypatch):
    monkeypatch.setattr(FakeNativeScheduler, 'update_from_output', lambda *args: None)
    result = capability.AccountingCapability(FakeEngine()).snapshot()
    assert result['status'] == 'unavailable' and not result['sources']
    assert 'runtime_sources_unavailable:ValueError' in result['errors']


def test_runtime_code_object_change_is_rejected_without_rehashing_disk(native_sources, monkeypatch):
    engine = FakeEngine()
    observed = capability.AccountingCapability(engine)
    original = FakeNativeScheduler.update_from_output
    monkeypatch.setattr(original, '__code__', (lambda *args: None).__code__)
    result = observed.snapshot()
    assert result['status'] == 'unavailable'
    assert 'accounting_method_changed:update_from_output' in result['errors']


def test_source_capture_failure_preserves_error_without_breaking_host_observation(native_sources, environment, monkeypatch):
    def unavailable(target):
        raise OSError('source inaccessible')
    monkeypatch.setattr(capability, 'source_descriptor', unavailable)
    engine = FakeEngine()
    server = live.bind_engine_observer(engine)
    try:
        record = asyncio.run(live.read_endpoint(live.endpoint_path(environment, 0, 0),
            {'model': 0, 'dp_rank': 0, 'role': 'engine'}, live.host_clock_id(), 1))
        assert record['status'] == 'observed' and not record['wave_active']
        assert record['accounting_capability']['status'] == 'unavailable'
        assert 'runtime_sources_unavailable:OSError' in record['accounting_capability']['errors']
    finally:
        server.close()


def test_live_socket_and_scheduler_bind_same_capability_under_original_message_limit(native_sources, environment):
    engine = FakeEngine()
    server = live.bind_engine_observer(engine)
    try:
        record = asyncio.run(live.read_endpoint(live.endpoint_path(environment, 0, 0),
            {'model': 0, 'dp_rank': 0, 'role': 'engine'}, live.host_clock_id(), 1))
        snapshot = engine.scheduler.placement_snapshot()
        assert snapshot['accounting_capability'] == record['accounting_capability']
        assert record['accounting_capability']['status'] == 'supported'
        assert len(json.dumps(record).encode()) < live.MAX_MESSAGE_BYTES == 4096
        assert snapshot['producer_pid'] == record['producer_pid']
        assert snapshot['producer_id'] == record['producer_id']
        snapshot['accounting_capability']['sources'].clear()
        assert engine.scheduler.placement_summary()['accounting_capability']['sources']
    finally:
        server.close()
    assert 'accounting_capability' not in engine.scheduler.placement_snapshot()


def test_complete_reference_binds_manifest_and_preserves_unknown_device_state(native_sources):
    packet, query = bound_input()
    original = copy.deepcopy(packet)
    reference = capability.prepare_bound_reference_work(packet, **query)
    assert reference['status'] == 'available' and reference['forecast_status'] == 'unavailable_not_estimated'
    assert reference['deployment_status'] == 'accounting_manifest_bound_not_runtime_acceptance'
    assert reference['deployment_manifest_sha256'] == query['deployment_manifest_sha256']
    assert len(reference['observations']['accounting_capabilities']) == 4
    assert reference['device_activity'] == 'unknown'
    validate_reference(reference)
    plan = project_reference_layout(reference, moves=[dict(model=1, id='conv-0-0', source=0, destination=1)])
    assert plan['reference_id'] == reference['reference_id'] and not plan['capacity_checked']
    assert packet == original


@pytest.mark.parametrize('role', capability.SOURCE_ROLES)
def test_every_source_role_must_match_predeclared_manifest(native_sources, role):
    packet, query = bound_input()
    captured = copy.deepcopy(packet['scheduler']['ranks'][0]['snapshot']['accounting_capability'])
    captured['sources'][role]['sha256'] = '0' * 64
    rehash(captured)
    replace_capability(packet, captured)
    rejected(capability.prepare_bound_reference_work(packet, **query), 'unreviewed_runtime_sources')


@pytest.mark.parametrize('mutation,reason', [
    ('missing', 'missing_accounting_capability'),
    ('one_changed', 'scheduler_wave_capability_mismatch'),
    ('corrupt_id', 'unsupported_or_invalid_accounting_capability'),
    ('schema_bool', 'unsupported_or_invalid_accounting_capability'),
    ('wrong_rank', 'unsupported_or_invalid_accounting_capability'),
    ('unsupported', 'unsupported_or_invalid_accounting_capability'),
    ('future_field', 'unsupported_or_invalid_accounting_capability'),
    ('malformed', 'malformed_accounting_capability'),
    ('topology', 'capability_query_topology_mismatch'),
    ('changed_limit', 'configuration_not_in_manifest'),
    ('async', 'unsupported_accounting_configuration'),
])
def test_partial_or_incompatible_capability_rejects_reference(native_sources, mutation, reason):
    packet, query = bound_input()
    captured = copy.deepcopy(packet['scheduler']['ranks'][0]['snapshot']['accounting_capability'])
    if mutation == 'missing':
        captured = None
    elif mutation == 'one_changed':
        packet['execution']['rank_waves'][0]['accounting_capability']['rank'] = 1
        rejected(capability.prepare_bound_reference_work(packet, **query), reason)
        return
    elif mutation == 'corrupt_id':
        captured['capability_id'] = '0' * 64
    elif mutation == 'schema_bool':
        captured['schema'] = True
    elif mutation == 'wrong_rank':
        captured['rank'] = 1
    elif mutation == 'unsupported':
        captured['status'] = 'unavailable'
    elif mutation == 'future_field':
        captured['remaining_output_oracle'] = 99999
    elif mutation == 'malformed':
        captured['configuration']['block_size'] = float('nan')
    elif mutation == 'topology':
        captured['configuration']['tensor_parallel_size'] = 1
    elif mutation == 'changed_limit':
        captured['configuration']['max_num_scheduled_tokens'] = 2048
    else:
        captured['configuration']['async_scheduling'] = True
    if mutation not in ('missing', 'corrupt_id', 'malformed'):
        rehash(captured)
    replace_capability(packet, captured)
    rejected(capability.prepare_bound_reference_work(packet, **query), reason)


@pytest.mark.parametrize('mutation', ['missing', 'wrong_hash', 'schema_bool', 'missing_tenant', 'missing_source',
                                    'source_sha', 'extra_field', 'async'])
def test_missing_or_invalid_manifest_never_self_calibrates_from_runtime(native_sources, mutation):
    packet, query = bound_input()
    manifest = query['deployment_manifest']
    if mutation == 'missing':
        query['deployment_manifest'] = None
    elif mutation == 'wrong_hash':
        query['deployment_manifest_sha256'] = '0' * 64
    else:
        if mutation == 'schema_bool':
            manifest['schema'] = True
        elif mutation == 'missing_tenant':
            manifest['tenants'].pop('1')
        elif mutation == 'missing_source':
            manifest['sources'].pop('engine')
        elif mutation == 'source_sha':
            manifest['sources']['engine']['sha256'] = 'unknown'
        elif mutation == 'extra_field':
            manifest['future_outcome'] = 1
        else:
            manifest['tenants']['0']['async_scheduling'] = True
        query['deployment_manifest_sha256'] = fingerprint(manifest)
    with pytest.raises(ValueError):
        capability.prepare_bound_reference_work(packet, **query)


@pytest.mark.parametrize('mutation', ['stale', 'late', 'clock', 'missing_lane', 'producer'])
def test_capability_binding_cannot_bypass_causal_join(native_sources, mutation):
    packet, query = bound_input()
    if mutation == 'stale':
        query['as_of'] = 20
    elif mutation == 'late':
        packet['scheduler']['ranks'][0]['available_at'] = 20
    elif mutation == 'clock':
        query['clock_id'] = 'wrong-clock'
    elif mutation == 'missing_lane':
        packet['execution']['workers'].pop()
    else:
        packet['scheduler']['ranks'][0]['snapshot']['producer_id'] = '0' * 32
    result = capability.prepare_bound_reference_work(packet, **query)
    assert result['status'] == 'unavailable' and 'reference_id' not in result


def test_bound_capability_is_in_work_hash_and_not_mutable_by_caller(native_sources):
    packet, query = bound_input()
    result = capability.prepare_bound_reference_work(packet, **query)
    result['observations']['accounting_capabilities'][0]['capability']['configuration']['block_size'] = 32
    with pytest.raises(ValueError):
        project_reference_layout(result)
    assert packet['scheduler']['ranks'][0]['snapshot']['accounting_capability']['configuration']['block_size'] == 16
