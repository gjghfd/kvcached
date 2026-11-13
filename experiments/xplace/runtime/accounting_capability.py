import copy
import hashlib
import importlib
import inspect
from pathlib import Path
import re
import sys

from reference_work import fingerprint, prepare_reference_work


CONFIG_PATHS = {
    'data_parallel_size': ('parallel_config', 'data_parallel_size'),
    'tensor_parallel_size': ('parallel_config', 'tensor_parallel_size'),
    'pipeline_parallel_size': ('parallel_config', 'pipeline_parallel_size'),
    'enable_expert_parallel': ('parallel_config', 'enable_expert_parallel'),
    'async_scheduling': ('scheduler_config', 'async_scheduling'),
    'enable_chunked_prefill': ('scheduler_config', 'enable_chunked_prefill'),
    'enable_prefix_caching': ('cache_config', 'enable_prefix_caching'),
    'block_size': ('cache_config', 'block_size'),
    'is_encoder_decoder': ('is_encoder_decoder',),
    'is_diffusion': ('vllm_config', 'model_config', 'is_diffusion'),
    'speculative_config_absent': ('vllm_config', 'speculative_config'),
    'kv_transfer_config_absent': ('vllm_config', 'kv_transfer_config'),
    'lora_config_absent': ('lora_config',),
    'connector_absent': ('connector',),
    'kv_cache_groups': ('kv_cache_config', 'kv_cache_groups'),
    'num_lookahead_tokens': ('num_lookahead_tokens',),
    'num_sampled_tokens_per_step': ('num_sampled_tokens_per_step',),
    'max_num_running_reqs': ('max_num_running_reqs',),
    'max_num_scheduled_tokens': ('max_num_scheduled_tokens',),
    'max_model_len': ('max_model_len',),
}
SOURCE_ROLES = ('vllm_scheduler', 'vllm_request', 'engine', 'placement_scheduler',
                'live_observation', 'accounting_capability')
ACCOUNTING_METHODS = ('_update_after_schedule', 'update_from_output')


def configuration_errors(config):
    if not isinstance(config, dict) or set(config) != set(CONFIG_PATHS):
        return ['missing_or_extra_configuration_fields']
    errors = []
    expected = {'pipeline_parallel_size': 1, 'enable_expert_parallel': True, 'async_scheduling': False,
                'enable_prefix_caching': False, 'is_encoder_decoder': False, 'is_diffusion': False,
                'speculative_config_absent': True, 'kv_transfer_config_absent': True,
                'lora_config_absent': True, 'connector_absent': True, 'kv_cache_groups': 1,
                'num_lookahead_tokens': 0, 'num_sampled_tokens_per_step': 1}
    positive = ('data_parallel_size', 'tensor_parallel_size', 'block_size', 'max_num_running_reqs',
                'max_num_scheduled_tokens', 'max_model_len')
    for key, value in config.items():
        valid = (type(value) is int and value >= (2 if key == 'data_parallel_size' else 1) if key in positive else
                 type(value) is bool if key == 'enable_chunked_prefill' else
                 type(value) is type(expected[key]) and value == expected[key])
        if not valid:
            errors.append('unsupported_configuration:' + key)
    return errors


def capture_configuration(scheduler):
    config, errors = {}, []
    for key, attributes in CONFIG_PATHS.items():
        try:
            value = scheduler
            for attribute in attributes:
                value = getattr(value, attribute)
            if key.endswith('_absent'):
                value = value is None
            elif key == 'kv_cache_groups':
                if not isinstance(value, (list, tuple)):
                    raise TypeError('KV groups are not a sequence')
                value = len(value)
            elif type(value) not in (int, bool):
                raise TypeError('Configuration field has unexpected type')
            config[key] = value
        except (AttributeError, TypeError):
            config[key] = None
            errors.append('missing_or_invalid_configuration:' + key)
    return config, errors + configuration_errors(config)


def source_descriptor(target):
    filename = inspect.getsourcefile(target)
    module = target.__name__ if inspect.ismodule(target) else target.__module__
    name = '<module>' if inspect.ismodule(target) else target.__qualname__
    if (filename is None or Path(filename).suffix != '.py' or not module or not name
            or len(module) > 200 or len(name) > 200):
        raise ValueError('Unavailable source identity')
    raw = Path(filename).read_bytes()
    if not raw or len(raw) > 8 * 1024 * 1024:
        raise ValueError('Unsupported source file size')
    return {'module': module, 'qualname': name, 'sha256': hashlib.sha256(raw).hexdigest()}


def method_function(scheduler, name):
    method = getattr(scheduler, name)
    return getattr(method, '__func__', method)


def runtime_sources(engine):
    scheduler_class = importlib.import_module('vllm.v1.core.sched.scheduler').Scheduler
    request_class = importlib.import_module('vllm.v1.request').Request
    if not isinstance(engine.scheduler, scheduler_class):
        raise ValueError('Unexpected scheduler implementation')
    methods = {name: getattr(scheduler_class, name) for name in ACCOUNTING_METHODS}
    if any(method_function(engine.scheduler, name) is not expected for name, expected in methods.items()):
        raise ValueError('Scheduler accounting methods are overridden')
    scheduler_source = Path(inspect.getsourcefile(scheduler_class)).resolve()
    for name, method in methods.items():
        if (not inspect.isfunction(method) or method.__module__ != scheduler_class.__module__
                or method.__qualname__ != scheduler_class.__qualname__ + '.' + name
                or method.__code__.co_name != name
                or Path(inspect.getsourcefile(method)).resolve() != scheduler_source):
            raise ValueError('Accounting method source does not match the declared scheduler')
    targets = {'vllm_scheduler': scheduler_class, 'vllm_request': request_class, 'engine': type(engine),
               'placement_scheduler': type(engine.scheduler), 'live_observation': sys.modules['live_observation'],
               'accounting_capability': sys.modules[__name__]}
    return {role: source_descriptor(target) for role, target in targets.items()}, methods


class AccountingCapability:
    def __init__(self, engine):
        self.engine = engine
        self.initial_config, self.initial_errors = capture_configuration(engine.scheduler)
        self.initial_rank = getattr(engine, 'dp_rank', None)
        self.sources, self.methods = {}, {}
        self.method_codes = {}
        try:
            self.sources, self.methods = runtime_sources(engine)
            self.method_codes = {name: method.__code__ for name, method in self.methods.items()}
        except (ImportError, AttributeError, TypeError, ValueError, OSError, KeyError) as error:
            self.initial_errors.append('runtime_sources_unavailable:' + type(error).__name__)

    def snapshot(self):
        config, errors = capture_configuration(self.engine.scheduler)
        errors = list(self.initial_errors) + errors
        if fingerprint(config) != fingerprint(self.initial_config):
            errors.append('configuration_changed_since_startup')
        rank = getattr(self.engine, 'dp_rank', None)
        configured_rank = getattr(getattr(self.engine.scheduler, 'parallel_config', None), 'data_parallel_rank', None)
        if type(rank) is not int or type(configured_rank) is not int or rank != configured_rank or rank != self.initial_rank:
            errors.append('engine_scheduler_rank_mismatch_or_change')
        elif type(config.get('data_parallel_size')) is not int or not 0 <= rank < config['data_parallel_size']:
            errors.append('configured_rank_out_of_bounds')
        for name, expected in self.methods.items():
            try:
                if (method_function(self.engine.scheduler, name) is not expected
                        or expected.__code__ is not self.method_codes[name]):
                    errors.append('accounting_method_changed:' + name)
            except AttributeError:
                errors.append('accounting_method_missing:' + name)
        record = {'kind': 'scheduler_accounting_capability_v1', 'schema': 1,
                  'status': 'unavailable' if errors else 'supported', 'rank': rank,
                  'configuration': config, 'sources': copy.deepcopy(self.sources), 'errors': sorted(set(errors))}
        record['capability_id'] = fingerprint(record)
        return record


def validate_manifest(manifest, expected_sha256):
    if (not isinstance(manifest, dict) or set(manifest) != {'kind', 'schema', 'sources', 'tenants'}
            or manifest.get('kind') != 'reference_accounting_manifest_v1'
            or type(manifest.get('schema')) is not int or manifest['schema'] != 1
            or not isinstance(expected_sha256, str) or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None
            or fingerprint(manifest) != expected_sha256):
        raise ValueError('Missing, changed, or invalid predeclared capability manifest')
    sources, tenants = manifest['sources'], manifest['tenants']
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_ROLES):
        raise ValueError('Manifest must bind every source role')
    for descriptor in sources.values():
        if (not isinstance(descriptor, dict) or set(descriptor) != {'module', 'qualname', 'sha256'}
                or any(not isinstance(descriptor[key], str) or not descriptor[key] for key in descriptor)
                or re.fullmatch('[0-9a-f]{64}', descriptor['sha256']) is None):
            raise ValueError('Invalid manifest source descriptor')
    if (not isinstance(tenants, dict) or set(tenants) != {'0', '1'}
            or any(configuration_errors(config) for config in tenants.values())):
        raise ValueError('Manifest must declare supported configuration for both tenants')


def capability_pair_error(scheduler_cap, wave_cap, identity):
    if not isinstance(scheduler_cap, dict) or not isinstance(wave_cap, dict):
        return 'missing_accounting_capability'
    try:
        if fingerprint(scheduler_cap) != fingerprint(wave_cap):
            return 'scheduler_wave_capability_mismatch'
        if (set(scheduler_cap) != {'kind', 'schema', 'status', 'rank', 'configuration', 'sources', 'errors', 'capability_id'}
                or scheduler_cap.get('kind') != 'scheduler_accounting_capability_v1'
                or type(scheduler_cap.get('schema')) is not int or scheduler_cap['schema'] != 1
                or scheduler_cap.get('status') != 'supported' or scheduler_cap.get('errors') != []
                or type(scheduler_cap.get('rank')) is not int or scheduler_cap['rank'] != identity[1]
                or fingerprint({key: value for key, value in scheduler_cap.items() if key != 'capability_id'})
                != scheduler_cap.get('capability_id')):
            return 'unsupported_or_invalid_accounting_capability'
    except (TypeError, ValueError):
        return 'malformed_accounting_capability'
    return None


def prepare_bound_reference_work(transport, *, deployment_manifest, deployment_manifest_sha256, **reference_query):
    """Bind causal reference work to predeclared accounting capabilities, not prediction accuracy."""
    validate_manifest(deployment_manifest, deployment_manifest_sha256)
    result = prepare_reference_work(transport, **reference_query)
    if result['status'] != 'available':
        return result
    waves = {(record['model'], record['dp_rank']): record for record in transport['execution']['rank_waves']}
    bound = []
    for record in transport['scheduler']['ranks']:
        identity = record['model'], record['dp_rank']
        scheduler_cap = record['snapshot'].get('accounting_capability')
        wave_cap = waves[identity].get('accounting_capability')
        reason = capability_pair_error(scheduler_cap, wave_cap, identity)
        if reason is None:
            config = scheduler_cap.get('configuration')
            if configuration_errors(config):
                reason = 'unsupported_accounting_configuration'
            elif (config['data_parallel_size'] != result['num_ranks']
                  or config['tensor_parallel_size'] != result['tensor_parallel_size']):
                reason = 'capability_query_topology_mismatch'
            elif scheduler_cap.get('sources') != deployment_manifest['sources']:
                reason = 'unreviewed_runtime_sources'
            elif fingerprint(config) != fingerprint(deployment_manifest['tenants'][str(identity[0])]):
                reason = 'configuration_not_in_manifest'
        if reason:
            result['errors'].append({'component': 'accounting_capability', 'identity': list(identity), 'reason': reason})
        else:
            bound.append({'model': identity[0], 'dp_rank': identity[1], 'capability': copy.deepcopy(scheduler_cap)})
    result.pop('reference_id')
    if result['errors']:
        result['status'] = 'unavailable'
        return result
    result['observations']['accounting_capabilities'] = sorted(bound, key=lambda record: (record['model'], record['dp_rank']))
    result['observation_sha256'] = fingerprint(result['observations'])
    result['deployment_manifest_sha256'] = deployment_manifest_sha256
    result['deployment_status'] = 'accounting_manifest_bound_not_runtime_acceptance'
    result['reference_id'] = fingerprint(result)
    return result
