import copy
import importlib.util
import io
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'tools/fit_host_residual.py'
SPEC = importlib.util.spec_from_file_location('fit_host_residual', SOURCE)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def worker():
    return dict(case='case', started_ns=0, ended_ns=5000000000, dropped=0, late_events=0, schema_errors=0,
        identity=dict(service='instance-1', dp_rank=0, tp_rank=0), events=[
            dict(kind='worker_sample:enter', at_ns=1000000000, scope_id=1, parent_scope_id=None, thread=1),
            dict(kind='worker_sample:return', at_ns=3000000000, scope_id=1, parent_scope_id=None, thread=1)])


def test_extract_retains_failed_and_censored_outer_calls():
    record = worker()
    assert analysis.extract_episodes(record)[0]['duration_s'] == 2
    record['events'][-1]['kind'] = 'worker_sample:error'
    assert analysis.extract_episodes(record)[0]['outcome'] == 'failed'
    record['events'].pop()
    episode = analysis.extract_episodes(record)[0]
    assert episode['outcome'] == 'right_censored' and episode['duration_s'] == 4


@pytest.mark.parametrize('mutation', ['drop', 'overlap', 'missing_entry', 'out_of_order'])
def test_bad_source_rejected_before_fit_not_population_filtered(mutation):
    record = worker()
    if mutation == 'drop':
        record['dropped'] = 1
    elif mutation == 'overlap':
        record['events'].insert(1, copy.deepcopy(record['events'][0]))
    elif mutation == 'missing_entry':
        record['events'].pop(0)
    else:
        record['events'][-1]['at_ns'] = 0
    with pytest.raises(ValueError):
        analysis.extract_episodes(record)


def test_virtual_queries_use_fixed_grid_not_fractions_of_future_duration():
    episode = analysis.extract_episodes(worker())[0]
    queries = list(analysis.virtual_queries(episode, start_ns=0, end_ns=5000000000, period_ns=1000000000,
                                           lag_s=1.5, restriction_s=2))
    assert [query['virtual_observed_ns'] for query in queries] == [1000000000, 2000000000]
    assert [query['target_s'] for query in queries] == [0.5, 0]


@pytest.mark.parametrize('case,end', [('current', 3), ('previous', 1000000001)])
def test_sequential_replay_refuses_same_case_or_future_training(case, end):
    training = [dict(analysis.extract_episodes(worker())[0], case=case, end_ns=end)]
    with pytest.raises(ValueError, match='leakage'):
        analysis.evaluate_case(dict(case='current', start_ns=1000000000), [], training, 'test', {}, io.StringIO())
