import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from host_residual import HostResidualModel, fit_host_residual
from reference_work import fingerprint


LABELS = ("worker_execute", "worker_sample", "worker_dummy")


def digest(filename):
    return hashlib.sha256(filename.read_bytes()).hexdigest()


def extract_episodes(worker):
    if any(worker.get(key) != 0 for key in ("dropped", "late_events", "schema_errors")):
        raise ValueError("Incomplete worker event stream")
    identity = worker["identity"]
    pending, episodes = {}, []
    previous = worker["started_ns"]
    seen = set()
    for event in worker["events"]:
        timestamp = event["at_ns"]
        if type(timestamp) is not int or not previous <= timestamp <= worker["ended_ns"]:
            raise ValueError("Worker timestamps are not ordered and bounded")
        previous = timestamp
        label, _, boundary = event["kind"].rpartition(":")
        if label not in LABELS:
            continue
        scope = event["scope_id"]
        if event["parent_scope_id"] is not None:
            raise ValueError("Outer worker scope unexpectedly nested")
        if boundary == "enter":
            if scope in seen or pending:
                raise ValueError("Duplicate or overlapping outer scope")
            seen.add(scope)
            pending[scope] = (label, timestamp, event["thread"])
            continue
        if boundary not in ("return", "error") or scope not in pending:
            raise ValueError("Missing outer entry or unknown terminal event")
        entered_label, start, thread = pending.pop(scope)
        if entered_label != label or thread != event["thread"]:
            raise ValueError("Outer scope identity changed")
        episodes.append(dict(id=f"{worker['case']}:{identity['service']}:{identity['dp_rank']}:"
            f"{identity['tp_rank']}:{scope}", case=worker["case"],
            group=[identity["service"], identity["dp_rank"], identity["tp_rank"], label],
            start_ns=start, end_ns=timestamp, duration_s=(timestamp - start) / 1e9,
            outcome="failed" if boundary == "error" else "returned"))
    for scope, (label, start, _) in pending.items():
        episodes.append(dict(id=f"{worker['case']}:{identity['service']}:{identity['dp_rank']}:"
            f"{identity['tp_rank']}:{scope}", case=worker["case"],
            group=[identity["service"], identity["dp_rank"], identity["tp_rank"], label],
            start_ns=start, end_ns=worker["ended_ns"], duration_s=(worker["ended_ns"] - start) / 1e9,
            outcome="right_censored"))
    return episodes


def virtual_queries(episode, *, start_ns, end_ns, period_ns, lag_s, restriction_s):
    first = max(episode["start_ns"], start_ns)
    stop = min(episode["end_ns"], end_ns)
    tick = ((first + period_ns - 1) // period_ns) * period_ns
    while tick < stop:
        residual = max(0.0, (episode["end_ns"] - tick) / 1e9 - lag_s)
        yield dict(episode_id=episode["id"], case=episode["case"], group=episode["group"],
            virtual_observed_ns=tick, observed_age_s=(tick - episode["start_ns"]) / 1e9,
            observation_lag_s=lag_s, outcome=episode["outcome"],
            target_s=None if episode["outcome"] == "right_censored" else min(residual, restriction_s),
            censored_target_lower_bound_s=min(residual, restriction_s) if episode["outcome"] == "right_censored" else None)
        tick += period_ns


def evaluate_case(descriptor, episodes, training, source_id, protocol, query_stream):
    if any(row["case"] == descriptor["case"] or row["end_ns"] >= descriptor["start_ns"] for row in training):
        raise ValueError("Same-case or future training leakage")
    artifact = fit_host_residual(training, source_id=source_id, minimum_at_risk=protocol["minimum_at_risk"],
                                minimum_cases=protocol["minimum_at_risk_cases"])
    estimator = HostResidualModel(artifact)
    statistics = dict(queries=0, available=0, labeled_available=0, unavailable={}, zero_targets=0,
                      censored_targets=0, failed_targets=0, absolute_error_sum_s=0.0, signed_error_sum_s=0.0,
                      target_sum_s=0.0, total_labeled_target_s=0.0)
    for episode in episodes:
        queries = virtual_queries(episode, start_ns=descriptor["start_ns"], end_ns=descriptor["end_ns"],
            period_ns=round(protocol["retrospective_poll_period_s"] * 1e9),
            lag_s=protocol["retrospective_receipt_lag_s"], restriction_s=protocol["restriction_s"])
        for query in queries:
            prediction = estimator.predict(query["group"], observed_age_s=query["observed_age_s"],
                observation_lag_s=query["observation_lag_s"], restriction_s=protocol["restriction_s"])
            statistics["queries"] += 1
            statistics["censored_targets"] += int(query["target_s"] is None)
            statistics["failed_targets"] += int(query["outcome"] == "failed")
            if query["target_s"] is not None:
                statistics["total_labeled_target_s"] += query["target_s"]
                statistics["zero_targets"] += int(query["target_s"] == 0)
            if prediction["status"] == "estimated_exploratory":
                statistics["available"] += 1
                if query["target_s"] is not None:
                    statistics["labeled_available"] += 1
                    error = prediction["restricted_mean_s"] - query["target_s"]
                    statistics["absolute_error_sum_s"] += abs(error)
                    statistics["signed_error_sum_s"] += error
                    statistics["target_sum_s"] += query["target_s"]
            else:
                reason = prediction["reason"]
                statistics["unavailable"][reason] = statistics["unavailable"].get(reason, 0) + 1
            query_stream.write(json.dumps(dict(query, prediction=prediction), separators=(",", ":")) + "\n")
    labeled = statistics["labeled_available"]
    statistics.update(coverage=statistics["available"] / statistics["queries"] if statistics["queries"] else None,
        mae_s=statistics["absolute_error_sum_s"] / labeled if labeled else None,
        signed_bias_s=statistics["signed_error_sum_s"] / labeled if labeled else None,
        wape=statistics["absolute_error_sum_s"] / statistics["target_sum_s"] if statistics["target_sum_s"] else None)
    return dict(descriptor, episodes=len(episodes), training_episodes=len(training),
                training_cases=sorted({row["case"] for row in training}), model_id=artifact["model_id"], **statistics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    report = dict(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), status="failed",
                  prediction_gate_passed=False, purpose="retrospective_sequential_exploration_not_independent_validation",
                  protocol_sha256=digest(args.protocol), auditor_sha256=digest(args.audit),
                  script_sha256=digest(Path(__file__)), runtime_sha256=digest(Path(__file__).parents[1] / "runtime/host_residual.py"),
                  errors=[], cases=[], used_inputs={})
    try:
        protocol = json.loads(args.protocol.read_text())
        audit = json.loads(args.audit.read_text())
        if audit["errors"] or audit["archive"]["sha256"] != protocol["archive_sha256"]:
            raise ValueError("Wrong or rejected audited archive")
        members = audit["archive"]["member_sha256"]
        def checked(filename):
            relative = str(filename.relative_to(args.artifact_root))
            checksum = digest(filename)
            if filename.is_symlink() or members.get(relative) != checksum:
                raise ValueError("Audited input SHA mismatch: " + relative)
            report["used_inputs"][relative] = checksum
            return json.loads(filename.read_text())
        root = args.artifact_root / 'job-309773/out/calibration/observability'
        cases = []
        for filename in root.glob('*/*/case.json'):
            descriptor = checked(filename)
            measurement = checked(filename.parent / 'measurement.json')
            cases.append(dict(descriptor, directory=filename.parent,
                start_ns=measurement['measurement_started_ns'], end_ns=measurement['measurement_ended_ns']))
        if len(cases) != 24 or sum(row['condition'] == 'timing_off' for row in cases) != 12:
            raise ValueError("Incomplete 24-case population")
        cases.sort(key=lambda row: row['start_ns'])
        training = []
        source_id = fingerprint(dict(archive=protocol['archive_sha256'], protocol=report['protocol_sha256']))
        with (args.output / 'episodes.jsonl').open('x') as episode_stream, (args.output / 'queries.jsonl').open('x') as query_stream:
            for descriptor in cases:
                directory = descriptor.pop('directory')
                if descriptor['condition'] == 'timing_off':
                    report['cases'].append(dict(descriptor, status='missing_observation_timing_disabled', episodes=0,
                        queries=0, training_episodes=len(training)))
                    continue
                stop = checked(directory / 'timing-stop.json')
                if stop['failures']:
                    raise ValueError("Failed timing stop, cannot silently use a successful subset")
                episodes, workers = [], set()
                for group in stop['results']:
                    for worker in group['workers']:
                        identity = worker['identity']
                        key = identity['service'], identity['dp_rank'], identity['tp_rank']
                        if key in workers or worker['case'] != descriptor['case']:
                            raise ValueError("Duplicate worker or wrong case")
                        workers.add(key)
                        episodes.extend(extract_episodes(worker))
                expected = {(f'instance-{model}', rank, lane) for model in (1, 2) for rank in (0, 1) for lane in (0, 1)}
                if workers != expected:
                    raise ValueError("Incomplete worker lanes")
                for episode in episodes:
                    episode_stream.write(json.dumps(episode, separators=(',', ':')) + '\n')
                evaluation = evaluate_case(descriptor, episodes, training, source_id, protocol, query_stream)
                report['cases'].append(dict(evaluation, status='exploratory_sequential_query_evaluation'))
                training.extend(episodes)
        model = fit_host_residual(training, source_id=source_id, minimum_at_risk=protocol['minimum_at_risk'],
                                  minimum_cases=protocol['minimum_at_risk_cases'])
        (args.output / 'model.json').write_text(json.dumps(model, separators=(',', ':')) + '\n')
        total = {key: sum(row.get(key, 0) for row in report['cases']) for key in
            ('episodes', 'queries', 'available', 'labeled_available', 'censored_targets', 'failed_targets',
             'absolute_error_sum_s', 'signed_error_sum_s', 'target_sum_s', 'zero_targets')}
        total.update(coverage=total['available'] / total['queries'] if total['queries'] else None,
                     mae_s=total['absolute_error_sum_s'] / total['labeled_available'] if total['labeled_available'] else None,
                     wape=total['absolute_error_sum_s'] / total['target_sum_s'] if total['target_sum_s'] else None)
        report.update(status='exploratory_fit_and_sequential_replay_complete_not_deployable', totals=total,
            model_id=model['model_id'], outcomes=dict(Counter(row['outcome'] for row in training)),
            limitations=['r4 already viewed;no independent holdout', 'virtual zero-lag samples,not live transport',
                'same node/process dependence and omitted workload covariates', 'host return only;not GPU/token progress',
                'no extrapolation or successful-subset accuracy gate', '12 timing-off cases remain unobserved'])
    except Exception as error:
        report['errors'].append(dict(type=type(error).__name__, error=str(error)))
    report['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report['outputs'] = {filename.name: digest(filename) for filename in args.output.iterdir() if filename.is_file()}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report.get(key) for key in ('status', 'totals', 'errors')}))
    if report['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
