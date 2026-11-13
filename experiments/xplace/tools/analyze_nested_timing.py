"""Describe audited nested host intervals without fitting or promoting experiment gates."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


LONG_GAP_NS = 100_000_000
LABELS = ("worker_execute", "worker_sample", "worker_dummy", "runner_sample",
          "runner_bookkeeping", "runner_to_list", "dp_coordinate")
GATES = ("measured_cost_gate", "independent_prediction_gate", "paired_ablation_gate", "mixed_topology_gate")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    def reject(value):
        raise ValueError("Nonfinite JSON number: " + value)
    def finite(value):
        number = float(value)
        require(math.isfinite(number), "Nonfinite JSON float")
        return number
    return json.loads(path.read_text(), parse_constant=reject, parse_float=finite)


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        require(type(start) is int and type(end) is int and start <= end, "Invalid host interval")
        if start == end:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def clipped(intervals, start, end):
    require(type(start) is int and type(end) is int and start <= end, "Invalid analysis window")
    return merged((max(left, start), min(right, end)) for left, right in intervals if left < end and right > start)


def duration(intervals):
    return sum(end - start for start, end in intervals)


def remainder(start, end, children):
    result, cursor = [], start
    for left, right in merged(children):
        require(start <= left <= right <= end, "Child interval outside parent")
        if cursor < left:
            result.append([cursor, left])
        cursor = right
    if cursor < end:
        result.append([cursor, end])
    return result


def prepare_intervals(spans):
    by_id = {span["scope_id"]: span for span in spans}
    require(len(by_id) == len(spans), "Repeated scope ID")
    result = []
    for span in spans:
        require(span["label"] in LABELS, "Unknown scope label")
        start, end = span["start_ns"], span["end_ns"]
        require(type(start) is int and type(end) is int and start <= end, "Backwards scope")
        children = merged((by_id[child]["start_ns"], by_id[child]["end_ns"]) for child in span["children"])
        exclusive = remainder(start, end, children)
        require(duration(children) + duration(exclusive) == end - start, "Scope partition differs")
        result.append({"label": span["label"], "scope_id": span["scope_id"], "thread": span["thread"],
                       "parent": span["parent"], "start_ns": start, "end_ns": end,
                       "children": children, "exclusive": exclusive})
    return result


def window_summary(intervals, start, end):
    selected = [span for span in intervals if span["start_ns"] < end and span["end_ns"] > start]
    outer = clipped(((span["start_ns"], span["end_ns"]) for span in selected if span["parent"] is None), start, end)
    exclusive = clipped((interval for span in selected for interval in span["exclusive"]), start, end)
    require(outer == exclusive, "Exclusive scopes do not cover outer host union")
    labels = {}
    for label in LABELS:
        calls = [span for span in selected if span["label"] == label]
        inclusive = clipped(((span["start_ns"], span["end_ns"]) for span in calls), start, end)
        residual = clipped((interval for span in calls for interval in span["exclusive"]), start, end)
        call_total = sum(duration(clipped([(span["start_ns"], span["end_ns"])], start, end)) for span in calls)
        child_total = sum(duration(clipped(span["children"], start, end)) for span in calls)
        residual_total = sum(duration(clipped(span["exclusive"], start, end)) for span in calls)
        require(call_total == child_total + residual_total, "Clipped parent partition differs")
        labels[label] = {"overlapping_calls": len(calls), "inclusive_union_ns": duration(inclusive),
                         "exclusive_union_ns": duration(residual), "sum_call_inclusive_ns": call_total,
                         "sum_call_direct_child_union_ns": child_total, "sum_call_exclusive_ns": residual_total}
    label_sum = sum(item["exclusive_union_ns"] for item in labels.values())
    outer_ns = duration(outer)
    require(0 <= outer_ns <= end - start and label_sum >= outer_ns, "Invalid wall coverage")
    return {"start_ns": start, "end_ns": end, "outer_union_ns": outer_ns,
            "exclusive_all_labels_union_ns": duration(exclusive), "exclusive_label_sum_ns": label_sum,
            "cross_label_overlap_ns": label_sum - outer_ns, "outside_observed_host_scopes_ns": end - start - outer_ns,
            "scopes": labels}


def callback_gaps(request):
    cadence = request["cadence"]
    require(isinstance(cadence, list) and cadence, "Missing callback cadence")
    for stamp, count in cadence:
        require(type(stamp) in (int, float) and math.isfinite(stamp) and stamp >= 0
                and type(count) is int and count > 0, "Invalid callback point")
    require(cadence[-1][1] == request["output_tokens"], "Callback cadence misses final tokens")
    gaps = []
    for index, (before, after) in enumerate(zip(cadence, cadence[1:])):
        elapsed, tokens = after[0] - before[0], after[1] - before[1]
        require(elapsed >= 0 and tokens > 0, "Backwards callback cadence")
        gaps.append({"gap_index": index, "start_ns": round(before[0] * 1e9), "end_ns": round(after[0] * 1e9),
                     "duration_s": elapsed, "visible_token_increment": tokens,
                     "merged_callback": tokens > 1, "long_gap": elapsed >= LONG_GAP_NS / 1e9})
    return gaps


def heartbeat_windows(frontend):
    intervals = {kind: [] for kind in ("async_heartbeat", "thread_heartbeat")}
    counts = dict.fromkeys(intervals, 0)
    for event in frontend["events"]:
        kind = event["kind"]
        if kind not in intervals:
            continue
        expected, actual, lateness = event["expected_ns"], event["at_ns"], event["lateness_ns"]
        require(all(type(value) is int for value in (expected, actual, lateness))
                and expected >= 0 and actual >= 0 and 0 <= lateness <= max(0, actual - expected), "Invalid heartbeat interval")
        counts[kind] += 1
        if lateness >= LONG_GAP_NS:
            intervals[kind].append((expected, actual))
    require(all(counts.values()), "Missing heartbeat kind")
    return {kind: merged(values) for kind, values in intervals.items()}


class AuditedInputs:
    def __init__(self, root, members):
        self.root = root.resolve()
        self.members = members
        self.used = {}

    def checked_path(self, relative):
        path = self.root / relative
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts
                and not path.is_symlink() and path.resolve().is_relative_to(self.root), "Unsafe audited input path")
        checksum = digest(path)
        require(checksum == self.members[relative], "Archived input changed: " + relative)
        self.used[relative] = checksum
        return path

    def read(self, relative):
        return load_json(self.checked_path(relative))


def checked(errors, label, operation):
    try:
        return operation()
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, ImportError) as error:
        errors.append({"check": label, "type": type(error).__name__, "error": str(error)})
        return None


def load_validator(reader, prefix, frozen):
    relative = prefix + "/runtime/nested_timing.py"
    path = reader.checked_path(relative)
    require(digest(path) == frozen["source_sha256"]["runtime/nested_timing.py"], "Frozen nested validator differs")
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("_audited_nested_timing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analyze_case(reader, prefix, descriptor, validator):
    relative = prefix + "/out/calibration/observability/" + descriptor["condition"] + "/" + descriptor["name"] + "-r" + str(descriptor["repeat"])
    require(reader.read(relative + "/case.json") == descriptor, "Case descriptor differs")
    row = reader.read(relative + "/measurement.json")
    requests = reader.read(relative + "/requests.json")
    require(row["measurement_schema"] == 2 and row["preemptions"] == 0, "Invalid or preempted measurement")
    require(all(row[key] == descriptor[key] for key in ("name", "repeat", "condition", "layout")), "Case layout differs")
    require(len(requests) == 1 + sum(descriptor["layout"]), "Missing case requests")
    code = [request for request in requests if request["model"] == 0]
    require(len(code) == 1 and code[0]["output_tokens"] == 64 and code[0]["prompt_tokens"] == 101167, "Code workload differs")
    require(all(request["model"] == 0 or (request["model"] == 1 and request["output_tokens"] == 3000
                and request["prompt_tokens"] == 4096) for request in requests), "Conv workload differs")
    gaps = callback_gaps(code[0])
    elapsed = sum(gap["duration_s"] for gap in gaps)
    result = {"requests": len(requests), "code_prefill_s": row["code_prefill_s"], "code_tpot_s": row["code_tpot_s"],
              "callback_gaps": gaps, "visible_token_increment": sum(gap["visible_token_increment"] for gap in gaps),
              "callback_duration_s": elapsed, "long_gap_count": sum(gap["long_gap"] for gap in gaps),
              "long_gap_time_fraction": sum(gap["duration_s"] for gap in gaps if gap["long_gap"]) / elapsed if elapsed else None,
              "workers": None}
    if descriptor["condition"] == "timing_off":
        return result
    stop = reader.read(relative + "/timing-stop.json")
    require(not stop["failures"], "Timing stop has failures")
    heartbeat = heartbeat_windows(reader.read(relative + "/timing-frontend.json"))
    timelines, identities = [], set()
    for group in stop["results"]:
        for worker in group["workers"]:
            validator.validate_worker_scopes(worker)
            require(worker["case"] == descriptor["case"] and not any(worker[key] for key in ("dropped", "late_events", "schema_errors")), "Invalid worker record")
            identity = worker["identity"]
            key = (identity["service"], identity["dp_rank"], identity["tp_rank"])
            require(key not in identities, "Repeated worker identity")
            identities.add(key)
            timelines.append((worker, prepare_intervals(validator.scope_intervals(worker))))
    expected = {(f"instance-{model + 1}", rank, lane) for model in range(2) for rank in range(2) for lane in range(2)}
    require(identities == expected, "Missing worker lane")
    start, end = row["measurement_started_ns"], row["measurement_ended_ns"]
    require(type(start) is int and type(end) is int and start < end, "Invalid measurement window")
    result["workers"] = [{"identity": worker["identity"], "configuration": worker["nested_timing"]["configuration"],
                          "measurement_window": window_summary(intervals, start, end)} for worker, intervals in timelines]
    for gap in gaps:
        if not gap["long_gap"]:
            continue
        start, end = gap["start_ns"], gap["end_ns"]
        gap["heartbeat_long_delay_overlap_ns"] = {kind: duration(clipped(values, start, end)) for kind, values in heartbeat.items()}
        gap["workers"] = [{"identity": worker["identity"], **window_summary(intervals, start, end)} for worker, intervals in timelines]
    return result


def analyze(evidence, audit_path, audit_sha256, job_id):
    report = {"schema": 1, "job": job_id, "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "analyzer_sha256": digest(Path(__file__)), "audit_sha256": audit_sha256, "errors": [], "cases": [],
              "threshold_ns": LONG_GAP_NS, "estimator_fitted": False, "gpu_completion_claim": False,
              "performance_claim": False, **dict.fromkeys(GATES, False),
              "semantics": "Retrospective host scopes only. Inclusive nested scopes must not be summed. Per-call child unions partition each parent; different threads may overlap. Uncovered time is unknown, not idle. Callback timestamps are approximately converted to ns; the fixed100ms threshold applies to original callback seconds. Heartbeat overlap may use a later event; not an online feature."}
    def prerequisites():
        require(job_id.isdigit(), "Invalid job ID")
        require(digest(audit_path) == audit_sha256, "Artifact audit SHA differs")
        audit = load_json(audit_path)
        require(audit["job"] == job_id and all(audit[key] is False for key in GATES), "Wrong audit job or gate contract")
        reader = AuditedInputs(evidence, audit["archive"]["member_sha256"])
        prefix = "job-" + job_id
        plan_names = [name for name, checksum in audit["frozen"]["source_sha256"].items()
                      if name.startswith("inputs/") and checksum == audit["frozen"]["plan_sha256"]]
        require(len(plan_names) == 1, "Missing or ambiguous frozen plan")
        plan = reader.read(prefix + "/" + plan_names[0])
        require(plan == reader.read(prefix + "/out/calibration/observability/plan.json"), "Observed plan differs")
        require(len(plan["cases"]) == 24 and plan["planned_requests"] == 336
                and plan["purpose"] == "mechanism_observation_not_predictive_validation", "Wrong fixed pilot")
        require(plan["nested_timing"]["scopes"] == list(LABELS[3:]), "Missing four-scope nested plan")
        validator = load_validator(reader, prefix, audit["frozen"])
        require(validator.source_sha256() == plan["nested_timing"]["source_sha256"], "Plan validator SHA differs")
        return audit, reader, prefix, plan, validator
    prepared = checked(report["errors"], "audited_inputs", prerequisites)
    if prepared is not None:
        audit, reader, prefix, plan, validator = prepared
        if audit["status"] != "artifact_checks_passed_not_predictive_validation" or audit["errors"]:
            report["errors"].append({"check": "artifact_acceptance", "error": "Artifact audit did not pass; all derived case entries remain partial evidence"})
        for descriptor in plan["cases"]:
            entry = {"case": descriptor["case"], "condition": descriptor["condition"], "layout": descriptor["layout"], "errors": []}
            details = checked(entry["errors"], "nested_case", lambda: analyze_case(reader, prefix, descriptor, validator))
            if details is not None:
                entry.update(details)
            entry["status"] = "failed" if entry["errors"] else "described"
            report["cases"].append(entry)
        report["pairs"] = checked(report["errors"], "paired_overhead", lambda: audit["observability"]["pairs"])
        checked(report["errors"], "all_pairs", lambda: require(isinstance(report["pairs"], list)
                and len(report["pairs"]) == 12, "Incomplete overhead pairs"))
        report["raw_sha256_used"] = reader.used
        checked(report["errors"], "all_requests", lambda: require(sum(case.get("requests", 0) for case in report["cases"]) == 336, "Incomplete requests; no subset aggregate accepted"))
    failed = bool(report["errors"] or len(report["cases"]) != 24 or any(case["errors"] for case in report["cases"]))
    report["status"] = "failed" if failed else "descriptive_only_not_predictive_validation"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite mechanism evidence")
    report = analyze(args.evidence, args.audit, args.audit_sha256, args.job_id)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "path": str(args.output), "sha256": digest(args.output)}))
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
