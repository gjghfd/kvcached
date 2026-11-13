import argparse
import csv
from decimal import Decimal
import json
import math
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as element_tree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from analyze_causal_calibration import digest, number, read, require, summarize_process_usage, write
from calibrate import steady_step
from causal_deployment import check_tree
from graph_execution import BYTE_LIMIT, EVENT_LIMIT, validate_graph_pair
from imbalance_calibration import scan_cases
from main import parse_args
from parallel_topology import verify_workers
from scan_job import BLOCKS, COMMON, load_protocol


ERRORS = (OSError, KeyError, TypeError, ValueError, RuntimeError, IndexError, AttributeError, OverflowError, element_tree.ParseError)
METRICS = ("steady_step_s", "mean_request_ttft_s", "mean_request_tpot_s")


def analysis_sources():
    project = Path(__file__).resolve().parents[1]
    paths = [*sorted((project / "runtime").rglob("*.py")), Path(__file__).resolve(),
             Path(__file__).resolve().with_name("analyze_causal_calibration.py")]
    return {str(path.relative_to(project)): digest(path) for path in paths}


def same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def integer(value, minimum=0):
    return type(value) is int and minimum <= value < 2 ** 63


def close(left, right):
    return number(left) and number(right) and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def expected_requests(case):
    result = []
    for rank, count in enumerate(case["layout"]):
        for _ in range(count):
            index = len(result)
            result.append(dict(request_id=f"dense_{case['name']}_{index}", rank_initial=rank, model=case["model"],
                               prompt_index=index, prompt_tokens=case["context_tokens"],
                               target_output_tokens=case["output_tokens"]))
    return result


def warmup_case(cases):
    largest = max(cases, key=lambda case: case["batch_size"])
    count = largest["batch_size"] // 2
    return dict(largest, name="warmup", layout=[count, count], output_tokens=256,
                delta=0, imbalance=0, max_count=count, capture_bucket=count)


def request_metrics(case, rows, started_at):
    expected = {row["request_id"]: row for row in expected_requests(case)}
    require(isinstance(rows, list) and len(rows) == len(expected), "request_count_differs")
    require(len({row["request_id"] for row in rows}) == len(rows)
            and {row["request_id"] for row in rows} == set(expected), "request_identity_differs")
    request_rows = []
    for row in rows:
        planned = expected[row["request_id"]]
        require(all(same(row[key], planned[key]) for key in ("model", "rank_initial", "prompt_tokens"))
                and row["role"] == "calibration" and same(row["output_tokens"], planned["target_output_tokens"]),
                "request_work_differs")
        times = [row[key] for key in ("arrived_at", "submitted_at", "first_token_at", "completed_at")]
        require(all(number(value) and value >= 0 for value in times) and times == sorted(times)
                and row["arrived_at"] == started_at and row["completed_at"] > row["first_token_at"],
                "request_timestamps_invalid")
        require(row["cadence_stride"] == 32 and type(row["cadence_stride"]) is int, "cadence_stride_differs")
        previous_time, previous_count = row["submitted_at"], 0
        require(isinstance(row["cadence"], list) and row["cadence"], "cadence_missing")
        for point in row["cadence"]:
            require(isinstance(point, list) and len(point) == 2, "cadence_shape_invalid")
            stamp, count = point
            require(number(stamp) and previous_time <= stamp <= row["completed_at"]
                    and integer(count) and previous_count <= count <= row["output_tokens"], "cadence_invalid")
            require(count == 0 and stamp < row["first_token_at"] or count > 0 and stamp >= row["first_token_at"],
                    "cadence_first_receipt_differs")
            previous_time, previous_count = stamp, count
        require(previous_count == row["output_tokens"], "cadence_final_count_missing")
        request_rows.append(dict(request_id=row["request_id"], ttft_s=row["first_token_at"] - row["submitted_at"],
                                 tpot_s=(row["completed_at"] - row["first_token_at"]) / (row["output_tokens"] - 1)))
    step, window = steady_step(rows)
    require(number(step) and step > 0 and window["samples"] == len(expected), "incomplete_common_decode_window")
    metrics = dict(steady_step_s=step, mean_request_ttft_s=statistics.mean(row["ttft_s"] for row in request_rows),
                   mean_request_tpot_s=statistics.mean(row["tpot_s"] for row in request_rows))
    require(all(number(value) and value > 0 for value in metrics.values()), "nonpositive_request_metric")
    return dict(metrics=metrics, window=window, requests=request_rows)


def graph_metrics(start, stop, case, workers, clock, started_at, completed_at):
    require(isinstance(clock, str) and clock.startswith("linux-monotonic:") and ":time:[" in clock,
            "attested_clock_missing")
    result = validate_graph_pair(start, stop, clock_id=clock)
    require(result["valid"], "graph_invalid:" + json.dumps(result["errors"]))
    require(start["case"] == case["name"], "graph_case_differs")
    for lane in result["lanes"]:
        identity = lane["identity"]
        key = (case["model"], identity["dp_rank"], identity["tp_rank"])
        require(identity["model"] == case["model"] and key in workers
                and identity["pid"] == workers[key]["pid"] and identity["gpu_uuid"] == workers[key]["gpu_uuid"],
                "graph_worker_not_attested")
        require(sum(lane["selections"].values()) > 0 and lane["host_replay_calls"] > 0,
                "no_observed_graph_calls")
    for ranks in stop["results"]:
        for row in ranks:
            require(row["event_limit"] == EVENT_LIMIT and row["byte_limit"] == BYTE_LIMIT,
                    "graph_bounds_differ")
            require(row["events"][0]["at"] <= started_at <= completed_at <= row["events"][-1]["at"],
                    "graph_window_does_not_enclose_requests")
    result["clock_source"] = "archived_frontend_and_eight_worker_attestations_not_analysis_host"
    return result


class ArtifactReader:
    def __init__(self, root):
        self.root = Path(root)
        self.hashes = {}

    def load(self, relative):
        filename = self.root / relative
        self.hashes[str(relative)] = digest(filename)
        return json.loads(filename.read_text(), parse_constant=reject_nonfinite, parse_float=finite_float)


def reject_nonfinite(value):
    raise ValueError("nonfinite_json:" + value)


def finite_float(value):
    result = float(value)
    require(math.isfinite(result), "nonfinite_json:" + value)
    return result


def process_usage(filename, pid):
    previous, identities, sample_count, units = -1, {}, 0, None
    with filename.open() as handle:
        for line in handle:
            row = json.loads(line, parse_constant=reject_nonfinite, parse_float=finite_float)
            require(same(row["group"], pid) and number(row["at"]) and row["at"] >= max(0, previous)
                    and integer(row["clock_ticks_per_s"], 1) and integer(row["page_bytes"], 1), "process_usage_identity_invalid")
            previous = row["at"]
            current_units = row["clock_ticks_per_s"], row["page_bytes"]
            require(units is None or units == current_units, "process_usage_units_changed")
            units = current_units
            require(isinstance(row["read_failures"], list) and isinstance(row["processes"], list), "process_usage_shape_invalid")
            seen = set()
            for process in row["processes"]:
                require(all(integer(process[key], 1 if key == "pid" else 0)
                            for key in ("pid", "start_ticks", "cpu_ticks", "rss_pages")), "process_usage_counter_invalid")
                identity = process["pid"], process["start_ticks"]
                require(process["pid"] not in seen and process["cpu_ticks"] >= identities.get(identity, 0),
                        "process_usage_duplicate_or_reversed_counter")
                seen.add(process["pid"])
                identities[identity] = process["cpu_ticks"]
            sample_count += 1
    require(sample_count and any(identity[0] == pid for identity in identities), "process_usage_main_pid_missing")
    return summarize_process_usage(filename)


def analyze_case(root, case, mode, workers, clock):
    reader = ArtifactReader(root)
    result = dict(case=case, mode=mode, available=False, errors=[], metrics={key: None for key in METRICS},
                  requests=[], planned_requests=expected_requests(case), raw_status=None, graph=None)
    name = case["name"]
    rows = []
    try:
        status = reader.load(f"{name}-status.json")
        result["raw_status"] = status
        attempt = reader.load(f"{name}-attempt.json")
        require(same(attempt["case"], case) and same(attempt["requests"], result["planned_requests"]), "attempt_plan_differs")
        require(all(same(status[key], value) for key, value in case.items()), "status_case_differs")
        failure_artifacts = []
        for suffix in ("incomplete", "cleanup", "cleanup-results"):
            if (Path(root) / f"{name}-{suffix}.json").exists():
                result[suffix.replace("-", "_")] = reader.load(f"{name}-{suffix}.json")
                failure_artifacts.append(suffix)
        rows = reader.load(f"{name}-requests.json")
        result["raw_completed_count"] = len(rows)
        require(status["status"] == "measured" and status.get("censored_at") is None
                and not failure_artifacts, "failed_or_censored_case")
        require(number(attempt["prepared_at"]) and attempt["prepared_at"] == status["prepared_at"]
                and attempt["prepared_at"] <= status["started_at"] <= status["finished_at"], "case_timestamps_invalid")
        performance = request_metrics(case, rows, status["started_at"])
        require(max(row["completed_at"] for row in rows) <= status["finished_at"], "completion_after_case_end")
        before, after = reader.load(f"{name}-before.json"), reader.load(f"{name}-after.json")
        changes = []
        for initial, final in zip(before[case["model"]], after[case["model"]]):
            require(integer(initial["preemptions_total"]) and integer(final["preemptions_total"])
                    and final["preemptions_total"] >= initial["preemptions_total"], "invalid_preemption_counter")
            changes.append(final["preemptions_total"] - initial["preemptions_total"])
        require(len(before[case["model"]]) == len(after[case["model"]]) == len(changes) == 2
                and sum(changes) == 0 and same(status["preemptions"], 0), "preemption_or_missing_lane")
        require(close(status["step_s"], performance["metrics"]["steady_step_s"])
                and same(status["window"], performance["window"]), "derived_performance_summary_differs")
        result.update(performance)
        result["request_identity"] = attempt["requests"]
    except ERRORS as error:
        result["errors"].append(str(error))
    try:
        if mode == "collect":
            start, stop = reader.load(f"{name}-graph-start.json"), reader.load(f"{name}-graph-stop.json")
            result["raw_graph_counters"] = [
                {key: row.get(key) for key in ("identity", "attempted", "retained", "dropped", "invalid", "bytes_retained")}
                for rank in stop.get("results", []) if isinstance(rank, list) for row in rank]
            require(result["raw_status"] is not None and result["raw_status"].get("graph_evidence", {}).get("valid") is True,
                    "runtime_graph_rejected")
            result["graph"] = graph_metrics(start, stop, case, workers, clock, result["raw_status"]["started_at"],
                                             max(row["completed_at"] for row in rows))
        else:
            require(not any(Path(root).glob(f"{name}-graph-*.json"))
                    and not (result["raw_status"] or {}).get("graph_evidence"), "off_condition_has_graph_artifacts")
    except ERRORS as error:
        result["errors"].append(str(error))
    result["available"] = not result["errors"]
    result["input_sha256"] = reader.hashes
    return result


def check_command(command, config_name, common, checkpoint):
    require(isinstance(command, list) and len(command) > 2 and Path(command[1]).name == "main.py", "child_command_invalid")
    options, values = {}, command[2:]
    while values:
        option, *values = values
        require(option.startswith("--") and "=" not in option and option not in options, "child_option_invalid")
        argument = []
        while values and not values[0].startswith("--"):
            argument.append(values.pop(0))
        options[option[2:].replace("-", "_")] = argument
    require(set(options) <= set(COMMON) | {"checkpoint", "prompt_bank", "output", "imbalance_config"},
            "child_unknown_option")
    require(options.get("checkpoint") == [checkpoint] and len(options.get("prompt_bank", [])) == 1
            and options.get("output") and len(options.get("imbalance_config", [])) == 1
            and Path(options["imbalance_config"][0]).name == config_name, "child_input_option_differs")
    try:
        parsed = parse_args(command[2:])
    except SystemExit as error:
        raise ValueError("child_argument_parse_failed") from error
    for key, expected in common.items():
        actual = getattr(parsed, key)
        if key == "max_inflight":
            actual = list(actual)
        require(same(actual, expected), "child_option_or_default_differs:" + key)
    return options


def audit_condition(job_root, block, mode, manifest, manifest_sha, protocol, configs, checkpoint):
    root = Path(job_root) / "out/scan-overhead-block" / mode
    reader = ArtifactReader(root)
    result = dict(status="not_passed", errors=[], workers={}, clock=None, physical_conditions=None)
    try:
        process = reader.load("process.json")
        require(process["status"] == "passed" and same(process["returncode"], 0) and process["timeout"] is False
                and process["forced_cleanup"] is False and not process.get("kill_remaining_group", False)
                and integer(process["pid"], 1), "unclean_condition_process")
        require(number(process["started_at"]) and number(process["finished_at"])
                and process["started_at"] < process["finished_at"], "process_wall_invalid")
        result["process"] = process
        options = check_command(process["command"], configs[f"{block}-{mode}"].name, protocol["common_args"], checkpoint)
        require(tuple(Path(options["output"][0]).parts[-3:]) == ("scan-overhead-block", mode, "out"),
                "child_output_path_differs")
        bank = Path(job_root) / "inputs" / Path(options["prompt_bank"][0]).name
        require(digest(bank) == manifest["prompt_bank_sha256"], "prompt_bank_differs")
        front, final = reader.load("out/causal-frontend-attestation.json"), reader.load("out/causal-final-attestation.json")
        for record in (front, final):
            require(record["status"] == "verified" and record["manifest_sha256"] == manifest_sha
                    and record["native_checked"] is True and same(record["versions"], manifest["versions"])
                    and record["pid"] == process["pid"], "frontend_attestation_invalid")
        require(front["clock_id"] == final["clock_id"] and front["clock_id"].startswith("linux-monotonic:"),
                "frontend_clock_differs")
        result["clock"] = front["clock_id"]
        aggregate = reader.load("out/causal-worker-attestations.json")
        require(aggregate["status"] == "verified" and aggregate["clock_id"] == front["clock_id"], "worker_aggregate_invalid")
        aggregated = {(row["model"], row["dp_rank"], row["tp_rank"]): row["pid"] for row in aggregate["workers"]}
        require(len(aggregated) == len(aggregate["workers"]) == 8, "worker_aggregate_count_differs")
        for model in (0, 1):
            for rank in (0, 1):
                for lane in (0, 1):
                    worker = reader.load(f"out/worker-instance-{model + 1}-dp{rank}-tp{lane}.json")
                    attestation = worker["causal_deployment"]
                    require(attestation["status"] == "verified" and attestation["manifest_sha256"] == manifest_sha
                            and same(attestation["configuration"], manifest["configuration"])
                            and attestation["native_checked"] is False and attestation["clock_id"] == front["clock_id"]
                            and integer(worker["pid"], 1) and attestation["pid"] == worker["pid"] == aggregated[(model, rank, lane)],
                            "worker_attestation_invalid")
                    require(worker["service"] == f"instance-{model + 1}" and same(worker["dp_rank"], rank)
                            and worker["checkpoint"] == checkpoint
                            and same(worker["groups"]["dp"], dict(rank=rank, world_size=2))
                            and same(worker["groups"]["tp"], dict(rank=lane, world_size=2))
                            and "A800" in worker["gpu_name"] and isinstance(worker["gpu_uuid"], str) and worker["gpu_uuid"],
                            "worker_topology_invalid")
                    result["workers"][(model, rank, lane)] = dict(pid=worker["pid"], gpu_uuid=worker["gpu_uuid"])
        result["topology"] = verify_workers(root / "out", 2, 2)
        workers = result["workers"]
        require(len({worker["pid"] for worker in workers.values()}) == 8, "worker_pids_not_distinct")
        devices = [workers[(0, rank, lane)]["gpu_uuid"] for rank in (0, 1) for lane in (0, 1)]
        require(len(set(devices)) == 4 and all(workers[(0, rank, lane)]["gpu_uuid"] == workers[(1, rank, lane)]["gpu_uuid"]
                                             for rank in (0, 1) for lane in (0, 1)), "gpu_colocation_differs")
        with (Path(job_root) / "gpus.csv").open() as handle:
            gpus = list(csv.reader(handle, skipinitialspace=True))
        require(len(gpus) == 4 and {row[1] for row in gpus} == set(devices)
                and all("A800" in row[2] and float(row[4]) >= 70000 for row in gpus), "allocation_gpu_inventory_differs")
        budget = reader.load("out/budget.json")
        require(same(budget["margin_gib"], 3.0) and len(budget["budget_gib"]) == 2
                and same(budget["natural_budget_gib"], budget["budget_gib"])
                and all(number(value) and value > 0 for value in budget["budget_gib"]), "budget_conditions_differ")
        result["physical_conditions"] = dict(clock=front["clock_id"], gpu_lanes=devices,
                                             budget_gib=budget["budget_gib"], margin_gib=budget["margin_gib"], checkpoint=checkpoint)
        require(reader.load("out/calibration/dataplane.json")["status"] == "passed", "dataplane_failed")
        phases = reader.load("out/phases.json")["phases"]
        require(not any(row["status"] == "failed" for row in phases)
                and any(row["phase"] == "job" and row["status"] == "passed" for row in phases), "main_phases_failed")
        startup = [row for row in phases if row["phase"] == "startup" and row["status"] == "passed"]
        require(len(startup) == 1 and number(startup[0]["wall"])
                and process["started_at"] <= startup[0]["wall"] <= process["finished_at"], "startup_time_missing")
        result["startup_wall_s"] = startup[0]["wall"] - process["started_at"]
        result["condition_wall_s"] = process["finished_at"] - process["started_at"]
        plan = reader.load("out/calibration/imbalance/plan.json")
        config = read(configs[f"{block}-{mode}"])
        cases = scan_cases(config, 2, 2)
        require(same(plan, dict(config=config, cases=cases)), "scan_plan_differs")
        summary = reader.load("out/calibration/imbalance/summary.json")
        require(summary["status"] == "passed" and summary["stop_reason"] == "completed" and summary["failure"] is None
                and summary["planned"] == summary["completed"] == summary["measured"] == 4
                and summary["remaining_cases"] == [] and summary["case_timeout_s"] == 150 and summary["max_scan_s"] == 900,
                "scan_summary_incomplete")
        for filename in ("results.json", "case-statuses.json"):
            rows = reader.load("out/calibration/imbalance/" + filename)
            require([row["name"] for row in rows] == [case["name"] for case in cases], "scan_case_order_differs")
            for row in rows:
                require(same(row, reader.load(f"out/calibration/imbalance/{row['name']}-status.json")), "case_summary_differs")
        require(same(reader.load("out/calibration/imbalance/warmup.json"),
                     reader.load("out/calibration/imbalance/warmup-status.json")), "warmup_summary_differs")
        result["resource_usage"] = process_usage(root / "process-usage.jsonl", process["pid"])
        reader.hashes["process-usage.jsonl"] = digest(root / "process-usage.jsonl")
        require(result["resource_usage"]["status"] == "sampled", "process_usage_missing_or_invalid")
        result["status"] = "passed"
    except ERRORS as error:
        result["errors"].append(str(error))
    result.setdefault("resource_usage", dict(status="unavailable", exact_cumulative_cpu=False, true_peak_rss=False))
    result["input_sha256"] = reader.hashes
    return result


def audit_block(root, block, bundle, manifest, manifest_sha, protocol, configs, startup_ids=None):
    reader = ArtifactReader(root)
    result = dict(status="not_passed", errors=[], conditions={})
    checkpoint = None
    try:
        for relative in ("exit-status", "inputs/deployment-manifest.json", "startup-tests.xml", "gpus.csv"):
            reader.hashes[relative] = digest(Path(root) / relative)
        require((Path(root) / "exit-status").read_text().strip() == "0", "job_exit_not_zero")
        require(digest(Path(root) / "inputs/deployment-manifest.json") == manifest_sha, "manifest_differs")
        check_tree(Path(root) / "runtime", manifest["runtime_sha256"])
        check_tree(Path(root) / "tests", {str(path.relative_to(Path(bundle) / "tests")): digest(path)
                                         for path in (Path(bundle) / "tests").rglob("*.py")})
        for filename in (Path(bundle) / "inputs").iterdir():
            if filename.is_file():
                require(digest(Path(root) / "inputs" / filename.name) == digest(filename), "archived_input_differs")
        startup = element_tree.parse(Path(root) / "startup-tests.xml")
        suites = list(startup.getroot().iter("testsuite"))
        actual_ids = sorted((row.attrib.get("classname"), row.attrib.get("name")) for row in startup.getroot().iter("testcase"))
        require(len(set(actual_ids)) == 238 and all(all(isinstance(value, str) and value for value in identity)
                                                   for identity in actual_ids), "startup_identity_invalid")
        if startup_ids is not None:
            require(actual_ids == sorted(tuple(identity) for identity in startup_ids), "startup_node_ids_differ")
        require(suites and len(list(startup.getroot().iter("testcase"))) == 238
                and not any(list(startup.getroot().iter(key)) for key in ("failure", "error", "skipped"))
                and sum(int(suite.attrib.get("tests", "0")) for suite in suites) == 238
                and all(int(suite.attrib.get(key, "0")) == 0 for suite in suites for key in ("errors", "failures", "skipped")),
                "startup_selection_failed_or_changed")
        status = reader.load("out/scan-overhead-block/status.json")
        result["raw_status"] = status
        require(status["block"] == block and type(status["block"]) is int
                and status["status"] == "collected_not_overhead_accepted" and status["not_executed"] == []
                and status["planned"] == status["started"] == list(BLOCKS[block])
                and [row["mode"] for row in status["completed"]] == list(BLOCKS[block]), "block_incomplete_or_order_changed")
        require(same(reader.load("out/scan-overhead-block/protocol.json"), protocol)
                and status["protocol_sha256"] == manifest["protocol_sha256"]
                and same(status["inputs"]["common"], protocol["common_args"]), "driver_protocol_differs")
        checkpoint = status["inputs"]["checkpoint"]
        require(isinstance(checkpoint, str) and checkpoint, "checkpoint_missing")
        for row in status["completed"]:
            require(same({key: value for key, value in row.items() if key != "mode"},
                         reader.load(f"out/scan-overhead-block/{row['mode']}/process.json")), "driver_process_receipt_differs")
    except ERRORS as error:
        result["errors"].append(str(error))
    for mode in BLOCKS[block]:
        result["conditions"][mode] = audit_condition(root, block, mode, manifest, manifest_sha, protocol, configs, checkpoint)
    try:
        first, second = (result["conditions"][mode] for mode in BLOCKS[block])
        require(all(row["status"] == "passed" for row in (first, second)), "condition_audit_failed")
        require(same(first["physical_conditions"], second["physical_conditions"]), "paired_physical_conditions_differ")
        require(first["process"]["pid"] != second["process"]["pid"]
                and first["process"]["finished_at"] <= second["process"]["started_at"], "engines_not_restarted_in_order")
        require(float(result["raw_status"]["allocation_started_at"]) <= first["process"]["started_at"]
                and second["process"]["finished_at"] - float(result["raw_status"]["allocation_started_at"]) <= 3300,
                "allocation_soft_deadline_exceeded")
        require(all(row["process"]["started_at"] - float(result["raw_status"]["allocation_started_at"])
                    <= protocol["allocation_soft_deadline_s"] - protocol["minimum_remaining_to_start_condition_s"]
                    for row in (first, second)), "condition_started_without_minimum_remaining_time")
    except ERRORS as error:
        result["errors"].append(str(error))
    result["status"] = "passed" if not result["errors"] else "not_passed"
    result["input_sha256"] = reader.hashes
    return result


def paired_overhead(cases, blocks):
    result = dict(planned_pairs=8, available_pairs=0, pairs=[], metrics={}, passed=False)
    for block in (0, 1):
        for name in sorted({row["case"]["name"] for row in cases if row["block"] == block}):
            rows = {row["mode"]: row for row in cases if row["block"] == block and row["case"]["name"] == name}
            pair = dict(block=block, case=name, available=False, errors=[], loss={key: None for key in METRICS})
            try:
                require(blocks[str(block)]["status"] == "passed", "block_audit_failed")
                require(set(rows) == {"off", "collect"} and all(row["available"] for row in rows.values()), "case_pair_unavailable")
                require(same(rows["off"]["request_identity"], rows["collect"]["request_identity"]), "pair_work_differs")
                for key in METRICS:
                    baseline, collected = rows["off"]["metrics"][key], rows["collect"]["metrics"][key]
                    require(number(baseline) and number(collected) and baseline > 0 and collected > 0, "pair_metric_invalid")
                    pair["loss"][key] = float(Decimal(str(collected)) / Decimal(str(baseline)) - 1)
                pair["available"] = True
                result["available_pairs"] += 1
            except ERRORS as error:
                pair["errors"].append(str(error))
            result["pairs"].append(pair)
    for metric in METRICS:
        losses = [row["loss"][metric] for row in result["pairs"] if row["available"]]
        median, worst = (statistics.median(losses), max(losses)) if losses else (None, None)
        result["metrics"][metric] = dict(available_pairs=len(losses), median_loss=median, worst_loss=worst,
                                          passed=len(losses) == 8 and median <= 0.05 and worst <= 0.15)
    result["coverage"] = result["available_pairs"] / 8
    result["passed"] = len(result["pairs"]) == 8 and all(row["passed"] for row in result["metrics"].values())
    return result


def analyze_jobs(jobs, bundle, output, manifest_sha256, *, startup_ids=None):
    bundle, output = Path(bundle), Path(output)
    require(set(jobs) <= {0, 1}, "unknown_block")
    require(len({Path(root).resolve() for root in jobs.values()}) == len(jobs), "duplicate_job_root")
    require(digest(bundle / "inputs/deployment-manifest.json") == manifest_sha256, "reference_manifest_hash_differs")
    manifest = read(bundle / "inputs/deployment-manifest.json")
    check_tree(bundle / "runtime", manifest["runtime_sha256"])
    protocol, configs = load_protocol(bundle / "inputs/graph-overhead-protocol.json", manifest["protocol_sha256"])
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema=1, purpose=protocol["purpose"], source_manifest_sha256=manifest_sha256,
                  protocol_sha256=manifest["protocol_sha256"], blocks={}, cases=[], warmups=[], errors=[],
                  independent_slurm_archive_acceptance_required=True, prediction_gate_passed=False, system_gain_gate=False,
                  gpu_completion_observed=False, complete_93_point_scan_validated=False)
    for block in (0, 1):
        root = Path(jobs[block]) if block in jobs else output / f"missing-block{block}"
        audited = audit_block(root, block, bundle, manifest, manifest_sha256, protocol, configs, startup_ids)
        report["blocks"][str(block)] = audited
        for mode in BLOCKS[block]:
            condition = audited["conditions"][mode]
            cases = scan_cases(read(configs[f"{block}-{mode}"]), 2, 2)
            scan_root = root / "out/scan-overhead-block" / mode / "out/calibration/imbalance"
            for case in [warmup_case(cases), *cases]:
                result = analyze_case(scan_root, case, mode, condition["workers"], condition["clock"])
                result["block"] = block
                report["warmups" if case["name"] == "warmup" else "cases"].append(result)
            if not report["warmups"][-1]["available"]:
                audited["errors"].append(f"{mode}_warmup_unavailable")
                audited["status"] = "not_passed"
    report["overhead"] = paired_overhead(report["cases"], report["blocks"])
    checkpoints = {condition["physical_conditions"]["checkpoint"] for block in report["blocks"].values()
                   for condition in block["conditions"].values() if condition["physical_conditions"] is not None}
    if len(checkpoints) != 1:
        report["errors"].append("cross_block_checkpoint_missing_or_changed")
        report["overhead"]["passed"] = False
    report["status"] = "artifact_overhead_passed_pending_slurm_acceptance" if report["overhead"]["passed"] else "not_passed"
    report["request_coverage"] = dict(planned=3072, valid=sum(len(row["requests"]) for row in report["cases"] if row["available"]))
    for block in report["blocks"].values():
        for condition in block["conditions"].values():
            condition["workers"] = [dict(model=key[0], dp_rank=key[1], tp_rank=key[2], **worker)
                                      for key, worker in sorted(condition["workers"].items())]
    report["analysis_sources"] = analysis_sources()
    write(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Audit both fixed graph-observation order blocks; exit zero is not a passed gate")
    parser.add_argument("--job", action="append", default=[], help="0=job-root or 1=job-root")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--analysis-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jobs = {}
    for value in args.job:
        block, root = value.split("=", 1)
        require(block in ("0", "1") and int(block) not in jobs, "duplicate_or_unknown_block")
        jobs[int(block)] = Path(root)
    rules = read(args.analysis_protocol)
    require(rules["schema"] == 1 and rules["deployment_manifest_sha256"] == args.manifest_sha256
            and rules["startup_expected"] == 238 and len(rules["startup_case_ids"]) == 238,
            "analysis_protocol_identity_differs")
    require(rules["deployment_protocol_sha256"] == digest(args.bundle / "inputs/graph-overhead-protocol.json"),
            "analysis_deployment_protocol_differs")
    for relative, expected in rules["analysis_sources"].items():
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts and path.parts[0] in ("runtime", "tools"),
                "analysis_source_path_invalid")
        require(digest(Path(__file__).parents[1] / path) == expected, "analysis_source_changed:" + relative)
    require(rules["analysis_sources"] == analysis_sources(), "analysis_dependency_set_not_frozen")
    result = analyze_jobs(jobs, args.bundle, args.output, args.manifest_sha256, startup_ids=rules["startup_case_ids"])
    write(args.output / "analysis-protocol.json", rules)
    print(json.dumps(dict(status=result["status"], overhead=result["overhead"], output=str(args.output)), allow_nan=False))


if __name__ == "__main__":
    main()
