"""Offline artifact audit for the frozen TP2 mechanism pilot, never a cost gate."""
import argparse
import datetime
import hashlib
import importlib
import json
import math
import re
import statistics
import sys
import tarfile
from pathlib import Path


CONDITIONS = ("timing_off", "timing_on")
METRICS = ("case_elapsed_s", "code_prefill_s", "code_tpot_s", "conv_whole_mean_tpot_s")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as handle:
        return stream_digest(handle)


def stream_digest(handle):
    checksum = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        checksum.update(chunk)
    return checksum.hexdigest()


def load(path, expected_type=None):
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    def finite_float(value):
        number = float(value)
        require(math.isfinite(number), "Nonfinite JSON number")
        return number
    result = json.loads(Path(path).read_text(), parse_constant=invalid, parse_float=finite_float)
    require(expected_type is None or isinstance(result, expected_type), "Unexpected JSON container type")
    return result


def checked(errors, label, operation):
    try:
        return operation()
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, ImportError, tarfile.TarError) as error:
        errors.append({"check": label, "type": type(error).__name__, "error": str(error)})
        return None


def safe_relative(name):
    path = Path(name)
    require(bool(name) and not path.is_absolute() and ".." not in path.parts, "Unsafe artifact path")
    return path


def verify_archive(evidence, job_id):
    fetch = load(evidence / f"fetch-{job_id}.json")
    archive_path = evidence / safe_relative(fetch["archive"])
    require(digest(archive_path) == fetch["sha256"], "Archive SHA mismatch")
    members = {}
    with tarfile.open(archive_path) as archive:
        for member in archive:
            relative = safe_relative(member.name)
            require(member.isfile() or member.isdir(), "Archive contains a link or special file")
            if member.isdir():
                continue
            require(member.name not in members, "Duplicate archive member")
            with archive.extractfile(member) as handle:
                expected = stream_digest(handle)
            local = evidence / relative
            require(not local.is_symlink() and local.resolve().is_relative_to(evidence.resolve()),
                    "Extracted artifact escapes evidence root")
            require(digest(local) == expected, "Extracted artifact differs: " + member.name)
            members[member.name] = expected
    prefix = f"job-{job_id}/"
    actual = {str(path.relative_to(evidence)) for path in (evidence / f"job-{job_id}").rglob("*") if path.is_file()}
    require(actual and actual == {name for name in members if name.startswith(prefix)},
            "Extracted job files missing from or added to archive")
    return {"sha256": fetch["sha256"], "files": len(members), "member_sha256": members}


def verify_frozen(staging, job, manifest_sha256, tarball_sha256, plan_sha256):
    upload = staging / "upload"
    manifest = upload / "MANIFEST.sha256"
    require(digest(manifest) == manifest_sha256, "Frozen manifest SHA mismatch")
    require(digest(staging / "upload.tar.gz") == tarball_sha256, "Frozen tarball SHA mismatch")
    sources = {}
    for line in manifest.read_text().splitlines():
        expected, name = line.split(None, 1)
        name = name.lstrip("*")
        relative = safe_relative(name)
        require(name not in sources, "Duplicate frozen manifest entry")
        require(digest(upload / relative) == expected, "Frozen file mismatch: " + name)
        if name != "xplace.sbatch":
            require(digest(job / relative) == expected, "Job differs from frozen file: " + name)
        sources[name] = expected
    build = load(staging / "build.json")
    require(len(sources) == build["files"], "Unexpected frozen file count")
    require(build["manifest_sha256"] == manifest_sha256 and build["tarball_sha256"] == tarball_sha256,
            "Build record differs from external SHA anchors")
    plan_names = [name for name, checksum in sources.items()
                  if name.startswith("inputs/") and name.endswith(".json") and checksum == plan_sha256]
    require(len(plan_names) == 1, "Frozen observation plan SHA mismatch or duplicate plan")
    plan = load(upload / plan_names[0])
    if "nested_timing" in plan:
        require({"runtime/nested_timing.py", "tests/test_nested_timing.py", "inputs/startup-test-contract.json"}
                <= set(sources), "Nested timing source or startup contract missing")
    else:
        require(len(sources) == 39, "Unexpected r3 frozen file count")
    require((job / "manifest-check.log").read_text() == "", "Startup manifest check failed")
    return {"manifest_sha256": manifest_sha256, "tarball_sha256": tarball_sha256,
            "plan_sha256": plan_sha256, "source_sha256": sources}, plan


def frozen_validators(upload):
    runtime = (upload / "runtime").resolve()
    names = ("cost_model", "engine", "controller", "calibrate", "measured_cost", "timing_probe", "timing_calibration")
    if (runtime / "nested_timing.py").exists():
        names += ("nested_timing",)
    for name in names:
        if name in sys.modules:
            require(Path(sys.modules[name].__file__).resolve() == runtime / f"{name}.py",
                    "Start a clean process; validator module is already loaded from another tree")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(runtime))
    return {name: importlib.import_module(name) for name in names}


def startup_test_summary(job, nested=False):
    expected = (218, 1)
    if nested:
        contract = load(job / "inputs" / "startup-test-contract.json")
        require(contract.get("schema") == 1 and contract.get("non_timesliced_only") is True,
                "Invalid nested startup test contract")
        nodes = contract["nodes"]
        require(isinstance(nodes, list) and nodes and len(nodes) == len(set(nodes))
                and all(isinstance(node, str) and re.fullmatch(r"tests/test_[a-z0-9_]+\.py(?:::test_[a-z0-9_]+)?", node)
                        for node in nodes), "Invalid explicit startup test nodes")
        require("tests/test_general.py" not in nodes and "tests/test_runtime_fake.py" not in nodes,
                "Mixed time-sliced files may not run wholesale")
        expected = (contract["passed"], contract["skipped"])
        require(type(expected[0]) is int and expected[0] > 0 and type(expected[1]) is int and expected[1] == 0,
                "Unexpected startup test counts")
        require((job / "startup-test-nodes.txt").read_text().splitlines() == nodes, "Executed startup test list differs")
    pattern = r"(\d+) passed(?:, (\d+) skipped)?(?:, \d+ warnings?)? in ([\d.]+)s(?: \([0-9:]+\))?"
    matches = [match.groups() for line in (job / "startup-tests.log").read_text().splitlines()
               for match in [re.fullmatch(pattern, line.strip("= "))] if match]
    require(len(matches) == 1 and (int(matches[0][0]), int(matches[0][1] or 0)) == expected,
            "Unexpected startup test summary")
    return {"passed": expected[0], "skipped": expected[1], "duration_s": float(matches[0][2])}


def lifecycle(job, validators, plan=None):
    require((job / "exit-status").read_text().strip() == "0", "Nonzero job exit status")
    nested = plan is not None and "nested_timing" in plan
    summary = startup_test_summary(job, nested=nested)
    phases = load(job / "out" / "phases.json")
    require({entry["phase"] for entry in phases["phases"]} == {"startup", "dataplane", "observability", "job"},
            "Unexpected phase records")
    records = validators["measured_cost"].verified_phases(phases, 2, 2, {"dataplane", "observability"})
    require(phases["args"]["phases"] == "dataplane,observability", "Unexpected job phases")
    require(phases["args"]["enable_tp_migration"] == "True", "TP migration disabled")
    require(records["dataplane"]["migrations"] == 8, "Wrong migration count")
    interface = load(job / "out" / "timing-interface-check.json")
    expected_sources = (validators["timing_probe"].interface_sources(nested=True) if nested
                        else validators["timing_probe"].INTERFACE_SHA256)
    require(interface == {"status": "passed", "source_sha256": expected_sources},
            "Deployed vLLM interface SHA evidence mismatch")
    gpu_lines = (job / "gpus.csv").read_text().strip().splitlines()
    require(len(gpu_lines) == 4 and all("A800" in line for line in gpu_lines), "Wrong GPU budget")
    uuids = {line.split(",")[1].strip().removeprefix("GPU-") for line in gpu_lines}
    require(uuids == set(records["startup"]["topology"]["gpu_uuids"]), "GPU identity mismatch")
    skip_sites = [line.strip() for path in (job / "tests").glob("*.py") for line in path.read_text().splitlines()
                  if "pytest.skip(" in line or "importorskip(" in line]
    require(len(skip_sites) == 1 and "analysis tools run locally" in skip_sites[0], "Unexpected test skip scope")
    started = datetime.datetime.fromisoformat((job / "started-at").read_text().strip())
    finished = datetime.datetime.fromisoformat((job / "finished-at").read_text().strip())
    require(0 < (finished - started).total_seconds() <= 3600, "Job lifecycle exceeds one-hour budget")
    return {"phases": phases, "startup_tests": summary,
            "started_at": (job / "started-at").read_text().strip(),
            "finished_at": (job / "finished-at").read_text().strip(), "node": (job / "nodelist").read_text().strip()}


def dataplane(job):
    record = load(job / "out" / "calibration" / "dataplane.json")
    transactions, moves, requests = record["transactions"], record["migrations"], record["requests"]
    require(record["status"] == "passed" and len(transactions) == 2 and len(moves) == 8, "Incomplete dataplane")
    require([item["path"] for item in transactions] == ["dataplane_out", "dataplane_back"], "Wrong migration order")
    require(all(item["status"] == "completed" and not item.get("move_failure") and len(item["moves"]) == 4
                for item in transactions), "Failed migration transaction")
    require([move for item in transactions for move in item["moves"]] == moves, "Migration event mismatch")
    for move in moves:
        require(move["tensor_parallel_size"] == 2 and move["bitwise_verified"] is True, "Missing TP2 integrity proof")
        require(move["model"] == 1 and {move["source"], move["destination"]} == {0, 1}, "Wrong migration endpoints")
        require(move["elapsed_s"] > 0 and move["kv_bytes"] > 0, "Invalid migration cost")
        for role in ("source", "destination"):
            shards = move["worker_shards"][role]
            require(len(shards) == 2 and {shard["tp_rank"] for shard in shards} == {0, 1}, "Missing TP lane")
            require(sum(shard["bytes"] for shard in shards) == move["kv_bytes"], "Incomplete shard byte count")
            require(all(shard["role"] == role and shard["rank"] == move[role] and shard["tp_size"] == 2
                        and shard["num_blocks"] == move["num_blocks"] and shard["bitwise_verified"] is True
                        and shard["integrity"] == "device_echo_full_bytewise" for shard in shards), "Invalid shard proof")
    require(len(requests) == 4 and {request["request_id"] for request in requests} ==
            {f"dp_{rank}_{index}" for rank in range(2) for index in range(2)}, "Missing dataplane request")
    require(all(request["output_tokens"] == 2500 and re.fullmatch(r"[0-9a-f]{64}", request["token_sha256"])
                for request in requests), "Incomplete dataplane outputs")
    elapsed = [move["elapsed_s"] for move in moves]
    return {"moves": 8, "requests": 4, "kv_bytes": sum(move["kv_bytes"] for move in moves),
            "elapsed_s": {"min": min(elapsed), "median": statistics.median(elapsed), "max": max(elapsed)},
            "scope": "Short-context all-shard copy evidence, not long-context or cost validation"}


def request_checks(requests, row, case):
    require(len(requests) == 1 + sum(case["layout"]), "Missing or extra requests")
    expected_ids = {f"if_code_{case['name']}_{case['repeat']}"} | {
        f"if_{case['name']}_{case['repeat']}_{rank}_{index}"
        for rank, count in enumerate(case["layout"]) for index in range(count)}
    require({request["request_id"] for request in requests} == expected_ids, "Request IDs differ from fixed plan")
    for request in requests:
        require(request["model"] in (0, 1), "Unknown tenant")
        expected_prompt, expected_output = (101167, 64) if request["model"] == 0 else (4096, 3000)
        require((request["prompt_tokens"], request["output_tokens"]) == (expected_prompt, expected_output),
                "Original workload lengths changed")
        require(re.fullmatch(r"[0-9a-f]{64}", request["token_sha256"]) is not None, "Missing output checksum")
        stamps = [request[name] for name in ("arrived_at", "submitted_at", "first_token_at", "completed_at")]
        require(all(math.isfinite(stamp) for stamp in stamps) and stamps == sorted(stamps), "Invalid request times")
        require(row["measurement_started_ns"] / 1e9 <= stamps[1] <= stamps[-1] <= row["measurement_ended_ns"] / 1e9,
                "Request outside measured case window")
    elapsed = (row["measurement_ended_ns"] - row["measurement_started_ns"]) / 1e9
    require(elapsed > 0 and math.isclose(row["case_elapsed_s"], elapsed, rel_tol=1e-9), "Case elapsed mismatch")
    conv = [request for request in requests if request["model"] == 1]
    if conv:
        observed = statistics.mean((request["completed_at"] - request["first_token_at"]) /
                                   (request["output_tokens"] - 1) for request in conv)
        require(math.isclose(row["conv_whole_mean_tpot_s"], observed, rel_tol=1e-9), "Whole Conv TPOT mismatch")
    return len(requests)


def telemetry(case_root, case, row, validator, source_sha256, nested=False):
    stop = load(case_root / "timing-stop.json")
    start = load(case_root / "timing-start.json")
    for command, record in (("start", start), ("stop", stop)):
        require(record["case"] == case["case"] and record["command"] == command and not record["failures"],
                "Failed or mismatched timing control")
        require(len(record["results"]) == 4, "Incomplete timing control acknowledgments")
    frontend = load(case_root / "timing-frontend.json")
    result = validator(stop["results"], frontend, case["case"], row["measurement_started_ns"], row["measurement_ended_ns"],
                       **({"nested": True} if nested else {}))
    require(load(case_root / "timing-validation.json") == result, "Persisted telemetry validation differs")
    records = [frontend]
    for before, after in zip(start["results"], stop["results"]):
        initial = [before["engine"], *before["workers"]]
        final = [after["engine"], *after["workers"]]
        require(len(initial) == len(final) == 3, "Missing timing start lane")
        for first, last in zip(initial, final):
            require(all(first[name] == last[name] for name in ("case", "identity", "started_ns")),
                    "Timing start/stop identity mismatch")
        records.extend(final)
    require(all(record["limit"] == 100000 and record["source_sha256"] == source_sha256 for record in records),
            "Timing source SHA or event bound mismatch")
    for boundary in ("start", "stop"):
        idle = load(case_root / f"idle-{boundary}.json")
        require(idle and idle[-1]["idle"] == [True] * 4, "Timing boundary was not idle on every engine")
    heartbeats = {}
    for kind in ("async_heartbeat", "thread_heartbeat"):
        values = [event["lateness_ns"] / 1e9 for event in frontend["events"] if event["kind"] == kind]
        require(values and all(math.isfinite(value) and value >= 0 for value in values), "Invalid heartbeat lateness")
        heartbeats[kind] = {"count": len(values), "max_s": max(values)}
    gpu_uuids = sorted({worker["identity"]["gpu_uuid"] for group in stop["results"] for worker in group["workers"]})
    return dict(result, heartbeat=heartbeats, gpu_uuids=gpu_uuids, timing_semantics="host boundaries, not GPU completion")


def paired_metrics(rows):
    pairs = []
    for repeat in range(3):
        for count in (0, 2, 8, 16):
            key = f"bal{count}-r{repeat}"
            pair = {"case": key}
            for condition in CONDITIONS:
                row = rows[condition].get(key, {})
                pair[condition] = {name: row.get(name) for name in METRICS}
            pair["relative_change_on_over_off"] = {}
            for name in METRICS:
                before, after = pair["timing_off"][name], pair["timing_on"][name]
                valid = all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                            for value in (before, after))
                pair["relative_change_on_over_off"][name] = after / before - 1 if valid else None
            pairs.append(pair)
    return pairs


def audit_matrix(root, plan, validators, source_sha256):
    errors, cases = [], []
    rows = {condition: {} for condition in CONDITIONS}
    raw = {condition: {} for condition in CONDITIONS}
    checked(errors, "plan", lambda: require(load(root / "plan.json") == plan ==
                                           validators["timing_calibration"].observation_plan(
                                               **({"nested": True} if "nested_timing" in plan else {})), "Frozen plan mismatch"))
    expected_cases = [case["case"] for case in plan["cases"]]
    expected_directories = {f"{case['condition']}/{case['name']}-r{case['repeat']}" for case in plan["cases"]}
    observed_directories = {str(path.relative_to(root)) for condition in CONDITIONS
                            for path in (root / condition).glob("*") if path.is_dir()}
    checked(errors, "case_directories", lambda: require(observed_directories == expected_directories,
                                                       "Missing or extra case directories"))
    for case in plan["cases"]:
        case_errors = []
        key = f"{case['name']}-r{case['repeat']}"
        case_root = root / case["condition"] / key
        entry = {"case": case["case"], "errors": case_errors,
                 "failure_artifacts": [str(path.relative_to(root)) for path in case_root.glob("*error*.json")]}
        partial = case_root / "failed-requests.json"
        if partial.exists():
            entry["failure_artifacts"].append(str(partial.relative_to(root)))
        checked(case_errors, "failure_artifacts", lambda: require(not entry["failure_artifacts"], "Case has failure artifacts"))
        checked(case_errors, "descriptor", lambda: require(load(case_root / "case.json") == case, "Case descriptor mismatch"))
        row = checked(case_errors, "measurement", lambda: load(case_root / "measurement.json", dict))
        requests = checked(case_errors, "requests", lambda: load(case_root / "requests.json", list))
        if row is not None:
            rows[case["condition"]][key] = row
            checked(case_errors, "layout", lambda: require(all(row[name] == case[name] for name in
                    ("name", "repeat", "layout", "condition")), "Case layout/condition mismatch"))
            checked(case_errors, "schema_preemptions", lambda: require(row["measurement_schema"] == 2 and
                    row["preemptions"] == 0, "Wrong measurement schema or nonzero preemptions"))
        if requests is not None:
            raw[case["condition"]][key] = requests
        if row is not None and requests is not None:
            entry["requests"] = checked(case_errors, "requests_contract", lambda: request_checks(requests, row, case))
        if case["condition"] == "timing_on" and row is not None:
            entry["telemetry"] = checked(case_errors, "telemetry", lambda: telemetry(
                case_root, case, row, validators["timing_probe"].validate_timing, source_sha256,
                nested="nested_timing" in plan))
        elif case["condition"] == "timing_off":
            checked(case_errors, "timing_disabled", lambda: require(not list(case_root.glob("timing-*.json")),
                                                                     "Timing-off case contains enabled telemetry"))
        entry["status"] = "failed" if case_errors else "passed"
        cases.append(entry)
    conditions = {}
    for condition in CONDITIONS:
        condition_errors = []
        ordered_rows = list(rows[condition].values())
        samples = checked(condition_errors, "raw_schema2", lambda: validators["measured_cost"].interference_samples(
            ordered_rows, raw[condition], 2, plan["raw_contract"]))
        checked(condition_errors, "aggregate", lambda: require(load(root / f"{condition}-interference.json") == ordered_rows,
                                                                "Aggregate differs from per-case measurements"))
        conditions[condition] = {"cases": len(ordered_rows), "requests": sum(len(value) for value in raw[condition].values()),
                                 "samples": len(samples) if samples is not None else None, "errors": condition_errors,
                                 "status": "failed" if condition_errors else "passed"}
    def chronology():
        previous_end = None
        for case in plan["cases"]:
            row = rows[case["condition"]][f"{case['name']}-r{case['repeat']}"]
            require(previous_end is None or row["measurement_started_ns"] > previous_end,
                    "Case time windows overlap or differ from frozen execution order")
            previous_end = row["measurement_ended_ns"]
    checked(errors, "chronology", chronology)
    def completeness():
        status, progress = load(root / "status.json"), load(root / "progress.json")
        require(status["status"] == "passed" and status["completed"] == expected_cases and
                not status.get("not_completed") and status["independent_prediction_gate"] is False, "Incomplete pilot status")
        require(progress == {"completed": expected_cases, "planned": 24}, "Progress does not cover fixed ordered matrix")
        validation = {condition: {name: conditions[condition][name] for name in ("status", "cases", "samples")}
                      for condition in CONDITIONS}
        require(load(root / "raw-validation.json") == status["conditions"] == validation, "Raw validation report mismatch")
        require(sum(value["requests"] for value in conditions.values()) == 336, "Pilot request count mismatch")
    checked(errors, "completeness", completeness)
    pairs = paired_metrics(rows)
    def pair_consistency():
        saved = load(root / "paired-overhead.json")
        expected = [{name: pair[name] for name in ("case", *CONDITIONS)} for pair in pairs]
        require(saved["pairs"] == expected and saved["performance_claim"] is False, "Persisted paired overhead mismatch")
    checked(errors, "paired_overhead", pair_consistency)
    failed = bool(errors or any(case["errors"] for case in cases) or any(value["errors"] for value in conditions.values()))
    return {"status": "failed" if failed else "passed", "errors": errors, "cases": cases, "conditions": conditions,
            "pairs": pairs, "performance_claim": False, "independent_prediction_gate": False}


def input_checks(job, report):
    root = job / "out" / "calibration" / "observability"
    inputs = load(root / "inputs.json")
    bank = load(job / "inputs" / "prompt-bank.json")["prompts"]
    key = sorted(name for name in bank if name.startswith("swe-"))[0]
    prompt = bank[key]["token_ids"]
    require(len(prompt) == inputs["code_prompt_tokens"] == 101167 and inputs["code_prompt_sha256"] ==
            hashlib.sha256(json.dumps(prompt).encode()).hexdigest(), "Original Code prompt identity mismatch")
    remaining = inputs["deadline_monotonic_s"] - inputs["phase_entry_monotonic_s"]
    require(0 < remaining <= 3300 and float(inputs["sbatch_entry_wall_s"]) > 0, "Invalid allocation time budget")
    for case in report["observability"]["cases"]:
        if case.get("telemetry"):
            topology = next(phase["topology"] for phase in report["lifecycle"]["phases"]["phases"]
                            if phase["phase"] == "startup" and phase["status"] == "passed")
            expected = {value.removeprefix("GPU-") for value in topology["gpu_uuids"]}
            require({value.removeprefix("GPU-") for value in case["telemetry"]["gpu_uuids"]} == expected,
                    "Telemetry GPU set differs from startup topology")
    for condition in CONDITIONS:
        for path in (root / condition).glob("*/measurement.json"):
            row = load(path)
            require(inputs["phase_entry_monotonic_s"] <= row["measurement_started_ns"] / 1e9 <
                    row["measurement_ended_ns"] / 1e9 <= inputs["deadline_monotonic_s"], "Case outside allocation budget")
    return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--tarball-sha256", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite audit evidence")
    require(args.job_id.isdigit(), "Invalid job ID")
    job = args.evidence / f"job-{args.job_id}"
    report = {"schema": 1, "job": args.job_id, "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "auditor_sha256": digest(Path(__file__)), "scope": "offline mechanism artifacts only",
              "errors": [], "measured_cost_gate": False, "independent_prediction_gate": False,
              "paired_ablation_gate": False, "mixed_topology_gate": False}
    report["archive"] = checked(report["errors"], "archive", lambda: verify_archive(args.evidence, args.job_id))
    frozen = checked(report["errors"], "frozen_sources", lambda: verify_frozen(
        args.staging, job, args.manifest_sha256, args.tarball_sha256, args.plan_sha256))
    if frozen is not None and report["archive"] is not None:
        report["frozen"], plan = frozen
        validators = checked(report["errors"], "validator_imports", lambda: frozen_validators(args.staging / "upload"))
        if validators is not None:
            report["lifecycle"] = checked(report["errors"], "lifecycle", lambda: lifecycle(job, validators, plan))
            report["dataplane"] = checked(report["errors"], "dataplane", lambda: dataplane(job))
            report["observability"] = checked(report["errors"], "matrix", lambda: audit_matrix(
                job / "out" / "calibration" / "observability", plan, validators,
                report["frozen"]["source_sha256"]["runtime/timing_probe.py"]))
            report["inputs"] = checked(report["errors"], "inputs", lambda: input_checks(job, report))
    failed = bool(report["errors"] or (report.get("observability") or {}).get("status") != "passed")
    report["status"] = "failed" if failed else "artifact_checks_passed_not_predictive_validation"
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "path": str(args.output), "sha256": digest(args.output)}))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
