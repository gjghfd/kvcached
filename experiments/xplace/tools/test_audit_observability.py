import copy
import io
import json
import math
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

import measured_cost
import timing_calibration
import timing_probe
import nested_timing
from audit_observability import audit_matrix, digest, load, paired_metrics, startup_test_summary, verify_archive
from calibrate import window_rate
from test_measured_cost import repeated_interference_fixture


VALIDATORS = {"measured_cost": measured_cost, "timing_probe": timing_probe, "timing_calibration": timing_calibration}
SOURCE_SHA256 = digest(Path(timing_probe.__file__))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def timing_records(case, row, nested=False):
    started = row["measurement_started_ns"] - 1000
    ended = row["measurement_ended_ns"] + 1000
    def record(identity, kinds):
        events = [{"kind": "start", "at_ns": started + 1}, *kinds, {"kind": "stop", "at_ns": ended - 1}]
        for sequence, event in enumerate(events, 1):
            event.update(sequence=sequence, thread=1)
            event.setdefault("at_ns", started + 10 + sequence)
        return {"schema": 1, "case": case, "identity": identity, "limit": 100000, "started_ns": started,
                "ended_ns": ended, "events": events, "attempted": len(events), "exported": len(events),
                "dropped": 0, "late_events": 0, "schema_errors": 0, "source_sha256": SOURCE_SHA256}
    frontend = record({"role": "frontend", "pid": 1}, [{"kind": kind, "lateness_ns": 20}
                                                       for kind in ("async_heartbeat", "thread_heartbeat")])
    reports = []
    for model in range(2):
        for rank in range(2):
            identity = {"service": f"instance-{model + 1}", "dp_rank": rank, "pid": 10 + model * 10 + rank * 3}
            engine = record(dict(identity, role="engine"), [{"kind": "schedule:return", "scheduled_tokens": 1}])
            workers = [record(dict(identity, role="worker", tp_rank=lane, pid=identity["pid"] + lane + 1,
                                   gpu_uuid=f"gpu-{rank}-{lane}"), [{"kind": "worker_execute:enter", "scheduled_tokens": 1}])
                       for lane in range(2)]
            if nested:
                workers = []
                for lane in range(2):
                    events = []
                    for kind, scope_id, parent in (
                        ("worker_execute:enter", 2, None), ("dp_coordinate:enter", 3, 2),
                        ("dp_coordinate:return", 3, 2), ("worker_execute:return", 2, None),
                        ("worker_sample:enter", 6, None), ("runner_sample:enter", 7, 6),
                        ("runner_sample:return", 7, 6), ("runner_bookkeeping:enter", 9, 6),
                        ("runner_to_list:enter", 10, 9), ("runner_to_list:return", 10, 9),
                        ("runner_bookkeeping:return", 9, 6), ("worker_sample:return", 6, None),
                    ):
                        events.append({"kind": kind, "scope_id": scope_id, "parent_scope_id": parent})
                    events[0]["scheduled_tokens"] = 1
                    events[4]["runner_state_present"] = True
                    worker = record(dict(identity, role="worker", tp_rank=lane, pid=identity["pid"] + lane + 1,
                                         gpu_uuid=f"gpu-{rank}-{lane}"), events)
                    groups = {"dp": {"name": "dp:0", "ranks": [lane, lane + 2], "rank": rank, "world_size": 2},
                              "tp": {"name": "tp:0", "ranks": [rank * 2, rank * 2 + 1], "rank": lane, "world_size": 2},
                              "ep": {"name": "ep:0", "ranks": [0, 1, 2, 3], "rank": rank * 2 + lane, "world_size": 4}}
                    worker["nested_timing"] = {"schema": 1, "scopes": list(nested_timing.SCOPES), "restored": True,
                        "source_sha256": nested_timing.source_sha256(), "lookup_module": "vllm.v1.worker.gpu_model_runner",
                        "configuration": {"groups": groups, "async_scheduling": False, "enable_dbo": False,
                            "enable_eplb": False, "speculative_decoding": False, "pipeline_parallel_size": 1,
                            "tensor_parallel_size": 2, "data_parallel_size": 2, "data_parallel_rank": rank,
                            "disable_nccl_for_dp_synchronization": False, "use_sequence_parallel_moe": True,
                            "all2all_backend": "allgather_reducescatter"}}
                    workers.append(worker)
            reports.append({"engine": engine, "workers": workers})
    starts = [{"engine": {key: group["engine"][key] for key in ("case", "identity", "started_ns")},
               "workers": [{key: worker[key] for key in ("case", "identity", "started_ns")} for worker in group["workers"]]}
              for group in reports]
    return starts, reports, frontend


@pytest.mark.parametrize("damage", [None, "count", "skip", "failed", "mixed_file", "executed_nodes"])
def test_nested_startup_contract(tmp_path, damage):
    nodes = ["tests/test_nested_timing.py"]
    contract = {"schema": 1, "non_timesliced_only": True, "nodes": nodes, "passed": 34, "skipped": 0}
    summary = "34 passed, 1 warning in 0.49s"
    if damage == "count":
        summary = "33 passed in 0.49s"
    elif damage == "skip":
        summary = "34 passed, 1 skipped in 0.49s"
    elif damage == "failed":
        summary = "1 failed, 34 passed in 0.49s"
    elif damage == "mixed_file":
        contract["nodes"] = ["tests/test_general.py"]
    save(tmp_path / "inputs/startup-test-contract.json", contract)
    (tmp_path / "startup-test-nodes.txt").write_text("\n".join(nodes if damage != "executed_nodes" else []) + "\n")
    (tmp_path / "startup-tests.log").write_text(summary + "\n")
    if damage is None:
        assert startup_test_summary(tmp_path, nested=True) == {"passed": 34, "skipped": 0, "duration_s": 0.49}
    else:
        with pytest.raises(ValueError):
            startup_test_summary(tmp_path, nested=True)


@pytest.fixture(scope="module", params=[False, True], ids=["outer", "nested"])
def complete_matrix(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("observation-matrix")
    nested = request.param
    plan = timing_calibration.observation_plan(nested=nested)
    original, raw, _ = repeated_interference_fixture()
    base = {measured_cost.interference_key(row): row for row in original}
    rows = {condition: {} for condition in ("timing_off", "timing_on")}
    save(root / "plan.json", plan)
    offset = 0
    for case in plan["cases"]:
        key = measured_cost.interference_key(case)
        row = copy.deepcopy(base[key])
        row["condition"] = case["condition"]
        requests = copy.deepcopy(raw[key])
        for request in requests:
            if request["model"] == 0:
                request.update(request_id=f"if_code_{case['name']}_{case['repeat']}", prompt_tokens=101167)
            else:
                index = request["request_id"].split("-")[-1]
                request["request_id"] = f"if_{case['name']}_{case['repeat']}_{request['rank_initial']}_{index}"
                request.update(arrived_at=89.0, submitted_at=89.0)
                request["cadence"] = request["cadence"][::30] + [request["cadence"][-1]]
            request["token_sha256"] = "a" * 64
        code, conv = requests[0], requests[1:]
        if conv:
            row.update(conv_before_s=window_rate(conv[0], 97, 100),
                       conv_during_prefill_s=window_rate(conv[0], 101, code["first_token_at"] - 1),
                       conv_during_decode_s=window_rate(conv[0], code["first_token_at"], code["completed_at"]),
                       conv_whole_mean_tpot_s=sum((request["completed_at"] - request["first_token_at"]) / 2999
                                                 for request in conv) / len(conv))
        ended = math.ceil(max(request["completed_at"] for request in requests)) + 1
        row.update(measurement_started_ns=(80 + offset) * 10**9, measurement_ended_ns=(ended + offset) * 10**9,
                   case_elapsed_s=ended - 80)
        for request in requests:
            for name in ("arrived_at", "submitted_at", "first_token_at", "completed_at"):
                request[name] += offset
            request["cadence"] = [[stamp + offset, count] for stamp, count in request["cadence"]]
        offset += ended - 80 + 1
        case_root = root / case["condition"] / key
        save(case_root / "case.json", case)
        save(case_root / "measurement.json", row)
        save(case_root / "requests.json", requests)
        rows[case["condition"]][key] = row
        if case["condition"] == "timing_on":
            starts, reports, frontend = timing_records(case["case"], row, nested=nested)
            for command, records in (("start", starts), ("stop", reports)):
                save(case_root / f"timing-{command}.json", {"case": case["case"], "command": command,
                                                           "failures": [], "results": records})
                save(case_root / f"idle-{command}.json", [{"at_ns": 1, "idle": [True] * 4}])
            save(case_root / "timing-frontend.json", frontend)
            save(case_root / "timing-validation.json", timing_probe.validate_timing(
                reports, frontend, case["case"], row["measurement_started_ns"], row["measurement_ended_ns"], nested=nested))
    validation = {condition: {"status": "passed", "cases": 12, "samples": 12} for condition in rows}
    completed = [case["case"] for case in plan["cases"]]
    save(root / "status.json", {"status": "passed", "completed": completed, "conditions": validation,
                               "independent_prediction_gate": False})
    save(root / "progress.json", {"completed": completed, "planned": 24})
    save(root / "raw-validation.json", validation)
    for condition, records in rows.items():
        save(root / f"{condition}-interference.json", list(records.values()))
    pairs = [{name: pair[name] for name in ("case", "timing_off", "timing_on")} for pair in paired_metrics(rows)]
    save(root / "paired-overhead.json", {"pairs": pairs, "performance_claim": False})
    return root, plan


def test_complete_matrix_is_not_a_prediction_gate(complete_matrix):
    root, plan = complete_matrix
    result = audit_matrix(root, plan, VALIDATORS, SOURCE_SHA256)
    assert result["status"] == "passed", result
    assert [result["conditions"][condition]["requests"] for condition in ("timing_off", "timing_on")] == [168, 168]
    assert len(result["cases"]) == 24 and len(result["pairs"]) == 12
    assert all(case["telemetry"]["process_records"] == 13 for case in result["cases"] if "telemetry" in case)
    assert result["independent_prediction_gate"] is False and result["performance_claim"] is False
    assert result["pairs"][0]["relative_change_on_over_off"]["conv_whole_mean_tpot_s"] is None


@pytest.mark.parametrize("damage", ["missing_case", "extra_case", "reordered_plan", "short_code", "short_conv",
                                    "wrong_prompt", "preemption", "missing_lane", "overflow", "false_validation",
                                    "start_pid", "timing_in_off", "source_sha", "bad_limit", "control_failed",
                                    "not_idle", "missing_cadence", "bad_heartbeat", "missing_pair", "changed_overhead",
                                    "partial_failure", "incomplete_status", "malformed_row", "request_window", "overlap",
                                    "worker_nested_source", "worker_scope_parent", "worker_restore"])
def test_bad_cases_are_retained_without_merging_conditions(tmp_path, complete_matrix, damage):
    original, plan = complete_matrix
    root = tmp_path / "matrix"
    shutil.copytree(original, root)
    case_root = root / "timing_on" / "bal2-r0"
    if damage == "missing_case":
        shutil.rmtree(case_root)
    elif damage == "extra_case":
        (root / "timing_on" / "extra").mkdir()
    elif damage == "reordered_plan":
        record = load(root / "plan.json")
        record["cases"].reverse()
        save(root / "plan.json", record)
    elif damage in ("short_code", "short_conv", "wrong_prompt", "missing_cadence"):
        path = case_root / "requests.json"
        records = load(path)
        if damage == "short_code":
            records[0]["output_tokens"] = 63
        elif damage == "short_conv":
            records[1]["output_tokens"] = 2999
        elif damage == "wrong_prompt":
            records[0]["prompt_tokens"] = 101168
        else:
            records[1]["cadence"] = records[1]["cadence"][:1]
        save(path, records)
    elif damage in ("preemption", "request_window", "malformed_row", "overlap"):
        path = case_root / "measurement.json"
        record = load(path)
        if damage == "malformed_row":
            record = []
        elif damage == "overlap":
            record["measurement_started_ns"] = 0
        else:
            if damage == "preemption":
                record["preemptions"] = 1
            else:
                record["measurement_started_ns"] = record["measurement_ended_ns"] - 1
        save(path, record)
    elif damage in ("missing_lane", "overflow", "source_sha", "bad_limit", "control_failed"):
        path = case_root / "timing-stop.json"
        record = load(path)
        if damage == "missing_lane":
            record["results"][0]["workers"].pop()
        elif damage == "control_failed":
            record["failures"] = [{"error": "failure"}]
        else:
            key, value = {"overflow": ("dropped", 1), "source_sha": ("source_sha256", "b" * 64),
                          "bad_limit": ("limit", 99999)}[damage]
            record["results"][0]["engine"][key] = value
        save(path, record)
    elif damage == "false_validation":
        save(case_root / "timing-validation.json", {"status": "passed"})
    elif damage == "start_pid":
        path = case_root / "timing-start.json"
        record = load(path)
        record["results"][0]["engine"]["identity"]["pid"] = 999
        save(path, record)
    elif damage == "timing_in_off":
        save(root / "timing_off" / "bal2-r0" / "timing-start.json", {})
    elif damage == "not_idle":
        save(case_root / "idle-stop.json", [{"idle": [True, True, True, False]}])
    elif damage == "bad_heartbeat":
        path = case_root / "timing-frontend.json"
        record = load(path)
        record["events"][1]["lateness_ns"] = -1
        save(path, record)
    elif damage in ("missing_pair", "changed_overhead"):
        path = root / "paired-overhead.json"
        record = load(path)
        if damage == "missing_pair":
            record["pairs"].pop()
        else:
            record["pairs"][1]["timing_on"]["code_tpot_s"] *= 0.1
        save(path, record)
    elif damage == "partial_failure":
        save(case_root / "failed-requests.json", {"status": "failed", "partial_progress": {}})
    elif damage == "incomplete_status":
        path = root / "status.json"
        record = load(path)
        record["completed"].pop()
        save(path, record)
    elif damage in ("worker_nested_source", "worker_scope_parent", "worker_restore"):
        path = case_root / "timing-stop.json"
        record = load(path)
        worker = record["results"][0]["workers"][0]
        metadata = worker.setdefault("nested_timing", {})
        if damage == "worker_nested_source":
            metadata["source_sha256"] = "0" * 64
        elif damage == "worker_restore":
            metadata["restored"] = False
        else:
            worker["events"][1]["parent_scope_id"] = 999
        save(path, record)
    result = audit_matrix(root, plan, VALIDATORS, SOURCE_SHA256)
    assert result["status"] == "failed", damage
    assert len(result["cases"]) == 24 and len(result["pairs"]) == 12
    assert set(result["conditions"]) == {"timing_off", "timing_on"}
    assert result["independent_prediction_gate"] is False


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_json_is_rejected(tmp_path, value):
    path = tmp_path / "bad.json"
    path.write_text('{"value":' + value + '}')
    with pytest.raises(ValueError, match="Nonfinite"):
        load(path)


@pytest.mark.parametrize("damage", [None, "link", "traversal", "duplicate", "modified", "extra"])
def test_archive_verification(tmp_path, damage):
    archive_path = tmp_path / "job.tar.gz"
    job = tmp_path / "job-123"
    job.mkdir()
    (job / "data.json").write_text("{}")
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(job, arcname="job-123")
        if damage in ("link", "traversal", "duplicate"):
            member = tarfile.TarInfo("../escape" if damage == "traversal" else "job-123/data.json")
            if damage == "link":
                member.type = tarfile.SYMTYPE
                member.linkname = "/etc/passwd"
                archive.addfile(member)
            else:
                member.size = 2
                archive.addfile(member, io.BytesIO(b"{}"))
    save(tmp_path / "fetch-123.json", {"archive": archive_path.name, "sha256": digest(archive_path)})
    if damage == "modified":
        (job / "data.json").write_text('{"modified":true}')
    elif damage == "extra":
        (job / "extra.json").write_text("{}")
    if damage is None:
        assert verify_archive(tmp_path, "123")["files"] == 1
    else:
        with pytest.raises(ValueError):
            verify_archive(tmp_path, "123")


def test_failed_cli_writes_report_but_never_overwrites(tmp_path):
    output = tmp_path / "audit.json"
    command = [sys.executable, str(Path(__file__).with_name("audit_observability.py")), "--evidence", str(tmp_path),
               "--staging", str(tmp_path), "--job-id", "123", "--manifest-sha256", "a" * 64,
               "--tarball-sha256", "b" * 64, "--plan-sha256", "c" * 64, "--output", str(output)]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 1 and load(output)["status"] == "failed"
    checksum = digest(output)
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0 and "Refusing to overwrite" in second.stderr
    assert digest(output) == checksum


def archive_job(evidence):
    archive_path = evidence / "job.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(evidence / "job-123", arcname="job-123")
    save(evidence / "fetch-123.json", {"archive": archive_path.name, "sha256": digest(archive_path)})


@pytest.fixture(scope="module")
def complete_archive(tmp_path_factory, complete_matrix):
    root = tmp_path_factory.mktemp("synthetic-archived-job")
    staging = root / "staging"
    upload = staging / "upload"
    job = root / "evidence" / "job-123"
    runtime = Path(__file__).resolve().parents[1] / "runtime"
    names = ("cost_model", "engine", "controller", "calibrate", "measured_cost", "timing_probe", "timing_calibration")
    (upload / "runtime").mkdir(parents=True)
    for name in names:
        shutil.copyfile(runtime / f"{name}.py", upload / "runtime" / f"{name}.py")
    original, plan = complete_matrix
    nested = "nested_timing" in plan
    plan_name = "observability-plan-nested.json" if nested else "observability-plan-r3-20261109.json"
    save(upload / "inputs" / plan_name, plan)
    prompt = [1] * 101167
    save(upload / "inputs" / "prompt-bank.json", {"prompts": {"swe-test": {"token_ids": prompt}}})
    (upload / "tests").mkdir()
    (upload / "tests" / "test_skip.py").write_text('pytest.skip("analysis tools run locally")\n')
    if nested:
        shutil.copyfile(runtime / "nested_timing.py", upload / "runtime" / "nested_timing.py")
        (upload / "tests" / "test_nested_timing.py").write_text("synthetic startup node\n")
        save(upload / "inputs" / "startup-test-contract.json", {"schema": 1, "non_timesliced_only": True,
             "nodes": ["tests/test_nested_timing.py"], "passed": 34, "skipped": 0})
    (upload / "xplace.sbatch").write_text("synthetic unit-test fixture, never submitted\n")
    for index in range(28):
        (upload / f"placeholder-{index}").write_text("synthetic\n")
    sources = sorted(path for path in upload.rglob("*") if path.is_file())
    assert len(sources) == (42 if nested else 39)
    (upload / "MANIFEST.sha256").write_text("".join(f"{digest(path)}  {path.relative_to(upload)}\n" for path in sources))
    shutil.copytree(upload, job)
    (job / "xplace.sbatch").unlink()
    (job / "manifest-check.log").write_text("")
    with tarfile.open(staging / "upload.tar.gz", "w:gz") as archive:
        archive.add(upload, arcname="upload")
    save(staging / "build.json", {"files": len(sources), "manifest_sha256": digest(upload / "MANIFEST.sha256"),
                                  "tarball_sha256": digest(staging / "upload.tar.gz")})
    shutil.copytree(original, job / "out" / "calibration" / "observability")
    import hashlib
    save(job / "out" / "calibration" / "observability" / "inputs.json",
         {"code_prompt_tokens": 101167, "code_prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
          "sbatch_entry_wall_s": "1000.0", "phase_entry_monotonic_s": 70.0, "deadline_monotonic_s": 3300.0})
    topology = {"num_ranks": 2, "tensor_parallel_size": 2, "expert_parallel_size": 4, "physical_gpus": 4,
                "gpu_uuids": [f"gpu-{rank}-{lane}" for rank in range(2) for lane in range(2)]}
    phases = [{"phase": "startup", "status": "passed", "topology": topology},
              {"phase": "dataplane", "status": "passed", "migrations": 8},
              {"phase": "observability", "status": "passed"}, {"phase": "job", "status": "passed"}]
    save(job / "out" / "phases.json", {"phases": phases, "args": {"phases": "dataplane,observability",
                                                                 "enable_tp_migration": "True"}})
    save(job / "out" / "timing-interface-check.json", {"status": "passed", "source_sha256": timing_probe.interface_sources(nested=nested)})
    for name, value in {"exit-status": "0", "startup-tests.log": "218 passed, 1 skipped in 20.00s",
                        "started-at": "2026-11-10T02:00:00+08:00", "finished-at": "2026-11-10T02:20:00+08:00",
                        "nodelist": "synthetic-node", "gpus.csv": "\n".join(f"0, GPU-{uuid}, A800"
                                                                             for uuid in topology["gpu_uuids"])}.items():
        (job / name).write_text(value)
    if nested:
        (job / "startup-tests.log").write_text("34 passed in 20.00s\n")
        (job / "startup-test-nodes.txt").write_text("tests/test_nested_timing.py\n")
    moves = []
    for index in range(8):
        source, destination = (0, 1) if index < 4 else (1, 0)
        move = {"model": 1, "source": source, "destination": destination, "tensor_parallel_size": 2,
                "bitwise_verified": True, "elapsed_s": 0.01, "kv_bytes": 128, "num_blocks": 1}
        move["worker_shards"] = {role: [{"tp_rank": lane, "role": role, "rank": move[role], "tp_size": 2,
                                        "num_blocks": 1, "bitwise_verified": True, "bytes": 64,
                                        "integrity": "device_echo_full_bytewise"} for lane in range(2)]
                                  for role in ("source", "destination")}
        moves.append(move)
    transactions = [{"path": path, "status": "completed", "moves": moves[index * 4:(index + 1) * 4]}
                    for index, path in enumerate(("dataplane_out", "dataplane_back"))]
    requests = [{"request_id": f"dp_{rank}_{index}", "output_tokens": 2500, "token_sha256": "a" * 64}
                for rank in range(2) for index in range(2)]
    save(job / "out" / "calibration" / "dataplane.json", {"status": "passed", "transactions": transactions,
                                                         "migrations": moves, "requests": requests})
    archive_job(root / "evidence")
    return root


@pytest.mark.parametrize("damage", [None, "frozen_code", "external_sha", "prompt_identity", "gpu_identity",
                                    "missing_shard", "exit_status", "deadline", "startup_tests", "interface_sha"])
def test_full_cli_uses_frozen_validators_and_preserves_failed_evidence(tmp_path, complete_archive, damage):
    root = tmp_path / "run"
    shutil.copytree(complete_archive, root)
    evidence, staging = root / "evidence", root / "staging"
    job = evidence / "job-123"
    build = load(staging / "build.json")
    manifest_sha = build["manifest_sha256"]
    if damage == "frozen_code":
        (job / "runtime" / "timing_probe.py").write_text("modified")
    elif damage == "external_sha":
        manifest_sha = "0" * 64
    elif damage in ("prompt_identity", "deadline"):
        path = job / "out" / "calibration" / "observability" / "inputs.json"
        record = load(path)
        record["code_prompt_sha256" if damage == "prompt_identity" else "deadline_monotonic_s"] = "0" * 64 if damage == "prompt_identity" else 90
        save(path, record)
    elif damage == "gpu_identity":
        path = job / "out" / "phases.json"
        record = load(path)
        record["phases"][0]["topology"]["gpu_uuids"][0] = "wrong"
        save(path, record)
    elif damage == "missing_shard":
        path = job / "out" / "calibration" / "dataplane.json"
        record = load(path)
        record["migrations"][0]["worker_shards"]["source"].pop()
        save(path, record)
    elif damage == "exit_status":
        (job / "exit-status").write_text("1")
    elif damage == "startup_tests":
        (job / "startup-tests.log").write_text("217 passed, 2 skipped in 20.00s")
    elif damage == "interface_sha":
        save(job / "out" / "timing-interface-check.json", {"status": "passed", "source_sha256": {}})
    archive_job(evidence)
    output = root / "audit.json"
    command = [sys.executable, str(Path(__file__).with_name("audit_observability.py")), "--evidence", str(evidence),
               "--staging", str(staging), "--job-id", "123", "--manifest-sha256", manifest_sha,
               "--tarball-sha256", build["tarball_sha256"], "--plan-sha256",
               digest(next((staging / "upload" / "inputs").glob("observability-plan*.json"))), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    report = load(output)
    assert result.returncode == int(damage is not None), (result.stdout, result.stderr, report)
    assert (report["status"] == "artifact_checks_passed_not_predictive_validation") == (damage is None)
    assert all(report[key] is False for key in ("measured_cost_gate", "independent_prediction_gate",
                                               "paired_ablation_gate", "mixed_topology_gate"))
    assert not list((staging / "upload").rglob("__pycache__"))
