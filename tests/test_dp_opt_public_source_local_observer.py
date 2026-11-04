import argparse
import ast
import datetime
import fcntl
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def observer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("public_source_local_observer")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def prepared(observer, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    roots = {role: tmp_path / role for role in observer.ROLES}
    for root in roots.values():
        root.mkdir()
    args = argparse.Namespace(**{role + "_root": root for role, root in roots.items()}, repo=repo,
                              output=None, interval=300, freshness=660, deadline_utc="2030-01-01T00:00:00Z")
    return observer.configuration(args), args


def live_process(observer, root, pid=42, identity="original"):
    return {"pid": pid, "ppid": 1, "uid": os.getuid(), "alive": True, "stat": "S",
            "command": f"python public_workload_campaign.py --root {root}",
            "process_identity_sha256": identity}


def active_campaign(config, now, **changes):
    root = config["roots"]["c0_cal"]
    state = {"repo": str(config["repo"]), "phase": "submitted_public_search", "job": 271324,
             "state_last_observed": "RUNNING", "last_poll_at": now.isoformat(), "last_poll_ssh_exit": 0}
    state.update(changes)
    dump(root / "campaign-state.json", state)
    dump(root / "local-monitor-identity.json", {"pid": 42, "parent_pid": 1})
    return root, state


@pytest.mark.parametrize("interval,freshness", [(0, 660), (-1, 660), (float("nan"), 660),
                                               (float("inf"), 660), (300, 299), (300, 0)])
def test_invalid_interval_or_freshness_rejected(observer, prepared, interval, freshness):
    _, args = prepared
    args.interval, args.freshness = interval, freshness
    with pytest.raises(ValueError):
        observer.configuration(args)


def test_distinct_roots_output_scope_and_timezone(observer, prepared):
    config, args = prepared
    args.c1_eval_root = args.c1_cal_root
    with pytest.raises(ValueError, match="distinct"):
        observer.configuration(args)
    args.c1_eval_root = config["roots"]["c1_eval"]
    args.output = config["roots"]["recovery"] / "elsewhere"
    with pytest.raises(ValueError):
        observer.configuration(args)
    with pytest.raises(ValueError, match="timezone"):
        observer.timestamp("2026-11-10T00:00:00")
    assert observer.DEFAULT_DEADLINE == "2026-11-10T00:00:00Z"


def test_process_inspection_only_local_ps_and_stable_fingerprint(observer, monkeypatch):
    commands = []
    def inspect(command, **kwargs):
        commands.append(command)
        assert kwargs["timeout"] == 10
        return subprocess.CompletedProcess(command, 0, "42 1 501 Sat Oct 31 05:45:49 2026 S python source.py\n", "")
    monkeypatch.setattr(observer.subprocess, "run", inspect)
    process = observer.process_identity(42)
    assert process["alive"] and process["uid"] == 501 and process["ppid"] == 1
    assert process["started_at_local_ps"] == "Sat Oct 31 05:45:49 2026"
    assert process["process_identity_sha256"] == observer.process_identity(42)["process_identity_sha256"]
    assert all(command[0] == "ps" for command in commands)


@pytest.mark.parametrize("pid", [None, True, "42", -1, 0])
def test_invalid_pid_never_executes_a_command(observer, monkeypatch, pid):
    monkeypatch.setattr(observer.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not execute"))
    assert observer.process_identity(pid)["alive"] is None


def test_process_dead_timeout_and_zombie_not_reported_alive(observer, monkeypatch):
    monkeypatch.setattr(observer.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", ""))
    assert observer.process_identity(42)["alive"] is False
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 10)
    monkeypatch.setattr(observer.subprocess, "run", timeout)
    assert observer.process_identity(42)["alive"] is None
    monkeypatch.setattr(observer.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "42 1 501 Sat Oct 31 05:45:49 2026 Z python source.py", ""))
    assert observer.process_identity(42)["alive"] is False


def test_poll_current_job_multiline_progress_and_third_field_not_exit(observer, prepared):
    config, _ = prepared
    root = config["roots"]["c0_cal"]
    (root / "poll-271324.log").write_text('=== 2026-10-31T00:00:00Z actual_gap_s=300 ssh_exit=0 ===\n'
        '270947|FAILED|1:0\n271324|RUNNING|2:00:15\nphase.json\n{"phase":"calibration"}\n'
        'progress.json\n{\n "completed_cases": 91,\n "total_cases": 144\n}\n')
    poll = observer.latest_poll(root, 271324)
    assert poll["job_state"] == "RUNNING"
    assert poll["uninterpreted_third_field"] == "2:00:15" and "job_exit" not in poll
    assert poll["progress"]["completed_cases"] == 91 and poll["phase"] == "calibration"


def test_waiting_prepared_roots_and_missing_progress_are_not_success_or_stall(observer, prepared, monkeypatch):
    config, _ = prepared
    now = observer.timestamp("2026-10-31T00:00:00Z")
    root, _ = active_campaign(config, now)
    monkeypatch.setattr(observer, "process_identity", lambda pid: live_process(observer, root) if pid else {"pid": None, "alive": None})
    waiting = observer.snapshot_root("c1_cal", config["roots"]["c1_cal"], config, {}, now)
    assert waiting["status"] == "waiting_not_success" and not waiting["alerts"]
    active = observer.snapshot_root("c0_cal", root, config, {}, now)
    assert not active["alerts"] and active["local_poll_freshness_s"] == 0
    assert "waiting" in active["progress_semantics"]


@pytest.mark.parametrize("changes,kind", [({"last_poll_at": "2026-10-30T23:00:00Z"}, "active_poll_stale_or_future_timestamp"),
    ({"last_poll_monotonic_gap_s": 700}, "measured_campaign_poll_gap"),
    ({"last_poll_ssh_exit": 255}, "observed_producer_query_error"),
    ({"state_last_observed": "FAILED"}, "local_campaign_failure")])
def test_campaign_alerts_use_local_evidence(observer, prepared, monkeypatch, changes, kind):
    config, _ = prepared
    now = observer.timestamp("2026-10-31T00:00:00Z")
    root, _ = active_campaign(config, now, **changes)
    monkeypatch.setattr(observer, "process_identity", lambda pid: live_process(observer, root))
    value = observer.snapshot_root("c0_cal", root, config, {}, now)
    assert kind in {alert["kind"] for alert in value["alerts"]}


def test_dead_pid_after_terminal_not_false_failure_and_reuse_alerts_when_active(observer, prepared, monkeypatch):
    config, _ = prepared
    now = observer.timestamp("2026-10-31T00:00:00Z")
    root, state = active_campaign(config, now, state_last_observed="COMPLETED")
    monkeypatch.setattr(observer, "process_identity", lambda pid: {"pid": 42, "alive": False})
    assert not observer.snapshot_root("c0_cal", root, config, {}, now)["alerts"]
    state["state_last_observed"] = "RUNNING"
    dump(root / "campaign-state.json", state)
    monkeypatch.setattr(observer, "process_identity", lambda pid: live_process(observer, root, identity="reused"))
    value = observer.snapshot_root("c0_cal", root, config, {"process_identity_sha256": "original"}, now)
    assert "active_local_process_dead_missing_or_changed" in {alert["kind"] for alert in value["alerts"]}


def test_caffeinate_pid_reuse_is_detected_without_any_signal(observer, prepared, monkeypatch):
    config, _ = prepared
    now = observer.timestamp("2026-10-31T00:00:00Z")
    root, _ = active_campaign(config, now)
    dump(root / "local-monitor-identity.json", {"pid": 42, "caffeinate_pid": 43})
    def inspect(pid):
        if pid == 42:
            return live_process(observer, root)
        return {"pid": 43, "alive": True, "command": "caffeinate -w 42", "process_identity_sha256": "reused"}
    monkeypatch.setattr(observer, "process_identity", inspect)
    value = observer.snapshot_root("c0_cal", root, config, {"caffeinate_identity_sha256": "original"}, now)
    assert "active_caffeinate_dead_or_changed" in {alert["kind"] for alert in value["alerts"]}


def test_frozen_sha_no_rebinding_new_files_and_deletion(observer, prepared):
    config, _ = prepared
    root, repo = config["roots"]["c0_cal"], config["repo"]
    source = repo / "runtime.py"
    source.write_text("original")
    bank = root / "frozen/inputs/prompt-bank.json"
    bank.parent.mkdir(parents=True)
    bank.write_text("{}")
    dump(bank.parent / "input-manifest.json", {"files_sha256": {bank.name: observer.digest(bank)}})
    state = {"repo": str(repo), "frozen_runtime_sha256": {"runtime.py": observer.digest(source)}}
    baseline = {}
    _, alerts = observer.check_hashes(root, repo, state, baseline)
    assert not alerts
    original_manifest = baseline["runtime_manifest"]
    source.write_text("changed")
    state["frozen_runtime_sha256"]["runtime.py"] = observer.digest(source)
    _, alerts = observer.check_hashes(root, repo, state, baseline)
    assert {"frozen_runtime_identity_changed", "observed_frozen_SHA_changed"} <= {alert["kind"] for alert in alerts}
    assert baseline["runtime_manifest"] == original_manifest
    source.write_text("original")
    state["frozen_runtime_sha256"]["runtime.py"] = observer.digest(source)
    trace = root / "frozen/new-native.json"
    trace.write_text("{}")
    assert not observer.check_hashes(root, repo, state, baseline)[1]
    trace.unlink()
    assert "frozen_file_disappeared" in {alert["kind"] for alert in observer.check_hashes(root, repo, state, baseline)[1]}


def test_declared_input_sha_and_path_escape_rejected(observer, prepared):
    config, _ = prepared
    root = config["roots"]["c1_cal"]
    bank = root / "frozen/inputs/prompt-bank.json"
    bank.parent.mkdir(parents=True)
    bank.write_text("{}")
    manifest = bank.parent / "input-manifest.json"
    dump(manifest, {"files_sha256": {bank.name: "0" * 64}})
    assert observer.check_hashes(root, config["repo"], {}, {})[1][0]["kind"] == "declared_frozen_SHA_mismatch"
    dump(manifest, {"files_sha256": {"../../outside": "0" * 64}})
    with pytest.raises(ValueError):
        observer.check_hashes(root, config["repo"], {}, {})
    with pytest.raises(ValueError, match="repo differs"):
        observer.check_hashes(root, config["repo"], {"repo": "/outside", "frozen_runtime_sha256": {"runtime.py": "0" * 64}}, {})


def test_retained_historical_exit_binding_preserved_not_active_failure(observer, prepared):
    config, _ = prepared
    root = config["roots"]["recovery"]
    path = root / "future-quality-full-isolated.exit"
    path.write_text("1\n")
    reference = "future-quality-full-isolated.log/exit1_No_module_named_pytest"
    dump(root / "future-quality-prepared-v2.json", {"first_isolated_direct_module_failure_retained": reference,
         "final_isolated_regression": {"passed": 923, "exit": 0}})
    value = observer.snapshot_root("recovery", root, config, {}, observer.utc_now())
    evidence = value["local_exit_files"][path.name]
    assert evidence["classification"] == "retained_historical_exit_evidence" and evidence["raw"] == "1"
    assert evidence["prepared_v2_binding"]["value"] == reference
    assert not value["alerts"] and path.read_text() == "1\n"
    dump(root / "campaign-state.json", {"phase": "diagnosis_required", "controller_exit_code": 1})
    failed = observer.snapshot_root("recovery", root, config, {}, observer.utc_now())
    assert {"local_campaign_failure", "declared_controller_or_producer_failure"} <= {alert["kind"] for alert in failed["alerts"]}


def test_unattributed_exit_retained_not_invented_as_current_job_failure(observer, prepared):
    config, _ = prepared
    root = config["roots"]["recovery"]
    (root / "historical-cpu.exit").write_text("2\n")
    value = observer.snapshot_root("recovery", root, config, {}, observer.utc_now())
    assert not value["alerts"] and value["notices"][0]["kind"] == "retained_unattributed_exit_evidence"
    assert value["local_exit_files"]["historical-cpu.exit"]["raw"] == "2"


def test_root_level_descriptor_declared_sha_preferred_and_tampering_detected(observer, prepared):
    config, _ = prepared
    root = config["roots"]["s0_future"]
    descriptor = root / "quality-followup.json"
    dump(descriptor, {"enabled": True})
    expected = observer.digest(descriptor)
    dump(config["roots"]["recovery"] / "future-quality-prepared-v2.json", {"descriptor_sha256": expected})
    baseline = {}
    value = observer.snapshot_root("s0_future", root, config, baseline, observer.utc_now())
    assert not value["alerts"] and value["descriptor_declared_sha256"] == expected
    assert value["frozen_SHA"]["actual_files_sha256"]["quality-followup.json"] == expected
    dump(descriptor, {"enabled": False})
    value = observer.snapshot_root("s0_future", root, config, baseline, observer.utc_now())
    assert {"declared_frozen_SHA_mismatch", "observed_frozen_SHA_changed"} <= {alert["kind"] for alert in value["alerts"]}


def test_prepared_v2_source_sha_observed_without_importing_source(observer, prepared):
    config, _ = prepared
    root = config["roots"]["recovery"]
    source = config["repo"] / "future.py"
    source.write_text("raise RuntimeError('must never import')\n")
    dump(root / "future-quality-prepared-v2.json", {"source_sha256": {"future.py": observer.digest(source)}})
    value, alerts = observer.check_hashes(root, config["repo"], {}, {})
    assert not alerts and "prepared-v2/repo/future.py" in value["actual_files_sha256"]


def test_recovery_controller_declared_sha_verified_without_import(observer, prepared):
    config, _ = prepared
    root = config["roots"]["recovery"]
    controller = config["repo"] / "experiments/dp_opt/public_source_recovery.py"
    controller.parent.mkdir(parents=True)
    controller.write_text("raise RuntimeError('never execute controller')\n")
    dump(root / "campaign-state.json", {"controller_sha256": observer.digest(controller)})
    value = observer.snapshot_root("recovery", root, config, {}, observer.utc_now())
    assert not value["alerts"] and "controller/public_source_recovery.py" in value["frozen_SHA"]["actual_files_sha256"]
    controller.write_text("changed controller")
    value = observer.snapshot_root("recovery", root, config, {}, observer.utc_now())
    assert "declared_frozen_SHA_mismatch" in {alert["kind"] for alert in value["alerts"]}


def test_frozen_symlink_and_output_symlink_cannot_escape(observer, prepared):
    config, _ = prepared
    root = config["roots"]["c1_cal"]
    (root / "frozen").mkdir()
    outside = config["repo"] / "outside.txt"
    outside.write_text("must remain untouched")
    (root / "frozen/link").symlink_to(outside)
    with pytest.raises(ValueError):
        observer.check_hashes(root, config["repo"], {}, {})
    config["output"].mkdir(parents=True)
    (config["output"] / "observer.jsonl").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        observer.run(config, once=True)
    assert outside.read_text() == "must remain untouched"


def test_predecessor_must_be_confirmed_dead_no_kill_or_launch(observer, prepared, monkeypatch):
    config, _ = prepared
    dump(config["roots"]["recovery"] / "subagent-monitor-identity.json", {"pid": 65227})
    monkeypatch.setattr(observer, "process_identity", lambda pid: {"pid": pid, "alive": True})
    monkeypatch.setattr(observer.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must not launch"))
    with pytest.raises(RuntimeError, match="Predecessor"):
        observer.launch(config, [sys.executable, observer.__file__])
    assert not config["output"].exists()


def test_exclusive_lock_refuses_duplicate_observer(observer, prepared):
    config, _ = prepared
    config["output"].mkdir(parents=True)
    with (config["output"] / "observer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            observer.run(config, once=True)


def test_detached_launch_mock_pending_receipt_prevents_race(observer, prepared, monkeypatch):
    config, _ = prepared
    calls = []
    def spawn(command, **kwargs):
        calls.append((command, kwargs))
        return argparse.Namespace(pid=123321)
    monkeypatch.setattr(observer.subprocess, "Popen", spawn)
    receipt = observer.launch(config, [sys.executable, observer.__file__])
    assert receipt["pid"] == 123321 and "not_yet_verified" in receipt["status"]
    assert calls[0][1]["start_new_session"] is True and calls[0][1]["cwd"] == config["repo"]
    assert calls[0][1]["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    monkeypatch.setattr(observer, "process_identity", lambda pid: {"pid": pid, "alive": True})
    with pytest.raises(RuntimeError, match="Previous detached launch"):
        observer.launch(config, [sys.executable, observer.__file__])
    assert len(calls) == 1


def test_expired_deadline_never_launches(observer, prepared, monkeypatch):
    config, _ = prepared
    config["deadline_utc"] = "2000-01-01T00:00:00Z"
    monkeypatch.setattr(observer.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="past"):
        observer.launch(config, [sys.executable, observer.__file__])


@pytest.mark.parametrize("command", [["ssh", "host"], ["sbatch", "script"], ["python", "other.py"]])
def test_launch_rejects_any_other_executable_without_spawn(observer, prepared, monkeypatch, command):
    config, _ = prepared
    monkeypatch.setattr(observer.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must not spawn"))
    with pytest.raises(ValueError, match="Launch only"):
        observer.launch(config, command)


def test_actual_monotonic_gap_and_wall_jump_recorded_not_assumed_300(observer, prepared, monkeypatch):
    config, _ = prepared
    origin = observer.timestamp("2026-10-31T00:00:00Z")
    clock = {"monotonic": 1000, "wall": 0}
    class Event:
        stopped = False
        waits = 0
        def is_set(self):
            return self.stopped
        def set(self):
            self.stopped = True
        def wait(self, seconds):
            assert seconds == 300
            self.waits += 1
            clock["monotonic"] += 700
            clock["wall"] += 400
            self.stopped = self.waits == 2
    monkeypatch.setattr(observer.threading, "Event", Event)
    monkeypatch.setattr(observer, "utc_now", lambda: origin + datetime.timedelta(seconds=clock["wall"]))
    monkeypatch.setattr(observer.time, "monotonic", lambda: clock["monotonic"])
    observer.run(config)
    rows = [json.loads(line) for line in (config["output"] / "observer.jsonl").read_text().splitlines()]
    assert rows[0]["observer_actual_monotonic_gap_s"] is None
    assert rows[1]["observer_actual_monotonic_gap_s"] == 700 and rows[1]["observer_actual_wall_gap_s"] == 400
    assert {"observer_monotonic_gap_exceeded", "observer_wall_clock_discontinuity"} <= {alert["kind"] for alert in rows[1]["alerts"]}


def test_real_stdlib_only_cli_one_snapshot_and_no_source_writes(observer, prepared):
    config, _ = prepared
    root = config["roots"]["c1_cal"]
    dump(root / "frozen/inputs/input-manifest.json", {"files_sha256": {}})
    before = {path: path.read_bytes() for campaign in config["roots"].values() for path in campaign.rglob("*") if path.is_file()}
    command = ["/usr/bin/python3", "-S", observer.__file__, "--repo", str(config["repo"]), "--deadline-utc", "2030-01-01T00:00:00Z", "--once"]
    for role, campaign in config["roots"].items():
        command.extend(["--" + role.replace("_", "-") + "-root", str(campaign)])
    result = subprocess.run(command, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=15)
    assert result.returncode == 0, result.stderr
    assert all(path.read_bytes() == contents for path, contents in before.items())
    rows = [json.loads(line) for line in (config["output"] / "observer.jsonl").read_text().splitlines()]
    assert len(rows[0]["roots"]) == 5 and not rows[0]["alerts"]
    identity = json.loads((config["output"] / "observer-identity.json").read_text())
    assert identity["status"] == "stopped" and identity["exit_reason"] == "requested_one_snapshot"
    assert identity["no_network_or_job_operations"] and identity["cwd"]


def test_prepare_command_never_launches_or_inspects_pid(observer, prepared, monkeypatch, capsys):
    config, _ = prepared
    monkeypatch.setattr(observer, "process_identity", lambda *args: pytest.fail("prepare must not inspect PID"))
    monkeypatch.setattr(observer.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("prepare must not launch"))
    arguments = ["--print-start-command"]
    for role, root in config["roots"].items():
        arguments.extend(["--" + role.replace("_", "-") + "-root", str(root)])
    observer.main(arguments)
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "prepared_only_not_started" and "--launch" in value["launch_command"]
    assert value["config"]["deadline_utc"] == "2026-11-10T00:00:00+00:00"
    assert not config["output"].exists()


def test_relative_cli_roots_are_canonicalized_before_detached_cwd_change(observer, prepared, monkeypatch, capsys):
    config, _ = prepared
    monkeypatch.chdir(config["repo"].parent)
    captured = []
    def launch(configuration, command):
        captured.append((configuration, command))
        return {"status": "mock_only"}
    monkeypatch.setattr(observer, "launch", launch)
    arguments = ["--launch", "--repo", "repo"]
    for role in observer.ROLES:
        arguments.extend(["--" + role.replace("_", "-") + "-root", role])
    observer.main(arguments)
    configuration, command = captured[0]
    assert configuration["repo"] == config["repo"]
    for role, root in config["roots"].items():
        flag = "--" + role.replace("_", "-") + "-root"
        assert command[command.index(flag) + 1] == str(root)
    assert command[command.index("--repo") + 1] == str(config["repo"])


def test_source_imports_stdlib_only_and_never_imports_existing_gpu_modules(observer):
    tree = ast.parse(Path(observer.__file__).read_text())
    names = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert not names & {"torch", "tokenizers", "pyarrow", "scipy", "socket", "requests", "public_workload_campaign"}
    assert {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)} == {"pathlib"}
