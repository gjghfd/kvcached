import importlib
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_watcher_retries_timeout_and_exits_at_terminal(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    watcher = importlib.import_module("watch_trace_job")
    monkeypatch.setattr("sys.argv", ["watch", "123", "--archive", str(tmp_path)])
    calls = []

    def query(command, **options):
        calls.append(command)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, 120)
        return SimpleNamespace(returncode=0, stdout="123|COMPLETED|0:0|start|end|duration|node\n", stderr="")

    waits = []
    monkeypatch.setattr(watcher.subprocess, "run", query)
    monkeypatch.setattr(watcher.time, "sleep", waits.append)
    watcher.main()
    assert len(calls) == 2 and len(waits) == 1
    assert 0 < waits[0] <= 300
    assert "timed out" in (tmp_path / "poll-123.log").read_text()


def test_terminal_processing_archives_and_verifies_agentic_artifacts(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    watcher = importlib.import_module("watch_trace_job")
    raw = tmp_path / "live-trace-123"
    (raw / "calibration").mkdir(parents=True)
    (raw / "calibration/calibration-summary.json").write_text('{}')
    (raw / "summary.json").write_text('{"status":"passed"}')
    calls = []

    def execute(command, **options):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(watcher.subprocess, "run", execute)
    result = watcher.process_terminal(123, tmp_path, tmp_path / "inputs")
    assert result["status"] == "initial_trace_verified"
    assert [step["name"] for step in result["steps"]] == ["archive", "calibration", "verification", "diagnostics"]
    assert "--prompt-bank" in calls[-1]
    assert json.loads((tmp_path / "terminal-processing-123.json").read_text()) == result


def test_terminal_processing_independently_verifies_joint_admission_pilot(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    watcher = importlib.import_module("watch_trace_job")
    raw = tmp_path / "live-trace-123"
    (raw / "transfer").mkdir(parents=True)
    (raw / "transfer/admission-trials.json").write_text("[]")
    commands = []

    def execute(command, **options):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(watcher.subprocess, "run", execute)
    result = watcher.process_terminal(123, tmp_path, tmp_path / "inputs")
    assert result["status"] == "admission_cost_pilot_verified"
    assert [step["name"] for step in result["steps"]] == ["archive", "admission_cost"]
    assert "verify_admission_cost.py" in commands[-1][1]


@pytest.mark.parametrize("policies,baselines", [
    (["no_move", "no_move_fair", "kv_only", "ilp"], ["no_move", "no_move_fair"]),
    (["no_move", "kv_only"], ["no_move"]),
    (["no_move", "ilp"], []),
])
def test_terminal_processing_automatically_analyzes_available_queue_baselines(monkeypatch, tmp_path,
                                                                           policies, baselines):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    watcher = importlib.import_module("watch_trace_job")
    raw = tmp_path / "live-trace-123"
    raw.mkdir()
    (raw / "summary.json").write_text('{"status":"passed"}')
    (raw / "config.json").write_text(json.dumps({"policies": policies}))
    commands = []

    def execute(command, **options):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(watcher.subprocess, "run", execute)
    result = watcher.process_terminal(123, tmp_path, tmp_path / "inputs")
    assert result["status"] == "initial_trace_verified"
    tails = [command for command in commands if command[1].endswith("analyze_tail_slo.py")]
    assert [command[command.index("--baseline") + 1] for command in tails] == baselines
    assert [step["name"] for step in result["steps"]] == ["archive", "verification", "diagnostics"] + [
        f"tail_{baseline}" for baseline in baselines]


def test_terminal_processing_rejects_failed_tail_analysis(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    watcher = importlib.import_module("watch_trace_job")
    raw = tmp_path / "live-trace-123"
    raw.mkdir()
    (raw / "summary.json").write_text('{"status":"passed"}')
    (raw / "config.json").write_text('{"policies":["no_move","no_move_fair","kv_only"]}')

    def execute(command, **options):
        failed = command[1].endswith("analyze_tail_slo.py")
        return SimpleNamespace(returncode=int(failed), stdout="", stderr="incomplete tail" if failed else "")

    monkeypatch.setattr(watcher.subprocess, "run", execute)
    result = watcher.process_terminal(123, tmp_path, tmp_path / "inputs")
    assert result["status"] == "verification_failed"
    assert result["steps"][-1] == {"name": "tail_no_move", "exit_code": 1}


def test_terminal_processing_stops_at_failed_archive(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    watcher = importlib.import_module("watch_trace_job")
    monkeypatch.setattr(watcher.subprocess, "run", lambda *args, **options: SimpleNamespace(
        returncode=1, stdout="", stderr="download interrupted"))
    result = watcher.process_terminal(123, tmp_path, tmp_path / "inputs")
    assert result["status"] == "archive_failed"
    assert len(result["steps"]) == 1
    assert "download interrupted" in (tmp_path / "terminal-123-archive.log").read_text()


def test_terminal_processing_retries_network_and_resumes_verified_raw(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    watcher = importlib.import_module("watch_trace_job")
    calls, waits = [], []

    def execute(command, **options):
        calls.append(list(command))
        if len(calls) == 1:
            (tmp_path / "live-trace-123").mkdir()
            return SimpleNamespace(returncode=1, stdout="", stderr="returned non-zero exit status 255")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(watcher.subprocess, "run", execute)
    monkeypatch.setattr(watcher.time, "sleep", waits.append)
    result = watcher.process_terminal(123, tmp_path, tmp_path / "inputs", archive_attempts=2)
    assert result["status"] == "archived_without_complete_formal_trace"
    assert "--resume" not in calls[0] and "--resume" in calls[1]
    assert waits == [300]
    assert [item["exit_code"] for item in result["archive_attempts"]] == [1, 0]
    assert "255" in (tmp_path / "terminal-123-archive.log").read_text()


def test_terminal_processing_never_retries_checksum_failure(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    watcher = importlib.import_module("watch_trace_job")
    waits = []
    monkeypatch.setattr(watcher.subprocess, "run", lambda *args, **options: SimpleNamespace(
        returncode=1, stdout="", stderr="Downloaded evidence checksum mismatch"))
    monkeypatch.setattr(watcher.time, "sleep", waits.append)
    result = watcher.process_terminal(123, tmp_path, tmp_path / "inputs", archive_attempts=3)
    assert result["status"] == "archive_failed"
    assert len(result["archive_attempts"]) == 1 and waits == []


def test_terminal_processing_does_not_claim_benefit_after_calibration_only(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    watcher = importlib.import_module("watch_trace_job")
    raw = tmp_path / "live-trace-123"
    (raw / "calibration").mkdir(parents=True)
    (raw / "calibration/calibration-summary.json").write_text('{}')
    monkeypatch.setattr(watcher.subprocess, "run", lambda *args, **options: SimpleNamespace(
        returncode=0, stdout="ok", stderr=""))
    result = watcher.process_terminal(123, tmp_path, tmp_path / "inputs")
    assert result["status"] == "archived_without_complete_formal_trace"
    assert [step["name"] for step in result["steps"]] == ["archive", "calibration"]


def test_terminal_processing_verifies_transport_without_claiming_trace_benefit(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    watcher = importlib.import_module("watch_trace_job")
    raw = tmp_path / "live-trace-123/transfer"
    raw.mkdir(parents=True)
    (raw / "transfer-summary.json").write_text('{"status":"transfer_pilot_passed"}')
    monkeypatch.setattr(watcher.subprocess, "run", lambda *args, **options: SimpleNamespace(
        returncode=0, stdout="ok", stderr=""))
    result = watcher.process_terminal(123, tmp_path, tmp_path / "inputs")
    assert result["status"] == "transfer_pilot_verified"
    assert [step["name"] for step in result["steps"]] == ["archive", "transfer"]
    assert "before_freezing_wait" in result["next_action"]


def test_terminal_processing_verifies_a800_capacity_without_claiming_policy_benefit(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    watcher = importlib.import_module("watch_trace_job")
    raw = tmp_path / "live-trace-123"
    (raw / "capacity").mkdir(parents=True)
    (raw / "capacity/summary.json").write_text('{"status":"capacity_passed"}')
    monkeypatch.setattr(watcher.subprocess, "run", lambda *args, **options: SimpleNamespace(
        returncode=0, stdout="ok", stderr=""))
    result = watcher.process_terminal(123, tmp_path, tmp_path / "inputs")
    assert result["status"] == "archived_without_complete_formal_trace"
    assert [step["name"] for step in result["steps"]] == ["archive", "capacity"]


def test_raw_resume_requires_exact_checksum_file_set(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    archive = importlib.import_module("archive_trace_job")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "evidence.json").write_text("{}")
    manifest = hashlib.sha256(b"{}").hexdigest() + "  ./evidence.json\n"
    archive.verify_raw(raw, manifest)
    (raw / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="file set"):
        archive.verify_raw(raw, manifest)
    with pytest.raises(ValueError, match="Unsafe"):
        archive.verify_raw(raw, hashlib.sha256(b"{}").hexdigest() + "  ../outside.json\n")


def test_archive_resume_verifies_existing_raw_and_logs_without_redownload(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    archive = importlib.import_module("archive_trace_job")
    raw = tmp_path / "live-trace-123"
    raw.mkdir()
    (raw / "evidence.json").write_text("{}")
    digest = hashlib.sha256(b"{}").hexdigest()
    manifest = digest + "  ./evidence.json\n"
    (tmp_path / "raw-123.sha256").write_text(manifest)
    for extension in ("out", "err"):
        (tmp_path / f"live-trace-123.{extension}").write_text("{}")
    monkeypatch.setattr("sys.argv", ["archive", "123", "--archive", str(tmp_path), "--resume"])

    def execute(command, **options):
        assert command[0] == "ssh"
        text = "123|FAILED|1:0|submit|start|end|elapsed|node|gpu\nWATCHER_BARRIER\nno_watcher\nRAW_MANIFEST\n" + manifest
        if "sha256sum" in command[-1] and "RAW_MANIFEST" not in command[-1]:
            text = digest + "  logfile\n"
        return SimpleNamespace(returncode=0, stdout=text, check_returncode=lambda: None)

    monkeypatch.setattr(archive.subprocess, "run", execute)
    archive.main()
    report = json.loads((tmp_path / "archive-123.json").read_text())
    assert report["resumed_verified_raw"] and report["raw_files_verified"] == 1
    assert report["log_sha256"] == {"out": digest, "err": digest}


@pytest.fixture
def remote_watcher_environment(tmp_path):
    home = tmp_path / "home"
    root = home / "run/kvcached-paracloud/state/live-trace-123"
    root.mkdir(parents=True)
    commands = tmp_path / "commands"
    commands.mkdir()
    scripts = {
        "date": '#!/bin/sh\nprintf "2026-10-27T13:00:00Z\\n"\n',
        "hostname": '#!/bin/sh\nprintf "ln03\\n"\n',
        "sacct": '#!/bin/sh\nprintf "%s\\n" "$FAKE_ACCOUNTING"\n',
        "sleep": '#!/bin/sh\nprintf "%s\\n" "$*" >> "$SLEEP_CALLS"\n'
                 'if [ -n "${WATCHER_SIGNAL:-}" ]; then kill -"$WATCHER_SIGNAL" "$PPID"; fi\n',
        "mv": '#!/bin/sh\nprintf "%s\\n" "$2" >> "$MARKER_RENAMES"\n'
              'case "$2" in *.exit) /bin/cp "$QUERY_LOG" "$EXIT_LOG_SNAPSHOT" ;; esac\n'
              'exec /bin/mv "$@"\n',
    }
    for name, script in scripts.items():
        executable = commands / name
        executable.write_text(script)
        executable.chmod(0o755)
    environment = dict(os.environ, HOME=str(home), PATH=str(commands) + os.pathsep + os.environ["PATH"],
                       FAKE_ACCOUNTING="123|COMPLETED|0:0|start|end|elapsed|node",
                       DP_OPT_REMOTE_MAX_POLLS="1", SLEEP_CALLS=str(tmp_path / "sleeps"),
                       MARKER_RENAMES=str(tmp_path / "renames"), QUERY_LOG=str(root / "remote-query.log"),
                       EXIT_LOG_SNAPSHOT=str(tmp_path / "exit-log-snapshot"),
                       LC_ALL="C", LC_CTYPE="C", LANG="C")
    return root, environment


def run_remote_watcher(environment):
    script = Path(__file__).resolve().parents[1] / "experiments/dp_opt/watch-trace-remote.sh"
    with Path(environment["QUERY_LOG"]).open("ab") as output:
        return subprocess.run(["bash", str(script), "123"], env=environment,
                              stdout=output, stderr=subprocess.STDOUT, timeout=10)


@pytest.mark.parametrize("state,signal,exit_status,sleeps", [
    ("COMPLETED", "", 0, []), ("FAILED", "", 0, []),
    ("RUNNING", "", 124, ["300"]), ("RUNNING", "TERM", 143, ["300"]),
    ("RUNNING", "HUP", 129, ["300"]),
])
def test_remote_watcher_atomically_publishes_exit_after_final_log(
        remote_watcher_environment, state, signal, exit_status, sleeps):
    root, environment = remote_watcher_environment
    environment["FAKE_ACCOUNTING"] = f"123|{state}|0:0|start|end|elapsed|node"
    environment["WATCHER_SIGNAL"] = signal
    expected = root / "remote-query.expected"
    expected.write_text("coordinator launch intent\n")
    result = run_remote_watcher(environment)
    assert result.returncode == exit_status
    started = dict(line.split("=", 1) for line in (root / "remote-query.started").read_text().splitlines())
    exited = dict(line.split("=", 1) for line in (root / "remote-query.exit").read_text().splitlines())
    for key in ("job", "hostname", "pid", "run_id"):
        assert started[key] == exited[key]
    assert started["hostname"] == "ln03"
    assert started["time"] == exited["time"] == "2026-10-27T13:00:00Z"
    assert started["exit_status"] == "running"
    assert exited["exit_status"] == str(exit_status)
    assert expected.read_text() == "coordinator launch intent\n"
    assert Path(environment["MARKER_RENAMES"]).read_text().splitlines() == [
        str(root / "remote-query.started"), str(root / "remote-query.exit")]
    log = Path(environment["QUERY_LOG"]).read_bytes()
    assert log == Path(environment["EXIT_LOG_SNAPSHOT"]).read_bytes()
    assert log.endswith(f"remote-query exit hostname=ln03 time=2026-10-27T13:00:00Z exit_status={exit_status}\n".encode())
    actual_sleeps = Path(environment["SLEEP_CALLS"])
    assert (actual_sleeps.read_text().splitlines() if actual_sleeps.exists() else []) == sleeps
    assert not list(root.glob(".remote-query.*.??????"))
    assert not (root / ".remote-query-watcher.lock").exists()


def test_remote_watcher_refuses_to_overwrite_previous_markers_or_append_after_exit(remote_watcher_environment):
    root, environment = remote_watcher_environment
    assert run_remote_watcher(environment).returncode == 0
    previous = {path: path.read_bytes() for path in root.iterdir() if path.is_file()}
    assert run_remote_watcher(environment).returncode == 2
    assert all(path.read_bytes() == content for path, content in previous.items())


def test_remote_watcher_legacy_coordinator_still_produces_shared_markers(remote_watcher_environment):
    root, environment = remote_watcher_environment
    assert run_remote_watcher(environment).returncode == 0
    assert not (root / "remote-query.expected").exists()
    assert (root / "remote-query.started").is_file()
    assert (root / "remote-query.exit").is_file()
