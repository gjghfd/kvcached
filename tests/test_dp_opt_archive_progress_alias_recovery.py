import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import archive_progress_alias_recovery as recovery


def binding():
    stage = {"job": 123, "plan": {"phase_name": "native-N4"}, "remote_home": "/data/home/owner",
             "terminal": "COMPLETED", "accounting_exit_code": "0:0"}
    return recovery.pointer_binding(stage, "owner", "/data/home/owner", "/data/run01/owner/kvcached-paracloud",
        "/data/run01/owner/kvcached-paracloud/state/live-trace-123/native-N4/progress.json")


@pytest.mark.parametrize("field,value", [("terminal", "FAILED"), ("accounting_exit_code", "1:0"),
                                       ("job", True), ("remote_home", "/other/home")])
def test_pointer_proof_rejects_wrong_ledger(field, value):
    stage = {"job": 123, "plan": {"phase_name": "native-N4"}, "remote_home": "/data/home/owner",
             "terminal": "COMPLETED", "accounting_exit_code": "0:0", field: value}
    with pytest.raises(ValueError):
        recovery.pointer_binding(stage, "owner", "/data/home/owner", "/data/run01/owner/kvcached-paracloud",
            "/data/run01/owner/kvcached-paracloud/state/live-trace-123/native-N4/progress.json")


@pytest.mark.parametrize("wrong", ["../other", "/data/run01/owner/kvcached-paracloud/state/live-trace-124/native-N4/progress.json",
                                 "/data/run01/owner/kvcached-paracloud/state/live-trace-123/native-N3/progress.json"])
def test_pointer_proof_rejects_wrong_target(wrong):
    stage = {"job": 123, "plan": {"phase_name": "native-N4"}, "remote_home": "/data/home/owner",
             "terminal": "COMPLETED", "accounting_exit_code": "0:0"}
    with pytest.raises(ValueError):
        recovery.pointer_binding(stage, "owner", "/data/home/owner", "/data/run01/owner/kvcached-paracloud", wrong)


@pytest.mark.parametrize("extra_link", [False, True])
def test_parser_alias_preserves_bytes_and_rejects_other_symlinks(tmp_path, extra_link):
    proof = binding()
    bundle = tmp_path / "original.tar.gz"
    payload = b'{"complete": true}\n'
    with tarfile.open(bundle, "w:gz") as archive:
        regular = tarfile.TarInfo("./native-N4/progress.json")
        regular.size = len(payload)
        archive.addfile(regular, io.BytesIO(payload))
        pointer = tarfile.TarInfo("./progress.json")
        pointer.type = tarfile.SYMTYPE
        pointer.linkname = proof["remote_target"]
        pointer.pax_headers = {"linkpath": proof["remote_target"]}
        archive.addfile(pointer)
        if extra_link:
            other = tarfile.TarInfo("./other")
            other.type, other.linkname = tarfile.SYMTYPE, "/unsafe"
            archive.addfile(other)
    before = recovery.digest(bundle)
    manifest = hashlib.sha256(payload).hexdigest() + "  ./native-N4/progress.json\n"
    destination = tmp_path / "raw"
    destination.mkdir()
    if extra_link:
        with pytest.raises(ValueError, match="Unsafe raw archive member"):
            recovery.extract_bound_alias(bundle, destination, manifest, 123, proof)
    else:
        pointers = recovery.extract_bound_alias(bundle, destination, manifest, 123, proof)
        assert pointers[0]["remote_target"] == proof["remote_target"]
        assert not (destination / "progress.json").exists()
        assert (destination / "native-N4/progress.json").read_bytes() == payload
    assert recovery.digest(bundle) == before


def test_archive_caller_must_match_receipt(tmp_path):
    with pytest.raises(ValueError, match="caller job"):
        recovery.extract_bound_alias(tmp_path / "absent", tmp_path, "", 124, binding())


@pytest.mark.parametrize("terminal,exit_code,error,expected", [
    ("COMPLETED", "0:0", "ValueError: Unsafe raw progress pointer", 1),
    ("FAILED", "1:0", "ValueError: Unsafe raw progress pointer", 0),
    ("COMPLETED", "0:0", "ValueError: Different failure", 0),
    ("COMPLETED", "1:0", "ValueError: Unsafe raw progress pointer", 0),
])
def test_follow_only_recovers_exact_completed_pointer_failure(tmp_path, monkeypatch, terminal, exit_code, error, expected):
    stage = {"job": 123, "terminal": terminal, "accounting_exit_code": exit_code}
    state = {"stages": {"native-N4": stage}, "frozen_runtime_sha256": {}, "repo": str(tmp_path)}
    (tmp_path / "campaign-state.json").write_text(json.dumps(state))
    (tmp_path / "local-monitor-identity.json").write_text(json.dumps({"pid": 456}))
    directory = tmp_path / "stages/native-N4"
    directory.mkdir(parents=True)
    (directory / "archive-123-attempt-1.log").write_text("original error\n" + error + "\n")
    calls, sleeps = [], []
    monkeypatch.setattr(recovery.os, "kill", lambda pid, signal: None)
    monkeypatch.setattr(recovery, "verify_performance_identity", lambda root, expected=None: ({"pid": 456}, {"pid": 456}))
    monkeypatch.setattr(recovery, "recover", lambda root, stage_id: calls.append(stage_id) or {"job": 123})
    monkeypatch.setattr(recovery.time, "sleep", sleeps.append)
    monkeypatch.setattr(recovery.subprocess, "run", lambda *args, **kwargs: pytest.fail("normal observer must not SSH"))
    if terminal == "COMPLETED" and exit_code == "0:0" and not expected:
        with pytest.raises(RuntimeError, match="permanent diagnosis"):
            recovery.follow(tmp_path, 1)
        assert sleeps == []
        row = json.loads((tmp_path / "progress-alias-follow.jsonl").read_text())
        assert row["status"] == "failed_preserved_stopped"
        assert Path(row["diagnosis"]).is_file()
    else:
        recovery.follow(tmp_path, 1)
        assert len(sleeps) == 1 and 299 <= sleeps[0] <= 300
    assert len(calls) == expected
    assert json.loads((tmp_path / "campaign-state.json").read_text()) == state


@pytest.fixture
def recovery_case(tmp_path, monkeypatch):
    directory = tmp_path / "stages/native-N4"
    directory.mkdir(parents=True)
    runtime = tmp_path / "repo/frozen.py"
    runtime.parent.mkdir()
    runtime.write_bytes(b"frozen performance source\n")
    stage = {"stage_id": "native-N4", "job": 123, "plan": {"phase_name": "native-N4"},
             "remote_home": "/data/home/owner", "terminal": "COMPLETED", "accounting_exit_code": "0:0"}
    state = {"repo": str(runtime.parent), "stages": {"native-N4": stage},
             "frozen_runtime_sha256": {"frozen.py": recovery.digest(runtime)}}
    (tmp_path / "campaign-state.json").write_text(json.dumps(state))
    identity = {"pid": 456, "argv": [sys.executable, str(runtime.parent / "performance_followup.py"),
                                      "--root", str(tmp_path)],
                "actual_process_start": "Mon Nov  2 07:59:30 2026", "monitor_interval_s": 300}
    (tmp_path / "local-monitor-identity.json").write_text(json.dumps(identity))
    proof = binding()
    payload = b'{"complete":true}\n'
    checksum = hashlib.sha256(payload).hexdigest()
    manifest = checksum + "  ./native-N4/progress.json\n"
    suffix = "20261102-100000-000000-fixture"
    original_attempt = directory / f"archive-123-attempt-{suffix}"
    original_attempt.mkdir()
    bundle = directory / f"raw-123-{suffix}.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        member = tarfile.TarInfo("./native-N4/progress.json")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
        pointer = tarfile.TarInfo("./progress.json")
        pointer.type, pointer.linkname = tarfile.SYMTYPE, proof["remote_target"]
        archive.addfile(pointer)
    accounting = "123|COMPLETED|0:0|submit|start|end|elapsed|node|gpu\n"
    query = accounting + "\nWATCHER_BARRIER\nshared_markers\nRAW_MANIFEST\n" + manifest
    (original_attempt / "query.stdout").write_text(query)
    (original_attempt / "accounting.txt").write_text(accounting)
    (original_attempt / "raw.sha256").write_text(manifest)
    log = directory / "archive-123-attempt-1.log"
    log.write_text("Traceback retained\nValueError: Unsafe raw progress pointer\n")
    timestamp = 1700000000000000000
    for path in [bundle, original_attempt, *original_attempt.iterdir()]:
        os.utime(path, ns=(timestamp, timestamp))
    os.utime(log, ns=(timestamp + 1000000000, timestamp + 1000000000))
    case = SimpleNamespace(root=tmp_path, directory=directory, stage=stage, state=state,
                           original_attempt=original_attempt, bundle=bundle, log=log,
                           manifest=manifest, payload=payload, proof=proof, runtime=runtime,
                           failure=None, calls=[], sleeps=[], logs={"out": b"stdout\n", "err": b""}, identity=identity,
                           actual_identity={"command": shlex.join(identity["argv"]), "lstart": identity["actual_process_start"],
                                            "uid": str(os.getuid()), "state": "S"})
    original_files = [bundle, log, runtime, *original_attempt.iterdir(), tmp_path / "campaign-state.json"]
    case.original_sha256 = {path: recovery.digest(path) for path in original_files}

    def execute(command, **options):
        case.calls.append(command)
        text, code = "", 0
        if command[0] == "ps":
            text = case.actual_identity[command[-1].removesuffix("=")] + "\n"
        elif command[0] == "ssh" and "printf 'OWNER|" in command[-1]:
            if case.failure == "alias_timeout":
                raise subprocess.TimeoutExpired(command, 180, output=b"partial owner\n", stderr=b"partial stderr\n")
            target_checksum = "0" * 64 if case.failure == "target_checksum" else checksum
            text = ("OWNER|owner\nHOME|/data/home/owner\nDATA|/data/run01/owner/kvcached-paracloud\n"
                    "123|owner|qwen-public-perf-followup|gpu_a800|COMPLETED|0:0\n"
                    f"TARGET_SHA|{target_checksum}  {proof['alias_target']}\n")
        elif command[0] == "ssh" and "RAW_MANIFEST" in command[-1]:
            if case.failure == "query_timeout":
                raise subprocess.TimeoutExpired(command, 540, output=b"partial accounting\n", stderr=b"query error\n")
            text = query if case.failure != "manifest_checksum" else query.replace(checksum, "0" * 64)
        elif command[0] == "ssh" and "find . -type f" in command[-1]:
            text = manifest if case.failure != "post_manifest" else manifest.replace(checksum, "0" * 64)
        elif command[0] == "ssh" and command[-1].startswith("test -f"):
            code = int(case.failure == "missing_log" and command[-1].endswith('.err"'))
        elif command[0] == "ssh" and command[-1].startswith("sha256sum"):
            extension = "err" if command[-1].endswith('.err"') else "out"
            text = hashlib.sha256(case.logs[extension]).hexdigest() + "  original-log\n"
        elif command[0] == "scp":
            extension = "err" if command[-2].endswith(".err") else "out"
            target = Path(command[-1])
            if case.failure == "log_transport" and extension == "err":
                target.write_bytes(b"partial download\n")
                raise subprocess.CalledProcessError(255, command, stderr="download failed")
            target.write_bytes(case.logs[extension])
        else:
            pytest.fail("Unmocked command, no real network allowed: " + repr(command))
        return subprocess.CompletedProcess(command, code, stdout=text, stderr="")

    monkeypatch.setattr(recovery.subprocess, "run", execute)
    monkeypatch.setattr(recovery.os, "kill", lambda pid, signal: None)
    monkeypatch.setattr(recovery.time, "sleep", case.sleeps.append)
    return case


def assert_original_bytes(case):
    assert {path: recovery.digest(path) for path in case.original_sha256} == case.original_sha256


def retained_attempt(case):
    attempts = sorted(case.directory.glob("progress-alias-attempt-123-*"))
    assert attempts
    return attempts[-1]


def test_real_archiver_install_retains_permanent_reports_and_independent_receipt(recovery_case):
    case = recovery_case
    result = recovery.recover(case.root, "native-N4")
    receipt_path = Path(result["installation_receipt"])
    assert receipt_path.is_file() and not receipt_path.stat().st_mode & 0o222
    receipt = json.loads(receipt_path.read_text())
    assert receipt["coordinator_shares_lock"] is False
    assert (case.directory / "archive-123.json").stat().st_mode & 0o200
    attempt = retained_attempt(case)
    assert json.loads((attempt / "attempt-status.json").read_text())["status"] == "complete_verified"
    reports = list(attempt.rglob("report.json"))
    assert reports
    for original in reports:
        relocation = json.loads(original.with_name("report-relocation.json").read_text())
        resolved = json.loads(Path(relocation["resolved_report"]).read_text())
        assert relocation["original_report_sha256"] == recovery.digest(original)
        assert all(record["exists"] for record in relocation["paths"].values())
        assert all(Path(resolved[field]).exists() for field in relocation["paths"])
        assert Path(resolved["raw"]) == case.directory / "live-trace-123"
    assert (case.directory / "live-trace-123/native-N4/progress.json").read_bytes() == case.payload
    assert_original_bytes(case)


def test_coordinator_real_resume_overwrite_does_not_erase_immutable_proof(recovery_case, monkeypatch):
    case = recovery_case
    result = recovery.recover(case.root, "native-N4")
    receipt = Path(result["installation_receipt"])
    receipt_before = receipt.read_bytes()
    report_path = case.directory / "archive-123.json"
    assert "alias_recovery_proof" in json.loads(report_path.read_text())
    monkeypatch.setattr(sys, "argv", ["archive", "123", "--archive", str(case.directory), "--resume"])
    recovery.archive_trace_job.main()
    assert "alias_recovery_proof" not in json.loads(report_path.read_text())
    assert receipt.read_bytes() == receipt_before
    calls_before = len(case.calls)
    checked = recovery.recover(case.root, "native-N4")
    assert checked["already_SHA_archived"] is True
    assert checked["installation_receipt"] == str(receipt)
    assert len(case.calls) == calls_before
    assert_original_bytes(case)


@pytest.mark.parametrize("failure", ["alias_timeout", "query_timeout", "manifest_checksum", "target_checksum",
                                     "post_manifest", "log_transport", "missing_log"])
def test_all_recovery_failures_retain_new_attempt_and_partial_evidence(recovery_case, failure):
    case = recovery_case
    case.failure = failure
    with pytest.raises((ValueError, subprocess.TimeoutExpired, subprocess.CalledProcessError)):
        recovery.recover(case.root, "native-N4")
    attempt = retained_attempt(case)
    status = json.loads((attempt / "attempt-status.json").read_text())
    assert status["status"] == "failed_preserved" and status["attempt"] == str(attempt)
    assert (attempt / "failure.txt").read_text()
    assert (attempt / "closed-input.json").exists()
    assert not (case.directory / "live-trace-123").exists()
    if failure == "alias_timeout":
        assert (attempt / "samefile-query.stdout").read_bytes() == b"partial owner\n"
    if failure == "query_timeout":
        assert any(path.read_bytes() == b"partial accounting\n" for path in attempt.rglob("query.stdout"))
    if failure == "log_transport":
        assert any(path.read_bytes() == b"partial download\n" for path in attempt.rglob("*.partial-*"))
    if failure in ("manifest_checksum", "target_checksum", "post_manifest"):
        assert list(attempt.rglob("parser-only.tar"))
    if failure == "missing_log":
        report = next(attempt.rglob("report-relocation-resolved-report.json"))
        assert Path(json.loads(report.read_text())["raw"]).exists()
    assert_original_bytes(case)


@pytest.mark.parametrize("missing", ["raw-123.sha256", "live-trace-123.out", "live-trace-123.err", "archive-123.json",
                                     "raw-123-pointers.json", "progress-alias-recovery.json", "accounting-123.txt",
                                     "progress-alias-123-installation.json"])
def test_raw_exists_never_skips_incomplete_installation(recovery_case, missing):
    case = recovery_case
    recovery.recover(case.root, "native-N4")
    (case.directory / missing).unlink()
    raw = case.directory / "live-trace-123/native-N4/progress.json"
    before = raw.read_bytes()
    calls_before = len(case.calls)
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        recovery.recover(case.root, "native-N4")
    assert raw.read_bytes() == before
    assert len(case.calls) == calls_before
    assert any(json.loads(path.read_text())["status"] == "failed_preserved"
               for path in case.directory.glob("progress-alias-attempt-123-*/attempt-status.json"))


@pytest.mark.parametrize("tamper", ["log", "manifest", "proof", "receipt", "retained_input", "report"])
def test_existing_archive_rejects_mismatched_installation_chain(recovery_case, tamper):
    case = recovery_case
    result = recovery.recover(case.root, "native-N4")
    receipt_path = Path(result["installation_receipt"])
    receipt = json.loads(receipt_path.read_text())
    if tamper == "log":
        (case.directory / "live-trace-123.out").write_bytes(b"different\n")
    elif tamper == "manifest":
        (case.directory / "raw-123.sha256").write_text("0" * 64 + case.manifest[64:])
    elif tamper == "proof":
        Path(receipt["proof"]).chmod(0o600)
        Path(receipt["proof"]).write_text("{}")
    elif tamper == "receipt":
        receipt["artifacts"] = []
        receipt_path.chmod(0o600)
        receipt_path.write_text(json.dumps(receipt))
    elif tamper == "retained_input":
        closed = json.loads((Path(receipt["proof"]).parent / "closed-input.json").read_text())
        copied = Path(next(iter(closed["retained_copies"].values())))
        copied.chmod(0o600)
        copied.write_text("different retained input")
    else:
        report_path = case.directory / "archive-123.json"
        report = json.loads(report_path.read_text())
        report["raw_files_verified"] += 1
        report_path.write_text(json.dumps(report))
    with pytest.raises((ValueError, KeyError)):
        recovery.verify_installed(case.directory, case.stage, "native-N4")


@pytest.mark.parametrize("stop_after", ["raw-123.sha256", "archive-123.json", "progress-alias-123-installation.json"])
def test_half_installation_is_preserved_and_never_success(recovery_case, monkeypatch, stop_after):
    case = recovery_case
    original_copy = recovery.atomic_copy

    def interrupted_copy(source, destination, **kwargs):
        original_copy(source, destination, **kwargs)
        if destination.name == stop_after:
            raise RuntimeError("fixture interrupted metadata installation")

    monkeypatch.setattr(recovery, "atomic_copy", interrupted_copy)
    with pytest.raises(RuntimeError, match="interrupted metadata"):
        recovery.recover(case.root, "native-N4")
    attempt = retained_attempt(case)
    assert not (case.directory / "live-trace-123").exists()
    assert (attempt / "verified-stage/live-trace-123/native-N4/progress.json").read_bytes() == case.payload
    assert (case.directory / stop_after).exists()
    before = {path: path.read_bytes() for path in case.directory.iterdir() if path.is_file()}
    with pytest.raises(ValueError):
        recovery.verify_installed(case.directory, case.stage, "native-N4")
    assert {path: path.read_bytes() for path in before} == before
    assert json.loads((attempt / "attempt-status.json").read_text())["status"] == "failed_preserved"
    assert_original_bytes(case)


def test_local_lock_refuses_second_installer_without_claiming_coordinator_lock(recovery_case):
    case = recovery_case
    with recovery.local_install_lock(case.directory):
        with pytest.raises(RuntimeError, match="local alias installer"):
            recovery.recover(case.root, "native-N4")
    assert not case.calls
    assert not list(case.directory.glob("progress-alias-attempt-*"))
    assert_original_bytes(case)


def test_coordinator_raw_appearance_before_commit_is_preserved(recovery_case, monkeypatch):
    case = recovery_case
    original_copy = recovery.atomic_copy
    destination = case.directory / "live-trace-123"

    def concurrent_copy(source, target, **kwargs):
        original_copy(source, target, **kwargs)
        if target.name == "archive-123.json":
            destination.mkdir()
            (destination / "coordinator-evidence").write_bytes(b"preserve concurrent evidence\n")

    monkeypatch.setattr(recovery, "atomic_copy", concurrent_copy)
    with pytest.raises(ValueError, match="final handoff"):
        recovery.recover(case.root, "native-N4")
    assert (destination / "coordinator-evidence").read_bytes() == b"preserve concurrent evidence\n"
    assert (retained_attempt(case) / "verified-stage/live-trace-123/native-N4/progress.json").exists()
    assert_original_bytes(case)


@pytest.mark.parametrize("current_log", ["", "Traceback\nValueError: Different current failure\n"])
def test_follow_records_current_failure_and_stops_not_old_alias_log(recovery_case, current_log):
    case = recovery_case
    (case.directory / "archive-123-attempt-2.log").write_text(current_log)
    with pytest.raises(RuntimeError, match="permanent diagnosis"):
        recovery.follow(case.root, 4)
    row = json.loads((case.root / "progress-alias-follow.jsonl").read_text())
    assert row["status"] == "failed_preserved_stopped" and row["recoveries"] == []
    assert Path(row["diagnosis"]).is_file()
    assert all(command[0] == "ps" for command in case.calls) and not case.sleeps
    assert row["actual_gap_s"] is None and row["elapsed_s"] >= 0


def test_follow_recovery_failure_has_diagnostic_and_retained_attempt(recovery_case):
    case = recovery_case
    case.failure = "log_transport"
    with pytest.raises(RuntimeError, match="permanent diagnosis"):
        recovery.follow(case.root, 2)
    row = json.loads((case.root / "progress-alias-follow.jsonl").read_text())
    assert row["status"] == "failed_preserved_stopped"
    assert list(retained_attempt(case).rglob("*.partial-*"))
    assert Path(row["diagnosis"]).is_file() and not case.sleeps
    assert_original_bytes(case)


def test_follow_does_not_hide_independent_verifier_failure_or_mutate_risk(recovery_case):
    case = recovery_case
    case.state["stages"]["native-N4"].update({"phase": "diagnosis_required", "independent_verification_error": "risk rejection"})
    (case.root / "campaign-state.json").write_text(json.dumps(case.state))
    before = (case.root / "campaign-state.json").read_bytes()
    with pytest.raises(RuntimeError, match="permanent diagnosis"):
        recovery.follow(case.root, 2)
    row = json.loads((case.root / "progress-alias-follow.jsonl").read_text())
    assert "risk rejection" in row["error"]
    assert (case.root / "campaign-state.json").read_bytes() == before
    assert all(command[0] == "ps" for command in case.calls) and not case.sleeps


def test_follow_raw_exists_requires_full_install_verification(recovery_case):
    case = recovery_case
    recovery.recover(case.root, "native-N4")
    (case.directory / "live-trace-123.err").unlink()
    calls_before = len(case.calls)
    with pytest.raises(RuntimeError, match="permanent diagnosis"):
        recovery.follow(case.root, 2)
    row = json.loads((case.root / "progress-alias-follow.jsonl").read_text())
    assert row["status"] == "failed_preserved_stopped"
    assert all(command[0] == "ps" for command in case.calls[calls_before:]) and not case.sleeps


def test_follow_read_failure_still_writes_permanent_diagnosis(recovery_case):
    case = recovery_case
    (case.root / "campaign-state.json").write_text("broken fixture json")
    with pytest.raises(RuntimeError, match="permanent diagnosis"):
        recovery.follow(case.root, 1)
    row = json.loads((case.root / "progress-alias-follow.jsonl").read_text())
    assert "JSONDecodeError" in row["error"] and Path(row["diagnosis"]).is_file()
    assert all(command[0] == "ps" for command in case.calls) and not case.sleeps


@pytest.mark.parametrize("unclosed", ["new_bundle", "new_attempt", "changed_closed_bundle"])
def test_closed_failure_binding_rejects_new_or_mutating_download(recovery_case, unclosed):
    case = recovery_case
    if unclosed == "new_bundle":
        case.bundle.touch()
    elif unclosed == "new_attempt":
        (case.directory / "archive-123-attempt-20261102-110000-000000-new").mkdir()
    else:
        retained = case.directory / "fixture-retained"
        retained.mkdir()
        closed = recovery.bind_closed_attempt(case.directory, 123, retained)
        case.bundle.touch()
        with pytest.raises(ValueError, match="evidence changed"):
            recovery.check_witnesses(closed["witnesses"])
        return
    with pytest.raises(ValueError, match="closed"):
        recovery.recover(case.root, "native-N4")
    assert not case.calls
    assert json.loads((retained_attempt(case) / "attempt-status.json").read_text())["status"] == "failed_preserved"


def test_atomic_publication_preserves_differing_existing_file_and_no_report_alias(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"first\n")
    recovery.atomic_copy(source, target)
    assert source.stat().st_ino != target.stat().st_ino
    target.write_bytes(b"coordinator overwrite\n")
    assert source.read_bytes() == b"first\n"
    with pytest.raises(ValueError, match="differing prior artifact"):
        recovery.atomic_copy(source, target)
    assert target.read_bytes() == b"coordinator overwrite\n"


@pytest.mark.parametrize("field", ["command", "lstart", "uid", "state"])
def test_follow_rejects_actual_pid_reuse_argv_start_owner_and_zombie(recovery_case, field):
    case = recovery_case
    case.actual_identity[field] = "Z" if field == "state" else "unbound actual value"
    with pytest.raises(RuntimeError, match="permanent diagnosis"):
        recovery.follow(case.root, 2)
    row = json.loads((case.root / "progress-alias-follow.jsonl").read_text())
    assert "Actual performance" in row["error"]
    assert all(command[0] == "ps" for command in case.calls) and not case.sleeps
    assert not list(case.directory.glob("progress-alias-attempt-123-*"))


@pytest.mark.parametrize("field,value", [("pid", True), ("actual_process_start", None), ("monitor_interval_s", 30),
                                        ("argv", ["python", "performance_followup.py", "--root", "/different/root"])])
def test_follow_requires_recorded_actual_identity_not_launch_timestamp(recovery_case, field, value):
    case = recovery_case
    case.identity[field] = value
    case.identity["started_at"] = "2026-11-02T07:59:30+08:00"
    (case.root / "local-monitor-identity.json").write_text(json.dumps(case.identity))
    with pytest.raises(RuntimeError, match="permanent diagnosis"):
        recovery.follow(case.root, 2)
    assert not case.calls and not case.sleeps


def test_follow_records_actual_gap_elapsed_and_waits_300_after_recovery(recovery_case, monkeypatch):
    case = recovery_case
    clock = [0.0]

    def bounded_recovery(root, stage_id):
        clock[0] += 7
        return {"job": 123, "already_SHA_archived": False}

    def sleep(delay):
        case.sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr(recovery, "recover", bounded_recovery)
    monkeypatch.setattr(recovery.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(recovery.time, "sleep", sleep)
    recovery.follow(case.root, 2)
    rows = [json.loads(line) for line in (case.root / "progress-alias-follow.jsonl").read_text().splitlines()]
    assert rows[0]["actual_gap_s"] is None and rows[1]["actual_gap_s"] == 307
    assert all(row["elapsed_s"] == 7 and row["next_delay_s"] == 300 for row in rows)
    assert case.sleeps == [300, 300]
    assert rows[0]["performance_identity"]["argv"] == case.identity["argv"]
    assert rows[0]["performance_identity"]["actual_process_start"] == case.identity["actual_process_start"]
    assert rows[0]["performance_identity"]["root"] == str(case.root)


def test_follow_identity_change_after_sleep_is_diagnosed_without_more_recovery(recovery_case, monkeypatch):
    case = recovery_case
    calls = []
    monkeypatch.setattr(recovery, "recover", lambda root, stage_id: calls.append(stage_id) or {"job": 123})

    def change_identity(delay):
        case.sleeps.append(delay)
        case.identity["pid"] += 1
        (case.root / "local-monitor-identity.json").write_text(json.dumps(case.identity))

    monkeypatch.setattr(recovery.time, "sleep", change_identity)
    with pytest.raises(RuntimeError, match="permanent diagnosis"):
        recovery.follow(case.root, 3)
    rows = [json.loads(line) for line in (case.root / "progress-alias-follow.jsonl").read_text().splitlines()]
    assert len(calls) == 1 and len(case.sleeps) == 1
    assert rows[-1]["status"] == "failed_preserved_stopped" and "identity changed" in rows[-1]["error"]


def test_follow_skips_already_closed_legacy_stage_without_claiming_installation_verification(recovery_case, monkeypatch):
    case = recovery_case
    state = json.loads((case.root / "campaign-state.json").read_text())
    state["stages"]["native-N4"]["phase"] = "verified"
    (case.root / "campaign-state.json").write_text(json.dumps(state))
    monkeypatch.setattr(recovery, "recover", lambda *args: pytest.fail("closed workload must not be rerun or recovered"))
    monkeypatch.setattr(recovery, "verify_installed", lambda *args: pytest.fail("legacy stage has no NEW installation receipt"))
    recovery.follow(case.root, 1)
    row = json.loads((case.root / "progress-alias-follow.jsonl").read_text())
    assert row["status"] == "observed_not_quality_verified" and row["recoveries"] == []
    assert row["closed_stages_not_reverified"] == ["native-N4"]
    assert json.loads((case.root / "campaign-state.json").read_text()) == state


@pytest.mark.parametrize("empty_destination", [False, True])
def test_exclusive_raw_rename_preserves_existing_even_empty_directory(tmp_path, empty_destination):
    source, target = tmp_path / "verified", tmp_path / "canonical"
    source.mkdir()
    (source / "payload").write_bytes(b"verified\n")
    target.mkdir()
    if not empty_destination:
        (target / "prior").write_bytes(b"existing\n")
    with pytest.raises(FileExistsError):
        recovery.exclusive_raw_rename(source, target)
    assert (source / "payload").read_bytes() == b"verified\n"
    assert set(path.name for path in target.iterdir()) == (set() if empty_destination else {"prior"})


def test_new_coordinator_attempt_during_recovery_blocks_install(recovery_case, monkeypatch):
    case = recovery_case
    original_archiver = recovery.run_archiver

    def begin_new_attempt(recovered, bundle, proof):
        original_archiver(recovered, bundle, proof)
        (case.directory / "archive-123-attempt-20261102-110000-000000-incoming").mkdir()

    monkeypatch.setattr(recovery, "run_archiver", begin_new_attempt)
    with pytest.raises(ValueError, match="New coordinator attempt"):
        recovery.recover(case.root, "native-N4")
    assert not (case.directory / "live-trace-123").exists()
    assert (retained_attempt(case) / "verified-stage/live-trace-123/native-N4/progress.json").exists()
    assert_original_bytes(case)


def test_historical_report_remap_is_new_immutable_and_never_rewrites_original(recovery_case):
    case = recovery_case
    recovery.recover(case.root, "native-N4")
    canonical = json.loads((case.directory / "archive-123.json").read_text())
    original = case.directory / "historical-attempt/report.json"
    original.parent.mkdir()
    old = dict(canonical)
    for field in ("raw", "attempt", "raw_manifest", "pointer_metadata"):
        old[field] = "/already-removed/alias-archive-old/" + Path(canonical[field]).name
    original.write_text(json.dumps(old))
    original.chmod(0o444)
    before = original.read_bytes()
    remap_path = case.directory / "new-historical-path-remap.json"
    recovery.write_report_path_remap(original, {field: canonical[field] for field in old}, remap_path)
    remap = json.loads(remap_path.read_text())
    assert original.read_bytes() == before and remap["original_report_sha256"] == recovery.digest(original)
    assert all(Path(record["permanent"]).exists() for record in remap["paths"].values())
    assert remap["path_mapping_only"] is True and remap["numerical_quality_verified"] is False
    assert not remap_path.stat().st_mode & 0o222
    resolved = Path(remap["resolved_report"])
    assert recovery.digest(resolved) == remap["resolved_report_sha256"] and not resolved.stat().st_mode & 0o222
    with pytest.raises(FileExistsError):
        recovery.write_report_path_remap(original, canonical, remap_path)
    assert original.read_bytes() == before


def test_path_remap_refuses_missing_permanent_target_without_rewriting_report(recovery_case):
    case = recovery_case
    recovery.recover(case.root, "native-N4")
    original = case.directory / "archive-123.json"
    before = original.read_bytes()
    remap_path = case.directory / "new-invalid-remap.json"
    with pytest.raises(ValueError, match="permanent report path"):
        recovery.write_report_path_remap(original, {"raw": case.root / "missing"}, remap_path)
    assert original.read_bytes() == before and not remap_path.exists()


def test_attempt_is_permanent_and_marked_incomplete_before_archiver_runs(recovery_case, monkeypatch):
    case = recovery_case

    def interrupted(recovered, bundle, proof):
        retained = recovered.parent
        start = json.loads((retained / "attempt-start.json").read_text())
        assert start["status"] == "started_incomplete" and start["attempt"] == str(retained)
        assert not (retained / "attempt-status.json").exists()
        (recovered / "partial-query-evidence").write_bytes(b"preserve interrupted attempt\n")
        raise KeyboardInterrupt("fixture interrupted archiver")

    monkeypatch.setattr(recovery, "run_archiver", interrupted)
    with pytest.raises(KeyboardInterrupt):
        recovery.recover(case.root, "native-N4")
    retained = retained_attempt(case)
    assert (retained / "verified-stage/partial-query-evidence").read_bytes() == b"preserve interrupted attempt\n"
    assert json.loads((retained / "attempt-status.json").read_text())["status"] == "failed_preserved"
    assert_original_bytes(case)


def test_report_overwrite_during_verification_fails_closed(recovery_case, monkeypatch):
    case = recovery_case
    recovery.recover(case.root, "native-N4")
    original_digest = recovery.digest
    progress = case.directory / "live-trace-123/native-N4/progress.json"
    report_path = case.directory / "archive-123.json"
    changed = []

    def concurrent_digest(path):
        if Path(path) == progress and not changed:
            report = json.loads(report_path.read_text())
            report["concurrent_resume_receipt"] = True
            report_path.write_text(json.dumps(report))
            changed.append(True)
        return original_digest(path)

    monkeypatch.setattr(recovery, "digest", concurrent_digest)
    with pytest.raises(ValueError, match="report changed"):
        recovery.verify_installed(case.directory, case.stage, "native-N4")


def test_follow_checks_actual_frozen_source_even_when_raw_exists(recovery_case):
    case = recovery_case
    recovery.recover(case.root, "native-N4")
    case.runtime.write_bytes(b"fixture changed current performance source\n")
    with pytest.raises(RuntimeError, match="permanent diagnosis"):
        recovery.follow(case.root, 2)
    row = json.loads((case.root / "progress-alias-follow.jsonl").read_text())
    assert "source SHA changed" in row["error"] and not case.sleeps


def test_exclusive_raw_rename_unsupported_platform_preserves_stage(tmp_path, monkeypatch):
    source, target = tmp_path / "verified", tmp_path / "canonical"
    source.mkdir()
    (source / "payload").write_bytes(b"verified\n")
    monkeypatch.setattr(recovery.sys, "platform", "unsupported-host")
    with pytest.raises(RuntimeError, match="exclusive raw rename"):
        recovery.exclusive_raw_rename(source, target)
    assert (source / "payload").read_bytes() == b"verified\n" and not target.exists()


def test_linux_exclusive_raw_rename_uses_reviewed_no_replace_abi(tmp_path, monkeypatch):
    source, target = tmp_path / "verified", tmp_path / "canonical"
    calls = []

    class Operation:
        def __call__(self, *arguments):
            calls.append(arguments)
            return 0

    operation = Operation()
    monkeypatch.setattr(recovery.sys, "platform", "linux")
    monkeypatch.setattr(recovery.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(renameat2=operation))
    recovery.exclusive_raw_rename(source, target)
    assert calls == [(-100, os.fsencode(source), -100, os.fsencode(target), 1)]
    assert operation.restype is recovery.ctypes.c_int
