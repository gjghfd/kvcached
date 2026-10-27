import hashlib
import importlib
import io
import os
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

import pytest


@pytest.fixture
def archiver(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("archive_trace_job")


def make_bundle(tmp_path, link_name="./progress.json", target=None, kind=tarfile.SYMTYPE):
    bundle = tmp_path / "raw.tar.gz"
    payload = b'{"phase":"done"}\n'
    with tarfile.open(bundle, "w:gz") as contents:
        member = tarfile.TarInfo("./recovery-2/progress.json")
        member.size = len(payload)
        contents.addfile(member, io.BytesIO(payload))
        pointer = tarfile.TarInfo(link_name)
        pointer.type = kind
        pointer.linkname = target or "/data/home/user/run/kvcached-paracloud/state/live-trace-123/recovery-2/progress.json"
        contents.addfile(pointer)
    manifest = hashlib.sha256(payload).hexdigest() + "  ./recovery-2/progress.json\n"
    return bundle, manifest


def test_progress_pointer_is_recorded_not_extracted(archiver, tmp_path):
    bundle, manifest = make_bundle(tmp_path)
    raw = tmp_path / "raw"
    pointers = archiver.extract_raw(bundle, raw, manifest, 123)
    assert pointers[0]["archived_target"] == "recovery-2/progress.json"
    assert pointers[0]["action"] == "recorded_not_extracted"
    assert not (raw / "progress.json").exists()
    archiver.verify_raw(raw, manifest)


@pytest.mark.parametrize("kwargs", [
    {"link_name": "./other.json"},
    {"link_name": "../progress.json"},
    {"link_name": "/progress.json"},
    {"kind": tarfile.LNKTYPE},
    {"target": "/etc/passwd"},
    {"target": "/data/run/kvcached-paracloud/state/live-trace-124/recovery-2/progress.json"},
    {"target": "/data/run/kvcached-paracloud/state/live-trace-123/missing/progress.json"},
    {"target": "/data/run/kvcached-paracloud/state/live-trace-123/../recovery-2/progress.json"},
    {"target": "recovery-2/progress.json"},
])
def test_archive_rejects_unsafe_links_before_extracting(archiver, tmp_path, kwargs):
    bundle, manifest = make_bundle(tmp_path, **kwargs)
    raw = tmp_path / "raw"
    with pytest.raises(ValueError):
        archiver.extract_raw(bundle, raw, manifest, 123)
    assert not raw.exists()


def test_archive_still_requires_exact_manifest(archiver, tmp_path):
    bundle, manifest = make_bundle(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        archiver.extract_raw(bundle, tmp_path / "raw", "0" * 64 + manifest[64:], 123)


@pytest.fixture
def remote_environment(tmp_path):
    root = tmp_path / "home/run/kvcached-paracloud/state/live-trace-123"
    root.mkdir(parents=True)
    commands = tmp_path / "commands"
    commands.mkdir()
    scripts = {
        "sacct": '#!/bin/sh\nprintf "%s\\n" "$FAKE_ACCOUNTING"\n',
        "sleep": '#!/bin/sh\nprintf "%s\\n" "$*" >> "$SLEEP_CALLS"\n'
                 'if [ -n "${BARRIER_EVENT:-}" ]; then /bin/sh "$BARRIER_EVENT"; fi\n',
        "pgrep": '#!/bin/sh\necho forbidden-host-local-pgrep >&2\nexit 99\n',
        "sha256sum": f'#!{sys.executable}\nimport hashlib, pathlib, sys\n'
                     'for filename in sys.argv[1:]:\n'
                     '    print(hashlib.sha256(pathlib.Path(filename).read_bytes()).hexdigest() + "  " + filename)\n',
    }
    for name, script in scripts.items():
        executable = commands / name
        executable.write_text(script)
        executable.chmod(0o755)
    environment = dict(os.environ, HOME=str(tmp_path / "home"),
                       PATH=str(commands) + os.pathsep + os.environ["PATH"],
                       FAKE_ACCOUNTING="123|COMPLETED|0:0|submit|start|end|elapsed|gpu-node|gpu",
                       SLEEP_CALLS=str(tmp_path / "sleeps"), LC_ALL="C", LC_CTYPE="C", LANG="C")
    return root, environment


def marker_text(status="running", **fields):
    marker = {"job": "123", "hostname": "ln03", "time": "2026-10-27T13:00:00Z",
              "pid": "456", "run_id": "ln03:456:2026-10-27T13:00:00Z", "exit_status": status}
    marker.update(fields)
    return "".join(f"{key}={value}\n" for key, value in marker.items())


def run_barrier(archiver, environment, legacy=False):
    command = archiver.archive_command(123, legacy)
    assert "pgrep" not in command
    return subprocess.run(["/bin/sh", "-c", command], env=environment,
                          text=True, capture_output=True, timeout=10)


@pytest.mark.parametrize("status", ["0", "124", "143"])
def test_shared_barrier_accepts_matching_exit_from_another_login_host(archiver, remote_environment, status):
    root, environment = remote_environment
    (root / "remote-query.expected").write_text("coordinator expects watcher\n")
    (root / "remote-query.started").write_text(marker_text())
    (root / "remote-query.exit").write_text(marker_text(status))
    (root / "remote-query.log").write_text("final log\n")
    result = run_barrier(archiver, environment)
    assert result.returncode == 0, result.stderr
    assert "WATCHER_BARRIER\nshared_markers\nRAW_MANIFEST\n" in result.stdout
    assert hashlib.sha256(b"final log\n").hexdigest() in result.stdout
    assert not Path(environment["SLEEP_CALLS"]).exists()


@pytest.mark.parametrize("markers", [("expected",), ("started",), ("expected", "started"), ("exit",)])
def test_shared_barrier_waits_boundedly_without_complete_markers(archiver, remote_environment, markers):
    root, environment = remote_environment
    for name in markers:
        (root / f"remote-query.{name}").write_text(marker_text("0" if name == "exit" else "running"))
    result = run_barrier(archiver, environment, legacy=True)
    assert result.returncode == 3
    assert "barrier incomplete" in result.stderr
    assert "RAW_MANIFEST" not in result.stdout
    assert Path(environment["SLEEP_CALLS"]).read_text().splitlines() == ["20"] * 20


def test_expected_marker_closes_watcher_launch_gap_before_hashing(archiver, remote_environment, tmp_path):
    root, environment = remote_environment
    (root / "remote-query.expected").write_text("")
    started = tmp_path / "started"
    exited = tmp_path / "exited"
    started.write_text(marker_text())
    exited.write_text(marker_text("0"))
    event = tmp_path / "event.sh"
    event.write_text(f'cp "{started}" "{root}/.started.tmp"\n'
                     f'mv "{root}/.started.tmp" "{root}/remote-query.started"\n'
                     f'printf "last query\\n" >> "{root}/remote-query.log"\n'
                     f'cp "{exited}" "{root}/.exit.tmp"\n'
                     f'mv "{root}/.exit.tmp" "{root}/remote-query.exit"\n')
    environment["BARRIER_EVENT"] = str(event)
    result = run_barrier(archiver, environment)
    assert result.returncode == 0, result.stderr
    assert Path(environment["SLEEP_CALLS"]).read_text() == "20\n"
    assert hashlib.sha256(b"last query\n").hexdigest() in result.stdout


@pytest.mark.parametrize("fields", [
    {"job": "124"}, {"hostname": "ln02"}, {"pid": "789"}, {"run_id": "old-watcher"},
    {"exit_status": "running"}, {"exit_status": "256"}, {"exit_status": "-1"}, {"time": ""},
])
def test_shared_barrier_rejects_stale_or_malformed_exit(archiver, remote_environment, fields):
    root, environment = remote_environment
    (root / "remote-query.started").write_text(marker_text())
    (root / "remote-query.exit").write_text(marker_text("0", **fields))
    result = run_barrier(archiver, environment)
    assert result.returncode == 3
    assert "mismatched shared watcher markers" in result.stderr
    assert "RAW_MANIFEST" not in result.stdout


@pytest.mark.parametrize("name", ["expected", "started", "exit"])
def test_shared_barrier_rejects_marker_symlinks(archiver, remote_environment, name):
    root, environment = remote_environment
    (root / f"remote-query.{name}").symlink_to(root / "missing")
    result = run_barrier(archiver, environment, legacy=True)
    assert result.returncode == 3
    assert "Unsafe shared watcher marker" in result.stderr


def test_markerless_query_log_requires_explicit_legacy_attestation(archiver, remote_environment):
    root, environment = remote_environment
    (root / "remote-query.log").write_text("old query log\n")
    refused = run_barrier(archiver, environment)
    assert refused.returncode == 3
    assert "--legacy-watcher-stopped" in refused.stderr
    assert "host-local process lookup is not a barrier" in refused.stderr
    accepted = run_barrier(archiver, environment, legacy=True)
    assert accepted.returncode == 0, accepted.stderr
    assert "WATCHER_BARRIER\nlegacy_watcher_stopped_attestation\n" in accepted.stdout
    assert "caller attests" in accepted.stderr
    assert hashlib.sha256(b"old query log\n").hexdigest() in accepted.stdout


def test_no_watcher_without_markers_or_query_log_is_explicit(archiver, remote_environment):
    root, environment = remote_environment
    (root / "evidence.json").write_text("{}")
    result = run_barrier(archiver, environment)
    assert result.returncode == 0, result.stderr
    assert "WATCHER_BARRIER\nno_watcher\n" in result.stdout


def test_active_job_is_rejected_before_watcher_barrier(archiver, remote_environment):
    _, environment = remote_environment
    environment["FAKE_ACCOUNTING"] = "123|RUNNING|0:0"
    result = run_barrier(archiver, environment)
    assert result.returncode == 2
    assert "active job" in result.stderr
    assert not Path(environment["SLEEP_CALLS"]).exists()


def raw_bundle(payload):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as contents:
        member = tarfile.TarInfo("./remote-query.log")
        member.size = len(payload)
        contents.addfile(member, io.BytesIO(payload))
    manifest = hashlib.sha256(payload).hexdigest() + "  ./remote-query.log\n"
    return buffer.getvalue(), manifest


def mock_archive_remote(archiver, monkeypatch, manifest, bundle, failure=None, state="COMPLETED"):
    calls = []

    def execute(command, **options):
        calls.append(command)
        text = ""
        code = 0
        if "RAW_MANIFEST" in command[-1]:
            text = f"123|{state}|0:0|submit|start|end|elapsed|node|gpu\n\nWATCHER_BARRIER\nshared_markers\nRAW_MANIFEST\n" + manifest
        elif command[-1].startswith("tar -C"):
            options["stdout"].write(bundle)
            if failure == "transport":
                raise subprocess.CalledProcessError(255, command, stderr="interrupted download")
        elif "find . -type f" in command[-1]:
            text = manifest if failure != "post_manifest" else manifest.replace("./remote-query.log", "./changed.log")
        elif command[-1].startswith("test -f"):
            code = 1
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, code, stdout=text, stderr="")

    monkeypatch.setattr(archiver.subprocess, "run", execute)
    return calls


@pytest.mark.parametrize("failure", ["transport", "checksum", "post_manifest"])
def test_archive_attempts_preserve_first_expected_manifest_and_residual_bundle(
        archiver, monkeypatch, tmp_path, failure):
    class FrozenDatetime:
        @staticmethod
        def now():
            return datetime(2026, 10, 27, 21, 18)

    monkeypatch.setattr(archiver, "datetime", FrozenDatetime)
    monkeypatch.setattr(sys, "argv", ["archive", "123", "--archive", str(tmp_path)])
    first_bundle, first_manifest = raw_bundle(b"before\n")
    next_bundle, next_manifest = raw_bundle(b"after\n")
    downloaded = next_bundle if failure == "checksum" else first_bundle
    mock_archive_remote(archiver, monkeypatch, first_manifest, downloaded, failure, state="FAILED")
    expected_error = subprocess.CalledProcessError if failure == "transport" else ValueError
    with pytest.raises(expected_error):
        archiver.main()
    first_attempt = next(tmp_path.glob("archive-123-attempt-*"))
    first_files = {path: path.read_bytes() for path in first_attempt.iterdir() if path.is_file()}
    residual = next(tmp_path.glob("raw-123-*.tar.gz"))
    assert residual.read_bytes() == downloaded
    assert (first_attempt / "raw.sha256").read_text() == first_manifest
    assert not (tmp_path / "raw-123.sha256").exists()
    assert not (tmp_path / "accounting-123.txt").exists()
    assert not (tmp_path / "live-trace-123").exists()
    mock_archive_remote(archiver, monkeypatch, next_manifest, next_bundle)
    archiver.main()
    attempts = list(tmp_path.glob("archive-123-attempt-*"))
    assert len(attempts) == 2
    assert all(path.read_bytes() == content for path, content in first_files.items())
    assert (tmp_path / "accounting-123.txt").read_text().startswith("123|COMPLETED|")
    assert (tmp_path / "raw-123.sha256").read_text() == next_manifest
    assert residual.read_bytes() == downloaded
    assert len(list(tmp_path.glob("raw-123-*.tar.gz"))) == 2
    assert (tmp_path / "live-trace-123/remote-query.log").read_bytes() == b"after\n"
    monkeypatch.setattr(sys, "argv", ["archive", "123", "--archive", str(tmp_path), "--resume"])
    calls = mock_archive_remote(archiver, monkeypatch, next_manifest, next_bundle)
    archiver.main()
    assert not any(command[-1].startswith("tar -C") for command in calls)
    assert (tmp_path / "raw-123.sha256").read_text() == next_manifest
    assert residual.read_bytes() == downloaded


def test_existing_canonical_manifest_rejects_difference_without_overwrite(archiver, monkeypatch, tmp_path):
    bundle, manifest = raw_bundle(b"new snapshot\n")
    old_manifest = "original expected manifest\n"
    (tmp_path / "raw-123.sha256").write_text(old_manifest)
    (tmp_path / "accounting-123.txt").write_text("original accounting\n")
    monkeypatch.setattr(sys, "argv", ["archive", "123", "--archive", str(tmp_path)])
    calls = mock_archive_remote(archiver, monkeypatch, manifest, bundle)
    with pytest.raises(ValueError, match="prior expected manifest"):
        archiver.main()
    assert len(calls) == 1
    attempt = next(tmp_path.glob("archive-123-attempt-*"))
    assert (attempt / "raw.sha256").read_text() == manifest
    assert (tmp_path / "raw-123.sha256").read_text() == old_manifest
    assert (tmp_path / "accounting-123.txt").read_text() == "original accounting\n"


@pytest.mark.parametrize("timeout", [False, True])
def test_failed_remote_query_preserves_per_attempt_accounting(archiver, monkeypatch, tmp_path, timeout):
    monkeypatch.setattr(sys, "argv", ["archive", "123", "--archive", str(tmp_path)])
    (tmp_path / "accounting-123.txt").write_text("original accounting\n")
    (tmp_path / "raw-123.sha256").write_text("original expected manifest\n")
    output = "123|COMPLETED|0:0\n"
    error_text = "Shared watcher exit barrier incomplete\n"

    def execute(command, **options):
        if timeout:
            raise subprocess.TimeoutExpired(command, 540, output=output.encode(), stderr=error_text.encode())
        return subprocess.CompletedProcess(command, 3, stdout=output, stderr=error_text)

    monkeypatch.setattr(archiver.subprocess, "run", execute)
    with pytest.raises(subprocess.TimeoutExpired if timeout else subprocess.CalledProcessError):
        archiver.main()
    attempt = next(tmp_path.glob("archive-123-attempt-*"))
    assert (attempt / "accounting.txt").read_text() == output
    assert (attempt / "query.stdout").read_text() == output
    assert (attempt / "query.stderr").read_text() == error_text
    assert not (attempt / "raw.sha256").exists()
    assert (tmp_path / "accounting-123.txt").read_text() == "original accounting\n"
    assert (tmp_path / "raw-123.sha256").read_text() == "original expected manifest\n"
