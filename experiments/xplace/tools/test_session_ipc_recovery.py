import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import sys
import tempfile
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import session_ipc_recovery as recovery


@pytest.fixture
def context(tmp_path, monkeypatch):
    source = tmp_path / "campaign.json"
    source.write_text(json.dumps({"status": "active"}))
    config = {"root": str(tmp_path), "workspace": str(tmp_path), "campaign_file": str(source)}
    request = {"source_campaign": str(source), "campaign_sha256": recovery.digest(source),
               "expected_turn_id": "current-turn", "wait_seconds": 10, "verify_seconds": 10}
    latest = {"turn_id": "current-turn", "status": "completed", "user_pending": False}
    monkeypatch.setattr(recovery.watchdog, "load_campaign", lambda config: {"status": "active"})
    monkeypatch.setattr(recovery.watchdog, "read_rollout", lambda config: dict(latest))
    return config, request, latest


@pytest.mark.parametrize("change", ["stop", "campaign", "user", "new_turn", "failed"])
def test_guard_rejects_changed_context(context, change):
    config, request, latest = context
    if change == "stop":
        (Path(config["root"]) / "STOP").touch()
    elif change == "campaign":
        Path(request["source_campaign"]).write_text("{}")
    elif change == "user":
        latest["user_pending"] = True
    elif change == "new_turn":
        latest["turn_id"] = "manual-turn"
    else:
        latest["status"] = "failed"
    with pytest.raises(ValueError):
        recovery.guard(config, request)


def test_active_turn_wait_times_out_without_reload(context, monkeypatch):
    config, request, latest = context
    latest["status"] = "active"
    clock = iter([0, 0, 11])
    monkeypatch.setattr(recovery.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(recovery.time, "sleep", lambda seconds: None)
    with pytest.raises(TimeoutError, match="no_reload"):
        recovery.wait_for_completion(config, request)


def test_normal_completion_required(context):
    config, request, latest = context
    assert recovery.wait_for_completion(config, request) == latest


def test_expired_recovery_never_reloads(context):
    config, request, latest = context
    request["expires_at"] = "2000-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="expired"):
        recovery.guard(config, request)


@pytest.mark.parametrize("outside", [None, "source_campaign", "output", "config", "bindings"])
def test_background_paths_must_stay_in_runtime(tmp_path, outside):
    config = {"root": str(tmp_path)}
    request = {name: str(tmp_path / name) for name in ("source_campaign", "output", "config")}
    request["bindings"] = [{"path": str(tmp_path / "helper.py")}]
    if outside == "bindings":
        request["bindings"][0]["path"] = str(tmp_path.parent / "desktop.py")
    elif outside:
        request[outside] = str(tmp_path.parent / "desktop")
    if outside:
        with pytest.raises(ValueError, match="runtime_directory"):
            recovery.validate_runtime_paths(config, request)
    else:
        recovery.validate_runtime_paths(config, request)


def test_receipt_never_overwrites(tmp_path):
    path = tmp_path / "receipt.json"
    recovery.write_once(path, {"original": True})
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        recovery.write_once(path, {"original": False})
    assert recovery.digest(path) == before


@pytest.fixture
def bridge(context, tmp_path, monkeypatch):
    config, request, latest = context
    directory = tempfile.TemporaryDirectory(prefix="ipc-recovery-", dir="/tmp")
    bridge_path = Path(directory.name) / "bridge.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(bridge_path))
    bridge_path.chmod(0o600)
    server.listen(1)
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps({
        "workspaceName": tmp_path.name, "socketPath": str(bridge_path),
        "marketplaceID": "openai.chatgpt", "capabilities": {"reload": 1},
    }))
    request.update(bridge_registration=str(registration_path),
                   bridge_registration_sha256=recovery.digest(registration_path),
                   bridge_socket=str(bridge_path), bridge_inode=bridge_path.stat().st_ino,
                   bridge_peer_pid=os.getpid())
    monkeypatch.setattr(recovery, "peer_pid", lambda connection: os.getpid())
    try:
        yield config, request, latest, server, tmp_path
    finally:
        server.close()
        directory.cleanup()


@pytest.mark.parametrize("change", ["permissions", "registration", "inode", "peer", "active"])
def test_wrong_or_active_window_never_reloaded(bridge, change):
    config, request, latest, server, output = bridge
    if change == "permissions":
        Path(request["bridge_socket"]).chmod(0o666)
    elif change == "registration":
        Path(request["bridge_registration"]).write_text("{}")
    elif change == "inode":
        request["bridge_inode"] += 1
    elif change == "peer":
        request["bridge_peer_pid"] += 1
    else:
        latest["status"] = "active"
    with pytest.raises(ValueError):
        recovery.reload_original_window(config, request, output)
    assert not (output / "reload-claim.json").exists()


def test_reload_framing_and_no_retry(bridge):
    config, request, latest, server, output = bridge
    received = []

    def accept():
        connection, address = server.accept()
        with connection:
            length = struct.unpack("<I", connection.recv(4))[0]
            received.append(json.loads(connection.recv(length)))

    receiver = threading.Thread(target=accept)
    receiver.start()
    recovery.reload_original_window(config, request, output)
    receiver.join(timeout=2)
    assert received == [{"command": "reload", "payload": {}}]
    with pytest.raises(FileExistsError):
        recovery.reload_original_window(config, request, output)


def test_snapshot_must_pass_existing_owner_guard(context, monkeypatch):
    config, request, latest = context
    calls = []

    class Client:
        def __init__(self, config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            pass

        def snapshot(self):
            return {"owner": "unchanged", "state": {}}

    clock = iter([0, 11])
    monkeypatch.setattr(recovery.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(recovery.watchdog, "verify_extension", lambda config: None)
    monkeypatch.setattr(recovery.watchdog, "snapshot_guard",
                        lambda snapshot, config, latest: calls.append(snapshot) or "owner_not_idle")
    with pytest.raises(TimeoutError, match="owner_not_idle"):
        recovery.verify_reconnected(config, request, Client)
    assert len(calls) == 1


def test_successful_snapshot_is_read_only(context, monkeypatch):
    config, request, latest = context
    snapshot = {"owner": "owner", "state": {}}

    class Client:
        def __init__(self, config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            pass

        def snapshot(self):
            return snapshot

    monkeypatch.setattr(recovery.watchdog, "verify_extension", lambda config: None)
    monkeypatch.setattr(recovery.watchdog, "snapshot_guard", lambda *args: None)
    monkeypatch.setattr(recovery.watchdog, "summarize_snapshot", lambda value: value)
    observed, failures = recovery.verify_reconnected(config, request, Client)
    assert observed == snapshot
    assert failures == []


def test_user_turn_during_wait_aborts(context, monkeypatch):
    config, request, latest = context
    latest["status"] = "active"
    monkeypatch.setattr(recovery.time, "sleep", lambda seconds: latest.update(turn_id="manual-turn"))
    with pytest.raises(ValueError, match="turn_changed"):
        recovery.wait_for_completion(config, request)


def test_publish_does_not_advance_experiment_gates(context, tmp_path, monkeypatch):
    config, request, latest = context
    campaign = {
        "status": "active", "stage": "capacity_rejected", "query_policy": {},
        "next_actions": ["recover IPC", "repair capacity"], "findings": ["failure retained"],
        "completion_gate": [{"name": "tp2", "passed": False, "evidence": []}],
        "jobs": [{"job": "312336", "state": "FAILED"}],
    }
    source = Path(request["source_campaign"])
    source.write_text(json.dumps(campaign))
    request["campaign_sha256"] = recovery.digest(source)
    result = tmp_path / "result.json"
    result.write_text('{"status":"same_thread_snapshot_restored"}')
    monkeypatch.setattr(recovery.watchdog, "validate_campaign", lambda value, config: value)
    monkeypatch.setattr(recovery.watchdog, "publish_campaign", lambda config, source: {"published": True})
    recovery.publish_recovery(config, request, result)
    updated = json.loads(source.read_text())
    assert updated["completion_gate"] == campaign["completion_gate"]
    assert updated["jobs"] == campaign["jobs"]
    assert updated["status"] == "active"
    assert updated["watchdog_health"]["snapshot_recovery_verified"]
    assert updated["query_policy"]["automatic_continuation_available"]
