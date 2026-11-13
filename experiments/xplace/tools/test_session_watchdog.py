import datetime
import fcntl
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
import session_watchdog as watchdog


OWNER = "c8f7d869-bce2-45f9-8409-4a232a17dcb5"
NOW = datetime.datetime(2026, 11, 8, 15, tzinfo=datetime.timezone.utc)


def append_event(config, kind, turn="failed-turn", **fields):
    payload = {"type": kind, **fields}
    if turn:
        payload["turn_id"] = turn
    with Path(config["rollout"]).open("a") as stream:
        stream.write(json.dumps({"type": "event_msg", "timestamp": "2026-11-08T14:55:00Z",
                                 "payload": payload}) + "\n")


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    config = {"root": str(tmp_path), "workspace": str(tmp_path), "rollout": str(tmp_path / "rollout.jsonl"),
              "thread_id": "01a1b49f-9449-7941-9407-da3901aa53a2", "interval_seconds": 300,
              "armed_at": "2026-11-08T14:30:00Z", "error_grace_seconds": 30}
    Path(config["rollout"]).write_text(json.dumps({"type": "session_meta", "payload": {
        "id": config["thread_id"], "cwd": config["workspace"]}}) + "\n")
    append_event(config, "task_started")
    append_event(config, "task_complete", error={"message": "stream disconnected"})
    monkeypatch.setattr(watchdog, "utc_now", lambda: NOW)
    monkeypatch.setattr(watchdog, "verify_extension", lambda config: None)
    return config


class FakeClient:
    def __init__(self, config):
        self.config = config
        self.sent = []
        self.messages = []
        self.before_snapshot = lambda: None
        self.runtime = "idle"
        self.requests = []
        self.cwd = config["workspace"]
        self.response = {"resultType": "success", "handledByClientId": OWNER,
                         "result": {"result": {"turn": {"id": "new-turn", "status": "inProgress"}}}}

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        pass

    def snapshot(self):
        self.before_snapshot()
        return {"owner": OWNER, "state": {
            "id": self.config["thread_id"], "cwd": self.cwd, "resumeState": "resumed",
            "threadRuntimeStatus": {"type": self.runtime}, "requests": self.requests}}

    def start_turn(self, owner, failed_turn, message=None):
        self.sent.append((owner, failed_turn))
        self.messages.append(message)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def run_tick(config, client=None):
    client = client or FakeClient(config)
    result = watchdog.tick(config, client_factory=lambda unused: client, now=NOW)
    return result, client


@pytest.mark.parametrize("interval_seconds", [300, 1800])
def test_config_accepts_supported_monitor_intervals(campaign, interval_seconds):
    campaign["interval_seconds"] = interval_seconds
    config_path = Path(campaign["root"]) / "config.json"
    config_path.write_text(json.dumps(campaign))
    assert watchdog.load_config(config_path)["interval_seconds"] == interval_seconds


@pytest.mark.parametrize("interval_seconds", [True, 0, -1, 60, 300.0, "1800"])
def test_config_rejects_unsupported_monitor_intervals(campaign, interval_seconds):
    campaign["interval_seconds"] = interval_seconds
    config_path = Path(campaign["root"]) / "config.json"
    config_path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="interval must be"):
        watchdog.load_config(config_path)


def test_half_hour_check_records_next_check_and_preserves_once_only_delivery(campaign):
    campaign["interval_seconds"] = 1800
    enable_completion(campaign)
    append_event(campaign, "task_complete")
    result, client = run_tick(campaign)
    assert result["delivery"] == "accepted"
    assert "每30分钟查询一次" in client.messages[0]
    assert "在该时点设置提醒" in client.messages[0]
    state = watchdog.load_state(Path(campaign["root"]))
    assert state["next_expected_check_at"] == "2026-11-08T15:30:00Z"
    result, client = run_tick(campaign, client)
    assert result["reason"] == "already_attempted"
    assert len(client.sent) == 1


def test_failed_turn_continues_once_with_durable_claim(campaign):
    result, client = run_tick(campaign)
    assert result["delivery"] == "accepted"
    assert client.sent == [(OWNER, "failed-turn")]
    result, client = run_tick(campaign, client)
    assert result["reason"] == "already_attempted"
    assert len(client.sent) == 1
    state = watchdog.load_state(Path(campaign["root"]))
    assert state["attempts"]["failed-turn"]["new_turn_id"] == "new-turn"
    assert state["next_expected_check_at"] == "2026-11-08T15:05:00Z"


@pytest.mark.parametrize("kind,fields,reason", [
    ("task_started", {}, "turn_active"),
    ("task_complete", {}, "turn_completed"),
    ("turn_aborted", {}, "turn_aborted"),
    ("user_message", {"message": "continue"}, "user_message_pending"),
])
def test_non_errors_do_not_send(campaign, kind, fields, reason):
    append_event(campaign, kind, **fields)
    result, client = run_tick(campaign)
    assert result["reason"] == reason
    assert not client.sent


def test_tool_errors_and_late_old_completions_do_not_interrupt(campaign):
    append_event(campaign, "task_started", "current-turn")
    append_event(campaign, "task_complete", "failed-turn", error="late old failure")
    append_event(campaign, "error", None, message="tool command failed")
    result, client = run_tick(campaign)
    assert result["reason"] == "turn_active"
    assert result["turn_id"] == "current-turn"
    assert not client.sent


def test_error_before_arming_is_not_replayed(campaign):
    campaign["armed_at"] = "2026-11-08T14:59:00Z"
    result, client = run_tick(campaign)
    assert result["reason"] == "historical_error"
    assert not client.sent


def test_error_grace_period(campaign):
    latest = watchdog.read_rollout(campaign)
    assert watchdog.eligible(latest, campaign, NOW - datetime.timedelta(minutes=4, seconds=50)) == "error_settling"


def test_partial_rollout_fails_closed(campaign):
    with Path(campaign["rollout"]).open("a") as stream:
        stream.write('{"type": "event_msg"')
    result, client = run_tick(campaign)
    assert result["reason"] == "rollout_write_in_progress"
    assert not client.sent


@pytest.mark.parametrize("field,value", [("thread_id", "wrong-thread"), ("workspace", "/other")])
def test_rollout_identity_is_pinned(campaign, field, value):
    campaign[field] = value
    result, client = run_tick(campaign)
    assert result["action"] == "blocked"
    assert not client.sent


@pytest.mark.parametrize("attribute,value", [("runtime", "active"), ("runtime", "unknown"),
                                           ("requests", [{"method": "approval"}]), ("cwd", "/other")])
def test_owner_must_be_idle_in_same_workspace(campaign, attribute, value):
    client = FakeClient(campaign)
    setattr(client, attribute, value)
    result, client = run_tick(campaign, client)
    assert result["action"] == "skip"
    assert not client.sent


def test_system_error_owner_continues_failed_turn_once(campaign):
    client = FakeClient(campaign)
    client.runtime = "systemError"
    result, client = run_tick(campaign, client)
    assert result["action"] == "continue"
    assert result["delivery"] == "accepted"
    assert client.sent == [(OWNER, "failed-turn")]
    assert run_tick(campaign, client)[0]["reason"] == "already_attempted"
    assert len(client.sent) == 1


@pytest.mark.parametrize("requests", [None, [{"method": "approval"}]])
def test_system_error_does_not_bypass_pending_requests(campaign, requests):
    client = FakeClient(campaign)
    client.runtime = "systemError"
    client.requests = requests
    result, client = run_tick(campaign, client)
    assert result["action"] == "skip"
    assert not client.sent


def test_system_error_owner_requires_failed_rollout(campaign):
    enable_completion(campaign)
    append_event(campaign, "task_complete")
    client = FakeClient(campaign)
    client.runtime = "systemError"
    result, client = run_tick(campaign, client)
    assert result["reason"] == "owner_not_idle"
    assert not client.sent


@pytest.mark.parametrize("runtime", ["idle", "systemError"])
@pytest.mark.parametrize("race", ["turn", "stop"])
def test_final_recheck_catches_user_turn_or_stop(campaign, race, runtime):
    client = FakeClient(campaign)
    client.runtime = runtime
    client.before_snapshot = (lambda: append_event(campaign, "task_started", "manual-turn")) if race == "turn" else (
        lambda: (Path(campaign["root"]) / "STOP").touch())
    result, client = run_tick(campaign, client)
    assert result["reason"] == "state_changed_before_send"
    assert not client.sent


def test_stop_prevents_all_actions(campaign):
    (Path(campaign["root"]) / "STOP").touch()
    result, client = run_tick(campaign)
    assert result["action"] == "stopped"
    assert not client.sent


@pytest.mark.parametrize("response", [TimeoutError(), ConnectionError(),
                                     {"resultType": "error", "error": "client-disconnected"},
                                     {"resultType": "success", "result": {}}])
def test_ambiguous_delivery_never_blindly_retries(campaign, response):
    client = FakeClient(campaign)
    client.response = response
    result, client = run_tick(campaign, client)
    assert result["delivery"] == "uncertain"
    result, client = run_tick(campaign, client)
    assert result["reason"] == "already_attempted"
    assert len(client.sent) == 1


def test_definite_non_delivery_can_retry(campaign):
    client = FakeClient(campaign)
    client.response = {"resultType": "error", "error": "no-client-found"}
    assert run_tick(campaign, client)[0]["delivery"] == "not_delivered"
    run_tick(campaign, client)
    assert len(client.sent) == 2


def test_new_failed_turn_can_always_continue(campaign):
    result, client = run_tick(campaign)
    append_event(campaign, "task_started", "next-failed-turn")
    append_event(campaign, "task_complete", "next-failed-turn", error="stream failed again")
    result, client = run_tick(campaign, client)
    assert result["delivery"] == "accepted"
    assert len(client.sent) == 2


def test_crash_after_claim_does_not_duplicate(campaign):
    watchdog.save_json(Path(campaign["root"]) / "state.json", {
        "attempts": {"failed-turn": {"status": "sending"}}})
    result, client = run_tick(campaign)
    assert result["reason"] == "already_attempted"
    assert not client.sent


def test_uncertain_delivery_reconciles_with_rollout_marker(campaign):
    client = FakeClient(campaign)
    client.response = TimeoutError()
    run_tick(campaign, client)
    append_event(campaign, "user_message", None, message="继续 [session-watchdog:failed-turn]")
    run_tick(campaign, client)
    state = watchdog.load_state(Path(campaign["root"]))
    assert state["attempts"]["failed-turn"]["status"] == "observed_in_rollout"
    assert len(client.sent) == 1


def test_single_instance_lock(campaign):
    with (Path(campaign["root"]) / "tick.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result, client = run_tick(campaign)
    assert result["reason"] == "another_check_running"
    assert not client.sent


def test_version_drift_blocks_dispatch(campaign, monkeypatch):
    def drift(config):
        raise ValueError("extension protocol changed")
    monkeypatch.setattr(watchdog, "verify_extension", drift)
    result, client = run_tick(campaign)
    assert result["action"] == "blocked"
    assert not client.sent


def test_config_copies_share_runtime_state_and_lock(campaign, tmp_path):
    campaign["runtime_root"] = str(tmp_path / "runtime")
    original = tmp_path / "config.json"
    copied = tmp_path / "backup.json"
    original.write_text(json.dumps(campaign))
    copied.write_text(original.read_text())
    assert watchdog.load_config(original)["root"] == watchdog.load_config(copied)["root"]
    assert watchdog.load_config(copied)["root"] == campaign["runtime_root"]


@pytest.mark.parametrize("normal_completion", [False, True])
def test_real_framing_and_owner_forwarding_with_mock_ipc(campaign, normal_completion):
    if normal_completion:
        enable_completion(campaign)
        append_event(campaign, "task_complete")
    received = []
    failures = []

    def read_exact(connection, count):
        data = bytearray()
        while len(data) < count:
            chunk = connection.recv(count - len(data))
            if not chunk:
                raise EOFError()
            data.extend(chunk)
        return data

    def serve(listener):
        try:
            with listener.accept()[0] as connection:
                connection.settimeout(5)
                while True:
                    length = struct.unpack("<I", read_exact(connection, 4))[0]
                    request = json.loads(read_exact(connection, length))
                    received.append(request)
                    method = request["method"]
                    if method == "initialize":
                        response = {"type": "response", "requestId": request["requestId"],
                                    "resultType": "success", "result": {"clientId": "mock-client"}}
                    elif method == "thread-stream-following-changed":
                        if not request["params"]["following"]:
                            break
                        response = {"type": "broadcast", "method": "thread-stream-state-changed",
                                    "version": 11, "sourceClientId": OWNER, "params": {
                                        "conversationId": campaign["thread_id"], "hostId": "local",
                                        "change": {"type": "snapshot", "conversationState":
                                                   FakeClient(campaign).snapshot()["state"]}}}
                    else:
                        assert method == "thread-follower-start-turn"
                        assert request["targetClientId"] == OWNER
                        assert set(request["params"]["turnStartParams"]) == {"input", "clientUserMessageId"}
                        response = {"type": "response", "requestId": request["requestId"],
                                    **FakeClient(campaign).response}
                    data = json.dumps(response).encode()
                    framed = struct.pack("<I", len(data)) + data
                    connection.sendall(framed[:3])
                    connection.sendall(framed[3:])
        except BaseException as error:
            failures.append(error)

    with tempfile.TemporaryDirectory(prefix="watchdog-ipc-", dir="/tmp") as directory:
        campaign["socket"] = str(Path(directory) / "ipc.sock")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(campaign["socket"])
            os.chmod(campaign["socket"], 0o600)
            listener.listen(1)
            thread = threading.Thread(target=serve, args=(listener,), daemon=True)
            thread.start()
            result = watchdog.tick(campaign, now=NOW)
            thread.join(timeout=6)
    assert not thread.is_alive()
    assert not failures
    assert result["delivery"] == "accepted"
    sent = [request for request in received if request["method"] == "thread-follower-start-turn"]
    assert len(sent) == 1
    assert "[session-watchdog:failed-turn]" in sent[0]["params"]["turnStartParams"]["input"][0]["text"]


def test_oversized_ipc_frame_rejected(campaign):
    client_socket, server_socket = socket.socketpair()
    with client_socket, server_socket:
        client = watchdog.IpcFollower(campaign)
        client.connection = client_socket
        server_socket.sendall(struct.pack("<I", watchdog.MAX_FRAME_BYTES + 1))
        with pytest.raises(ValueError, match="frame size"):
            client.receive(watchdog.time.monotonic() + 1)


def enable_completion(config):
    config.update(continue_on_completion=True, completion_armed_at=config["armed_at"],
                  campaign_file=str(Path(config["root"]) / "campaign.json"))
    context = {"thread_id": config["thread_id"], "workspace": config["workspace"], "status": "active",
               "updated_at": "2026-11-08T14:54:00Z", "campaign_id": "tp2-then-mixed", "stage": "tp2_calibration",
               "objective": "EP4TP2消融，验证后进行EP4TP1+EP4TP2", "jobs": [{"job": "123", "state": "RUNNING"}],
               "findings": ["72/72 calibration passed"], "next_actions": ["check job 123 then inspect dataplane"],
               "evidence": ["evidence/job-123"], "constraints": ["single allocation; no PAUSE deletion"],
               "completion_gate": [{"name": "tp2_ablation", "passed": False, "evidence": []}]}
    watchdog.save_json(Path(config["campaign_file"]), context)
    return context


def test_normal_completion_sends_context_once(campaign):
    enable_completion(campaign)
    append_event(campaign, "task_complete", last_agent_message="Submitted job 123; calibration pending")
    result, client = run_tick(campaign)
    assert result["delivery"] == "accepted"
    message = client.messages[0]
    for text in ("正常完成", "Submitted job 123", "tp2_calibration", "123", "72/72", "check job 123", "EP4TP1"):
        assert text in message
    result, client = run_tick(campaign, client)
    assert result["reason"] == "already_attempted"
    assert len(client.sent) == 1
    attempt = watchdog.load_state(Path(campaign["root"]))["attempts"]["failed-turn"]
    assert attempt["trigger"] == "completed"
    assert attempt["campaign_sha256"] and attempt["message_sha256"]


def test_error_also_includes_experiment_context(campaign):
    enable_completion(campaign)
    result, client = run_tick(campaign)
    assert result["delivery"] == "accepted"
    assert "异常结束" in client.messages[0]
    assert "stream disconnected" in client.messages[0]
    assert "tp2_calibration" in client.messages[0]


@pytest.mark.parametrize("status", ["paused", "blocked", "complete"])
def test_inactive_campaign_does_not_continue(campaign, status):
    context = enable_completion(campaign)
    context["status"] = status
    context["completion_gate"] = [{"name": "tp2_ablation", "passed": True, "evidence": ["accepted.json"]}]
    watchdog.save_json(Path(campaign["campaign_file"]), context)
    append_event(campaign, "task_complete")
    result, client = run_tick(campaign)
    assert result["reason"] == "campaign_" + status
    assert not client.sent


def test_completion_requires_explicit_context(campaign):
    campaign.update(continue_on_completion=True, completion_armed_at=campaign["armed_at"])
    append_event(campaign, "task_complete")
    result, client = run_tick(campaign)
    assert result["action"] == "blocked"
    assert not client.sent


def test_old_completion_does_not_trigger_new_campaign(campaign):
    enable_completion(campaign)
    campaign["completion_armed_at"] = "2026-11-08T14:59:00Z"
    append_event(campaign, "task_complete")
    result, client = run_tick(campaign)
    assert result["reason"] == "historical_completion"
    assert not client.sent


def test_campaign_pause_race_prevents_send(campaign):
    context = enable_completion(campaign)
    client = FakeClient(campaign)
    def pause():
        context["status"] = "paused"
        watchdog.save_json(Path(campaign["campaign_file"]), context)
    client.before_snapshot = pause
    result, client = run_tick(campaign, client)
    assert result["reason"] == "state_changed_before_send"
    assert not client.sent


def test_cannot_declare_complete_without_gate_evidence(campaign):
    context = enable_completion(campaign)
    context["status"] = "complete"
    with pytest.raises(ValueError, match="all evidence gates"):
        watchdog.validate_campaign(context, campaign)
    context["completion_gate"][0]["passed"] = True
    with pytest.raises(ValueError, match="requires evidence"):
        watchdog.validate_campaign(context, campaign)


def test_publish_rejects_stale_or_wrong_thread_context(campaign):
    context = enable_completion(campaign)
    source = Path(campaign["root"]) / "checkpoint.json"
    context["updated_at"] = "2026-11-08T14:53:00Z"
    watchdog.save_json(source, context)
    with pytest.raises(ValueError, match="stale"):
        watchdog.publish_campaign(campaign, source)
    context["updated_at"] = "2026-11-08T14:56:00Z"
    context["thread_id"] = "other-thread"
    watchdog.save_json(source, context)
    with pytest.raises(ValueError, match="identity"):
        watchdog.publish_campaign(campaign, source)
    context["thread_id"] = campaign["thread_id"]
    context["jobs"][0]["state"] = "COMPLETED"
    watchdog.save_json(source, context)
    result = watchdog.publish_campaign(campaign, source)
    assert result["status"] == "active"
    assert watchdog.load_campaign(campaign)["jobs"][0]["state"] == "COMPLETED"
