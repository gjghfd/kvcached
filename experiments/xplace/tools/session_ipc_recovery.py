import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import struct
import subprocess
import time

import session_watchdog as watchdog


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, value):
    with Path(path).open("x") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_runtime_paths(config, request):
    root = Path(config["root"]).resolve()
    paths = [request[name] for name in ("source_campaign", "output", "config")]
    paths.extend(binding["path"] for binding in request["bindings"])
    try:
        for path in paths:
            Path(path).resolve().relative_to(root)
    except ValueError as error:
        raise ValueError("recovery_paths_must_be_in_runtime_directory") from error


def guard(config, request):
    if request.get("expires_at") and watchdog.utc_now() > watchdog.parse_timestamp(request["expires_at"]):
        raise ValueError("recovery_request_expired")
    if (Path(config["root"]) / "STOP").exists():
        raise ValueError("watchdog_stopped")
    if digest(request["source_campaign"]) != request["campaign_sha256"]:
        raise ValueError("campaign_changed")
    campaign = watchdog.load_campaign(config)
    if campaign["status"] != "active":
        raise ValueError("campaign_not_active")
    if Path(config["campaign_file"]).read_bytes() != Path(request["source_campaign"]).read_bytes():
        raise ValueError("published_campaign_changed")
    latest = watchdog.read_rollout(config)
    if latest["turn_id"] != request["expected_turn_id"] or latest.get("user_pending"):
        raise ValueError("turn_changed_or_user_pending")
    if latest["status"] not in ("active", "completed"):
        raise ValueError("turn_not_normally_completed")
    return latest


def wait_for_completion(config, request):
    deadline = time.monotonic() + request["wait_seconds"]
    while True:
        latest = guard(config, request)
        if latest["status"] == "completed":
            return latest
        if time.monotonic() >= deadline:
            raise TimeoutError("current_turn_did_not_complete_no_reload")
        time.sleep(5)


def peer_pid(connection):
    return struct.unpack("i", connection.getsockopt(0, 2, 4))[0]


def reload_original_window(config, request, output):
    registration_path = Path(request["bridge_registration"])
    if digest(registration_path) != request["bridge_registration_sha256"]:
        raise ValueError("bridge_registration_changed")
    registration = json.loads(registration_path.read_text())
    if (registration["workspaceName"] != Path(config["workspace"]).name
            or registration["socketPath"] != request["bridge_socket"]
            or registration["marketplaceID"] != "openai.chatgpt"
            or registration["capabilities"].get("reload") != 1):
        raise ValueError("wrong_window_bridge")
    metadata = Path(request["bridge_socket"]).lstat()
    if (not stat.S_ISSOCK(metadata.st_mode) or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077 or metadata.st_ino != request["bridge_inode"]):
        raise ValueError("bridge_socket_changed_or_not_private")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(5)
        connection.connect(request["bridge_socket"])
        if peer_pid(connection) != request["bridge_peer_pid"]:
            raise ValueError("bridge_peer_changed")
        if guard(config, request)["status"] != "completed":
            raise ValueError("turn_active_before_reload")
        write_once(output / "reload-claim.json", {
            "at": watchdog.timestamp(watchdog.utc_now()),
            "peer_pid": request["bridge_peer_pid"],
            "expected_turn_id": request["expected_turn_id"],
            "action": "reload_original_window_once_no_retry",
        })
        data = json.dumps({"command": "reload", "payload": {}}).encode()
        connection.sendall(struct.pack("<I", len(data)) + data)


def verify_reconnected(config, request, client_factory=watchdog.IpcFollower):
    deadline = time.monotonic() + request["verify_seconds"]
    attempts = []
    while True:
        latest = guard(config, request)
        if latest["status"] != "completed":
            raise ValueError("turn_changed_during_reconnection")
        watchdog.verify_extension(config)
        try:
            with client_factory(config) as client:
                snapshot = client.snapshot()
            reason = watchdog.snapshot_guard(snapshot, config, latest)
            if reason:
                raise ValueError(reason)
            guard(config, request)
            return watchdog.summarize_snapshot(snapshot), attempts
        except (OSError, ConnectionError, ValueError) as error:
            attempts.append({"at": watchdog.timestamp(watchdog.utc_now()),
                             "error": type(error).__name__, "detail": str(error)[:160]})
        if time.monotonic() >= deadline:
            raise TimeoutError("reconnection_not_verified: " + json.dumps(attempts))
        time.sleep(10)


def publish_recovery(config, request, result_path):
    guard(config, request)
    campaign = json.loads(Path(request["source_campaign"]).read_text())
    campaign["updated_at"] = watchdog.timestamp(watchdog.utc_now())
    campaign["stage"] = "tp2_r7_capacity_rejected_ipc_restored"
    campaign["watchdog_health"] = {
        "path": str(result_path), "sha256": digest(result_path),
        "status": "same_thread_snapshot_restored_delivery_pending",
        "snapshot_recovery_verified": True,
        "deadline_repaired_source_not_deployed": True,
    }
    campaign["query_policy"]["automatic_continuation_available"] = True
    campaign["query_policy"]["next_expected_check_at"] = campaign["updated_at"]
    campaign["query_policy"]["reason"] = "原窗口重载后同会话快照验证通过；保持1800秒周期及原发送保护。"
    campaign["next_actions"][0] = (
        "先核验活动目录watchdog_health恢复回执及发送记录，将较新的活动检查点同步回仓库，"
        "不得用旧仓库副本覆盖。连接已恢复，不再重载；继续容量修复，禁止重复取回/盲重跑r7。")
    campaign["findings"][0] = (
        "原窗口在恢复轮正常结束后仅重载一次，同thread/workspace/resumed/idle/无待处理请求快照通过。"
        "不新建会话、不杀进程、不删除socket或PAUSE；下一消息仍由原watchdog完整保护决定。")
    watchdog.validate_campaign(campaign, config)
    watchdog.save_json(Path(request["source_campaign"]), campaign)
    receipt = watchdog.publish_campaign(config, Path(request["source_campaign"]))
    if Path(config["campaign_file"]).read_bytes() != Path(request["source_campaign"]).read_bytes():
        raise ValueError("publish_byte_mismatch")
    write_once(result_path.parent / "publish-after-recovery.json", receipt)


def run(request_path):
    request = json.loads(request_path.read_text())
    output = Path(request["output"])
    result_path = output / "result.json"
    config = watchdog.load_config(Path(request["config"]))
    validate_runtime_paths(config, request)
    with (output / "recovery.lock").open("a") as recovery_lock:
        fcntl.flock(recovery_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if result_path.exists() or (output / "reload-claim.json").exists():
            raise ValueError("recovery_already_attempted_no_retry")
        with (Path(config["root"]) / "tick.lock").open("a") as tick_lock:
            fcntl.flock(tick_lock.fileno(), fcntl.LOCK_EX)
            try:
                for binding in request["bindings"]:
                    if digest(binding["path"]) != binding["sha256"]:
                        raise ValueError("recovery_binding_changed")
                watchdog.verify_extension(config)
                write_once(output / "waiting.json", {
                    "at": watchdog.timestamp(watchdog.utc_now()),
                    "expected_turn_id": request["expected_turn_id"],
                    "status": "waiting_for_current_turn_normal_completion_no_reload_yet",
                })
                wait_for_completion(config, request)
                reload_original_window(config, request, output)
                time.sleep(3)
                snapshot, attempts = verify_reconnected(config, request)
                write_once(result_path, {
                    "at": watchdog.timestamp(watchdog.utc_now()),
                    "status": "same_thread_snapshot_restored",
                    "snapshot": snapshot, "verification_attempts": attempts,
                    "reloads": 1, "new_turn_requested": False,
                    "no_experiment_command": True,
                })
                publish_recovery(config, request, result_path)
            except Exception as error:
                failure_path = output / "failure.json"
                if not failure_path.exists():
                    write_once(failure_path, {
                        "at": watchdog.timestamp(watchdog.utc_now()),
                        "status": "recovery_not_completed_no_automatic_retry",
                        "error": type(error).__name__, "detail": str(error),
                    })
                raise
        command = ["/bin/launchctl", "kickstart",
                   "gui/" + str(os.getuid()) + "/local.paracloud.codex-session-watchdog"]
        completed = subprocess.run(command, capture_output=True, text=True)
        write_once(output / "watchdog-wake.json", {
            "at": watchdog.timestamp(watchdog.utc_now()), "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr,
            "delivery_still_subject_to_original_guards": True,
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    run(args.request)


if __name__ == "__main__":
    main()
