import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import struct
import time
import uuid


INTERVAL_SECONDS = 300
MAX_FRAME_BYTES = 45 * 1024 * 1024
MARKER_PATTERN = re.compile(r"\[session-watchdog:([^\]\s]+)\]")


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def timestamp(value):
    return value.isoformat().replace("+00:00", "Z")


def parse_timestamp(value):
    parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return parsed


def save_json(path, value):
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x") as stream:
            os.chmod(temporary, 0o600)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def load_config(path):
    config = json.loads(path.read_text())
    config["root"] = str(Path(config.get("runtime_root", path.resolve().parent)).expanduser().resolve())
    if type(config["interval_seconds"]) is not int or config["interval_seconds"] not in (300, 1800):
        raise ValueError("watchdog interval must be 300 or 1800 seconds")
    uuid.UUID(config["thread_id"])
    parse_timestamp(config["armed_at"])
    return config


def read_rollout(config):
    latest = {"status": "unknown", "turn_id": None, "user_pending": False}
    identity = None
    markers = set()
    incomplete = False
    with Path(config["rollout"]).open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                incomplete = True
                break
            event = json.loads(line)
            payload = event.get("payload", {})
            if event.get("type") == "session_meta":
                identity = payload
            if event.get("type") != "event_msg":
                continue
            kind = payload.get("type")
            turn_id = payload.get("turn_id")
            if kind == "task_started":
                latest = {"status": "active", "turn_id": turn_id,
                          "timestamp": event.get("timestamp"), "user_pending": False}
            elif kind in ("task_complete", "turn_aborted"):
                if not turn_id or turn_id != latest["turn_id"]:
                    continue
                latest.update(status="aborted" if kind == "turn_aborted" else
                              "failed" if payload.get("error") else "completed",
                              timestamp=event.get("timestamp"), user_pending=False,
                              last_agent_message=str(payload.get("last_agent_message") or "")[:2400])
                error = payload.get("error")
                latest["error_summary"] = str(error.get("message", "") if isinstance(error, dict) else error or "")[:500]
            elif kind == "user_message":
                message = payload.get("message", "")
                if isinstance(message, str):
                    markers.update(MARKER_PATTERN.findall(message))
                if latest["status"] != "active":
                    latest["user_pending"] = True
    if not identity or identity.get("id") != config["thread_id"]:
        raise ValueError("rollout thread identity mismatch")
    if Path(identity.get("cwd", "")).resolve() != Path(config["workspace"]).resolve():
        raise ValueError("rollout workspace mismatch")
    latest.update(incomplete=incomplete, markers=sorted(markers))
    return latest


def eligible(latest, config, now):
    if latest["incomplete"]:
        return "rollout_write_in_progress"
    completion = latest["status"] == "completed" and config.get("continue_on_completion", False)
    if latest["status"] != "failed" and not completion:
        return "turn_" + latest["status"]
    if latest["user_pending"]:
        return "user_message_pending"
    ended_at = parse_timestamp(latest["timestamp"])
    armed_at = config["completion_armed_at"] if completion else config["armed_at"]
    if ended_at < parse_timestamp(armed_at):
        return "historical_completion" if completion else "historical_error"
    if (now - ended_at).total_seconds() < config.get("error_grace_seconds", 30):
        return "completion_settling" if completion else "error_settling"
    return None


def validate_campaign(campaign, config):
    if campaign.get("thread_id") != config["thread_id"]:
        raise ValueError("campaign thread identity mismatch")
    if Path(campaign.get("workspace", "")).resolve() != Path(config["workspace"]).resolve():
        raise ValueError("campaign workspace mismatch")
    if campaign.get("status") not in ("active", "paused", "complete", "blocked"):
        raise ValueError("invalid campaign status")
    parse_timestamp(campaign["updated_at"])
    for key in ("campaign_id", "objective", "stage"):
        if not isinstance(campaign.get(key), str) or not campaign[key].strip():
            raise ValueError("missing campaign " + key)
    for key in ("jobs", "findings", "next_actions", "evidence", "constraints", "completion_gate"):
        if not isinstance(campaign.get(key), list):
            raise ValueError("missing campaign list " + key)
    if campaign["status"] == "active" and not campaign["next_actions"]:
        raise ValueError("active campaign needs concrete next actions")
    gates = campaign["completion_gate"]
    if not gates or any(not isinstance(gate, dict) or not isinstance(gate.get("passed"), bool)
                        or not gate.get("name") or not isinstance(gate.get("evidence"), list) for gate in gates):
        raise ValueError("campaign needs explicit evidence gates")
    if any(gate["passed"] and not gate["evidence"] for gate in gates):
        raise ValueError("passed gate requires evidence")
    if campaign["status"] == "complete" and not all(gate["passed"] for gate in gates):
        raise ValueError("cannot complete campaign before all evidence gates pass")
    if len(json.dumps(campaign, ensure_ascii=False)) > 16000:
        raise ValueError("campaign context exceeds message budget")
    return campaign


def load_campaign(config):
    if not config.get("campaign_file"):
        if config.get("continue_on_completion"):
            raise ValueError("normal completion requires explicit campaign context")
        return None
    return validate_campaign(json.loads(Path(config["campaign_file"]).read_text()), config)


def campaign_digest(campaign):
    return hashlib.sha256(json.dumps(campaign, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def continuation_message(config, latest, campaign):
    reason = {"failed": "异常结束", "completed": "正常完成"}.get(latest["status"], latest["status"])
    header = (f"继续当前实验任务（后台自动续接，不是新会话）。触发原因：上一轮{reason}。\n"
              f"检查时间：{timestamp(utc_now())}；上轮 turn：{latest['turn_id']}；"
              f"结束时间：{latest.get('timestamp', 'unknown')}。\n")
    previous = ""
    if latest.get("last_agent_message"):
        previous += "上轮会话结论（供核对，不代表整个实验目标完成）：\n" + latest["last_agent_message"] + "\n"
    if latest.get("error_summary"):
        previous += "上轮错误：" + latest["error_summary"] + "\n"
    context = ("当前实验检查点（状态是记录时的快照，先核实实时作业和文件，不能据此重复提交）：\n"
               + json.dumps(campaign, ensure_ascii=False, indent=2) + "\n") if campaign else ""
    instructions = (
        "先检查已有命令、squeue/sacct 与输出，不能重复占用 allocation。作业结束后验收并分析结果；"
        "失败须先定位原因、记录修复与差异，再提交新实验，不能原包盲重跑。"
        "不要只回复计划：执行当前可推进的下一步；有作业运行则检查/等待并更新检查点。"
        "EP4TP2 先完成实测代价接入与同条件消融，真实结果通过门槛后再做一个模型 EP4TP1、"
        "另一个 EP4TP2 的混合拓扑。保留所有失败和负结果，不通过筛选种子或不公平条件制造最佳性能。"
        "每轮结束前更新 campaign 检查点并用 session_watchdog.py publish 发布至活动目录；"
        "全部门槛完成后将 campaign.status 置为 complete，停止正常完成续接。"
        f"任务等待/运行中每{config['interval_seconds'] // 60}分钟查询一次；"
        "若仍未结束，记录下次检查时间并结束本轮，不再短间隔前台轮询。"
        "若有可信的预计结束时间，在该时点设置提醒；排队启动估计不等于结束时间。"
        "不恢复旧 observer/自动链，不删除 PAUSE，不运行分时算法。\n")
    marker = "[session-watchdog:" + latest["turn_id"] + "]"
    return header + previous + context + instructions + marker


def publish_campaign(config, source):
    campaign = validate_campaign(json.loads(source.read_text()), config)
    destination = Path(config["campaign_file"])
    if destination.exists():
        current = json.loads(destination.read_text())
        if parse_timestamp(campaign["updated_at"]) < parse_timestamp(current["updated_at"]):
            raise ValueError("refusing stale campaign checkpoint")
    save_json(destination, campaign)
    return {"published": str(destination), "stage": campaign["stage"], "status": campaign["status"],
            "sha256": campaign_digest(campaign)}


def verify_extension(config):
    extension = Path(config["extension_path"])
    package = json.loads((extension / "package.json").read_text())
    if package["version"] != config["extension_version"]:
        raise ValueError("unverified extension version")
    for filename, expected in config["extension_sha256"].items():
        actual = hashlib.sha256((extension / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("extension protocol source changed")


def snapshot_guard(snapshot, config, latest):
    state = snapshot["state"]
    if state.get("id") != config["thread_id"]:
        return "owner_thread_mismatch"
    if Path(state.get("cwd", "")).resolve() != Path(config["workspace"]).resolve():
        return "owner_workspace_mismatch"
    if state.get("resumeState") != "resumed":
        return "owner_not_resumed"
    runtime = state.get("threadRuntimeStatus", {}).get("type")
    recoverable_error = runtime == "systemError" and latest["status"] == "failed"
    if runtime != "idle" and not recoverable_error:
        return "owner_not_idle"
    if state.get("requests") is None or state["requests"]:
        return "owner_requests_pending_or_unknown"
    return None


def summarize_snapshot(snapshot):
    state = snapshot["state"]
    return {"owner": snapshot["owner"], "thread_id": state.get("id"),
            "cwd": state.get("cwd"), "runtime": state.get("threadRuntimeStatus"),
            "resume_state": state.get("resumeState"),
            "pending_requests": len(state.get("requests") or [])}


class IpcFollower:
    def __init__(self, config):
        self.config = config
        self.connection = None
        self.client_id = str(uuid.uuid4())
        self.following = False

    def __enter__(self):
        path = Path(self.config["socket"])
        metadata = path.lstat()
        parent = path.parent.stat()
        if (not stat.S_ISSOCK(metadata.st_mode) or metadata.st_uid != os.getuid()
                or metadata.st_mode & 0o077 or parent.st_uid != os.getuid()
                or parent.st_mode & 0o077):
            raise ValueError("IPC socket must be private and owned by current user")
        self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.connection.settimeout(10)
        try:
            self.connection.connect(str(path))
            response = self.request("initialize", {"clientType": "xplace-session-watchdog"}, version=0)
            if response.get("resultType") != "success":
                raise ValueError("IPC initialization rejected")
            self.client_id = response["result"]["clientId"]
            return self
        except BaseException:
            self.connection.close()
            raise

    def __exit__(self, *unused):
        if self.following:
            with contextlib.suppress(OSError):
                self.follow(False)
        self.connection.close()

    def send(self, message):
        data = json.dumps(message, ensure_ascii=False).encode()
        self.connection.sendall(struct.pack("<I", len(data)) + data)

    def receive(self, deadline):
        def read_exact(count):
            output = bytearray()
            while len(output) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("IPC response timeout")
                self.connection.settimeout(remaining)
                data = self.connection.recv(count - len(output))
                if not data:
                    raise ConnectionError("IPC disconnected")
                output.extend(data)
            return output

        length = struct.unpack("<I", read_exact(4))[0]
        if not 0 < length <= MAX_FRAME_BYTES:
            raise ValueError("invalid IPC frame size")
        return json.loads(read_exact(length))

    def request(self, method, params, version=1, owner=None):
        request_id = str(uuid.uuid4())
        message = {"type": "request", "requestId": request_id, "sourceClientId": self.client_id,
                   "method": method, "params": params, "version": version, "timeoutMs": 30000}
        if owner:
            message["targetClientId"] = owner
        self.send(message)
        deadline = time.monotonic() + 35
        while True:
            response = self.receive(deadline)
            if response.get("type") == "response" and response.get("requestId") == request_id:
                return response

    def follow(self, following):
        self.send({"type": "broadcast", "method": "thread-stream-following-changed", "version": 1,
                   "sourceClientId": self.client_id, "params": {
                       "conversationId": self.config["thread_id"], "hostId": "local", "following": following}})
        self.following = following

    def snapshot(self):
        self.follow(True)
        deadline = time.monotonic() + 10
        while True:
            message = self.receive(deadline)
            if message.get("method") != "thread-stream-state-changed":
                continue
            params = message.get("params", {})
            if params.get("conversationId") != self.config["thread_id"] or params.get("hostId") != "local":
                continue
            if message.get("version") != 11:
                raise ValueError("unverified IPC snapshot protocol")
            change = params.get("change", {})
            if change.get("type") != "snapshot":
                continue
            owner = message["sourceClientId"]
            uuid.UUID(owner)
            return {"owner": owner, "state": change["conversationState"]}

    def start_turn(self, owner, failed_turn, message=None):
        marker = "[session-watchdog:" + failed_turn + "]"
        message = message or ("继续。后台监控检测到本会话上一轮异常结束，请从现有记录和工具状态续接当前任务。"
                   "先确认已有命令或实验是否仍在运行；补跑实验前须分析原因并修复，不得原封不动重跑。"
                   "不恢复已停用的旧自动链，不删除 PAUSE 文件。\n" + marker)
        return self.request("thread-follower-start-turn", {
            "conversationId": self.config["thread_id"], "turnStartParams": {
                "input": [{"type": "text", "text": message}],
                "clientUserMessageId": str(uuid.uuid5(uuid.NAMESPACE_URL,
                    self.config["thread_id"] + ":session-watchdog:" + failed_turn))}}, owner=owner)


def load_state(root):
    path = root / "state.json"
    return json.loads(path.read_text()) if path.exists() else {"attempts": {}}


def record_check(root, state, result, now, interval_seconds=INTERVAL_SECONDS):
    state.update(last_check_at=timestamp(now), last_result=result,
                 next_expected_check_at=timestamp(now + datetime.timedelta(seconds=interval_seconds)))
    save_json(root / "state.json", state)
    descriptor = os.open(str(root / "checks.jsonl"), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "a") as stream:
        stream.write(json.dumps({"at": timestamp(now), **result}, ensure_ascii=False) + "\n")
    return result


def tick(config, client_factory=IpcFollower, now=None):
    now = now or utc_now()
    root = Path(config["root"])
    with (root / "tick.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"action": "skip", "reason": "another_check_running"}
        state = load_state(root)
        try:
            result = check_session(config, root, state, now, client_factory)
        except Exception as error:
            result = {"action": "blocked", "reason": type(error).__name__, "detail": str(error)[:160]}
        return record_check(root, state, result, now, config["interval_seconds"])


def check_session(config, root, state, now, client_factory, expected_campaign=None):
    if (root / "STOP").exists():
        return {"action": "stopped"}
    latest = read_rollout(config)
    state["last_turn"] = {key: value for key, value in latest.items() if key != "markers"}
    for failed_turn in latest["markers"]:
        attempt = state["attempts"].get(failed_turn)
        if attempt and attempt["status"] in ("sending", "uncertain"):
            attempt["status"] = "observed_in_rollout"
    reason = eligible(latest, config, now)
    if reason:
        return {"action": "skip", "reason": reason, "turn_id": latest["turn_id"]}
    campaign = load_campaign(config)
    if expected_campaign is not None and campaign != expected_campaign:
        return {"action": "skip", "reason": "reminder_campaign_changed", "turn_id": latest["turn_id"]}
    if campaign and campaign["status"] != "active":
        return {"action": "skip", "reason": "campaign_" + campaign["status"], "turn_id": latest["turn_id"]}
    failed_turn = latest["turn_id"]
    previous = state["attempts"].get(failed_turn)
    if previous and previous["status"] != "not_delivered":
        return {"action": "skip", "reason": "already_attempted", "turn_id": failed_turn,
                "delivery": previous["status"]}
    verify_extension(config)
    with client_factory(config) as client:
        snapshot = client.snapshot()
        reason = snapshot_guard(snapshot, config, latest)
        state["last_owner"] = summarize_snapshot(snapshot)
        if reason:
            return {"action": "skip", "reason": reason, "turn_id": failed_turn}
        current = read_rollout(config)
        if ((root / "STOP").exists() or current != latest or eligible(current, config, utc_now())
                or load_campaign(config) != campaign):
            return {"action": "skip", "reason": "state_changed_before_send", "turn_id": failed_turn}
        message = continuation_message(config, latest, campaign)
        state["attempts"][failed_turn] = {"status": "sending", "at": timestamp(now),
                                        "owner": snapshot["owner"],
                                        "trigger": latest["status"],
                                        "campaign_sha256": campaign_digest(campaign),
                                        "message_sha256": hashlib.sha256(message.encode()).hexdigest(),
                                        "try_count": (previous or {}).get("try_count", 0) + 1}
        save_json(root / "state.json", state)
        attempt = state["attempts"][failed_turn]
        try:
            response = client.start_turn(snapshot["owner"], failed_turn, message)
            if response.get("resultType") == "error" and response.get("error") == "no-client-found":
                attempt["status"] = "not_delivered"
            elif (response.get("resultType") == "success"
                  and response.get("handledByClientId") == snapshot["owner"]
                  and isinstance(response.get("result", {}).get("result", {}).get("turn"), dict)
                  and response["result"]["result"]["turn"].get("id")):
                attempt.update(status="accepted", new_turn_id=response["result"]["result"]["turn"]["id"])
            else:
                attempt["status"] = "uncertain"
        except Exception as error:
            attempt.update(status="uncertain", error_type=type(error).__name__)
        save_json(root / "state.json", state)
        return {"action": "continue", "turn_id": failed_turn, "delivery": attempt["status"]}


def probe(config):
    verify_extension(config)
    with IpcFollower(config) as client:
        snapshot = client.snapshot()
    latest = read_rollout(config)
    return {"read_only": True, "rollout": latest, "owner": summarize_snapshot(snapshot),
            "rollout_guard": eligible(latest, config, utc_now()),
            "owner_guard": snapshot_guard(snapshot, config, latest)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("probe", "tick", "status", "stop", "preview", "publish"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--context", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["root"])
    if args.command == "tick":
        result = tick(config)
    elif args.command == "probe":
        result = probe(config)
    elif args.command == "stop":
        (root / "STOP").touch()
        result = {"stopped": True, "root": str(root)}
    elif args.command == "publish":
        if args.context is None:
            parser.error("publish requires --context")
        result = publish_campaign(config, args.context)
    elif args.command == "preview":
        latest = read_rollout(config)
        result = {"read_only": True, "guard": eligible(latest, config, utc_now()),
                  "message": continuation_message(config, latest, load_campaign(config))}
    else:
        campaign = load_campaign(config)
        result = {"stopped": (root / "STOP").exists(), "interval_seconds": config["interval_seconds"],
                  "thread_id": config["thread_id"], "campaign": campaign, **load_state(root)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
