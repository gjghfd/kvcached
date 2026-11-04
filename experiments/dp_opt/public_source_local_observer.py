import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback


ROLES = ("recovery", "c0_cal", "s0_future", "c1_cal", "c1_eval")
DEFAULT_DEADLINE = "2026-11-10T00:00:00Z"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}
SCRIPTS = {"recovery": ("public_source_recovery.py",),
           "c0_cal": ("public_workload_campaign.py",),
           "s0_future": ("public_poisson_followup.py",),
           "c1_cal": ("public_workload_campaign.py",),
           "c1_eval": ("public_poisson_followup.py",)}


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def timestamp(value):
    result = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Require a timezone-qualified timestamp")
    return result.astimezone(datetime.timezone.utc)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def digest(path):
    before = path.stat()
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("File changed during SHA read")
    return hasher.hexdigest()


def contained(base, path):
    resolved = path.resolve()
    resolved.relative_to(base.resolve())
    return resolved


def read_json(path):
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("Require a JSON object")
        return value, None
    except FileNotFoundError:
        return {}, "not_yet_available"
    except (OSError, ValueError) as error:
        return {}, str(error)


def save(path, value):
    temporary = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    with temporary.open("x") as target:
        target.write(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def tail(path, maximum=2 * 1024 * 1024):
    with path.open("rb") as source:
        source.seek(0, os.SEEK_END)
        source.seek(max(0, source.tell() - maximum))
        return source.read().decode("utf-8", errors="replace")


def process_identity(pid):
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return {"pid": None, "alive": None, "status": "identity_not_yet_available"}
    try:
        result = subprocess.run(["ps", "-p", str(pid), "-o", "pid=", "-o", "ppid=", "-o", "uid=",
                                 "-o", "lstart=", "-o", "stat=", "-o", "command="],
                                capture_output=True, text=True, timeout=10)
        fields = result.stdout.strip().split(None, 9)
        if not result.stdout.strip() and result.returncode == 1:
            return {"pid": pid, "alive": False}
        if result.returncode or len(fields) != 10 or int(fields[0]) != pid:
            raise ValueError("Unrecognized local ps result")
        command = fields[9]
        value = {"pid": pid, "ppid": int(fields[1]), "uid": int(fields[2]), "started_at_local_ps": " ".join(fields[3:8]),
                 "stat": fields[8], "alive": not fields[8].startswith("Z"),
                 "command_sha256": hashlib.sha256(command.encode()).hexdigest(), "command": command}
        value["process_identity_sha256"] = fingerprint({key: value[key] for key in
                                                        ("pid", "uid", "started_at_local_ps", "command_sha256")})
        return value
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {"pid": pid, "alive": None, "inspection_error": str(error)}


def latest_poll(root, job):
    if isinstance(job, bool) or not isinstance(job, int) or job <= 0:
        return {"status": "no_current_job"}
    path = contained(root, root / f"poll-{job}.log")
    try:
        text = tail(path)
    except FileNotFoundError:
        return {"status": "not_yet_available"}
    headers = list(re.finditer(r"^=== (\S+) (.*?) ===$", text, re.MULTILINE))
    if not headers:
        return {"status": "no_complete_header", "path": str(path)}
    header = headers[-1]
    attributes = dict(re.findall(r"(\w+)=(\S+)", header[2]))
    value = {"status": "observed_local_poll", "path": str(path), "at": header[1], "attributes": attributes}
    timestamp(value["at"])
    section = text[header.end():]
    for line in section.splitlines():
        if line.startswith(str(job) + "|"):
            fields = line.split("|")
            value.update(job_state=fields[1], uninterpreted_third_field=fields[2] if len(fields) > 2 else None)
    for match in re.finditer(r"^\s*\{", section, re.MULTILINE):
        try:
            payload, _ = json.JSONDecoder().raw_decode(section[match.start():].lstrip())
            if isinstance(payload, dict):
                if "phase" in payload:
                    value["phase"] = payload["phase"]
                if "completed_cases" in payload or "completed_requests" in payload:
                    value["progress"] = payload
        except ValueError:
            pass
    return value


def check_hashes(root, repo, state, baseline, descriptor_sha=None, controller_binding=None):
    paths = {str(path.relative_to(root)): (contained(root, path), None)
             for path in (root / "frozen").rglob("*") if path.is_file()}
    runtime = state.get("frozen_runtime_sha256", {})
    if not isinstance(runtime, dict):
        raise ValueError("Invalid frozen runtime manifest")
    if runtime and Path(state.get("repo", "")).resolve() != repo:
        raise ValueError("Campaign repo differs from configured read-only repo")
    for name, expected in runtime.items():
        if Path(name).is_absolute():
            raise ValueError("Absolute runtime member")
        paths["repo/" + name] = (contained(repo, repo / name), expected)
    if controller_binding:
        controller_path, expected = controller_binding
        paths["controller/public_source_recovery.py"] = (contained(repo, controller_path), expected)
    descriptor = root / "quality-followup.json"
    if descriptor.exists() or descriptor_sha is not None:
        paths[descriptor.name] = (contained(root, descriptor), descriptor_sha)
    prepared_path = root / "future-quality-prepared-v2.json"
    if prepared_path.exists():
        paths[prepared_path.name] = (contained(root, prepared_path), None)
        prepared, error = read_json(contained(root, prepared_path))
        if error:
            raise ValueError("Unreadable prepared-v2 identity: " + error)
        for name, expected in prepared.get("source_sha256", {}).items():
            if Path(name).is_absolute():
                raise ValueError("Absolute prepared-v2 source member")
            paths["prepared-v2/repo/" + name] = (contained(repo, repo / name), expected)
    for path in (root / "frozen").rglob("input-manifest.json"):
        manifest, error = read_json(contained(root, path))
        if error:
            raise ValueError("Unreadable frozen input manifest: " + error)
        for name, expected in manifest.get("files_sha256", {}).items():
            target = contained(path.parent, path.parent / name)
            paths[str(target.relative_to(root))] = (target, expected)
    alerts, observed = [], {}
    expected_runtime_sha = fingerprint(runtime) if runtime else None
    if baseline.get("runtime_manifest") not in (None, expected_runtime_sha):
        alerts.append({"kind": "frozen_runtime_identity_changed"})
    if runtime:
        baseline.setdefault("runtime_manifest", expected_runtime_sha)
    known = baseline.setdefault("files", {})
    for name, (path, expected) in paths.items():
        try:
            actual = digest(path)
            observed[name] = actual
            if expected is not None and actual != expected:
                alerts.append({"kind": "declared_frozen_SHA_mismatch", "path": str(path), "expected": expected, "actual": actual})
            if name in known and known[name] != actual:
                alerts.append({"kind": "observed_frozen_SHA_changed", "path": str(path), "expected": known[name], "actual": actual})
            known.setdefault(name, actual)
        except (OSError, ValueError) as error:
            alerts.append({"kind": "frozen_SHA_read_error", "path": str(path), "error": str(error)})
    for name in known.keys() - paths.keys():
        alerts.append({"kind": "frozen_file_disappeared", "path": name})
    return {"checked_files": len(observed), "runtime_manifest_sha256": expected_runtime_sha,
            "actual_files_sha256": observed, "baseline_scope": "first_observed_local_bytes_not_upstream_revision"}, alerts


def snapshot_root(role, root, config, baseline, now):
    alerts = []
    value = {"root": str(root), "role": role, "alerts": alerts, "notices": []}
    prepared_path = contained(config["roots"]["recovery"], config["roots"]["recovery"] / "future-quality-prepared-v2.json")
    prepared, prepared_error = read_json(prepared_path)
    if prepared_error not in (None, "not_yet_available"):
        alerts.append({"kind": "prepared_v2_read_error", "error": prepared_error})
    state, state_error = read_json(contained(root, root / "campaign-state.json"))
    identity, identity_error = read_json(contained(root, root / "local-monitor-identity.json"))
    phase, job_state = str(state.get("phase", "")), str(state.get("state_last_observed", ""))
    terminal = job_state in TERMINAL
    paused = (root / "PAUSE").exists()
    active = bool(state) and not terminal and not paused and bool(state.get("job") or "running" in phase or "submitted" in phase)
    value.update(phase=phase or None, job=state.get("job"), job_state=job_state or None, active=active,
                 status="observed" if state else "waiting_not_success", paused=paused,
                 campaign_read_error=state_error, identity_read_error=identity_error)
    if "diagnosis" in phase.lower() or job_state in TERMINAL - {"COMPLETED"}:
        alerts.append({"kind": "local_campaign_failure", "phase": phase, "job_state": job_state})
    for field in ("producer_exit_code", "controller_exit_code", "child_exit_code"):
        if state.get(field) not in (None, 0, "0"):
            alerts.append({"kind": "declared_controller_or_producer_failure", "field": field, "raw": state[field]})
    if state_error not in (None, "not_yet_available"):
        alerts.append({"kind": "campaign_read_error", "error": state_error})
    for field in ("recorded_at", "last_poll_at"):
        if state.get(field):
            value[field + "_age_s"] = (now - timestamp(state[field])).total_seconds()
    proc = process_identity(identity.get("pid"))
    command = proc.pop("command", "")
    proc["command_identity_matches"] = str(root) in command and any(script in command for script in SCRIPTS[role]) if command else None
    proc["declared_parent_pid"] = identity.get("parent_pid")
    proc["declared_cwd"] = str(config["repo"])
    value["main"] = proc
    previous = baseline.get("process_identity_sha256")
    if active and (proc.get("alive") is not True or proc["command_identity_matches"] is not True
                   or previous is not None and previous != proc.get("process_identity_sha256")):
        alerts.append({"kind": "active_local_process_dead_missing_or_changed", "evidence": proc})
    if proc.get("alive") and previous is None:
        baseline["process_identity_sha256"] = proc["process_identity_sha256"]
    if identity.get("caffeinate_pid"):
        caffeinate = process_identity(identity["caffeinate_pid"])
        command = caffeinate.pop("command", "")
        caffeinate["command_identity_matches"] = "caffeinate" in command and str(identity.get("pid")) in command if command else None
        value["caffeinate"] = caffeinate
        previous_caffeinate = baseline.get("caffeinate_identity_sha256")
        if active and (caffeinate.get("alive") is not True or caffeinate["command_identity_matches"] is not True
                       or previous_caffeinate is not None and previous_caffeinate != caffeinate.get("process_identity_sha256")):
            alerts.append({"kind": "active_caffeinate_dead_or_changed", "evidence": caffeinate})
        if caffeinate.get("alive") and previous_caffeinate is None:
            baseline["caffeinate_identity_sha256"] = caffeinate["process_identity_sha256"]
    poll = latest_poll(root, state.get("job"))
    value["latest_poll"] = poll
    value["progress_semantics"] = "missing_progress_is_waiting_not_completion_or_GPU_stall"
    dates = [timestamp(raw) for raw in (state.get("last_poll_at"), poll.get("at")) if raw]
    if dates:
        age = (now - max(dates)).total_seconds()
        value["local_poll_freshness_s"] = age
        if active and (age > config["freshness_s"] or age < -30):
            alerts.append({"kind": "active_poll_stale_or_future_timestamp", "age_s": age})
    elif active and state.get("job"):
        alerts.append({"kind": "active_job_missing_local_poll"})
    value["campaign_poll_gaps_s"] = {field: state.get(field) for field in
                                     ("last_poll_actual_gap_s", "last_poll_monotonic_gap_s", "last_poll_wall_clock_gap_s")}
    for field, gap in value["campaign_poll_gaps_s"].items():
        if active and isinstance(gap, (float, int)) and gap > config["freshness_s"]:
            alerts.append({"kind": "measured_campaign_poll_gap", "field": field, "gap_s": gap})
    exits = [state.get("last_poll_ssh_exit"), poll.get("attributes", {}).get("ssh_exit")]
    if active and any(exit_code not in (None, 0, "0") for exit_code in exits):
        alerts.append({"kind": "observed_producer_query_error", "exits": exits})
    value["local_exit_files"] = {}
    for path in root.glob("*.exit"):
        raw = contained(root, path).read_text().strip()
        reference = prepared.get("first_isolated_direct_module_failure_retained")
        retained = (isinstance(reference, str) and path.name == reference.split(".log", 1)[0] + ".exit"
                    and raw == "1" and prepared.get("final_isolated_regression", {}).get("exit") == 0)
        evidence = {"raw": raw, "sha256": digest(path), "classification": "successful_local_exit_evidence" if raw == "0"
                    else "retained_historical_exit_evidence" if retained else "retained_unattributed_exit_evidence",
                    "not_controller_or_GPU_failure_without_campaign_evidence": True}
        if retained:
            evidence["prepared_v2_binding"] = {"file": str(prepared_path), "field": "first_isolated_direct_module_failure_retained", "value": reference}
        value["local_exit_files"][path.name] = evidence
        if raw != "0":
            value["notices"].append({"kind": evidence["classification"], "file": str(path), "raw": raw})
    descriptor_sha = prepared.get("descriptor_sha256") if role == "s0_future" else None
    value["descriptor_declared_sha256"] = descriptor_sha
    controller_binding = (config["repo"] / "experiments/dp_opt/public_source_recovery.py", state["controller_sha256"]) if role == "recovery" and state.get("controller_sha256") else None
    value["frozen_SHA"], sha_alerts = check_hashes(root, config["repo"], state, baseline, descriptor_sha, controller_binding)
    alerts.extend(sha_alerts)
    return value


def configuration(arguments):
    roots = {role: getattr(arguments, role + "_root").resolve() for role in ROLES}
    if len(set(roots.values())) != 5:
        raise ValueError("Require five distinct roots")
    output = arguments.output.resolve() if arguments.output else roots["recovery"] / "local-observer-v2"
    output.relative_to(roots["recovery"] / "local-observer-v2")
    if any(output == root or root in output.parents and role != "recovery" for role, root in roots.items()):
        raise ValueError("Observer output overlaps a monitored campaign")
    if any(not math.isfinite(value) or value <= 0 for value in (arguments.interval, arguments.freshness)):
        raise ValueError("Intervals must be finite and positive")
    if arguments.freshness < arguments.interval:
        raise ValueError("Freshness bound must cover the interval")
    return {"roots": roots, "repo": arguments.repo.resolve(), "output": output,
            "interval_s": arguments.interval, "freshness_s": arguments.freshness,
            "deadline_utc": timestamp(arguments.deadline_utc).isoformat()}


def public_config(config):
    return {key: {role: str(path) for role, path in value.items()} if key == "roots"
            else str(value) if isinstance(value, Path) else value for key, value in config.items()}


def predecessor_guard(config):
    previous, error = read_json(config["roots"]["recovery"] / "subagent-monitor-identity.json")
    if error not in (None, "not_yet_available"):
        raise RuntimeError("Reconcile unreadable predecessor identity before handoff")
    if previous:
        inspected = process_identity(previous.get("pid"))
        if inspected.get("alive") is not False:
            raise RuntimeError("Predecessor PID still alive or unverified; no automatic stop or duplicate observer")


def prepare_output(config):
    output = config["output"]
    output.mkdir(parents=True, exist_ok=True)
    for name in ("observer.lock", "launch.lock", "observer.jsonl", "launcher.log",
                 "observer-baseline.json", "observer-identity.json"):
        if (output / name).is_symlink():
            raise ValueError("Never follow an observer output file symlink")
    return output


def run(config, once=False):
    if timestamp(config["deadline_utc"]) <= utc_now():
        raise ValueError("Deadline is already in the past")
    predecessor_guard(config)
    output = prepare_output(config)
    stopped = threading.Event()
    with (output / "observer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_sha = fingerprint(public_config(config))
        baseline, error = read_json(output / "observer-baseline.json")
        if error not in (None, "not_yet_available") or baseline and baseline.get("config_sha256") != config_sha:
            raise ValueError("Preserve existing observer baseline/config identity")
        baseline.setdefault("config_sha256", config_sha)
        identity = process_identity(os.getpid())
        identity.pop("command", None)
        identity.update(cwd=str(Path.cwd()), session_id=os.getsid(0), program_sha256=digest(Path(__file__)),
                        config=public_config(config), config_sha256=config_sha, status="running",
                        no_network_or_job_operations=True, started_at_utc=utc_now().isoformat())
        save(output / "observer-identity.json", identity)
        previous_monotonic, previous_wall, cycle = None, None, 0
        handlers = {name: signal.getsignal(name) for name in (signal.SIGTERM, signal.SIGINT)}
        for name in handlers:
            signal.signal(name, lambda signum, frame: stopped.set())
        reason = "deadline_reached"
        try:
            with (output / "observer.jsonl").open("a") as log:
                while not stopped.is_set() and utc_now() < timestamp(config["deadline_utc"]):
                    now, monotonic = utc_now(), time.monotonic()
                    cycle += 1
                    roots = []
                    for role, root in config["roots"].items():
                        try:
                            roots.append(snapshot_root(role, root, config, baseline.setdefault(role, {}), now))
                        except (OSError, ValueError, TypeError, AttributeError) as error:
                            roots.append({"role": role, "root": str(root), "alerts": [{"kind": "local_read_error", "error": str(error)}]})
                    gap = monotonic - previous_monotonic if previous_monotonic is not None else None
                    wall_gap = (now - previous_wall).total_seconds() if previous_wall else None
                    alerts = [{"role": row["role"], **alert} for row in roots for alert in row["alerts"]]
                    if gap is not None and gap > config["freshness_s"]:
                        alerts.append({"kind": "observer_monotonic_gap_exceeded", "actual_gap_s": gap})
                    if gap is not None and abs(wall_gap - gap) > 30:
                        alerts.append({"kind": "observer_wall_clock_discontinuity", "wall_gap_s": wall_gap, "monotonic_gap_s": gap})
                    if digest(Path(__file__)) != identity["program_sha256"]:
                        alerts.append({"kind": "observer_source_changed"})
                    row = {"event": "snapshot", "cycle": cycle, "at_utc": now.isoformat(), "monotonic_s": monotonic,
                           "observer_pid": os.getpid(), "observer_ppid": os.getppid(), "cwd": str(Path.cwd()),
                           "observer_process_identity_sha256": identity.get("process_identity_sha256"),
                           "observer_actual_monotonic_gap_s": gap, "observer_actual_wall_gap_s": wall_gap,
                           "interval_s": config["interval_s"], "deadline_utc": config["deadline_utc"],
                           "roots": roots, "alerts": alerts, "GPU_benefit_not_measured": True}
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    os.fsync(log.fileno())
                    save(output / "observer-baseline.json", baseline)
                    identity.update(last_cycle=cycle, last_snapshot_at_utc=row["at_utc"], last_monotonic_s=monotonic,
                                    last_alerts=alerts, real_300s_continuity_not_inferred_from_CPU_tests=True)
                    save(output / "observer-identity.json", identity)
                    previous_monotonic, previous_wall = monotonic, now
                    if once:
                        reason = "requested_one_snapshot"
                        break
                    stopped.wait(max(0, min(config["interval_s"] - (time.monotonic() - monotonic),
                                            (timestamp(config["deadline_utc"]) - utc_now()).total_seconds())))
                if stopped.is_set():
                    reason = "observer_only_signal_stop"
                log.write(json.dumps({"event": "observer_exit", "at_utc": utc_now().isoformat(),
                                      "observer_pid": os.getpid(), "reason": reason, "last_cycle": cycle}) + "\n")
                log.flush()
        except BaseException:
            reason = "observer_error_not_GPU_failure"
            with (output / "observer.jsonl").open("a") as log:
                log.write(json.dumps({"event": "observer_error", "at_utc": utc_now().isoformat(),
                                      "observer_pid": os.getpid(), "traceback": traceback.format_exc(),
                                      "reason": reason}) + "\n")
            raise
        finally:
            identity.update(status="stopped", exit_reason=reason, stopped_at_utc=utc_now().isoformat())
            save(output / "observer-identity.json", identity)
            for name, handler in handlers.items():
                signal.signal(name, handler)


def launch(config, command):
    if command[:2] != [sys.executable, str(Path(__file__).resolve())]:
        raise ValueError("Launch only this local stdlib observer with the current Python")
    if timestamp(config["deadline_utc"]) <= utc_now():
        raise ValueError("Deadline is already in the past")
    predecessor_guard(config)
    output = prepare_output(config)
    with (output / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (output / "observer.lock").open("a") as observer_lock:
            fcntl.flock(observer_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous, error = read_json(output / "observer-identity.json")
        if error not in (None, "not_yet_available"):
            raise RuntimeError("Reconcile previous observer identity")
        if previous.get("pid") and process_identity(previous["pid"]).get("alive") is not False:
            raise RuntimeError("Observer PID still alive or unverified; never duplicate launch")
        for path in output.glob("launch-*.json"):
            receipt, error = read_json(contained(output, path))
            if error or process_identity(receipt.get("pid")).get("alive") is not False:
                raise RuntimeError("Previous detached launch still alive or unverified; reconcile its PID")
        with (output / "launcher.log").open("a") as log:
            child = subprocess.Popen(command, cwd=config["repo"], stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        receipt = {"pid": child.pid, "cwd": str(config["repo"]), "command": command,
                   "started_at_utc": utc_now().isoformat(), "start_new_session": True,
                   "status": "spawned_not_yet_verified_no_continuity_claim", "no_GPU_operations": True}
        with (output / f"launch-{child.pid}.json").open("x") as target:
            target.write(json.dumps(receipt, indent=2) + "\n")
        return receipt


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Independent local-only five-root observer; never queries or controls jobs")
    for role in ROLES:
        parser.add_argument("--" + role.replace("_", "-") + "-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--interval", type=float, default=300)
    parser.add_argument("--freshness", type=float, default=660)
    parser.add_argument("--deadline-utc", default=DEFAULT_DEADLINE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--print-start-command", action="store_true")
    args = parser.parse_args(argv)
    config = configuration(args)
    command = [sys.executable, str(Path(__file__).resolve()), "--repo", str(config["repo"]),
               "--output", str(config["output"]), "--interval", str(config["interval_s"]),
               "--freshness", str(config["freshness_s"]), "--deadline-utc", config["deadline_utc"]]
    for role, root in config["roots"].items():
        command.extend(["--" + role.replace("_", "-") + "-root", str(root)])
    if args.print_start_command:
        print(json.dumps({"config": public_config(config), "launch_command": shlex.join(command + ["--launch"]),
                          "status": "prepared_only_not_started", "predecessor_guard": "old observer must be independently verified stopped"}, indent=2))
    elif args.launch:
        print(json.dumps(launch(config, command), indent=2))
    else:
        run(config, once=args.once)


if __name__ == "__main__":
    main()
