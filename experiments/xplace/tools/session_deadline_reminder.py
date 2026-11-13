import argparse
import datetime
import fcntl
import json
from pathlib import Path
import time

import session_watchdog as watchdog


def target_campaign(config, job, deadline):
    if (Path(config["root"]) / "STOP").exists():
        return None, "stopped"
    campaign = watchdog.load_campaign(config)
    if not campaign or campaign["status"] != "active":
        return None, "campaign_inactive"
    if campaign.get("last_status_query", {}).get("current_job") != job:
        return None, "job_changed"
    current = next((item for item in campaign["jobs"] if item["job"] == job), None)
    if not current or current["state"] not in ("RUNNING", "COMPLETING"):
        return None, "job_not_running"
    reminder = campaign.get("query_policy", {}).get("expected_end_reminder")
    if (not isinstance(reminder, dict) or reminder.get("job") != job
            or reminder.get("kind") != "slurm_time_limit"
            or watchdog.parse_timestamp(reminder["at"]) != deadline):
        return None, "reminder_changed"
    return campaign, None


def run_reminder(config, job, due_at, client_factory=watchdog.IpcFollower,
                 clock=watchdog.utc_now, sleeper=time.sleep):
    if not job.isdigit():
        raise ValueError("job must be numeric")
    deadline = watchdog.parse_timestamp(due_at)
    root = Path(config["root"])
    stamp = deadline.astimezone(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"deadline-{job}-{stamp}"
    receipt = root / (name + ".json")
    identity = {"job": job, "at": watchdog.timestamp(deadline), "kind": "slurm_time_limit"}
    with (root / (name + ".lock")).open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"action": "skip", "reason": "reminder_running"}
        if receipt.exists():
            return {"action": "skip", "reason": "reminder_already_handled"}
        try:
            campaign, reason = target_campaign(config, job, deadline)
            delay = (deadline - clock()).total_seconds()
            if reason:
                result = {"action": "skip", "reason": reason}
            elif delay > 3600:
                result = {"action": "skip", "reason": "deadline_more_than_one_hour_away"}
            else:
                watchdog.save_json(root / (name + "-waiting.json"), {
                    **identity, "started_at": watchdog.timestamp(clock()), "status": "waiting"})
                now = clock()
                while now < deadline:
                    sleeper((deadline - now).total_seconds())
                    after_sleep = clock()
                    if after_sleep <= now:
                        break
                    now = after_sleep
                if now < deadline:
                    result = {"action": "skip", "reason": "clock_not_advancing"}
                elif (now - deadline).total_seconds() > 1800:
                    result = {"action": "skip", "reason": "deadline_expired"}
                else:
                    with (root / "tick.lock").open("a") as tick_lock:
                        try:
                            fcntl.flock(tick_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            result = {"action": "skip", "reason": "another_check_running"}
                        else:
                            campaign, reason = target_campaign(config, job, deadline)
                            if reason:
                                result = {"action": "skip", "reason": reason}
                            else:
                                watchdog.save_json(receipt, {
                                    **identity, "status": "claimed", "claimed_at": watchdog.timestamp(now)})
                                state = watchdog.load_state(root)
                                result = watchdog.check_session(
                                    config, root, state, now, client_factory, expected_campaign=campaign)
                                state["last_deadline_reminder"] = {
                                    **identity, "checked_at": watchdog.timestamp(now), "result": result}
                                watchdog.save_json(root / "state.json", state)
        except Exception as error:
            result = {"action": "blocked", "reason": type(error).__name__, "detail": str(error)[:160]}
        watchdog.save_json(receipt, {
            **identity, "status": "handled", "handled_at": watchdog.timestamp(clock()), "result": result})
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--job", required=True)
    parser.add_argument("--at", required=True)
    args = parser.parse_args()
    result = run_reminder(watchdog.load_config(args.config), args.job, args.at)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if result["action"] == "blocked" else 0)


if __name__ == "__main__":
    main()
