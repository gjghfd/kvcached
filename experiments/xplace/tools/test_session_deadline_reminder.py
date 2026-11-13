import datetime
import fcntl
import json
from pathlib import Path

import pytest

import session_deadline_reminder as reminder
import session_watchdog as watchdog
from test_session_watchdog import FakeClient, NOW, append_event, campaign, enable_completion


def prepare(config, deadline=NOW):
    context = enable_completion(config)
    context["last_status_query"] = {"current_job": "123"}
    context["query_policy"] = {"expected_end_reminder": {
        "job": "123", "kind": "slurm_time_limit", "at": watchdog.timestamp(deadline)}}
    watchdog.save_json(Path(config["campaign_file"]), context)
    append_event(config, "task_complete")
    return context


def invoke(config, client, deadline=NOW, **kwargs):
    return reminder.run_reminder(config, "123", watchdog.timestamp(deadline),
                                 client_factory=lambda unused: client, clock=lambda: NOW, **kwargs)


def test_deadline_dispatches_once_without_moving_periodic_clock(campaign):
    prepare(campaign)
    root = Path(campaign["root"])
    watchdog.save_json(root / "state.json", {"attempts": {}, "next_expected_check_at": "2026-11-08T15:05:00Z"})
    client = FakeClient(campaign)
    assert invoke(campaign, client)["delivery"] == "accepted"
    assert invoke(campaign, client)["reason"] == "reminder_already_handled"
    assert len(client.sent) == 1
    assert watchdog.load_state(root)["next_expected_check_at"] == "2026-11-08T15:05:00Z"


@pytest.mark.parametrize("change,reason", [
    ("terminal", "job_not_running"), ("new_job", "job_changed"),
    ("paused", "campaign_inactive"), ("removed", "reminder_changed"),
    ("stop", "stopped"),
])
def test_stale_or_stopped_deadlines_do_not_send(campaign, change, reason):
    context = prepare(campaign)
    if change == "terminal":
        context["jobs"][0]["state"] = "COMPLETED"
    elif change == "new_job":
        context["last_status_query"]["current_job"] = "456"
    elif change == "paused":
        context["status"] = "paused"
    elif change == "removed":
        context["query_policy"]["expected_end_reminder"] = None
    else:
        (Path(campaign["root"]) / "STOP").touch()
    watchdog.save_json(Path(campaign["campaign_file"]), context)
    client = FakeClient(campaign)
    assert invoke(campaign, client)["reason"] == reason
    assert not client.sent


def test_waits_once_and_rechecks_campaign_after_sleep(campaign, monkeypatch):
    deadline = NOW + datetime.timedelta(seconds=20)
    context = prepare(campaign, deadline)
    current = [NOW]
    sleeps = []
    client = FakeClient(campaign)

    def sleep(seconds):
        sleeps.append(seconds)
        current[0] = deadline
        context["jobs"][0]["state"] = "COMPLETED"
        watchdog.save_json(Path(campaign["campaign_file"]), context)

    monkeypatch.setattr(watchdog, "utc_now", lambda: current[0])
    result = reminder.run_reminder(campaign, "123", watchdog.timestamp(deadline),
        client_factory=lambda unused: client, clock=lambda: current[0], sleeper=sleep)
    assert sleeps == [20]
    assert result["reason"] == "job_not_running"
    assert not client.sent


def test_early_sleep_return_waits_until_the_actual_deadline(campaign, monkeypatch):
    deadline = NOW + datetime.timedelta(seconds=20)
    prepare(campaign, deadline)
    current, sleeps = [NOW], []
    client = FakeClient(campaign)

    def sleep(seconds):
        assert not client.sent
        sleeps.append(seconds)
        current[0] = deadline - datetime.timedelta(milliseconds=52) if len(sleeps) == 1 else deadline

    monkeypatch.setattr(watchdog, "utc_now", lambda: current[0])
    result = reminder.run_reminder(campaign, "123", watchdog.timestamp(deadline),
        client_factory=lambda unused: client, clock=lambda: current[0], sleeper=sleep)
    assert sleeps == pytest.approx([20, 0.052])
    assert result["delivery"] == "accepted" and len(client.sent) == 1


def test_nonadvancing_clock_does_not_spin_or_send(campaign):
    deadline = NOW + datetime.timedelta(seconds=20)
    prepare(campaign, deadline)
    client, sleeps = FakeClient(campaign), []
    result = reminder.run_reminder(campaign, "123", watchdog.timestamp(deadline),
        client_factory=lambda unused: client, clock=lambda: NOW, sleeper=sleeps.append)
    assert sleeps == [20]
    assert result["reason"] == "clock_not_advancing" and not client.sent


@pytest.mark.parametrize("offset,reason", [(3601, "deadline_more_than_one_hour_away"), (-1801, "deadline_expired")])
def test_invalid_time_window_does_not_send(campaign, offset, reason):
    deadline = NOW + datetime.timedelta(seconds=offset)
    prepare(campaign, deadline)
    client = FakeClient(campaign)
    assert invoke(campaign, client, deadline)["reason"] == reason
    assert not client.sent


def test_active_turn_is_not_interrupted(campaign):
    prepare(campaign)
    append_event(campaign, "task_started", "current-turn")
    client = FakeClient(campaign)
    assert invoke(campaign, client)["reason"] == "turn_active"
    assert not client.sent


def test_deadline_respects_periodic_lock(campaign):
    prepare(campaign)
    client = FakeClient(campaign)
    with (Path(campaign["root"]) / "tick.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert invoke(campaign, client)["reason"] == "another_check_running"
    assert not client.sent


def test_crash_claim_is_never_retried(campaign):
    prepare(campaign)
    root = Path(campaign["root"])
    watchdog.save_json(root / "deadline-123-20261108T150000Z.json", {"status": "claimed"})
    client = FakeClient(campaign)
    assert invoke(campaign, client)["reason"] == "reminder_already_handled"
    assert not client.sent


def test_campaign_change_before_common_guard_blocks_dispatch(campaign, monkeypatch):
    context = prepare(campaign)
    expected = json.loads(json.dumps(context))
    context["last_status_query"]["current_job"] = "456"
    watchdog.save_json(Path(campaign["campaign_file"]), context)
    client = FakeClient(campaign)
    result = watchdog.check_session(campaign, Path(campaign["root"]), {"attempts": {}}, NOW,
                                   lambda unused: client, expected_campaign=expected)
    assert result["reason"] == "reminder_campaign_changed"
    assert not client.sent
