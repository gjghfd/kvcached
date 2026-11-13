from types import SimpleNamespace

import pytest

import test_causal_observation as causal_test


@pytest.fixture(autouse=True)
def restore_process_globals(monkeypatch):
    monkeypatch.setattr(causal_test.os, "environ", dict(causal_test.os.environ))
    monkeypatch.setattr(causal_test.time, "monotonic", causal_test.time.monotonic)
    monkeypatch.setattr(causal_test.live, "host_clock_id", causal_test.live.host_clock_id)


def fixture(monkeypatch, ready_at, exit_at=None):
    clock = [0.0]
    monkeypatch.setattr(causal_test.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(causal_test.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    class Process:
        name = "delayed-lane"
        pid = 123

        @property
        def exitcode(self):
            return 7 if exit_at is not None and clock[0] >= exit_at else None

    ready = SimpleNamespace(is_set=lambda: clock[0] >= ready_at)
    return clock, Process(), ready


def test_slow_spawn_waits_for_readiness_not_old_eight_second_limit(monkeypatch):
    clock, process, ready = fixture(monkeypatch, 9)
    states = causal_test.wait_for_producers([process], [ready])
    assert 9 <= clock[0] < 9.1 and states == [dict(name="delayed-lane", pid=123, exitcode=None, ready=True)]


def test_dead_child_fails_before_deadline_with_pid_and_state(monkeypatch):
    clock, process, ready = fixture(monkeypatch, 90, exit_at=0.1)
    with pytest.raises(AssertionError, match="Producer exited.*delayed-lane.*123.*7.*False"):
        causal_test.wait_for_producers([process], [ready])
    assert clock[0] < 0.2


def test_shared_deadline_is_not_multiplied_by_twelve_children(monkeypatch):
    clock, process, ready = fixture(monkeypatch, 90)
    with pytest.raises(AssertionError, match="startup deadline.*123"):
        causal_test.wait_for_producers([process] * 12, [ready] * 12, timeout_s=1)
    assert clock[0] == 1


def test_ready_but_exited_child_cannot_pass(monkeypatch):
    _, process, ready = fixture(monkeypatch, 0, exit_at=0)
    with pytest.raises(AssertionError, match="Producer exited"):
        causal_test.wait_for_producers([process], [ready])


@pytest.mark.parametrize("timeout", [0, -1, 61, True, float("nan")])
def test_unbounded_or_invalid_deadline_rejected(monkeypatch, timeout):
    _, process, ready = fixture(monkeypatch, 0)
    with pytest.raises(ValueError):
        causal_test.wait_for_producers([process], [ready], timeout_s=timeout)


def test_all_producers_and_matching_events_required(monkeypatch):
    _, process, ready = fixture(monkeypatch, 0)
    for processes, events in (([], []), ([process], []), ([], [ready])):
        with pytest.raises(ValueError):
            causal_test.wait_for_producers(processes, events)


def test_worker_lifetime_covers_startup_and_collection(monkeypatch):
    released = []
    monkeypatch.setattr(causal_test.live, "start_worker_observer", lambda worker: SimpleNamespace(close=lambda: None))
    ready = SimpleNamespace(set=lambda: released.append("ready"))
    stop = SimpleNamespace(wait=lambda seconds: released.append(seconds) or True)
    causal_test.producer_process("/tmp/causal-fixture-test", 0, 0, 0, {}, ready, stop, None)
    assert released == ["ready", 90]


def test_unreleased_worker_is_a_failure_not_silent_expiry(monkeypatch):
    closed = []
    monkeypatch.setattr(causal_test.live, "start_worker_observer", lambda worker: SimpleNamespace(close=lambda: closed.append(True)))
    ready = SimpleNamespace(set=lambda: None)
    with pytest.raises(TimeoutError, match="Coordinator"):
        causal_test.producer_process("/tmp/causal-fixture-test", 0, 0, 0, {}, ready, SimpleNamespace(wait=lambda seconds: False), None)
    assert closed == [True]
