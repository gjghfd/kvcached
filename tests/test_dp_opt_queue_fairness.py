import asyncio
import importlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("queue_fairness"), importlib.import_module("live_trace")


def entry(identifier, model, arrival):
    return {"mark": {"request_id": identifier, "model": model}, "arrived_at": arrival}


@pytest.mark.parametrize("threshold", [0, -1, float("nan"), float("inf")])
def test_threshold_must_be_finite_positive(modules, threshold):
    with pytest.raises(ValueError):
        modules[0].FairAdmissionGate(threshold)


def test_oldest_overdue_request_latches_until_actual_reservation(modules):
    gate = modules[0].FairAdmissionGate(30)
    entries = [entry("code", 0, 0), entry("conv", 1, 5)]
    assert gate.update(entries, 29.99) is None
    assert gate.authorization(1, "conv", 29.99)["allowed"]
    assert gate.update(entries, 30)["owner"]["request_id"] == "code"
    assert not gate.authorization(1, "conv", 30)["allowed"]
    assert gate.update([entry("even-older", 1, -10)], 40) is None
    assert gate.reserved("conv", 35, 40) is None
    assert gate.owner["request_id"] == "code"
    assert gate.reserved("code", 40, 41)["hold_s"] == 11
    assert gate.owner is None
    assert gate.authorization(1, "conv", 41)["allowed"]


def test_block_journal_deduplicates_polls_and_release_rejects_future_time(modules):
    gate = modules[0].FairAdmissionGate(1)
    gate.update([entry("old", 1, 0)], 1)
    assert gate.blocked(0, "new", 2)["allowed"] is False
    assert gate.blocked(0, "new", 3) is None
    assert gate.summary()["suppressed_attempts"] == 2
    with pytest.raises(ValueError):
        gate.reserved("old", 5, 4)
    assert gate.owner is not None


@pytest.mark.parametrize("threshold", ["30", "0", "nan"])
def test_trace_cli_accepts_new_baseline_and_validates_threshold(modules, monkeypatch, tmp_path, threshold):
    runner = modules[1]
    run = AsyncMock()
    monkeypatch.setattr(runner, "run", run)
    monkeypatch.setattr(sys, "argv", ["live_trace.py", "--checkpoint", "unused", "--fit", "fit.json",
                                     "--trace", "trace.json", "--output", str(tmp_path), "--policies",
                                     "no_move", "no_move_fair", "kv_only", "--fairness-queue-threshold-s", threshold])
    if threshold == "30":
        runner.main()
        assert run.call_args.args[0].policies == ["no_move", "no_move_fair", "kv_only"]
        assert run.call_args.args[0].fairness_queue_threshold_s == 30
    else:
        with pytest.raises(SystemExit):
            runner.main()
        run.assert_not_awaited()


def test_fair_launcher_keeps_historical_controls_and_explicit_threshold():
    scripts = Path(__file__).resolve().parents[1] / "experiments/dp_opt"
    launcher = (scripts / "agentic-fair-baselines.sbatch").read_text()
    assert "no_move no_move_fair own_model kv_only ilp" in launcher
    assert "DP_OPT_FAIRNESS_QUEUE_THRESHOLD_S:-30" in launcher
    assert "agentic-revalidated.sbatch" in launcher
    serving = (scripts / "trace-benefit.sbatch").read_text()
    assert 'extra_args+=(--policies "${trace_policies[@]}")' in serving
    assert 'extra_args+=(--fairness-queue-threshold-s "$DP_OPT_FAIRNESS_QUEUE_THRESHOLD_S")' in serving


@pytest.mark.parametrize("policy", ["no_move", "no_move_fair"])
def test_fair_baseline_blocks_new_competitors_without_migration_or_pause(modules, monkeypatch, tmp_path, policy):
    runner = modules[1]
    marks = [{"request_id": identifier, "model": model, "arrival_s": arrival,
              "prompt_tokens": 16, "output_tokens": 2, "source_fragment": "sample", "source_internal_row": index}
             for index, (identifier, model, arrival) in enumerate(
                 [("conv-0", 1, 0), ("code", 0, .001)] + [(f"conv-{index}", 1, .008 * index) for index in range(1, 8)])]
    trace = {"duration_s": .15, "requests": marks}
    empty = {"requests": [], "pages": [], "mapped_bytes": 0, "block_size": 16, "block_bytes": 1024,
             "page_bytes": 1024, "page_block_capacity": 1, "mapped_free_block_slots": 0}
    observed = [[dict(empty), dict(empty)], [dict(empty), dict(empty)]]
    blocked_until = 0
    submitted = []

    async def direct(prompt, output, budgets, admit, arriving_model):
        nonlocal blocked_until
        if arriving_model == 0 and time.monotonic() < blocked_until:
            return {"admitted": False, "submitted": False, "moves": [], "observations": observed,
                    "end_to_end_control_s": .0001}
        if arriving_model == 1:
            blocked_until = time.monotonic() + .025
        await admit(0)
        return {"admitted": True, "submitted": True, "moves": [], "observations": observed,
                "end_to_end_control_s": .0001}

    async def stream(engine, identifier, prompt, output, rank, model, role, arrival):
        now = time.monotonic()
        submitted.append((identifier, model, now))
        await asyncio.sleep(.04)
        return {"request_id": identifier, "model": model, "prompt_tokens": len(prompt), "arrived_at": arrival,
                "submitted_at": now, "first_token_at": now + .02,
                "completed_at": time.monotonic(), "token_ids": [1, 2]}

    async def reservation(*arguments, **options):
        await asyncio.sleep(0)
        return {"reservation_pending": False, "reserved_at": time.monotonic(),
                "reservation_time_source": "scheduler_monotonic"}

    monkeypatch.setattr(runner, "wait_empty", AsyncMock())
    monkeypatch.setattr(runner, "capture_control", AsyncMock(return_value=[]))
    monkeypatch.setattr(runner, "stream", stream)
    monkeypatch.setattr(runner, "wait_reserved", reservation)
    args = SimpleNamespace(kv_budget_mib=4096, request_timeout_s=3, queue_poll_s=.002, deadline_s=120,
                           max_moves=8, candidate_limit=8, output=tmp_path, fairness_queue_threshold_s=.012,
                           prepared_prompts={"16": [1] * 16})

    async def run():
        controller = SimpleNamespace(lock=asyncio.Lock(), events=[], concurrent_admission=True,
                                     snapshots=AsyncMock(return_value=observed), try_direct_admission=direct,
                                     schedule_admission=AsyncMock(), pause=AsyncMock())
        result = await runner.run_trace([None, None], controller, args, trace, None, policy, 0, tmp_path / "trial")
        controller.schedule_admission.assert_not_awaited()
        controller.pause.assert_not_awaited()
        return result

    summary = asyncio.run(run())
    assert summary["completed_requests"] == len(marks)
    assert summary["moves"] == 0
    assert len({identifier for identifier, model, stamp in submitted}) == len(marks)
    if policy == "no_move_fair":
        events = json.loads((tmp_path / "trial/fairness-events.json").read_text())
        owner = next(event for event in events if event["event"] == "latch" and event["owner"]["request_id"] == "code")
        release = next(event for event in events if event["event"] == "release" and event["owner"]["request_id"] == "code")
        assert not any(model == 1 and owner["observed_at"] <= stamp < release["observed_at"]
                       for identifier, model, stamp in submitted)
        assert summary["fairness"]["owner_at_end"] is None
        assert summary["fairness"]["suppressed_attempts"] > 0
        verifier = importlib.import_module("verify_queue_fairness")
        assert verifier.verify(tmp_path / "trial", .012)["status"] == "raw_queue_fairness_verified"
    else:
        assert "fairness" not in summary
        assert not (tmp_path / "trial/fairness-events.json").exists()


def test_pending_allocator_reservation_does_not_release_fairness_on_submission(modules, monkeypatch, tmp_path):
    runner = modules[1]
    marks = [{"request_id": identifier, "model": model, "arrival_s": arrival, "prompt_tokens": 16,
              "output_tokens": 2, "source_fragment": "sample", "source_internal_row": index}
             for index, (identifier, model, arrival) in enumerate((("code", 0, 0), ("conv", 1, .015)))]
    empty = {"requests": [], "pages": [], "mapped_bytes": 0, "block_size": 16, "block_bytes": 1024}
    observed = [[dict(empty), dict(empty)], [dict(empty), dict(empty)]]
    stamps = {}

    async def streaming(engine, identifier, prompt, output, rank, model, role, arrival):
        now = time.monotonic()
        stamps[identifier] = now
        await asyncio.sleep(.08)
        return {"request_id": identifier, "model": model, "prompt_tokens": len(prompt), "arrived_at": arrival,
                "submitted_at": now, "first_token_at": now + .07,
                "completed_at": time.monotonic(), "token_ids": [1, 2]}

    async def reservation(controller, task, identifier, model, rank, **options):
        if identifier == "code" and not options.get("wait_for_capacity"):
            return {"reservation_pending": True}
        if identifier == "code":
            await asyncio.sleep(.045)
        await asyncio.sleep(0)
        return {"reservation_pending": False, "reserved_at": time.monotonic(),
                "reservation_time_source": "scheduler_monotonic"}

    async def direct(prompt, output, budgets, admit, arriving_model):
        status = await admit(0)
        return {"admitted": not status["reservation_pending"], "submitted": True, "moves": [],
                "observations": observed, "end_to_end_control_s": .0001}

    monkeypatch.setattr(runner, "wait_empty", AsyncMock())
    monkeypatch.setattr(runner, "capture_control", AsyncMock(return_value=[]))
    monkeypatch.setattr(runner, "stream", streaming)
    monkeypatch.setattr(runner, "wait_reserved", reservation)
    args = SimpleNamespace(kv_budget_mib=4096, request_timeout_s=3, queue_poll_s=.002, deadline_s=120,
                           max_moves=8, candidate_limit=8, output=tmp_path, fairness_queue_threshold_s=.005,
                           prepared_prompts={"16": [1] * 16})

    async def run():
        controller = SimpleNamespace(lock=asyncio.Lock(), events=[], concurrent_admission=True,
                                     snapshots=AsyncMock(return_value=observed), try_direct_admission=direct,
                                     check_capacity=AsyncMock(), schedule_admission=AsyncMock())
        return await runner.run_trace([None, None], controller, args, {"duration_s": .15, "requests": marks},
                                      None, "no_move_fair", 0, tmp_path / "trial")

    asyncio.run(run())
    events = json.loads((tmp_path / "trial/fairness-events.json").read_text())
    release = next(event for event in events if event["event"] == "release")
    assert stamps["code"] < release["owner"]["activated_at"] < release["reserved_at"] <= stamps["conv"]
    assert stamps["conv"] < stamps["code"] + .08
    verifier = importlib.import_module("verify_queue_fairness")
    verified = verifier.verify(tmp_path / "trial", .005)
    assert verified["activations"] == sum(event["event"] == "latch" for event in events)
    assert sum(event["event"] == "latch" and event["owner"]["request_id"] == "code" for event in events) == 1


def fair_raw(root):
    root.mkdir()
    requests = [{"request_id": identifier, "model": model, "arrived_at": arrival, "submitted_at": submitted,
                 "reserved_at": reserved, "reservation_observed_at": reserved, "arrival_dispatch_lag_s": 0}
                for identifier, model, arrival, submitted, reserved in (("code", 0, 0, 40, 40), ("conv", 1, 1, 42, 42))]
    owner = {"request_id": "code", "model": 0, "arrived_at": 0, "activated_at": 30}
    events = [{"event": "latch", "observed_at": 30, "owner": owner, "queue_age_s": 30, "threshold_s": 30},
              {"event": "blocked", "observed_at": 31, "request_id": "conv", "model": 1,
               "owner_request_id": "code", "owner_model": 0, "allowed": False},
              {"event": "release", "observed_at": 41, "reserved_at": 40, "owner": owner, "hold_s": 11}]
    decisions = [{"request_id": identifier, "admitted": True, "submitted": True, "moves": [], "pause_call_s": 0,
                  "fairness_authorization": {"observed_at": now, "request_id": identifier, "model": model,
                                             "owner_request_id": active, "owner_model": 0 if active else None, "allowed": True}}
                 for identifier, model, now, active in (("code", 0, 39, "code"), ("conv", 1, 42, None))]
    summary = {"policy": "no_move_fair", "moves": 0, "ilp_attempts": 0,
               "fairness": {"threshold_s": 30, "owner_at_end": None, "activations": 1, "releases": 1,
                            "suppressed_attempts": 1}}
    for name, data in (("requests", requests), ("decisions", decisions), ("summary", summary), ("fairness-events", events)):
        (root / f"{name}.json").write_text(json.dumps(data))
    (root / "workload.json").write_text(json.dumps({"requests": [{"request_id": record["request_id"]} for record in requests]}))
    (root / "fairness-events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))


@pytest.mark.parametrize("mutation", [None, "early", "foreign", "release", "late_foreign_KV", "journal", "oldest", "moves"])
def test_independent_fairness_replay_rejects_false_gate_evidence(modules, tmp_path, mutation):
    verifier = importlib.import_module("verify_queue_fairness")
    root = tmp_path / "raw"
    fair_raw(root)
    events = json.loads((root / "fairness-events.json").read_text())
    requests = json.loads((root / "requests.json").read_text())
    decisions = json.loads((root / "decisions.json").read_text())
    if mutation == "early":
        events[0]["observed_at"] = 29
    elif mutation == "foreign":
        decisions[1]["fairness_authorization"]["observed_at"] = 35
    elif mutation == "release":
        events[2]["reserved_at"] = 39
    elif mutation == "late_foreign_KV":
        requests[1]["reserved_at"] = 35
    elif mutation == "oldest":
        requests[1]["arrived_at"] = -1
    elif mutation == "moves":
        decisions[0]["moves"] = [{"request_id": "conv"}]
    for name, data in (("requests", requests), ("decisions", decisions), ("fairness-events", events)):
        (root / f"{name}.json").write_text(json.dumps(data))
    (root / "fairness-events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    if mutation == "journal":
        (root / "fairness-events.jsonl").write_text("\n")
    if mutation is None:
        assert verifier.verify(root, 30)["requests"] == 2
    else:
        with pytest.raises(ValueError):
            verifier.verify(root, 30)
