import asyncio
import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("live_trace")


def observations(epoch):
    empty = {"requests": [], "pages": [], "mapped_bytes": 0, "block_size": 16,
             "block_bytes": 1024**3, "page_bytes": 1024**3, "uniform_page_geometry": True,
             "page_block_capacity": 1, "mapped_free_block_slots": 0}
    records = [[dict(empty) for rank in (0, 1)] for model in (0, 1)]
    for rank in (0, 1):
        requests = [{"id": f"conv-{epoch}-{rank}-{index}", "movable": True, "blocks": [index]}
                    for index in range(3)]
        records[1][rank] = {**empty, "mapped_bytes": 3 * 1024**3, "requests": requests,
                            "pages": [{"id": index, "owners": [request["id"]], "pinned": False,
                                       "memory_gib": 1} for index, request in enumerate(requests)]}
    return records


@pytest.mark.parametrize("policy", ["own_model", "ilp", "kv_only", "kv_only_rebalance", "kv_only_amortized"])
@pytest.mark.parametrize("concurrent", [False, True])
@pytest.mark.parametrize("wait_aware", [False, True])
def test_planning_rejection_or_cooldown_preserves_direct_fifo(runner, monkeypatch, tmp_path, policy, concurrent,
                                                             wait_aware):
    marks = [{"request_id": identifier, "model": model, "arrival_s": arrival,
              "prompt_tokens": 16, "output_tokens": 2,
              "source_fragment": "sample", "source_internal_row": index}
             for index, (identifier, model, arrival) in enumerate((("a", 0, 0), ("b", 1, .001), ("c", 0, .002)))]
    trace = {"duration_s": .08, "requests": marks}
    submissions, blocked_attempts = [], []

    async def stream(engine, identifier, prompt, output, rank, model, role, arrival):
        submitted = time.monotonic()
        submissions.append(identifier)
        await asyncio.sleep(.01)
        return {"request_id": identifier, "model": model, "prompt_tokens": len(prompt),
                "arrived_at": arrival, "submitted_at": submitted, "first_token_at": submitted,
                "completed_at": time.monotonic(), "token_ids": [1, 2]}

    async def direct(prompt, output, budgets, admit, arriving_model):
        records = observations(len(blocked_attempts))
        if arriving_model == 0 and len(blocked_attempts) < 4:
            blocked_attempts.append(records)
            return {"admitted": False, "observations": records, "end_to_end_control_s": .001}
        await admit(0)
        return {"admitted": True, "observations": records, "end_to_end_control_s": .001}

    async def schedule(*args, **options):
        await asyncio.sleep(.002)
        return {"admitted": False, "observations": blocked_attempts[-1], "end_to_end_control_s": .002}

    monkeypatch.setattr(runner, "stream", stream)
    async def reserved(*args, **kwargs):
        return {"reserved_at": time.monotonic(), "reservation_time_source": "fake_runtime",
                "reservation_pending": False}
    monkeypatch.setattr(runner, "wait_reserved", reserved)
    monkeypatch.setattr(runner, "wait_empty", AsyncMock())
    monkeypatch.setattr(runner, "capture_control", AsyncMock(return_value=[]))
    args = SimpleNamespace(kv_budget_mib=4096, request_timeout_s=3, queue_poll_s=.01,
                           deadline_s=120, max_moves=8, candidate_limit=8, solver_limit_s=5,
                           move_fixed_s=.01, planning_retry_min_s=60, planning_retry_cost_factor=4,
                           wait_aware=wait_aware, wait_pause_floor_s=.35, wait_solve_floor_s=.15,
                           wait_copy_seconds_per_gib=5, wait_safety_fraction=.8,
                           output=tmp_path, prepared_prompts={"16": [1] * 16})

    async def run():
        controller = SimpleNamespace(lock=asyncio.Lock(), events=[], concurrent_admission=concurrent,
                                     snapshots=AsyncMock(return_value=observations(0)),
                                     try_direct_admission=AsyncMock(side_effect=direct),
                                     schedule_admission=AsyncMock(side_effect=schedule),
                                     rebalance_after_release=AsyncMock(return_value={"path": "release_rebalance_ilp",
                                         "moves": [], "end_to_end_control_s": .001}))
        result = await runner.run_trace([None, None], controller, args, trace, None, policy, 0, tmp_path / "trial")
        assert controller.schedule_admission.await_count == int(policy != "own_model" and not (policy == "ilp" and wait_aware))
        for call in controller.schedule_admission.await_args_list:
            assert call.kwargs["policy"] == ("kv_only" if policy in ("kv_only_rebalance", "kv_only_amortized") else policy)
        return result

    result = asyncio.run(run())
    assert len(blocked_attempts) == 4
    assert submissions == ["b", "a", "c"]
    assert result["completed_requests"] == 3
    prechecks = result["no_pause_planning_prechecks"]
    assert prechecks == ({"checked": 4, "rejected": 4} if policy == "own_model" or (policy == "ilp" and wait_aware)
                         else {"checked": 1, "rejected": 0})
    if policy == "ilp" and wait_aware:
        assert result["wait_cost_prechecks"] == {"checked": 4, "rejected": 4}
        journal = (tmp_path / "trial/wait-prechecks.jsonl").read_text().splitlines()
        assert len(journal) == 4
    requests = json.loads((tmp_path / "trial/requests.json").read_text())
    assert {request["request_id"] for request in requests} == {"a", "b", "c"}


def test_no_candidate_wait_timeout_preserves_evidence_without_pause_or_submission(runner, monkeypatch, tmp_path):
    trace = {"duration_s": .01, "requests": [{"request_id": "a", "model": 0, "arrival_s": 0,
                                            "prompt_tokens": 16, "output_tokens": 2}]}
    records = observations(0)
    monkeypatch.setattr(runner, "wait_empty", AsyncMock())
    monkeypatch.setattr(runner, "capture_control", AsyncMock(return_value=[]))
    streaming = AsyncMock()
    monkeypatch.setattr(runner, "stream", streaming)
    args = SimpleNamespace(kv_budget_mib=4096, request_timeout_s=.6, queue_poll_s=.01,
                           deadline_s=120, max_moves=8, candidate_limit=8, output=tmp_path,
                           prepared_prompts={"16": [1] * 16})

    async def run():
        controller = SimpleNamespace(lock=asyncio.Lock(), events=[], snapshots=AsyncMock(return_value=records),
                                     try_direct_admission=AsyncMock(return_value={"admitted": False,
                                         "observations": records, "end_to_end_control_s": .001}),
                                     schedule_admission=AsyncMock())
        with pytest.raises(asyncio.TimeoutError):
            await runner.run_trace([None, None], controller, args, trace, None, "own_model", 0, tmp_path / "trial")
        controller.schedule_admission.assert_not_awaited()
        streaming.assert_not_awaited()

    asyncio.run(run())
    assert json.loads((tmp_path / "trial/requests.json").read_text()) == []
    assert json.loads((tmp_path / "trial/decisions.json").read_text()) == []
    assert (tmp_path / "trial/workload.json").exists()


@pytest.mark.parametrize("policy", ["own_model", "kv_only", "ilp"])
def test_prepared_dispatch_keeps_same_cost_parameters_and_accounts_for_planning(runner, monkeypatch, tmp_path, policy):
    records = observations(0)
    mark = {"request_id": "code", "model": 0, "arrival_s": 0, "prompt_tokens": 16,
            "output_tokens": 2, "source_fragment": "sample", "source_internal_row": 1}
    trace = {"duration_s": .03, "requests": [mark]}
    prepared_options, scheduled_options = [], []

    async def prepare(*arguments, **options):
        prepared_options.append(options)
        await asyncio.sleep(.005)
        return {"elapsed_s": .005}

    async def schedule(prompt, output, budgets, admit, **options):
        scheduled_options.append(options)
        await admit(0)
        return {"admitted": True, "moves": [], "observations": records,
                "end_to_end_control_s": .01, "preplan_s": .005}

    async def streaming(engine, identifier, prompt, output, rank, model, role, arrival):
        now = time.monotonic()
        return {"request_id": identifier, "model": model, "prompt_tokens": len(prompt),
                "arrived_at": arrival, "submitted_at": now, "first_token_at": now,
                "completed_at": now + .001, "token_ids": [1, 2]}

    monkeypatch.setattr(runner, "migration_precheck", lambda *args: {"may_fit": True})
    monkeypatch.setattr(runner, "wait_migration_precheck", lambda *args, **options: {
        "may_reduce_latency_sum": True, "estimated_wait_s": 20, "active_plus_incoming": 7})
    monkeypatch.setattr(runner, "stream", streaming)
    monkeypatch.setattr(runner, "wait_reserved", AsyncMock())
    monkeypatch.setattr(runner, "wait_empty", AsyncMock())
    monkeypatch.setattr(runner, "capture_control", AsyncMock(return_value=[]))
    args = SimpleNamespace(kv_budget_mib=4096, request_timeout_s=3, queue_poll_s=.005,
                           deadline_s=120, max_moves=8, candidate_limit=8, solver_limit_s=1,
                           move_fixed_s=0, move_bandwidth_gib_s=1 / .07,
                           planning_retry_min_s=2, planning_retry_cost_factor=4, preplan=True, wait_aware=True,
                           wait_pause_floor_s=.35, wait_solve_floor_s=.59, wait_copy_seconds_per_gib=.07,
                           wait_safety_fraction=.8, wait_incoming_handshake_s=3.06,
                           wait_old_transaction_fixed_s=.09, output=tmp_path, prepared_prompts={"16": [1] * 16})

    async def run():
        controller = SimpleNamespace(lock=asyncio.Lock(), events=[], snapshots=AsyncMock(return_value=records),
            try_direct_admission=AsyncMock(return_value={"admitted": False, "observations": records,
                                                         "end_to_end_control_s": .001}),
            prepare_admission=prepare, schedule_admission=schedule)
        return await runner.run_trace([None, None], controller, args, trace, None, policy, 0, tmp_path / "trial")

    result = asyncio.run(run())
    assert result["completed_requests"] == 1
    assert result["control_s_including_preplan"] == pytest.approx(.015)
    assert len(prepared_options) == len(scheduled_options) == 1
    for name in ("policy", "move_fixed_s", "move_bandwidth_gib_s", "max_moves", "candidate_limit"):
        assert prepared_options[0][name] == scheduled_options[0][name]
    assert scheduled_options[0]["prepared"] == {"elapsed_s": .005}
    if policy == "ilp":
        assert scheduled_options[0]["wait_cost_s"] == 16
        assert scheduled_options[0]["wait_fixed_cost_s"] == 0
        assert scheduled_options[0]["future_transaction_fixed_s"] == .09
    else:
        assert "wait_cost_s" not in scheduled_options[0]
