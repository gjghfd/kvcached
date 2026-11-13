import asyncio
import json
import math
import sys
import time
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "runtime"))

from fake_vllm import FREE_BYTES, PAGE_BYTES, FakeTenant, install_fake_vllm, link_gpu  # noqa: E402

install_fake_vllm()

from calibrate import run_dataplane  # noqa: E402
from controller import PlacementController  # noqa: E402
from cost_model import default_dp2_params  # noqa: E402
from engine import stream  # noqa: E402
from trace_runner import run_trace  # noqa: E402

RANKS = 4


def make_engines(step_s=0.002):
    return [FakeTenant(0, RANKS, step_s), FakeTenant(1, RANKS, step_s)]


def trace_args(**overrides):
    values = dict(planner_params={"migrate_min_gib": 0.5, "code_need_gib": 0.84, "horizon_s": 1.0,
                                  "return_margin_s": 0.0, "return_margin_ratio": 0.0},
                  step_params=default_dp2_params(), trace_sha256="test", seed=1, fairness_threshold_s=0.3,
                  blocked_recheck_s=0.05, plan_retry_s=0.05, poll_s=0.005, return_period_s=0.2,
                  return_cooldown_s=0.2, monitor_s=0.1, drain_timeout_s=60)
    values.update(overrides)
    return types.SimpleNamespace(**values)


def small_trace():
    requests = []
    for index in range(16):
        requests.append({"request_id": f"conv-{index}", "model": 1, "arrival_s": 0.01 * index,
                         "prompt_tokens": 85, "output_tokens": 400, "prompt_key": "conv"})
    requests.append({"request_id": "code-0", "model": 0, "arrival_s": 0.3, "prompt_tokens": 8192,
                     "output_tokens": 64, "prompt_key": "code"})
    return {"duration_s": 2.0, "requests": requests}


PROMPTS = {"conv": list(range(85)), "code": list(range(8192))}


def test_dataplane_moves_across_all_rank_pairs(tmp_path):
    async def scenario():
        engines = make_engines()
        controller = PlacementController(engines, RANKS, [40.0] * RANKS)
        result = await run_dataplane(engines, controller, tmp_path, RANKS)
        calls = [call for call in engines[1].calls if call[0] == "kv_peer"]
        return result, calls

    result, calls = asyncio.run(scenario())
    assert result["status"] == "passed"
    assert len(result["migrations"]) == 8
    assert len(calls) == 8 * RANKS
    assert {(migration["source"], migration["destination"]) for migration in result["migrations"]} >= {(0, 1), (0, 3), (2, 1), (3, 0)}
    assert json.loads((tmp_path / "dataplane.json").read_text())["status"] == "passed"


def run_policy(tmp_path, policy, budget=1.2):
    async def scenario():
        engines = make_engines()
        controller = PlacementController(engines, RANKS, [budget] * RANKS)
        summary = await run_trace(engines, controller, small_trace(), PROMPTS, policy, tmp_path / policy,
                                  trace_args())
        return summary, controller

    return asyncio.run(scenario())


def test_no_move_completes_without_migrations(tmp_path):
    summary, _ = run_policy(tmp_path, "no_move")
    assert summary["status"] == "passed" and summary["requests"] == 17 and summary["moves"] == 0
    requests = json.loads((tmp_path / "no_move" / "requests.json").read_text())
    assert all(request["output_tokens"] == (64 if request["model"] == 0 else 400) for request in requests)


def test_migration_policies_admit_code_by_moving_conv(tmp_path):
    for policy in ("mig_random", "ca_full"):
        summary, controller = run_policy(tmp_path, policy)
        assert summary["status"] == "passed", summary
        assert summary["moves"] > 0, policy
        decisions = json.loads((tmp_path / policy / "decisions.json").read_text())
        admissions = [item for item in decisions if item.get("path") == "admission" and item.get("submitted")]
        assert admissions, decisions
        code = next(request for request in json.loads((tmp_path / policy / "requests.json").read_text())
                    if request["model"] == 0)
        assert code["admission_path"] == "migration"


def test_fair_baseline_runs(tmp_path):
    summary, _ = run_policy(tmp_path, "no_move_fair")
    assert summary["status"] == "passed" and summary["moves"] == 0


def test_kvcached_native_hands_requests_straight_to_vllm(tmp_path):
    async def scenario():
        engines = make_engines()
        link_gpu(engines)
        controller = PlacementController(engines, RANKS, [1.2] * RANKS)
        summary = await run_trace(engines, controller, small_trace(), PROMPTS, "kvcached_native",
                                  tmp_path / "native", trace_args(monitor_s=0.02))
        return summary, engines, controller

    summary, engines, controller = asyncio.run(scenario())
    assert summary["status"] == "passed" and summary["requests"] == 17, summary
    assert summary["native"] and summary["preemptions"] == [0, 0] and summary["rank_unobserved"] == 0
    assert summary["moves"] == summary["transactions"] == summary["holds"] == 0
    # The 1.2 GiB budget is observed, not enforced: the Code lands on arrival on top of the Conv KV.
    assert max(summary["max_mapped_gib"]) > 1.2
    requests = json.loads((tmp_path / "native" / "requests.json").read_text())
    assert {request["admission_path"] for request in requests} == {"native"}
    assert {request["rank_initial"] for request in requests if request["model"] == 1} == set(range(RANKS))
    code = next(request for request in requests if request["model"] == 0)
    assert code["first_token_at"] - code["arrived_at"] < 0.2
    assert json.loads((tmp_path / "native" / "decisions.json").read_text()) == []
    assert summary["kv_mode"] == "native" and engines[0].mode == engines[1].mode == "native"
    assert controller.enforce_budget


def long_output_trace():
    """8 Conv whose final length (85 + 2000) needs 3 pages each, but which hold 1 page when the Code arrives."""
    requests = [{"request_id": f"conv-{index}", "model": 1, "arrival_s": 0.002 * index, "prompt_tokens": 85,
                 "output_tokens": 2000, "prompt_key": "conv"} for index in range(8)]
    requests.append({"request_id": "code-0", "model": 0, "arrival_s": 0.1, "prompt_tokens": 8192,
                     "output_tokens": 64, "prompt_key": "code"})
    return {"duration_s": 1.0, "requests": requests}


QUICK_HEADROOM = {"relief_horizon_s": 0.01, "decode_window_s": 0.01, "slack_pages": 0}


def run_mode(tmp_path, policy, kv_mode, budget=1.2, capacity_pages=None):
    async def scenario():
        engines = make_engines()
        link_gpu(engines)
        if capacity_pages is not None:
            for rank in range(RANKS):
                engines[0].gpu["ballast"][rank] = FREE_BYTES - capacity_pages * PAGE_BYTES
        controller = PlacementController(engines, RANKS, [budget] * RANKS)
        args = trace_args(kv_mode=kv_mode, reserve_extra_tokens=0, monitor_s=0.02,
                          planner_params=dict(trace_args().planner_params, **QUICK_HEADROOM))
        summary = await run_trace(engines, controller, long_output_trace(), PROMPTS, policy,
                                  tmp_path / f"{policy}-{kv_mode}", args)
        return summary, engines

    summary, engines = asyncio.run(scenario())
    requests = json.loads((tmp_path / f"{policy}-{kv_mode}" / "requests.json").read_text())
    return summary, requests, engines


def test_current_length_admission_takes_code_without_migration(tmp_path):
    full, full_requests, _ = run_mode(tmp_path, "ca_full", "full")
    current, current_requests, _ = run_mode(tmp_path, "ca_full", "prompt")
    for summary, requests in ((full, full_requests), (current, current_requests)):
        assert summary["status"] == "passed" and summary["requests"] == 9, summary
        assert all(request["output_tokens"] == (64 if request["model"] == 0 else 2000) for request in requests)
    code = {name: next(request for request in requests if request["model"] == 0)
            for name, requests in (("full", full_requests), ("current", current_requests))}
    # The legacy reservation books every Conv's (replayed) final length, so the Code needs moves; booking only the
    # current length lets it in directly.
    assert code["full"]["admission_path"] == "migration" and full["moves"] > 0
    assert code["current"]["admission_path"] == "direct" and current["moves"] == 0
    assert current["kv_mode"] == "prompt" and full["kv_mode"] == "full"


def test_growth_beyond_capacity_retracts_and_still_completes(tmp_path):
    for policy, kv_mode in (("kvcached_native", "native"), ("no_move", "prompt")):
        summary, requests, engines = run_mode(tmp_path, policy, kv_mode, budget=10.0, capacity_pages=12)
        assert summary["status"] == "passed" and summary["requests"] == 9, (policy, summary)
        assert sum(summary["preemptions"]) > 0, (policy, summary["preemptions"])
        assert sum(summary["recompute_tokens"]) > 0
        assert all(request["output_tokens"] == (64 if request["model"] == 0 else 2000) for request in requests)
        assert summary["preempted_requests"] > 0


def test_transaction_moves_decoding_requests_while_others_are_retracted():
    async def scenario():
        engines = make_engines(step_s=0.01)
        link_gpu(engines)
        controller = PlacementController(engines, RANKS, [10.0] * RANKS)
        await controller.set_mode("prompt")
        streams = [asyncio.create_task(stream(engines[1], f"c{index}", list(range(85)), 400, index % 2, 1, "t",
                                              time.monotonic())) for index in range(4)]
        await asyncio.sleep(0.1)
        retracted = engines[1].requests["c0"]
        retracted.retracted = True
        mover = next(request for request in engines[1].requests.values()
                     if request.rank == 1 and not request.retracted)

        def plan_fn(observed):
            return {"moves": [{"id": mover.id, "model": 1, "source": 1, "destination": 3,
                               "blocks": len(engines[1].blocks_of(mover))}]}

        report = await controller.run_transaction(plan_fn, "test", pause_models=(1,))
        retracted.retracted = False
        await asyncio.gather(*streams)
        return report, mover.rank

    report, rank = asyncio.run(scenario())
    assert report["status"] == "completed", report
    assert len(report["moves"]) == 1 and rank == 3


def test_ballast_sets_shared_kv_capacity_to_whole_budget_pages():
    async def scenario():
        engines = make_engines()
        link_gpu(engines)
        controller = PlacementController(engines, RANKS, [16.0] * RANKS)
        applied = await controller.apply_ballast(16.0, 0.98)
        held = [engines[model].ballast_held.get(rank, 0) for model in (0, 1) for rank in range(RANKS)]
        released = await controller.release_ballast(0.98)
        return applied, held, released

    applied, held, released = asyncio.run(scenario())
    page_gib = PAGE_BYTES / 1024**3
    assert applied["pages"] == math.floor(16.0 / page_gib)
    assert all(abs(value - applied["target_gib"]) <= page_gib / 4 for value in applied["capacity_gib"])
    assert all(math.floor(value / page_gib) == applied["pages"] for value in applied["capacity_gib"])
    assert all(value > 0 for value in held)
    assert released["released_bytes"] == sum(held)
    assert all(abs(after - before) < 1e-9
               for after, before in zip(released["capacity_gib"], applied["natural_capacity_gib"]))


def run_with_overlap(tmp_path, policy, budget):
    async def scenario():
        engines = make_engines()
        engines[1].peer = engines[0]
        controller = PlacementController(engines, RANKS, [budget] * RANKS)
        summary = await run_trace(engines, controller, small_trace(), PROMPTS, policy, tmp_path / policy,
                                  trace_args())
        return summary, controller, engines[1].overlap_steps

    return asyncio.run(scenario())


def test_time_multiplexing_holds_conv_while_code_is_resident(tmp_path):
    baseline, _, baseline_overlap = run_with_overlap(tmp_path, "no_move", 40.0)
    summary, controller, overlap = run_with_overlap(tmp_path, "no_move_tm", 40.0)
    assert baseline["status"] == "passed" and summary["status"] == "passed", summary
    assert summary["requests"] == 17 and summary["holds"] == 1 and summary["held_s"] > 0
    assert not controller.held
    assert baseline_overlap >= 60 and overlap <= 3, (baseline_overlap, overlap)
    requests = json.loads((tmp_path / "no_move_tm" / "requests.json").read_text())
    assert all(request["output_tokens"] == (64 if request["model"] == 0 else 400) for request in requests)
    holds = json.loads((tmp_path / "no_move_tm" / "holds.json").read_text())
    assert len(holds) == 1 and holds[0]["held_s"] > 0


def test_time_multiplexing_keeps_conv_paused_after_migration_admission(tmp_path):
    summary, controller, overlap = run_with_overlap(tmp_path, "ca_full_tm", 1.2)
    assert summary["status"] == "passed", summary
    assert summary["moves"] > 0 and summary["holds"] >= 1 and overlap <= 3, (summary, overlap)
    assert not controller.held
    code = next(request for request in json.loads((tmp_path / "ca_full_tm" / "requests.json").read_text())
                if request["model"] == 0)
    assert code["admission_path"] == "migration"
    holds = json.loads((tmp_path / "ca_full_tm" / "holds.json").read_text())
    assert holds[0].get("from_transaction") is True
    transactions = json.loads((tmp_path / "ca_full_tm" / "transactions.json").read_text())
    assert all(item["status"] in ("completed", "deferred_prefill_in_progress", "deferred_not_movable_after_pause")
               for item in transactions)


def streaming_trace():
    """Conv keep arriving while the Code is resident, so Conv admissions overlap a Code."""
    requests = [{"request_id": f"conv-{index}", "model": 1, "arrival_s": 0.02 * index, "prompt_tokens": 85,
                 "output_tokens": 300, "prompt_key": "conv"} for index in range(24)]
    requests.append({"request_id": "code-0", "model": 0, "arrival_s": 0.15, "prompt_tokens": 8192,
                     "output_tokens": 64, "prompt_key": "code"})
    return {"duration_s": 2.0, "requests": sorted(requests, key=lambda request: request["arrival_s"])}


def run_streaming(tmp_path, policy, budget):
    async def scenario():
        engines = make_engines()
        engines[1].peer = engines[0]
        controller = PlacementController(engines, RANKS, [budget] * RANKS)
        summary = await run_trace(engines, controller, streaming_trace(), PROMPTS, policy, tmp_path / policy,
                                  trace_args(monitor_s=0.5))
        return summary, controller, engines

    return asyncio.run(scenario())


def test_fast_admission_uses_cached_code_view(tmp_path):
    base, _, base_engines = run_streaming(tmp_path, "no_move", 40.0)
    fast, controller, fast_engines = run_streaming(tmp_path, "no_move_v2", 40.0)
    assert base["status"] == "passed" and fast["status"] == "passed", (base, fast)
    assert fast["admission_view"]["cached"] > 0 and base["admission_view"]["cached"] == 0
    code_rpcs = lambda engines: engines[0].engine_core.rpc_counts.get("placement_summary", 0)
    assert code_rpcs(fast_engines) < code_rpcs(base_engines), (code_rpcs(fast_engines), code_rpcs(base_engines))
    requests = json.loads((tmp_path / "no_move_v2" / "requests.json").read_text())
    assert len(requests) == 25 and all(r["output_tokens"] == (64 if r["model"] == 0 else 300) for r in requests)


def test_fast_admission_respects_budget_with_migration(tmp_path):
    summary, controller, _ = run_streaming(tmp_path, "ca_full_v2", 1.2)
    assert summary["status"] == "passed", summary
    assert summary["moves"] > 0 and summary["admission_view"]["cached"] > 0, summary
    random_summary, _, _ = run_streaming(tmp_path, "mig_random_v2", 1.2)
    assert random_summary["status"] == "passed", random_summary
    assert random_summary["moves"] > 0 and random_summary["admission_view"]["cached"] > 0, random_summary


def test_hold_first_pauses_conv_before_code_starts(tmp_path):
    summary, controller, overlap = run_with_overlap(tmp_path, "no_move_tm_v2", 40.0)
    assert summary["status"] == "passed" and summary["holds"] == 1, summary
    assert overlap == 0, overlap
    code = next(request for request in json.loads((tmp_path / "no_move_tm_v2" / "requests.json").read_text())
                if request["model"] == 0)
    hold = json.loads((tmp_path / "no_move_tm_v2" / "holds.json").read_text())[0]
    assert hold["paused_at"] <= code["submitted_at"]


def test_code_view_epoch_rejects_stale_refresh():
    async def make():
        return PlacementController(make_engines(), RANKS, [40.0] * RANKS)

    controller = asyncio.run(make())
    controller.fast_admission = True
    epoch = controller.code_epoch
    controller.begin_code_change()
    controller.offer_code_view(epoch, ["before-admission"])
    assert controller.code_view is None
    controller.end_code_change(["after-admission"])
    controller.offer_code_view(epoch + 1, ["raced-with-admission"])
    assert controller.code_view == ["after-admission"]
    controller.offer_code_view(controller.code_epoch, ["fresh"])
    assert controller.code_view == ["fresh"]


def test_sliced_tm_bounds_conv_stall_without_overlap(tmp_path, monkeypatch):
    import trace_runner

    monkeypatch.setitem(trace_runner.POLICIES, "ca_full_tm_s5",
                        dict(trace_runner.POLICIES["ca_full_tm_s5"], slice=(0.03, 0.01)))

    async def scenario():
        engines = [FakeTenant(0, RANKS, 0.002, chunk_tokens=128), FakeTenant(1, RANKS, 0.002)]
        engines[1].peer = engines[0]
        controller = PlacementController(engines, RANKS, [40.0] * RANKS)
        summary = await run_trace(engines, controller, streaming_trace(), PROMPTS, "ca_full_tm_s5",
                                  tmp_path / "s5", trace_args(monitor_s=0.5, poll_s=0.002))
        return summary, controller, engines[1].overlap_steps

    summary, controller, overlap = asyncio.run(scenario())
    assert summary["status"] == "passed", summary
    assert summary["slices"] >= 2 and summary["holds"] >= summary["slices"] + 1, summary
    assert overlap == 0, overlap
    assert not controller.held
    holds = json.loads((tmp_path / "s5" / "holds.json").read_text())
    assert max(hold["held_s"] for hold in holds) < 0.2, [hold["held_s"] for hold in holds]
    requests = json.loads((tmp_path / "s5" / "requests.json").read_text())
    assert len(requests) == 25 and all(r["output_tokens"] == (64 if r["model"] == 0 else 300) for r in requests)


def test_slo_aware_slicing_triggers_on_waiting_conv(tmp_path, monkeypatch):
    import trace_runner

    monkeypatch.setitem(trace_runner.POLICIES, "ca_full_tm_slo",
                        dict(trace_runner.POLICIES["ca_full_tm_slo"], slice=(0.02, 0.01, 5.0)))

    async def scenario():
        engines = [FakeTenant(0, RANKS, 0.002, chunk_tokens=128), FakeTenant(1, RANKS, 0.002)]
        engines[1].peer = engines[0]
        controller = PlacementController(engines, RANKS, [40.0] * RANKS)
        summary = await run_trace(engines, controller, streaming_trace(), PROMPTS, "ca_full_tm_slo",
                                  tmp_path / "slo", trace_args(monitor_s=0.5, poll_s=0.002))
        return summary, controller, engines[1].overlap_steps

    summary, controller, overlap = asyncio.run(scenario())
    assert summary["status"] == "passed", summary
    slices = json.loads((tmp_path / "slo" / "slices.json").read_text())
    assert slices and {item["reason"] for item in slices} == {"ttft"}, slices
    assert overlap == 0 and not controller.held, overlap
    requests = json.loads((tmp_path / "slo" / "requests.json").read_text())
    assert len(requests) == 25 and all(r["output_tokens"] == (64 if r["model"] == 0 else 300) for r in requests)


@pytest.mark.parametrize("policy", ["no_move_tm_slo", "mig_random_tm_slo", "ca_admit_tm_slo", "ca_return_tm_slo",
                                    "ca_full_tm_slo_v1", "no_move_tm_slo_nh"])
def test_final_scheme_ablations_run_to_completion(tmp_path, monkeypatch, policy):
    import trace_runner

    monkeypatch.setitem(trace_runner.POLICIES, policy, dict(trace_runner.POLICIES[policy], slice=(0.02, 0.01, 5.0)))

    async def scenario():
        engines = [FakeTenant(0, RANKS, 0.002, chunk_tokens=128), FakeTenant(1, RANKS, 0.002)]
        engines[1].peer = engines[0]
        controller = PlacementController(engines, RANKS, [40.0] * RANKS)
        summary = await run_trace(engines, controller, streaming_trace(), PROMPTS, policy, tmp_path / policy,
                                  trace_args(monitor_s=0.5, poll_s=0.002))
        return summary, controller, engines[1].overlap_steps

    summary, controller, overlap = asyncio.run(scenario())
    assert summary["status"] == "passed", summary
    assert summary["holds"] >= 1 and not controller.held, summary
    if trace_runner.POLICIES[policy].get("hold_first"):  # v1 admits Code before Conv reaches its pause boundary
        assert overlap == 0, overlap
    requests = json.loads((tmp_path / policy / "requests.json").read_text())
    assert len(requests) == 25 and all(r["output_tokens"] == (64 if r["model"] == 0 else 300) for r in requests)


def first_code_headroom(tmp_path, policy, planner_overrides=None):
    """Per-rank headroom of the first Code direct-admission attempt, and the trial summary."""
    async def scenario():
        engines = make_engines()
        link_gpu(engines)
        controller = PlacementController(engines, RANKS, [10.0] * RANKS)
        seen = []
        direct = controller.try_direct_admission

        async def spy(model, tokens, admit, route=None, before_admit=None, headroom=None):
            report = await direct(model, tokens, admit, route, before_admit, headroom)
            if model == 0:
                seen.append((report.get("headroom_gib"), report.get("counts")))
            return report

        controller.try_direct_admission = spy
        args = trace_args(kv_mode="prompt", reserve_extra_tokens=0, monitor_s=0.02)
        args.planner_params = dict(args.planner_params, **(planner_overrides or {}))
        label = policy + "".join(f"-{value}" for value in (planner_overrides or {}).values())
        summary = await run_trace(engines, controller, long_output_trace(), PROMPTS, policy, tmp_path / label, args)
        return summary, seen

    summary, seen = asyncio.run(scenario())
    assert summary["status"] == "passed", summary
    return seen[0]


@pytest.mark.parametrize("base,ablated", [("ca_full_tm_slo", "ca_full_tm_slo_nh"), ("no_move_v2", "no_move_v2_nh"),
                                          ("no_move_tm_slo", "no_move_tm_slo_nh")])
def test_no_headroom_ablation_disables_growth_allowance(tmp_path, base, ablated):
    """The *_nh variants must admit with zero growth headroom; the default keeps per-Conv growth + slack."""
    assert all(value == 0.0 for value in first_code_headroom(tmp_path, ablated)[0])
    assert any(value > 0.0 for value in first_code_headroom(tmp_path, base)[0])


def test_relief_scope_destinations_drops_the_floor_on_the_code_rank(tmp_path):
    """Under a plain hold, Conv on the Code's rank does not grow: with relief_scope "destinations" the Code's rank
    only keeps the slack page, while the default also keeps the relief floor for every Conv there."""
    slack = {"slack_pages": 1}
    floor, counts = first_code_headroom(tmp_path, "no_move_tm_v2", dict(slack, relief_scope="all"))
    scoped, _ = first_code_headroom(tmp_path, "no_move_tm_v2", dict(slack, relief_scope="destinations"))
    assert len(set(scoped)) == 1 and scoped[0] > 0.0
    assert all(value > scoped[0] for value, conv in zip(floor, counts[1]) if conv > 0)


def test_no_return_rebalance_while_a_code_is_resident(tmp_path):
    """Current-length mode: the returner waits until no Code is resident (its prefill slows peer copies)."""
    summary, requests, _ = run_mode(tmp_path, "ca_full", "prompt", budget=10.0)
    assert summary["status"] == "passed", summary
    code = next(request for request in requests if request["model"] == 0)
    origin = json.loads((tmp_path / "ca_full-prompt" / "workload.json").read_text())["origin"]
    resident = (code["arrived_at"] - origin, code["completed_at"] - origin)
    decisions = json.loads((tmp_path / "ca_full-prompt" / "decisions.json").read_text())
    returns = [item for item in decisions if item.get("path") in ("return", "return_preplan_rejected")]
    assert all(not (resident[0] < item["elapsed_s"] < resident[1]) for item in returns), (resident, returns)


def test_deadline_censors_trial_records_partial_progress_and_frees_engines(tmp_path):
    """A trial that misses its deadline is censored (not failed): unfinished requests get partial records (TTFT lower
    bound if never started), are aborted in the engine, and the next trial on the same engines runs normally."""
    slow = {"duration_s": 0.3, "requests": [
        *[{"request_id": f"conv-{index}", "model": 1, "arrival_s": 0.01 * index, "prompt_tokens": 85,
           "output_tokens": 20000, "prompt_key": "conv"} for index in range(4)],
        {"request_id": "code-0", "model": 0, "arrival_s": 0.05, "prompt_tokens": 8192, "output_tokens": 64,
         "prompt_key": "code"}]}

    async def scenario():
        engines = make_engines()
        link_gpu(engines)
        controller = PlacementController(engines, RANKS, [10.0] * RANKS)
        censored = await run_trace(engines, controller, slow, PROMPTS, "no_move", tmp_path / "censored",
                                   trace_args(drain_timeout_s=1.5, monitor_s=0.02))  # origin is start + 1 s
        left = sum(len(engine.requests) for engine in engines)
        after = await run_trace(engines, controller, small_trace(), PROMPTS, "no_move", tmp_path / "after",
                                trace_args())
        return censored, left, after

    censored, left, after = asyncio.run(scenario())
    assert censored["status"] == "censored" and censored["errors"] == [], censored
    assert censored["censored"] == 4 and censored["requests"] == 1
    incomplete = json.loads((tmp_path / "censored" / "incomplete.json").read_text())
    assert {record["trace_request_id"] for record in incomplete} == {f"conv-{index}" for index in range(4)}
    for record in incomplete:
        assert record["censored"] and record["first_token_observed"]
        assert 1 < record["output_tokens"] < record["target_output_tokens"]
        assert record["completed_at"] > record["first_token_at"] >= record["arrived_at"]
    assert left == 0
    assert after["status"] == "passed" and after["requests"] == 17


def test_report_includes_censored_records_as_lower_bounds(tmp_path):
    tools = HERE.parent / "tools"
    if not (tools / "report_main.py").exists():
        pytest.skip("analysis tools run locally and are not part of the job upload")
    sys.path.insert(0, str(tools))
    sys.path.insert(0, str(HERE.parent / "sim"))
    import report_main

    root = tmp_path / "trace"
    for policy, censored in (("a", False), ("b", True)):
        directory = root / policy
        directory.mkdir(parents=True)
        done = [{"trace_request_id": "x", "model": 0, "arrived_at": 10.0, "arrival_s": 0.0, "first_token_at": 15.0,
                 "completed_at": 16.0, "output_tokens": 64, "reserved_at": 10.0}]
        (directory / "requests.json").write_text(json.dumps(done))
        (directory / "workload.json").write_text(json.dumps({"origin": 10.0}))
        (directory / "transactions.json").write_text("[]")
        if censored:
            (directory / "incomplete.json").write_text(json.dumps([
                {"trace_request_id": "y", "model": 0, "arrived_at": 20.0, "arrival_s": 10.0, "first_token_at": 500.0,
                 "completed_at": 500.0, "output_tokens": 0, "reserved_at": None, "censored": True,
                 "first_token_observed": False}]))
    requests, origin, _, _ = report_main.load(root, "b")
    assert len(requests) == 2 and any(request.get("censored") for request in requests)
    from analyze import per_request

    item = per_request(next(request for request in requests if request.get("censored")), origin)
    assert item["ttft_s"] == 480.0 and item["tpot_s"] is None
