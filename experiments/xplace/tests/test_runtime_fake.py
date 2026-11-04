import asyncio
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "runtime"))

from fake_vllm import FakeTenant, install_fake_vllm  # noqa: E402

install_fake_vllm()

from calibrate import run_dataplane  # noqa: E402
from controller import PlacementController  # noqa: E402
from cost_model import default_dp2_params  # noqa: E402
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
