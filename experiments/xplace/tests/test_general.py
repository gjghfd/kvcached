"""General (type-agnostic) policies *_g: planner properties and end-to-end runs on the fake engine.

The algorithm must not depend on which tenant a request belongs to: swapping the tenants' roles must swap the
decisions, every admission keeps growth headroom, and requests of the arrival's own tenant may be moved.
"""
import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "runtime"))

from fake_vllm import FakeTenant, install_fake_vllm  # noqa: E402

install_fake_vllm()

from controller import PlacementController  # noqa: E402
from cost_model import StepModel, default_dp2_params  # noqa: E402
from planner import (DEFAULT_PARAMS, admission_risk, build_state, growth_tokens, plan_admission,  # noqa: E402
                     plan_return, run_seconds, tm_gain)
from test_planner import rank_observation  # noqa: E402
import trace_runner  # noqa: E402

RANKS = 4
PARAMS = dict(DEFAULT_PARAMS, max_moves=24)


def symmetric_params():
    params = default_dp2_params()
    params["interference"]["0"] = dict(params["interference"]["1"])
    return params


MODEL = StepModel(symmetric_params())


# ---------- planner ----------

def test_growth_is_rate_times_time_after_prefill_and_spare():
    rate, solo = 50.0, 26.4e-5
    assert growth_tokens(rate, 100000, 0, 100256, solo, 15.0) == 0.0  # still prefilling through the window
    assert growth_tokens(rate, 85, 2000, 341, solo, 15.0) == pytest.approx(rate * 15.0)  # past its reservation
    fresh = growth_tokens(rate, 85, 0, 341, solo, 15.0)
    assert fresh == pytest.approx(rate * (15.0 - 85 * solo) - 256)  # the spare reserved tokens come first
    slices = (2.0, 1.0, 15.0)
    assert run_seconds(PARAMS, True, 26.0, slices) == pytest.approx(3.0 + 15.0)  # held: slices only, then W
    assert run_seconds(PARAMS, False, 26.0, slices) == pytest.approx(26.0 + 15.0)
    assert run_seconds(PARAMS, True, 26.0, None) == pytest.approx(15.0)


def test_tm_gain_depends_on_prefill_length_not_on_the_tenant():
    def others(tenant):
        return [(1 - tenant, [6] * RANKS, [6 * 4000.0] * RANKS)]

    for tenant in (0, 1):
        assert tm_gain(MODEL, PARAMS, tenant, 100000, others(tenant)) > 0
        assert tm_gain(MODEL, PARAMS, tenant, 86, others(tenant)) < 0
    assert tm_gain(MODEL, PARAMS, 0, 100000, others(0)) == pytest.approx(tm_gain(MODEL, PARAMS, 1, 100000, others(1)))
    assert tm_gain(MODEL, PARAMS, 0, 100000, [(1, [0] * RANKS, [0.0] * RANKS)]) == -PARAMS["pause_fixed_s"]


def tenant_observed(per_rank, prefix):
    return [rank_observation(rank, [{"id": f"{prefix}-{rank}-{index}", "tokens": 13300, "context": 3000 + index}
                                    for index in range(count)]) for rank, count in enumerate(per_rank)]


def observed_for(arrival_model, residents_model, per_rank):
    observed = [None, None]
    observed[residents_model] = tenant_observed(per_rank, "r")
    observed[1 - residents_model] = tenant_observed([0] * RANKS, "empty")
    return observed


def general_state(observed, budget):
    return build_state(observed, [budget] * RANKS, request_growth={}, arrival_growth_gib=0.0)


ARRIVAL_TOKENS = 101626 + 64


def test_general_admission_may_move_the_arrivals_own_tenant():
    """Only the arrival's own tenant holds KV: the legacy planner (other tenant only) finds nothing to move."""
    arrival = {"model": 0, "tokens": ARRIVAL_TOKENS, "prompt_tokens": 101626}
    observed = observed_for(0, 0, [3, 2, 2, 2])
    legacy = plan_admission(build_state(observed, [11.5] * RANKS), arrival, "ca", MODEL, PARAMS)
    general = plan_admission(general_state(observed, 11.5), arrival, "ca", MODEL, PARAMS)
    assert legacy["target"] is None
    assert general["target"] is not None and general["moves"]
    assert all(move["model"] == 0 for move in general["moves"])


@pytest.mark.parametrize("residents_model", [0, 1])
def test_general_admission_mirrors_when_tenants_swap(residents_model):
    """The same situation with the tenants' roles swapped yields the same plan, tenant labels swapped."""
    plans = {}
    for arrival_model in (0, 1):
        residents = residents_model if arrival_model == 0 else 1 - residents_model
        arrival = {"model": arrival_model, "tokens": ARRIVAL_TOKENS, "prompt_tokens": 101626}
        plan = plan_admission(general_state(observed_for(arrival_model, residents, [3, 2, 2, 2]), 11.5),
                              arrival, "ca", MODEL, PARAMS)
        assert plan["target"] is not None
        plans[arrival_model] = (plan["target"], sorted((move["id"], move["source"], move["destination"])
                                                       for move in plan["moves"]),
                                pytest.approx(plan["objective_s"]))
    assert plans[0] == plans[1]


def test_general_return_levels_every_tenant():
    observed = observed_for(0, 0, [12, 1, 1, 1])  # leveling crosses the 16 -> 8 CUDA graph bucket
    params = dict(PARAMS, return_margin_s=0.0, return_margin_ratio=0.0, horizon_s=30.0)
    legacy = plan_return(build_state(observed, [40.0] * RANKS), MODEL, params)
    general = plan_return(general_state(observed, 40.0), MODEL,
                          dict(params, return_tenants=(0, 1), arrival_needs=[], arrival_window_s=500.0))
    assert not legacy["moves"]  # legacy rebalances the Conv tenant (1) only
    assert general["moves"] and all(move["model"] == 0 for move in general["moves"])


def test_admission_risk_prices_needs_that_would_no_longer_fit():
    state = general_state(observed_for(0, 1, [4, 3, 3, 0]), 12.0)
    before = max(budget - used for budget, used in zip(state["budgets"], state["used"]))
    shrunk = list(state["used"])
    shrunk[3] += 6.0  # a plan filling the empty rank: the largest free drops below a 9.4 GiB need
    risky = dict(PARAMS, arrival_needs=[9.4, 0.05, 0.05], arrival_window_s=600.0)
    assert before > 9.4 and admission_risk(state, shrunk, risky) > 0
    assert admission_risk(state, shrunk, dict(risky, arrival_needs=[0.05, 0.05])) == 0.0
    assert admission_risk(state, state["used"], risky) == 0.0


def test_cached_view_is_used_only_while_it_cannot_understate_kv():
    """A view stays exact while no request can allocate past its reservation: a long prefill must finish first
    (at the solo prefill speed), then decode through the spare reserved tokens; a request past its reservation
    grows at once; a held tenant does not grow at all."""
    import time as clock

    async def scenario():
        controller = PlacementController([FakeTenant(0, RANKS), FakeTenant(1, RANKS)], RANKS, [16.0] * RANKS)
        controller.max_decode_rate, controller.min_prefill_s_per_token = 100.0, 25e-5

        def view(age, requests):
            controller.views[0] = {"at": clock.monotonic() - age, "summaries": [{"requests": requests}]}

        prefilling = {"prompt_tokens": 100000, "computed_tokens": 40000, "reservation_tokens": 100256}
        view(10.0, [prefilling])
        assert controller.view_exact(0)
        view(20.0, [prefilling])
        assert not controller.view_exact(0)
        decoding = {"prompt_tokens": 85, "computed_tokens": 3000, "reservation_tokens": 341}
        view(0.5, [decoding])
        assert not controller.view_exact(0)
        controller.held[0] = {"paused_at": clock.monotonic() - 1.0}
        assert controller.view_exact(0)

    asyncio.run(scenario())


# ---------- end to end on the fake engine ----------

GENERAL_PLANNER = {"horizon_s": 1.0, "return_margin_s": 0.0, "return_margin_ratio": 0.0, "tm_grace_s": 0.3,
                   "relief_horizon_s": 0.05, "history_s": 10.0, "rate_prior_s": 1.0}


def trace_args(**overrides):
    values = dict(planner_params=dict(GENERAL_PLANNER), step_params=symmetric_params(), trace_sha256="test", seed=1,
                  fairness_threshold_s=0.3, blocked_recheck_s=0.05, plan_retry_s=0.05, poll_s=0.002,
                  return_period_s=0.2, return_cooldown_s=0.2, monitor_s=0.5, drain_timeout_s=60,
                  kv_mode="prompt", reserve_extra_tokens=0)
    values.update(overrides)
    return types.SimpleNamespace(**values)


def streaming_trace(code_model=0):
    """Short requests keep arriving at one tenant while a long prompt prefills at the other."""
    requests = [{"request_id": f"short-{index}", "model": 1 - code_model, "arrival_s": 0.02 * index,
                 "prompt_tokens": 85, "output_tokens": 300, "prompt_key": "short"} for index in range(24)]
    requests.append({"request_id": "long-0", "model": code_model, "arrival_s": 0.15, "prompt_tokens": 8192,
                     "output_tokens": 64, "prompt_key": "long"})
    return {"duration_s": 2.0, "requests": sorted(requests, key=lambda request: request["arrival_s"])}


PROMPTS = {"short": list(range(85)), "long": list(range(8192))}


def run_general(tmp_path, monkeypatch, policy, code_model=0, budget=40.0, spy=None):
    monkeypatch.setitem(trace_runner.POLICIES, policy, dict(trace_runner.POLICIES[policy], slice=(0.02, 0.01, 5.0))
                        if trace_runner.POLICIES[policy].get("slice") else trace_runner.POLICIES[policy])

    async def scenario():
        engines = [FakeTenant(0, RANKS, 0.002, chunk_tokens=128 if code_model == 0 else None),
                   FakeTenant(1, RANKS, 0.002, chunk_tokens=128 if code_model == 1 else None)]
        engines[1 - code_model].peer = engines[code_model]
        controller = PlacementController(engines, RANKS, [budget] * RANKS)
        if spy is not None:
            spy(controller)
        label = f"{policy}-long{code_model}"
        summary = await trace_runner.run_trace(engines, controller, streaming_trace(code_model), PROMPTS, policy,
                                               tmp_path / label, trace_args())
        return summary, controller, engines[1 - code_model].overlap_steps, tmp_path / label

    return asyncio.run(scenario())


def exact(root):
    requests = json.loads((root / "requests.json").read_text())
    return len(requests) == 25 and all(request["output_tokens"] == (64 if request["prompt_tokens"] == 8192 else 300)
                                       for request in requests)


def test_general_policies_exist_for_every_controlled_policy():
    for name, spec in list(trace_runner.POLICIES.items()):
        if spec.get("native") or spec.get("general"):
            continue
        assert trace_runner.POLICIES[f"{name}_g"]["general"] is True
    assert "kvcached_native_g" not in trace_runner.POLICIES


@pytest.mark.parametrize("code_model", [0, 1])
def test_general_final_scheme_holds_the_other_tenant_whichever_it_is(tmp_path, monkeypatch, code_model):
    """The long prompt starts an episode in either tenant; the other tenant is held (no co-run step) and sliced."""
    summary, controller, overlap, root = run_general(tmp_path, monkeypatch, "ca_full_tm_slo_g", code_model)
    assert summary["status"] == "passed", summary
    assert exact(root)
    episodes = json.loads((root / "episodes.json").read_text())
    assert episodes and {item["model"] for item in episodes} == {code_model}, episodes
    assert summary["holds"] >= 1 and not controller.held, summary
    assert overlap == 0, overlap
    slices = json.loads((root / "slices.json").read_text())
    assert slices and all(item["exclusive"] == code_model for item in slices), slices


def test_general_parallel_policy_never_holds(tmp_path, monkeypatch):
    summary, controller, overlap, root = run_general(tmp_path, monkeypatch, "no_move_v2_nh_g")
    assert summary["status"] == "passed" and exact(root)
    assert summary["holds"] == 0 and summary["episodes"] == 0 and overlap > 0


@pytest.mark.parametrize("policy,positive", [("ca_full_tm_slo_g", True), ("no_move_v2_nh_g", False)])
def test_general_headroom_applies_to_every_admission(tmp_path, monkeypatch, policy, positive):
    """Short requests are admitted with growth headroom too (no tenant is assumed not to grow); *_nh keeps none."""
    seen = {0: [], 1: []}

    def spy(controller):
        direct = controller.try_direct_admission

        async def wrapped(model, tokens, admit, route=None, before_admit=None, headroom=None):
            report = await direct(model, tokens, admit, route, before_admit, headroom)
            seen[model].append(report.get("headroom_gib"))
            return report

        controller.try_direct_admission = wrapped

    summary, _, _, root = run_general(tmp_path, monkeypatch, policy, spy=spy)
    assert summary["status"] == "passed" and exact(root)
    for model in (0, 1):
        assert seen[model], seen
        values = [value for item in seen[model] for value in item]
        assert all(value > 0 for value in values) if positive else all(value == 0 for value in values)
