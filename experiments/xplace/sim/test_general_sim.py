"""Simulator, general policies (*_g): episodes follow the long prompt whichever tenant serves it (local only)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from cost_model import default_dp2_params  # noqa: E402
from simulator import simulate  # noqa: E402


def symmetric_params():
    params = default_dp2_params()
    params["interference"]["0"] = dict(params["interference"]["1"])
    return params


def trace(long_model):
    requests = [{"request_id": f"short-{index}", "model": 1 - long_model, "arrival_s": 2.0 * index,
                 "prompt_tokens": 90, "output_tokens": 400} for index in range(40)]
    requests += [{"request_id": f"long-{index}", "model": long_model, "arrival_s": 20.0 + 30.0 * index,
                  "prompt_tokens": 60000, "output_tokens": 64} for index in range(2)]
    return {"duration_s": 100.0, "requests": sorted(requests, key=lambda request: request["arrival_s"])}


def run(policy, long_model):
    return simulate(trace(long_model), policy, step_params=symmetric_params(),
                    sim_params={"budget_gib": 16.0, "kv_mode": "prompt", "reserve_extra_tokens": 256})


@pytest.mark.parametrize("long_model", [0, 1])
def test_episodes_follow_the_long_prompt(long_model):
    out = run("ca_full_tm_slo_g", long_model)
    assert len(out["requests"]) == 42 and out["preemptions"] == [0, 0]
    assert out["episodes"] and {episode["model"] for episode in out["episodes"]} == {long_model}


def test_swapping_roles_mirrors_the_outcome():
    ttft = {}
    for long_model in (0, 1):
        requests = run("ca_full_tm_slo_g", long_model)["requests"]
        ttft[long_model] = sorted(request["first_token_at"] - request["arrival_s"] for request in requests
                                  if request["prompt_tokens"] == 60000)
    assert ttft[0] == pytest.approx(ttft[1], rel=0.05)


def test_parallel_general_policy_has_no_episodes():
    out = run("ca_full_v2_g", 0)
    assert len(out["requests"]) == 42 and out["episodes"] == []
