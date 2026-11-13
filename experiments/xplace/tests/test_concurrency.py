import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from concurrency import InFlightLimiter, validate_limits
from controller import PlacementController
from fake_vllm import FakeTenant
from main import parse_args
from test_runtime_fake import trace_args
from trace_runner import run_trace


@pytest.mark.parametrize("limits", [(-1, 2), (1,), (1, 2, 3), (True, 1), (1.0, 2), None])
def test_invalid_limits(limits):
    with pytest.raises(ValueError):
        validate_limits(limits)


def test_slot_lifecycle_independent_tenants_and_queued_wait():
    limiter = InFlightLimiter((1, 0))
    first, waiting, other = [{"mark": {"model": model}} for model in (0, 0, 1)]
    limiter.acquire(first, 0)
    assert not limiter.allow(waiting, 1)
    assert limiter.allow(other, 1)
    limiter.acquire(other, 1)
    assert limiter.wait_s(waiting, 3) == 2
    with pytest.raises(RuntimeError):
        limiter.acquire(first, 3)
    limiter.release(first)
    limiter.release(first)
    limiter.acquire(waiting, 4)
    assert limiter.wait_s(waiting, 5) == 3
    assert limiter.active == [1, 1]
    assert limiter.peak == [1, 1]


def test_cli_limit_validation():
    common = ["--checkpoint", "test", "--output", "out", "--prompt-bank", "bank.json", "--phases", "dataplane"]
    assert parse_args(common).max_inflight == (0, 0)
    assert parse_args(common + ["--max-inflight", "3", "64"]).max_inflight == [3, 64]
    with pytest.raises(SystemExit):
        parse_args(common + ["--max-inflight", "-1", "64"])


@pytest.mark.parametrize("policy", ["no_move", "no_move_v2", "ca_full_v2", "kvcached_native", "no_move_v2_g",
                                    "mig_random_v2_g", "ca_full_v2_g", "ca_admit_v2_g", "ca_periodic_v2_g",
                                    "ca_no_imbalance_v2_g"])
def test_every_policy_obeys_same_limit_and_preserves_central_queue(tmp_path, policy):
    prompts = {"short": list(range(85))}
    trace = {"duration_s": 2, "requests": [
        {"request_id": f"request-{model}-{index}", "model": model, "arrival_s": index * 0.001,
         "prompt_tokens": 85, "output_tokens": 48, "prompt_key": "short"}
        for index in range(5) for model in (0, 1)]}

    async def scenario():
        engines = [FakeTenant(model, 2, 0.002) for model in (0, 1)]
        controller = PlacementController(engines, 2, [40, 40])
        summary = await run_trace(engines, controller, trace, prompts, policy, tmp_path,
                                  trace_args(max_inflight=(1, 2), kv_mode="prompt", reserve_extra_tokens=0))
        return summary

    summary = asyncio.run(scenario())
    assert summary["status"] == "passed", summary
    assert summary["requests"] == summary["expected"] == 10
    assert summary["concurrency"]["max_inflight"] == [1, 2]
    assert summary["concurrency"]["peak_inflight"] == [1, 2]
    assert summary["concurrency"]["inflight_after_cleanup"] == [0, 0]
    assert all(count > 0 for count in summary["concurrency"]["blocked_checks"])
    workload = json.loads((tmp_path / "workload.json").read_text())
    assert workload["max_inflight"] == [1, 2]
    requests = json.loads((tmp_path / "requests.json").read_text())
    assert any(request["concurrency_head_wait_s"] > 0 for request in requests)
    for model, limit in enumerate((1, 2)):
        entries = [request for request in requests if request["model"] == model]
        active = 0
        for _, delta in sorted([(request["submitted_at"], 1) for request in entries]
                               + [(request["completed_at"], -1) for request in entries]):
            active += delta
            assert 0 <= active <= limit
        ordered = sorted(entries, key=lambda request: request["submitted_at"])
        assert [request["trace_request_id"] for request in ordered] == [f"request-{model}-{index}" for index in range(5)]
        assert all(request["first_token_at"] >= request["submitted_at"] >= request["arrived_at"] for request in entries)


@pytest.mark.parametrize("policy", ["no_move_v2_g", "kvcached_native"])
def test_censoring_retains_cap_wait_and_releases_slots(tmp_path, policy):
    trace = {"duration_s": 1.2, "requests": [
        {"request_id": f"request-{index}", "model": 1, "arrival_s": 0, "prompt_tokens": 85,
         "output_tokens": 100000, "prompt_key": "short"} for index in range(3)]}

    async def scenario():
        engines = [FakeTenant(model, 2, 0.002) for model in (0, 1)]
        controller = PlacementController(engines, 2, [40, 40])
        return await run_trace(engines, controller, trace, {"short": list(range(85))}, policy, tmp_path,
                               trace_args(max_inflight=(1, 1), kv_mode="prompt", drain_timeout_s=0.1))

    summary = asyncio.run(scenario())
    assert summary["status"] == "censored", summary
    assert summary["concurrency"]["peak_inflight"] == [0, 1]
    assert summary["concurrency"]["inflight_after_cleanup"] == [0, 0]
    incomplete = json.loads((tmp_path / "incomplete.json").read_text())
    assert len(incomplete) == 3
    assert sum(request["submitted_at"] is not None for request in incomplete) == 1
    assert incomplete[1]["concurrency_head_wait_s"] > 0
