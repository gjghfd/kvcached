import asyncio
import importlib
import json
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return tuple(importlib.import_module(name) for name in ("agentic_trace", "live_trace", "verify_trace"))


def marks():
    return ([{"prompt_tokens": 100084, "output_tokens": 64, "prompt_key": "swe"}],
            [{"prompt_tokens": 4096, "output_tokens": 24576, "count": 5}])


def test_zero_code_realization_keeps_the_original_poisson_draw(modules):
    generator = modules[0]
    code, conv = marks()
    trace = generator.generate_mixed_trace(code, conv, 1e-9, .0064, 1200, 42, allow_empty_tenants=True)
    assert trace["allow_empty_tenants"] is True
    assert {mark["model"] for mark in trace["requests"]} == {1}
    assert trace["requests"][0]["arrival_s"] == random.Random(42 * 4 + 1).expovariate(.0064)
    assert trace == generator.generate_mixed_trace(code, conv, 1e-9, .0064, 1200, 42, allow_empty_tenants=True)
    with pytest.raises(ValueError, match="both tenants"):
        generator.generate_mixed_trace(code, conv, 1e-9, .0064, 1200, 42)


def test_fully_empty_realization_is_retained_without_inserting_requests(modules):
    code, conv = marks()
    trace = modules[0].generate_mixed_trace(code, conv, 1e-9, 1e-9, 1200, 42, allow_empty_tenants=True)
    assert trace["requests"] == []
    assert trace["seed"] == 42
    assert trace["model_rates_rps"] == [1e-9, 1e-9]


def test_nonempty_arrivals_and_mark_sampling_are_unchanged(modules):
    code, conv = marks()
    historical = modules[0].generate_mixed_trace(code, conv, .02, .01, 1200, 42)
    retained = modules[0].generate_mixed_trace(code, conv, .02, .01, 1200, 42, allow_empty_tenants=True)
    retained.pop("allow_empty_tenants")
    assert retained == historical


@pytest.mark.parametrize("value", [1, "true", None])
def test_empty_realization_option_is_not_truthy_coercion(modules, value):
    code, conv = marks()
    with pytest.raises(ValueError):
        modules[0].generate_mixed_trace(code, conv, .02, .01, 1200, 42, allow_empty_tenants=value)


def test_absent_tenant_has_no_fabricated_latency_or_throughput(modules):
    summary = modules[2].request_metrics([], 0, 1200, 120)
    assert summary["completed_requests"] == summary["output_tokens"] == 0
    assert summary["mean_ttft_s"] is summary["p95_ttft_s"] is summary["p99_ttft_s"] is None
    assert summary["mean_tpot_s"] is summary["makespan_s"] is summary["output_tokens_per_complete_drain_s"] is None
    assert summary["deadline_goodput_req_s"] == 0
    assert summary == modules[1].summarize_trace([], 0, 1200, 120, allow_empty=True)
    with pytest.raises(ValueError):
        modules[1].summarize_trace([], 0, 1200, 120)


def test_empty_pair_does_not_create_a_latency_reduction(modules):
    summary = modules[2].request_metrics([], 0, 1200, 120)
    comparisons = modules[2].paired_summary_comparisons([
        {**summary, "policy": "no_move", "repeat": 0}, {**summary, "policy": "ilp", "repeat": 0}], ("no_move",), "ilp")
    assert comparisons == [{"repeat": 0, "baseline": "no_move", "treatment": "ilp",
                            "deadline_met_request_difference": 0}]


def test_fully_empty_native_trace_waits_offered_window_then_terminates(modules, monkeypatch, tmp_path):
    runner = modules[1]
    trace = {"duration_s": .01, "allow_empty_tenants": True, "requests": []}
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace))
    observation = {"requests": [], "mapped_bytes": 0, "block_size": 16, "block_bytes": 1024}
    records = [[observation, observation], [observation, observation]]
    monkeypatch.setattr(runner, "wait_empty", AsyncMock(return_value=records))
    monkeypatch.setattr(runner, "capture_control", AsyncMock(return_value=[]))
    args = SimpleNamespace(kv_budget_mib=10240, trace=trace_path, output=tmp_path, request_timeout_s=3,
                           queue_poll_s=.01, deadline_s=120, solver_limit_s=.1, move_fixed_s=.01)

    async def execute():
        controller = SimpleNamespace(lock=asyncio.Lock(), events=[], snapshots=AsyncMock(return_value=records),
                                     schedule_admission=AsyncMock(), try_direct_admission=AsyncMock())
        summary = await runner.run_trace([None, None], controller, args, trace,
            SimpleNamespace(encode=lambda *args, **kwargs: [1, 2]), "no_move", 0, tmp_path / "trial")
        controller.schedule_admission.assert_not_awaited()
        controller.try_direct_admission.assert_not_awaited()
        return summary

    result = asyncio.run(execute())
    assert result["completed_requests"] == result["moves"] == 0
    assert result["arrival_window_s"] == .01
    assert json.loads((tmp_path / "trial/requests.json").read_text()) == []
