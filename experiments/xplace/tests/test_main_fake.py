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

import calibrate  # noqa: E402
import main  # noqa: E402
import trace_runner  # noqa: E402

RANKS = 4


def small_grid(num_ranks, output_tokens=120):
    cases = []
    for count in (1, 2, 4, 8, 9, 16):
        layout = [count] * num_ranks
        contexts = [[512] * count for count in layout]
        cases.append({"name": f"bal{count}", "model": 1, "layout": layout, "contexts": contexts,
                      "output_tokens": output_tokens, "max_count": count, "max_context": 512 * count})
    return cases


def test_run_phases_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate, "conv_grid", small_grid)
    original_interference = calibrate.run_interference

    async def quick_interference(engines, controller, root, code_prompt, num_ranks):
        return await original_interference(engines, controller, root, code_prompt, num_ranks, conv_counts=(0, 2))

    monkeypatch.setattr(calibrate, "run_interference", quick_interference)
    trace = {"duration_s": 1.5, "requests": [
        *[{"request_id": f"conv-{index}", "model": 1, "arrival_s": 0.01 * index, "prompt_tokens": 85,
           "output_tokens": 300, "prompt_key": "public-a"} for index in range(16)],
        {"request_id": "code-0", "model": 0, "arrival_s": 0.25, "prompt_tokens": 8192, "output_tokens": 64,
         "prompt_key": "swe-a"}]}
    (tmp_path / "trace.json").write_text(json.dumps(trace))
    (tmp_path / "planner.json").write_text(json.dumps({"migrate_min_gib": 0.5, "code_need_gib": 0.84,
                                                        "horizon_s": 1.0}))
    bank = {"public-a": list(range(85)), "swe-a": list(range(8192))}
    args = types.SimpleNamespace(output=tmp_path / "out", phases="dataplane,grid,interference,fit,trace",
                                 policies="no_move,ca_full", num_ranks=RANKS, budget_margin_gib=0.0,
                                 budget_cap_gib=1.2, step_params=None, planner_params=tmp_path / "planner.json",
                                 trace=tmp_path / "trace.json", seed=1, fairness_threshold_s=0.3,
                                 drain_timeout_s=60)
    args.output.mkdir()
    marks = []

    async def scenario():
        engines = [FakeTenant(0, RANKS, 0.0005), FakeTenant(1, RANKS, 0.0005)]
        original = trace_runner.run_trace

        async def fast_trace(engines, controller, trace, prompts, policy, root, trace_args):
            trace_args.poll_s, trace_args.blocked_recheck_s, trace_args.plan_retry_s = .005, .05, .05
            trace_args.monitor_s, trace_args.return_period_s, trace_args.return_cooldown_s = .1, .2, .2
            return await original(engines, controller, trace, prompts, policy, root, trace_args)

        monkeypatch.setattr(trace_runner, "run_trace", fast_trace)
        return await main.run_phases(engines, args, bank, None, lambda *items, **extra: marks.append(items),
                                     verify=False)

    controller = asyncio.run(scenario())
    statuses = {name: status for name, status in marks}
    assert statuses["dataplane"] == "passed" and statuses["grid"] == "passed"
    assert statuses["interference"] == "passed" and statuses["trace"] == "passed"
    assert statuses["trace:no_move"] == "passed" and statuses["trace:ca_full"] == "passed"
    assert controller.budgets == [1.2] * RANKS
    out = args.output
    assert json.loads((out / "calibration" / "dataplane.json").read_text())["status"] == "passed"
    assert len(json.loads((out / "calibration" / "conv-grid.json").read_text())) == 6
    assert json.loads((out / "calibration" / "step-params.json").read_text())["tenants"]
    summary = json.loads((out / "trace" / "ca_full" / "summary.json").read_text())
    assert summary["status"] == "passed" and summary["moves"] > 0


def test_probe_gate_aborts_on_slow_node(tmp_path, monkeypatch):
    async def fake_probe(engines, controller, case, seed):
        return {**case, "status": "measured", "step_s": 0.030, "window": {"window_s": 5.0}}

    monkeypatch.setattr(calibrate, "measure_case", fake_probe)
    bank = {"public-a": list(range(85))}

    def run(threshold_ms):
        args = types.SimpleNamespace(output=tmp_path / f"out{threshold_ms}", phases="", policies="", num_ranks=RANKS,
                                     budget_margin_gib=0.0, budget_cap_gib=None, step_params=None,
                                     planner_params=None, trace=None, seed=1, fairness_threshold_s=0.3,
                                     drain_timeout_s=60, probe_max_ms=threshold_ms)
        args.output.mkdir()
        marks = []

        async def scenario():
            engines = [FakeTenant(0, RANKS, 0.0005), FakeTenant(1, RANKS, 0.0005)]
            await main.run_phases(engines, args, bank, None, lambda *items, **extra: marks.append((items, extra)),
                                  verify=False)

        try:
            asyncio.run(scenario())
            return None, marks
        except RuntimeError as error:
            return str(error), marks

    error, marks = run(40.0)
    assert error is None and any(items == ("probe", "passed") for items, _ in marks)
    error, marks = run(16.5)
    assert error and "Slow node" in error
    assert any(items == ("probe", "failed") and extra["step_ms"] == 30.0 for items, extra in marks)
