import asyncio
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "runtime"))

from fake_vllm import FakeTenant, install_fake_vllm, link_gpu  # noqa: E402

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


def test_native_trial_is_bracketed_by_ballast(tmp_path, monkeypatch):
    trace = {"duration_s": 1.0, "requests": [
        *[{"request_id": f"conv-{index}", "model": 1, "arrival_s": 0.01 * index, "prompt_tokens": 85,
           "output_tokens": 200, "prompt_key": "public-a"} for index in range(8)],
        {"request_id": "code-0", "model": 0, "arrival_s": 0.1, "prompt_tokens": 8192, "output_tokens": 64,
         "prompt_key": "swe-a"}]}
    (tmp_path / "trace.json").write_text(json.dumps(trace))
    bank = {"public-a": list(range(85)), "swe-a": list(range(8192))}
    args = types.SimpleNamespace(output=tmp_path / "out", phases="trace", policies="kvcached_native,no_move",
                                 num_ranks=RANKS, budget_margin_gib=0.0, budget_cap_gib=1.2, step_params=None,
                                 planner_params=None, trace=tmp_path / "trace.json", seed=1,
                                 fairness_threshold_s=0.3, drain_timeout_s=60)
    args.output.mkdir()
    marks = []

    async def scenario():
        engines = [FakeTenant(0, RANKS, 0.0005), FakeTenant(1, RANKS, 0.0005)]
        link_gpu(engines)
        original = trace_runner.run_trace

        async def fast_trace(engines, controller, trace, prompts, policy, root, trace_args):
            trace_args.poll_s, trace_args.blocked_recheck_s, trace_args.plan_retry_s = .005, .05, .05
            trace_args.monitor_s = .02
            return await original(engines, controller, trace, prompts, policy, root, trace_args)

        monkeypatch.setattr(trace_runner, "run_trace", fast_trace)
        await main.run_phases(engines, args, bank, None, lambda *items, **extra: marks.append(items), verify=False)
        return engines

    engines = asyncio.run(scenario())
    assert marks.index(("ballast:kvcached_native", "applied")) < marks.index(("trace:kvcached_native", "started"))
    assert marks.index(("ballast:kvcached_native", "released")) < marks.index(("trace:kvcached_native", "passed"))
    assert marks.index(("trace:kvcached_native", "passed")) < marks.index(("trace:no_move", "started"))
    assert ("trace:no_move", "passed") in marks
    applied = json.loads((args.output / "ballast-kvcached_native.json").read_text())
    released = json.loads((args.output / "ballast-kvcached_native-released.json").read_text())
    assert all(abs(value - applied["target_gib"]) < 0.03 for value in applied["capacity_gib"])
    assert all(abs(after - before) < 1e-9
               for after, before in zip(released["capacity_gib"], applied["natural_capacity_gib"]))
    assert not any(engine.ballast_held for engine in engines)


def test_current_length_mode_caps_every_trial(tmp_path, monkeypatch):
    trace = {"duration_s": 0.6, "requests": [
        *[{"request_id": f"conv-{index}", "model": 1, "arrival_s": 0.01 * index, "prompt_tokens": 85,
           "output_tokens": 150, "prompt_key": "public-a"} for index in range(8)],
        {"request_id": "code-0", "model": 0, "arrival_s": 0.1, "prompt_tokens": 8192, "output_tokens": 64,
         "prompt_key": "swe-a"}]}
    (tmp_path / "trace.json").write_text(json.dumps(trace))
    bank = {"public-a": list(range(85)), "swe-a": list(range(8192))}
    args = types.SimpleNamespace(output=tmp_path / "out", phases="trace", policies="no_move,ca_full",
                                 num_ranks=RANKS, budget_margin_gib=0.0, budget_cap_gib=1.2, step_params=None,
                                 planner_params=None, trace=tmp_path / "trace.json", seed=1,
                                 fairness_threshold_s=0.3, drain_timeout_s=60, kv_mode="prompt",
                                 reserve_extra_tokens=128)
    args.output.mkdir()
    marks = []

    async def scenario():
        engines = [FakeTenant(0, RANKS, 0.0005), FakeTenant(1, RANKS, 0.0005)]
        link_gpu(engines)
        original = trace_runner.run_trace

        async def fast_trace(engines, controller, trace, prompts, policy, root, trace_args):
            trace_args.poll_s, trace_args.blocked_recheck_s, trace_args.plan_retry_s = .005, .05, .05
            trace_args.monitor_s = .02
            return await original(engines, controller, trace, prompts, policy, root, trace_args)

        monkeypatch.setattr(trace_runner, "run_trace", fast_trace)
        await main.run_phases(engines, args, bank, None, lambda *items, **extra: marks.append(items), verify=False)
        return engines

    engines = asyncio.run(scenario())
    for policy in ("no_move", "ca_full"):
        assert marks.index((f"ballast:{policy}", "applied")) < marks.index((f"trace:{policy}", "started"))
        assert marks.index((f"ballast:{policy}", "released")) < marks.index((f"trace:{policy}", "passed"))
        summary = json.loads((args.output / "trace" / policy / "summary.json").read_text())
        assert summary["kv_mode"] == "prompt" and summary["status"] == "passed"
        workload = json.loads((args.output / "trace" / policy / "workload.json").read_text())
        assert workload["reserve_extra_tokens"] == 128
    assert not any(engine.ballast_held for engine in engines)


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


def segment_trace(tmp_path, name, conv, output_tokens):
    trace = {"duration_s": 0.6, "requests": [
        *[{"request_id": f"conv-{index}", "model": 1, "arrival_s": 0.01 * index, "prompt_tokens": 85,
           "output_tokens": output_tokens, "prompt_key": "public-a"} for index in range(conv)],
        {"request_id": "code-0", "model": 0, "arrival_s": 0.1, "prompt_tokens": 8192, "output_tokens": 64,
         "prompt_key": "swe-a"}]}
    (tmp_path / name).write_text(json.dumps(trace))


def segment_args(tmp_path, segments):
    (tmp_path / "segments.json").write_text(json.dumps({"segments": segments}))
    args = types.SimpleNamespace(output=tmp_path / "out", phases="trace", policies="no_move", num_ranks=RANKS,
                                 budget_margin_gib=0.0, budget_cap_gib=1.2, step_params=None, planner_params=None,
                                 trace=None, segments=tmp_path / "segments.json", seed=1, fairness_threshold_s=0.3,
                                 drain_timeout_s=60, kv_mode="prompt", reserve_extra_tokens=128)
    args.output.mkdir()
    return args


def run_segments(args, bank, monkeypatch, marks):
    async def scenario():
        engines = [FakeTenant(0, RANKS, 0.0005), FakeTenant(1, RANKS, 0.0005)]
        link_gpu(engines)
        original = trace_runner.run_trace

        async def fast_trace(engines, controller, trace, prompts, policy, root, trace_args):
            trace_args.poll_s, trace_args.blocked_recheck_s, trace_args.plan_retry_s = .005, .05, .05
            trace_args.monitor_s = .02
            return await original(engines, controller, trace, prompts, policy, root, trace_args)

        monkeypatch.setattr(trace_runner, "run_trace", fast_trace)
        await main.run_phases(engines, args, bank, None, lambda *items, **extra: marks.append(items), verify=False)
        return engines

    return asyncio.run(scenario())


def test_segments_run_each_trace_with_its_own_kv_cap(tmp_path, monkeypatch):
    segment_trace(tmp_path, "a.json", 8, 150)
    segment_trace(tmp_path, "b.json", 6, 120)
    args = segment_args(tmp_path, [{"label": "a-c12", "trace": "a.json", "budget_cap_gib": 1.2,
                                    "policies": ["no_move", "ca_full"]},
                                   {"label": "b-c20", "trace": "b.json", "budget_cap_gib": 2.0,
                                    "policies": ["no_move", "kvcached_native"]}])
    bank = {"public-a": list(range(85)), "swe-a": list(range(8192))}
    marks = []
    engines = run_segments(args, bank, monkeypatch, marks)
    out = args.output
    order = [name for name, status in marks if name.startswith("trace:") and status == "started"]
    assert order == ["trace:a-c12/no_move", "trace:a-c12/ca_full", "trace:b-c20/no_move", "trace:b-c20/kvcached_native"]
    for label, policy, cap, conv in (("a-c12", "no_move", 1.2, 8), ("a-c12", "ca_full", 1.2, 8),
                                     ("b-c20", "no_move", 2.0, 6), ("b-c20", "kvcached_native", 2.0, 6)):
        summary = json.loads((out / "trace" / label / policy / "summary.json").read_text())
        assert summary["status"] == "passed" and summary["requests"] == conv + 1
        applied = json.loads((out / f"ballast-{label}-{policy}.json").read_text())
        assert applied["cap_gib"] == cap
        assert (out / f"ballast-{label}-{policy}-released.json").exists()
        assert marks.index((f"ballast:{label}/{policy}", "released")) < marks.index((f"trace:{label}/{policy}", "passed"))
    assert ("trace", "passed") in marks
    assert not (out / "trace" / "no_move").exists()
    assert not any(engine.ballast_held for engine in engines)


def test_segments_fail_fast_before_any_trial(tmp_path, monkeypatch):
    segment_trace(tmp_path, "a.json", 4, 100)
    args = segment_args(tmp_path, [{"label": "ok", "trace": "a.json", "budget_cap_gib": 1.2, "policies": ["no_move"]},
                                   {"label": "bad", "trace": "a.json", "budget_cap_gib": 1.2, "policies": ["typo"]}])
    marks = []
    try:
        run_segments(args, {"public-a": list(range(85)), "swe-a": list(range(8192))}, monkeypatch, marks)
        raise AssertionError("an unknown policy in a later segment must fail before the first trial")
    except ValueError as error:
        assert "typo" in str(error)
    assert not any(name.startswith("trace:") for name, _ in marks)
    (tmp_path / "dup.json").write_text(json.dumps({"segments": [
        {"label": "x", "trace": "a.json", "policies": ["no_move"]},
        {"label": "x", "trace": "a.json", "policies": ["no_move"]}]}))
    try:
        main.trace_segments(types.SimpleNamespace(segments=tmp_path / "dup.json", budget_cap_gib=1.2))
        raise AssertionError("duplicate segment labels must be rejected")
    except ValueError as error:
        assert "unique" in str(error)
