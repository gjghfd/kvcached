import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def motivation(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "experiments/dp_opt"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("dp_motivation", root / "motivation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def phase_worker(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "experiments/dp_opt"

    class Event:
        def __init__(self, enable_timing=True):
            pass

        def record(self):
            pass

        def elapsed_time(self, ended):
            return 1.0

    cuda = types.SimpleNamespace(Event=Event, synchronize=lambda device: None)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    monkeypatch.setitem(sys.modules, "worker", types.SimpleNamespace(VerificationWorker=object))
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.distributed", types.ModuleType("vllm.distributed"))
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state",
                        types.SimpleNamespace(get_dp_group=lambda: None))
    spec = importlib.util.spec_from_file_location("dp_phase_worker", root / "motivation_worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = module.MotivationWorker()
    instance.device = "cuda:0"
    instance.parallel_config = types.SimpleNamespace(data_parallel_rank=0)
    instance._motivation_phase_hooks = []
    instance._motivation_phase_events = None
    instance._motivation_phase_pending = {}
    instance._motivation_collective_originals = []
    instance._motivation_collective_calls = None
    return instance, Event


def trace(rank, count, steps=256):
    events = []
    for step in range(steps):
        events.append({
            "sequence": step, "kind": "execute" if count else "dummy",
            "start_ns": step * 100000000 + rank * 1000,
            "end_ns": step * 100000000 + 50000000 + rank * 1000,
            "new_requests": 0, "resumed_requests": 0,
            "requests": [{"id": f"r{rank}-{request}", "tokens": 1, "output_tokens": step + 1,
                          "computed_tokens": 128 + step, "prompt_tokens": 128}
                         for request in range(count)],
            "batch": {"should_ubatch": False, "cudagraph_mode": "NONE",
                      "padded_tokens": count or 1, "unpadded_tokens": count or 1},
        })
    return {"rank": rank, "events": events}


def test_plan_includes_three_paired_repeats_and_hot_rank_swaps(motivation):
    plan = motivation.build_plan()
    assert len(plan) == 30
    assert len({trial["id"] for trial in plan}) == 30
    for repeat in range(3):
        cohort = [trial for trial in plan if trial["repeat"] == repeat]
        assert cohort[0]["role"] == "baseline_start"
        assert cohort[-1]["role"] == "baseline_end"
        assert sum(trial["layout"] == [8, 8] for trial in cohort) == 2
        assert {tuple(trial["layout"]) for trial in cohort} == set(motivation.layouts())


def test_extremes_plan_keeps_paired_baselines(motivation):
    plan = motivation.build_plan(extremes_only=True)
    assert len(plan) == 12
    for repeat in range(3):
        cohort = [trial for trial in plan if trial["repeat"] == repeat]
        assert cohort[0]["role"] == "baseline_start"
        assert cohort[-1]["role"] == "baseline_end"
        assert {tuple(trial["layout"]) for trial in cohort} == {(8, 8), (16, 0), (0, 16)}


@pytest.mark.parametrize("total", [8, 32, 64, 128, 256])
def test_variable_batch_plan_and_capacity(motivation, total):
    plan = motivation.build_plan(extremes_only=True, total=total)
    assert len(plan) == 12
    assert {tuple(trial["layout"]) for trial in plan} == {
        (total // 2, total // 2), (total, 0), (0, total),
    }
    workers = [{"kv_num_blocks": 32768, "max_num_seqs": 256}] * 2
    assert motivation.validate_capacity(workers, 1024, 512, total=total)["required_hot_rank_blocks"] == total * 96
    with pytest.raises(ValueError, match="sequence slots"):
        motivation.validate_capacity([{"kv_num_blocks": 32768, "max_num_seqs": 4}], 1024, 512, total=total)


def test_variable_batch_summary_and_screening_interval(motivation):
    trials = [{**trial, "stable_window": {"critical_cadence_median_s": 0.1},
               "summary": {"output_tokens_per_second": 640}, "stable_client_itl_s": {"p95": 0.11}}
              for trial in motivation.build_plan(repeats=1, extremes_only=True, total=64)]
    summary = motivation.paired_summary(trials)
    assert [row["layout"] for row in summary] == [[32, 32], [64, 0], [0, 64]]
    assert all(row["paired_slowdown_t95_ci"] is None for row in summary)


def test_sweep_requires_both_orientation_confidence_bounds(motivation):
    root = Path(motivation.__file__).parent
    spec = importlib.util.spec_from_file_location("dp_batch_sweep", root / "batch_sweep.py")
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)
    report = {"summary": [
        {"imbalance": 0, "output_tokens_per_second": 100},
        *[{"imbalance": 1, "layout": layout, "paired_slowdown": 1.5,
           "paired_slowdown_t95_ci": [1.45, 1.55], "output_tokens_per_second": 60,
           "cadence_s": 0.15} for layout in ([64, 0], [0, 64])],
    ]}
    metrics = sweep.case_metrics(report)
    assert metrics["minimum_orientation_decode_loss"] == pytest.approx(1/3)
    assert sweep.confirmed_target(metrics)
    metrics["orientations"][0]["steady_decode_loss_t95_ci"] = [0.29, 0.4]
    assert not sweep.confirmed_target(metrics)
    metrics["orientations"][0]["steady_decode_loss_t95_ci"] = None
    assert not sweep.confirmed_target(metrics)


def test_resume_reuses_only_verified_matching_screening(motivation, tmp_path):
    root = Path(motivation.__file__).parent
    spec = importlib.util.spec_from_file_location("dp_resume_sweep", root / "batch_sweep.py")
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)
    case = {"name": "batch-16-ctx1024", "requests": 16, "input_tokens": 1024,
            "output_tokens": 512, "family": "batch"}
    workers = [{"dp_rank": rank, "gpu_name": "A800", "checkpoint": "qwen"}
               for rank in (0, 1)]
    path = tmp_path / "motivation/cases/screen-batch-16-ctx1024/results.json"
    assert sweep.resumed_screen(case, tmp_path, workers) is None
    path.parent.mkdir(parents=True)
    result = {"status": "failed"}
    path.write_text(json.dumps(result))
    assert sweep.resumed_screen(case, tmp_path, workers) is None
    result = {"status": "passed", "workers": workers,
              "protocol": {"requests": 16, "input_tokens": 1024, "output_tokens": 512,
                           "repeats": 1, "screening": True, "extremes_only": True,
                           "trim_steps_each_end": 32, "minimum_stable_steps": 128},
              "summary": [{"imbalance": 0, "output_tokens_per_second": 100},
                          *[{"imbalance": 1, "layout": layout, "paired_slowdown": 1.1,
                             "paired_slowdown_t95_ci": None, "output_tokens_per_second": 90,
                             "cadence_s": 0.1} for layout in ([16, 0], [0, 16])]]}
    path.write_text(json.dumps(result))
    resumed = sweep.resumed_screen(case, tmp_path, workers)
    assert resumed["selection_only"]
    assert not sweep.confirmed_target(resumed)
    result["protocol"]["requests"] = 32
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="protocol"):
        sweep.resumed_screen(case, tmp_path, workers)
    result["protocol"]["requests"] = 16
    result["workers"][0]["gpu_name"] = "H200"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="hardware/configuration"):
        sweep.resumed_screen(case, tmp_path, [{"dp_rank": rank, "gpu_name": "A800",
                                              "checkpoint": "qwen"} for rank in (0, 1)])


def test_long_context_capacity_checks_all_hot_requests(motivation):
    workers = [{"kv_num_blocks": 16384}, {"kv_num_blocks": 16384}]
    capacity = motivation.validate_capacity(workers, 8192, 1024)
    assert capacity["required_hot_rank_blocks"] == 9216
    with pytest.raises(ValueError, match="Insufficient"):
        motivation.validate_capacity([{"kv_num_blocks": 1365}] * 2, 8192, 1024)
    with pytest.raises(ValueError, match="positive"):
        motivation.validate_capacity(workers, 0, 1024)
    with pytest.raises(ValueError, match="block size"):
        motivation.validate_capacity([{"kv_num_blocks": 16384, "kv_block_size": 32}], 8192, 1024)
    with pytest.raises(ValueError, match="max_model_len"):
        motivation.validate_capacity([{"kv_num_blocks": 16384, "max_model_len": 4096}], 8192, 1024)


@pytest.mark.parametrize("counts", [(8, 8), (12, 4), (16, 0), (0, 16)])
def test_stable_window_verifies_populations_and_idle_rank_dummy(motivation, counts):
    window = motivation.stable_window([trace(rank, count) for rank, count in enumerate(counts)], counts)
    assert all(rank["steps"] >= 128 for rank in window["ranks"].values())
    assert window["critical_cadence_median_s"] == pytest.approx(0.1)
    for rank, count in enumerate(counts):
        assert bool(window["ranks"][str(rank)]["dummy_steps"]) == (count == 0)


def test_population_change_in_middle_rejects_window(motivation):
    traces = [trace(rank, 8) for rank in (0, 1)]
    traces[0]["events"][100]["requests"].pop()
    with pytest.raises(RuntimeError, match="changing populations"):
        motivation.stable_window(traces, [8, 8])


def test_short_trace_is_not_reported_as_stable(motivation):
    with pytest.raises(RuntimeError, match="Insufficient"):
        motivation.stable_window([trace(rank, 8, 150) for rank in (0, 1)], [8, 8])


def test_metrics_include_both_rank_population_and_kv(motivation):
    lines = []
    for rank in (0, 1):
        for name in ("vllm:num_requests_running", "vllm:num_requests_waiting",
                     "vllm:num_preemptions_total", "vllm:kv_cache_usage_perc"):
            lines.append(f'{name}{{model_name="qwen",engine="{rank}"}} 0')
    assert set(motivation.parse_metrics("\n".join(lines), "qwen")) == {"0", "1"}
    with pytest.raises(RuntimeError, match="Missing"):
        motivation.parse_metrics("\n".join(lines[:-1]), "qwen")


def test_preemption_or_chunked_prefill_never_counts_as_stable(motivation):
    events = trace(0, 8)["events"]
    ids = {request["id"] for request in events[0]["requests"]}
    events[0]["resumed_requests"] = 1
    assert not motivation.valid_decode(events[0], ids)
    events[1]["requests"][0]["tokens"] = 2
    assert not motivation.valid_decode(events[1], ids)
    events[2]["requests"][0]["output_tokens"] = 0
    assert not motivation.valid_decode(events[2], ids)


def test_paired_summary_uses_same_repeat_balanced_controls(motivation):
    trials = []
    for repeat in range(3):
        for layout in motivation.layouts():
            slowdown = 1 + abs(layout[0] - layout[1]) / 32
            trials.append({"repeat": repeat, "layout": list(layout),
                           "stable_window": {"critical_cadence_median_s": 0.1 * slowdown},
                           "summary": {"output_tokens_per_second": 160 / slowdown},
                           "stable_client_itl_s": {"p95": 0.11 * slowdown}})
        trials.append(dict(trials[repeat * 10]))
    summary = motivation.paired_summary(trials)
    for result in summary:
        expected = 1 + result["imbalance"] / 2
        assert result["paired_slowdown"] == pytest.approx(expected)
        assert result["paired_slowdown_t95_ci"] == pytest.approx([expected, expected])


def test_missing_batch_evidence_invalidates_trace(motivation):
    traces = [trace(rank, 8) for rank in (0, 1)]
    for event in traces[0]["events"]:
        event["batch"] = None
    with pytest.raises(RuntimeError, match="Missing batch"):
        motivation.stable_window(traces, [8, 8])


def test_graph_population_and_padding_are_explicitly_validated(motivation):
    traces = [trace(rank, count) for rank, count in enumerate((16, 0))]
    for item in traces:
        for event in item["events"]:
            event["batch"]["cudagraph_mode"] = "FULL"
            event["batch"]["padded_tokens"] = 16
    with pytest.raises(RuntimeError, match="Expected NONE"):
        motivation.stable_window(traces, [16, 0])
    window = motivation.stable_window(traces, [16, 0], expected_cudagraph_mode="FULL")
    assert window["ranks"]["1"]["unpadded_tokens"] == [1]
    assert window["ranks"]["1"]["padded_tokens"] == [16]


def test_eager_padding_is_rejected(motivation):
    traces = [trace(rank, count) for rank, count in enumerate((16, 0))]
    for event in traces[1]["events"]:
        event["batch"]["padded_tokens"] = 16
    with pytest.raises(RuntimeError, match="silently pad"):
        motivation.stable_window(traces, [16, 0])


def test_graph_padding_must_match_the_registered_batch(motivation):
    traces = [trace(rank, 8) for rank in (0, 1)]
    for item in traces:
        for event in item["events"]:
            event["batch"]["cudagraph_mode"] = "FULL"
            event["batch"]["padded_tokens"] = 16
    with pytest.raises(RuntimeError, match="declared max-rank padding"):
        motivation.stable_window(traces, [8, 8], expected_cudagraph_mode="FULL")


def test_summary_writes_csv_and_valid_svg(motivation, tmp_path):
    import xml.etree.ElementTree as element_tree

    summary = [{"layout": [8, 8], "imbalance": 0, "paired_slowdown": 1,
                "paired_slowdown_t95_ci": [0.99, 1.01], "cadence_s": 0.1,
                "output_tokens_per_second": 160, "stable_client_itl_p95_s": 0.11}]
    motivation.write_summary(tmp_path, summary)
    assert (tmp_path / "summary.csv").read_text().splitlines()[0].startswith("rank0_requests")
    element_tree.parse(tmp_path / "motivation.svg")


def test_extremes_summary_does_not_require_intermediate_layouts(motivation):
    trials = []
    for trial in motivation.build_plan(extremes_only=True):
        trials.append({**trial, "stable_window": {"critical_cadence_median_s": 0.1},
                       "summary": {"output_tokens_per_second": 160},
                       "stable_client_itl_s": {"p95": 0.11}})
    summary = motivation.paired_summary(trials)
    assert [result["layout"] for result in summary] == [[8, 8], [16, 0], [0, 16]]


def test_phase_scaling_separates_linear_share_from_measured_batch_delta(motivation):
    diagnostics = []
    for layout in ([8, 8], [16, 0], [0, 16]):
        durations = {"attention": 10 if layout == [8, 8] else 12,
                     "attention_core": 2, "moe": 80, "worker_execute": 100}
        diagnostics.append({"layout": layout, "phase_profile": {
            str(rank): {"ranges_ms": {phase: {"mean": value} for phase, value in durations.items()}}
            for rank in (0, 1)
        }})
    summary = motivation.phase_scaling(diagnostics)
    assert [result["hot_rank"] for result in summary] == [0, 1]
    assert all(result["range_mean_ratio_skew_to_balanced"]["attention"] == pytest.approx(1.2)
               for result in summary)
    assert all(result["linear_half_share_prediction"] == pytest.approx(0.06) for result in summary)
    assert all(result["measured_attention_delta_fraction"] == pytest.approx(0.02) for result in summary)


def test_phase_events_resolve_only_after_trial_and_hooks_are_removed(phase_worker, monkeypatch, tmp_path):
    instance, event = phase_worker
    removed = []
    monkeypatch.setenv("DP_OPT_EVIDENCE_ROOT", str(tmp_path))
    instance._motivation_phase_hooks = [types.SimpleNamespace(remove=lambda: removed.append(True))]
    cuda_events = [(phase, event(), event())
                   for phase in ("attention", "attention_core", "moe", "dispatch", "combine")
                   for layer in range(48)]
    cuda_events.append(("worker_execute", event(), event()))
    instance._motivation_events = [{"phase_cuda_events": cuda_events}]
    instance.motivation_control("stop", "test-trial", "long")
    result = json.loads((tmp_path / "motivation/cases/long/traces/test-trial-dp0.json").read_text())
    assert removed == [True]
    assert instance._motivation_phase_hooks == []
    assert result["events"][0]["phase_cuda_ms"] == {
        "attention": 48.0, "attention_core": 48.0, "moe": 48.0,
        "dispatch": 48.0, "combine": 48.0, "worker_execute": 1.0,
    }
    assert "phase_cuda_events" not in result["events"][0]


def test_unprofiled_trial_never_installs_phase_hooks(phase_worker):
    instance, event = phase_worker
    instance.install_phase_hooks = lambda: pytest.fail("Unprofiled trial installed hooks")
    instance.motivation_control("start", "ordinary")
    assert not instance._motivation_profile
    assert instance.begin_phase_sample() is None
    assert instance.end_phase_sample(None) == []


def test_incomplete_phase_layer_coverage_is_rejected(phase_worker, monkeypatch, tmp_path):
    instance, event = phase_worker
    monkeypatch.setenv("DP_OPT_EVIDENCE_ROOT", str(tmp_path))
    instance._motivation_events = [{"phase_cuda_events": [("attention", event(), event())]}]
    with pytest.raises(RuntimeError, match="48 model layers"):
        instance.motivation_control("stop", "incomplete")
