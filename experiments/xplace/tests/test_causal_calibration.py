import asyncio
import copy
import json
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "runtime"))

import causal_calibration as calibration
import causal_collection as collection
from controller import PlacementController
from cost_model import default_dp2_params
from fake_vllm import FakeTenant, install_fake_vllm
import main
import trace_runner


@pytest.fixture(scope="module")
def bank_file(tmp_path_factory):
    root = tmp_path_factory.mktemp("causal-bank")
    prompts = {f"swe-{index:02}": dict(prompt_tokens=100000, token_ids=[1] * 100000) for index in range(10)}
    prompts.update({f"public-{index:03}": dict(prompt_tokens=8, token_ids=[2] * 8,
                                              original_output_tokens=1000 + index) for index in range(256)})
    target = root / "bank.json"
    target.write_text(json.dumps(dict(prompts=prompts)))
    return target


@pytest.fixture
def plan(bank_file):
    return calibration.calibration_plan(bank_file)


def cli_args(tmp_path, bank_file, plan):
    filename = tmp_path / "plan.json"
    filename.write_text(json.dumps(plan))
    return ["--checkpoint", "fake", "--output", str(tmp_path / "out"), "--prompt-bank", str(bank_file),
            "--phases", "dataplane,causal", "--num-ranks", "2", "--tensor-parallel-size", "2",
            "--enable-tp-migration", "--kv-mode", "prompt", "--causal-config", str(filename),
            "--causal-mode", "collect", "--reserve-extra-tokens", "256", "--max-inflight", "1", "16",
            "--drain-timeout-s", "210", "--seed", "20261110"]


def test_plan_preserves_bank_lengths_and_independent_partition(plan):
    assert [len(case["requests"]) for case in plan["cases"]] == [5, 17, 5, 17]
    keys = {request["prompt_key"] for case in plan["cases"] for request in case["requests"]}
    assert not keys.intersection(plan["holdout"]["code_keys"] + plan["holdout"]["conv_keys"])
    for case in plan["cases"]:
        for request in case["requests"]:
            assert request["output_tokens"] == (64 if request["model"] == 0
                                                else 1000 + int(request["prompt_key"].split("-")[1]))
    assert plan["reference"]["k"] == 16 and plan["reference"]["queued_included"]
    assert not plan["reference"]["output_length_is_feature"]
    assert plan["max_inflight"] == [1, 16] and plan["reserve_extra_tokens"] == 256
    assert plan["holdout"]["relative_rmse_limit"] == 0.15
    assert plan["holdout"]["max_relative_error_limit"] == 0.30


def test_revised_resources_are_fixed_without_changing_workload(plan):
    assert plan["schema"] == 2
    assert plan["paired_budget_gib"] == [38.0, 39.75]
    assert plan["capacity_record_schema"] == 1
    assert plan["paired_capacity_basis"] == "r6_r7_raw_free_lower_bound_minus_512MiB"
    assert plan["collection"]["max_bytes"] == 512 * 1024 ** 2
    assert plan["collection"]["max_events"] == 1_000_000
    assert plan["case_window_s"] == 240 and plan["drain_timeout_s"] == 210
    assert plan["budget_margin_gib"] == 3 and plan["reserve_extra_tokens"] == 256
    assert plan["collection"]["period_s"] == 0.1 and plan["collection"]["deadline_s"] == 0.08


def test_natural_budget_drift_cannot_change_the_paired_budget(plan):
    budgets = [dict(natural_budget_gib=values, budget_gib=values, margin_gib=3.0)
               for values in ([38.75, 40.25], [38.75, 40.5])]
    bound = [calibration.bind_paired_budget(budget, plan) for budget in budgets]
    assert [item["budget_gib"] for item in bound] == [[38.0, 39.75]] * 2
    assert all(item["paired_budget_status"] == "verified" for item in bound)
    assert budgets[1]["budget_gib"] == [38.75, 40.5]


@pytest.mark.parametrize("natural", [[37.75, 40.25], [38.75, 39.5]])
def test_insufficient_natural_budget_is_not_silently_clipped(plan, natural):
    record = calibration.bind_paired_budget(dict(natural_budget_gib=natural, budget_gib=natural,
                                                margin_gib=3.0), plan)
    assert record["paired_budget_status"] == "unavailable"
    assert record["required_budget_gib"] == [38.0, 39.75]


@pytest.mark.parametrize("natural", [[float("nan"), 42], [41], [True, 42], [41, float("inf")]])
def test_invalid_capacity_evidence_rejects(plan, natural):
    with pytest.raises(ValueError, match="budget"):
        calibration.bind_paired_budget(dict(natural_budget_gib=natural, budget_gib=natural, margin_gib=3.0), plan)


@pytest.mark.parametrize("field,value", [("policy", "ca_full"), ("reserve_extra_tokens", 0),
                                         ("max_inflight", [2, 16]), ("case_window_s", 300)])
def test_changed_plan_rejects_before_execution(tmp_path, bank_file, plan, field, value):
    plan[field] = value
    filename = tmp_path / "plan.json"
    filename.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="differs"):
        calibration.load_plan(filename, bank_file)


def test_output_length_tampering_rejects(tmp_path, bank_file, plan):
    plan["cases"][0]["requests"][0]["output_tokens"] *= 4
    with pytest.raises(SystemExit):
        main.parse_args(cli_args(tmp_path, bank_file, plan))


def test_cli_accepts_only_training_calibration(tmp_path, bank_file, plan):
    parsed = main.parse_args(cli_args(tmp_path, bank_file, plan))
    assert parsed.phases == "dataplane,causal" and parsed.causal_mode == "collect"


@pytest.mark.parametrize("mode", calibration.MODES)
def test_main_configures_both_engines_before_start_and_shuts_down(tmp_path, monkeypatch, bank_file, plan, mode):
    import engine
    import causal_deployment

    monkeypatch.setattr(causal_deployment, "frontend_attestation", lambda: {"status": "fake_cpu_test"})

    args = main.parse_args(cli_args(tmp_path, bank_file, plan) + ["--causal-mode", mode])
    for name in ("XPLACE_NESTED_TIMING", "XPLACE_LIVE_OBSERVATION", "XPLACE_BATCH_LINK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XPLACE_JOB_STARTED_AT", str(time.time()))
    monkeypatch.setenv("DP_OPT_TRANSFER_ROOT", str(tmp_path))
    events = []

    class Engine:
        @classmethod
        def from_engine_args(cls, configuration):
            events.append(("boot", calibration.os.environ["DP_OPT_SERVICE"],
                           calibration.os.environ["XPLACE_LIVE_OBSERVATION"],
                           calibration.os.environ["XPLACE_BATCH_LINK"]))
            return cls()

        def shutdown(self):
            events.append(("shutdown",))

    module = ModuleType("vllm.v1.engine.async_llm")
    module.AsyncLLM = Engine
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.async_llm", module)
    monkeypatch.setattr(engine, "engine_arguments", lambda *args, **kwargs: kwargs)

    async def phases(engines, configuration, prompts, bank_sha, mark):
        assert len(engines) == 2 and configuration.causal_plan == plan
        assert bank_sha == plan["prompt_bank_sha256"] and len(prompts) == 266
        assert configuration.timing_deadline > time.monotonic()
        events.append(("phases",))

    monkeypatch.setattr(main, "run_phases", phases)
    asyncio.run(main.run(args))
    assert events[:2] == [("boot", f"instance-{index}", "0" if mode == "off" else "1",
                           "0" if mode == "off" else "1") for index in (1, 2)]
    assert events[2:] == [("phases",), ("shutdown",), ("shutdown",)]


@pytest.mark.parametrize("extra", [["--phases", "causal"], ["--phases", "dataplane,causal,fit"],
                                   ["--num-ranks", "4"], ["--kv-mode", "full"],
                                   ["--budget-cap-gib", "16"], ["--budget-margin-gib", "1"],
                                   ["--reserve-extra-tokens", "0"], ["--max-inflight", "0", "0"],
                                   ["--drain-timeout-s", "5400"], ["--seed", "1"],
                                   ["--max-model-len", "65536"], ["--cache-bytes", "1024"]])
def test_cli_rejects_unsafe_phase_or_capacity_changes(tmp_path, bank_file, plan, extra):
    with pytest.raises(SystemExit):
        main.parse_args(cli_args(tmp_path, bank_file, plan) + extra)


@pytest.mark.parametrize("mode", calibration.MODES)
def test_instrumentation_mode_applies_before_engine_boot(tmp_path, monkeypatch, mode):
    for name in ("XPLACE_NESTED_TIMING", "XPLACE_TIMING_ALLOWED", "XPLACE_LIVE_OBSERVATION", "XPLACE_BATCH_LINK"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "private"
    calibration.configure_collection(mode, root)
    assert calibration.os.environ["XPLACE_LIVE_OBSERVATION"] == ("0" if mode == "off" else "1")
    assert calibration.os.environ["XPLACE_BATCH_LINK"] == ("0" if mode == "off" else "1")
    assert root.exists() == (mode != "off")
    if root.exists():
        assert root.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("name", ["XPLACE_NESTED_TIMING", "XPLACE_TIMING_ALLOWED"])
def test_legacy_timing_is_not_combined(tmp_path, monkeypatch, name):
    monkeypatch.setenv(name, "1")
    with pytest.raises(ValueError, match="stop-export"):
        calibration.configure_collection("collect", tmp_path / "root")


def patch_collection(monkeypatch):
    monkeypatch.setattr(calibration, "host_clock_id", lambda: "cpu-fixture")
    monkeypatch.setattr(collection, "host_clock_id", lambda: "cpu-fixture")

    async def missing(*args):
        await asyncio.sleep(0)
        return {"status": "incomplete"}

    monkeypatch.setattr(collection, "collect_batch_links", missing)
    return SimpleNamespace(num_ranks=2, tensor_parallel_size=2, prediction_observations=missing)


@pytest.mark.parametrize("mode", calibration.MODES)
def test_matrix_persists_all_cases_without_accepting_prediction(tmp_path, monkeypatch, plan, mode):
    controller = patch_collection(monkeypatch)
    calls = []

    async def trace(engines, control, case, prompts, policy, root, arguments, receipt_observer=None):
        assert policy == "no_move" and arguments.kv_mode == "prompt"
        assert arguments.reserve_extra_tokens == 256 and arguments.max_inflight == [1, 16]
        assert (receipt_observer is not None) == (mode == "collect")
        calls.append(case["case"])
        if receipt_observer:
            assert receipt_observer.register_arrival("nm-test", 1, 8, time.monotonic())
            await asyncio.sleep(0.002)
        return {"status": "censored"}

    monkeypatch.setattr(trace_runner, "run_trace", trace)
    root = tmp_path / "pilot"
    result = asyncio.run(calibration.run_causal_calibration([], controller, root, {}, plan, mode, tmp_path,
                                                           time.monotonic() + 4000))
    assert len(calls) == 4 and len(result["completed"]) == 4 and result["status"] == "collected"
    assert not result["prediction_gate_passed"] and not result["coverage_accepted"]
    assert len(list(root.glob("*/usage.json"))) == 4
    reports = list(root.glob("*/collection.json"))
    assert len(reports) == (4 if mode == "collect" else 0)
    for report in reports:
        data = json.loads(report.read_text())
        assert data["stopped"] and data["task_ended"] and data["requests"][0]["submitted_at"] is None
    with pytest.raises(FileExistsError):
        asyncio.run(calibration.run_causal_calibration([], controller, root, {}, plan, mode, tmp_path,
                                                       time.monotonic() + 4000))


@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
def test_matrix_failure_retains_partial_observations_and_remaining_plan(tmp_path, monkeypatch, plan, error_type):
    controller = patch_collection(monkeypatch)

    async def failure(*args, receipt_observer):
        receipt_observer.register_arrival("failed", 0, 8, time.monotonic())
        raise error_type("fixture")

    monkeypatch.setattr(trace_runner, "run_trace", failure)
    root = tmp_path / "failed"
    with pytest.raises(error_type):
        asyncio.run(calibration.run_causal_calibration([], controller, root, {}, plan, "collect", tmp_path,
                                                       time.monotonic() + 4000))
    status = json.loads((root / "status.json").read_text())
    report = json.loads((root / plan["cases"][0]["case"] / "collection.json").read_text())
    assert status["status"] == "failed" and len(status["planned"]) == 4 and not status["completed"]
    assert report["task_ended"] and report["requests"][0]["request_id"] == "failed"


def test_deadline_refuses_to_launch_another_case(tmp_path, monkeypatch, plan):
    controller = patch_collection(monkeypatch)
    with pytest.raises(TimeoutError, match="Insufficient"):
        asyncio.run(calibration.run_causal_calibration([], controller, tmp_path / "deadline", {}, plan,
                                                       "off", tmp_path, time.monotonic() + 20))
    record = json.loads((tmp_path / "deadline/status.json").read_text())
    assert len(record["planned"]) == 4 and not record["completed"]


def test_holdout_cannot_be_relabelled_into_executable_training(tmp_path, monkeypatch, plan):
    controller = patch_collection(monkeypatch)
    plan["cases"][0]["split"] = "validation"
    with pytest.raises(ValueError, match="cannot execute"):
        asyncio.run(calibration.run_causal_calibration([], controller, tmp_path / "holdout", {}, plan,
                                                       "collect", tmp_path, time.monotonic() + 4000))
    assert not (tmp_path / "holdout").exists()


def test_causal_phase_dispatch_follows_same_job_dataplane(tmp_path, monkeypatch, plan):
    import calibrate

    sampling_fixture(monkeypatch)
    calls = []

    async def dataplane(*args):
        calls.append("dataplane")
        return {"migrations": []}

    async def causal(engines, controller, root, prompts, runtime_plan, mode, live_root, deadline):
        calls.append("causal")
        assert controller.tensor_parallel_size == controller.num_ranks == 2
        assert controller.budgets == plan["paired_budget_gib"]
        assert runtime_plan == plan and mode == "collect" and deadline > time.monotonic()
        return {"status": "collected", "completed": [1, 2, 3, 4]}

    monkeypatch.setattr(calibrate, "run_dataplane", dataplane)
    monkeypatch.setattr(calibration, "run_causal_calibration", causal)

    async def budget(*args):
        return dict(natural_budget_gib=[41, 42], budget_gib=[41, 42], margin_gib=3.0)

    monkeypatch.setattr(main, "measure_budget", budget)
    args = SimpleNamespace(output=tmp_path, phases="dataplane,causal", num_ranks=2, tensor_parallel_size=2,
             budget_margin_gib=3.0, budget_cap_gib=None, step_params=None, causal_plan=plan, causal_mode="collect",
             causal_live_root=tmp_path, timing_deadline=time.monotonic() + 3000)
    marks = []

    async def scenario():
        engines = [CumulativeTenant(model, 2, 0.001) for model in (0, 1)]
        await main.run_phases(engines, args, {}, plan["prompt_bank_sha256"],
                              lambda *args, **kwargs: marks.append((args, kwargs)), verify=False)

    asyncio.run(scenario())
    assert calls == ["dataplane", "causal"]
    assert marks[-1][0] == ("causal", "collected") and not marks[-1][1]["prediction_gate_passed"]


def test_insufficient_paired_capacity_fails_before_warmup(tmp_path, monkeypatch, plan):
    import engine

    async def unexpected_stream(*args, **kwargs):
        pytest.fail("Insufficient paired capacity must stop before warmup")

    monkeypatch.setattr(engine, "stream", unexpected_stream)
    args = SimpleNamespace(output=tmp_path, phases="dataplane,causal", num_ranks=2, tensor_parallel_size=2,
                           budget_margin_gib=3.0, budget_cap_gib=None, causal_plan=plan)
    engines = [CumulativeTenant(model, 2, 0.001) for model in (0, 1)]
    with pytest.raises(ValueError, match="paired budget"):
        asyncio.run(main.run_phases(engines, args, {}, plan["prompt_bank_sha256"],
                                   lambda *args, **kwargs: pytest.fail("No phase should pass"), verify=False))
    budget = json.loads((tmp_path / "budget.json").read_text())
    assert budget["natural_budget_gib"] == [37, 37]
    assert budget["required_budget_gib"] == [38.0, 39.75]
    assert budget["paired_budget_status"] == "unavailable" and budget["budget_gib"] is None


class CumulativeTenant(FakeTenant):
    def collective_rpc(self, rank, method, timeout, args, kwargs):
        result = super().collective_rpc(rank, method, timeout, args, kwargs)
        if method == "placement_memory_snapshot":
            return [dict(result[0], rank=rank, tp_rank=lane, tp_size=2, gpu_uuid=f"cpu-{rank}-{lane}")
                    for lane in range(2)]
        return result

    async def generate(self, prompt, params, request_id, data_parallel_rank=None):
        async for item in super().generate(prompt, params, request_id, data_parallel_rank):
            completion = item.outputs[0]
            item.request_id = request_id
            item.finished = len(completion.token_ids) == params.max_tokens
            completion.index = 0
            completion.finish_reason = "length" if item.finished else None
            completion.stop_reason = None
            yield item


def tiny_arguments(**changes):
    fields = dict(planner_params={}, step_params=default_dp2_params(), trace_sha256="cpu-fixture", seed=1,
                  fairness_threshold_s=30.0, blocked_recheck_s=0.001, plan_retry_s=0.001, poll_s=0.001,
                  return_period_s=10.0, return_cooldown_s=10.0, monitor_s=0.01, drain_timeout_s=3.0,
                  kv_mode="prompt", reserve_extra_tokens=256, max_inflight=[1, 1])
    fields.update(changes)
    return SimpleNamespace(**fields)


def sampling_fixture(monkeypatch):
    install_fake_vllm()
    original = sys.modules["vllm"].SamplingParams

    class CumulativeSampling(original):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.output_kind = SimpleNamespace(name="CUMULATIVE")

    monkeypatch.setattr(sys.modules["vllm"], "SamplingParams", CumulativeSampling)


def test_censor_deadline_covers_full_window_after_trace_origin(tmp_path, monkeypatch):
    sampling_fixture(monkeypatch)
    trace = dict(duration_s=0.05, requests=[dict(request_id="conv-0", model=1, arrival_s=0.0,
                 prompt_tokens=8, output_tokens=4096, prompt_key="prompt")])
    arguments = tiny_arguments(drain_timeout_s=0.05)

    async def scenario():
        engines = [CumulativeTenant(model, 2, 0.002) for model in (0, 1)]
        controller = PlacementController(engines, 2, [40.0, 40.0])
        controller.tensor_parallel_size = 2
        return await trace_runner.run_trace(engines, controller, trace, {"prompt": [2] * 8}, "no_move",
                                            tmp_path, arguments)

    summary = asyncio.run(scenario())
    window = trace["duration_s"] + arguments.drain_timeout_s
    assert summary["status"] == "censored" and not summary["errors"]
    assert window <= summary["censored_at_s"] <= window + 0.5
    incomplete = json.loads((tmp_path / "incomplete.json").read_text())
    assert len(incomplete) == 1 and incomplete[0]["output_tokens"] > 0
    assert incomplete[0]["target_output_tokens"] == 4096


def test_real_shared_queue_hook_records_waiters_and_preserves_outputs(tmp_path, monkeypatch):
    missing = patch_collection(monkeypatch)
    sampling_fixture(monkeypatch)

    async def scenario():
        trace = dict(duration_s=0.0, requests=[dict(request_id=f"req-{index}", model=1, arrival_s=0.0,
                     prompt_tokens=8, output_tokens=16, prompt_key="prompt") for index in range(3)])
        outputs = []
        for enabled in (False, True):
            engines = [CumulativeTenant(model, 2, 0.001) for model in (0, 1)]
            controller = PlacementController(engines, 2, [40.0, 40.0])
            controller.tensor_parallel_size = 2
            controller.prediction_observations = missing.prediction_observations
            observer = collection.CausalCollection(case_id="queue", clock_id="cpu-fixture", period_s=0.005,
                         deadline_s=0.01, max_events=10000, max_bytes=8000000, max_requests=4) if enabled else None
            if observer:
                await observer.start(controller, tmp_path)
            root = tmp_path / str(enabled)
            summary = await trace_runner.run_trace(engines, controller, trace, {"prompt": [2] * 8}, "no_move",
                                                   root, tiny_arguments(), receipt_observer=observer)
            assert summary["status"] == "passed" and summary["moves"] == summary["holds"] == 0
            outputs.append(json.loads((root / "requests.json").read_text()))
        report = observer.report()
        samples = [event for event in report["events"] if event["kind"] == "sample"]
        assert any(any(item["submitted_at"] is None for item in event["cohort_at_completion"])
                   for event in samples if event["cohort_at_completion"])
        assert len(report["trace_cutoff"]["cohort"]) == 3 and report["task_ended"]
        assert report["counts"]["invalid"] == 0
        for baseline, measured in zip(*outputs):
            for key in ("token_sha256", "output_tokens", "prompt_tokens", "rank_initial"):
                assert baseline[key] == measured[key]

    asyncio.run(scenario())


def test_cutoff_survives_event_overflow_and_does_not_decode_a_report(monkeypatch):
    patch_collection(monkeypatch)
    observer = collection.CausalCollection(case_id="overflow", clock_id="cpu-fixture", period_s=0.1,
                 deadline_s=0.01, max_events=1, max_bytes=10000, max_requests=1)
    observer.register_arrival("queued", 1, 8, time.monotonic())
    observer.mark_trace_cutoff(time.monotonic(), None)
    original = copy.deepcopy(observer.trace_cutoff)
    observer.mark_trace_cutoff(time.monotonic(), 123)
    monkeypatch.setattr(observer, "report", lambda: pytest.fail("report decoding requested"))
    assert asyncio.run(observer.stop(include_report=False)) is None
    assert observer.trace_cutoff == original and observer.dropped > 0


def test_cancelled_trace_closes_waiters_sampler_and_streams(tmp_path, monkeypatch):
    missing = patch_collection(monkeypatch)
    sampling_fixture(monkeypatch)

    async def scenario():
        engines = [CumulativeTenant(model, 2, 0.001) for model in (0, 1)]
        controller = PlacementController(engines, 2, [40.0, 40.0])
        controller.tensor_parallel_size = 2
        controller.prediction_observations = missing.prediction_observations
        observer = collection.CausalCollection(case_id="cancel", clock_id="cpu-fixture", period_s=0.005,
                     deadline_s=0.01, max_events=10000, max_bytes=8000000, max_requests=4)
        await observer.start(controller, tmp_path)
        trace = dict(duration_s=0.0, requests=[dict(request_id="slow", model=1, arrival_s=0.0,
                     prompt_tokens=8, output_tokens=100000, prompt_key="prompt")])
        task = asyncio.create_task(trace_runner.run_trace(engines, controller, trace, {"prompt": [2] * 8},
                           "no_move", tmp_path / "cancel", tiny_arguments(), receipt_observer=observer))
        deadline = time.monotonic() + 3
        while not any(item["count"] > 0 for item in observer.requests.values()):
            if task.done():
                await task
            assert time.monotonic() < deadline
            await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert observer.closed and observer.task.done() and observer.trace_cutoff is not None
        assert not engines[1].requests
        await asyncio.sleep(0)
        remaining = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        assert all(task.get_coro().__qualname__ == "FakeTenant.run" for task in remaining)

    asyncio.run(scenario())
