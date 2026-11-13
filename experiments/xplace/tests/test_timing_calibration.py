import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

import main
import timing_calibration as calibration
from measured_cost import interference_key, repeated_interference_plan


class Controller:
    num_ranks = 2
    tensor_parallel_size = 2

    async def summaries(self):
        return [[{"requests": [], "preemptions_total": 0} for _ in range(2)] for _ in range(2)]

    async def call(self, model, rank, method, command, case):
        assert method == "placement_timing"
        if command == "ready":
            return {"idle": True}
        return {"command": command, "engine": {"model": model, "rank": rank}, "workers": [{}, {}]}


async def no_wait(*arguments):
    await asyncio.sleep(0)


def test_fixed_matrix_preserves_all_pairs_and_original_inputs():
    plan = calibration.observation_plan()
    assert len(plan["cases"]) == 24 and len({case["case"] for case in plan["cases"]}) == 24
    assert sum(sum(case["layout"]) + 1 for case in plan["cases"]) == plan["planned_requests"] == 336
    assert plan["raw_contract"] == repeated_interference_plan(2, 23)
    assert (plan["code_output_tokens"], plan["conv_output_tokens"], plan["cadence_stride"]) == (64, 3000, 1)
    assert plan["background_settle_s"] == 3
    for repeat in range(3):
        rows = [case for case in plan["cases"] if case["repeat"] == repeat]
        assert [case["layout"] for case in rows] == [[count, count] for count in (0, 2, 8, 16) for _ in range(2)]
        assert [case["condition"] for case in rows[:2]] == (["timing_on", "timing_off"] if repeat == 1 else
                                                          ["timing_off", "timing_on"])


@pytest.mark.parametrize("started", [None, "nan", "inf", "not-a-time", "0"])
def test_deadline_rejects_missing_or_invalid_sbatch_time(monkeypatch, started):
    if started is None:
        monkeypatch.delenv("XPLACE_JOB_STARTED_AT", raising=False)
    else:
        monkeypatch.setenv("XPLACE_JOB_STARTED_AT", started)
    with pytest.raises(ValueError):
        calibration.allocation_deadline()


def test_deadline_includes_time_before_python_startup(monkeypatch):
    monkeypatch.setenv("XPLACE_JOB_STARTED_AT", str(time.time() - 300))
    remaining = calibration.allocation_deadline() - time.monotonic()
    assert 2999 < remaining <= 3000


def test_cli_allows_only_explicit_isolated_observability_phase(tmp_path):
    required = ["--checkpoint", "model", "--output", str(tmp_path), "--prompt-bank", "bank.json",
                "--num-ranks", "2", "--tensor-parallel-size", "2", "--enable-tp-migration"]
    assert main.parse_args(required + ["--phases", "dataplane,observability"]).phases == "dataplane,observability"
    for phases in ("observability", "observability,dataplane", "dataplane,observability,fit"):
        with pytest.raises(SystemExit):
            main.parse_args(required + ["--phases", phases])


def fake_matrix(monkeypatch):
    calls = []
    class Frontend:
        def __init__(self, case):
            self.case = case
            self.buffer = SimpleNamespace(active=True)
        async def start(self):
            calls.append(("front_start", self.case))
        async def stop(self):
            assert self.buffer.active
            self.buffer.active = False
            calls.append(("front_stop", self.case))
            return {"case": self.case}
    async def measure(engines, controller, case, code_prompt, root):
        calls.append(("measure", case["case"]))
        row = dict(case, measurement_started_ns=1, measurement_ended_ns=2, case_elapsed_s=1,
                   code_prefill_s=1, code_tpot_s=1, conv_whole_mean_tpot_s=1)
        calibration.save(root / "requests.json", [])
        calibration.save(root / "measurement.json", row)
        return row, []
    def samples(rows, raw, num_ranks, plan):
        assert len(rows) == len(raw) == 12 and num_ranks == 2
        assert Counter(row["repeat"] for row in rows) == {0: 4, 1: 4, 2: 4}
        assert set(raw) == {interference_key(row) for row in rows}
        return ["fixture"]
    monkeypatch.setattr(calibration, "FrontendTiming", Frontend)
    monkeypatch.setattr(calibration, "measure_observation_case", measure)
    monkeypatch.setattr(calibration, "interference_samples", samples)
    monkeypatch.setattr(calibration, "validate_timing", lambda *arguments: {"status": "passed", "fixture": True})
    return calls


def test_complete_matrix_runs_both_conditions_without_promoting_parameters(tmp_path, monkeypatch):
    calls = fake_matrix(monkeypatch)
    output = tmp_path / "observability"
    result = asyncio.run(calibration.run_observability([], Controller(), output, [1] * 101167, time.monotonic() + 30))
    assert result["status"] == "passed" and len(result["completed"]) == 24
    assert result["independent_prediction_gate"] is False
    assert Counter(kind for kind, _ in calls) == {"measure": 24, "front_start": 12, "front_stop": 12}
    assert len(json.loads((output / "paired-overhead.json").read_text())["pairs"]) == 12
    assert not list(output.rglob("step-params.json"))
    with pytest.raises(FileExistsError):
        asyncio.run(calibration.run_observability([], Controller(), output, [1] * 101167, time.monotonic() + 30))


def test_timeout_preserves_all_uncompleted_cases(tmp_path, monkeypatch):
    calls = fake_matrix(monkeypatch)
    output = tmp_path / "observability"
    with pytest.raises(TimeoutError):
        asyncio.run(calibration.run_observability([], Controller(), output, [1] * 101167, time.monotonic() - 1))
    status = json.loads((output / "status.json").read_text())
    assert not calls and status["completed"] == [] and len(status["not_completed"]) == 24


def test_live_deadline_cancels_the_active_case(tmp_path, monkeypatch):
    fake_matrix(monkeypatch)
    cancelled = []
    async def unfinished(*arguments):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)
    monkeypatch.setattr(calibration, "measure_observation_case", unfinished)
    output = tmp_path / "observability"
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(calibration.run_observability([], Controller(), output, [1] * 101167, time.monotonic() + 0.05))
    assert cancelled == [True]
    assert len(json.loads((output / "status.json").read_text())["not_completed"]) == 24


def test_raw_failure_keeps_both_complete_conditions(tmp_path, monkeypatch):
    fake_matrix(monkeypatch)
    attempted = []
    def reject(rows, *arguments):
        attempted.append(rows[0]["condition"])
        raise ValueError("fixture raw rejection")
    monkeypatch.setattr(calibration, "interference_samples", reject)
    output = tmp_path / "observability"
    with pytest.raises(ValueError, match="raw acceptance"):
        asyncio.run(calibration.run_observability([], Controller(), output, [1] * 101167, time.monotonic() + 30))
    assert attempted == ["timing_off", "timing_on"]
    assert all(len(json.loads((output / f"{condition}-interference.json").read_text())) == 12 for condition in attempted)
    assert len(json.loads((output / "status.json").read_text())["completed"]) == 24


def test_telemetry_failure_is_saved_without_selecting_a_subset(tmp_path, monkeypatch):
    calls = fake_matrix(monkeypatch)
    def reject(*arguments):
        raise ValueError("fixture overflow")
    monkeypatch.setattr(calibration, "validate_timing", reject)
    output = tmp_path / "observability"
    with pytest.raises(ValueError, match="overflow"):
        asyncio.run(calibration.run_observability([], Controller(), output, [1] * 101167, time.monotonic() + 30))
    status = json.loads((output / "status.json").read_text())
    assert len(status["completed"]) == 1 and len(status["not_completed"]) == 23
    assert (output / "timing_on/bal0-r0/timing-frontend.json").is_file()
    assert (output / "timing_on/bal0-r0/requests.json").is_file()
    assert Counter(kind for kind, _ in calls)["front_stop"] == 1


def test_request_failure_cancels_children_and_retains_partial_progress(tmp_path, monkeypatch):
    tasks = []
    seeds = []
    async def fake_stream(engine, request_id, prompt, output, rank, model, role, arrived_at, stride, progress):
        tasks.append(asyncio.current_task())
        progress.update(count=1, cadence=[[time.monotonic(), 1]])
        assert stride == 1 and output == (64 if model == 0 else 3000)
        if model == 0:
            raise RuntimeError("fixture stream failure")
        await asyncio.Event().wait()
    def prompt(tokens, seed):
        seeds.append(seed)
        return [1] * tokens
    monkeypatch.setattr(calibration, "stream", fake_stream)
    monkeypatch.setattr(calibration, "synthetic_prompt", prompt)
    monkeypatch.setattr(calibration, "wait_decoding", no_wait)
    monkeypatch.setattr(calibration, "BACKGROUND_SETTLE_S", 0)
    case = {"name": "bal2", "repeat": 1, "layout": [2, 2], "condition": "timing_off"}
    with pytest.raises(RuntimeError, match="stream failure"):
        asyncio.run(calibration.measure_observation_case([None, None], Controller(), case, [1] * 101167, tmp_path))
    failure = json.loads((tmp_path / "failed-requests.json").read_text())
    assert len(failure["partial_progress"]) == 5 and len(tasks) == 5 and all(task.done() for task in tasks)
    assert seeds == [1032, 1033, 1133, 1134]
