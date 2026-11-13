import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
import graph_execution
import imbalance_calibration as calibration


def config():
    return dict(num_ranks=2, tensor_parallel_size=2, seed=7, context_tokens=8, output_tokens=512,
                repeats=1, case_timeout_s=1, max_scan_s=10, series=[dict(batch_size=4, deltas=[0, 1, 2])])


class Controller:
    budgets, tensor_parallel_size, num_ranks = [40.0] * 4, 2, 2

    async def memory(self, **kwargs):
        return None

    async def summaries(self):
        return {1: [dict(preemptions_total=0), dict(preemptions_total=0)]}


def result(request_id):
    return dict(request_id=request_id, first_token_at=1.0, completed_at=15.0, output_tokens=512,
                cadence=[[2.0, 32], [6.0, 256], [12.0, 512]])


@pytest.fixture
def prepared(monkeypatch):
    async def idle(controller):
        return None

    async def stream(engine, request_id, *args, progress=None):
        return result(request_id)

    monkeypatch.setattr(calibration, "wait_idle", idle)
    monkeypatch.setattr(calibration, "fits_budget", lambda *args: True)
    monkeypatch.setattr(calibration, "stream", stream)


@pytest.mark.parametrize("flag", ["1", 1, None, [], {}])
def test_graph_switch_requires_boolean(flag):
    with pytest.raises(ValueError):
        calibration.scan_cases(dict(config(), observe_graph_execution=flag), 2, 2)


def test_graph_switch_keeps_same_cases():
    plain = calibration.scan_cases(config(), 2, 2)
    observed = calibration.scan_cases(dict(config(), observe_graph_execution=True), 2, 2)
    assert [{key: value for key, value in case.items() if key != "observe_graph_execution"} for case in observed] == plain
    assert all(case["observe_graph_execution"] is True for case in observed)


@pytest.mark.parametrize("failure_at", ["warmup", "first_case"])
@pytest.mark.parametrize("failure_type", ["exception", "timeout", "cancel"])
def test_scan_failure_preserves_plan_and_unrun_points(tmp_path, monkeypatch, failure_at, failure_type):
    reached = None

    async def measure(engines, controller, case, prompts, root):
        if (case["name"] == "warmup") == (failure_at == "warmup"):
            reached.set()
            if failure_type == "exception":
                raise RuntimeError("deliberate request failure")
            await asyncio.Event().wait()
        return dict(case, status="measured", step_s=0.02)

    monkeypatch.setattr(calibration, "measure_layout", measure)

    async def run():
        nonlocal reached
        reached = asyncio.Event()
        task = asyncio.create_task(calibration.run_imbalance_scan([], Controller(), tmp_path, config(), 2))
        if failure_type == "cancel":
            await reached.wait()
            task.cancel()
        expected = dict(exception=RuntimeError, timeout=asyncio.TimeoutError, cancel=asyncio.CancelledError)
        with pytest.raises(expected[failure_type]):
            await task

    asyncio.run(run())
    summary = json.loads((tmp_path / "summary.json").read_text())
    records = json.loads((tmp_path / "results.json").read_text())
    statuses = json.loads((tmp_path / "case-statuses.json").read_text())
    assert summary["status"] == "failed" and summary["failure"]
    assert len(statuses) == summary["planned"] == 3
    assert len(records) == (0 if failure_at == "warmup" else 1)
    failed = json.loads((tmp_path / "warmup.json").read_text()) if failure_at == "warmup" else records[0]
    assert failed["status"] == dict(exception="error", timeout="timed_out", cancel="cancelled")[failure_type]
    assert all(row["status"] == "not_run" for row in statuses[len(records):])


def test_immutable_cutoff_precedes_cleanup_results(tmp_path, monkeypatch, prepared):
    case = calibration.scan_cases(config(), 2, 2)[0]
    failure = RuntimeError("stream failed")

    async def stream(engine, request_id, *args, progress=None):
        index = int(request_id.rsplit("_", 1)[1])
        progress.update(count=8, cadence=[[2.0, 8]])
        if index == 0:
            return result(request_id)
        if index == 1:
            await asyncio.sleep(0.01)
            raise failure
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            progress["count"] = 512
            progress["cadence"].append([3.0, 512])
            return result(request_id)

    monkeypatch.setattr(calibration, "stream", stream)
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(calibration.measure_layout([None, None], Controller(), case, [[1]] * 4, tmp_path))
    assert caught.value is failure
    cutoff = json.loads((tmp_path / f"{case['name']}-requests.json").read_text())
    incomplete = json.loads((tmp_path / f"{case['name']}-incomplete.json").read_text())
    cleanup = json.loads((tmp_path / f"{case['name']}-cleanup-results.json").read_text())
    assert len(cutoff) == 1 and len(incomplete) == 3 and len(cleanup) == 3
    assert all(row["partial"]["count"] == 8 and len(row["partial"]["cadence"]) == 1 for row in incomplete)
    assert len(json.loads((tmp_path / f"{case['name']}-attempt.json").read_text())["requests"]) == 4


def test_preparation_failure_retains_all_requests(tmp_path, monkeypatch, prepared):
    async def fail(controller):
        raise RuntimeError("idle failed")

    monkeypatch.setattr(calibration, "wait_idle", fail)
    case = calibration.scan_cases(config(), 2, 2)[0]
    with pytest.raises(RuntimeError):
        asyncio.run(calibration.measure_layout([None, None], Controller(), case, [[1]] * 4, tmp_path))
    incomplete = json.loads((tmp_path / f"{case['name']}-incomplete.json").read_text())
    assert len(incomplete) == 4 and all(row["partial"] == {} for row in incomplete)


@pytest.mark.parametrize("mode", ["start_error", "stop_error", "stop_cancel", "pair_invalid"])
def test_graph_control_failure_retains_measurement_and_cancel(tmp_path, monkeypatch, prepared, mode):
    calls = []
    reached = None

    async def control(controller, model, command, case, root):
        calls.append(command)
        if mode == f"{command}_error":
            raise RuntimeError("control failed")
        if mode == "stop_cancel" and command == "stop":
            reached.set()
            await asyncio.Event().wait()
        return {"command": command}

    monkeypatch.setattr(graph_execution, "graph_control", control)
    monkeypatch.setattr(graph_execution, "validate_graph_pair", lambda *args: dict(valid=False))
    case = calibration.scan_cases(dict(config(), observe_graph_execution=True), 2, 2)[0]

    async def run():
        nonlocal reached
        reached = asyncio.Event()
        task = asyncio.create_task(calibration.measure_layout([None, None], Controller(), case, [[1]] * 4, tmp_path))
        if mode == "stop_cancel":
            await reached.wait()
            task.cancel()
        if mode in ("start_error", "stop_cancel"):
            with pytest.raises(RuntimeError if mode == "start_error" else asyncio.CancelledError):
                await task
        else:
            assert (await task)["status"] == "invalid_graph_evidence"

    asyncio.run(run())
    status = json.loads((tmp_path / f"{case['name']}-status.json").read_text())
    assert calls == ["start", "stop"] and not status["graph_evidence"]["valid"]
    if mode != "start_error":
        assert status["step_s"] is not None
        assert len(json.loads((tmp_path / f"{case['name']}-requests.json").read_text())) == 4
    assert status["status"] == dict(start_error="error", stop_error="invalid_graph_evidence",
                                    stop_cancel="cancelled", pair_invalid="invalid_graph_evidence")[mode]


def test_budget_failure_stops_warmup_and_keeps_plan(tmp_path, monkeypatch, prepared):
    monkeypatch.setattr(calibration, "fits_budget", lambda *args: False)
    summary = asyncio.run(calibration.run_imbalance_scan([None, None], Controller(), tmp_path, config(), 2))
    assert summary["stop_reason"] == "warmup_not_measured" and summary["measured"] == 0
    statuses = json.loads((tmp_path / "case-statuses.json").read_text())
    assert len(statuses) == 3 and all(row["status"] == "not_run" for row in statuses)
    before = (tmp_path / "plan.json").read_bytes()
    with pytest.raises(FileExistsError):
        asyncio.run(calibration.run_imbalance_scan([None, None], Controller(), tmp_path, config(), 2))
    assert (tmp_path / "plan.json").read_bytes() == before


def test_graph_failure_stops_even_when_performance_window_is_invalid(tmp_path, monkeypatch):
    async def measure(engines, controller, case, prompts, root):
        if case["name"] == "warmup":
            return dict(case, status="measured", step_s=0.02)
        return dict(case, status="invalid_window_or_preemption", step_s=None, graph_evidence=dict(valid=False))

    monkeypatch.setattr(calibration, "measure_layout", measure)
    summary = asyncio.run(calibration.run_imbalance_scan([], Controller(), tmp_path, config(), 2))
    assert summary["stop_reason"] == "graph_evidence_failed" and summary["completed"] == 1
    statuses = json.loads((tmp_path / "case-statuses.json").read_text())
    assert [row["status"] for row in statuses] == ["invalid_window_or_preemption", "not_run", "not_run"]
