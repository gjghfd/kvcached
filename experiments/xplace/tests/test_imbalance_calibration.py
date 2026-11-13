import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_split_repeats_preserve_complete_plan():
    full = config()
    full["repeats"] = 3
    expected = calibration.scan_cases(full, 4)
    actual = []
    for repeat in range(3):
        actual.extend(calibration.scan_cases({**full, "repeats": 1, "repeat_start": repeat}, 4))
    assert actual == expected
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError, match="repeat_start"):
            calibration.scan_cases({**full, "repeat_start": invalid}, 4)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "runtime"))

import imbalance_calibration as calibration
from main import parse_args


def config():
    return {"seed": 7, "context_tokens": 4096, "output_tokens": 1024, "repeats": 2,
            "case_timeout_s": 120, "max_scan_s": 2800,
            "series": [{"batch_size": 32, "deltas": [0, 1, 2, 8]},
                       {"batch_size": 128, "deltas": [0, 1, 16, 32]}]}


def test_dense_cases_preserve_batch_and_context():
    cases = calibration.scan_cases(config(), 4)
    assert cases == calibration.scan_cases(config(), 4)
    assert len(cases) == 16 and len({case["name"] for case in cases}) == 16
    for case in cases:
        assert sum(case["layout"]) == case["batch_size"]
        assert min(case["layout"]) >= 0
        assert case["imbalance"] == max(case["layout"]) - min(case["layout"]) == 2 * case["delta"]
        assert case["context_tokens"] == 4096 and case["output_tokens"] == 1024
        assert case["capture_bucket"] >= max(case["layout"])
    first = next(case for case in cases if case["name"] == "bs128_d32_r0")
    second = next(case for case in cases if case["name"] == "bs128_d32_r1")
    assert first["layout"] == [64, 32, 32, 0]
    assert second["layout"] == [32, 32, 0, 64]


@pytest.mark.parametrize("field,value", [("repeats", 0), ("context_tokens", -1), ("output_tokens", True)])
def test_invalid_settings_rejected(field, value):
    settings = config()
    settings[field] = value
    with pytest.raises(ValueError):
        calibration.scan_cases(settings, 4)


def test_invalid_layout_rejected():
    settings = config()
    settings["series"][0]["deltas"].append(9)
    with pytest.raises(ValueError):
        calibration.scan_cases(settings, 4)
    with pytest.raises(ValueError):
        calibration.scan_cases(config(), 2)


def test_scan_keeps_all_raw_results_and_excludes_warmup(tmp_path, monkeypatch):
    seen = []

    async def measure(engines, controller, case, prompts, root):
        seen.append((case["name"], prompts[0]))
        return {**case, "status": "measured", "step_s": 0.02}

    monkeypatch.setattr(calibration, "measure_layout", measure)
    result = asyncio.run(calibration.run_imbalance_scan([], None, tmp_path, config(), 4))
    assert result["status"] == "passed" and result["measured"] == 16
    records = json.loads((tmp_path / "results.json").read_text())
    assert len(records) == 16 and all(record["name"] != "warmup" for record in records)
    assert seen[0][0] == "warmup" and all(prompt == seen[0][1] for _, prompt in seen)
    assert json.loads((tmp_path / "warmup.json").read_text())["output_tokens"] == 256


def test_cli_requires_scan_config(tmp_path):
    args = ["--checkpoint", "test", "--output", "out", "--prompt-bank", "bank.json", "--phases", "imbalance"]
    with pytest.raises(SystemExit):
        parse_args(args)
    path = tmp_path / "scan.json"
    path.write_text(json.dumps(config()))
    parsed = parse_args(args + ["--imbalance-config", str(path)])
    assert parsed.imbalance_config == path


@pytest.mark.parametrize("scan_budget,completed", [(2800, 69), (3100, 72)])
def test_tp2_time_budget_reports_truncation_without_dropping_cases(tmp_path, monkeypatch, scan_budget, completed):
    settings = {**config(), "num_ranks": 2, "tensor_parallel_size": 2, "repeats": 3,
                "max_scan_s": scan_budget,
                "series": [{"batch_size": 32, "deltas": [0, 1, 2, 4, 8, 12, 16]},
                           {"batch_size": 64, "deltas": [0, 1, 2, 4, 8, 16, 24, 32]},
                           {"batch_size": 128, "deltas": [0, 1, 2, 4, 8, 16, 32, 48, 64]}]}
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(calibration, "time", SimpleNamespace(monotonic=lambda: clock.now))

    async def measure(engines, controller, case, prompts, root):
        clock.now += 20 if case["name"] == "warmup" else 39
        return {**case, "status": "measured", "step_s": 0.02}

    monkeypatch.setattr(calibration, "measure_layout", measure)
    result = asyncio.run(calibration.run_imbalance_scan(
        [], SimpleNamespace(tensor_parallel_size=2), tmp_path, settings, 2))
    cases = calibration.scan_cases(settings, 2, 2)
    assert result["planned"] == len(cases) == 72
    assert result["completed"] == result["measured"] == completed
    assert result["remaining_cases"] == [case["name"] for case in cases[completed:]]
    assert result["stop_reason"] == ("completed" if completed == 72 else "scan_budget_exhausted")
    assert result["status"] == ("passed" if completed == 72 else "incomplete")
    assert result["max_scan_s"] == scan_budget and result["case_timeout_s"] == 120
    assert calibration.scan_cases({**settings, "max_scan_s": 2800}, 2, 2) == cases


@pytest.mark.parametrize("preemptions,expected", [(0, "measured"), (1, "invalid_window_or_preemption")])
def test_measurement_saves_cadence_and_rejects_preemptions(tmp_path, monkeypatch, preemptions, expected):
    class Controller:
        budgets = [40.0] * 4

        def __init__(self):
            self.reads = 0

        async def memory(self, release_unused=False):
            assert release_unused

        async def summaries(self):
            self.reads += 1
            return {1: [{"preemptions_total": preemptions if self.reads > 1 and rank == 0 else 0}
                        for rank in range(4)]}

    async def idle(controller):
        return None

    async def stream(engine, request_id, prompt, output_tokens, rank, model, role, arrived_at, progress=None):
        return {"request_id": request_id, "first_token_at": 1.0, "completed_at": 15.0,
                "cadence": [[2.0, 64], [6.0, 256], [12.0, output_tokens]],
                "output_tokens": output_tokens, "rank_initial": rank}

    monkeypatch.setattr(calibration, "wait_idle", idle)
    monkeypatch.setattr(calibration, "fits_budget", lambda *args: True)
    monkeypatch.setattr(calibration, "stream", stream)
    case = calibration.scan_cases(config(), 4)[0]
    result = asyncio.run(calibration.measure_layout([None, None], Controller(), case,
                                                    [[1234]] * case["batch_size"], tmp_path))
    assert result["status"] == expected and result["preemptions"] == preemptions
    assert result["window"]["samples"] == case["batch_size"]
    assert result["common_start"] == 2.0 and result["common_end"] == 14.0
    records = json.loads((tmp_path / result["requests_file"]).read_text())
    assert len(records) == case["batch_size"] and all(record["cadence"] for record in records)
