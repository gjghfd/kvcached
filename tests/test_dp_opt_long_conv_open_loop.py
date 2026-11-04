import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def calibration(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("long_conv_open_loop")


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    files = {"code-marks.json": [{"prompt_tokens": 100123, "output_tokens": 64, "prompt_key": "real-history"}],
        "conv-marks.json": [{"prompt_tokens": prompt, "output_tokens": output, "count": count,
            "source_internal_row": ordinal, "source_fragment": "original-paired-slice"}
            for ordinal, (prompt, output, count) in enumerate(((2500, 512, 1), (4000, 800, 3)))],
        "prompt-bank.json": {"prompts": {}}}
    for name, value in files.items():
        (root / name).write_text(json.dumps(value))
    return root


def test_frozen_rates_rebuild_and_preserve_Code_stream_and_original_marks(calibration, inputs, tmp_path):
    frozen = tmp_path / "frozen"
    config = calibration.prepare(inputs, frozen)
    assert calibration.validate_upload(frozen, inputs) == config
    traces = [json.loads((frozen / f"rate-{ordinal}.json").read_text()) for ordinal in range(3)]
    codes = [[mark for mark in trace["requests"] if mark["model"] == 0] for trace in traces]
    assert codes[0] == codes[1] == codes[2]
    for trace in traces:
        assert trace["allow_empty_tenants"]
        assert trace["arrival"] == "two_independent_Poisson_streams"
        assert trace["duration_s"] == 3600 and trace["evaluation_start_s"] == 600
        for mark in trace["requests"]:
            if mark["model"] == 1:
                assert mark["output_tokens"] == mark["synthetic_output_tokens"] == 24576
                assert mark["original_output_tokens"] in (512, 800)
                assert mark["original_source_fragment"] == "original-paired-slice"
                assert "count" not in mark
    assert not config["selection"]["select_using_policy_benefit"]
    assert not config["research_goal_complete"]


def test_empty_draw_remains_empty_without_forced_requests(calibration, inputs, monkeypatch):
    monkeypatch.setattr(calibration, "DURATION_S", .001)
    trace = calibration.make_trace(json.loads((inputs / "code-marks.json").read_text()),
        json.loads((inputs / "conv-marks.json").read_text()), "a" * 64, .00565)
    assert trace["requests"] == [] and trace["allow_empty_tenants"]


@pytest.mark.parametrize("fault", ["rate", "seed", "guard", "bank", "trace", "refrozen_trace", "pointer"])
def test_independent_upload_validator_rejects_changed_controls_or_draw(calibration, inputs, tmp_path, fault):
    frozen = tmp_path / "frozen"
    calibration.prepare(inputs, frozen)
    config_path = frozen / "open-loop-config.json"
    config = json.loads(config_path.read_text())
    if fault == "rate":
        config["conv_rates_rps"][0] = .006
    elif fault == "seed":
        config["seed"] += 1
    elif fault == "guard":
        config["selection"]["maximum_pending_growth"] = 100
    elif fault == "bank":
        (inputs / "conv-marks.json").write_text("[]")
    elif fault == "pointer":
        (frozen / "trace.json").write_text("{}")
    else:
        path = frozen / "rate-1.json"
        trace = json.loads(path.read_text())
        trace["requests"][0]["arrival_s"] += 1
        path.write_text(json.dumps(trace))
        if fault == "refrozen_trace":
            config["trace_sha256"]["rate-1.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        calibration.validate_upload(frozen, inputs)


def test_preserve_previous_frozen_input_directory(calibration, inputs, tmp_path):
    destination = tmp_path / "frozen"
    calibration.prepare(inputs, destination)
    with pytest.raises(ValueError, match="Preserve"):
        calibration.prepare(inputs, destination)


@pytest.mark.parametrize("prompt,output", [(99999, 64), (100123, 63)])
def test_Code_lengths_are_not_shortened_for_capacity(calibration, inputs, tmp_path, prompt, output):
    (inputs / "code-marks.json").write_text(json.dumps([{"prompt_tokens": prompt, "output_tokens": output}]))
    with pytest.raises(ValueError, match=">=100K"):
        calibration.prepare(inputs, tmp_path / "frozen")


def test_population_half_windows_measure_real_queue_not_completion_rate(calibration):
    population = [{"elapsed_s": elapsed, "capacity_pending": pending, "counts": [[0, 0], [1, 1]],
        "admission_stable_across_observation": True} for elapsed, pending in ((600, 0), (2100, 1), (3600, 0))]
    first, second = calibration.population_windows(population)
    assert first["mean_pending"] == 0 and second["mean_pending"] == 1
    assert first["covered_s"] == second["covered_s"] == 1500
    assert first["mean_active"] == second["mean_active"] == 2


@pytest.mark.parametrize("later", [600, 599, float("nan")])
def test_population_order_and_nonfinite_values_are_rejected(calibration, later):
    with pytest.raises(ValueError, match="timestamps"):
        calibration.population_windows([{"elapsed_s": 600}, {"elapsed_s": later}])


def test_candidate_screen_never_chooses_by_TTFT_gain(calibration, tmp_path, monkeypatch):
    def phase(path):
        ordinal = int(path.name[-1])
        return {"eligible_for_separate_evaluation_rate_candidate": ordinal != 0,
            "physical_pressure": {"local_physical_ratio": (1.2, 1.21, 1.24)[ordinal]},
            "rate_rps": [.001, calibration.CONV_RATES[ordinal]], "TTFT_benefit": (99, -10, 50)[ordinal]}
    monkeypatch.setattr(calibration, "analyze_phase", phase)
    result = calibration.analyze(tmp_path)
    assert result["candidate_case"] == "rate-1"
    assert not result["final_trace_frozen"] and not result["research_goal_complete"]


def test_no_qualifying_rate_is_retained_not_retried_or_extrapolated(calibration, tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "analyze_phase", lambda path:
        {"eligible_for_separate_evaluation_rate_candidate": False})
    assert calibration.analyze(tmp_path)["candidate_case"] is None


@pytest.fixture
def phase(tmp_path):
    root = tmp_path / "rate-0"
    case = root / "r0-no_move"
    case.mkdir(parents=True)
    offered = [{"request_id": "conv", "model": 1, "arrival_s": 700, "output_tokens": 4}]
    (root / "trace.json").write_text(json.dumps({"model_rates_rps": [.001, .00565], "requests": offered}))
    (case / "workload.json").write_text(json.dumps({"origin": 10000}))
    (case / "requests.json").write_text(json.dumps([{"request_id": "conv", "model": 1,
        "arrived_at": 10700, "submitted_at": 10700.1, "reserved_at": 10700.2,
        "first_token_at": 10701, "completed_at": 10710, "token_ids": [1, 2, 3, 4]}]))
    population = [{"elapsed_s": elapsed, "capacity_pending": pending, "counts": [[0, 0], [1, 1]],
        "admission_stable_across_observation": True} for elapsed, pending in ((600, 0), (2100, 0), (3600, 0))]
    (case / "population.json").write_text(json.dumps(population))
    (case / "decisions.json").write_text("[]")
    return root


@pytest.mark.parametrize("ratio,coverage,pending_growth,eligible", [
    (1.2, 3000, 0, True), (1.09, 3000, 0, False), (1.31, 3000, 0, False),
    (1.2, 2000, 0, False), (1.2, 3000, 1, False)])
def test_measured_screen_uses_pressure_coverage_and_queue_growth(calibration, phase, monkeypatch,
                                                               ratio, coverage, pending_growth, eligible):
    monkeypatch.setattr(calibration, "background_pressure", lambda *args:
        {"local_physical_ratio": ratio, "background_covered_s": coverage})
    path = phase / "r0-no_move/population.json"
    population = json.loads(path.read_text())
    population[1]["capacity_pending"] = pending_growth
    path.write_text(json.dumps(population))
    result = calibration.analyze_phase(phase)
    assert result["eligible_for_separate_evaluation_rate_candidate"] is eligible
    assert result["measurement_metrics"]["Code"]["mean_ttft_s"] is None
    assert result["measurement_metrics"]["Conv"]["mean_ttft_s"] == 1
    assert result["measured_admitted_service"]["Conv"]["count"] == 1
    assert not result["stationarity_established"]


def test_missing_physical_witness_is_explicit_not_invented(calibration, phase):
    result = calibration.analyze_phase(phase)
    assert result["physical_pressure"] is None
    assert result["physical_pressure_unavailable_reason"]
    assert not result["eligible_for_separate_evaluation_rate_candidate"]


def test_empty_offered_trace_keeps_empty_tenant_metrics(calibration, phase):
    path = phase / "trace.json"
    trace = json.loads(path.read_text())
    trace["requests"] = []
    path.write_text(json.dumps(trace))
    (phase / "r0-no_move/requests.json").write_text("[]")
    result = calibration.analyze_phase(phase)
    assert result["offered_counts"] == [0, 0]
    assert result["measurement_metrics"]["Conv"]["completed_requests"] == 0
    assert result["measurement_metrics"]["Conv"]["mean_ttft_s"] is None


def test_incomplete_drain_cannot_pass_calibration_analysis(calibration, phase):
    (phase / "r0-no_move/requests.json").write_text("[]")
    with pytest.raises(ValueError, match="drain all"):
        calibration.analyze_phase(phase)
