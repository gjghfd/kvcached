import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from fit_measured_cost import build_bundle, fit_decode, fit_interference
from test_measured_cost import interference_fixture, repeated_interference_fixture, write_measured_cost_fixture
from calibrate import steady_step
from cost_model import StepModel, default_dp2_params
from measured_cost import interference_samples, load_measured_cost


def decode_fixture():
    model = StepModel(default_dp2_params())
    rows, raw = [], {}
    cases = [([1, 1], 4096), ([2, 2], 4096), ([4, 4], 4096), ([8, 8], 4096), ([9, 7], 4096),
             ([16, 16], 4096), ([20, 12], 4096), ([8, 8], 1024), ([8, 8], 8192), ([8, 8], 16384)]
    for index, (layout, context) in enumerate(cases):
        step = model.decode_step(1, layout, [count * (context + 512) for count in layout])
        filename = f"case-{index}-requests.json"
        requests = [{"request_id": f"request-{rank}-{offset}", "rank_initial": rank, "model": 1,
                     "prompt_tokens": context, "output_tokens": 1024, "first_token_at": 10,
                     "completed_at": 10 + 1023 * step,
                     "cadence": [[10 + (count - 1) * step, count] for count in [1, *range(16, 1025, 16)]]}
                    for rank, count in enumerate(layout) for offset in range(count)]
        measured, window = steady_step(requests)
        rows.append({"name": f"case-{index}", "layout": layout, "contexts": [[context] * count for count in layout],
                     "output_tokens": 1024, "model": 1, "status": "measured", "preemptions": 0,
                     "step_s": measured, "window": window, "requests_file": filename})
        raw[filename] = requests
    return rows, raw


def job_fixture(root):
    existing = root / "existing"
    existing.mkdir()
    path = write_measured_cost_fixture(existing)
    params = json.loads(path.read_text())
    dense = root / "job-123"
    cost = root / "job-456"
    dense_out = dense / "out" / "calibration" / "imbalance"
    cost_out = cost / "out" / "calibration"
    dense_out.mkdir(parents=True)
    cost_out.mkdir(parents=True)
    rows, raw = decode_fixture()
    for descriptor in params["provenance"]["sources"]:
        role = descriptor["role"]
        data = json.loads((existing / descriptor["file"]).read_text())
        if role in ("imbalance", "grid"):
            target = dense_out / "results.json" if role == "imbalance" else cost_out / "conv-grid.json"
            data = rows
        elif role == "interference":
            target = cost_out / "interference.json"
        else:
            target = (dense if role == "imbalance_phases" else cost) / "out" / "phases.json"
            for phase in data["phases"]:
                if phase["phase"] in ("imbalance", "grid"):
                    phase.update(measured=len(rows), planned=len(rows))
        target.write_text(json.dumps(data))
    for directory in (dense_out, cost_out):
        for filename, value in raw.items():
            (directory / filename).write_text(json.dumps(value))
    for descriptor in params["provenance"]["interference_requests"]:
        shutil.copyfile(existing / descriptor["file"], cost_out / descriptor["file"])
    return dense, cost


def test_relative_monotone_fit_reports_final_model_and_every_point():
    rows, _ = decode_fixture()
    params, report = fit_decode(rows)
    assert params["fit"]["cases"] == len(rows)
    assert params["fit"]["max_relative_error"] < 1e-10
    graph = list(params["tenants"]["1"]["graph_s"].values())
    assert graph == sorted(graph)
    assert report["design_rank"] == report["design_columns"]
    assert len(report["rows"]) == len(rows)
    rows[5]["step_s"] *= 0.7
    params, report = fit_decode(rows)
    actual = sum(row["relative_error"] ** 2 for row in report["rows"]) / len(rows)
    assert params["fit"]["relative_rmse"] ** 2 == pytest.approx(actual)


def test_fixed_context_scan_is_not_a_long_context_fit():
    rows, _ = decode_fixture()
    with pytest.raises(ValueError, match="fixed-context"):
        fit_decode(rows[:7])


def test_prefill_uses_real_tokens_not_nominal_100k():
    params = default_dp2_params()
    rows, raw = interference_fixture(params)
    samples = interference_samples(rows, raw, 2)
    fitted, _ = fit_decode(decode_fixture()[0])
    report = fit_interference(fitted, samples)
    assert fitted["code_prefill"]["solo_s_per_100k"] == pytest.approx(26.4)
    assert fitted["code_prefill"]["with_other_s_per_100k"]["8"] == pytest.approx(54)
    assert report["code_prompt_tokens"] == [123456, 123456]


def test_bundle_is_reproducible_loadable_and_never_overwritten(tmp_path):
    dense, cost = job_fixture(tmp_path)
    output = tmp_path / "bundle"
    report = build_bundle(dense, cost, output)
    assert report["status"] == "passed_in_sample_checks"
    assert report["independent_validation"] is False
    params = load_measured_cost(output / "step-params.json", 2, 2)
    assert params["fit"]["cases"] == 20
    repeat = tmp_path / "repeat"
    build_bundle(dense, cost, repeat)
    assert (output / "step-params.json").read_bytes() == (repeat / "step-params.json").read_bytes()
    with pytest.raises(FileExistsError):
        build_bundle(dense, cost, output)


@pytest.mark.parametrize("damage", [None, "missing_plan", "bad_dataplane", "source_job"])
def test_separate_interference_job_reuses_every_decode_row_with_strict_provenance(tmp_path, damage):
    dense, cost = job_fixture(tmp_path)
    separate = tmp_path / "job-789"
    calibration = separate / "out" / "calibration"
    calibration.mkdir(parents=True)
    phases = json.loads((cost / "out/phases.json").read_text())
    phases["phases"] = [phase for phase in phases["phases"] if phase["phase"] != "grid"]
    if damage == "bad_dataplane":
        next(phase for phase in phases["phases"] if phase["phase"] == "dataplane")["migrations"] = 7
    (separate / "out/phases.json").write_text(json.dumps(phases))
    rows, raw, plan = repeated_interference_fixture()
    (calibration / "interference.json").write_text(json.dumps(rows))
    if damage != "missing_plan":
        (calibration / "interference-plan.json").write_text(json.dumps(plan))
    for key, records in raw.items():
        (calibration / f"interference-{key}-requests.json").write_text(json.dumps(records))
    output = tmp_path / "joined"
    if damage in ("missing_plan", "bad_dataplane"):
        with pytest.raises(ValueError):
            build_bundle(dense, cost, output, separate)
        assert not (output / "step-params.json").exists()
        return
    build_bundle(dense, cost, output, separate)
    path = output / "step-params.json"
    params = load_measured_cost(path, 2, 2)
    assert params["fit"]["cases"] == 20
    assert params["provenance"]["schema_version"] == 2
    source_jobs = {source["role"]: source["job"] for source in params["provenance"]["sources"]}
    assert source_jobs == {"imbalance": "123", "imbalance_phases": "123", "grid": "456", "cost_phases": "456",
                           "interference": "789", "interference_plan": "789", "interference_phases": "789"}
    assert len(params["provenance"]["interference_requests"]) == 12
    if damage == "source_job":
        next(source for source in params["provenance"]["sources"] if source["role"] == "interference_phases")["job"] = "456"
        path.write_text(json.dumps(params))
        with pytest.raises(ValueError, match="mismatched jobs"):
            load_measured_cost(path, 2, 2)


@pytest.mark.parametrize("damage", ["missing_grid", "wrong_context", "bad_window", "bad_topology", "bad_fit"])
def test_rejected_bundle_preserves_failure_without_usable_params(tmp_path, damage):
    dense, cost = job_fixture(tmp_path)
    path = cost / "out" / "calibration" / "conv-grid.json"
    if damage == "missing_grid":
        path.unlink()
    elif damage == "bad_topology":
        path = cost / "out" / "phases.json"
        value = json.loads(path.read_text())
        value["phases"][0]["topology"]["tensor_parallel_size"] = 1
        path.write_text(json.dumps(value))
    elif damage == "wrong_context":
        path = path.parent / "case-0-requests.json"
        value = json.loads(path.read_text())
        value[0]["prompt_tokens"] *= 2
        path.write_text(json.dumps(value))
    elif damage == "bad_window":
        value = json.loads(path.read_text())
        value[0]["step_s"] *= 2
        path.write_text(json.dumps(value))
    else:
        interference = path.parent / "interference.json"
        value = json.loads(interference.read_text())
        solo = value[0]
        raw = interference.parent / f"interference-{solo['name']}-r0-requests.json"
        requests = json.loads(raw.read_text())
        code = requests[0]
        code["completed_at"] = code["first_token_at"] + 63 * solo["code_tpot_s"] * 3
        code["cadence"] = [[code["first_token_at"] + (count - 1) * solo["code_tpot_s"] * 3, count]
                           for count in range(1, 65)]
        solo["code_tpot_s"] *= 3
        raw.write_text(json.dumps(requests))
        interference.write_text(json.dumps(value))
    output = tmp_path / "failed"
    with pytest.raises((ValueError, FileNotFoundError)):
        build_bundle(dense, cost, output)
    assert not (output / "step-params.json").exists()
    report = json.loads((output / "fit-report.json").read_text())
    assert report["status"] == "failed"
    assert report["error"]
