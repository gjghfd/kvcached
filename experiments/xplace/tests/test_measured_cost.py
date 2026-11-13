import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from cost_model import StepModel, capture_bucket, default_dp2_params
from measured_cost import (fit_residuals, interference_key, interference_residuals, interference_samples,
                           load_measured_cost, repeated_interference_plan)


def interference_fixture(params, layouts=([0, 0], [2, 2])):
    from calibrate import window_rate

    model = StepModel(copy.deepcopy(params))
    prompt_tokens = 123456
    solo_step = model.decode_step(0, [1, 0], [prompt_tokens + 32, 0])
    rows, raw = [], {}
    for layout in layouts:
        active = any(layout)
        bucket = capture_bucket(max(layout)) if active else None
        first = 100 + model.prefill_s_per_token(0, bucket) * prompt_tokens
        step = model.interfere(0, solo_step, True, bucket) if active else solo_step
        code = {"request_id": "code", "model": 0, "rank_initial": 0, "prompt_tokens": prompt_tokens,
                "output_tokens": 64, "arrived_at": 100, "submitted_at": 100, "first_token_at": first,
                "completed_at": first + 63 * step, "cadence_stride": 1,
                "cadence": [[first + (count - 1) * step, count] for count in range(1, 65)]}
        row = {"name": "bal" + str(max(layout)), "repeat": 0, "layout": layout,
               "code_prefill_s": first - 100, "code_tpot_s": (code["completed_at"] - first) / 63}
        requests = [code]
        if active:
            cadence, now = [[90.0, 1]], 90.0
            for count in range(2, 3001):
                now += model.interfere(1, 0.02, True, bucket) if 100 <= now <= code["completed_at"] else 0.02
                cadence.append([now, count])
            for rank, count in enumerate(layout):
                for index in range(count):
                    requests.append({"request_id": f"conv-{rank}-{index}", "model": 1, "rank_initial": rank,
                                     "prompt_tokens": 4096, "output_tokens": 3000, "cadence": cadence,
                                     "cadence_stride": 1, "first_token_at": 90.0, "completed_at": cadence[-1][0]})
            row.update(conv_before_s=window_rate(requests[1], 97, 100),
                       conv_during_prefill_s=window_rate(requests[1], 101, first - 1),
                       conv_during_decode_s=window_rate(requests[1], first, code["completed_at"]))
        rows.append(row)
        raw[interference_key(row)] = requests
    return rows, raw


def repeated_interference_fixture(decode_duration=None):
    from calibrate import window_rate

    plan = repeated_interference_plan(2)
    original, requests = interference_fixture(default_dp2_params(), [item["layout"] for item in plan["layouts"]])
    rows, raw = [], {}
    for repeat in range(plan["repeats"]):
        for source in original:
            row = copy.deepcopy(source)
            row.update(repeat=repeat, measurement_schema=2, preemptions=0)
            records = copy.deepcopy(requests[interference_key(source)])
            code = records[0]
            if decode_duration is not None and any(row["layout"]):
                code["completed_at"] = code["first_token_at"] + decode_duration
                code["cadence"] = [[code["first_token_at"] + (count - 1) * decode_duration / 63, count]
                                   for count in range(1, 65)]
                row["code_tpot_s"] = decode_duration / 63
                row["conv_during_decode_s"] = window_rate(records[1], code["first_token_at"], code["completed_at"])
            rows.append(row)
            raw[interference_key(row)] = records
    return rows, raw, plan


def test_fixed_dense_repeats_measure_short_code_without_changing_lengths():
    rows, raw, plan = repeated_interference_fixture(decode_duration=2)
    assert len(interference_samples(rows, raw, 2, plan)) == 12
    assert all(records[0]["output_tokens"] == 64 for records in raw.values())
    legacy = copy.deepcopy(rows)
    for row in legacy:
        row.pop("measurement_schema")
    with pytest.raises(ValueError, match="observation window"):
        interference_samples(legacy, raw, 2)


@pytest.mark.parametrize("damage", ["missing_round", "reordered", "short_total", "sparse", "missing_window",
                                     "partial_window", "preempted", "code_length", "conv_length", "loose_gate", "no_plan"])
def test_repeated_interference_never_drops_or_relaxes_bad_observations(damage):
    rows, raw, plan = repeated_interference_fixture(decode_duration=0.8 if damage == "short_total" else 2)
    active = raw["bal2-r0"]
    if damage == "missing_round":
        raw.pop(interference_key(rows.pop()))
    elif damage == "reordered":
        rows.reverse()
    elif damage == "sparse":
        active[1]["cadence_stride"] = 32
    elif damage == "missing_window":
        active[1]["cadence"] = [active[1]["cadence"][0], active[1]["cadence"][-1]]
    elif damage == "partial_window":
        active[1]["first_token_at"] = active[0]["first_token_at"] + 1
    elif damage == "preempted":
        rows[1]["preemptions"] = 1
    elif damage == "code_length":
        active[0]["output_tokens"] = 65
    elif damage == "conv_length":
        active[1]["output_tokens"] = 3001
    elif damage == "loose_gate":
        plan["min_total_decode_window_s"] = 2
    elif damage == "no_plan":
        plan = None
    with pytest.raises(ValueError):
        interference_samples(rows, raw, 2, plan)


@pytest.mark.parametrize("damaged", [False, True])
def test_tp2_calibration_runs_all_dense_repeats_and_persists_failures(tmp_path, monkeypatch, damaged):
    import asyncio
    from types import SimpleNamespace
    import calibrate

    _, raw, plan = repeated_interference_fixture()
    if damaged:
        records = raw["bal2-r0"]
        records[1]["cadence"] = [records[1]["cadence"][0], records[1]["cadence"][-1]]
    calls = []

    async def fake_stream(engine, request_id, prompt, output_tokens, rank, model, role, arrived_at, stride=32):
        parts = request_id.split("_")
        key = f"{parts[2]}-r{parts[3]}" if model == 0 else f"{parts[1]}-r{parts[2]}"
        record = raw[key][0] if model == 0 else next(item for item in raw[key][1:]
                                                     if item["request_id"] == f"conv-{rank}-{parts[4]}")
        calls.append((model, len(prompt), output_tokens, stride))
        return {**copy.deepcopy(record), "request_id": request_id}

    async def noop(*args, **kwargs):
        return None

    async def summaries():
        return [[{"preemptions_total": 0} for rank in range(2)] for model in range(2)]

    monkeypatch.setattr(calibrate, "stream", fake_stream)
    monkeypatch.setattr(calibrate, "wait_idle", noop)
    monkeypatch.setattr(calibrate, "wait_decoding", noop)
    monkeypatch.setattr(calibrate, "time", SimpleNamespace(monotonic=lambda: 100.0))
    monkeypatch.setattr(calibrate, "asyncio", SimpleNamespace(create_task=asyncio.create_task,
                                                              gather=asyncio.gather, sleep=noop))
    controller = SimpleNamespace(tensor_parallel_size=2, summaries=summaries)
    scenario = calibrate.run_interference([None, None], controller, tmp_path, list(range(123456)), 2)
    if damaged:
        with pytest.raises(ValueError):
            asyncio.run(scenario)
    else:
        assert len(asyncio.run(scenario)) == 12
    assert len(calls) == 168 and all(stride == 1 for _, _, _, stride in calls)
    assert all(output == (64 if model == 0 else 3000) for model, _, output, _ in calls)
    assert json.loads((tmp_path / "interference-plan.json").read_text()) == plan
    assert len(list(tmp_path.glob("interference-*-requests.json"))) == 12
    report = json.loads((tmp_path / "interference-validation.json").read_text())
    assert report["status"] == ("failed" if damaged else "passed") and report["cases"] == 12


def write_measured_cost_fixture(root):
    params = default_dp2_params()
    model = StepModel(params)
    rows = []
    for index, layout in enumerate(([1, 1], [2, 2], [8, 8], [9, 7], [16, 16], [20, 12])):
        rows.append({"name": "case-" + str(index), "layout": list(layout), "model": 1,
                     "context_tokens": 4096, "output_tokens": 1024, "status": "measured", "preemptions": 0,
                     "window": {"window_s": 10, "samples": sum(layout)},
                     "step_s": model.decode_step(1, list(layout), [count * 4608 for count in layout])})
    topology = {"num_ranks": 2, "tensor_parallel_size": 2, "expert_parallel_size": 4, "physical_gpus": 4}
    phases = [{"phase": "startup", "status": "passed", "topology": topology},
              {"phase": "job", "status": "passed"}]
    interference, raw = interference_fixture(params)
    sources = {
        "imbalance": rows,
        "grid": [{**row, "name": "grid-" + row["name"]} for row in rows],
        "imbalance_phases": {"phases": phases + [{"phase": "imbalance", "status": "passed",
                                                  "measured": len(rows), "planned": len(rows)}]},
        "cost_phases": {"phases": phases + [{"phase": "dataplane", "status": "passed", "migrations": 8},
                                             {"phase": "grid", "status": "passed", "measured": len(rows)},
                                             {"phase": "interference", "status": "passed"}]},
        "interference": interference,
    }
    descriptors = []
    for role, value in sources.items():
        source = root / (role + ".json")
        source.write_text(json.dumps(value))
        descriptors.append({"role": role, "file": source.name, "job": "123",
                            "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    raw_sources = []
    for key, value in raw.items():
        source = root / ("interference-" + key + "-requests.json")
        source.write_text(json.dumps(value))
        raw_sources.append({"case": key, "file": source.name, "job": "123",
                            "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    params["provenance"] = {"schema_version": 1, "topology": {key: value for key, value in topology.items()
                                                           if key != "physical_gpus"}, "sources": descriptors,
                            "interference_requests": raw_sources}
    params["fit"] = fit_residuals(copy.deepcopy(params), sources["grid"] + sources["imbalance"])
    params["interference_fit"] = interference_residuals(copy.deepcopy(params), interference_samples(interference, raw, 2))
    path = root / "measured.json"
    path.write_text(json.dumps(params))
    return path


def replace_source(path, role, mutate):
    params = json.loads(path.read_text())
    source = next(item for item in params["provenance"]["sources"] if item["role"] == role)
    source_path = path.parent / source["file"]
    value = json.loads(source_path.read_text())
    mutate(value)
    source_path.write_text(json.dumps(value))
    source["sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(params))


def test_verified_sources_and_recomputed_model_residual(tmp_path):
    path = write_measured_cost_fixture(tmp_path)
    params = load_measured_cost(path, 2, 2)
    assert params["verified_fit"]["cases"] == 12
    assert params["verified_fit"]["max_relative_error"] == pytest.approx(0)


@pytest.mark.parametrize("damage", ["prior", "tp1", "source_hash", "source_missing", "reported_fit", "poor_fit"])
def test_invalid_cost_artifact_rejected(tmp_path, damage):
    path = write_measured_cost_fixture(tmp_path)
    params = json.loads(path.read_text())
    if damage == "prior":
        params.pop("provenance")
    elif damage == "tp1":
        params["provenance"]["topology"]["tensor_parallel_size"] = 1
    elif damage == "source_hash":
        (tmp_path / "grid.json").write_text("[]")
    elif damage == "source_missing":
        params["provenance"]["sources"].pop()
    elif damage == "reported_fit":
        params["fit"]["relative_rmse"] = 0.1
    else:
        params["tenants"]["1"]["base_s"] = 10
    path.write_text(json.dumps(params))
    with pytest.raises(ValueError):
        load_measured_cost(path, 2, 2)


@pytest.mark.parametrize("role,mutate", [
    ("grid", lambda rows: rows[0].update(preemptions=1)),
    ("grid", lambda rows: rows[0]["window"].update(samples=1)),
    ("grid", lambda rows: rows.pop()),
    ("cost_phases", lambda value: value["phases"][1].update(status="failed")),
    ("cost_phases", lambda value: value["phases"][0]["topology"].update(tensor_parallel_size=1)),
    ("interference", lambda rows: rows[1].update(conv_during_prefill_s=None)),
])
def test_bad_evidence_rejected_even_with_matching_hash(tmp_path, role, mutate):
    path = write_measured_cost_fixture(tmp_path)
    replace_source(path, role, mutate)
    with pytest.raises(ValueError):
        load_measured_cost(path, 2, 2)


@pytest.mark.parametrize("damage", ["missing_raw", "hash", "prompt_length", "fallback", "phase", "prefill", "solo"])
def test_interference_must_match_raw_and_fitted_model(tmp_path, damage):
    path = write_measured_cost_fixture(tmp_path)
    params = json.loads(path.read_text())
    if damage == "missing_raw":
        params["provenance"].pop("interference_requests")
    elif damage in ("hash", "prompt_length"):
        descriptor = params["provenance"]["interference_requests"][0]
        raw_path = path.parent / descriptor["file"]
        records = json.loads(raw_path.read_text())
        records[0]["prompt_tokens"] *= 2
        raw_path.write_text(json.dumps(records))
        if damage == "prompt_length":
            descriptor["sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    elif damage == "fallback":
        params["interference"]["1"]["additive_s"] = 1
    elif damage == "phase":
        params["interference"]["1"]["decode"] = {"8": 1}
    elif damage == "prefill":
        params["code_prefill"]["solo_s_per_100k"] *= 2
    else:
        params["tenants"]["0"]["base_s"] = 1
    path.write_text(json.dumps(params))
    with pytest.raises(ValueError):
        load_measured_cost(path, 2, 2)
