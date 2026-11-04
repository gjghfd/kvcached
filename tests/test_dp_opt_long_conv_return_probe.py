import copy
import importlib
import json
from pathlib import Path

import pytest

@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("long_conv_return_probe")


def banks():
    return ([{"prompt_tokens": 100084, "output_tokens": 64, "prompt_key": "swe-one"},
             {"prompt_tokens": 101675, "output_tokens": 64, "prompt_key": "swe-two"}],
            [{"prompt_tokens": 2048, "output_tokens": 512, "count": 2},
             {"prompt_tokens": 7999, "output_tokens": 700, "count": 5}])


def test_long_context_plan_covers_supported_remaining_work(probe):
    result = probe.screen_plan(probe.calibration_plan())
    assert result["cases"] == 112
    assert result["maximum_mapped_mib"] == 9984
    assert max(probe.calibration_plan()["populations"][0]["prompt_tokens"]) + 3072 - 1 > 7999 + 24576


def test_probe_changes_only_output_and_arrivals_not_models_or_source_lengths(probe):
    code, conv = banks()
    original = copy.deepcopy((code, conv))
    trace = probe.make_probe(code, conv, "a" * 64, 20261030)
    assert (code, conv) == original
    assert trace == probe.make_probe(code, conv, "a" * 64, 20261030)
    assert trace["arrival"] == "closed_N2_mechanism_probe_not_Poisson_final_trace"
    assert len(trace["requests"]) == 3
    assert [request["arrival_s"] for request in trace["requests"]] == [0., 1., 120.]
    assert all(request["output_tokens"] == 24576 for request in trace["requests"][:2])
    assert all((request["prompt_tokens"], request["original_output_tokens"])
               in [(mark["prompt_tokens"], mark["output_tokens"]) for mark in conv]
               for request in trace["requests"][:2])
    assert trace["requests"][2] == {**next(mark for mark in code if mark["prompt_key"] == trace["requests"][2]["prompt_key"]),
                                    "request_id": "long-conv-20261030-code", "model": 0, "arrival_s": 120.}


@pytest.mark.parametrize("output", [8192, 0, -1, 24576.0, True, 131072])
def test_output_must_be_explicitly_longer_and_fit_model(probe, output):
    with pytest.raises(ValueError):
        probe.make_probe(*banks(), "a" * 64, 20261030, output)


@pytest.mark.parametrize("weights", [[], [{"count": 0}], [{"count": -1}], [{"count": 1.5}], [{"count": True}]])
def test_invalid_weighted_bank_rejected(probe, weights):
    import random

    with pytest.raises(ValueError):
        probe.weighted_mark(weights, random.Random(1))


def write_inputs(root):
    root.mkdir()
    code, conv = banks()
    for name, value in (("code-marks.json", code), ("conv-marks.json", conv), ("prompt-bank.json", {})):
        (root / name).write_text(json.dumps(value))


def test_frozen_probe_rebuild_and_existing_destination_protection(tmp_path, probe):
    inputs, frozen = tmp_path / "inputs", tmp_path / "frozen"
    write_inputs(inputs)
    probe.prepare(inputs, frozen)
    assert probe.validate_upload(frozen, inputs)["calibration"]["cases"] == 112
    with pytest.raises(ValueError, match="Preserve"):
        probe.prepare(inputs, frozen)


@pytest.mark.parametrize("filename", ["probe-20261030.json", "calibration-plan.json", "probe-config.json"])
def test_modified_frozen_files_rejected(tmp_path, filename, probe):
    inputs, frozen = tmp_path / "inputs", tmp_path / "frozen"
    write_inputs(inputs)
    probe.prepare(inputs, frozen)
    (frozen / filename).write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        probe.validate_upload(frozen, inputs)


def test_changed_original_bank_rejected(tmp_path, probe):
    inputs, frozen = tmp_path / "inputs", tmp_path / "frozen"
    write_inputs(inputs)
    probe.prepare(inputs, frozen)
    (inputs / "conv-marks.json").write_text("[]")
    with pytest.raises(ValueError, match="bank changed"):
        probe.validate_upload(frozen, inputs)


def test_forged_rehashed_trace_cannot_change_draws(tmp_path, probe):
    import hashlib

    inputs, frozen = tmp_path / "inputs", tmp_path / "frozen"
    write_inputs(inputs)
    probe.prepare(inputs, frozen)
    name = "probe-20261030.json"
    trace = json.loads((frozen / name).read_text())
    trace["requests"][0]["prompt_tokens"] += 1
    (frozen / name).write_text(json.dumps(trace))
    config = json.loads((frozen / "probe-config.json").read_text())
    config["frozen_sha256"][name] = hashlib.sha256((frozen / name).read_bytes()).hexdigest()
    (frozen / "probe-config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="reproduce"):
        probe.validate_upload(frozen, inputs)
