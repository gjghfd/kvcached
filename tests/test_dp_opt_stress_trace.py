import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("stress_trace")


def test_output_variant_preserves_source_and_original_length(module):
    original = [{"prompt_tokens": 4096, "output_tokens": 512, "count": 3, "source_internal_row": 7}]
    transformed = module.long_output_marks(original, 8192)
    assert original[0]["output_tokens"] == 512
    assert transformed[0]["original_output_tokens"] == 512
    assert transformed[0]["output_tokens"] == 8192
    assert transformed[0]["source_internal_row"] == 7 and transformed[0]["count"] == 3
    assert "synthetic" in transformed[0]["source_fragment"]


def test_natural_two_stream_warmup_is_frozen_not_bootstrap(module):
    code = [{"prompt_tokens": 101371, "output_tokens": 64, "prompt_key": "real"}]
    conv = [{"prompt_tokens": 4096, "output_tokens": 512, "source_internal_row": 1}]
    trace = module.make_stress_trace(code, conv, 8192, .001, .025, 1800, 600, 20261024, "a" * 64)
    assert trace == module.make_stress_trace(code, conv, 8192, .001, .025, 1800, 600, 20261024, "a" * 64)
    assert trace["duration_s"] == 2400 and trace["evaluation_start_s"] == 600
    assert all(0 < mark["arrival_s"] < 2400 for mark in trace["requests"])
    assert all(mark["output_tokens"] == (64 if mark["model"] == 0 else 8192) for mark in trace["requests"])
    assert [round(mark["arrival_s"], 3) for mark in trace["requests"] if mark["model"] == 0] == [1231.721, 1331.572]


def test_measured_rate_decreases_with_residency_and_memory(module):
    marks = [{"prompt_tokens": 4096, "output_tokens": 512}]
    before = module.rate_from_profile(marks, 8192, 160)
    slower = module.rate_from_profile(marks, 8192, 320)
    longer = module.rate_from_profile(marks, 16384, 160)
    assert slower["conv_rate_rps"] == pytest.approx(before["conv_rate_rps"] / 2)
    assert longer["conv_rate_rps"] < before["conv_rate_rps"]
    assert before["conv_rate_rps"] * before["mean_full_reserved_mib"] * 160 == pytest.approx(4800)


@pytest.mark.parametrize("output", [0, -1, 1.5, True])
def test_invalid_synthetic_outputs_rejected(module, output):
    with pytest.raises(ValueError):
        module.long_output_marks([{"prompt_tokens": 1, "output_tokens": 1}], output)
