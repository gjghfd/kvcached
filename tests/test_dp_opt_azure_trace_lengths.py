import importlib
import math
from pathlib import Path

import pytest


@pytest.fixture
def lengths(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("azure_trace_lengths")


def test_32k_support_preserves_paired_lengths(lengths):
    rows = [(3072, 512), (8192, 768), (32256, 512), (32257, 512)]
    result = lengths.analyze_lengths({"conv": rows}, 32768)["datasets"]["conv"]
    assert result["supported"]["requests"] == 3
    assert result["supported"]["mean_context_times_output"] == (3072 * 512 + 8192 * 768 + 32256 * 512) / 3
    assert result["longer_decode_slice"]["requests"] == 3
    historical = lengths.analyze_lengths({"conv": rows}, 4096)["datasets"]["conv"]
    assert historical["supported"]["requests"] == 1


def test_kv_service_integral_is_joint_mean(lengths):
    rows = [(2048, 256), (8192, 512)]
    result = lengths.analyze_lengths({"conv": rows}, 32768)["datasets"]["conv"]["supported"]
    integral = result["service_assumptions"][1]["mean_kv_mib_seconds"]
    expected = sum(math.ceil((context + output) / 16) * 1.5 * (.2 + .04 * output)
                   for context, output in rows) / len(rows)
    assert integral == expected
    assert integral != result["mean_reserved_mib"] * result["service_assumptions"][1]["mean_resident_s"]


@pytest.mark.parametrize("limit", (0, -1))
def test_analysis_rejects_nonpositive_length(lengths, limit):
    with pytest.raises(ValueError, match="positive"):
        lengths.analyze_lengths({"conv": [(2048, 256)]}, limit)


def test_empty_slice_is_reported_without_fabricated_statistics(lengths):
    result = lengths.analyze_lengths({"conv": [(64, 16)]}, 32768)["datasets"]["conv"]
    assert result["long_slice"] == {"requests": 0}
    assert result["longer_decode_slice"] == {"requests": 0}
