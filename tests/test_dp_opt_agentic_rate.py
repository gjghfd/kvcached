import importlib
from pathlib import Path

import pytest


@pytest.fixture
def rate_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("agentic_rate")


def pilot():
    return {"status": "passed", "experiments": [
        {"population": population, "conv_mean_service_s": 10 + .4 * population,
         "conv_mark": {"output_tokens": 640}, "code_service_s": 20}
        for population in (16, 32, 48)]}


def test_rates_preserve_memory_service_correlation(rate_module):
    marks = [{"prompt_tokens": 2048, "output_tokens": 512, "count": 2},
             {"prompt_tokens": 4096, "output_tokens": 1024, "count": 1}]
    result = rate_module.calculate_rates(pilot(), marks)
    assert result["intercept_s"] == pytest.approx(10)
    assert result["slope_s_per_request"] == pytest.approx(.4)
    assert result["service_biased_resident_memory_mib"] == pytest.approx(360)
    assert result["code_rate_rps"] == .012
    assert result["population_extrapolated"]
    assert result["conv_rate_rps"] <= result["conv_rate_raw_rps"]


def test_rates_require_successful_pilot(rate_module):
    data = pilot()
    data["status"] = "failed"
    with pytest.raises(ValueError, match="successful pilot"):
        rate_module.calculate_rates(data, [{"prompt_tokens": 16, "output_tokens": 16}])


def test_rates_require_distinct_population(rate_module):
    data = pilot()
    data["experiments"] = [data["experiments"][0]] * 2
    with pytest.raises(ValueError, match="distinct"):
        rate_module.calculate_rates(data, [{"prompt_tokens": 16, "output_tokens": 16}])
