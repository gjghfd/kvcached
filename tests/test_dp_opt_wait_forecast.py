import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def forecast(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("wait_forecast")


def observed():
    empty = {"requests": [], "pages": [{"id": 0, "owners": [], "pinned": True}], "page_bytes": 1024**3,
             "page_block_capacity": 64, "block_size": 16, "free_blocks": 10000}
    result = [[copy.deepcopy(empty) for rank in (0, 1)] for model in (0, 1)]
    for rank in (0, 1):
        result[1][rank]["requests"] = [{"id": f"conv-{rank}", "blocks": list(range(64, 96)),
                                        "estimated_remaining_decode_s": 1}]
        result[1][rank]["pages"].append({"id": 1, "owners": [f"conv-{rank}"], "pinned": False})
    return result


def context(rate):
    return {"incoming_request_id": "Code-0", "competing_model": 1, "arrival_rate_rps": rate, "as_of": 100,
            "samples": [{"request_id": f"old-{index}", "reserved_at": 0, "completed_at": 100,
                         "blocks": 32, "service_s": 100} for index in range(16)]}


def test_future_births_can_pin_shared_pages_after_current_owner_completes(forecast):
    low = forecast.forecast_wait(observed(), 0, 4096, [5, 5], context(.001))
    high = forecast.forecast_wait(observed(), 0, 4096, [5, 5], context(10))
    assert low["estimated_wait_s"] == 1
    assert high["estimated_wait_s"] > low["estimated_wait_s"]
    assert high == forecast.forecast_wait(observed(), 0, 4096, [5, 5], context(10))


def test_shadow_pool_keeps_shared_and_pinned_owners_until_actual_release(forecast):
    observation = observed()[1][0]
    pool = forecast.Pool(observation)
    pool.allocate("new", 64)
    assert "new" in pool.pages[1]["owners"]
    pool.remove("conv-0")
    assert 1 in pool.pages and pool.pages[1]["owners"]
    pool.remove("new")
    assert set(pool.pages) == {0}
    assert pool.free(0) == 63


@pytest.mark.parametrize("field,value", [("arrival_rate_rps", 0), ("arrival_rate_rps", float("nan")),
                                         ("competing_model", 0), ("as_of", 99)])
def test_invalid_or_future_forecast_context_rejected(forecast, field, value):
    options = context(1)
    options[field] = value
    with pytest.raises(ValueError):
        forecast.forecast_wait(observed(), 0, 4096, [5, 5], options)


def test_unknown_current_rate_preserves_legacy_fallback(forecast):
    observations = observed()
    observations[1][0]["requests"][0]["estimated_remaining_decode_s"] = None
    assert forecast.forecast_wait(observations, 0, 4096, [5, 5], context(1))["status"].startswith("unavailable")


def test_current_direct_fit_is_zero_wait(forecast):
    result = forecast.forecast_wait(observed(), 0, 16, [5, 5], context(1))
    assert result["estimated_wait_s"] == 0 and result["censored_paths"] == 0


def test_forecast_context_selects_only_recent_completed_past_joint_marks(forecast):
    records = [{"request_id": f"request-{index}", "model": 1, "reserved_at": index - 1,
                "completed_at": index, "prompt_tokens": 2048, "token_ids": [1] * 64} for index in range(100)]
    options = forecast.forecast_context("Code", 0, records, [.004, .61], observed(), 70)
    assert len(options["samples"]) == 64 and options["samples"][-1]["completed_at"] == 70
    assert options["samples"][0]["completed_at"] == 7
    assert all(sample["service_s"] == 1 and sample["blocks"] == 132 for sample in options["samples"])
    assert forecast.forecast_context("Code", 0, records, [.004, .61], observed(), 10) is None
