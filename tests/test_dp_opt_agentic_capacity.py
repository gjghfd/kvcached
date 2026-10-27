import importlib
from pathlib import Path

import pytest


@pytest.fixture
def capacity(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("agentic_capacity")


def memory(free=12000, mapped=192):
    return {"gpu_memory": [{"total_mib": 81920, "free_mib": free},
                           {"total_mib": 81920, "free_mib": free + 100}], "mapped_mib": [mapped, mapped]}


def test_capacity_uses_minimum_headroom_over_all_samples(capacity):
    code = [{"prompt_tokens": 101675, "output_tokens": 64}]
    setting = capacity.capacity_setting([memory(), memory(9000, 3000)], code)
    assert setting["kv_budget_mib_per_gpu"] == 10944
    assert setting["non_KV_guard_mib"] == 1024
    lower, upper = setting["ideal_conv_fragmentation_window_mib"]
    target = setting["target_conv_reserved_mib"]
    assert lower < target < upper
    assert setting["target_offset_above_local_blocking_boundary_mib"] == 768
    assert setting["longest_code_logical_mib"] == 9538.5


def test_capacity_rejects_unfit_100k_instead_of_truncating(capacity):
    with pytest.raises(MemoryError, match="longest 100K"):
        capacity.capacity_setting([memory(9000)], [{"prompt_tokens": 101675, "output_tokens": 64}])


def test_capacity_ceiling_is_page_aligned(capacity):
    setting = capacity.capacity_setting([memory(40000)], [{"prompt_tokens": 100000, "output_tokens": 64}])
    assert setting["kv_budget_mib_per_gpu"] == 14304
    assert setting["kv_budget_mib_per_gpu"] % 96 == 0


def test_runtime_float_page_geometry_exports_integer_cli_budget(capacity):
    setting = capacity.capacity_setting([memory()], [{"prompt_tokens": 101675, "output_tokens": 64}],
                                        page_mib=96.0)
    assert type(setting["kv_budget_mib_per_gpu"]) is int
    assert int(str(setting["kv_budget_mib_per_gpu"])) == 11136


def test_observed_a800_headroom_accounts_for_allocator_reserve(capacity):
    sample = {"gpu_memory": [{"free_mib": 12962, "total_mib": 81152.75},
                              {"free_mib": 13805.125, "total_mib": 81152.75}],
              "mapped_mib": [192, 192]}
    code = [{"prompt_tokens": 101675, "output_tokens": 64}]
    with pytest.raises(MemoryError, match="longest 100K"):
        capacity.capacity_setting([sample], code, allocator_gpu_utilization=.95)
    setting = capacity.capacity_setting([sample], code, allocator_gpu_utilization=.98)
    assert setting["kv_budget_mib_per_gpu"] == 11520
    assert setting["effective_non_KV_guard_mib_per_gpu"] == pytest.approx([1623.055] * 2)


@pytest.mark.parametrize("utilization", [0, -1, 1.1, float("nan")])
def test_rejects_invalid_allocator_utilization(capacity, utilization):
    with pytest.raises(ValueError):
        capacity.capacity_setting([memory()], [{"prompt_tokens": 100000, "output_tokens": 64}],
                                  allocator_gpu_utilization=utilization)


def test_rates_scale_with_target_and_measured_service(capacity):
    marks = [{"prompt_tokens": 4000, "output_tokens": 640, "count": 2}]
    setting = {"target_conv_reserved_mib": 6000}
    rates = capacity.calibration_rates(setting, marks, 20, 640)
    slower = capacity.calibration_rates(setting, marks, 40, 640)
    assert len(rates["conv_rates_rps"]) == 4
    assert slower["central_conv_rate_rps"] * 2 == rates["central_conv_rate_rps"]
    assert rates["multipliers"] == [.5, .8, 1.2, 1.6]


@pytest.mark.parametrize("samples", [[], [memory(-1)], [memory(90000)]])
def test_capacity_rejects_invalid_memory(capacity, samples):
    with pytest.raises(ValueError):
        capacity.capacity_setting(samples, [{"prompt_tokens": 100000, "output_tokens": 64}])
