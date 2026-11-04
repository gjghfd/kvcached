import importlib
from pathlib import Path

import pytest


@pytest.fixture
def followup(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("public_poisson_followup")


def test_rate_grid_is_finite_and_capacity_selected_not_benefit_selected(followup):
    cases = {name: {"eligible_for_separate_evaluation_rate_candidate": eligible,
                   "physical_pressure": {"local_physical_ratio": pressure}, "rate_rps": [.001, rate]}
             for name, eligible, pressure, rate in (("rate-0", True, 1.11, .008),
                                                   ("rate-1", True, 1.2, .012),
                                                   ("rate-2", False, 1.4, .02))}
    grid = followup.search_rates(cases, "rate-1")
    assert len(grid) == 4
    assert {setting["code_rate"] for setting in grid} == {.0005, .001}
    assert {setting["conv_rate"] for setting in grid} == {.008, .012}
    assert all(setting["calibration_case"] != "rate-2" for setting in grid)
    with pytest.raises(ValueError):
        followup.search_rates(cases, None)


def metrics_fixture():
    measurement = {"mean_ttft_s": 10., "p95_ttft_s": 20., "p99_ttft_s": 30., "mean_tpot_s": .02}
    return {policy: {group: {"measurement": dict(measurement)} for group in ("Code", "Conv", "combined")}
            for policy in ("no_move", "kv_only_workaware", "random_no_return")}


def test_selection_requires_return_proxy_not_only_code_admission_gain(followup):
    metrics = metrics_fixture()
    metrics["kv_only_workaware"]["Code"]["measurement"]["mean_ttft_s"] = 1.
    assert followup.score(metrics) is None
    metrics["kv_only_workaware"]["Conv"]["measurement"]["mean_tpot_s"] = .018
    assert followup.score(metrics) > 0


def test_negative_conv_tails_penalize_score_even_when_code_improves(followup):
    metrics = metrics_fixture()
    metrics["kv_only_workaware"]["Conv"]["measurement"]["mean_tpot_s"] = .018
    original = followup.score(metrics)
    metrics["kv_only_workaware"]["Conv"]["measurement"]["p99_ttft_s"] = 300.
    assert followup.score(metrics) < original


def test_empty_code_draw_cannot_receive_positive_score(followup):
    metrics = metrics_fixture()
    metrics["kv_only_workaware"]["Conv"]["measurement"]["mean_tpot_s"] = .018
    metrics["kv_only_workaware"]["Code"]["measurement"]["p99_ttft_s"] = None
    assert followup.score(metrics) is None


def test_baselines_and_fixed_heldout_preserve_default_pause_and_public_prompts(followup):
    assert len(followup.POLICIES) == 7
    assert followup.HELDOUT_SEED == 2026103105
    assert followup.HELDOUT_S == 10800
    batch = (Path(followup.__file__).parent / "public-poisson-search.sbatch").read_text()
    assert "DP_OPT_PAUSE_SYNC_INTERVAL=32" in batch
    assert "KVCACHED_GPU_UTILIZATION=0.98" in batch
    assert "heldout-trace.json" in batch
    assert "agentic-100k-input/prompt-bank.json" not in batch
    assert "DP_OPT_WORKAWARE_OPTIMIZER=robust" in batch
    command = followup.submit_command({}, "/one-ledger")
    assert "git@github.com:gjghfd/kvcached.git" in command
    assert 'squeue -h -u "$USER"' in command
    assert "--time=80:00:00" in command
    assert "public-population-calibration" in batch
    assert "DP_OPT_WORKAWARE_OPTIMIZER=point" in batch
    assert "--anchor-supplement" in batch
    assert "public_population_calibration.py" in batch
    assert batch.index("run_phase public-population-calibration") < batch.index("for ((ordinal=0;")
    assert "public_population_calibration.py" in " ".join(followup.SYNC_FILES)
