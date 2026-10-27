import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("workaware_profile_campaign")


def test_submission_uses_only_profile_wrapper_and_preserves_safety_guards(module):
    environment = module.frozen_environment("/frozen trace.json", "a" * 64, "/old/cadence", .019)
    command = module.submission_command(environment, "/sync/submitted-job.txt")
    assert command.count("sbatch --parsable") == 1
    assert "workaware-profile-return.sbatch" in command
    assert "test -z" in command and "squeue -h" in command
    assert "remote get-url origin" in command and "gjghfd/kvcached.git" in command
    assert "DP_OPT_WORKAWARE_CALIBRATION_SOURCE=/old/cadence" in command
    assert "'/frozen trace.json'" in command


def test_runtime_inventory_includes_profile_verifier_and_token_reference_tests(module):
    assert "experiments/dp_opt/verify_workaware_profile.py" in module.SYNC_FILES
    assert "experiments/dp_opt/workaware_profile_cost.py" in module.SYNC_FILES
    assert "tests/test_dp_opt_trace_reference.py" in module.SYNC_FILES
    assert len(set(module.SYNC_FILES)) == len(module.SYNC_FILES)


@pytest.mark.parametrize("rate", (0, -1, float("inf"), float("nan")))
def test_rate_prior_requires_finite_positive_value(module, rate):
    with pytest.raises(ValueError, match="positive"):
        module.frozen_environment("/trace", "a" * 64, "/cadence", rate)


def test_summary_keeps_both_tenants_and_combined_request_average_tpot(module):
    metrics = {"ttft_s": {"mean_s": 1, "p95_s": 2, "p99_s": 3}, "tpot_s": {"mean_s": .01}}
    group = {"requests": 2, "baseline": {"metrics": metrics}, "treatment": {"metrics": metrics},
             "changes": {"ttft_s": {"mean_s": {"delta_s": 0}}}}
    summary = module.summarize_primary({"kv_only_headroom": {"groups": {name: group for name in ("Code", "Conv", "all_requests")}}})
    assert set(summary["kv_only_headroom"]) == {"code", "conv", "combined"}
    assert summary["kv_only_headroom"]["conv"]["treatment_request_average_tpot"]["mean_s"] == .01
