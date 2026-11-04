import copy
import importlib
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("workaware_robust_live_campaign")


def completed_parent():
    return ({"GPU_experiment_complete": True, "state_last_observed": "COMPLETED",
             "phase": "long_Conv_probe_SHA_verified_analyzed"},
            {"phase": "complete_independent_candidate_replay_not_GPU_benefit"},
            {"status": "all_three_probe_analyses_verified_not_research_goal_complete",
             "phases": {f"probe-{seed}": {} for seed in (20261030, 20261031, 20261032)}})


def test_only_complete_gpu_and_both_followups_release_the_parent_barrier(campaign):
    assert campaign.parent_ready(*completed_parent())


@pytest.mark.parametrize("missing", ["GPU", "Slurm", "main_analysis", "followup", "phase_analysis", "third_seed"])
def test_incomplete_parent_never_permits_source_promotion_or_submission(campaign, missing):
    state, followup, analysis = copy.deepcopy(completed_parent())
    if missing == "GPU":
        state["GPU_experiment_complete"] = False
    elif missing == "Slurm":
        state["state_last_observed"] = "RUNNING"
    elif missing == "main_analysis":
        state["phase"] = "archiving"
    elif missing == "followup":
        followup["phase"] = "waiting_for_complete_GPU_archive"
    elif missing == "phase_analysis":
        analysis["status"] = "verifying"
    else:
        analysis["phases"].pop("probe-20261032")
    assert not campaign.parent_ready(state, followup, analysis)


def test_native_submission_uses_one_a800_request_and_protected_user_fork(campaign):
    command = campaign.submission_command({"DP_OPT_ROBUST_PARENT_JOB": "268245"}, "/tmp/ledger")
    assert command.count("sbatch ") == 1
    assert 'test -z "$(squeue -h -u "$USER")"' in command
    assert "test ! -e /tmp/ledger" in command
    assert "git@github.com:gjghfd/kvcached.git" in command
    assert "branch --show-current" in command and "main" in command
    assert "workaware-robust-live.sbatch" in command
    assert "git push" not in command and "scancel" not in command


def test_sync_inventory_is_explicit_unique_and_contains_live_optimizer_dependencies(campaign):
    repo = Path(__file__).resolve().parents[1]
    assert len(campaign.SYNC_FILES) == len(set(campaign.SYNC_FILES))
    assert all((repo / name).is_file() for name in campaign.SYNC_FILES)
    assert "experiments/dp_opt/workaware_robust_adapter.py" in campaign.SYNC_FILES
    assert "experiments/dp_opt/verify_workaware_robust_rebalance.py" in campaign.SYNC_FILES
    assert "tests/test_dp_opt_zero_arrival_trace.py" in campaign.SYNC_FILES


def test_native_batch_script_keeps_default_graph_and_real_paused_copy(campaign):
    path = Path(campaign.__file__).with_name("workaware-robust-live.sbatch")
    text = path.read_text()
    assert "#SBATCH --partition=gpu_a800" in text and "#SBATCH --gpus=2" in text
    assert "DP_OPT_PAUSE_SYNC_INTERVAL=32" in text and "DP_OPT_KV_TRANSPORT=dp_echo" in text
    assert "DP_OPT_WORKAWARE_OPTIMIZER=robust" in text
    assert "DP_OPT_TRACE_BUDGET_MIB=10240" in text and "KVCACHED_GPU_UTILIZATION=0.98" in text
    assert "DP_OPT_MAX_MODEL_LEN=131072" in text
    assert "enforce-eager" not in text and "capture-sizes" not in text
    subprocess.run(["bash", "-n", str(path)], check=True)


@pytest.mark.parametrize("optimizer", ["robust", "unknown"])
def test_cli_rejects_missing_workaware_policy_or_unknown_optimizer_before_importing_gpu(campaign, optimizer, tmp_path):
    path = Path(campaign.__file__).with_name("live_trace.py")
    result = subprocess.run([sys.executable, str(path), "--checkpoint", "missing", "--fit", "missing",
        "--trace", "missing", "--output", str(tmp_path), "--policies", "no_move", "--workaware-optimizer", optimizer],
        text=True, capture_output=True, timeout=30)
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "requires the workaware policy" in result.stderr or "invalid choice" in result.stderr
