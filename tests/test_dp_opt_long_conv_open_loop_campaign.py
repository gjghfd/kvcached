import importlib
import json
import re
import sys
from pathlib import Path

import pytest


@pytest.fixture
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("long_conv_open_loop_campaign")


def test_parent_barrier_requires_native_SHA_and_result_consumer_closure(campaign):
    state = {"GPU_experiment_complete": True, "state_last_observed": "COMPLETED", "job": 269567,
             "phase": "robust_native_pairs_SHA_verified_analyzed"}
    report = {"job": 269567, "status": "complete_native_report_not_final_Poisson_research_goal"}
    assert campaign.parent_ready(state, report)
    assert not campaign.parent_ready(state, {})
    assert not campaign.parent_ready({**state, "state_last_observed": "RUNNING"}, report)
    assert not campaign.parent_ready(state, {**report, "job": 268245})
    assert not campaign.parent_ready({**state, "GPU_experiment_complete": False}, report)


def test_one_A800_submission_is_ledger_guarded_and_separate_from_native(campaign):
    command = campaign.submission_command({"DP_OPT_OPEN_LOOP_UPLOAD": "/protected upload"}, "/ledger/submitted.txt")
    assert command.count("sbatch --parsable") == 1
    assert "test ! -e /ledger/submitted.txt" in command
    assert "squeue -h -u" in command
    assert "long-conv-open-loop.sbatch" in command
    assert "qwen-robust-native" not in command
    assert "'/protected upload'" in command


def test_next_stage_sync_keeps_frozen_native_dependency_set(campaign):
    previous = importlib.import_module("workaware_robust_live_campaign")
    assert set(previous.SYNC_FILES) <= set(campaign.SYNC_FILES)
    assert len(campaign.SYNC_FILES) == len(set(campaign.SYNC_FILES)) == 114
    assert "experiments/dp_opt/record_robust_native_result.py" not in campaign.SYNC_FILES


def test_slurm_driver_freezes_default_graph_budget_Code_and_single_policy(campaign):
    path = Path(campaign.__file__).with_name("long-conv-open-loop.sbatch")
    text = path.read_text()
    assert "#SBATCH --partition=gpu_a800" in text and "#SBATCH --gpus=2" in text
    assert "DP_OPT_TRACE_POLICIES=no_move" in text
    assert "DP_OPT_PAUSE_SYNC_INTERVAL=32" in text and "DP_OPT_TRACE_BUDGET_MIB=10240" in text
    assert "DP_OPT_MAX_MODEL_LEN=131072" in text and "KVCACHED_GPU_UTILIZATION=0.98" in text
    assert 'DP_OPT_WORKAWARE_RATE_PRIORS="0.001 ${rates[$ordinal]}"' in text
    assert "for ordinal in 0 1 2" in text and "verify_trace.py" in text
    assert "ENFORCE_EAGER" not in text and "srun" not in text


def test_waiting_parent_only_reads_existing_state_and_never_queries_GPU(campaign, tmp_path, monkeypatch):
    root, parent = tmp_path / "next", tmp_path / "parent"
    root.mkdir()
    parent.mkdir()
    previous = {"phase": "submitted_native_robust_pairs", "GPU_experiment_complete": False,
                "state_last_observed": "RUNNING", "job": 269567}
    (parent / "campaign-state.json").write_text(json.dumps(previous))
    state = {"repo": str(tmp_path), "frozen_runtime_sha256": {}}
    monkeypatch.setattr(campaign, "checked_ssh", lambda *args: pytest.fail("No SSH while native is active"))
    monkeypatch.setattr(campaign, "protected_sync", lambda *args, **kwargs: pytest.fail("No active-source sync"))
    def stop_after_one_interval(seconds):
        assert seconds == 300
        raise RuntimeError("bounded test interval")
    monkeypatch.setattr(campaign.time, "sleep", stop_after_one_interval)
    with pytest.raises(RuntimeError, match="bounded test"):
        campaign.await_parent(root, parent, state)
    saved = json.loads((root / "campaign-state.json").read_text())
    assert saved["parent_job"] == 269567 and not saved["SSH_queries_while_waiting"]


def test_parent_failure_stops_next_stage_without_automatic_allocation(campaign, tmp_path, monkeypatch):
    root, parent = tmp_path / "next", tmp_path / "parent"
    root.mkdir()
    parent.mkdir()
    (parent / "campaign-state.json").write_text(json.dumps({"phase": "diagnosis_required"}))
    monkeypatch.setattr(campaign, "checked_ssh", lambda *args: pytest.fail("No SSH after parent failure"))
    with pytest.raises(RuntimeError, match="Native failure"):
        campaign.await_parent(root, parent, {"repo": str(tmp_path), "frozen_runtime_sha256": {}})


def test_no_move_driver_passes_actual_live_cli_optimizer_contract(campaign, tmp_path, monkeypatch):
    live = importlib.import_module("live_trace")
    text = Path(campaign.__file__).with_name("long-conv-open-loop.sbatch").read_text()
    optimizer = re.search(r"DP_OPT_WORKAWARE_OPTIMIZER=(\w+)", text).group(1)
    assert optimizer == "point"
    observed = []
    async def collect(args):
        observed.append(args)
    monkeypatch.setattr(live, "run", collect)
    monkeypatch.setattr(sys, "argv", ["live_trace.py", "--checkpoint", "test-checkpoint", "--fit", "test-fit",
        "--trace", "test-trace", "--output", str(tmp_path), "--policies", "no_move",
        "--workaware-optimizer", optimizer, "--max-model-len", "131072", "--kv-budget-mib", "10240"])
    live.main()
    assert observed[0].policies == ["no_move"] and observed[0].workaware_optimizer == "point"


def test_robust_without_workaware_policy_is_rejected_before_any_model_start(campaign, tmp_path, monkeypatch):
    live = importlib.import_module("live_trace")
    async def unexpected(args):
        pytest.fail("Invalid optimizer/policy must fail before model initialization")
    monkeypatch.setattr(live, "run", unexpected)
    monkeypatch.setattr(sys, "argv", ["live_trace.py", "--checkpoint", "test-checkpoint", "--fit", "test-fit",
        "--trace", "test-trace", "--output", str(tmp_path), "--policies", "no_move", "--workaware-optimizer", "robust"])
    with pytest.raises(SystemExit) as error:
        live.main()
    assert error.value.code == 2
