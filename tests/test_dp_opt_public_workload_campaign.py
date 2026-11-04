import importlib
from pathlib import Path

import pytest


@pytest.fixture
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("public_workload_campaign")


def test_public_job_has_one_a800_allocation_and_user_fork_guard(campaign):
    command = campaign.submission_command({"DP_OPT_PUBLIC_UPLOAD": "/upload"}, "/unique-ledger")
    assert command.count("sbatch") == 2
    assert 'squeue -h -u "$USER"' in command
    assert "git@github.com:gjghfd/kvcached.git" in command
    assert "test ! -e /unique-ledger" in command
    assert "--time=18:00:00" in command
    assert "public-workload-search.sbatch" in command


def test_sync_inventory_keeps_existing_runtime_and_adds_public_bank_contract(campaign):
    assert len(campaign.SYNC_FILES) == len(set(campaign.SYNC_FILES))
    for name in ("experiments/dp_opt/public_response_inputs.py", "experiments/dp_opt/public_workload_search.py",
                 "experiments/dp_opt/workaware_robust_ilp.py", "tests/test_dp_opt_public_workload_search.py"):
        assert name in campaign.SYNC_FILES


def test_batch_keeps_graph_pause_allocator_and_rebuilds_original_prompts(campaign):
    batch = (Path(campaign.__file__).parent / "public-workload-search.sbatch").read_text()
    assert "#SBATCH --partition=gpu_a800" in batch
    assert "DP_OPT_PAUSE_SYNC_INTERVAL=32" in batch
    assert "DP_OPT_WORKAWARE_OPTIMIZER=robust" in batch
    assert "KVCACHED_GPU_UTILIZATION=0.98" in batch
    assert 'DP_OPT_PROMPT_BANK="$evidence/frozen/inputs/prompt-bank.json"' in batch
    assert "--action validate" in batch
    assert "--action freeze-calibration" in batch
    assert "long_output_marks" not in batch
    assert "24576" not in batch
