import importlib
from pathlib import Path

import pytest


@pytest.fixture
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("long_conv_return_campaign")


def test_waiting_previous_is_not_ready(campaign):
    assert not campaign.previous_ready({"GPU_experiment_complete": False, "state_last_observed": "RUNNING"})


def test_complete_flag_without_full_verification_phase_is_not_ready(campaign):
    assert not campaign.previous_ready({"GPU_experiment_complete": True, "state_last_observed": "COMPLETED", "phase": "archiving"})


def test_failed_previous_blocks_successor(campaign):
    with pytest.raises(RuntimeError, match="diagnose"):
        campaign.previous_ready({"phase": "diagnosis_required_no_automatic_duplicate_submission"})


def test_successor_requires_terminal_and_complete_verification(campaign):
    assert campaign.previous_ready({"GPU_experiment_complete": True, "state_last_observed": "COMPLETED",
                           "phase": "early_return_lower_rate_SHA_verified_analyzed"})


def test_submission_is_unique_A800_not_Aegaeon_and_preserves_ledger(campaign):
    command = campaign.submission_command({"DP_OPT_LONG_CONV_UPLOAD": "/frozen input"}, "/ledger/job.txt")
    assert 'test -z "$(squeue' in command
    assert "test ! -e /ledger/job.txt" in command
    assert "git@github.com:gjghfd/kvcached.git" in command
    assert "long-conv-return-probe.sbatch" in command
    assert "--time=12:00:00" in command
    assert "'/frozen input'" in command
    assert "git push" not in command
