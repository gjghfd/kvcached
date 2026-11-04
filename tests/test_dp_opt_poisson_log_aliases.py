import importlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def follower(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("follow_poisson_log_aliases")


@pytest.mark.parametrize("job", [True, 0, -1, "269991", "1; exit 0", None])
def test_rejects_non_integer_or_unscoped_job(follower, job):
    with pytest.raises(ValueError):
        follower.alias_command(job)


@pytest.mark.parametrize("state", [{}, {"job": 42, "state_last_observed": "PENDING"},
                                  {"job": True, "state_last_observed": "RUNNING"}])
def test_pending_or_invalid_job_never_triggers_ssh(follower, state):
    assert not follower.ready(state)


def test_alias_command_preserves_actual_logs_and_is_idempotent(follower, tmp_path):
    logs = tmp_path / "run/kvcached-paracloud/logs"
    logs.mkdir(parents=True)
    for extension in ("out", "err"):
        (logs / f"long-poisson-42.{extension}").write_text(extension)
    environment = {**os.environ, "HOME": str(tmp_path)}
    for _ in range(2):
        result = subprocess.run(["bash", "-c", follower.alias_command(42)], env=environment,
                                text=True, capture_output=True)
        assert result.returncode == 0
        for extension in ("out", "err"):
            actual, alias = logs / f"long-poisson-42.{extension}", logs / f"live-trace-42.{extension}"
            assert actual.read_text() == extension and alias.samefile(actual)
            assert f"verified_log_alias|{extension}|" in result.stdout


@pytest.mark.parametrize("conflict", ["file", "wrong_symlink"])
def test_alias_command_refuses_existing_evidence(follower, tmp_path, conflict):
    logs = tmp_path / "run/kvcached-paracloud/logs"
    logs.mkdir(parents=True)
    actual, alias = logs / "long-poisson-42.out", logs / "live-trace-42.out"
    actual.write_text("original")
    if conflict == "file":
        alias.write_text("protected")
    else:
        alias.symlink_to("different.out")
    result = subprocess.run(["bash", "-c", follower.alias_command(42)],
                            env={**os.environ, "HOME": str(tmp_path)}, capture_output=True)
    assert result.returncode != 0 and actual.read_text() == "original"
    assert alias.read_text() == "protected" if conflict == "file" else os.readlink(alias) == "different.out"


@pytest.fixture
def campaign(tmp_path):
    root, parent = tmp_path / "followup", tmp_path / "campaign"
    parent.mkdir()
    (parent / "campaign-state.json").write_text(json.dumps({"phase": "submitted", "job": 42,
                                                            "state_last_observed": "RUNNING"}))
    return root, parent


def test_one_ssh_installs_logs_without_gpu_or_slurm_queries(follower, campaign, monkeypatch):
    calls = []
    def fake_run(command, **options):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="verified_log_alias|out|a|b\nverified_log_alias|err|c|d\n", stderr="")
    monkeypatch.setattr(follower.subprocess, "run", fake_run)
    root, parent = campaign
    follower.run(root, parent)
    status = json.loads((root / "followup-state.json").read_text())
    assert len(calls) == 1 and calls[0][-2] == "paracloud"
    assert "squeue" not in calls[0][-1] and "sbatch" not in calls[0][-1]
    assert status["status"] == "log_aliases_verified_not_experiment_completion"
    assert status["research_goal_complete"] is False
    with pytest.raises(FileExistsError):
        follower.run(root, parent)
    assert len(calls) == 1


@pytest.mark.parametrize("condition", ["PAUSE", "diagnosis_required"])
def test_pause_and_diagnosis_do_not_access_remote(follower, campaign, monkeypatch, condition):
    root, parent = campaign
    monkeypatch.setattr(follower.subprocess, "run", lambda *args, **kwargs: pytest.fail("No SSH allowed"))
    if condition == "PAUSE":
        (parent / "PAUSE").touch()
        follower.run(root, parent)
    else:
        (parent / "campaign-state.json").write_text(json.dumps({"phase": condition}))
        with pytest.raises(RuntimeError, match="diagnosis"):
            follower.run(root, parent)


def test_failed_receipt_is_preserved_without_retry(follower, campaign, monkeypatch):
    root, parent = campaign
    monkeypatch.setattr(follower.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(returncode=2, stdout="partial", stderr="protected alias"))
    with pytest.raises(RuntimeError, match="Both actual"):
        follower.run(root, parent)
    assert (root / "alias-receipt.log").read_text() == "partialprotected alias"
    assert json.loads((root / "followup-state.json").read_text())["status"] == "diagnosis_required_no_automatic_retry"
