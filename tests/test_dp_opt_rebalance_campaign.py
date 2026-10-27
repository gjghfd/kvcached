import hashlib
import importlib
import json
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("rebalance_campaign")


def config():
    return {"max_model_len": 131072, "cache_bytes": 25769803776, "deadline_s": 120,
            "max_moves": 8, "candidate_limit": 8, "kv_budget_mib": 10240,
            "move_fixed_s": 1.77, "move_bandwidth_gib_s": 11.64, "seed": 20261024,
            "frozen_trace_sha256": "a" * 64}


def test_frozen_environment_preserves_measured_case(campaign):
    environment = campaign.frozen_environment(config(), "/trace/paired.json")
    assert environment["DP_OPT_REBALANCE_BUDGET_MIB"] == "10240"
    assert environment["DP_OPT_TRACE_SEED"] == "20261024"
    assert environment["DP_OPT_MOVE_FIXED_S"] == "1.77"
    assert environment["DP_OPT_REBALANCE_TRACE_SHA256"] == "a" * 64


@pytest.mark.parametrize("field,value", [("max_model_len", 32768), ("cache_bytes", 1), ("deadline_s", 200),
                                       ("max_moves", 2), ("candidate_limit", 2)])
def test_frozen_environment_rejects_silent_protocol_change(campaign, field, value):
    changed = config()
    changed[field] = value
    with pytest.raises(ValueError):
        campaign.frozen_environment(changed, "/trace.json")


def test_submission_has_durable_single_job_ledger_and_safe_shell(campaign):
    command = campaign.submission_command(campaign.frozen_environment(config(), "/a path/trace.json"), "/ledger/job.txt")
    subprocess.run(["bash", "-n"], input=command, text=True, check=True)
    assert 'test -z "$(squeue -h -u "$USER")"' in command
    assert "test ! -e /ledger/job.txt" in command and "> /ledger/job.txt" in command
    assert "gjghfd/kvcached.git" in command and "rebalance-trace.sbatch" in command
    assert "stress-sweep.sbatch" not in command and command.count("sbatch --parsable") == 1
    assert "'/a path/trace.json'" in command


def test_handoff_refuses_unrelated_or_reused_pid(campaign, monkeypatch, tmp_path):
    monkeypatch.setattr(campaign, "process_command", lambda pid: "Ss unrelated-daemon")
    terminate = Mock()
    monkeypatch.setattr(campaign.os, "killpg", terminate)
    with pytest.raises(RuntimeError, match="unrelated/reused"):
        campaign.handoff_old_campaign(1234, tmp_path)
    terminate.assert_not_called()


def test_handoff_targets_only_verified_isolated_old_campaign(campaign, monkeypatch, tmp_path):
    command = f"Ss python stress_campaign.py --archive {tmp_path.resolve()}"
    observations = iter([command, None])
    monkeypatch.setattr(campaign, "process_command", lambda pid: next(observations))
    monkeypatch.setattr(campaign.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(campaign.os, "getpgrp", lambda: 9876)
    terminate = Mock()
    monkeypatch.setattr(campaign.os, "killpg", terminate)
    result = campaign.handoff_old_campaign(1234, tmp_path)
    terminate.assert_called_once_with(1234, signal.SIGTERM)
    assert result["current_GPU_job_not_cancelled"] is True


def test_handoff_refuses_current_process_group(campaign, monkeypatch, tmp_path):
    monkeypatch.setattr(campaign, "process_command", lambda pid: f"Ss python stress_campaign.py --archive {tmp_path.resolve()}")
    monkeypatch.setattr(campaign.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(campaign.os, "getpgrp", lambda: 1234)
    with pytest.raises(RuntimeError, match="isolated"):
        campaign.handoff_old_campaign(1234, tmp_path)


def test_deployment_refuses_runtime_modified_after_launch(campaign, tmp_path):
    source = tmp_path / "runtime.py"
    source.write_text("changed")
    state = {"frozen_runtime_sha256": {"runtime.py": hashlib.sha256(b"original").hexdigest()}}
    with pytest.raises(RuntimeError, match="changed after coordinator start"):
        campaign.protected_sync(tmp_path, tmp_path, state)


def test_sync_scope_excludes_protected_data_planes_and_checkpoint(campaign):
    assert len(campaign.SYNC_FILES) == len(set(campaign.SYNC_FILES))
    assert "experiments/dp_opt/live_rebalance.py" in campaign.SYNC_FILES
    assert "experiments/dp_opt/ilp_reference.py" in campaign.SYNC_FILES
    assert all(filename.startswith(("experiments/dp_opt/", "tests/test_dp_opt_")) for filename in campaign.SYNC_FILES)


def test_checked_ssh_does_not_accept_failed_query(campaign, monkeypatch):
    monkeypatch.setattr(campaign, "ssh", lambda *args: SimpleNamespace(returncode=255, stdout="", stderr="SSH failure"))
    with pytest.raises(RuntimeError, match="255"):
        campaign.checked_ssh("read-only query")


def test_failed_old_job_cannot_advance_without_explicit_diagnosis(campaign, tmp_path):
    with pytest.raises(RuntimeError, match="explicit raw-verified diagnosis"):
        campaign.verify_old_failure(tmp_path, 123, "FAILED", None)


def test_old_failure_diagnosis_must_match_raw_manifest(campaign, tmp_path):
    (tmp_path / "raw-123.sha256").write_text("immutable manifest")
    diagnosis = tmp_path / "diagnosis.json"
    diagnosis.write_text(json.dumps({"job":123,"state":"FAILED","ready_for_new_method":True,
                                     "raw_manifest_sha256":"mismatched"}))
    with pytest.raises(RuntimeError, match="immutable raw archive"):
        campaign.verify_old_failure(tmp_path, 123, "FAILED", diagnosis)
