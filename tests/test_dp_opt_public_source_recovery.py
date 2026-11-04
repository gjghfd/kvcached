import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def recovery(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    module = importlib.import_module("public_source_recovery")
    monkeypatch.setattr(module, "process_alive", lambda pid: False)
    return module


def failed_parent(tmp_path):
    root = tmp_path / "parent"
    raw = root / "live-trace-11"
    raw.mkdir(parents=True)
    (raw / "evidence.json").write_text('{"failed":true}\n')
    digest = hashlib.sha256((raw / "evidence.json").read_bytes()).hexdigest()
    (root / "raw-11.sha256").write_text(f"{digest}  evidence.json\n")
    (root / "campaign-state.json").write_text(json.dumps({"job": 11, "state_last_observed": "FAILED",
                                                          "GPU_calibration_complete": False}))
    (root / "archive-11.json").write_text(json.dumps({"state": "FAILED", "watcher_barrier": "shared_markers",
                                                      "raw_files_verified": 1}))
    (root / "local-monitor-identity.json").write_text('{"pid":12345}\n')
    (root / "live-trace-11.err").write_text("live_trace.py complete\nKeyError: 'source_internal_row'\n")
    return root


def test_closed_failed_parent_remains_failed_and_sha_verified(recovery, tmp_path):
    root = failed_parent(tmp_path)
    result = recovery.verify_closed_failure(root)
    assert result == {"job": 11, "state": "FAILED", "raw_files": 1,
                      "SHA_verified": True, "not_reclassified_as_success": True}
    assert json.loads((root / "campaign-state.json").read_text())["GPU_calibration_complete"] is False


def test_alive_parent_prevents_recovery(recovery, tmp_path, monkeypatch):
    root = failed_parent(tmp_path)
    monkeypatch.setattr(recovery, "process_alive", lambda pid: True)
    with pytest.raises(RuntimeError, match="still alive"):
        recovery.verify_closed_failure(root)


def test_failed_raw_sha_cannot_be_bypassed(recovery, tmp_path):
    root = failed_parent(tmp_path)
    (root / "live-trace-11/evidence.json").write_text("changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        recovery.verify_closed_failure(root)


def test_unrelated_oom_requires_different_diagnosis(recovery, tmp_path):
    root = failed_parent(tmp_path)
    (root / "live-trace-11.err").write_text("CUDA out of memory")
    with pytest.raises(RuntimeError, match="only addresses"):
        recovery.verify_closed_failure(root)


def test_failed_parent_cannot_be_labeled_calibration_success(recovery, tmp_path):
    root = failed_parent(tmp_path)
    (root / "campaign-state.json").write_text(json.dumps({"job": 11, "state_last_observed": "FAILED",
                                                          "GPU_calibration_complete": True}))
    with pytest.raises(RuntimeError, match="diagnosed failed parent"):
        recovery.verify_closed_failure(root)


def verifier_failed_parent(tmp_path):
    root = failed_parent(tmp_path)
    raw = root / "live-trace-11"
    phase = "native-2026103101-N2"
    native = raw / phase
    native.mkdir()
    (native / "exit-status").write_text("0\n")
    (raw / "phase.json").write_text(json.dumps({"phase": phase}))
    (raw / "public-context-fit.json").write_text('{"verified":true}\n')
    (root / "live-trace-11.err").write_text("verify_trace.py\nKeyError: 'traj_id'\n")
    paths = sorted(path for path in raw.rglob("*") if path.is_file())
    (root / "raw-11.sha256").write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(raw)}\n" for path in paths))
    (root / "archive-11.json").write_text(json.dumps({"state": "FAILED", "watcher_barrier": "shared_markers",
                                                      "raw_files_verified": len(paths)}))
    return root


def mock_native_verification(recovery, monkeypatch, policies=None):
    policies = policies if policies is not None else ["no_move", "random_no_return", "kv_only_headroom", "kv_only_workaware"]
    result = {"status": "raw_trace_verified", "trials": [{"policy": policy} for policy in policies],
              "trace_sha256": "f" * 64}
    monkeypatch.setattr(recovery, "verify", lambda root, bank: result)
    monkeypatch.setattr(recovery, "verify_profile", lambda root: {"verified": True})


def test_prompt_bank_recovery_reverifies_drained_native_and_keeps_failure(recovery, tmp_path, monkeypatch):
    root = verifier_failed_parent(tmp_path)
    mock_native_verification(recovery, monkeypatch)
    result = recovery.verify_closed_failure(root)
    assert result["failure_kind"] == "public_prompt_bank_verifier_schema"
    assert result["native_policy_count"] == 4
    assert result["cost_profile_independently_verified"] is True
    assert result["not_reclassified_as_success"] is True
    assert json.loads((root / "campaign-state.json").read_text())["state_last_observed"] == "FAILED"


def test_prompt_bank_recovery_rejects_missing_native_policy(recovery, tmp_path, monkeypatch):
    root = verifier_failed_parent(tmp_path)
    mock_native_verification(recovery, monkeypatch, ["no_move"])
    with pytest.raises(RuntimeError, match="every native policy"):
        recovery.verify_closed_failure(root)


def test_prompt_bank_recovery_rejects_independent_profile_difference(recovery, tmp_path, monkeypatch):
    root = verifier_failed_parent(tmp_path)
    mock_native_verification(recovery, monkeypatch)
    monkeypatch.setattr(recovery, "verify_profile", lambda root: {"verified": False})
    with pytest.raises(RuntimeError, match="cost profile differs"):
        recovery.verify_closed_failure(root)


def test_prompt_bank_recovery_propagates_native_verification_failure(recovery, tmp_path, monkeypatch):
    root = verifier_failed_parent(tmp_path)
    mock_native_verification(recovery, monkeypatch)

    def reject(root, bank):
        raise ValueError("ownership invalid")

    monkeypatch.setattr(recovery, "verify", reject)
    with pytest.raises(ValueError, match="ownership invalid"):
        recovery.verify_closed_failure(root)
