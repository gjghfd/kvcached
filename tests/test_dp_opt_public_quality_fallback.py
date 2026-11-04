import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def fallback(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    module = importlib.import_module("public_quality_fallback")
    monkeypatch.setattr(module, "process_alive", lambda pid: False)
    monkeypatch.setattr(module, "verify_closed", lambda root, field: {"job": 99})
    return module


@pytest.fixture
def roots(fallback, tmp_path):
    parent, recovery, future = (tmp_path / name for name in ("parent", "recovery", "future"))
    write(parent / "campaign-state.json", {"phase": "complete", "GPU_calibration_complete": True, "candidate_case": None})
    write(recovery / "campaign-state.json", {"phase": "diagnosis_required", "error": fallback.UNQUALIFIED})
    write(recovery / "local-monitor-identity.json", {"pid": 12345})
    return SimpleNamespace(parent=parent, recovery=recovery, future=future, root=tmp_path / "fallback", repo=tmp_path / "repo")


def test_only_independently_complete_unqualified_source_can_use_fallback(fallback, roots):
    assert fallback.readiness(roots.parent, roots.recovery, roots.future) == "complete_C0_unqualified_rate_not_GPU_failure"


def test_qualified_source_keeps_original_owner_and_never_starts_another_producer(fallback, roots):
    write(roots.parent / "campaign-state.json", {"phase": "complete", "GPU_calibration_complete": True, "candidate_case": "rate-1"})
    assert fallback.readiness(roots.parent, roots.recovery, roots.future) == "qualified_parent_owned_by_existing_recovery"


def test_running_calibration_waits_without_ssh_or_submission(fallback, roots):
    write(roots.parent / "campaign-state.json", {"phase": "running", "GPU_calibration_complete": False})
    assert fallback.readiness(roots.parent, roots.recovery, roots.future) == "waiting"


def test_live_original_owner_must_exit_before_fallback(fallback, roots, monkeypatch):
    monkeypatch.setattr(fallback, "process_alive", lambda pid: True)
    assert fallback.readiness(roots.parent, roots.recovery, roots.future) == "waiting_outer_exit"


@pytest.mark.parametrize("kind", ["failed_C0", "unknown_outer_failure", "existing_identity", "existing_state"])
def test_gpu_failure_unknown_exit_and_existing_owner_are_not_bypassed(fallback, roots, kind):
    if kind == "failed_C0":
        write(roots.parent / "campaign-state.json", {"phase": "diagnosis_required", "GPU_calibration_complete": False})
    elif kind == "unknown_outer_failure":
        write(roots.recovery / "campaign-state.json", {"phase": "diagnosis_required", "error": "CUDA OOM"})
    else:
        write(roots.future / ("local-monitor-identity.json" if kind == "existing_identity" else "campaign-state.json"), {})
    with pytest.raises(RuntimeError):
        fallback.readiness(roots.parent, roots.recovery, roots.future)


def test_frozen_future_code_and_descriptor_are_verified(fallback, roots):
    write(roots.future / fallback.PLAN_FILE, {"original": True})
    code = roots.repo / "future.py"
    code.parent.mkdir()
    code.write_text("original\n")
    write(roots.recovery / "future-quality-prepared-v2.json", {"descriptor_activated": True,
        "descriptor_sha256": fallback.digest(roots.future / fallback.PLAN_FILE), "source_sha256": {"future.py": fallback.digest(code)}})
    fallback.verify_freeze(roots.repo, roots.recovery, roots.future)
    code.write_text("changed\n")
    with pytest.raises(RuntimeError, match="runtime changed"):
        fallback.verify_freeze(roots.repo, roots.recovery, roots.future)


@pytest.mark.parametrize("qualified", [False, True])
def test_fallback_launches_only_one_local_child_after_terminal_gate(fallback, roots, monkeypatch, qualified):
    monkeypatch.setattr(fallback, "check_pause", lambda root: None)
    monkeypatch.setattr(fallback, "verify_freeze", lambda *args: None)
    monkeypatch.setattr(fallback, "load_plan", lambda *args: ({}, roots.parent, roots.future))
    status = "qualified_parent_owned_by_existing_recovery" if qualified else "complete_C0_unqualified_rate_not_GPU_failure"
    monkeypatch.setattr(fallback, "readiness", lambda *args: status)
    commands = []
    monkeypatch.setattr(fallback.subprocess, "Popen", lambda command, **kwargs:
                        commands.append(command) or SimpleNamespace(pid=33333, wait=lambda: 0))
    fallback.run(roots.root, roots.parent, roots.recovery, roots.future)
    assert len(commands) == (0 if qualified else 1)
    assert not any("ssh" in item or "sbatch" in item for command in commands for item in command)
    assert json.loads((roots.root / "campaign-state.json").read_text())["total_research_goal_completed"] is False


def test_failed_terminal_raw_verification_prevents_fallback(fallback, roots, monkeypatch):
    def reject(*args):
        raise ValueError("SHA mismatch")
    monkeypatch.setattr(fallback, "verify_closed", reject)
    with pytest.raises(ValueError, match="SHA mismatch"):
        fallback.readiness(roots.parent, roots.recovery, roots.future)
