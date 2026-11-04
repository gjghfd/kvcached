import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def setup_recovery(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    recovery = importlib.import_module("public_source_recovery")
    quality = importlib.import_module("public_quality_followup")
    root, failed, bootstrap, calibration, evaluation, plan_root = [tmp_path / name for name in
        ("recovery", "failed", "bootstrap", "quality-cal", "quality-eval", "plan")]
    bootstrap.mkdir()
    (bootstrap / "campaign-state.json").write_text(json.dumps({
        "GPU_calibration_complete": True, "state_last_observed": "COMPLETED", "job": 10}))
    (bootstrap / "raw-10.sha256").write_text("test-only-mocked-archive\n")
    inputs = calibration / "frozen/inputs"
    inputs.mkdir(parents=True)
    (inputs / "input-manifest.json").write_text("{}\n")
    plan_root.mkdir()
    (plan_root / quality.PLAN_FILE).write_text("{}\n")
    plan = {"source_selection": "original_marker_quality_only_not_output_length_or_GPU_gain",
            "input_manifest_sha256": recovery.digest(inputs / "input-manifest.json")}
    failure = {"failure_kind": "public_prompt_bank_verifier_schema", "not_reclassified_as_success": True}
    monkeypatch.setattr(recovery, "verify_closed_failure", lambda parent: failure)
    monkeypatch.setattr(recovery, "verify_raw", lambda raw, manifest: None)
    monkeypatch.setattr(recovery, "validate", lambda path: {"dataset_label": quality.LABEL, "bank_size": 256})
    monkeypatch.setattr(quality, "load_plan", lambda parent, failed: (plan, calibration, evaluation))
    calls = []

    def stage(repo, root, state, child_root, script, arguments, role):
        recovery.check_quality_source(state)
        calls.append((role, child_root))
        return {"GPU_calibration_complete": True, "candidate_case": "rate-1"}

    monkeypatch.setattr(recovery, "run_stage", stage)
    return recovery, quality, (root, failed, bootstrap, calibration, evaluation, plan_root), plan, failure, calls


def test_explicit_quality_recovery_keeps_failed_parent_and_uses_declared_roots(setup_recovery):
    recovery, quality, arguments, plan, failure, calls = setup_recovery
    recovery.run(*arguments)
    state = json.loads((arguments[0] / "campaign-state.json").read_text())
    assert state["dataset_label"] == quality.LABEL
    assert state["closed_failure"]["not_reclassified_as_success"] is True
    assert state["C0_failure_not_reclassified_as_success"] is True
    assert state["C0_costs_not_reused_for_C1"] is True
    assert state["Stratos_inputs_replaced"] is False
    assert state["total_research_goal_completed"] is False
    assert calls == [("calibration", arguments[3]), ("finite_grid_and_optional_fixed_heldout", arguments[4])]


def test_quality_recovery_cannot_bypass_a_different_failure(setup_recovery):
    recovery, quality, arguments, plan, failure, calls = setup_recovery
    failure.clear()
    with pytest.raises(RuntimeError, match="diagnosed and reverified prompt-bank failure"):
        recovery.run(*arguments)
    assert not calls


def test_quality_recovery_cannot_change_predeclared_roots(setup_recovery, monkeypatch):
    recovery, quality, arguments, plan, failure, calls = setup_recovery
    monkeypatch.setattr(quality, "load_plan", lambda parent, failed: (plan, arguments[3], arguments[0]))
    with pytest.raises(RuntimeError, match="unchanged predeclared independent roots"):
        recovery.run(*arguments)
    assert not calls


def test_quality_recovery_unqualified_calibration_never_starts_grid(setup_recovery, monkeypatch):
    recovery, quality, arguments, plan, failure, calls = setup_recovery
    monkeypatch.setattr(recovery, "run_stage", lambda *args: {"GPU_calibration_complete": True, "candidate_case": None})
    with pytest.raises(RuntimeError, match="no pressure/queue-qualified rate"):
        recovery.run(*arguments)
    state = json.loads((arguments[0] / "campaign-state.json").read_text())
    assert "diagnosis_required" in state["phase"]
    assert not state["total_research_goal_completed"]


def test_quality_recovery_rejects_changed_declaration_between_stages(setup_recovery, monkeypatch):
    recovery, quality, arguments, plan, failure, calls = setup_recovery

    def stage(repo, root, state, child_root, script, child_arguments, role):
        recovery.check_quality_source(state)
        calls.append(role)
        (arguments[5] / quality.PLAN_FILE).write_text('{"changed":true}\n')
        return {"GPU_calibration_complete": True, "candidate_case": "rate-1"}

    monkeypatch.setattr(recovery, "run_stage", stage)
    with pytest.raises(RuntimeError, match="declaration changed"):
        recovery.run(*arguments)
    assert calls == ["calibration"]
