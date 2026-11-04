import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("robust_return_followup")


def state():
    return {"job": 268245, "GPU_experiment_complete": True, "state_last_observed": "COMPLETED",
            "phase": "long_Conv_probe_SHA_verified_analyzed", "probe_config": {"seeds": [1, 2, 3]}}


@pytest.mark.parametrize("field,value", [("GPU_experiment_complete", False), ("state_last_observed", "RUNNING"),
                                        ("phase", "submitted_long_context_then_return_probes")])
def test_pending_or_incomplete_campaign_cannot_start_replay(module, field, value):
    current = state()
    current[field] = value
    assert not module.ready(current)


def test_terminal_diagnosis_stops_without_GPU_retry(module):
    current = state()
    current["phase"] = "diagnosis_required_no_automatic_duplicate_submission"
    with pytest.raises(RuntimeError, match="failure diagnosis"):
        module.ready(current)


def test_replay_requires_raw_SHA_and_all_three_independent_verifications(module, monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    files = [repo / f"candidate-{index}.py" for index in range(4)]
    for path in files:
        path.write_text("pass\n")
    (tmp_path / "robust-candidate-source.sha256").write_text(
        "".join(module.digest(path) + "  " + path.name + "\n" for path in files))
    raw = tmp_path / "live-trace-268245"
    raw.mkdir()
    (raw / "exit-status").write_text("0\n")
    (tmp_path / "raw-268245.sha256").write_text("verified manifest")
    verification = {"probes": {f"probe-{seed}": {} for seed in (1, 2, 3)}, "research_goal_complete": False,
                    "calibration": {"source_verification": "raw_verified"}}
    (tmp_path / "independent-verification.json").write_text(json.dumps(verification))
    verified, analyses = [], []
    monkeypatch.setattr(module, "verify_raw", lambda path, manifest: verified.append((path, manifest)))
    monkeypatch.setattr(module, "analyze", lambda path, **options: analyses.append(path) or {"offline": True})
    result = module.replay_completed(tmp_path, repo, state())
    assert len(verified) == 1
    assert len(analyses) == 3
    assert result["new_allocation_submitted"] is False
    assert result["runtime_changed"] is False
    files[0].write_text("changed\n")
    with pytest.raises(RuntimeError, match="source changed"):
        module.replay_completed(tmp_path, repo, state())


def test_source_manifest_cannot_escape_repository(module, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("pass\n")
    (tmp_path / "robust-candidate-source.sha256").write_text(
        (module.digest(outside) + "  ../outside.py\n") * 4)
    with pytest.raises(RuntimeError, match="source changed"):
        module.check_sources(tmp_path, repo)
