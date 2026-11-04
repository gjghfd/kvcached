from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
import public_source_continuation as continuation


def promotion_fixture(tmp_path):
    repo, stage, root = (tmp_path / name for name in ("repo", "stage", "evidence"))
    for name in continuation.PROMOTIONS:
        for base, text in ((repo, "original"), (stage, "staged source label")):
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    state = {"old_sha256": {name: continuation.digest(repo / name) for name in continuation.PROMOTIONS},
             "staged_sha256": {name: continuation.digest(stage / name) for name in continuation.PROMOTIONS}}
    return repo, stage, root, state


def test_promotion_keeps_exact_old_and_new_source_sha(tmp_path):
    repo, stage, root, state = promotion_fixture(tmp_path)
    continuation.promote(repo, stage, root, state)
    for name in continuation.PROMOTIONS:
        assert continuation.digest(repo / name) == state["staged_sha256"][name]
        assert continuation.digest(root / "pre-promotion-source" / name) == state["old_sha256"][name]


def test_user_local_edit_is_not_overwritten(tmp_path):
    repo, stage, root, state = promotion_fixture(tmp_path)
    (repo / continuation.PROMOTIONS[-1]).write_text("user edit")
    with pytest.raises(RuntimeError, match="do not overwrite local edits"):
        continuation.promote(repo, stage, root, state)
    assert (repo / continuation.PROMOTIONS[0]).read_text() == "original"
    assert (repo / continuation.PROMOTIONS[-1]).read_text() == "user edit"


def test_existing_second_backup_fails_before_first_promotion(tmp_path):
    repo, stage, root, state = promotion_fixture(tmp_path)
    path = root / "pre-promotion-source" / continuation.PROMOTIONS[-1]
    path.parent.mkdir(parents=True)
    path.write_text("prior evidence")
    with pytest.raises(RuntimeError, match="existing source promotion evidence"):
        continuation.promote(repo, stage, root, state)
    assert (repo / continuation.PROMOTIONS[0]).read_text() == "original"


def test_pending_parent_does_not_promote_or_read_raw(tmp_path):
    assert continuation.ready_parent(tmp_path, {"phase": "running", "GPU_calibration_complete": False}) is False


def test_failed_parent_requires_diagnosis(tmp_path):
    with pytest.raises(RuntimeError, match="Parent needs diagnosis"):
        continuation.ready_parent(tmp_path, {"phase": "diagnosis_required", "GPU_calibration_complete": False})


def test_completed_accounting_is_not_independent_closure(tmp_path):
    with pytest.raises(RuntimeError, match="independent public parent closure"):
        continuation.ready_parent(tmp_path, {"phase": "running", "GPU_calibration_complete": True,
                                              "state_last_observed": "COMPLETED"})
