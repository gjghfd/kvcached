import importlib
import json
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("random_baseline_campaign")


@pytest.mark.parametrize("phase,complete,ready", [
    ("submitted_pressure_1p2_five_policy_trace", False, False),
    ("profile_return_trace_SHA_verified_analyzed", True, False),
    ("pressure_1p2_trace_verified_analyzed", False, False),
    ("pressure_1p2_trace_verified_analyzed", True, True)])
def test_waits_for_full_previous_verification(campaign, phase, complete, ready):
    assert campaign.previous_ready({"phase": phase, "GPU_experiment_complete": complete}) is ready


def test_previous_failure_never_triggers_another_submission(campaign):
    with pytest.raises(RuntimeError, match="requires diagnosis"):
        campaign.previous_ready({"phase": "diagnosis_required_no_automatic_duplicate_submission"})


@pytest.mark.parametrize("changed", [False, True])
def test_promotion_is_guarded_and_uses_apply_patch(campaign, tmp_path, changed):
    tool = shutil.which("apply_patch")
    if tool is None:
        pytest.skip("Official apply_patch executable unavailable")
    repo, stage, root = (tmp_path / name for name in ("repo", "stage", "evidence"))
    for directory in (repo, stage, root):
        directory.mkdir()
    (repo / "existing.txt").write_text("old\nshared\n")
    (stage / "existing.txt").write_text("new\nshared\nextra\n")
    (stage / "added.txt").write_text("added\n")
    manifest, patch = campaign.promotion_patch(repo, stage, ("existing.txt", "added.txt"))
    (root / "promotion.patch").write_text(patch)
    state = {"promotion_manifest": manifest, "promotion_patch_sha256": campaign.digest(root / "promotion.patch")}
    if changed:
        (repo / "existing.txt").write_text("user-edit\n")
        with pytest.raises(RuntimeError, match="do not overwrite"):
            campaign.promote(root, repo, stage, state, Path(tool))
        assert not (repo / "added.txt").exists()
    else:
        campaign.promote(root, repo, stage, state, Path(tool))
        assert (repo / "existing.txt").read_bytes() == (stage / "existing.txt").read_bytes()
        assert (repo / "added.txt").read_bytes() == (stage / "added.txt").read_bytes()


def test_six_policies_keep_existing_baselines_and_add_random(campaign):
    assert campaign.POLICIES == ("no_move", "no_move_fair", "random_no_return", "kv_only",
                                 "kv_only_headroom", "kv_only_workaware")
    assert "experiments/dp_opt/random_placement.py" in campaign.SYNC_FILES
    assert "tests/test_dp_opt_random_placement.py" in campaign.SYNC_FILES


def test_no_return_baseline_has_no_release_trigger(campaign, tmp_path):
    for policy in ("random_no_return", "kv_only_headroom", "kv_only_workaware"):
        directory = tmp_path / f"r0-{policy}"
        directory.mkdir()
        (directory / "decisions.json").write_text(json.dumps([] if policy == "random_no_return" else [
            {"path": "release_rebalance_ilp", "reason": "domain_guard"}]))
        (directory / "migrations.json").write_text("[]")
        (directory / "summary.json").write_text("{}")
    result = campaign.summarize_rebalance_reasons(tmp_path)
    assert result["random_no_return"]["release_rebalance_reasons"] == {}
    assert result["kv_only_workaware"]["release_rebalance_reasons"] == {"domain_guard": 1}
