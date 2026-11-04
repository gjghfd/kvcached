from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
import public_source_continuation_v2 as continuation


def watchdog_fixture(tmp_path):
    source = Path(continuation.__file__).parent / ".public-source-stage-r2"
    stage = tmp_path / "stage"
    for name in continuation.WATCHDOGS:
        target = stage / "experiments/dp_opt" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "experiments/dp_opt" / name, target)
    return stage


def test_future_watchdogs_keep_the_same_slo_for_every_policy(tmp_path):
    stage = watchdog_fixture(tmp_path)
    assert continuation.validate_watchdogs(stage) == {
        "public-workload-search.sbatch": 7200,
        "public-poisson-search.sbatch": 18000,
    }
    for name in continuation.WATCHDOGS:
        script = (stage / "experiments/dp_opt" / name).read_text()
        assert script.count("DP_OPT_REQUEST_TIMEOUT_S=") == 1
        assert script.count("DP_OPT_DEADLINE_S=120") == 1
        assert "DP_OPT_MAX_MODEL_LEN=131072" in script
        assert "DP_OPT_TRACE_BUDGET_MIB=10240" in script
        assert "DP_OPT_PAUSE_SYNC_INTERVAL=32" in script


@pytest.mark.parametrize("name", tuple(continuation.WATCHDOGS))
@pytest.mark.parametrize("change", ("old_timeout", "relaxed_slo", "duplicate_timeout"))
def test_changed_or_policy_local_watchdog_is_rejected(tmp_path, name, change):
    stage = watchdog_fixture(tmp_path)
    path = stage / "experiments/dp_opt" / name
    content = path.read_text()
    timeout = continuation.WATCHDOGS[name]
    if change == "old_timeout":
        content = content.replace(f"DP_OPT_REQUEST_TIMEOUT_S={timeout}", "DP_OPT_REQUEST_TIMEOUT_S=1800")
    elif change == "relaxed_slo":
        content = content.replace("DP_OPT_DEADLINE_S=120", "DP_OPT_DEADLINE_S=7200")
    else:
        content += "\nexport DP_OPT_REQUEST_TIMEOUT_S=1\n"
    path.write_text(content)
    with pytest.raises(RuntimeError, match="watchdog/SLO"):
        continuation.validate_watchdogs(stage)


def test_all_four_promotions_preserve_exact_backups(tmp_path):
    repo, stage, root = (tmp_path / name for name in ("repo", "stage", "evidence"))
    for name in continuation.PROMOTIONS:
        for base, text in ((repo, "original frozen source"), (stage, "future source")):
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    state = {"old_sha256": {name: continuation.digest(repo / name) for name in continuation.PROMOTIONS},
             "staged_sha256": {name: continuation.digest(stage / name) for name in continuation.PROMOTIONS}}
    continuation.promote(repo, stage, root, state)
    assert len(continuation.PROMOTIONS) == 4
    for name in continuation.PROMOTIONS:
        assert continuation.digest(repo / name) == state["staged_sha256"][name]
        assert continuation.digest(root / "pre-promotion-source" / name) == state["old_sha256"][name]


def test_pending_parent_cannot_promote_a_longer_watchdog(tmp_path):
    assert not continuation.ready_parent(tmp_path, {"phase": "running", "GPU_calibration_complete": False})


def test_failed_parent_requires_diagnosis_before_future_work(tmp_path):
    with pytest.raises(RuntimeError, match="Parent needs diagnosis"):
        continuation.ready_parent(tmp_path, {"phase": "diagnosis_required", "GPU_calibration_complete": False})
