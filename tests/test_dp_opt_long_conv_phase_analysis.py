import importlib
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def observer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("long_conv_phase_analysis")


def test_only_phases_before_current_phase_can_be_archived(observer):
    assert observer.eligible_phases({}, "probe-20261030") == ()
    assert observer.eligible_phases({}, "probe-20261031") == ("probe-20261030",)
    assert observer.eligible_phases({}, "probe-20261032") == ("probe-20261030", "probe-20261031")
    assert observer.eligible_phases({}, "long-context-calibration") == ()


def test_terminal_source_requires_complete_verified_GPU_marker(observer):
    with pytest.raises(ValueError, match="verified GPU"):
        observer.eligible_phases({"GPU_experiment_complete": True}, "probe-20261032")
    state = {"GPU_experiment_complete": True, "state_last_observed": "COMPLETED",
             "phase": "long_Conv_probe_SHA_verified_analyzed"}
    assert observer.eligible_phases(state, None) == observer.PHASES


def test_phase_is_read_from_last_existing_main_poll(observer, tmp_path):
    path = tmp_path / "poll"
    assert observer.latest_phase(path) is None
    path.write_text('{"phase":"probe-20261030"}\nignore\n{"phase":"probe-20261031"}\n')
    assert observer.latest_phase(path) == "probe-20261031"


def test_old_large_log_is_bounded_and_missing_phase_does_not_authorize_archive(observer, tmp_path):
    path = tmp_path / "poll"
    path.write_text('{"phase":"probe-20261030"}\n' + "padding\n" * 20000)
    assert observer.latest_phase(path) is None


@pytest.mark.parametrize("job,phase", [(0, "probe-20261030"), (True, "probe-20261030"),
                                      ("123;rm", "probe-20261030"), (123, "../active")])
def test_archive_command_rejects_unsafe_identity(observer, job, phase):
    with pytest.raises(ValueError):
        observer.pack_command(job, phase)


def test_archive_command_has_exit_zero_and_no_symlink_SHA_barriers(observer):
    command = observer.pack_command(123, "probe-20261030")
    assert 'test "$(cat "$source/exit-status")" = 0' in command
    assert 'cmp "$attempt/raw-before.sha256" "$attempt/raw-after.sha256"' in command
    assert command.count('find "$source" -type l') == 2
    assert "sbatch" not in command and "squeue" not in command and "sacct" not in command
    assert "python" not in command


def test_archive_shell_syntax_and_active_phase_barrier(observer, tmp_path):
    command = observer.pack_command(123, "probe-20261030")
    assert subprocess.run(["bash", "-n"], input=command, text=True, capture_output=True).returncode == 0
    root = tmp_path / "run/kvcached-paracloud/state/live-trace-123/probe-20261030"
    root.mkdir(parents=True)
    (root / "exit-status").write_text("1\n")
    result = subprocess.run(["bash", "-s"], input=command, text=True, capture_output=True,
                            env={**os.environ, "HOME": str(tmp_path)})
    assert result.returncode != 0
    assert not (tmp_path / "run/kvcached-paracloud/sync").exists()


def metadata(stage="/data/home/user/run/kvcached-paracloud/sync/closed-long-conv-123.abc123",
             attempt="phase-probe-20261030.xyz123", digest="a" * 64):
    return f"stage={stage}\nattempt={attempt}\nbundle_sha256={digest}\n"


def test_archive_metadata_is_strict_and_confined(observer):
    assert observer.pack_metadata(metadata(), 123, "probe-20261030")["bundle_sha256"] == "a" * 64


@pytest.mark.parametrize("text", [metadata(stage="/tmp/closed-long-conv-123.abc123"),
    metadata(stage="/data/../home/user/run/kvcached-paracloud/sync/closed-long-conv-123.abc123"),
    metadata(stage="/data/home/evil;command/run/kvcached-paracloud/sync/closed-long-conv-123.abc123"),
    metadata(attempt="../raw"), metadata(digest="b" * 63), metadata() + "stage=duplicate\n", "bad-line\n"])
def test_unsafe_or_duplicate_archive_metadata_is_rejected(observer, text):
    with pytest.raises(ValueError):
        observer.pack_metadata(text, 123, "probe-20261030")


def test_local_poll_only_does_not_run_SSH_or_submit_GPU(observer, tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    campaign = {"job": 123, "repo": str(repo), "inputs": str(tmp_path),
                "phase": "submitted_long_context_then_return_probes", "frozen_runtime_sha256": {}}
    (tmp_path / "campaign-state.json").write_text(json.dumps(campaign))
    (tmp_path / "poll-123.log").write_text('{"phase":"probe-20261030"}\n')
    manifest = tmp_path / "sources.sha256"
    source = repo / "dummy.py"
    source.write_text("pass\n")
    manifest.write_text(hashlib.sha256(source.read_bytes()).hexdigest() + "  dummy.py\n")
    monkeypatch.setattr(observer.subprocess, "run", lambda *arguments, **options: pytest.fail("Unexpected SSH"))
    observer.observe(tmp_path, manifest, once=True)
    state = json.loads((tmp_path / "pause-inclusive-phase-analysis/observer-state.json").read_text())
    assert state["phases"] == {}
    assert state["Slurm_queries"] is False
    assert state["GPU_submissions"] is False


def test_PAUSE_stops_without_archive_or_submission(observer, tmp_path, monkeypatch):
    manifest = tmp_path / "sources.sha256"
    manifest.write_text("a" * 64 + "  dummy.py\n")
    (tmp_path / "PAUSE").touch()
    monkeypatch.setattr(observer.subprocess, "run", lambda *arguments, **options: pytest.fail("Unexpected SSH"))
    with pytest.raises(RuntimeError, match="pause marker"):
        observer.observe(tmp_path, manifest, once=True)


def test_empty_manifest_does_not_disable_source_freeze(observer, tmp_path):
    manifest = tmp_path / "sources.sha256"
    manifest.write_text("")
    with pytest.raises(ValueError, match="nonempty frozen"):
        observer.observe(tmp_path, manifest, once=True)
