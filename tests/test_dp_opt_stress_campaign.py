import importlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("stress_campaign")


def best():
    return {"setting": {"output_tokens": 8192, "budget_mib_per_gpu": 10240,
                        "conv_rate_rps": .023, "code_rate_rps": .001}}


def test_complete_grid_and_fixed_heldout_rate(campaign):
    assert campaign.training_plan(0)["budgets"] == [10240, 11328]
    for stage, output in ((2, 12288), (3, 16384)):
        plan = campaign.training_plan(stage, best())
        assert plan["outputs"] == [output] and plan["budgets"] == [10240, 11328]
        assert plan["multipliers"] == [.85, 1, 1.15]
    with pytest.raises(ValueError):
        campaign.training_plan(1)
    plan = campaign.confirmation_plan(best(), 20261025)
    assert plan["fixed_conv_rate"] == .023 and plan["multipliers"] == [1]
    command = campaign.submit_command(plan)
    assert 'test -z "$(squeue' in command and "gjghfd/kvcached.git" in command
    assert "DP_OPT_STRESS_FIXED_CONV_RATE=0.023" in command and "--time=32:00:00" in command


def test_archive_argv_and_resumes_completed_download(campaign, monkeypatch, tmp_path):
    (tmp_path / "live-trace-123").mkdir()
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout="passed", stderr=""))
    monkeypatch.setattr(campaign.subprocess, "run", run)
    campaign.archive_job(tmp_path, 123, Path("scripts"), {})
    command = run.call_args.args[0]
    assert command[1:] == ["scripts/archive_trace_job.py", "123", "--archive", str(tmp_path), "--resume"]
    assert "--log-prefix" not in command


def test_remote_monitor_failure_preserves_job_without_resubmit(campaign, monkeypatch, tmp_path):
    commands = []

    def ssh(command):
        commands.append(command)
        return SimpleNamespace(returncode=0 if len(commands) == 1 else 1,
                               stdout="123\n" if len(commands) == 1 else "", stderr="")

    monkeypatch.setattr(campaign, "ssh", ssh)
    state = {"jobs": []}
    with pytest.raises(RuntimeError, match="do not submit again"):
        campaign.start_job(tmp_path, campaign.training_plan(0), state)
    saved = json.loads((tmp_path / "campaign-state.json").read_text())
    assert saved["job"] == 123 and len(saved["jobs"]) == 1 and len(commands) == 2


def test_pause_marker_prevents_submission(campaign, monkeypatch, tmp_path):
    (tmp_path / "PAUSE").touch()
    ssh = Mock()
    monkeypatch.setattr(campaign, "ssh", ssh)
    with pytest.raises(RuntimeError, match="pause"):
        campaign.start_job(tmp_path, campaign.training_plan(0), {"jobs": []})
    ssh.assert_not_called()


@pytest.mark.parametrize("terminal", ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"])
def test_terminal_poll_does_not_submit(campaign, monkeypatch, tmp_path, terminal):
    monkeypatch.setattr(campaign, "ssh", lambda *args: SimpleNamespace(
        returncode=0, stdout=f"123|{terminal}|0:0|start|end|duration|node\n", stderr=""))
    monkeypatch.setattr(campaign.time, "sleep", Mock(side_effect=AssertionError("terminal must return")))
    state = {}
    assert campaign.poll_job(tmp_path, 123, state) == terminal
    assert state["state_last_observed"] == terminal


@pytest.mark.parametrize("wall_gap", [300, 1800])
def test_poll_records_wallclock_gaps_without_changing_monotonic_cadence(campaign, monkeypatch, tmp_path, wall_gap):
    start = datetime(2026, 10, 26, tzinfo=timezone.utc)
    moments = iter((start, start + timedelta(seconds=wall_gap)))

    class WallClock:
        @classmethod
        def now(cls):
            return next(moments)

    monkeypatch.setattr(campaign, "datetime", WallClock)
    monkeypatch.setattr(campaign, "save_state", lambda root, state: (
        root / "campaign-state.json").write_text(json.dumps(state)))
    monotonic = iter((0, 1, 300, 301))
    monkeypatch.setattr(campaign.time, "monotonic", lambda: next(monotonic))
    sleep = Mock()
    monkeypatch.setattr(campaign.time, "sleep", sleep)
    ssh = Mock(side_effect=[SimpleNamespace(returncode=0, stdout=(
        f"123|{status}|0:0|start|end|duration|node\n"), stderr="") for status in ("RUNNING", "COMPLETED")])
    monkeypatch.setattr(campaign, "ssh", ssh)
    state = {}
    assert campaign.poll_job(tmp_path, 123, state) == "COMPLETED"
    assert state["last_poll_actual_gap_s"] == wall_gap
    assert state["last_poll_wall_clock_gap_s"] == wall_gap
    assert state["last_poll_monotonic_gap_s"] == 300
    assert f"actual_gap_s={float(wall_gap)} ssh_exit=0" in (tmp_path / "poll-123.log").read_text()
    assert ssh.call_count == 2
    sleep.assert_called_once_with(299)


def test_stress_branch_precedes_calibration(campaign):
    source = (Path(campaign.__file__).parent / "live_trace.py").read_text()
    assert source.index('if getattr(args, "stress_sweep", False):') < source.index("if args.calibration_conv_marks is not None:")
