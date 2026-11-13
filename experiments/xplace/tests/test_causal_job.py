import json
import os
from pathlib import Path
import signal
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

import causal_job as job


@pytest.mark.parametrize("age", [-1, 3300, 4000, float("nan"), float("inf")])
def test_invalid_shared_deadline_rejects(age):
    with pytest.raises(ValueError):
        job.remaining_seconds(10000 - age, now=10000)


def test_deadline_keeps_original_sbatch_age():
    assert job.remaining_seconds(100, now=400) == 3000


@pytest.mark.parametrize("block", [0, 1])
def test_fixed_order_restarts_and_separates_outputs(tmp_path, block):
    calls = []

    def runner(command, environment, root, deadline):
        calls.append((command, environment, root, deadline))
        return dict(status="passed", returncode=0)

    environment = dict(XPLACE_JOB_STARTED_AT=str(time.time()), DP_OPT_TRANSFER_ROOT=str(tmp_path))
    result = job.run_block(block=block, root=tmp_path / "out", common_args=["--checkpoint", "fake"],
                           environment=environment, runner=runner)
    assert result["status"] == "collected" and result["not_executed"] == []
    assert [call[0][-1] for call in calls] == list(job.BLOCKS[block])
    assert len({call[1]["DP_OPT_TRANSFER_ROOT"] for call in calls}) == 2
    assert calls[0][3] == calls[1][3]
    assert all(call[1]["XPLACE_JOB_STARTED_AT"] == environment["XPLACE_JOB_STARTED_AT"] for call in calls)
    assert not result["prediction_gate_passed"] and not result["overhead_accepted"]


@pytest.mark.parametrize("error", [None, TimeoutError("test_timeout")])
def test_failed_condition_is_retained_and_never_retried(tmp_path, error):
    calls = []

    def runner(command, environment, root, deadline):
        calls.append(command)
        if error:
            raise error
        return dict(status="failed", returncode=7)

    with pytest.raises((RuntimeError, TimeoutError)):
        job.run_block(block=0, root=tmp_path / "out", common_args=[], runner=runner,
                      environment=dict(XPLACE_JOB_STARTED_AT=str(time.time()), DP_OPT_TRANSFER_ROOT=str(tmp_path)))
    record = json.loads((tmp_path / "out/status.json").read_text())
    assert len(calls) == 1 and record["status"] == "failed" and record["not_executed"] == ["collect"]
    assert record["started"] == ["off"]


def test_exhausted_condition_budget_has_no_subprocess(tmp_path):
    with pytest.raises(TimeoutError):
        job.run_block(block=0, root=tmp_path / "out", common_args=[],
                      environment=dict(XPLACE_JOB_STARTED_AT=str(time.time() - 1900), DP_OPT_TRANSFER_ROOT=str(tmp_path)))
    record = json.loads((tmp_path / "out/status.json").read_text())
    assert record["not_executed"] == ["off", "collect"] and record["started"] == []


@pytest.mark.parametrize("returncode", [0, 7])
def test_real_cpu_child_exit_logs_and_returncode(tmp_path, returncode):
    command = [sys.executable, "-c", f"import sys; print('CPU child'); sys.exit({returncode})"]
    result = job.run_child(command, dict(os.environ), tmp_path / "child", time.monotonic() + 5,
                           sample=lambda group: dict(group=group, fixture=True))
    assert result["returncode"] == returncode
    assert result["status"] == ("passed" if returncode == 0 else "failed")
    assert not result["forced_cleanup"]
    assert "CPU child" in (tmp_path / "child/stdout.log").read_text()


def test_real_timeout_kills_and_reaps_cpu_child(tmp_path):
    with pytest.raises(TimeoutError):
        job.run_child([sys.executable, "-c", "import time; time.sleep(60)"], dict(os.environ),
                      tmp_path / "child", time.monotonic() + 0.1,
                      sample=lambda group: dict(group=group, fixture=True))
    record = json.loads((tmp_path / "child/process.json").read_text())
    assert record["timeout"] and record["forced_cleanup"] and record["status"] == "failed"
    assert record["returncode"] == -signal.SIGTERM
    with pytest.raises(ProcessLookupError):
        os.kill(record["pid"], 0)


def test_proc_parser_handles_parentheses_and_group_members(tmp_path):
    for pid, group in ((123, 123), (124, 123), (125, 9)):
        directory = tmp_path / str(pid)
        directory.mkdir()
        fields = ["0"] * 22
        fields[0:3] = ["S", "12", str(group)]
        fields[11], fields[12], fields[19], fields[21] = "7", "9", "100", "20"
        (directory / "stat").write_text(f"{pid} (name with ) chars) " + " ".join(fields))
    record = job.process_group_usage(123, tmp_path)
    assert {item["pid"] for item in record["processes"]} == {123, 124}
    assert all(item["cpu_ticks"] == 16 and item["rss_pages"] == 20 for item in record["processes"])
    assert record["read_failures"] == []
