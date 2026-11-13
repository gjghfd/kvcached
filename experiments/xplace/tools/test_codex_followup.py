import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import codex_followup as followup


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    root = tmp_path / "control"
    root.mkdir()
    config = {"_path": str(root / "config.json"), "workspace": str(tmp_path), "thread_id": "pinned-thread",
              "rollout": str(tmp_path / "rollout.jsonl"), "idle_grace_s": 0, "model": "existing-model"}
    state = {"enabled": True, "phase": "waiting_for_idle", "events": [], "submissions": {},
             "observed": {"1": {"state": "FAILED", "in_queue": False}},
             "jobs": [{"job": "1", "tag": "original", "dp": 4, "tp": 1, "config_name": "scan.json"}]}
    followup.save(root / "state.json", state)
    records = [{"type": "session_meta", "payload": {"id": config["thread_id"]}},
               {"type": "event_msg", "timestamp": "2000-01-01T00:00:00Z",
                "payload": {"type": "task_started", "turn_id": "turn-1"}},
               {"type": "event_msg", "timestamp": "2000-01-01T00:00:01Z",
                "payload": {"type": "task_complete", "turn_id": "turn-1"}}]
    Path(config["rollout"]).write_text("".join(json.dumps(record) + "\n" for record in records))
    monkeypatch.setattr(followup.xplace_ctl, "STAGING", tmp_path / "staging")
    return config, root, state


def bundle(tag, code="value = 1\n", seed=7, gpus=4):
    root = followup.xplace_ctl.STAGING / tag
    upload = root / "upload"
    for name in ("runtime", "inputs", "tests"):
        (upload / name).mkdir(parents=True)
    (upload / "runtime/main.py").write_text(code)
    followup.save(upload / "inputs/scan.json", {"seed": seed, "context_tokens": 4096, "series": []})
    (upload / "tests/test_smoke.py").write_text("def test_smoke():\n    assert True\n")
    (upload / "xplace.sbatch").write_text(f"#!/bin/bash\n#SBATCH --nodes=1\n#SBATCH --gpus={gpus}\ntrue\n")
    files = sorted(path for path in upload.rglob("*") if path.is_file())
    manifest = "".join(f"{followup.xplace_ctl.sha256(path)}  {path.relative_to(upload)}\n" for path in files)
    (upload / "MANIFEST.sha256").write_text(manifest)
    (root / "upload.tar.gz").write_bytes(b"frozen-test-bundle")
    followup.save(root / "build.json", {"manifest_sha256": followup.xplace_ctl.sha256(upload / "MANIFEST.sha256"),
                  "tarball_sha256": followup.xplace_ctl.sha256(root / "upload.tar.gz"), "uploaded_at": "test"})
    return root


def repair(config):
    workspace = Path(config["workspace"])
    evidence = workspace / "evidence/xplace-original/job-1/main.log"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("Observed failure in the original experiment\n")
    report = workspace / "repair-report.json"
    followup.save(report, {"parent_job": "1", "diagnosis": "Fault identified in the recorded configuration",
                          "fix": "Correct the runtime value", "evidence_files": [str(evidence.relative_to(workspace))]})
    return report


def test_current_session_claim_validate_and_finish_without_model(campaign, monkeypatch):
    config, root, state = campaign
    config["dispatch_mode"] = "current_session"
    monkeypatch.setenv("CODEX_THREAD_ID", config["thread_id"])
    state.update(enabled=False, phase="needs_attention", error="model unavailable")
    followup.save(root / "state.json", state)
    monkeypatch.setattr(followup, "job_states", lambda jobs: state["observed"])
    claimed = followup.claim_current(config)
    assert followup.read(root / "state.json")["phase"] == "current_session"
    assert not followup.read(root / "state.json")["enabled"]
    with pytest.raises(ValueError, match="never launches"):
        followup.run_codex(config, root, Path(claimed["event"]))
    bundle("original")
    bundle("fixed", code="value = 2\n")
    proof = followup.validate_repair(config, "fixed", "1", repair(config))
    assert proof["parent_job"] == "1"
    followup.save(Path(config["workspace"]) / "analysis.json", {"status": "passed"})
    followup.save(Path(claimed["event"]) / "outcome.json", {
        "jobs": {"1": {"status": "verified", "analysis": "analysis.json"}}})
    finished = followup.finish_current(config)
    assert finished["phase"] == "complete" and not finished["enabled"]


@pytest.mark.parametrize("problem", ["wrong_thread", "enabled", "ambiguous", "running", "stop", "live_process"])
def test_current_claim_rejects_unsafe_recovery(campaign, monkeypatch, problem):
    config, root, state = campaign
    config["dispatch_mode"] = "current_session"
    monkeypatch.setenv("CODEX_THREAD_ID", "wrong" if problem == "wrong_thread" else config["thread_id"])
    state.update(enabled=problem == "enabled", phase="needs_attention")
    if problem == "ambiguous":
        state["submissions"]["fixed"] = {"status": "needs_reconciliation"}
    if problem == "running":
        state["observed"]["1"] = {"state": "RUNNING", "in_queue": True}
    if problem == "stop":
        (root / "STOP").touch()
    if problem == "live_process":
        event = root / "event-0001"
        event.mkdir()
        followup.save(event / "process.json", {"pid": 12345})
        state["events"].append(str(event))
        monkeypatch.setattr(followup.os, "kill", lambda pid, signal: None)
    followup.save(root / "state.json", state)
    monkeypatch.setattr(followup, "job_states", lambda jobs: state["observed"])
    with pytest.raises(ValueError):
        followup.claim_current(config)
    assert followup.read(root / "state.json")["events"] == state["events"]


def test_current_finish_retry_enters_foreground_monitor_state(campaign, monkeypatch):
    config, root, state = campaign
    config["dispatch_mode"] = "current_session"
    monkeypatch.setenv("CODEX_THREAD_ID", config["thread_id"])
    state.update(enabled=False, phase="needs_attention")
    followup.save(root / "state.json", state)
    monkeypatch.setattr(followup, "job_states", lambda jobs: state["observed"])
    claimed = followup.claim_current(config)
    state = followup.read(root / "state.json")
    state["jobs"].append({"job": "2", "tag": "fixed", "parent_job": "1"})
    followup.save(root / "state.json", state)
    followup.save(Path(claimed["event"]) / "outcome.json", {
        "jobs": {"1": {"status": "retry", "retry_job": "2"}}})
    finished = followup.finish_current(config)
    assert finished["phase"] == "waiting_for_jobs_current_session"
    assert not finished["enabled"]


def test_current_monitor_returns_on_terminal_job_without_codex(campaign, monkeypatch):
    config, root, state = campaign
    config["dispatch_mode"] = "current_session"
    monkeypatch.setenv("CODEX_THREAD_ID", config["thread_id"])
    state.update(enabled=False, phase="waiting_for_jobs_current_session")
    followup.save(root / "state.json", state)
    observed = iter([{"1": {"state": "RUNNING", "in_queue": True}},
                     {"1": {"state": "COMPLETED", "in_queue": False}}])
    monkeypatch.setattr(followup, "job_states", lambda jobs: next(observed))
    waits = []
    monkeypatch.setattr(followup.time, "sleep", waits.append)
    followup.monitor_current(config)
    assert waits == [120]
    assert followup.read(root / "state.json")["phase"] == "ready_current_session"
    assert not followup.read(root / "state.json")["enabled"]


def test_idle_guard_rejects_active_wrong_identity_and_partial_record(campaign):
    config, root, state = campaign
    assert followup.thread_idle(config)
    path = Path(config["rollout"])
    original = path.read_text()
    path.write_text(original + json.dumps({"type": "event_msg", "payload": {"type": "task_started", "turn_id": "new"}}) + "\n")
    assert not followup.thread_idle(config)
    path.write_text(original + "{unfinished")
    assert not followup.thread_idle(config)
    path.write_text(original)
    assert not followup.thread_idle({**config, "thread_id": "another-thread"})


def test_queue_overrides_terminal_accounting(monkeypatch):
    monkeypatch.setattr(followup.xplace_ctl, "remote", lambda *args, **kwargs:
                        "1|COMPLETED|0:0\n2|CANCELLED by 12|0:15\nQUEUE\n1|RUNNING\n")
    states = followup.job_states([{"job": "1"}, {"job": "2"}])
    assert states["1"]["in_queue"] and states["1"]["state"] == "RUNNING"
    assert states["2"]["state"] == "CANCELLED"


def test_success_triggers_once_and_stops(campaign, monkeypatch):
    config, root, state = campaign
    calls = []
    followup.save(Path(config["workspace"]) / "analysis.json", {"status": "passed"})
    monkeypatch.setattr(followup, "job_states", lambda jobs: {"1": {"state": "COMPLETED", "in_queue": False}})

    def run(settings, directory, event):
        calls.append(settings["thread_id"])
        followup.save(event / "outcome.json", {"jobs": {"1": {"status": "verified", "analysis": "analysis.json"}}})

    monkeypatch.setattr(followup, "run_codex", run)
    followup.tick(config)
    followup.tick(config)
    result = followup.read(root / "state.json")
    assert calls == ["pinned-thread"] and result["phase"] == "complete" and not result["enabled"]


def test_waits_for_whole_registered_batch(campaign, monkeypatch):
    config, root, state = campaign
    state["jobs"].append({"job": "2", "tag": "other"})
    followup.save(root / "state.json", state)
    monkeypatch.setattr(followup, "job_states", lambda jobs: {"1": {"state": "COMPLETED"}, "2": {"state": "RUNNING"}})
    monkeypatch.setattr(followup, "run_codex", lambda *args: pytest.fail("premature dispatch"))
    followup.tick(config)
    assert not followup.read(root / "state.json")["events"]


def test_interrupted_dispatch_is_not_replayed(campaign, monkeypatch):
    config, root, state = campaign
    state["phase"] = "dispatching"
    followup.save(root / "state.json", state)
    monkeypatch.setattr(followup, "run_codex", lambda *args: pytest.fail("duplicate dispatch"))
    followup.tick(config)
    assert followup.read(root / "state.json")["phase"] == "needs_attention"


def test_incomplete_background_response_blocks_instead_of_repeating(campaign, monkeypatch):
    config, root, state = campaign
    monkeypatch.setattr(followup, "job_states", lambda jobs: {"1": {"state": "FAILED"}})
    monkeypatch.setattr(followup, "run_codex", lambda *args: None)
    with pytest.raises(FileNotFoundError):
        followup.tick(config)
    assert not followup.read(root / "state.json")["enabled"]


@pytest.mark.parametrize("code,seed", [("value = 1\n", 7), ("# changed comment\nvalue=1\n", 7),
                                     ('"""changed documentation"""\nvalue = 1\n', 7), ("value = 1\n", 99)])
def test_noop_or_seed_only_retry_is_rejected(campaign, code, seed):
    config, root, state = campaign
    bundle("original")
    bundle("retry", code, seed)
    with pytest.raises(ValueError, match="Unchanged/repeated"):
        followup.validate_repair(config, "retry", "1", repair(config))


def test_tests_and_repair_proof_are_required_before_serial_submission(campaign, monkeypatch):
    config, root, state = campaign
    bundle("original")
    bundle("retry", "value = 2\n")
    proof = followup.validate_repair(config, "retry", "1", repair(config))
    assert proof["changed_files"] == ["runtime/main.py"]
    state["phase"] = "dispatching"
    state["submissions"] = {f"old-{index}": {"status": "submitted"} for index in range(10)}
    followup.save(root / "state.json", state)
    calls = []
    monkeypatch.setattr(followup.xplace_ctl, "remote", lambda *args, **kwargs: "55\n56\n")

    def submit(tag, env, limit, name, dependency, exclude):
        calls.append((env, limit, dependency))
        return {"job": "57"}

    monkeypatch.setattr(followup.xplace_ctl, "submit", submit)
    result = followup.submit_retry(config, "retry", "1")
    assert result["job"] == "57"
    assert calls[0][1:] == ("01:00:00", "afterany:55:56")
    assert calls[0][0]["XPLACE_PHASES"] == "imbalance"
    assert followup.read(root / "state.json")["jobs"][-1]["parent_job"] == "1"
    with pytest.raises(ValueError):
        followup.submit_retry(config, "retry", "1")


def test_submission_timeout_stops_and_preserves_reservation(campaign, monkeypatch):
    config, root, state = campaign
    bundle("original")
    bundle("retry", "value = 2\n")
    followup.validate_repair(config, "retry", "1", repair(config))
    state["phase"] = "dispatching"
    followup.save(root / "state.json", state)
    monkeypatch.setattr(followup.xplace_ctl, "remote", lambda *args, **kwargs: "")

    def fail(*args, **kwargs):
        raise TimeoutError("SSH disconnected after possible sbatch acceptance")

    monkeypatch.setattr(followup.xplace_ctl, "submit", fail)
    with pytest.raises(TimeoutError):
        followup.submit_retry(config, "retry", "1")
    result = followup.read(root / "state.json")
    assert not result["enabled"] and result["submissions"]["retry"]["status"] == "needs_reconciliation"


def test_changed_staging_and_eight_gpu_requests_are_rejected(campaign):
    config, root, state = campaign
    directory = bundle("modified")
    (directory / "upload/runtime/main.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="Staging files"):
        followup.verify_bundle("modified")
    bundle("eight", gpus=8)
    with pytest.raises(ValueError, match="four GPUs"):
        followup.verify_bundle("eight")


@pytest.mark.parametrize("identity", ["pinned-thread", "wrong-thread"])
def test_actual_subprocess_protocol_keeps_thread_id(campaign, identity):
    config, root, state = campaign
    executable = root / "fake-codex"
    executable.write_text(f"#!{sys.executable}\nimport json, sys\n"
                          f"print(json.dumps({{'type': 'thread.started', 'thread_id': '{identity}'}}))\n"
                          "print(json.dumps({'type': 'turn.completed'}))\n"
                          "assert sys.argv[-2] == 'pinned-thread'\n")
    executable.chmod(0o700)
    config["codex"] = str(executable)
    event = root / "protocol-test"
    event.mkdir()
    (event / "prompt.txt").write_text("test fixture, no model request\n")
    if identity == config["thread_id"]:
        followup.run_codex(config, root, event)
    else:
        with pytest.raises(RuntimeError, match="pinned thread"):
            followup.run_codex(config, root, event)
