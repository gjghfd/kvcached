import copy
import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import performance_archive_bridge as bridge
import performance_path_recovery as original


@pytest.fixture
def campaign(tmp_path):
    repo = Path(bridge.__file__).resolve().parents[2]
    stage_id = "native-second-N2"
    directory = tmp_path / "stages" / stage_id
    directory.mkdir(parents=True)
    entry = {"stage_id": stage_id, "job": 123, "phase": "archiving", "terminal": "COMPLETED",
             "accounting_exit_code": "0:0", "plan": {"stage_id": stage_id, "phase_name": stage_id,
                                                       "trace_sha256": "1" * 64}}
    state = {"repo": str(repo), "stages": {stage_id: entry},
             "frozen_runtime_sha256": {bridge.ARCHIVER_NAME: bridge.digest(repo / bridge.ARCHIVER_NAME)}}
    (tmp_path / "campaign-state.json").write_text(json.dumps(state))
    manifest = tmp_path / "archive-bridge-manifest.json"
    manifest.write_text(json.dumps({name: bridge.digest(repo / name)
                                   for name in (bridge.ENTRY_NAME, bridge.BRIDGE_NAME, bridge.ALIAS_NAME)}))
    host_manifest = tmp_path / "host-source-manifest-r2.json"
    host_manifest.write_text(json.dumps({name: bridge.digest(repo / name)
                                        for name in (bridge.ORIGINAL_NAME, bridge.RISK_NAME)}))
    return SimpleNamespace(root=tmp_path, repo=repo, directory=directory, entry=entry, state=state,
                           stage_id=stage_id, manifest=manifest, host_manifest=host_manifest,
                           calls=[], sleeps=[], events=[], original=original)


def source_guard(case, loaded_entry_sha256=None):
    return bridge.SourceGuard(case.root, case.manifest, case.host_manifest, case.repo / bridge.ENTRY_NAME,
                              loaded_entry_sha256 or bridge.digest(case.repo / bridge.ENTRY_NAME), case.original)


def callback(case, results, *, recover=None, guard=None):
    def run(command, **options):
        case.calls.append((command, options))
        case.events.append("frozen_child")
        return results.pop(0)

    return bridge.ArchiveBridge(source_guard(case) if guard is None else guard,
                                run=run, sleep=case.sleeps.append, recover=recover)


def child(code=0, error=""):
    return subprocess.CompletedProcess(["fake-frozen-archiver"], code, stdout="", stderr=error)


def fake_recovery(case):
    def recover(root, stage_id):
        assert root == case.root and stage_id == case.stage_id
        assert os.getpid() == case.coordinator_pid
        log = max(case.directory.glob("archive-123-attempt-*.log"), key=lambda path: path.stat().st_mtime_ns)
        assert log.read_text().rstrip().splitlines()[-1] == bridge.ALIAS_ERROR
        assert case.events == ["frozen_child"]
        case.events.append("same_coordinator_recovery")
        raw = case.directory / "live-trace-123"
        raw.mkdir()
        payload = b"original complete output\n"
        (raw / "result.json").write_bytes(payload)
        (case.directory / "raw-123.sha256").write_text(hashlib.sha256(payload).hexdigest() + "  ./result.json\n")
        proof = case.directory / "new-fake-installation-receipt.json"
        proof.write_text(json.dumps({"job": 123, "stage_id": stage_id, "fixture_only": True}))
        return {"job": 123, "stage_id": stage_id, "installation_receipt": str(proof), "already_SHA_archived": False}

    return recover


def test_same_coordinator_alias_recovery_then_real_execute_stage_strict_path(campaign, monkeypatch):
    case = campaign
    case.coordinator_pid = os.getpid()
    job = callback(case, [child(1, "Traceback\n" + bridge.ALIAS_ERROR + "\n")], recover=fake_recovery(case))

    def strict(actual_root, entry):
        assert actual_root == case.root and entry["job"] == 123
        assert case.events == ["frozen_child", "same_coordinator_recovery"]
        case.events.append("strict_downstream")
        return {"complete_requests_verified": True}

    monkeypatch.setattr(original.followup, "verify_stage", strict)
    subprocess_before = subprocess.run
    with bridge.installed_archive_bridge(job):
        entry, report = original.followup.execute_stage(case.root, case.state, case.entry["plan"])
    assert entry["phase"] == "verified" and report["complete_requests_verified"] is True
    assert case.events == ["frozen_child", "same_coordinator_recovery", "strict_downstream"]
    assert len(case.calls) == 1 and case.sleeps == []
    assert subprocess.run is subprocess_before
    assert bridge.stress_campaign.archive_job is bridge.ORIGINAL_ARCHIVE_JOB
    returned = json.loads(next(job.evidence.rglob("returned.json")).read_text())
    assert returned["action"] == "same_coordinator_alias_recovery"
    assert returned["strict_downstream_verified"] is False
    assert returned["root"] == str(case.root) and returned["job"] == 123


def test_strict_failure_after_recovery_stays_diagnosis_not_success_or_resubmit(campaign, monkeypatch):
    case = campaign
    case.coordinator_pid = os.getpid()
    job = callback(case, [child(1, bridge.ALIAS_ERROR)], recover=fake_recovery(case))
    monkeypatch.setattr(original.followup, "verify_stage", lambda *args: (_ for _ in ()).throw(ValueError("strict rejection")))
    with bridge.installed_archive_bridge(job), pytest.raises(ValueError, match="strict rejection"):
        original.followup.execute_stage(case.root, case.state, case.entry["plan"])
    state = json.loads((case.root / "campaign-state.json").read_text())
    assert state["stages"][case.stage_id]["phase"] == "diagnosis_required"
    assert len(case.calls) == 1 and not case.sleeps
    assert bridge.stress_campaign.archive_job is bridge.ORIGINAL_ARCHIVE_JOB


def test_nonalias_child_retries_300_and_preserves_original_phase_save(campaign):
    case = campaign
    job = callback(case, [child(1, "ValueError: different error\n"), child()])
    job(case.directory, 123, case.repo / "experiments/dp_opt", case.entry)
    assert len(case.calls) == 2 and case.sleeps == [300]
    assert (case.directory / "archive-123-attempt-1.log").read_text() == "ValueError: different error\n"
    saved = json.loads((case.directory / "campaign-state.json").read_text())
    assert saved["phase"] == "terminal_SHA_archive_and_independent_verification"
    assert saved["job"] == 123 and saved["plan"] == case.entry["plan"]
    assert case.entry["phase"] == saved["phase"]


def test_bounded_nonalias_failures_keep_exact_12_attempts_and_300_sequence(campaign):
    case = campaign
    job = callback(case, [child(1, "transport error\n") for count in range(12)])
    with pytest.raises(RuntimeError, match="bounded retries"):
        job(case.directory, 123, case.repo / "experiments/dp_opt", case.entry)
    assert len(case.calls) == 12 and case.sleeps == [300] * 12
    assert len(list(case.directory.glob("archive-123-attempt-*.log"))) == 12
    failed = json.loads(next(job.evidence.rglob("failed.json")).read_text())
    assert failed["GPU_resubmission_authorized"] is False


@pytest.mark.parametrize("terminal,exit_code", [("FAILED", "1:0"), ("COMPLETED", "1:0")])
def test_alias_tail_not_recovered_for_noncompleted_or_nonzero_exit(campaign, terminal, exit_code):
    case = campaign
    case.entry.update({"terminal": terminal, "accounting_exit_code": exit_code})
    (case.root / "campaign-state.json").write_text(json.dumps(case.state))
    job = callback(case, [child(1, bridge.ALIAS_ERROR), child()], recover=lambda *args: pytest.fail("not attributable alias"))
    job(case.directory, 123, case.repo / "experiments/dp_opt", case.entry)
    assert case.sleeps == [300] and len(case.calls) == 2


@pytest.mark.parametrize("tail", ["Unsafe raw progress pointer", bridge.ALIAS_ERROR + " extra", bridge.ALIAS_ERROR + "\nnext failure"])
def test_only_exact_current_tail_authorizes_recovery(campaign, tail):
    case = campaign
    job = callback(case, [child(1, tail), child()], recover=lambda *args: pytest.fail("must not recover"))
    job(case.directory, 123, case.repo / "experiments/dp_opt", case.entry)
    assert case.sleeps == [300]


def test_recovery_failure_retains_closed_log_and_stops_without_next_child(campaign):
    case = campaign

    def failure(root, stage_id):
        assert (case.directory / "archive-123-attempt-1.log").read_text().rstrip() == bridge.ALIAS_ERROR
        retained = case.root / "fake-recovery-partial"
        retained.mkdir()
        (retained / "partial").write_bytes(b"preserve\n")
        raise ValueError("samefile proof rejected")

    job = callback(case, [child(1, bridge.ALIAS_ERROR)], recover=failure)
    with pytest.raises(ValueError, match="samefile proof"):
        job(case.directory, 123, case.repo / "experiments/dp_opt", case.entry)
    assert (case.root / "fake-recovery-partial/partial").read_bytes() == b"preserve\n"
    assert len(case.calls) == 1 and not case.sleeps
    assert list(job.evidence.rglob("failed.json")) and list(job.evidence.rglob("closed-1.json"))


def test_restart_logs_are_append_only_never_overwritten(campaign):
    case = campaign
    prior = {number: f"original closed {number}\n" for number in (1, 2, 10)}
    for number, text in prior.items():
        (case.directory / f"archive-123-attempt-{number}.log").write_text(text)
    job = callback(case, [child(1, "different failure\n"), child()])
    job(case.directory, 123, case.repo / "experiments/dp_opt", case.entry)
    assert all((case.directory / f"archive-123-attempt-{number}.log").read_text() == text for number, text in prior.items())
    assert (case.directory / "archive-123-attempt-11.log").read_text() == "different failure\n"
    assert (case.directory / "archive-123-attempt-12.log").read_text() == ""


def test_legacy_existing_raw_uses_frozen_resume_not_new_helper(campaign):
    case = campaign
    case.entry["phase"] = "verified"
    (case.root / "campaign-state.json").write_text(json.dumps(case.state))
    raw = case.directory / "live-trace-123"
    raw.mkdir()
    (raw / "old").write_bytes(b"old archived bytes\n")
    report = case.directory / "archive-123.json"
    report.write_bytes(b"old canonical report\n")
    job = callback(case, [child()], recover=lambda *args: pytest.fail("legacy must use original resume"))
    job(case.directory, 123, case.repo / "experiments/dp_opt", case.entry)
    assert "--resume" in case.calls[0][0]
    assert (raw / "old").read_bytes() == b"old archived bytes\n" and report.read_bytes() == b"old canonical report\n"
    assert not (case.directory / "progress-alias-123-installation.json").exists()


@pytest.mark.parametrize("wrong", ["job", "boolean_job", "stage", "plan", "archive", "scripts", "duplicate_owner", "terminal"])
def test_owned_job_stage_master_and_archive_mapping_fail_closed(campaign, wrong):
    case = campaign
    job = callback(case, [])
    entry = copy.deepcopy(case.entry)
    actual_job, archive, scripts = 123, case.directory, case.repo / "experiments/dp_opt"
    if wrong == "job":
        actual_job = 999
    elif wrong == "boolean_job":
        entry["job"] = True
    elif wrong == "stage":
        entry["stage_id"] = "other-stage"
    elif wrong == "plan":
        entry["plan"]["trace_sha256"] = "0" * 64
    elif wrong == "archive":
        archive = case.root
    elif wrong == "scripts":
        scripts = case.root
    elif wrong == "terminal":
        entry["terminal"] = "FAILED"
    else:
        case.state["stages"]["duplicate"] = copy.deepcopy(case.entry)
        (case.root / "campaign-state.json").write_text(json.dumps(case.state))
    before = (case.root / "campaign-state.json").read_bytes()
    with pytest.raises(ValueError):
        job(archive, actual_job, scripts, entry)
    assert not case.calls and not case.sleeps
    assert (case.root / "campaign-state.json").read_bytes() == before
    assert not list(case.directory.glob("archive-123-attempt-*.log"))


@pytest.mark.parametrize("tamper", ["extra", "missing", "entry_sha", "alias_sha", "loaded_entry_sha", "original_manifest"])
def test_independent_three_file_and_original_two_file_manifests_are_strict(campaign, tamper):
    case = campaign
    contents = json.loads(case.manifest.read_text())
    loaded_entry_sha256 = None
    if tamper == "extra":
        contents["experiments/dp_opt/extra.py"] = "0" * 64
    elif tamper == "missing":
        del contents[bridge.ALIAS_NAME]
    elif tamper in ("entry_sha", "alias_sha"):
        contents[bridge.ENTRY_NAME if tamper == "entry_sha" else bridge.ALIAS_NAME] = "0" * 64
    elif tamper == "loaded_entry_sha":
        loaded_entry_sha256 = "0" * 64
    else:
        case.host_manifest.write_text("{}")
    case.manifest.write_text(json.dumps(contents))
    with pytest.raises(ValueError):
        source_guard(case, loaded_entry_sha256)
    assert not case.calls


def test_manifest_changes_after_load_stop_before_phase_or_child(campaign):
    case = campaign
    job = callback(case, [])
    before = copy.deepcopy(case.entry)
    case.manifest.write_text("{}")
    with pytest.raises(ValueError, match="manifest changed"):
        job(case.directory, 123, case.repo / "experiments/dp_opt", case.entry)
    assert case.entry == before and not case.calls


def test_extra_host_sources_cannot_be_declared_as_gpu_closure(campaign):
    case = campaign
    case.state["frozen_runtime_sha256"][bridge.BRIDGE_NAME] = bridge.digest(case.repo / bridge.BRIDGE_NAME)
    (case.root / "campaign-state.json").write_text(json.dumps(case.state))
    with pytest.raises(ValueError, match="host-only"):
        source_guard(case)


def test_loaded_source_receipt_binds_root_job_stage_and_host_only_inventory(campaign):
    case = campaign
    job = callback(case, [child()])
    loaded = json.loads((job.evidence / "loaded-source.json").read_text())
    assert loaded["root"] == str(case.root) and loaded["coordinator_pid"] == os.getpid()
    assert loaded["bound_stages_at_load"] == [{"stage_id": case.stage_id, "job": 123, "phase": "archiving"}]
    assert loaded["extra_sources_host_only"] is True and loaded["GPU_manifest_modified"] is False
    assert loaded["loaded_source_sha256"] == json.loads(case.manifest.read_text())
    assert loaded["host_source_manifest_sha256"] == bridge.digest(case.host_manifest)
    assert not (job.evidence / "loaded-source.json").stat().st_mode & 0o222


def test_only_archive_job_is_hooked_and_exception_restores_it(campaign):
    case = campaign
    job = callback(case, [])
    run_before, save_before, verify_before = subprocess.run, bridge.stress_campaign.save_state, original.followup.verify_stage
    with pytest.raises(RuntimeError, match="fixture"), bridge.installed_archive_bridge(job):
        assert bridge.stress_campaign.archive_job is job
        assert subprocess.run is run_before and bridge.stress_campaign.save_state is save_before
        assert original.followup.verify_stage is verify_before
        raise RuntimeError("fixture")
    assert bridge.stress_campaign.archive_job is bridge.ORIGINAL_ARCHIVE_JOB


def test_entry_main_imports_original_same_basename_not_itself(monkeypatch):
    entry = Path(bridge.__file__).parent / "host_archive/performance_path_recovery.py"
    called = []
    before_path = list(sys.path)
    monkeypatch.setattr(bridge, "run_entry", lambda module, source, checksum, argv:
                        called.append((module, source, checksum, argv, sys.path[0])))
    monkeypatch.setattr(sys, "argv", [str(entry), "fixture-args"])
    try:
        runpy.run_path(str(entry), run_name="__main__")
    finally:
        sys.path[:] = before_path
    assert called[0][0] is original and Path(called[0][0].__file__) == entry.parent.parent / "performance_path_recovery.py"
    assert called[0][1] == entry.resolve() and called[0][2] == bridge.digest(entry)
    assert called[0][3] == ["fixture-args"] and called[0][4] == str(entry.parent.parent)


def test_entry_spawn_import_is_noop_without_recursion(monkeypatch):
    entry = Path(bridge.__file__).parent / "host_archive/performance_path_recovery.py"
    before_path = list(sys.path)
    monkeypatch.setattr(bridge, "run_entry", lambda *args: pytest.fail("spawn must not reenter host coordinator"))
    runpy.run_path(str(entry), run_name="__mp_main__")
    assert sys.path == before_path


def test_entry_preserves_old_host_cli_and_removes_only_new_manifest_option(campaign, monkeypatch):
    case = campaign
    forwarded = ["--root", str(case.root), "--native-root", str(case.root / "parent"), "--risk-portability-audit",
                 "--host-source-manifest", str(case.host_manifest), "--audit-only"]
    arguments = [*forwarded, "--archive-bridge-manifest", str(case.manifest)]
    captured = []
    monkeypatch.setattr(original, "main", lambda: captured.append((list(sys.argv), bridge.stress_campaign.archive_job)))
    before = list(sys.argv)
    bridge.run_entry(original, case.repo / bridge.ENTRY_NAME, bridge.digest(case.repo / bridge.ENTRY_NAME), arguments)
    assert captured[0][0] == [str(case.repo / bridge.ENTRY_NAME), *forwarded]
    assert isinstance(captured[0][1], bridge.ArchiveBridge)
    assert sys.argv == before and bridge.stress_campaign.archive_job is bridge.ORIGINAL_ARCHIVE_JOB


@pytest.mark.parametrize("missing", ["--archive-bridge-manifest", "--host-source-manifest", "--risk-portability-audit"])
def test_new_entry_does_not_relax_original_host_manifest_or_risk_flags(campaign, missing):
    case = campaign
    options = {"--root": str(case.root), "--archive-bridge-manifest": str(case.manifest),
               "--host-source-manifest": str(case.host_manifest), "--risk-portability-audit": None}
    del options[missing]
    arguments = [argument for option, value in options.items() for argument in ([option] if value is None else [option, value])]
    with pytest.raises(SystemExit):
        bridge.run_entry(original, case.repo / bridge.ENTRY_NAME, bridge.digest(case.repo / bridge.ENTRY_NAME), arguments)


def test_closed_log_collision_preserves_prior_log_and_child_output(campaign):
    case = campaign

    def competing_child(command, **options):
        case.calls.append(command)
        (case.directory / "archive-123-attempt-1.log").write_bytes(b"unexpected existing writer\n")
        return child(1, bridge.ALIAS_ERROR + "\n")

    job = bridge.ArchiveBridge(source_guard(case), run=competing_child, sleep=case.sleeps.append,
                               recover=lambda *args: pytest.fail("unowned closed log cannot authorize recovery"))
    with pytest.raises(FileExistsError):
        job(case.directory, 123, case.repo / "experiments/dp_opt", case.entry)
    assert (case.directory / "archive-123-attempt-1.log").read_bytes() == b"unexpected existing writer\n"
    evidence = json.loads(next(job.evidence.rglob("child-1.json")).read_text())
    assert evidence["stderr"] == bridge.ALIAS_ERROR + "\n" and evidence["child_returncode"] == 1
    assert list(job.evidence.rglob("failed.json")) and not case.sleeps


@pytest.mark.parametrize("wrong", ["job", "stage"])
def test_recovery_return_must_match_owned_stage(campaign, wrong):
    case = campaign
    result = {"job": 999 if wrong == "job" else 123, "stage_id": "other" if wrong == "stage" else case.stage_id}
    job = callback(case, [child(1, bridge.ALIAS_ERROR)], recover=lambda *args: result)
    with pytest.raises(ValueError, match="owning coordinator stage"):
        job(case.directory, 123, case.repo / "experiments/dp_opt", case.entry)
    assert len(case.calls) == 1 and not case.sleeps
    assert list(job.evidence.rglob("failed.json"))


def test_bridge_rejects_concurrent_same_process_calls(campaign):
    case = campaign
    job = callback(case, [])
    assert job.busy.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="synchronous owning coordinator"):
            job(case.directory, 123, case.repo / "experiments/dp_opt", case.entry)
    finally:
        job.busy.release()
    assert not case.calls


def test_root_binding_is_rechecked_between_child_and_recovery(campaign):
    case = campaign

    def changed_owner(command, **options):
        case.calls.append(command)
        changed = copy.deepcopy(case.state)
        changed["stages"][case.stage_id]["job"] = 999
        (case.root / "campaign-state.json").write_text(json.dumps(changed))
        return child(1, bridge.ALIAS_ERROR)

    job = bridge.ArchiveBridge(source_guard(case), run=changed_owner,
                               recover=lambda *args: pytest.fail("master ownership changed"), sleep=case.sleeps.append)
    with pytest.raises(ValueError, match="owned master stage"):
        job(case.directory, 123, case.repo / "experiments/dp_opt", case.entry)
    assert (case.directory / "archive-123-attempt-1.log").exists()
    assert len(case.calls) == 1 and not case.sleeps


def test_private_alias_source_is_compiled_from_manifest_bytes_not_cached_module(campaign, monkeypatch):
    case = campaign
    monkeypatch.setitem(sys.modules, "archive_progress_alias_recovery", SimpleNamespace(recover=lambda *args: pytest.fail("cached module")))
    guard = source_guard(case)
    assert guard.alias.__name__ == "_performance_archive_bridge_alias"
    assert guard.alias.recover.__module__ == guard.alias.__name__
    assert Path(guard.alias.__file__).resolve() == case.repo / bridge.ALIAS_NAME
    assert guard.loaded[bridge.ALIAS_NAME] == bridge.digest(guard.alias.__file__)


def test_new_entry_rejects_self_import_origin_before_original_main(campaign):
    case = campaign
    case.original = SimpleNamespace(__file__=str(case.repo / bridge.ENTRY_NAME))
    with pytest.raises(ValueError, match="source origin differs"):
        source_guard(case)
    assert not case.calls


@pytest.mark.parametrize("option", ["--root", "--archive-bridge-manifest", "--host-source-manifest", "--risk-portability-audit"])
def test_duplicate_manifest_or_root_flags_fail_closed(campaign, option):
    case = campaign
    options = {"--root": str(case.root), "--archive-bridge-manifest": str(case.manifest),
               "--host-source-manifest": str(case.host_manifest), "--risk-portability-audit": None}
    arguments = [argument for name, value in options.items() for argument in ([name] if value is None else [name, value])]
    value = options[option]
    arguments.extend([option] if value is None else [option, value])
    with pytest.raises(ValueError, match="unique explicit host option"):
        bridge.run_entry(original, case.repo / bridge.ENTRY_NAME, bridge.digest(case.repo / bridge.ENTRY_NAME), arguments)
