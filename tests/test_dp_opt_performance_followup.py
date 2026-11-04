import copy
import importlib
import json
import os
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("performance_followup")


@pytest.fixture
def analysis(driver):
    return importlib.import_module("performance_followup_analysis")


def test_exact_deployment_includes_shell_startup_tests_and_wildcard_sources(driver):
    repo = Path(__file__).resolve().parents[1]
    manifest = driver.runtime_manifest(repo, {})
    launch = (repo / "experiments/dp_opt/trace-benefit.sbatch").read_text()
    required = driver.re.findall(r'\$repo/(tests/[A-Za-z0-9_.-]+\.py)', launch)
    assert required
    assert all(name in manifest for name in required)
    assert "tests/test_dp_opt_live_placement.py" in manifest
    assert "tests/test_vllm_engine_core_shutdown.py" in manifest
    assert "pyproject.toml" in manifest
    assert "experiments/dp_opt/analyze_trace.py" in manifest
    assert "experiments/dp_opt/analyze_layout_cost.py" in manifest
    assert "kvcached/integration/vllm/patches.py" in manifest
    assert "experiments/dp_opt/docs/ilp-example.json" in manifest
    assert all(str(path.relative_to(repo)) in manifest for path in (repo / "experiments/dp_opt").glob("live_*.py"))


def test_protected_patch_is_copy_only_and_bound_to_SHA(driver):
    state = {"frozen_runtime_sha256": {"kvcached/integration/vllm/patches.py": "bound",
             "experiments/dp_opt/analyze_trace.py": "extra"}, "native_runtime_sha256": {},
             "protected_vllm_sha256": "bound"}
    files = driver.deployment_files(state)
    assert "kvcached/integration/vllm/patches.py" not in files
    assert "experiments/dp_opt/analyze_trace.py" in files
    state["protected_vllm_sha256"] = "different"
    with pytest.raises(ValueError, match="protected SHA"):
        driver.deployment_files(state)


def put_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


@pytest.mark.parametrize("has_state", [False, True])
def test_cli_failure_persists_without_reclassifying_or_submitting(driver, tmp_path, monkeypatch, has_state):
    root = tmp_path / "campaign"
    if has_state:
        put_json(root / "campaign-state.json", {"phase": "rate_analysis", "stages": {"rate-0": {"job": 278600}}})
    monkeypatch.setattr(driver.sys, "argv", ["performance_followup.py", "--root", str(root),
                                           "--native-root", str(tmp_path / "native")])

    def failed_run(*args, **kwargs):
        raise ValueError("Independent measurement mismatch")

    monkeypatch.setattr(driver, "run", failed_run)
    with pytest.raises(ValueError, match="Independent measurement mismatch"):
        driver.main()
    failures = list(root.glob("coordinator-error-*.json"))
    assert len(failures) == 1
    assert driver.read_json(failures[0])["automatic_duplicate_submission_authorized"] is False
    if has_state:
        state = driver.read_json(root / "campaign-state.json")
        assert state["phase"] == "diagnosis_required_followup"
        assert state["failure_phase"] == "rate_analysis"
        assert state["stages"] == {"rate-0": {"job": 278600}}
    else:
        assert not (root / "campaign-state.json").exists()


@pytest.fixture
def shell_environment(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    scripts = {
        "id": '#!/bin/bash\nprintf "%s\\n" "${PF_OWNER:-tester}"\n',
        "squeue": '#!/bin/bash\nprintf "%s\\n" "$PF_QUEUE"\n',
        "sacct": '#!/bin/bash\nwhile test "$#" -gt 0; do\n'
                 'if test "$1" = -j; then shift; job="$1"; fi\nshift\ndone\n'
                 'if test "$job" = 278529; then printf "%s\\n" "$PF_DEBUG_ACCOUNT"; '
                 'elif test "$job" = 278700; then printf "%s\\n" "$PF_NEW_DEBUG_ACCOUNT"; '
                 'elif test "$job" = 278701; then printf "%s\\n" "$PF_THIRD_DEBUG_ACCOUNT"; '
                 'else printf "%s\\n" "$PF_PERF_ACCOUNT"; fi\n',
        "sbatch": '#!/bin/bash\ncount=0\n'
                  'if test -f "$PF_SUBMISSION_COUNT"; then count="$(cat "$PF_SUBMISSION_COUNT")"; fi\n'
                  'printf "%s\\n" "$((count + 1))" > "$PF_SUBMISSION_COUNT"\nprintf "%s\\n" 278600\n',
    }
    for name, text in scripts.items():
        path = binaries / name
        path.write_text(text)
        path.chmod(0o755)
    return {**os.environ, "HOME": str(tmp_path), "USER": "tester", "PATH": str(binaries) + ":" + os.environ["PATH"],
            "PF_QUEUE": "278529|tester|qwen-head-hidden-probe|gpu_a800|PENDING",
            "PF_DEBUG_ACCOUNT": "278529|tester|qwen-head-hidden-probe|gpu_a800|PENDING|",
            "PF_PERF_ACCOUNT": "278600|tester|qwen-public-perf-followup|gpu_a800|RUNNING|",
            "PF_NEW_DEBUG_ACCOUNT": "278700|tester|qwen-upstream-hidden-probe|gpu_a800|RUNNING|",
            "PF_THIRD_DEBUG_ACCOUNT": "278701|tester|qwen-another-upstream-probe|gpu_a800|RUNNING|",
            "PF_SUBMISSION_COUNT": str(tmp_path / "submission-count")}


def run_shell(script, environment):
    return subprocess.run(["bash", "-c", "set -eu\n" + script], env=environment, text=True, capture_output=True)


@pytest.mark.parametrize("status", ["PENDING", "RUNNING", "COMPLETING"])
def test_verified_debug_is_allowed_without_waiting(driver, shell_environment, status):
    shell_environment["PF_QUEUE"] = f"278529|tester|qwen-head-hidden-probe|gpu_a800|{status}"
    shell_environment["PF_DEBUG_ACCOUNT"] = f"278529|tester|qwen-head-hidden-probe|gpu_a800|{status}|"
    assert run_shell(driver.queue_guard(278529), shell_environment).returncode == 0


@pytest.mark.parametrize("status", ["FAILED", "COMPLETED", "CANCELLED by 1000", "TIMEOUT"])
def test_terminal_debug_never_blocks_performance(driver, shell_environment, status):
    shell_environment["PF_QUEUE"] = ""
    shell_environment["PF_DEBUG_ACCOUNT"] = f"278529|tester|qwen-head-hidden-probe|gpu_a800|{status}|"
    assert run_shell(driver.queue_guard(278529), shell_environment).returncode == 0


@pytest.mark.parametrize("field,value", [(0, "12345"), (1, "another-owner"), (2, "qwen-other"),
                                       (3, "gpu_h100"), (4, "UNKNOWN")])
def test_debug_queue_identity_is_strict(driver, shell_environment, field, value):
    fields = shell_environment["PF_QUEUE"].split("|")
    fields[field] = value
    shell_environment["PF_QUEUE"] = "|".join(fields)
    result = run_shell(driver.queue_guard(278529), shell_environment)
    assert result.returncode == 73
    assert "PERFORMANCE_FOLLOWUP_GUARD_REJECTED" in result.stderr


@pytest.mark.parametrize("bad_account", ["", "278529|outsider|qwen-head-hidden-probe|gpu_a800|PENDING|",
                                        "278529|tester|qwen-other|gpu_a800|PENDING|"])
def test_accounting_cannot_be_substituted_by_queue(driver, shell_environment, bad_account):
    shell_environment["PF_DEBUG_ACCOUNT"] = bad_account
    assert run_shell(driver.queue_guard(278529), shell_environment).returncode == 73


def test_monitor_allows_only_own_performance_and_known_debug(driver, shell_environment):
    shell_environment["PF_QUEUE"] += "\n278600|tester|qwen-public-perf-followup|gpu_a800|RUNNING"
    assert run_shell(driver.queue_guard(278529, performance_job=278600), shell_environment).returncode == 0
    assert run_shell(driver.queue_guard(278529), shell_environment).returncode == 73
    shell_environment["PF_QUEUE"] += "\n278777|tester|qwen-public-perf-followup|gpu_a800|PENDING"
    assert run_shell(driver.queue_guard(278529, performance_job=278600), shell_environment).returncode == 73


def test_empty_queue_requires_no_unknown_allocation(driver, shell_environment):
    shell_environment["PF_QUEUE"] = ""
    assert run_shell(driver.queue_guard(None), shell_environment).returncode == 0
    shell_environment["PF_QUEUE"] = "278529|tester|qwen-head-hidden-probe|gpu_a800|PENDING"
    assert run_shell(driver.queue_guard(None), shell_environment).returncode == 73


@pytest.mark.parametrize("job", [0, -1, True, "01", "278529; echo unsafe", "278529.0"])
def test_job_identity_is_not_shell_input(driver, job):
    with pytest.raises(ValueError):
        driver.queue_guard(job)


def test_sync_scope_rewrites_string_and_stdin_and_restores(driver, shell_environment, monkeypatch):
    calls = []

    def original(text, *args, **kwargs):
        calls.append(text)
        result = run_shell(text, shell_environment)
        if result.returncode:
            raise RuntimeError(result.stderr)
        return result.stdout

    monkeypatch.setattr(driver.rebalance_campaign, "checked_ssh", original)
    monkeypatch.setattr(driver.rebalance_campaign, "checked_ssh_script", original)
    with pytest.raises(RuntimeError, match="GUARD_REJECTED"):
        with driver.sync_scope(278529, "qwen-head-hidden-probe"):
            driver.rebalance_campaign.checked_ssh(driver.EMPTY_QUEUE_GUARD + '; printf "string passed"')
            driver.rebalance_campaign.checked_ssh_script(driver.EMPTY_QUEUE_GUARD + "\n" + driver.EMPTY_QUEUE_GUARD)
            shell_environment["PF_QUEUE"] += "\n999|tester|unknown|gpu_a800|RUNNING"
            driver.rebalance_campaign.checked_ssh_script(driver.EMPTY_QUEUE_GUARD)
    assert len(calls) == 3
    assert all(driver.EMPTY_QUEUE_GUARD not in text and "sacct" in text for text in calls)
    assert driver.rebalance_campaign.checked_ssh is original
    assert driver.rebalance_campaign.checked_ssh_script is original


def submission_fixture(tmp_path, driver):
    stage_id = "rate-0"
    stage = tmp_path / "stages" / stage_id
    stage.mkdir(parents=True)
    entry = {"stage_id": stage_id, "phase": "staged", "remote_stage": str(stage),
             "submission_ledger": str(stage / "remote-ledger.txt"), "plan": {"walltime": "04:30:00"}}
    state = {"debug_job": 278529, "debug_name": "qwen-head-hidden-probe", "stages": {stage_id: entry}}
    return entry, state


def test_submission_ledger_is_atomic_and_idempotent(driver, tmp_path, shell_environment):
    entry, state = submission_fixture(tmp_path, driver)
    command = driver.submission_command(entry, state)
    assert "--gpus=2" in command and "--time=04:30:00" in command
    for repeat in range(2):
        result = run_shell(command, shell_environment)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "278600"
    assert Path(shell_environment["PF_SUBMISSION_COUNT"]).read_text().strip() == "1"


def test_lost_submission_response_recovers_committed_ledger_not_second_job(driver, tmp_path, shell_environment, monkeypatch):
    entry, state = submission_fixture(tmp_path, driver)
    commands = []

    def remote(command):
        commands.append(command)
        result = run_shell(command, shell_environment)
        assert result.returncode == 0, result.stderr
        if len(commands) == 1:
            raise RuntimeError("lost SSH response after committed sbatch")
        return result.stdout.strip()

    monkeypatch.setattr(driver.rebalance_campaign, "checked_ssh", remote)
    with pytest.raises(RuntimeError, match="lost SSH"):
        driver.reconcile_submission(tmp_path, state, entry)
    assert entry["phase"] == "submission_intent"
    assert driver.reconcile_submission(tmp_path, state, entry) == 278600
    assert "sbatch" not in commands[1]
    assert Path(shell_environment["PF_SUBMISSION_COUNT"]).read_text().strip() == "1"


def test_ambiguous_submission_intent_cannot_resubmit(driver, tmp_path, shell_environment, monkeypatch):
    entry, state = submission_fixture(tmp_path, driver)
    entry["phase"] = "submission_intent"
    commands = []

    def remote(command):
        commands.append(command)
        result = run_shell(command, shell_environment)
        if result.returncode:
            raise RuntimeError("No committed ledger; require manual reconciliation")
        return result.stdout

    monkeypatch.setattr(driver.rebalance_campaign, "checked_ssh", remote)
    with pytest.raises(RuntimeError, match="manual reconciliation"):
        driver.reconcile_submission(tmp_path, state, entry)
    assert all("sbatch" not in command for command in commands)
    assert not Path(shell_environment["PF_SUBMISSION_COUNT"]).exists()


def test_poll_ignores_terminal_text_from_failed_ssh(driver, tmp_path, monkeypatch):
    entry, state = submission_fixture(tmp_path, driver)
    entry.update({"job": 278600, "phase": "monitoring"})
    commands, sleeps = [], []
    results = iter([SimpleNamespace(returncode=255, stdout="278600|COMPLETED|0:0|a|b|c|d\n", stderr="broken SSH"),
                    SimpleNamespace(returncode=0, stdout="278600|COMPLETED|0:0|a|b|c|d\n", stderr="")])

    def query(command):
        commands.append(command)
        return next(results)

    monkeypatch.setattr(driver.stress_campaign, "ssh", query)
    monkeypatch.setattr(driver.time, "sleep", sleeps.append)
    assert driver.poll_stage(tmp_path, state, entry) == "COMPLETED"
    assert len(commands) == 2 and len(sleeps) == 1 and 295 < sleeps[0] <= 300
    assert all("278529" in command and "278600" in command and "squeue" in command for command in commands)
    assert entry["accounting_exit_code"] == "0:0"


def test_poll_third_job_rejection_preserves_recorded_job(driver, tmp_path, monkeypatch):
    entry, state = submission_fixture(tmp_path, driver)
    entry.update({"job": 278600, "phase": "monitoring"})
    monkeypatch.setattr(driver.stress_campaign, "ssh", lambda command: SimpleNamespace(
        returncode=73, stdout="", stderr="PERFORMANCE_FOLLOWUP_GUARD_REJECTED"))
    with pytest.raises(RuntimeError, match="never launch a third"):
        driver.poll_stage(tmp_path, state, entry)
    assert entry["job"] == 278600 and "terminal" not in entry


@pytest.fixture
def debug_registry_fixture(tmp_path, driver, shell_environment, monkeypatch):
    entry, state = submission_fixture(tmp_path, driver)
    entry.update({"job": 278600, "phase": "monitoring", "plan": {"walltime": "04:30:00", "seed": 2026103103,
                   "trace_sha256": "frozen-trace-SHA", "controls": driver.CONTROLS}})
    repo = Path(__file__).resolve().parents[1]
    state.update({"repo": str(repo), "frozen_runtime_sha256": {"experiments/dp_opt/performance_followup.py":
        driver.digest(repo / "experiments/dp_opt/performance_followup.py")},
        "debug_registry": driver.initialize_debug_registry(tmp_path, 278529, "qwen-head-hidden-probe")})
    driver.campaign_queue_guard(state, 278600)
    driver.stress_campaign.save_state(tmp_path, state)
    shell_environment.update({"PF_QUEUE": "278700|tester|qwen-upstream-hidden-probe|gpu_a800|RUNNING\n"
        "278600|tester|qwen-public-perf-followup|gpu_a800|RUNNING",
        "PF_DEBUG_ACCOUNT": "278529|tester|qwen-head-hidden-probe|gpu_a800|COMPLETED|"})
    calls = []
    def remote(command, *args, **kwargs):
        calls.append(command)
        result = run_shell(command, shell_environment)
        if result.returncode:
            raise RuntimeError(result.stderr)
        return result.stdout.strip()
    monkeypatch.setattr(driver.rebalance_campaign, "checked_ssh", remote)
    monkeypatch.setattr(driver.rebalance_campaign, "checked_ssh_script", remote)
    return tmp_path, state, entry, shell_environment, calls


def test_explicit_debug_registration_appends_sha_and_never_changes_frozen_campaign(driver, debug_registry_fixture):
    root, state, entry, environment, calls = debug_registry_fixture
    before = driver.digest(root / "campaign-state.json")
    initial = driver.digest(root / "authorized-debug/00000000.json")
    head = driver.register_debug(root, 278700, "qwen-upstream-hidden-probe")
    record = driver.read_json(root / "authorized-debug/00000001.json")
    assert record["previous_sha256"] == initial and record["identities"] == [
        {"job": 278700, "name": "qwen-upstream-hidden-probe", "owner": "tester"}]
    assert record["remote_verified"] and record["operator_explicit_registration"]
    assert "DEBUG_REGISTRATION_OWNER|tester" in record["verification_stdout"]
    assert head["sha256"] == driver.digest(root / "authorized-debug/00000001.json")
    assert driver.register_debug(root, 278700, "qwen-upstream-hidden-probe") == head and len(calls) == 1
    assert driver.digest(root / "campaign-state.json") == before and driver.digest(root / "authorized-debug/00000000.json") == initial
    assert run_shell(driver.campaign_queue_guard(state, 278600), environment).returncode == 0
    assert state["debug_job"] == 278529 and entry["plan"]["seed"] == 2026103103


@pytest.mark.parametrize("field,value", [(0, "278701"), (1, "another-owner"), (2, "qwen-wrong-name"),
                                       (3, "gpu_h100"), (4, "UNVERIFIED_STATE")])
def test_new_debug_registration_requires_matching_remote_accounting(driver, debug_registry_fixture, field, value):
    root, state, entry, environment, calls = debug_registry_fixture
    fields = environment["PF_NEW_DEBUG_ACCOUNT"].rstrip("|").split("|")
    fields[field] = value
    environment["PF_NEW_DEBUG_ACCOUNT"] = "|".join(fields) + "|"
    with pytest.raises(RuntimeError, match="GUARD_REJECTED"):
        driver.register_debug(root, 278700, "qwen-upstream-hidden-probe")
    assert not (root / "authorized-debug/00000001.json").exists()


@pytest.mark.parametrize("queue_has_old", [True, False])
def test_registry_allows_at_most_one_active_debug_in_queue_and_accounting(driver, debug_registry_fixture, queue_has_old):
    root, state, entry, environment, calls = debug_registry_fixture
    environment["PF_DEBUG_ACCOUNT"] = "278529|tester|qwen-head-hidden-probe|gpu_a800|RUNNING|"
    if queue_has_old:
        environment["PF_QUEUE"] += "\n278529|tester|qwen-head-hidden-probe|gpu_a800|RUNNING"
    with pytest.raises(RuntimeError, match="GUARD_REJECTED"):
        driver.register_debug(root, 278700, "qwen-upstream-hidden-probe")
    assert not (root / "authorized-debug/00000001.json").exists()


def test_unregistered_same_qwen_name_and_unknown_third_job_remain_rejected(driver, debug_registry_fixture):
    root, state, entry, environment, calls = debug_registry_fixture
    driver.register_debug(root, 278700, "qwen-upstream-hidden-probe")
    environment["PF_QUEUE"] += "\n278701|tester|qwen-upstream-hidden-probe|gpu_a800|PENDING"
    assert run_shell(driver.campaign_queue_guard(state, 278600), environment).returncode == 73
    with pytest.raises(ValueError, match="separate allocations"):
        driver.register_debug(root, 278600, driver.JOB_NAME)


@pytest.mark.parametrize("concurrent_guard_rejection", [False, True])
def test_poll_reloads_registered_exact_debug_ids_every_iteration(driver, debug_registry_fixture, monkeypatch, concurrent_guard_rejection):
    root, state, entry, environment, registrations = debug_registry_fixture
    environment["PF_QUEUE"] = "278529|tester|qwen-head-hidden-probe|gpu_a800|RUNNING\n"
    environment["PF_QUEUE"] += "278600|tester|qwen-public-perf-followup|gpu_a800|RUNNING"
    environment["PF_DEBUG_ACCOUNT"] = "278529|tester|qwen-head-hidden-probe|gpu_a800|RUNNING|"
    commands, sleeps = [], []
    frozen_plan = copy.deepcopy(entry["plan"])
    def rotate():
        environment["PF_DEBUG_ACCOUNT"] = "278529|tester|qwen-head-hidden-probe|gpu_a800|COMPLETED|"
        environment["PF_QUEUE"] = "278700|tester|qwen-upstream-hidden-probe|gpu_a800|RUNNING\n"
        environment["PF_QUEUE"] += "278600|tester|qwen-public-perf-followup|gpu_a800|RUNNING"
        driver.register_debug(root, 278700, "qwen-upstream-hidden-probe")
    def poll(command):
        commands.append(command)
        if concurrent_guard_rejection and len(commands) == 1:
            rotate()
        guard = command.split("\ndate -Is\n")[0]
        result = run_shell(guard, environment)
        if result.returncode:
            return SimpleNamespace(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        status = "RUNNING" if len(commands) == 1 else "COMPLETED"
        return SimpleNamespace(returncode=0, stdout=f"278600|{status}|0:0|start|end|elapsed|node\n", stderr="")
    def sleep(seconds):
        sleeps.append(seconds)
        if not concurrent_guard_rejection:
            rotate()
    monkeypatch.setattr(driver.stress_campaign, "ssh", poll)
    monkeypatch.setattr(driver.time, "sleep", sleep)
    assert driver.poll_stage(root, state, entry) == "COMPLETED"
    assert len(commands) == 2 and len(sleeps) == 1 and 290 < sleeps[0] <= 300
    assert "278700:qwen-upstream-hidden-probe" not in commands[0]
    assert "278700:qwen-upstream-hidden-probe" in commands[1]
    assert entry["plan"] == frozen_plan and state["debug_job"] == 278529
    assert state["debug_registry_observed"]["sequence"] == 1
    assert '"sequence": 0' in (root / "stages/rate-0/poll-278600.log").read_text()
    assert '"sequence": 1' in (root / "stages/rate-0/poll-278600.log").read_text()


def test_sync_scope_reads_current_registry_in_both_string_and_stdin_guards(driver, debug_registry_fixture):
    root, state, entry, environment, calls = debug_registry_fixture
    environment["PF_QUEUE"] = ""
    with driver.sync_scope(state["debug_job"], state["debug_name"], state):
        driver.rebalance_campaign.checked_ssh(driver.EMPTY_QUEUE_GUARD)
        environment["PF_QUEUE"] = "278700|tester|qwen-upstream-hidden-probe|gpu_a800|RUNNING"
        driver.register_debug(root, 278700, "qwen-upstream-hidden-probe")
        driver.rebalance_campaign.checked_ssh_script(driver.EMPTY_QUEUE_GUARD)
    assert "278700:qwen-upstream-hidden-probe" not in calls[0]
    assert "278700:qwen-upstream-hidden-probe" in calls[-1]
    assert all(driver.EMPTY_QUEUE_GUARD not in command for command in calls)


@pytest.mark.parametrize("change", ["initial", "last_record", "rollback", "gap", "unverified"])
def test_debug_authorization_sha_chain_cannot_be_tampered_or_rolled_back(driver, debug_registry_fixture, change):
    root, state, entry, environment, calls = debug_registry_fixture
    driver.register_debug(root, 278700, "qwen-upstream-hidden-probe")
    driver.campaign_queue_guard(state, 278600)
    initial, latest = root / "authorized-debug/00000000.json", root / "authorized-debug/00000001.json"
    if change == "rollback":
        latest.unlink()
    elif change == "gap":
        latest.rename(root / "authorized-debug/00000002.json")
    else:
        path = initial if change == "initial" else latest
        record = driver.read_json(path)
        if change == "unverified":
            record["remote_verified"] = False
        else:
            record["identities"][0]["name"] = "qwen-tampered"
        put_json(path, record)
    with pytest.raises(ValueError):
        driver.campaign_queue_guard(state, 278600)


def test_operator_debug_register_cli_requires_explicit_name_and_bound_prepared_root(driver, debug_registry_fixture, monkeypatch, capsys):
    import sys
    root, state, entry, environment, calls = debug_registry_fixture
    monkeypatch.setattr(sys, "argv", ["performance_followup.py", "--root", str(root), "--register-debug-job", "278700"])
    with pytest.raises(SystemExit):
        driver.main()
    assert not calls
    monkeypatch.setattr(sys, "argv", ["performance_followup.py", "--root", str(root), "--register-debug-job", "278700",
        "--register-debug-name", "qwen-upstream-hidden-probe"])
    driver.main()
    assert json.loads(capsys.readouterr().out)["sequence"] == 1 and len(calls) == 1


@pytest.fixture
def native_fixture(tmp_path, driver, analysis, monkeypatch):
    parent = tmp_path / "native"
    root = tmp_path / "followup"
    root.mkdir()
    raw = parent / "live-trace-278514"
    put_json(raw / "frozen/inputs/code-marks.json", [{"prompt_key": "code", "prompt_tokens": 100000, "output_tokens": 64}])
    put_json(raw / "frozen/inputs/conv-marks.json", [{"prompt_key": "conv", "prompt_tokens": 128, "output_tokens": 8,
        "original_output_tokens": 8, "original_response_sha256": "a" * 64}])
    put_json(raw / "frozen/inputs/prompt-bank.json", {"prompts": {"conv": {}, "code": {}}})
    put_json(raw / "frozen/inputs/input-manifest.json", {"dataset_label": "openthoughts3_closed"})
    put_json(raw / "public-context-calibration/source.json", {"default_head": True})
    put_json(raw / "public-context-fit.json", {"coefficients": {"cadence": 1.0}})
    put_json(root / "campaign-state.json", {"stages": {}})
    results = {}
    for name in analysis.native_names():
        put_json(raw / name / "marker.json", {"name": name})
        put_json(raw / "frozen" / (name + ".json"), {"seed": int(name.split("-")[1]), "duration_s": 900})
        put_json(raw / name / "trace.json", analysis.read_json(raw / "frozen" / (name + ".json")))
        put_json(raw / name / "config.json", {"move_fixed_s": .475, "move_bandwidth_gib_s": 1,
            "policies": list(analysis.search.POLICIES), "repeats": 1, "max_model_len": 131072,
            "cache_bytes": 25769803776, "kv_budget_mib": 10240, "request_timeout_s": 7200,
            "allocator_gpu_utilization": .98, "fairness_queue_threshold_s": 30,
            "workaware_optimizer": "robust", "seed": 20261021, "workaware_rate_priors": [.001, .012],
            "workaware_fit_sha256": analysis.digest(raw / "public-context-fit.json")})
        put_json(raw / name / "placement-fit.json", {"placement": "original"})
        (raw / name / "source.sha256").write_text(analysis.digest(raw / name / "placement-fit.json") + "  placement-fit.json\n")
        put_json(raw / name / "summary.json", {"status": "passed"})
        (raw / name / "exit-status").write_text("0\n")
        results[name] = {"status": "performance_verified", "greedy_equal": False}

    def freeze(building):
        put_json(building / "calibration-frozen/calibration-config.json", {"conv_rates_rps": [.01, .02, .03]})
        for ordinal in range(3):
            put_json(building / f"calibration-frozen/rate-{ordinal}.json", analysis.poisson.make_poisson(
                building / "frozen/inputs", 2026103103, [.01, .02, .03][ordinal]))

    freeze(raw)
    (raw / "exit-status").write_text("0\n")
    put_json(parent / "campaign-state.json", {"job": 278514, "native_performance_complete": True,
        "state_last_observed": "COMPLETED", "phase": "native_performance_SHA_verified_rate_calibration_ready"})
    put_json(parent / "independent-native-performance.json", results)
    (parent / "accounting-278514.txt").write_text("278514|COMPLETED|0:0|submit|start|end|elapsed|\n")
    (parent / "raw-278514.sha256").write_text("".join(f"{sha}  {name}\n" for name, sha in analysis.file_hashes(raw).items()))
    monkeypatch.setattr(analysis.search, "validate_frozen", lambda frozen: True)
    monkeypatch.setattr(analysis.search, "freeze_calibration", freeze)
    monkeypatch.setattr(analysis, "validate_inputs", lambda inputs: True)
    monkeypatch.setattr(analysis, "verify_profile", lambda source: {"coefficients": {"cadence": 1.0}})

    def verify(phase, bank):
        assert (phase / "marker.json").exists()
        return results[phase.name]

    monkeypatch.setattr(analysis, "verify", verify)
    return root, parent, raw, results


def test_six_native_cases_and_gpu_rate_freeze_rebuild_without_overwrite(native_fixture, analysis):
    root, parent, raw, results = native_fixture
    before = analysis.file_hashes(raw)
    proof = {"pid": 12345, "stopped": True}
    workspace = analysis.prepare_workspace(root, parent, 278514, coordinator_exit=proof)
    assert not (workspace / "calibration-frozen").exists()
    analysis.finalize_native_workspace(root, parent, 278514, None)
    assert len(analysis.read_json(root / "native-provenance.json")["six_cases"]) == 6
    assert analysis.prepare_workspace(root, parent, 278514, coordinator_exit=proof) == workspace
    analysis.finalize_native_workspace(root, parent, 278514, None)
    assert analysis.file_hashes(raw) == before
    assert analysis.file_hashes(workspace / "calibration-frozen") == analysis.file_hashes(raw / "calibration-frozen")


def test_incomplete_native_independent_review_blocks_followup(native_fixture, analysis):
    root, parent, raw, results = native_fixture
    truncated = {name: value for name, value in results.items() if name != analysis.native_names()[0]}
    put_json(parent / "independent-native-performance.json", truncated)
    with pytest.raises(ValueError, match="All six"):
        analysis.prepare_workspace(root, parent, 278514, coordinator_exit={"stopped": True})
    assert not (root / "workspace").exists()


def test_native_failed_status_is_never_reclassified(native_fixture, analysis):
    root, parent, raw, results = native_fixture
    state = analysis.read_json(parent / "campaign-state.json")
    state["state_last_observed"] = "FAILED"
    put_json(parent / "campaign-state.json", state)
    (parent / "accounting-278514.txt").write_text("278514|FAILED|1:0|\n")
    with pytest.raises(ValueError, match="existing native coordinator"):
        analysis.prepare_workspace(root, parent, 278514, coordinator_exit={"stopped": True})


def test_parent_raw_sha_is_required(native_fixture, analysis):
    root, parent, raw, results = native_fixture
    (raw / "frozen/inputs/code-marks.json").write_text("changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        analysis.prepare_workspace(root, parent, 278514, coordinator_exit={"stopped": True})


def test_immutable_json_and_safe_sha_paths(driver, tmp_path):
    path = tmp_path / "freeze.json"
    driver.write_json(path, {"seed": 42})
    driver.write_json(path, {"seed": 42})
    with pytest.raises(ValueError, match="immutable"):
        driver.write_json(path, {"seed": 43})
    with pytest.raises(ValueError, match="SHA differs"):
        driver.check_hashes(tmp_path, {"../escape": "0" * 64})


@pytest.mark.parametrize("kind,policy,duration,walltime", [("rate", "no_move", 3600, "04:30:00"),
    ("explore", "kv_only_workaware", 3600, "04:30:00"), ("heldout", "random_no_return", 10800, "06:00:00")])
def test_one_complete_original_trace_per_allocation(driver, tmp_path, kind, policy, duration, walltime):
    trace = tmp_path / "trace.json"
    put_json(trace, {"duration_s": duration, "seed": 2026103105, "model_rates_rps": [.001, .01], "requests": [1, 2]})
    put_json(tmp_path / "workspace/public-context-fit.json", {})
    put_json(tmp_path / "workspace/placement-fit.json", {})
    put_json(tmp_path / "native-provenance.json", {"movement_costs": {"move_fixed_s": .475, "move_bandwidth_gib_s": 1}})
    plan = driver.plan_for(tmp_path, "stage", "heldout" if kind == "heldout" else "rate-0", kind, policy, trace, walltime)
    assert plan["policy"] == policy and plan["duration_s"] == duration and plan["request_timeout_s"] == 7200
    assert plan["walltime"] == walltime and plan["seed"] == 2026103105
    assert plan["trace_sha256"] == driver.digest(trace)


@pytest.mark.parametrize("walltime", ["03:00:00", "4:99:00", "UNLIMITED", "6:00:00;echo unsafe"])
def test_walltime_requires_explicit_valid_budget(driver, walltime):
    with pytest.raises(ValueError):
        driver.policy_walltime(walltime)


def make_tail_trial(root, analysis):
    marks = [{"request_id": identifier, "model": model, "prompt_tokens": 16,
              "output_tokens": 2, "arrival_s": 0} for identifier, model in (("code", 0), ("conv", 1))]
    put_json(root / "trace.json", {"duration_s": 100, "requests": marks})
    put_json(root / "config.json", {"deadline_s": 120, "policies": ["no_move", "kv_only"]})
    put_json(root / "summary.json", {"status": "passed"})
    (root / "exit-status").write_text("0\n")
    for policy, origin in (("no_move", 100), ("kv_only", 1000)):
        records = [{"request_id": mark["request_id"], "model": mark["model"], "prompt_tokens": 16,
                    "arrived_at": origin, "submitted_at": origin, "reserved_at": origin,
                    "first_token_at": origin + 1, "completed_at": origin + 10,
                    "token_ids": [1, 2] if policy == "no_move" else [1, 3]} for mark in marks]
        directory = root / ("r0-" + policy)
        put_json(directory / "requests.json", records)
        put_json(directory / "workload.json", {"origin": origin, "requests": marks,
            "trace_sha256": analysis.digest(root / "trace.json"), "duration_s": 100})
        put_json(directory / "summary.json", analysis.analyze_tail_slo.request_metrics(records, origin, 100, 120))


def test_rate_tail_analysis_does_not_reintroduce_fatal_greedy_gate(analysis, tmp_path):
    make_tail_trial(tmp_path, analysis)
    original = analysis.amortized_campaign.analyze
    with pytest.raises(ValueError, match="different output token IDs"):
        original(tmp_path)
    with analysis.tail_observation():
        result = analysis.amortized_campaign.analyze(tmp_path)
        assert result["quality_equivalence_not_proven"]
        assert result["groups"]["all_requests"]["requests"] == 2
    assert analysis.amortized_campaign.analyze is original


@pytest.mark.parametrize("change", ["length", "missing", "metrics", "failed"])
def test_observation_tail_preserves_all_non_greedy_checks(analysis, tmp_path, change):
    make_tail_trial(tmp_path, analysis)
    requests = analysis.read_json(tmp_path / "r0-kv_only/requests.json")
    if change == "length":
        requests[0]["token_ids"].append(4)
    elif change == "missing":
        requests.pop()
    elif change == "metrics":
        summary = analysis.read_json(tmp_path / "r0-kv_only/summary.json")
        summary["mean_ttft_s"] = 999
        put_json(tmp_path / "r0-kv_only/summary.json", summary)
    else:
        (tmp_path / "exit-status").write_text("1\n")
    put_json(tmp_path / "r0-kv_only/requests.json", requests)
    with analysis.tail_observation(), pytest.raises(ValueError):
        analysis.amortized_campaign.analyze(tmp_path)


def test_public_adapters_restore_all_functions_on_failure(driver, analysis, tmp_path):
    originals = {name: getattr(driver.poisson, name) for name in ("verify", "validate_frozen", "verify_calibration")}
    original_tail = analysis.amortized_campaign.analyze
    with pytest.raises(RuntimeError):
        with analysis.split_public_analysis(tmp_path, lambda *args: None):
            assert all(getattr(driver.poisson, name) is not original for name, original in originals.items())
            raise RuntimeError("stop")
    assert all(getattr(driver.poisson, name) is original for name, original in originals.items())
    assert analysis.amortized_campaign.analyze is original_tail


def test_split_join_requires_all_seven_policies_and_never_fabricates_success(analysis, tmp_path):
    root = tmp_path
    (root / "evaluation").mkdir()
    entries = {}
    for ordinal, policy in enumerate(analysis.poisson.POLICIES):
        stage_id = "explore-0-" + policy
        entry = {"stage_id": stage_id, "job": 280000 + ordinal,
                 "plan": {"phase_name": "explore-0", "policy": policy,
                          "optimizer": "robust" if policy == "kv_only_workaware" else "point"}}
        entries[policy] = entry
        phase = root / "stages" / stage_id / f"live-trace-{entry['job']}" / "explore-0"
        put_json(phase / "config.json", {"policies": [policy], "repeats": 1, "workaware_fit_sha256": "same"})
        put_json(phase / "trace.json", {"seed": 10, "requests": []})
        put_json(phase / "references.json", [{"trace_request_id": "req", "token_ids": [1, 2]}])
        put_json(phase / f"r0-{policy}/requests.json", [{"request_id": "req", "token_ids": [1, ordinal + 2]}])
        (phase.parent.parent / f"raw-{entry['job']}.sha256").write_text("raw manifest retained")
    with pytest.raises(ValueError, match="all seven"):
        analysis.join_phase(root, "explore-0", {"no_move": entries["no_move"]})
    phase = analysis.join_phase(root, "explore-0", entries)
    put_json(root / "campaign-state.json", {"stages": {entry["stage_id"]: entry for entry in entries.values()}})
    assert (phase / "exit-status").read_text().strip() == "NOT_AN_ALLOCATION"
    assert analysis.read_json(phase / "summary.json")["status"] != "passed"
    calls = []

    def verifier(root, entry):
        calls.append(entry["stage_id"])
        return {"status": "verified", "greedy_observed": True}

    result = analysis.verify_join(root, phase, verifier)
    assert len(calls) == 7 and len(result["raw_SHA_provenance"]) == 7
    assert not result["cross_policy_greedy_observations"]["full_greedy_equivalence_passed"]
    assert result["cross_policy_greedy_observations"]["paired"][1]["first_difference_zero_based"] == 1
    wrong = copy.deepcopy(analysis.read_json(phase / "split-provenance.json"))
    wrong["sources"][1]["stage_id"] = entries["no_move"]["stage_id"]
    put_json(phase / "split-provenance.json", wrong)
    with pytest.raises(ValueError, match="policy binding"):
        analysis.verify_join(root, phase, verifier)


def test_explicit_deployment_files_and_shell_allocation_is_single_stage(driver):
    assert len(driver.FOLLOWUP_FILES) == 4
    assert all("performance_followup" in name or "performance-followup" in name for name in driver.FOLLOWUP_FILES)
    assert driver.NEW_DEPLOYMENT_MODULES == ("experiments/dp_opt/public_poisson_followup.py",
        "experiments/dp_opt/public_population_calibration.py", "experiments/dp_opt/public_quality_followup.py")
    assert driver.SYNC_FILES == driver.FOLLOWUP_FILES + driver.NEW_DEPLOYMENT_MODULES
    repo = Path(__file__).resolve().parents[1]
    script = repo / "experiments/dp_opt/performance-followup.sbatch"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    source = script.read_text()
    assert "#SBATCH --time=04:30:00" in source and "#SBATCH --partition=gpu_a800" in source
    assert "#SBATCH --gpus=2" in source and "head_hidden" not in source
    assert "--compute-upload" in source and "for policy" not in source


def test_frozen_runtime_closure_adds_poisson_dependencies_without_changing_parent(driver):
    repo = Path(__file__).resolve().parents[1]
    parent = {"experiments/dp_opt/live_trace.py": driver.digest(repo / "experiments/dp_opt/live_trace.py")}
    manifest = driver.runtime_manifest(repo, parent)
    assert manifest["experiments/dp_opt/live_trace.py"] == parent["experiments/dp_opt/live_trace.py"]
    assert "experiments/dp_opt/public_poisson_followup.py" in manifest
    assert set(driver.SYNC_FILES).issubset(manifest)
    assert not any("head_hidden" in name for name in manifest)
    with pytest.raises(ValueError, match="native frozen runtime changed"):
        driver.runtime_manifest(repo, {"experiments/dp_opt/live_trace.py": "0" * 64})


@pytest.mark.parametrize("modules_present", [False, True])
def test_stage_installs_missing_modules_via_real_protected_sync_then_copies_runtime(
        native_fixture, driver, analysis, shell_environment, monkeypatch, modules_present):
    root, parent, raw, results, templates = checkpoint_native_fixture(native_fixture, analysis, complete=1)
    workspace = analysis.prepare_workspace(root, parent, 278514, coordinator_exit={"stopped": True})
    repo = Path(__file__).resolve().parents[1]
    runtime = driver.runtime_manifest(repo, {})
    additional_existing = {"experiments/dp_opt/" + name for name in (
        "admission_cost_pilot.py", "agentic_capacity.py", "analyze_stress_sweep.py", "analyze_ttft_phases.py",
        "freeze_measured_control.py", "stress_sweep.py", "transfer_pilot.py", "verify_admission_cost.py")}
    parent_manifest = {name: sha for name, sha in runtime.items()
        if name not in additional_existing | set(driver.SYNC_FILES)}
    assert set(runtime) - set(parent_manifest) - set(driver.FOLLOWUP_FILES) == additional_existing | set(driver.NEW_DEPLOYMENT_MODULES)
    remote_home = root.parent / "mock-remote-home"
    remote_repo = remote_home / "kvcached/kvcached"
    missing = set(driver.FOLLOWUP_FILES) | (set() if modules_present else set(driver.NEW_DEPLOYMENT_MODULES))
    for name in set(runtime) - missing:
        target = remote_repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, target)
    protected = "kvcached/integration/vllm/patches.py"
    (remote_repo / protected).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo / protected, remote_repo / protected)
    remote_parent = remote_home / "run/kvcached-paracloud/state/live-trace-278514"
    shutil.copytree(raw, remote_parent)
    before_remote = driver.file_hashes(remote_repo)
    before_sources = {name: driver.digest(repo / name) for name in driver.NEW_DEPLOYMENT_MODULES}
    native_name = analysis.native_names()[1]
    plan = driver.plan_for(root, native_name, native_name, "native", None, workspace / "frozen" / (native_name + ".json"))
    entry = {"stage_id": native_name, "plan": plan, "phase": "prepared"}
    state = {"repo": str(repo), "frozen_runtime_sha256": runtime, "native_runtime_sha256": parent_manifest,
        "protected_vllm_sha256": driver.digest(repo / protected), "native_job": 278514,
        "debug_job": 278529, "debug_name": "qwen-head-hidden-probe", "stages": {native_name: entry}}
    environment = {**shell_environment, "HOME": str(remote_home),
        "PF_QUEUE": "278529|tester|qwen-head-hidden-probe|gpu_a800|RUNNING",
        "PF_DEBUG_ACCOUNT": "278529|tester|qwen-head-hidden-probe|gpu_a800|RUNNING|"}
    git = Path(shell_environment["PATH"].split(":")[0]) / "git"
    git.write_text('#!/bin/bash\ncase "$*" in\n'
        '  *"branch --show-current"*) printf "%s\\n" main ;;\n'
        '  *"remote get-url origin"*) printf "%s\\n" git@github.com:gjghfd/kvcached.git ;;\n'
        '  *"diff --"*) exit 0 ;;\n  *) exit 2 ;;\nesac\n')
    git.chmod(0o755)
    import hashlib
    monkeypatch.setattr(driver.rebalance_campaign, "PROTECTED_AEG_DIFF", hashlib.sha256(b"").hexdigest())
    commands, uploads = [], []
    real_run = subprocess.run

    def remote(command, *args, **kwargs):
        commands.append(command)
        result = real_run(["bash", "-c", "set -eu\n" + command], env=environment, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return result.stdout.strip()

    def transfer(command, *args, **kwargs):
        if command[0] == "scp":
            assert command[-1].startswith("paracloud:"), "Only a local mock transfer is permitted"
            uploads.append(command)
            shutil.copy2(command[-2], command[-1].split(":", 1)[1])
            return subprocess.CompletedProcess(command, 0)
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(driver.rebalance_campaign, "checked_ssh", remote)
    monkeypatch.setattr(driver.rebalance_campaign, "checked_ssh_script", remote)
    monkeypatch.setattr(subprocess, "run", transfer)
    driver.stage_inputs(root, state, entry)
    assert entry["phase"] == "staged" and len(uploads) == 1
    stage = root / "stages" / native_name
    upload = Path(entry["remote_stage"]) / "upload"
    assert driver.file_hashes(upload / "runtime") == runtime
    receipt = driver.read_json(stage / "protected-sync-manifest.json")
    assert {item["local"] for item in receipt} == set(driver.deployment_files(state))
    for name in driver.NEW_DEPLOYMENT_MODULES:
        item = next(item for item in receipt if item["local"] == name)
        assert item["old_sha256"] == (before_sources[name] if modules_present else "ABSENT")
        backup = Path(entry["remote_stage"]) / "old" / name
        assert backup.exists() is modules_present
        if modules_present:
            assert driver.digest(backup) == before_sources[name]
        assert driver.digest(remote_repo / name) == driver.digest(upload / "runtime" / name) == before_sources[name]
        assert driver.digest(repo / name) == before_sources[name]
    with tarfile.open(stage / "protected-sync.tar.gz") as bundle:
        assert set(driver.SYNC_FILES).issubset(bundle.getnames())
    assert all(driver.digest(remote_repo / name) == sha for name, sha in before_remote.items())
    assert any(".rebalance-sync" in command and "mv " in command for command in commands)
    assert all(driver.EMPTY_QUEUE_GUARD not in command for command in commands)
    assert not any("sbatch --parsable" in command for command in commands)


@pytest.mark.parametrize("csv,accepted", [
    ("GPU-one, NVIDIA A800-SXM4-80GB, 81920, 80000\nGPU-two, NVIDIA A800-SXM4-80GB, 81920, 70000\n", True),
    ("GPU-one, NVIDIA A800-SXM4-80GB, 81920, 69999\nGPU-two, NVIDIA A800-SXM4-80GB, 81920, 80000\n", False),
    ("GPU-one, NVIDIA A800-SXM4-80GB, 81920, 80000\n", False),
    ("GPU-one, NVIDIA A800-SXM4-80GB, 81920, 80000\n" * 3, False),
    ("GPU-one, NVIDIA H100, 81920, 80000\nGPU-two, NVIDIA A800-SXM4-80GB, 81920, 80000\n", False),
    ("GPU-one, NVIDIA A800-SXM4-80GB, 81920, N/A\nGPU-two, NVIDIA A800-SXM4-80GB, 81920, 80000\n", False),
    ("GPU-one, NVIDIA A800-SXM4-80GB, 81920\nGPU-two, NVIDIA A800-SXM4-80GB, 81920, 80000\n", False),
    ("", False),
])
def test_sbatch_gpu_csv_guard_uses_free_fourth_column_and_exact_two_a800(driver, tmp_path, csv, accepted):
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "experiments/dp_opt/performance-followup.sbatch").read_text()
    command = source.split("awk -F, ", 1)[1].split(' "$evidence/gpu-memory-before-load.csv"', 1)[0]
    gpu_file = tmp_path / "gpu.csv"
    gpu_file.write_text(csv)
    result = subprocess.run(["bash", "-c", "awk -F, " + command + " " + shlex.quote(str(gpu_file))], capture_output=True)
    assert (result.returncode == 0) is accepted


def test_prepare_only_cli_starts_waiting_native_with_checkpoint_cancel_authorized(driver, tmp_path, monkeypatch):
    import sys
    repo = Path(__file__).resolve().parents[1]
    root, parent = tmp_path / "followup", tmp_path / "native"
    manifest = {"experiments/dp_opt/live_trace.py": driver.digest(repo / "experiments/dp_opt/live_trace.py")}
    put_json(parent / "campaign-state.json", {"job": 278514, "phase": "monitoring_public_native_performance",
        "state_last_observed": "RUNNING", "native_performance_complete": False,
        "frozen_runtime_sha256": manifest, "protected_vllm_sha256": "not-deployed-in-prepare-only"})
    monkeypatch.setattr(sys, "argv", ["performance_followup.py", "--root", str(root), "--native-root", str(parent),
        "--native-coordinator-pid", "7777", "--allow-checkpoint-cancelled", "--prepare-only"])
    monkeypatch.setattr(driver.rebalance_campaign, "checked_ssh", lambda *args: pytest.fail("Prepare must not use SSH"))
    monkeypatch.setattr(driver.rebalance_campaign, "checked_ssh_script", lambda *args: pytest.fail("Prepare must not use SSH stdin"))
    monkeypatch.setattr(driver, "wait_native_archive", lambda *args: pytest.fail("Prepare must not enter polling or allocation"))
    monkeypatch.setattr(driver.subprocess, "run", lambda *args, **kwargs: pytest.fail("Prepare must not invoke external commands"))
    driver.main()
    before = driver.read_json(root / "campaign-state.json")
    assert before["phase"] == "waiting_existing_native_archive" and before["allow_checkpoint_cancelled"] is True
    assert before["native_job"] == 278514 and before["debug_job"] == 278529 and not before["stages"]
    assert before["native_runtime_sha256"] == manifest and set(driver.SYNC_FILES).issubset(before["frozen_runtime_sha256"])
    driver.main()
    after = driver.read_json(root / "campaign-state.json")
    assert {name: value for name, value in after.items() if name != "recorded_at"} == {
        name: value for name, value in before.items() if name != "recorded_at"}


def test_original_gpu_rate_freeze_mismatch_is_fatal_without_overwrite(native_fixture, analysis):
    root, parent, raw, results = native_fixture
    put_json(raw / "calibration-frozen/rate-0.json", {"seed": 999, "ordinal": 0})
    before = analysis.file_hashes(raw)
    (parent / "raw-278514.sha256").write_text("".join(f"{sha}  {name}\n" for name, sha in before.items()))
    with pytest.raises(ValueError, match="original measured-service freeze"):
        analysis.prepare_workspace(root, parent, 278514, coordinator_exit={"stopped": True})
        analysis.finalize_native_workspace(root, parent, 278514, None)
    assert analysis.file_hashes(raw) == before
    assert not (root / "workspace/calibration-frozen").exists()


@pytest.mark.parametrize("kind", ["rate", "native"])
def test_failed_allocation_is_archived_once_and_never_automatically_resubmitted(driver, tmp_path, monkeypatch, kind):
    entry, state = submission_fixture(tmp_path, driver)
    entry.update({"job": 278600, "terminal": "TIMEOUT", "accounting_exit_code": "0:15", "phase": "archiving"})
    entry["plan"]["stage_id"] = entry["stage_id"]
    entry["plan"]["kind"] = kind
    state["repo"] = str(tmp_path)
    calls = []

    def archive(directory, job, scripts, item):
        calls.append(job)
        raw = directory / f"live-trace-{job}"
        raw.mkdir()
        (raw / "exit-status").write_text("143\n")
        (directory / f"raw-{job}.sha256").write_text(f"{driver.digest(raw / 'exit-status')}  exit-status\n")

    monkeypatch.setattr(driver.stress_campaign, "archive_job", archive)
    monkeypatch.setattr(driver.rebalance_campaign, "checked_ssh", lambda *args: pytest.fail("No SSH or sbatch for failed job"))
    with pytest.raises(RuntimeError, match="do not use incomplete work"):
        driver.execute_stage(tmp_path, state, entry["plan"])
    assert calls == [278600] and entry["phase"] == "diagnosis_required"
    with pytest.raises(RuntimeError, match="explicit diagnosis"):
        driver.execute_stage(tmp_path, state, entry["plan"])
    assert calls == [278600]


def checkpoint_native_fixture(native_fixture, analysis, complete=3, terminal="TIMEOUT"):
    root, parent, raw, results = native_fixture
    templates = root.parent / "native-case-templates"
    for name in analysis.native_names():
        shutil.copytree(raw / name, templates / name)
    for index, name in enumerate(analysis.native_names()):
        if index > complete:
            shutil.rmtree(raw / name)
        elif index == complete:
            (raw / name / "exit-status").unlink()
            put_json(raw / name / "summary.json", {"status": "running"})
    shutil.rmtree(raw / "calibration-frozen")
    (raw / "exit-status").write_text("143\n")
    previous = analysis.read_json(parent / "campaign-state.json")
    previous.update({"native_performance_complete": False, "state_last_observed": terminal,
        "phase": "diagnosis_required_performance_not_blocked_by_greedy",
        "frozen_runtime_sha256": {}, "protected_vllm_sha256": "CPU-fixture-only"})
    put_json(parent / "campaign-state.json", previous)
    (parent / "independent-native-performance.json").unlink()
    (parent / "accounting-278514.txt").write_text(f"278514|{terminal}|0:15|submit|start|end|elapsed|\n")
    (parent / "raw-278514.sha256").write_text("".join(
        f"{sha}  {name}\n" for name, sha in analysis.file_hashes(raw).items()))
    return root, parent, raw, results, templates


def test_timeout_reuses_only_complete_native_cases_never_partial_timings(native_fixture, analysis, monkeypatch):
    root, parent, raw, results, templates = checkpoint_native_fixture(native_fixture, analysis)
    calls = []
    original = analysis.verify
    monkeypatch.setattr(analysis, "verify", lambda phase, bank: (calls.append(phase.name), original(phase, bank))[1])
    before = analysis.file_hashes(raw)
    workspace = analysis.prepare_workspace(root, parent, 278514, coordinator_exit={"stopped": True})
    review = analysis.read_json(root / "native-parent-review.json")
    assert list(review["available_cases"]) == analysis.native_names()[:3]
    assert list(review["excluded_cases"]) == analysis.native_names()[3:]
    assert calls == analysis.native_names()[:3]
    assert not (workspace / analysis.native_names()[3]).exists()
    assert not (workspace / "calibration-frozen").exists()
    assert review["terminal"] == "TIMEOUT" and review["raw_exit_status"] == "143"
    with pytest.raises(ValueError, match="before all six"):
        analysis.finalize_native_workspace(root, parent, 278514, None)
    assert analysis.file_hashes(raw) == before


@pytest.mark.parametrize("terminal", ["CANCELLED", "COMPLETED"])
def test_missing_raw_exit_marker_requires_cancelled_accounting_not_fabrication(native_fixture, analysis, terminal):
    root, parent, raw, results, templates = checkpoint_native_fixture(native_fixture, analysis, terminal=terminal)
    (raw / "exit-status").unlink()
    (parent / "raw-278514.sha256").write_text("".join(
        f"{sha}  {name}\n" for name, sha in analysis.file_hashes(raw).items()))
    if terminal == "COMPLETED":
        with pytest.raises(ValueError, match="exit status"):
            analysis.parent_checkpoint(parent, 278514, allow_checkpoint_cancelled=True)
    else:
        result = analysis.parent_checkpoint(parent, 278514, allow_checkpoint_cancelled=True)
        assert result["raw_exit_status"] is None
        assert result["terminal"] == "CANCELLED"
        assert not (raw / "exit-status").exists()


def test_cancelled_requires_explicit_scientific_checkpoint_authorization(native_fixture, analysis):
    root, parent, raw, results, templates = checkpoint_native_fixture(native_fixture, analysis, terminal="CANCELLED")
    with pytest.raises(ValueError, match="explicit scientific-checkpoint"):
        analysis.prepare_workspace(root, parent, 278514, coordinator_exit={"stopped": True})
    assert not (root / "workspace").exists()
    analysis.prepare_workspace(root, parent, 278514, True, {"stopped": True})
    review = analysis.read_json(root / "native-parent-review.json")
    assert review["scientific_checkpoint_cancelled_authorized"] and review["terminal"] == "CANCELLED"
    assert analysis.read_json(parent / "campaign-state.json")["native_performance_complete"] is False


def test_partially_written_next_native_configuration_is_excluded_not_used(native_fixture, analysis):
    root, parent, raw, results, templates = checkpoint_native_fixture(native_fixture, analysis)
    partial = raw / analysis.native_names()[3]
    (partial / "placement-fit.json").unlink()
    (partial / "source.sha256").unlink()
    (parent / "raw-278514.sha256").write_text("".join(
        f"{sha}  {name}\n" for name, sha in analysis.file_hashes(raw).items()))
    workspace = analysis.prepare_workspace(root, parent, 278514, coordinator_exit={"stopped": True})
    review = analysis.read_json(root / "native-parent-review.json")
    assert len(review["available_cases"]) == 3 and not (workspace / partial.name).exists()
    assert review["placement_fit_source_phase"] == analysis.native_names()[0]


@pytest.mark.parametrize("change", ["completed_structural_failure", "non_greedy_partial_error", "wrong_window", "wrong_policy"])
def test_checkpoint_does_not_relax_native_invariants(native_fixture, analysis, monkeypatch, change):
    root, parent, raw, results, templates = checkpoint_native_fixture(native_fixture, analysis)
    phase = raw / analysis.native_names()[0]
    if change == "completed_structural_failure":
        monkeypatch.setattr(analysis, "verify", lambda *args: (_ for unused in ()).throw(ValueError("ownership failure")))
    elif change == "non_greedy_partial_error":
        put_json(raw / analysis.native_names()[3] / "error.json", {"type": "RuntimeError", "message": "copy corrupted"})
    elif change == "wrong_window":
        payload = analysis.read_json(phase / "trace.json")
        payload["duration_s"] = 800
        put_json(phase / "trace.json", payload)
    else:
        config = analysis.read_json(phase / "config.json")
        config["policies"] = ["no_move"]
        put_json(phase / "config.json", config)
    (parent / "raw-278514.sha256").write_text("".join(
        f"{sha}  {name}\n" for name, sha in analysis.file_hashes(raw).items()))
    with pytest.raises(ValueError):
        analysis.prepare_workspace(root, parent, 278514, coordinator_exit={"stopped": True})
    assert not (root / "workspace").exists()


def test_native_rescue_original_seeds_four_policies_and_resume_then_freeze_once(native_fixture, driver, analysis, monkeypatch):
    root, parent, raw, results, templates = checkpoint_native_fixture(native_fixture, analysis, terminal="CANCELLED")
    (root / "campaign-state.json").unlink()
    before = analysis.file_hashes(raw)
    monkeypatch.setattr(driver, "native_coordinator_stopped", lambda parent, identity: {**identity, "stopped": True})
    monkeypatch.setattr(driver.rebalance_campaign, "checked_ssh", lambda *args: pytest.fail("No remote operations in CPU rescue"))
    allocated, freezes = [], []
    original_freeze = analysis.search.freeze_calibration

    def freeze(workspace):
        freezes.append(analysis.native_names())
        assert all((workspace / name / "exit-status").read_text().strip() == "0" for name in analysis.native_names())
        original_freeze(workspace)

    monkeypatch.setattr(analysis.search, "freeze_calibration", freeze)

    def verify_stage(root, entry):
        archive = root / "stages" / entry["stage_id"]
        driver.verify_raw(archive / f"live-trace-{entry['job']}", (archive / f"raw-{entry['job']}.sha256").read_text())
        review = analysis.read_json(root / "native-parent-review.json")
        controls = {field: review[field] for field in ("movement_costs", "placement_fit_sha256", "native_controller_seed")}
        return analysis.verify_native_case(archive / f"live-trace-{entry['job']}" / entry["plan"]["phase_name"],
            root / "workspace/frozen", controls, driver.digest(root / "workspace/public-context-fit.json"))

    def allocation(root, state, plan):
        stage_id = plan["stage_id"]
        if stage_id in state["stages"]:
            entry = state["stages"][stage_id]
            assert entry["plan"] == plan
            return entry, verify_stage(root, entry)
        if plan["kind"] == "rate":
            assert plan["policies"] == ["no_move"] and plan["duration_s"] == 3600
            assert plan["seed"] == 2026103103 and plan["controller_seed"] == 2026103103
            assert len(allocated) == 3 and len(freezes) == 1
            allocated.append(stage_id)
            job = 281504
            entry = {"stage_id": stage_id, "job": job, "plan": plan, "phase": "verified", "terminal": "COMPLETED"}
            state["stages"][stage_id] = entry
            phase = root / "stages" / stage_id / f"live-trace-{job}" / stage_id
            put_json(phase / "trace.json", driver.read_json(root / plan["trace_relative"]))
            put_json(phase / "summary.json", {"status": "passed"})
            (phase / "exit-status").write_text("0\n")
            archive = phase.parent.parent
            (archive / f"raw-{job}.sha256").write_text("".join(f"{sha}  {name}\n" for name, sha
                in driver.file_hashes(phase.parent).items()))
            driver.stress_campaign.save_state(root, state)
            return entry, {"CPU_simulated_rate": True}
        assert plan["kind"] == "native" and plan["policies"] == list(analysis.search.POLICIES)
        assert plan["duration_s"] == 900 and plan["request_timeout_s"] == 7200 and plan["walltime"] == "04:30:00"
        assert plan["seed"] == int(stage_id.split("-")[1]) and plan["controller_seed"] == 20261021
        assert plan["rates_rps"] == [.001, .012] and plan["optimizer"] == "robust"
        allocated.append(stage_id)
        job = 281500 + len(allocated)
        entry = {"stage_id": stage_id, "job": job, "plan": plan, "phase": "verified", "terminal": "COMPLETED"}
        state["stages"][stage_id] = entry
        archive = root / "stages" / stage_id
        shutil.copytree(templates / stage_id, archive / f"live-trace-{job}" / stage_id)
        (archive / f"raw-{job}.sha256").write_text("".join(f"{sha}  {name}\n" for name, sha
            in driver.file_hashes(archive / f"live-trace-{job}").items()))
        driver.stress_campaign.save_state(root, state)
        return entry, verify_stage(root, entry)

    monkeypatch.setattr(driver, "execute_stage", allocation)
    monkeypatch.setattr(driver, "verify_stage", verify_stage)
    for index in range(3):
        driver.run(root, parent, once=True, native_only=True, allow_checkpoint_cancelled=True, native_coordinator_pid=7777)
        assert len(allocated) == index + 1 and not freezes
        assert not (root / "workspace/calibration-frozen").exists()
    final = driver.run(root, parent, native_only=True, allow_checkpoint_cancelled=True, native_coordinator_pid=7777)
    assert allocated == analysis.native_names()[3:] and len(freezes) == 1
    receipt = analysis.read_json(root / "native-provenance.json")
    assert len(receipt["six_cases"]) == 6 and len(receipt["case_provenance"]) == 6
    assert receipt["original_native_terminal"] == "CANCELLED"
    assert [receipt["case_provenance"][name]["job"] for name in analysis.native_names()[:3]] == [278514] * 3
    assert [receipt["case_provenance"][name]["terminal"] for name in analysis.native_names()[:3]] == ["CANCELLED"] * 3
    assert len(set(receipt["case_provenance"][name]["job"] for name in analysis.native_names()[3:])) == 3
    assert final["phase"] == "six_native_cases_independently_verified_original_terminals_preserved"
    driver.run(root, parent, native_only=True, allow_checkpoint_cancelled=True, native_coordinator_pid=7777)
    assert len(freezes) == 1 and len(allocated) == 3 and analysis.file_hashes(raw) == before
    driver.run(root, parent, once=True, rate_only=True, allow_checkpoint_cancelled=True, native_coordinator_pid=7777)
    assert allocated == analysis.native_names()[3:] + ["rate-0"] and len(freezes) == 1
    assert analysis.file_hashes(raw) == before


def test_waits_for_archive_and_original_coordinator_exit_not_complete_flag(native_fixture, driver, analysis, monkeypatch):
    root, parent, raw, results, templates = checkpoint_native_fixture(native_fixture, analysis)
    state = {"native_job": 278514, "native_coordinator_identity": {"pid": 7777}}
    proof = {"pid": 7777, "stopped": True}
    monkeypatch.setattr(driver, "native_coordinator_stopped", lambda *args: None)
    assert driver.wait_native_archive(root, parent, state, once=True) is None
    monkeypatch.setattr(driver, "native_coordinator_stopped", lambda *args: proof)
    manifest = parent / "raw-278514.sha256"
    content = manifest.read_text()
    manifest.unlink()
    assert driver.wait_native_archive(root, parent, state, once=True) is None
    manifest.write_text(content)
    assert driver.wait_native_archive(root, parent, state, once=True) == proof
    assert "diagnosis_required" in analysis.read_json(parent / "campaign-state.json")["phase"]


@pytest.mark.parametrize("returncode,stdout,stopped", [(1, "", True), (0, "Z dead", True),
    (0, "S python performance_campaign.py --root ROOT", False), (0, "S unrelated-reused-pid", True), (2, "", None)])
def test_native_coordinator_stop_check_is_local_query_only(driver, tmp_path, monkeypatch, returncode, stdout, stopped):
    calls = []
    def process(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=returncode, stdout=stdout.replace("ROOT", str(tmp_path)), stderr="")
    monkeypatch.setattr(driver.subprocess, "run", process)
    if stopped is None:
        with pytest.raises(RuntimeError, match="Cannot independently verify"):
            driver.native_coordinator_stopped(tmp_path, {"pid": 7777})
    else:
        proof = driver.native_coordinator_stopped(tmp_path, {"pid": 7777})
        assert bool(proof) is stopped
    assert calls == [["ps", "-ww", "-p", "7777", "-o", "state=", "-o", "command="]]


def test_interrupted_native_rate_freeze_commit_resumes_without_regeneration(native_fixture, analysis, monkeypatch):
    root, parent, raw, results = native_fixture
    analysis.prepare_workspace(root, parent, 278514, coordinator_exit={"stopped": True})
    original_write = analysis.write_json
    original_freeze = analysis.search.freeze_calibration
    calls = []
    def freeze(workspace):
        calls.append(workspace)
        original_freeze(workspace)
    def interrupted(path, payload):
        if Path(path).name == "native-provenance.json":
            raise RuntimeError("simulated interrupted immutable receipt install")
        return original_write(path, payload)
    monkeypatch.setattr(analysis.search, "freeze_calibration", freeze)
    monkeypatch.setattr(analysis, "write_json", interrupted)
    with pytest.raises(RuntimeError, match="simulated interrupted"):
        analysis.finalize_native_workspace(root, parent, 278514, None)
    assert (root / "native-rate-freeze.pending/calibration-frozen").exists()
    monkeypatch.setattr(analysis, "write_json", original_write)
    analysis.finalize_native_workspace(root, parent, 278514, None)
    assert len(calls) == 1 and not (root / "native-rate-freeze.pending").exists()


@pytest.mark.parametrize("change", [None, "terminal", "accounting_exit", "trace", "policies", "controller_seed",
                                   "budget", "inputs", "runtime", "placement"])
def test_independent_native_shard_verifier_keeps_all_non_greedy_guards(native_fixture, driver, analysis, monkeypatch, change):
    root, parent, raw_parent, results, templates = checkpoint_native_fixture(native_fixture, analysis, complete=0)
    analysis.prepare_workspace(root, parent, 278514, coordinator_exit={"stopped": True})
    workspace = root / "workspace"
    name = analysis.native_names()[0]
    plan = driver.plan_for(root, name, name, "native", None, workspace / "frozen" / (name + ".json"))
    job = 281777
    entry = {"stage_id": name, "job": job, "terminal": "COMPLETED", "accounting_exit_code": "0:0", "plan": plan}
    archive = root / "stages" / name
    raw = archive / f"live-trace-{job}"
    shutil.copytree(templates / name, raw / name)
    for group in ("frozen/inputs", "public-context-calibration"):
        shutil.copytree(workspace / group, raw / group)
    for filename in ("public-context-fit.json", "placement-fit.json"):
        shutil.copy2(workspace / filename, raw / filename)
    copies = {group: driver.file_hashes(workspace / group) for group in ("frozen/inputs", "public-context-calibration")}
    put_json(raw / "copy-manifest.json", copies)
    put_json(archive / "copy-manifest.json", copies)
    put_json(raw / "runtime-manifest.json", {})
    put_json(archive / "runtime-manifest.json", {})
    put_json(root / "campaign-state.json", {"frozen_runtime_sha256": {}})
    (raw / "runtime").mkdir()
    put_json(raw / "stage.json", plan)
    (raw / "exit-status").write_text("0\n")
    monkeypatch.setattr(driver, "validate_inputs", lambda *args: True)
    monkeypatch.setattr(driver, "verify_profile", lambda *args: analysis.read_json(workspace / "public-context-fit.json"))
    monkeypatch.setattr(driver, "verify", lambda *args: results[name])
    if change in ("policies", "controller_seed", "budget"):
        config = analysis.read_json(raw / name / "config.json")
        field, value = {"policies": ("policies", ["no_move"]), "controller_seed": ("seed", plan["seed"]),
                        "budget": ("kv_budget_mib", 10239)}[change]
        config[field] = value
        put_json(raw / name / "config.json", config)
    elif change == "terminal":
        entry["terminal"] = "TIMEOUT"
    elif change == "accounting_exit":
        entry["accounting_exit_code"] = "0:15"
    elif change == "trace":
        put_json(raw / name / "trace.json", {"duration_s": 899, "seed": plan["seed"]})
    elif change == "inputs":
        put_json(raw / "frozen/inputs/code-marks.json", [{"prompt_tokens": 99999, "output_tokens": 64}])
    elif change == "runtime":
        put_json(raw / "runtime-manifest.json", {"foreign.py": "a" * 64})
    elif change == "placement":
        put_json(raw / name / "placement-fit.json", {"different": True})
    (archive / f"raw-{job}.sha256").write_text("".join(f"{sha}  {filename}\n" for filename, sha in driver.file_hashes(raw).items()))
    if change is None:
        assert driver.verify_stage(root, entry) == results[name]
    else:
        with pytest.raises(ValueError):
            driver.verify_stage(root, entry)


def test_native_coordinator_identity_requires_exact_performance_root(driver, tmp_path):
    with pytest.raises(ValueError, match="native-coordinator-pid"):
        driver.native_coordinator_identity(tmp_path)
    assert driver.native_coordinator_identity(tmp_path, 7777)["explicit_binding"]
    put_json(tmp_path / "local-monitor-identity.json", {"pid": 7777,
        "argv": ["python", "performance_campaign.py", "--root", str(tmp_path)]})
    assert driver.native_coordinator_identity(tmp_path)["pid"] == 7777
    with pytest.raises(ValueError, match="differs"):
        driver.native_coordinator_identity(tmp_path, 8888)
    put_json(tmp_path / "local-monitor-identity.json", {"pid": 7777,
        "argv": ["python", "performance_campaign.py", "--root", str(tmp_path / "other-root")]})
    with pytest.raises(ValueError, match="not bound"):
        driver.native_coordinator_identity(tmp_path)


@pytest.mark.parametrize("change", ["movement_costs", "original_native_terminal", "extra_case_file"])
def test_native_provenance_and_original_case_copy_are_immutable(native_fixture, analysis, change):
    root, parent, raw, results = native_fixture
    analysis.prepare_workspace(root, parent, 278514, coordinator_exit={"stopped": True})
    analysis.finalize_native_workspace(root, parent, 278514, None)
    if change == "extra_case_file":
        (root / "workspace" / analysis.native_names()[0] / "unarchived-extra.json").write_text("{}\n")
    else:
        receipt = analysis.read_json(root / "native-provenance.json")
        receipt[change] = {"move_fixed_s": 0, "move_bandwidth_gib_s": 99} if change == "movement_costs" else "COMPLETED_RELABELED"
        put_json(root / "native-provenance.json", receipt)
    with pytest.raises(ValueError):
        analysis.finalize_native_workspace(root, parent, 278514, None)


def test_full_split_campaign_uses_original_score_fixed_heldout_and_resume(driver, analysis, tmp_path, monkeypatch):
    root = tmp_path / "followup"
    parent = tmp_path / "native"
    workspace = root / "workspace"
    inputs = workspace / "frozen/inputs"
    conv = [{"prompt_key": "conv", "prompt_tokens": 128, "output_tokens": 8,
             "original_output_tokens": 8, "original_response_sha256": "a" * 64, "count": 1}]
    code = [{"prompt_key": "code", "prompt_tokens": 100001, "output_tokens": 64, "count": 1}]
    put_json(inputs / "conv-marks.json", conv)
    put_json(inputs / "code-marks.json", code)
    put_json(inputs / "prompt-bank.json", {"prompts": {"conv": {}, "code": {}}})
    put_json(inputs / "input-manifest.json", {"dataset_label": "openthoughts3_closed"})
    put_json(workspace / "public-context-fit.json", {"coefficients": {"cadence": 1.0}})
    put_json(workspace / "placement-fit.json", {"placement": "original"})
    put_json(root / "native-provenance.json", {"movement_costs": {"move_fixed_s": .475, "move_bandwidth_gib_s": 1}})
    put_json(root / "native-parent-review.json", {"terminal": "COMPLETED", "available_cases":
        {name: {} for name in analysis.native_names()}})
    put_json(parent / "campaign-state.json", {"job": 278514, "native_performance_complete": True,
        "frozen_runtime_sha256": {}, "protected_vllm_sha256": "unused-by-CPU-simulation"})
    calibration = {"native": {}, "candidate_case": "rate-1", "calibration": {
        "rate-0": {"eligible_for_separate_evaluation_rate_candidate": True,
                   "physical_pressure": {"local_physical_ratio": 1.1}, "rate_rps": [.001, .008]},
        "rate-1": {"eligible_for_separate_evaluation_rate_candidate": True,
                   "physical_pressure": {"local_physical_ratio": 1.2}, "rate_rps": [.001, .01]},
        "rate-2": {"eligible_for_separate_evaluation_rate_candidate": False,
                   "physical_pressure": {"local_physical_ratio": 1.4}, "rate_rps": [.001, .02]}}}
    for ordinal, rate in enumerate((.008, .01, .02)):
        put_json(workspace / f"calibration-frozen/rate-{ordinal}.json",
                 driver.poisson.make_poisson(inputs, 2026103103, rate))
    monkeypatch.setattr(driver, "prepare_workspace", lambda *args: workspace)
    monkeypatch.setattr(driver, "native_coordinator_identity", lambda *args: {"pid": 12345})
    monkeypatch.setattr(driver, "wait_native_archive", lambda *args: {"pid": 12345, "stopped": True})
    monkeypatch.setattr(driver, "finalize_native_workspace", lambda *args: None)
    monkeypatch.setattr(analysis, "validate_inputs", lambda *args: True)
    monkeypatch.setattr(analysis, "verify_profile", lambda *args: {"coefficients": {"cadence": 1.0}})

    def rate_analysis(root):
        driver.write_json(root / "rate-analysis.json", calibration)
        return calibration

    monkeypatch.setattr(driver, "analyze_calibration", rate_analysis)
    allocated = []

    def verify_shard(root, entry):
        directory = root / "stages" / entry["stage_id"]
        driver.verify_raw(directory / f"live-trace-{entry['job']}", (directory / f"raw-{entry['job']}.sha256").read_text())
        return {"status": "CPU_simulated_performance_verified", "quality_equivalence_not_proven": True}

    def allocation(root, state, plan):
        stage_id, policy = plan["stage_id"], plan["policy"]
        if stage_id in state["stages"]:
            entry = state["stages"][stage_id]
            assert entry["plan"] == plan
            return entry, verify_shard(root, entry)
        allocated.append(stage_id)
        job = 281000 + len(allocated)
        entry = {"stage_id": stage_id, "job": job, "plan": plan, "phase": "verified", "terminal": "COMPLETED"}
        state["stages"][stage_id] = entry
        archive = root / "stages" / stage_id
        phase = archive / f"live-trace-{job}" / plan["phase_name"]
        trace = driver.read_json(root / plan["trace_relative"])
        put_json(phase / "trace.json", trace)
        put_json(phase / "config.json", {"policies": [policy], "repeats": 1, "deadline_s": 120,
                                        "workaware_fit_sha256": plan["fit_sha256"]})
        put_json(phase / "summary.json", {"status": "passed"})
        (phase / "exit-status").write_text("0\n")
        factor = 1.0
        if policy == "kv_only_workaware":
            factor = (2.0 if plan["kind"] == "heldout" else .5 if plan["phase_name"] == "explore-1" else .8)
        origin = 1000 + job
        requests = []
        references = []
        for mark in trace["requests"]:
            arrived = origin + mark["arrival_s"]
            first = arrived + 10 * factor
            tokens = [2 if policy == "kv_only_workaware" else 1] * mark["output_tokens"]
            requests.append({"request_id": mark["request_id"], "model": mark["model"],
                "prompt_tokens": mark["prompt_tokens"], "arrived_at": arrived, "submitted_at": arrived,
                "reserved_at": arrived, "first_token_at": first,
                "completed_at": first + .02 * factor * (mark["output_tokens"] - 1), "token_ids": tokens})
            references.append({"trace_request_id": mark["request_id"], "token_ids": [1] * mark["output_tokens"]})
        put_json(phase / "references.json", references)
        case = phase / f"r0-{policy}"
        put_json(case / "requests.json", requests)
        put_json(case / "workload.json", {"origin": origin, "requests": trace["requests"],
                                          "duration_s": trace["duration_s"], "trace_sha256": driver.digest(phase / "trace.json")})
        put_json(case / "summary.json", analysis.analyze_tail_slo.request_metrics(requests, origin, trace["duration_s"], 120))
        put_json(case / "population.json", [])
        put_json(case / "decisions.json", [])
        raw = phase.parent
        (archive / f"raw-{job}.sha256").write_text("".join(f"{sha}  {name}\n" for name, sha in driver.file_hashes(raw).items()))
        driver.stress_campaign.save_state(root, state)
        return entry, verify_shard(root, entry)

    monkeypatch.setattr(driver, "execute_stage", allocation)
    monkeypatch.setattr(driver, "verify_stage", verify_shard)
    monkeypatch.setattr(driver.rebalance_campaign, "checked_ssh", lambda *args: pytest.fail("No real network in CPU simulation"))
    first = driver.run(root, parent, once=True)
    assert allocated == ["rate-0"] and len(first["stages"]) == 1
    final = driver.run(root, parent)
    assert len(allocated) == 3 + 4 * 7 + 7
    assert final["phase"] == "split_exploration_and_optional_heldout_SHA_verified"
    result = driver.read_json(root / "independent-evaluation-analysis.json")
    assert result["best_case"] == "explore-1"
    assert result["heldout"]["score"] is None
    assert result["all_heldout_negative_results_retained"] and result["P99_statistically_significant"] is False
    assert all(len(case["metrics"]) == 7 for case in result["cases"].values())
    frozen_before = driver.file_hashes(root / "evaluation/frozen")
    heldout_before = driver.digest(root / "evaluation/heldout-trace.json")
    driver.run(root, parent)
    assert len(allocated) == 38 and driver.file_hashes(root / "evaluation/frozen") == frozen_before
    assert driver.digest(root / "evaluation/heldout-trace.json") == heldout_before


def test_remote_query_watcher_uses_frozen_source_and_idempotent_intent(driver, tmp_path, monkeypatch):
    entry, state = submission_fixture(tmp_path, driver)
    entry.update({"job": 278600, "remote_home": "/home/tester"})
    state["frozen_runtime_sha256"] = {"experiments/dp_opt/watch-trace-remote.sh": "a" * 64}
    commands = []

    def remote(command):
        commands.append(command)
        return "job=278600\nhostname=login02\ntime=2026-11-02T00:00:00Z\npid=12345\nrun_id=login02:12345:start\nexit_status=running"

    monkeypatch.setattr(driver.rebalance_campaign, "checked_ssh", remote)
    driver.ensure_remote_watcher(tmp_path, state, entry)
    assert entry["remote_watcher_started"]["pid"] == 12345
    assert "DP_OPT_REMOTE_MAX_POLLS=1201" in commands[0]
    assert ".performance-followup-watcher-intent" in commands[0] and "remote-query.expected" in commands[0]
    assert "upload/runtime/experiments/dp_opt/watch-trace-remote.sh" in commands[0]
    assert "sbatch" not in commands[0] and "278529" in commands[0]


def test_compute_has_no_inherited_debug_hooks_or_changed_workload(driver, tmp_path, monkeypatch):
    upload = tmp_path / "upload"
    evidence = tmp_path / "evidence"
    runtime = evidence / "runtime"
    launch = runtime / "experiments/dp_opt/trace-benefit.sbatch"
    launch.parent.mkdir(parents=True)
    launch.write_text('sha256sum "$root/state/sparse-fit-231467.json"\n'
                     'python "$repo/experiments/dp_opt/live_trace.py" --fit "$root/state/sparse-fit-231467.json" '
                     '--output "$DP_OPT_PHASE_EVIDENCE_ROOT"\n')
    put_json(upload / "frozen/trace.json", {"duration_s": 10800, "seed": 2026103105, "requests": [1, 2, 3]})
    put_json(upload / "frozen/inputs/prompt-bank.json", {"prompts": {"original": [1, 2, 3]}})
    put_json(upload / "public-context-fit.json", {"cost": 1})
    put_json(upload / "placement-fit.json", {"placement": "original"})
    put_json(upload / "public-context-calibration/source.json", {"cost": 1})
    copies = {"frozen/inputs": driver.file_hashes(upload / "frozen/inputs"),
              "public-context-calibration": driver.file_hashes(upload / "public-context-calibration")}
    put_json(upload / "copy-manifest.json", copies)
    put_json(upload / "runtime-manifest.json", driver.file_hashes(runtime))
    plan = {"controls": driver.CONTROLS, "phase_name": "heldout", "policy": "kv_only_workaware",
            "policies": ["kv_only_workaware"], "controller_seed": 2026103105,
            "stage_id": "heldout-kv_only_workaware", "optimizer": "robust", "request_timeout_s": 7200,
            "seed": 2026103105, "rates_rps": [.001, .01], "trace_sha256": driver.digest(upload / "frozen/trace.json"),
            "fit_sha256": driver.digest(upload / "public-context-fit.json"),
            "placement_fit_sha256": driver.digest(upload / "placement-fit.json"),
            "movement_costs": {"move_fixed_s": .475, "move_bandwidth_gib_s": 1}}
    put_json(upload / "stage.json", plan)
    monkeypatch.setenv("DP_OPT_HEAD_DTYPE", "float32")
    monkeypatch.setenv("DP_OPT_HEAD_HIDDEN_DIAGNOSTIC", "1")
    monkeypatch.setenv("DP_OPT_TRACE_POLICIES", "no_move own_model")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(driver, "validate_inputs", lambda inputs: True)
    monkeypatch.setattr(driver, "verify_profile", lambda source: {"cost": 1})
    monkeypatch.setattr(driver, "verify", lambda *args: {"full_greedy_equivalence_passed": False})
    calls = []
    monkeypatch.setattr(driver.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    before = driver.digest(upload / "frozen/trace.json")
    driver.compute(upload, evidence)
    assert len(calls) == 1
    environment = calls[0][1]["env"]
    assert "VLLM_BATCH_INVARIANT" not in environment
    assert not any("HEAD" in name or "HIDDEN" in name for name in environment if name.startswith("DP_OPT_"))
    assert environment["DP_OPT_TRACE_POLICIES"] == "kv_only_workaware"
    assert environment["DP_OPT_TRACE_SEED"] == "2026103105" and environment["DP_OPT_REQUEST_TIMEOUT_S"] == "7200"
    assert environment["DP_OPT_TRACE_BUDGET_MIB"] == "10240" and environment["DP_OPT_PAUSE_SYNC_INTERVAL"] == "32"
    assert environment["KVCACHED_GPU_UTILIZATION"] == "0.98"
    assert environment["DP_OPT_MOVE_FIXED_S"] == "0.475" and environment["DP_OPT_MOVE_BANDWIDTH_GIB_S"] == "1"
    assert driver.digest(evidence / "placement-fit.json") == plan["placement_fit_sha256"]
    assert driver.digest(evidence / "frozen/trace.json") == before


@pytest.mark.parametrize("policy", ["no_move", "own_model", "kv_only", "no_move_fair", "random_no_return",
                                  "kv_only_headroom", "kv_only_workaware"])
def test_each_split_policy_is_accepted_by_actual_frozen_live_cli(driver, tmp_path, monkeypatch, policy):
    import sys

    trace = tmp_path / "trace.json"
    put_json(trace, {"duration_s": 3600, "seed": 2026103104, "model_rates_rps": [.001, .01]})
    put_json(tmp_path / "workspace/public-context-fit.json", {})
    put_json(tmp_path / "workspace/placement-fit.json", {})
    put_json(tmp_path / "native-provenance.json", {"movement_costs": {"move_fixed_s": .475, "move_bandwidth_gib_s": 1}})
    plan = driver.plan_for(tmp_path, "explore-0-" + policy, "explore-0", "explore", policy, trace)
    live = importlib.import_module("live_trace")
    received = []
    arguments = ["live_trace.py", "--checkpoint", "unused", "--fit", "unused", "--trace", "unused",
                 "--output", "unused", "--policies", policy, "--workaware-optimizer", plan["optimizer"]]
    if policy == "kv_only_workaware":
        arguments += ["--workaware-fit", "unused", "--workaware-fit-sha256", "a" * 64,
                      "--workaware-source-root", "unused", "--workaware-rate-priors", ".001", ".01"]
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(live, "run", lambda args: args)
    monkeypatch.setattr(live.asyncio, "run", received.append)
    live.main()
    assert received[0].workaware_optimizer == ("robust" if policy == "kv_only_workaware" else "point")
