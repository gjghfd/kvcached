import argparse
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import fp32_head_campaign
import rebalance_campaign
from amortized_campaign import check_frozen_sources
from archive_trace_job import verify_raw
from fp32_head_probe import analyze as analyze_parent
from head_hidden_probe import analyze
from rebalance_campaign import checked_ssh
from static_batch_campaign import digest
from stress_campaign import archive_job, check_pause, poll_job, save_state


SYNC_FILES = ("experiments/dp_opt/head_hidden_probe.py", "experiments/dp_opt/head_hidden_worker.py",
              "experiments/dp_opt/head_hidden_campaign.py", "experiments/dp_opt/head-hidden-probe.sbatch",
              "experiments/dp_opt/docs/head-hidden-diagnostic.md", "tests/test_dp_opt_head_hidden_probe.py")
EMPTY_QUEUE_GUARD = 'test -z "$(squeue -h -u "$USER")"'


def positive_job_id(value):
    try:
        job = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("Require a positive integer performance job ID") from error
    if job <= 0 or str(job) != str(value):
        raise argparse.ArgumentTypeError("Require a positive integer performance job ID")
    return job


def submission_guard(concurrent_performance_job=None):
    if concurrent_performance_job is None:
        return EMPTY_QUEUE_GUARD
    job = positive_job_id(concurrent_performance_job)
    return f'''dp_opt_guard_user="$(id -un)"
test "$dp_opt_guard_user" = "$USER"
dp_opt_guard_queue="$(squeue -h -u "$dp_opt_guard_user" -o '%i|%u|%j|%T')"
printf '%s\\n' "$dp_opt_guard_queue" | awk -F'|' -v owner="$dp_opt_guard_user" -v job={job} '
    /^[[:space:]]*$/ {{ next }}
    {{
        for (field = 1; field <= NF; field++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $field)
        count++
        if (NF != 4 || $1 != job || $2 != owner ||
            $3 !~ /^qwen-public-perf(-[[:alnum:]_.-]+)?$/ ||
            $4 !~ /^(PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|RESIZING|REQUEUED|REQUEUE_FED|REQUEUE_HOLD|SIGNALING|STAGE_OUT)$/) invalid = 1
    }}
    END {{ exit (invalid || count != 1) }}'
dp_opt_guard_accounting="$(sacct -n -X -P -j {job} --format=JobIDRaw,User,JobName%128,State%64)"
printf '%s\\n' "$dp_opt_guard_accounting" | awk -F'|' -v owner="$dp_opt_guard_user" -v job={job} '
    /^[[:space:]]*$/ {{ next }}
    {{
        for (field = 1; field <= NF; field++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $field)
        count++
        if (NF != 4 || $1 != job || $2 != owner ||
            $3 !~ /^qwen-public-perf(-[[:alnum:]_.-]+)?$/ ||
            $4 !~ /^(PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|RESIZING|REQUEUED|REQUEUE_FED|REQUEUE_HOLD|SIGNALING|STAGE_OUT)$/) invalid = 1
    }}
    END {{ exit (invalid || count != 1) }}'
test "$(printf '%s\\n' "$dp_opt_guard_queue" | cut -d'|' -f3 | tr -d '[:space:]')" = \
     "$(printf '%s\\n' "$dp_opt_guard_accounting" | cut -d'|' -f3 | tr -d '[:space:]')"'''


def verify_concurrent_job(root, concurrent_performance_job, phase):
    if concurrent_performance_job is None:
        return
    observation = checked_ssh("set -eu\n" + submission_guard(concurrent_performance_job) + '''
date -Is
printf 'USER=%s\\nSQUEUE\\n%s\\nSACCT\\n%s\\n' "$dp_opt_guard_user" "$dp_opt_guard_queue" "$dp_opt_guard_accounting"''')
    (root / f"concurrent-performance-{phase}.txt").write_text(observation + "\n")


@contextmanager
def campaign_lock(root):
    with (root / "campaign-run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Existing debug coordinator owns this campaign; do not duplicate") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextmanager
def sync_file_scope(concurrent_performance_job=None):
    original = fp32_head_campaign.SYNC_FILES
    original_parent_ssh = fp32_head_campaign.checked_ssh
    original_sync_ssh = rebalance_campaign.checked_ssh
    original_sync_script = rebalance_campaign.checked_ssh_script
    guard = submission_guard(concurrent_performance_job)

    def guarded_parent_ssh(command, *args, **kwargs):
        return original_parent_ssh(command.replace(EMPTY_QUEUE_GUARD, guard), *args, **kwargs)

    def guarded_sync_ssh(command, *args, **kwargs):
        return original_sync_ssh(command.replace(EMPTY_QUEUE_GUARD, guard), *args, **kwargs)

    def guarded_sync_script(script, *args, **kwargs):
        return original_sync_script(script.replace(EMPTY_QUEUE_GUARD, guard), *args, **kwargs)

    fp32_head_campaign.SYNC_FILES = SYNC_FILES
    if concurrent_performance_job is not None:
        fp32_head_campaign.checked_ssh = guarded_parent_ssh
        rebalance_campaign.checked_ssh = guarded_sync_ssh
        rebalance_campaign.checked_ssh_script = guarded_sync_script
    try:
        yield
    finally:
        fp32_head_campaign.SYNC_FILES = original
        fp32_head_campaign.checked_ssh = original_parent_ssh
        rebalance_campaign.checked_ssh = original_sync_ssh
        rebalance_campaign.checked_ssh_script = original_sync_script


def prepare(root, parent, repo, concurrent_performance_job=None):
    if concurrent_performance_job is not None:
        concurrent_performance_job = positive_job_id(concurrent_performance_job)
    previous = json.loads((parent / "campaign-state.json").read_text())
    if previous.get("job") != 276677 or previous.get("static_strict_passed") is not False:
        raise ValueError("Require the archived FP32-head full-token failure")
    raw = parent / "live-trace-276677"
    verify_raw(raw, (parent / "raw-276677.sha256").read_text())
    if analyze_parent(raw / "static", "static") != json.loads((parent / "independent-static-verification.json").read_text()):
        raise ValueError("Parent precision diagnosis differs")
    check_frozen_sources(repo, previous)
    shutil.copytree(raw / "frozen", root / "frozen")
    return {"repo": str(repo), "parent": str(parent), "parent_job": 276677,
            "phase": "head_hidden_prepared", "original_trace_sha256": digest(root / "frozen/trace.json"),
            "frozen_runtime_sha256": {**previous["frozen_runtime_sha256"],
                                      **{name: digest(repo / name) for name in SYNC_FILES}},
            "protected_vllm_sha256": previous["protected_vllm_sha256"],
            "concurrent_performance_job": concurrent_performance_job,
            "monitor_interval_s": 300, "is_algorithm_benefit": False, "research_goal_complete": False,
            "publication_authorized": False, "observational_CPU_sync_not_benefit": True}


def preflight(root, repo, state):
    fp32_head_campaign.preflight(root, repo, state)
    environment = {**os.environ, "PYTHONPATH": str(repo / "experiments/dp_opt")}
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_dp_opt_head_hidden_probe.py"],
                            cwd=repo, env=environment, text=True, capture_output=True)
    (root / "head-hidden-cpu-tests.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("Head/hidden focused tests failed")
    with tempfile.TemporaryDirectory(prefix="head-hidden-actual-isolation-") as directory:
        isolated = Path(directory)
        for name in state["frozen_runtime_sha256"]:
            target = isolated / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / name, target)
        command = [sys.executable, "-I", "-c", "import sys;sys.path.insert(0," +
                   repr(str(isolated / "experiments/dp_opt")) +
                   ");import head_hidden_probe;import head_hidden_campaign;print('head/hidden actual deployment imports passed')"]
        result = subprocess.run(command, cwd=isolated, text=True, capture_output=True)
        (root / "head-hidden-isolated-runtime-imports.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError("Head/hidden actual import closure failed")
    check_frozen_sources(repo, state)
    state["head_hidden_focused_tests_and_actual_imports_passed"] = True
    save_state(root, state)


def run(root, parent, concurrent_performance_job=None, prepare_only=False, preflight_only=False):
    if concurrent_performance_job is not None:
        concurrent_performance_job = positive_job_id(concurrent_performance_job)
    root.mkdir(parents=True, exist_ok=True)
    with campaign_lock(root):
        run_locked(root, parent, concurrent_performance_job, prepare_only, preflight_only)


def run_locked(root, parent, concurrent_performance_job, prepare_only, preflight_only):
    repo = Path(__file__).resolve().parents[2]
    root.mkdir(parents=True, exist_ok=True)
    path = root / "campaign-state.json"
    state = json.loads(path.read_text()) if path.exists() else prepare(root, parent, repo, concurrent_performance_job)
    if state["repo"] != str(repo) or state["parent"] != str(parent):
        raise ValueError("Preserve diagnostic campaign identity")
    if state.get("concurrent_performance_job") != concurrent_performance_job:
        raise ValueError("Preserve concurrent performance job identity on recovery; pass the same explicit job ID")
    if state.get("GPU_diagnostic_complete"):
        return
    if "diagnosis_required" in state["phase"]:
        raise RuntimeError("Diagnose the existing allocation; do not duplicate")
    save_state(root, state)
    if prepare_only or preflight_only:
        check_pause(root)
        check_frozen_sources(repo, state)
        if state["phase"] != "head_hidden_prepared" or "job" in state:
            raise RuntimeError("Prepare/preflight only applies to an unsubmitted debug campaign")
        if preflight_only:
            preflight(root, repo, state)
        return
    try:
        check_pause(root)
        check_frozen_sources(repo, state)
        if "job" not in state:
            if state["phase"] != "head_hidden_prepared":
                raise RuntimeError("Reconcile submission intent before recovery")
            preflight(root, repo, state)
            verify_concurrent_job(root, concurrent_performance_job, "before-sync")
            with sync_file_scope(concurrent_performance_job):
                stage, home = fp32_head_campaign.stage_inputs(repo, root, state)
            ledger = stage + "/submitted-job.txt"
            state.update({"phase": "submission_intent_do_not_duplicate", "remote_sync_stage": stage,
                          "submission_ledger": ledger})
            save_state(root, state)
            verify_concurrent_job(root, concurrent_performance_job, "before-submit")
            submitted = checked_ssh(
                'set -eu\n' + submission_guard(concurrent_performance_job) + '\n' +
                f'test ! -e {shlex.quote(ledger)}; '
                f'export DP_OPT_HEAD_HIDDEN_UPLOAD={shlex.quote(stage + "/upload")}; '
                'sbatch --parsable --export=ALL '
                '--output="$HOME/run/kvcached-paracloud/logs/live-trace-%j.out" '
                '--error="$HOME/run/kvcached-paracloud/logs/live-trace-%j.err" '
                f'"$HOME/kvcached/kvcached/experiments/dp_opt/head-hidden-probe.sbatch" > {shlex.quote(ledger)}; '
                f'cat {shlex.quote(ledger)}')
            state.update({"job": int(submitted.split(";")[0]), "phase": "monitoring_common_prefix_projection_control"})
            (root / "submit.txt").write_text(submitted + "\n")
            save_state(root, state)
            remote = home + f"/run/kvcached-paracloud/state/live-trace-{state['job']}"
            watcher = checked_ssh(f'''set -eu
mkdir -p {shlex.quote(remote)}
nohup bash "$HOME/kvcached/kvcached/experiments/dp_opt/watch-trace-remote.sh" {state['job']} \\
    >> {shlex.quote(remote + '/remote-query.log')} 2>&1 < /dev/null &
printf '%s\\n' "$!"''')
            state["remote_watcher_pid"] = int(watcher)
            save_state(root, state)
        terminal = poll_job(root, state["job"], state)
        archive_job(root, state["job"], repo / "experiments/dp_opt", state)
        check_frozen_sources(repo, state)
        raw = root / f"live-trace-{state['job']}"
        if terminal != "COMPLETED" or (raw / "exit-status").read_text().strip() != "0":
            raise RuntimeError("Head/hidden control failed; preserve full raw evidence and diagnose")
        result = analyze(raw)
        for name in ("head-hidden-diagnostic.json", "independent-head-hidden-verification.json"):
            if result != json.loads((raw / name).read_text()):
                raise RuntimeError("Independent projection reconstruction differs")
        for name in ("head_hidden_probe.py", "head_hidden_worker.py", "fp32_head_probe.py", "fp32_head_worker.py"):
            if digest(raw / name) != digest(repo / "experiments/dp_opt" / name):
                raise RuntimeError("Executed observer/precision source differs")
        if digest(raw / "trace.json") != state["original_trace_sha256"]:
            raise RuntimeError("Original complete trace differs")
        (root / "independent-head-hidden-verification.json").write_text(json.dumps(result, indent=2) + "\n")
        state.update({"phase": "common_prefix_projection_SHA_verified_not_benefit", "GPU_diagnostic_complete": True,
                      "strict_correctness_passed": result["full_static_analysis"]["strict_correctness_passed"],
                      "all_observed_first_divergences_attributed": result["all_observed_first_divergences_attributed"],
                      "next_action": ("use_first_common_prefix_decomposition_to_choose_head_or_upstream_layer_diagnostic"
                                      if result["all_observed_first_divergences_attributed"]
                                      else "diagnose_first_divergence_outside_captured_positions_no_automatic_submission")})
        save_state(root, state)
    except Exception as error:
        state.update({"phase": "diagnosis_required_no_automatic_duplicate_submission", "error": repr(error)})
        save_state(root, state)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--concurrent-performance-job", type=positive_job_id)
    preparation = parser.add_mutually_exclusive_group()
    preparation.add_argument("--prepare-only", action="store_true")
    preparation.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(args.root.resolve(), args.parent.resolve(), concurrent_performance_job=args.concurrent_performance_job,
        prepare_only=args.prepare_only, preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()
