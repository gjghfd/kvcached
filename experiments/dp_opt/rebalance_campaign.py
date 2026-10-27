import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

from stress_campaign import archive_and_verify, archive_job, poll_job, save_state, ssh


SYNC_FILES = [
    "experiments/dp_opt/ilp_reference.py", "experiments/dp_opt/live_controller.py",
    "experiments/dp_opt/live_trace.py", "experiments/dp_opt/live_rebalance.py",
    "experiments/dp_opt/verify_trace.py", "experiments/dp_opt/verify_rebalance.py",
    "experiments/dp_opt/trace-benefit.sbatch", "experiments/dp_opt/rebalance-trace.sbatch",
    "tests/test_dp_opt_rebalance.py", "tests/test_dp_opt_trace.py", "tests/test_dp_opt_trace_planning.py",
    "experiments/dp_opt/README.md", "experiments/dp_opt/docs/ilp-scheduler-design.md",
    "experiments/dp_opt/docs/release-rebalance-protocol.md",
    "experiments/dp_opt/reports/release-rebalance-status.md",
    "experiments/dp_opt/reports/remaining-optimization-space.md",
]
PROTECTED_AEG_DIFF = "896eb42a0552e378f72adc8dbd721a06c87beb48a6f4f2aaafe60b45bd258ca8"


def checked_ssh(command, timeout=240):
    result = ssh(command, timeout)
    if result.returncode:
        raise RuntimeError(f"SSH failed ({result.returncode}): {result.stderr}")
    return result.stdout.strip()


def process_command(pid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "state=", "-o", "command="], text=True, capture_output=True)
    if result.returncode or not result.stdout.strip() or result.stdout.lstrip().startswith("Z"):
        return None
    return result.stdout.strip()


def handoff_old_campaign(pid, archive):
    command = process_command(pid)
    if command is None:
        return {"old_pid": pid, "old_process_already_exited": True}
    if "stress_campaign.py " not in command or str(archive.resolve()) not in command:
        raise RuntimeError("Refuse to stop an unrelated/reused old campaign PID")
    if os.getpgid(pid) != pid or pid == os.getpgrp():
        raise RuntimeError("Old campaign is not its own isolated process group")
    os.killpg(pid, signal.SIGTERM)
    for attempt in range(30):
        if process_command(pid) is None:
            return {"old_pid": pid, "old_command": command, "old_process_stopped": True,
                    "current_GPU_job_not_cancelled": True}
        time.sleep(1)
    raise RuntimeError("Old local submission campaign did not exit; do not synchronize or submit")


def frozen_environment(config, trace_path):
    if (config["max_model_len"] != 131072 or config["cache_bytes"] != 25769803776
            or config["deadline_s"] != 120 or config["max_moves"] != 8 or config["candidate_limit"] != 8):
        raise ValueError("Source case differs from the declared A800 paired rebalancing protocol")
    return {"DP_OPT_REBALANCE_TRACE_PATH": trace_path,
            "DP_OPT_REBALANCE_TRACE_SHA256": config["frozen_trace_sha256"],
            "DP_OPT_REBALANCE_BUDGET_MIB": str(config["kv_budget_mib"]),
            "DP_OPT_MOVE_FIXED_S": str(config["move_fixed_s"]),
            "DP_OPT_MOVE_BANDWIDTH_GIB_S": str(config["move_bandwidth_gib_s"]),
            "DP_OPT_TRACE_SEED": str(config["seed"])}


def submission_command(environment, ledger, wall_hours=20):
    exports = "; ".join(f"export {name}={shlex.quote(value)}" for name, value in environment.items())
    return (f'set -eu; test -z "$(squeue -h -u "$USER")"; test ! -e {shlex.quote(ledger)}; '
            'test "$(git -C "$HOME/kvcached/kvcached" branch --show-current)" = main; '
            'test "$(git -C "$HOME/kvcached/kvcached" remote get-url origin)" = git@github.com:gjghfd/kvcached.git; '
            + exports + '; '
            f'sbatch --parsable --time={wall_hours}:00:00 --export=ALL '
            '--output="$HOME/run/kvcached-paracloud/logs/live-trace-%j.out" '
            '--error="$HOME/run/kvcached-paracloud/logs/live-trace-%j.err" '
            f'"$HOME/kvcached/kvcached/experiments/dp_opt/rebalance-trace.sbatch" > {shlex.quote(ledger)}; '
            f'cat {shlex.quote(ledger)}')


def verify_old_failure(archive, job, terminal, diagnosis_path):
    from archive_trace_job import verify_raw

    if diagnosis_path is None:
        raise RuntimeError("Old allocation failed; require explicit raw-verified diagnosis before new deployment")
    diagnosis = json.loads(diagnosis_path.read_text())
    manifest = archive / f"raw-{job}.sha256"
    if (diagnosis.get("job") != job or diagnosis.get("state") != terminal
            or diagnosis.get("ready_for_new_method") is not True
            or diagnosis.get("raw_manifest_sha256") != hashlib.sha256(manifest.read_bytes()).hexdigest()):
        raise RuntimeError("Old failure diagnosis does not match the immutable raw archive")
    verify_raw(archive / f"live-trace-{job}", manifest.read_text())
    return diagnosis


def protected_sync(repo, root, state, files=None, prefix="rebalance"):
    files = SYNC_FILES if files is None else files
    for filename, expected in state.get("frozen_runtime_sha256", {}).items():
        if hashlib.sha256((repo / filename).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Local tested runtime changed after coordinator start; diagnose before deployment")
    home = checked_ssh('printf "%s" "$HOME"')
    stage = f"{home}/run/kvcached-paracloud/sync/{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    guard = ('git -C "$HOME/aegaeon/aegaeon" diff -- aegaeon/llm.py '
             'aegaeon/loader/quick_loader.py aegaeon/simple_dispatcher.py | sha256sum')
    safety = ('test -z "$(squeue -h -u "$USER")"; '
              'test "$(git -C "$HOME/kvcached/kvcached" branch --show-current)" = main; '
              'test "$(git -C "$HOME/kvcached/kvcached" remote get-url origin)" = git@github.com:gjghfd/kvcached.git; '
              f'test "$( {guard} | cut -d\' \' -f1)" = {PROTECTED_AEG_DIFF}; '
              'test "$(sha256sum "$HOME/kvcached/kvcached/kvcached/integration/vllm/patches.py" | cut -d\' \' -f1)" = '
              + state["protected_vllm_sha256"])
    checked_ssh("set -eu; " + safety)
    targets = {filename: f"{home}/kvcached/kvcached/{filename}" for filename in files}
    old_text = checked_ssh("; ".join(f'if test -f {shlex.quote(target)}; then sha256sum {shlex.quote(target)}; '
                                    f'else printf "ABSENT %s\\n" {shlex.quote(target)}; fi' for target in targets.values()))
    old = {line.split(maxsplit=1)[1]: line.split()[0] for line in old_text.splitlines()}
    manifest = [{"local": filename, "target": target, "old_sha256": old[target],
                 "sha256": hashlib.sha256((repo / filename).read_bytes()).hexdigest()} for filename, target in targets.items()]
    (root / "protected-sync-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "protected-sync-old.sha256").write_text(old_text + "\n")
    bundle = root / "protected-sync.tar.gz"
    with tarfile.open(bundle, "w:gz") as contents:
        for filename in files:
            contents.add(repo / filename, arcname=filename)
        contents.add(root / "frozen/trace.json", arcname="frozen/trace.json")
    checked_ssh(f"mkdir -p {shlex.quote(stage + '/upload')}")
    subprocess.run(["scp", "-o", "ConnectTimeout=30", str(bundle), f"paracloud:{stage}/bundle.tar.gz"], check=True, timeout=240)
    checked_ssh(f"tar -xzf {shlex.quote(stage + '/bundle.tar.gz')} -C {shlex.quote(stage + '/upload')}")
    script = ["set -eu", safety]
    for item in manifest:
        target, uploaded = shlex.quote(item["target"]), shlex.quote(stage + "/upload/" + item["local"])
        script.append(f"test ! -e {target}" if item["old_sha256"] == "ABSENT" else
                      f'test "$(sha256sum {target} | cut -d\' \' -f1)" = {item["old_sha256"]}')
        script.append(f'test "$(sha256sum {uploaded} | cut -d\' \' -f1)" = {item["sha256"]}')
    for item in manifest:
        target = shlex.quote(item["target"])
        backup = stage + "/old/" + item["local"]
        temporary = shlex.quote(item["target"] + ".rebalance-sync")
        script.append("mkdir -p " + shlex.quote(str(Path(item["target"]).parent)) + " " + shlex.quote(str(Path(backup).parent)))
        if item["old_sha256"] != "ABSENT":
            script.append(f"cp -p {target} {shlex.quote(backup)}")
        script.extend([f"cp {shlex.quote(stage + '/upload/' + item['local'])} {temporary}", f"mv {temporary} {target}"])
    script.extend([safety, "sha256sum " + " ".join(shlex.quote(target) for target in targets.values())])
    installed = checked_ssh("\n".join(script))
    if installed.splitlines() != [f'{item["sha256"]}  {item["target"]}' for item in manifest]:
        raise RuntimeError("Installed protected source SHA mismatch")
    (root / "protected-sync-result.txt").write_text(installed + "\nBACKUP " + stage + "/old\n")
    return stage, home


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--old-archive", type=Path, required=True)
    parser.add_argument("--old-job", type=int, required=True)
    parser.add_argument("--old-campaign-pid", type=int, required=True)
    parser.add_argument("--source-case", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--old-diagnosis", type=Path)
    args = parser.parse_args()
    root, scripts = args.archive.resolve(), Path(__file__).resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    if (root / "campaign-state.json").exists():
        parser.error("Preserve existing coordinator ledger; reconcile its recorded jobs before resuming")
    from verify_trace import verify

    source = verify(args.source_case, args.inputs / "prompt-bank.json")
    config = json.loads((args.source_case / "config.json").read_text())
    config["frozen_trace_sha256"] = source["trace_sha256"]
    frozen_environment(config, "validated-before-any-process-handoff")
    frozen = root / "frozen"
    frozen.mkdir()
    (frozen / "trace.json").write_bytes((args.source_case / "trace.json").read_bytes())
    (frozen / "source-config.json").write_text(json.dumps(config, indent=2) + "\n")
    state = {"phase": "preparing_safe_old_job_handoff", "old_job": args.old_job, "jobs": [],
             "source_case": str(args.source_case.resolve()), "trace_sha256": source["trace_sha256"],
             "new_method_GPU_experiment_complete": False, "total_research_goal_completed": False,
             "publication_authorized": False}
    state["frozen_runtime_sha256"] = {filename: hashlib.sha256((scripts.parents[1] / filename).read_bytes()).hexdigest()
                                      for filename in SYNC_FILES[:11]}
    save_state(root, state)
    try:
        initial = checked_ssh(f'date -Is; sacct -n -X -j {args.old_job} --format=JobID,State,ExitCode -P; '
                              'squeue -h -u "$USER" -o "%i|%T|%j"')
        (root / "handoff-initial-query.txt").write_text(initial + "\n")
        state["handoff"] = handoff_old_campaign(args.old_campaign_pid, args.old_archive)
        state["protected_vllm_sha256"] = checked_ssh('sha256sum "$HOME/kvcached/kvcached/kvcached/integration/vllm/patches.py"').split()[0]
        state["phase"] = "monitoring_old_frozen_job_before_new_source_deployment"
        save_state(root, state)
        terminal = poll_job(root, args.old_job, state)
        if terminal == "COMPLETED":
            old_verified = archive_and_verify(root, args.old_job, args.inputs, scripts, state)
            state["old_verification_status"] = old_verified["status"]
        else:
            diagnosis = verify_old_failure(args.old_archive, args.old_job, terminal, args.old_diagnosis)
            state["old_diagnosis"] = diagnosis
            state["old_verification_status"] = "failed_raw_archived_completed_cases_verified_new_goal_separate"
        state["old_terminal"] = terminal
        state["phase"] = "old_immutable_archive_verified_ready_for_new_method_sync"
        save_state(root, state)
        stage, home = protected_sync(scripts.parents[1], root, state)
        remote_trace = stage + "/upload/frozen/trace.json"
        actual_trace_sha = checked_ssh(f"sha256sum {shlex.quote(remote_trace)}").split()[0]
        if actual_trace_sha != state["trace_sha256"]:
            raise RuntimeError("Frozen paired trace upload SHA differs")
        environment = frozen_environment(config, remote_trace)
        ledger = stage + "/submitted-job.txt"
        state.update({"phase": "submitting_new_method_once", "submission_ledger": ledger,
                      "frozen_environment": environment, "source_backup": stage + "/old"})
        save_state(root, state)
        result = ssh(submission_command(environment, ledger))
        (root / "new-method-submit.txt").write_text(result.stdout + result.stderr)
        if result.returncode:
            recorded = checked_ssh(f"if test -f {shlex.quote(ledger)}; then cat {shlex.quote(ledger)}; fi")
            if not recorded:
                raise RuntimeError("Submission failed or uncertain; inspect queue and ledger, do not resubmit")
            job = int(recorded.split(";")[0])
        else:
            job = int(result.stdout.strip().split(";")[0])
        state["jobs"].append({"job": job, "policy": "original_five_plus_kv_only_rebalance"})
        state.update({"phase": "monitoring_new_method_frozen_trace", "job": job})
        save_state(root, state)
        remote_watcher = checked_ssh(f'''mkdir -p "{home}/run/kvcached-paracloud/state/live-trace-{job}"
nohup bash "{home}/kvcached/kvcached/experiments/dp_opt/watch-trace-remote.sh" {job} \
    >> "{home}/run/kvcached-paracloud/state/live-trace-{job}/remote-query.log" 2>&1 < /dev/null &
printf '%s\\n' "$!"''')
        state["remote_watcher_pid"] = int(remote_watcher)
        save_state(root, state)
        state["new_terminal"] = poll_job(root, job, state)
        archive_job(root, job, scripts, state)
        result = verify(root / f"live-trace-{job}", args.inputs / "prompt-bank.json")
        (root / f"verification-{job}.json").write_text(json.dumps(result, indent=2) + "\n")
        from analyze_tail_slo import analyze

        comparisons = {baseline: analyze(root / f"live-trace-{job}", baseline, "kv_only_rebalance")
                       for baseline in ("no_move", "no_move_fair", "kv_only")}
        (root / "new-method-tail-ttft-results.json").write_text(json.dumps(comparisons, indent=2) + "\n")
        state.update({"phase": "new_method_GPU_trace_archived_and_independently_verified",
                      "new_method_GPU_experiment_complete": True,
                      "next_action": "report_actual_Code_Conv_and_combined_costs_not_assume_positive_benefit"})
        save_state(root, state)
    except BaseException as error:
        state.update({"phase": "needs_diagnosis_no_automatic_duplicate_submission", "error": str(error)})
        save_state(root, state)
        raise


if __name__ == "__main__":
    main()
