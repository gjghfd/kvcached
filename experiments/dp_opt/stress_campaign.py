import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}


def training_plan(stage, best=None):
    common = {"code_rate": .001, "window_s": 1800, "warmup_s": 600, "seed": 20261024}
    if stage == 0:
        return {**common, "outputs": [8192], "budgets": [10240, 11328], "multipliers": [1], "wall_hours": 16}
    if best is None:
        raise ValueError("Refinement requires a verified prior best, not a predicted result")
    setting = best["setting"]
    if stage == 1:
        return {**common, "outputs": [setting["output_tokens"]], "budgets": [setting["budget_mib_per_gpu"]],
                "multipliers": [.85, 1, 1.15], "fixed_conv_rate": setting["conv_rate_rps"], "wall_hours": 20}
    if stage in (2, 3):
        return {**common, "outputs": [12288 if stage == 2 else 16384], "budgets": [10240, 11328],
                "multipliers": [.85, 1, 1.15], "wall_hours": 40}
    raise ValueError("Unknown bounded exploration stage")


def confirmation_plan(best, seed):
    setting = best["setting"]
    return {"outputs": [setting["output_tokens"]], "budgets": [setting["budget_mib_per_gpu"]],
            "multipliers": [1], "fixed_conv_rate": setting["conv_rate_rps"], "code_rate": setting["code_rate_rps"],
            "window_s": 16000, "warmup_s": 600, "seed": seed, "wall_hours": 32}


def submit_command(plan):
    variables = {"DP_OPT_STRESS_OUTPUTS": " ".join(map(str, plan["outputs"])),
                 "DP_OPT_STRESS_BUDGETS": " ".join(map(str, plan["budgets"])),
                 "DP_OPT_STRESS_RATE_MULTIPLIERS": " ".join(map(str, plan["multipliers"])),
                 "DP_OPT_STRESS_CODE_RATE": str(plan["code_rate"]), "DP_OPT_STRESS_WINDOW_S": str(plan["window_s"]),
                 "DP_OPT_STRESS_WARMUP_S": str(plan["warmup_s"]), "DP_OPT_STRESS_SEED": str(plan["seed"])}
    if "fixed_conv_rate" in plan:
        variables["DP_OPT_STRESS_FIXED_CONV_RATE"] = str(plan["fixed_conv_rate"])
    exports = "; ".join(f"export {name}={shlex.quote(value)}" for name, value in variables.items())
    return ('set -eu; test -z "$(squeue -h -u "$USER")"; '
            'test "$(git -C "$HOME/kvcached/kvcached" branch --show-current)" = main; '
            'test "$(git -C "$HOME/kvcached/kvcached" remote get-url origin)" = git@github.com:gjghfd/kvcached.git; '
            'unset DP_OPT_STRESS_FIXED_CONV_RATE; ' + exports + '; '
            f'sbatch --parsable --time={plan["wall_hours"]}:00:00 --export=ALL '
            '--output="$HOME/run/kvcached-paracloud/logs/live-trace-%j.out" '
            '--error="$HOME/run/kvcached-paracloud/logs/live-trace-%j.err" '
            '"$HOME/kvcached/kvcached/experiments/dp_opt/stress-sweep.sbatch"')


def ssh(command, timeout=120):
    return subprocess.run(["ssh", "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=20",
                           "-o", "ServerAliveCountMax=3", "paracloud", command],
                          text=True, capture_output=True, timeout=timeout)


def save_state(root, state):
    state["recorded_at"] = datetime.now().astimezone().isoformat()
    temporary = root / "campaign-state.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(root / "campaign-state.json")


def check_pause(root):
    if (root / "PAUSE").exists():
        raise RuntimeError("User pause marker present; do not submit or resume another allocation")


def poll_job(root, job, state):
    command = f'''date -Is
squeue -h -j {job} -o '%i|%T|%M|%R'
sacct -n -X -j {job} --format=JobID,State,ExitCode,Start,End,Elapsed,NodeList -P
root="$HOME/run/kvcached-paracloud/state/live-trace-{job}"
for filename in phase.json progress.json error.json exit-status capacity/summary.json; do
    if [ -f "$root/$filename" ]; then printf '\\n%s\\n' "$filename"; cat "$root/$filename"; fi
done
for filename in "$root"/cases/case-*/progress.json "$root"/cases/case-*/error.json; do
    if [ -f "$filename" ]; then printf '\\n%s\\n' "$filename"; cat "$filename"; fi
done
tail -n 12 "$HOME/run/kvcached-paracloud/logs/live-trace-{job}.err" 2>/dev/null
true'''
    previous = None
    previous_wall = None
    while True:
        check_pause(root)
        started = time.monotonic()
        polled_at = datetime.now().astimezone()
        stamp = polled_at.isoformat()
        monotonic_gap = started - previous if previous is not None else None
        gap = (polled_at - previous_wall).total_seconds() if previous_wall is not None else None
        previous = started
        previous_wall = polled_at
        try:
            result = ssh(command)
            stdout, stderr, code = result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            stdout, stderr, code = "", "SSH poll timeout; retry at next scheduled 300s interval", 124
        text = f"\n=== {stamp} actual_gap_s={gap} ssh_exit={code} ===\n{stdout}{stderr}"
        with (root / f"poll-{job}.log").open("a") as output:
            output.write(text)
        print(text, flush=True)
        observed = next((line.split("|")[1].split()[0] for line in stdout.splitlines()
                         if line.startswith(str(job) + "|") and len(line.split("|")) >= 7), None)
        state.update({"job": job, "state_last_observed": observed or "query_failed_or_accounting_unknown",
                      "last_poll_at": stamp, "last_poll_ssh_exit": code, "last_poll_actual_gap_s": gap,
                      "last_poll_wall_clock_gap_s": gap, "last_poll_monotonic_gap_s": monotonic_gap})
        save_state(root, state)
        if observed in TERMINAL:
            return observed
        time.sleep(max(0, 300 - (time.monotonic() - started)))


def archive_job(root, job, scripts, state):
    state["phase"] = "terminal_SHA_archive_and_independent_verification"
    save_state(root, state)
    for attempt in range(1, 13):
        check_pause(root)
        command = [sys.executable, str(scripts / "archive_trace_job.py"), str(job), "--archive", str(root)]
        if (root / f"live-trace-{job}").exists():
            command.append("--resume")
        result = subprocess.run(command, text=True, capture_output=True)
        (root / f"archive-{job}-attempt-{attempt}.log").write_text(result.stdout + result.stderr)
        if result.returncode == 0:
            break
        time.sleep(300)
    else:
        raise RuntimeError("Archive failed after bounded retries; preserve downloads and stop resubmission")


def archive_and_verify(root, job, inputs, scripts, state):
    archive_job(root, job, scripts, state)
    output = root / f"verification-{job}.json"
    result = subprocess.run([sys.executable, str(scripts / "verify_stress_sweep.py"), str(root / f"live-trace-{job}"),
                             "--inputs", str(inputs), "--output", str(output)], text=True, capture_output=True)
    (root / f"verification-{job}.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("Stress raw verification failed; diagnose before changing or resubmitting")
    return json.loads(output.read_text())


def start_job(root, plan, state):
    check_pause(root)
    result = ssh(submit_command(plan))
    (root / f"submit-stage-{len(state['jobs'])}.txt").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("Slurm submission rejected; no duplicate allocation request")
    job = int(result.stdout.strip().split(";")[0])
    entry = {"job": job, "plan": plan, "remote_watcher_pid": None,
             "submitted_at_local": datetime.now().astimezone().isoformat()}
    state["jobs"].append(entry)
    state.update({"phase": "submitted_waiting_for_monitor", "job": job, "plan": plan})
    save_state(root, state)
    remote = f'''mkdir -p "$HOME/run/kvcached-paracloud/state/live-trace-{job}"
nohup bash "$HOME/kvcached/kvcached/experiments/dp_opt/watch-trace-remote.sh" {job} \\
    >> "$HOME/run/kvcached-paracloud/state/live-trace-{job}/remote-query.log" 2>&1 < /dev/null &
printf '%s\\n' "$!"'''
    watcher = ssh(remote)
    if watcher.returncode:
        raise RuntimeError(f"Job {job} exists but remote query-only monitor failed; do not submit again")
    entry["remote_watcher_pid"] = int(watcher.stdout.strip())
    state.update({"phase": "running_frozen_stage", "job": job, "plan": plan})
    save_state(root, state)
    return job


def paired_replication(base, root):
    records = {}
    for job in (245583, 247027):
        records[str(job)] = {}
        for baseline, prefix in (("no_move", ""), ("no_move_fair", "fair-")):
            path = base / f"tail-slo-{prefix}{job}.json"
            records[str(job)][baseline] = json.loads(path.read_text())
    for baseline in ("no_move", "no_move_fair"):
        before, after = records["245583"][baseline], records["247027"][baseline]
        if (before["source_sha256"]["trace.json"] != after["source_sha256"]["trace.json"]
                or before["deadline_s"] != after["deadline_s"]):
            raise ValueError("Old paired replay changed frozen trace or SLO")
    (root / "paired-replication-245583-247027.json").write_text(json.dumps(records, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--sync-script", type=Path, required=True)
    parser.add_argument("--wait-for-old", type=int, default=247027)
    parser.add_argument("--confirmation-seeds", nargs="+", type=int,
                        default=list(range(20261025, 20261033)))
    args = parser.parse_args()
    args.archive.mkdir(parents=True, exist_ok=True)
    if (args.archive / "campaign-state.json").exists():
        parser.error("Preserve existing campaign; resume must reconcile recorded Slurm jobs explicitly")
    state = {"phase": "waiting_for_old_terminal_archive", "old_job": args.wait_for_old, "jobs": [],
             "policy_objective": "Code_P95_reduction_vs_both_baselines_all_Conv_costs_reported",
             "confirmation_seeds_predeclared": args.confirmation_seeds,
             "exploration_plan_predeclared": {"outputs": [8192, 12288, 16384], "budgets": [10240, 11328],
                 "stages": 4, "rate_multipliers": [.85, 1, 1.15], "training_seed": 20261024},
             "total_research_goal_completed": False, "publication_authorized": False}
    scripts = Path(__file__).resolve().parent
    save_state(args.archive, state)
    try:
        while True:
            check_pause(args.archive)
            terminal = args.base_archive / f"terminal-processing-{args.wait_for_old}.json"
            if terminal.exists():
                prior = json.loads(terminal.read_text())
                if prior["status"] != "initial_trace_verified":
                    raise RuntimeError("Old run needs archive/verification diagnosis; do not overwrite frozen runtime")
                break
            time.sleep(300)
        paired_replication(args.base_archive, args.archive)
        state["phase"] = "protected_sync_after_old_archive"
        save_state(args.archive, state)
        result = subprocess.run([sys.executable, str(args.sync_script)], text=True, capture_output=True)
        (args.archive / "protected-sync.txt").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError("Protected sync rejected; no new allocation submitted")
        best = None
        all_cases = []
        for stage in range(4):
            plan = training_plan(stage, best)
            job = start_job(args.archive, plan, state)
            terminal_state = poll_job(args.archive, job, state)
            if terminal_state != "COMPLETED":
                archive_job(args.archive, job, scripts, state)
                raise RuntimeError("GPU stage failed; preserve raw files and diagnose before resubmission")
            verified = archive_and_verify(args.archive, job, args.base_archive / "input", scripts, state)
            for case in verified["cases"]:
                case["job"] = job
            all_cases.extend(verified["cases"])
            available = [case for case in all_cases if case["analysis"]["minimum_Code_p95_reduction_fraction"] is not None]
            if not available:
                raise RuntimeError("No measured Code samples; cannot select an optimum from missing evidence")
            best = max(available, key=lambda case: case["analysis"]["minimum_Code_p95_reduction_fraction"])
            state.update({"phase": "verified_exploration_stage_complete", "best_observed": best,
                          "all_exploration_cases": all_cases})
            save_state(args.archive, state)
        confirmations = []
        confirmation_roots = []
        for seed in args.confirmation_seeds:
            job = start_job(args.archive, confirmation_plan(best, seed), state)
            terminal_state = poll_job(args.archive, job, state)
            if terminal_state != "COMPLETED":
                archive_job(args.archive, job, scripts, state)
                raise RuntimeError("Confirmation allocation failed; do not omit the failed seed")
            verified = archive_and_verify(args.archive, job, args.base_archive / "input", scripts, state)
            confirmation_roots.extend(args.archive / f"live-trace-{job}" / case["directory"]
                                      for case in verified["cases"])
            confirmations.extend(verified["cases"])
            state["verified_confirmations"] = confirmations
            save_state(args.archive, state)
        from analyze_stress_sweep import analyze_confirmations

        conclusion = analyze_confirmations(confirmation_roots, args.confirmation_seeds)
        (args.archive / "heldout-conclusion.json").write_text(json.dumps(conclusion, indent=2) + "\n")
        state.update({"phase": "declared_grid_and_all_heldout_seeds_verified", "verified_confirmations": confirmations,
                      "heldout_conclusion": conclusion,
                      "total_research_goal_completed": conclusion["confirmed_50pct_Code_p95_both_baselines"],
                      "next_action": "report_all_costs" if conclusion["confirmed_50pct_Code_p95_both_baselines"]
                                     else "diagnose_heldout_results_and_predeclare_next_design_iteration",
                      "scope": "bounded_grid_observed_best_all_seeds_and_negative_results_retained_no_global_optimum_claim"})
        save_state(args.archive, state)
    except BaseException as error:
        state.update({"phase": "stopped_for_diagnosis_or_user_pause", "error_type": type(error).__name__,
                      "error": str(error), "automatic_resubmit_after_failure": False})
        save_state(args.archive, state)
        raise


if __name__ == "__main__":
    main()
