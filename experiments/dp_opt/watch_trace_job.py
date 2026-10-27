import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def process_terminal(job, archive, inputs, archive_attempts=1):
    scripts = Path(__file__).resolve().parent
    commands = [([sys.executable, str(scripts / "archive_trace_job.py"), str(job), "--archive", str(archive)],
                 "archive")]
    raw = archive / f"live-trace-{job}"
    if raw.exists():
        commands[0][0].append("--resume")
    report = {"job": job, "status": "processing", "steps": []}
    for command, name in commands:
        attempts = []
        for attempt in range(archive_attempts):
            stamp = datetime.now().astimezone().isoformat()
            result = subprocess.run(command, text=True, capture_output=True)
            text = result.stdout + result.stderr
            attempts.append({"attempt": attempt + 1, "started_at": stamp, "exit_code": result.returncode})
            with (archive / f"terminal-{job}-{name}.log").open("a") as output:
                output.write(f"\n=== attempt={attempt + 1} started_at={stamp} ===\n" + text)
            if result.returncode == 0:
                break
            transient = any(marker in text for marker in (
                "returned non-zero exit status 255", "timed out", "Broken pipe", "Connection closed",
                "Could not resolve hostname", "Remote query-only watcher is still writing"))
            if not transient or attempt + 1 == archive_attempts:
                break
            report.update({"status": "archive_retry_pending", "archive_attempts": attempts,
                           "next_action": "retry_transient_network_failure_after_300s_preserving_partial_downloads"})
            (archive / f"terminal-processing-{job}.json").write_text(json.dumps(report, indent=2) + "\n")
            time.sleep(300)
            if raw.exists() and "--resume" not in command:
                command.append("--resume")
        report["archive_attempts"] = attempts
        report["steps"].append({"name": name, "exit_code": result.returncode})
        if result.returncode:
            report.update({"status": "archive_failed", "next_action": "preserve_incomplete_download_and_retry_archive"})
            break
    else:
        commands = []
        if (raw / "capacity" / "summary.json").exists():
            commands.append(([sys.executable, str(scripts / "verify_agentic_capacity.py"),
                              str(raw / "capacity"), "--inputs", str(inputs), "--output",
                              str(archive / f"capacity-verification-{job}.json")], "capacity"))
        if (raw / "calibration" / "calibration-summary.json").exists():
            commands.append(([sys.executable, str(scripts / "verify_agentic_calibration.py"),
                              str(raw / "calibration"), "--inputs", str(inputs), "--output",
                              str(archive / f"calibration-verification-{job}.json")], "calibration"))
        if (raw / "transfer" / "transfer-summary.json").exists():
            commands.append(([sys.executable, str(scripts / "verify_transfer_pilot.py"), str(raw / "transfer"),
                              "--marks", str(inputs / "conv-marks.json"), "--output",
                              str(archive / f"transfer-verification-{job}.json")], "transfer"))
        if (raw / "transfer" / "admission-trials.json").exists():
            commands.append(([sys.executable, str(scripts / "verify_admission_cost.py"),
                              str(raw / "transfer"), "--inputs", str(inputs), "--output",
                              str(archive / f"admission-cost-verification-{job}.json")], "admission_cost"))
        if (raw / "transfer" / "layout-trials.json").exists():
            commands.append(([sys.executable, str(scripts / "layout_cost_pilot.py"),
                              str(raw / "transfer"), "--output",
                              str(archive / f"layout-cost-verification-{job}.json")], "layout_cost"))
        if (raw / "summary.json").exists() and json.loads((raw / "summary.json").read_text()).get("status") == "passed":
            commands.extend([
                ([sys.executable, str(scripts / "verify_trace.py"), str(raw), "--prompt-bank",
                  str(inputs / "prompt-bank.json"), "--output", str(archive / f"verification-{job}.json")], "verification"),
                ([sys.executable, str(scripts / "analyze_trace.py"), str(raw), "--prompt-bank",
                  str(inputs / "prompt-bank.json"), "--output", str(archive / f"diagnostics-{job}.json")], "diagnostics")])
            if (raw / "config.json").exists():
                policies = json.loads((raw / "config.json").read_text()).get("policies", [])
                if "kv_only" in policies:
                    for baseline, suffix in (("no_move", ""), ("no_move_fair", "fair-")):
                        if baseline in policies:
                            commands.append(([sys.executable, str(scripts / "analyze_tail_slo.py"), str(raw),
                                              "--baseline", baseline, "--treatment", "kv_only", "--output",
                                              str(archive / f"tail-slo-{suffix}{job}.json")], f"tail_{baseline}"))
        for command, name in commands:
            result = subprocess.run(command, text=True, capture_output=True)
            (archive / f"terminal-{job}-{name}.log").write_text(result.stdout + result.stderr)
            report["steps"].append({"name": name, "exit_code": result.returncode})
            if result.returncode:
                report.update({"status": "verification_failed", "next_action": "diagnose_preserved_raw_evidence_before_resubmission"})
                break
        else:
            formal = any(step["name"] == "diagnostics" for step in report["steps"])
            transfer = any(step["name"] == "transfer" for step in report["steps"])
            admission_cost = any(step["name"] == "admission_cost" for step in report["steps"])
            layout_cost = any(step["name"] == "layout_cost" for step in report["steps"])
            report.update({"status": "initial_trace_verified" if formal else
                           "layout_cost_pilot_verified" if layout_cost else
                           "admission_cost_pilot_verified" if admission_cost else
                           "transfer_pilot_verified" if transfer else "archived_without_complete_formal_trace",
                           "next_action": "review_all_policy_metrics_and_negative_outcomes_before_next_experiment" if formal
                                          else "freeze_raw_verified_small_population_cost_then_same_trace_comparison" if layout_cost
                                          else "calibrate_wait_cost_and_guardrails_before_frozen_trace_replay" if admission_cost
                                          else "review_paired_transfer_cost_before_freezing_wait_parameters_or_trace" if transfer
                                          else "review_calibration_or_runtime_failure_before_next_experiment"})
    (archive / f"terminal-processing-{job}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=int)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--agentic-inputs", type=Path)
    parser.add_argument("--terminal-only", action="store_true")
    parser.add_argument("--archive-attempts", type=int, default=12)
    args = parser.parse_args()
    if args.job <= 0:
        parser.error("Require positive Slurm job ID")
    if args.archive_attempts < 1:
        parser.error("Require positive archive attempt count")
    if args.terminal_only and args.agentic_inputs is None:
        parser.error("Terminal-only recovery requires agentic inputs")
    args.archive.mkdir(parents=True, exist_ok=True)
    if args.terminal_only:
        print(json.dumps(process_terminal(args.job, args.archive, args.agentic_inputs,
                                          args.archive_attempts), indent=2), flush=True)
        return
    command = f'''date -Is
squeue -h -j {args.job} -o '%i|%T|%M|%R'
sacct -n -X -j {args.job} --format=JobID,State,ExitCode,Start,End,Elapsed,NodeList -P
for filename in phase.json progress.json error.json exit-status capacity/summary.json capacity/error.json capacity/capacity-setting-initial.json capacity/capacity-setting.json transfer/progress.json transfer/error.json transfer/transfer-summary.json pilot/summary.json pilot/error.json calibration/progress.json calibration/error.json calibration/summary.json; do
  artifact="$HOME/run/kvcached-paracloud/state/live-trace-{args.job}/$filename"
  if [ -f "$artifact" ]; then printf '\\n%s\\n' "$filename"; cat "$artifact"; fi
done
for filename in pilot/pilot-progress.json pilot/pilot-summary.json calibration/calibration-summary.json; do
  artifact="$HOME/run/kvcached-paracloud/state/live-trace-{args.job}/$filename"
  if [ -f "$artifact" ]; then
    printf '\\n%s compact\\n' "$filename"
    sed -n '/"population":/p; /"conv_mean_service_s":/p; /"code_service_s":/p; /"suggested_conv_rate_rps":/p; /"suggested_code_rate_rps":/p; /"status":/p; /"scope":/p; /"conv_rate_rps":/p; /"mean_sampled_conv_reserved_mib":/p; /"mean_sampled_external_queue_count":/p; /"global_enough_local_blocked_code_samples":/p' "$artifact"
  fi
done
printf '\\nLOG_TAIL\\n'
tail -n 14 "$HOME/run/kvcached-paracloud/logs/live-trace-{args.job}.out" 2>/dev/null
tail -n 14 "$HOME/run/kvcached-paracloud/logs/live-trace-{args.job}.err" 2>/dev/null
true
'''
    previous = None
    while True:
        now = time.time()
        gap = now - previous if previous is not None else None
        previous = now
        stamp = datetime.now().astimezone().isoformat()
        try:
            result = subprocess.run(["ssh", "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=20",
                                     "-o", "ServerAliveCountMax=3", "paracloud", command],
                                    text=True, capture_output=True, timeout=120)
            stdout, stderr, returncode = result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            stdout, stderr, returncode = "", "SSH query timed out; retry on the next scheduled poll\n", 124
        text = f"\n=== {stamp} actual_gap_s={gap} ssh_exit={returncode} ===\n{stdout}{stderr}"
        with (args.archive / f"poll-{args.job}.log").open("a") as output:
            output.write(text)
        print(text, flush=True)
        state = next((line.split("|")[1] for line in stdout.splitlines()
                      if line.startswith(str(args.job) + "|") and len(line.split("|")) >= 7), None)
        if state and state.split()[0] in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                                         "NODE_FAIL", "PREEMPTED"):
            if args.agentic_inputs is not None:
                print(json.dumps(process_terminal(args.job, args.archive, args.agentic_inputs,
                                                  args.archive_attempts), indent=2), flush=True)
            return
        time.sleep(max(0, 300 - (time.time() - now)))


if __name__ == "__main__":
    main()
