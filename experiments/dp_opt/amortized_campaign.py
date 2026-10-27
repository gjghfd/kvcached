import argparse
import hashlib
import json
from pathlib import Path

from analyze_tail_slo import analyze, compare_group, validate_records
from rebalance_campaign import SYNC_FILES as PREVIOUS_SYNC_FILES
from rebalance_campaign import checked_ssh, frozen_environment, protected_sync, submission_command
from stress_campaign import archive_job, check_pause, poll_job, save_state, ssh
from verify_trace import verify


POLICIES = ("no_move", "no_move_fair", "own_model", "kv_only", "ilp", "kv_only_rebalance", "kv_only_amortized")
HEADROOM_POLICIES = ("no_move", "no_move_fair", "own_model", "kv_only", "ilp", "kv_only_headroom")
SYNC_FILES = list(dict.fromkeys(PREVIOUS_SYNC_FILES + [
    "experiments/dp_opt/amortized_rebalance.py", "experiments/dp_opt/amortized_campaign.py",
    "experiments/dp_opt/rebalance_campaign.py", "tests/test_dp_opt_amortized_rebalance.py",
    "tests/test_dp_opt_amortized_campaign.py", "tests/test_dp_opt_ilp_reference.py",
    "experiments/dp_opt/docs/amortized-rebalance-protocol.md",
    "experiments/dp_opt/reports/amortized-rebalance-status.md",
    "experiments/dp_opt/docs/experiment-matrix.json", "experiments/dp_opt/headroom_rebalance.py",
    "experiments/dp_opt/analyze_ttft_phases.py", "tests/test_dp_opt_headroom_rebalance.py",
    "experiments/dp_opt/docs/headroom-tail-protocol.md",
    "experiments/dp_opt/reports/headroom-tail-status.md"]))


def paired_window(root, baseline, treatment, start_s=600, end_s=2400):
    result = analyze(root, baseline, treatment)
    trace = json.loads((root / "trace.json").read_text())
    marks = {mark["request_id"]: mark for mark in trace["requests"]}
    before = validate_records(json.loads((root / f"r0-{baseline}/requests.json").read_text()), marks)
    after = validate_records(json.loads((root / f"r0-{treatment}/requests.json").read_text()), marks)
    identifiers = {identifier for identifier, mark in marks.items() if start_s <= mark["arrival_s"] < end_s}
    groups = {"all_requests": identifiers,
              "Code": {identifier for identifier in identifiers if marks[identifier]["model"] == 0},
              "Conv": {identifier for identifier in identifiers if marks[identifier]["model"] == 1}}
    result.update({"evaluation_start_s": start_s, "evaluation_end_s": end_s,
                   "groups": {name: compare_group(before, after, selected, result["deadline_s"])
                              for name, selected in groups.items()},
                   "Code_paired_requests": [record for record in result["Code_paired_requests"]
                                            if record["request_id"] in identifiers]})
    return result


def deployment_environment(config, trace_path, headroom=False):
    environment = frozen_environment(config, trace_path)
    environment.update({"DP_OPT_TRACE_POLICIES": " ".join(HEADROOM_POLICIES if headroom else POLICIES),
                        "DP_OPT_REBALANCE_OVERHEAD_FRACTION": "0.01",
                        "DP_OPT_REBALANCE_MAX_MOVES": "4",
                        "DP_OPT_REBALANCE_CANDIDATE_LIMIT": "8",
                        "DP_OPT_REBALANCE_MIN_GAIN_S": "1",
                        "DP_OPT_REBALANCE_COOLDOWN_S": "10",
                        "DP_OPT_REBALANCE_RETRY_S": "2"})
    if headroom:
        environment.update({"DP_OPT_REBALANCE_HEADROOM_QUANTILE": "0.95",
                            "DP_OPT_REBALANCE_HEADROOM_HISTORY_LIMIT": "64"})
    return environment


def check_frozen_sources(repo, state):
    for name, expected in state["frozen_runtime_sha256"].items():
        if hashlib.sha256((repo / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Frozen runtime changed; do not deploy or verify using an untested version")


def finish_job(root, state, inputs, scripts):
    job = state["job"]
    state["phase"] = "monitoring_amortized_frozen_trace"
    save_state(root, state)
    state["terminal"] = poll_job(root, job, state)
    archive_job(root, job, scripts, state)
    if state["terminal"] != "COMPLETED":
        raise RuntimeError("GPU job failed; raw SHA archive retained, diagnose before resubmission")
    check_frozen_sources(scripts.parents[1], state)
    result = verify(root / f"live-trace-{job}", inputs / "prompt-bank.json")
    (root / f"verification-{job}.json").write_text(json.dumps(result, indent=2) + "\n")
    treatment = state.get("treatment", "kv_only_amortized")
    baselines = ("no_move", "no_move_fair", "kv_only") if treatment == "kv_only_headroom" else ("no_move", "no_move_fair", "kv_only", "kv_only_rebalance")
    comparisons = {baseline: analyze(root / f"live-trace-{job}", baseline, treatment)
                   for baseline in baselines}
    windows = {baseline: paired_window(root / f"live-trace-{job}", baseline, treatment)
               for baseline in comparisons}
    (root / "all-request-comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    (root / "evaluation-window-comparisons.json").write_text(json.dumps(windows, indent=2) + "\n")
    if treatment == "kv_only_headroom":
        from analyze_ttft_phases import analyze_phases

        (root / "ttft-phase-attribution.json").write_text(json.dumps(analyze_phases(root / f"live-trace-{job}"), indent=2) + "\n")
    state.update({"phase": "amortized_trace_SHA_archived_independently_verified",
                  "GPU_experiment_complete": True, "total_research_goal_completed": False,
                  "next_action": "report_Code_Conv_combined_tail_and_costs_then_decide_next_experiment"})
    save_state(root, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--variant", choices=("amortized", "headroom"), default="amortized")
    args = parser.parse_args()
    root, scripts = args.root.resolve(), Path(__file__).resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    check_pause(root)
    if args.resume:
        state = json.loads((root / "campaign-state.json").read_text())
        if not state.get("job"):
            raise RuntimeError("Uncertain pre-submission state requires queue/ledger diagnosis, not another submit")
        if state.get("GPU_experiment_complete"):
            return
    else:
        if (root / "campaign-state.json").exists():
            raise RuntimeError("Existing coordinator state; use resume to prevent duplicate submission")
        frozen = root / "frozen"
        frozen.mkdir()
        (frozen / "trace.json").write_bytes((args.source / "trace.json").read_bytes())
        config = json.loads((args.source / "source-config.json").read_text())
        config["frozen_trace_sha256"] = hashlib.sha256((frozen / "trace.json").read_bytes()).hexdigest()
        (frozen / "source-config.json").write_text(json.dumps(config, indent=2) + "\n")
        policies = HEADROOM_POLICIES if args.variant == "headroom" else POLICIES
        state = {"phase": "preparing_amortized_same_trace", "jobs": [], "policies": list(policies),
                 "variant": args.variant, "treatment": "kv_only_headroom" if args.variant == "headroom" else "kv_only_amortized",
                 "trace_sha256": config["frozen_trace_sha256"], "GPU_experiment_complete": False,
                 "total_research_goal_completed": False, "publication_authorized": False,
                 "frozen_runtime_sha256": {name: hashlib.sha256((scripts.parents[1] / name).read_bytes()).hexdigest()
                                           for name in SYNC_FILES if name.endswith((".py", ".sbatch"))}}
        save_state(root, state)
        try:
            initial = checked_ssh('date -Is; squeue -h -u "$USER" -o "%i|%T|%j"')
            (root / "initial-query.txt").write_text(initial + "\n")
            state["protected_vllm_sha256"] = checked_ssh(
                'sha256sum "$HOME/kvcached/kvcached/kvcached/integration/vllm/patches.py"').split()[0]
            stage, home = protected_sync(scripts.parents[1], root, state, files=SYNC_FILES, prefix=args.variant)
            trace_path = stage + "/upload/frozen/trace.json"
            if checked_ssh(f"sha256sum '{trace_path}'").split()[0] != state["trace_sha256"]:
                raise RuntimeError("Uploaded trace differs from frozen paired input")
            environment = deployment_environment(config, trace_path, headroom=args.variant == "headroom")
            ledger = stage + "/submitted-job.txt"
            state.update({"phase": "submitting_amortized_once", "submission_ledger": ledger,
                          "frozen_environment": environment, "source_backup": stage + "/old"})
            save_state(root, state)
            result = ssh(submission_command(environment, ledger))
            (root / "submit.txt").write_text(result.stdout + result.stderr)
            submitted = result.stdout.strip() if result.returncode == 0 else checked_ssh(
                f"if test -f '{ledger}'; then cat '{ledger}'; fi")
            if not submitted:
                raise RuntimeError("Uncertain submission; inspect queue and durable ledger, never blindly retry")
            job = int(submitted.split(";")[0])
            state.update({"job": job, "jobs": [{"job": job, "policies": list(policies)}]})
            save_state(root, state)
            remote = checked_ssh(f'''mkdir -p "{home}/run/kvcached-paracloud/state/live-trace-{job}"
nohup bash "{home}/kvcached/kvcached/experiments/dp_opt/watch-trace-remote.sh" {job} \
    >> "{home}/run/kvcached-paracloud/state/live-trace-{job}/remote-query.log" 2>&1 < /dev/null &
printf '%s\\n' "$!"''')
            state["remote_watcher_pid"] = int(remote)
            save_state(root, state)
        except BaseException as error:
            state.update({"phase": "needs_diagnosis_no_duplicate_submission", "error": str(error)})
            save_state(root, state)
            raise
    try:
        finish_job(root, state, args.inputs.resolve(), scripts)
    except BaseException as error:
        state.update({"phase": "needs_diagnosis_no_duplicate_submission", "error": str(error)})
        save_state(root, state)
        raise


if __name__ == "__main__":
    main()
