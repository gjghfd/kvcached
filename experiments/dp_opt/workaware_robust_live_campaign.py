import argparse
import hashlib
import json
import shlex
import shutil
from pathlib import Path

from amortized_campaign import check_frozen_sources, paired_window
from analyze_return_repayment import analyze as repayment
from archive_trace_job import verify_raw
from long_conv_return_campaign import SYNC_FILES as PREVIOUS_SYNC_FILES
from long_conv_return_probe import validate_upload
from rebalance_campaign import checked_ssh, protected_sync
from stress_campaign import archive_job, check_pause, poll_job, save_state
from verify_trace import verify


SEEDS = (20261030, 20261031)
SYNC_FILES = tuple(dict.fromkeys((*PREVIOUS_SYNC_FILES,
    "experiments/dp_opt/workaware_robust_adapter.py", "experiments/dp_opt/workaware_robust_rebalance.py",
    "experiments/dp_opt/workaware_robust_ilp.py", "experiments/dp_opt/verify_workaware_robust_rebalance.py",
    "experiments/dp_opt/workaware_robust_live_campaign.py", "experiments/dp_opt/workaware-robust-live.sbatch",
    "experiments/dp_opt/docs/workaware-robust-native-protocol.md",
    "experiments/dp_opt/agentic_trace.py", "tests/test_dp_opt_robust_live_integration.py",
    "tests/test_dp_opt_zero_arrival_trace.py", "tests/test_dp_opt_robust_adapter.py",
    "tests/test_dp_opt_robust_ilp.py", "tests/test_dp_opt_workaware_robust_rebalance.py",
    "tests/test_dp_opt_workaware_robust_rebalance_verify.py", "tests/test_dp_opt_robust_live_campaign.py")))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parent_ready(state, followup, analysis):
    return (state.get("GPU_experiment_complete") is True and state.get("state_last_observed") == "COMPLETED"
            and state.get("phase") == "long_Conv_probe_SHA_verified_analyzed"
            and followup.get("phase") == "complete_independent_candidate_replay_not_GPU_benefit"
            and analysis.get("status") == "all_three_probe_analyses_verified_not_research_goal_complete"
            and set(analysis.get("phases", {})) == {f"probe-{seed}" for seed in (20261030, 20261031, 20261032)})


def validate_parent(parent):
    check_pause(parent)
    state = json.loads((parent / "campaign-state.json").read_text())
    followup = json.loads((parent / "robust-followup-status.json").read_text())
    analysis = json.loads((parent / "pause-inclusive-phase-analysis/observer-state.json").read_text())
    if not parent_ready(state, followup, analysis):
        raise RuntimeError("Preserve old allocation and frozen sources until all three independent analyses close")
    if digest(parent / "robust-followup-comparison.json") != followup.get("comparison_sha256"):
        raise RuntimeError("Parent robust follow-up digest differs from its completed status")
    raw = parent / f"live-trace-{state['job']}"
    verify_raw(raw, (parent / f"raw-{state['job']}.sha256").read_text())
    validation = json.loads((parent / "independent-verification.json").read_text())
    if (set(validation.get("probes", {})) != set(analysis["phases"])
            or validation.get("calibration", {}).get("source_verification") != "raw_verified"
            or followup.get("job") != state["job"] or analysis.get("job") != state["job"]):
        raise RuntimeError("Parent validation changed job, calibrated source or three-seed coverage")
    return state, raw


def submission_command(environment, ledger):
    exports = "; ".join(f"export {name}={shlex.quote(value)}" for name, value in environment.items())
    return (f'set -eu; test -z "$(squeue -h -u "$USER")"; test ! -e {shlex.quote(ledger)}; '
            'test "$(git -C "$HOME/kvcached/kvcached" branch --show-current)" = main; '
            'test "$(git -C "$HOME/kvcached/kvcached" remote get-url origin)" = git@github.com:gjghfd/kvcached.git; '
            + exports + '; sbatch --parsable --time=06:00:00 --export=ALL '
            '--output="$HOME/run/kvcached-paracloud/logs/live-trace-%j.out" '
            '--error="$HOME/run/kvcached-paracloud/logs/live-trace-%j.err" '
            '"$HOME/kvcached/kvcached/experiments/dp_opt/workaware-robust-live.sbatch" | '
            f'tee {shlex.quote(ledger)}')


def finish(root, state):
    repo = Path(state["repo"])
    terminal = poll_job(root, state["job"], state)
    archive_job(root, state["job"], repo / "experiments/dp_opt", state)
    check_frozen_sources(repo, state)
    raw = root / f"live-trace-{state['job']}"
    if terminal != "COMPLETED" or (raw / "exit-status").read_text().strip() != "0":
        raise RuntimeError("Robust native job failed; diagnose complete SHA archive before any retry")
    results = {"scope": "closed_N2_native_robust_pairs_not_final_Poisson_trace", "phases": {},
               "research_goal_complete": False}
    for seed in SEEDS:
        name = f"native-{seed}"
        phase = raw / name
        if digest(phase / "trace.json") != state["trace_sha256"][str(seed)]:
            raise RuntimeError("Native robust evaluation changed frozen requests or timing")
        verified = verify(phase, Path(state["inputs"]) / "prompt-bank.json")
        decisions = json.loads((phase / "r0-kv_only_workaware/decisions.json").read_text())
        plans = [decision for decision in decisions if decision["path"] == "release_rebalance_ilp"
                 and "before_pause" in decision]
        if not plans or any(decision.get("workaware_optimizer") != "robust" for decision in plans):
            raise RuntimeError("Native verification did not execute the typed robust release path")
        results["phases"][name] = {"verification": verified,
            "comparison": paired_window(phase, "random_no_return", "kv_only_workaware", 0, 1200),
            "repayment": repayment(phase), "robust_planned_attempts": len(plans),
            "committed_returns": sum(bool(decision["moves"]) for decision in plans)}
    (root / "independent-verification.json").write_text(json.dumps(results, indent=2) + "\n")
    state.update({"phase": "robust_native_pairs_SHA_verified_analyzed", "GPU_experiment_complete": True,
                  "next_action": "measure_open_loop_service_KV_stability_then_freeze_Poisson_evaluation",
                  "total_research_goal_completed": False})
    save_state(root, state)


def run(root, parent):
    repo = Path(__file__).resolve().parents[2]
    root.mkdir(parents=True, exist_ok=True)
    check_pause(root)
    state_path = root / "campaign-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "phase": "preparing_native_robust_pairs", "repo": str(repo), "parent_root": str(parent),
        "GPU_experiment_complete": False, "total_research_goal_completed": False, "publication_authorized": False}
    if state["repo"] != str(repo) or state["parent_root"] != str(parent):
        raise RuntimeError("Preserve native campaign identity")
    if state.get("GPU_experiment_complete"):
        return
    if state["phase"] == "diagnosis_required_no_automatic_duplicate_submission":
        raise RuntimeError("Reconcile archived failure before resuming")
    try:
        if "job" not in state:
            if state["phase"] != "preparing_native_robust_pairs":
                raise RuntimeError("Reconcile submission intent and remote ledger; never duplicate allocation")
            previous, raw = validate_parent(parent)
            state.update({"parent_job": previous["job"], "inputs": previous["inputs"],
                          "protected_vllm_sha256": previous["protected_vllm_sha256"],
                          "parent_fit_sha256": digest(raw / "long-context-fit.json"),
                          "trace_sha256": {str(seed): digest(raw / f"frozen/probe-{seed}.json") for seed in SEEDS},
                          "frozen_runtime_sha256": {name: digest(repo / name) for name in SYNC_FILES}})
            save_state(root, state)
            check_frozen_sources(repo, state)
            shutil.copytree(raw / "frozen", root / "frozen")
            shutil.copy2(raw / "frozen/probe-20261030.json", root / "frozen/trace.json")
            validate_upload(root / "frozen", Path(state["inputs"]))
            uploads = ("frozen/probe-config.json", "frozen/calibration-plan.json",
                       *(f"frozen/probe-{seed}.json" for seed in (20261030, 20261031, 20261032)))
            stage, home = protected_sync(repo, root, state, files=SYNC_FILES, prefix="robust-native", upload_files=uploads)
            environment = {"DP_OPT_ROBUST_UPLOAD": stage + "/upload", "DP_OPT_ROBUST_PARENT_JOB": str(previous["job"]),
                "DP_OPT_ROBUST_FIT_SHA256": state["parent_fit_sha256"],
                "DP_OPT_MOVE_FIXED_S": "1.7743364535272121", "DP_OPT_MOVE_BANDWIDTH_GIB_S": "11.64086282917902"}
            ledger = stage + "/submitted-job.txt"
            state.update({"phase": "submission_intent_do_not_duplicate", "remote_sync_stage": stage,
                          "submission_ledger": ledger, "frozen_environment": environment})
            save_state(root, state)
            output = checked_ssh(submission_command(environment, ledger))
            state.update({"job": int(output.split(";")[0]), "phase": "submitted_native_robust_pairs"})
            (root / "submit.txt").write_text(output + "\n")
            save_state(root, state)
            remote = home + "/run/kvcached-paracloud/state/live-trace-" + str(state["job"])
            watcher = checked_ssh(f'''set -eu
mkdir -p {shlex.quote(remote)}
nohup bash "$HOME/kvcached/kvcached/experiments/dp_opt/watch-trace-remote.sh" {state['job']} \\
    >> {shlex.quote(remote + '/remote-query.log')} 2>&1 < /dev/null &
printf '%s\\n' "$!"''')
            state["remote_watcher_pid"] = int(watcher)
            save_state(root, state)
        finish(root, state)
    except Exception as error:
        state.update({"phase": "diagnosis_required_no_automatic_duplicate_submission", "error": repr(error)})
        save_state(root, state)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.parent_root.resolve())


if __name__ == "__main__":
    main()
