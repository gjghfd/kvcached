import argparse
import difflib
import json
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from analyze_tail_slo import analyze
from amortized_campaign import paired_window
from rebalance_campaign import checked_ssh, protected_sync
from stress_campaign import archive_job, check_pause, poll_job, save_state
from verify_trace import verify
from verify_workaware_profile import verify_profile
from workaware_profile_campaign import digest, frozen_environment, submission_command, summarize_primary
from workaware_pressure_campaign import SYNC_FILES as PREVIOUS_SYNC_FILES


POLICIES = ("no_move", "no_move_fair", "random_no_return", "kv_only", "kv_only_headroom", "kv_only_workaware")
PROMOTION_FILES = (
    "experiments/dp_opt/random_placement.py", "experiments/dp_opt/random_baseline_campaign.py",
    "experiments/dp_opt/live_controller.py", "experiments/dp_opt/live_trace.py",
    "experiments/dp_opt/workaware_evidence.py", "experiments/dp_opt/verify_trace.py",
    "experiments/dp_opt/verify_candidate_revalidation.py", "experiments/dp_opt/trace-benefit.sbatch",
    "experiments/dp_opt/workaware-profile-return.sbatch", "tests/test_dp_opt_random_placement.py",
    "tests/test_dp_opt_trace_planning.py", "tests/test_dp_opt_random_baseline_campaign.py",
    "experiments/dp_opt/docs/random-baseline-protocol.md")
SYNC_FILES = tuple(dict.fromkeys((*PREVIOUS_SYNC_FILES, *PROMOTION_FILES)))


def promotion_patch(repo, stage, files=PROMOTION_FILES):
    manifest, patch = [], ["*** Begin Patch\n"]
    for name in files:
        source, target = stage / name, repo / name
        before = target.read_text() if target.exists() else None
        after = source.read_text()
        manifest.append({"path": name, "old_sha256": digest(target) if target.exists() else None,
                         "sha256": digest(source)})
        if before == after:
            continue
        if before is None:
            patch.append(f"*** Add File: {name}\n")
            patch.extend("+" + line + "\n" for line in after.splitlines())
        else:
            patch.append(f"*** Update File: {name}\n")
            for line in list(difflib.unified_diff(before.splitlines(True), after.splitlines(True)))[2:]:
                patch.append("@@\n" if line.startswith("@@") else line)
    patch.append("*** End Patch\n")
    return manifest, "".join(patch)


def previous_ready(previous):
    if previous.get("phase") == "diagnosis_required_no_automatic_duplicate_submission":
        raise RuntimeError("Previous experiment requires diagnosis; do not replace it or submit another allocation")
    return (previous.get("phase") == "pressure_1p2_trace_verified_analyzed"
            and previous.get("GPU_experiment_complete") is True)


def prepare(root, previous_root, repo, stage):
    if (root / "campaign-state.json").exists():
        raise RuntimeError("Preserve the existing continuation ledger; never submit twice")
    previous = json.loads((previous_root / "campaign-state.json").read_text())
    trace = json.loads((previous_root / "frozen/trace.json").read_text())
    if digest(previous_root / "frozen/trace.json") != previous["trace_sha256"]:
        raise ValueError("Previous frozen trace changed")
    inputs = Path(previous["inputs"])
    for name, expected in previous["input_sha256"].items():
        if digest(inputs / name) != expected:
            raise ValueError("Input bank changed before baseline comparison")
    if trace["duration_s"] != 4800 or trace["model_rates_rps"] != [.0005, previous["conv_rate_rps"]]:
        raise ValueError("Baseline comparison must reuse the original rates and 4800-second trace")
    root.mkdir(parents=True, exist_ok=True)
    (root / "frozen").mkdir()
    shutil.copy2(previous_root / "frozen/trace.json", root / "frozen/trace.json")
    manifest, patch = promotion_patch(repo, stage)
    (root / "promotion.patch").write_text(patch)
    protocol = {"policies": list(POLICIES), "trace_sha256": previous["trace_sha256"],
                "model_rates_rps": trace["model_rates_rps"], "duration_s": trace["duration_s"],
                "repeats": 1, "recovery_populations": [2, 9],
                "recovery_policies": ["random_no_return", "kv_only_headroom", "kv_only_workaware"],
                "setting_change": "add_random_no_return_only_not_select_favorable_arrivals_or_shorten_requests",
                "scope": "first_paired_comparison_two_Code_samples_not_statistical_P99_confirmation"}
    (root / "frozen/protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    state = {"phase": "waiting_previous_verified_completion", "previous_job": previous["job"],
             "previous_root": str(previous_root), "repo": str(repo), "stage": str(stage),
             "inputs": str(inputs), "input_sha256": previous["input_sha256"],
             "trace_sha256": previous["trace_sha256"], "conv_rate_rps": previous["conv_rate_rps"],
             "protected_vllm_sha256": previous["protected_vllm_sha256"],
             "calibration": previous["calibration"], "promotion_manifest": manifest,
             "promotion_patch_sha256": digest(root / "promotion.patch"),
             "GPU_experiment_complete": False, "total_research_goal_completed": False,
             "publication_authorized": False, "next_action": "wait_300s_no_second_allocation_then_promote_and_run_six_policies"}
    save_state(root, state)
    return state


def promote(root, repo, stage, state, patch_tool):
    for item in state["promotion_manifest"]:
        target = repo / item["path"]
        if (digest(target) if target.exists() else None) != item["old_sha256"]:
            raise RuntimeError("Main source changed after isolated testing; do not overwrite user or running changes")
        if digest(stage / item["path"]) != item["sha256"]:
            raise RuntimeError("Isolated tested candidate changed before promotion")
    if digest(root / "promotion.patch") != state["promotion_patch_sha256"]:
        raise RuntimeError("Promotion patch changed")
    result = subprocess.run([str(patch_tool), (root / "promotion.patch").read_text()], cwd=repo,
                            text=True, capture_output=True, check=True)
    (root / "promotion-result.txt").write_text(result.stdout + result.stderr)
    for item in state["promotion_manifest"]:
        if digest(repo / item["path"]) != item["sha256"]:
            raise RuntimeError("Promoted source differs from the tested candidate")


def summarize_rebalance_reasons(phase):
    result = {}
    for policy in ("random_no_return", "kv_only_headroom", "kv_only_workaware"):
        reasons = {}
        decisions = json.loads((phase / f"r0-{policy}/decisions.json").read_text())
        for decision in decisions:
            if decision["path"] == "release_rebalance_ilp":
                reason = decision.get("reason", decision.get("status", "unspecified"))
                reasons[reason] = reasons.get(reason, 0) + 1
        result[policy] = {"release_rebalance_reasons": reasons,
                          "migrations": json.loads((phase / f"r0-{policy}/migrations.json").read_text()),
                          "summary": json.loads((phase / f"r0-{policy}/summary.json").read_text())}
    return result


def run(root, patch_tool):
    state = json.loads((root / "campaign-state.json").read_text())
    if state["phase"] != "waiting_previous_verified_completion":
        raise RuntimeError("Continuation ledger already advanced; reconcile instead of resubmitting")
    repo, stage, previous_root = (Path(state[name]) for name in ("repo", "stage", "previous_root"))
    try:
        while True:
            check_pause(root)
            check_pause(previous_root)
            previous = json.loads((previous_root / "campaign-state.json").read_text())
            state["last_previous_check_at"] = datetime.now().astimezone().isoformat()
            state["previous_last_poll_at"] = previous.get("last_poll_at")
            state["previous_state_last_observed"] = previous.get("state_last_observed")
            save_state(root, state)
            print(json.dumps({name: state.get(name) for name in (
                "last_previous_check_at", "phase", "previous_job", "previous_last_poll_at",
                "previous_state_last_observed")}), flush=True)
            if previous_ready(previous):
                break
            time.sleep(300)
        pressure = json.loads((previous_root / "pressure-results.json").read_text())
        if pressure.get("GPU_trace_complete_and_verified") is not True:
            raise RuntimeError("Previous pressure trace has not passed independent verification")
        state["previous_actual_no_move_ratio"] = pressure["actual_no_move_ratio"]
        for name, expected in previous["frozen_runtime_sha256"].items():
            if digest(repo / name) != expected:
                raise RuntimeError("Previous frozen local source changed before its verified completion")
        checked_ssh('test -z "$(squeue -h -u "$USER")"')
        reused_fit = verify_profile(Path(state["calibration"]))
        (root / "independent-reused-profile-fit.json").write_text(json.dumps(reused_fit, indent=2) + "\n")
        promote(root, repo, stage, state, patch_tool)
        state["frozen_runtime_sha256"] = {name: digest(repo / name) for name in SYNC_FILES
                                          if name.endswith((".py", ".sbatch", ".sh"))}
        state["phase"] = "promoted_tested_random_baseline_preparing_protected_sync"
        save_state(root, state)
        remote_stage, home = protected_sync(repo, root, state, files=SYNC_FILES, prefix="workaware-random-baseline")
        calibration = home + "/run/kvcached-paracloud/state/live-trace-256440/cadence"
        environment = frozen_environment(remote_stage + "/upload/frozen/trace.json", state["trace_sha256"],
                                         calibration, state["conv_rate_rps"])
        environment.update({"DP_OPT_WORKAWARE_POLICIES": " ".join(POLICIES), "DP_OPT_RANDOM_BASELINE": "1",
                            "DP_OPT_WORKAWARE_RECOVERY_POLICIES": "random_no_return kv_only_headroom kv_only_workaware"})
        ledger = remote_stage + "/submitted-job.txt"
        state.update({"phase": "submission_intent_do_not_duplicate", "submission_ledger": ledger,
                      "frozen_environment": environment})
        save_state(root, state)
        output = checked_ssh(submission_command(environment, ledger))
        job = int(output.split(";")[0])
        (root / "submit.txt").write_text(output + "\n")
        state.update({"job": job, "phase": "submitted_random_baseline_six_policy_comparison"})
        save_state(root, state)
        remote = f"{home}/run/kvcached-paracloud/state/live-trace-{job}"
        watcher = checked_ssh(f'''set -eu
mkdir -p {shlex.quote(remote)}
printf '%s\n' '{job}' > {shlex.quote(remote + '/remote-query.expected')}
nohup bash "$HOME/kvcached/kvcached/experiments/dp_opt/watch-trace-remote.sh" {job} \\
    >> {shlex.quote(remote + '/remote-query.log')} 2>&1 < /dev/null &
printf '%s\n' "$!"''')
        state["remote_watcher_pid"] = int(watcher)
        save_state(root, state)
        poll_job(root, job, state)
        archive_job(root, job, stage / "experiments/dp_opt", state)
        raw = root / f"live-trace-{job}"
        if (raw / "exit-status").read_text().strip() != "0":
            raise RuntimeError("GPU experiment failed; raw preserved, diagnose before retry")
        phase = raw / "paired-trace"
        if digest(phase / "trace.json") != state["trace_sha256"]:
            raise ValueError("Formal workload differs from the frozen paired trace")
        bank = Path(state["inputs"]) / "prompt-bank.json"
        checks = {"formal": verify(phase, bank),
                  **{f"recovery-{population}": verify(raw / f"recovery-{population}", bank) for population in (2, 9)}}
        (root / "independent-verification.json").write_text(json.dumps(checks, indent=2) + "\n")
        comparisons = {baseline: analyze(phase, baseline, "kv_only_workaware") for baseline in POLICIES[:-1]}
        windows = {baseline: paired_window(phase, baseline, "kv_only_workaware", 600, 4800) for baseline in POLICIES[:-1]}
        for name, value in (("all-request-comparisons", comparisons), ("evaluation-window-comparisons", windows),
                            ("primary-metrics", summarize_primary(comparisons)),
                            ("evaluation-window-primary-metrics", summarize_primary(windows)),
                            ("rebalance-diagnosis", summarize_rebalance_reasons(phase))):
            (root / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
        for name, expected in state["frozen_runtime_sha256"].items():
            if digest(repo / name) != expected:
                raise RuntimeError("Frozen local source changed during the new experiment")
        state.update({"phase": "random_baseline_six_policy_SHA_verified_analyzed", "GPU_experiment_complete": True,
                      "next_action": "inspect_Conv_Code_combined_TTFT_and_Conv_TPOT_then_general_cost_or_domain_iteration"})
        save_state(root, state)
    except Exception as error:
        state.update({"phase": "diagnosis_required_no_automatic_duplicate_submission", "error": repr(error)})
        save_state(root, state)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--patch-tool", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root, repo = args.root.resolve(), args.repo.resolve()
    stage = Path(__file__).resolve().parents[2]
    if not args.resume:
        prepare(root, args.previous_root.resolve(), repo, stage)
    if not args.prepare_only:
        run(root, args.patch_tool.absolute())


if __name__ == "__main__":
    main()
