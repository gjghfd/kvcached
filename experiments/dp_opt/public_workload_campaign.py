import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from amortized_campaign import check_frozen_sources
from archive_trace_job import verify_raw
from public_response_inputs import digest
from public_workload_search import analyze, prepare, validate_frozen
from rebalance_campaign import checked_ssh, protected_sync
from stress_campaign import archive_job, check_pause, poll_job, save_state
from workaware_robust_live_campaign import SYNC_FILES as PREVIOUS_SYNC_FILES


SYNC_FILES = tuple(dict.fromkeys((*PREVIOUS_SYNC_FILES,
    "experiments/dp_opt/public_response_profile.py", "experiments/dp_opt/public_response_inputs.py",
    "experiments/dp_opt/public_workload_search.py", "experiments/dp_opt/public_workload_campaign.py",
    "experiments/dp_opt/public-workload-search.sbatch", "experiments/dp_opt/long_conv_open_loop.py",
    "experiments/dp_opt/kv_pressure.py", "experiments/dp_opt/docs/public-response-workload.md",
    "experiments/dp_opt/reports/public-response-profile.md", "tests/test_dp_opt_public_response_profile.py",
    "tests/test_dp_opt_public_response_inputs.py", "tests/test_dp_opt_public_workload_search.py",
    "tests/test_dp_opt_public_workload_campaign.py")))


def submission_command(environment, ledger):
    exports = "; ".join(f"export {name}={shlex.quote(value)}" for name, value in environment.items())
    return (f'set -eu; test -z "$(squeue -h -u "$USER")"; test ! -e {shlex.quote(ledger)}; '
            'test "$(git -C "$HOME/kvcached/kvcached" branch --show-current)" = main; '
            'test "$(git -C "$HOME/kvcached/kvcached" remote get-url origin)" = git@github.com:gjghfd/kvcached.git; '
            + exports + '; sbatch --parsable --time=18:00:00 --export=ALL '
            '--output="$HOME/run/kvcached-paracloud/logs/live-trace-%j.out" '
            '--error="$HOME/run/kvcached-paracloud/logs/live-trace-%j.err" '
            '"$HOME/kvcached/kvcached/experiments/dp_opt/public-workload-search.sbatch" | '
            f'tee {shlex.quote(ledger)}')


def finish(root, state):
    repo = Path(state["repo"])
    terminal = poll_job(root, state["job"], state)
    archive_job(root, state["job"], repo / "experiments/dp_opt", state)
    check_frozen_sources(repo, state)
    raw = root / f"live-trace-{state['job']}"
    if terminal != "COMPLETED" or (raw / "exit-status").read_text().strip() != "0":
        raise RuntimeError("Public GPU search failed; diagnose its complete SHA archive before any retry")
    result = analyze(raw)
    if result != json.loads((raw / "search-analysis.json").read_text()):
        raise RuntimeError("Independent public search reconstruction differs from compute analysis")
    (root / "independent-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    state.update({"phase": "public_source_mechanism_calibration_SHA_verified",
                  "GPU_calibration_complete": True, "candidate_case": result["candidate_case"],
                  "total_research_goal_completed": False,
                  "next_action": "freeze_bounded_seven_policy_grid_and_heldout_or_diagnose_unqualified_pressure"})
    save_state(root, state)


def run(root, previous):
    repo = Path(__file__).resolve().parents[2]
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "campaign-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "repo": str(repo), "previous_root": str(previous), "phase": "new_public_source_search_prepared",
        "publication_authorized": False, "total_research_goal_completed": False,
        "GPU_calibration_complete": False}
    if state["repo"] != str(repo) or state["previous_root"] != str(previous):
        raise RuntimeError("Preserve public campaign identity")
    if state.get("GPU_calibration_complete"):
        return
    if "diagnosis_required" in state["phase"]:
        raise RuntimeError("Diagnose and reconcile the current campaign; never duplicate its allocation")
    save_state(root, state)
    try:
        check_pause(root)
        if "job" not in state:
            if state["phase"] != "new_public_source_search_prepared":
                raise RuntimeError("Reconcile the submission intent/remote ledger before retry")
            old = json.loads((previous / "campaign-state.json").read_text())
            if old.get("GPU_calibration_complete") is not True or old.get("state_last_observed") != "COMPLETED":
                raise RuntimeError("Old synthetic calibration must fully close before new source deployment")
            verify_raw(previous / f"live-trace-{old['job']}", (previous / f"raw-{old['job']}.sha256").read_text())
            frozen = root / "frozen"
            config = validate_frozen(frozen) if (frozen / "search-config.json").exists() else prepare(root)
            state.update({"inputs": str(frozen / "inputs"), "configuration": config,
                "previous_job": old["job"], "protected_vllm_sha256": old["protected_vllm_sha256"],
                "frozen_runtime_sha256": {name: digest(repo / name) for name in SYNC_FILES
                                           if name.endswith((".py", ".sh", ".sbatch"))}})
            save_state(root, state)
            check_frozen_sources(repo, state)
            tests = sorted(name for name in SYNC_FILES if name.startswith("tests/"))
            with (root / "cpu-regression.log").open("w") as output:
                regression = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests], cwd=repo,
                                            stdout=output, stderr=subprocess.STDOUT)
            (root / "cpu-regression.exit").write_text(str(regression.returncode) + "\n")
            if regression.returncode:
                raise RuntimeError("Public source/runtime CPU regression failed; do not synchronize or submit")
            check_frozen_sources(repo, state)
            uploads = tuple(str(path.relative_to(root)) for path in sorted(frozen.rglob("*"))
                            if path.is_file() and path != frozen / "trace.json")
            stage, home = protected_sync(repo, root, state, files=SYNC_FILES, prefix="public-response-search",
                                         upload_files=uploads)
            environment = {"DP_OPT_PUBLIC_UPLOAD": stage + "/upload",
                "DP_OPT_MOVE_FIXED_S": "1.7743364535272121", "DP_OPT_MOVE_BANDWIDTH_GIB_S": "11.64086282917902"}
            ledger = stage + "/submitted-job.txt"
            state.update({"phase": "submission_intent_do_not_duplicate", "remote_sync_stage": stage,
                          "submission_ledger": ledger, "frozen_environment": environment})
            save_state(root, state)
            output = checked_ssh(submission_command(environment, ledger))
            state.update({"job": int(output.split(";")[0]), "phase": "submitted_public_cost_native_open_loop_search"})
            (root / "submit.txt").write_text(output + "\n")
            save_state(root, state)
            remote = home + f"/run/kvcached-paracloud/state/live-trace-{state['job']}"
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
    parser.add_argument("--previous-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.previous_root.resolve())


if __name__ == "__main__":
    main()
