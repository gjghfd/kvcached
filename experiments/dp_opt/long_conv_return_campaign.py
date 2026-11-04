import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from amortized_campaign import check_frozen_sources, paired_window
from archive_trace_job import verify_raw
from long_conv_return_probe import POLICIES, prepare, validate_upload
from rebalance_campaign import checked_ssh, protected_sync
from stress_campaign import archive_job, check_pause, poll_job, save_state
from verify_trace import verify
from verify_workaware_profile import verify_profile
from workaware_phase_campaign import SYNC_FILES as PREVIOUS_SYNC_FILES


SYNC_FILES = tuple(dict.fromkeys((*PREVIOUS_SYNC_FILES,
    "experiments/dp_opt/long_conv_return_probe.py", "experiments/dp_opt/long_conv_return_campaign.py",
    "experiments/dp_opt/long-conv-return-probe.sbatch", "experiments/dp_opt/analyze_return_recovery.py",
    "experiments/dp_opt/analyze_return_token_windows.py", "tests/test_dp_opt_long_conv_return_probe.py",
    "tests/test_dp_opt_long_conv_return_campaign.py", "experiments/dp_opt/docs/long-conv-return-probe.md")))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def previous_ready(previous):
    if previous.get("phase") == "diagnosis_required_no_automatic_duplicate_submission":
        raise RuntimeError("Previous allocation failed; diagnose archived raw before another experiment")
    return (previous.get("GPU_experiment_complete") is True
            and previous.get("state_last_observed") == "COMPLETED"
            and previous.get("phase") == "early_return_lower_rate_SHA_verified_analyzed")


def await_previous(root, previous_root, repo):
    while True:
        check_pause(root)
        check_pause(previous_root)
        previous = json.loads((previous_root / "campaign-state.json").read_text())
        check_frozen_sources(repo, previous)
        if previous_ready(previous):
            manifest = (previous_root / f"raw-{previous['job']}.sha256").read_text()
            raw = previous_root / f"live-trace-{previous['job']}"
            verify_raw(raw, manifest)
            verification = json.loads((previous_root / "independent-verification.json").read_text())
            if set(verification) != {"supplement", "composite", "cache-fixed-original", "lower-code-rate"}:
                raise RuntimeError("Previous allocation lacks complete calibration and trace verification")
            return previous
        print(json.dumps({"waiting_for_verified_previous_job": previous.get("job"),
                          "phase": previous.get("phase"), "last_poll_at": previous.get("last_poll_at")}), flush=True)
        time.sleep(300)


def submission_command(environment, ledger):
    exports = "; ".join(f"export {name}={shlex.quote(value)}" for name, value in environment.items())
    return (f'set -eu; test -z "$(squeue -h -u "$USER")"; test ! -e {shlex.quote(ledger)}; '
            'test "$(git -C "$HOME/kvcached/kvcached" branch --show-current)" = main; '
            'test "$(git -C "$HOME/kvcached/kvcached" remote get-url origin)" = git@github.com:gjghfd/kvcached.git; '
            + exports + '; sbatch --parsable --time=12:00:00 --export=ALL '
            '--output="$HOME/run/kvcached-paracloud/logs/live-trace-%j.out" '
            '--error="$HOME/run/kvcached-paracloud/logs/live-trace-%j.err" '
            f'"$HOME/kvcached/kvcached/experiments/dp_opt/long-conv-return-probe.sbatch" > {shlex.quote(ledger)}; '
            f'cat {shlex.quote(ledger)}')


def finish(root, state):
    from analyze_return_recovery import analyze_recovery
    from analyze_return_token_windows import analyze as token_windows

    repo, inputs = Path(state["repo"]), Path(state["inputs"])
    terminal = poll_job(root, state["job"], state)
    archive_job(root, state["job"], repo / "experiments/dp_opt", state)
    raw = root / f"live-trace-{state['job']}"
    check_frozen_sources(repo, state)
    if terminal != "COMPLETED" or (raw / "exit-status").read_text().strip() != "0":
        raise RuntimeError("Long Conv probe failed; raw retained, diagnose before retry")
    validate_upload(raw / "frozen", inputs)
    fit = verify_profile(raw / "long-context-calibration")
    expected_profiles = {f"{model}:{population}" for model in (0, 1) for population in (2, 3)}
    if set(fit["population_profiles"]["profiles"]) != expected_profiles:
        raise RuntimeError("Independent long-context fit changed its declared population coverage")
    results = {"calibration": fit, "probes": {}, "research_goal_complete": False}
    for seed in state["probe_config"]["seeds"]:
        name = f"probe-{seed}"
        phase = raw / name
        if digest(phase / "trace.json") != state["probe_config"]["frozen_sha256"][name + ".json"]:
            raise RuntimeError("GPU probe changed frozen arrivals, prompts or output lengths")
        results["probes"][name] = {"verification": verify(phase, inputs / "prompt-bank.json"),
            "comparisons": {baseline: paired_window(phase, baseline, "kv_only_workaware", 0, 1200)
                            for baseline in POLICIES if baseline != "kv_only_workaware"},
            "recovery": analyze_recovery(phase, start_s=0), "token_windows": token_windows(phase),
            "scope": "closed_N2_mechanism_not_final_Poisson_trace_or_Code_tail_significance"}
    (root / "independent-verification.json").write_text(json.dumps(results, indent=2) + "\n")
    state.update({"phase": "long_Conv_probe_SHA_verified_analyzed", "GPU_experiment_complete": True,
                  "total_research_goal_completed": False,
                  "next_action": "report_repaid_pause_return_vs_natural_recovery_then_calibrate_open_loop_1p2_pressure"})
    save_state(root, state)


def run(root, previous_root):
    repo = Path(__file__).resolve().parents[2]
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "campaign-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "phase": "waiting_for_previous_complete_archive", "repo": str(repo),
        "previous_root": str(previous_root), "GPU_experiment_complete": False,
        "total_research_goal_completed": False, "publication_authorized": False}
    if state["repo"] != str(repo) or state["previous_root"] != str(previous_root):
        raise RuntimeError("Preserve campaign identity")
    save_state(root, state)
    try:
        if "job" not in state:
            if state["phase"] != "waiting_for_previous_complete_archive":
                raise RuntimeError("Reconcile the submission ledger before retry; do not duplicate allocation")
            previous = await_previous(root, previous_root, repo)
            inputs = Path(previous["inputs"])
            frozen = root / "frozen"
            config = prepare(inputs, frozen) if not frozen.exists() else json.loads((frozen / "probe-config.json").read_text())
            validate_upload(frozen, inputs)
            if not (frozen / "trace.json").exists():
                shutil.copy2(frozen / "probe-20261030.json", frozen / "trace.json")
            state.update({"inputs": str(inputs), "previous_job": previous["job"], "probe_config": config,
                "protected_vllm_sha256": previous["protected_vllm_sha256"],
                "frozen_runtime_sha256": {name: digest(repo / name) for name in SYNC_FILES
                                           if name.endswith((".py", ".sh", ".sbatch"))}})
            save_state(root, state)
            tests = sorted({name for name in SYNC_FILES if name.startswith("tests/")})
            with (root / "cpu-regression.log").open("w") as output:
                result = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests], cwd=repo,
                                        stdout=output, stderr=subprocess.STDOUT)
            (root / "cpu-regression.exit").write_text(str(result.returncode) + "\n")
            if result.returncode:
                raise RuntimeError("CPU regression failed; do not synchronize or submit")
            check_frozen_sources(repo, state)
            uploads = ("frozen/probe-config.json", "frozen/calibration-plan.json",
                       *(f"frozen/probe-{seed}.json" for seed in config["seeds"]))
            stage, home = protected_sync(repo, root, state, files=SYNC_FILES, prefix="long-conv-return", upload_files=uploads)
            environment = {"DP_OPT_LONG_CONV_UPLOAD": stage + "/upload",
                "DP_OPT_MOVE_FIXED_S": "1.7743364535272121", "DP_OPT_MOVE_BANDWIDTH_GIB_S": "11.64086282917902"}
            ledger = stage + "/submitted-job.txt"
            state.update({"phase": "submission_intent_do_not_duplicate", "remote_sync_stage": stage,
                          "submission_ledger": ledger, "frozen_environment": environment})
            save_state(root, state)
            output = checked_ssh(submission_command(environment, ledger))
            state.update({"job": int(output.split(";")[0]), "phase": "submitted_long_context_then_return_probes"})
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
    parser.add_argument("--previous-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.previous_root.resolve())


if __name__ == "__main__":
    main()
