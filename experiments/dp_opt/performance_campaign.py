import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from amortized_campaign import check_frozen_sources
from archive_trace_job import verify_raw
from performance_numerics import digest, verify
from rebalance_campaign import checked_ssh, checked_ssh_script, protected_sync
from stress_campaign import archive_job, check_pause, poll_job, save_state
from verify_workaware_profile import same_payload, verify_profile


SYNC_FILES = ("experiments/dp_opt/performance_numerics.py", "experiments/dp_opt/performance_trace.py",
              "experiments/dp_opt/performance_public.py", "experiments/dp_opt/performance_campaign.py",
              "experiments/dp_opt/performance-native.sbatch", "tests/test_dp_opt_performance_numerics.py")
SEEDS = (2026103101, 2026103102)
POPULATIONS = (2, 3, 4)


def prepare(root, parent, repo):
    previous = json.loads((parent / "campaign-state.json").read_text())
    if previous["job"] != 274385:
        raise ValueError("Require the archived original public-source/default-head calibration")
    raw = parent / "live-trace-274385"
    verify_raw(raw, (parent / "raw-274385.sha256").read_text())
    fit = verify_profile(raw / "public-context-calibration")
    if not same_payload(fit, json.loads((raw / "public-context-fit.json").read_text())):
        raise ValueError("Default-head cost calibration does not independently reconstruct")
    source_parent = Path(__file__).resolve().parents[3] / "evidence/public-fp32-head-diagnostic-20261101"
    baseline = json.loads((source_parent / "campaign-state.json").read_text())
    check_frozen_sources(repo, baseline)
    shutil.copytree(raw / "frozen", root / "frozen")
    shutil.copytree(raw / "public-context-calibration", root / "reused-context-calibration")
    shutil.copy2(raw / "public-context-fit.json", root / "reused-context-fit.json")
    state = {"repo": str(repo), "parent": str(parent), "parent_job": 274385,
             "phase": "performance_prepared", "monitor_interval_s": 300,
             "user_authorized_greedy_observation_on": "2026-11-02",
             "performance_independent_of_debug": True, "quality_equivalence_unresolved": True,
             "default_BF16_head_and_sparse_graphs_unchanged": True,
             "protected_vllm_sha256": baseline["protected_vllm_sha256"],
             "frozen_runtime_sha256": {**baseline["frozen_runtime_sha256"],
                                      **{name: digest(repo / name) for name in SYNC_FILES}},
             "publication_authorized": False, "research_goal_complete": False,
             "native_seeds": list(SEEDS), "native_populations": list(POPULATIONS),
             "default_head_costs_independently_verified": True}
    return state


def preflight(root, repo, state):
    check_frozen_sources(repo, state)
    environment = {**os.environ, "PYTHONPATH": str(repo / "experiments/dp_opt")}
    tests = ["test_dp_opt_performance_numerics.py", "test_dp_opt_trace.py", "test_dp_opt_live_placement.py",
             "test_dp_opt_candidate_revalidation.py", "test_dp_opt_headroom_rebalance.py",
             "test_dp_opt_random_placement.py", "test_dp_opt_solver_efficiency_verifier.py"]
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", *["tests/" + name for name in tests]],
                            cwd=repo, env=environment, text=True, capture_output=True)
    (root / "performance-preflight-tests.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("Performance opt-in/invariant regression failed")
    with tempfile.TemporaryDirectory(prefix="performance-runtime-isolation-") as directory:
        isolated = Path(directory)
        for name in state["frozen_runtime_sha256"]:
            target = isolated / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / name, target)
        command = [sys.executable, "-I", "-c", "import sys;sys.path.insert(0," +
                   repr(str(isolated / "experiments/dp_opt")) +
                   ");import performance_trace;import performance_campaign;import performance_public;print('performance actual deployment imports passed')"]
        result = subprocess.run(command, cwd=isolated, text=True, capture_output=True)
        (root / "performance-isolated-imports.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError("Performance exact-copy import closure failed")
    subprocess.run(["bash", "-n", str(repo / "experiments/dp_opt/performance-native.sbatch")], check=True)
    state["performance_preflight_passed"] = True
    save_state(root, state)


def stage_inputs(root, repo, state):
    home = checked_ssh('printf "%s" "$HOME"')
    existing = {name: sha for name, sha in state["frozen_runtime_sha256"].items() if name not in SYNC_FILES}
    paths = [home + "/kvcached/kvcached/" + name for name in existing]
    actual = checked_ssh("sha256sum " + " ".join(map(shlex.quote, paths)))
    (root / "performance-parent-runtime-remote.sha256").write_text(actual + "\n")
    if actual.splitlines() != [f"{sha}  {path}" for sha, path in zip(existing.values(), paths)]:
        raise RuntimeError("Frozen parent runtime changed before performance source sync")
    stage, home = protected_sync(repo, root, state, files=SYNC_FILES, prefix="performance-parallel")
    remote_parent = home + "/run/kvcached-paracloud/state/live-trace-274385"
    commands = ["set -eu", 'test -z "$(squeue -h -u "$USER")"']
    sources = {}
    for directory, original in (("frozen", "frozen"), ("reused-context-calibration", "public-context-calibration")):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file():
                relative = str(path.relative_to(root / directory))
                sources[directory + "/" + relative] = (remote_parent + "/" + original + "/" + relative, digest(path))
    sources["reused-context-fit.json"] = (remote_parent + "/public-context-fit.json", digest(root / "reused-context-fit.json"))
    for name, (source, expected) in sources.items():
        target = stage + "/upload/" + name
        commands.extend([f'test "$(sha256sum {shlex.quote(source)} | cut -d\' \' -f1)" = {expected}',
                         "mkdir -p " + shlex.quote(str(Path(target).parent)),
                         "cp -p " + shlex.quote(source) + " " + shlex.quote(target),
                         f'test "$(sha256sum {shlex.quote(target)} | cut -d\' \' -f1)" = {expected}'])
    checked_ssh_script("\n".join(commands))
    (root / "performance-inputs-double-SHA.json").write_text(json.dumps(sources, indent=2) + "\n")
    state.update({"remote_sync_stage": stage, "remote_home": home, "phase": "performance_staged"})
    save_state(root, state)
    return stage, home


def run(root, parent, prepare_only=False, stage_only=False):
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    path = root / "campaign-state.json"
    state = json.loads(path.read_text()) if path.exists() else prepare(root, parent, repo)
    if state["repo"] != str(repo) or state["parent"] != str(parent):
        raise ValueError("Preserve immutable performance campaign identity")
    save_state(root, state)
    check_pause(root)
    check_frozen_sources(repo, state)
    if prepare_only:
        preflight(root, repo, state)
        return
    if "diagnosis_required" in state["phase"]:
        raise RuntimeError("Diagnose the recorded performance job, never duplicate it")
    try:
        if "job" not in state:
            if state["phase"] == "performance_prepared":
                preflight(root, repo, state)
                stage_inputs(root, repo, state)
            if stage_only:
                return
            if state["phase"] != "performance_staged":
                raise RuntimeError("Reconcile performance submission intent before retrying")
            stage = state["remote_sync_stage"]
            ledger = stage + "/submitted-performance-job.txt"
            state.update({"phase": "performance_submission_intent_no_duplicate", "submission_ledger": ledger})
            save_state(root, state)
            submitted = checked_ssh(f'set -eu; test -z "$(squeue -h -u "$USER")"; test ! -e {shlex.quote(ledger)}; '
                f'export DP_OPT_PERFORMANCE_UPLOAD={shlex.quote(stage + "/upload")}; '
                'sbatch --parsable --export=ALL --output="$HOME/run/kvcached-paracloud/logs/live-trace-%j.out" '
                '--error="$HOME/run/kvcached-paracloud/logs/live-trace-%j.err" '
                f'"$HOME/kvcached/kvcached/experiments/dp_opt/performance-native.sbatch" > {shlex.quote(ledger)}; cat {shlex.quote(ledger)}')
            state.update({"job": int(submitted.split(";")[0]), "phase": "monitoring_public_native_performance"})
            (root / "submit.txt").write_text(submitted + "\n")
            save_state(root, state)
            remote = state["remote_home"] + f"/run/kvcached-paracloud/state/live-trace-{state['job']}"
            watcher = checked_ssh(f'mkdir -p {shlex.quote(remote)}; nohup bash "$HOME/kvcached/kvcached/experiments/dp_opt/watch-trace-remote.sh" '
                                  f'{state["job"]} >> {shlex.quote(remote + "/remote-query.log")} 2>&1 < /dev/null & printf "%s\\n" "$!"')
            state["remote_watcher_pid"] = int(watcher)
            save_state(root, state)
        terminal = poll_job(root, state["job"], state)
        archive_job(root, state["job"], repo / "experiments/dp_opt", state)
        raw = root / f"live-trace-{state['job']}"
        results = {}
        for seed in SEEDS:
            for population in POPULATIONS:
                name = f"native-{seed}-N{population}"
                if (raw / name / "exit-status").exists():
                    results[name] = verify(raw / name, raw / "frozen/inputs/prompt-bank.json")
        (root / "independent-native-performance.json").write_text(json.dumps(results, indent=2) + "\n")
        if terminal != "COMPLETED" or len(results) != len(SEEDS) * len(POPULATIONS):
            raise RuntimeError("Native performance incomplete; preserve valid cases and diagnose non-greedy failure")
        state.update({"phase": "native_performance_SHA_verified_rate_calibration_ready", "native_performance_complete": True,
                      "next_action": "rate_calibration_then_seven_policy_Poisson_and_heldout_without_debug_dependency"})
        save_state(root, state)
    except Exception as error:
        state.update({"phase": "diagnosis_required_performance_not_blocked_by_greedy", "error": repr(error)})
        save_state(root, state)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    run(args.root.resolve(), args.parent.resolve(), args.prepare_only, args.stage_only)


if __name__ == "__main__":
    main()
