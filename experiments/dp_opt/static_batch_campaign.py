import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from amortized_campaign import check_frozen_sources
from analyze_native_divergence import analyze as analyze_failure
from archive_trace_job import verify_raw
from rebalance_campaign import checked_ssh, protected_sync
from static_batch_probe import analyze
from stress_campaign import archive_job, check_pause, poll_job, save_state


SYNC_FILES = (
    "experiments/dp_opt/static_batch_probe.py",
    "experiments/dp_opt/static_batch_campaign.py",
    "experiments/dp_opt/static-batch-probe.sbatch",
    "experiments/dp_opt/analyze_native_divergence.py",
    "experiments/dp_opt/batch_numerics_probe.py",
    "experiments/dp_opt/analyze_numerics_contrast.py",
    "experiments/dp_opt/docs/static-batch-diagnostic.md",
    "tests/test_dp_opt_static_batch_probe.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(root, repo, state):
    check_frozen_sources(repo, state)
    environment = {**os.environ, "PYTHONPATH": str(repo / "experiments/dp_opt")}
    with (root / "cpu-tests.log").open("w") as output:
        result = subprocess.run([sys.executable, "-m", "pytest", "-q",
                                 "tests/test_dp_opt_static_batch_probe.py", "tests/test_dp_opt_batch_numerics_probe.py",
                                 "tests/test_dp_opt_numerics_contrast.py", "tests/test_dp_opt_trace.py",
                                 "tests/test_dp_opt_live_placement.py", "tests/test_dp_opt_candidate_revalidation.py"],
                                cwd=repo, env=environment, stdout=output, stderr=subprocess.STDOUT)
    (root / "cpu-tests.exit").write_text(str(result.returncode) + "\n")
    if result.returncode:
        raise RuntimeError("Static batch CPU deployment tests failed before installation")
    with tempfile.TemporaryDirectory(prefix="static-batch-runtime-isolation-") as directory:
        isolated = Path(directory)
        for name in state["frozen_runtime_sha256"]:
            target = isolated / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / name, target)
        command = [sys.executable, "-I", "-c", "import sys;sys.path.insert(0," +
                   repr(str(isolated / "experiments/dp_opt")) +
                   ");import static_batch_probe;import static_batch_campaign;print('actual deployment imports passed')"]
        result = subprocess.run(command, cwd=isolated, text=True, capture_output=True)
        (root / "isolated-runtime-imports.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError("Actual frozen deployment import closure failed")
    check_frozen_sources(repo, state)
    state["CPU_deployment_tests_passed"] = True
    state["actual_runtime_isolated_imports_passed"] = True
    save_state(root, state)


def prepare(root, parent, repo):
    previous = json.loads((parent / "campaign-state.json").read_text())
    if previous.get("state_last_observed") != "FAILED" or previous.get("job") != 274385:
        raise ValueError("Require the explicitly diagnosed full native failure")
    raw = parent / f'live-trace-{previous["job"]}'
    verify_raw(raw, (parent / f'raw-{previous["job"]}.sha256').read_text())
    diagnosis = analyze_failure(raw / "native-2026103101-N2")
    saved = json.loads((parent.parent / "independent-native-divergence-20261101.json").read_text())
    if saved != diagnosis:
        raise ValueError("Full parent diagnostic does not independently reproduce")
    check_frozen_sources(repo, previous)
    shutil.copytree(raw / "frozen", root / "frozen")
    if digest(root / "frozen/trace.json") != diagnosis["trace_sha256"]:
        raise ValueError("Do not change the original native trace")
    return {"repo": str(repo), "parent": str(parent), "parent_job": previous["job"],
            "phase": "static_batch_prepared", "original_trace_sha256": diagnosis["trace_sha256"],
            "frozen_runtime_sha256": {**previous["frozen_runtime_sha256"],
                                      **{name: digest(repo / name) for name in SYNC_FILES}},
            "protected_vllm_sha256": previous["protected_vllm_sha256"],
            "monitor_interval_s": 300, "is_algorithm_benefit": False,
            "research_goal_complete": False, "publication_authorized": False,
            "parent_failure_reclassified_as_success": False}


def stage_inputs(repo, root, state):
    home = checked_ssh('printf "%s" "$HOME"')
    deployed = state["frozen_runtime_sha256"]
    existing = {name: sha for name, sha in deployed.items() if name not in SYNC_FILES}
    expected = [f'{sha}  {home}/kvcached/kvcached/{name}' for name, sha in existing.items()]
    actual = checked_ssh("sha256sum " + " ".join(shlex.quote(line.split("  ", 1)[1]) for line in expected))
    (root / "parent-runtime-remote.sha256").write_text(actual + "\n")
    if actual.splitlines() != expected:
        raise RuntimeError("Installed parent runtime differs; do not deploy diagnostic")
    files = {str(path.relative_to(root)): digest(path)
             for path in sorted((root / "frozen/inputs").rglob("*")) if path.is_file()}
    if not files:
        raise ValueError("Require original immutable prompt-bank inputs")
    source = home + f'/run/kvcached-paracloud/state/live-trace-{state["parent_job"]}/'
    expected = [f"{sha}  {source}{name}" for name, sha in files.items()]
    actual = checked_ssh("sha256sum " + " ".join(shlex.quote(source + name) for name in files))
    (root / "parent-inputs-remote.sha256").write_text(actual + "\n")
    if actual.splitlines() != expected:
        raise RuntimeError("Original remote prompt bank changed")
    stage, home = protected_sync(repo, root, state, files=SYNC_FILES, prefix="static-batch")
    script = ['set -eu; test -z "$(squeue -h -u "$USER")"']
    for name, sha in files.items():
        original, target = shlex.quote(source + name), shlex.quote(stage + "/upload/" + name)
        script.extend([f'test "$(sha256sum {original} | cut -d\' \' -f1)" = {sha}',
                       "mkdir -p " + shlex.quote(str(Path(stage + "/upload/" + name).parent)),
                       f"cp -p {original} {target}",
                       f'test "$(sha256sum {target} | cut -d\' \' -f1)" = {sha}'])
    script.append(f'test "$(sha256sum {shlex.quote(stage + "/upload/frozen/trace.json")} | cut -d\' \' -f1)" = {state["original_trace_sha256"]}')
    checked_ssh("; ".join(script))
    state["input_reuse_double_SHA"] = files
    return stage, home


def run(root, parent):
    repo = Path(__file__).resolve().parents[2]
    root.mkdir(parents=True, exist_ok=True)
    path = root / "campaign-state.json"
    state = json.loads(path.read_text()) if path.exists() else prepare(root, parent, repo)
    if state["repo"] != str(repo) or state["parent"] != str(parent):
        raise ValueError("Preserve campaign identity")
    if "diagnosis_required" in state["phase"]:
        raise RuntimeError("Diagnose existing campaign; never submit duplicate allocations")
    if state.get("GPU_diagnostic_complete"):
        return
    save_state(root, state)
    try:
        check_pause(root)
        check_frozen_sources(repo, state)
        if "job" not in state:
            if state["phase"] != "static_batch_prepared":
                raise RuntimeError("Reconcile submission intent/ledger before recovery")
            preflight(root, repo, state)
            stage, home = stage_inputs(repo, root, state)
            ledger = stage + "/submitted-job.txt"
            state.update({"phase": "submission_intent_do_not_duplicate", "remote_sync_stage": stage,
                          "submission_ledger": ledger})
            save_state(root, state)
            submitted = checked_ssh(
                f'set -eu; test -z "$(squeue -h -u "$USER")"; test ! -e {shlex.quote(ledger)}; '
                f'export DP_OPT_STATIC_BATCH_UPLOAD={shlex.quote(stage + "/upload")}; '
                'sbatch --parsable --export=ALL '
                '--output="$HOME/run/kvcached-paracloud/logs/live-trace-%j.out" '
                '--error="$HOME/run/kvcached-paracloud/logs/live-trace-%j.err" '
                f'"$HOME/kvcached/kvcached/experiments/dp_opt/static-batch-probe.sbatch" > {shlex.quote(ledger)}; '
                f'cat {shlex.quote(ledger)}')
            state.update({"job": int(submitted.split(";")[0]), "phase": "monitoring_static_no_copy_diagnostic"})
            (root / "submit.txt").write_text(submitted + "\n")
            save_state(root, state)
            remote = home + f'/run/kvcached-paracloud/state/live-trace-{state["job"]}'
            watcher = checked_ssh(f'''set -eu
mkdir -p {shlex.quote(remote)}
nohup bash "$HOME/kvcached/kvcached/experiments/dp_opt/watch-trace-remote.sh" {state["job"]} \\
    >> {shlex.quote(remote + '/remote-query.log')} 2>&1 < /dev/null &
printf '%s\\n' "$!"''')
            state["remote_watcher_pid"] = int(watcher)
            save_state(root, state)
        terminal = poll_job(root, state["job"], state)
        archive_job(root, state["job"], repo / "experiments/dp_opt", state)
        check_frozen_sources(repo, state)
        raw = root / f'live-trace-{state["job"]}'
        if terminal != "COMPLETED" or (raw / "exit-status").read_text().strip() != "0":
            raise RuntimeError("Static GPU diagnostic failed; inspect the full terminal SHA archive")
        result = analyze(raw)
        for name in ("static-batch-diagnostic.json", "independent-static-batch-verification.json"):
            if result != json.loads((raw / name).read_text()):
                raise RuntimeError("Independent common-prefix diagnostic differs")
        if (digest(raw / "static-batch-probe-source.py") != digest(repo / "experiments/dp_opt/static_batch_probe.py")
                or digest(raw / "trace.json") != state["original_trace_sha256"]):
            raise RuntimeError("Executed source or original trace differs")
        with (root / "independent-static-batch-verification.json").open("x") as output:
            output.write(json.dumps(result, indent=2) + "\n")
        state.update({"phase": "static_batch_SHA_and_common_prefix_verified_not_benefit",
                      "GPU_diagnostic_complete": True,
                      "no_copy_divergence_reproduced": result["no_copy_divergence_reproduced"],
                      "next_action": "inspect_first_shared_prefix_and_execution_shape_before_any_new_benefit_trial"})
        save_state(root, state)
    except Exception as error:
        state.update({"phase": "diagnosis_required_no_automatic_duplicate_submission", "error": repr(error)})
        save_state(root, state)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.parent.resolve())


if __name__ == "__main__":
    main()
