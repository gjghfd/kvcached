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
from analyze_numerics_contrast import candidate_contrast
from archive_trace_job import verify_raw
from fp32_head_probe import analyze
from rebalance_campaign import checked_ssh, protected_sync
from static_batch_campaign import digest
from static_batch_probe import analyze as analyze_parent
from stress_campaign import archive_job, check_pause, poll_job, save_state


SYNC_FILES = ("experiments/dp_opt/fp32_head_probe.py", "experiments/dp_opt/fp32_head_worker.py",
              "experiments/dp_opt/fp32_head_campaign.py", "experiments/dp_opt/fp32-head-probe.sbatch",
              "experiments/dp_opt/docs/fp32-head-diagnostic.md", "tests/test_dp_opt_fp32_head_probe.py")


def prepare(root, parent, repo):
    previous = json.loads((parent / "campaign-state.json").read_text())
    if (previous.get("job") != 276017 or not previous.get("GPU_diagnostic_complete")
            or not previous.get("no_copy_divergence_reproduced")):
        raise ValueError("Require the archived zero-copy divergence before the precision control")
    raw = parent / f"live-trace-{previous['job']}"
    verify_raw(raw, (parent / f"raw-{previous['job']}.sha256").read_text())
    if analyze_parent(raw) != json.loads((parent / "independent-static-batch-verification.json").read_text()):
        raise ValueError("Parent full-token diagnosis does not reproduce")
    check_frozen_sources(repo, previous)
    shutil.copytree(raw / "frozen", root / "frozen")
    return {"repo": str(repo), "parent": str(parent), "parent_job": previous["job"],
            "phase": "fp32_head_prepared", "original_trace_sha256": digest(root / "frozen/trace.json"),
            "frozen_runtime_sha256": {**previous["frozen_runtime_sha256"],
                                      **{name: digest(repo / name) for name in SYNC_FILES}},
            "protected_vllm_sha256": previous["protected_vllm_sha256"],
            "monitor_interval_s": 300, "is_algorithm_benefit": False,
            "research_goal_complete": False, "publication_authorized": False,
            "head_setting_change_disclosed": True, "old_default_head_cost_reuse_authorized": False}


def preflight(root, repo, state):
    check_frozen_sources(repo, state)
    environment = {**os.environ, "PYTHONPATH": str(repo / "experiments/dp_opt")}
    tests = ["test_dp_opt_fp32_head_probe.py", "test_dp_opt_static_batch_probe.py", "test_dp_opt_batch_numerics_probe.py",
             "test_dp_opt_numerics_contrast.py", "test_dp_opt_trace.py", "test_dp_opt_live_placement.py",
             "test_dp_opt_candidate_revalidation.py", "test_dp_opt_headroom_rebalance.py", "test_dp_opt_random_placement.py"]
    with (root / "cpu-tests.log").open("w") as output:
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", *["tests/" + name for name in tests]],
                                cwd=repo, env=environment, stdout=output, stderr=subprocess.STDOUT)
    (root / "cpu-tests.exit").write_text(str(result.returncode) + "\n")
    if result.returncode:
        raise RuntimeError("FP32 head deployment CPU tests failed")
    with tempfile.TemporaryDirectory(prefix="fp32-head-runtime-isolation-") as directory:
        isolated = Path(directory)
        for name in state["frozen_runtime_sha256"]:
            target = isolated / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / name, target)
        command = [sys.executable, "-I", "-c", "import sys;sys.path.insert(0," +
                   repr(str(isolated / "experiments/dp_opt")) +
                   ");import fp32_head_probe;import fp32_head_campaign;print('actual FP32 deployment imports passed')"]
        result = subprocess.run(command, cwd=isolated, text=True, capture_output=True)
        (root / "isolated-runtime-imports.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError("FP32 head actual deployment import closure failed")
    state["CPU_deployment_tests_passed"] = True
    state["actual_runtime_isolated_imports_passed"] = True
    save_state(root, state)


def stage_inputs(repo, root, state):
    home = checked_ssh('printf "%s" "$HOME"')
    existing = {name: sha for name, sha in state["frozen_runtime_sha256"].items() if name not in SYNC_FILES}
    paths = [home + "/kvcached/kvcached/" + name for name in existing]
    actual = checked_ssh("sha256sum " + " ".join(map(shlex.quote, paths)))
    (root / "parent-runtime-remote.sha256").write_text(actual + "\n")
    if actual.splitlines() != [f"{sha}  {path}" for sha, path in zip(existing.values(), paths)]:
        raise RuntimeError("Installed frozen parent runtime differs")
    files = {str(path.relative_to(root)): digest(path)
             for path in sorted((root / "frozen/inputs").rglob("*")) if path.is_file()}
    if not files:
        raise ValueError("Original public-source immutable inputs are missing")
    source = home + f"/run/kvcached-paracloud/state/live-trace-{state['parent_job']}/"
    actual = checked_ssh("sha256sum " + " ".join(shlex.quote(source + name) for name in files))
    (root / "parent-inputs-remote.sha256").write_text(actual + "\n")
    if actual.splitlines() != [f"{sha}  {source}{name}" for name, sha in files.items()]:
        raise RuntimeError("Original prompt bank SHA differs")
    stage, home = protected_sync(repo, root, state, files=SYNC_FILES, prefix="fp32-head")
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
    state_path = root / "campaign-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else prepare(root, parent, repo)
    if state["repo"] != str(repo) or state["parent"] != str(parent):
        raise ValueError("Preserve precision campaign identity")
    if state.get("GPU_diagnostic_complete"):
        return
    if "diagnosis_required" in state["phase"]:
        raise RuntimeError("Diagnose the existing allocation before any resubmission")
    save_state(root, state)
    try:
        check_pause(root)
        check_frozen_sources(repo, state)
        if "job" not in state:
            if state["phase"] != "fp32_head_prepared":
                raise RuntimeError("Reconcile submission intent; do not duplicate an allocation")
            preflight(root, repo, state)
            stage, home = stage_inputs(repo, root, state)
            ledger = stage + "/submitted-job.txt"
            state.update({"phase": "submission_intent_do_not_duplicate", "remote_sync_stage": stage,
                          "submission_ledger": ledger})
            save_state(root, state)
            submitted = checked_ssh(
                f'set -eu; test -z "$(squeue -h -u "$USER")"; test ! -e {shlex.quote(ledger)}; '
                f'export DP_OPT_FP32_HEAD_UPLOAD={shlex.quote(stage + "/upload")}; '
                'sbatch --parsable --export=ALL '
                '--output="$HOME/run/kvcached-paracloud/logs/live-trace-%j.out" '
                '--error="$HOME/run/kvcached-paracloud/logs/live-trace-%j.err" '
                f'"$HOME/kvcached/kvcached/experiments/dp_opt/fp32-head-probe.sbatch" > {shlex.quote(ledger)}; '
                f'cat {shlex.quote(ledger)}')
            state.update({"job": int(submitted.split(";")[0]), "phase": "monitoring_fp32_head_correctness_control"})
            (root / "submit.txt").write_text(submitted + "\n")
            save_state(root, state)
            remote = home + f"/run/kvcached-paracloud/state/live-trace-{state['job']}"
            watcher = checked_ssh(f'''set -eu
mkdir -p {shlex.quote(remote)}
nohup bash "$HOME/kvcached/kvcached/experiments/dp_opt/watch-trace-remote.sh" {state['job']} \\
    >> {shlex.quote(remote + '/remote-query.log')} 2>&1 < /dev/null &
printf '%s\\n' "$!"''')
            state["remote_watcher_pid"] = int(watcher)
            save_state(root, state)
        terminal = poll_job(root, state["job"], state)
        archive_job(root, state["job"], repo / "experiments/dp_opt", state)
        check_frozen_sources(repo, state)
        raw = root / f"live-trace-{state['job']}"
        if (raw / "static/fp32-head-diagnostic.json").exists():
            result = analyze(raw / "static", "static")
            if (result != json.loads((raw / "static/fp32-head-diagnostic.json").read_text())
                    or digest(raw / "static/trace.json") != state["original_trace_sha256"]
                    or digest(raw / "static/fp32_head_probe.py") != digest(repo / "experiments/dp_opt/fp32_head_probe.py")
                    or digest(raw / "static/fp32_head_worker.py") != digest(repo / "experiments/dp_opt/fp32_head_worker.py")
                    or result["precision"]["logits_processor_sha256"] != digest(raw / "installed-numerics-source/logits_processor.py")):
                raise RuntimeError("Independent FP32 static reconstruction differs")
            (root / "independent-static-verification.json").write_text(json.dumps(result, indent=2) + "\n")
            old = json.loads((parent / f"live-trace-{state['parent_job']}/references.json").read_text())
            new = json.loads((raw / "static/references.json").read_text())
            old_by_id = {record["trace_request_id"]: record for record in old}
            comparison = [candidate_contrast({**record, "request_id": record["trace_request_id"]},
                                            old_by_id[record["trace_request_id"]]) for record in new]
            (root / "old-new-serial-numerical-observation.json").write_text(json.dumps({
                "comparisons": comparison, "used_for_new_configuration_acceptance": False,
                "is_algorithm_benefit": False}, indent=2) + "\n")
            state["static_strict_passed"] = result["strict_correctness_passed"]
            save_state(root, state)
        if terminal != "COMPLETED" or (raw / "exit-status").read_text().strip() != "0":
            raise RuntimeError("Precision control failed; preserve full diagnostics and stop for root-cause analysis")
        for stage in ("static", "migration"):
            result = analyze(raw / stage, stage)
            if (not result["strict_correctness_passed"]
                    or result != json.loads((raw / stage / "independent-fp32-head-verification.json").read_text())
                    or digest(raw / stage / "trace.json") != state["original_trace_sha256"]
                    or digest(raw / stage / "fp32_head_probe.py") != digest(repo / "experiments/dp_opt/fp32_head_probe.py")
                    or digest(raw / stage / "fp32_head_worker.py") != digest(repo / "experiments/dp_opt/fp32_head_worker.py")
                    or result["precision"]["logits_processor_sha256"] != digest(raw / "installed-numerics-source/logits_processor.py")):
                raise RuntimeError("Full independent precision-stage/source verification failed")
            (root / f"independent-{stage}-verification.json").write_text(json.dumps(result, indent=2) + "\n")
        state.update({"phase": "FP32_head_static_and_migration_strict_verified_not_benefit",
                      "GPU_diagnostic_complete": True, "migration_strict_passed": True,
                      "next_action": "revalidate_FP32_head_costs_before_public_workaware_trace_chain"})
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
