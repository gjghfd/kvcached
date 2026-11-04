import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from archive_trace_job import verify_raw
from batch_numerics_probe import EXPECTED_FAILURES, analyze
from rebalance_campaign import checked_ssh, protected_sync
from stress_campaign import archive_job, poll_job, save_state


SYNC_FILES = (
    "experiments/dp_opt/batch_numerics_probe.py",
    "experiments/dp_opt/batch_numerics_campaign.py",
    "experiments/dp_opt/batch-numerics-probe.sbatch",
    "experiments/dp_opt/docs/batch-numerics-diagnostic.md",
    "tests/test_dp_opt_batch_numerics_probe.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root, parent, repo):
    previous = json.loads((parent / "campaign-state.json").read_text())
    if previous["state_last_observed"] != "FAILED" or previous["job"] != 272106:
        raise ValueError("Require the preserved diagnosed C1 failure, not a successful benefit parent")
    raw = parent / f"live-trace-{previous['job']}"
    verify_raw(raw, (parent / f"raw-{previous['job']}.sha256").read_text())
    error = json.loads((raw / "native-2026103101-N2/error.json").read_text())
    if error["type"] != "RuntimeError" or error["message"] not in EXPECTED_FAILURES:
        raise ValueError("Diagnose any unrelated parent failure before numerics probing")
    shutil.copytree(raw / "frozen", root / "frozen")
    if digest(root / "frozen/trace.json") != digest(raw / "native-2026103101-N2/trace.json"):
        raise ValueError("Numerics probe must retain the exact failed native trace")
    sources = {**previous["frozen_runtime_sha256"],
               **{name: digest(repo / name) for name in SYNC_FILES}}
    return {"repo": str(repo), "parent": str(parent), "parent_job": previous["job"],
            "phase": "numerics_probe_prepared", "inputs": str(root / "frozen/inputs"),
            "protected_vllm_sha256": previous["protected_vllm_sha256"],
            "frozen_runtime_sha256": sources, "original_trace_sha256": digest(root / "frozen/trace.json"),
            "is_algorithm_benefit": False, "publication_authorized": False,
            "parent_failure_reclassified_as_success": False}


def run(root, parent):
    repo = Path(__file__).resolve().parents[2]
    root.mkdir(parents=True, exist_ok=True)
    path = root / "campaign-state.json"
    state = json.loads(path.read_text()) if path.exists() else prepare(root, parent, repo)
    if state["repo"] != str(repo) or state["parent"] != str(parent):
        raise ValueError("Preserve numerics campaign identity")
    if "diagnosis_required" in state["phase"]:
        raise RuntimeError("Diagnose existing numerics state; do not duplicate submission")
    if state.get("GPU_diagnostic_complete"):
        return
    save_state(root, state)
    try:
        if "job" not in state:
            if state["phase"] != "numerics_probe_prepared":
                raise RuntimeError("Reconcile the submission ledger; never submit a duplicate")
            with (root / "cpu-tests.log").open("w") as output:
                result = subprocess.run([sys.executable, "-m", "pytest", "-q",
                    "tests/test_dp_opt_batch_numerics_probe.py", "tests/test_dp_opt_trace.py",
                    "tests/test_dp_opt_public_quality_recovery.py"], cwd=repo,
                    stdout=output, stderr=subprocess.STDOUT)
            (root / "cpu-tests.exit").write_text(str(result.returncode) + "\n")
            if result.returncode:
                raise RuntimeError("Numerics CPU tests failed before sync/submission")
            deployed = json.loads((parent / "protected-sync-manifest.json").read_text())
            actual = checked_ssh("sha256sum " + " ".join(shlex.quote(item["target"]) for item in deployed))
            (root / "parent-remote-source-check.txt").write_text(actual + "\n")
            if actual.splitlines() != [f'{item["sha256"]}  {item["target"]}' for item in deployed]:
                raise RuntimeError("Existing remote parent sources changed before diagnostic")
            uploads = tuple(str(path.relative_to(root)) for path in sorted((root / "frozen").rglob("*"))
                            if path.is_file() and path != root / "frozen/trace.json")
            stage, home = protected_sync(repo, root, state, files=SYNC_FILES,
                                          prefix="batch-numerics", upload_files=uploads)
            ledger = stage + "/submitted-job.txt"
            state.update({"phase": "submission_intent_do_not_duplicate", "remote_sync_stage": stage,
                          "submission_ledger": ledger})
            save_state(root, state)
            command = (f'set -eu; test -z "$(squeue -h -u "$USER")"; test ! -e {shlex.quote(ledger)}; '
                f'export DP_OPT_NUMERICS_UPLOAD={shlex.quote(stage + "/upload")}; '
                'sbatch --parsable --export=ALL '
                '--output="$HOME/run/kvcached-paracloud/logs/live-trace-%j.out" '
                '--error="$HOME/run/kvcached-paracloud/logs/live-trace-%j.err" '
                '"$HOME/kvcached/kvcached/experiments/dp_opt/batch-numerics-probe.sbatch" | '
                f'tee {shlex.quote(ledger)}')
            submitted = checked_ssh(command)
            state.update({"job": int(submitted.split(";")[0]), "phase": "submitted_no_move_numerics_probe"})
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
        raw = root / f"live-trace-{state['job']}"
        if terminal != "COMPLETED" or (raw / "exit-status").read_text().strip() != "0":
            raise RuntimeError("Numerics GPU probe failed; preserve terminal archive for diagnosis")
        result = analyze(raw)
        if result != json.loads((raw / "batch-numerics-diagnostic.json").read_text()):
            raise RuntimeError("Independent common-prefix numerics reconstruction differs")
        if digest(raw / "batch-numerics-probe-source.py") != digest(repo / "experiments/dp_opt/batch_numerics_probe.py"):
            raise RuntimeError("Diagnostic did not execute the frozen probe source")
        with (root / "independent-numerics-verification.json").open("x") as output:
            output.write(json.dumps(result, indent=2) + "\n")
        state.update({"phase": "numerics_SHA_and_common_prefix_verified_not_benefit",
                      "GPU_diagnostic_complete": True, "is_algorithm_benefit": False})
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
