import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from archive_trace_job import verify_raw
from rebalance_campaign import checked_ssh, protected_sync
from stress_campaign import archive_job, check_pause, poll_job, save_state
from verify_roundtrip_phase import verify as verify_controls
from verify_trace import verify as verify_trace


SYNC_FILES = (
    "experiments/dp_opt/migration_roundtrip_control.py",
    "experiments/dp_opt/roundtrip_state_probe.py",
    "experiments/dp_opt/roundtrip-state-probe.sbatch",
    "experiments/dp_opt/verify_roundtrip_transaction.py",
    "experiments/dp_opt/verify_roundtrip_phase.py",
    "experiments/dp_opt/roundtrip_state_campaign.py",
    "experiments/dp_opt/docs/roundtrip-state-control.md",
    "tests/test_dp_opt_migration_roundtrip_control.py",
    "tests/test_dp_opt_roundtrip_transaction_verifier.py",
    "tests/test_dp_opt_roundtrip_phase_verifier.py",
    "tests/test_dp_opt_roundtrip_probe_entry.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_parent_inputs(repo, root, state, home, parent_job):
    files = {str(path.relative_to(root / "frozen")): digest(path)
             for path in sorted((root / "frozen").rglob("*")) if path.is_file()}
    source = home + f"/run/kvcached-paracloud/state/live-trace-{parent_job}/frozen"
    targets = {name: source + "/" + name for name in files}
    actual = checked_ssh("sha256sum " + " ".join(shlex.quote(target) for target in targets.values()))
    expected = [f"{files[name]}  {target}" for name, target in targets.items()]
    (root / "remote-parent-frozen-inputs.sha256").write_text(actual + "\n")
    if actual.splitlines() != expected:
        raise RuntimeError("Remote immutable parent inputs differ; do not reuse or install a new runtime")
    stage, home = protected_sync(repo, root, state, files=SYNC_FILES, prefix="roundtrip-state", upload_files=())
    script = ['set -eu; test -z "$(squeue -h -u "$USER")"']
    for name, expected_sha in files.items():
        original, target = shlex.quote(targets[name]), shlex.quote(stage + "/upload/frozen/" + name)
        script.extend([f'test "$(sha256sum {original} | cut -d\' \' -f1)" = {expected_sha}',
                       "mkdir -p " + shlex.quote(str(Path(stage + "/upload/frozen/" + name).parent)),
                       f"cp -p {original} {target}",
                       f'test "$(sha256sum {target} | cut -d\' \' -f1)" = {expected_sha}'])
    checked_ssh("; ".join(script))
    state["remote_parent_inputs_reused_after_SHA_verification"] = {"job": parent_job, "files": files}
    return stage, home


def prepare(root, parent, repo):
    previous = json.loads((parent / "campaign-state.json").read_text())
    numerics_parent = (previous.get("job") == 272931 and previous.get("state_last_observed") == "COMPLETED"
                       and previous.get("GPU_diagnostic_complete") is True)
    retry_parent = previous.get("job") == 273701 and previous.get("state_last_observed") == "FAILED"
    if not numerics_parent and not retry_parent:
        raise ValueError("Require terminal verified numerics or the explicitly diagnosed transport failure")
    raw = parent / f"live-trace-{previous['job']}"
    verify_raw(raw, (parent / f"raw-{previous['job']}.sha256").read_text())
    if numerics_parent:
        diagnostic = json.loads((parent / "independent-numerics-verification.json").read_text())
        if diagnostic != json.loads((raw / "batch-numerics-diagnostic.json").read_text()):
            raise ValueError("Parent independent diagnostic differs from immutable raw")
        divergence = diagnostic["baseline_serial_divergence_reproduced"]
    else:
        diagnosis = json.loads((parent / "transport-failure-diagnosis.json").read_text())
        error = json.loads((raw / "state-control/pilot/error.json").read_text())
        if (diagnosis.get("job") != previous["job"] or diagnosis.get("state") != "FAILED"
                or diagnosis.get("ready_for_transport_retry") is not True
                or diagnosis.get("raw_manifest_sha256") != digest(parent / f"raw-{previous['job']}.sha256")
                or diagnosis.get("strict_token_check_relaxed") is not False
                or error != {"type": "ValueError", "message": "Roundtrip requires exact device bytewise echo for both hops"}
                or diagnosis.get("error") != error):
            raise ValueError("Require immutable independent transport diagnosis before a sole retry")
        divergence = previous["parent_divergence_reproduced"]
    shutil.copytree(raw / "frozen", root / "frozen")
    if digest(root / "frozen/trace.json") != previous["original_trace_sha256"]:
        raise ValueError("Do not replace, shorten or resample the original control inputs")
    sources = {**previous["frozen_runtime_sha256"], **{name: digest(repo / name) for name in SYNC_FILES}}
    return {"repo": str(repo), "parent": str(parent), "parent_job": previous["job"],
            "phase": "state_controls_prepared", "frozen_runtime_sha256": sources,
            "protected_vllm_sha256": previous["protected_vllm_sha256"],
            "original_trace_sha256": previous["original_trace_sha256"],
            "is_algorithm_benefit": False, "publication_authorized": False,
            "original_native_failure_reclassified_as_success": False,
            "parent_divergence_reproduced": divergence,
            "transport_retry_parent": retry_parent}


def run(root, parent):
    repo = Path(__file__).resolve().parents[2]
    root.mkdir(parents=True, exist_ok=True)
    check_pause(root)
    path = root / "campaign-state.json"
    state = json.loads(path.read_text()) if path.exists() else prepare(root, parent, repo)
    if state["repo"] != str(repo) or state["parent"] != str(parent):
        raise ValueError("Preserve state-control campaign identity")
    if "diagnosis_required" in state["phase"]:
        raise RuntimeError("Diagnose the existing campaign; never duplicate its allocation")
    if state.get("GPU_controls_complete"):
        return
    save_state(root, state)
    try:
        if "job" not in state:
            if state["phase"] != "state_controls_prepared":
                raise RuntimeError("Reconcile the submission ledger before any new action")
            tests = [name for name in SYNC_FILES if name.startswith("tests/")]
            with (root / "cpu-tests.log").open("w") as output:
                result = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests], cwd=repo,
                                        stdout=output, stderr=subprocess.STDOUT)
            (root / "cpu-tests.exit").write_text(str(result.returncode) + "\n")
            if result.returncode:
                raise RuntimeError("State-control CPU regressions failed before deployment")
            home = checked_ssh('printf "%s" "$HOME"')
            previous = json.loads((parent / "campaign-state.json").read_text())
            expected = [f"{value}  {home}/kvcached/kvcached/{name}"
                        for name, value in previous["frozen_runtime_sha256"].items()]
            actual = checked_ssh("sha256sum " + " ".join(shlex.quote(line.split("  ", 1)[1])
                                                        for line in expected))
            (root / "parent-remote-source-check.txt").write_text(actual + "\n")
            if actual.splitlines() != expected:
                raise RuntimeError("Original runtime changed before independent controls")
            stage, home = stage_parent_inputs(repo, root, state, home, previous["job"])
            ledger = stage + "/submitted-job.txt"
            state.update({"phase": "submission_intent_do_not_duplicate", "remote_sync_stage": stage,
                          "submission_ledger": ledger})
            save_state(root, state)
            submitted = checked_ssh(
                f'set -eu; test -z "$(squeue -h -u "$USER")"; test ! -e {shlex.quote(ledger)}; '
                f'export DP_OPT_STATE_CONTROL_UPLOAD={shlex.quote(stage + "/upload")}; '
                'sbatch --parsable --export=ALL '
                '--output="$HOME/run/kvcached-paracloud/logs/live-trace-%j.out" '
                '--error="$HOME/run/kvcached-paracloud/logs/live-trace-%j.err" '
                '"$HOME/kvcached/kvcached/experiments/dp_opt/roundtrip-state-probe.sbatch" '
                f'> {shlex.quote(ledger)}; cat {shlex.quote(ledger)}')
            state.update({"job": int(submitted.split(";")[0]), "phase": "submitted_state_and_plain_controls"})
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
            raise RuntimeError("Control GPU stage failed; preserve and diagnose the terminal raw")
        bank = raw / "frozen/inputs/prompt-bank.json"
        controls = verify_controls(raw / "pilot", bank)
        baseline = verify_trace(raw / "plain-no-move", bank)
        if controls != json.loads((raw / "state-independent-verification.json").read_text()):
            raise RuntimeError("Compute and local independent control reconstructions differ")
        if baseline != json.loads((raw / "plain-independent-verification.json").read_text()):
            raise RuntimeError("Compute and local strict baseline reconstructions differ")
        if any(result["trace_sha256"] != state["original_trace_sha256"] for result in (controls, baseline)):
            raise RuntimeError("Control phases did not retain the original failed trace")
        for name in ("roundtrip_state_probe.py", "migration_roundtrip_control.py"):
            if digest(raw / "pilot" / (name + ".executed")) != digest(repo / "experiments/dp_opt" / name):
                raise RuntimeError("Controls executed an unfrozen source version")
        with (root / "independent-controls-verification.json").open("x") as output:
            output.write(json.dumps({"state_controls": controls, "plain_no_move": baseline,
                                     "is_algorithm_benefit": False}, indent=2) + "\n")
        state.update({"phase": "controls_SHA_and_full_tokens_verified_not_benefit",
                      "GPU_controls_complete": True, "is_algorithm_benefit": False})
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
