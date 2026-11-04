import argparse
import json
import shlex
import time
from datetime import datetime
from pathlib import Path

from amortized_campaign import check_frozen_sources
from archive_trace_job import verify_raw
from long_conv_open_loop import CONV_RATES, analyze, digest, prepare, validate_upload
from rebalance_campaign import checked_ssh, protected_sync
from stress_campaign import archive_job, check_pause, poll_job, save_state
from verify_trace import verify
from workaware_robust_live_campaign import SYNC_FILES as PREVIOUS_SYNC_FILES


SYNC_FILES = tuple(dict.fromkeys((*PREVIOUS_SYNC_FILES,
    "experiments/dp_opt/long_conv_open_loop.py", "experiments/dp_opt/long_conv_open_loop_campaign.py",
    "experiments/dp_opt/long-conv-open-loop.sbatch", "tests/test_dp_opt_long_conv_open_loop.py",
    "tests/test_dp_opt_long_conv_open_loop_campaign.py", "experiments/dp_opt/docs/long-conv-open-loop-calibration.md")))


def parent_ready(state, report):
    return (state.get("GPU_experiment_complete") is True and state.get("state_last_observed") == "COMPLETED"
        and state.get("phase") == "robust_native_pairs_SHA_verified_analyzed"
        and report.get("job") == state.get("job")
        and report.get("status") == "complete_native_report_not_final_Poisson_research_goal")


def await_parent(root, parent, state):
    while True:
        check_pause(root)
        check_pause(parent)
        check_frozen_sources(Path(state["repo"]), state)
        previous = json.loads((parent / "campaign-state.json").read_text())
        report_path = parent / "report-followup-state.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        if "diagnosis_required" in previous["phase"] or "diagnosis_required" in report.get("status", ""):
            raise RuntimeError("Native failure requires diagnosis; do not allocate calibration GPUs")
        state.update({"parent_job": previous.get("job"), "last_parent_phase": previous["phase"],
            "last_parent_poll_at": previous.get("last_poll_at"),
            "parent_observed_at": datetime.now().astimezone().isoformat(), "SSH_queries_while_waiting": False})
        save_state(root, state)
        if parent_ready(previous, report):
            repo = Path(state["repo"])
            check_frozen_sources(repo, previous)
            raw = parent / f"live-trace-{previous['job']}"
            manifest_path = parent / f"raw-{previous['job']}.sha256"
            verify_raw(raw, manifest_path.read_text())
            pooled = json.loads((parent / "native-pooled-metrics.json").read_text())
            if (pooled["job"] != previous["job"]
                    or pooled["raw_manifest_sha256"] != digest(manifest_path)
                    or pooled["independent_verification_sha256"] != digest(parent / "independent-verification.json")):
                raise RuntimeError("Parent native pooled result differs from its verified raw archive")
            return previous
        print(json.dumps({"job": previous.get("job"), "parent_phase": previous["phase"],
                          "next": "wait_SHA_and_native_report_before_any_sync_or_submission"}), flush=True)
        time.sleep(300)


def submission_command(environment, ledger):
    exports = "; ".join(f"export {name}={shlex.quote(value)}" for name, value in sorted(environment.items()))
    return ("set -eu; test -z \"$(squeue -h -u \"$USER\")\"; "
        f"test ! -e {shlex.quote(ledger)}; {exports}; "
        "sbatch --parsable -o \"$HOME/run/kvcached-paracloud/logs/long-open-loop-%j.out\" "
        "-e \"$HOME/run/kvcached-paracloud/logs/long-open-loop-%j.err\" "
        "\"$HOME/kvcached/kvcached/experiments/dp_opt/long-conv-open-loop.sbatch\" | "
        f"tee {shlex.quote(ledger)}")


def finish(root, state):
    repo = Path(state["repo"])
    terminal = poll_job(root, state["job"], state)
    archive_job(root, state["job"], repo / "experiments/dp_opt", state)
    check_frozen_sources(repo, state)
    raw = root / f"live-trace-{state['job']}"
    if terminal != "COMPLETED" or (raw / "exit-status").read_text().strip() != "0":
        raise RuntimeError("Open-loop calibration failed; diagnose archived evidence before retry")
    validate_upload(raw / "frozen", Path(state["inputs"]))
    validation = {}
    for ordinal in range(len(CONV_RATES)):
        name = f"rate-{ordinal}"
        if digest(raw / name / "trace.json") != state["trace_sha256"][f"{name}.json"]:
            raise RuntimeError("Open-loop serving changed a frozen Poisson trace")
        validation[name] = verify(raw / name, Path(state["inputs"]) / "prompt-bank.json")
    (root / "independent-verification.json").write_text(json.dumps(validation, indent=2) + "\n")
    result = analyze(raw)
    (root / "calibration-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    state.update({"phase": "open_loop_calibration_SHA_verified_not_final_benefit",
        "GPU_calibration_complete": True, "candidate_case": result["candidate_case"],
        "next_action": "review_service_pressure_queue_then_separately_freeze_multi_policy_Poisson",
        "total_research_goal_completed": False})
    save_state(root, state)


def run(root, parent):
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    state_path = root / "campaign-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "repo": str(repo), "parent_root": str(parent), "phase": "awaiting_verified_native_report",
        "frozen_runtime_sha256": {name: digest(repo / name) for name in SYNC_FILES},
        "GPU_calibration_complete": False, "total_research_goal_completed": False,
        "publication_authorized": False}
    if state["repo"] != str(repo) or state["parent_root"] != str(parent):
        raise RuntimeError("Preserve existing open-loop campaign identity")
    if state.get("GPU_calibration_complete"):
        return
    if "diagnosis_required" in state["phase"]:
        raise RuntimeError("Diagnose existing failure before any retry")
    try:
        check_pause(root)
        check_frozen_sources(repo, state)
        if "job" not in state:
            if state["phase"] != "awaiting_verified_native_report":
                raise RuntimeError("Reconcile existing submission intent and ledger, never allocate twice")
            parent_state = json.loads((parent / "campaign-state.json").read_text())
            inputs = Path(parent_state["inputs"])
            config = (validate_upload(root / "frozen", inputs) if (root / "frozen").exists()
                      else prepare(inputs, root / "frozen"))
            validate_upload(root / "frozen", inputs)
            state.update({"inputs": str(inputs),
                "trace_sha256": config["trace_sha256"], "configuration": config})
            save_state(root, state)
            previous = await_parent(root, parent, state)
            state["protected_vllm_sha256"] = previous["protected_vllm_sha256"]
            save_state(root, state)
            stage, home = protected_sync(repo, root, state, files=SYNC_FILES, prefix="long-conv-open-loop",
                upload_files=("frozen/open-loop-config.json", *(f"frozen/rate-{ordinal}.json"
                               for ordinal in range(len(CONV_RATES)))))
            environment = {"DP_OPT_OPEN_LOOP_UPLOAD": stage + "/upload",
                "DP_OPT_OPEN_LOOP_FIT_JOB": str(previous["parent_job"]),
                "DP_OPT_OPEN_LOOP_FIT_SHA256": previous["parent_fit_sha256"],
                "DP_OPT_MOVE_FIXED_S": "1.7743364535272121", "DP_OPT_MOVE_BANDWIDTH_GIB_S": "11.64086282917902"}
            ledger = stage + "/submitted-job.txt"
            state.update({"phase": "submission_intent_do_not_duplicate", "remote_sync_stage": stage,
                "submission_ledger": ledger, "frozen_environment": environment})
            save_state(root, state)
            output = checked_ssh(submission_command(environment, ledger))
            state.update({"job": int(output.split(";")[0]), "phase": "submitted_open_loop_baseline_calibration"})
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
    parser.add_argument("--parent-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.parent_root.resolve())


if __name__ == "__main__":
    main()
