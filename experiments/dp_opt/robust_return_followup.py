import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from archive_trace_job import verify_raw
from diagnose_return_gates import analyze, digest
from stress_campaign import check_pause


def ready(state):
    if state.get("phase") == "diagnosis_required_no_automatic_duplicate_submission":
        raise RuntimeError("GPU campaign requires failure diagnosis; do not start candidate replays")
    return (state.get("GPU_experiment_complete") is True and state.get("state_last_observed") == "COMPLETED"
            and state.get("phase") == "long_Conv_probe_SHA_verified_analyzed")


def check_sources(root, repo):
    manifest = root / "robust-candidate-source.sha256"
    entries = manifest.read_text().splitlines()
    if len(entries) != 4:
        raise RuntimeError("Require the four frozen candidate runtime/test files")
    for line in entries:
        expected, name = line.split("  ", 1)
        path = (repo / name).resolve()
        if repo.resolve() not in path.parents or digest(path) != expected:
            raise RuntimeError("Candidate source changed before independent follow-up")


def replay_completed(root, repo, state):
    check_pause(root)
    check_sources(root, repo)
    if not ready(state):
        raise RuntimeError("Candidate replays require complete SHA-verified GPU campaign")
    raw = root / f"live-trace-{state['job']}"
    manifest = root / f"raw-{state['job']}.sha256"
    verify_raw(raw, manifest.read_text())
    if (raw / "exit-status").read_text().strip() != "0":
        raise RuntimeError("GPU exit marker is not successful")
    verification_path = root / "independent-verification.json"
    verification = json.loads(verification_path.read_text())
    expected = {f"probe-{seed}" for seed in state["probe_config"]["seeds"]}
    if (set(verification.get("probes", {})) != expected
            or verification.get("research_goal_complete") is not False
            or verification.get("calibration", {}).get("source_verification") != "raw_verified"):
        raise RuntimeError("Independent GPU verification lacks declared probe/calibration coverage")
    result = {"job": state["job"], "raw_manifest_sha256": digest(manifest),
              "GPU_verification_sha256": digest(verification_path),
              "candidate_source_manifest_sha256": digest(root / "robust-candidate-source.sha256"),
              "scope": "independent_counterfactual_replays_not_new_GPU_policy_results", "probes": {},
              "runtime_changed": False, "new_allocation_submitted": False, "research_goal_complete": False}
    for name in sorted(expected):
        check_pause(root)
        check_sources(root, repo)
        result["probes"][name] = analyze(raw / name / "r0-kv_only_workaware", release_window_s=120,
                                         replay_max_evaluations=1500, robust_ilp_budget_s=8)
    check_sources(root, repo)
    return result


def write_status(root, status):
    target = root / "robust-followup-status.json"
    temporary = root / f".robust-followup-status-{os.getpid()}.json"
    temporary.write_text(json.dumps({**status, "observed_at": datetime.now().astimezone().isoformat(),
                                    "pid": os.getpid()}, indent=2) + "\n")
    temporary.replace(target)


def run(root, repo):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    lock = root / ".robust-followup.lock"
    lock.mkdir()
    try:
        while True:
            check_pause(root)
            check_sources(root, repo)
            state = json.loads((root / "campaign-state.json").read_text())
            if ready(state):
                result = replay_completed(root, repo, state)
                target = root / "robust-followup-comparison.json"
                temporary = root / ".robust-followup-comparison.json"
                temporary.write_text(json.dumps(result, indent=2) + "\n")
                temporary.replace(target)
                write_status(root, {"phase": "complete_independent_candidate_replay_not_GPU_benefit",
                                    "job": state["job"], "comparison_sha256": digest(target)})
                return
            status = {"phase": "waiting_for_complete_GPU_archive", "job": state.get("job"),
                      "GPU_state": state.get("state_last_observed"), "last_main_poll_at": state.get("last_poll_at"),
                      "interval_s": 300, "SSH_queries": False, "runtime_changes": False}
            write_status(root, status)
            print(json.dumps({**status, "observed_at": datetime.now().astimezone().isoformat()}), flush=True)
            time.sleep(300)
    except Exception as error:
        write_status(root, {"phase": "diagnosis_required_no_automatic_GPU_retry", "diagnosis": str(error)})
        raise
    finally:
        lock.rmdir()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    run(args.root, args.repo)


if __name__ == "__main__":
    main()
