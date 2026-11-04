import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from public_quality_followup import PLAN_FILE, load_plan, verify_closed
from public_response_inputs import digest
from public_source_recovery import process_alive
from stress_campaign import check_pause, save_state


UNQUALIFIED = "Stratos calibration has no pressure/queue-qualified rate; retain all negatives"


def readiness(parent, recovery, future):
    current = json.loads((parent / "campaign-state.json").read_text())
    if not current.get("GPU_calibration_complete"):
        if "diagnosis_required" in current["phase"]:
            raise RuntimeError("C0 GPU calibration failed; archive and diagnose, never bypass it")
        return "waiting"
    if current.get("candidate_case") is not None:
        return "qualified_parent_owned_by_existing_recovery"
    outer = json.loads((recovery / "campaign-state.json").read_text())
    identity = json.loads((recovery / "local-monitor-identity.json").read_text())
    if process_alive(identity["pid"]):
        return "waiting_outer_exit"
    if "diagnosis_required" not in outer["phase"] or UNQUALIFIED not in outer.get("error", ""):
        raise RuntimeError("Unknown recovery exit; reconcile without creating another producer")
    if (future / "local-monitor-identity.json").exists() or (future / "campaign-state.json").exists():
        raise RuntimeError("An existing source continuation must be reconciled, never duplicated")
    verify_closed(parent, "GPU_calibration_complete")
    return "complete_C0_unqualified_rate_not_GPU_failure"


def verify_freeze(repo, recovery, future):
    prepared = json.loads((recovery / "future-quality-prepared-v2.json").read_text())
    if prepared.get("descriptor_activated") is not True or digest(future / PLAN_FILE) != prepared["descriptor_sha256"]:
        raise RuntimeError("Original quality continuation descriptor changed")
    if any(digest(repo / name) != expected for name, expected in prepared["source_sha256"].items()):
        raise RuntimeError("Frozen future runtime changed; reconcile and rerun regression")


def run(root, parent, recovery, future, interval=300):
    repo = Path(__file__).resolve().parents[2]
    root.mkdir(parents=True, exist_ok=True)
    if (root / "campaign-state.json").exists():
        raise RuntimeError("Preserve existing fallback identity; do not restart it automatically")
    state = {"phase": "local_only_waiting_C0_terminal_rate_qualification", "parent_root": str(parent),
             "recovery_root": str(recovery), "future_root": str(future), "interval_s": interval,
             "controller_sha256": digest(Path(__file__)),
             "SSH_queries_while_waiting": False, "publication_authorized": False,
             "total_research_goal_completed": False}
    save_state(root, state)
    try:
        while True:
            for directory in (root, parent, recovery, future):
                check_pause(directory)
            if digest(Path(__file__)) != state["controller_sha256"]:
                raise RuntimeError("Active local-only fallback controller changed")
            verify_freeze(repo, recovery, future)
            condition = readiness(parent, recovery, future)
            state["condition"] = condition
            save_state(root, state)
            if condition == "qualified_parent_owned_by_existing_recovery":
                state.update(phase="passive_exit_existing_recovery_owns_qualified_parent", new_child_started=False)
                save_state(root, state)
                return
            if condition == "complete_C0_unqualified_rate_not_GPU_failure":
                break
            time.sleep(interval)
        load_plan(future, parent)
        future.mkdir(parents=True, exist_ok=True)
        state.update(phase="launch_intent_do_not_duplicate_verified_unqualified_source", C0_negative_rates_retained=True)
        save_state(root, state)
        with (root / "continuation-coordinator.log").open("x") as output:
            command = [sys.executable, str(repo / "experiments/dp_opt/public_poisson_followup.py"),
                       "--root", str(future), "--parent-root", str(parent)]
            child = subprocess.Popen(command, cwd=repo, stdout=output, stderr=subprocess.STDOUT)
            with (future / "local-monitor-identity.json").open("x") as identity:
                identity.write(json.dumps({"pid": child.pid, "parent_pid": os.getpid(), "interval_s": interval,
                                           "role": "qualified_quality_source_after_unqualified_C0"}, indent=2) + "\n")
            state.update(phase="running_independent_quality_source_continuation", child_pid=child.pid)
            save_state(root, state)
            result = child.wait()
        state["child_exit"] = result
        if result:
            raise RuntimeError("Quality child failed; retain raw evidence and diagnose")
        measured = verify_closed(Path(load_plan(future, parent)[2]), "GPU_evaluation_complete")
        state.update(phase="quality_child_SHA_closed_scientific_conclusion_still_required", quality_job=measured["job"])
        save_state(root, state)
    except Exception as error:
        state.update(phase="diagnosis_required_no_duplicate_fallback", error=repr(error))
        save_state(root, state)
        raise


def main():
    parser = argparse.ArgumentParser(description="Local-only fallback for a completed C0 with no qualified rate")
    for name in ("root", "parent", "recovery", "future"):
        parser.add_argument("--" + name, type=Path, required=True)
    arguments = parser.parse_args()
    run(*(getattr(arguments, name).resolve() for name in ("root", "parent", "recovery", "future")))


if __name__ == "__main__":
    main()
