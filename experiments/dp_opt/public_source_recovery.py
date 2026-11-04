import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from amortized_campaign import check_frozen_sources
from archive_trace_job import verify_raw
from public_response_inputs import digest, validate
from public_workload_campaign import SYNC_FILES
from stress_campaign import check_pause, save_state
from verify_trace import verify
from verify_workaware_profile import same_payload, verify_profile


def process_alive(pid):
    return subprocess.run(["ps", "-p", str(pid), "-o", "pid="], capture_output=True).returncode == 0


def verify_closed_failure(parent):
    state = json.loads((parent / "campaign-state.json").read_text())
    job = state["job"]
    archive = json.loads((parent / f"archive-{job}.json").read_text())
    if (state.get("state_last_observed") != "FAILED" or state.get("GPU_calibration_complete")
            or archive.get("state") != "FAILED" or archive.get("watcher_barrier") != "shared_markers"):
        raise RuntimeError("Require a diagnosed failed parent with complete terminal SHA archive")
    identity = json.loads((parent / "local-monitor-identity.json").read_text())
    if process_alive(identity["pid"]):
        raise RuntimeError("Parent coordinator still alive; do not overwrite its runtime")
    raw = parent / f"live-trace-{job}"
    manifest = (parent / f"raw-{job}.sha256").read_text()
    verify_raw(raw, manifest)
    if archive["raw_files_verified"] != len(manifest.splitlines()):
        raise RuntimeError("Failed parent archive file count differs")
    error = (parent / f"live-trace-{job}.err").read_text()
    failure = {"job": job, "state": "FAILED", "raw_files": archive["raw_files_verified"],
               "SHA_verified": True, "not_reclassified_as_success": True}
    if "KeyError: 'traj_id'" in error and "verify_trace.py" in error:
        phase = json.loads((raw / "phase.json").read_text())["phase"]
        native = raw / phase
        if not phase.startswith("native-") or (native / "exit-status").read_text().strip() != "0":
            raise RuntimeError("Prompt-bank recovery requires a completely drained native phase")
        result = verify(native, parent / "frozen/inputs/prompt-bank.json")
        if (result["status"] != "raw_trace_verified" or len(result["trials"]) != 4
                or {trial["policy"] for trial in result["trials"]}
                != {"no_move", "random_no_return", "kv_only_headroom", "kv_only_workaware"}):
            raise RuntimeError("Prompt-bank recovery requires independent verification of every native policy")
        fit = verify_profile(raw / "public-context-calibration")
        if not same_payload(fit, json.loads((raw / "public-context-fit.json").read_text())):
            raise RuntimeError("Prompt-bank recovery cost profile differs from independent reconstruction")
        failure.update({"failure_kind": "public_prompt_bank_verifier_schema", "native_phase": phase,
                        "native_independent_status": result["status"], "native_policy_count": len(result["trials"]),
                        "native_trace_sha256": result["trace_sha256"], "cost_profile_independently_verified": True})
        return failure
    if "KeyError: 'source_internal_row'" not in error or "live_trace.py" not in error:
        raise RuntimeError("This recovery only addresses the diagnosed public provenance failure")
    return failure


def check_quality_source(state):
    if "quality_plan_root" not in state:
        return
    from public_quality_followup import PLAN_FILE, load_plan

    plan_root = Path(state["quality_plan_root"])
    if digest(plan_root / PLAN_FILE) != state["quality_followup_plan_sha256"]:
        raise RuntimeError("Active quality recovery declaration changed")
    plan, calibration, evaluation = load_plan(plan_root, Path(state["failed_parent_root"]))
    if (str(calibration) != state["calibration_root"] or str(evaluation) != state["evaluation_root"]
            or plan["input_manifest_sha256"] != state["input_manifest_sha256"]):
        raise RuntimeError("Active quality recovery source/roots changed")


def run_stage(repo, root, state, child_root, script, arguments, role):
    label = state.get("dataset_label", "stratos")
    check_pause(root)
    check_quality_source(state)
    check_frozen_sources(repo, state)
    if digest(Path(__file__)) != state["controller_sha256"]:
        raise RuntimeError("Active recovery controller changed")
    child_root.mkdir(parents=True, exist_ok=True)
    identity = child_root / "local-monitor-identity.json"
    if identity.exists() or (child_root / "campaign-state.json").exists():
        raise RuntimeError("Preserve existing child identity/state; never duplicate a submission")
    with (root / f"{role}-coordinator.log").open("w") as output:
        child = subprocess.Popen([sys.executable, str(repo / "experiments/dp_opt" / script),
                                  "--root", str(child_root), *arguments], cwd=repo,
                                 stdout=output, stderr=subprocess.STDOUT)
        identity.write_text(json.dumps({"pid": child.pid, "parent_pid": os.getpid(),
                                       "interval_s": 300, "role": role}, indent=2) + "\n")
        state.update({"phase": f"running_{label}_{role}", "child_pid": child.pid})
        save_state(root, state)
        result = child.wait()
    if result:
        raise RuntimeError(f"{label} {role} failed; diagnose its complete raw evidence before retry")
    check_frozen_sources(repo, state)
    check_quality_source(state)
    if digest(Path(__file__)) != state["controller_sha256"]:
        raise RuntimeError("Active recovery controller changed")
    return json.loads((child_root / "campaign-state.json").read_text())


def run(root, failed_parent, bootstrap, calibration, evaluation, quality_plan_root=None):
    repo = Path(__file__).resolve().parents[2]
    root.mkdir(parents=True, exist_ok=True)
    if (root / "campaign-state.json").exists():
        raise RuntimeError("Preserve recorded recovery state; reconcile before any restart")
    failure = verify_closed_failure(failed_parent)
    previous = json.loads((bootstrap / "campaign-state.json").read_text())
    if not previous.get("GPU_calibration_complete") or previous.get("state_last_observed") != "COMPLETED":
        raise RuntimeError("Require a genuinely completed environment bootstrap, not the failed public parent")
    verify_raw(bootstrap / f"live-trace-{previous['job']}", (bootstrap / f"raw-{previous['job']}.sha256").read_text())
    inputs = calibration / "frozen/inputs"
    manifest = validate(inputs)
    label, sync_files = "stratos", SYNC_FILES
    quality_plan = None
    if quality_plan_root is not None:
        from public_poisson_followup import SYNC_FILES as QUALITY_SYNC_FILES
        from public_quality_followup import LABEL, PLAN_FILE, load_plan

        if failure.get("failure_kind") != "public_prompt_bank_verifier_schema":
            raise RuntimeError("Independent quality recovery requires the diagnosed and reverified prompt-bank failure")
        quality_plan, planned_calibration, planned_evaluation = load_plan(quality_plan_root, failed_parent)
        if calibration != planned_calibration or evaluation != planned_evaluation:
            raise RuntimeError("Quality recovery must use the unchanged predeclared independent roots")
        label, sync_files = LABEL, QUALITY_SYNC_FILES
    if manifest["dataset_label"] != label or manifest["bank_size"] != 256:
        raise RuntimeError("Require the unchanged original-distribution " + label + "256 bank")
    state = {"repo": str(repo), "failed_parent_root": str(failed_parent), "closed_failure": failure,
             "bootstrap_root": str(bootstrap), "bootstrap_job": previous["job"],
             "calibration_root": str(calibration), "evaluation_root": str(evaluation),
             "phase": "verified_failure_fixed_starting_original_" + label, "dataset_label": label,
             "input_manifest_sha256": digest(inputs / "input-manifest.json"),
             "frozen_runtime_sha256": {name: digest(repo / name) for name in sync_files
                                       if name.endswith((".py", ".sh", ".sbatch"))},
             "controller_sha256": digest(Path(__file__)), "publication_authorized": False,
             "total_research_goal_completed": False}
    if quality_plan is not None:
        state.update({"quality_plan_root": str(quality_plan_root),
                      "quality_followup_plan_sha256": digest(quality_plan_root / PLAN_FILE),
                      "source_selection": quality_plan["source_selection"], "Stratos_inputs_replaced": False,
                      "C0_failure_not_reclassified_as_success": True,
                      "C0_costs_not_reused_for_C1": True})
    check_quality_source(state)
    save_state(root, state)
    try:
        result = run_stage(repo, root, state, calibration, "public_workload_campaign.py",
                           ["--previous-root", str(bootstrap)], "calibration")
        if not result.get("GPU_calibration_complete") or result.get("candidate_case") is None:
            raise RuntimeError(label + " calibration has no pressure/queue-qualified rate; retain all negatives")
        if digest(inputs / "input-manifest.json") != state["input_manifest_sha256"]:
            raise RuntimeError("Frozen original Stratos input manifest changed")
        run_stage(repo, root, state, evaluation, "public_poisson_followup.py",
                  ["--parent-root", str(calibration)], "finite_grid_and_optional_fixed_heldout")
        state.update({"phase": label + "_grid_optional_heldout_SHA_closed_requires_scientific_conclusion",
                      "next_action": "compare_all_tenants_TTFT_Conv_TPOT_and_return_cost_vs_natural_recovery"})
        save_state(root, state)
    except Exception as error:
        state.update({"phase": "diagnosis_required_no_automatic_duplicate_submission", "error": repr(error)})
        save_state(root, state)
        raise


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "failed-parent", "bootstrap", "calibration", "evaluation"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--quality-plan-root", type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.failed_parent.resolve(), args.bootstrap.resolve(),
        args.calibration.resolve(), args.evaluation.resolve(),
        args.quality_plan_root.resolve() if args.quality_plan_root is not None else None)


if __name__ == "__main__":
    main()
