import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from amortized_campaign import check_frozen_sources
from archive_trace_job import verify_raw
from public_response_inputs import digest, validate
from stress_campaign import check_pause, save_state


PLAN_FILE = "quality-followup.json"
LABEL = "openthoughts3_closed"


def load_plan(root, parent):
    plan = json.loads((root / PLAN_FILE).read_text())
    if (plan.get("schema_version") != 1 or plan.get("dataset_label") != LABEL
            or plan.get("source_selection") != "original_marker_quality_only_not_output_length_or_GPU_gain"
            or plan.get("Stratos_bank_replaced") is not False):
        raise ValueError("Quality study must predeclare original-source selection and preserve Stratos")
    calibration = Path(plan["calibration_root"]).resolve()
    evaluation = Path(plan["evaluation_root"]).resolve()
    if (calibration.parent != root.parent.resolve() or evaluation.parent != root.parent.resolve()
            or len({calibration, evaluation, root.resolve(), parent.resolve()}) != 4
            or (evaluation / PLAN_FILE).exists()):
        raise ValueError("Independent quality study needs distinct local evidence roots")
    inputs, original = calibration / "frozen/inputs", parent / "frozen/inputs"
    manifest = validate(inputs)
    if (digest(inputs / "input-manifest.json") != plan["input_manifest_sha256"]
            or digest(original / "input-manifest.json") != plan["parent_input_manifest_sha256"]
            or manifest["dataset_label"] != LABEL or validate(original)["dataset_label"] != "stratos"):
        raise ValueError("Quality/parent bank differs from its original source freeze")
    code = json.loads((inputs / "code-marks.json").read_text())
    old_code = json.loads((original / "code-marks.json").read_text())
    bank = json.loads((inputs / "prompt-bank.json").read_text())
    old_bank = json.loads((original / "prompt-bank.json").read_text())
    if (code != old_code or not code or any(mark["prompt_tokens"] < 100000 or mark["output_tokens"] != 64 for mark in code)
            or any(bank["prompts"][mark["prompt_key"]] != old_bank["prompts"][mark["prompt_key"]] for mark in code)):
        raise ValueError("Quality source cannot change the Code request stream or token IDs")
    cohort = Path(plan["cohort_row_map"]).resolve()
    if root.parent.resolve() not in cohort.parents or digest(cohort) != plan["cohort_row_map_sha256"]:
        raise ValueError("Quality source needs a frozen local original-response cohort")
    records = [json.loads(line) for line in cohort.read_text().splitlines()]
    identity_path = cohort.parent / "predeclared-identity.json"
    identity = json.loads(identity_path.read_text())
    identity_sha = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
    archive = json.loads((cohort.parent / "archive-manifest.json").read_text())
    quality = manifest.get("source_quality", {})
    definition = identity["cohort_definition"]
    if (identity_sha != plan["cohort_identity_sha256"] or quality.get("cohort_identity_sha256") != identity_sha
            or bank.get("source_quality") != quality or archive.get("identity_sha256") != identity_sha
            or quality.get("C0_manifest_sha256") != plan["parent_input_manifest_sha256"]
            or quality.get("cohort_rows") != len(records) or quality.get("natural_eos_verified") is not False
            or quality.get("not_full_population") is not True
            or definition.get("no_token_length_selection") is not True
            or definition.get("no_GPU_result_selection") is not True
            or definition.get("no_response_trim_concat_padding_or_extension") is not True):
        raise ValueError("Quality source must preserve its predeclared marker-only identity and limitations")
    for name, expected in quality["cohort_files_sha256"].items():
        if (name not in ("predeclared-identity.json", "full-source-response-row-map.jsonl",
                        "cohort-response-row-map.jsonl", "summary.json")
                or digest(cohort.parent / name) != expected or archive["outputs"][name]["sha256"] != expected):
            raise ValueError("Original quality cohort evidence changed")
    for record in records:
        flags = record["marker_flags"]
        if (flags.get("open_marker_count") != 1 or flags.get("close_marker_count") != 1
                or flags.get("unique_pair_in_order") is not True
                or flags.get("nonempty_original_final_text") is not True
                or flags.get("cohort_eligible") is not True):
            raise ValueError("Original quality cohort contains an ineligible response")
    known = {(record["source_sha256"], record["source_row"], record["assistant_index"],
              record["original_response_sha256"], record["profile_selection_sha256"]): record for record in records}
    conv = json.loads((inputs / "conv-marks.json").read_text())
    if (len(conv) != 256 or len(known) != len(records)
            or [mark["selection_sha256"] for mark in conv]
            != [record["profile_selection_sha256"] for record in sorted(records, key=lambda record: record["profile_selection_sha256"])[:256]]):
        raise ValueError("Quality study must keep the whole predeclared hash bank256")
    for mark in conv:
        record = known.get((mark["source_sha256"], mark["source_row"], mark["assistant_index"],
                            mark["original_response_sha256"], mark["selection_sha256"]))
        if (record is None or mark["prompt_tokens"] != record["prompt_tokens"]
                or mark["output_tokens"] != record["output_tokens"]):
            raise ValueError("Quality bank changes an original cohort request or response length")
    return plan, calibration, evaluation


def verify_closed(root, completed_field):
    state = json.loads((root / "campaign-state.json").read_text())
    if not state.get(completed_field) or state.get("state_last_observed") != "COMPLETED":
        raise RuntimeError("Quality child must independently finish, not merely submit or run")
    job = state["job"]
    archive = json.loads((root / f"archive-{job}.json").read_text())
    manifest = (root / f"raw-{job}.sha256").read_text()
    raw = root / f"live-trace-{job}"
    verify_raw(raw, manifest)
    if (archive.get("state") != "COMPLETED" or archive.get("watcher_barrier") != "shared_markers"
            or archive["raw_files_verified"] != len(manifest.splitlines())
            or (raw / "exit-status").read_text().strip() != "0"):
        raise RuntimeError("Quality child requires complete terminal SHA and watcher closure")
    analysis_file = "search-analysis.json" if completed_field == "GPU_calibration_complete" else "evaluation-analysis.json"
    result = json.loads((root / "independent-verification.json").read_text())
    if result != json.loads((raw / analysis_file).read_text()):
        raise RuntimeError("Quality child independent reconstruction differs from compute analysis")
    if completed_field == "GPU_calibration_complete" and state.get("candidate_case") != result.get("candidate_case"):
        raise RuntimeError("Quality child rate qualification differs from independent reconstruction")
    return state


def mirror_child(root, state, child_root):
    path = child_root / "campaign-state.json"
    if not path.exists():
        return
    observed = json.loads(path.read_text())
    for name in ("job", "state_last_observed", "last_poll_at", "last_poll_ssh_exit", "last_poll_actual_gap_s",
                 "last_poll_monotonic_gap_s", "last_poll_wall_clock_gap_s", "remote_watcher_pid"):
        if name in observed:
            state[name] = observed[name]
        else:
            state.pop(name, None)
    state["source_child_phase"] = observed["phase"]
    save_state(root, state)


def run_child(repo, root, state, child_root, script, arguments, role):
    check_pause(root)
    check_pause(child_root)
    check_frozen_sources(repo, state)
    if digest(root / PLAN_FILE) != state["quality_followup_plan_sha256"]:
        raise RuntimeError("Active original-source plan changed")
    child_root.mkdir(parents=True, exist_ok=True)
    identity = child_root / "local-monitor-identity.json"
    if identity.exists() or (child_root / "campaign-state.json").exists():
        raise RuntimeError("Preserve existing quality child identity; reconcile before any restart")
    with (root / f"{role}-coordinator.log").open("x") as output:
        child = subprocess.Popen([sys.executable, str(repo / "experiments/dp_opt" / script),
                                  "--root", str(child_root), *arguments], cwd=repo,
                                 stdout=output, stderr=subprocess.STDOUT)
        identity.write_text(json.dumps({"pid": child.pid, "parent_pid": os.getpid(), "interval_s": 300,
                                       "role": role}, indent=2) + "\n")
        state.update({"phase": "running_" + role, "active_source_root": str(child_root), "source_child_pid": child.pid})
        save_state(root, state)
        while child.poll() is None:
            check_pause(root)
            check_pause(child_root)
            check_frozen_sources(repo, state)
            if digest(root / PLAN_FILE) != state["quality_followup_plan_sha256"]:
                raise RuntimeError("Active original-source plan changed")
            mirror_child(root, state, child_root)
            time.sleep(300)
        mirror_child(root, state, child_root)
        state["source_child_exit"] = child.returncode
        save_state(root, state)
        if child.returncode:
            raise RuntimeError(f"Quality {role} failed; retain all raw evidence and diagnose before retry")
    check_frozen_sources(repo, state)
    if digest(root / PLAN_FILE) != state["quality_followup_plan_sha256"]:
        raise RuntimeError("Active original-source plan changed")


def follow_quality(root, parent, repo, state):
    if not (root / PLAN_FILE).exists():
        return False
    plan, calibration, evaluation = load_plan(root, parent)
    verify_closed(parent, "GPU_calibration_complete")
    if state.get("quality_followup_plan_sha256"):
        raise RuntimeError("Reconcile recorded quality-source continuation before restarting")
    state.update({"quality_followup_plan_sha256": digest(root / PLAN_FILE), "quality_source_plan": plan,
                  "source_parent_root": str(parent), "dataset_label": LABEL})
    save_state(root, state)
    run_child(repo, root, state, calibration, "public_workload_campaign.py",
              ["--previous-root", str(parent)], "quality_source_calibration")
    measured = verify_closed(calibration, "GPU_calibration_complete")
    if measured.get("candidate_case") is None:
        raise RuntimeError("Original quality source has no pressure/queue-qualified rate; retain negatives")
    run_child(repo, root, state, evaluation, "public_poisson_followup.py",
              ["--parent-root", str(calibration)], "quality_source_grid_and_fixed_heldout")
    result = verify_closed(evaluation, "GPU_evaluation_complete")
    state.update({"phase": "quality_source_grid_optional_heldout_SHA_verified_requires_scientific_conclusion",
                  "quality_source_study_complete": True, "GPU_evaluation_complete": True,
                  "Stratos_bank_replaced": False, "Stratos_grid_run_by_this_delegation": False,
                  "quality_evaluation_root": str(evaluation), "quality_evaluation_job": result["job"],
                  "total_research_goal_completed": False,
                  "next_action": "compare_all_tenant_tail_TTFT_Conv_TPOT_and_causal_return_vs_random"})
    save_state(root, state)
    return True
