import argparse
import hashlib
import json
import math
from pathlib import Path

from analyze_stress_sweep import analyze_case, select_best
from verify_agentic_capacity import verify as verify_capacity
from verify_benefit import verify_ownership
from verify_trace import verify as verify_trace


def verify_profile(root):
    summary = json.loads((root / "summary.json").read_text())
    setting = json.loads((root / "setting.json").read_text())
    references = json.loads((root / "references.json").read_text())
    trials = json.loads((root / "trials.json").read_text())
    if (summary["status"] != "profile_passed" or summary["trials"] != trials
            or [trial["layout"] for trial in trials] != ["balanced", "consolidated", "migrated"]):
        raise ValueError("Incomplete declared mixed-cost profile")
    for trial in trials:
        verify_ownership(trial["decode_ready"], trial["budgets_gib"])
        for ranks in trial["decode_ready"]:
            for rank in ranks:
                if (rank["page_bytes"] != 96 * 1024**2 or rank["block_bytes"] != 1.5 * 1024**2
                        or rank["block_size"] != 16 or rank["page_block_capacity"] != 64):
                    raise ValueError("Profile geometry differs from declared Qwen BF16 layout")
        for request in trial["requests"]:
            reference = references[0 if request["model"] == 1 else 1]
            if (request["token_ids"] != reference["token_ids"] or request["prompt_tokens"] != reference["prompt_tokens"]
                    or len(request["token_ids"]) != (setting["Conv"]["output_tokens"] if request["model"] == 1 else setting["Code"]["output_tokens"])):
                raise ValueError("Profile tokens or source input length differ from serial reference")
    migrated = trials[-1]
    verify_ownership(migrated["before"], migrated["budgets_gib"])
    verify_ownership(migrated["after"], migrated["budgets_gib"])
    if (len(migrated["events"]) != 2 or migrated["after"][1][0]["requests"]
            or not migrated["probe_after"]["allocated"] or not migrated["probe_after"]["fits_budget"]
            or not migrated["probe_after"]["rollback_verified"]
            or any(probe["rollback_verified"] is not True for probe in migrated["probes_before"])):
        raise ValueError("Whole-source migration failed real Code allocation witness")
    originals = {request["id"]: request for request in migrated["before"][1][0]["requests"]}
    if {event["request_id"] for event in migrated["events"]} != set(originals):
        raise ValueError("Profile migration events differ from source owners")
    for event in migrated["events"]:
        if (event["model"] != 1 or event["source"] != 0 or event["destination"] != 1
                or not event["bitwise_verified"] or event["kv_bytes"] <= 0
                or event["kv_bytes"] != len(originals[event["request_id"]]["blocks"]) * migrated["before"][1][0]["block_bytes"]
                or any(peak > budget + 1e-6 for peak, budget in zip(event["copy_peak_gib"], migrated["budgets_gib"]))):
            raise ValueError("Profile migration lacks checked content, destination peak or ownership")
        if any(request["id"] == event["request_id"] for request in migrated["after"][1][0]["requests"]):
            raise ValueError("Moved profile request still owns source KV")
        if sum(request["id"] == event["request_id"] for request in migrated["after"][1][1]["requests"]) != 1:
            raise ValueError("Moved profile request lacks destination owner")
    measured = sum(request["completed_at"] - request["submitted_at"] for request in trials[0]["requests"]) / 4
    if not math.isclose(measured, summary["rate"]["representative_service_s"], rel_tol=1e-12):
        raise ValueError("Profile rate omitted actual Conv service")
    return {"status": "profile_raw_verified", "Code_input_tokens": setting["Code"]["prompt_tokens"],
            "Conv_output_tokens": setting["Conv"]["output_tokens"], "migrations": 2,
            "both_before_probes_blocked": not any(probe["fits_budget"] for probe in migrated["probes_before"]),
            "scope": "joint_finite_population_correctness_and_cost_not_trace_benefit"}


def verify(root, inputs):
    root = Path(root)
    manifest = json.loads((root / "stress-manifest.json").read_text())
    frozen = json.loads((root / "stress-freeze.json").read_text())
    if ((root / "exit-status").read_text().strip() != "0" or manifest["status"] != "passed"
            or len(manifest["completed_cases"]) != manifest["planned_cases"]
            or len(frozen["settings_and_traces"]) != manifest["planned_cases"]
            or frozen["frozen_before_policy_outcomes"] is not True):
        raise ValueError("Sweep incomplete or frozen settings missing")
    capacity = verify_capacity(root / "capacity", inputs)
    profiles = {}
    for name, digest in frozen["profile_files_sha256"].items():
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Calibration profile changed after settings freeze")
        profiles[name] = verify_profile(path.parent)
    cases = []
    for index, (case, entry) in enumerate(zip(manifest["completed_cases"], frozen["settings_and_traces"])):
        directory = root / case["directory"]
        trace = json.loads((directory / "trace.json").read_text())
        setting = json.loads((directory / "stress-setting.json").read_text())
        if (trace != entry["trace"] or setting != entry["setting"] or case["setting"] != setting
                or directory.name != f"case-{index}"):
            raise ValueError("Executed stress workload differs from frozen planned case")
        for mark in trace["requests"]:
            if mark["model"] == 1 and (mark["synthetic_output_tokens"] != mark["output_tokens"]
                    or not 512 <= mark["original_output_tokens"] <= 1024 or "synthetic" not in mark["source_fragment"]):
                raise ValueError("Long-output variant concealed original Azure output")
        verified = verify_trace(directory, inputs / "prompt-bank.json")
        trace_digest = hashlib.sha256((directory / "trace.json").read_bytes()).hexdigest()
        if not any(line.split(maxsplit=1)[0] == trace_digest for line in (directory / "source.sha256").read_text().splitlines()):
            raise ValueError("Case trace missing frozen source lineage")
        analysis = analyze_case(directory)
        if analysis != case["analysis"]:
            raise ValueError("Archived selection/tail metrics differ from independent recomputation")
        cases.append({"directory": case["directory"], "setting": setting, "analysis": analysis,
                      "verification": verified})
    return {"status": "stress_sweep_independently_verified", "capacity": capacity,
            "profiles": profiles, "cases": cases, "best_observed": select_best(cases),
            "scope": "declared_grid_descriptive_best_all_outcomes_retained_no_generalized50pct_claim"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root, args.inputs)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]),
                      "best_setting": result["best_observed"]["setting"] if result["best_observed"] else None}))


if __name__ == "__main__":
    main()
