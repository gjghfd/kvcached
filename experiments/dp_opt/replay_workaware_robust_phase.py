"""Replay verified point-policy release states without changing live inference."""

import argparse
import copy
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path

from amortized_campaign import check_frozen_sources
from archive_trace_job import verify_raw
from verify_trace import verify
from verify_workaware_robust_rebalance import verify_plan
from workaware_robust_rebalance import plan_workaware_robust_rebalance


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(repo, source_manifest, runtime_state):
    sources = {}
    for line in source_manifest.read_text().splitlines():
        expected, filename = line.split(maxsplit=1)
        path = repo / filename
        if (len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected)
                or not path.resolve().is_relative_to(repo.resolve()) or path.is_symlink()
                or filename in sources or digest(path) != expected):
            raise ValueError("Frozen robust interface source mismatch or unsafe manifest")
        sources[filename] = expected
    if not sources:
        raise ValueError("Require a nonempty frozen source manifest")
    state = json.loads(runtime_state.read_text())
    if not state.get("frozen_runtime_sha256"):
        raise ValueError("Require the original frozen runtime hashes before historical verification")
    check_frozen_sources(repo, state)
    return sources, state["frozen_runtime_sha256"]


def replay_decisions(case, fit, solver_budgets):
    workload = json.loads((case / "workload.json").read_text())
    if workload["policy"] != "kv_only_workaware":
        raise ValueError("Only original workaware release states can be compared")
    budgets = workload["capacity_gib"]
    decisions = json.loads((case / "decisions.json").read_text())
    result = {"case": case.name, "source_decisions_SHA256": digest(case / "decisions.json"),
              "capacity_gib": budgets, "replays": [], "unplanned_releases": []}
    for index, decision in enumerate(decisions):
        if decision["path"] != "release_rebalance_ilp":
            continue
        if "before_pause" not in decision:
            if "after_pause" in decision:
                raise ValueError("After-pause evidence lacks its original before-pause plan")
            result["unplanned_releases"].append({"decision_index": index, "reason": decision["reason"]})
            continue
        for phase, observation_key in (("before_pause", "observations"), ("after_pause", "frozen_observations")):
            if phase not in decision:
                continue
            blocking = decision[phase]["gate"]["measured_blocking_so_far_request_s"] if phase == "after_pause" else 0.
            for solver_budget in solver_budgets:
                settings = copy.deepcopy(decision["settings"])
                settings["solver_limit_s"] = solver_budget
                observed = copy.deepcopy(decision[observation_key])
                started = time.monotonic()
                check = plan_workaware_robust_rebalance(
                    copy.deepcopy(observed), copy.deepcopy(budgets), copy.deepcopy(fit), True,
                    copy.deepcopy(settings), None, blocking, phase)
                elapsed = time.monotonic() - started
                verification = verify_plan(check, observed, budgets, fit, settings, phase, blocking,
                                           controller_planning_upper_s=elapsed)
                old_gate = decision[phase]["gate"]
                old_moves = decision[phase]["winner"]["moves"]
                result["replays"].append({"decision_index": index, "phase": phase,
                    "solver_budget_s": solver_budget, "CPU_elapsed_s": elapsed,
                    "old_gate": old_gate, "same_selected_moves_as_point_plan": check["winner"]["moves"] == old_moves,
                    "new_guard_passing_opportunity": check["gate"]["accepted"] and not old_gate["accepted"],
                    "verification": verification, "check": check})
    return result


def replay_phase(raw, raw_manifest, prompt_bank, source_manifest, runtime_state, output, solver_budgets, repo):
    if (not solver_budgets or len(set(solver_budgets)) != len(solver_budgets)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value <= 0 for value in solver_budgets)):
        raise ValueError("Require distinct finite positive total solver budgets")
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(raw.resolve()):
        raise ValueError("Preserve existing output and immutable raw evidence")
    sources, runtime_hashes = verify_sources(repo, source_manifest, runtime_state)
    manifest_text = raw_manifest.read_text()
    verify_raw(raw, manifest_text)
    verification = verify(raw, prompt_bank)
    cases = sorted(raw.glob("r*-kv_only_workaware"))
    if not cases:
        raise ValueError("Verified phase has no original workaware case")
    fit_path = raw / "placement-fit.json"
    fit = json.loads(fit_path.read_text())
    result = {"recorded_at": datetime.now().astimezone().isoformat(),
              "scope": "CPU_counterfactual_original_release_states_not_new_GPU_policy_benefit",
              "raw_manifest_SHA256": digest(raw_manifest), "prompt_bank_SHA256": digest(prompt_bank),
              "source_manifest_SHA256": digest(source_manifest), "interface_source_SHA256": sources,
              "original_runtime_source_SHA256": runtime_hashes, "placement_fit_SHA256": digest(fit_path),
              "verification": verification, "solver_budgets_s": solver_budgets,
              "legacy_structural_fit_feasibility_override": True, "performance_extrapolation": False,
              "cases": [replay_decisions(case, fit, solver_budgets) for case in cases],
              "runtime_changed": False, "GPU_submissions": False, "research_goal_complete": False}
    verify_raw(raw, manifest_text)
    final_sources, final_runtime_hashes = verify_sources(repo, source_manifest, runtime_state)
    if final_sources != sources or final_runtime_hashes != runtime_hashes:
        raise ValueError("Source hashes changed during replay")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as target:
        json.dump(result, target, indent=2)
        target.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--prompt-bank", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--runtime-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--solver-budgets-s", type=float, nargs="+", default=[1.])
    args = parser.parse_args()
    result = replay_phase(args.raw, args.raw_manifest, args.prompt_bank, args.source_manifest,
                          args.runtime_state, args.output, args.solver_budgets_s, Path(__file__).resolve().parents[2])
    entries = [entry for case in result["cases"] for entry in case["replays"]]
    print(json.dumps({"verified_replays": len(entries),
                      "new_guard_passing_opportunities": sum(entry["new_guard_passing_opportunity"] for entry in entries),
                      "scope": result["scope"]}, indent=2))


if __name__ == "__main__":
    main()
