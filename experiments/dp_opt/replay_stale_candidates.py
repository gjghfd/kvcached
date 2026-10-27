import argparse
import hashlib
import json
from pathlib import Path

from ilp_reference import replay_candidate
from live_controller import admission_planning_snapshot
from layout_cost_pilot import load_verified_lookup
from verify_benefit import verify_ownership
from verify_candidate_revalidation import verify_projection


def replay(root, fit_path):
    config = json.loads((root / "config.json").read_text())
    fit = json.loads(fit_path.read_text())
    lookup = load_verified_lookup(root / "layout-cost-lookup.json", config["layout_cost_lookup_sha256"],
                                  root / "layout-cost-source")
    source_files = [root / "config.json", fit_path, root / "layout-cost-lookup.json"]
    cases = []
    for policy in ("own_model", "kv_only", "ilp"):
        path = root / f"r0-{policy}/decisions.json"
        source_files.append(path)
        for index, decision in enumerate(json.loads(path.read_text())):
            if not str(decision.get("reason", "")).startswith("preplan_stale_"):
                continue
            prompt, output, budgets, original_policy, tenant, max_moves, limit, fixed, bandwidth, solver_limit = decision["preplan"]["settings"]
            observed = decision["observations"]
            verify_ownership(observed, budgets)
            original = [candidate for candidate in decision["plan"]["candidates"] if "objective_s" in candidate]
            selected = {identifier for candidate in original for identifier in candidate["moves"]}
            snapshot, horizon, population = admission_planning_snapshot(observed, tenant, prompt + output, output,
                budgets, fit, True, policy, limit, fixed, bandwidth, lookup, selected)
            previous = {request["id"]: request for request in decision["snapshot"]["movable"]}
            current = {request["id"]: request for request in snapshot["movable"]}
            projections = []
            for candidate in original:
                changed = [identifier for identifier in candidate["moves"] if identifier not in current
                           or any(current[identifier][name] != previous[identifier][name]
                                  for name in ("model", "source", "memory_gib"))]
                if changed:
                    projections.append({"status": "selected_rank_or_rounded_reservation_changed", "selected": changed})
                else:
                    projected = replay_candidate(snapshot, candidate)
                    verify_projection(snapshot, candidate, projected)
                    projections.append(projected)
            cases.append({"policy": policy, "decision_index": index, "request_id": decision["request_id"],
                          "reason": decision["reason"], "old_counts": decision["snapshot"]["counts"],
                          "current_counts": snapshot["counts"], "current_horizon_steps": horizon,
                          "current_active_plus_incoming": population, "projections": projections,
                          "any_current_projection_feasible": any("objective_s" in item for item in projections),
                          "exact_original_blocks_available": False})
    return {"status": "current_raw_owner_copy_admission_projection_independently_verified",
            "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
            "cases": cases,
            "scope": "offline_original_actions_current_owner_projection_not_GPU_benefit_or_exact_original_block_identity_proof"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.root, args.fit)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]),
                      "feasible": sum(case["any_current_projection_feasible"] for case in result["cases"])}))


if __name__ == "__main__":
    main()
