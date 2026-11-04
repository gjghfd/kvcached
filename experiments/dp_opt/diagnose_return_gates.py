import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

from workaware_robust_ilp import enumerate_robust_reference, solve_robust_candidate


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Require finite timestamps and request-second costs")
    return value


def summarize_plan(plan):
    gate = plan["gate"]
    reference, winner = plan["reference"], plan["winner"]
    gain = finite(reference["objective_s"]) - finite(winner["objective_s"])
    uncertainty = finite(gate["uncertainty_request_s"])
    minimum = finite(gate["min_gain_request_s"])
    if (uncertainty < 0 or minimum < 0
            or not math.isclose(gain, finite(gate["predicted_net_gain_request_s"]), abs_tol=1e-6)
            or gate["accepted"] != bool(winner["moves"] and gain > uncertainty + minimum)):
        raise ValueError("Recorded workaware gate disagrees with its objective or uncertainty")
    blocker = ("accepted" if gate["accepted"] else "no_move_or_nonpositive_point_gain"
               if not winner["moves"] or gain <= minimum else "uncertainty_exceeds_point_gain")
    return {"population": plan["active_population"], "forecast_horizon_s": plan["forecast_horizon_s"],
            "point_gain_request_s": gain, "uncertainty_request_s": uncertainty,
            "minimum_gain_request_s": minimum, "guarded_margin_request_s": gain - uncertainty - minimum,
            "gate_blocker": blocker, "accepted": gate["accepted"], "candidate_moves": list(winner["moves"]),
            "migration_cost_request_s": winner["migration_cost_s"],
            "activation_cost_request_s": winner.get("migration_activation_cost_s", 0),
            "reference_risk_request_s": reference.get("work_cost", {}).get("placement_risk_cost_s", 0),
            "winner_risk_request_s": winner.get("work_cost", {}).get("placement_risk_cost_s", 0),
            "candidate_status": winner["status"], "candidate_mip_gap": winner.get("mip_gap")}


def replay_scenarios(plan, max_moves, max_evaluations):
    scenarios = plan.get("plan", {}).get("scenarios", [])
    snapshots = [entry["snapshot"] for entry in scenarios] if scenarios else [plan["snapshot"]]
    rebuilt = [enumerate_robust_reference(snapshot, max_moves, max_evaluations) for snapshot in snapshots]
    feasible = [entry["winner"] for entry in rebuilt if entry["winner"] is not None]
    winner = min(feasible, key=lambda entry: (entry["robust_objective_request_s"], len(entry["moves"]))) if feasible else None
    margin = (plan["reference"]["objective_s"] - winner["robust_objective_request_s"]
              - plan["gate"]["min_gain_request_s"]) if winner else None
    return {"winner": {key: winner[key] for key in ("moves", "objective_s", "uncertainty_request_s",
                                                   "robust_objective_request_s")} if winner else None,
            "guarded_margin_request_s": margin,
            "passes_unchanged_guard": bool(winner and winner["moves"] and margin > 0),
            "all_scenarios_exhaustive": all(entry["status"] == "complete_reference_enumeration" for entry in rebuilt),
            "evaluations": sum(entry["evaluations"] for entry in rebuilt),
            "scope": "recorded_current_state_counterfactual_no_future_arrivals_not_causal_GPU_benefit"}


def solve_scenarios(plan, max_moves, budget_s):
    scenarios = plan.get("plan", {}).get("scenarios", [])
    snapshots = [entry["snapshot"] for entry in scenarios] if scenarios else [plan["snapshot"]]
    started = time.monotonic()
    solved = [solve_robust_candidate(snapshot, max_moves, budget_s / len(snapshots)) for snapshot in snapshots]
    feasible = [entry for entry in solved if "robust_objective_request_s" in entry]
    winner = min(feasible, key=lambda entry: (entry["robust_objective_request_s"], len(entry["moves"]))) if feasible else None
    margin = (plan["reference"]["objective_s"] - winner["robust_objective_request_s"]
              - plan["gate"]["min_gain_request_s"]) if winner else None
    return {"winner": {key: winner[key] for key in ("moves", "objective_s", "uncertainty_request_s",
                                                   "robust_objective_request_s")} if winner else None,
            "guarded_margin_request_s": margin, "passes_unchanged_guard": bool(winner and winner["moves"] and margin > 0),
            "scenario_statuses": [entry["status"] for entry in solved],
            "all_scenarios_proven": all(entry["status"] in ("optimal", "infeasible") for entry in solved),
            "planning_wall_s": time.monotonic() - started,
            "scope": "local_candidate_ILP_not_installed_not_GPU_benefit"}


def analyze(root, release_window_s=75, replay_max_evaluations=0, robust_ilp_budget_s=0):
    root = Path(root)
    if (finite(release_window_s) <= 0 or type(replay_max_evaluations) is not int or replay_max_evaluations < 0
            or finite(robust_ilp_budget_s) < 0):
        raise ValueError("Require a positive release window and nonnegative replay budget")
    workload = json.loads((root / "workload.json").read_text())
    origin = finite(workload["origin"])
    records = json.loads((root / "requests.json").read_text())
    code = sorted((record for record in records if record["model"] == 0), key=lambda row: row["completed_at"])
    counts, blockers, events = Counter(), Counter(), []
    with (root / "decisions.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            if row.get("path") != "release_rebalance_ilp":
                continue
            reason = row["reason"]
            counts[reason] += 1
            event = {"elapsed_s": finite(row["started_at"]) - origin, "reason": reason,
                     "preplan_s": row.get("preplan_s"), "pause_call_s": row.get("pause_call_s"),
                     "calibration_deferral": row.get("calibration_deferral"),
                     "resume_elapsed_s": finite(row["resume_completed_at"]) - origin if row.get("resume_completed_at") else None}
            if row.get("observations"):
                fit = row.get("settings", {}).get("workaware_fit", {})
                profiles = fit.get("population_profiles", {}).get("profiles", {})
                domains = []
                for model, ranks in enumerate(row["observations"]):
                    requests = [request for rank in ranks for request in rank["requests"]]
                    profile = profiles.get(f"{model}:{len(requests)}", {})
                    domains.append({"model": model, "counts": [len(rank["requests"]) for rank in ranks],
                        "population_profile_available": bool(profile),
                        "max_observed_context_tokens": max((finite(request["computed_tokens"]) for request in requests), default=0),
                        "peak_observed_attention_tokens": max(sum(finite(request["computed_tokens"]) for request in rank["requests"])
                                                              for rank in ranks),
                        "measured_max_context_tokens": profile.get("max_context_tokens", fit.get("max_context_tokens")),
                        "measured_max_attention_tokens": profile.get("max_attention_tokens", fit.get("max_attention_tokens"))})
                event["observed_domains"] = domains
            for phase in ("before_pause", "after_pause"):
                if row.get(phase):
                    event[phase] = summarize_plan(row[phase])
            if "before_pause" in event:
                blockers[event["before_pause"]["gate_blocker"]] += 1
                if replay_max_evaluations:
                    event["robust_reference"] = replay_scenarios(row["before_pause"], row["settings"]["max_moves"],
                                                                 replay_max_evaluations)
                if robust_ilp_budget_s:
                    event["robust_ilp"] = solve_scenarios(row["before_pause"], row["settings"]["max_moves"],
                                                         robust_ilp_budget_s)
            events.append(event)
    windows = []
    for record in code:
        released = finite(record["completed_at"]) - origin
        windows.append({"Code_request_id": record["request_id"], "release_elapsed_s": released,
                        "window_s": release_window_s,
                        "events": [{**event, "delay_from_release_s": event["elapsed_s"] - released}
                                   for event in events if released <= event["elapsed_s"] < released + release_window_s]})
    return {"status": "offline_recorded_gate_diagnosis_not_GPU_benefit",
            "source_sha256": {name: digest(root / name) for name in ("workload.json", "requests.json", "decisions.jsonl")},
            "release_decision_counts": dict(counts), "before_pause_gate_blockers": dict(blockers),
            "Code_release_windows": windows, "all_release_events": events,
            "runtime_changed": False, "error_bound_reduced": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--release-window-s", type=float, default=75)
    parser.add_argument("--replay-max-evaluations", type=int, default=0)
    parser.add_argument("--robust-ilp-budget-s", type=float, default=0)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.release_window_s, args.replay_max_evaluations, args.robust_ilp_budget_s), indent=2))


if __name__ == "__main__":
    main()
