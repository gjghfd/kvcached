import math

from ilp_reference import enumerate_targets, replay_candidate
from live_controller import admission_planning_snapshot


class MigrationInferenceLedger:
    def __init__(self, overhead_fraction, pause_fixed_s):
        if (not math.isfinite(overhead_fraction) or not 0 < overhead_fraction < 1
                or not math.isfinite(pause_fixed_s) or pause_fixed_s < 0):
            raise ValueError("Invalid migration/inference overhead policy")
        self.overhead_fraction = overhead_fraction
        self.pause_fixed_s = pause_fixed_s
        self.completed = {}
        self.epochs = {}
        self.spent_blocking_request_s = 0

    def admission(self, decision):
        if decision.get("admitted") and decision.get("moves"):
            identifier = decision["request_id"]
            self.epochs[identifier] = {"beneficiary": identifier,
                                       "models": sorted({move["model"] for move in decision["moves"]})}

    def completion(self, request):
        identifier = request["request_id"]
        if identifier in self.completed:
            raise ValueError("Duplicate completion in inference-work ledger")
        reserved, finished = request["reserved_at"], request["completed_at"]
        if not all(math.isfinite(value) for value in (reserved, finished)) or finished < reserved:
            raise ValueError("Inference-work ledger requires ordered reservation/completion times")
        self.completed[identifier] = finished - reserved

    def eligible(self):
        return any(identifier in self.completed for identifier in self.epochs)

    def evidence(self, observations, sampled_at):
        active = [request for ranks in observations for rank in ranks for request in rank["requests"]
                  if request["id"] not in self.completed]
        if any(not math.isfinite(request["reserved_at"]) or request["reserved_at"] > sampled_at for request in active):
            raise ValueError("Active request lacks a valid inference-work interval")
        gross = sum(self.completed.values()) + sum(sampled_at - request["reserved_at"] for request in active)
        service = max(0, gross - self.spent_blocking_request_s)
        credit = self.overhead_fraction * service - self.spent_blocking_request_s
        return {"sampled_at": sampled_at, "completed_ids": sorted(self.completed),
                "gross_service_request_s": gross, "inference_service_request_s": service,
                "spent_blocking_request_s": self.spent_blocking_request_s,
                "credit_request_s": max(0, credit), "unclamped_credit_request_s": credit,
                "overhead_fraction": self.overhead_fraction, "pause_fixed_s": self.pause_fixed_s,
                "ready_epochs": [self.epochs[identifier] for identifier in sorted(self.epochs)
                                 if identifier in self.completed],
                "scope": "past_reservation_to_completion_exposure_minus_measured_pause_no_future_trace"}

    def settle(self, decision):
        blocking = decision.get("active_blocking_upper_request_s", 0)
        if not math.isfinite(blocking) or blocking < 0:
            raise ValueError("Invalid measured migration blocking cost")
        self.spent_blocking_request_s += blocking
        observed = decision.get("observations", [])
        population = sum(len(rank["requests"]) for ranks in observed for rank in ranks)
        if blocking and population:
            self.pause_fixed_s = max(self.pause_fixed_s, blocking / population)
        if not decision.get("deferred"):
            for epoch in decision.get("balance_evidence", {}).get("ready_epochs", []):
                self.epochs.pop(epoch["beneficiary"], None)


def plan_amortized_rebalance(observed, budgets, fit, allow_extrapolation, settings, lookup,
                            blocking_cost_s, phase):
    snapshot, horizon, population = admission_planning_snapshot(
        observed, 0, 0, 0, budgets, fit, allow_extrapolation, "ilp", settings["candidate_limit"],
        0, settings["move_bandwidth_gib_s"], lookup, new_request_count=0)
    fixed = population * settings["pause_fixed_s"] if phase == "before_pause" else 0
    credit = max(0, settings["credit_request_s"] - blocking_cost_s)
    snapshot["migration_activation_cost_s"] = fixed
    snapshot["migration_budget_s"] = credit
    reference = replay_candidate(snapshot, {"target": None, "moves": [], "status": "unchanged_layout"})
    plan = enumerate_targets(snapshot, max_moves=settings["max_moves"], time_limit_s=settings["solver_limit_s"])
    winner = plan["winner"]
    if winner is None or winner["objective_s"] >= reference["objective_s"]:
        winner = reference
    gate = amortized_gate(reference, winner, settings, population, blocking_cost_s, phase)
    return {"snapshot": snapshot, "horizon_steps": horizon, "active_population": population,
            "plan": plan, "reference": reference, "winner": winner, "gate": gate}


def amortized_gate(reference, winner, settings, population, blocking_cost_s, phase):
    fixed = population * settings["pause_fixed_s"] if phase == "before_pause" else 0
    migration = winner["migration_cost_s"] + winner.get("migration_activation_cost_s", 0)
    available = max(0, settings["credit_request_s"] - blocking_cost_s)
    gain = reference["performance_cost_s"] - winner["objective_s"]
    return {"accepted": bool(winner["moves"]) and gain > settings["min_gain_s"]
                         and migration <= available + 1e-6,
            "phase": phase, "predicted_net_gain_request_s": gain,
            "predicted_migration_request_s": migration,
            "fixed_control_request_s": fixed,
            "measured_blocking_so_far_request_s": blocking_cost_s,
            "available_credit_request_s": available,
            "min_gain_request_s": settings["min_gain_s"],
            "sunk_cost_excluded_from_continuation_profit": phase == "after_pause"}


def verify_amortized_history(decisions, config):
    epochs, spent = {}, 0
    pause = config.get("rebalance_pause_fixed_s")
    pause = config["move_fixed_s"] if pause is None else pause
    for decision in decisions:
        if decision["path"] != "release_rebalance_ilp":
            if decision.get("admitted") and decision.get("moves"):
                identifier = decision["request_id"]
                epochs[identifier] = {"beneficiary": identifier,
                                      "models": sorted({move["model"] for move in decision["moves"]})}
            continue
        evidence = decision.get("balance_evidence")
        if evidence is None:
            if decision["moves"] or decision.get("pause_call_s", 0):
                raise ValueError("Amortized migration lacks a displacement and inference-credit history")
            continue
        expected = [epochs[identifier] for identifier in sorted(epochs) if identifier in evidence["completed_ids"]]
        if (evidence["ready_epochs"] != expected or not math.isclose(evidence["spent_blocking_request_s"], spent, abs_tol=1e-8)
                or evidence["pause_fixed_s"] != pause):
            raise ValueError("Amortized policy reset credit or fabricated displacement/pause history")
        if decision.get("pause_call_s", 0):
            finished = decision["resume_completed_at"]
            blocking = sum(len(decision["observations"][model][rank]["requests"]) * max(0, finished - receipt["reached_at"])
                           for model, ranks in enumerate(decision["pause_receipts"]) for rank, receipt in enumerate(ranks))
            if not math.isclose(blocking, decision["active_blocking_upper_request_s"], abs_tol=1e-8):
                raise ValueError("Amortized pause charge differs from four-rank receipts")
            spent += blocking
            population = sum(len(rank["requests"]) for ranks in decision["observations"] for rank in ranks)
            if population:
                pause = max(pause, blocking / population)
        if not decision.get("deferred"):
            for epoch in expected:
                epochs.pop(epoch["beneficiary"], None)
