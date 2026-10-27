import math
import time

import ilp_reference
from headroom_rebalance import HeadroomInferenceLedger
from live_controller import admission_planning_snapshot
from workaware_cost import headroom_modes, reference_horizon_s, remaining_segments


class WorkawareInferenceLedger(HeadroomInferenceLedger):
    def __init__(self, rate_priors, origin, pause_fixed_s, quantile=.95, history_limit=64, prior_strength=2):
        super().__init__(.01, pause_fixed_s, quantile, history_limit)
        if (len(rate_priors) != 2 or any(not math.isfinite(rate) or rate <= 0 for rate in rate_priors)
                or not math.isfinite(origin) or not math.isfinite(prior_strength) or prior_strength <= 0):
            raise ValueError("Require frozen external rate priors and positive Poisson prior strength")
        self.rate_priors = list(rate_priors)
        self.origin = origin
        self.prior_strength = prior_strength
        self.arrival_counts = [0, 0]
        self.seen_arrivals = set()
        self.cold_costs = [[], []]

    def arrival(self, mark, arrived_at, observed_at):
        if mark["request_id"] in self.seen_arrivals or arrived_at < self.origin:
            raise ValueError("Duplicate or pre-origin arrival in online rate ledger")
        sample = super().arrival(mark, arrived_at, observed_at)
        self.seen_arrivals.add(mark["request_id"])
        self.arrival_counts[mark["model"]] += 1
        return sample

    def admission(self, decision, observed_at=None):
        decision["workaware_ledger_observed_at"] = time.monotonic() if observed_at is None else observed_at
        super().admission(decision)
        if not decision.get("admitted") or not decision.get("moves"):
            return
        model = decision["arriving_model"]
        observations = decision["observations"]
        population = 1 + sum(len(rank["requests"]) for ranks in observations for rank in ranks)
        control = decision.get("end_to_end_control_s_including_preplan",
                               decision["end_to_end_control_s"] + decision.get("preplan_s", 0))
        if model not in (0, 1) or not math.isfinite(control) or control < 0:
            raise ValueError("Invalid real admission-cost observation")
        self.cold_costs[model].append(population * control)
        del self.cold_costs[model][:-self.history_limit]

    def completion(self, request, observed_at=None):
        request["workaware_completion_observed_at"] = time.monotonic() if observed_at is None else observed_at
        super().completion(request)

    def settle(self, decision, observed_at=None):
        decision["workaware_settled_at"] = time.monotonic() if observed_at is None else observed_at
        super().settle(decision)

    def evidence(self, observations, sampled_at):
        if not math.isfinite(sampled_at) or sampled_at < self.origin:
            raise ValueError("Rate estimate predates workload origin")
        result = super().evidence(observations, sampled_at)
        elapsed = sampled_at - self.origin
        profiles = []
        for model, history in enumerate(self.arrivals):
            if any(sample["observed_at"] > sampled_at for sample in history):
                raise ValueError("Online model contains a future arrival")
            lengths = sorted(sample["total_tokens"] for sample in history)
            tokens = lengths[math.ceil(self.quantile * len(lengths)) - 1] if lengths else 0
            rate = ((self.prior_strength + self.arrival_counts[model])
                    / (self.prior_strength / self.rate_priors[model] + elapsed))
            profiles.append({"rate_rps": rate, "tokens": tokens, "observed_arrivals": self.arrival_counts[model],
                             "cold_cost_samples_s": list(self.cold_costs[model])})
        result["workaware_profile"] = {"origin": self.origin, "observed_duration_s": elapsed,
                                        "rate_priors_rps": list(self.rate_priors), "prior_strength": self.prior_strength,
                                        "models": profiles, "rate_scope": "past_counts_and_frozen_external_priors_no_future_trace",
                                        "migration_budget_policy": "net_gain_with_uncertainty_not_past_credit_hard_cap"}
        return result


def workaware_gate(reference, winner, settings, uncertainty_request_s, blocking_cost_s, phase):
    values = (uncertainty_request_s, blocking_cost_s, settings["min_gain_s"])
    if any(not math.isfinite(value) or value < 0 for value in values) or phase not in ("before_pause", "after_pause"):
        raise ValueError("Invalid gain uncertainty, control history or planning phase")
    gain = reference["objective_s"] - winner["objective_s"]
    return {"accepted": bool(winner["moves"]) and gain > settings["min_gain_s"] + uncertainty_request_s,
            "predicted_net_gain_request_s": gain, "uncertainty_request_s": uncertainty_request_s,
            "min_gain_request_s": settings["min_gain_s"], "phase": phase,
            "measured_blocking_so_far_request_s": blocking_cost_s,
            "sunk_cost_excluded_from_continuation_profit": phase == "after_pause",
            "migration_budget_policy": "net_gain_with_uncertainty_not_past_credit_hard_cap"}


def plan_workaware_rebalance(observed, budgets, fit, allow_extrapolation, settings, lookup,
                            blocking_cost_s, phase):
    if not getattr(ilp_reference, "SUPPORTS_WORK_SEGMENTS", False):
        raise RuntimeError("Work-aware staged ILP integration is not installed; never omit remaining-work costs")
    calibration = settings["workaware_fit"]
    segments = remaining_segments(observed, calibration, settings.get("work_interval_steps", 128))
    snapshot, _, population = admission_planning_snapshot(
        observed, 0, 0, 0, budgets, fit, allow_extrapolation, "ilp", settings["candidate_limit"],
        0, settings["move_bandwidth_gib_s"], None, new_request_count=0)
    snapshot["performance_cost_tables_s"] = [[0] * len(table) for table in snapshot["performance_cost_tables_s"]]
    snapshot["work_segments"] = segments
    snapshot["migration_activation_cost_s"] = population * settings["pause_fixed_s"] if phase == "before_pause" else 0
    profiles = settings["workaware_profile"]["models"]
    cold_costs = []
    for model, profile in enumerate(profiles):
        samples = profile["cold_cost_samples_s"]
        tokens = profile["tokens"]
        observation = observed[model][0]
        size_gib = math.ceil(tokens / observation["block_size"]) * observation["block_bytes"] / 1024**3
        prior = (population + 1) * (settings["pause_fixed_s"] + size_gib / settings["move_bandwidth_gib_s"])
        cold_costs.append(max(prior, sum(samples) / len(samples)) if samples else prior)
    horizon = reference_horizon_s(segments, calibration)
    scenarios = headroom_modes(snapshot, observed, profiles, horizon, cold_costs)
    plans, references = [], []
    for scenario in scenarios:
        selected = {**scenario["snapshot"], "placement_risk_cost_s": scenario["risk_cost_s"]}
        if all(used <= cap + 1e-6 for used, cap in zip(selected["initial_gib"], selected["final_capacity_gib"])):
            references.append({"snapshot": selected, "reference": ilp_reference.replay_candidate(
                selected, {"target": None, "moves": [], "status": "unchanged_layout"})})
        plan = ilp_reference.enumerate_targets(selected, max_moves=settings["max_moves"],
                                             time_limit_s=settings["solver_limit_s"] / max(1, len(scenarios)))
        plans.append({**scenario, "snapshot": selected, "plan": plan})
    reference_entry = min(references, key=lambda candidate: candidate["reference"]["objective_s"])
    reference = reference_entry["reference"]
    selected_plan = min((plan for plan in plans if plan["plan"]["winner"] is not None),
                        key=lambda plan: plan["plan"]["winner"]["objective_s"], default=None)
    winner = reference if selected_plan is None else selected_plan["plan"]["winner"]
    if winner["objective_s"] >= reference["objective_s"]:
        winner, selected_plan = reference, None
    if "population_profiles" in calibration:
        from workaware_profile_cost import changed_segment_uncertainty

        uncertainty = changed_segment_uncertainty(segments, winner["moves"])
    else:
        uncertainty = sum(segment["uncertainty_request_s"] for segment in segments)
    return {"snapshot": reference_entry["snapshot"] if selected_plan is None else selected_plan["snapshot"],
            "unconstrained_snapshot": snapshot, "horizon_steps": max((segment["end_step"] for segment in segments), default=0),
            "forecast_horizon_s": horizon, "active_population": population,
            "reference": reference, "winner": winner, "work_segments": segments,
            "plan": {"status": "workaware_soft_headroom_enumerated", "winner": winner, "scenarios": plans},
            "gate": workaware_gate(reference, winner, settings, uncertainty, blocking_cost_s, phase),
            "cost_scope": "remaining_decode_graph_and_context_weighted_attention_with_Poisson_headroom_risk"}
