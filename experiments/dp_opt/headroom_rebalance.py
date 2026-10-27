import itertools
import math

from amortized_rebalance import MigrationInferenceLedger, amortized_gate
from ilp_reference import enumerate_targets


class HeadroomInferenceLedger(MigrationInferenceLedger):
    def __init__(self, overhead_fraction, pause_fixed_s, quantile=.95, history_limit=64):
        super().__init__(overhead_fraction, pause_fixed_s)
        if (not math.isfinite(quantile) or not 0 < quantile <= 1
                or type(history_limit) is not int or history_limit < 1):
            raise ValueError("Invalid online headroom profile")
        self.quantile = quantile
        self.history_limit = history_limit
        self.arrivals = [[], []]

    def arrival(self, mark, arrived_at, observed_at):
        if (mark["model"] not in (0, 1) or any(type(mark[name]) is not int or mark[name] <= 0
                for name in ("prompt_tokens", "output_tokens"))
                or not all(math.isfinite(value) for value in (arrived_at, observed_at))
                or observed_at < arrived_at):
            raise ValueError("Headroom history requires an actually dispatched past arrival")
        sample = {"request_id": mark["request_id"], "model": mark["model"],
                  "total_tokens": mark["prompt_tokens"] + mark["output_tokens"],
                  "arrived_at": arrived_at, "observed_at": observed_at}
        history = self.arrivals[mark["model"]]
        if any(item["request_id"] == sample["request_id"] for item in history):
            raise ValueError("Duplicate online arrival")
        history.append(sample)
        del history[:-self.history_limit]
        return sample

    def evidence(self, observations, sampled_at):
        result = super().evidence(observations, sampled_at)
        result["headroom_profile"] = {"quantile": self.quantile, "history_limit": self.history_limit,
                                      "arrivals": [list(history) for history in self.arrivals]}
        return result


def headroom_scenarios(snapshot, observed, profile):
    quantile, limit = profile["quantile"], profile["history_limit"]
    if (not math.isfinite(quantile) or not 0 < quantile <= 1 or type(limit) is not int
            or limit < 1 or len(profile["arrivals"]) != 2):
        raise ValueError("Invalid bounded past-arrival headroom profile")
    guards = []
    for model, history in enumerate(profile["arrivals"]):
        if not history:
            continue
        if (len(history) > limit or len({item["request_id"] for item in history}) != len(history)
                or any(item["model"] != model or type(item["total_tokens"]) is not int
                       or item["total_tokens"] <= 0 for item in history)):
            raise ValueError("Malformed online headroom history")
        lengths = sorted(item["total_tokens"] for item in history)
        tokens = lengths[math.ceil(quantile * len(lengths)) - 1]
        demands = []
        for rank in observed[model]:
            if (not rank.get("uniform_page_geometry") or any(rank[name] <= 0
                    for name in ("block_size", "page_block_capacity", "page_bytes"))):
                raise ValueError("Headroom certification requires positive uniform page geometry")
            blocks = math.ceil(tokens / rank["block_size"])
            demands.append(math.ceil(blocks / rank["page_block_capacity"]) * rank["page_bytes"] / 1024**3)
        targets = [rank for rank in (0, 1) if snapshot["initial_gib"][rank] + demands[rank]
                   <= snapshot["capacity_gib"][rank] + 1e-6]
        if targets:
            guards.append({"model": model, "total_tokens": tokens, "demand_gib": demands,
                           "reference_safe_ranks": targets})
    scenarios = []
    for ranks in itertools.product((0, 1), repeat=len(guards)):
        capacities = list(snapshot["capacity_gib"])
        choices = []
        for guard, rank in zip(guards, ranks):
            capacities[rank] = min(capacities[rank], snapshot["capacity_gib"][rank] - guard["demand_gib"][rank])
            choices.append({"model": guard["model"], "rank": rank})
        if any(capacity < 0 for capacity in capacities):
            continue
        scenarios.append({"guards": choices, "snapshot": {**snapshot, "final_capacity_gib": capacities}})
    return guards, scenarios


def plan_headroom_rebalance(observed, budgets, fit, allow_extrapolation, settings, lookup,
                            blocking_cost_s, phase):
    from live_controller import admission_planning_snapshot
    from ilp_reference import replay_candidate

    snapshot, horizon, population = admission_planning_snapshot(
        observed, 0, 0, 0, budgets, fit, allow_extrapolation, "ilp", settings["candidate_limit"],
        0, settings["move_bandwidth_gib_s"], lookup, new_request_count=0)
    snapshot["migration_activation_cost_s"] = population * settings["pause_fixed_s"] if phase == "before_pause" else 0
    snapshot["migration_budget_s"] = max(0, settings["credit_request_s"] - blocking_cost_s)
    reference = replay_candidate(snapshot, {"target": None, "moves": [], "status": "unchanged_layout"})
    guards, scenarios = headroom_scenarios(snapshot, observed, settings["headroom_profile"])
    plans = []
    winner, selected = reference, snapshot
    for scenario in scenarios:
        plan = enumerate_targets(scenario["snapshot"], max_moves=settings["max_moves"],
                                 time_limit_s=settings["solver_limit_s"] / max(1, len(scenarios)))
        plans.append({**scenario, "plan": plan})
        candidate = plan["winner"]
        if candidate is not None and candidate["objective_s"] < winner["objective_s"]:
            winner, selected = candidate, scenario["snapshot"]
    gate = amortized_gate(reference, winner, settings, population, blocking_cost_s, phase)
    return {"snapshot": selected, "unconstrained_snapshot": snapshot, "horizon_steps": horizon,
            "active_population": population, "plan": {"status": "headroom_enumerated",
            "winner": winner, "scenarios": plans}, "reference": reference, "winner": winner,
            "gate": gate, "headroom_guards": guards}


def verify_headroom_history(decisions, requests, config, origin, marks, arrivals):
    from amortized_rebalance import verify_amortized_history

    verify_amortized_history(decisions, config)
    expected = {}
    for sample in arrivals:
        identifier = sample["request_id"]
        mark = marks.get(identifier)
        if (mark is None or identifier in expected or sample["model"] != mark["model"]
                or sample["total_tokens"] != mark["prompt_tokens"] + mark["output_tokens"]
                or not all(math.isfinite(sample[field]) for field in ("arrived_at", "observed_at"))
                or sample["arrived_at"] != origin + mark["arrival_s"]
                or sample["observed_at"] < sample["arrived_at"]):
            raise ValueError("Online headroom sample differs from its real dispatched arrival")
        expected[identifier] = sample
    if set(expected) != set(requests):
        raise ValueError("Headroom policy must record every real arrival exactly once")
    for decision in decisions:
        evidence = decision.get("balance_evidence")
        if evidence is None:
            continue
        profile = evidence["headroom_profile"]
        if (profile["quantile"] != config.get("rebalance_headroom_quantile", .95)
                or profile["history_limit"] != config.get("rebalance_headroom_history_limit", 64)):
            raise ValueError("Headroom policy changed its frozen tail profile")
        histories = [[sample for sample in arrivals if sample["model"] == model
                      and sample["observed_at"] <= evidence["sampled_at"]][-profile["history_limit"]:]
                     for model in (0, 1)]
        if profile["arrivals"] != histories:
            raise ValueError("Headroom planning skipped past arrivals or looked ahead in the trace")
