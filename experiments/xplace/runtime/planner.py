import math
import random

from cost_model import capture_bucket, integrated_cost

GIB = 1024**3


def build_state(observed, budgets):
    """Compact planning view of full placement snapshots (observed[model][rank])."""
    num_ranks = len(observed[0])
    used = [sum(observed[model][rank]["mapped_bytes"] for model in (0, 1)) / GIB for rank in range(num_ranks)]
    requests, pools = [], {}
    for model in (0, 1):
        for rank in range(num_ranks):
            observation = observed[model][rank]
            pools[(model, rank)] = {
                "block_size": observation["block_size"], "page_block_capacity": observation["page_block_capacity"],
                "page_gib": observation["page_bytes"] / GIB, "block_gib": observation["block_bytes"] / GIB,
                "free_slots": observation["mapped_free_block_slots"], "free_blocks": observation["free_blocks"],
                "pages": [(frozenset(page["owners"]), page["pinned"], page["memory_gib"])
                          for page in observation.get("pages", [])]}
            for request in observation["requests"]:
                blocks = (request["num_blocks"] if "num_blocks" in request else len(request["blocks"])
                          if "blocks" in request else
                          math.ceil((request["prompt_tokens"] + request["max_tokens"]) / observation["block_size"]))
                requests.append({"id": request["id"], "model": model, "rank": rank,
                                 "context": request["computed_tokens"],
                                 "remaining": max(0, request["max_tokens"] - request["output_tokens"]),
                                 "blocks": blocks, "movable": bool(request.get("movable", request.get("decoding")))})
    return {"num_ranks": num_ranks, "budgets": [float(value) for value in budgets], "used": used,
            "requests": requests, "pools": pools}


def pool_increment(pool, blocks):
    if blocks <= 0:
        return 0.0
    return math.ceil(max(0, blocks - pool["free_slots"]) / pool["page_block_capacity"]) * pool["page_gib"]


def released_gib(state, rank, identifiers):
    return sum(gib for model in (0, 1) for owners, pinned, gib in state["pools"][(model, rank)]["pages"]
               if owners and not pinned and owners <= identifiers)


def plan_usage(state, moves, arrival=None, target=None):
    """Per-rank KV usage after moves (+ arrival at target) under the conservative page model, or None."""
    used = list(state["used"])
    sources, added = {}, {}
    for move in moves:
        sources.setdefault(move["source"], set()).add(move["id"])
        key = (move["model"], move["destination"])
        added[key] = added.get(key, 0) + move["blocks"]
    for rank, identifiers in sources.items():
        used[rank] -= released_gib(state, rank, frozenset(identifiers))
    for (model, rank), blocks in added.items():
        pool = state["pools"][(model, rank)]
        if blocks > pool["free_blocks"]:
            return None
        used[rank] += pool_increment(pool, blocks)
    if arrival is not None:
        pool = state["pools"][(arrival["model"], target)]
        blocks = math.ceil(arrival["tokens"] / pool["block_size"])
        if blocks > pool["free_blocks"]:
            return None
        used[target] += pool_increment(pool, blocks)
    if any(value > budget + 1e-6 for value, budget in zip(used, state["budgets"])):
        return None
    return used


def layout(state, moves=(), arrival=None, target=None):
    """(rank, context, remaining) per tenant after moves and an optional arrival."""
    destination = {move["id"]: move["destination"] for move in moves}
    tenants = {0: [], 1: []}
    for request in state["requests"]:
        tenants[request["model"]].append((destination.get(request["id"], request["rank"]),
                                          request["context"], request["remaining"]))
    if arrival is not None:
        tenants[arrival["model"]].append((target, arrival["prompt_tokens"], arrival["output_tokens"]))
    return tenants


def layout_cost(state, model, tenants, horizon_steps):
    return sum(integrated_cost(model, tenant, requests, state["num_ranks"], horizon_steps)
               for tenant, requests in tenants.items())


def migration_cost(state, moves, params):
    if not moves:
        return 0.0
    copy = sum(params["move_fixed_s"] + move["blocks"] * state["pools"][(move["model"], move["source"])]["block_gib"]
               / params["copy_gib_s"] for move in moves)
    return len(state["requests"]) * (params["pause_fixed_s"] + copy)


def horizon_steps(state, model, params):
    tenants = layout(state)
    counts = [0] * state["num_ranks"]
    contexts = [0.0] * state["num_ranks"]
    for rank, context, _ in tenants[1]:
        counts[rank] += 1
        contexts[rank] += context
    step = model.decode_step(1, counts, contexts) or 0.02
    return max(1, int(params["horizon_s"] / step))


def _assign(state, model, selected, target, used, rng, cost_aware, arrival=None):
    """Choose destinations for requests leaving `target`; returns moves or None if capacity blocks."""
    num_ranks = state["num_ranks"]
    moving = {request["id"] for request in selected}
    counts = {tenant: [0] * num_ranks for tenant in (0, 1)}
    contexts = {tenant: [0.0] * num_ranks for tenant in (0, 1)}
    for request in state["requests"]:
        if request["id"] not in moving:
            counts[request["model"]][request["rank"]] += 1
            contexts[request["model"]][request["rank"]] += request["context"]
    if arrival is not None:
        counts[arrival["model"]][target] += 1
        contexts[arrival["model"]][target] += arrival["prompt_tokens"]
    added = {}
    moves = []
    order = sorted(selected, key=lambda request: (-request["context"], request["id"])) if cost_aware == "cost" else selected
    for request in order:
        options = []
        for destination in range(num_ranks):
            if destination == target:
                continue
            key = (request["model"], destination)
            pool = state["pools"][key]
            blocks = added.get(key, 0) + request["blocks"]
            if blocks > pool["free_blocks"]:
                continue
            increment = pool_increment(pool, blocks) - pool_increment(pool, added.get(key, 0))
            if used[destination] + increment > state["budgets"][destination] + 1e-6:
                continue
            options.append((destination, increment))
        if not options:
            return None
        if cost_aware == "cost":
            tenant = request["model"]

            def score(option):
                destination = option[0]
                trial_counts = list(counts[tenant])
                trial_contexts = list(contexts[tenant])
                trial_counts[destination] += 1
                trial_contexts[destination] += request["context"]
                return (model.placement_terms(tenant, trial_counts, trial_contexts),
                        capture_bucket(trial_counts[destination]), trial_contexts[destination], destination)

            destination, increment = min(options, key=score)
        elif cost_aware == "bestfit":
            destination, increment = min(options, key=lambda option: (
                state["budgets"][option[0]] - used[option[0]] - option[1], option[0]))
        else:
            destination, increment = rng.choice(options)
        key = (request["model"], destination)
        added[key] = added.get(key, 0) + request["blocks"]
        used[destination] += increment
        counts[request["model"]][destination] += 1
        contexts[request["model"]][destination] += request["context"]
        moves.append({"id": request["id"], "model": request["model"], "source": target,
                      "destination": destination, "blocks": request["blocks"]})
    return moves


def _select_for_target(state, target, arrival, ordered, max_moves):
    """Shortest prefix of `ordered` whose release lets the arrival fit on `target` (ignoring destinations)."""
    selected = []
    for request in [None] + ordered:
        if request is not None:
            if len(selected) >= max_moves:
                return None
            selected.append(request)
        identifiers = frozenset(item["id"] for item in selected)
        pool = state["pools"][(arrival["model"], target)]
        blocks = math.ceil(arrival["tokens"] / pool["block_size"])
        if blocks > pool["free_blocks"]:
            return None
        used = state["used"][target] - released_gib(state, target, identifiers) + pool_increment(pool, blocks)
        if used <= state["budgets"][target] + 1e-6:
            return selected
    return None


ORDERINGS = {
    "kv_desc": lambda request: (-request["blocks"], request["id"]),
    "remaining_asc": lambda request: (request["remaining"], -request["blocks"], request["id"]),
    "remaining_desc": lambda request: (-request["remaining"], -request["blocks"], request["id"]),
}


def plan_admission(state, arrival, policy, model, params, seed=0):
    """Plan moves that let `arrival` ({model, tokens, prompt_tokens, output_tokens}) fit on some rank.

    policy: "mig_random" (random feasible target, random destinations), "mig_bestfit" (KV-only: fewest moves,
    best-fit destinations) or "ca" (cost-aware). Only the other tenant's requests are moved.
    """
    if policy not in ("mig_random", "mig_bestfit", "ca"):
        raise ValueError(f"Unknown admission policy {policy}")
    rng = random.Random(seed)
    num_ranks = state["num_ranks"]
    movable = [request for request in state["requests"] if request["movable"] and request["model"] != arrival["model"]]
    horizon = horizon_steps(state, model, params)
    candidates = []
    targets = list(range(num_ranks))
    if policy == "mig_random":
        rng.shuffle(targets)
    mode = {"mig_random": "random", "mig_bestfit": "bestfit", "ca": "cost"}[policy]
    for target in targets:
        on_target = [request for request in movable if request["rank"] == target]
        orderings = list(ORDERINGS) if policy == "ca" else ["kv_desc"]
        for name in orderings:
            selected = _select_for_target(state, target, arrival,
                                          sorted(on_target, key=ORDERINGS[name]), params["max_moves"])
            if selected is None:
                continue
            moves = _assign(state, model, selected, target, list(state["used"]), rng, mode, arrival)
            if moves is None:
                continue
            used = plan_usage(state, moves, arrival, target)
            if used is None:
                continue
            candidate = {"target": target, "moves": moves, "ordering": name, "used_after_gib": used}
            if policy == "mig_random":
                candidate["selection"] = "first_feasible_in_random_target_order"
                return {**candidate, "candidates_evaluated": len(candidates) + 1, "horizon_steps": horizon}
            if policy == "mig_bestfit":
                candidate["objective_s"] = float(len(moves))
                candidates.append(candidate)
                continue
            performance = layout_cost(state, model, layout(state, moves, arrival, target), horizon)
            candidate.update(performance_cost_s=performance, migration_cost_s=migration_cost(state, moves, params),
                             objective_s=performance + migration_cost(state, moves, params))
            candidates.append(candidate)
    if not candidates:
        return {"target": None, "moves": [], "reason": "no_feasible_plan", "horizon_steps": horizon}
    best = min(candidates, key=lambda item: (item["objective_s"], len(item["moves"]), item["target"]))
    return {**best, "candidates_evaluated": len(candidates), "horizon_steps": horizon,
            "alternatives": [{key: item.get(key) for key in ("target", "ordering", "objective_s", "performance_cost_s",
                                                              "migration_cost_s")} | {"moves": len(item["moves"])}
                             for item in candidates]}


def _headroom_ranks(state, used, need_gib):
    return [rank for rank in range(state["num_ranks"]) if state["budgets"][rank] - used[rank] >= need_gib - 1e-6]


def future_admission_cost(state, used, params):
    """Request-seconds a future Code admission would pay to migrate if no rank keeps enough headroom."""
    conv = [request for request in state["requests"] if request["model"] == 1]
    if not conv:
        return 0.0
    average_gib = sum(request["blocks"] for request in conv) / len(conv) * state["pools"][(1, 0)]["block_gib"]
    best_free = max(budget - value for budget, value in zip(state["budgets"], used))
    moves = max(0, math.ceil((params["code_need_gib"] - best_free) / max(average_gib, 1e-9)))
    if not moves:
        return 0.0
    return (len(state["requests"]) + 1) * (params["pause_fixed_s"]
                                           + moves * (params["move_fixed_s"] + average_gib / params["copy_gib_s"]))


def _level_moves(state, tenant, level, key, max_moves):
    """Moves bringing every rank's `tenant` count down to `level`, filling the lightest ranks first."""
    num_ranks = state["num_ranks"]
    counts = [0] * num_ranks
    for request in state["requests"]:
        if request["model"] == tenant:
            counts[request["rank"]] += 1
    moves = []
    for source in sorted(range(num_ranks), key=lambda rank: -counts[rank]):
        excess = counts[source] - level
        if excess <= 0:
            continue
        pool = sorted([request for request in state["requests"] if request["model"] == tenant
                       and request["rank"] == source and request["movable"]], key=key)
        for request in pool[:excess]:
            destination = min((rank for rank in range(num_ranks) if rank != source and counts[rank] < level),
                              key=lambda rank: (counts[rank], rank), default=None)
            if destination is None:
                return None
            moves.append({"id": request["id"], "model": tenant, "source": source,
                          "destination": destination, "blocks": request["blocks"]})
            counts[source] -= 1
            counts[destination] += 1
            if len(moves) > max_moves:
                return None
    return moves


RETURN_KEYS = {
    "remaining_desc": lambda request: (-request["remaining"], -request["context"], request["id"]),
    "context_desc": lambda request: (-request["context"], -request["remaining"], request["id"]),
}


def plan_return(state, model, params):
    """Cost-aware rebalance after a release: level per-rank counts when gain > migration cost + headroom risk."""
    num_ranks = state["num_ranks"]
    horizon = horizon_steps(state, model, params)
    base_cost = layout_cost(state, model, layout(state), horizon)
    need = params.get("code_need_gib", 0.0)
    before = _headroom_ranks(state, state["used"], need) if need else []
    probability = 1 - math.exp(-params.get("code_rate", 0.0) * params["horizon_s"])
    decisions, seen = [], set()
    for tenant in params.get("return_tenants", (1,)):
        counts = [0] * num_ranks
        for request in state["requests"]:
            if request["model"] == tenant:
                counts[request["rank"]] += 1
        floor = math.ceil(sum(counts) / num_ranks)
        for level in range(max(counts) - 1, floor - 1, -1):
            for name, key in RETURN_KEYS.items():
                moves = _level_moves(state, tenant, level, key, params["return_max_moves"])
                if not moves:
                    continue
                identity = tuple(sorted((move["id"], move["destination"]) for move in moves))
                if identity in seen:
                    continue
                seen.add(identity)
                used = plan_usage(state, moves)
                if used is None:
                    continue
                gain = base_cost - layout_cost(state, model, layout(state, moves), horizon)
                cost = migration_cost(state, moves, params)
                after = _headroom_ranks(state, used, need) if need else []
                risk = probability * future_admission_cost(state, used, params) if before and not after else 0.0
                decisions.append({"tenant": tenant, "level": level, "ordering": name, "moves": moves,
                                  "gain_s": gain, "migration_cost_s": cost, "headroom_risk_s": risk,
                                  "net_s": gain - cost - risk, "used_after_gib": used})
    accepted = [item for item in decisions
                if item["net_s"] > params["return_margin_s"] + params["return_margin_ratio"] * item["migration_cost_s"]]
    chosen = max(accepted, key=lambda item: (item["net_s"], -len(item["moves"]))) if accepted else None
    return {"moves": chosen["moves"] if chosen else [], "accepted": chosen is not None, "base_cost_s": base_cost,
            "horizon_steps": horizon, "decisions": [{key: value for key, value in item.items() if key != "moves"}
                                                    | {"moves": len(item["moves"])} for item in decisions],
            "chosen": {key: value for key, value in chosen.items() if key != "moves"} if chosen else None}


DEFAULT_PARAMS = {"max_moves": 8, "return_max_moves": 12, "horizon_s": 90.0, "pause_fixed_s": 1.0,
                  "move_fixed_s": 0.05, "copy_gib_s": 10.0, "return_margin_s": 5.0, "return_margin_ratio": 0.5,
                  "code_rate": 0.002, "code_need_gib": 9.375, "migrate_min_gib": 4.0}
