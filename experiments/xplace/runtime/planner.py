import math
import random

from cost_model import capture_bucket, integrated_cost
from scoring_state import capture_scoring_observations

GIB = 1024**3
# Output lengths are unknown online: every resident is assumed to keep decoding through the planning horizon.
UNKNOWN_REMAINING = 10**9


def build_state(observed, budgets, growth_gib=(0.0, 0.0), slack_gib=0.0, target_growth_gib=None,
                request_growth=None, arrival_growth_gib=0.0):
    """Compact planning view of full placement snapshots (observed[model][rank]).

    growth_gib[model]: KV one resident request of `model` may add within the planning window (decode rate x window,
    never its unknown final length). Ranks whose usage a plan increases must keep that growth plus `slack_gib` free.
    target_growth_gib: the same allowance on the rank receiving an arrival (defaults to growth_gib).
    request_growth: general (type-agnostic) planning -- per-request growth in GiB keyed by request id, replacing the
    per-model allowance; arrival_growth_gib is the arriving request's own growth.
    """
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
                          math.ceil(request["computed_tokens"] / observation["block_size"]))
                requests.append({"id": request["id"], "model": model, "rank": rank,
                                 "context": request["computed_tokens"], "remaining": UNKNOWN_REMAINING,
                                 "blocks": blocks, "movable": bool(request.get("movable", request.get("decoding")))})
    target_growth_gib = growth_gib if target_growth_gib is None else target_growth_gib
    state = {"num_ranks": num_ranks, "budgets": [float(value) for value in budgets], "used": used,
             "requests": requests, "pools": pools, "growth_gib": [float(value) for value in growth_gib],
             "scoring_observations": capture_scoring_observations(observed),
             "target_growth_gib": [float(value) for value in target_growth_gib], "slack_gib": float(slack_gib)}
    if request_growth is not None:
        state.update(general=True, request_growth=dict(request_growth), arrival_growth=float(arrival_growth_gib))
    return state


def rank_growth(state, moves=(), arrival=None, target=None):
    """General planning: per-rank sum of request growth (GiB) after `moves` (+ the arrival on `target`)."""
    destination = {move["id"]: move["destination"] for move in moves}
    totals = [0.0] * state["num_ranks"]
    for request in state["requests"]:
        totals[destination.get(request["id"], request["rank"])] += state["request_growth"].get(request["id"], 0.0)
    if arrival is not None:
        totals[target] += state["arrival_growth"]
    return totals


def freed_for(state, rank, identifiers, model):
    """General planning: what the requests in `identifiers` leaving `rank` free for a `model` request there.

    Returns (KV released, free blocks `model` gains, mapped holes `model` gains). Blocks of `model`'s own requests
    become holes in its pages at once; another tenant's page helps only once all its owners left (the page returns to
    the shared physical pool).
    """
    released, own_pages, other_pages = 0.0, 0, 0
    for owner in (0, 1):
        for owners, pinned, gib in state["pools"][(owner, rank)]["pages"]:
            if owners and not pinned and owners <= identifiers:
                released += gib
                if owner == model:
                    own_pages += 1
                else:
                    other_pages += 1
    capacity = state["pools"][(model, rank)]["page_block_capacity"]
    own_blocks = sum(request["blocks"] for request in state["requests"]
                     if request["id"] in identifiers and request["model"] == model and request["rank"] == rank)
    return released, own_blocks + other_pages * capacity, own_blocks - own_pages * capacity


def headroom_gib(state, counts, rank, target=False):
    """Growth allowance a rank must keep free for its residents (counts[model][rank]) within the planning window;
    `target` marks the rank receiving the arrival."""
    growth = state.get("target_growth_gib", state["growth_gib"]) if target else state["growth_gib"]
    return sum(counts[model][rank] * growth[model] for model in (0, 1)) + state["slack_gib"]


def resident_counts(state, moves=(), arrival=None, target=None):
    destination = {move["id"]: move["destination"] for move in moves}
    counts = [[0] * state["num_ranks"] for _ in (0, 1)]
    for request in state["requests"]:
        counts[request["model"]][destination.get(request["id"], request["rank"])] += 1
    if arrival is not None:
        counts[arrival["model"]][target] += 1
    return counts


def conv_growth_tokens(model, params, conv_counts, conv_contexts, arrival_prompt=None, isolate=False,
                       slice_cfg=None, floor=True):
    """Tokens one resident Conv may decode within the planning window (output length unknown, so rate x time).

    At least `relief_horizon_s` at the solo rate, so a rank receiving requests keeps room until the next rebalance.
    For a long arrival, also its residency (calibrated prefill time + `decode_window_s`) at the rate Conv runs during
    it: interfered when the tenants co-run, only within slices under time-multiplexing, zero for a plain hold.
    floor=False drops the relief-horizon floor: residents of the arrival's own rank only grow during its residency,
    and its KV is released when it completes.
    """
    solo = model.decode_step(1, conv_counts, conv_contexts) or 0.02
    tokens = params["relief_horizon_s"] / solo if floor else 0.0
    if arrival_prompt:
        prefill = model.params.get("code_prefill", {})
        per_100k = prefill.get("solo_s_per_100k" if isolate else "with_other_s_per_100k", 54.0)
        if isinstance(per_100k, dict):
            per_100k = max(float(value) for value in per_100k.values())
        window = arrival_prompt / 100000 * float(per_100k) + params["decode_window_s"]
        if not isolate:
            during = window / (model.decode_step(1, conv_counts, conv_contexts, "prefill") or solo)
        elif slice_cfg:
            period = slice_cfg[2] if len(slice_cfg) == 3 else slice_cfg[0]
            during = (math.ceil(window / period) + 1) * slice_cfg[1] / solo
        else:
            during = 0.0
        tokens = max(tokens, during)
    return tokens


def pool_increment(pool, blocks):
    if blocks <= 0:
        return 0.0
    return math.ceil(max(0, blocks - pool["free_slots"]) / pool["page_block_capacity"]) * pool["page_gib"]


def released_gib(state, rank, identifiers):
    return sum(gib for model in (0, 1) for owners, pinned, gib in state["pools"][(model, rank)]["pages"]
               if owners and not pinned and owners <= identifiers)


def plan_usage(state, moves, arrival=None, target=None):
    """Per-rank KV usage after moves (+ arrival at target) under the conservative page model, or None.

    Ranks whose usage grows must also keep the growth headroom of their residents after the plan.
    """
    used = list(state["used"])
    sources, added, released = {}, {}, {}
    for move in moves:
        sources.setdefault(move["source"], set()).add(move["id"])
        key = (move["model"], move["destination"])
        added[key] = added.get(key, 0) + move["blocks"]
    for rank, identifiers in sources.items():
        released[rank] = released_gib(state, rank, frozenset(identifiers))
        used[rank] -= released[rank]
    for (model, rank), blocks in added.items():
        pool = state["pools"][(model, rank)]
        if blocks > pool["free_blocks"]:
            return None
        used[rank] += pool_increment(pool, blocks)
    if arrival is not None:
        pool = state["pools"][(arrival["model"], target)]
        blocks = math.ceil(arrival["tokens"] / pool["block_size"])
        if state.get("general"):
            _, gained, holes = freed_for(state, target, frozenset(sources.get(target, ())), arrival["model"])
            if blocks > pool["free_blocks"] + gained:
                return None
            used[target] += pool_increment(dict(pool, free_slots=pool["free_slots"] + holes), blocks)
        else:
            if blocks > pool["free_blocks"] + int(released.get(target, 0.0) / pool["block_gib"] + 1e-9):
                return None
            used[target] += pool_increment(pool, blocks)
    if state.get("general"):
        growth = rank_growth(state, moves, arrival, target)
        for rank, (value, budget) in enumerate(zip(used, state["budgets"])):
            grew = value > state["used"][rank] + 1e-9
            if value + (growth[rank] + state["slack_gib"] if grew else 0.0) > budget + 1e-6:
                return None
        return used
    counts = resident_counts(state, moves, arrival, target)
    for rank, (value, budget) in enumerate(zip(used, state["budgets"])):
        grew = value > state["used"][rank] + 1e-9
        headroom = headroom_gib(state, counts, rank, target=arrival is not None and rank == target) if grew else 0.0
        if value + headroom > budget + 1e-6:
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
        tenants[arrival["model"]].append((target, arrival["prompt_tokens"], UNKNOWN_REMAINING))
    return tenants


def layout_cost(state, model, tenants, horizon_steps):
    """horizon_steps: one horizon for all tenants, or {tenant: steps} (general planning)."""
    return sum(integrated_cost(model, tenant, requests, state["num_ranks"],
                               horizon_steps[tenant] if isinstance(horizon_steps, dict) else horizon_steps)
               for tenant, requests in tenants.items())


def imbalance_objective_enabled(state, params):
    enabled = params.get("score_imbalance_cost", True)
    if not isinstance(enabled, bool):
        raise ValueError("score_imbalance_cost must be boolean")
    if not enabled and state["num_ranks"] != 2:
        raise ValueError("Strict imbalance-cost ablation requires two attention groups")
    return enabled


def migration_cost(state, moves, params):
    """Request-seconds paused by a transaction. General planning counts only the tenants whose requests move (only
    they are paused); the legacy planner counts every request."""
    if not moves:
        return 0.0
    copy = sum(params["move_fixed_s"] + move["blocks"] * state["pools"][(move["model"], move["source"])]["block_gib"]
               / params["copy_gib_s"] for move in moves)
    if state.get("general"):
        paused = {move["model"] for move in moves}
        return sum(request["model"] in paused for request in state["requests"]) * (params["pause_fixed_s"] + copy)
    return len(state["requests"]) * (params["pause_fixed_s"] + copy)


def horizon_steps(state, model, params):
    """Decode steps in the planning window: the Conv tenant's (legacy), or each tenant's own (general planning)."""
    tenants = layout(state)
    horizons = {}
    for tenant in ((0, 1) if state.get("general") else (1,)):
        counts = [0] * state["num_ranks"]
        contexts = [0.0] * state["num_ranks"]
        for rank, context, _ in tenants[tenant]:
            counts[rank] += 1
            contexts[rank] += context
        step = model.decode_step(tenant, counts, contexts) or 0.02
        horizons[tenant] = max(1, int(params["horizon_s"] / step))
    return horizons if state.get("general") else horizons[1]


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
    general = state.get("general")
    growth = rank_growth(state) if general else None  # destinations are never the target: no arrival, no movers
    order = sorted(selected, key=lambda request: (-request["context"], request["id"])) if cost_aware == "cost" else selected
    for request in order:
        own = state["request_growth"].get(request["id"], 0.0) if general else 0.0
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
            if general:
                headroom = growth[destination] + own + state["slack_gib"]
            else:
                trial = {tenant: list(values) for tenant, values in counts.items()}
                trial[request["model"]][destination] += 1
                headroom = headroom_gib(state, trial, destination)
            if used[destination] + increment + headroom > state["budgets"][destination] + 1e-6:
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
        if general:
            growth[destination] += own
        counts[request["model"]][destination] += 1
        contexts[request["model"]][destination] += request["context"]
        moves.append({"id": request["id"], "model": request["model"], "source": target,
                      "destination": destination, "blocks": request["blocks"]})
    return moves


def _select_for_target(state, target, arrival, ordered, max_moves):
    """Shortest prefix of `ordered` whose release lets the arrival fit on `target` (ignoring destinations), keeping
    the growth headroom of what stays on `target` plus the arrival.

    The arrival needs physically free blocks: kvcached reports free_blocks = this pool's holes + physically free
    pages, so pages released by the moves add to it (both tenants map the same physical memory)."""
    base = resident_counts(state, arrival=arrival, target=target)
    pool = state["pools"][(arrival["model"], target)]
    blocks = math.ceil(arrival["tokens"] / pool["block_size"])
    general = state.get("general")
    if general:
        growth = rank_growth(state, arrival=arrival, target=target)[target]
    selected = []
    for request in [None] + ordered:
        if request is not None:
            if len(selected) >= max_moves:
                return None
            selected.append(request)
        identifiers = frozenset(item["id"] for item in selected)
        if general:
            released, gained, holes = freed_for(state, target, identifiers, arrival["model"])
            if blocks > pool["free_blocks"] + gained:
                continue
            used = (state["used"][target] - released
                    + pool_increment(dict(pool, free_slots=pool["free_slots"] + holes), blocks))
            headroom = (growth - sum(state["request_growth"].get(item["id"], 0.0) for item in selected)
                        + state["slack_gib"])
            if used + headroom <= state["budgets"][target] + 1e-6:
                return selected
            continue
        released = released_gib(state, target, identifiers)
        if blocks > pool["free_blocks"] + int(released / pool["block_gib"] + 1e-9):
            continue
        counts = [list(values) for values in base]
        for item in selected:
            counts[item["model"]][target] -= 1
        used = state["used"][target] - released + pool_increment(pool, blocks)
        if used + headroom_gib(state, counts, target, target=True) <= state["budgets"][target] + 1e-6:
            return selected
    return None


ORDERINGS = {
    "kv_desc": lambda request: (-request["blocks"], request["id"]),
    "kv_asc": lambda request: (request["blocks"], request["id"]),
}


def plan_admission(state, arrival, policy, model, params, seed=0):
    """Plan moves that let `arrival` ({model, tokens, prompt_tokens, output_tokens}) fit on some rank.

    policy: "mig_random" (random feasible target, random destinations), "mig_bestfit" (KV-only: fewest moves,
    best-fit destinations) or "ca" (cost-aware). The legacy planner moves only the other tenant's requests; general
    planning moves any decoding request (blocks of the arrival's own tenant are reusable at once).
    """
    if policy not in ("mig_random", "mig_bestfit", "ca"):
        raise ValueError(f"Unknown admission policy {policy}")
    rng = random.Random(seed)
    num_ranks = state["num_ranks"]
    score_imbalance = imbalance_objective_enabled(state, params)
    movable = [request for request in state["requests"]
               if request["movable"] and (state.get("general") or request["model"] != arrival["model"])]
    horizon = horizon_steps(state, model, params)
    candidates, diagnostics = [], []
    targets = list(range(num_ranks))
    if policy == "mig_random":
        rng.shuffle(targets)
    mode = {"mig_random": "random", "mig_bestfit": "bestfit", "ca": "cost"}[policy]
    for target in targets:
        on_target = [request for request in movable if request["rank"] == target]
        orderings = list(ORDERINGS) if policy == "ca" else ["kv_desc"]
        stages = []
        for name in orderings:
            selected = _select_for_target(state, target, arrival,
                                          sorted(on_target, key=ORDERINGS[name]), params["max_moves"])
            if selected is None:
                stages.append("select")
                continue
            moves = _assign(state, model, selected, target, list(state["used"]), rng, mode, arrival)
            if moves is None:
                stages.append("assign")
                continue
            used = plan_usage(state, moves, arrival, target)
            if used is None:
                stages.append("usage")
                continue
            stages.append("ok")
            candidate = {"target": target, "moves": moves, "ordering": name, "used_after_gib": used}
            if policy == "mig_random":
                candidate["selection"] = "first_feasible_in_random_target_order"
                return {**candidate, "candidates_evaluated": len(candidates) + 1, "horizon_steps": horizon}
            if policy == "mig_bestfit":
                candidate["objective_s"] = float(len(moves))
                candidates.append(candidate)
                continue
            performance = layout_cost(state, model, layout(state, moves, arrival, target), horizon)
            scored_performance = performance if score_imbalance else 0.0
            candidate.update(performance_cost_s=performance, migration_cost_s=migration_cost(state, moves, params),
                             scored_performance_cost_s=scored_performance,
                             objective_s=scored_performance + migration_cost(state, moves, params))
            candidates.append(candidate)
        if "ok" not in stages:
            pool = state["pools"][(arrival["model"], target)]
            residents = [request for request in state["requests"] if request["rank"] == target
                         and (state.get("general") or request["model"] != arrival["model"])]
            diagnostics.append({
                "target": target, "stages": stages, "residents": len(residents), "movable": len(on_target),
                "need_blocks": math.ceil(arrival["tokens"] / pool["block_size"]), "free_blocks": pool["free_blocks"],
                "released_if_all_moved_gib": released_gib(state, target, frozenset(item["id"] for item in on_target)),
                "used_gib": state["used"][target], "budget_gib": state["budgets"][target]})
    if not candidates:
        return {"target": None, "moves": [], "reason": "no_feasible_plan", "horizon_steps": horizon,
                "diagnostics": diagnostics}
    best = min(candidates, key=lambda item: (item["objective_s"], len(item["moves"]), item["target"]))
    return {**best, "candidates_evaluated": len(candidates), "horizon_steps": horizon,
            "score_imbalance_cost": score_imbalance,
            "alternatives": [{key: item.get(key) for key in ("target", "ordering", "objective_s", "performance_cost_s",
                                                              "scored_performance_cost_s", "migration_cost_s")}
                             | {"moves": len(item["moves"])}
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


# ---------- general (type-agnostic) planning: policies *_g ----------

def solo_rate(model, tenant, counts, contexts):
    """Decode tokens/s of a request of `tenant` running alone at this layout (an upper bound on its rate)."""
    return 1.0 / (model.decode_step(tenant, counts, contexts) or 0.02)


def run_seconds(params, held, episode_s, slice_cfg):
    """Seconds a tenant may decode in the protection window: the rest of the time-multiplexing episode (only its
    slices if it is held), then the relief window W until the next rebalance."""
    if not held:
        run = episode_s
    elif slice_cfg and episode_s > 0:
        period = slice_cfg[2] if len(slice_cfg) == 3 else slice_cfg[0]
        run = (math.ceil(episode_s / period) + 1) * slice_cfg[1]
    else:
        run = 0.0
    return run + params["relief_horizon_s"]


def growth_tokens(rate, prompt, computed, reservation, prefill_s_per_token, run_s):
    """Tokens a request may allocate beyond its reservation within run_s (rate x time, never its unknown output
    length): none while it still prefills, and the reservation's spare tokens are used first."""
    prefill_left = max(0, prompt - computed) * prefill_s_per_token
    tokens = rate * max(0.0, run_s - prefill_left)
    return max(0.0, tokens - max(0, reservation - max(computed, prompt)))


def tm_gain(model, params, tenant, remaining_prompt, others):
    """Request-seconds gained by holding the other tenants while `tenant` prefills `remaining_prompt` tokens alone.

    others: [(tenant, per-rank counts, per-rank contexts)] of the tenants that would be held. The prefill finishes
    sooner (alone vs. co-running, minus the pause), and each held request trades its interfered progress during the
    co-run for solo progress after the episode. Positive only for prefills long enough to amortize the pause.
    """
    held = [item for item in others if any(item[1])]
    if not held or remaining_prompt <= 0:
        return -params["pause_fixed_s"]
    bucket = max(capture_bucket(max(counts)) for _, counts, _ in held)
    co_run = remaining_prompt * model.prefill_s_per_token(tenant, bucket)
    alone = remaining_prompt * model.prefill_s_per_token(tenant) + params["pause_fixed_s"]
    gain = co_run - alone
    for other, counts, contexts in held:
        solo = model.decode_step(other, counts, contexts)
        interfered = model.decode_step(other, counts, contexts, "prefill")
        gain += sum(counts) * ((co_run - alone) - co_run * solo / interfered)
    return gain


def admission_risk(state, used, params):
    """General rebalance risk: request-seconds of future migrations caused by shrinking the largest per-rank free KV
    from F to F' -- arrivals whose KV need lies in (F', F] would have fit without moving anyone.

    params["arrival_needs"]: KV needs (GiB) of recent arrivals; params["arrival_window_s"]: the time they span plus a
    prior. The rate of needs above x is (#needs > x, plus one below the largest need seen) / window.
    """
    needs = params.get("arrival_needs") or ()
    before = max(budget - value for budget, value in zip(state["budgets"], state["used"]))
    after = max(budget - value for budget, value in zip(state["budgets"], used))
    if not needs or after >= before:
        return 0.0
    largest = max(needs)

    def rate(free):
        return (sum(need > free for need in needs) + (free < largest)) / params["arrival_window_s"]

    probability = 1 - math.exp(-max(0.0, rate(after) - rate(before)) * params["horizon_s"])
    movable = [request for request in state["requests"] if request["movable"]]
    if probability <= 0 or not movable:
        return 0.0
    block_gib = state["pools"][(movable[0]["model"], movable[0]["rank"])]["block_gib"]
    average = sum(request["blocks"] for request in movable) / len(movable) * block_gib
    moves = math.ceil((min(before, largest) - after) / max(average, 1e-9))
    if moves <= 0:
        return 0.0
    return probability * (len(state["requests"]) + 1) * (
        params["pause_fixed_s"] + moves * (params["move_fixed_s"] + average / params["copy_gib_s"]))


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
    "context_asc": lambda request: (request["context"], request["id"]),
    "context_desc": lambda request: (-request["context"], request["id"]),
}


def plan_return(state, model, params):
    """Cost-aware rebalance after a release: level per-rank counts when gain > migration cost + headroom risk.

    General planning (state from request growth) levels every tenant in params["return_tenants"] and prices the risk
    with admission_risk (the recent arrival needs) instead of a fixed Code need and Code arrival rate.
    """
    num_ranks = state["num_ranks"]
    score_imbalance = imbalance_objective_enabled(state, params)
    general = state.get("general")
    horizon = horizon_steps(state, model, params)
    base_cost = layout_cost(state, model, layout(state), horizon)
    need = 0.0 if general else params.get("code_need_gib", 0.0)
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
                scored_gain = gain if score_imbalance else 0.0
                cost = migration_cost(state, moves, params)
                if general:
                    risk = admission_risk(state, used, params)
                else:
                    after = _headroom_ranks(state, used, need) if need else []
                    risk = probability * future_admission_cost(state, used, params) if before and not after else 0.0
                decisions.append({"tenant": tenant, "level": level, "ordering": name, "moves": moves,
                                  "gain_s": gain, "migration_cost_s": cost, "headroom_risk_s": risk,
                                  "scored_gain_s": scored_gain,
                                  "net_s": scored_gain - cost - risk, "used_after_gib": used})
    accepted = [item for item in decisions
                if item["net_s"] > params["return_margin_s"] + params["return_margin_ratio"] * item["migration_cost_s"]]
    chosen = max(accepted, key=lambda item: (item["net_s"], -len(item["moves"]))) if accepted else None
    return {"moves": chosen["moves"] if chosen else [], "accepted": chosen is not None, "base_cost_s": base_cost,
            "score_imbalance_cost": score_imbalance,
            "horizon_steps": horizon, "decisions": [{key: value for key, value in item.items() if key != "moves"}
                                                    | {"moves": len(item["moves"])} for item in decisions],
            "chosen": {key: value for key, value in chosen.items() if key != "moves"} if chosen else None}


DEFAULT_PARAMS = {"max_moves": 8, "return_max_moves": 12, "horizon_s": 90.0, "pause_fixed_s": 1.0,
                  "move_fixed_s": 0.05, "copy_gib_s": 10.0, "return_margin_s": 5.0, "return_margin_ratio": 0.5,
                  "code_rate": 0.002, "code_need_gib": 9.375, "migrate_min_gib": 4.0,
                  # current-length admission: growth headroom (see conv_growth_tokens) and per-rank slack pages;
                  # relief_scope "all" applies the relief floor to every growing rank, "destinations" only to ranks
                  # receiving moved requests (not to the arrival's own rank)
                  "relief_horizon_s": 15.0, "decode_window_s": 10.0, "slack_pages": 1, "relief_scope": "all",
                  # general policies (*_g): grace after the episode's first tokens, arrival history for the risk
                  "tm_grace_s": 2.0, "history_s": 900.0, "rate_prior_s": 500.0}
RELIEF_FLOOR_ON_TARGET = {"all": True, "destinations": False}
