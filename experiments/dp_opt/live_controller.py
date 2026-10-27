import asyncio
import math
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from ilp_reference import enumerate_targets, replay_candidate
from motivation import graph_padding
from calibrated_cost import SPARSE_CAPTURE_SIZES


def snapshot_model(rank_snapshots, arriving_tokens, model_index, snapshot):
    for rank, observation in enumerate(rank_snapshots):
        if not observation["uniform_page_geometry"]:
            raise ValueError("Live conservative allocation requires uniform page/block geometry")
        snapshot["counts"][model_index][rank] = len(observation["requests"])
        snapshot["initial_gib"][rank] += observation["mapped_bytes"] / 1024**3
        for page in observation["pages"]:
            item = dict(page, id=f"{model_index}:{rank}:{page['id']}", rank=rank,
                        pool=f"{model_index}:{rank}")
            snapshot["source_pages"].append(item)
        for request in observation["requests"]:
            if request["movable"]:
                pages = math.ceil(len(request["blocks"]) * observation["block_bytes"] / observation["page_bytes"])
                snapshot["movable"].append({"id": request["id"], "model": model_index, "source": rank,
                                            "destination_pool": f"{model_index}:{1 - rank}",
                                            "memory_gib": pages * observation["page_bytes"] / 1024**3,
                                            "move_cost_s": 0.01})
    if model_index == 0:
        observation = rank_snapshots[0]
        blocks = math.ceil(arriving_tokens / observation["block_size"])
        pages = math.ceil(blocks * observation["block_bytes"] / observation["page_bytes"])
        snapshot["new_gib"] = pages * observation["page_bytes"] / 1024**3


def performance_tables(snapshot, fit, horizon, allow_extrapolation):
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("Prediction horizon must be positive")
    if fit["capture_sizes"] != SPARSE_CAPTURE_SIZES or fit["raw_verification"]["status"] != "raw_verified":
        raise ValueError("Live placement requires the raw-verified sparse fit")
    coefficients = fit["capture_shape_posthoc"]
    if coefficients["status"] != "descriptive_posthoc_fit":
        raise ValueError("Missing identifiable capture-aware surrogate")
    if (any(not math.isfinite(coefficients[name]) or coefficients[name] < 0 for name in ("a", "b"))
            or not math.isfinite(fit["baseline_cadence_s"]) or fit["baseline_cadence_s"] <= 0):
        raise ValueError("Nonnegative finite slopes and positive cadence required")
    tables = []
    for model, counts in enumerate(snapshot["counts"]):
        population = sum(counts) + snapshot.get("new_request_count", 1) * int(model == 0)
        if not population:
            tables.append([0])
            continue
        if population != fit["requests"] and not allow_extrapolation:
            raise ValueError("Post-admission population differs from calibration; use feasibility-only override explicitly")
        baseline = graph_padding([population // 2, population - population // 2], SPARSE_CAPTURE_SIZES)
        table = []
        for rank_zero in range(population + 1):
            padding = graph_padding([rank_zero, population - rank_zero], SPARSE_CAPTURE_SIZES)
            if padding is None:
                raise ValueError("Population exceeds fixed capture sizes")
            imbalance = abs(2 * rank_zero - population) / population if population else 0
            excess = coefficients["a"] * imbalance + coefficients["b"] * (padding / baseline - 1)
            table.append(horizon * population * fit["baseline_cadence_s"] * excess)
        tables.append(table)
    return tables


def stable_ownership(observations):
    return [[{"mapped_bytes": rank["mapped_bytes"],
              "requests": sorted((request["id"], tuple(request["blocks"])) for request in rank["requests"]),
              "pages": sorted((page["id"], tuple(page["owners"]), page["pinned"]) for page in rank["pages"])}
             for rank in model] for model in observations]


def preplan_metadata(prepared):
    return {"settings": prepared["settings"], "admission_generation": prepared["admission_generation"],
            "elapsed_s": prepared["elapsed_s"], "horizon_steps": prepared["horizon_steps"],
            "active_plus_incoming": prepared["active_plus_incoming"], "scope": prepared["scope"]}


def admission_planning_snapshot(observed, arriving_model, total_tokens, output_tokens, budgets_gib,
                                fit, allow_extrapolation, policy, candidate_limit,
                                move_fixed_s, move_bandwidth_gib_s, layout_lookup=None, retained_candidate_ids=None,
                                new_request_count=1):
    observations = [observed[arriving_model], observed[1 - arriving_model]]
    snapshot = {"counts": [[0, 0], [0, 0]], "initial_gib": [0, 0],
                "capacity_gib": list(budgets_gib), "new_gib": 0, "source_pages": [], "movable": []}
    if new_request_count != 1:
        snapshot["new_request_count"] = new_request_count
    for model in (0, 1):
        snapshot_model(observations[model], total_tokens, model, snapshot)
    remaining = [request["max_tokens"] - request["output_tokens"]
                 for ranks in observations for rank in ranks for request in rank["requests"]]
    horizon = max(1, sum(remaining) / len(remaining)) if remaining else max(1, output_tokens)
    snapshot["performance_cost_tables_s"] = performance_tables(snapshot, fit, horizon, allow_extrapolation)
    snapshot["performance_cost_scope_by_model"] = ["single_model_N128_capture_surrogate_population_extrapolated"] * 2
    if layout_lookup is not None:
        from layout_cost_pilot import measured_table

        for model, counts in enumerate(snapshot["counts"]):
            actual_model = arriving_model if model == 0 else 1 - arriving_model
            population = sum(counts) + new_request_count * int(model == 0)
            if actual_model == 1 and str(population) in layout_lookup["tables"]:
                snapshot["performance_cost_tables_s"][model] = measured_table(layout_lookup, population, horizon)
                snapshot["performance_cost_scope_by_model"][model] = (
                    "direct_small_population_Conv_count_lookup_context_and_Code_prefill_interference_not_joint_calibrated")
    population = sum(sum(counts) for counts in snapshot["counts"]) + new_request_count
    for request in snapshot["movable"]:
        source = next(item for item in observations[request["model"]][request["source"]]["requests"]
                      if item["id"] == request["id"])
        logical_gib = len(source["blocks"]) * observations[request["model"]][request["source"]]["block_bytes"] / 1024**3
        request["move_cost_s"] = population * (move_fixed_s + logical_gib / move_bandwidth_gib_s)
    selected = select_candidates(snapshot, candidate_limit)
    retained = set(retained_candidate_ids or [])
    selected_ids = {request["id"] for request in selected} | retained
    snapshot["movable"] = [request for request in snapshot["movable"] if request["id"] in selected_ids]
    if policy == "own_model":
        snapshot["movable"] = [request for request in snapshot["movable"] if request["model"] == 0]
    if policy == "kv_only":
        snapshot["performance_cost_tables_s"] = [[0] * len(table) for table in snapshot["performance_cost_tables_s"]]
    return snapshot, horizon, population


def revalidate_prepared_plan(prepared, observed, fit, allow_extrapolation, layout_lookup=None):
    from verify_benefit import verify_ownership

    prompt, output, budgets, policy, model, max_moves, limit, fixed, bandwidth, solver_limit = prepared["settings"]
    verify_ownership(observed, budgets)
    if len(observed) != 2 or any(len(ranks) != 2 for ranks in observed):
        raise ValueError("Candidate revalidation requires four tenant/rank inventories")
    for ranks in observed:
        for observation in ranks:
            pages = {page["id"] for page in observation["pages"]}
            if (len(pages) != len(observation["pages"]) or any(
                    block // observation["page_block_capacity"] not in pages
                    for request in observation["requests"] for block in request["blocks"])):
                raise ValueError("Candidate blocks lack unique mapped source pages")
            if any(not math.isclose(page["memory_gib"], observation["page_bytes"] / 1024**3)
                   for page in observation["pages"]):
                raise ValueError("Candidate source page bytes differ from physical geometry")
    original = prepared["observations"]
    geometry = ("block_size", "block_bytes", "page_bytes", "page_block_capacity", "uniform_page_geometry")
    if any(before.get(name) != after.get(name) for before_ranks, after_ranks in zip(original, observed)
           for before, after in zip(before_ranks, after_ranks) for name in geometry):
        raise ValueError("Prepared candidate KV geometry changed")
    current = {request["id"]: (tenant, rank, request) for tenant, ranks in enumerate(observed)
               for rank, snapshot in enumerate(ranks) for request in snapshot["requests"]}
    previous = {request["id"]: (tenant, rank, request) for tenant, ranks in enumerate(original)
                for rank, snapshot in enumerate(ranks) for request in snapshot["requests"]}
    candidates = prepared["plan"].get("candidates") or ([prepared["plan"]["winner"]]
                                                      if prepared["plan"]["winner"] is not None else [])
    retained = {identifier for candidate in candidates for identifier in candidate.get("moves", [])}
    snapshot, horizon, population = admission_planning_snapshot(
        observed, model, prompt + output, output, budgets, fit, allow_extrapolation,
        policy, limit, fixed, bandwidth, layout_lookup, retained)
    replayed = []
    for candidate in candidates:
        if "objective_s" not in candidate:
            continue
        invalid = [identifier for identifier in candidate["moves"] if identifier not in current
                   or identifier not in previous or current[identifier][:2] != previous[identifier][:2]
                   or current[identifier][2]["blocks"] != previous[identifier][2]["blocks"]
                   or not current[identifier][2]["movable"]]
        if invalid or len(candidate["moves"]) > max_moves:
            replayed.append({"status": "revalidation_selected_identity_or_blocks_rejected",
                             "target": candidate["target"], "invalid_selected_ids": invalid})
        else:
            replayed.append(replay_candidate(snapshot, candidate))
    for target in admission_candidates(observed, model, prompt + output, budgets)[1]:
        replayed.append(replay_candidate(snapshot, {"target": target, "moves": [],
                                                   "status": "current_direct_allocator_fits"}))
    feasible = [candidate for candidate in replayed if "objective_s" in candidate]
    direct = [candidate for candidate in feasible if not candidate["moves"]]
    if policy == "kv_only" and direct:
        winner = min(direct, key=lambda candidate: (snapshot["counts"][0][candidate["target"]],
                                                   snapshot["initial_gib"][candidate["target"]], candidate["target"]))
    else:
        winner = min(feasible, key=lambda candidate: (candidate["objective_s"], len(candidate["moves"]),
                                                    candidate["target"])) if feasible else None
    return snapshot, horizon, population, {"status": "revalidated_feasible_not_proven" if winner else "no_feasible_plan",
        "candidates": replayed, "winner": winner, "proven_infeasible": False,
        "scope": "original_two_target_ILP_candidates_and_current_direct_fits_not_current_global_optimum"}


def prepared_candidate_identity(prepared):
    selected = {identifier for candidate in prepared["plan"].get("candidates", [])
                for identifier in candidate.get("moves", [])}
    if prepared["plan"].get("winner") is not None:
        selected.update(prepared["plan"]["winner"]["moves"])
    geometry = ("block_size", "block_bytes", "page_bytes", "page_block_capacity", "uniform_page_geometry")
    return {"plan": prepared["plan"], "settings": prepared["settings"],
            "selected": [{"id": request["id"], "tenant": tenant, "rank": rank,
                          "blocks": list(request["blocks"]), "movable": request["movable"]}
                         for tenant, ranks in enumerate(prepared["observations"])
                         for rank, observation in enumerate(ranks) for request in observation["requests"]
                         if request["id"] in selected],
            "geometry": [[{name: observation[name] for name in geometry} for observation in ranks]
                         for ranks in prepared["observations"]]}


def allocation_increment(observation, tokens):
    if (not observation["uniform_page_geometry"] or observation["page_block_capacity"] <= 0
            or observation["block_size"] <= 0 or observation["page_bytes"] <= 0):
        raise ValueError("Admission requires uniform positive page geometry")
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
        raise ValueError("Admission token count must be a positive integer")
    blocks = math.ceil(tokens / observation["block_size"])
    free = observation["mapped_free_block_slots"]
    if not isinstance(free, int) or free < 0:
        raise ValueError("Mapped free slots must be a nonnegative integer")
    pages = math.ceil(max(0, blocks - free) / observation["page_block_capacity"])
    return pages * observation["page_bytes"] / 1024**3


def select_candidates(snapshot, limit=2):
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("Candidate limit must be positive")
    released = {request["id"]: sum(page["memory_gib"] for page in snapshot["source_pages"]
                                   if not page["pinned"] and page["owners"] == [request["id"]])
                for request in snapshot["movable"]}
    selected = []
    for model in (0, 1):
        for rank in (0, 1):
            pool = [request for request in snapshot["movable"]
                    if request["model"] == model and request["source"] == rank]
            selected.extend(sorted(pool, key=lambda request: (-released[request["id"]],
                                                              request["memory_gib"], request["id"]))[:limit])
    return selected


def migration_precheck(observations, model, tokens, budgets, policy, max_moves=2, candidate_limit=2):
    if policy not in ("own_model", "kv_only", "ilp") or model not in (0, 1):
        raise ValueError("Migration precheck requires a movable policy and valid tenant")
    if (not isinstance(max_moves, int) or isinstance(max_moves, bool) or max_moves < 0
            or not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0
            or len(budgets) != 2 or any(not math.isfinite(value) or value <= 0 for value in budgets)):
        raise ValueError("Migration precheck requires valid move limit, tokens and budgets")
    snapshot = {"counts": [[0, 0], [0, 0]], "initial_gib": [0, 0],
                "capacity_gib": list(budgets), "new_gib": 0, "source_pages": [], "movable": []}
    for index, ranks in enumerate((observations[model], observations[1 - model])):
        snapshot_model(ranks, tokens, index, snapshot)
    selected = select_candidates(snapshot, candidate_limit)
    if policy == "own_model":
        selected = [request for request in selected if request["model"] == 0]
    candidates = {request["id"]: request for request in selected}
    release_shares = {identifier: 0 for identifier in candidates}
    for page in snapshot["source_pages"]:
        owners = page["owners"]
        if (page["pinned"] or not owners or len(owners) > max_moves
                or any(owner not in candidates for owner in owners)):
            continue
        for owner in owners:
            release_shares[owner] += page["memory_gib"] / len(owners)
    release_upper = [sum(sorted((release_shares[request["id"]] for request in selected
                                 if request["source"] == rank), reverse=True)[:max_moves])
                     for rank in (0, 1)]
    needed = [max(0, snapshot["initial_gib"][rank] + snapshot["new_gib"] - budgets[rank])
              for rank in (0, 1)]
    possible = bool(selected) and max_moves > 0 and any(
        release_upper[rank] + 1e-6 >= needed[rank] for rank in (0, 1))
    return {"may_fit": possible, "selected_candidates": len(selected),
            "optimistic_source_release_gib": release_upper, "required_source_release_gib": needed,
            "scope": "necessary_bound_for_bounded_conservative_ILP_not_actual_allocator_infeasibility_or_copy_certificate"}


def admission_candidates(observations, model, tokens, budgets):
    used = [sum(observations[tenant][rank]["mapped_bytes"] for tenant in (0, 1)) / 1024**3
            for rank in (0, 1)]
    candidates = [rank for rank in (0, 1)
                  if used[rank] + allocation_increment(observations[model][rank], tokens) <= budgets[rank] + 1e-6
                  and math.ceil(tokens / observations[model][rank]["block_size"])
                  <= observations[model][rank].get("free_blocks", math.inf)]
    return used, candidates


def wait_migration_precheck(observations, model, tokens, budgets, max_moves=8, candidate_limit=8,
                            pause_floor_s=.35, solve_floor_s=.15, copy_seconds_per_gib=5, safety_fraction=.8,
                            boundary_cost=None):
    if (not 0 < safety_fraction <= 1 or any(not math.isfinite(value) or value < 0
                                           for value in (pause_floor_s, solve_floor_s, copy_seconds_per_gib))):
        raise ValueError("Wait gate requires finite nonnegative costs and unit safety fraction")
    bounds = migration_precheck(observations, model, tokens, budgets, "ilp", max_moves, candidate_limit)
    snapshot = {"counts": [[0, 0], [0, 0]], "initial_gib": [0, 0], "capacity_gib": list(budgets),
                "new_gib": 0, "source_pages": [], "movable": []}
    for index, ranks in enumerate((observations[model], observations[1 - model])):
        snapshot_model(ranks, tokens, index, snapshot)
    selected = select_candidates(snapshot, candidate_limit)
    requests = {request["id"]: request for ranks in observations for rank in ranks for request in rank["requests"]}
    shares = {request["id"]: 0 for request in selected}
    for page in snapshot["source_pages"]:
        if (page["pinned"] or not page["owners"] or len(page["owners"]) > max_moves
                or any(owner not in shares for owner in page["owners"])):
            continue
        for owner in page["owners"]:
            shares[owner] += page["memory_gib"] / len(page["owners"])
    waits, copy_bounds = [], []
    for rank in (0, 1):
        release_events = []
        for page in snapshot["source_pages"]:
            if page["rank"] != rank or page["pinned"] or not page["owners"]:
                continue
            remaining = [requests[owner].get("estimated_remaining_decode_s") for owner in page["owners"]]
            if all(value is not None and math.isfinite(value) and value >= 0 for value in remaining):
                release_events.append((max(remaining), page["memory_gib"]))
        needed = bounds["required_source_release_gib"][rank]
        released, wait = 0, None
        for duration, memory in sorted(release_events):
            released += memory
            if released + 1e-6 >= needed:
                wait = duration
                break
        if needed <= 1e-6:
            wait = 0
        waits.append(wait)
        weighted = []
        for candidate in selected:
            share = shares[candidate["id"]]
            if candidate["source"] != rank or share <= 0:
                continue
            request = requests[candidate["id"]]
            geometry = observations[model if candidate["model"] == 0 else 1 - model][rank]
            copy_s = len(request["blocks"]) * geometry["block_bytes"] / 1024**3 * copy_seconds_per_gib
            weighted.append((copy_s / share, share))
        needed_copy, copy_lower = needed, 0
        for price, share in sorted(weighted):
            portion = min(needed_copy, share)
            copy_lower += portion * price
            needed_copy -= portion
            if needed_copy <= 1e-6:
                break
        copy_bounds.append(copy_lower if needed_copy <= 1e-6 else None)
    available_waits = [wait for wait in waits if wait is not None]
    available_costs = [cost for cost in copy_bounds if cost is not None]
    wait = min(available_waits) if available_waits else None
    population = len(requests) + 1
    lower = population * (pause_floor_s + solve_floor_s + min(available_costs)) if available_costs else None
    if boundary_cost is not None:
        names = ("incoming_handshake_s", "old_transaction_fixed_s", "preplan_s")
        if any(not math.isfinite(boundary_cost[name]) or boundary_cost[name] < 0 for name in names):
            raise ValueError("Boundary wait costs must be finite nonnegative measured estimates")
        lower = (boundary_cost["incoming_handshake_s"] + boundary_cost["preplan_s"]
                 + population * (boundary_cost["old_transaction_fixed_s"] + min(available_costs))) if available_costs else None
    worthwhile = bounds["may_fit"] and wait is not None and lower is not None and lower < safety_fraction * wait
    return {"may_reduce_latency_sum": worthwhile, "estimated_wait_s": wait,
            "wait_by_rank_s": waits, "migration_latency_sum_lower_s": lower,
            "active_plus_incoming": population, "safety_fraction": safety_fraction,
            "boundary_cost": boundary_cost,
            "scope": "online_decode_extrapolation_and_fractional_owner_release_copy_cost_gate_not_proven_counterfactual_or_allocator_feasibility"}


class PlacementController:
    def __init__(self, engines, fit, transfer_root, allow_extrapolation=False):
        self.engines = engines
        self.fit = fit
        self.transfer_root = Path(transfer_root)
        self.transfer_root.mkdir(parents=True, exist_ok=True)
        self.kv_transport = os.environ.get("DP_OPT_KV_TRANSPORT", "host_file")
        if self.kv_transport not in ("host_file", "dp_allgatherv", "dp_echo"):
            raise ValueError("Unknown experimental KV transport")
        self.lock = asyncio.Lock()
        self.allow_extrapolation = allow_extrapolation
        self.events = []
        self.rank_local_admission = False
        self.admission_observations = None
        self.admission_generation = 0
        self.concurrent_admission = False
        self.layout_lookup = None
        self.revalidate_prepared = False
        self.model_admission_locks = [asyncio.Lock(), asyncio.Lock()]
        self.model_admission_generations = [0, 0]

    @asynccontextmanager
    async def direct_admission_epoch(self, model, tokens, budgets):
        if not self.concurrent_admission:
            async with self.lock:
                observations, scope = await self.admission_snapshot(model)
                used, candidates = admission_candidates(observations, model, tokens, budgets)
                if not candidates and self.rank_local_admission and scope != "fresh_all_tenants":
                    observations, scope = await self.admission_snapshot(model, refresh_all=True)
                    used, candidates = admission_candidates(observations, model, tokens, budgets)
                yield observations, scope, used, candidates
            return
        async with self.model_admission_locks[model]:
            refresh_all = False
            while True:
                async with self.lock:
                    generation = self.admission_generation
                    model_generation = self.model_admission_generations[model]
                    cached = self.admission_observations
                full = refresh_all or cached is None
                observed = await self.snapshots() if full else list(await asyncio.gather(*[
                    self.call(model, rank, "placement_snapshot") for rank in (0, 1)]))
                await self.lock.acquire()
                try:
                    if ((full and generation != self.admission_generation)
                            or (not full and (model_generation != self.model_admission_generations[model]
                                             or self.admission_observations is None))):
                        continue
                    if full:
                        observations, scope = observed, "fresh_all_tenants"
                    else:
                        observations = [list(ranks) for ranks in self.admission_observations]
                        observations[model] = observed
                        scope = "fresh_arriving_tenant_cached_other_upper_bounds_no_independent_growth"
                    used, candidates = admission_candidates(observations, model, tokens, budgets)
                    if not candidates and not full:
                        refresh_all = True
                        continue
                    yield observations, scope, used, candidates
                    return
                finally:
                    self.lock.release()

    async def admission_snapshot(self, model, refresh_all=False):
        if not self.rank_local_admission or refresh_all or self.admission_observations is None:
            return await self.snapshots(), "fresh_all_tenants"
        current = await asyncio.gather(*[self.call(model, rank, "placement_snapshot") for rank in (0, 1)])
        observations = [list(ranks) for ranks in self.admission_observations]
        observations[model] = list(current)
        return observations, "fresh_arriving_tenant_cached_other_upper_bounds_no_independent_growth"

    async def monitor_snapshot(self):
        async with self.lock:
            before_generation = self.admission_generation
        observations = await self.snapshots()
        async with self.lock:
            after_generation = self.admission_generation
        return {"observations": observations, "admission_generation_before": before_generation,
                "admission_generation_after": after_generation,
                "admission_stable_across_observation": before_generation == after_generation,
                "scope": "query_outside_control_lock_accept_only_same_admission_epoch_no_independent_growth"}

    async def call(self, model, rank, method, *args):
        client = self.engines[model].engine_core
        return await asyncio.wait_for(client._call_utility_async(
            method, *args, engine=client.core_engines[rank]), timeout=120)

    async def snapshots(self, refresh_physical_capacity=False):
        arguments = (True,) if refresh_physical_capacity else ()
        observations = await asyncio.gather(*[self.call(model, rank, "placement_snapshot", *arguments)
                                               for model in (0, 1) for rank in (0, 1)])
        return [list(observations[:2]), list(observations[2:])]

    async def release_transfer_cache(self):
        return await asyncio.gather(*[self.call(model, rank, "collective_rpc", "placement_memory_snapshot",
                                               120, (True,), {}) for model in (0, 1) for rank in (0, 1)])

    async def pause(self):
        await asyncio.wait_for(asyncio.gather(*[
            engine.engine_core.pause_scheduler_async(mode="keep", clear_cache=False)
            for engine in self.engines]), timeout=120)

    async def pause_boundary(self, observations):
        counts = [sum(len(rank["requests"]) for rank in ranks) for ranks in observations]
        receipts = [None, None]
        for model in sorted((0, 1), key=lambda index: (counts[index], index)):
            receipts[model] = list(await asyncio.gather(*[
                self.call(model, rank, "placement_pause_boundary") for rank in (0, 1)]))
        return receipts

    async def resume(self):
        await asyncio.wait_for(asyncio.gather(*[
            engine.engine_core.resume_scheduler_async() for engine in self.engines]), timeout=120)

    async def begin(self, transaction):
        return [await asyncio.gather(*[self.call(model, rank, "placement_begin", transaction) for rank in (0, 1)])
                for model in (0, 1)]

    async def end(self, transaction):
        await asyncio.gather(*[self.call(model, rank, "placement_end", transaction)
                               for model in (0, 1) for rank in (0, 1)])

    async def check_capacity(self, budgets_gib, observations=None):
        current = observations if observations is not None else await self.snapshots()
        used = [sum(current[model][rank]["mapped_bytes"] for model in (0, 1)) / 1024**3 for rank in (0, 1)]
        if budgets_gib is not None and any(used[rank] > budgets_gib[rank] + 1e-6 for rank in (0, 1)):
            raise MemoryError("Actual mapped KV exceeds controlled budget")
        return used

    async def retry_reservation(self, model, rank, request_id, budgets_gib):
        async with self.lock:
            observations = await self.snapshots()
            location = observations[model][rank]
            request = next((record for record in location["requests"] if record["id"] == request_id), None)
            if request is None or not request.get("reservation_pending", False):
                return
            tokens = request["prompt_tokens"] + request["max_tokens"]
            used = sum(observations[tenant][rank]["mapped_bytes"] for tenant in (0, 1)) / 1024**3
            if (used + allocation_increment(location, tokens) > budgets_gib[rank] + 1e-6
                    or math.ceil(tokens / location["block_size"]) > location["free_blocks"]):
                return
            self.admission_observations = None
            self.admission_generation += 1
            self.model_admission_generations[model] += 1
            await self.call(model, rank, "placement_retry_reservation", request_id)
            await self.check_capacity(budgets_gib)

    async def probe(self, transaction, model, rank, prompt_token_ids, max_tokens, budgets_gib):
        before = await self.snapshots()
        result = await self.call(model, rank, "placement_probe", transaction, prompt_token_ids, max_tokens)
        after = await self.snapshots()
        if stable_ownership(before) != stable_ownership(after):
            raise RuntimeError("Allocation probe did not restore physical pages and request ownership")
        used = [sum(before[tenant][gpu]["mapped_bytes"] for tenant in (0, 1)) / 1024**3
                for gpu in (0, 1)]
        peak = list(used)
        peak[rank] += (result["during"]["mapped_bytes"] - before[model][rank]["mapped_bytes"]) / 1024**3
        return {"model": model, "rank": rank, "allocated": result["allocated"],
                "fits_budget": result["allocated"] and all(value <= cap + 1e-6
                                                             for value, cap in zip(peak, budgets_gib)),
                "initial_gib": used, "allocation_peak_gib": peak,
                "additional_mapped_bytes": result["during"]["mapped_bytes"] - before[model][rank]["mapped_bytes"],
                "rollback_verified": True, "allocator": result,
                "scope": "real_allocator_temporary_reservation_controlled_kv_cap_not_device_oom"}

    async def try_direct_admission(self, prompt_tokens, output_tokens, budgets_gib, admit, arriving_model=0):
        if (arriving_model not in (0, 1) or min(prompt_tokens, output_tokens) <= 0
                or len(budgets_gib) != 2 or any(not math.isfinite(value) or value <= 0 for value in budgets_gib)):
            raise ValueError("Direct admission requires positive lengths and two finite budgets")
        started = time.monotonic()
        async with self.direct_admission_epoch(arriving_model, prompt_tokens + output_tokens, budgets_gib) as epoch:
            before, observation_scope, used, candidates = epoch
            target = min(candidates, key=lambda rank: (len(before[arriving_model][rank]["requests"]),
                                                        used[rank], rank)) if candidates else None
            after = None
            reservation = None
            if target is not None:
                self.admission_observations = None
                self.admission_generation += 1
                self.model_admission_generations[arriving_model] += 1
                reservation = await admit(target)
                if isinstance(reservation, dict) and reservation.get("rank_observation") is not None:
                    if reservation["rank_observed"] != target:
                        raise RuntimeError("Direct reservation moved outside serialized admission")
                    after = [list(ranks) for ranks in before]
                    after[arriving_model][target] = reservation["rank_observation"]
                else:
                    after = await self.snapshots()
                await self.check_capacity(budgets_gib, after)
            self.admission_observations = [list(ranks) for ranks in (after if after is not None else before)]
            pending = isinstance(reservation, dict) and reservation.get("reservation_pending", False)
            return {"admitted": target is not None and not pending, "submitted": target is not None,
                    "reservation_pending": pending, "arriving_model": arriving_model,
                    "target": target, "moves": [], "observations": before, "after_admission": after,
                    "observations_scope": observation_scope,
                    "after_admission_scope": "target_fresh_other_ranks_pre_admission_upper_bounds_no_independent_growth",
                    "end_to_end_control_s": time.monotonic() - started, "pause_call_s": 0,
                    "scope": "serialized_full_reservation_no_unmanaged_arrivals_no_growth_direct_fastpath"}

    async def rebalance_after_release(self, budgets_gib, **options):
        from live_rebalance import rebalance_after_release

        return await rebalance_after_release(self, budgets_gib, **options)

    async def schedule_admission(self, prompt_tokens, output_tokens, budgets_gib, admit, policy="ilp",
                                 arriving_model=0, max_moves=2, candidate_limit=2, move_fixed_s=0.475,
                                 move_bandwidth_gib_s=1, solver_limit_s=1, allow_migrations=True,
                                 wait_cost_s=None, wait_fixed_cost_s=0, prepared=None, future_transaction_fixed_s=0,
                                 wait_recheck_options=None):
        if policy not in ("no_move", "own_model", "kv_only", "ilp") or arriving_model not in (0, 1):
            raise ValueError("Unknown policy or arriving model")
        if (any(not math.isfinite(value) for value in (prompt_tokens, output_tokens, move_bandwidth_gib_s,
                                                       solver_limit_s, move_fixed_s))
                or min(prompt_tokens, output_tokens, move_bandwidth_gib_s, solver_limit_s) <= 0 or move_fixed_s < 0):
            raise ValueError("Admission lengths, bandwidth and limits must be positive")
        if len(budgets_gib) != 2 or any(not math.isfinite(value) or value <= 0 for value in budgets_gib):
            raise ValueError("Two finite positive KV budgets are required")
        if (wait_cost_s is not None and (not math.isfinite(wait_cost_s) or wait_cost_s < 0)
                or not math.isfinite(wait_fixed_cost_s) or wait_fixed_cost_s < 0
                or not math.isfinite(future_transaction_fixed_s) or future_transaction_fixed_s < 0):
            raise ValueError("Wait and fixed planning costs must be finite and nonnegative")
        if self.revalidate_prepared and prepared is not None and wait_cost_s is not None and wait_recheck_options is None:
            raise ValueError("Candidate revalidation requires current natural-release wait options")
        async with self.lock:
            started = time.monotonic()
            revalidations = []
            total_tokens = prompt_tokens + output_tokens
            original_candidate = prepared_candidate_identity(prepared) if prepared is not None and self.revalidate_prepared else None

            def refresh_candidate(observation, stage):
                refreshed_at = time.monotonic()
                result = revalidate_prepared_plan(prepared, observation, self.fit, self.allow_extrapolation,
                                                  self.layout_lookup)
                gate = (wait_migration_precheck(observation, arriving_model, total_tokens, budgets_gib,
                                               max_moves=max_moves, candidate_limit=candidate_limit,
                                               **wait_recheck_options) if wait_recheck_options is not None else None)
                revalidations.append({"stage": stage, "observations": observation, "snapshot": result[0],
                                      "horizon_steps": result[1], "active_plus_incoming": result[2],
                                      "plan": result[3], "wait_gate": gate, "wait_recheck_options": wait_recheck_options,
                                      "elapsed_s": time.monotonic() - refreshed_at})
                return result, gate

            if prepared is not None:
                current = await self.snapshots()
                if (prepared["settings"] != [prompt_tokens, output_tokens, list(budgets_gib), policy,
                                                     arriving_model, max_moves, candidate_limit,
                                                     move_fixed_s, move_bandwidth_gib_s, solver_limit_s]
                        or (not self.revalidate_prepared and (prepared["admission_generation"] != self.admission_generation
                                                             or stable_ownership(current) != prepared["ownership"]))):
                    return {"admitted": False, "submitted": False, "moves": [], "solve_s": 0,
                            "pause_call_s": 0, "reason": "preplan_stale_before_pause",
                            "preplan_s": prepared["elapsed_s"], "preplan": preplan_metadata(prepared),
                            "observations": current, "snapshot": prepared["snapshot"], "plan": prepared["plan"],
                            "winner": None,
                            "end_to_end_control_s": time.monotonic() - started, "resume_s": 0}
                if self.revalidate_prepared:
                    refreshed, gate = refresh_candidate(current, "before_pause")
                    if refreshed[3]["winner"] is None or any(not request["movable"]
                            for ranks in current for rank in ranks for request in rank["requests"]) or (
                            refreshed[3]["winner"]["moves"] and gate is not None and not gate["may_reduce_latency_sum"]):
                        return {"admitted": False, "submitted": False, "moves": [], "solve_s": 0,
                                "pause_call_s": 0, "reason": "preplan_candidate_revalidation_rejected_before_pause",
                                "observations": current, "snapshot": refreshed[0], "plan": refreshed[3], "winner": None,
                                "preplan_s": prepared["elapsed_s"], "preplan": preplan_metadata(prepared),
                                "preplan_candidate_revalidated": True, "original_candidate": original_candidate,
                                "candidate_revalidations": revalidations,
                                "end_to_end_control_s": time.monotonic() - started, "resume_s": 0}
                if prepared["plan"]["winner"] is None and not self.revalidate_prepared:
                    return {"admitted": False, "submitted": False, "moves": [], "solve_s": 0,
                            "pause_call_s": 0, "reason": "preplan_no_feasible_candidate",
                            "preplan_s": prepared["elapsed_s"], "preplan": preplan_metadata(prepared),
                            "observations": current, "snapshot": prepared["snapshot"], "plan": prepared["plan"],
                            "winner": None,
                            "end_to_end_control_s": time.monotonic() - started, "resume_s": 0}
            self.admission_observations = None
            self.admission_generation += 1
            self.model_admission_generations = [generation + 1 for generation in self.model_admission_generations]
            transaction = uuid.uuid4().hex
            pause_receipts = await self.pause_boundary(current) if prepared is not None else await self.pause()
            paused_at = time.monotonic()
            completed, report, observed = False, None, None
            try:
                eligibility = await self.snapshots()
                if any(not request["movable"] for ranks in eligibility for rank in ranks for request in rank["requests"]):
                    report = {"admitted": False, "deferred": True, "arriving_model": arriving_model,
                              "policy": policy, "moves": [], "solve_s": 0,
                              "observations": eligibility, "snapshot": prepared["snapshot"] if prepared else None,
                              "plan": prepared["plan"] if prepared else None, "winner": None,
                              "preplan_s": prepared["elapsed_s"] if prepared else 0,
                              "preplan_candidate_revalidated": original_candidate is not None,
                              "original_candidate": original_candidate, "candidate_revalidations": revalidations,
                              "pause_receipts": pause_receipts,
                              "pause_call_s": paused_at - started,
                              "reason": "population_not_all_decode_after_pause"}
                    completed = True
                    return report
                observed = await self.begin(transaction)
                if (prepared is not None and not self.revalidate_prepared
                        and stable_ownership(observed) != prepared["ownership"]):
                    await self.end(transaction)
                    report = {"admitted": False, "submitted": False, "moves": [], "solve_s": 0,
                              "pause_call_s": paused_at - started, "reason": "preplan_stale_after_boundary",
                              "preplan_s": prepared["elapsed_s"], "preplan": preplan_metadata(prepared),
                              "observations": observed, "snapshot": prepared["snapshot"], "plan": prepared["plan"],
                              "winner": None,
                              "pause_receipts": pause_receipts}
                    completed = True
                    return report
                observations = [observed[arriving_model], observed[1 - arriving_model]]
                total_tokens = prompt_tokens + output_tokens
                snapshot, horizon, population = admission_planning_snapshot(
                    observed, arriving_model, total_tokens, output_tokens, budgets_gib, self.fit,
                    self.allow_extrapolation, policy, candidate_limit, move_fixed_s, move_bandwidth_gib_s,
                    self.layout_lookup)
                if prepared is not None:
                    if self.revalidate_prepared:
                        refreshed, gate = refresh_candidate(observed, "after_boundary")
                        snapshot, horizon, population, refreshed_plan = refreshed
                        if gate is not None:
                            wait_cost_s = (gate["estimated_wait_s"] * gate["safety_fraction"]
                                           if gate["estimated_wait_s"] is not None else 0)
                    else:
                        snapshot = prepared["snapshot"]
                        horizon = prepared["horizon_steps"]
                candidates = []
                for target in (0, 1):
                    increment = allocation_increment(observations[0][target], total_tokens)
                    if snapshot["initial_gib"][target] + increment <= budgets_gib[target] + 1e-6:
                        final_counts = [list(counts) for counts in snapshot["counts"]]
                        final_counts[0][target] += 1
                        performance = sum(snapshot["performance_cost_tables_s"][model][counts[0]]
                                          for model, counts in enumerate(final_counts))
                        candidates.append({"target": target, "moves": [], "objective_s": performance,
                                           "status": "actual_mapped_slots_direct", "final_counts": final_counts,
                                           "allocation_increment_gib": increment})
                plan, solve_s = None, 0
                if prepared is not None:
                    plan = refreshed_plan if self.revalidate_prepared else prepared["plan"]
                    if self.revalidate_prepared:
                        candidates = [candidate for candidate in plan["candidates"] if "objective_s" in candidate]
                    elif plan["winner"] is not None:
                        candidates.append(plan["winner"])
                elif policy != "no_move" and allow_migrations:
                    solve_started = time.monotonic()
                    plan = await asyncio.to_thread(enumerate_targets, snapshot, max_moves=max_moves,
                                                   time_limit_s=solver_limit_s)
                    solve_s = time.monotonic() - solve_started
                    if plan["winner"] is not None:
                        candidates.append(plan["winner"])
                if policy in ("no_move", "kv_only"):
                    direct = [candidate for candidate in candidates if not candidate["moves"]]
                    if direct:
                        winner = min(direct, key=lambda candidate: (snapshot["counts"][0][candidate["target"]],
                                                                   snapshot["initial_gib"][candidate["target"]], candidate["target"]))
                    else:
                        winner = min(candidates, key=lambda candidate: (candidate["objective_s"], len(candidate["moves"]))) if candidates else None
                else:
                    winner = min(candidates, key=lambda candidate: (candidate["objective_s"], len(candidate["moves"]),
                                                                   candidate["target"])) if candidates else None
                wait_tradeoff = None
                if wait_cost_s is not None and winner is not None and winner["moves"]:
                    elapsed = time.monotonic() - started
                    if prepared is not None:
                        blocking = sum(len(observed[model][rank]["requests"]) *
                            max(0, time.monotonic() - pause_receipts[model][rank]["reached_at"])
                            for model in (0, 1) for rank in (0, 1))
                        fixed = wait_fixed_cost_s + blocking + elapsed + prepared["elapsed_s"]
                        baseline_counts = [list(counts) for counts in snapshot["counts"]]
                        baseline_counts[0][winner["target"]] += 1
                        baseline_performance = sum(snapshot["performance_cost_tables_s"][model][counts[0]]
                                                   for model, counts in enumerate(baseline_counts))
                        migration_cost = population * future_transaction_fixed_s + winner["migration_cost_s"] + max(
                            0, winner["performance_cost_s"] - baseline_performance)
                    else:
                        fixed = max(wait_fixed_cost_s, elapsed * population)
                        migration_cost = winner["objective_s"]
                    threshold = wait_cost_s if prepared is not None else max(0, wait_cost_s - elapsed)
                    wait_tradeoff = {"migration_cost_s": migration_cost, "fixed_cost_s": fixed,
                                     "threshold_s": threshold, "elapsed_at_gate_s": elapsed,
                                     "blocking_at_gate_request_s": blocking if prepared is not None else None,
                                     "accepted": migration_cost + fixed < threshold}
                    if migration_cost + fixed >= threshold:
                        winner = None
                moves, before_admission, after_admission, reservation = [], None, None, None
                submitted, allocator_blocked, transfer_memory = False, False, None
                if winner is not None:
                    by_id = {request["id"]: request for request in snapshot["movable"]}
                    for request_id in winner["moves"]:
                        request = dict(by_id[request_id])
                        request["model"] = arriving_model if request["model"] == 0 else 1 - arriving_model
                        moves.append(await self.migrate(transaction, request, budgets_gib))
                    if moves:
                        transfer_memory = await self.release_transfer_cache()
                    current = await self.snapshots(refresh_physical_capacity=bool(moves))
                    before_admission = current
                    target = winner["target"]
                    increment = allocation_increment(current[arriving_model][target], total_tokens)
                    used = sum(current[model][target]["mapped_bytes"] for model in (0, 1)) / 1024**3
                    if used + increment > budgets_gib[target] + 1e-6:
                        raise MemoryError("Actual mapped-slot admission witness changed after migration")
                    required_blocks = math.ceil(total_tokens / current[arriving_model][target]["block_size"])
                    allocator_blocked = required_blocks > current[arriving_model][target].get("free_blocks", math.inf)
                    if not allocator_blocked:
                        reservation = await admit(target)
                        submitted = True
                        await self.check_capacity(budgets_gib)
                        after_admission = await self.snapshots()
                await self.end(transaction)
                pending = isinstance(reservation, dict) and reservation.get("reservation_pending", False)
                report = {"admitted": submitted and not pending, "submitted": submitted,
                          "actual_allocator_blocked_after_plan": allocator_blocked,
                          "wait_cost_s": wait_cost_s, "wait_fixed_cost_s": wait_fixed_cost_s,
                          "future_transaction_fixed_s": future_transaction_fixed_s,
                          "wait_tradeoff": wait_tradeoff,
                          "transfer_memory_after_cache_release": transfer_memory,
                          "reservation_pending": pending, "arriving_model": arriving_model, "policy": policy,
                          "snapshot": snapshot, "observations": observed, "plan": plan, "winner": winner,
                          "before_admission": before_admission, "after_admission": after_admission,
                          "moves": moves, "solve_s": solve_s, "pause_call_s": paused_at - started,
                          "preplan": preplan_metadata(prepared) if prepared is not None else None,
                          "preplan_candidate_revalidated": prepared is not None and self.revalidate_prepared,
                          "original_candidate": original_candidate, "candidate_revalidations": revalidations,
                          "preplan_s": prepared["elapsed_s"] if prepared else 0,
                          "pause_receipts": pause_receipts,
                          "transaction_s_before_resume": time.monotonic() - started, "horizon_steps": horizon,
                          "cost_scope": ("small_population_Conv_lookup_with_explicit_surrogate_fallback_not_Code_prefill_joint_cost"
                                         if self.layout_lookup is not None else
                                         "single_model_surrogate_extrapolated_measured_fixed_latency_assumed_bandwidth"),
                          "candidate_scope": f"at_most_{candidate_limit}_requests_per_model_rank_not_global_optimum"}
                completed = True
            finally:
                if completed:
                    resume_started = time.monotonic()
                    await self.resume()
                    report["resume_s"] = time.monotonic() - resume_started
                    report["end_to_end_control_s"] = time.monotonic() - started
                    if prepared is not None:
                        report["end_to_end_control_s_including_preplan"] = report["end_to_end_control_s"] + prepared["elapsed_s"]
                        report["active_blocking_upper_request_s"] = sum(len(observed[model][rank]["requests"]) *
                            max(0, time.monotonic() - pause_receipts[model][rank]["reached_at"])
                            for model in (0, 1) for rank in (0, 1)) if observed is not None else None
            return report

    async def prepare_admission(self, prompt_tokens, output_tokens, budgets_gib, policy="ilp",
                                arriving_model=0, max_moves=2, candidate_limit=2, move_fixed_s=.475,
                                move_bandwidth_gib_s=1, solver_limit_s=1):
        if (policy not in ("own_model", "kv_only", "ilp") or arriving_model not in (0, 1)
                or len(budgets_gib) != 2 or any(not math.isfinite(value) or value <= 0 for value in budgets_gib)
                or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
                       for value in (prompt_tokens, output_tokens, candidate_limit))
                or not isinstance(max_moves, int) or isinstance(max_moves, bool) or max_moves < 0
                or any(not math.isfinite(value) or value <= 0 for value in (move_bandwidth_gib_s, solver_limit_s))
                or not math.isfinite(move_fixed_s) or move_fixed_s < 0):
            raise ValueError("Preplanning requires valid admission geometry, movable policy and finite costs")
        started = time.monotonic()
        async with self.lock:
            generation = self.admission_generation
        observed = await self.snapshots()
        snapshot, horizon, population = admission_planning_snapshot(
            observed, arriving_model, prompt_tokens + output_tokens, output_tokens, budgets_gib,
            self.fit, self.allow_extrapolation, policy, candidate_limit, move_fixed_s, move_bandwidth_gib_s,
            self.layout_lookup)
        plan = await asyncio.to_thread(enumerate_targets, snapshot, max_moves=max_moves, time_limit_s=solver_limit_s)
        return {"settings": [prompt_tokens, output_tokens, list(budgets_gib), policy, arriving_model,
                             max_moves, candidate_limit, move_fixed_s, move_bandwidth_gib_s, solver_limit_s],
                "admission_generation": generation, "ownership": stable_ownership(observed),
                "observations": observed, "snapshot": snapshot, "plan": plan, "horizon_steps": horizon,
                "active_plus_incoming": population, "elapsed_s": time.monotonic() - started,
                "scope": "decode_continues_during_ILP_revalidate_before_and_after_default32_boundary"}

    async def migrate(self, transaction, request, budgets_gib=None):
        model, source = request["model"], request["source"]
        destination = 1 - source
        request_id = request["id"]
        path = self.transfer_root / (uuid.uuid4().hex + ".pt")
        started = time.monotonic()
        stage_started = started
        stage_s = {}

        def mark(name):
            nonlocal stage_started
            now = time.monotonic()
            stage_s[name] = now - stage_started
            stage_started = now

        prepare_started, source_remove_started, peer_started, peer_completed = False, False, False, False
        try:
            payload = await self.call(model, source, "placement_export", transaction, request_id)
            mark("export_request_state")
            if budgets_gib is not None:
                current = await self.check_capacity(budgets_gib)
                if current[destination] + request["memory_gib"] > budgets_gib[destination] + 1e-6:
                    raise MemoryError("Actual destination headroom cannot cover conservative reservation")
            mark("source_capacity_check")
            prepare_started = True
            imported = await self.call(model, destination, "placement_prepare", transaction, payload)
            mark("destination_prepare")
            copy_peak = await self.check_capacity(budgets_gib)
            mark("copy_peak_capacity_check")
            if self.kv_transport in ("dp_allgatherv", "dp_echo"):
                if len(payload["blocks"]) != len(imported["blocks"][0]):
                    raise RuntimeError("Peer source and destination block counts differ")
                peer_started = True
                integrity = "device_echo" if self.kv_transport == "dp_echo" else "sha256"
                results = await asyncio.gather(
                    self.call(model, source, "placement_kv_peer", transaction, payload["blocks"], source, integrity),
                    self.call(model, destination, "placement_kv_peer", transaction, imported["blocks"][0], source, integrity),
                    return_exceptions=True)
                errors = [result for result in results if isinstance(result, BaseException)]
                if errors:
                    raise RuntimeError("Peer transfer failed; keep both engines paused and mappings alive") from errors[0]
                peer_completed = True
                exported, received = results
                mark("peer_kv_rpc")
            else:
                exported = await self.call(model, source, "placement_kv_file", transaction, "export", payload["blocks"], str(path))
                mark("export_kv_rpc")
                received = await self.call(model, destination, "placement_kv_file", transaction, "import", imported["blocks"][0], str(path))
                mark("import_kv_rpc")
            if (exported["sha256"], exported["bytes"]) != (received["sha256"], received["bytes"]):
                raise RuntimeError("Destination KV checksum or length differs from source")
            if self.kv_transport == "dp_echo" and not all(
                    result.get("integrity") == "device_echo_full_bytewise" and result.get("bitwise_verified") is True
                    and result.get("sha256_scope") == "verified_integrity_receipt_not_KV_content"
                    for result in (exported, received)):
                raise RuntimeError("Destination KV lacks the exact device bytewise integrity proof")
            await self.call(model, destination, "placement_runner_state", transaction, imported)
            mark("install_destination_runner")
            await self.call(model, destination, "placement_commit", transaction, request_id)
            mark("commit_destination")
            source_remove_started = True
            after = await self.call(model, source, "placement_remove", transaction, request_id)
            mark("release_source")
            client = self.engines[model].engine_core
            client.reqs_in_flight[request_id] = client.core_engines[destination]
            await self.call(model, source, "placement_runner_state", transaction, {"request_id": request_id, "remove": True})
            mark("reroute_and_remove_source_runner")
            event = {"request_id": request_id, "model": model, "source": source, "destination": destination,
                     "elapsed_s": time.monotonic() - started, "kv_bytes": exported["bytes"],
                     "sha256": exported["sha256"], "source_mapped_after": after["mapped_bytes"],
                     "copy_peak_gib": copy_peak,
                     "transport": ("existing_dp_allgatherv_device_payload_host_integrity_readback"
                                   if self.kv_transport == "dp_allgatherv" else
                                   "existing_dp_allgatherv_full_device_bytewise_echo" if self.kv_transport == "dp_echo"
                                   else "host_staged_file_not_nvlink_optimized"),
                     "integrity": exported.get("integrity", "full_content_sha256"),
                     "sha256_scope": exported.get("sha256_scope", "KV_content"),
                     "bitwise_verified": exported.get("bitwise_verified", False),
                     "stage_s": stage_s,
                     "worker_transfer_stages_s": {"export": exported.get("stage_s"),
                                                   "import": received.get("stage_s")}}
            self.events.append(event)
            return event
        except BaseException as error:
            if peer_started and not peer_completed:
                raise RuntimeError("Uncertain peer transfer completion; retain source and destination, keep engines paused and terminate experiment") from error
            if source_remove_started:
                raise RuntimeError("Uncertain/post-commit migration failure; retain destination, keep engines paused and terminate experiment") from error
            if prepare_started:
                await self.call(model, destination, "placement_rollback", transaction, request_id)
                await self.call(model, destination, "placement_runner_state", transaction, {"request_id": request_id, "remove": True})
            raise
        finally:
            path.unlink(missing_ok=True)

    async def place(self, arriving_tokens, budgets_gib=None, policy="ilp", required_counts=None,
                    headroom_pages=None, admit=None):
        if policy not in ("ilp", "own_model", "kv_only", "no_move"):
            raise ValueError("Unknown placement policy")
        async with self.lock:
            transaction = uuid.uuid4().hex
            started = time.monotonic()
            await self.pause()
            completed = False
            try:
                observations = await self.begin(transaction)
                if budgets_gib is None:
                    if headroom_pages is None or headroom_pages < 0:
                        raise ValueError("Provide a KV budget or nonnegative headroom pages")
                    budgets_gib = [(sum(observations[model][rank]["mapped_bytes"] for model in (0, 1))
                                    + headroom_pages * observations[0][rank]["page_bytes"]) / 1024**3
                                   for rank in (0, 1)]
                snapshot = {"counts": [[0, 0], [0, 0]], "initial_gib": [0, 0],
                            "capacity_gib": list(budgets_gib), "new_gib": 0,
                            "source_pages": [], "movable": []}
                for model in (0, 1):
                    snapshot_model(observations[model], arriving_tokens, model, snapshot)
                if required_counts is not None and snapshot["counts"] != required_counts:
                    raise RuntimeError(f"Population changed before placement: {snapshot['counts']}")
                snapshot["performance_cost_tables_s"] = performance_tables(
                    snapshot, self.fit, 32, self.allow_extrapolation)
                population = sum(sum(counts) for counts in snapshot["counts"]) + 1
                for request in snapshot["movable"]:
                    request["move_cost_s"] = population * (0.002 + request["memory_gib"] / 12)
                if policy == "own_model":
                    snapshot["movable"] = [request for request in snapshot["movable"] if request["model"] == 0]
                elif policy == "kv_only":
                    snapshot["performance_cost_tables_s"] = [[0] * len(table)
                                                              for table in snapshot["performance_cost_tables_s"]]
                elif policy not in ("ilp", "no_move"):
                    raise ValueError("Unknown experiment placement policy")
                plan = await asyncio.to_thread(enumerate_targets, snapshot,
                                               max_moves=0 if policy == "no_move" else 4, time_limit_s=5)
                moves = []
                before_admission = None
                if plan["winner"] is not None:
                    by_id = {request["id"]: request for request in snapshot["movable"]}
                    for request_id in plan["winner"]["moves"]:
                        moves.append(await self.migrate(transaction, by_id[request_id], budgets_gib))
                    if admit is not None:
                        target = plan["winner"]["target"]
                        current = await self.check_capacity(budgets_gib)
                        if current[target] + snapshot["new_gib"] > budgets_gib[target] + 1e-6:
                            raise MemoryError("Actual admission headroom cannot cover conservative reservation")
                        before_admission = await self.snapshots()
                        await admit(target)
                    await self.check_capacity(budgets_gib)
                await self.end(transaction)
                completed = True
                return {"plan": plan, "snapshot": snapshot, "observations": observations,
                        "before_admission": before_admission, "moves": moves,
                        "pause_s": time.monotonic() - started,
                        "coefficient_scope": "single_model_surrogate_not_concurrent_calibration",
                        "admitted_while_paused": admit is not None and plan["winner"] is not None,
                        "population_extrapolation_allowed": self.allow_extrapolation}
            finally:
                if completed:
                    await self.resume()
