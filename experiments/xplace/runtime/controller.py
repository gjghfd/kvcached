import asyncio
import math
import time
import uuid

GIB = 1024**3


class CleanMigrationFailure(RuntimeError):
    """Migration failed before any irreversible step; destination state was rolled back."""


def allocation_increment(observation, tokens):
    if (not observation["uniform_page_geometry"] or observation["page_block_capacity"] <= 0
            or observation["block_size"] <= 0 or observation["page_bytes"] <= 0):
        raise ValueError("Admission requires uniform positive page geometry")
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
        raise ValueError("Admission token count must be a positive integer")
    blocks = math.ceil(tokens / observation["block_size"])
    pages = math.ceil(max(0, blocks - observation["mapped_free_block_slots"]) / observation["page_block_capacity"])
    return pages * observation["page_bytes"] / GIB


def used_gib(observations):
    return [sum(observations[model][rank]["mapped_bytes"] for model in range(len(observations))) / GIB
            for rank in range(len(observations[0]))]


def fits(observations, model, rank, tokens, budgets, used=None, headroom=0.0):
    used = used_gib(observations) if used is None else used
    observation = observations[model][rank]
    return (used[rank] + allocation_increment(observation, tokens) + headroom <= budgets[rank] + 1e-6
            and math.ceil(tokens / observation["block_size"]) <= observation["free_blocks"])


def direct_candidates(observations, model, tokens, budgets, headroom=None):
    used = used_gib(observations)
    headroom = headroom or [0.0] * len(budgets)
    return used, [rank for rank in range(len(budgets))
                  if fits(observations, model, rank, tokens, budgets, used, headroom[rank])]


def count_least_loaded(observations, model, candidates, used):
    return min(candidates, key=lambda rank: (len(observations[model][rank]["requests"]), used[rank], rank))


def all_decoding(summaries):
    return all(request["decoding"] for ranks in summaries for rank in ranks for request in rank["requests"])


class PlacementController:
    def __init__(self, engines, num_ranks, budgets_gib, call_timeout_s=180):
        if len(engines) != 2 or len(budgets_gib) != num_ranks or num_ranks < 2:
            raise ValueError("Controller needs two tenants and one budget per DP rank")
        self.engines = engines
        self.num_ranks = num_ranks
        self.tensor_parallel_size = 1
        self.budgets = [float(value) for value in budgets_gib]
        self.lock = asyncio.Lock()
        self.events = []
        self.transactions = []
        self.held = {}
        self.holds = []
        self.call_timeout_s = call_timeout_s
        # Fast Conv admission: cached Code-tenant view. Code KV only grows through admissions made under self.lock,
        # which bump code_epoch first and refresh or drop the cache afterwards, so a cached view can only overstate
        # Code usage. Concurrent refreshes (monitor) are accepted only if the epoch did not move meanwhile.
        self.fast_admission = False
        self.code_view = None
        self.code_epoch = 0
        self.view_stats = {"cached": 0, "fetched": 0}
        # General policies (*_g): a per-tenant cached view serves another tenant's admission only while it is exact
        # (view_exact): the tenant was held since the view, or none of its requests can outgrow its reservation
        # within the view's age (e.g. a long prefill). Admissions of a tenant bump its epoch under self.lock.
        self.general = False
        self.views = {}
        self.view_epoch = [0, 0]
        self.max_decode_rate = 100.0  # tokens/s bound on one request's decode (set from the step model)
        self.min_prefill_s_per_token = 0.0  # fastest prefill (alone), set from the step model
        # Native runs cap KV physically (ballast) instead of through admission, so the budget is observed, not enforced.
        self.enforce_budget = True

    async def tenant_summaries(self, model):
        return list(await asyncio.gather(*[self.call(model, rank, "placement_summary")
                                           for rank in range(self.num_ranks)]))

    async def admission_view(self, model):
        """Summaries used to admit a `model` request (call under self.lock)."""
        if self.general:
            return await self.general_view(model)
        if not (self.fast_admission and model == 1):
            observations = await self.summaries()
            if self.fast_admission:
                self.code_view = observations[0]
            return observations
        if self.code_view is None:
            self.code_view = await self.tenant_summaries(0)
            self.view_stats["fetched"] += 1
        else:
            self.view_stats["cached"] += 1
        return [self.code_view, await self.tenant_summaries(1)]

    def begin_code_change(self):
        """Call under self.lock before Code KV may grow: drop the cache and reject in-flight refreshes."""
        self.code_epoch += 1
        self.code_view = None

    def end_code_change(self, view=None):
        """Call under self.lock once the Code reservation is in place; a refresh begun before this is rejected."""
        self.code_epoch += 1
        self.code_view = view

    def offer_code_view(self, epoch, view):
        if self.fast_admission and epoch == self.code_epoch:
            self.code_view = view

    async def general_view(self, model):
        """General policies: `model` live; each other tenant from its cached view while exact (fast admission),
        else live. Call under self.lock, so no admission changes a view meanwhile."""
        observations = [None, None]
        others = [other for other in (0, 1) if other != model]
        # read cached views before awaiting: a completion may drop a view meanwhile
        cached = {other: self.views[other]["summaries"] for other in others
                  if self.fast_admission and self.view_exact(other)}
        live = [model] + [other for other in others if other not in cached]
        started = time.monotonic()
        for item, value in zip(live, await asyncio.gather(*[self.tenant_summaries(item) for item in live])):
            observations[item] = value
            self.views[item] = {"at": started, "summaries": value}
        for other in others:
            if other in cached:
                observations[other] = cached[other]
                self.view_stats["cached"] += 1
            else:
                self.view_stats["fetched"] += 1
        return observations

    def view_exact(self, model):
        """True if `model`'s cached view still bounds its KV: held since the view was taken, or no request in it can
        allocate past its reservation within the view's age -- a request must first finish its prefill (at most at
        the solo prefill speed) and then decode through the reservation's spare tokens (at most max_decode_rate)."""
        view = self.views.get(model)
        if view is None:
            return False
        hold = self.held.get(model)
        if hold is not None and hold["paused_at"] <= view["at"]:
            return True
        age = time.monotonic() - view["at"]
        for rank in view["summaries"]:
            for request in rank["requests"]:
                prompt, computed = request["prompt_tokens"], request["computed_tokens"]
                prefill_s = max(0, prompt - computed) * self.min_prefill_s_per_token
                spare = max(0, request["reservation_tokens"] - max(computed, prompt))
                if age > prefill_s + spare / self.max_decode_rate:
                    return False
        return True

    def begin_change(self, model):
        """Call under self.lock before `model`'s KV may grow through an admission (Code only in legacy policies)."""
        if self.general:
            self.view_epoch[model] += 1
            self.views.pop(model, None)
        elif model == 0:
            self.begin_code_change()

    def end_change(self, model, view=None, at=None):
        """Call under self.lock once the admission's reservation is in place; `view` is the tenant's state now."""
        if self.general:
            self.view_epoch[model] += 1
            if view is not None:
                self.views[model] = {"at": time.monotonic() if at is None else at, "summaries": view}
        elif model == 0:
            self.end_code_change(view)

    def offer_view(self, model, epoch, view, at):
        """General policies: a monitor refresh begun at `at`, kept only if no admission changed `model` since."""
        if epoch == self.view_epoch[model]:
            self.views[model] = {"at": at, "summaries": view}

    def drop_views(self, models):
        for model in models:
            self.view_epoch[model] += 1
            self.views.pop(model, None)

    async def call(self, model, rank, method, *args):
        client = self.engines[model].engine_core
        return await asyncio.wait_for(client._call_utility_async(method, *args, engine=client.core_engines[rank]),
                                      timeout=self.call_timeout_s)

    async def gather(self, method, *args):
        values = await asyncio.gather(*[self.call(model, rank, method, *args)
                                        for model in (0, 1) for rank in range(self.num_ranks)])
        return [list(values[model * self.num_ranks:(model + 1) * self.num_ranks]) for model in (0, 1)]

    async def snapshots(self, refresh_physical_capacity=False):
        return await self.gather("placement_snapshot", *((True,) if refresh_physical_capacity else ()))

    async def summaries(self):
        return await self.gather("placement_summary")

    async def prediction_observations(self, root, timeout_s):
        from live_observation import collect_execution, collect_scheduler
        from observation_diagnostics import gather_drained

        scheduler, execution = await gather_drained(
            collect_scheduler(self, timeout_s),
            collect_execution(root, self.num_ranks, self.tensor_parallel_size, timeout_s))
        return {"scheduler": scheduler, "execution": execution, "collected_at": time.monotonic(),
                "forecast_status": "unavailable_not_estimated", "device_activity": "unknown"}

    async def memory(self, release_unused=False, models=(0, 1)):
        values = await asyncio.gather(*[self.call(model, rank, "collective_rpc", "placement_memory_snapshot", 120,
                                                  (release_unused,), {})
                                        for model in models for rank in range(self.num_ranks)])
        memory = {model: list(values[index * self.num_ranks:(index + 1) * self.num_ranks])
                  for index, model in enumerate(models)}
        if self.tensor_parallel_size > 1:
            from tp_migration import ordered_shards

            memory = {model: [ordered_shards(group, self.tensor_parallel_size, rank)
                              for rank, group in enumerate(groups)] for model, groups in memory.items()}
        return memory

    def check_capacity(self, observations):
        used = used_gib(observations)
        if self.enforce_budget and any(used[rank] > self.budgets[rank] + 1e-6 for rank in range(self.num_ranks)):
            raise MemoryError(f"Mapped KV {used} exceeds controlled budget {self.budgets}")
        return used

    async def set_mode(self, mode, reserve_extra=0):
        """Switch every scheduler's admission mode: "full" (legacy prompt+max_tokens), "prompt" (current length,
        then per-step growth with vLLM retract) or "native" (stock vLLM)."""
        return await self.gather("placement_set_mode", mode, int(reserve_extra))

    def worker_capacity(self, memory, summaries, utilization):
        capacity = []
        for rank in range(self.num_ranks):
            if any(len(memory[model][rank]) != self.tensor_parallel_size for model in (0, 1)):
                raise RuntimeError("Missing TP worker memory snapshot")
            mapped = sum(summaries[model][rank]["mapped_bytes"] for model in (0, 1))
            group = []
            for lane in range(self.tensor_parallel_size):
                first, second = (memory[model][rank][lane] for model in (0, 1))
                if first["gpu_uuid"] != second["gpu_uuid"] or first["total_mib"] != second["total_mib"]:
                    raise RuntimeError("Tenant memory snapshots refer to different GPUs")
                free = min(first["free_mib"], second["free_mib"]) * 1024**2
                group.append(mapped + free - first["total_mib"] * 1024**2 * (1 - utilization))
            capacity.append(group)
        return capacity

    def kv_capacity(self, memory, summaries, utilization):
        return [min(group) for group in self.worker_capacity(memory, summaries, utilization)]

    async def apply_ballast(self, cap_gib, utilization):
        """Hold device memory so each GPU's shared KV capacity is floor(cap / page) whole page bundles, as under the
        controlled budget. The first pass undershoots by a quarter page so no pass ever needs to give memory back."""
        summaries = await self.summaries()
        page = summaries[0][0]["page_bytes"]
        target = (math.floor(cap_gib * GIB / page) + 0.5) * page
        if any(rank["page_bytes"] != page for model in summaries for rank in model):
            raise RuntimeError("Ballast requires matching per-worker KV page geometry")
        natural = self.worker_capacity(await self.memory(), summaries, utilization)
        if any(value < target for group in natural for value in group):
            raise RuntimeError(f"Natural KV capacity {natural} is below the native target {target / GIB:.3f} GiB")
        capacity = natural
        for slack in (0.25 * page, 0.0):
            excess = [[max(0, int(value - target - slack)) for value in group] for group in capacity]
            share = (0, 1) if slack else (0,)
            await asyncio.gather(*[self.call(model, rank, "collective_rpc", "placement_ballast", 120,
                                             (([value // len(share) for value in excess[rank]]
                                               if self.tensor_parallel_size > 1 else excess[rank][0] // len(share)),), {})
                                   for rank in range(self.num_ranks) for model in share])
            capacity = self.worker_capacity(await self.memory(), await self.summaries(), utilization)
        if any(abs(value - target) > 0.25 * page for group in capacity for value in group):
            raise RuntimeError(f"Ballast missed the KV target {target / GIB:.3f} GiB: {capacity}")
        return {"cap_gib": cap_gib, "utilization": utilization, "page_bytes": page, "target_gib": target / GIB,
                "pages": math.floor(cap_gib * GIB / page),
                "natural_capacity_gib": [min(group) / GIB for group in natural],
                "capacity_gib": [min(group) / GIB for group in capacity],
                "worker_natural_capacity_gib": [[value / GIB for value in group] for group in natural],
                "worker_capacity_gib": [[value / GIB for value in group] for group in capacity]}

    async def release_ballast(self, utilization):
        released = await asyncio.gather(*[self.call(model, rank, "collective_rpc", "placement_ballast_release", 120,
                                                    (), {})
                                          for model in (0, 1) for rank in range(self.num_ranks)])
        capacity = self.kv_capacity(await self.memory(), await self.summaries(), utilization)
        return {"released_bytes": sum(shard["released_bytes"] for item in released for shard in item),
                "capacity_gib": [value / GIB for value in capacity]}

    async def pause_boundary(self, observations, models=(0, 1)):
        counts = [sum(len(rank["requests"]) for rank in ranks) for ranks in observations]
        receipts = [None, None]
        for model in sorted(models, key=lambda index: (counts[index], index)):
            receipts[model] = list(await asyncio.gather(*[self.call(model, rank, "placement_pause_boundary")
                                                          for rank in range(self.num_ranks)]))
        return receipts

    async def resume(self, models=(0, 1)):
        await asyncio.wait_for(asyncio.gather(*[self.engines[model].engine_core.resume_scheduler_async()
                                                for model in models]), timeout=self.call_timeout_s)

    async def set_hold(self, model, wanted):
        """Pause `model` at a DP boundary while wanted() is true, resume it once false (time-multiplexing).

        wanted is evaluated under the lock so concurrent callers converge. Transactions leave held tenants paused.
        """
        async with self.lock:
            if wanted() and model not in self.held:
                await self.hold_locked(model)
            elif not wanted() and model in self.held:
                await self.release_locked(model)
            return model in self.held

    async def release_locked(self, model, **extra):
        """Resume a held tenant and record the hold (call under self.lock)."""
        hold = self.held.pop(model)
        hold["resumed_at"] = time.monotonic()
        await self.resume((model,))
        hold["held_s"] = hold["resumed_at"] - hold["paused_at"]
        hold.update(extra)
        self.holds.append(hold)

    async def pause_native(self, model):
        """Stop scheduling `model` immediately (vLLM keep-pause, KV kept); returns once every DP rank is idle."""
        await asyncio.wait_for(self.engines[model].engine_core.pause_scheduler_async(mode="keep", clear_cache=False),
                               timeout=self.call_timeout_s)

    async def hold_locked(self, model):
        """Pause `model` at its next DP boundary and mark it held (call under self.lock; no-op if already held)."""
        if model in self.held:
            return
        requested = time.monotonic()
        if self.fast_admission:
            observations = [[], []]
            observations[model] = await self.tenant_summaries(model)
        else:
            observations = await self.summaries()
        receipts = (await self.pause_boundary(observations, (model,)))[model]
        self.held[model] = {"model": model, "requested_at": requested, "paused_at": time.monotonic(),
                            "requests_at_pause": sum(len(rank["requests"]) for rank in observations[model]),
                            "pause_reach_s": [receipt["reached_at"] - receipt["armed_at"] for receipt in receipts]}

    async def wait_reserved(self, task, request_id, model, rank, timeout=60):
        """Return once the target rank reports the request and whether its full reservation succeeded."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            observation = await self.call(model, rank, "placement_summary")
            record = next((request for request in observation["requests"]
                           if request["id"] == request_id or request["id"].rsplit("-", 1)[0] == request_id), None)
            if record is not None:
                return {"reservation_pending": record["reservation_pending"], "reserved_at": record["reserved_at"],
                        "rank_observation": observation}
            if task.done():
                result = await task
                return {"reservation_pending": False, "reserved_at": result["first_token_at"],
                        "reservation_time_source": "first_token_upper_bound"}
            await asyncio.sleep(.02)
        raise TimeoutError(f"Request {request_id} reservation was not observed on rank {rank}")

    async def retry_reservation(self, model, rank, request_id):
        """Retry a pending full reservation when the budget allows; returns {reservation_pending, reserved_at}."""
        async with self.lock:
            observations = await self.summaries()
            record = next((request for request in observations[model][rank]["requests"]
                           if request["id"] == request_id or request["id"].rsplit("-", 1)[0] == request_id), None)
            if record is None:
                return {"reservation_pending": True, "reserved_at": None, "missing": True}
            if not record["reservation_pending"]:
                return {"reservation_pending": False, "reserved_at": record["reserved_at"]}
            tokens = record["reservation_tokens"]
            if fits(observations, model, rank, tokens, self.budgets):
                self.begin_change(model)
                result = await self.call(model, rank, "placement_retry_reservation", record["id"])
                self.end_change(model)
                return {"reservation_pending": result.get("reservation_pending", True),
                        "reserved_at": result.get("reserved_at")}
            return {"reservation_pending": True, "reserved_at": None}

    async def try_direct_admission(self, model, tokens, admit, route=None, before_admit=None, headroom=None):
        async with self.lock:
            started = time.monotonic()
            observations = await self.admission_view(model)
            if callable(headroom):
                headroom = headroom(observations)
            used, candidates = direct_candidates(observations, model, tokens, self.budgets, headroom)
            report = {"path": "direct", "model": model, "tokens": tokens, "used_gib": used,
                      "counts": [[len(rank["requests"]) for rank in ranks] for ranks in observations],
                      "candidates": candidates, "admitted": False, "target": None, "headroom_gib": headroom}
            if candidates:
                target = (route or count_least_loaded)(observations, model, candidates, used)
                if before_admit is not None:
                    hold_started = time.monotonic()
                    await before_admit()
                    report["before_admit_s"] = time.monotonic() - hold_started
                self.begin_change(model)
                reservation = await admit(target)
                if model == 0 or self.general:
                    view = None
                    if self.fast_admission and reservation.get("rank_observation") is not None:
                        view = list(observations[model])
                        view[target] = reservation["rank_observation"]
                    self.end_change(model, view, at=started)
                report.update(target=target, admitted=not reservation.get("reservation_pending", False),
                              submitted=True, reservation_pending=reservation.get("reservation_pending", False))
            report["control_s"] = time.monotonic() - started
            report["observations"] = observations
            return report

    async def run_transaction(self, plan_fn, label, admission=None, admit=None, pause_models=(0, 1), hold_after=(),
                              admission_headroom=None):
        """Pause `pause_models` at a DP boundary, apply plan_fn's moves, optionally admit, then resume.

        Only tenants whose requests move need pausing. An unpaused tenant may still grow KV per step, so the
        admission is re-checked on fresh snapshots after the moves (with `admission_headroom(current)` for the target).
        Waiting or retracted requests may coexist; the planner only moves movable (decoding) requests.
        Tenants in `hold_after` stay paused as held after a successful admission (time-multiplexing).
        """
        pause_models = tuple(sorted(set(pause_models)))
        async with self.lock:
            started = time.monotonic()
            summaries = await self.summaries()
            report = {"path": label, "started_at": started, "moves": [], "admitted": False, "status": None,
                      "pause_models": list(pause_models)}
            transaction = uuid.uuid4().hex
            # Held tenants are already paused (time-multiplexing) and stay paused after the transaction.
            to_pause = tuple(model for model in pause_models if model not in self.held)
            report["held_models"] = [model for model in pause_models if model in self.held]
            report["pause_receipts"] = (await self.pause_boundary(summaries, to_pause) if to_pause
                                        else [None, None])
            paused_at = time.monotonic()
            report["pause_wait_s"] = paused_at - started
            completed = False
            try:
                if pause_models:
                    observed = await self.begin(transaction, pause_models)
                else:
                    observed = await self.snapshots()
                solve_started = time.monotonic()
                plan = await asyncio.to_thread(plan_fn, observed)
                report.update(solve_s=time.monotonic() - solve_started, plan=plan, observations=observed)
                moves = [move for move in plan.get("moves", []) if move["model"] in pause_models]
                if len(moves) != len(plan.get("moves", [])):
                    report["dropped_unpaused_moves"] = len(plan["moves"]) - len(moves)
                try:
                    for move in moves:
                        report["moves"].append(await self.migrate(
                            transaction, move["model"], move["id"], move["source"], move["destination"]))
                except CleanMigrationFailure as error:
                    report["move_failure"] = str(error)
                if self.general and report["moves"]:
                    self.drop_views({event["model"] for event in report["moves"]})
                if report["moves"]:
                    report["transfer_memory"] = await self.memory(release_unused=True, models=pause_models)
                current = await self.snapshots(refresh_physical_capacity=bool(report["moves"]))
                report["used_after_moves_gib"] = self.check_capacity(current)
                target = plan.get("target")
                if (admission is not None and target is not None and "move_failure" not in report
                        and "dropped_unpaused_moves" not in report):
                    margin = admission_headroom(current)[target] if admission_headroom is not None else 0.0
                    report["admission_headroom_gib"] = margin
                    if fits(current, admission["model"], target, admission["tokens"], self.budgets, headroom=margin):
                        model = admission["model"]
                        admitted_at = time.monotonic()
                        self.begin_change(model)
                        reservation = await admit(target)
                        if model == 0 or self.general:
                            view = None
                            if self.fast_admission and reservation.get("rank_observation") is not None:
                                view = list(current[model])
                                view[target] = reservation["rank_observation"]
                            self.end_change(model, view, at=admitted_at)
                        report.update(target=target, submitted=True,
                                      admitted=not reservation.get("reservation_pending", False))
                    else:
                        report["post_move_misfit"] = True
                if pause_models:
                    await self.end(transaction, pause_models)
                report["status"] = "completed"
                completed = True
            finally:
                if completed:
                    resumed_at = time.monotonic()
                    keep = tuple(model for model in to_pause if model in hold_after and report.get("submitted"))
                    if set(to_pause) - set(keep):
                        await self.resume(tuple(model for model in to_pause if model not in keep))
                    for model in keep:
                        self.held[model] = {"model": model, "requested_at": started, "paused_at": paused_at,
                                            "from_transaction": True,
                                            "requests_at_pause": sum(len(rank["requests"])
                                                                     for rank in summaries[model])}
                    finished = time.monotonic()
                    report["resume_s"] = finished - resumed_at
                    report["control_s"] = finished - started
                    report["paused_request_s"] = sum(
                        len(summaries[model][rank]["requests"]) * max(0.0, resumed_at - receipt["reached_at"])
                        for model in pause_models for rank, receipt in enumerate(report["pause_receipts"][model] or [])
                        if "reached_at" in receipt)
            self.transactions.append({key: value for key, value in report.items() if key != "observations"})
            return report

    async def begin(self, transaction, models=(0, 1)):
        observed = await self.snapshots()
        for model in models:
            observed[model] = list(await asyncio.gather(*[self.call(model, rank, "placement_begin", transaction)
                                                          for rank in range(self.num_ranks)]))
        return observed

    async def end(self, transaction, models=(0, 1)):
        await asyncio.gather(*[self.call(model, rank, "placement_end", transaction)
                               for model in models for rank in range(self.num_ranks)])

    async def migrate(self, transaction, model, request_id, source, destination):
        if source == destination or not (0 <= source < self.num_ranks and 0 <= destination < self.num_ranks):
            raise ValueError("Migration needs distinct valid ranks")
        started = time.monotonic()
        stage_started, stage_s = started, {}

        def mark(name):
            nonlocal stage_started
            now = time.monotonic()
            stage_s[name] = now - stage_started
            stage_started = now

        prepare_started = peer_started = peer_completed = remove_started = False
        try:
            payload = await self.call(model, source, "placement_export", transaction, request_id)
            num_blocks = len(payload["blocks"])
            summaries = await self.summaries()
            destination_view = summaries[model][destination]
            if not fits(summaries, model, destination, num_blocks * destination_view["block_size"], self.budgets):
                raise CleanMigrationFailure(f"Destination {destination} lacks headroom for {request_id}")
            mark("export_and_check")
            prepare_started = True
            imported = await self.call(model, destination, "placement_prepare", transaction, payload)
            if len(imported["blocks"][0]) != num_blocks:
                raise RuntimeError("Destination block count differs from source")
            copy_peak = self.check_capacity(await self.summaries())
            mark("destination_prepare")
            peer_started = True
            results = await asyncio.gather(*[
                self.call(model, rank, "placement_kv_peer", transaction,
                          payload["blocks"] if rank == source else imported["blocks"][0] if rank == destination else [],
                          source, destination, num_blocks)
                for rank in range(self.num_ranks)], return_exceptions=True)
            errors = [result for result in results if isinstance(result, BaseException)]
            if errors:
                raise RuntimeError("Peer transfer failed") from errors[0]
            from tp_migration import verify_transfer_endpoints

            verify_transfer_endpoints(results[source], results[destination], self.tensor_parallel_size)
            peer_completed = True
            mark("peer_copy")
            await self.call(model, destination, "placement_runner_state", transaction, imported)
            await self.call(model, destination, "placement_commit", transaction, request_id)
            mark("destination_commit")
            remove_started = True
            after = await self.call(model, source, "placement_remove", transaction, request_id)
            client = self.engines[model].engine_core
            client.reqs_in_flight[request_id] = client.core_engines[destination]
            await self.call(model, source, "placement_runner_state", transaction, {"request_id": request_id, "remove": True})
            mark("source_release")
            event = {"request_id": request_id, "model": model, "source": source, "destination": destination,
                     "transaction": transaction, "started_at": started, "elapsed_s": time.monotonic() - started,
                     "kv_bytes": results[source]["bytes"], "num_blocks": num_blocks,
                     "source_mapped_after": after["mapped_bytes"], "copy_peak_gib": copy_peak, "stage_s": stage_s,
                     "worker_stage_s": {"source": results[source]["stage_s"],
                                        "destination": results[destination]["stage_s"]},
                     "tensor_parallel_size": self.tensor_parallel_size,
                     "worker_shards": {"source": results[source].get("shards", []),
                                       "destination": results[destination].get("shards", [])},
                     "bitwise_verified": True}
            self.events.append(event)
            return event
        except CleanMigrationFailure:
            if prepare_started:
                await self._rollback(transaction, model, destination, request_id)
            raise
        except BaseException as error:
            if peer_started and not peer_completed:
                raise RuntimeError("Uncertain peer transfer; engines stay paused, experiment must stop") from error
            if remove_started:
                raise RuntimeError("Post-commit migration failure; engines stay paused, experiment must stop") from error
            if prepare_started:
                await self._rollback(transaction, model, destination, request_id)
            raise CleanMigrationFailure(f"{type(error).__name__}: {error}") from error

    async def _rollback(self, transaction, model, destination, request_id):
        await self.call(model, destination, "placement_rollback", transaction, request_id)
        await self.call(model, destination, "placement_runner_state", transaction, {"request_id": request_id, "remove": True})
