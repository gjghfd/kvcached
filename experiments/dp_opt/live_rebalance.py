import asyncio
import math
import time

from ilp_reference import enumerate_targets, replay_candidate
from live_controller import admission_planning_snapshot, stable_ownership


def plan_rebalance(observed, budgets, fit, allow_extrapolation, settings, layout_lookup=None,
                   blocking_cost_s=0, phase="before_pause"):
    if settings.get("policy_mode") == "workaware":
        from workaware_rebalance import plan_workaware_rebalance

        return plan_workaware_rebalance(observed, budgets, fit, allow_extrapolation, settings,
                                       layout_lookup, blocking_cost_s, phase)
    if settings.get("policy_mode") == "headroom":
        from headroom_rebalance import plan_headroom_rebalance

        return plan_headroom_rebalance(observed, budgets, fit, allow_extrapolation, settings,
                                       layout_lookup, blocking_cost_s, phase)
    if settings.get("policy_mode") == "amortized":
        from amortized_rebalance import plan_amortized_rebalance

        return plan_amortized_rebalance(observed, budgets, fit, allow_extrapolation, settings,
                                       layout_lookup, blocking_cost_s, phase)
    snapshot, horizon, population = admission_planning_snapshot(
        observed, 0, 0, 0, budgets, fit, allow_extrapolation, "ilp", settings["candidate_limit"],
        0, settings["move_bandwidth_gib_s"], layout_lookup, new_request_count=0)
    reference = replay_candidate(snapshot, {"target": None, "moves": [], "status": "unchanged_layout"})
    plan = enumerate_targets(snapshot, max_moves=settings["max_moves"], time_limit_s=settings["solver_limit_s"])
    winner = plan["winner"]
    if winner is None or winner["objective_s"] >= reference["objective_s"]:
        winner = reference
    gain = reference["performance_cost_s"] - winner["objective_s"]
    fixed = population * settings["pause_fixed_s"]
    gate = {"accepted": bool(winner["moves"]) and gain > fixed + blocking_cost_s + settings["min_gain_s"],
            "predicted_gain_after_copy_request_s": gain, "fixed_control_request_s": fixed,
            "measured_blocking_so_far_request_s": blocking_cost_s, "min_gain_request_s": settings["min_gain_s"]}
    return {"snapshot": snapshot, "horizon_steps": horizon, "active_population": population,
            "plan": plan, "reference": reference, "winner": winner, "gate": gate}


async def rebalance_after_release(controller, budgets, max_moves=4, candidate_limit=8,
                                  move_bandwidth_gib_s=1, solver_limit_s=1, pause_fixed_s=.475,
                                  min_gain_s=1, admission_busy=None, balance_ledger=None, workaware_fit=None):
    from verify_benefit import verify_ownership

    settings = {"max_moves": max_moves, "candidate_limit": candidate_limit,
                "move_bandwidth_gib_s": move_bandwidth_gib_s, "solver_limit_s": solver_limit_s,
                "pause_fixed_s": pause_fixed_s, "min_gain_s": min_gain_s}
    if (len(budgets) != 2 or any(not math.isfinite(value) or value <= 0 for value in budgets)
            or type(max_moves) is not int or max_moves < 0
            or type(candidate_limit) is not int or candidate_limit < 1
            or any(not math.isfinite(value) or value <= 0 for value in (move_bandwidth_gib_s, solver_limit_s))
            or any(not math.isfinite(value) or value < 0 for value in (pause_fixed_s, min_gain_s))):
        raise ValueError("Invalid release-rebalance budgets, limits or costs")
    started = time.monotonic()
    report = {"path": "release_rebalance_ilp", "started_at": started, "moves": [], "admitted": False, "submitted": False,
              "settings": settings, "pause_call_s": 0, "solve_s": 0, "preplan_s": 0,
              "end_to_end_control_s": 0, "new_request_count": 0,
              "cost_scope": "remaining_decode_single_model_capture_surrogate_or_Conv_lookup_not_joint_calibration"}

    def finish(reason, deferred=False):
        report.update({"reason": reason, "deferred": deferred,
                       "end_to_end_control_s": time.monotonic() - started - report["preplan_s"]})
        report["end_to_end_control_s_including_preplan"] = report["end_to_end_control_s"] + report["preplan_s"]
        return report

    def busy():
        return admission_busy is not None and admission_busy()

    async with controller.lock:
        if busy():
            return finish("admission_has_priority", True)
        observed = await controller.snapshots()
        generation = controller.admission_generation
    verify_ownership(observed, budgets)
    report["observations"] = observed
    if balance_ledger is not None:
        evidence = balance_ledger.evidence(observed, time.monotonic())
        report["balance_evidence"] = evidence
        settings.update({"policy_mode": "amortized", "credit_request_s": evidence["credit_request_s"],
                         "pause_fixed_s": evidence["pause_fixed_s"],
                         "overhead_fraction": evidence["overhead_fraction"]})
        if "headroom_profile" in evidence:
            settings.update({"policy_mode": "headroom", "headroom_profile": evidence["headroom_profile"]})
        if "workaware_profile" in evidence:
            if workaware_fit is None:
                raise ValueError("Workaware recovery requires independently verified raw calibration")
            from workaware_cost import remaining_segments

            settings.update({"policy_mode": "workaware", "workaware_profile": evidence["workaware_profile"],
                             "workaware_fit": workaware_fit, "work_interval_steps": 128})
            report["cost_scope"] = "remaining_work_graph_context_attention_and_soft_Poisson_headroom"
            try:
                remaining_segments(observed, workaware_fit)
            except ValueError as error:
                report["calibration_deferral"] = str(error)
                return finish("outside_verified_workaware_calibration", True)
        report["cost_scope"] = ("remaining_work_graph_context_attention_and_soft_Poisson_headroom"
                                if settings.get("policy_mode") == "workaware" else
                                "past_work_credit_activation_ILP_and_single_model_capture_surrogate_not_joint_calibration")
        if not evidence["ready_epochs"]:
            return finish("no_capacity_displacement_ready_for_recovery")
    if not any(rank["requests"] for ranks in observed for rank in ranks):
        return finish("empty_population")
    if any(not request["movable"] for ranks in observed for rank in ranks for request in rank["requests"]):
        return finish("population_not_all_decode", True)
    if (balance_ledger is not None and settings.get("policy_mode") != "workaware" and settings["credit_request_s"] < sum(
            len(rank["requests"]) for ranks in observed for rank in ranks) * settings["pause_fixed_s"]):
        return finish("migration_credit_not_yet_amortized", True)
    planned_at = time.monotonic()
    before = await asyncio.to_thread(plan_rebalance, observed, budgets, controller.fit,
                                    controller.allow_extrapolation, settings, controller.layout_lookup)
    report["preplan_s"] = time.monotonic() - planned_at
    report["before_pause"] = before
    if not before["gate"]["accepted"]:
        return finish("no_profitable_rebalance", settings.get("policy_mode") in ("headroom", "workaware"))
    async with controller.lock:
        current = await controller.snapshots()
        if busy():
            return finish("admission_has_priority", True)
        if generation != controller.admission_generation or stable_ownership(current) != stable_ownership(observed):
            report["current_before_pause"] = current
            return finish("rebalance_stale_before_pause", True)
        if any(not request["movable"] for ranks in current for rank in ranks for request in rank["requests"]):
            return finish("population_not_all_decode", True)
        controller.admission_observations = None
        controller.admission_generation += 1
        controller.model_admission_generations = [value + 1 for value in controller.model_admission_generations]
        pause_started = time.monotonic()
        receipts = await controller.pause_boundary(current)
        report["pause_receipts"] = receipts
        report["pause_call_s"] = time.monotonic() - pause_started
        completed = False
        try:
            eligible = await controller.snapshots()
            if any(not request["movable"] for ranks in eligible for rank in ranks for request in rank["requests"]):
                report["after_boundary"] = eligible
                completed = True
                return finish("population_not_all_decode_after_pause", True)
            transaction = f"rebalance-{time.monotonic_ns()}"
            frozen = await controller.begin(transaction)
            verify_ownership(frozen, budgets)
            blocking_sampled_at = time.monotonic()
            blocking = sum(len(frozen[model][rank]["requests"]) * max(0, blocking_sampled_at - receipt["reached_at"])
                           for model, ranks in enumerate(receipts) for rank, receipt in enumerate(ranks))
            solved_at = time.monotonic()
            fresh = await asyncio.to_thread(plan_rebalance, frozen, budgets, controller.fit,
                                           controller.allow_extrapolation, settings, controller.layout_lookup, blocking,
                                           phase="after_pause")
            report["solve_s"] = time.monotonic() - solved_at
            extra_blocking = fresh["active_population"] * report["solve_s"]
            if settings.get("policy_mode") == "workaware":
                fresh["gate"]["measured_blocking_so_far_request_s"] = blocking + extra_blocking
            elif balance_ledger is not None:
                from amortized_rebalance import amortized_gate

                fresh["gate"] = amortized_gate(fresh["reference"], fresh["winner"], settings,
                                               fresh["active_population"], blocking + extra_blocking, "after_pause")
            else:
                fresh["gate"]["measured_blocking_so_far_request_s"] += extra_blocking
                fresh["gate"]["accepted"] = (fresh["gate"]["accepted"] and
                    fresh["gate"]["predicted_gain_after_copy_request_s"] > fresh["gate"]["fixed_control_request_s"]
                    + fresh["gate"]["measured_blocking_so_far_request_s"] + min_gain_s)
            report["after_pause"] = fresh
            report["blocking_sampled_at"] = blocking_sampled_at
            report["frozen_observations"] = frozen
            if busy() or not fresh["gate"]["accepted"]:
                await controller.end(transaction)
                completed = True
                return finish("admission_has_priority_after_pause" if busy() else "no_profit_after_fresh_solve",
                              busy() or settings.get("policy_mode") in ("headroom", "workaware"))
            by_id = {request["id"]: request for request in fresh["snapshot"]["movable"]}
            for identifier in fresh["winner"]["moves"]:
                event = await controller.migrate(transaction, by_id[identifier], budgets)
                event["purpose"] = "release_rebalance"
                report["moves"].append(event)
            report["transfer_memory_after_cache_release"] = await controller.release_transfer_cache()
            after = await controller.snapshots(refresh_physical_capacity=True)
            verify_ownership(after, budgets)
            if settings.get("policy_mode") == "workaware":
                used = [sum(after[model][rank]["mapped_bytes"] for model in (0, 1)) / 1024**3 for rank in (0, 1)]
                if any(value > cap + 1e-6 for value, cap in zip(used, fresh["snapshot"]["final_capacity_gib"])):
                    raise RuntimeError("Actual post-copy pages violated selected soft-headroom protection")
            if settings.get("policy_mode") == "headroom":
                used = [sum(after[model][rank]["mapped_bytes"] for model in (0, 1)) / 1024**3 for rank in (0, 1)]
                if any(not any(used[rank] + guard["demand_gib"][rank] <= budgets[rank] + 1e-6
                        for rank in (0, 1)) for guard in fresh["headroom_guards"]):
                    raise RuntimeError("Actual post-copy pages violated the protected admission headroom")
            inventory = lambda observations: {request["id"]: (model, request["prompt_tokens"], request["max_tokens"],
                request["output_tokens"], request.get("computed_tokens"), len(request["blocks"]))
                for model, ranks in enumerate(observations) for rank in ranks for request in rank["requests"]}
            if inventory(frozen) != inventory(after):
                raise RuntimeError("Release rebalance changed the request population or decode state")
            if [[len(rank["requests"]) for rank in ranks] for ranks in after] != fresh["winner"]["final_counts"]:
                raise RuntimeError("Release rebalance final counts differ from the fresh ILP plan")
            report["after_rebalance"] = after
            await controller.end(transaction)
            completed = True
            return finish("rebalanced")
        finally:
            if completed:
                resumed_at = time.monotonic()
                await controller.resume()
                report["resume_s"] = time.monotonic() - resumed_at
                report["end_to_end_control_s"] = time.monotonic() - started - report["preplan_s"]
                report["end_to_end_control_s_including_preplan"] = report["end_to_end_control_s"] + report["preplan_s"]
                if balance_ledger is not None:
                    report["resume_completed_at"] = time.monotonic()
                report["active_blocking_upper_request_s"] = sum(
                    len(current[model][rank]["requests"]) * max(0,
                        report.get("resume_completed_at", time.monotonic()) - receipt["reached_at"])
                    for model, ranks in enumerate(receipts) for rank, receipt in enumerate(ranks))


class ReleaseRebalancer:
    def __init__(self, attempt, record, retry_s=2, cooldown_s=10, retry_on_release_only=False):
        if any(not math.isfinite(value) or value <= 0 for value in (retry_s, cooldown_s)):
            raise ValueError("Rebalance retry and cooldown must be finite and positive")
        self.attempt = attempt
        self.record = record
        self.retry_s = retry_s
        self.cooldown_s = cooldown_s
        self.retry_on_release_only = retry_on_release_only
        self.pending = {}
        self.event = asyncio.Event()
        self.closed = False
        self.not_before = 0
        self.task = asyncio.create_task(self.run())

    def notify(self, release):
        if self.closed:
            raise RuntimeError("Cannot notify a closed release rebalancer")
        self.pending[release["request_id"]] = release
        self.event.set()

    async def close(self):
        self.closed = True
        self.event.set()
        await self.task

    async def run(self):
        while not self.closed or self.pending:
            await self.event.wait()
            self.event.clear()
            if not self.closed:
                await asyncio.sleep(max(0, self.not_before - time.monotonic()))
            causes, self.pending = list(self.pending.values()), {}
            if not causes:
                continue
            report = dict(await self.attempt(causes))
            report["release_events"] = causes
            await self.record(report)
            if report["moves"]:
                self.not_before = time.monotonic() + self.cooldown_s
            if report.get("deferred") and not self.closed:
                self.pending = {**{item["request_id"]: item for item in causes}, **self.pending}
            delay = self.cooldown_s if report["moves"] else self.retry_s
            if self.pending and not self.closed and not self.retry_on_release_only:
                await asyncio.sleep(delay)
                self.event.set()
