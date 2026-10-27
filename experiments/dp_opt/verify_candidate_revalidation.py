import math

from verify_benefit import verify_ownership


def verify_projection(snapshot, original, projected):
    requests = {request["id"]: request for request in snapshot["movable"]}
    memory = list(snapshot["initial_gib"])
    peaks = list(memory)
    counts = [list(values) for values in snapshot["counts"]]
    counts[0][original["target"]] += 1
    moved, released = set(), set()
    reused = {requests[identifier].get("destination_pool") for identifier in original["moves"]}
    rejection = None
    for identifier in original["moves"]:
        request = requests[identifier]
        source, destination = request["source"], 1 - request["source"]
        memory[destination] += request["memory_gib"]
        peaks[destination] = max(peaks[destination], memory[destination])
        if memory[destination] > snapshot["capacity_gib"][destination] + 1e-6:
            rejection = "revalidation_copy_peak_rejected"
            break
        moved.add(identifier)
        for page in snapshot["source_pages"]:
            if (page["id"] not in released and page["owners"] and not page["pinned"]
                    and page["pool"] not in reused and all(owner in moved for owner in page["owners"])):
                released.add(page["id"])
                memory[page["rank"]] -= page["memory_gib"]
        counts[request["model"]][source] -= 1
        counts[request["model"]][destination] += 1
    if rejection is None:
        memory[original["target"]] += snapshot["new_gib"]
        if any(used > capacity + 1e-6 for used, capacity in zip(memory, snapshot["capacity_gib"])):
            rejection = "revalidation_admission_rejected"
    if rejection is not None:
        if projected["status"] != rejection or "objective_s" in projected:
            raise ValueError("Revalidated candidate accepted unsafe copy/admission")
        return
    performance = sum(table[counts[model][0]] for model, table in enumerate(snapshot["performance_cost_tables_s"]))
    migration = sum(requests[identifier]["move_cost_s"] for identifier in original["moves"])
    admission = snapshot.get("admission_cost_s", [0, 0])[original["target"]]
    if (projected["moves"] != original["moves"] or projected["final_counts"] != counts
            or projected["released_source_pages"] != sorted(released)
            or projected["status"] != "revalidated_feasible_not_current_optimum"):
        raise ValueError("Revalidated candidate identity/count/release/scope mismatch")
    expected = {"performance_cost_s": performance, "migration_cost_s": migration,
                "admission_cost_s": admission, "objective_s": performance + migration + admission}
    for name, value in expected.items():
        if not math.isclose(projected[name], value, abs_tol=1e-8, rel_tol=1e-9):
            raise ValueError("Revalidated candidate cost mismatch: " + name)
    for name, values in (("final_gib", memory), ("peak_gib", [max(peak, used) for peak, used in zip(peaks, memory)])):
        if any(not math.isclose(actual, value, abs_tol=1e-8) for actual, value in zip(projected[name], values)):
            raise ValueError("Revalidated candidate physical memory mismatch")


def verify_revalidation(decision, fit, lookup):
    from live_controller import admission_candidates, admission_planning_snapshot, wait_migration_precheck

    evidence = decision["original_candidate"]
    prompt, output, budgets, policy, tenant, max_moves, limit, fixed, bandwidth, solver_limit = evidence["settings"]
    previous = {request["id"]: request for request in evidence["selected"]}
    original = [candidate for candidate in evidence["plan"].get("candidates", []) if "objective_s" in candidate]
    if not original and evidence["plan"].get("winner") is not None:
        original = [evidence["plan"]["winner"]]
    retained = {identifier for candidate in original for identifier in candidate["moves"]}
    if retained != set(previous):
        raise ValueError("Original selected identity inventory differs from original candidates")
    checks = decision["candidate_revalidations"]
    if not checks or checks[0]["stage"] != "before_pause" or len(checks) > 2:
        raise ValueError("Missing candidate-level before-pause safety evidence")
    for index, check in enumerate(checks):
        if check["stage"] != ("before_pause" if index == 0 else "after_boundary"):
            raise ValueError("Candidate revalidation stages out of order")
        if not math.isfinite(check["elapsed_s"]) or check["elapsed_s"] < 0:
            raise ValueError("Invalid candidate revalidation elapsed cost")
        observed = check["observations"]
        verify_ownership(observed, budgets)
        current = {request["id"]: {"tenant": model, "rank": rank, **request}
                   for model, ranks in enumerate(observed) for rank, observation in enumerate(ranks)
                   for request in observation["requests"]}
        if evidence["geometry"] != [[{name: observation[name] for name in geometry}
                                      for observation, geometry in zip(ranks, old_ranks)]
                                     for ranks, old_ranks in zip(observed, evidence["geometry"])]:
            raise ValueError("Candidate page geometry changed")
        snapshot, horizon, population = admission_planning_snapshot(
            observed, tenant, prompt + output, output, budgets, fit, True, policy, limit, fixed, bandwidth, lookup, retained)
        if (check["snapshot"] != snapshot or check["horizon_steps"] != horizon
                or check["active_plus_incoming"] != population):
            raise ValueError("Candidate snapshot differs from current raw ownership/lengths/costs")
        candidates = original + [{"target": target, "moves": [], "status": "current_direct_allocator_fits"}
                                 for target in admission_candidates(observed, tenant, prompt + output, budgets)[1]]
        if len(check["plan"]["candidates"]) != len(candidates):
            raise ValueError("Candidate revalidation dropped or added an original action")
        for candidate, projected in zip(candidates, check["plan"]["candidates"]):
            invalid = [identifier for identifier in candidate["moves"] if identifier not in current
                       or any(current[identifier][name] != previous[identifier][name] for name in ("tenant", "rank", "blocks"))
                       or not current[identifier]["movable"]]
            if invalid or len(candidate["moves"]) > max_moves:
                if (projected["status"] != "revalidation_selected_identity_or_blocks_rejected"
                        or projected["invalid_selected_ids"] != invalid):
                    raise ValueError("Candidate accepted changed selected request")
            else:
                verify_projection(snapshot, candidate, projected)
        feasible = [candidate for candidate in check["plan"]["candidates"] if "objective_s" in candidate]
        direct = [candidate for candidate in feasible if not candidate["moves"]]
        if policy == "kv_only" and direct:
            winner = min(direct, key=lambda candidate: (snapshot["counts"][0][candidate["target"]],
                                                       snapshot["initial_gib"][candidate["target"]], candidate["target"]))
        else:
            winner = min(feasible, key=lambda candidate: (candidate["objective_s"], len(candidate["moves"]), candidate["target"])) if feasible else None
        if check["plan"]["winner"] != winner or check["plan"]["proven_infeasible"]:
            raise ValueError("Candidate selection is not the cheapest currently revalidated original action")
        gate = check["wait_gate"]
        if gate is not None:
            options = check["wait_recheck_options"]
            if not math.isclose(options["copy_seconds_per_gib"], 1 / bandwidth, abs_tol=1e-10):
                raise ValueError("Candidate wait cost differs from frozen migration bandwidth")
            rebuilt = wait_migration_precheck(observed, tenant, prompt + output, budgets,
                max_moves=max_moves, candidate_limit=limit, **options)
            if rebuilt != gate:
                raise ValueError("Revalidated wait gate differs from current natural-release estimate")
    if decision.get("winner") is not None:
        final = checks[-1]
        if len(checks) != 2 or decision["winner"] != final["plan"]["winner"]:
            raise ValueError("Executed candidate lacks frozen after-boundary safety replay")
        if decision["winner"]["moves"] and final["wait_gate"] is not None:
            threshold = final["wait_gate"]["estimated_wait_s"]
            threshold = threshold * final["wait_gate"]["safety_fraction"] if threshold is not None else 0
            if not math.isclose(decision["wait_cost_s"], threshold, abs_tol=1e-8):
                raise ValueError("Executed migration used stale pre-boundary wait prediction")
    if decision.get("moves"):
        final = checks[-1]
        if [move["request_id"] for move in decision["moves"]] != decision["winner"]["moves"]:
            raise ValueError("Actual migration order differs from revalidated selected IDs")
        requests = {request["id"]: (model, rank, request, observation)
                    for model, ranks in enumerate(final["observations"])
                    for rank, observation in enumerate(ranks) for request in observation["requests"]}
        for move in decision["moves"]:
            model, source, request, observation = requests[move["request_id"]]
            destination = 1 - source
            if (move["model"] != model or move["source"] != source or move["destination"] != destination
                    or move["kv_bytes"] != len(request["blocks"]) * observation["block_bytes"]
                    or move.get("integrity") != "device_echo_full_bytewise"
                    or move.get("bitwise_verified") is not True):
                raise ValueError("Actual migration tenant/source/bytes/exact echo differs from frozen selection")
            after = decision["before_admission"]
            owners = [item for item in after[model][destination]["requests"] if item["id"] == move["request_id"]]
            if (len(owners) != 1 or len(owners[0]["blocks"]) != len(request["blocks"])
                    or any(item["id"] == move["request_id"] for item in after[model][source]["requests"])):
                raise ValueError("Migrated request ownership/blocks did not move to the planned destination")
    tradeoff = decision.get("wait_tradeoff")
    if tradeoff is not None:
        final = checks[-1]
        winner, snapshot = final["plan"]["winner"], final["snapshot"]
        baseline = [list(values) for values in snapshot["counts"]]
        baseline[0][winner["target"]] += 1
        baseline_cost = sum(table[baseline[model][0]] for model, table in enumerate(snapshot["performance_cost_tables_s"]))
        migration_cost = (final["active_plus_incoming"] * decision["future_transaction_fixed_s"]
                          + winner["migration_cost_s"] + max(0, winner["performance_cost_s"] - baseline_cost))
        fixed = (decision["wait_fixed_cost_s"] + tradeoff["blocking_at_gate_request_s"]
                 + tradeoff["elapsed_at_gate_s"] + decision["preplan_s"])
        if (not math.isclose(tradeoff["migration_cost_s"], migration_cost, abs_tol=1e-8)
                or not math.isclose(tradeoff["fixed_cost_s"], fixed, abs_tol=1e-8)
                or tradeoff["accepted"] != (migration_cost + fixed < tradeoff["threshold_s"])
                or not math.isclose(tradeoff["threshold_s"], decision["wait_cost_s"], abs_tol=1e-8)
                or (decision.get("winner") is not None) != tradeoff["accepted"]):
            raise ValueError("Revalidated migration ignored current wait/performance/control tradeoff")
