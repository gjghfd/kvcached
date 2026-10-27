import argparse
import json
import math
from pathlib import Path

SUPPORTS_WORK_SEGMENTS = True


def added_work_cost(snapshot, moved_ids):
    if not snapshot.get("work_segments") and "placement_risk_cost_s" not in snapshot:
        return {"performance_cost_s": 0, "graph_cost_s": 0, "attention_cost_s": 0,
                "counts_by_segment": [], "placement_risk_cost_s": 0}
    from workaware_ilp import work_cost

    return work_cost(snapshot, moved_ids)


def validate_snapshot(snapshot):
    if snapshot.get("work_segments") or "placement_risk_cost_s" in snapshot:
        from workaware_ilp import validate_work_snapshot

        validate_work_snapshot(snapshot)
    incoming = snapshot.get("new_request_count", 1)
    if isinstance(incoming, bool) or not isinstance(incoming, int) or incoming not in (0, 1):
        raise ValueError("New request count must be zero or one")
    tables = snapshot.get("performance_cost_tables_s")
    joint_table = snapshot.get("joint_performance_costs_s")
    for name in ("counts", "capacity_gib", "initial_gib"):
        if len(snapshot[name]) != 2:
            raise ValueError("Reference scope is two models and two ranks")
    if any(len(counts) != 2 or any(not isinstance(count, int) or count < 0 for count in counts)
           for counts in snapshot["counts"]):
        raise ValueError("Populations must be nonnegative integer counts")
    imbalance_costs = snapshot.get("imbalance_cost_s", [])
    if ("imbalance_cost_s" in snapshot or (tables is None and joint_table is None)) and len(imbalance_costs) != 2:
        raise ValueError("Affine mode requires two imbalance coefficients")
    values = snapshot["capacity_gib"] + snapshot["initial_gib"] + imbalance_costs + [snapshot["new_gib"]]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Memory and costs must be finite and nonnegative")
    if any(cost <= 0 for cost in imbalance_costs):
        raise ValueError("Imbalance coefficients must be positive")
    admission_costs = snapshot.get("admission_cost_s", [0, 0])
    if len(admission_costs) != 2 or any(not math.isfinite(cost) or cost < 0 for cost in admission_costs):
        raise ValueError("Admission costs must be two finite nonnegative request-second values")
    if incoming == 0 and (snapshot["new_gib"] != 0 or any(admission_costs)):
        raise ValueError("Rebalancing has no new request memory or admission cost")
    for field in ("migration_activation_cost_s", "migration_budget_s"):
        if field in snapshot and (isinstance(snapshot[field], bool)
                                  or not math.isfinite(snapshot[field]) or snapshot[field] < 0):
            raise ValueError("Migration activation and budget must be finite nonnegative request-seconds")
    if tables is not None and joint_table is not None:
        raise ValueError("Select separate or joint layout costs, not both")
    populations = [sum(snapshot["counts"][0]) + incoming, sum(snapshot["counts"][1])]
    if tables is not None:
        if len(tables) != 2 or any(len(table) != population + 1 for table, population in zip(tables, populations)):
            raise ValueError("Each cost table must cover every final rank-zero population")
        if any(not math.isfinite(cost) or cost < 0 for table in tables for cost in table):
            raise ValueError("Layout table costs must be finite and nonnegative")
    if joint_table is not None:
        if (len(joint_table) != populations[0] + 1
                or any(len(row) != populations[1] + 1 for row in joint_table)):
            raise ValueError("Joint costs must cover every final pair of rank-zero populations")
        if any(not math.isfinite(cost) or cost < 0 for row in joint_table for cost in row):
            raise ValueError("Joint layout costs must be finite and nonnegative")
    if any(used > capacity for used, capacity in zip(snapshot["initial_gib"], snapshot["capacity_gib"])):
        raise ValueError("Snapshot already exceeds its safe KV budget")
    final_capacity = snapshot.get("final_capacity_gib", snapshot["capacity_gib"])
    if (len(final_capacity) != 2 or any(isinstance(value, bool) or not math.isfinite(value)
            or not 0 <= value <= capacity for value, capacity in zip(final_capacity, snapshot["capacity_gib"]))):
        raise ValueError("Final-placement caps must be within the physical copy-peak budgets")
    identifiers = [request["id"] for request in snapshot["movable"]]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Movable request identifiers must be unique")
    for request in snapshot["movable"]:
        if request["source"] not in (0, 1) or request["model"] not in (0, 1):
            raise ValueError("Invalid request model or source rank")
        if any(not math.isfinite(request[name]) or request[name] < 0 for name in ("memory_gib", "move_cost_s")):
            raise ValueError("Request memory and migration cost must be finite and nonnegative")
    for model in (0, 1):
        for rank in (0, 1):
            movable_count = sum(request["model"] == model and request["source"] == rank
                                for request in snapshot["movable"])
            if movable_count > snapshot["counts"][model][rank]:
                raise ValueError("More movable requests than the active population")
    for rank in (0, 1):
        private_memory = sum(request["memory_gib"] for request in snapshot["movable"] if request["source"] == rank)
        if snapshot.get("source_pages") is None and private_memory > snapshot["initial_gib"][rank]:
            raise ValueError("Private request memory exceeds initial occupied memory")
    pages = snapshot.get("source_pages")
    if pages is not None:
        if len({page["id"] for page in pages}) != len(pages):
            raise ValueError("Physical source page identifiers must be unique")
        request_by_id = {request["id"]: request for request in snapshot["movable"]}
        for page in pages:
            if (page["rank"] not in (0, 1) or not math.isfinite(page["memory_gib"])
                    or page["memory_gib"] <= 0 or len(set(page["owners"])) != len(page["owners"])):
                raise ValueError("Invalid physical source page")
            if any(owner in request_by_id and request_by_id[owner]["source"] != page["rank"]
                   for owner in page["owners"]):
                raise ValueError("A page owner belongs to a different source rank")
        for rank in (0, 1):
            if sum(page["memory_gib"] for page in pages if page["rank"] == rank) > snapshot["initial_gib"][rank] + 1e-6:
                raise ValueError("Physical source pages exceed initial mapped memory")


def solve_candidate(snapshot, target=None, max_moves=None, time_limit_s=10):
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    validate_snapshot(snapshot)
    incoming = snapshot.get("new_request_count", 1)
    if (incoming and target not in (0, 1)) or (not incoming and target is not None):
        raise ValueError("Admission requires a rank; rebalancing requires target=None")
    requests = snapshot["movable"]
    if max_moves is not None and (not isinstance(max_moves, int) or max_moves < 0):
        raise ValueError("Move limit must be a nonnegative integer")
    if not math.isfinite(time_limit_s) or time_limit_s <= 0:
        raise ValueError("Solver time limit must be positive")
    stages = len(requests) if max_moves is None else min(max_moves, len(requests))
    move_dimensions = len(requests) * stages
    imbalance_positions = [move_dimensions, move_dimensions + 1]
    tables = snapshot.get("performance_cost_tables_s")
    joint_table = snapshot.get("joint_performance_costs_s")
    table_positions = []
    joint_entries = []
    source_pages = snapshot.get("source_pages")
    release_positions = []
    dimensions = move_dimensions + 2
    if tables is not None:
        for table in tables:
            table_positions.append(list(range(dimensions, dimensions + len(table))))
            dimensions += len(table)
    if joint_table is not None:
        reachable_counts = []
        for model in (0, 1):
            initial_count = snapshot["counts"][model][0] + incoming * int(model == 0 and target == 0)
            outgoing = sum(request["model"] == model and request["source"] == 0 for request in requests)
            incoming_moves = sum(request["model"] == model and request["source"] == 1 for request in requests)
            reachable_counts.append(range(initial_count - min(stages, outgoing),
                                          initial_count + min(stages, incoming_moves) + 1))
        for count_a in reachable_counts[0]:
            for count_b in reachable_counts[1]:
                joint_entries.append((count_a, count_b, dimensions))
                dimensions += 1
    if source_pages is not None:
        for page in source_pages:
            release_positions.append(list(range(dimensions, dimensions + stages + 1)))
            dimensions += stages + 1
    activation_position = None
    if "migration_activation_cost_s" in snapshot or "migration_budget_s" in snapshot:
        activation_position = dimensions
        dimensions += 1
    work_variables = []
    if snapshot.get("work_segments"):
        from workaware_ilp import allocate_work_variables

        work_variables, dimensions = allocate_work_variables(snapshot, dimensions)
    objective = np.zeros(dimensions)
    lower = np.zeros(dimensions)
    upper = np.ones(dimensions)
    upper[imbalance_positions] = [sum(snapshot["counts"][0]) + incoming, sum(snapshot["counts"][1])]
    integrality = np.zeros(dimensions)
    integrality[:move_dimensions] = 1
    if activation_position is not None:
        integrality[activation_position] = 1
        objective[activation_position] = snapshot.get("migration_activation_cost_s", 0)
    for positions in table_positions:
        integrality[positions] = 1
    for count_a, count_b, position in joint_entries:
        integrality[position] = 1
        objective[position] = joint_table[count_a][count_b]
    request_positions = {request["id"]: request_index for request_index, request in enumerate(requests)}
    for page, positions in zip(source_pages or [], release_positions):
        integrality[positions] = 1
        upper[positions[0]] = 0
        if page.get("pinned", False) or not page["owners"] or any(owner not in request_positions for owner in page["owners"]):
            upper[positions] = 0
    binary_positions = list(range(move_dimensions)) + [position for positions in table_positions
                                                       for position in positions] + [entry[2] for entry in joint_entries]
    binary_positions += [position for positions in release_positions for position in positions]
    binary_positions += [position for variables in work_variables for position in variables["counts"]]
    rows, row_lower, row_upper = [], [], []

    def constraint(row, minimum=-np.inf, maximum=np.inf):
        rows.append(row)
        row_lower.append(minimum)
        row_upper.append(maximum)

    if work_variables:
        from workaware_ilp import install_work_constraints

        install_work_constraints(snapshot, work_variables, requests, stages, dimensions,
                                 objective, upper, integrality, constraint)

    for request_index, request in enumerate(requests):
        row = np.zeros(dimensions)
        positions = slice(request_index * stages, (request_index + 1) * stages)
        row[positions] = 1
        objective[positions] = request["move_cost_s"]
        constraint(row, maximum=1)
    if activation_position is not None:
        for position in range(move_dimensions):
            row = np.zeros(dimensions)
            row[position], row[activation_position] = 1, -1
            constraint(row, maximum=0)
        row = np.zeros(dimensions)
        row[activation_position] = 1
        row[:move_dimensions] = -1
        constraint(row, maximum=0)
        if "migration_budget_s" in snapshot:
            row = np.zeros(dimensions)
            row[:move_dimensions] = objective[:move_dimensions]
            row[activation_position] = objective[activation_position]
            constraint(row, maximum=snapshot["migration_budget_s"])
    for stage in range(stages):
        row = np.zeros(dimensions)
        for request_index in range(len(requests)):
            row[request_index * stages + stage] = 1
        constraint(row, maximum=1)
    for page, positions in zip(source_pages or [], release_positions):
        if upper[positions[-1]] == 0:
            continue
        for completed in range(1, stages + 1):
            aggregate = np.zeros(dimensions)
            aggregate[positions[completed]] = 1
            for request_index, request in enumerate(requests):
                if page.get("pool") is not None and request.get("destination_pool") == page["pool"]:
                    row = np.zeros(dimensions)
                    row[positions[completed]] = 1
                    offset = request_index * stages
                    row[offset:offset + stages] = 1
                    constraint(row, maximum=1)
                    aggregate[offset:offset + stages] += 1
            for owner in page["owners"]:
                row = np.zeros(dimensions)
                row[positions[completed]] = 1
                offset = request_positions[owner] * stages
                row[offset:offset + completed] = -1
                constraint(row, maximum=0)
                aggregate[offset:offset + completed] -= 1
            constraint(aggregate, minimum=1 - len(page["owners"]))
    if joint_entries:
        row = np.zeros(dimensions)
        row[[entry[2] for entry in joint_entries]] = 1
        constraint(row, minimum=1, maximum=1)
    for model in (0, 1):
        initial_difference = snapshot["counts"][model][0] - snapshot["counts"][model][1]
        initial_difference += incoming * (1 if target == 0 else -1) if model == 0 else 0
        if tables is None and joint_table is None:
            objective[imbalance_positions[model]] = snapshot["imbalance_cost_s"][model]
        else:
            row = np.zeros(dimensions)
            if tables is not None:
                positions = table_positions[model]
                objective[positions] = tables[model]
                row[positions] = 1
                constraint(row.copy(), minimum=1, maximum=1)
                row[positions] = np.arange(len(positions))
            else:
                for count_a, count_b, position in joint_entries:
                    row[position] = count_a if model == 0 else count_b
            for request_index, request in enumerate(requests):
                if request["model"] == model:
                    change = -1 if request["source"] == 0 else 1
                    row[request_index * stages:(request_index + 1) * stages] = -change
            initial_rank_zero = snapshot["counts"][model][0] + incoming * int(model == 0 and target == 0)
            constraint(row, minimum=initial_rank_zero, maximum=initial_rank_zero)
        for sign in (-1, 1):
            row = np.zeros(dimensions)
            row[imbalance_positions[model]] = 1
            for request_index, request in enumerate(requests):
                if request["model"] == model:
                    change = -2 if request["source"] == 0 else 2
                    row[request_index * stages:(request_index + 1) * stages] = -sign * change
            constraint(row, minimum=sign * initial_difference)
    for rank in (0, 1):
        row = np.zeros(dimensions)
        for request_index, request in enumerate(requests):
            change = (-request["memory_gib"] if source_pages is None else 0) if request["source"] == rank else request["memory_gib"]
            row[request_index * stages:(request_index + 1) * stages] = change
        for page, positions in zip(source_pages or [], release_positions):
            if page["rank"] == rank:
                row[positions[-1]] -= page["memory_gib"]
        demand = snapshot["new_gib"] if rank == target else 0
        constraint(row, maximum=snapshot.get("final_capacity_gib", snapshot["capacity_gib"])[rank]
                   - snapshot["initial_gib"][rank] - demand)
        for stage in range(stages):
            row = np.zeros(dimensions)
            for request_index, request in enumerate(requests):
                for previous in range(stage + 1):
                    position = request_index * stages + previous
                    if request["source"] != rank:
                        row[position] += request["memory_gib"]
                    elif previous < stage and source_pages is None:
                        row[position] -= request["memory_gib"]
            for page, positions in zip(source_pages or [], release_positions):
                if page["rank"] == rank:
                    row[positions[stage]] -= page["memory_gib"]
            constraint(row, maximum=snapshot["capacity_gib"][rank] - snapshot["initial_gib"][rank])
    result = milp(objective, integrality=integrality, bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(np.array(rows), np.array(row_lower), np.array(row_upper)),
                  options={"mip_rel_gap": 0, "time_limit": time_limit_s})
    if result.status == 2:
        return {"status": "infeasible", "target": target}
    if result.x is None or result.status not in (0, 1):
        return {"status": "no_verified_incumbent", "target": target, "solver_message": result.message}
    if any(abs(result.x[position] - round(result.x[position])) > 1e-6 for position in binary_positions):
        return {"status": "no_verified_incumbent", "target": target, "solver_message": "Nonintegral incumbent"}
    moves = sorted((stage, request_index) for request_index in range(len(requests)) for stage in range(stages)
                   if result.x[request_index * stages + stage] > 0.5)
    repeated_request = len({request_index for stage, request_index in moves}) != len(moves)
    concurrent_moves = len({stage for stage, request_index in moves}) != len(moves)
    if repeated_request or concurrent_moves:
        raise RuntimeError("Solver violated serial one-move-per-request constraints")
    memory = snapshot["initial_gib"].copy()
    counts = [population.copy() for population in snapshot["counts"]]
    if incoming:
        counts[0][target] += incoming
    peaks = memory.copy()
    moved_ids, released_ids = set(), set()
    reused_pools = {requests[request_index].get("destination_pool") for _, request_index in moves}
    for _, request_index in moves:
        request = requests[request_index]
        source, destination = request["source"], 1 - request["source"]
        memory[destination] += request["memory_gib"]
        if memory[destination] > snapshot["capacity_gib"][destination] + 1e-6:
            raise RuntimeError("Solver returned an infeasible copy peak")
        peaks[destination] = max(peaks[destination], memory[destination])
        moved_ids.add(request["id"])
        if source_pages is None:
            memory[source] -= request["memory_gib"]
        else:
            for page in source_pages:
                if (page["id"] not in released_ids and page["owners"] and not page.get("pinned", False)
                        and (page.get("pool") is None or page["pool"] not in reused_pools)
                        and set(page["owners"]).issubset(moved_ids)):
                    memory[page["rank"]] -= page["memory_gib"]
                    released_ids.add(page["id"])
        counts[request["model"]][source] -= 1
        counts[request["model"]][destination] += 1
    if incoming:
        memory[target] += snapshot["new_gib"]
    if any(used > capacity + 1e-6 for used, capacity in zip(memory, snapshot["capacity_gib"])):
        raise RuntimeError("Solver returned an infeasible admission")
    if any(used > capacity + 1e-6 for used, capacity in zip(memory, snapshot.get("final_capacity_gib", snapshot["capacity_gib"]))):
        raise RuntimeError("Solver violated the final-placement headroom")
    differences = [abs(population[0] - population[1]) for population in counts]
    if joint_table is not None:
        performance_cost = joint_table[counts[0][0]][counts[1][0]]
        performance_mode = "joint_layout_table"
    elif tables is not None:
        performance_cost = sum(table[counts[model][0]] for model, table in enumerate(tables))
        performance_mode = "layout_table"
    else:
        performance_cost = sum(cost * difference for cost, difference in zip(snapshot["imbalance_cost_s"], differences))
        performance_mode = "affine_imbalance"
    migration_cost = sum(requests[request_index]["move_cost_s"] for _, request_index in moves)
    work = added_work_cost(snapshot, moved_ids)
    performance_cost += work["performance_cost_s"]
    if snapshot.get("work_segments"):
        performance_mode = "remaining_work_graph_and_attention"
    activation_cost = snapshot.get("migration_activation_cost_s", 0) if moves else 0
    if migration_cost + activation_cost > snapshot.get("migration_budget_s", math.inf) + 1e-6:
        raise RuntimeError("Solver violated the migration overhead budget")
    variable_score = performance_cost + migration_cost + activation_cost
    if not math.isclose(variable_score, result.fun, abs_tol=1e-6):
        raise RuntimeError("Independently recomputed objective differs from incumbent")
    score = variable_score + work["placement_risk_cost_s"]
    admission_cost = snapshot.get("admission_cost_s", [0, 0])[target] if incoming else 0
    dual_bound = result.mip_dual_bound
    if dual_bound is not None:
        dual_bound += work["placement_risk_cost_s"]
    if dual_bound is None and result.status == 0:
        dual_bound = score
    total_cost = score + admission_cost
    total_bound = None if dual_bound is None else dual_bound + admission_cost
    relative_gap = (max(0, total_cost - total_bound) / max(abs(total_cost), 1e-10)
                    if total_bound is not None else None)
    answer = {"status": "optimal" if result.status == 0 else "feasible_time_limited", "target": target,
            "performance_cost_mode": performance_mode,
            "performance_cost_s": performance_cost, "migration_cost_s": migration_cost,
            "admission_cost_s": admission_cost,
            "objective_s": total_cost,
            "mip_gap": 0 if result.status == 0 else relative_gap,
            "dual_bound_s": total_bound,
            "moves": [requests[request_index]["id"] for _, request_index in moves],
            "memory_cost_mode": "page_owners_conservative_destination" if source_pages is not None else "request_private",
            "released_source_pages": sorted(released_ids),
            "imbalance_counts": differences, "final_counts": counts, "final_gib": memory,
            "peak_gib": [max(peak, final) for peak, final in zip(peaks, memory)]}
    if activation_position is not None:
        answer["migration_activation_cost_s"] = activation_cost
    if snapshot.get("work_segments") or "placement_risk_cost_s" in snapshot:
        answer["work_cost"] = work
    return answer


def replay_candidate(snapshot, candidate):
    validate_snapshot(snapshot)
    incoming = snapshot.get("new_request_count", 1)
    target, identifiers = candidate["target"], candidate["moves"]
    requests = {request["id"]: request for request in snapshot["movable"]}
    if ((incoming and target not in (0, 1)) or (not incoming and target is not None)
            or len(identifiers) != len(set(identifiers))
            or any(identifier not in requests for identifier in identifiers)):
        raise ValueError("Fixed candidate has unknown or duplicate requests/target")
    memory, peaks = list(snapshot["initial_gib"]), list(snapshot["initial_gib"])
    counts = [list(values) for values in snapshot["counts"]]
    if incoming:
        counts[0][target] += incoming
    moved, released = set(), set()
    pages = snapshot.get("source_pages")
    reused_pools = {requests[identifier].get("destination_pool") for identifier in identifiers}
    for identifier in identifiers:
        request = requests[identifier]
        source, destination = request["source"], 1 - request["source"]
        memory[destination] += request["memory_gib"]
        peaks[destination] = max(peaks[destination], memory[destination])
        if memory[destination] > snapshot["capacity_gib"][destination] + 1e-6:
            return {"status": "revalidation_copy_peak_rejected", "target": target}
        moved.add(identifier)
        if pages is None:
            memory[source] -= request["memory_gib"]
        else:
            for page in pages:
                if (page["id"] not in released and page["owners"] and not page.get("pinned", False)
                        and (page.get("pool") is None or page["pool"] not in reused_pools)
                        and set(page["owners"]).issubset(moved)):
                    memory[page["rank"]] -= page["memory_gib"]
                    released.add(page["id"])
        counts[request["model"]][source] -= 1
        counts[request["model"]][destination] += 1
    if incoming:
        memory[target] += snapshot["new_gib"]
    if any(used > capacity + 1e-6 for used, capacity in zip(memory, snapshot["capacity_gib"])):
        return {"status": "revalidation_admission_rejected", "target": target}
    if any(used > capacity + 1e-6 for used, capacity in zip(memory, snapshot.get("final_capacity_gib", snapshot["capacity_gib"]))):
        return {"status": "revalidation_headroom_rejected", "target": target}
    if snapshot.get("joint_performance_costs_s") is not None:
        performance = snapshot["joint_performance_costs_s"][counts[0][0]][counts[1][0]]
    elif snapshot.get("performance_cost_tables_s") is not None:
        performance = sum(table[counts[model][0]] for model, table in enumerate(snapshot["performance_cost_tables_s"]))
    else:
        performance = sum(cost * abs(values[0] - values[1])
                          for cost, values in zip(snapshot["imbalance_cost_s"], counts))
    migration = sum(requests[identifier]["move_cost_s"] for identifier in identifiers)
    work = added_work_cost(snapshot, identifiers)
    performance += work["performance_cost_s"]
    activation = snapshot.get("migration_activation_cost_s", 0) if identifiers else 0
    if migration + activation > snapshot.get("migration_budget_s", math.inf) + 1e-6:
        return {"status": "revalidation_migration_budget_rejected", "target": target}
    admission = snapshot.get("admission_cost_s", [0, 0])[target] if incoming else 0
    answer = {"status": "revalidated_feasible_not_current_optimum", "target": target, "moves": list(identifiers),
            "performance_cost_s": performance, "migration_cost_s": migration, "admission_cost_s": admission,
            "objective_s": performance + migration + activation + admission + work["placement_risk_cost_s"],
            "final_counts": counts, "final_gib": memory,
            "peak_gib": [max(peak, final) for peak, final in zip(peaks, memory)],
            "released_source_pages": sorted(released), "source_candidate_status": candidate.get("status"),
            "scope": "fixed_original_ILP_candidate_replayed_no_new_ILP_or_current_global_optimality"}
    if "migration_activation_cost_s" in snapshot or "migration_budget_s" in snapshot:
        answer["migration_activation_cost_s"] = activation
    if snapshot.get("work_segments") or "placement_risk_cost_s" in snapshot:
        answer["work_cost"] = work
    return answer


def enumerate_targets(snapshot, **options):
    targets = (0, 1) if snapshot.get("new_request_count", 1) else (None,)
    candidates = [solve_candidate(snapshot, target, **options) for target in targets]
    feasible = [candidate for candidate in candidates if "objective_s" in candidate]
    winner = min(feasible, key=lambda candidate: (candidate["objective_s"], len(candidate["moves"]),
                                                candidate["target"])) if feasible else None
    proven = winner is not None and all(candidate["status"] in ("optimal", "infeasible") for candidate in candidates)
    return {"status": "optimal" if proven else "feasible_not_proven" if winner else "no_feasible_plan",
            "proven_infeasible": winner is None and all(candidate["status"] == "infeasible" for candidate in candidates),
            "scope": "CPU two-rank planner; conservative destination pages and source release, no exact packing or incoming-pool release credit",
            "candidates": candidates, "winner": winner}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--time-limit-s", type=float, default=10)
    args = parser.parse_args()
    report = enumerate_targets(json.loads(args.snapshot.read_text()), max_moves=args.max_moves,
                               time_limit_s=args.time_limit_s)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
