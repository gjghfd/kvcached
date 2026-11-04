import itertools
import math

from calibrated_cost import SPARSE_CAPTURE_SIZES, graph_padding
from ilp_reference import replay_candidate, validate_snapshot
from workaware_ilp import allocate_work_variables, install_work_constraints


def segment_features(segment, moved_ids=()):
    moved = set(moved_ids)
    counts, work = [0, 0], [0., 0.]
    for identifier, source in segment["request_sources"].items():
        destination = 1 - source if identifier in moved else source
        counts[destination] += 1
        work[destination] += segment["attention_weights"][identifier]
    return (graph_padding(counts, SPARSE_CAPTURE_SIZES),
            segment["graph_costs_s"][counts[0]], abs(work[0] - work[1]))


def same_features(before, after):
    return (before[0] == after[0]
            and math.isclose(before[1], after[1], rel_tol=1e-12, abs_tol=1e-10)
            and math.isclose(before[2], after[2], rel_tol=1e-12, abs_tol=1e-6))


def changed_uncertainty(segments, moved_ids):
    moved = set(moved_ids)
    known = {identifier for segment in segments for identifier in segment["request_sources"]}
    if not moved <= known:
        raise ValueError("Moved request is absent from remaining-work forecast")
    return sum(segment["uncertainty_request_s"] for segment in segments
               if not same_features(segment_features(segment), segment_features(segment, moved)))


def allocate_uncertainty_variables(snapshot, dimensions):
    variables = []
    for segment in snapshot.get("work_segments", []):
        if segment["uncertainty_request_s"] > 0:
            variables.append({"changed": dimensions, "equivalent": [dimensions + 1, dimensions + 2]})
            dimensions += 3
        else:
            variables.append(None)
    return variables, dimensions


def install_uncertainty_constraints(snapshot, work_variables, uncertainty_variables, requests,
                                    stages, dimensions, objective, integrality, constraint):
    import numpy as np

    segments = snapshot.get("work_segments", [])
    if len(segments) != len(work_variables) or len(segments) != len(uncertainty_variables):
        raise ValueError("Uncertainty variables must cover each remaining-work segment")
    for segment, work_positions, positions in zip(segments, work_variables, uncertainty_variables):
        if positions is None:
            if segment["uncertainty_request_s"] != 0:
                raise ValueError("Nonzero uncertainty cannot be omitted from ILP")
            continue
        changed, equivalent = positions["changed"], positions["equivalent"]
        integrality[[changed, *equivalent]] = 1
        objective[changed] = segment["uncertainty_request_s"]
        row = np.zeros(dimensions)
        row[[changed, *equivalent]] = 1
        constraint(row, minimum=1, maximum=1)
        before = segment_features(segment)
        allowed_counts = [position for count, position in enumerate(work_positions["counts"])
                          if graph_padding([count, segment["population"] - count], SPARSE_CAPTURE_SIZES) == before[0]
                          and math.isclose(segment["graph_costs_s"][count], before[1], rel_tol=1e-12, abs_tol=1e-10)]
        total_work = sum(segment["attention_weights"].values())
        initial_difference = sum((1 if source == 0 else -1) * segment["attention_weights"][identifier]
                                 for identifier, source in segment["request_sources"].items()) / total_work
        tolerance = max(1e-6, before[2] * 1e-12) / total_work
        coefficients = np.zeros(dimensions)
        for index, request in enumerate(requests):
            if request["id"] in segment["request_sources"]:
                source = segment["request_sources"][request["id"]]
                weight = segment["attention_weights"][request["id"]] / total_work
                coefficients[index * stages:(index + 1) * stages] = (-2 if source == 0 else 2) * weight
        for sign, position in zip((1, -1), equivalent):
            row = np.zeros(dimensions)
            row[position] = 1
            row[allowed_counts] = -1
            constraint(row, maximum=0)
            target = sign * initial_difference
            row = coefficients.copy()
            row[position] = 2
            constraint(row, maximum=2 + target - initial_difference + tolerance)
            row = coefficients.copy()
            row[position] = -2
            constraint(row, minimum=-2 + target - initial_difference - tolerance)


def robust_replay(snapshot, candidate):
    validate_snapshot(snapshot)
    if snapshot.get("new_request_count", 1) != 0:
        raise ValueError("Robust remaining-work optimization is release-only, not prefill admission")
    try:
        replay = replay_candidate(snapshot, candidate)
    except ValueError as error:
        return {"status": "outside_verified_work_domain", "diagnosis": str(error)}
    if "objective_s" not in replay:
        return replay
    segments = snapshot.get("work_segments", [])
    uncertainty = changed_uncertainty(segments, replay["moves"]) if segments else 0
    return {**replay, "uncertainty_request_s": uncertainty,
            "robust_objective_request_s": replay["objective_s"] + uncertainty,
            "scope": "offline_unchanged_error_bound_and_independent_serial_copy_peak_replay"}


def unchanged_dominance_bound(snapshot, max_moves):
    if max_moves == 0 or not snapshot["movable"]:
        return 0.
    gross = 0.
    for segment in snapshot.get("work_segments", []):
        before = segment_features(segment)
        current = before[1] + segment["attention_difference_cost_s"] * before[2]
        relaxed_minimum = min(segment["graph_costs_s"])
        equivalent_cost_slack = max(1e-10, max(segment["graph_costs_s"]) * 1e-12)
        equivalent_cost_slack += segment["attention_difference_cost_s"] * max(
            1e-6, sum(segment["attention_weights"].values()) * 1e-12)
        gross += max(equivalent_cost_slack, current - relaxed_minimum - segment["uncertainty_request_s"])
    return gross - snapshot.get("migration_activation_cost_s", 0) - min(
        request["move_cost_s"] for request in snapshot["movable"])


def solve_robust_candidate(snapshot, max_moves=4, time_limit_s=10):
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    validate_snapshot(snapshot)
    if (snapshot.get("new_request_count", 1) != 0 or snapshot.get("joint_performance_costs_s") is not None
            or snapshot.get("performance_cost_tables_s") is None
            or any(value != 0 for table in snapshot["performance_cost_tables_s"] for value in table)):
        raise ValueError("Candidate supports release-only remaining-work costs, not admission or static layout surrogates")
    if (type(max_moves) is not int or max_moves < 0 or isinstance(time_limit_s, bool)
            or not math.isfinite(time_limit_s) or time_limit_s <= 0):
        raise ValueError("Require a nonnegative move limit and positive solver budget")
    fallback = robust_replay(snapshot, {"target": None, "moves": [], "status": "unchanged_layout"})
    upper_gain = unchanged_dominance_bound(snapshot, max_moves)
    if "robust_objective_request_s" in fallback and upper_gain <= 0:
        return {**fallback, "status": "optimal", "robust_dual_bound_request_s": fallback["robust_objective_request_s"],
                "robust_mip_gap": 0., "candidate_source": "unchanged_dominance_certificate",
                "upper_bound_move_gain_request_s": upper_gain,
                "scope": "proved_unchanged_optimal_within_empirical_cost_model_no_ILP_execution_not_installed"}
    requests = snapshot["movable"]
    stages = min(max_moves, len(requests))
    move_dimensions = len(requests) * stages
    work_variables, dimensions = allocate_work_variables(snapshot, move_dimensions)
    uncertainty_variables, dimensions = allocate_uncertainty_variables(snapshot, dimensions)
    activation = dimensions
    dimensions += 1
    pages = snapshot.get("source_pages")
    groups, page_positions = {}, []
    for page in pages or []:
        key = (page["rank"], page.get("pool"), tuple(sorted(set(page["owners"]))), page.get("pinned", False))
        if key not in groups:
            groups[key] = list(range(dimensions, dimensions + stages + 1))
            dimensions += stages + 1
        page_positions.append(groups[key])
    objective, lower, upper = np.zeros(dimensions), np.zeros(dimensions), np.ones(dimensions)
    integrality = np.zeros(dimensions)
    integrality[:move_dimensions] = 1
    integrality[activation] = 1
    objective[activation] = snapshot.get("migration_activation_cost_s", 0)
    rows, minima, maxima = [], [], []

    def constraint(row, minimum=-np.inf, maximum=np.inf):
        rows.append(row)
        minima.append(minimum)
        maxima.append(maximum)

    install_work_constraints(snapshot, work_variables, requests, stages, dimensions,
                             objective, upper, integrality, constraint)
    install_uncertainty_constraints(snapshot, work_variables, uncertainty_variables, requests, stages,
                                    dimensions, objective, integrality, constraint)
    for index, request in enumerate(requests):
        positions = slice(index * stages, (index + 1) * stages)
        objective[positions] = request["move_cost_s"]
        row = np.zeros(dimensions)
        row[positions] = 1
        constraint(row, maximum=1)
        row[activation] = -1
        constraint(row, maximum=0)
    row = np.zeros(dimensions)
    row[activation] = 1
    row[:move_dimensions] = -1
    constraint(row, maximum=0)
    if "migration_budget_s" in snapshot:
        row = np.zeros(dimensions)
        row[:move_dimensions] = objective[:move_dimensions]
        row[activation] = objective[activation]
        constraint(row, maximum=snapshot["migration_budget_s"])
    for stage in range(stages):
        row = np.zeros(dimensions)
        row[stage:move_dimensions:stages] = 1
        constraint(row, maximum=1)
    request_indices = {request["id"]: index for index, request in enumerate(requests)}
    for (rank, pool, owners, pinned), positions in groups.items():
        integrality[positions] = 1
        upper[positions[0]] = 0
        if pinned or not owners or any(owner not in request_indices for owner in owners):
            upper[positions] = 0
            continue
        for completed in range(1, stages + 1):
            aggregate = np.zeros(dimensions)
            aggregate[positions[completed]] = 1
            for index, request in enumerate(requests):
                if pool is not None and request.get("destination_pool") == pool:
                    selected = slice(index * stages, (index + 1) * stages)
                    row = np.zeros(dimensions)
                    row[positions[completed]] = 1
                    row[selected] = 1
                    constraint(row, maximum=1)
                    aggregate[selected] += 1
            for owner in owners:
                selected = slice(request_indices[owner] * stages, request_indices[owner] * stages + completed)
                row = np.zeros(dimensions)
                row[positions[completed]] = 1
                row[selected] = -1
                constraint(row, maximum=0)
                aggregate[selected] -= 1
            constraint(aggregate, minimum=1 - len(owners))
    for rank in (0, 1):
        final = np.zeros(dimensions)
        for index, request in enumerate(requests):
            change = (0 if pages is not None else -request["memory_gib"]) if request["source"] == rank else request["memory_gib"]
            final[index * stages:(index + 1) * stages] = change
        for page, positions in zip(pages or [], page_positions):
            if page["rank"] == rank:
                final[positions[-1]] -= page["memory_gib"]
        constraint(final, maximum=snapshot.get("final_capacity_gib", snapshot["capacity_gib"])[rank]
                   - snapshot["initial_gib"][rank])
        for stage in range(stages):
            peak = np.zeros(dimensions)
            for index, request in enumerate(requests):
                offset = index * stages
                if request["source"] != rank:
                    peak[offset:offset + stage + 1] = request["memory_gib"]
                elif pages is None:
                    peak[offset:offset + stage] = -request["memory_gib"]
            for page, positions in zip(pages or [], page_positions):
                if page["rank"] == rank:
                    peak[positions[stage]] -= page["memory_gib"]
            constraint(peak, maximum=snapshot["capacity_gib"][rank] - snapshot["initial_gib"][rank])
    result = milp(objective, integrality=integrality, bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(np.array(rows), minima, maxima),
                  options={"mip_rel_gap": 0, "time_limit": time_limit_s})
    if result.status == 2:
        return {"status": "infeasible", "scope": "candidate_release_robust_ILP_not_installed"}
    if result.x is None or result.status not in (0, 1):
        return {"status": "no_verified_robust_incumbent", "solver_message": result.message}
    if any(abs(result.x[position] - round(result.x[position])) > 1e-6
           for position in np.flatnonzero(integrality)):
        return {"status": "no_verified_robust_incumbent", "solver_message": "Nonintegral robust incumbent"}
    ordered = sorted((stage, index) for index in range(len(requests)) for stage in range(stages)
                     if result.x[index * stages + stage] > .5)
    if len({stage for stage, index in ordered}) != len(ordered) or len({index for stage, index in ordered}) != len(ordered):
        return {"status": "no_verified_robust_incumbent", "solver_message": "Nonserial or repeated moves"}
    replay = robust_replay(snapshot, {"target": None, "moves": [requests[index]["id"] for stage, index in ordered],
                                      "status": "robust_ILP_candidate"})
    risk = snapshot.get("placement_risk_cost_s", 0)
    if ("robust_objective_request_s" not in replay
            or not math.isclose(replay["robust_objective_request_s"], result.fun + risk, rel_tol=1e-8, abs_tol=1e-6)):
        return {"status": "no_verified_robust_incumbent", "solver_message": "Independent cost or physical-copy replay rejected",
                "replay_status": replay["status"]}
    bound = getattr(result, "mip_dual_bound", None)
    source = "robust_ILP_incumbent"
    if ("robust_objective_request_s" in fallback
            and fallback["robust_objective_request_s"] < replay["robust_objective_request_s"] - 1e-6):
        if result.status == 0:
            return {"status": "no_verified_robust_incumbent", "solver_message": "Claimed optimum is worse than verified unchanged layout"}
        replay, source = fallback, "verified_unchanged_fallback"
    total_bound = None if bound is None else bound + risk
    gap = (None if total_bound is None else max(0, replay["robust_objective_request_s"] - total_bound)
           / max(abs(replay["robust_objective_request_s"]), 1e-10))
    return {**replay, "status": "optimal" if result.status == 0 else "feasible_time_limited",
            "robust_dual_bound_request_s": total_bound, "robust_mip_gap": gap, "candidate_source": source,
            "scope": "candidate_robust_ILP_full_serial_copy_peak_and_source_owner_validation_not_installed"}


def enumerate_robust_reference(snapshot, max_moves, max_evaluations=10000):
    validate_snapshot(snapshot)
    if (type(max_moves) is not int or max_moves < 0
            or type(max_evaluations) is not int or max_evaluations <= 0):
        raise ValueError("Require a nonnegative move limit and positive reference search budget")
    identifiers = [request["id"] for request in snapshot["movable"]]
    reference = robust_replay(snapshot, {"target": None, "moves": [], "status": "unchanged_layout"})
    winner = reference if "robust_objective_request_s" in reference else None
    evaluations, complete = 0, True
    rejected = {}
    seen_sets = set()
    for count in range(1, min(max_moves, len(identifiers)) + 1):
        for ordered in itertools.permutations(identifiers, count):
            selected = frozenset(ordered)
            if selected in seen_sets:
                continue
            if evaluations >= max_evaluations:
                complete = False
                break
            evaluations += 1
            replay = robust_replay(snapshot, {"target": None, "moves": list(ordered), "status": "reference_order"})
            if "robust_objective_request_s" not in replay:
                rejected[replay["status"]] = rejected.get(replay["status"], 0) + 1
                continue
            seen_sets.add(selected)
            if winner is None or (replay["robust_objective_request_s"], len(ordered), ordered) < (
                    winner["robust_objective_request_s"], len(winner["moves"]), tuple(winner["moves"])):
                winner = replay
        if not complete:
            break
    return {"status": "complete_reference_enumeration" if complete else "bounded_reference_not_proven",
            "winner": winner, "evaluations": evaluations, "rejected": rejected,
            "scope": "CPU_oracle_not_runtime_ILP_not_GPU_benefit"}
