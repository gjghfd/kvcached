import math

from workaware_robust_ilp import install_uncertainty_constraints as legacy_install


def concentrated_equivalence_key(segment):
    sources = segment["request_sources"]
    weights = segment["attention_weights"]
    population = segment["population"]
    if len(sources) != population or len(set(sources.values())) != 1:
        return None
    source = next(iter(sources.values()))
    if (type(source) is not int or source not in (0, 1)
            or segment["initial_counts"] != [population if rank == source else 0 for rank in (0, 1)]):
        return None
    total = sum(weights.values())
    if (not weights or not math.isfinite(total) or total <= 0
            or any(not math.isfinite(weight) or weight <= max(1e-6, total * 1e-12)
                   for weight in weights.values())):
        return None
    if not math.isclose(segment["graph_costs_s"][0], segment["graph_costs_s"][population],
                        rel_tol=1e-12, abs_tol=1e-10):
        return None
    return (segment["model"], population, tuple(segment["initial_counts"]), tuple(sorted(sources.items())))


def allocate_grouped_uncertainty_variables(snapshot, dimensions):
    variables, groups = [], {}
    for segment in snapshot.get("work_segments", []):
        if segment["uncertainty_request_s"] == 0:
            variables.append(None)
            continue
        key = concentrated_equivalence_key(segment)
        if key is not None:
            if key not in groups:
                groups[key] = {"changed": dimensions, "mode": "concentrated_count_equivalence"}
                dimensions += 1
            variables.append(groups[key])
        else:
            variables.append({"changed": dimensions, "equivalent": [dimensions + 1, dimensions + 2]})
            dimensions += 3
    return variables, dimensions


def install_grouped_uncertainty_constraints(snapshot, work_variables, uncertainty_variables, requests,
                                           stages, dimensions, objective, integrality, constraint):
    import numpy as np

    segments = snapshot.get("work_segments", [])
    if len(segments) != len(work_variables) or len(segments) != len(uncertainty_variables):
        raise ValueError("Grouped uncertainty must cover every remaining-work segment")
    installed = {}
    for segment, work, positions in zip(segments, work_variables, uncertainty_variables):
        if positions is None:
            if segment["uncertainty_request_s"] != 0:
                raise ValueError("Cannot omit a nonzero uncertainty coefficient")
            continue
        if positions.get("mode") != "concentrated_count_equivalence":
            legacy_install({**snapshot, "work_segments": [segment]}, [work], [positions], requests,
                           stages, dimensions, objective, integrality, constraint)
            continue
        if concentrated_equivalence_key(segment) is None:
            raise ValueError("Grouped count equivalence requires its concentration proof")
        changed, counts = positions["changed"], work["counts"]
        identity = tuple(counts)
        if changed in installed and installed[changed] != identity:
            raise ValueError("Grouped uncertainty cannot mix different active request sets")
        objective[changed] += segment["uncertainty_request_s"]
        integrality[changed] = 1
        if changed not in installed:
            row = np.zeros(dimensions)
            row[changed] = 1
            row[counts[0]] = 1
            row[counts[-1]] = 1
            constraint(row, minimum=1, maximum=1)
            installed[changed] = identity
