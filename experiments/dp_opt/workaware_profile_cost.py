import math
from collections import defaultdict

from calibrated_cost import graph_padding, SPARSE_CAPTURE_SIZES
from workaware_cost import remaining_segments


def fit_rows(rows, baseline_graph):
    import numpy as np
    from scipy.optimize import nnls

    graphs = sorted({int(baseline_graph + row["graph_delta"]) for row in rows if row["graph_delta"] > 0})
    design = np.array([[int(baseline_graph + row["graph_delta"] == graph) for graph in graphs]
                       + [row["peak_context_delta_tokens"]] for row in rows], dtype=float)
    if not rows or any(row["peak_context_delta_tokens"] <= 0 for row in rows):
        raise ValueError("Profile requires measured positive context contrasts")
    scale = np.maximum(np.max(np.abs(design), axis=0), 1)
    if np.linalg.matrix_rank(design / scale) != len(graphs) + 1:
        raise ValueError("Population-specific graph and attention effects are not identifiable")
    response = np.array([row["measured_cadence_delta_s"] for row in rows])
    coefficients, _ = nnls(design / scale, response)
    coefficients = coefficients / scale
    return {"graph_penalties_s": {str(baseline_graph): 0.,
            **{str(graph): float(value) for graph, value in zip(graphs, coefficients[:-1])}},
            "attention_token_s": float(coefficients[-1])}


def predict_contrast(row, profile, baseline_graph):
    graph = str(int(baseline_graph + row["graph_delta"]))
    if graph not in profile["graph_penalties_s"]:
        raise ValueError("Held-out contrast requires an unmeasured graph transition")
    return (profile["graph_penalties_s"][graph]
            + profile["attention_token_s"] * row["peak_context_delta_tokens"])


def fit_profiles(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["population"])].append(row)
    profiles = {}
    for (model, population), group in sorted(grouped.items()):
        baseline = graph_padding([population // 2, population - population // 2], SPARSE_CAPTURE_SIZES)
        repeats = sorted({row["repeat"] for row in group})
        if len(repeats) < 2:
            raise ValueError("A population profile requires held-out repeat validation")
        residuals = []
        for repeat in repeats:
            trained = fit_rows([row for row in group if row["repeat"] != repeat], baseline)
            residuals.extend(row["measured_cadence_delta_s"] - predict_contrast(row, trained, baseline)
                             for row in group if row["repeat"] == repeat)
        profile = fit_rows(group, baseline)
        profile.update({"model": model, "population": population, "baseline_graph": baseline,
                        "contrasts": len(group), "heldout_repeats": repeats,
                        "contrast_residual_bound_s": max(abs(value) for value in residuals),
                        "contrast_rmse_s": math.sqrt(sum(value * value for value in residuals) / len(residuals)),
                        "uncertainty_scope": "worst_heldout_repeat_layout_contrast_not_statistical_confidence"})
        profiles[f"{model}:{population}"] = profile
    return {"status": "diagnostic_profiles_require_full_raw_verification_before_runtime",
            "scope": "per_tenant_population_paired_layout_loss_default_sparse_not_kernel_attention",
            "profiles": profiles}


def profile_segments(observed, fit, profiles, max_interval_steps=128):
    original = remaining_segments(observed, fit, max_interval_steps)
    result = []
    for segment in original:
        population = segment["population"]
        if population == 1:
            result.append({**segment, "uncertainty_request_s": 0,
                           "uncertainty_scope": "rank_symmetric_single_request_cost_is_placement_invariant"})
            continue
        model = segment["model"]
        key = f"{model}:{population}"
        if key not in profiles["profiles"]:
            break
        profile = profiles["profiles"][key]
        maximum_context = max(segment["attention_weights"].values())
        peak = max(sum(weight for identifier, weight in segment["attention_weights"].items()
                       if segment["request_sources"][identifier] == rank) for rank in (0, 1))
        maximum_peak = profile.get("max_attention_tokens", math.inf)
        if maximum_context > profile.get("max_context_tokens", math.inf) or peak > maximum_peak:
            break
        graphs = [graph_padding([count, population - count], SPARSE_CAPTURE_SIZES)
                  for count in range(population + 1)]
        if any(str(graph) not in profile["graph_penalties_s"] for graph in graphs):
            raise ValueError("Remaining population requires an uncalibrated graph transition")
        weight = segment["steps"] * population
        result.append({**segment,
            "graph_costs_s": [weight * profile["graph_penalties_s"][str(graph)] for graph in graphs],
            "attention_difference_cost_s": weight * profile["attention_token_s"] / 2,
            "uncertainty_request_s": 2 * weight * profile["contrast_residual_bound_s"],
            "attention_difference_limit_tokens": min(segment["attention_difference_limit_tokens"],
                                                       2 * maximum_peak - sum(segment["attention_weights"].values())),
            "uncertainty_scope": profile["uncertainty_scope"], "population_profile": key})
    if not result:
        raise ValueError("No measured population-specific remaining-work forecast")
    return result


def changed_segment_uncertainty(segments, moved_ids):
    moved = set(moved_ids)
    known = {identifier for segment in segments for identifier in segment["request_sources"]}
    if not moved <= known:
        raise ValueError("Moved request is absent from forecast")
    uncertainty = 0
    for segment in segments:
        before_counts, after_counts = [0, 0], [0, 0]
        before_work, after_work = [0., 0.], [0., 0.]
        for identifier, rank in segment["request_sources"].items():
            target = 1 - rank if identifier in moved else rank
            before_counts[rank] += 1
            after_counts[target] += 1
            before_work[rank] += segment["attention_weights"][identifier]
            after_work[target] += segment["attention_weights"][identifier]
        same_graph = graph_padding(before_counts, SPARSE_CAPTURE_SIZES) == graph_padding(after_counts, SPARSE_CAPTURE_SIZES)
        same_work = math.isclose(abs(before_work[0] - before_work[1]), abs(after_work[0] - after_work[1]),
                                 rel_tol=1e-12, abs_tol=1e-6)
        if not (same_graph and same_work):
            uncertainty += segment["uncertainty_request_s"]
    return uncertainty
