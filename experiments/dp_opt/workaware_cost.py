import math

from calibrated_cost import SPARSE_CAPTURE_SIZES
from motivation import graph_padding


def validate_fit(fit):
    if (fit.get("status") != "raw_verified_workaware_cadence_fit"
            or fit.get("capture_sizes") != SPARSE_CAPTURE_SIZES
            or fit.get("scope") != "one_active_tenant_other_resident_idle"
            or fit.get("source_verification") != "raw_verified"):
        raise ValueError("Require raw-verified, sparse, resident-idle cadence calibration")
    coefficients = fit["coefficients"]
    if (any(not math.isfinite(coefficients[name]) or coefficients[name] < 0
            for name in ("base_s", "graph_batch_s", "attention_token_s"))
            or coefficients["base_s"] <= 0
            or not math.isfinite(fit["residual_bound_s"]) or fit["residual_bound_s"] < 0
            or type(fit["max_population"]) is not int or fit["max_population"] < 1
            or not math.isfinite(fit["max_context_tokens"]) or fit["max_context_tokens"] <= 0):
        raise ValueError("Invalid measured cadence coefficients or calibration domain")
    if "max_attention_tokens" in fit and (not math.isfinite(fit["max_attention_tokens"]) or fit["max_attention_tokens"] <= 0):
        raise ValueError("Invalid measured peak-attention domain")


def fit_cadence(samples, capture_sizes=SPARSE_CAPTURE_SIZES):
    import numpy as np
    from scipy.optimize import nnls

    if len(samples) < 6 or capture_sizes != SPARSE_CAPTURE_SIZES:
        raise ValueError("Require varied measured cases and default sparse graphs")
    features, cadences = [], []
    for sample in samples:
        counts, contexts = sample["counts"], sample["context_tokens"]
        if (len(counts) != 2 or len(contexts) != 2 or any(type(count) is not int or count < 0 for count in counts)
                or not sum(counts) or any(not math.isfinite(value) or value < 0 for value in contexts)
                or any(count and context <= 0 for count, context in zip(counts, contexts))
                or not math.isfinite(sample["cadence_s"]) or sample["cadence_s"] <= 0):
            raise ValueError("Invalid stable decode calibration sample")
        padded = graph_padding(counts, capture_sizes)
        if padded is None:
            raise ValueError("Calibration population exceeds captured graphs")
        features.append([1, padded, max(contexts)])
        cadences.append(sample["cadence_s"])
    design = np.array(features, dtype=float)
    scale = np.maximum(design.max(axis=0), 1)
    if np.linalg.matrix_rank(design / scale) < 3:
        raise ValueError("Graph and attention contributions are not identifiable")
    fitted, _ = nnls(design / scale, np.array(cadences))
    coefficients = fitted / scale
    residuals = np.array(cadences) - design @ coefficients
    return {"coefficients": dict(zip(("base_s", "graph_batch_s", "attention_token_s"), coefficients.tolist())),
            "residual_bound_s": float(np.max(np.abs(residuals))),
            "rmse_s": float(np.sqrt(np.mean(residuals**2))), "samples": len(samples),
            "status": "unverified_fit_requires_raw_calibration_verification",
            "capture_sizes": list(capture_sizes), "scope": "one_active_tenant_other_resident_idle"}


def remaining_segments(observed, fit, max_interval_steps=128):
    validate_fit(fit)
    if "population_profiles" in fit:
        from workaware_profile_cost import profile_segments

        if fit["population_profiles"].get("status") != "raw_verified_workaware_population_profiles":
            raise ValueError("Require raw-verified population profiles")
        base_fit = {name: value for name, value in fit.items() if name != "population_profiles"}
        return profile_segments(observed, base_fit, fit["population_profiles"], max_interval_steps)
    if (len(observed) != 2 or any(len(ranks) != 2 for ranks in observed)
            or type(max_interval_steps) is not int or max_interval_steps < 1):
        raise ValueError("Require two tenants, two ranks and a positive integration interval")
    active_models = [model for model, ranks in enumerate(observed) if any(rank["requests"] for rank in ranks)]
    if len(active_models) > 1:
        raise ValueError("Mixed active tenants require separately verified joint calibration")
    coefficients = fit["coefficients"]
    segments, identifiers = [], set()
    for model, ranks in enumerate(observed):
        requests = []
        for rank, observation in enumerate(ranks):
            for request in observation["requests"]:
                if request["id"] in identifiers:
                    raise ValueError("Request identifiers must be globally unique")
                identifiers.add(request["id"])
                remaining = request["max_tokens"] - request["output_tokens"]
                context = request["computed_tokens"]
                if (not request["movable"] or type(remaining) is not int or remaining <= 0
                        or type(context) is not int or context <= 0):
                    raise ValueError("Require movable decode requests with known remaining work")
                requests.append({"id": request["id"], "rank": rank, "remaining": remaining, "context": context})
        if not requests:
            continue
        if len(requests) > fit["max_population"]:
            raise ValueError("Population is outside measured calibration domain")
        initial_peak = max(sum(request["context"] for request in requests if request["rank"] == rank) for rank in (0, 1))
        if initial_peak > fit.get("max_attention_tokens", math.inf):
            raise ValueError("Peak attention work is outside measured calibration domain")
        context_horizon = math.floor(fit["max_context_tokens"] - max(request["context"] for request in requests))
        horizon = min(max(request["remaining"] for request in requests), context_horizon)
        if horizon <= 0:
            raise ValueError("No remaining forecast within measured context domain")
        boundaries = sorted({0, horizon, *(request["remaining"] for request in requests if request["remaining"] < horizon),
                             *range(max_interval_steps, horizon, max_interval_steps)})
        for left, right in zip(boundaries, boundaries[1:]):
            active = [request for request in requests if request["remaining"] > left]
            counts = [sum(request["rank"] == rank for request in active) for rank in (0, 1)]
            population, steps = len(active), right - left
            midpoint = (left + right) / 2
            weights = {request["id"]: request["context"] + midpoint for request in active}
            sources = {request["id"]: request["rank"] for request in active}
            peak = max(sum(weights[request["id"]] for request in active if request["rank"] == rank) for rank in (0, 1))
            if peak > fit.get("max_attention_tokens", math.inf):
                break
            baseline = graph_padding([population // 2, population - population // 2], SPARSE_CAPTURE_SIZES)
            graph_costs = [steps * population * coefficients["graph_batch_s"]
                           * (graph_padding([count, population - count], SPARSE_CAPTURE_SIZES) - baseline)
                           for count in range(population + 1)]
            segments.append({"model": model, "start_step": left, "end_step": right,
                             "steps": steps, "population": population, "initial_counts": counts,
                             "request_sources": sources, "attention_weights": weights,
                             "graph_costs_s": graph_costs,
                             "attention_difference_cost_s": steps * population * coefficients["attention_token_s"] / 2,
                             "attention_difference_limit_tokens": min(sum(weights.values()),
                                 2 * fit.get("max_attention_tokens", sum(weights.values())) - sum(weights.values())),
                             "uncertainty_request_s": 2 * steps * population * fit["residual_bound_s"],
                             "forecast_scope": "current_requests_only_completion_events_and_midpoint_context_integration"})
    if active_models and not segments:
        raise ValueError("No stable remaining forecast within measured peak-attention domain")
    return segments


def evaluate_segments(segments, moved_ids=()):
    moved = set(moved_ids)
    known = {identifier for segment in segments for identifier in segment["request_sources"]}
    if not moved <= known:
        raise ValueError("Moved request is absent from current forecast")
    graph, attention = 0, 0
    counts_by_segment = []
    for segment in segments:
        counts = list(segment["initial_counts"])
        difference = 0
        for identifier, source in segment["request_sources"].items():
            destination = 1 - source if identifier in moved else source
            if destination != source:
                counts[source] -= 1
                counts[destination] += 1
            difference += (1 if destination == 0 else -1) * segment["attention_weights"][identifier]
        if abs(difference) > segment.get("attention_difference_limit_tokens", math.inf) + 1e-6:
            raise ValueError("Candidate peak attention work is outside measured calibration domain")
        graph += segment["graph_costs_s"][counts[0]]
        attention += segment["attention_difference_cost_s"] * abs(difference)
        counts_by_segment.append(counts)
    return {"performance_cost_s": graph + attention, "graph_cost_s": graph,
            "attention_cost_s": attention, "counts_by_segment": counts_by_segment}


def reference_horizon_s(segments, fit):
    coefficients = fit["coefficients"]
    times = [0, 0]
    for segment in segments:
        contexts = [0, 0]
        for identifier, source in segment["request_sources"].items():
            contexts[source] += segment["attention_weights"][identifier]
        cadence = (coefficients["base_s"]
                   + coefficients["graph_batch_s"] * graph_padding(segment["initial_counts"], SPARSE_CAPTURE_SIZES)
                   + coefficients["attention_token_s"] * max(contexts))
        times[segment["model"]] += segment["steps"] * cadence
    return max(times)


def headroom_modes(snapshot, observed, arrival_profile, horizon_s, cold_cost_s):
    import itertools

    if (len(arrival_profile) != 2 or len(cold_cost_s) != 2 or not math.isfinite(horizon_s) or horizon_s < 0
            or any(not math.isfinite(value) or value < 0 for value in cold_cost_s)):
        raise ValueError("Require two past-arrival profiles and finite planning costs")
    models = []
    for model, profile in enumerate(arrival_profile):
        rate, tokens = profile["rate_rps"], profile["tokens"]
        if not math.isfinite(rate) or rate < 0 or type(tokens) is not int or tokens < 0:
            raise ValueError("Invalid past-arrival rate or KV quantile")
        if not rate or not tokens:
            continue
        demands = []
        for rank in observed[model]:
            if not rank["uniform_page_geometry"] or any(rank[field] <= 0 for field in ("block_size", "page_block_capacity", "page_bytes")):
                raise ValueError("Headroom requires measured uniform page geometry")
            blocks = math.ceil(tokens / rank["block_size"])
            demands.append(math.ceil(blocks / rank["page_block_capacity"]) * rank["page_bytes"] / 1024**3)
        probability = -math.expm1(-rate * horizon_s)
        models.append({"model": model, "tokens": tokens, "demands_gib": demands,
                       "arrival_probability": probability, "unprotected_cost_s": probability * cold_cost_s[model]})
    scenarios = []
    for choices in itertools.product((None, 0, 1), repeat=len(models)):
        capacities = list(snapshot["capacity_gib"])
        penalty, decisions = 0, []
        for model, rank in zip(models, choices):
            if rank is None:
                penalty += model["unprotected_cost_s"]
            else:
                capacities[rank] = min(capacities[rank], snapshot["capacity_gib"][rank] - model["demands_gib"][rank])
            decisions.append({**model, "protected_rank": rank})
        if all(value >= 0 for value in capacities):
            scenarios.append({"guards": decisions, "risk_cost_s": penalty,
                              "snapshot": {**snapshot, "final_capacity_gib": capacities}})
    return scenarios
