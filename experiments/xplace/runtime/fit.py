import math
import statistics

from cost_model import capture_bucket


def case_features(case):
    """(bucket of max per-rank count, max per-rank context at mid-window) for a measured grid case."""
    middle = case["output_tokens"] / 2
    contexts = case.get("contexts")
    if contexts is not None:
        maximum = max(sum(row) + len(row) * middle for row in contexts)
    else:
        maximum = case["max_context"] + case["max_count"] * middle
    return capture_bucket(case["max_count"]), maximum


def fit_step_model(grid, interference=None, prior=None):
    """Least squares: step = base + graph[bucket] + attention * max_context, graph[1] = 0, all terms >= 0."""
    import numpy as np

    rows = [case for case in grid if case.get("status") == "measured" and case.get("step_s")]
    if len(rows) < 6:
        raise ValueError("Too few measured calibration cases to fit a step model")
    buckets = sorted({case_features(case)[0] for case in rows} - {1})
    design, target = [], []
    for case in rows:
        bucket, context = case_features(case)
        design.append([1.0] + [1.0 if bucket == size else 0.0 for size in buckets] + [context])
        target.append(case["step_s"])
    design, target = np.array(design), np.array(target)
    scale = np.maximum(np.abs(design).max(axis=0), 1e-12)
    try:
        from scipy.optimize import nnls

        coefficients, _ = nnls(design / scale, target)
    except ImportError:
        coefficients = np.clip(np.linalg.lstsq(design / scale, target, rcond=None)[0], 0, None)
    coefficients = coefficients / scale
    predicted = design @ coefficients
    residuals = target - predicted
    graph = {1: 0.0}
    graph.update({size: float(coefficients[1 + index]) for index, size in enumerate(buckets)})
    previous = 0.0
    for size in sorted(graph):
        graph[size] = max(graph[size], previous)
        previous = graph[size]
    tenant = {"base_s": float(coefficients[0]), "graph_s": graph, "attention_s_per_token": float(coefficients[-1])}
    if len(graph) >= 2:
        largest = sorted(graph)[-2:]
        tenant["graph_slope_s"] = max(0.0, (graph[largest[1]] - graph[largest[0]]) / (largest[1] - largest[0]))
    params = {"source": "dp4_job_calibration_fit", "tenants": {"0": dict(tenant), "1": dict(tenant)},
              "fit": {"cases": len(rows), "rmse_s": float(math.sqrt(float(np.mean(residuals ** 2)))),
                      "max_abs_residual_s": float(np.max(np.abs(residuals))),
                      "relative_rmse": float(math.sqrt(float(np.mean((residuals / target) ** 2))))}}
    params["interference"] = (prior or {}).get("interference", {"0": {"multiplier": 1.0, "additive_s": 0.15},
                                                                "1": {"multiplier": 1.0, "additive_s": 0.15}})
    params["code_prefill"] = dict((prior or {}).get("code_prefill", {}))
    if interference:
        with_conv = [item for item in interference if item["conv_per_rank"] and item.get("conv_before_s")]
        solo = next((item for item in interference if not item["conv_per_rank"]), None)
        if with_conv:
            during = [statistics.median([value for value in (item.get("conv_during_prefill_s"),
                                                              item.get("conv_during_decode_s")) if value])
                      - item["conv_before_s"] for item in with_conv
                      if item.get("conv_during_prefill_s") or item.get("conv_during_decode_s")]
            if during:
                params["interference"]["1"] = {"multiplier": 1.0, "additive_s": max(0.0, statistics.median(during))}
        if solo is not None and with_conv:
            code_extra = [item["code_tpot_s"] - solo["code_tpot_s"] for item in with_conv]
            params["interference"]["0"] = {"multiplier": 1.0, "additive_s": max(0.0, statistics.median(code_extra))}
            params["code_prefill"].update(solo_s_per_100k=solo["code_prefill_s"],
                                          with_other_s_per_100k=statistics.median(
                                              item["code_prefill_s"] for item in with_conv))
    return params
