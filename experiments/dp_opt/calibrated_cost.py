import argparse
import copy
import json
import math
from pathlib import Path

from motivation import graph_padding


SPARSE_CAPTURE_SIZES = [1, 8, 16, 32, 64, 128, 256]


def layout_cost_table(fit, horizon_steps=1, request_weight=1, mode="capture_shape"):
    if fit["status"] != "verified_curve_fit" or fit.get("raw_verification", {}).get("status") != "raw_verified":
        raise ValueError("Cost construction requires a raw-verified calibration")
    if any(not math.isfinite(value) or value <= 0 for value in (horizon_steps, request_weight)):
        raise ValueError("Horizon and request weight must be finite and positive")
    population = fit["requests"]
    if not isinstance(population, int) or population < 1:
        raise ValueError("Calibration population must be a positive integer")
    captures = fit["capture_sizes"]
    baseline_padding = graph_padding([population // 2, population - population // 2], captures)
    if graph_padding([population, 0], captures) is None or baseline_padding is None:
        raise ValueError("Capture sizes do not cover the calibration population")
    if mode == "affine":
        if not fit["passes_predeclared_0_05_residual"]:
            raise ValueError("The rejected primary affine fit cannot become the default cost")
        imbalance_slope, shape_slope = fit["alpha"], 0
    elif mode == "capture_shape":
        shape_fit = fit["capture_shape_posthoc"]
        if shape_fit["status"] != "descriptive_posthoc_fit":
            raise ValueError("Capture-shape coefficients are not separately identifiable")
        imbalance_slope, shape_slope = shape_fit["a"], shape_fit["b"]
    else:
        raise ValueError("Unknown calibration cost mode")
    if (any(not math.isfinite(value) or value < 0 for value in (imbalance_slope, shape_slope))
            or not math.isfinite(fit["baseline_cadence_s"]) or fit["baseline_cadence_s"] <= 0):
        raise ValueError("This reference requires nonnegative calibrated slopes and positive cadence")
    costs, ratios = [], []
    for rank_zero in range(population + 1):
        counts = [rank_zero, population - rank_zero]
        imbalance = abs(counts[0] - counts[1]) / population
        padding = graph_padding(counts, captures)
        excess = imbalance_slope * imbalance + shape_slope * (padding / baseline_padding - 1)
        ratios.append(1 + excess)
        costs.append(request_weight * horizon_steps * population * fit["baseline_cadence_s"] * excess)
    return {"status": "constructed_surrogate_not_all_counts_measured", "requests": population,
            "mode": mode, "horizon_steps": horizon_steps, "request_weight": request_weight,
            "cost_units": "weighted_request_seconds_above_balanced", "costs_s": costs,
            "predicted_cadence_ratios": ratios, "capture_sizes": captures,
            "orientation": "symmetric_mean_of_both_hot_rank_orientations",
            "coefficient_scope": "single_model_fixed_population_context_hardware_and_capture_profile",
            "concurrent_models_calibrated": False,
            "unmeasured_counts": "model_predictions_not_direct_measurements",
            "primary_affine_test_passed": fit["passes_predeclared_0_05_residual"]}


def attach_sparse_costs(snapshot, calibrations):
    if len(calibrations) != 2 or len(snapshot["counts"]) != 2:
        raise ValueError("Sparse admission requires two model calibrations")
    if snapshot.get("joint_performance_costs_s") is not None:
        raise ValueError("Single-model calibration cannot overwrite joint costs")
    populations = [sum(snapshot["counts"][0]) + 1, sum(snapshot["counts"][1])]
    for calibration, population in zip(calibrations, populations):
        if calibration["status"] != "constructed_surrogate_not_all_counts_measured":
            raise ValueError("Expected a labelled constructed calibration surrogate")
        if calibration["capture_sizes"] != SPARSE_CAPTURE_SIZES:
            raise ValueError("Admission calibration must use the fixed sparse capture sizes")
        if calibration["requests"] != population or len(calibration["costs_s"]) != population + 1:
            raise ValueError("Calibration must match each post-admission population; no extrapolation")
        if calibration["cost_units"] != "weighted_request_seconds_above_balanced":
            raise ValueError("Performance and migration costs must share request-second units")
        if any(not math.isfinite(cost) or cost < 0 for cost in calibration["costs_s"]):
            raise ValueError("Calibration costs must be finite and nonnegative")
    result = copy.deepcopy(snapshot)
    result["performance_cost_tables_s"] = [list(calibration["costs_s"]) for calibration in calibrations]
    result["cost_calibration"] = {
        "status": "single_model_surrogates_not_concurrent_calibration", "capture_profile": "sparse",
        "concurrent_models_calibrated": False,
        "models": [{key: value for key, value in calibration.items()
                    if key not in ("costs_s", "predicted_cadence_ratios")} for calibration in calibrations]}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("affine", "capture_shape"), default="capture_shape")
    parser.add_argument("--horizon-steps", type=float, default=1)
    parser.add_argument("--request-weight", type=float, default=1)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    report = layout_cost_table(json.loads(args.fit.read_text()), args.horizon_steps, args.request_weight, args.mode)
    report["source_fit"] = str(args.fit)
    if args.snapshot is not None:
        report = attach_sparse_costs(json.loads(args.snapshot.read_text()), [report, report])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("costs_s", "predicted_cadence_ratios", "performance_cost_tables_s")}, indent=2))


if __name__ == "__main__":
    main()
