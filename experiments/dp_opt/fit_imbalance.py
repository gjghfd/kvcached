import argparse
import json
import math
import statistics
from pathlib import Path


def fit_curve(report):
    if report["status"] != "passed" or report["protocol"].get("screening"):
        raise ValueError("Calibration requires verified, nonscreening trials")
    rows = report["summary"]
    measured = [row for row in rows if row["imbalance"] > 0]
    if not measured or len({row["imbalance"] for row in measured}) < 2:
        raise ValueError("Intermediate imbalance measurements are required")
    observations = [(row["imbalance"], ratio - 1) for row in measured for ratio in row["paired_ratios"]]
    denominator = sum(imbalance ** 2 for imbalance, _ in observations)
    alpha = sum(imbalance * excess for imbalance, excess in observations) / denominator
    extremes = [row for row in measured if row["imbalance"] == 1]
    if len(extremes) != 2:
        raise ValueError("Both extreme hot-rank orientations are required")
    endpoint_alpha = statistics.mean(row["paired_slowdown"] for row in extremes) - 1
    orientation_alphas = {}
    for rank in (0, 1):
        rank_points = [(row["imbalance"], ratio - 1) for row in measured
                       if row["layout"][rank] > row["layout"][1 - rank] for ratio in row["paired_ratios"]]
        orientation_alphas[str(rank)] = (sum(imbalance * excess for imbalance, excess in rank_points)
                                         / sum(imbalance ** 2 for imbalance, _ in rank_points))
    errors = [excess - alpha * imbalance for imbalance, excess in observations]
    mean_excess = statistics.mean(excess for _, excess in observations)
    total_variation = sum((excess - mean_excess) ** 2 for _, excess in observations)
    repeat_alphas = []
    for repeat in sorted({trial["repeat"] for trial in report["trials"]}):
        cohort = [trial for trial in report["trials"] if trial["repeat"] == repeat]
        baselines = [trial for trial in cohort if trial["role"].startswith("baseline_")]
        if len(baselines) != 2:
            raise ValueError("Every repeat needs two balanced endpoints")
        reference = statistics.mean(trial["stable_window"]["critical_cadence_median_s"] for trial in baselines)
        repeat_points = [(trial["imbalance"], trial["stable_window"]["critical_cadence_median_s"] / reference - 1)
                         for trial in cohort if trial["role"] == "measured"]
        repeat_alphas.append(sum(imbalance * excess for imbalance, excess in repeat_points)
                             / sum(imbalance ** 2 for imbalance, _ in repeat_points))
    if len(repeat_alphas) != 3:
        raise ValueError("Three independent paired repeats are required")
    margin = 4.303 * statistics.stdev(repeat_alphas) / math.sqrt(3)
    points = []
    for row in measured:
        observed = row["paired_slowdown"]
        predicted = 1 + alpha * row["imbalance"]
        padding = sorted({trial["stable_window"]["ranks"][str(rank)]["padded_tokens"][0]
                          for trial in report["trials"] if trial["layout"] == row["layout"] for rank in (0, 1)})
        points.append({"layout": row["layout"], "imbalance": row["imbalance"], "graph_sizes": padding,
                       "paired_slowdown": observed, "slowdown_t95_ci": row["paired_slowdown_t95_ci"],
                       "loss_from_mean_ratio": 1 - 1 / observed, "predicted_slowdown": predicted,
                       "slowdown_residual": observed - predicted,
                       "endpoint_surrogate_residual": observed - (1 + endpoint_alpha * row["imbalance"])})
    maximum = max(abs(point["slowdown_residual"]) for point in points)
    baseline = next(row for row in rows if row["imbalance"] == 0)
    balanced_padding = sorted({trial["stable_window"]["ranks"][str(rank)]["padded_tokens"][0]
                               for trial in report["trials"] if trial["imbalance"] == 0 for rank in (0, 1)})
    if len(balanced_padding) != 1 or any(len(point["graph_sizes"]) != 1 for point in points):
        raise ValueError("Graph shape must be stable within each measured layout")
    shape_points = [(point["imbalance"], point["graph_sizes"][0] / balanced_padding[0] - 1,
                     point["paired_slowdown"] - 1) for point in points]
    square_imbalance = sum(imbalance ** 2 for imbalance, shape, excess in shape_points)
    square_shape = sum(shape ** 2 for imbalance, shape, excess in shape_points)
    cross = sum(imbalance * shape for imbalance, shape, excess in shape_points)
    determinant = square_imbalance * square_shape - cross ** 2
    shape_fit = {"status": "not_identifiable_collinear_shape_and_imbalance"}
    if determinant > 1e-12:
        imbalance_response = sum(imbalance * excess for imbalance, shape, excess in shape_points)
        shape_response = sum(shape * excess for imbalance, shape, excess in shape_points)
        imbalance_slope = (imbalance_response * square_shape - shape_response * cross) / determinant
        shape_slope = (shape_response * square_imbalance - imbalance_response * cross) / determinant
        shape_residuals = [excess - imbalance_slope * imbalance - shape_slope * shape
                           for imbalance, shape, excess in shape_points]
        shape_fit = {"status": "descriptive_posthoc_fit", "formula": "R=1+a*I+b*(p/p0-1)",
                     "a": imbalance_slope, "b": shape_slope,
                     "max_absolute_mean_slowdown_residual": max(abs(error) for error in shape_residuals),
                     "rmse_mean_slowdown": math.sqrt(statistics.mean(error ** 2 for error in shape_residuals)),
                     "causal_kernel_attribution": False}
    return {"status": "verified_curve_fit", "requests": report["protocol"]["requests"],
            "capture_sizes": report["protocol"].get("capture_sizes"),
            "baseline_cadence_s": baseline["cadence_s"], "alpha": alpha,
            "alpha_by_hot_rank": orientation_alphas,
            "repeat_alphas": repeat_alphas, "alpha_t95_ci": [alpha - margin, alpha + margin],
            "endpoint_alpha": endpoint_alpha, "rmse_slowdown": math.sqrt(statistics.mean(error ** 2 for error in errors)),
            "descriptive_r_squared": 1 - sum(error ** 2 for error in errors) / total_variation if total_variation else None,
            "max_absolute_mean_slowdown_residual": maximum,
            "passes_predeclared_0_05_residual": maximum <= 0.05, "points": points,
            "capture_shape_posthoc": shape_fit,
            "scope": "Single-model calibration; not proof of linearity, causal attention share or dual-model prediction"}


def main():
    from verify_motivation import verify_case

    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verification = verify_case(args.results.parent)
    result = fit_curve(json.loads(args.results.read_text()))
    result["raw_verification"] = verification
    result["source_results"] = str(args.results)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
