import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from analyze_workaware_cost import analyze
from calibrated_cost import SPARSE_CAPTURE_SIZES, graph_padding
from workaware_profile_cost import fit_rows, predict_contrast


def maximum_count(row):
    population, rank_zero = row["population"], row["rank_zero"]
    if (type(population) is not int or population < 2 or type(rank_zero) is not int
            or not 0 <= rank_zero <= population):
        raise ValueError("Invalid measured attention population or rank count")
    count = max(rank_zero, population - rank_zero)
    if count <= (population + 1) // 2:
        raise ValueError("Require an unbalanced measured layout contrast")
    if (not math.isfinite(row["peak_context_delta_tokens"]) or row["peak_context_delta_tokens"] <= 0
            or not math.isfinite(row["measured_cadence_delta_s"])):
        raise ValueError("Invalid measured context or cadence contrast")
    return count


def fit_partition_rows(rows):
    import numpy as np
    from scipy.optimize import nnls

    if not rows or len({(row["model"], row["population"]) for row in rows}) != 1:
        raise ValueError("Fit one nonempty tenant/population group")
    counts = [maximum_count(row) for row in rows]
    population = rows[0]["population"]
    baseline = (population + 1) // 2
    levels = sorted(set(counts))
    design = np.array([[int(count == level) for level in levels] + [row["peak_context_delta_tokens"]]
                       for count, row in zip(counts, rows)], dtype=float)
    scale = np.maximum(np.max(np.abs(design), axis=0), 1)
    if np.linalg.matrix_rank(design / scale) != len(levels) + 1:
        raise ValueError("Actual-count and context effects are not identifiable")
    coefficients, _ = nnls(design / scale, np.array([row["measured_cadence_delta_s"] for row in rows]))
    coefficients = coefficients / scale
    missing = sorted(set(range(baseline, population + 1)) - {baseline, *levels})
    return {"baseline_max_count": baseline,
            "partition_penalties_s": {str(baseline): 0.,
                **{str(level): float(value) for level, value in zip(levels, coefficients[:-1])}},
            "attention_token_s": float(coefficients[-1]), "missing_max_counts": missing,
            "complete_count_domain": not missing}


def predict_partition_contrast(row, profile):
    count = str(maximum_count(row))
    if count not in profile["partition_penalties_s"]:
        raise ValueError("Held-out layout requires an unmeasured attention count")
    return profile["partition_penalties_s"][count] + profile["attention_token_s"] * row["peak_context_delta_tokens"]


def residual_statistics(values):
    return {"bound_ms": 1000 * max(abs(value) for value in values),
            "rmse_ms": 1000 * math.sqrt(sum(value * value for value in values) / len(values))}


def compare_cost_models(rows):
    if not rows:
        raise ValueError("Require raw-verified layout contrasts")
    grouped = defaultdict(list)
    identities = set()
    for row in rows:
        maximum_count(row)
        if type(row["model"]) is not int or row["model"] not in (0, 1):
            raise ValueError("Unknown measured tenant")
        if type(row["repeat"]) is not int or row["repeat"] < 0:
            raise ValueError("Invalid held-out repeat")
        identity = tuple(row[name] for name in ("model", "population", "prompt_tokens", "repeat", "rank_zero"))
        if identity in identities:
            raise ValueError("Duplicate measured layout contrast")
        identities.add(identity)
        grouped[(row["model"], row["population"])].append(row)
    profiles, all_old, all_new = {}, [], []
    for (model, population), group in sorted(grouped.items()):
        repeats = sorted({row["repeat"] for row in group})
        if len(repeats) < 2:
            raise ValueError("Require independent held-out repeats")
        baseline = graph_padding([population // 2, population - population // 2], SPARSE_CAPTURE_SIZES)
        old_residuals, new_residuals = [], []
        for repeat in repeats:
            train = [row for row in group if row["repeat"] != repeat]
            old, candidate = fit_rows(train, baseline), fit_partition_rows(train)
            for row in group:
                if row["repeat"] == repeat:
                    old_residuals.append(row["measured_cadence_delta_s"] - predict_contrast(row, old, baseline))
                    new_residuals.append(row["measured_cadence_delta_s"] - predict_partition_contrast(row, candidate))
        all_old.extend(old_residuals)
        all_new.extend(new_residuals)
        profiles[f"{model}:{population}"] = {"samples": len(group), "heldout_repeats": repeats,
            "old": residual_statistics(old_residuals), "candidate": residual_statistics(new_residuals),
            "candidate_fit": fit_partition_rows(group)}
    return {"status": "offline_heldout_diagnostic_not_installed_not_GPU_benefit",
            "scope": "every_raw_verified_layout_and_repeat_no_trace_metric_fitting_or_model_selection",
            "contrasts": len(rows), "old": residual_statistics(all_old),
            "candidate": residual_statistics(all_new), "profiles": profiles}


def analyze_partition_cost(root):
    verified = analyze(root)
    result = compare_cost_models(verified["contrasts"])
    result["raw_trials_sha256"] = verified["raw_trials_sha256"]
    result["source_verification"] = verified["status"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing cost-model diagnostic")
    result = analyze_partition_cost(args.root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
