import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from analyze_workaware_cost import paired_contrasts
from calibrated_cost import SPARSE_CAPTURE_SIZES, graph_padding
from verify_workaware_profile import verify_profile
from workaware_profile_cost import fit_rows, predict_contrast


def peak_count(row):
    return max(row["rank_zero"], row["population"] - row["rank_zero"])


def count_features(row, variant):
    baseline = (row["population"] + 1) // 2
    if variant == "linear_count":
        counts = [peak_count(row) - baseline]
    elif variant == "monotone_count":
        counts = [int(peak_count(row) >= count)
                  for count in range(baseline + 1, row["population"] + 1)]
    else:
        raise ValueError("Unknown diagnostic variant")
    return counts + [row["peak_context_delta_tokens"]]


def fit_count(rows, variant):
    import numpy as np
    from scipy.optimize import nnls

    if not rows or len({(row["model"], row["population"]) for row in rows}) != 1:
        raise ValueError("Require one nonempty tenant/population group")
    if any(row["graph_delta"] != 0 or row["peak_context_delta_tokens"] <= 0 for row in rows):
        raise ValueError("Diagnostic is restricted to verified same-graph positive contrasts")
    design = np.asarray([count_features(row, variant) for row in rows], dtype=float)
    scale = np.maximum(np.max(np.abs(design), axis=0), 1)
    normalized = design / scale
    if np.linalg.matrix_rank(normalized) != design.shape[1]:
        raise ValueError("Count diagnostic design is not identifiable")
    response = np.asarray([row["measured_cadence_delta_s"] for row in rows])
    coefficients, _ = nnls(normalized, response)
    return {"variant": variant, "coefficients": (coefficients / scale).tolist(),
            "normalized_condition_number": float(np.linalg.cond(normalized))}


def predict_count(row, fitted):
    return sum(feature * coefficient for feature, coefficient in
               zip(count_features(row, fitted["variant"]), fitted["coefficients"]))


def summarize(errors):
    if not errors or any(not math.isfinite(error) for error in errors):
        raise ValueError("Require finite held-out prediction errors")
    return {"samples": len(errors), "bound_ms": 1000 * max(abs(error) for error in errors),
            "rmse_ms": 1000 * math.sqrt(sum(error * error for error in errors) / len(errors))}


def diagnose(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["population"])].append(row)
    if not groups:
        raise ValueError("Require raw-verified same-graph contrast groups")
    report = {"status": "offline_all_domain_diagnostic_not_installed_not_GPU_benefit",
              "scope": "same_sparse_graph_population_contrasts_no_trace_metric_selection",
              "profiles": {}}
    for (model, population), group in sorted(groups.items()):
        baseline = graph_padding([population // 2, population - population // 2], SPARSE_CAPTURE_SIZES)
        fitted = {variant: fit_count(group, variant) for variant in ("linear_count", "monotone_count")}
        profile = {"samples": len(group), "candidate_fits": fitted, "heldout": {}}
        for field in ("repeat", "prompt_tokens", "rank_zero"):
            values = sorted({row[field] for row in group})
            if len(values) < 2:
                raise ValueError("Require held-out repeat, context and layout domains")
            errors = {variant: [] for variant in ("graph_only", "linear_count", "monotone_count")}
            fold_diagnostics = []
            for value in values:
                training = [row for row in group if row[field] != value]
                heldout = [row for row in group if row[field] == value]
                old = fit_rows(training, baseline)
                candidates = {variant: fit_count(training, variant) for variant in fitted}
                errors["graph_only"].extend(row["measured_cadence_delta_s"] - predict_contrast(row, old, baseline)
                                             for row in heldout)
                for variant, candidate in candidates.items():
                    errors[variant].extend(row["measured_cadence_delta_s"] - predict_count(row, candidate)
                                           for row in heldout)
                fold_diagnostics.append({"heldout_value": value, "training_samples": len(training),
                                         "heldout_samples": len(heldout), "candidate_fits": candidates})
            profile["heldout"][field] = {"values": values, "errors": {
                variant: summarize(residuals) for variant, residuals in errors.items()}, "folds": fold_diagnostics}
        profile["all_domain_bound_ms"] = {variant: max(
            domain["errors"][variant]["bound_ms"] for domain in profile["heldout"].values())
            for variant in ("graph_only", "linear_count", "monotone_count")}
        report["profiles"][f"{model}:{population}"] = profile
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    fitted = verify_profile(args.root)
    source = args.root / "workaware-trials.json"
    trials = json.loads(source.read_text())
    rebuilt = [{"case": trial["case"], "sample": trial["sample"]} for trial in trials]
    contrasts = paired_contrasts(rebuilt, fitted)
    if contrasts["unmatched_groups"]:
        raise ValueError("Require complete balanced paired calibration groups")
    report = diagnose(contrasts["contrasts"])
    report["source_trials_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    report["source_verification"] = fitted["status"]
    report["runtime_algorithm_changed"] = False
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
