import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from calibrated_cost import graph_padding
from workaware_cost import fit_cadence
from workaware_pilot import stable_sample


def describe(values):
    return {"samples": len(values), "mean": statistics.mean(values),
            "max_abs": max(abs(value) for value in values),
            "rmse": math.sqrt(statistics.mean(value * value for value in values))} if values else None


def paired_contrasts(trials, fit):
    groups = defaultdict(list)
    identities = set()
    coefficients = fit["coefficients"]
    for trial in trials:
        case, sample = trial["case"], trial["sample"]
        identity = tuple(case[name] for name in ("model", "population", "prompt_tokens", "repeat", "rank_zero"))
        if identity in identities:
            raise ValueError("Duplicate calibration case in paired contrasts")
        identities.add(identity)
        counts, contexts = sample["counts"], sample["context_tokens"]
        if (len(counts) != 2 or len(contexts) != 2 or counts != [case["rank_zero"], case["population"] - case["rank_zero"]]
                or any(type(count) is not int or count < 0 for count in counts)
                or any(not math.isfinite(value) or value < 0 for value in contexts)
                or not math.isfinite(sample["cadence_s"]) or sample["cadence_s"] <= 0):
            raise ValueError("Paired contrast has invalid measured features or counts")
        graph = graph_padding(counts, fit["capture_sizes"])
        if graph is None:
            raise ValueError("Paired contrast exceeds captured graphs")
        groups[identity[:-1]].append({"rank_zero": case["rank_zero"], "imbalance": abs(counts[0] - counts[1]),
            "graph": graph, "peak_context": max(contexts), "cadence_s": sample["cadence_s"]})
    rows, unmatched = [], []
    for identity, cases in sorted(groups.items()):
        model, population, prompt, repeat = identity
        balanced = [case for case in cases if case["imbalance"] == population % 2]
        if not balanced:
            unmatched.append({"model": model, "population": population, "prompt_tokens": prompt, "repeat": repeat})
            continue
        baseline = {name: statistics.mean(case[name] for case in balanced)
                    for name in ("graph", "peak_context", "cadence_s")}
        for case in cases:
            if case in balanced:
                continue
            graph_delta = case["graph"] - baseline["graph"]
            context_delta = case["peak_context"] - baseline["peak_context"]
            measured = case["cadence_s"] - baseline["cadence_s"]
            predicted = coefficients["graph_batch_s"] * graph_delta + coefficients["attention_token_s"] * context_delta
            rows.append({"model": model, "population": population, "prompt_tokens": prompt, "repeat": repeat,
                "rank_zero": case["rank_zero"], "balanced_direction_samples": len(balanced),
                "graph_delta": graph_delta, "peak_context_delta_tokens": context_delta,
                "measured_cadence_delta_s": measured, "global_predicted_delta_s": predicted,
                "global_prediction_error_s": measured - predicted,
                "global_uncertainty_allows_positive_gain_before_control": predicted > 2 * fit["residual_bound_s"]})
    same_graph = [row for row in rows if row["graph_delta"] == 0 and row["peak_context_delta_tokens"] > 0]
    slope = (sum(row["peak_context_delta_tokens"] * row["measured_cadence_delta_s"] for row in same_graph)
             / sum(row["peak_context_delta_tokens"] ** 2 for row in same_graph)) if same_graph else None
    return {"scope": "same_population_prompt_model_repeat_layout_contrasts_not_kernel_attention_or_independent_trials",
        "contrast_count": len(rows), "unmatched_groups": unmatched,
        "global_prediction_error_s": describe([row["global_prediction_error_s"] for row in rows]),
        "same_graph_attention": {"contrasts": len(same_graph), "slope_s_per_peak_context_token": slope,
            "fit_error_s": describe([row["measured_cadence_delta_s"] - slope * row["peak_context_delta_tokens"]
                                     for row in same_graph]),
            "measured_positive_fraction": statistics.mean(row["measured_cadence_delta_s"] > 0 for row in same_graph)
                                          if same_graph else None,
            "slope_is_unconstrained_descriptive_not_installed_runtime_coefficient": True},
        "by_population": {str(population): {
            "contrasts": sum(row["population"] == population for row in rows),
            "global_error_s": describe([row["global_prediction_error_s"] for row in rows if row["population"] == population]),
            "cadence_uncertainty_passes": sum(row["population"] == population and
                row["global_uncertainty_allows_positive_gain_before_control"] for row in rows)}
            for population in sorted({row["population"] for row in rows})}, "contrasts": rows}


def analyze(root, partial=False):
    root = Path(root)
    if partial:
        fit = None
    else:
        from verify_workaware_pilot import verify

        fit = verify(root)
    trials = json.loads((root / "workaware-trials.json").read_text())
    rebuilt = [{"case": trial["case"], "sample": stable_sample(trial["requests"])} for trial in trials]
    if any(trial["sample"] != rebuilt_trial["sample"] for trial, rebuilt_trial in zip(trials, rebuilt)):
        raise ValueError("Stored cadence sample disagrees with independently rebuilt request window")
    fit = fit if fit is not None else fit_cadence([trial["sample"] for trial in rebuilt])
    result = paired_contrasts(rebuilt, fit)
    result.update({"status": "partial_diagnostic_not_verified_not_for_runtime" if partial else "verified_cadence_contrast_analysis",
                   "runtime_changes": False, "global_fit": fit,
                   "raw_trials_sha256": hashlib.sha256((root / "workaware-trials.json").read_bytes()).hexdigest()})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    result = analyze(args.root, args.partial)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: value for name, value in result.items() if name != "contrasts"}, indent=2))


if __name__ == "__main__":
    main()
