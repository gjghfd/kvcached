import argparse
import json
import statistics
from pathlib import Path

from layout_cost_pilot import POPULATIONS, cadence, verify


def summarize(lookup, trials):
    result = []
    for population in POPULATIONS:
        row = lookup["tables"][str(population)]
        values, balanced = row["cadence_s"], row["balanced_cadence_s"]
        imbalance = [abs(2 * count - population) / population for count in range(population + 1)]
        ratios = [value / balanced for value in values]
        alpha = sum(weight * (ratio - 1) for weight, ratio in zip(imbalance, ratios)) / sum(
            weight**2 for weight in imbalance)
        repeats = [[statistics.mean(cadence(request) for trial in trials
                     if trial["case"] == {"population": population, "rank_zero": count, "repeat": repeat}
                     for request in trial["requests"]) for repeat in (0, 1)] for count in range(population + 1)]
        result.append({"population": population, "balanced_cadence_ms": balanced * 1000,
                       "extreme_cadence_ms": [values[0] * 1000, values[-1] * 1000],
                       "extreme_throughput_loss_fraction": [1 - balanced / values[0], 1 - balanced / values[-1]],
                       "best_count_rank_zero": min(range(population + 1), key=values.__getitem__),
                       "worst_cadence_ratio_to_balanced": max(ratios),
                       "max_hot_direction_cadence_difference_ms": max(abs(values[count] - values[-count - 1])
                                                                       for count in range(population + 1)) * 1000,
                       "max_pair_repeat_relative_spread": max(abs(pair[1] - pair[0]) / statistics.mean(pair)
                                                                  for pair in repeats),
                       "descriptive_affine_alpha": alpha,
                       "descriptive_affine_max_abs_ratio_residual": max(abs(ratio - 1 - alpha * weight)
                                                                          for ratio, weight in zip(ratios, imbalance)),
                       "cadence_ms_all_counts": [value * 1000 for value in values]})
    return {"status": "independently_verified_layout_diagnostics", "populations": result,
            "scope": "two_repeats_one_active_Conv_not_context_or_Code_prefill_joint_generalization"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    lookup = verify(options.root)
    result = summarize(lookup, json.loads((options.root / "layout-trials.json").read_text()))
    options.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**result, "populations": [{key: value for key, value in row.items()
                                               if key != "cadence_ms_all_counts"} for row in result["populations"]]}, indent=2))


if __name__ == "__main__":
    main()
