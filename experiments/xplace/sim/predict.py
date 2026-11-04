"""Pre-registered forecast for a frozen trace: per-policy metric distributions over noisy simulator replicas.

Each policy gets independent noise seeds (a real job runs every policy once with independent noise), so the
forecast spread of a comparison is the spread of (metric_b / metric_a - 1) over independent replica pairs.
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))

from validate_live import quantile, sim_run  # noqa: E402

METRICS = ("code_ttft_avg", "code_ttft_p90", "code_ttft_p95", "conv_tpot_avg", "conv_tpot_p95", "conv_tpot_p99",
           "conv_ttft_avg", "conv_ttft_p99", "none_b16_share", "moves", "admissions", "returns")
COMPARISONS = (("no_move", "mig_random"), ("no_move", "ca_full"), ("no_move_fair", "ca_full"),
               ("mig_random", "ca_full"), ("mig_random", "ca_admit"), ("mig_random", "ca_return"),
               ("ca_return", "ca_full"), ("ca_admit", "ca_full"),
               ("ca_full", "ca_full_tm"), ("no_move", "no_move_tm"), ("no_move_tm", "ca_full_tm"),
               ("mig_random", "mig_random_tm"), ("no_move", "ca_full_tm"),
               ("ca_full", "ca_full_v2"), ("ca_full_tm", "ca_full_tm_v2"), ("no_move", "ca_full_tm_v2"),
               ("no_move", "ca_full_v2"), ("ca_full_tm_v2", "ca_full_tm_s5"), ("ca_full_v2", "ca_full_tm_s5"),
               ("ca_full_v2", "ca_full_tm_v2"), ("ca_full_tm_v2", "ca_full_tm_s4"), ("ca_full_v2", "ca_full_tm_s4"),
               ("no_move", "ca_full_tm_s4"), ("ca_full_tm_s4", "ca_full_tm_slo"), ("ca_full_tm_v2", "ca_full_tm_slo"),
               ("no_move", "ca_full_tm_slo"), ("ca_full_v2", "ca_full_tm_slo"))


def summarize(values):
    values = [value for value in values if not math.isnan(value)]
    return {"mean": statistics.mean(values), "p5": quantile(values, .05), "p95": quantile(values, .95)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--belief", type=Path, required=True)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--policies", default="no_move,ca_full,mig_random,ca_admit,no_move_fair,ca_return")
    parser.add_argument("--budget-gib", type=float, default=16.0)
    parser.add_argument("--warmup-s", type=float, default=300)
    parser.add_argument("--replicas", type=int, default=30)
    parser.add_argument("--interference-cv", type=float, default=0.37)
    parser.add_argument("--prefill-cv", type=float, default=0.09)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text())
    truth, belief, planner = (json.loads(path.read_text()) for path in (args.truth, args.belief, args.planner))
    window = (args.warmup_s, trace["duration_s"])
    policies = args.policies.split(",")
    runs = {}
    for index, policy in enumerate(policies):
        runs[policy] = [sim_run(trace, policy, truth, belief, planner, args.budget_gib, window,
                                {"interference_cv": args.interference_cv, "prefill_cv": args.prefill_cv,
                                 "noise_seed": 1000 * (index + 1) + replica}) for replica in range(args.replicas)]
    forecast = {policy: {key: summarize([run.get(key, math.nan) for run in items]) for key in METRICS}
                for policy, items in runs.items()}
    comparisons = {}
    for base, other in COMPARISONS:
        if base not in runs or other not in runs:
            continue
        comparisons[f"{other}_vs_{base}"] = {}
        for key in ("code_ttft_avg", "conv_tpot_avg", "conv_tpot_p99", "conv_ttft_avg"):
            changes = [b[key] / a[key] - 1 for a in runs[base] for b in runs[other] if a[key] and b[key]]
            comparisons[f"{other}_vs_{base}"][key] = {**summarize(changes),
                                                      "p_negative": sum(change < 0 for change in changes) / len(changes)}
    report = {"trace": str(args.trace), "truth": str(args.truth), "belief": str(args.belief),
              "planner": str(args.planner), "budget_gib": args.budget_gib, "window_s": window,
              "replicas": args.replicas, "interference_cv": args.interference_cv, "prefill_cv": args.prefill_cv,
              "forecast": forecast, "comparisons": comparisons}
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"{'policy':13s} {'Code TTFT avg':>22s} {'Conv TPOT avg ms':>22s} {'Conv TPOT p99 ms':>22s} "
          f"{'Conv TTFT avg':>20s} {'noCode G16':>11s} {'moves':>6s}")
    for policy, item in forecast.items():
        cell = lambda key, scale=1: (f"{scale * item[key]['mean']:7.1f} [{scale * item[key]['p5']:5.1f},"
                                     f"{scale * item[key]['p95']:6.1f}]")
        print(f"{policy:13s} {cell('code_ttft_avg'):>22s} {cell('conv_tpot_avg', 1e3):>22s} "
              f"{cell('conv_tpot_p99', 1e3):>22s} {cell('conv_ttft_avg'):>20s} "
              f"{item['none_b16_share']['mean']:11.2f} {item['moves']['mean']:6.1f}")
    print()
    for name, values in comparisons.items():
        print(f"{name:26s} " + "  ".join(
            f"{key}: {100 * value['mean']:+.1f}% [{100 * value['p5']:+.1f},{100 * value['p95']:+.1f}] "
            f"P(<0)={value['p_negative']:.2f}" for key, value in values.items() if key in ("code_ttft_avg", "conv_tpot_avg")))


if __name__ == "__main__":
    main()
