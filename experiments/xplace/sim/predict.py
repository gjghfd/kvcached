"""Pre-registered forecast for a frozen trace: per-policy metric distributions over noisy simulator replicas.

Each policy gets independent noise seeds (a real job runs every policy once with independent noise), so the
forecast spread of a comparison is the spread of (metric_b / metric_a - 1) over independent replica pairs.
"""
import argparse
import json
import math
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))

from validate_live import quantile, sim_run  # noqa: E402

METRICS = ("code_ttft_avg", "code_ttft_p90", "code_ttft_p95", "conv_tpot_avg", "conv_tpot_p95", "conv_tpot_p99",
           "conv_ttft_avg", "conv_ttft_p99", "none_b16_share", "moves", "admissions", "returns", "preemptions",
           "wall_s")
COMPARISONS = (("no_move", "mig_random"), ("no_move", "ca_full"), ("no_move_fair", "ca_full"),
               ("mig_random", "ca_full"), ("mig_random", "ca_admit"), ("mig_random", "ca_return"),
               ("ca_return", "ca_full"), ("ca_admit", "ca_full"),
               ("ca_full", "ca_full_tm"), ("no_move", "no_move_tm"), ("no_move_tm", "ca_full_tm"),
               ("mig_random", "mig_random_tm"), ("no_move", "ca_full_tm"),
               ("ca_full", "ca_full_v2"), ("ca_full_tm", "ca_full_tm_v2"), ("no_move", "ca_full_tm_v2"),
               ("no_move", "ca_full_v2"), ("ca_full_tm_v2", "ca_full_tm_s5"), ("ca_full_v2", "ca_full_tm_s5"),
               ("ca_full_v2", "ca_full_tm_v2"), ("ca_full_tm_v2", "ca_full_tm_s4"), ("ca_full_v2", "ca_full_tm_s4"),
               ("no_move", "ca_full_tm_s4"), ("ca_full_tm_s4", "ca_full_tm_slo"), ("ca_full_tm_v2", "ca_full_tm_slo"),
               ("no_move", "ca_full_tm_slo"), ("ca_full_v2", "ca_full_tm_slo"),
               # master table (current-length setting): each pair ablates one component
               ("kvcached_native", "no_move_v2"), ("kvcached_native", "ca_full_tm_slo"),
               ("kvcached_native", "ca_full_v2"), ("no_move_v2", "no_move_fair_v2"),
               ("no_move_v2", "mig_random_v2"), ("mig_random_v2", "ca_admit_v2"),
               ("mig_random_v2", "ca_return_v2"), ("mig_random_v2", "ca_full_v2"), ("ca_admit_v2", "ca_full_v2"),
               ("ca_return_v2", "ca_full_v2"), ("no_move_v2", "no_move_tm_v2"), ("no_move_tm_v2", "mig_random_tm_v2"),
               ("mig_random_v2", "mig_random_tm_v2"), ("mig_random_tm_v2", "ca_full_tm_v2"),
               ("no_move_v2", "ca_full_tm_slo"), ("no_move_tm_v2", "ca_full_tm_v2"),
               ("ca_full_v2_nh", "ca_full_v2"), ("ca_full_tm_slo_nh", "ca_full_tm_slo"),
               ("mig_random_v2", "ca_full_v2_nh"), ("ca_full_tm_s4", "ca_full_tm_slo"),
               # ablations of the final scheme: each variant -> ca_full_tm_slo
               ("no_move_tm_slo", "ca_full_tm_slo"), ("mig_random_tm_slo", "ca_full_tm_slo"),
               ("ca_admit_tm_slo", "ca_full_tm_slo"), ("ca_return_tm_slo", "ca_full_tm_slo"),
               ("ca_full_tm_slo_v1", "ca_full_tm_slo"), ("mig_random_tm_slo", "ca_return_tm_slo"),
               ("mig_random_tm_slo", "ca_admit_tm_slo"),
               # baselines without growth headroom
               ("no_move_v2_nh", "ca_full_tm_slo"), ("no_move_tm_slo_nh", "ca_full_tm_slo"),
               ("no_move_v2", "no_move_v2_nh"), ("no_move_tm_slo", "no_move_tm_slo_nh"),
               ("kvcached_native", "no_move_v2_nh"),
               # general (type-agnostic) algorithm *_g: overall, migration, placement, headroom, scheduling
               ("no_move_v2_nh_g", "ca_full_tm_slo_g"), ("no_move_tm_slo_nh_g", "ca_full_tm_slo_g"),
               ("mig_random_tm_slo_g", "ca_full_tm_slo_g"), ("ca_admit_tm_slo_g", "ca_full_tm_slo_g"),
               ("ca_return_tm_slo_g", "ca_full_tm_slo_g"), ("mig_random_tm_slo_g", "ca_admit_tm_slo_g"),
               ("mig_random_tm_slo_g", "ca_return_tm_slo_g"), ("ca_full_tm_slo_nh_g", "ca_full_tm_slo_g"),
               ("ca_full_tm_v2_g", "ca_full_tm_slo_g"), ("ca_full_v2_g", "ca_full_tm_slo_g"),
               ("no_move_v2_nh_g", "no_move_tm_slo_nh_g"), ("no_move_v2_nh_g", "ca_full_v2_g"),
               ("ca_full_tm_slo", "ca_full_tm_slo_g"))


def summarize(values):
    values = [value for value in values if not math.isnan(value)]
    if not values:  # e.g. a mechanism metric with no qualifying period in any replica
        return {"mean": math.nan, "p5": math.nan, "p95": math.nan}
    return {"mean": statistics.mean(values), "p5": quantile(values, .05), "p95": quantile(values, .95)}


def _replica(task):
    return sim_run(*task)


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
    parser.add_argument("--kv-mode", choices=("full", "prompt"), default="full")
    parser.add_argument("--reserve-extra-tokens", type=int, default=0)
    parser.add_argument("--page-model", choices=("blocks", "high_water"), default="blocks",
                        help="high_water reproduces the forecasts up to V12")
    parser.add_argument("--vary-admission-seed", action="store_true",
                        help="also vary the admission RNG per replica (random migration's choices), not only noise")
    parser.add_argument("--workers", type=int, default=1, help="processes; replicas and noise seeds are unchanged")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text())
    truth, belief, planner = (json.loads(path.read_text()) for path in (args.truth, args.belief, args.planner))
    window = (args.warmup_s, trace["duration_s"])
    policies = args.policies.split(",")
    def replica_params(index, replica):
        params = {"interference_cv": args.interference_cv, "prefill_cv": args.prefill_cv,
                  "noise_seed": 1000 * (index + 1) + replica, "kv_mode": args.kv_mode,
                  "reserve_extra_tokens": args.reserve_extra_tokens, "page_model": args.page_model}
        if args.vary_admission_seed:  # replica 0 keeps the simulator's default admission seed
            params["seed"] = 20261103 + 7919 * replica
        return params

    tasks = [(policy, (trace, policy, truth, belief, planner, args.budget_gib, window, replica_params(index, replica)))
             for index, policy in enumerate(policies) for replica in range(args.replicas)]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_replica, [task for _, task in tasks]))
    else:
        results = [_replica(task) for _, task in tasks]
    runs = {policy: [] for policy in policies}
    for (policy, _), result in zip(tasks, results):
        runs[policy].append(result)
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
              "kv_mode": args.kv_mode, "reserve_extra_tokens": args.reserve_extra_tokens, "page_model": args.page_model,
              "vary_admission_seed": args.vary_admission_seed, "forecast": forecast, "comparisons": comparisons}
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"{'policy':17s} {'Code TTFT avg':>22s} {'Conv TPOT avg ms':>22s} {'Conv TPOT p99 ms':>22s} "
          f"{'Conv TTFT avg':>20s} {'noCode G16':>11s} {'moves':>6s} {'retract':>8s} {'wall s':>7s}")
    for policy, item in forecast.items():
        cell = lambda key, scale=1: (f"{scale * item[key]['mean']:7.1f} [{scale * item[key]['p5']:5.1f},"
                                     f"{scale * item[key]['p95']:6.1f}]")
        print(f"{policy:17s} {cell('code_ttft_avg'):>22s} {cell('conv_tpot_avg', 1e3):>22s} "
              f"{cell('conv_tpot_p99', 1e3):>22s} {cell('conv_ttft_avg'):>20s} "
              f"{item['none_b16_share']['mean']:11.2f} {item['moves']['mean']:6.1f} {item['preemptions']['mean']:8.1f} "
              f"{item['wall_s']['p95']:7.0f}")
    print()
    for name, values in comparisons.items():
        print(f"{name:26s} " + "  ".join(
            f"{key}: {100 * value['mean']:+.1f}% [{100 * value['p5']:+.1f},{100 * value['p95']:+.1f}] "
            f"P(<0)={value['p_negative']:.2f}" for key, value in values.items() if key in ("code_ttft_avg", "conv_tpot_avg")))


if __name__ == "__main__":
    main()
