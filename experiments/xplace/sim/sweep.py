"""Sweep workload regimes in the simulator: Code TTFT and Conv TPOT per policy (DP=4)."""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE.parent / "runtime"))

from analyze import metrics  # noqa: E402
from cost_model import default_dp2_params  # noqa: E402
from make_trace import DEFAULT_INPUTS, make_trace  # noqa: E402
from simulator import POLICIES, simulate  # noqa: E402


def run_one(job):
    trace, policy, sim_params, step_params, planner_params, warmup_s = job
    output = simulate(trace, policy, sim_params=sim_params, step_params=step_params, planner_params=planner_params)
    result = metrics(output["requests"], warmup_s=warmup_s, end_s=trace["duration_s"])
    result["moves"] = len(output["migrations"])
    result["return_moves"] = sum(item["kind"] == "return" for item in output["migrations"])
    result["paused_request_s"] = sum(item["paused_request_s"] for item in output["decisions"])
    utilization = [sum(sample["used_gib"]) / sum(output["budgets_gib"]) for sample in output["population"]
                   if warmup_s <= sample["t"] < trace["duration_s"]]
    result["kv_utilization_avg"] = sum(utilization) / len(utilization) if utilization else float("nan")
    return policy, result


def sweep(regime, policies, step_params=None, planner_params=None, workers=6):
    step_params = step_params or default_dp2_params()
    trace = make_trace(DEFAULT_INPUTS, regime["conv_rate"], regime["code_rate"], regime["duration_s"], regime["seed"])
    sim_params = {"num_ranks": regime.get("num_ranks", 4), "budget_gib": regime["budget_gib"], "transaction_fixed_s": 0.2, "move_fixed_s": 0.015, "copy_gib_s": 20.0}
    jobs = [(trace, policy, sim_params, step_params, planner_params or {}, regime.get("warmup_s", 600))
            for policy in policies]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(run_one, jobs))


def summarize(results):
    base = results.get("no_move", {})
    reference = results.get("mig_random", {})
    lines = []
    for policy, result in results.items():
        conv = result.get("conv_tpot_avg", float("nan"))
        line = (f"{policy:12s} code n={result.get('code_ttft_n', 0):3d} ttft avg {result.get('code_ttft_avg', float('nan')):7.1f}"
                f" p95 {result.get('code_ttft_p95', float('nan')):7.1f} | conv n={result.get('conv_tpot_n', 0):4d}"
                f" tpot {1e3 * conv:6.2f}ms")
        if base.get("conv_tpot_avg"):
            line += f" ({100 * (conv / base['conv_tpot_avg'] - 1):+5.1f}% vs nm)"
        if reference.get("conv_tpot_avg") and policy not in ("no_move", "no_move_fair", "mig_random"):
            line += f" ({100 * (conv / reference['conv_tpot_avg'] - 1):+5.1f}% vs mr)"
        line += (f" ttft p99 {result.get('conv_ttft_p99', float('nan')):5.2f}s | moves {result['moves']:4d}"
                 f" (ret {result['return_moves']:3d}) util {result['kv_utilization_avg']:.2f}")
        lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-gib", type=float, nargs="+", default=[40.0])
    parser.add_argument("--conv-rate", type=float, nargs="+", default=[0.16])
    parser.add_argument("--code-rate", type=float, nargs="+", default=[0.002])
    parser.add_argument("--duration-s", type=float, default=4200)
    parser.add_argument("--warmup-s", type=float, default=600)
    parser.add_argument("--seed", type=int, nargs="+", default=[1])
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--step-params", type=Path)
    parser.add_argument("--planner-params", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    step_params = json.loads(args.step_params.read_text()) if args.step_params else None
    planner_params = json.loads(args.planner_params.read_text()) if args.planner_params else None
    collected = []
    for budget in args.budget_gib:
        for conv_rate in args.conv_rate:
            for code_rate in args.code_rate:
                for seed in args.seed:
                    regime = {"budget_gib": budget, "conv_rate": conv_rate, "code_rate": code_rate,
                              "duration_s": args.duration_s, "warmup_s": args.warmup_s, "seed": seed}
                    results = sweep(regime, args.policies.split(","), step_params, planner_params)
                    print(f"=== budget {budget} GiB conv {conv_rate} code {code_rate} seed {seed}", flush=True)
                    print(summarize(results), flush=True)
                    collected.append({"regime": regime, "results": results})
    if args.json:
        args.json.write_text(json.dumps(collected, indent=1))


if __name__ == "__main__":
    main()
