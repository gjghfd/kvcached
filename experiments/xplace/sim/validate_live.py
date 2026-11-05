"""Validate the simulator against a live trace run: sim truth = live-fitted params, planner belief = frozen params.

Prints per-policy live vs sim: Code TTFT, Conv TPOT/TTFT, migrations, and the mechanism metric (share of
no-Code time with Conv max-rank count > 8, i.e. CUDA-graph bucket 16).
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

from analyze import metrics  # noqa: E402
from simulator import simulate  # noqa: E402


def code_spans(requests, origin):
    return [(r["reserved_at"] - origin, r["completed_at"] - origin) for r in requests if r["model"] == 0]


def mechanism(population, spans, window):
    """Share of no-Code population samples with Conv max count > 8; mean Conv total in those samples."""
    rows = [p for p in population if window[0] <= p["t"] < window[1]
            and not any(s <= p["t"] < e for s, e in spans) and sum(p["counts"][1])]
    if not rows:
        return {"none_samples": 0}
    return {"none_samples": len(rows), "none_b16_share": sum(max(p["counts"][1]) > 8 for p in rows) / len(rows),
            "none_conv_total": statistics.mean(sum(p["counts"][1]) for p in rows),
            "none_imbalance": statistics.mean(max(p["counts"][1]) - sum(p["counts"][1]) / 4 for p in rows)}


def live_run(root, window):
    workload = json.loads((root / "workload.json").read_text())
    requests = json.loads((root / "requests.json").read_text())
    transactions = json.loads((root / "transactions.json").read_text())
    population = [json.loads(line) for line in (root / "population.jsonl").read_text().splitlines()]
    result = metrics(requests, workload["origin"], *window)
    result.update(mechanism(population, code_spans(requests, workload["origin"]), window))
    result.update(moves=sum(len(t.get("moves", [])) for t in transactions),
                  admissions=sum(t["path"] == "admission" for t in transactions),
                  returns=sum(t["path"] == "return" for t in transactions))
    return result


def sim_run(trace, policy, truth, belief, planner, budget, window, extra=None):
    sim_params = {"num_ranks": 4, "budget_gib": budget, **truth.get("sim", {}), **(extra or {})}
    out = simulate(trace, policy, sim_params=sim_params, step_params=truth, planner_step_params=belief,
                   planner_params=planner)
    result = metrics(out["requests"], 0.0, *window)
    result.update(mechanism(out["population"], code_spans(out["requests"], 0.0), window))
    result.update(moves=len(out["migrations"]), admissions=sum(d["kind"] == "admission" for d in out["decisions"]),
                  returns=sum(d["kind"] == "return" and d["moves"] > 0 for d in out["decisions"]),
                  preemptions=sum(out.get("preemptions", [0, 0])),
                  wall_s=max(request["completed_at"] for request in out["requests"]))
    return result


ROWS = [("code_ttft_avg", "Code TTFT avg s", 1), ("code_ttft_p95", "Code TTFT p95 s", 1),
        ("conv_tpot_avg", "Conv TPOT avg ms", 1e3), ("conv_tpot_p99", "Conv TPOT p99 ms", 1e3),
        ("conv_ttft_avg", "Conv TTFT avg s", 1), ("conv_ttft_p99", "Conv TTFT p99 s", 1),
        ("none_b16_share", "no-Code G16 share", 1), ("none_conv_total", "no-Code Conv total", 1),
        ("moves", "moves", 1), ("admissions", "admission txns", 1), ("returns", "return txns", 1)]


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("live_root", type=Path, help=".../out/trace")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--belief", type=Path, required=True)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--budget-gib", type=float, default=16.0)
    parser.add_argument("--warmup-s", type=float, default=300)
    parser.add_argument("--noise-seeds", type=int, default=0, help="0: deterministic; N: N noisy replicas")
    parser.add_argument("--interference-cv", type=float, default=0.37)
    parser.add_argument("--prefill-cv", type=float, default=0.09)
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text())
    truth, belief, planner = (json.loads(path.read_text()) for path in (args.truth, args.belief, args.planner))
    window = (args.warmup_s, trace["duration_s"])
    policies = [p.name for p in sorted(args.live_root.iterdir()) if (p / "requests.json").exists()]
    if args.noise_seeds:
        print(f"{'metric':22s}" + "".join(f"{policy:>34s}" for policy in policies))
        print(f"{'':22s}" + "".join(f"{'live':>10s}{'sim mean [p5, p95]':>24s}" for _ in policies))
        results = {}
        for policy in policies:
            live = live_run(args.live_root / policy, window)
            sims = [sim_run(trace, policy, truth, belief, planner, args.budget_gib, window,
                            {"interference_cv": args.interference_cv, "prefill_cv": args.prefill_cv,
                             "noise_seed": seed}) for seed in range(args.noise_seeds)]
            results[policy] = (live, sims)
        for key, label, scale in ROWS:
            cells = []
            for policy in policies:
                live, sims = results[policy]
                values = [scale * sim.get(key, math.nan) for sim in sims]
                cells.append(f"{scale * live.get(key, math.nan):10.2f}{statistics.mean(values):10.2f} "
                             f"[{quantile(values, .05):5.1f},{quantile(values, .95):5.1f}]")
            print(f"{label:22s}" + "".join(cells))
        return
    table = {}
    for policy in policies:
        table[policy] = (live_run(args.live_root / policy, window),
                         sim_run(trace, policy, truth, belief, planner, args.budget_gib, window))
    print(f"{'metric':22s}" + "".join(f"{policy:>24s}" for policy in policies))
    print(f"{'':22s}" + "".join(f"{'live':>12s}{'sim':>12s}" for _ in policies))
    for key, label, scale in ROWS:
        cells = []
        for policy in policies:
            live, sim = table[policy]
            cells.append(f"{scale * live.get(key, math.nan):12.2f}{scale * sim.get(key, math.nan):12.2f}")
        print(f"{label:22s}" + "".join(cells))


if __name__ == "__main__":
    main()
