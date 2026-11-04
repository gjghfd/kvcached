"""Fit simulator ground-truth step parameters from one live trace run (calibration policy, e.g. pilot no_move).

The planner keeps its frozen belief (step-params-dp4-v1.json); this file only makes the simulator's clock match
what live runs measured: no-Code Conv steps by capture bucket + max context, cross-tenant interference by Code
phase, Code chunked-prefill and decode speed with Conv active, and migration cost while Code prefills.
"""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from live_samples import code_samples, conv_step_samples, load_run


def fit_conv(samples, window):
    """Step = base + graph[bucket] + attention * max context on all no-Code ticks (bucket 1 pins the intercept);
    interference = mean residual over Code-prefill / Code-decode ticks inside the evaluation window."""
    none = [s for s in samples if s["state"] == "none"]
    rows = [[1.0, s["bucket"] >= 8, s["bucket"] >= 16, s["max_context"]] for s in none]
    target = [s["step_s"] for s in none]
    (base, g8, g16, attention), *_ = np.linalg.lstsq(np.array(rows, float), np.array(target), rcond=None)
    graph = {"1": 0.0, "8": float(g8), "16": float(g8 + g16)}
    tenant = {"base_s": float(base), "graph_s": graph, "graph_slope_s": float(g16) / 16,
              "attention_s_per_token": float(attention)}

    def own(s):
        bucket_cost = graph["16"] if s["bucket"] >= 16 else graph["8"] if s["bucket"] >= 8 else 0.0
        return base + bucket_cost + attention * s["max_context"]

    in_window = [s for s in samples if window[0] <= s["t"] < window[1]]
    # Code decode phases (~15 s) are shorter than a slowed 32-token cadence segment, so no segment lies fully
    # inside one; use ticks inside a decode phase with no concurrent prefill instead.
    residual = {"prefill": [s["step_s"] - own(s) for s in in_window if s["state"] == "prefill"],
                "decode": [s["step_s"] - own(s) for s in in_window
                           if s["code_decode"] and not s["code_prefill"] and s["state"] in ("decode", "transition")]}
    interference = {state: float(np.mean(values)) for state, values in residual.items() if values}
    fit = {"none_n": len(none), "none_rmse_s": float(np.sqrt(np.mean([(own(s) - s["step_s"]) ** 2 for s in none]))),
           "interference_n": {state: len(values) for state, values in residual.items()}}
    return tenant, interference, fit


def fit_code(rows):
    single = [row for row in rows if not row["others_prefill"] and not row["others_decode"]]
    prefill = statistics.median(row["prefill_s"] * 1e5 / row["prompt"] for row in single)
    tpot = statistics.median(row["decode_tpot_s"] for row in single
                             if row["decode_overlap_other_prefill_s"] == 0)
    return prefill, tpot, len(single)


def fit_moves(transactions, prefill_spans, origin):
    normal, slow, fixed = [], [], []
    for item in transactions:
        moves = item.get("moves") or []
        if item.get("status") != "completed" or not moves:
            continue
        for move in moves:
            t = move["started_at"] - origin
            (slow if any(s <= t < e for s, e in prefill_spans) else normal).append(move["elapsed_s"])
        fixed.append(item["control_s"] - item.get("pause_wait_s", 0.0) - sum(move["elapsed_s"] for move in moves))
    return {"move_s": statistics.median(normal) if normal else None,
            "move_s_during_code_prefill": statistics.median(slow) if slow else None,
            "transaction_fixed_s": statistics.median(fixed) if fixed else None,
            "moves_n": [len(normal), len(slow)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration_run", type=Path, help=".../out/trace/<policy> used to fit")
    parser.add_argument("--transaction-runs", type=Path, nargs="*", default=[],
                        help="runs with migrations for transaction costs")
    parser.add_argument("--prior", type=Path, required=True, help="step-params-dp4-v1.json (Code solo numbers)")
    parser.add_argument("--warmup-s", type=float, default=300)
    parser.add_argument("--end-s", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = json.loads(args.prior.read_text())
    samples = conv_step_samples(args.calibration_run, (0.0, args.end_s))
    conv, interference, fit = fit_conv(samples, (args.warmup_s, args.end_s))
    prefill_with_conv, code_tpot, single = fit_code(
        [row for row in code_samples(args.calibration_run) if args.warmup_s <= row["arrival_s"] < args.end_s])
    code = dict(prior["tenants"]["0"])
    code.pop("full_bucket_bonus_s", None)
    code_solo = code["base_s"] + float(code["graph_s"]["1"]) + code["attention_s_per_token"] * 1e5
    moves = {"transaction_fixed_s": [], "move_s": [], "move_s_during_code_prefill": []}
    for run in args.transaction_runs:
        workload, requests, transactions, _ = load_run(run)
        origin = workload["origin"]
        spans = [(r["reserved_at"] - origin, r["first_token_at"] - origin) for r in requests if r["model"] == 0]
        for key, value in fit_moves(transactions, spans, origin).items():
            if key in moves and value is not None:
                moves[key].append(value)
    params = {
        "source": f"live fit from {args.calibration_run} window [{args.warmup_s}, {args.end_s})",
        "tenants": {"0": code, "1": conv},
        "interference": {"1": interference,
                         "0": {"prefill": code_tpot - code_solo, "decode": code_tpot - code_solo}},
        "code_prefill": {"solo_s_per_100k": prior["code_prefill"]["solo_s_per_100k"],
                         "with_other_s_per_100k": prefill_with_conv, "chunk_tokens": 4096},
        "sim": {key: statistics.median(values) for key, values in moves.items() if values},
        "fit": {**fit, "code_single_n": single, "code_tpot_with_conv_s": code_tpot},
    }
    args.output.write_text(json.dumps(params, indent=1) + "\n")
    print(json.dumps(params, indent=1))


if __name__ == "__main__":
    main()
