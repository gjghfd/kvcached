"""Decompose live Conv avg TPOT by concurrent Code state (none / prefill / decode) and transaction pauses.

Each 32-token cadence segment of a Conv request is assigned to the state covering most of its duration.
avg TPOT = sum over states of mean_r(time of r in state / (O_r - 1)), so contributions add up exactly.
"""
import argparse
import json
import math
from pathlib import Path


def intervals_for(requests, transactions, origin, holds=()):
    code = [r for r in requests if r["model"] == 0]
    prefill = [(r["reserved_at"] - origin, r["first_token_at"] - origin) for r in code if r.get("reserved_at")]
    decode = [(r["first_token_at"] - origin, r["completed_at"] - origin) for r in code]
    paused = []
    for item in transactions:
        if 1 not in item.get("pause_models", (0, 1)):
            continue
        receipts = (item.get("pause_receipts") or [None, None])[1] or []
        reached = [receipt["reached_at"] for receipt in receipts if receipt and receipt.get("reached_at")]
        start = (min(reached) if reached else item["started_at"] + item.get("pause_wait_s", 0)) - origin
        paused.append((start, item["started_at"] + item["control_s"] - origin))
    paused += [(hold["paused_at"] - origin, hold["resumed_at"] - origin) for hold in holds if hold.get("model") == 1]
    return prefill, decode, paused


def count_at(intervals, t):
    return sum(start <= t < end for start, end in intervals)


def overlap(a0, a1, intervals):
    return sum(max(0.0, min(a1, end) - max(a0, start)) for start, end in intervals)


def state_of(t0, t1, prefill, decode, paused):
    if overlap(t0, t1, paused) > (t1 - t0) / 2:
        return "paused"
    middle = (t0 + t1) / 2
    p, d = count_at(prefill, middle), count_at(decode, middle)
    if p:
        return f"prefill{min(p, 2)}" + (f"+d" if d else "")
    if d:
        return f"decode{min(d, 2)}"
    return "none"


def decompose(root, window):
    workload = json.loads((root / "workload.json").read_text())
    origin = workload["origin"]
    requests = json.loads((root / "requests.json").read_text())
    transactions = json.loads((root / "transactions.json").read_text())
    holds = json.loads((root / "holds.json").read_text()) if (root / "holds.json").exists() else []
    prefill, decode, paused = intervals_for(requests, transactions, origin, holds)
    contribution, tokens, seconds, n = {}, {}, {}, 0
    for request in requests:
        if request["model"] != 1 or not window[0] <= request["arrival_s"] < window[1]:
            continue
        n += 1
        denominator = request["output_tokens"] - 1
        cadence = request["cadence"]
        for (t0, c0), (t1, c1) in zip(cadence, cadence[1:]):
            t0, t1 = t0 - origin, t1 - origin
            state = state_of(t0, t1, prefill, decode, paused)
            contribution[state] = contribution.get(state, 0.0) + (t1 - t0) / denominator
            tokens[state] = tokens.get(state, 0) + (c1 - c0)
            seconds[state] = seconds.get(state, 0.0) + (t1 - t0)
    end = max(r["completed_at"] for r in requests) - origin
    horizon = (window[0], min(window[1], end))
    span = horizon[1] - horizon[0]
    covered = {"prefill": overlap(*horizon, prefill) / span, "decode": overlap(*horizon, decode) / span,
               "paused": overlap(*horizon, paused) / span}
    states = sorted(contribution, key=lambda key: -contribution[key])
    if n == 0:  # e.g. a censored trial where no Conv of the window completed
        return {"conv_n": 0, "tpot_avg_ms": float("nan"), "states": {}, "code_residency_share": covered}
    return {"conv_n": n, "tpot_avg_ms": 1e3 * sum(contribution.values()) / n,
            "states": {state: {"tpot_contribution_ms": 1e3 * contribution[state] / n,
                               "token_share": tokens[state] / sum(tokens.values()),
                               "ms_per_token": 1e3 * seconds[state] / max(tokens[state], 1)} for state in states},
            "code_residency_share": covered}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("--warmup-s", type=float, default=300)
    parser.add_argument("--end-s", type=float, required=True)
    args = parser.parse_args()
    window = (args.warmup_s, args.end_s)
    for root in sorted(path for path in args.trace_root.iterdir() if (path / "requests.json").exists()):
        result = decompose(root, window)
        share = result["code_residency_share"]
        print(f"\n{root.name}: Conv n={result['conv_n']} avg TPOT {result['tpot_avg_ms']:.2f} ms | wall share: "
              f"Code prefill {share['prefill']:.2f}, Code decode {share['decode']:.2f}, Conv paused {share['paused']:.3f}")
        for state, item in result["states"].items():
            print(f"  {state:12s} contrib {item['tpot_contribution_ms']:6.2f} ms  tokens {100 * item['token_share']:5.1f}%"
                  f"  {item['ms_per_token']:7.1f} ms/token")


if __name__ == "__main__":
    main()
