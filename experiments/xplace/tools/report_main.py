"""Main-experiment report: A/B/ablation tables with paired bootstrap CIs (same trace => per-request pairing).

The request bootstrap only covers request-level variation within one run; run-to-run noise (interference mode,
Code queueing) is larger and is quantified separately with the simulator forecast (sim/predict.py).
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

from analyze import metrics, per_request  # noqa: E402
from decompose import decompose  # noqa: E402
from live_samples import conv_step_samples  # noqa: E402

SLO = {"code": {"ttft_s": 120.0}, "conv": {"ttft_s": 5.0, "tpot_s": 0.050}}


def load(root, policy):
    """Completed requests plus, for a censored trial, the partial records of requests unfinished at the deadline
    (TTFT of a request with no first token is the lower bound deadline - arrival)."""
    directory = root / policy
    requests = json.loads((directory / "requests.json").read_text())
    if (directory / "incomplete.json").exists():
        requests = requests + json.loads((directory / "incomplete.json").read_text())
    workload = json.loads((directory / "workload.json").read_text())
    transactions = json.loads((directory / "transactions.json").read_text())
    holds = json.loads((directory / "holds.json").read_text()) if (directory / "holds.json").exists() else []
    return requests, workload["origin"], transactions, holds


def paired(requests_a, origin_a, requests_b, origin_b, model, key, window):
    a = {r["trace_request_id"]: per_request(r, origin_a) for r in requests_a if r["model"] == model}
    b = {r["trace_request_id"]: per_request(r, origin_b) for r in requests_b if r["model"] == model}
    # sorted: set order of str keys follows the per-process hash seed, which would make the bootstrap CI vary by run
    pairs = [(a[k][key], b[k][key]) for k in sorted(a.keys() & b.keys())
             if window[0] <= a[k]["arrival_s"] < window[1] and a[k][key] is not None and b[k][key] is not None]
    return pairs


def bootstrap_ratio(pairs, rounds=2000, seed=7):
    """Ratio of means (b/a - 1) with a percentile bootstrap CI over paired requests."""
    if not pairs:
        return None
    rng = random.Random(seed)

    def ratio(sample):
        return sum(b for _, b in sample) / sum(a for a, _ in sample) - 1

    point = ratio(pairs)
    draws = sorted(ratio([pairs[rng.randrange(len(pairs))] for _ in pairs]) for _ in range(rounds))
    return {"change": point, "ci95": [draws[int(.025 * rounds)], draws[int(.975 * rounds) - 1]], "n": len(pairs)}


def control(transactions, holds=()):
    return {"transactions": len(transactions),
            "admission": sum(t["path"] == "admission" for t in transactions),
            "return": sum(t["path"] == "return" for t in transactions),
            "moves": sum(len(t.get("moves", [])) for t in transactions),
            "moved_gib": sum(m["kv_bytes"] for t in transactions for m in t.get("moves", [])) / 1024**3,
            "control_s": sum(t.get("control_s", 0) for t in transactions),
            "paused_request_s": sum(t.get("paused_request_s", 0) for t in transactions),
            "holds": len(holds), "held_s": sum(h.get("held_s", 0) for h in holds)}


def mechanism(root, window):
    """Conv avg TPOT split by concurrent Code state, plus no-Code step time and CUDA-graph bucket share."""
    split = decompose(root, window)
    samples = [s for s in conv_step_samples(root, window) if s["state"] == "none"]
    groups = {"none": "none", "paused": "paused"}
    contribution = {}
    for state, item in split["states"].items():
        key = groups.get(state, "prefill" if state.startswith("prefill") else "decode")
        contribution[key] = contribution.get(key, 0.0) + item["tpot_contribution_ms"]
    return {"tpot_by_state_ms": contribution,
            "none_step_ms": 1e3 * sum(s["step_s"] for s in samples) / len(samples) if samples else float("nan"),
            "none_b16_share": sum(s["bucket"] >= 16 for s in samples) / len(samples) if samples else float("nan"),
            "none_avoidable_b16_share": sum(s["bucket"] >= 16 and s["total"] <= 32 for s in samples) / len(samples)
            if samples else float("nan"),
            "code_residency_share": split["code_residency_share"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_root", type=Path, help=".../out/trace")
    parser.add_argument("--warmup-s", type=float, default=300)
    parser.add_argument("--end-s", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    window = (args.warmup_s, args.end_s)
    policies = [p.name for p in sorted(args.trace_root.iterdir()) if (p / "requests.json").exists()]
    data = {policy: load(args.trace_root, policy) for policy in policies}
    table = {policy: {**metrics(requests, origin, *window, slo=SLO), **control(transactions, holds)}
                 for policy, (requests, origin, transactions, holds) in data.items()}
    mechanisms = {policy: mechanism(args.trace_root / policy, window) for policy in policies}
    comparisons = {}
    for base in policies:
        if base not in data:
            continue
        for policy in policies:
            if policy == base:
                continue
            ra, oa = data[base][:2]
            rb, ob = data[policy][:2]
            comparisons[f"{policy}_vs_{base}"] = {
                "code_ttft": bootstrap_ratio(paired(ra, oa, rb, ob, 0, "ttft_s", window)),
                "conv_tpot": bootstrap_ratio(paired(ra, oa, rb, ob, 1, "tpot_s", window)),
                "conv_ttft": bootstrap_ratio(paired(ra, oa, rb, ob, 1, "ttft_s", window))}
    report = {"window_s": window, "slo": SLO, "table": table, "comparisons": comparisons, "mechanism": mechanisms}
    for policy, (requests, origin, _, _) in data.items():
        gaps = sorted(r["max_token_gap_s"] for r in requests if r["model"] == 1 and r.get("max_token_gap_s") is not None
                      and window[0] <= per_request(r, origin)["arrival_s"] < window[1])
        if gaps:
            table[policy]["conv_max_gap_p50"] = gaps[len(gaps) // 2]
            table[policy]["conv_max_gap_p99"] = gaps[min(len(gaps) - 1, int(.99 * len(gaps)))]
        censored = [r for r in requests if r.get("censored")]
        table[policy]["censored"] = len(censored)
        table[policy]["censored_code_no_first_token"] = sum(1 for r in censored if r["model"] == 0
                                                            and not r.get("first_token_observed"))
        # vLLM recompute preemptions over all requests of the trial (completed and censored)
        table[policy]["preemptions"] = sum(r.get("preemptions") or 0 for r in requests)
    rows = ["| policy | Code TTFT avg / p90 / p95 / p99 (s) | Conv TPOT avg / p95 / p99 (ms) | Conv TTFT avg / p95 / p99 (s) | "
            "Conv max token gap p50 / p99 (s) | SLO Code / Conv | moves | preemptions | paused req·s | held s | censored |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for policy, m in table.items():
        rows.append(f"| {policy} | {m['code_ttft_avg']:.1f} / {m['code_ttft_p90']:.1f} / {m['code_ttft_p95']:.1f} / "
                    f"{m['code_ttft_p99']:.1f} | {1e3 * m['conv_tpot_avg']:.2f} / {1e3 * m['conv_tpot_p95']:.1f} / "
                    f"{1e3 * m['conv_tpot_p99']:.1f} | {m['conv_ttft_avg']:.2f} / {m['conv_ttft_p95']:.2f} / "
                    f"{m['conv_ttft_p99']:.2f} | {m.get('conv_max_gap_p50', float('nan')):.1f} / "
                    f"{m.get('conv_max_gap_p99', float('nan')):.1f} | {m.get('code_slo_attainment', float('nan')):.2f} / "
                    f"{m.get('conv_slo_attainment', float('nan')):.2f} | {m['moves']} | {m['preemptions']} | "
                    f"{m['paused_request_s']:.0f} | {m['held_s']:.0f} | {m['censored']} "
                    f"({m['censored_code_no_first_token']} Code unstarted) |")
    rows += ["", "| comparison | Code TTFT change [95% CI] | Conv TPOT change [95% CI] | Conv TTFT change [95% CI] |",
             "|---|---|---|---|"]
    for name, values in comparisons.items():
        cells = []
        for key in ("code_ttft", "conv_tpot", "conv_ttft"):
            item = values[key]
            cells.append("n/a" if item is None else
                         f"{100 * item['change']:+.1f}% [{100 * item['ci95'][0]:+.1f}, {100 * item['ci95'][1]:+.1f}] (n={item['n']})")
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    rows += ["", "| policy | Conv TPOT from: no Code / Code prefill / Code decode / paused (ms) | no-Code step (ms) | "
             "no-Code G16 share (avoidable) | Code resident share (prefill+decode) |", "|---|---|---|---|---|"]
    for policy, m in mechanisms.items():
        parts = m["tpot_by_state_ms"]
        share = m["code_residency_share"]
        rows.append(f"| {policy} | {parts.get('none', 0):.2f} / {parts.get('prefill', 0):.2f} / "
                    f"{parts.get('decode', 0):.2f} / {parts.get('paused', 0):.2f} | {m['none_step_ms']:.2f} | "
                    f"{m['none_b16_share']:.2f} ({m['none_avoidable_b16_share']:.2f}) | "
                    f"{share['prefill'] + share['decode']:.2f} |")
    print("\n".join(rows))
    if args.output:
        args.output.write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
