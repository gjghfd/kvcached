"""Per-policy metrics shared by live runs and the simulator.

Request records need: model, arrival_s (trace-relative) or arrived_at, reserved_at, first_token_at,
completed_at, output_tokens. Times may be absolute (live, monotonic) or trace-relative (simulator).
"""
import argparse
import json
import math
from pathlib import Path

TENANTS = {0: "code", 1: "conv"}


def percentile(values, quantile):
    values = sorted(values)
    if not values:
        return math.nan
    position = (len(values) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def describe(values, prefix):
    if not values:
        return {f"{prefix}_n": 0}
    return {f"{prefix}_n": len(values), f"{prefix}_avg": sum(values) / len(values),
            f"{prefix}_p50": percentile(values, .5), f"{prefix}_p90": percentile(values, .9),
            f"{prefix}_p95": percentile(values, .95), f"{prefix}_p99": percentile(values, .99),
            f"{prefix}_max": max(values)}


def per_request(request, origin=0.0):
    arrived = request["arrived_at"]
    arrival_s = request.get("arrival_s", arrived - origin)
    output = request["output_tokens"]
    item = {"model": request["model"], "arrival_s": arrival_s,
            "ttft_s": request["first_token_at"] - arrived,
            "latency_s": request["completed_at"] - arrived,
            "tpot_s": (request["completed_at"] - request["first_token_at"]) / (output - 1) if output > 1 else None,
            "queue_s": (request["reserved_at"] - arrived) if request.get("reserved_at") is not None else None,
            "output_tokens": output, "decode_s": request["completed_at"] - request["first_token_at"]}
    return item


def metrics(requests, origin=0.0, warmup_s=0.0, end_s=math.inf, slo=None):
    items = [per_request(request, origin) for request in requests]
    items = [item for item in items if warmup_s <= item["arrival_s"] < end_s]
    result = {"requests": len(items), "window_s": [warmup_s, end_s]}
    for model, name in TENANTS.items():
        group = [item for item in items if item["model"] == model]
        result.update(describe([item["ttft_s"] for item in group], f"{name}_ttft"))
        result.update(describe([item["queue_s"] for item in group if item["queue_s"] is not None], f"{name}_queue"))
        tpots = [item["tpot_s"] for item in group if item["tpot_s"] is not None]
        result.update(describe(tpots, f"{name}_tpot"))
        tokens = sum(item["output_tokens"] - 1 for item in group if item["output_tokens"] > 1)
        result[f"{name}_tpot_token_weighted"] = (sum(item["decode_s"] for item in group) / tokens
                                                 if tokens else math.nan)
        if slo and name in slo:
            limits = slo[name]
            met = [item for item in group if item["ttft_s"] <= limits.get("ttft_s", math.inf)
                   and (item["tpot_s"] is None or item["tpot_s"] <= limits.get("tpot_s", math.inf))]
            result[f"{name}_slo_attainment"] = len(met) / len(group) if group else math.nan
    result.update(describe([item["ttft_s"] for item in items], "all_ttft"))
    return result


def compare(by_policy, baseline="no_move", reference="mig_random"):
    """Relative changes: Code TTFT vs baseline; Conv TPOT degradation vs baseline and vs reference."""
    rows = {}
    base = by_policy.get(baseline)
    ref = by_policy.get(reference)
    for policy, result in by_policy.items():
        row = {}
        for key in ("code_ttft_avg", "code_ttft_p90", "code_ttft_p95", "code_ttft_p99", "conv_tpot_avg",
                    "conv_tpot_p95", "conv_tpot_p99", "conv_ttft_avg", "conv_ttft_p99", "all_ttft_avg",
                    "all_ttft_p99"):
            value = result.get(key, math.nan)
            row[key] = value
            if base and base.get(key):
                row[f"{key}_vs_{baseline}"] = value / base[key] - 1
            if ref and ref.get(key) and policy != reference:
                row[f"{key}_vs_{reference}"] = value / ref[key] - 1
        rows[policy] = row
    return rows


def load_live_policy(root):
    root = Path(root)
    workload = json.loads((root / "workload.json").read_text())
    requests = json.loads((root / "requests.json").read_text())
    return requests, workload["origin"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path, help="per-policy live result directories")
    parser.add_argument("--warmup-s", type=float, default=0.0)
    parser.add_argument("--end-s", type=float, default=math.inf)
    args = parser.parse_args()
    by_policy = {}
    for root in args.roots:
        requests, origin = load_live_policy(root)
        by_policy[root.name] = metrics(requests, origin, args.warmup_s, args.end_s)
    print(json.dumps({"metrics": by_policy, "comparison": compare(by_policy)}, indent=2))


if __name__ == "__main__":
    main()
