"""Generate manuscript tables exclusively from the migration-only evidence allowlist."""
import json
import statistics

import paper_data as D

OUT = D.ROOT / "Paper/draft"


def median(values):
    return statistics.median(values) if values else None


def table_row(seed, sid):
    data, metrics = D.trial(seed, sid), D.summary(seed, sid)
    gaps = [request["max_token_gap_s"] for request in D.cohort(seed, sid, 1)]
    return {"sid": sid, "name": D.NAMES[sid], "seed": seed, "policy": D.POLICIES[sid],
            "code_ttft_avg": metrics["code_ttft_avg"], "conv_tpot_avg_ms": metrics["conv_tpot_avg"] * 1000,
            "conv_tpot_p95_ms": metrics["conv_tpot_p95"] * 1000,
            "conv_tpot_p99_ms": metrics["conv_tpot_p99"] * 1000,
            "conv_ttft_avg": metrics["conv_ttft_avg"], "conv_ttft_p99": metrics["conv_ttft_p99"],
            "conv_gap_p50": median(gaps), "code_slo": metrics["code_slo_attainment"],
            "conv_slo": metrics["conv_slo_attainment"], "moves": len(data["migrations"]),
            "preemptions": sum(request.get("preemptions", 0) for request in data["requests"])}


def overheads(seed):
    data = D.trial(seed, "X")
    migrations, transactions = data["migrations"], data["transactions"]
    by_path = {name: [item for item in transactions if item["path"] == name] for name in ("admission", "return")}
    return {"seed": seed,
            "transactions_admission": len(by_path["admission"]), "transactions_return": len(by_path["return"]),
            "moves_admission": sum(len(item["moves"]) for item in by_path["admission"]),
            "moves_return": sum(len(item["moves"]) for item in by_path["return"]),
            "moved_gib_median": median([item["kv_bytes"] / 1024**3 for item in migrations]),
            "move_s_median": median([item["elapsed_s"] for item in migrations]),
            "move_s_max": max(item["elapsed_s"] for item in migrations),
            "pause_wait_s_median": median([item["pause_wait_s"] for item in transactions]),
            "pause_wait_s_max": max(item["pause_wait_s"] for item in transactions),
            "control_s_admission_median": median([item["control_s"] for item in by_path["admission"]]),
            "control_s_return_median": median([item["control_s"] for item in by_path["return"]]),
            "solve_s_admission_median": median([item["solve_s"] for item in by_path["admission"]]),
            "paused_request_s": sum(item["paused_request_s"] for item in transactions),
            "verified": sum(bool(item["bitwise_verified"]) for item in migrations), "migrations": len(migrations)}


def correctness():
    result = {"trials": 0, "requests": 0, "exact": 0, "migrations": 0, "verified": 0}
    for seed in D.SEEDS:
        trace = D.read(D.V16 / "inputs" / f"cur-b16-c012-p120-s{seed}-d900.json")
        expected = {request["request_id"]: request["output_tokens"] for request in trace["requests"]}
        for sid in D.POLICIES:
            data = D.trial(seed, sid)
            result["trials"] += 1
            result["requests"] += len(data["requests"])
            result["exact"] += sum(request["output_tokens"] == expected[request["trace_request_id"]]
                                   for request in data["requests"])
            result["migrations"] += len(data["migrations"])
            result["verified"] += sum(bool(move["bitwise_verified"]) for move in data["migrations"])
    return result


def workload(seed):
    trace = D.read(D.V16 / "inputs" / f"cur-b16-c012-p120-s{seed}-d900.json")
    conv = [request for request in trace["requests"] if request["model"] == 1]
    code = [request for request in trace["requests"] if request["model"] == 0]
    return {"seed": seed, "requests": len(trace["requests"]), "conv": len(conv), "code": len(code),
            "conv_window": sum(D.WINDOW[0] <= request["arrival_s"] < D.WINDOW[1] for request in conv),
            "code_prompt": [request["prompt_tokens"] for request in code],
            "conv_prompt_mean": statistics.mean(request["prompt_tokens"] for request in conv),
            "conv_output_mean": statistics.mean(request["output_tokens"] for request in conv),
            "conv_output_max": max(request["output_tokens"] for request in conv)}


def scan_group(rows, axis, values):
    result = []
    for value in values:
        selected = [row for row in rows if row[axis] == value]
        baseline = {(row["seed"], row["replica"]): row for row in selected if row["policy"] == D.POLICIES["B"]}
        entry = {axis: value, "policies": {}}
        for sid, policy in D.POLICIES.items():
            trials = [row for row in selected if row["policy"] == policy]
            metrics = {}
            for key in ("code_ttft_avg", "conv_tpot_avg", "preemptions"):
                samples = [row[key] for row in trials]
                metrics[key] = {"median": median(samples), "min": min(samples), "max": max(samples)}
            if sid == "X":
                deltas = [100 * (row["conv_tpot_avg"] / baseline[(row["seed"], row["replica"])]["conv_tpot_avg"] - 1)
                          for row in trials]
                metrics["conv_tpot_change_pct"] = {"median": median(deltas), "min": min(deltas), "max": max(deltas)}
            entry["policies"][sid] = {"n": len(trials), **metrics}
        result.append(entry)
    return result


def simulation():
    rows = D.sweep_rows()
    rates = scan_group([row for row in rows if row["budget"] == 16], "rate", (0.08, 0.10, 0.12, 0.14))
    budgets = scan_group([row for row in rows if row["rate"] == 0.12], "budget", (16, 20, 24))
    pairs = D.imbalance_rows()
    deltas = [row["b_pct"] for row in pairs]
    return {"rate": rates, "budget": budgets,
            "imbalance": {"pairs": len(pairs), "improved": sum(value < 0 for value in deltas),
                          "change_median_pct": median(deltas),
                          "random_preemptions": sum(row["random_preemptions"] for row in pairs),
                          "ours_preemptions": sum(row["cost_aware_preemptions"] for row in pairs)}}


def forecast_error():
    rows = []
    for seed in D.SEEDS:
        forecast = D.read(D.EVIDENCE / "xplace-inputs" / f"forecast-v16-general-s{seed}.json")["forecast"]
        for sid, policy in D.POLICIES.items():
            live = D.summary(seed, sid)
            rows.append({"seed": seed, "sid": sid, "policy": policy,
                         **{key: forecast[policy][key]["mean"] / live[key] - 1
                            for key in ("code_ttft_avg", "conv_tpot_avg")}})
    return {"runs": rows, **{key: {"n": len(rows), "median": median([abs(row[key]) for row in rows]),
                                  "max": max(abs(row[key]) for row in rows)}
                            for key in ("code_ttft_avg", "conv_tpot_avg")}}


def change_text(change):
    lower, upper = change["ci95"]
    return f"{change['change'] * 100:+.1f}% [{lower * 100:+.1f}, {upper * 100:+.1f}]"


def main():
    result = {"scope": "migration-only", "policies": D.POLICIES, "names": D.NAMES,
              "table2": [table_row(seed, sid) for seed in D.SEEDS for sid in D.POLICIES],
              "table3": [{"seed": seed, **D.comparison(seed)} for seed in D.SEEDS],
              "table4": [overheads(seed) for seed in D.SEEDS], "correctness": correctness(),
              "workload": [workload(seed) for seed in D.SEEDS],
              "mechanism": {str(seed): {sid: D.report(seed)["mechanism"][policy] for sid, policy in D.POLICIES.items()}
                            for seed in D.SEEDS},
              "free_at_arrivals": D.free_at_arrivals(), "simulation": simulation(), "forecast_error": forecast_error()}
    (OUT / "tables.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 论文数据（仅限迁移方案，由 paper_tables.py 生成）", "", "## 主表", "",
             "| 踪迹种子 | 策略 | Code TTFT (s) | Conv TPOT均值/P95/P99 (ms) | Conv TTFT均值/P99 (s) | SLO Code/Conv | 迁移/抢占 |",
             "|---|---|---|---|---|---|---|"]
    for row in result["table2"]:
        lines.append(f"| {row['seed']} | {row['name']} | {row['code_ttft_avg']:.1f} | "
                     f"{row['conv_tpot_avg_ms']:.2f}/{row['conv_tpot_p95_ms']:.1f}/{row['conv_tpot_p99_ms']:.1f} | "
                     f"{row['conv_ttft_avg']:.2f}/{row['conv_ttft_p99']:.2f} | {row['code_slo']:.2f}/{row['conv_slo']:.2f} | "
                     f"{row['moves']}/{row['preemptions']} |")
    lines += ["", "## 配对比较（XPlace相对不迁移基线）", "",
              "| 种子 | Code TTFT | Conv TPOT | Conv TTFT |", "|---|---|---|---|"]
    for row in result["table3"]:
        lines.append(f"| {row['seed']} | {change_text(row['code_ttft'])} | {change_text(row['conv_tpot'])} | "
                     f"{change_text(row['conv_ttft'])} |")
    for key in ("table4", "correctness", "mechanism", "simulation", "forecast_error"):
        lines += ["", f"## {key}", "", "```json", json.dumps(result[key], ensure_ascii=False, indent=2), "```"]
    (OUT / "tables.md").write_text("\n".join(lines) + "\n")
    return OUT / "tables.json"


if __name__ == "__main__":
    print(main())
