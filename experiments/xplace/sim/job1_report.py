"""Summarize a fetched Job 1 (calibration + pilot) and validate the simulator against the live pilot."""
import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE.parent / "runtime"))

from analyze import metrics  # noqa: E402
from cost_model import StepModel  # noqa: E402
from fit import case_features  # noqa: E402
from simulator import simulate  # noqa: E402


def load(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def transaction_costs(transactions):
    """Fixed (non-copy) control time and per-GiB copy time from live admission/return transactions."""
    fixed, per_move, copy_rates = [], [], []
    for item in transactions:
        moves = item.get("moves") or []
        if item.get("status") != "completed":
            continue
        copy = sum(move["stage_s"].get("peer_copy", 0) for move in moves)
        other = sum(move["elapsed_s"] - move["stage_s"].get("peer_copy", 0) for move in moves)
        fixed.append(item["control_s"] - copy - other)
        per_move.extend(move["elapsed_s"] - move["stage_s"].get("peer_copy", 0) for move in moves)
        copy_rates.extend(move["kv_bytes"] / 1024**3 / move["stage_s"]["peer_copy"] for move in moves
                          if move["stage_s"].get("peer_copy"))
    summary = {"transactions": len(fixed)}
    if fixed:
        summary.update(fixed_s_median=statistics.median(fixed), fixed_s_max=max(fixed))
    if per_move:
        summary.update(move_overhead_s_median=statistics.median(per_move))
    if copy_rates:
        summary.update(copy_gib_s_median=statistics.median(copy_rates))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_root", type=Path, help="evidence/xplace-<tag>/job-<id>")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--warmup-s", type=float, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.job_root / "out"
    report = {"phases": load(out / "phases.json", {}).get("phases", []), "budget": load(out / "budget.json")}
    dataplane = load(out / "calibration" / "dataplane.json")
    if dataplane:
        stages = [migration["stage_s"] for migration in dataplane["migrations"]]
        report["dataplane"] = {"status": dataplane["status"], "migrations": len(dataplane["migrations"]),
                               "elapsed_s": [round(m["elapsed_s"], 3) for m in dataplane["migrations"]],
                               "peer_copy_s": [round(stage.get("peer_copy", 0), 3) for stage in stages],
                               "transactions": [{key: item.get(key) for key in ("status", "control_s", "pause_wait_s",
                                                                                  "solve_s", "resume_s")}
                                                for item in dataplane["transactions"]]}
    grid = load(out / "calibration" / "conv-grid.json", [])
    params = load(out / "calibration" / "step-params.json")
    if grid and params:
        model = StepModel(params)
        rows = []
        for case in grid:
            if case.get("status") != "measured" or not case.get("step_s"):
                rows.append({"name": case["name"], "status": case.get("status")})
                continue
            bucket, context = case_features(case)
            counts = [case["max_count"]] + [0] * 3
            predicted = model.decode_step(1, counts, [context] + [0.0] * 3)
            rows.append({"name": case["name"], "bucket": bucket, "max_context": round(context),
                         "measured_ms": round(1e3 * case["step_s"], 2), "predicted_ms": round(1e3 * predicted, 2),
                         "error_pct": round(100 * (predicted / case["step_s"] - 1), 1)})
        report["grid"] = rows
        report["step_params"] = params
    report["interference"] = load(out / "calibration" / "interference.json")
    trace = json.loads(args.trace.read_text())
    pilot = {}
    for root in sorted((out / "trace").glob("*")) if (out / "trace").exists() else []:
        requests = load(root / "requests.json", [])
        workload = load(root / "workload.json")
        if not requests or not workload:
            continue
        live = metrics(requests, workload["origin"], args.warmup_s, trace["duration_s"])
        costs = transaction_costs(load(root / "transactions.json", []))
        pilot[root.name] = {"live": live, "costs": costs, "summary": load(root / "summary.json")}
    if pilot and params:
        all_costs = [item["costs"] for item in pilot.values() if item["costs"].get("transactions")]
        sim_params = {"num_ranks": 4, "budget_gib": report["budget"]["budget_gib"]}
        if all_costs:
            sim_params["transaction_fixed_s"] = statistics.median(item["fixed_s_median"] for item in all_costs)
            rates = [item["copy_gib_s_median"] for item in all_costs if "copy_gib_s_median" in item]
            if rates:
                sim_params["copy_gib_s"] = statistics.median(rates)
        planner = json.loads((args.job_root / "inputs" / "planner-params-v1.json").read_text())
        for policy, item in pilot.items():
            simulated = simulate(trace, policy, sim_params=sim_params, step_params=params, planner_params=planner)
            item["sim"] = metrics(simulated["requests"], 0.0, args.warmup_s, trace["duration_s"])
            item["sim_moves"] = len(simulated["migrations"])
        report["sim_params"] = sim_params
    report["pilot"] = pilot
    for policy, item in pilot.items():
        live, sim = item["live"], item.get("sim", {})
        print(f"{policy:11s} live CodeTTFT {live.get('code_ttft_avg', float('nan')):7.1f} ConvTPOT "
              f"{1e3 * live.get('conv_tpot_avg', float('nan')):6.2f}ms moves {item['summary']['moves'] if item['summary'] else '?'}"
              f" | sim CodeTTFT {sim.get('code_ttft_avg', float('nan')):7.1f} ConvTPOT "
              f"{1e3 * sim.get('conv_tpot_avg', float('nan')):6.2f}ms moves {item.get('sim_moves')}")
    if args.output:
        args.output.write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
