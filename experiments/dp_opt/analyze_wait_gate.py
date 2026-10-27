import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


def analyze(root):
    config = json.loads((root / "config.json").read_text())
    trace = json.loads((root / "trace.json").read_text())
    directory = root / "r0-ilp"
    requests = json.loads((directory / "requests.json").read_text())
    by_id = {request["request_id"]: request for request in requests}
    decisions = json.loads((directory / "decisions.json").read_text())
    gates = []
    for decision in decisions:
        if decision.get("wait_tradeoff") is None:
            continue
        checks = decision["candidate_revalidations"]
        first, final = checks[0], checks[-1]
        population = first["active_plus_incoming"]
        snapshot, candidate = first["snapshot"], first["plan"]["winner"]
        counts = [list(values) for values in snapshot["counts"]]
        counts[0][candidate["target"]] += 1
        baseline_cost = sum(table[counts[model][0]] for model, table in enumerate(snapshot["performance_cost_tables_s"]))
        future = (population * decision["future_transaction_fixed_s"] + candidate["migration_cost_s"]
                  + max(0, candidate["performance_cost_s"] - baseline_cost))
        before_gate_cost = future + decision["preplan_s"] + config["wait_incoming_handshake_s"]
        now = max(observation["observed_at"] for ranks in final["observations"] for observation in ranks)
        mark = by_id[decision["request_id"]]
        competing = [request for request in requests if request["model"] != mark["model"]
                     and now < request["reserved_at"] < mark["reserved_at"]]
        geometry = final["observations"][1 - mark["model"]][0]
        gates.append({"request_id": decision["request_id"], "current_population": population,
                      "performance_scope": snapshot["performance_cost_scope_by_model"],
                      "pre_pause_candidate_cost_s": before_gate_cost,
                      "pre_pause_wait_threshold_s": first["wait_gate"]["estimated_wait_s"] * config["wait_safety_fraction"],
                      "would_reject_before_pause": before_gate_cost >= first["wait_gate"]["estimated_wait_s"] * config["wait_safety_fraction"],
                      "after_boundary_tradeoff": decision["wait_tradeoff"],
                      "predicted_no_future_arrivals_wait_s": final["wait_gate"]["estimated_wait_s"],
                      "actual_later_reservation_wait_s": max(0, mark["reserved_at"] - now),
                      "future_competing_admissions": len(competing),
                      "future_competing_reserved_gib": sum(
                          ((request["prompt_tokens"] + len(request["token_ids"]) + geometry["block_size"] - 1)
                           // geometry["block_size"]) * geometry["block_bytes"] / 1024**3 for request in competing),
                      "actual_pause_call_s": decision["pause_call_s"]})
    summary = {}
    for policy in config["policies"]:
        trial = root / f"r0-{policy}"
        result = json.loads((trial / "summary.json").read_text())
        records = json.loads((trial / "requests.json").read_text())
        summary[policy] = {key: result[key] for key in ("mean_queue_s", "mean_latency_s", "deadline_met_requests", "moves")}
        summary[policy]["latency_sum_s"] = sum(request["completed_at"] - request["arrived_at"] for request in records)
        summary[policy]["Code_latency_sum_s"] = sum(request["completed_at"] - request["arrived_at"] for request in records if request["model"] == 0)
    return {"status": "raw_wait_gate_diagnosis", "policies": summary, "rejected_after_boundary": gates,
            "all_detailed_rejections_known_cost_before_pause": all(gate["would_reject_before_pause"] for gate in gates),
            "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                              for name in ("config.json", "trace.json", "r0-ilp/decisions.json", "r0-ilp/requests.json")},
            "scope": "retrospective_labels_only_not_causal_migration_benefit_or_forecast_training"}


def continuity(archive, job):
    local = (archive / f"watcher-{job}-detached.txt").read_text()
    records = re.findall(r"^=== (\S+) actual_gap_s=(\S+) ssh_exit=(\d+) ===", local, re.MULTILINE)
    remote = (archive / f"live-trace-{job}/remote-watch.log").read_text()
    dates = [datetime.fromisoformat(value) for value in re.findall(r"^\d{4}-\d\d-\d\dT[^\n]+", remote, re.MULTILINE)]
    return {"job": job, "local_polls": len(records),
            "local_max_gap_s": max(float(gap) for stamp, gap, code in records if gap != "None"),
            "local_ssh_failed_polls": sum(code != "0" for stamp, gap, code in records),
            "remote_polls": len(dates), "remote_first": dates[0].isoformat(), "remote_last": dates[-1].isoformat(),
            "remote_max_gap_s": max((right - left).total_seconds() for left, right in zip(dates, dates[1:]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job", type=int)
    args = parser.parse_args()
    result = analyze(args.root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.job:
        report = continuity(args.root.parent, args.job)
        (args.root.parent / f"watch-continuity-{args.job}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rejections": len(result["rejected_after_boundary"]),
                      "known_before_pause": result["all_detailed_rejections_known_cost_before_pause"],
                      "policies": result["policies"]}, indent=2))


if __name__ == "__main__":
    main()
