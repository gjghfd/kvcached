import argparse
import json
import re
import statistics
from datetime import datetime
from pathlib import Path


def summarize(values):
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values), "count": len(values)}


def analyze(archive, job):
    verification = json.loads((archive / f"transfer-verification-{job}.json").read_text())
    costs = verification["pause_costs"]
    if len(costs) != 36 or verification["baseline_transport"] != "native_keep":
        raise ValueError("Expected independently verified 36-case boundary pilot")
    modes = {}
    fields = ("incoming_control_delay_s", "active_tenant_blocking_upper_s",
              "max_client_token_gap_s", "client_progress_during_handshake_tokens")
    for mode in ("native_keep", "default32_boundary"):
        selected = [row for row in costs if row["case"]["pause_mode"] == mode]
        modes[mode] = {field: summarize([row[field] for row in selected]) for field in fields}
        modes[mode]["by_target_tokens"] = {
            str(target): {field: summarize([row[field] for row in selected
                                            if row["case"]["target_tokens"] == target]) for field in fields}
            for target in (2560, 4608, 8192)}
    pairs = []
    for treatment in costs:
        if treatment["case"]["pause_mode"] != "default32_boundary":
            continue
        baseline = next(row for row in costs if row["case"]["pause_mode"] == "native_keep"
                        and all(row["case"][key] == treatment["case"][key]
                                for key in ("target_tokens", "repeat", "model")))
        pairs.append({"case": treatment["case"], "blocking_reduction_fraction":
                      1 - treatment["active_tenant_blocking_upper_s"] / baseline["active_tenant_blocking_upper_s"],
                      "incoming_delay_delta_s": treatment["incoming_control_delay_s"] - baseline["incoming_control_delay_s"]})
    local = re.findall(r"^=== (\S+) actual_gap_s=([^ ]+) ssh_exit=(\d+) ===",
                       (archive / f"poll-{job}.log").read_text(), re.MULTILINE)
    remote = re.findall(r"^2026-\S+", (archive / f"live-trace-{job}/remote-watch.log").read_text(), re.MULTILINE)
    if len(local) < 2 or len(remote) < 2:
        raise ValueError("Cannot establish actual watcher continuity")
    times = [datetime.fromisoformat(value) for value in remote]
    continuity = {"job": job, "local_polls": len(local),
                  "local_max_gap_s": max(float(row[1]) for row in local if row[1] != "None"),
                  "local_all_ssh_zero": all(row[2] == "0" for row in local),
                  "remote_polls": len(remote), "remote_first": remote[0], "remote_last": remote[-1],
                  "remote_max_gap_s": max((later - earlier).total_seconds()
                                          for earlier, later in zip(times, times[1:]))}
    return {"job": job, "modes": modes, "paired": pairs,
            "scope": "single_active_request_control_cost_not_joint_population_or_trace_net_benefit"}, continuity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=int)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    result, continuity = analyze(args.archive, args.job)
    for name, value in ((f"results-{args.job}-compact.json", result),
                        (f"watch-continuity-{args.job}.json", continuity)):
        (args.archive / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"modes": result["modes"], "continuity": continuity}, indent=2))


if __name__ == "__main__":
    main()
