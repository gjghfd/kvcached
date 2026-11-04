import argparse
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path

from amortized_campaign import check_frozen_sources
from archive_trace_job import verify_raw
from stress_campaign import check_pause
from verify_trace import request_metrics


POLICIES = ("random_no_return", "kv_only_workaware")
GROUPS = (("Code", 0), ("Conv", 1), ("combined", None))
PHASES = ("native-20261030", "native-20261031")


def ready(state):
    return (state.get("GPU_experiment_complete") is True and state.get("state_last_observed") == "COMPLETED"
            and state.get("phase") == "robust_native_pairs_SHA_verified_analyzed")


def summarize(raw, validation):
    records = {policy: [] for policy in POLICIES}
    duration = 0
    for name in PHASES:
        phase = raw / name
        trace = json.loads((phase / "trace.json").read_text())
        if trace["duration_s"] != 1200:
            raise ValueError("Native report requires the declared 1200-second offered window")
        offered = {request["request_id"]: request for request in trace["requests"]}
        if len(offered) != 3 or len(offered) != len(trace["requests"]):
            raise ValueError("Native report requires all three unique offered requests per seed")
        token_reference = None
        for policy in POLICIES:
            case = phase / f"r0-{policy}"
            workload = json.loads((case / "workload.json").read_text())
            origin = workload["origin"]
            requests = json.loads((case / "requests.json").read_text())
            if (workload["duration_s"] != trace["duration_s"] or len(requests) != len(offered)
                    or {request["request_id"] for request in requests} != set(offered)):
                raise ValueError("Native report must include every offered request without duplicates")
            tokens = {request["request_id"]: request["token_ids"] for request in requests}
            if token_reference is not None and tokens != token_reference:
                raise ValueError("Native paired outputs differ")
            token_reference = tokens
            for request in requests:
                source = offered[request["request_id"]]
                times = [request[field] for field in
                         ("arrived_at", "submitted_at", "reserved_at", "first_token_at", "completed_at")]
                if (request["model"] != source["model"] or len(request["token_ids"]) != source["output_tokens"]
                        or not all(math.isfinite(value) for value in times)
                        or times != sorted(times) or not math.isfinite(origin)
                        or abs(request["arrived_at"] - origin - source["arrival_s"]) > .001):
                    raise ValueError("Native request model, tokens, arrival or lifecycle differs")
                records[policy].append({**request, **{field: request[field] - origin for field in
                    ("arrived_at", "submitted_at", "reserved_at", "first_token_at", "completed_at")}})
        duration += trace["duration_s"]
    pooled = {policy: {name: request_metrics([request for request in values
        if model is None or request["model"] == model], 0, duration, 120) for name, model in GROUPS}
        for policy, values in records.items()}
    for groups in pooled.values():
        for metric in groups.values():
            for field in ("makespan_s", "drain_s", "output_tokens_per_complete_drain_s"):
                metric.pop(field)
    reduction = {name: {field: 1 - pooled[POLICIES[1]][name][field] / pooled[POLICIES[0]][name][field]
        for field in ("mean_ttft_s", "p95_ttft_s", "p99_ttft_s", "mean_tpot_s")
        if pooled[POLICIES[0]][name][field] not in (None, 0)} for name, _ in GROUPS}
    return {"pooled": pooled, "workaware_reduction_fractions_vs_random": reduction,
            "phase_return_diagnostics": {name: {field: validation["phases"][name][field]
                for field in ("robust_planned_attempts", "committed_returns", "repayment")}
                for name in PHASES},
            "scope": "two_seed_closed_native_robust_pairs_not_final_Poisson_or_P99_significance",
            "initial_TTFT_is_return_effect": False, "research_goal_complete": False}


def markdown(result, job):
    lines = ["# Native robust release results", "", f"Job{job}: two-seed closed N2 validation, not final Poisson.",
             "TTFT seconds, avg/P95/P99; Conv avg TPOT milliseconds. Report empirical sample counts,",
             "not P99 significance. Initial TTFT precedes return and cannot establish return TTFT benefit.", "",
             "| Policy | Code TTFT | Conv TTFT | Combined TTFT | Conv avg TPOT |",
             "|---|---|---|---|---:|"]
    for policy, groups in result["pooled"].items():
        cells = ["/".join(f"{groups[name][field]:.4f}" for field in
                 ("mean_ttft_s", "p95_ttft_s", "p99_ttft_s")) for name, _ in GROUPS]
        lines.append("|" + "|".join([policy, *cells, f"{groups['Conv']['mean_tpot_s'] * 1000:.4f}"]) + "|")
    lines.extend(["", "Samples per policy: " + ", ".join(f"{name}={result['pooled'][POLICIES[0]][name]['completed_requests']}"
                  for name, _ in GROUPS), "",
        "All offered requests and pause/planning/copy cost remain included. Negative reductions are regressions.",
        "The random baseline migrates without return; no_move/fairness are not part of this new native pair.",
        "Do not infer improvement over the old point optimizer when plans coincide. Preserve all exclusions",
        "in the independently verified pause-inclusive repayment diagnostics, including unavailable anchors.", ""])
    if "workaware_reduction_fractions_vs_random" in result:
        lines.extend(["Workaware reductions versus random (%; negative means regression):", "",
                      "| Group | Avg TTFT | P95 TTFT | P99 TTFT | Avg TPOT |",
                      "|---|---:|---:|---:|---:|"])
        for name, _ in GROUPS:
            values = result["workaware_reduction_fractions_vs_random"][name]
            lines.append("|" + "|".join([name, *(f"{values[field] * 100:.3f}" for field in
                ("mean_ttft_s", "p95_ttft_s", "p99_ttft_s", "mean_tpot_s"))]) + "|")
        lines.extend(["", "Full per-seed return counts, pause-inclusive repayment and excluded requests:",
                      "`native-pooled-metrics.json` / `phase_return_diagnostics` in this job's evidence root.", ""])
    lines.extend([
        "Next: measured open-loop service/KV/queue-stability calibration, then separately frozen Poisson",
        "evaluation with ordinary/fair/random/headroom/workaware baselines. Rates remain priors and zero-Code",
        "realizations remain valid. Do not reroll seeds, relax uncertainty or declare the research goal complete.", ""])
    return "\n".join(lines)


def record(root, repo):
    state = json.loads((root / "campaign-state.json").read_text())
    if not ready(state):
        raise ValueError("Require complete native archive and independent validation before reporting")
    check_frozen_sources(repo, state)
    raw = root / f"live-trace-{state['job']}"
    manifest = (root / f"raw-{state['job']}.sha256").read_text()
    verify_raw(raw, manifest)
    validation_path = root / "independent-verification.json"
    validation = json.loads(validation_path.read_text())
    if set(validation.get("phases", {})) != set(PHASES):
        raise ValueError("Native report requires both independently verified declared seeds")
    result = {**summarize(raw, validation), "job": state["job"],
        "raw_manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
        "independent_verification_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest()}
    verify_raw(raw, manifest)
    check_frozen_sources(repo, state)
    report = repo / "experiments/dp_opt/reports/workaware-robust-native-results.md"
    output = root / "native-pooled-metrics.json"
    if report.exists() or output.exists():
        raise ValueError("Preserve prior report and pooled metrics; never overwrite native evidence")
    with output.open("x") as target:
        json.dump(result, target, indent=2)
        target.write("\n")
    with report.open("x") as target:
        target.write(markdown(result, state["job"]))
    return {"report": str(report), "pooled_metrics": str(output), "research_goal_complete": False}


def save_status(root, status):
    temporary = root / "report-followup-state.tmp"
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(root / "report-followup-state.json")


def run(root, repo):
    if (root / "report-followup-state.json").exists():
        raise RuntimeError("Preserve existing report consumer status; do not start a duplicate")
    try:
        watch(root, repo)
    except Exception as error:
        save_status(root, {"observed_at": datetime.now().astimezone().isoformat(),
            "status": "diagnosis_required_no_automatic_report_or_GPU_submission", "error": repr(error),
            "SSH_queries": False, "GPU_submissions": False, "research_goal_complete": False})
        raise


def watch(root, repo):
    while True:
        check_pause(root)
        state = json.loads((root / "campaign-state.json").read_text())
        if "diagnosis_required" in state["phase"]:
            raise RuntimeError("Native failure requires diagnosis, not an automatic positive report")
        status = {"observed_at": datetime.now().astimezone().isoformat(), "job": state.get("job"),
                  "main_phase": state["phase"], "last_main_poll_at": state.get("last_poll_at"),
                  "SSH_queries": False, "GPU_submissions": False, "runtime_changes": False}
        if ready(state):
            status.update(record(root, repo))
            status["status"] = "complete_native_report_not_final_Poisson_research_goal"
            save_status(root, status)
            return
        status["status"] = "waiting_for_main_complete_SHA_verified_native_result"
        save_status(root, status)
        print(json.dumps(status), flush=True)
        time.sleep(300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), Path(__file__).resolve().parents[2])


if __name__ == "__main__":
    main()
