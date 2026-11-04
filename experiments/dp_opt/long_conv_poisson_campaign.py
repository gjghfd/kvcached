import argparse
import json
import shlex
import time
from datetime import datetime
from pathlib import Path

from agentic_trace import generate_mixed_trace
from amortized_campaign import check_frozen_sources
from archive_trace_job import verify_raw
from long_conv_open_loop import CODE_RATE, CONV_RATES, analyze, digest, validate_upload
from long_conv_open_loop_campaign import SYNC_FILES as PREVIOUS_SYNC_FILES
from rebalance_campaign import checked_ssh, protected_sync
from stress_campaign import archive_job, check_pause, poll_job, save_state
from stress_trace import long_output_marks
from verify_trace import request_metrics, verify


SEEDS = (2026103002, 2026103003)
POLICIES = ("no_move", "own_model", "kv_only", "no_move_fair", "random_no_return", "kv_only_headroom", "kv_only_workaware")
DURATION_S = 7200
WARMUP_S = 600
GROUPS = (("Code", 0), ("Conv", 1), ("combined", None))
SYNC_FILES = tuple(dict.fromkeys((*PREVIOUS_SYNC_FILES,
    "experiments/dp_opt/long_conv_poisson_campaign.py", "experiments/dp_opt/long-conv-poisson.sbatch",
    "tests/test_dp_opt_long_conv_poisson_campaign.py", "experiments/dp_opt/docs/long-conv-poisson-evaluation.md")))


def parent_ready(state):
    return (state.get("GPU_calibration_complete") is True and state.get("state_last_observed") == "COMPLETED"
        and state.get("phase") == "open_loop_calibration_SHA_verified_not_final_benefit")


def await_calibration(root, parent, state):
    while True:
        check_pause(root)
        check_pause(parent)
        check_frozen_sources(Path(state["repo"]), state)
        previous = json.loads((parent / "campaign-state.json").read_text())
        if "diagnosis_required" in previous["phase"]:
            raise RuntimeError("Calibration failure requires diagnosis, not another GPU allocation")
        state.update({"parent_job": previous.get("job"), "parent_phase": previous["phase"],
            "parent_observed_at": datetime.now().astimezone().isoformat(),
            "parent_last_poll_at": previous.get("last_poll_at"), "SSH_queries_while_waiting": False})
        save_state(root, state)
        if parent_ready(previous):
            check_frozen_sources(Path(state["repo"]), previous)
            raw = parent / f"live-trace-{previous['job']}"
            verify_raw(raw, (parent / f"raw-{previous['job']}.sha256").read_text())
            validate_upload(raw / "frozen", Path(previous["inputs"]))
            recalculated = analyze(raw)
            if recalculated != json.loads((parent / "calibration-analysis.json").read_text()):
                raise RuntimeError("Calibration analysis differs from independent raw reconstruction")
            candidate = recalculated["candidate_case"]
            if candidate is None or candidate != previous.get("candidate_case"):
                raise RuntimeError("No independently qualifying rate; preserve all cases without retuning or reroll")
            validation = json.loads((parent / "independent-verification.json").read_text())
            if set(validation) != {"rate-0", "rate-1", "rate-2"}:
                raise RuntimeError("Calibration requires independent verification of every declared rate")
            return previous, recalculated["cases"][candidate]["rate_rps"][1]
        print(json.dumps({"parent_job": previous.get("job"), "phase": previous["phase"],
                          "next": "wait_independent_calibration_before_freezing_any_evaluation_trace"}), flush=True)
        time.sleep(300)


def make_trace(inputs, conv_rate, seed):
    if conv_rate not in CONV_RATES or seed not in SEEDS:
        raise ValueError("Require a predeclared independently calibrated rate and evaluation seed")
    code = json.loads((inputs / "code-marks.json").read_text())
    conv = json.loads((inputs / "conv-marks.json").read_text())
    if any(mark["prompt_tokens"] < 100000 or mark["output_tokens"] != 64 for mark in code):
        raise ValueError("Evaluation Code retains real >=100K/O64 lengths")
    trace = generate_mixed_trace(code, long_output_marks(conv, 24576), CODE_RATE, conv_rate,
        DURATION_S, seed, max_model_len=131072, allow_empty_tenants=True)
    trace.update({"prompt_bank_sha256": digest(inputs / "prompt-bank.json"),
        "synthetic_Conv_output_tokens": 24576, "evaluation_start_s": WARMUP_S,
        "evaluation_duration_s": DURATION_S - WARMUP_S,
        "sampling": "verified_SWE_smith_history_Azure_input_explicit_synthetic_Conv_output",
        "warmup": "natural_Poisson_no_bootstrap_all_requests_recorded_and_drained",
        "scope": "first_two_seed_measured_rate_multi_policy_Poisson_not_P99_significance"})
    return trace


def prepare(root, inputs, conv_rate, parent_job):
    frozen = root / "frozen"
    if frozen.exists():
        raise RuntimeError("Preserve prior evaluation trace; never reroll or overwrite")
    traces = {f"evaluation-{seed}.json": make_trace(inputs, conv_rate, seed) for seed in SEEDS}
    frozen.mkdir(parents=True)
    for name, trace in traces.items():
        (frozen / name).write_text(json.dumps(trace, indent=2) + "\n")
    (frozen / "trace.json").write_bytes((frozen / f"evaluation-{SEEDS[0]}.json").read_bytes())
    config = {"schema_version": 1, "seeds": list(SEEDS), "policies": list(POLICIES),
        "model_rates_rps": [CODE_RATE, conv_rate], "duration_s": DURATION_S, "warmup_s": WARMUP_S,
        "fairness_threshold_s": 30, "budget_mib": 10240, "allow_empty_tenants": True,
        "parent_calibration_job": parent_job,
        "input_sha256": {name: digest(inputs / name) for name in
                         ("code-marks.json", "conv-marks.json", "prompt-bank.json")},
        "trace_sha256": {name: digest(frozen / name) for name in traces},
        "scope": "two_seed_first_Poisson_benefit_trace_not_tail_significance_or_generalization",
        "research_goal_complete": False}
    (frozen / "evaluation-config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def validate_evaluation(frozen, inputs, conv_rate):
    config = json.loads((frozen / "evaluation-config.json").read_text())
    if (config.get("schema_version") != 1 or config.get("seeds") != list(SEEDS)
            or config.get("policies") != list(POLICIES) or config.get("model_rates_rps") != [CODE_RATE, conv_rate]
            or config.get("duration_s") != DURATION_S or config.get("warmup_s") != WARMUP_S
            or config.get("fairness_threshold_s") != 30 or config.get("budget_mib") != 10240
            or config.get("allow_empty_tenants") is not True):
        raise ValueError("Evaluation controls differ from the predeclared measured-rate protocol")
    if set(config["input_sha256"]) != {"code-marks.json", "conv-marks.json", "prompt-bank.json"}:
        raise ValueError("Freeze every original evaluation input bank")
    for name, expected in config["input_sha256"].items():
        if digest(inputs / name) != expected:
            raise ValueError("Evaluation input bank changed")
    if set(config["trace_sha256"]) != {f"evaluation-{seed}.json" for seed in SEEDS}:
        raise ValueError("Retain both predeclared Poisson draws")
    for seed in SEEDS:
        name = f"evaluation-{seed}.json"
        if (digest(frozen / name) != config["trace_sha256"][name]
                or json.loads((frozen / name).read_text()) != make_trace(inputs, conv_rate, seed)):
            raise ValueError("Poisson evaluation draw cannot be independently rebuilt")
    if (frozen / "trace.json").read_bytes() != (frozen / f"evaluation-{SEEDS[0]}.json").read_bytes():
        raise ValueError("Evaluation default trace pointer changed")
    return config


def pooled_metrics(raw):
    records = {policy: [] for policy in POLICIES}
    for seed in SEEDS:
        phase = raw / f"evaluation-{seed}"
        trace = json.loads((phase / "trace.json").read_text())
        expected = {mark["request_id"] for mark in trace["requests"]}
        for policy in POLICIES:
            case = phase / f"r0-{policy}"
            workload = json.loads((case / "workload.json").read_text())
            origin = workload["origin"]
            requests = json.loads((case / "requests.json").read_text())
            if len(requests) != len(expected) or {request["request_id"] for request in requests} != expected:
                raise ValueError("Every policy must drain all offered requests, including warmup")
            for request in requests:
                records[policy].append({**request, **{field: request[field] - origin for field in
                    ("arrived_at", "submitted_at", "reserved_at", "first_token_at", "completed_at")}})
    fields = ("mean_ttft_s", "p95_ttft_s", "p99_ttft_s", "mean_tpot_s", "completed_requests", "output_tokens")
    result = {}
    for policy, values in records.items():
        result[policy] = {}
        for group, model in GROUPS:
            selected = [request for request in values if model is None or request["model"] == model]
            all_metrics = request_metrics(selected, 0, DURATION_S * len(SEEDS), 120)
            measured = request_metrics([request for request in selected if WARMUP_S <= request["arrived_at"]
                < DURATION_S], WARMUP_S, (DURATION_S - WARMUP_S) * len(SEEDS), 120)
            result[policy][group] = {"all_offered": {field: all_metrics[field] for field in fields},
                "measurement_arrivals": {field: measured[field] for field in fields}}
    return result


def markdown(job, metrics):
    lines = ["# Long-output Poisson policy comparison", "",
        f"Job{job}: two independent7200s draws; same traces and budgets across seven policies.",
        "All offered requests remain included; measurement-arrival subset follows600s natural warmup.",
        "Code/Conv/combined TTFT seconds (avg/P95/P99), Conv avg TPOT milliseconds.",
        "Report request counts and empirical quantiles, not P99 significance or causal return TTFT proof.", ""]
    for scope in ("all_offered", "measurement_arrivals"):
        lines.extend([f"## {scope}", "", "| Policy | Code TTFT | Conv TTFT | Combined TTFT | Conv TPOT | Counts Code/Conv |",
                      "|---|---|---|---|---:|---|"])
        for policy in POLICIES:
            groups = {name: metrics[policy][name][scope] for name, _ in GROUPS}
            cells = ["/".join("N/A" if groups[name][field] is None else f"{groups[name][field]:.4f}"
                for field in ("mean_ttft_s", "p95_ttft_s", "p99_ttft_s")) for name, _ in GROUPS]
            tpot = groups["Conv"]["mean_tpot_s"]
            lines.append("|" + "|".join([policy, *cells, "N/A" if tpot is None else f"{tpot * 1000:.4f}",
                f"{groups['Code']['completed_requests']}/{groups['Conv']['completed_requests']}"]) + "|")
        lines.append("")
    lines.extend(["no_move is the same full-reservation custom admission, not unmodified native vLLM.",
        "no_move_fair blocks cross-tenant admissions after30s queue age; random migrates without return.",
        "Headroom and robust workaware include real planning, pause, dp_echo copies and old-request costs.",
        "Sparse graphs, BF16/128K/two A800 and10240MiB/GPU remain identical. Negative results stay visible.",
        "Raw phase summaries include ownership, physical bytes, fairness events, return controls and domain rejections.",
        "No seed reroll or posthoc request deletion. Further causal repayment and sensitivity review are required",
        "before declaring the total research goal complete.", ""])
    return "\n".join(lines)


def comparisons(metrics):
    treatment = metrics["kv_only_workaware"]
    fields = ("mean_ttft_s", "p95_ttft_s", "p99_ttft_s", "mean_tpot_s")
    return {baseline: {group: {scope: {field: (
        1 - treatment[group][scope][field] / metrics[baseline][group][scope][field]
        if metrics[baseline][group][scope][field] not in (None, 0)
        and treatment[group][scope][field] is not None else None) for field in fields}
        for scope in ("all_offered", "measurement_arrivals")} for group, _ in GROUPS}
        for baseline in POLICIES if baseline != "kv_only_workaware"}


def submission_command(environment, ledger):
    exports = "; ".join(f"export {name}={shlex.quote(value)}" for name, value in sorted(environment.items()))
    return ("set -eu; test -z \"$(squeue -h -u \"$USER\")\"; "
        f"test ! -e {shlex.quote(ledger)}; {exports}; "
        "sbatch --parsable -o \"$HOME/run/kvcached-paracloud/logs/long-poisson-%j.out\" "
        "-e \"$HOME/run/kvcached-paracloud/logs/long-poisson-%j.err\" "
        "\"$HOME/kvcached/kvcached/experiments/dp_opt/long-conv-poisson.sbatch\" | "
        f"tee {shlex.quote(ledger)}")


def finish(root, state):
    repo = Path(state["repo"])
    terminal = poll_job(root, state["job"], state)
    archive_job(root, state["job"], repo / "experiments/dp_opt", state)
    check_frozen_sources(repo, state)
    raw = root / f"live-trace-{state['job']}"
    if terminal != "COMPLETED" or (raw / "exit-status").read_text().strip() != "0":
        raise RuntimeError("Poisson allocation failed; diagnose archived evidence without hiding failed requests")
    validation = {}
    for seed in SEEDS:
        name = f"evaluation-{seed}"
        if digest(raw / name / "trace.json") != state["configuration"]["trace_sha256"][name + ".json"]:
            raise RuntimeError("Poisson serving changed frozen requests or timing")
        validation[name] = verify(raw / name, Path(state["inputs"]) / "prompt-bank.json")
    (root / "independent-verification.json").write_text(json.dumps(validation, indent=2) + "\n")
    metrics = pooled_metrics(raw)
    with (root / "pooled-metrics.json").open("x") as output:
        json.dump(metrics, output, indent=2)
        output.write("\n")
    with (root / "pooled-comparisons.json").open("x") as output:
        json.dump(comparisons(metrics), output, indent=2)
        output.write("\n")
    report = repo / "experiments/dp_opt/reports/long-conv-poisson-results.md"
    with report.open("x") as output:
        output.write(markdown(state["job"], metrics))
    state.update({"phase": "two_seed_Poisson_SHA_verified_requires_causal_and_sensitivity_review",
        "GPU_evaluation_complete": True, "report": str(report), "total_research_goal_completed": False,
        "next_action": "review_all_baselines_tail_TTFT_Conv_TPOT_return_costs_domain_and_sensitivity"})
    save_state(root, state)


def run(root, parent):
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    state_path = root / "campaign-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "repo": str(repo), "parent_root": str(parent), "phase": "awaiting_independent_rate_calibration",
        "frozen_runtime_sha256": {name: digest(repo / name) for name in SYNC_FILES},
        "GPU_evaluation_complete": False, "total_research_goal_completed": False, "publication_authorized": False}
    if state["repo"] != str(repo) or state["parent_root"] != str(parent):
        raise RuntimeError("Preserve existing Poisson campaign identity")
    if state.get("GPU_evaluation_complete"):
        return
    if "diagnosis_required" in state["phase"]:
        raise RuntimeError("Diagnose previous failure before any retry")
    try:
        check_pause(root)
        check_frozen_sources(repo, state)
        if "job" not in state:
            if state["phase"] != "awaiting_independent_rate_calibration":
                raise RuntimeError("Reconcile existing submission intent and ledger; never duplicate allocation")
            previous, conv_rate = await_calibration(root, parent, state)
            inputs = Path(previous["inputs"])
            config = prepare(root, inputs, conv_rate, previous["job"])
            validate_evaluation(root / "frozen", inputs, conv_rate)
            state.update({"inputs": str(inputs), "configuration": config,
                          "protected_vllm_sha256": previous["protected_vllm_sha256"]})
            save_state(root, state)
            stage, home = protected_sync(repo, root, state, files=SYNC_FILES, prefix="long-conv-poisson",
                upload_files=("frozen/evaluation-config.json", *(f"frozen/evaluation-{seed}.json" for seed in SEEDS)))
            old_environment = previous["frozen_environment"]
            environment = {"DP_OPT_POISSON_UPLOAD": stage + "/upload", "DP_OPT_POISSON_CONV_RATE": str(conv_rate),
                "DP_OPT_POISSON_FIT_JOB": old_environment["DP_OPT_OPEN_LOOP_FIT_JOB"],
                "DP_OPT_POISSON_FIT_SHA256": old_environment["DP_OPT_OPEN_LOOP_FIT_SHA256"],
                "DP_OPT_MOVE_FIXED_S": "1.7743364535272121", "DP_OPT_MOVE_BANDWIDTH_GIB_S": "11.64086282917902"}
            ledger = stage + "/submitted-job.txt"
            state.update({"phase": "submission_intent_do_not_duplicate", "remote_sync_stage": stage,
                          "submission_ledger": ledger, "frozen_environment": environment})
            save_state(root, state)
            output = checked_ssh(submission_command(environment, ledger))
            state.update({"job": int(output.split(";")[0]), "phase": "submitted_seven_policy_Poisson_evaluation"})
            (root / "submit.txt").write_text(output + "\n")
            save_state(root, state)
            remote = home + f"/run/kvcached-paracloud/state/live-trace-{state['job']}"
            watcher = checked_ssh(f'''set -eu
mkdir -p {shlex.quote(remote)}
nohup bash "$HOME/kvcached/kvcached/experiments/dp_opt/watch-trace-remote.sh" {state['job']} \\
    >> {shlex.quote(remote + '/remote-query.log')} 2>&1 < /dev/null &
printf '%s\\n' "$!"''')
            state["remote_watcher_pid"] = int(watcher)
            save_state(root, state)
        finish(root, state)
    except Exception as error:
        state.update({"phase": "diagnosis_required_no_automatic_duplicate_submission", "error": repr(error)})
        save_state(root, state)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--parent-root", type=Path)
    parser.add_argument("--validate-upload", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--conv-rate", type=float)
    args = parser.parse_args()
    if args.validate_upload:
        if args.inputs is None or args.conv_rate is None or args.root is not None or args.parent_root is not None:
            parser.error("Validation requires inputs and calibrated Conv rate, not a campaign root")
        print(json.dumps(validate_evaluation(args.validate_upload, args.inputs, args.conv_rate), indent=2))
        return
    if args.root is None or args.parent_root is None:
        parser.error("Campaign requires root and parent-root")
    run(args.root.resolve(), args.parent_root.resolve())


if __name__ == "__main__":
    main()
