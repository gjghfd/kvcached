import argparse
import hashlib
import json
import math
import shlex
import sys
from pathlib import Path

from analyze_tail_slo import analyze
from amortized_campaign import check_frozen_sources, paired_window
from rebalance_campaign import checked_ssh, protected_sync, verify_old_failure
from stress_campaign import archive_job, poll_job, save_state
from verify_trace import verify
from verify_workaware_profile import verify_profile
from workaware_campaign import TEST_FILES
from workaware_evidence import PROFILE_RUNTIME_FILES
from workaware_fixture import make_fixture


BASELINES = ("no_move", "no_move_fair", "kv_only", "kv_only_headroom")
EXTRA_TESTS = ("test_dp_opt_workaware_cost_analysis.py", "test_dp_opt_workaware_profile_cost.py",
               "test_dp_opt_workaware_profile_verify.py", "test_dp_opt_workaware_profile_integration.py",
               "test_dp_opt_trace_reference.py", "test_dp_opt_archive_trace_job.py",
               "test_dp_opt_workaware_profile_campaign.py")
SYNC_FILES = tuple(dict.fromkeys((
    *(f"experiments/dp_opt/{name}" for name in PROFILE_RUNTIME_FILES),
    "experiments/dp_opt/workaware-return.sbatch", "experiments/dp_opt/verify_trace.py",
    "experiments/dp_opt/verify_rebalance.py", "experiments/dp_opt/trace-benefit.sbatch",
    "experiments/dp_opt/workaware_profile_campaign.py", "experiments/dp_opt/archive_trace_job.py",
    "experiments/dp_opt/workaware_campaign.py", "experiments/dp_opt/analyze_workaware_cost.py",
    "experiments/dp_opt/README.md", "experiments/dp_opt/docs/workaware-return-protocol.md",
    "experiments/dp_opt/reports/workaware-return-status.md",
    *(f"tests/{name}" for name in (*TEST_FILES, *EXTRA_TESTS)),
)))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_environment(trace, trace_sha256, calibration, conv_rate):
    if not math.isfinite(conv_rate) or conv_rate <= 0:
        raise ValueError("Require positive frozen Conv rate")
    return {"DP_OPT_WORKAWARE_TRACE_PATH": trace, "DP_OPT_WORKAWARE_TRACE_SHA256": trace_sha256,
            "DP_OPT_WORKAWARE_CONV_RATE": str(conv_rate), "DP_OPT_WORKAWARE_CALIBRATION_SOURCE": calibration,
            "DP_OPT_MOVE_FIXED_S": "1.7743364535272121", "DP_OPT_MOVE_BANDWIDTH_GIB_S": "11.64086282917902"}


def submission_command(environment, ledger):
    exports = "; ".join(f"export {name}={shlex.quote(value)}" for name, value in environment.items())
    return ('set -eu; test -z "$(squeue -h -u "$USER")"; '
            'test "$(git -C "$HOME/kvcached/kvcached" branch --show-current)" = main; '
            'test "$(git -C "$HOME/kvcached/kvcached" remote get-url origin)" = git@github.com:gjghfd/kvcached.git; '
            f'test ! -e {shlex.quote(ledger)}; ' + exports + '; '
            'sbatch --parsable --export=ALL --output="$HOME/run/kvcached-paracloud/logs/live-trace-%j.out" '
            '--error="$HOME/run/kvcached-paracloud/logs/live-trace-%j.err" '
            f'"$HOME/kvcached/kvcached/experiments/dp_opt/workaware-profile-return.sbatch" > {shlex.quote(ledger)}; '
            f'cat {shlex.quote(ledger)}')


def validate_frozen_trace(old_root, conv_rate):
    protocol = json.loads((old_root / "frozen/protocol.json").read_text())
    if (protocol["conv_rate_rps"] != conv_rate or protocol["code_rate_rps"] != .0005
            or protocol["seed"] != 20261024 or protocol["duration_s"] != 4800 or protocol["warmup_s"] != 600):
        raise ValueError("Reuse the declared paired trace without rate or mark changes")
    return protocol


def summarize_primary(comparisons):
    result = {}
    for baseline, comparison in comparisons.items():
        selected = {}
        for tenant, source in (("code", "Code"), ("conv", "Conv"), ("combined", "all_requests")):
            group = comparison["groups"][source]
            selected[tenant] = {"requests": group["requests"],
                "baseline_ttft": group["baseline"]["metrics"]["ttft_s"],
                "treatment_ttft": group["treatment"]["metrics"]["ttft_s"],
                "ttft_changes": group["changes"]["ttft_s"],
                "baseline_request_average_tpot": group["baseline"]["metrics"]["tpot_s"],
                "treatment_request_average_tpot": group["treatment"]["metrics"]["tpot_s"]}
        result[baseline] = selected
    return result


def finish(root, repo, inputs, state):
    archive_job(root, state["job"], repo / "experiments/dp_opt", state)
    raw = root / f"live-trace-{state['job']}"
    phase_checks = {}
    for population in (2, 9):
        phase = raw / f"recovery-{population}"
        if not (phase / "exit-status").exists() or (phase / "exit-status").read_text().strip() != "0":
            continue
        expected = make_fixture(json.loads((inputs / "code-marks.json").read_text()),
            json.loads((inputs / "conv-marks.json").read_text()), population, digest(inputs / "prompt-bank.json"))
        if json.loads((phase / "trace.json").read_text()) != expected:
            raise ValueError("Mechanism changed its predeclared input recipe")
        phase_checks[str(population)] = verify(phase, inputs / "prompt-bank.json")
    (root / "independent-mechanism-verification.json").write_text(json.dumps(phase_checks, indent=2) + "\n")
    if (raw / "exit-status").read_text().strip() != "0":
        raise RuntimeError("Job failed; completed phases independently verified, raw preserved, diagnose before retry")
    phase = raw / "paired-trace"
    if digest(phase / "trace.json") != state["trace_sha256"]:
        raise ValueError("Formal workload differs from the frozen paired trace")
    verification = verify(phase, inputs / "prompt-bank.json")
    (root / "independent-trace-verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    comparisons = {baseline: analyze(phase, baseline, "kv_only_workaware") for baseline in BASELINES}
    windows = {baseline: paired_window(phase, baseline, "kv_only_workaware", 600, 4800) for baseline in BASELINES}
    (root / "all-request-comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    (root / "evaluation-window-comparisons.json").write_text(json.dumps(windows, indent=2) + "\n")
    (root / "primary-metrics.json").write_text(json.dumps(summarize_primary(comparisons), indent=2) + "\n")
    (root / "evaluation-window-primary-metrics.json").write_text(json.dumps(summarize_primary(windows), indent=2) + "\n")
    state.update({"phase": "profile_return_trace_SHA_verified_analyzed", "GPU_experiment_complete": True,
                  "total_research_goal_completed": False,
                  "next_action": "compare_both_tenant_avg_P95_P99_TTFT_and_Conv_TPOT_then_general_iteration",
                  "statistical_scope": "single_paired_trace_two_Code_marks_descriptive_not_tail_significance"})
    save_state(root, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--old-diagnosis", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--conv-rate", type=float, default=.01956673845194241)
    args = parser.parse_args()
    root, old_root, inputs = args.root.resolve(), args.old_root.resolve(), args.inputs.resolve()
    repo = Path(__file__).resolve().parents[2]
    root.mkdir(parents=True, exist_ok=True)
    if (root / "campaign-state.json").exists():
        raise RuntimeError("Preserve existing submission ledger; reconcile instead of submitting twice")
    state = {"phase": "validating_archived_failure_and_completed_calibration",
             "GPU_experiment_complete": False, "total_research_goal_completed": False,
             "publication_authorized": False, "old_job": 256440}
    save_state(root, state)
    try:
        verify_old_failure(old_root, 256440, "FAILED", args.old_diagnosis)
        protocol = validate_frozen_trace(old_root, args.conv_rate)
        fit = verify_profile(old_root / "live-trace-256440/cadence")
        (root / "independent-reused-profile-fit.json").write_text(json.dumps(fit, indent=2) + "\n")
        frozen = root / "frozen"
        frozen.mkdir()
        (frozen / "trace.json").write_bytes((old_root / "frozen/trace.json").read_bytes())
        (frozen / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        old_state = json.loads((old_root / "campaign-state.json").read_text())
        state.update({"trace_sha256": digest(frozen / "trace.json"),
                      "protected_vllm_sha256": old_state["protected_vllm_sha256"],
                      "frozen_runtime_sha256": {name: digest(repo / name) for name in SYNC_FILES
                                                 if name.endswith((".py", ".sbatch"))}})
        save_state(root, state)
        stage, home = protected_sync(repo, root, state, files=SYNC_FILES, prefix="workaware-profile")
        environment = frozen_environment(stage + "/upload/frozen/trace.json", state["trace_sha256"],
            home + "/run/kvcached-paracloud/state/live-trace-256440/cadence", args.conv_rate)
        ledger = stage + "/submitted-job.txt"
        state.update({"phase": "submission_intent_do_not_duplicate", "submission_ledger": ledger,
                      "frozen_environment": environment})
        save_state(root, state)
        output = checked_ssh(submission_command(environment, ledger))
        job = int(output.split(";")[0])
        (root / "submit.txt").write_text(output + "\n")
        state.update({"job": job, "phase": "submitted_profile_mechanisms_and_five_policy_trace"})
        save_state(root, state)
        watcher = checked_ssh(f'''mkdir -p "$HOME/run/kvcached-paracloud/state/live-trace-{job}"
nohup bash "$HOME/kvcached/kvcached/experiments/dp_opt/watch-trace-remote.sh" {job} \\
    >> "$HOME/run/kvcached-paracloud/state/live-trace-{job}/remote-query.log" 2>&1 < /dev/null &
printf '%s\\n' "$!"''')
        state["remote_watcher_pid"] = int(watcher)
        save_state(root, state)
        poll_job(root, job, state)
        check_frozen_sources(repo, state)
        finish(root, repo, inputs, state)
    except Exception as error:
        state.update({"phase": "diagnosis_required_no_automatic_duplicate_submission", "error": repr(error)})
        save_state(root, state)
        raise


if __name__ == "__main__":
    main()
