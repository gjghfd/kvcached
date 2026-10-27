import argparse
import difflib
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from analyze_tail_slo import analyze
from amortized_campaign import check_frozen_sources, paired_window
from rebalance_campaign import checked_ssh, protected_sync
from stress_campaign import archive_job, check_pause, poll_job, save_state
from stress_trace import make_stress_trace
from verify_trace import verify
from workaware_evidence import RUNTIME_FILES
from workaware_fixture import make_fixture


PATCH_FILES = ("ilp_reference.py", "live_rebalance.py", "live_trace.py", "verify_rebalance.py", "verify_trace.py", "trace-benefit.sbatch")
TEST_FILES = ("test_dp_opt_workaware_cost.py", "test_dp_opt_workaware_rebalance.py", "test_dp_opt_workaware_ilp.py",
              "test_dp_opt_workaware_pilot.py", "test_dp_opt_workaware_live.py", "test_dp_opt_workaware_campaign.py", "test_dp_opt_ilp_reference.py",
              "test_dp_opt_rebalance.py", "test_dp_opt_trace.py", "test_dp_opt_trace_planning.py",
              "test_dp_opt_headroom_rebalance.py", "test_dp_opt_amortized_rebalance.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integration_patch(repo, staging):
    lines = ["*** Begin Patch"]
    for name in PATCH_FILES:
        relative = f"experiments/dp_opt/{name}"
        before, after = (repo / relative).read_text(), (staging / relative).read_text()
        if before == after:
            continue
        lines.append(f"*** Update File: {relative}")
        difference = list(difflib.unified_diff(before.splitlines(), after.splitlines(), n=3))[2:]
        lines.extend("@@" if line.startswith("@@") else line for line in difference)
    lines.append("*** End Patch")
    return "\n".join(lines) + "\n"


def prepare(root, inputs, conv_rate):
    frozen = root / "frozen"
    frozen.mkdir()
    read = lambda name: json.loads((inputs / name).read_text())
    trace = make_stress_trace(read("code-marks.json"), read("conv-marks.json"), 8192, .0005,
                              conv_rate, 4200, 600, 20261024, digest(inputs / "prompt-bank.json"))
    (frozen / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    (frozen / "protocol.json").write_text(json.dumps({"code_rate_rps": .0005, "conv_rate_rps": conv_rate,
        "warmup_s": 600, "duration_s": 4800, "seed": 20261024, "baseline": "kv_only_headroom",
        "policies": ["no_move", "no_move_fair", "kv_only", "kv_only_headroom", "kv_only_workaware"],
        "Code_samples": sum(mark["model"] == 0 for mark in trace["requests"]),
        "tail_scope": "first_paired_iteration_descriptive_Code_tails_not_multi_seed_P99_confirmation"}, indent=2) + "\n")


def wait_old(root, old_root, supervision, repo, state):
    old = json.loads((old_root / "campaign-state.json").read_text())
    while True:
        check_pause(root)
        check_frozen_sources(repo, old)
        latest = json.loads((old_root / "campaign-state.json").read_text())
        state.update({"phase": "waiting_old_SHA_archive_before_runtime_replacement", "old_job": latest["job"],
                      "old_state_observed": latest.get("state_last_observed"), "old_last_poll": latest.get("last_poll_at")})
        save_state(root, state)
        for name, expected in state["pending_runtime_sha256"].items():
            if name not in {f"experiments/dp_opt/{name}" for name in PATCH_FILES} and digest(repo / name) != expected:
                raise RuntimeError("Prepared new helpers changed while waiting for old immutable experiment")
        side = json.loads((supervision / "status.json").read_text())
        if (latest.get("phase") == "lower_Conv_rate_trace_and_pressure_independently_verified"
                and latest.get("GPU_experiment_complete") and side.get("supervision_complete")):
            if side.get("terminal_audit_status") != "passed":
                raise RuntimeError("Old independent terminal audit did not pass")
            verify(old_root / f"live-trace-{latest['job']}", Path(state["inputs"]) / "prompt-bank.json")
            break
        if latest.get("state_last_observed") in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"):
            raise RuntimeError("Old job failed; diagnose its archived evidence before starting new algorithm")
        time.sleep(300)


def install(root, repo, staging, state):
    manifest = json.loads((root / "staged-manifest.json").read_text())
    for relative, expected in manifest.items():
        if digest(staging / relative) != expected:
            raise RuntimeError("Validated isolated integration changed before installation")
    backup = root / "old-runtime-backup"
    backup.mkdir()
    for name in PATCH_FILES:
        (backup / name).write_bytes((repo / "experiments/dp_opt" / name).read_bytes())
    patch = integration_patch(repo, staging)
    (root / "applied-integration.patch").write_text(patch)
    subprocess.run(["apply_patch", patch], cwd=repo, check=True)
    for name, expected in state["pending_runtime_sha256"].items():
        if digest(repo / name) != expected:
            raise RuntimeError("Installed source differs from validated pending runtime")
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", *(f"tests/{name}" for name in TEST_FILES)],
                            cwd=repo, text=True, capture_output=True)
    (root / "installed-regression.txt").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("Installed algorithm regression failed; no synchronization or new GPU submission")
    state["phase"] = "installed_CPU_verified_ready_for_protected_sync"
    save_state(root, state)


def submit(root, repo, state, conv_rate):
    files = list(dict.fromkeys([*(f"experiments/dp_opt/{name}" for name in RUNTIME_FILES),
                               *(f"experiments/dp_opt/{name}" for name in PATCH_FILES),
                               *(f"tests/{name}" for name in TEST_FILES), "experiments/dp_opt/workaware_campaign.py",
                               "experiments/dp_opt/docs/workaware-return-protocol.md", "experiments/dp_opt/README.md"]))
    stage, home = protected_sync(repo, root, state, files=files, prefix="workaware")
    trace = stage + "/upload/frozen/trace.json"
    ledger = stage + "/submitted-job.txt"
    environment = {"DP_OPT_WORKAWARE_TRACE_PATH": trace, "DP_OPT_WORKAWARE_TRACE_SHA256": digest(root / "frozen/trace.json"),
        "DP_OPT_WORKAWARE_CONV_RATE": str(conv_rate), "DP_OPT_MOVE_FIXED_S": "1.7743364535272121",
        "DP_OPT_MOVE_BANDWIDTH_GIB_S": "11.64086282917902"}
    exports = "; ".join(f"export {name}={shlex.quote(value)}" for name, value in environment.items())
    command = ('set -eu; test -z "$(squeue -h -u "$USER")"; '
        'test "$(git -C "$HOME/kvcached/kvcached" branch --show-current)" = main; '
        'test "$(git -C "$HOME/kvcached/kvcached" remote get-url origin)" = git@github.com:gjghfd/kvcached.git; '
        f'test ! -e {shlex.quote(ledger)}; ' + exports + '; '
        f'sbatch --parsable --export=ALL --output="$HOME/run/kvcached-paracloud/logs/live-trace-%j.out" '
        '--error="$HOME/run/kvcached-paracloud/logs/live-trace-%j.err" '
        f'"$HOME/kvcached/kvcached/experiments/dp_opt/workaware-return.sbatch" > {shlex.quote(ledger)}; '
        f'cat {shlex.quote(ledger)}')
    state.update({"phase": "submission_intent_do_not_duplicate", "submission_ledger": ledger,
                  "frozen_runtime_sha256": {name: digest(repo / name) for name in files}, "frozen_environment": environment})
    save_state(root, state)
    output = checked_ssh(command)
    job = int(output.split(";")[0])
    (root / "submit.txt").write_text(output + "\n")
    state.update({"job": job, "phase": "submitted_workaware_calibration_and_paired_experiments"})
    save_state(root, state)
    watcher = checked_ssh(f'''mkdir -p "$HOME/run/kvcached-paracloud/state/live-trace-{job}"
nohup bash "$HOME/kvcached/kvcached/experiments/dp_opt/watch-trace-remote.sh" {job} \\
    >> "$HOME/run/kvcached-paracloud/state/live-trace-{job}/remote-query.log" 2>&1 < /dev/null &
printf '%s\\n' "$!"''')
    state["remote_watcher_pid"] = int(watcher)
    save_state(root, state)
    return job


def finish(root, repo, inputs, scripts, state):
    job = state["job"]
    archive_job(root, job, scripts, state)
    archived = root / f"live-trace-{job}"
    if (archived / "exit-status").read_text().strip() != "0":
        raise RuntimeError("GPU phase failed; raw evidence archived, diagnose before modifying/retrying")
    from verify_workaware_pilot import verify as verify_cadence

    fit = verify_cadence(archived / "cadence", root / "independent-cadence-fit.json")
    for population in (2, 9):
        expected = make_fixture(json.loads((inputs / "code-marks.json").read_text()),
            json.loads((inputs / "conv-marks.json").read_text()), population, digest(inputs / "prompt-bank.json"))
        actual = json.loads((archived / f"recovery-{population}/trace.json").read_text())
        if actual != expected:
            raise ValueError("Recovery mechanism altered its predeclared input recipe")
        verify(archived / f"recovery-{population}", inputs / "prompt-bank.json")
    trace_root = archived / "paired-trace"
    if digest(trace_root / "trace.json") != digest(root / "frozen/trace.json"):
        raise ValueError("Paired GPU trace differs from independently frozen offered workload")
    verification = verify(trace_root, inputs / "prompt-bank.json")
    (root / "independent-trace-verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    comparisons = {baseline: analyze(trace_root, baseline, "kv_only_workaware") for baseline in
                   ("no_move", "no_move_fair", "kv_only", "kv_only_headroom")}
    (root / "all-request-comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    windows = {baseline: paired_window(trace_root, baseline, "kv_only_workaware", 600, 4800) for baseline in comparisons}
    (root / "evaluation-window-comparisons.json").write_text(json.dumps(windows, indent=2) + "\n")
    state.update({"phase": "first_workaware_paired_trace_SHA_verified_analyzed", "GPU_experiment_complete": True,
                  "total_research_goal_completed": False, "cadence_coefficients": fit["coefficients"],
                  "next_action": "report_both_tenants_TTFT_and_Conv_TPOT_cost_error_then_general_algorithm_iteration"})
    save_state(root, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-campaign", type=Path, required=True)
    parser.add_argument("--old-supervision", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--conv-rate", type=float, default=.01956673845194241)
    parser.add_argument("--after-install", action="store_true")
    args = parser.parse_args()
    root, scripts = args.root.resolve(), Path(__file__).resolve().parent
    repo, inputs = scripts.parents[1], args.inputs.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "campaign-state.json").exists() and not args.after_install:
        raise RuntimeError("Existing campaign must be audited/resumed explicitly, never submitted twice")
    state = (json.loads((root / "campaign-state.json").read_text()) if args.after_install else
        {"phase": "prepared_waiting_for_old_runtime", "GPU_experiment_complete": False,
         "total_research_goal_completed": False, "publication_authorized": False, "inputs": str(inputs)})
    try:
        if not args.after_install:
            prepare(root, inputs, args.conv_rate)
            manifest = {f"experiments/dp_opt/{name}": digest(args.staging / "experiments/dp_opt" / name) for name in PATCH_FILES}
            (root / "staged-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            state["pending_runtime_sha256"] = {
                **{f"experiments/dp_opt/{name}": digest(scripts / name) for name in RUNTIME_FILES},
                **{f"tests/{name}": digest(repo / "tests" / name) for name in TEST_FILES},
                "experiments/dp_opt/workaware_campaign.py": digest(Path(__file__)), **manifest}
            old_state = json.loads((args.old_campaign / "campaign-state.json").read_text())
            state["protected_vllm_sha256"] = old_state["protected_vllm_sha256"]
            save_state(root, state)
            wait_old(root, args.old_campaign, args.old_supervision, repo, state)
            install(root, repo, args.staging, state)
            os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--after-install"])
        if state["phase"] != "installed_CPU_verified_ready_for_protected_sync" or state.get("job"):
            raise RuntimeError("Invalid restart/submission state; refuse duplicate allocation")
        job = submit(root, repo, state, args.conv_rate)
        poll_job(root, job, state)
        check_frozen_sources(repo, state)
        finish(root, repo, inputs, scripts, state)
    except Exception as error:
        state.update({"phase": "diagnosis_required_no_automatic_duplicate_submission", "error": repr(error)})
        save_state(root, state)
        raise


if __name__ == "__main__":
    main()
