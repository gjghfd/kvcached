import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from amortized_campaign import check_frozen_sources, paired_window
from random_baseline_campaign import POLICIES, SYNC_FILES as PREVIOUS_SYNC_FILES
from rebalance_campaign import checked_ssh, protected_sync
from stress_campaign import archive_job, check_pause, poll_job, save_state
from stress_trace import make_stress_trace
from verify_trace import verify
from verify_workaware_profile import same_payload, verify_profile
from workaware_calibration_plan import screen_plan, supplemental_plan


EXTRA_FILES = ("workaware_calibration_plan.py", "workaware_profile_bundle.py", "workaware_phase_campaign.py",
               "workaware-phase-return.sbatch", "workaware_profile_cost.py", "verify_workaware_pilot.py",
               "verify_workaware_profile.py", "workaware_pilot.py", "rebalance_campaign.py")
EXTRA_TESTS = ("test_dp_opt_release_observation.py", "test_dp_opt_workaware_phase.py",
               "test_dp_opt_queue_fairness.py", "test_dp_opt_workaware_profile_integration.py",
               "test_dp_opt_exact_constraint_reuse.py", "test_dp_opt_anchored_cadence.py", "test_dp_opt_phase_reuse.py")
SYNC_FILES = tuple(dict.fromkeys((*PREVIOUS_SYNC_FILES,
    *("experiments/dp_opt/" + name for name in EXTRA_FILES), *("tests/" + name for name in EXTRA_TESTS),
    "experiments/dp_opt/reports/random-baseline-results.md", "experiments/dp_opt/docs/workaware-phase-calibration-plan.md")))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(config, original, lower, plan):
    if (config["code_rate_rps"] != .00025 or config["duration_s"] != 9600
            or config["seed"] != 20261024 or config["warmup_s"] != 600
            or original["duration_s"] != 4800 or original["model_rates_rps"] != [.0005, config["conv_rate_rps"]]
            or lower["duration_s"] != 9600 or lower["model_rates_rps"] != [.00025, config["conv_rate_rps"]]
            or lower["prompt_bank_sha256"] != original["prompt_bank_sha256"]
            or any(request["prompt_tokens"] < 100000 for request in lower["requests"] if request["model"] == 0)
            or not any(request["model"] == 0 for request in lower["requests"])
            or lower["max_model_len"] != 131072 or plan != supplemental_plan()):
        raise ValueError("Phase config must preserve long inputs and freeze real reduced-rate arrivals")
    if "reuse_source_root" in config:
        if (not re.fullmatch(r"/[A-Za-z0-9/_-]+/run/kvcached-paracloud/state/live-trace-[1-9][0-9]*/calibration-n567",
                             config["reuse_source_root"])
                or not config.get("reuse_source_files_sha256")
                or any(not isinstance(name, str) or name.startswith("/") or ".." in Path(name).parts
                       or not re.fullmatch(r"[0-9a-f]{64}", digest)
                       for name, digest in config["reuse_source_files_sha256"].items())
                or config.get("reuse_fit_data", {}).get("population_profiles", {}).get("status")
                    != "raw_verified_workaware_population_profiles"):
            raise ValueError("Reuse requires frozen relative raw hashes and an independently verified composite")
    return screen_plan(plan)


def reuse_source_files(source):
    from workaware_pilot import SOURCE_FILES
    from workaware_profile_bundle import source_files

    names = source_files(source, {"source_files_sha256": {name: digest(source / name) for name in SOURCE_FILES}})
    return {name: digest(source / name) for name in names}


def validate_reuse_source(source, config):
    if (str(source) != config.get("reuse_source_root")
            or reuse_source_files(source) != config.get("reuse_source_files_sha256")
            or (source / "exit-status").read_text().strip() != "0"):
        raise ValueError("Reused calibration differs from the frozen successful GPU phase")
    return {"status": "frozen_GPU_calibration_raw_reused_not_new_GPU_cases", "files": len(config["reuse_source_files_sha256"])}


def validate_reused_profile(source, config):
    if not same_payload(verify_profile(source), config.get("reuse_fit_data")):
        raise ValueError("Rebuilt reused profile differs from independent local verification")
    return {"status": "reused_composite_rebuilt_and_independently_verified"}


def validate_upload(root, inputs=None):
    config = json.loads((root / "phase-config.json").read_text())
    if set(config["frozen_sha256"]) != {"original-trace.json", "lower-trace.json", "calibration-plan.json"}:
        raise ValueError("Frozen phase upload omits declared inputs")
    for name, expected in config["frozen_sha256"].items():
        if name not in ("original-trace.json", "lower-trace.json", "calibration-plan.json") or digest(root / name) != expected:
            raise ValueError("Frozen phase upload changed")
    original = json.loads((root / "original-trace.json").read_text())
    lower = json.loads((root / "lower-trace.json").read_text())
    plan = json.loads((root / "calibration-plan.json").read_text())
    if inputs is not None:
        for name, expected in config["input_sha256"].items():
            if digest(inputs / name) != expected:
                raise ValueError("Original prompt bank or length marks changed")
        read = lambda name: json.loads((inputs / name).read_text())
        rebuilt = make_stress_trace(read("code-marks.json"), read("conv-marks.json"), 8192, .00025,
            config["conv_rate_rps"], 9000, 600, 20261024, digest(inputs / "prompt-bank.json"))
        if lower != rebuilt:
            raise ValueError("Reduced-rate Poisson trace does not reproduce from frozen source marks")
    return validate_config(config, original, lower, plan)


def prepare(root, previous_root, repo, reuse_source=None, reuse_remote=None, reuse_profile=None):
    if (root / "campaign-state.json").exists():
        raise RuntimeError("Preserve existing phase submission ledger; use explicit resume")
    previous = json.loads((previous_root / "campaign-state.json").read_text())
    if previous["phase"] != "random_baseline_six_policy_SHA_verified_analyzed" or not previous["GPU_experiment_complete"]:
        raise ValueError("Previous six-policy experiment must be terminal and independently verified")
    inputs = Path(previous["inputs"])
    for name, expected in previous["input_sha256"].items():
        if digest(inputs / name) != expected:
            raise ValueError("Original dataset inputs changed")
    original_path = previous_root / "frozen/trace.json"
    if digest(original_path) != previous["trace_sha256"]:
        raise ValueError("Previous frozen trace changed")
    original = json.loads(original_path.read_text())
    read = lambda name: json.loads((inputs / name).read_text())
    lower = make_stress_trace(read("code-marks.json"), read("conv-marks.json"), 8192, .00025,
        previous["conv_rate_rps"], 9000, 600, 20261024, digest(inputs / "prompt-bank.json"))
    root.mkdir(parents=True, exist_ok=True)
    frozen = root / "frozen"
    frozen.mkdir()
    shutil.copy2(original_path, frozen / "trace.json")
    (frozen / "lower-trace.json").write_text(json.dumps(lower, indent=2) + "\n")
    plan = supplemental_plan()
    (frozen / "calibration-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    config = {"code_rate_rps": .00025, "conv_rate_rps": previous["conv_rate_rps"], "duration_s": 9600,
        "warmup_s": 600, "seed": 20261024, "policies": list(POLICIES), "repeats": 1,
        "cache_fixed_original_policies": ["random_no_return", "kv_only_workaware"],
        "input_sha256": previous["input_sha256"],
        "frozen_sha256": {name: digest(frozen / filename) for name, filename in (
            ("original-trace.json", "trace.json"), ("lower-trace.json", "lower-trace.json"),
            ("calibration-plan.json", "calibration-plan.json"))},
        "scope": "single_paired_lower_rate_trace_not_statistical_tail_confirmation", "migration_overlap": False,
        "setting_changes": "all_policy_completion_cache_invalidation_supplement_N5_N6_N7_then_halve_Code_rate"}
    if any(value is not None for value in (reuse_source, reuse_remote, reuse_profile)):
        if any(value is None for value in (reuse_source, reuse_remote, reuse_profile)):
            raise ValueError("Supply all local/remote raw and verified composite reuse inputs")
        config.update({"reuse_source_root": reuse_remote,
            "reuse_source_files_sha256": reuse_source_files(reuse_source),
            "reuse_fit_data": verify_profile(reuse_profile)})
        if not same_payload(config["reuse_fit_data"], json.loads((reuse_profile / "fit.json").read_text())):
            raise ValueError("Reuse composite fit must reproduce from archived raw")
    screen = validate_config(config, original, lower, plan)
    (frozen / "phase-config.json").write_text(json.dumps(config, indent=2) + "\n")
    state = {"phase": "prepared_early_return_and_real_lower_rate_trace", "repo": str(repo), "inputs": str(inputs),
        "GPU_experiment_complete": False, "total_research_goal_completed": False, "publication_authorized": False,
        "previous_job": previous["job"], "protected_vllm_sha256": previous["protected_vllm_sha256"],
        "trace_sha256": digest(frozen / "trace.json"), "input_sha256": previous["input_sha256"],
        "conv_rate_rps": previous["conv_rate_rps"], "screen": screen,
        "reuse_source_root": config.get("reuse_source_root", ""),
        "lower_counts": [sum(mark["model"] == model for mark in lower["requests"]) for model in (0, 1)],
        "frozen_runtime_sha256": {name: digest(repo / name) for name in SYNC_FILES if name.endswith((".py", ".sbatch", ".sh"))}}
    save_state(root, state)


def submission_command(environment, ledger):
    exports = "; ".join(f"export {name}={shlex.quote(value)}" for name, value in environment.items())
    return ('set -eu; test -z "$(squeue -h -u "$USER")"; '
            'test "$(git -C "$HOME/kvcached/kvcached" branch --show-current)" = main; '
            'test "$(git -C "$HOME/kvcached/kvcached" remote get-url origin)" = git@github.com:gjghfd/kvcached.git; '
            f'test ! -e {shlex.quote(ledger)}; ' + exports + '; '
            'sbatch --parsable --export=ALL --output="$HOME/run/kvcached-paracloud/logs/live-trace-%j.out" '
            '--error="$HOME/run/kvcached-paracloud/logs/live-trace-%j.err" '
            f'"$HOME/kvcached/kvcached/experiments/dp_opt/workaware-phase-return.sbatch" > {shlex.quote(ledger)}; '
            f'cat {shlex.quote(ledger)}')


def run(root):
    state = json.loads((root / "campaign-state.json").read_text())
    repo = Path(state["repo"])
    try:
        check_pause(root)
        check_frozen_sources(repo, state)
        if "job" not in state:
            if state["phase"] != "prepared_early_return_and_real_lower_rate_trace":
                raise RuntimeError("Reconcile the remote submission ledger before any retry")
            tests = sorted({name for name in SYNC_FILES if name.startswith("tests/")})
            with (root / "cpu-regression.log").open("w") as output:
                result = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests], cwd=repo,
                                        stdout=output, stderr=subprocess.STDOUT)
            (root / "cpu-regression.exit").write_text(str(result.returncode) + "\n")
            if result.returncode:
                raise RuntimeError("CPU regression failed; do not synchronize or submit")
            check_frozen_sources(repo, state)
            stage, home = protected_sync(repo, root, state, files=SYNC_FILES, prefix="workaware-early-return",
                upload_files=("frozen/lower-trace.json", "frozen/calibration-plan.json", "frozen/phase-config.json"))
            environment = {"DP_OPT_WORKAWARE_PHASE_UPLOAD": stage + "/upload",
                "DP_OPT_WORKAWARE_REUSE_SOURCE": state.get("reuse_source_root", ""),
                "DP_OPT_WORKAWARE_CONV_RATE": str(state["conv_rate_rps"]),
                "DP_OPT_WORKAWARE_LEGACY_SOURCE": home + "/run/kvcached-paracloud/state/live-trace-256440/cadence",
                "DP_OPT_MOVE_FIXED_S": "1.7743364535272121", "DP_OPT_MOVE_BANDWIDTH_GIB_S": "11.64086282917902"}
            ledger = stage + "/submitted-job.txt"
            state.update({"phase": "submission_intent_do_not_duplicate", "submission_ledger": ledger,
                          "remote_sync_stage": stage, "frozen_environment": environment})
            save_state(root, state)
            output = checked_ssh(submission_command(environment, ledger))
            state.update({"job": int(output.split(";")[0]), "phase": "submitted_calibration_then_paired_traces"})
            (root / "submit.txt").write_text(output + "\n")
            save_state(root, state)
            remote = home + "/run/kvcached-paracloud/state/live-trace-" + str(state["job"])
            watcher = checked_ssh(f'''set -eu
mkdir -p {shlex.quote(remote)}
nohup bash "$HOME/kvcached/kvcached/experiments/dp_opt/watch-trace-remote.sh" {state['job']} \\
    >> {shlex.quote(remote + '/remote-query.log')} 2>&1 < /dev/null &
printf '%s\\n' "$!"''')
            state["remote_watcher_pid"] = int(watcher)
            save_state(root, state)
        poll_job(root, state["job"], state)
        archive_job(root, state["job"], repo / "experiments/dp_opt", state)
        raw = root / f"live-trace-{state['job']}"
        if (raw / "exit-status").read_text().strip() != "0":
            raise RuntimeError("GPU phase failed; diagnose immutable raw before retry")
        checks = {"supplement": verify_profile(raw / "composite-profile/supplement"),
                  "composite": verify_profile(raw / "composite-profile")}
        for phase in ("cache-fixed-original", "lower-code-rate"):
            checks[phase] = verify(raw / phase, Path(state["inputs"]) / "prompt-bank.json")
            duration = 4800 if phase == "cache-fixed-original" else 9600
            baselines = ("random_no_return",) if phase == "cache-fixed-original" else POLICIES[:-1]
            metrics = {baseline: paired_window(raw / phase, baseline, "kv_only_workaware", 600, duration)
                       for baseline in baselines}
            (root / (phase + "-comparisons.json")).write_text(json.dumps(metrics, indent=2) + "\n")
        (root / "independent-verification.json").write_text(json.dumps(checks, indent=2) + "\n")
        check_frozen_sources(repo, state)
        state.update({"phase": "early_return_lower_rate_SHA_verified_analyzed", "GPU_experiment_complete": True,
                      "next_action": "record_all_adverse_metrics_then_paired_seed_confirmation_or_cost_diagnosis"})
        save_state(root, state)
    except Exception as error:
        state.update({"phase": "diagnosis_required_no_automatic_duplicate_submission", "error": repr(error)})
        save_state(root, state)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--previous-root", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-upload", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--reuse-source", type=Path)
    parser.add_argument("--reuse-remote")
    parser.add_argument("--reuse-profile", type=Path)
    parser.add_argument("--validate-reuse-source", type=Path)
    parser.add_argument("--validate-reused-profile", type=Path)
    parser.add_argument("--phase-config", type=Path)
    args = parser.parse_args()
    if args.validate_upload is not None:
        print(json.dumps(validate_upload(args.validate_upload, args.inputs), indent=2))
        return
    if args.validate_reuse_source is not None or args.validate_reused_profile is not None:
        if args.phase_config is None:
            parser.error("Require frozen phase config for calibration reuse")
        config = json.loads(args.phase_config.read_text())
        result = (validate_reuse_source(args.validate_reuse_source, config) if args.validate_reuse_source is not None
                  else validate_reused_profile(args.validate_reused_profile, config))
        print(json.dumps(result, indent=2))
        return
    if args.root is None or (not args.resume and args.previous_root is None):
        parser.error("Require campaign root and previous verified root")
    root, repo = args.root.resolve(), Path(__file__).resolve().parents[2]
    if not args.resume:
        prepare(root, args.previous_root.resolve(), repo, args.reuse_source, args.reuse_remote, args.reuse_profile)
    if not args.prepare_only:
        run(root)


if __name__ == "__main__":
    main()
