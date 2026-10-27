import argparse
import json
import shlex
from pathlib import Path

from archive_trace_job import verify_raw
from kv_pressure import background_pressure, rate_for_ratio, read_pressure
from rebalance_campaign import checked_ssh, protected_sync
from stress_campaign import poll_job, save_state
from stress_trace import make_stress_trace
from verify_workaware_profile import verify_profile
from workaware_profile_campaign import SYNC_FILES as PROFILE_SYNC_FILES
from workaware_profile_campaign import digest, finish, frozen_environment, submission_command


SYNC_FILES = tuple(dict.fromkeys((*PROFILE_SYNC_FILES,
    "experiments/dp_opt/workaware_pressure_campaign.py", "experiments/dp_opt/kv_pressure.py",
    "experiments/dp_opt/stress_trace.py", "experiments/dp_opt/stress_campaign.py",
    "experiments/dp_opt/rebalance_campaign.py", "experiments/dp_opt/amortized_campaign.py",
    "experiments/dp_opt/watch-trace-remote.sh", "tests/test_dp_opt_workaware_pressure_campaign.py",
    "tests/test_dp_opt_watch_trace.py", "tests/test_dp_opt_workaware_numeric_replay.py",
    "experiments/dp_opt/docs/workaware-pressure-1p2-protocol.md")))


def retarget_trace(old_trace, inputs, conv_rate):
    read = lambda name: json.loads((inputs / name).read_text())
    parameters = (read("code-marks.json"), read("conv-marks.json"), 8192, .0005)
    old_rate = old_trace["model_rates_rps"][1]
    prompt_sha = digest(inputs / "prompt-bank.json")
    original = make_stress_trace(*parameters, old_rate, 4200, 600, 20261024, prompt_sha)
    if original != old_trace:
        raise ValueError("Archived trace cannot be rebuilt from the unchanged input banks")
    new_trace = make_stress_trace(*parameters, conv_rate, 4200, 600, 20261024, prompt_sha)
    if ([mark for mark in original["requests"] if mark["model"] == 0]
            != [mark for mark in new_trace["requests"] if mark["model"] == 0]):
        raise ValueError("Code stream must remain exactly unchanged")
    old_conv = [mark for mark in original["requests"] if mark["model"] == 1]
    new_conv = [mark for mark in new_trace["requests"] if mark["model"] == 1]
    if len(new_conv) < len(old_conv):
        raise ValueError("Pressure increase must preserve every old Conv mark")
    for old, new in zip(old_conv, new_conv):
        if {key: value for key, value in old.items() if key != "arrival_s"} != {
                key: value for key, value in new.items() if key != "arrival_s"}:
            raise ValueError("Conv time dilation changed sampled lengths or request identity")
    return new_trace


def arrival_waits(phase):
    trace = json.loads((phase / "trace.json").read_text())
    marks = {mark["request_id"]: mark for mark in trace["requests"]}
    result = {}
    for policy in ("no_move", "no_move_fair", "kv_only", "kv_only_headroom", "kv_only_workaware"):
        records = json.loads((phase / f"r0-{policy}/requests.json").read_text())
        result[policy] = [{"request_id": record["request_id"],
            "arrival_s": marks[record["request_id"]]["arrival_s"],
            "queue_s": record["submitted_at"] - record["arrived_at"],
            "reservation_to_first_token_s": record["first_token_at"] - record["submitted_at"],
            "ttft_s": record["first_token_at"] - record["arrived_at"]}
            for record in records if record["model"] == 0]
    return result


def prepare(root, old_root, inputs):
    if (root / "prepared.json").exists() or (root / "campaign-state.json").exists():
        raise RuntimeError("Preserve the existing frozen configuration and submission ledger")
    old_job = 257484
    archive = json.loads((old_root / f"archive-{old_job}.json").read_text())
    review = json.loads((old_root / "postterminal-parent-review.json").read_text())
    if archive["state"] != "COMPLETED" or not review["GPU_trace_verified"]:
        raise ValueError("Require the completed independently verified prior baseline")
    verify_raw(old_root / f"live-trace-{old_job}", (old_root / f"raw-{old_job}.sha256").read_text())
    phase = old_root / f"live-trace-{old_job}/paired-trace"
    old_trace = json.loads((phase / "trace.json").read_text())
    pressure = background_pressure(json.loads((phase / "r0-no_move/population.json").read_text()),
        json.loads((phase / "r0-no_move/decisions.json").read_text()), 600, 4800, 10240)
    old_rate = old_trace["model_rates_rps"][1]
    setting = rate_for_ratio(pressure, old_rate, 1.2, maximum_rate=2 * old_rate)
    rate = setting["Conv_rate_rps_candidate"]
    trace = retarget_trace(old_trace, inputs, rate)
    frozen = root / "frozen"
    frozen.mkdir(parents=True)
    (frozen / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    protocol = {"code_rate_rps": .0005, "conv_rate_rps": rate, "warmup_s": 600,
        "duration_s": 4800, "seed": 20261024, "target_local_physical_ratio": 1.2,
        "budget_mib_per_gpu": 10240, "rate_setting": setting, "prior_measured_pressure": pressure,
        "counts_per_policy": [sum(mark["model"] == model for mark in trace["requests"]) for model in (0, 1)],
        "change_scope": "Conv_rate_only_same_Poisson_seed_Code_marks_lengths_budget_graph_algorithm",
        "pressure_tolerance": .02, "pressure_target_is_measured_not_assumed": True,
        "all_rates_and_adverse_results_retained": True, "Code_tail_scope": "two_samples_descriptive_only"}
    (frozen / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    old_state = json.loads((old_root / "campaign-state.json").read_text())
    prepared = {"old_job": old_job, "old_root": str(old_root), "inputs": str(inputs),
        "trace_sha256": digest(frozen / "trace.json"), "protocol_sha256": digest(frozen / "protocol.json"),
        "input_sha256": {name: digest(inputs / name) for name in ("code-marks.json", "conv-marks.json", "prompt-bank.json")},
        "protected_vllm_sha256": old_state["protected_vllm_sha256"], "conv_rate_rps": rate,
        "calibration": str(old_root.parent / "campaign-v1/live-trace-256440/cadence")}
    (root / "prepared.json").write_text(json.dumps(prepared, indent=2) + "\n")
    return prepared


def run(root, repo):
    if (root / "campaign-state.json").exists():
        raise RuntimeError("Submission ledger exists; reconcile instead of submitting twice")
    prepared = json.loads((root / "prepared.json").read_text())
    inputs = Path(prepared["inputs"])
    for filename, expected in prepared["input_sha256"].items():
        if digest(inputs / filename) != expected:
            raise ValueError("Input bank changed after pressure freeze")
    for filename in ("trace", "protocol"):
        if digest(root / f"frozen/{filename}.json") != prepared[f"{filename}_sha256"]:
            raise ValueError("Frozen pressure workload changed before submission")
    fit = verify_profile(Path(prepared["calibration"]))
    (root / "independent-reused-profile-fit.json").write_text(json.dumps(fit, indent=2) + "\n")
    state = {**prepared, "phase": "prepared_for_protected_sync", "GPU_experiment_complete": False,
        "total_research_goal_completed": False, "publication_authorized": False,
        "frozen_runtime_sha256": {filename: digest(repo / filename) for filename in SYNC_FILES
                                   if filename.endswith((".py", ".sbatch", ".sh"))}}
    save_state(root, state)
    try:
        stage, home = protected_sync(repo, root, state, files=SYNC_FILES, prefix="workaware-pressure-1p2")
        environment = frozen_environment(stage + "/upload/frozen/trace.json", state["trace_sha256"],
            home + "/run/kvcached-paracloud/state/live-trace-256440/cadence", prepared["conv_rate_rps"])
        environment["DP_OPT_WORKAWARE_SKIP_RECOVERY"] = "1"
        ledger = stage + "/submitted-job.txt"
        state.update({"phase": "submission_intent_do_not_duplicate", "submission_ledger": ledger,
            "frozen_environment": environment})
        save_state(root, state)
        output = checked_ssh(submission_command(environment, ledger))
        job = int(output.split(";")[0])
        (root / "submit.txt").write_text(output + "\n")
        state.update({"job": job, "phase": "submitted_pressure_1p2_five_policy_trace"})
        save_state(root, state)
        remote = f"{home}/run/kvcached-paracloud/state/live-trace-{job}"
        watcher = checked_ssh(f'''set -eu
mkdir -p {shlex.quote(remote)}
printf '%s\\n' '{job}' > {shlex.quote(remote + '/remote-query.expected')}
nohup bash "$HOME/kvcached/kvcached/experiments/dp_opt/watch-trace-remote.sh" {job} \\
    >> {shlex.quote(remote + '/remote-query.log')} 2>&1 < /dev/null &
printf '%s\\n' "$!"''')
        state["remote_watcher_pid"] = int(watcher)
        save_state(root, state)
        poll_job(root, job, state)
        for filename, expected in state["frozen_runtime_sha256"].items():
            if digest(repo / filename) != expected:
                raise RuntimeError("Frozen local campaign/runtime source changed while running")
        finish(root, repo, inputs, state)
        phase = root / f"live-trace-{job}/paired-trace"
        pressures = read_pressure(phase)
        actual = pressures["no_move"]["local_physical_ratio"]
        result = {"job": job, "target_ratio": 1.2, "actual_no_move_ratio": actual,
            "within_prespecified_pressure_tolerance": abs(actual - 1.2) <= .02,
            "measured_pressure": pressures, "Code_waits": arrival_waits(phase),
            "GPU_trace_complete_and_verified": True, "total_research_goal_completed": False}
        (root / "pressure-results.json").write_text(json.dumps(result, indent=2) + "\n")
        state.update({"phase": "pressure_1p2_trace_verified_analyzed", "pressure_result": "pressure-results.json",
            "next_action": "review_actual_ratio_Code_blockage_and_both_tenant_tail_before_rate_refinement"})
        save_state(root, state)
    except Exception as error:
        state.update({"phase": "diagnosis_required_no_automatic_duplicate_submission", "error": repr(error)})
        save_state(root, state)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--old-root", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        if args.old_root is None or args.inputs is None:
            parser.error("Preparation requires --old-root and --inputs")
        print(json.dumps(prepare(root, args.old_root.resolve(), args.inputs.resolve()), indent=2))
    else:
        run(root, Path(__file__).resolve().parents[2])


if __name__ == "__main__":
    main()
