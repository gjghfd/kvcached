import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import public_poisson_followup
import public_workload_campaign
import public_workload_search
from amortized_campaign import check_frozen_sources
from archive_trace_job import verify_raw
from efficient_public_launch import LAUNCHES, launch_payload, make_launch, verify_launch, verify_phase
from rebalance_campaign import checked_ssh, protected_sync
from stress_campaign import check_pause, save_state
from verify_roundtrip_phase import verify as verify_controls
from verify_trace import verify as verify_trace


EXTRA_FILES = (
    "experiments/dp_opt/efficient_workaware_trace.py",
    "experiments/dp_opt/solver_efficiency.py",
    "experiments/dp_opt/grouped_robust_uncertainty.py",
    "experiments/dp_opt/prune_headroom_scenarios.py",
    "experiments/dp_opt/verify_solver_efficiency.py",
    "experiments/dp_opt/verify_efficient_trace.py",
    "experiments/dp_opt/efficient_public_continuation.py",
    "experiments/dp_opt/efficient_public_launch.py",
    "experiments/dp_opt/efficient-public-search.sbatch",
    "experiments/dp_opt/efficient-public-poisson.sbatch",
    "tests/test_dp_opt_grouped_robust_uncertainty.py",
    "tests/test_dp_opt_prune_headroom_scenarios.py",
    "tests/test_dp_opt_solver_efficiency.py",
    "tests/test_dp_opt_solver_efficiency_verifier.py",
    "tests/test_dp_opt_efficient_trace_verifier.py",
    "tests/test_dp_opt_efficient_public_continuation.py",
    "tests/test_dp_opt_efficient_public_launch.py",
    "experiments/dp_opt/analyze_tail_slo.py",
    "experiments/dp_opt/analyze_return_repayment.py",
    "experiments/dp_opt/benchmark.py",
    "experiments/dp_opt/single_model_benchmark.py",
    "experiments/dp_opt/queue_fairness.py",
    "experiments/dp_opt/verify_benefit.py",
    "experiments/dp_opt/verify_queue_fairness.py",
)
CAMPAIGN_ENABLED = False


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def adapted_verifiers():
    originals = [(module, module.verify) for module in (public_workload_search, public_poisson_followup)]
    if any(original is verify_phase for module, original in originals):
        raise RuntimeError("Equivalent public verifier context must not be nested")
    for module, original in originals:
        module.verify = verify_phase
    try:
        yield
    finally:
        for module, original in originals:
            module.verify = original


def sync_without_prompt_bank(repo, root, state, input_parent_job, **kwargs):
    root = Path(root)
    home = checked_ssh('printf "%s" "$HOME"')
    inputs = {str(path.relative_to(root)): digest(path)
              for path in sorted((root / "frozen/inputs").rglob("*")) if path.is_file()}
    if not inputs:
        raise ValueError("Require immutable public prompt-bank inputs before source installation")
    source = home + f"/run/kvcached-paracloud/state/live-trace-{input_parent_job}/"
    expected = [f"{value}  {source}{name}" for name, value in inputs.items()]
    actual = checked_ssh("sha256sum " + " ".join(shlex.quote(source + name) for name in inputs))
    (root / "reused-inputs-parent.sha256").write_text(actual + "\n")
    if actual.splitlines() != expected:
        raise RuntimeError("Remote public inputs differ; do not install or submit")
    options = dict(kwargs)
    options["upload_files"] = tuple(name for name in options.get("upload_files", ()) if name not in inputs)
    stage, home = protected_sync(repo, root, state, **options)
    commands = ['set -eu; test -z "$(squeue -h -u "$USER")"']
    for name, expected_sha in inputs.items():
        original, target = shlex.quote(source + name), shlex.quote(stage + "/upload/" + name)
        commands.extend([f'test "$(sha256sum {original} | cut -d\' \' -f1)" = {expected_sha}',
                         "mkdir -p " + shlex.quote(str(Path(stage + "/upload/" + name).parent)),
                         f"cp -p {original} {target}",
                         f'test "$(sha256sum {target} | cut -d\' \' -f1)" = {expected_sha}'])
    checked_ssh("; ".join(commands))
    state["public_input_parent_reused_after_double_SHA"] = {"job": input_parent_job, "files": inputs}
    return stage, home


@contextmanager
def campaign_adapters(input_parent_job):
    global CAMPAIGN_ENABLED
    if CAMPAIGN_ENABLED:
        raise RuntimeError("Equivalent public campaign context must not be nested")
    CAMPAIGN_ENABLED = True
    originals = []
    for module, role, command_name in (
            (public_workload_campaign, "search", "submission_command"),
            (public_poisson_followup, "poisson", "submit_command")):
        for name in ("SYNC_FILES", "protected_sync", command_name):
            originals.append((module, name, getattr(module, name)))
        module.SYNC_FILES = tuple(dict.fromkeys((*module.SYNC_FILES, *EXTRA_FILES)))
        original_command = getattr(module, command_name)
        canonical, wrapper, calls = LAUNCHES[role]

        def submit(*args, _original=original_command, _canonical=canonical, _wrapper=wrapper, **kwargs):
            command = _original(*args, **kwargs)
            if command.count(_canonical) != 1:
                raise ValueError("Canonical submission target changed")
            return command.replace(_canonical, _wrapper)

        setattr(module, command_name, submit)
        module.protected_sync = lambda repo, root, state, **kwargs: sync_without_prompt_bank(
            repo, root, state, input_parent_job, **kwargs)
    try:
        with adapted_verifiers():
            yield
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)
        CAMPAIGN_ENABLED = False


def verified_control_parent(parent):
    parent = Path(parent)
    previous = json.loads((parent / "campaign-state.json").read_text())
    if "diagnosis_required" in previous["phase"]:
        raise RuntimeError("Control parent failed; archive/diagnose it without a new submission")
    if not previous.get("GPU_controls_complete"):
        return None
    if (previous.get("state_last_observed") != "COMPLETED"
            or previous["phase"] != "controls_SHA_and_full_tokens_verified_not_benefit"):
        raise RuntimeError("Control completion declaration is not independent terminal closure")
    raw = parent / f"live-trace-{previous['job']}"
    verify_raw(raw, (parent / f"raw-{previous['job']}.sha256").read_text())
    bank = raw / "frozen/inputs/prompt-bank.json"
    rebuilt = {"state_controls": verify_controls(raw / "pilot", bank),
               "plain_no_move": verify_trace(raw / "plain-no-move", bank), "is_algorithm_benefit": False}
    if rebuilt != json.loads((parent / "independent-controls-verification.json").read_text()):
        raise RuntimeError("Independent control/token reconstruction differs before deployment")
    if any(rebuilt[name]["trace_sha256"] != previous["original_trace_sha256"]
           for name in ("state_controls", "plain_no_move")):
        raise RuntimeError("Do not substitute original strict-control inputs")
    identity = json.loads((parent / "local-monitor-identity.json").read_text())
    if subprocess.run(["ps", "-p", str(identity["coordinator_pid"]), "-o", "pid="],
                      capture_output=True).returncode == 0:
        return None
    return previous


def run(root, parent, bootstrap):
    repo = Path(__file__).resolve().parents[2]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "campaign-state.json").exists():
        raise RuntimeError("Preserve existing continuation state; reconcile before any restart")
    previous = json.loads((parent / "campaign-state.json").read_text())
    deployment = tuple(dict.fromkeys((*public_workload_campaign.SYNC_FILES,
                                     *public_poisson_followup.SYNC_FILES, *EXTRA_FILES)))
    state = {"repo": str(repo), "parent": str(parent), "bootstrap": str(bootstrap),
             "phase": "waiting_for_independent_strict_control_closure",
             "frozen_runtime_sha256": {**previous["frozen_runtime_sha256"],
                                       **{name: digest(repo / name) for name in deployment
                                          if name not in previous["frozen_runtime_sha256"]}},
             "is_algorithm_benefit": False, "publication_authorized": False,
             "research_goal_complete": False, "monitor_interval_s": 300}
    save_state(root, state)
    try:
        while True:
            check_pause(root)
            check_pause(parent)
            check_frozen_sources(repo, state)
            closed = verified_control_parent(parent)
            if closed is not None:
                break
            observed = json.loads((parent / "campaign-state.json").read_text())
            state.update({"parent_job": observed.get("job"), "parent_phase": observed["phase"],
                          "parent_GPU_state": observed.get("state_last_observed")})
            save_state(root, state)
            time.sleep(300)
        search, evaluation = root / "search", root / "evaluation"
        if search.exists() or evaluation.exists():
            raise RuntimeError("Preserve existing child evidence; never duplicate its allocation")
        shutil.copytree(parent / f"live-trace-{closed['job']}/frozen", search / "frozen")
        state.update({"phase": "running_equivalent_public_native_and_rate_search",
                      "parent_job": closed["job"], "search": str(search), "evaluation": str(evaluation)})
        save_state(root, state)
        with campaign_adapters(closed["job"]):
            public_workload_campaign.run(search, bootstrap)
            check_frozen_sources(repo, state)
            search_state = json.loads((search / "campaign-state.json").read_text())
            search_launch = verify_launch(search / f"live-trace-{search_state['job']}", repo, "search")
            with (search / "independent-launch-verification.json").open("x") as handle:
                handle.write(json.dumps(search_launch, indent=2) + "\n")
            if search_state.get("candidate_case") is None:
                state["phase"] = "public_native_rate_closed_no_qualified_pressure_preserve_negative_results"
            else:
                state["phase"] = "running_equivalent_seven_policy_and_heldout_evaluation"
                save_state(root, state)
                check_pause(root)
                public_poisson_followup.run(evaluation, search)
                evaluation_state = json.loads((evaluation / "campaign-state.json").read_text())
                evaluation_launch = verify_launch(evaluation / f"live-trace-{evaluation_state['job']}", repo, "poisson")
                with (evaluation / "independent-launch-verification.json").open("x") as handle:
                    handle.write(json.dumps(evaluation_launch, indent=2) + "\n")
                state["phase"] = "public_finite_grid_closed_results_require_scientific_review"
        check_frozen_sources(repo, state)
        save_state(root, state)
    except Exception as error:
        state.update({"phase": "diagnosis_required_no_automatic_duplicate_submission", "error": repr(error)})
        save_state(root, state)
        raise


def main():
    if "--module" in sys.argv:
        index = sys.argv.index("--module")
        role = sys.argv[index + 1]
        del sys.argv[index:index + 2]
        module = {"search": public_workload_search, "poisson": public_poisson_followup}[role]
        if "--root" not in sys.argv:
            raise ValueError("Independent public action requires its immutable job root")
        verify_launch(Path(sys.argv[sys.argv.index("--root") + 1]), Path(__file__).resolve().parents[2], role)
        with adapted_verifiers():
            module.main()
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--bootstrap", type=Path)
    parser.add_argument("--make-launch", choices=LAUNCHES)
    parser.add_argument("--verify-phase", type=Path)
    parser.add_argument("--prompt-bank", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.verify_phase is not None:
        result = verify_phase(args.verify_phase, args.prompt_bank)
        with args.output.open("x") as handle:
            handle.write(json.dumps(result, indent=2) + "\n")
    elif args.make_launch is not None:
        print(make_launch(Path(__file__).resolve().parents[2], args.output, args.make_launch))
    else:
        if args.root is None or args.parent is None or args.bootstrap is None:
            parser.error("Continuation requires root, control parent and verified bootstrap")
        run(args.root.resolve(), args.parent.resolve(), args.bootstrap.resolve())


if __name__ == "__main__":
    main()
