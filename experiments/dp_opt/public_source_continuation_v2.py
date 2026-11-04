import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from amortized_campaign import check_frozen_sources
from archive_trace_job import verify_raw
from public_response_inputs import digest, validate
from stress_campaign import check_pause, save_state
from verify_dataset_viewer_snapshot import verify as verify_snapshot


PROMOTIONS = ("experiments/dp_opt/public_workload_search.py", "experiments/dp_opt/public_poisson_followup.py",
              "experiments/dp_opt/public-workload-search.sbatch", "experiments/dp_opt/public-poisson-search.sbatch")
WATCHDOGS = {"public-workload-search.sbatch": 7200, "public-poisson-search.sbatch": 18000}


def validate_watchdogs(stage):
    for name, timeout in WATCHDOGS.items():
        script = stage / "experiments/dp_opt" / name
        subprocess.run(["bash", "-n", str(script)], check=True)
        exports = [word for line in script.read_text().splitlines() if line.startswith("export ")
                   for word in shlex.split(line)[1:]]
        for key, value in (("DP_OPT_REQUEST_TIMEOUT_S", timeout), ("DP_OPT_DEADLINE_S", 120)):
            settings = [word.split("=", 1)[1] for word in exports if word.startswith(key + "=")]
            if settings != [str(value)]:
                raise RuntimeError("Future watchdog/SLO configuration differs from the frozen protocol")
    return dict(WATCHDOGS)


def ready_parent(parent, previous):
    if "diagnosis_required" in previous.get("phase", ""):
        raise RuntimeError("Parent needs diagnosis; preserve its raw evidence and do not duplicate jobs")
    if not previous.get("GPU_calibration_complete"):
        return False
    if (previous.get("state_last_observed") != "COMPLETED"
            or previous.get("phase") != "public_source_mechanism_calibration_SHA_verified"):
        raise RuntimeError("Require independent public parent closure, not just a Slurm terminal state")
    raw = parent / f"live-trace-{previous['job']}"
    verify_raw(raw, (parent / f"raw-{previous['job']}.sha256").read_text())
    if (json.loads((parent / "independent-verification.json").read_text())
            != json.loads((raw / "search-analysis.json").read_text())):
        raise RuntimeError("Parent independent analysis differs from its raw result")
    identity = json.loads((parent / "local-monitor-identity.json").read_text())
    alive = subprocess.run(["ps", "-p", str(identity["pid"]), "-o", "pid="], capture_output=True).returncode == 0
    return not alive


def promote(repo, stage, root, state):
    backup = root / "pre-promotion-source"
    for name in PROMOTIONS:
        if digest(repo / name) != state["old_sha256"][name] or digest(stage / name) != state["staged_sha256"][name]:
            raise RuntimeError("Public label promotion source changed; do not overwrite local edits")
        if (backup / name).exists() or (repo / name).with_name(Path(name).name + ".source-promotion.tmp").exists():
            raise RuntimeError("Preserve existing source promotion evidence; reconcile before retry")
    for name in PROMOTIONS:
        target = repo / name
        old = backup / name
        old.parent.mkdir(parents=True, exist_ok=True)
        if old.exists():
            raise RuntimeError("Preserve previous promotion backup; reconcile before retry")
        shutil.copy2(target, old)
        temporary = target.with_name(target.name + ".source-promotion.tmp")
        if temporary.exists():
            raise RuntimeError("Preserve unknown partial source promotion")
        shutil.copy2(stage / name, temporary)
        if digest(temporary) != state["staged_sha256"][name]:
            raise RuntimeError("Staged public label copy failed SHA verification")
        temporary.replace(target)
        if digest(target) != state["staged_sha256"][name]:
            raise RuntimeError("Public label promotion failed final SHA verification")


def stage_tests(repo, stage, inputs, root):
    script = '''import importlib.util,sys,pathlib,pytest
repo,stage,inputs=map(pathlib.Path,sys.argv[1:])
sys.path.insert(0,str(repo/'experiments/dp_opt'))
for name in ('public_workload_search','public_poisson_followup'):
    spec=importlib.util.spec_from_file_location(name,stage/'experiments/dp_opt'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
assert sys.modules['public_workload_search'].public_tenant_name(inputs)=='Bespoke-Stratos-original-reasoning'
raise SystemExit(pytest.main(['-q',str(repo/'tests/test_dp_opt_public_workload_search.py'),str(repo/'tests/test_dp_opt_public_poisson_followup.py'),str(repo/'tests/test_dp_opt_public_workload_campaign.py')]))
'''
    with (root / "staged-CPU-regression.log").open("w") as output:
        result = subprocess.run([sys.executable, "-c", script, str(repo), str(stage), str(inputs)],
                                stdout=output, stderr=subprocess.STDOUT)
    (root / "staged-CPU-regression.exit").write_text(str(result.returncode) + "\n")
    if result.returncode:
        raise RuntimeError("Staged source-label regression failed; no promotion or new allocation")


def run(root, parent, calibration, evaluation, snapshot):
    repo = Path(__file__).resolve().parents[2]
    stage = repo / "experiments/dp_opt/.public-source-stage-r2"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "campaign-state.json").exists():
        raise RuntimeError("Preserve recorded source continuation; reconcile owned processes before restarting")
    inputs = calibration / "frozen/inputs"
    manifest = validate(inputs)
    if manifest["dataset_label"] != "stratos" or manifest["bank_size"] != 256:
        raise RuntimeError("Require the outcome-independent full-distribution Stratos256 bank")
    verification = verify_snapshot(snapshot)
    if (verification != json.loads((snapshot / "independent-original-cell-verification.json").read_text())
            or verification["source_sha256"] != manifest["sources"][0]["sha256"]):
        raise RuntimeError("Stratos original viewer export/provenance differs from its bank")
    watchdogs = validate_watchdogs(stage)
    state = {"repo": str(repo), "parent_root": str(parent), "calibration_root": str(calibration),
             "evaluation_root": str(evaluation), "phase": "waiting_parent_SHA_closure_then_Stratos",
             "old_sha256": {name: digest(repo / name) for name in PROMOTIONS},
             "staged_sha256": {name: digest(stage / name) for name in PROMOTIONS},
             "frozen_runtime_sha256": json.loads((parent / "campaign-state.json").read_text())["frozen_runtime_sha256"],
             "input_manifest_sha256": digest(inputs / "input-manifest.json"),
             "controller_sha256": digest(Path(__file__)), "parent_job": None,
             "watchdogs_s": watchdogs, "SLO_s": 120, "parent_runtime_unchanged_until_independent_closure": True,
             "SSH_queries_while_waiting": False, "publication_authorized": False,
             "total_research_goal_completed": False}
    save_state(root, state)
    try:
        stage_tests(repo, stage, inputs, root)
        while True:
            check_pause(root)
            check_pause(parent)
            check_frozen_sources(repo, state)
            validate(inputs)
            if digest(inputs / "input-manifest.json") != state["input_manifest_sha256"]:
                raise RuntimeError("Frozen Stratos input manifest changed")
            if digest(Path(__file__)) != state["controller_sha256"]:
                raise RuntimeError("Active continuation controller changed")
            previous = json.loads((parent / "campaign-state.json").read_text())
            state.update({"parent_job": previous.get("job"), "parent_phase": previous.get("phase"),
                          "parent_last_poll_at": previous.get("last_poll_at")})
            save_state(root, state)
            if ready_parent(parent, previous):
                break
            time.sleep(300)
        if verify_snapshot(snapshot) != verification:
            raise RuntimeError("Original Stratos raw snapshot changed before deployment")
        validate_watchdogs(stage)
        promote(repo, stage, root, state)
        state["frozen_runtime_sha256"].update(state["staged_sha256"])
        state["phase"] = "running_Stratos_native_and_open_loop_calibration"
        save_state(root, state)
        with (root / "Stratos-calibration-coordinator.log").open("w") as output:
            result = subprocess.run([sys.executable, str(repo / "experiments/dp_opt/public_workload_campaign.py"),
                "--root", str(calibration), "--previous-root", str(parent)], stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("Stratos calibration failed; diagnose its complete raw archive before retry")
        result = json.loads((calibration / "campaign-state.json").read_text())
        if not result.get("GPU_calibration_complete") or result.get("candidate_case") is None:
            raise RuntimeError("Stratos physical pressure/queue calibration has no qualified rate; retain all negatives")
        check_frozen_sources(repo, state)
        state["phase"] = "running_Stratos_seven_policy_grid_and_fixed_heldout"
        save_state(root, state)
        with (root / "Stratos-evaluation-coordinator.log").open("w") as output:
            result = subprocess.run([sys.executable, str(repo / "experiments/dp_opt/public_poisson_followup.py"),
                "--root", str(evaluation), "--parent-root", str(calibration)], stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("Stratos grid/heldout failed; diagnose its complete raw evidence before retry")
        state.update({"phase": "Stratos_grid_heldout_SHA_closed_requires_scientific_conclusion",
                      "total_research_goal_completed": False,
                      "next_action": "compare_all_tenants_TTFT_and_Conv_TPOT_and_causal_return_repayment"})
        save_state(root, state)
    except Exception as error:
        state.update({"phase": "diagnosis_required_no_automatic_duplicate_submission", "error": repr(error)})
        save_state(root, state)
        raise


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "parent", "calibration", "evaluation", "snapshot"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(*(getattr(args, name).resolve() for name in ("root", "parent", "calibration", "evaluation", "snapshot")))


if __name__ == "__main__":
    main()
