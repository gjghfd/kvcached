import hashlib
import json
import os
import shutil
from pathlib import Path


RUNTIME_FILES = (
    "ilp_reference.py", "live_controller.py", "live_scheduler.py", "live_trace.py", "live_rebalance.py",
    "live_benefit.py", "live_benchmark.py", "worker.py", "peer_transfer.py", "amortized_rebalance.py",
    "headroom_rebalance.py", "calibrated_cost.py", "motivation.py", "layout_cost_pilot.py",
    "workaware_cost.py", "workaware_rebalance.py", "workaware_ilp.py", "workaware_worker.py",
    "workaware_pilot.py", "verify_workaware_pilot.py", "workaware_evidence.py",
    "verify_workaware_rebalance.py", "workaware_fixture.py", "workaware-return.sbatch",
)

PROFILE_RUNTIME_FILES = (*RUNTIME_FILES, "analyze_workaware_cost.py", "workaware_profile_cost.py",
                         "verify_workaware_profile.py", "workaware-profile-return.sbatch")
RANDOM_RUNTIME_FILES = (*PROFILE_RUNTIME_FILES, "random_placement.py")
PHASE_RUNTIME_FILES = (*PROFILE_RUNTIME_FILES, "workaware_calibration_plan.py", "workaware_profile_bundle.py",
                       "workaware-phase-return.sbatch")
PHASE_RANDOM_RUNTIME_FILES = (*PHASE_RUNTIME_FILES, "random_placement.py")
ROBUST_RUNTIME_FILES = ("workaware_robust_adapter.py", "workaware_robust_rebalance.py",
                        "workaware_robust_ilp.py", "verify_workaware_robust_rebalance.py",
                        "verify_rebalance.py", "verify_trace.py")


def runtime_files(optimizer="point"):
    if optimizer not in ("point", "robust"):
        raise ValueError("Unknown workaware runtime optimizer")
    if optimizer == "robust":
        return (*runtime_files(), *ROBUST_RUNTIME_FILES)
    if os.environ.get("DP_OPT_WORKAWARE_PHASE_PROFILES") == "1":
        return PHASE_RANDOM_RUNTIME_FILES if os.environ.get("DP_OPT_RANDOM_BASELINE") == "1" else PHASE_RUNTIME_FILES
    if os.environ.get("DP_OPT_RANDOM_BASELINE") == "1":
        return RANDOM_RUNTIME_FILES
    return PROFILE_RUNTIME_FILES if os.environ.get("DP_OPT_WORKAWARE_PROFILES") == "1" else RUNTIME_FILES


def archive_runtime(root, optimizer="point"):
    source_root = Path(__file__).resolve().parent
    archive = root / "workaware-runtime-source"
    archive.mkdir()
    sources = {}
    for name in runtime_files(optimizer):
        source = source_root / name
        shutil.copy2(source, archive / name)
        sources[name] = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    manifest = {"sources": sources, "scope": "immutable_launch_sources_for_workaware_calibration_and_serving"}
    if optimizer == "robust":
        manifest["workaware_optimizer"] = "robust"
    if os.environ.get("DP_OPT_WORKAWARE_PROFILES") == "1":
        manifest["algorithm_version"] = "population_profile_v2"
    if os.environ.get("DP_OPT_RANDOM_BASELINE") == "1":
        manifest["algorithm_version"] = "population_profile_v2_random_admission_v1"
    if os.environ.get("DP_OPT_WORKAWARE_PHASE_PROFILES") == "1":
        manifest["algorithm_version"] = ("population_profile_v3_release_observation_random_v1"
            if os.environ.get("DP_OPT_RANDOM_BASELINE") == "1" else "population_profile_v3_release_observation")
    (root / "workaware-runtime-source.json").write_text(json.dumps(manifest, indent=2) + "\n")


def verify_runtime(root):
    manifest = json.loads((root / "workaware-runtime-source.json").read_text())
    inventory = {line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
                 for line in (root / "source.sha256").read_text().splitlines()}
    version = manifest.get("algorithm_version")
    optimizer = manifest.get("workaware_optimizer", "point")
    if optimizer not in ("point", "robust"):
        raise ValueError("Unknown archived workaware optimizer")
    if version not in (None, "population_profile_v2", "population_profile_v2_random_admission_v1",
                       "population_profile_v3_release_observation", "population_profile_v3_release_observation_random_v1"):
        raise ValueError("Unknown workaware runtime algorithm version")
    expected = (RANDOM_RUNTIME_FILES if version == "population_profile_v2_random_admission_v1" else
                PROFILE_RUNTIME_FILES if version == "population_profile_v2" else RUNTIME_FILES)
    if version == "population_profile_v3_release_observation":
        expected = PHASE_RUNTIME_FILES
    elif version == "population_profile_v3_release_observation_random_v1":
        expected = PHASE_RANDOM_RUNTIME_FILES
    if optimizer == "robust":
        expected = (*expected, *ROBUST_RUNTIME_FILES)
    if set(manifest["sources"]) != set(expected):
        raise ValueError("Workaware runtime source manifest is incomplete")
    for name, source in manifest["sources"].items():
        actual = hashlib.sha256((root / "workaware-runtime-source" / name).read_bytes()).hexdigest()
        if actual != source["sha256"] or inventory.get(source["path"]) != actual:
            raise ValueError("Workaware runtime differs from frozen launch source SHA")
    return manifest
