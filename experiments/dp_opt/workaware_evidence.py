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


def runtime_files():
    return PROFILE_RUNTIME_FILES if os.environ.get("DP_OPT_WORKAWARE_PROFILES") == "1" else RUNTIME_FILES


def archive_runtime(root):
    source_root = Path(__file__).resolve().parent
    archive = root / "workaware-runtime-source"
    archive.mkdir()
    sources = {}
    for name in runtime_files():
        source = source_root / name
        shutil.copy2(source, archive / name)
        sources[name] = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    manifest = {"sources": sources, "scope": "immutable_launch_sources_for_workaware_calibration_and_serving"}
    if os.environ.get("DP_OPT_WORKAWARE_PROFILES") == "1":
        manifest["algorithm_version"] = "population_profile_v2"
    (root / "workaware-runtime-source.json").write_text(json.dumps(manifest, indent=2) + "\n")


def verify_runtime(root):
    manifest = json.loads((root / "workaware-runtime-source.json").read_text())
    inventory = {line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
                 for line in (root / "source.sha256").read_text().splitlines()}
    version = manifest.get("algorithm_version")
    if version not in (None, "population_profile_v2"):
        raise ValueError("Unknown workaware runtime algorithm version")
    expected = PROFILE_RUNTIME_FILES if version == "population_profile_v2" else RUNTIME_FILES
    if set(manifest["sources"]) != set(expected):
        raise ValueError("Workaware runtime source manifest is incomplete")
    for name, source in manifest["sources"].items():
        actual = hashlib.sha256((root / "workaware-runtime-source" / name).read_bytes()).hexdigest()
        if actual != source["sha256"] or inventory.get(source["path"]) != actual:
            raise ValueError("Workaware runtime differs from frozen launch source SHA")
    return manifest
