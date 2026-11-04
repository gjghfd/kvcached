#!/usr/bin/env bash
#SBATCH --job-name=qwen-upstream-qkv-mirror
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=03:00:00
set -euo pipefail
root="${KVCACHED_DATA_ROOT:-$HOME/run/kvcached-paracloud}"
upload="${DP_OPT_QKV_MIRROR_UPLOAD:?Supply immutable mirror upload}"
evidence="$root/state/live-trace-${SLURM_JOB_ID:?Require actual allocation}"
test ! -e "$evidence/runtime"
mkdir -p "$evidence"
finish() {
    result=$?
    trap - EXIT
    printf '%s\n' "$result" > "$evidence/exit-status"
    exit "$result"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
source "$HOME/aegaeon/paracloud/scripts/common.sh"
require_compute
load_toolchains
export PATH="$root/env/bin:$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib/python3.10/site-packages/torch/lib:$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cp -a "$upload/runtime" "$evidence/runtime"
find "$evidence/runtime" -type f -exec chmod u+w {} +
cd "$evidence"
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_ADDOPTS='-p no:cacheprovider'
export PYTHONPATH="$evidence/runtime/experiments/dp_opt:$DEPLOY_ROOT/scripts"
export DP_OPT_UPSTREAM_COMPUTE_CONFIRMED=1 DP_OPT_QKV_MIRROR_COMPUTE_CONFIRMED=1 DP_OPT_QKV_MIRROR_REPETITIONS=3
python - "$upload" "$evidence/runtime" <<'PY'
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath

upload, runtime = [Path(value).resolve() for value in sys.argv[1:]]
def check(directory, expected):
    actual = {}
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("Compute immutable tree contains a symlink")
        if path.is_file():
            actual[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in expected:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or str(relative) != name:
            raise RuntimeError("Unsafe immutable manifest path")
    if actual != expected:
        raise RuntimeError("Compute immutable SHA/file set differs before tests/imports")
integrity = upload / "compute-integrity.json"
check(upload, {**json.loads(integrity.read_text()), "compute-integrity.json": hashlib.sha256(integrity.read_bytes()).hexdigest()})
check(runtime, json.loads((upload / "runtime-manifest.json").read_text()))
PY
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT \
    python "$evidence/runtime/experiments/dp_opt/upstream_qkv_mirror_campaign.py" \
    --compute-upload "$upload" --compute-evidence "$evidence" --compute-preflight \
    > "$evidence/mirror-input-preflight.log" 2>&1
nvidia-smi --query-gpu=uuid,name,memory.total,memory.free --format=csv,noheader,nounits > "$evidence/gpu-memory-before-load.csv"
awk -F, 'NF != 4 || $2 !~ /A800/ || ($4 + 0) < 70000 { bad=1 }
    END { exit (NR != 2 || bad) }' "$evidence/gpu-memory-before-load.csv"
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT -u VLLM_BATCH_INVARIANT \
    python -m pytest -q -p no:cacheprovider \
    "$evidence/runtime/tests/test_dp_opt_upstream_qkv_mirror_probe.py" \
    "$evidence/runtime/tests/test_dp_opt_upstream_qkv_mirror_campaign.py" \
    "$evidence/runtime/tests/test_dp_opt_upstream_first_layer_probe.py" \
    "$evidence/runtime/tests/test_dp_opt_head_hidden_probe.py" \
    "$evidence/runtime/tests/test_dp_opt_fp32_head_probe.py" \
    > "$evidence/mirror-startup-tests.log" 2>&1
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT \
    python -c 'import os; assert os.environ["PYTEST_ADDOPTS"] == "-p no:cacheprovider"; from upstream_qkv_mirror_campaign import InstalledProvenanceMirrorWorker; print("actual compute mirror imports passed", InstalledProvenanceMirrorWorker.__module__)' \
    > "$evidence/mirror-compute-imports.log" 2>&1
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT \
    python "$evidence/runtime/experiments/dp_opt/upstream_qkv_mirror_campaign.py" \
    --compute-upload "$upload" --compute-evidence "$evidence"
