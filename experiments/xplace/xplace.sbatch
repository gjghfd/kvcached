#!/usr/bin/env bash
#SBATCH --job-name=xplace
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=01:00:00
# Submit with: sbatch --time=... --output=<tag>/logs/slurm-%j.out --export=ALL,XPLACE_TAG=...,XPLACE_PHASES=... xplace.sbatch
set -euo pipefail
export XPLACE_JOB_STARTED_AT="$(date +%s)"
unset XPLACE_SCAN_OVERHEAD
source "$HOME/aegaeon/paracloud/scripts/common.sh"
require_compute
load_toolchains
root="${KVCACHED_DATA_ROOT:-$HOME/run/kvcached-paracloud}"
tag="${XPLACE_TAG:?Set XPLACE_TAG to the uploaded campaign tag}"
base="$root/xplace/$tag"
upload="$base/upload"
work="$base/job-$SLURM_JOB_ID"
out="$work/out"
mkdir -p "$out"
finish() {
    code=$?
    trap - EXIT
    rm -rf "${DP_OPT_TRANSFER_ROOT:-/nonexistent-xplace}"
    printf '%s\n' "$code" > "$work/exit-status"
    date -Is > "$work/finished-at"
    exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
date -Is > "$work/started-at"
printf '%s\n' "$SLURM_JOB_NODELIST" > "$work/nodelist"
(cd "$upload" && sha256sum -c --quiet MANIFEST.sha256) > "$work/manifest-check.log" 2>&1
cp -a "$upload/runtime" "$upload/tests" "$upload/inputs" "$work/"
chmod -R u+w "$work/runtime" "$work/tests" "$work/inputs"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv,noheader,nounits > "$work/gpus.csv"
nvidia-smi topo -m > "$work/topology.txt" 2>&1 || true
python3 - "$work/gpus.csv" <<'EOF'
import sys
rows = [line.split(", ") for line in open(sys.argv[1]).read().strip().splitlines()]
assert len(rows) == 4, f"expected 4 GPUs, got {len(rows)}"
assert all("A800" in row[2] for row in rows), rows
assert all(float(row[4]) >= 70000 for row in rows), f"GPU memory already in use: {rows}"
EOF
export PATH="$root/env/bin:$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib/python3.10/site-packages/torch/lib:$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$work/runtime:$DEPLOY_ROOT/scripts"
export ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 DP_OPT_LIVE_PLACEMENT=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENGINE_READY_TIMEOUT_S=1800
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export KVCACHED_MIN_RESERVED_PAGES=0 KVCACHED_MAX_RESERVED_PAGES=0 KVCACHED_PAGE_PREALLOC_ENABLED=false
export KVCACHED_GPU_UTILIZATION=0.98 DP_OPT_PAUSE_SYNC_INTERVAL=32 DP_OPT_TRANSFER_CHUNK_MIB=32
export DP_OPT_TRANSFER_ROOT="$(mktemp -d /tmp/xplace-$SLURM_JOB_ID.XXXXXX)"
unset VLLM_BATCH_INVARIANT
cd "$work"
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT python - <<'EOF' > "$work/import-check.log" 2>&1
import importlib
import importlib.metadata
import hashlib
import json
from pathlib import Path

for name in ("engine", "cost_model", "planner", "fit", "fairness", "controller", "peer_transfer",
             "trace_runner", "calibrate", "imbalance_calibration", "graph_execution", "scan_job", "parallel_topology", "live_scheduler", "worker", "main"):
    importlib.import_module(name)
import kvcached
import torch
import vllm

vllm_root = Path(vllm.__file__).parent
kv_root = Path(kvcached.__file__).parent
paths = [vllm_root / "config/parallel.py", vllm_root / "distributed/parallel_state.py",
         kv_root / "integration/vllm/patches.py", kv_root / "dp_namespace.py"]
record = {"versions": {name: importlib.metadata.version(name) for name in ("vllm", "kvcached", "torch")},
          "cuda": torch.version.cuda,
          "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
Path("runtime-environment.json").write_text(json.dumps(record, indent=2) + "\n")
print("imports ok")
EOF
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT \
    -u XPLACE_NESTED_TIMING -u XPLACE_LIVE_OBSERVATION -u XPLACE_TIMING_ALLOWED -u XPLACE_BATCH_LINK \
    -u XPLACE_CAUSAL_MANIFEST -u XPLACE_CAUSAL_MANIFEST_SHA256 \
    -u XPLACE_SCAN_OVERHEAD -u XPLACE_GRAPH_EXECUTION_ALLOWED \
    PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    python - "$work/inputs/${XPLACE_STARTUP_CONTRACT:?Set an explicit non-timesliced startup contract}" <<'EOF' > "$work/startup-tests.log" 2>&1
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

contract = json.loads(Path(sys.argv[1]).read_text())
assert contract["non_timesliced_only"] is True and contract["nodes"]
assert all(node.startswith("tests/test_") and ".." not in node and "timesliced" not in node
           for node in contract["nodes"])
result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                         "--junitxml=startup-tests.xml", *contract["nodes"]])
if result.returncode:
    raise SystemExit(result.returncode)
suites = ET.parse("startup-tests.xml").getroot().iter("testsuite")
assert all(int(suite.attrib.get("skipped", "0")) == 0 for suite in suites), "Startup tests cannot skip"
EOF
checkpoint="$(python -c 'from qwen_common import checkpoint_path; print(checkpoint_path())')"
phases="${XPLACE_PHASES:?Set XPLACE_PHASES}"
policies="${XPLACE_POLICIES:-no_move+ca_full}"
args=(--checkpoint "$checkpoint" --output "$out" --prompt-bank "$work/inputs/prompt-bank.json"
      --num-ranks "${XPLACE_NUM_RANKS:-4}" --tensor-parallel-size "${XPLACE_TENSOR_PARALLEL_SIZE:-1}"
      --phases "${phases//+/,}" --policies "${policies//+/,}"
      --budget-margin-gib "${XPLACE_BUDGET_MARGIN_GIB:-3}" --seed "${XPLACE_SEED:-20261103}"
      --max-inflight "${XPLACE_MAX_INFLIGHT_MODEL0:-0}" "${XPLACE_MAX_INFLIGHT_MODEL1:-0}"
      --drain-timeout-s "${XPLACE_DRAIN_TIMEOUT_S:-5400}")
[[ -n "${XPLACE_TRACE:-}" ]] && args+=(--trace "$work/inputs/$XPLACE_TRACE")
[[ -n "${XPLACE_STEP_PARAMS:-}" ]] && args+=(--step-params "$work/inputs/$XPLACE_STEP_PARAMS")
[[ -n "${XPLACE_PLANNER_PARAMS:-}" ]] && args+=(--planner-params "$work/inputs/$XPLACE_PLANNER_PARAMS")
[[ -n "${XPLACE_BUDGET_CAP_GIB:-}" ]] && args+=(--budget-cap-gib "$XPLACE_BUDGET_CAP_GIB")
[[ -n "${XPLACE_PROBE_MAX_MS:-}" ]] && args+=(--probe-max-ms "$XPLACE_PROBE_MAX_MS")
[[ -n "${XPLACE_KV_MODE:-}" ]] && args+=(--kv-mode "$XPLACE_KV_MODE")
[[ -n "${XPLACE_RESERVE_EXTRA_TOKENS:-}" ]] && args+=(--reserve-extra-tokens "$XPLACE_RESERVE_EXTRA_TOKENS")
[[ -n "${XPLACE_SEGMENTS:-}" ]] && args+=(--segments "$work/inputs/$XPLACE_SEGMENTS")
[[ -n "${XPLACE_IMBALANCE_CONFIG:-}" ]] && args+=(--imbalance-config "$work/inputs/$XPLACE_IMBALANCE_CONFIG")
[[ -n "${XPLACE_CAUSAL_CONFIG:-}" ]] && args+=(--causal-config "$work/inputs/$XPLACE_CAUSAL_CONFIG")
[[ -n "${XPLACE_CAUSAL_MODE:-}" ]] && args+=(--causal-mode "$XPLACE_CAUSAL_MODE")
[[ "${XPLACE_ENABLE_TP_MIGRATION:-0}" == "1" ]] && args+=(--enable-tp-migration)
if [[ -n "${XPLACE_SCAN_BLOCK:-}" ]]; then
    if [[ -n "${XPLACE_CAUSAL_BLOCK:-}" || -n "${XPLACE_CAUSAL_MODE:-}" || -n "${XPLACE_CAUSAL_CONFIG:-}" || -n "${XPLACE_IMBALANCE_CONFIG:-}" ]]; then
        printf 'Scan block conflicts with causal or per-condition overrides\n' >&2
        exit 2
    fi
    export XPLACE_CAUSAL_MANIFEST="$work/inputs/${XPLACE_SCAN_MANIFEST_FILE:?}"
    export XPLACE_CAUSAL_MANIFEST_SHA256="${XPLACE_SCAN_MANIFEST_SHA256:?}"
    python "$work/runtime/scan_job.py" --block "$XPLACE_SCAN_BLOCK" \
        --protocol "$work/inputs/${XPLACE_SCAN_PROTOCOL_FILE:?}" "${args[@]}" > "$work/main.log" 2>&1
elif [[ -n "${XPLACE_CAUSAL_BLOCK:-}" ]]; then
    [[ -z "${XPLACE_CAUSAL_MODE:-}" ]]
    export XPLACE_CAUSAL_MANIFEST="$work/inputs/${XPLACE_CAUSAL_MANIFEST_FILE:?}"
    export XPLACE_CAUSAL_PROTOCOL="$work/inputs/${XPLACE_CAUSAL_PROTOCOL_FILE:?}"
    : "${XPLACE_CAUSAL_MANIFEST_SHA256:?}"
    python "$work/runtime/causal_job.py" --block "$XPLACE_CAUSAL_BLOCK" "${args[@]}" > "$work/main.log" 2>&1
else
    python "$work/runtime/main.py" "${args[@]}" > "$work/main.log" 2>&1
fi
