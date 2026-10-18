#!/usr/bin/env bash
#SBATCH --job-name=kvcached-build
#SBATCH --partition=gpu_a800,gpu_h100,gpu_h200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH --time-min=00:20:00
set -euo pipefail
source "$HOME/aegaeon/paracloud/scripts/common.sh"
require_compute
load_toolchains
root="${KVCACHED_DATA_ROOT:-$HOME/run/kvcached-paracloud}"
repo="${KVCACHED_REPO_ROOT:-$HOME/kvcached/kvcached}"
deploy="${KVCACHED_DEPLOY_ROOT:-$HOME/kvcached/paracloud}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib/python3.10/site-packages/torch/lib:$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset ENABLE_KVCACHED KVCACHED_AUTOPATCH
mkdir -p "$root" "$deploy/state" "$deploy/logs"
exec 9>"$deploy/state/prepare.lock"
flock --nonblock 9
if [[ ! -x "$root/env/bin/python" ]]; then
    "$ENV_PREFIX/bin/python" -m venv --system-site-packages "$root/env"
fi
export PATH="$root/env/bin:$ENV_PREFIX/bin:$PATH"
python -c 'import sqlite3, torch; print(torch.__version__, torch.version.cuda, sqlite3.sqlite_version)'
python -m pip install --no-index --find-links "$root/wheels" posix_ipc wrapt
python -m pip install --no-deps --no-build-isolation -e "$repo"
python -c 'import kvcached._C, wrapt, posix_ipc; print("kvcached extension imported")'
cd "$repo"
python -m pytest -q tests/test_dp_namespace.py
python -m pip freeze > "$deploy/state/environment-$SLURM_JOB_ID.txt"
git rev-parse HEAD > "$deploy/state/source-$SLURM_JOB_ID.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$deploy/state/build-ready"
