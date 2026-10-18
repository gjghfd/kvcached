#!/usr/bin/env bash
#SBATCH --job-name=kvcached-dual-qwen
#SBATCH --partition=gpu_a800,gpu_h100,gpu_h200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=01:00:00
#SBATCH --time-min=00:30:00
exec bash "${KVCACHED_REPO_ROOT:-$HOME/kvcached/kvcached}/experiments/dp_opt/run-dual-qwen.sh"
