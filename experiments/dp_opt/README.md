# Cross-model attention-DP / KV-headroom research

This user fork develops on `main` with **native vLLM + kvcached** as the research
data plane. This directory owns the idea, protocols, experiment matrix, reports
and execution code. Aegaeon is a separate mechanism/model-switching reference;
no current research documents live in that repository.

## Read in this order

| Question | Document |
| --- | --- |
| Motivation settings, results and interpretation | [reports/motivation-summary.md](reports/motivation-summary.md) |
| CUDA-graph main experiment | [reports/graph-results.md](reports/graph-results.md) |
| Placement idea and required migration data plane | [docs/placement-design.md](docs/placement-design.md) |
| Measurement protocol and planned B/C scenarios | [docs/motivation-protocol.md](docs/motivation-protocol.md) |
| Machine-readable experiment status/configuration | [docs/experiment-matrix.json](docs/experiment-matrix.json) |
| kvcached / Prism mechanism and residency review | [docs/kvcached-prism-review.md](docs/kvcached-prism-review.md) |
| Dual-tenant residency / independent-client baseline | [reports/dual-qwen-results.md](reports/dual-qwen-results.md) |
| Historical eager imbalance curve | [reports/motivation-results.md](reports/motivation-results.md) |
| Historical eager length / batch-KV controls | [reports/length-results.md](reports/length-results.md), [reports/batch-results.md](reports/batch-results.md) |

CUDA graph is the main serving baseline; eager results remain historical
controls. Measured motivation is single-model, not cross-model KV interference
or placement benefit. Detailed numbers, confidence intervals and validity
checks live only in reports, not README copies.

## Deployment

Remote source: `~/kvcached/kvcached`; branch `main`; origin
`git@github.com:gjghfd/kvcached.git`, with the original project retained as
`upstream`. Base commit: `04ccd1da0653d251622122d0695d72bb288fb8ec`.
The isolated environment is `~/run/kvcached-paracloud/env`; checkpoint is
`~/run/aegaeon-paracloud/models/Qwen3-30B-A3B-Instruct-2507`.
State/logs live under `~/run/kvcached-paracloud/` with compatibility symlinks
from `~/kvcached/paracloud/{state,logs}`.

Cluster constraints and synchronization are maintained in the outer workspace
`ParaCloud/docs/cluster-operations.md`; ownership is in
`ParaCloud/docs/documentation-layout.md`. Do not overwrite protected Aegaeon
runtime patches. All imports/builds/tests/inference run in Slurm allocations;
login nodes only read/sync/query/download. Check recorded markers/accounting
before submitting a new job. Completed reports do not imply a live endpoint.

## Execution entries

Scripts remain at this directory's root; document moves do not alter runtime paths.

| Stage | Entry |
| --- | --- |
| Isolated preparation/build | `prepare.sbatch` |
| Two resident tenants / correctness / concurrency | `dual-qwen.sbatch` → `run-dual-qwen.sh` → `benchmark.py` |
| Two independent clients | `DP_OPT_BENCHMARK_MODE=independent` + `dual-qwen.sbatch` → `independent_benchmarks.py` |
| Main graph motivation | `motivation-graph.sbatch` → `graph_sweep.py` → `motivation.py` |
| Historical eager curve | `motivation.sbatch` |
| Historical eager length / batch-KV sweeps | `motivation-lengths.sbatch`, `motivation-batch.sbatch` |
| Regression | `validate.sbatch` |

```bash
ssh paracloud 'root="$HOME/kvcached/paracloud"; sbatch --output="$root/logs/prepare-%j.out" --error="$root/logs/prepare-%j.err" "$HOME/kvcached/kvcached/experiments/dp_opt/prepare.sbatch"'
ssh paracloud 'root="$HOME/kvcached/paracloud"; sbatch --export=ALL,DP_OPT_BENCHMARK_MODE=independent --output="$root/logs/independent-%j.out" --error="$root/logs/independent-%j.err" "$HOME/kvcached/kvcached/experiments/dp_opt/dual-qwen.sbatch"'
ssh paracloud 'root="$HOME/kvcached/paracloud"; sbatch --output="$root/logs/graph-%j.out" --error="$root/logs/graph-%j.err" "$HOME/kvcached/kvcached/experiments/dp_opt/motivation-graph.sbatch"'
```

The independent-client default is 16 requests/model/round (8 per DP rank),
128 input / 256 output tokens, three overlapping rounds, with correctness and
warmup excluded. Graph settings and the paired protocol are versioned in the
matrix and protocol, not inferred from these client defaults.

Raw archives stay in outer `ParaCloud/evidence/`; its [README.md](README.md) indexes jobs,
verification and document-migration evidence. Never rewrite archived JSON/logs
or their captured old source paths merely to match the new document layout.
