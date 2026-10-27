# Cross-model attention-DP / KV-headroom research

This user fork develops on `main` with **native vLLM + kvcached** as the research
data plane. This directory owns the idea, protocols, experiment matrix, reports
and execution code. Aegaeon is a separate mechanism/model-switching reference;
no current research documents live in that repository.

## Read in this order

Start with [the research summary](reports/research-summary.md) for the idea,
current admission/return ILP, verified results for both tenants and combined
metrics, limitations, and the running experiment. Individual reports below
retain their original settings and negative results.

Current follow-up: [Conv pressure target 1.2](docs/workaware-pressure-1p2-protocol.md)
raises only Conv to `.030863253171591748RPS` from measured physical KV pressure;
Code stays `.0005RPS` with unchanged arrival marks. See the protocol for the
fixed five-policy trace and measured-pressure/queue verification requirements.

| Question | Document |
| --- | --- |
| Idea, current algorithm and consolidated experimental conclusions | [reports/research-summary.md](reports/research-summary.md) |
| Motivation settings, results and interpretation | [reports/motivation-summary.md](reports/motivation-summary.md) |
| CUDA-graph main experiment | [reports/graph-results.md](reports/graph-results.md) |
| Placement idea and required migration data plane | [docs/placement-design.md](docs/placement-design.md) |
| Sparse-aware admission algorithm and cross-model example | [docs/sparse-scheduler.md](docs/sparse-scheduler.md) |
| Rank enumeration, calibrated surrogate and staged ILP | [docs/ilp-scheduler-design.md](docs/ilp-scheduler-design.md) |
| Release-triggered ILP Conv rebalancing: implementation and experiment boundary | [docs/release-rebalance-protocol.md](docs/release-rebalance-protocol.md) |
| Verified A800 six-policy release rebalancing: tail/SLO and negative incremental benefit | [reports/release-rebalance-status.md](reports/release-rebalance-status.md) |
| Cost-amortized ILP: migration budget and inference-service tradeoff | [docs/amortized-rebalance-protocol.md](docs/amortized-rebalance-protocol.md) |
| Verified A800 seven-policy cost-amortized ILP: less control, no incremental TTFT gain | [reports/amortized-rebalance-status.md](reports/amortized-rebalance-status.md) |
| Online headroom-preserving ILP and declared tail-TTFT experiment | [docs/headroom-tail-protocol.md](docs/headroom-tail-protocol.md) |
| Headroom-tail six-policy results, tail tradeoffs and independent verification | [reports/headroom-tail-status.md](reports/headroom-tail-status.md) |
| Lower Conv arrival rate toward 1.1 local KV pressure: paired six-policy experiment | [reports/pressure-ratio-status.md](reports/pressure-ratio-status.md) |
| Next tail stage: true packing feasibility and selective-transaction design (not implemented) | [docs/tail-next-stage-design.md](docs/tail-next-stage-design.md) |
| Intermediate graph-imbalance calibration protocol | [docs/curve-protocol.md](docs/curve-protocol.md) |
| Measured intermediate graph curves and fit limits | [reports/curve-results.md](reports/curve-results.md) |
| Current calibration and scheduler validation status | [reports/ilp-calibration-status.md](reports/ilp-calibration-status.md) |
| Live DP/KV migration prototype and experiment limits | [docs/live-placement.md](docs/live-placement.md) |
| Verified live migration/admission and remaining limits | [reports/live-placement-status.md](reports/live-placement-status.md) |
| Actual allocator controls / B-to-A capacity witness | [reports/allocator-pressure-status.md](reports/allocator-pressure-status.md) |
| End-to-end four-policy paired benchmark | [docs/benefit-protocol.md](docs/benefit-protocol.md) |
| Distinct-placement confirmation and late-arrival boundary | [docs/benefit-confirmation-protocol.md](docs/benefit-confirmation-protocol.md) |
| Longer-decode migration-cost amortization | [docs/benefit-long-decode-protocol.md](docs/benefit-long-decode-protocol.md) |
| End-to-end benchmark progress | [reports/benefit-status.md](reports/benefit-status.md) |
| Verified prototype benefit conclusions and limits | [reports/benefit-results.md](reports/benefit-results.md) |
| Primary long-KV admission despite sufficient global capacity | [docs/benefit-long-kv-protocol.md](docs/benefit-long-kv-protocol.md) |
| Azure-length Poisson trace and current 32K configuration | [docs/azure-poisson-trace-design.md](docs/azure-poisson-trace-design.md) |
| First real 32K Poisson-trace four-policy comparison | [docs/trace-benefit-protocol.md](docs/trace-benefit-protocol.md) |
| Verified 32K Poisson-trace negative result and control-cost diagnosis | [reports/trace-benefit-results.md](reports/trace-benefit-results.md) |
| Complete Azure/LongBench mixed-workload length profile and 1M limits | [reports/mixed-workload-profile.md](reports/mixed-workload-profile.md) |
| Verified 100K coding-agent history inputs from SWE-smith | [reports/agentic-100k-dataset.md](reports/agentic-100k-dataset.md) |
| Authorized 128K Azure Conv / SWE-smith trace and rate pilot | [docs/agentic-trace-protocol.md](docs/agentic-trace-protocol.md) |
| Verified A800 100K admission, tail latency/SLO tradeoffs and ILP cost diagnosis | [reports/agentic-trace-results.md](reports/agentic-trace-results.md) |
| Paired host-file / device-collective real-KV cost pilot | [docs/transfer-cost-protocol.md](docs/transfer-cost-protocol.md) |
| Default32 preplan / inside-pause mixed-population 100K admission cost | [docs/admission-cost-protocol.md](docs/admission-cost-protocol.md) |
| Complete small-population attention-DP cost lookup | [docs/layout-cost-protocol.md](docs/layout-cost-protocol.md) |
| Frozen measured boundary/placement trace control | [docs/measured-control-trace-protocol.md](docs/measured-control-trace-protocol.md) |
| Candidate-level safety revalidation on the unchanged trace | [docs/candidate-revalidation-protocol.md](docs/candidate-revalidation-protocol.md) |
| Independent and timeout-fair no-migration queue baselines | [docs/queue-baselines-protocol.md](docs/queue-baselines-protocol.md) |
| Long-resident synthetic Conv / 100K Code stress: two settings verified, rate sweep running | [docs/long-resident-conv-stress-protocol.md](docs/long-resident-conv-stress-protocol.md) |
| KV-only 50% latency threshold and conditional CPU design envelope | [reports/stress-trace-design-analysis.md](reports/stress-trace-design-analysis.md) |
| Long-input Code arrival with resident Conv migration: next trace | [docs/mixed-workload-trace-design.md](docs/mixed-workload-trace-design.md) |
| Measurement protocol and planned B/C scenarios | [docs/motivation-protocol.md](docs/motivation-protocol.md) |
| Machine-readable experiment status/configuration | [docs/experiment-matrix.json](docs/experiment-matrix.json) |
| Work-aware soft-headroom ILP / KV return: implementation and calibration in progress | [docs/workaware-return-protocol.md](docs/workaware-return-protocol.md) |
| Verified five-policy work-aware trace: modest Conv gains, combined P95 regression | [reports/workaware-return-results.md](reports/workaware-return-results.md) |
| kvcached / Prism mechanism and residency review | [docs/kvcached-prism-review.md](docs/kvcached-prism-review.md) |
| Dual-tenant residency / independent-client baseline | [reports/dual-qwen-results.md](reports/dual-qwen-results.md) |
| Historical eager imbalance curve | [reports/motivation-results.md](reports/motivation-results.md) |
| Historical eager length / batch-KV controls | [reports/length-results.md](reports/length-results.md), [reports/batch-results.md](reports/batch-results.md) |

CUDA graph is the main serving baseline; eager results remain historical
controls. Per the 2026-10-19 user decision, subsequent calibration and placement
experiments retain the existing sparse capture sizes `1,8,16,32,64,128,256`;
do not add dense capture sizes. Completed dense results remain historical
mechanism-ablation evidence, not scheduler calibration for this baseline.
Measured motivation is single-model, not cross-model KV interference
or placement benefit. Detailed numbers, confidence intervals and validity
checks live only in reports, not README copies.

The authorized long-resident Conv / 100K Code iteration uses `stress-sweep.sbatch`
and the sequential `stress_campaign.py`; the executable grid, heldout seeds and
validity boundaries are declared in `docs/long-resident-conv-stress-protocol.md`.
Recover operations from `../../../evidence/kvonly-stress-campaign-20261024/campaign-state.json`.
Job233056 verifies small dual-tenant GPU migration and ILP admission, not
cross-tenant capacity handoff or a performance advantage over live controls.

New live serving defaults to `max_model_len=32768` (prompt plus output), with
`--max-model-len` on both Python runners or `DP_OPT_MAX_MODEL_LEN` on launchers.
Sparse captures and the 4096-token per-step budget remain unchanged. Each
model/rank defaults to 4 GiB virtual KV capacity, not eagerly mapped physical
KV or a change to the controlled aggregate budget. Historical 4K results and
their explicit reproduction settings remain historical, not 32K validation.

The next heterogeneous scenario uses long-input Code arrivals and longer-resident
Conv requests as migration candidates; Code need not decode for a long time.
The 2026-10-21 full-dataset profile and ILP witnesses are local CPU evidence only.
Analysis up to 1M tokens does not change the 32K serving default, KV pools or
sparse graphs; current unsharded dual-tenant memory cannot serve a 1M request.

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

KV control-cost pilots: `agentic-transfer.sbatch` (host/DP-device+CPU SHA),
`agentic-echo.sbatch` (CPU SHA/exact GPU bytewise echo), and
`agentic-boundary.sbatch` (native keep/default32 consensus-boundary pause with
the same exact echo). See `docs/transfer-cost-protocol.md`; these do not alone
establish mixed-trace placement benefit.

| Stage | Entry |
| --- | --- |
| Isolated preparation/build | `prepare.sbatch` |
| Two resident tenants / correctness / concurrency | `dual-qwen.sbatch` → `run-dual-qwen.sh` → `benchmark.py` |
| Two independent clients | `DP_OPT_BENCHMARK_MODE=independent` + `dual-qwen.sbatch` → `independent_benchmarks.py` |
| Main graph motivation | `motivation-graph.sbatch` → `graph_sweep.py` → `motivation.py` |
| Intermediate graph curve, sparse only | `motivation-curve.sbatch` → `motivation.py`; offline `fit_imbalance.py` |
| CPU-only rank-enumerated ILP algebra reference | `ilp_reference.py` + `docs/ilp-example.json` (optional SciPy/HiGHS) |
| Complete Azure / exact Qwen LongBench length profiling, local CPU | `workload_profile.py` (LongBench requires optional isolated `tokenizers`) |
| Reconstruct 100K agent inputs at whole assistant-call boundaries | `agentic_profile.py` (optional isolated `pyarrow`, `jinja2`, `tokenizers`) |
| Real-agent prompt bank / independent Azure Conv and Code streams | `agentic_trace.py` → `agentic-trace.sbatch` (A800 capacity pilot, open-loop calibration, automatic freeze, four policies) |
| A800 sampled headroom / full-input page fit / paired-length rate bracket | `agentic_capacity.py` (guarded pilot estimates, not a peak-memory or benefit certificate) |
| Same-A800 real-KV transport correctness / cost, not trace benefit | `agentic-transfer.sbatch` → `transfer_pilot.py`; independent `verify_transfer_pilot.py` |
| Finite-population pilot RPS extrapolation with paired memory/output correlation | `agentic_rate.py` (not a steady-state or benefit guarantee) |
| Independent calibration replay, journals, allocation inventory and queue/guardrail verification | `verify_agentic_calibration.py` (local archive analysis; never imports vLLM) |
| Verified calibration control intervals / Code-prefill overlap / allocator waiting | `analyze_agentic_calibration.py --inputs <inputs> --output <json>` (derived RPC/wait intervals, not kernel profiling) |
| Reapply declared Code residency target to complete A800 measurements | `recalibrate_agentic_rates.py --inputs <inputs> --output <json>` (Code rate only; no policy-outcome selection) |
| Five-minute query with optional terminal archive / verification | `watch_trace_job.py <job> --archive <dir> --agentic-inputs <inputs>` (local; stops on failure, no auto-resubmit) |
| Heterogeneous Poisson offered-state page geometry / offline ILP | `mixed_workload_envelope.py` (not a GPU or queue simulator) |
| Long-output stress setting / service and control-cost sensitivity | `stress_trace_envelope.py --sensitivity` (conditional single-Code CPU proxy, not GPU benefit) |
| Build labelled layout-cost tables from calibrated curves | `calibrated_cost.py` (single-model surrogate, not live placement) |
| Sparse-calibrated synthetic admission snapshot | `calibrated_cost.py --snapshot docs/sparse-admission-example.json` → `ilp_reference.py` |
| Live dual-tenant DP/KV migration and admission prototype | `live-placement.sbatch` → `live_benchmark.py` (GPU functionality verified; limits in reports) |
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

## 端到端收益阶段

长KV容量准入主场景结果见`reports/benefit-long-kv-results.md`：总空间够、当前任一卡
均不能放新请求、跨模型迁移后准入；保留老请求latency/SLO退化及受控预算边界。
默认32step六配对确认也已归档验证；不把相同full-reservation scheduler的no_move称作
未修改native-vLLM动态KV调度，不把更早准入概括为全请求SLO收益。

32K Azure长度sample+两独立Poisson stream首轮实跑已完成并独立复核，**ILP净收益为负**。
完整对照和诊断见`reports/trace-benefit-results.md`；此前理论设计保留在
`docs/azure-poisson-trace-design.md`。尚未完成多seed/多rate最终泛化实验；下一阶段需
wait-vs-migrate、轻量控制和小batch联合cost，而非从机制收益直接外推。

四策略双模型配对协议见 `docs/benefit-protocol.md`；执行入口
`DP_OPT_LIVE_MODE=benefit sbatch experiments/dp_opt/live-placement.sbatch`。
容量不足的baseline继续排队，全部15请求完成，控制/复制成本计入原始指标；
`verify_benefit.py`独立复算，不把容量功能验证当作性能收益。
