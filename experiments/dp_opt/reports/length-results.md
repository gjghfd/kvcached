# Long-token attention-DP motivation: job 227012

本报告属于 kvcached 研究 `reports/`；执行脚本在上一级，`evidence/` 路径相对于外层 ParaCloud。

## Verified run

Job **227012** completed on **2026-10-18 00:11:28–01:07:25 CST**,
`COMPLETED/0:0`, node `d1n41a28g03`, two **A800-SXM4-80GB** GPUs. One BF16
Qwen3-30B-A3B-Instruct-2507 service, attention TP=1 / DP=2 / EP=2. All three
length cases use this same allocation, checkpoint and GPU pair. They run
sequentially, not in randomized length order, so comparisons between lengths
also contain time drift; paired layout comparisons stay within each case.

Runtime remains eager Triton MoE / allgather-reducescatter, no DBO, EPLB, APC
or async scheduling. Each rank has 24 GiB virtual KV, 16,384 blocks of 16
tokens, max-model-len 10,240, max-num-seqs 32 and a 512-token scheduling budget.
Each length has three repetitions of two extreme layouts bracketed by 8:8
controls: 12 main trials and three separate diagnostic trials per case.

- **36 main trials, 576 successful requests, 442,368 output tokens**, no failures.
- Nine diagnostic trials add 144 requests / 110,592 output tokens, excluded
  from the primary results. Correctness, warmup and untraced controls are
  excluded from both counts.
- Allocation-side regression: **225 tests passed** in 6.16 seconds.
- All main and diagnostic stable windows were rechecked from raw traces:
  correct populations and token counts, no waiting or preemption, one token
  per SSE chunk, and mapped KV reclaimed before the next trial.

## Measured performance

Decode loss is `1 - 1 / paired_cadence_slowdown`. Burst loss compares mean
full-request-span output throughput with the case's balanced reference. These
are different windows: burst throughput includes admission, chunked prefill,
decode ramp-up and completion, not just constant-population decode.

| Input/output | DP layout | Stable cadence ms | Paired decode loss | Burst output token/s | Burst loss | Mean TTFT s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 128/256 | 8:8 | 69.138 | reference | 227.987 | reference | 0.183 |
| 128/256 | 16:0 | 70.612 | 2.09% | 221.254 | 2.95% | 0.247 |
| 128/256 | 0:16 | 71.499 | 3.30% | 219.169 | 3.87% | 0.250 |
| 128/1024 | 8:8 | 67.920 | reference | 234.186 | reference | 0.167 |
| 128/1024 | 16:0 | 69.139 | 1.78% | 229.506 | 2.00% | 0.247 |
| 128/1024 | 0:16 | 70.695 | 3.93% | 225.130 | 3.87% | 0.246 |
| 8192/1024 | 8:8 | 68.510 | reference | 207.378 | reference | 4.940 |
| 8192/1024 | 16:0 | 70.937 | 3.42% | 179.431 | 13.48% | 9.626 |
| 8192/1024 | 0:16 | 70.951 | 3.44% | 179.178 | 13.60% | 9.758 |

The 95% paired cadence ratio intervals for 16:0 / 0:16 are:

- 128/256: 0.9856–1.0571 / 1.0230–1.0454.
- 128/1024: 0.9948–1.0414 / 1.0251–1.0566.
- 8192/1024: 1.0085–1.0624 / 1.0113–1.0599.

Thus the first two 16:0 intervals include no slowdown. Only three paired
repeats were taken, and balanced start/end drift reaches 3.43% in output_long
and 2.34% in input_output_long. Small differences should not be overinterpreted.

## Interpretation and validity

Increasing output length alone does not materially amplify the observed
imbalance penalty. Increasing input to 8,192 tokens makes the **full-burst**
loss substantially larger, approximately 13.5%, while stable decode loss
remains approximately 3.4%. Mean TTFT nearly doubles, 4.94 to 9.63–9.76 seconds.
Mean maximum request TTFT grows from 8.79 to 18.68–18.85 seconds; first-token
arrival spread grows from 7.67 to 17.53–17.69 seconds. Burst makespan grows
from 79.01 to 91.32–91.44 seconds. This supports a prefill/ramp-up contribution
to the larger end-to-end penalty, not an isolated attention-kernel attribution.

KV capacity does not explain these losses: main-trial maximum usage is
2.35%, 7.04% and 55.52% respectively. Mapped backing peaks are 672, 1,824 and
13,728 MiB per rank; reserved backing stays zero. Verified common windows
retain 185–187, 953–955 and 713–844 steps respectively. Long-prefill ramp-up
shortens the common population window, but it still exceeds the 128-step
minimum. Sampled stable contexts in the long case span 8,225–9,181 tokens,
with approximately 8,703 computed tokens on average in the extreme layouts.

### Phase diagnostics are not reliable isolation

Diagnostics completed and confirm actual collective sizes `[8,8]`, `[16,1]`
and `[1,16]`; empty ranks still run dummy model/EP work. Full-attention,
attention-core, MoE and dispatch/combine ranges cover all 48 layers. However:

- Sampled steps' median host execution is about **9.5–19.2% longer** than
  unsampled steps within the same diagnostic trial. Event insertion and hooks
  introduce substantial local perturbation despite sampling only every 32 steps.
- Sampled worker CUDA ranges are approximately 81–87 ms, while unprofiled
  primary cadence is approximately 68–71 ms. These are different statistics
  and trials, not interchangeable denominators.
- Balanced ranks have markedly different module splits despite identical
  8-request populations. For 8192/1024, attention ranges are 27.14 / 16.82 ms,
  while dispatch ranges are 5.21 / 24.64 ms. Stream gaps and collective wait
  are charged to different ranges; these cannot identify intrinsic GPU work.

Consequently, the raw `linear_half_share_prediction` and
`measured_attention_delta_fraction` fields are **diagnostic estimates only**.
They must not be presented as measured idle-GPU fractions or causal throughput
predictions. The user's `0.5*A_skew/T_skew` hypothesis still requires a
low-perturbation critical-path measurement and attention batch-scaling check.
Likewise, the communicator source's equal/unequal NCCL path change is verified
as a mechanism difference, not quantitatively isolated as the ~3–4% penalty.

The next profiling experiment should separate GPU kernel-active time, CPU
launch gaps and collective waits, using sparse or layer-subset tracing with
an explicit perturbation control. Placement should account for prefill/ramp-up
as well as steady decode and live KV. Request/KV migration is not implemented
or validated by this experiment.

## Small-batch bandwidth hypothesis

The user's hypothesis is plausible and should be tested before concluding
that attention inherently occupies only a few percent of decode time.
At 16 global decode tokens, local attention batch is eight when balanced.
Qwen has 128 routed experts and selects eight per token: across the EP group
there are only 128 token-expert assignments per layer, averaging one token
per expert (including inactive experts). Actual routing may be concentrated;
per-expert occupancy and padded GEMM shapes have not been measured.

From the checkpoint dimensions, BF16 attention QKV/output projections contain
36 MiB of weights per layer, 1.6875 GiB across 48 layers. With weights ideally
read once and dominating activation traffic, arithmetic intensity is roughly
`B` FLOP/byte: only about 8 or 16 for these local batches. Doubling tokens
need not double weight traffic and can improve matrix reuse/utilization.
Each expert's three matrices contain 9 MiB, and 64 local experts across all
layers contain 27 GiB. Expert batches around one token provide poor weight
reuse even though the model has only approximately 3B active parameters/token.
An independent-uniform routing model predicts approximately 41.2 distinct
local experts touched by 16 tokens, or 17.4 GiB of ideal one-read local expert
weights per step. This is a conditional illustration, not measured traffic:
the prompts are related and routing may be highly correlated.

GQA has only four KV heads for 32 query heads. Unique BF16 K+V storage is
96 KiB per request/context-token across 48 layers. At approximately 8,703
context tokens, ideal unique KV reads are 6.37 GiB for eight requests and
12.75 GiB for 16. Actual HBM transactions depend on kernel tiling, reuse and
reloads; this does not establish bandwidth saturation. Runtime selected
FlashAttention v2 and Triton MoE, with CUDA graphs and compilation disabled.
Tiny expert GEMMs, launches and EP collectives can therefore limit cadence
even without the HBM interface running at its bandwidth limit.

Three different mechanisms must be kept separate:

- **Projection/expert weight bandwidth:** batch growth amortizes weight reads;
  MoE remains EP-distributed, so moving attention homes does not remove its
  dominant shared cost. This can make layout sensitivity small.
- **KV-read bandwidth:** each independent request has distinct KV. If the
  attention kernel already saturates HBM, concentrating twice the KV reads
  on one GPU should increase local attention time rather than hide imbalance.
- **Launch/low-utilization/collective stalls:** small kernels and eager gaps
  can make duration relatively flat with batch; this is not equivalent to
  measured memory-bandwidth saturation.

Thus a memory-bound whole model does not imply memory-bound attention core,
and memory-bound attention alone does not imply negligible imbalance cost.
The precise layout-sensitive quantity is `A(16)-A(8)`, together with changes
to collective/dense stages, not simply the absolute attention share.
The existing phase diagnostics cannot separate these mechanisms because
their launch overhead and wait attribution are substantial.

A controlled follow-up should first sweep global batch 16/32/64/128 at a
fixed short input/output length (for example 1024/256), comparing balanced
and both extreme layouts. Sequence slots and KV budget must be increased
uniformly and checked for the hottest rank; 8K context at batch 128 is not
feasible under the current memory budget. Then vary context at a fixed
feasible batch. Measure expert occupancy, kernel-active/launch-gap time,
DRAM bandwidth and Tensor Core activity with perturbation controls. Use a
separate eager-versus-CUDA-graph control only if the kvcached path supports it;
do not silently change execution mode in the main comparison. This follow-up
now has an implemented batch/KV sweep. Job229639 completed seven screens
before a separate home-quota write failure; resume229747 and independent
graph control229748 were submitted on 2026-10-18. See the experiment README
for current progress. Hardware-counter attribution remains unimplemented.

Dimension-based estimates and runtime source evidence are saved as
`decode-traffic-estimates.txt`, `model-config.json` and
`decode-source-evidence.txt` in the local evidence directory. Arithmetic
intensity reasoning follows NVIDIA's *Matrix Multiplication Background User's
Guide*; a fetched copy is `nvidia-matmul-performance.html` there. These are
analytical hypotheses, not a replacement for hardware-counter measurements.

## Evidence and operations

Local raw evidence: `evidence/kvcached-lengths-227012/`, including all results,
trials, traces, diagnostics, CSV/SVG, worker metadata, source hashes, logs and
final accounting. Rechecks are recorded in `verification-analysis.txt`,
`diagnostic-sensitivity.txt` and `prefill-rampup-analysis.txt` in that directory.
Remote evidence: `~/kvcached/paracloud/state/dual-qwen-227012/`.

The job exited and has **no online endpoint**. Monitor delivery failed after
the run; Slurm accounting and exit marker confirm experiment success.
Shutdown logs contain worker-socket preservation and multiprocessing resource
tracker warnings; these are separate from benchmark validity, and no global
cleanup was performed. Old unrelated pending job 223200 was not modified.
