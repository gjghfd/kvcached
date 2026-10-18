# Single-model attention-DP imbalance: measured motivation

本报告属于 kvcached 研究 `reports/`；执行脚本在上一级，`evidence/` 路径相对于外层 ParaCloud。

## Half-attention-share hypothesis and follow-up

The extreme layout leaves one rank without real attention/KV work, not without
expert work. Under **linear attention scaling**, write skewed time as
`T_s=A_s+M` and balanced time as `T_b=A_s/2+M`. The throughput loss is exactly
`1-T_b/T_s=0.5*A_s/T_s`. If attention share is instead defined on the balanced
run, `f=A_b/T_b`, the same loss is `f/(1+f)`, with step slowdown `1+f`.

More generally, if only attention time changes, loss is `(A_s-A_b)/T_s`.
An idle peer alone does not establish `A_s=2*A_b`: request count, arithmetic
work and kernel duration need not scale identically. The dummy path also
executes a one-token model forward and its local projections, while real
attention metadata and KV writes are absent in this eager configuration.

For the earlier 0:16 paired cadence ratio 1.036608, the implied steady-state
throughput loss is `1-1/1.036608 = 3.5315%`. Under the ideal linear model only,
this would imply a skewed attention fraction of approximately 7.063%.
This is **not a measured attention fraction**; the full-burst throughput uses
a different window and must not be substituted into a decode decomposition.

Job **227012** completed on **2026-10-18 00:11:28–01:07:25 CST**, two A800s,
`COMPLETED/0:0`: 36 main trials, 576 requests and 442,368 output tokens passed.
All lengths share one GPU pair and a 24 GiB virtual KV budget/rank. At
8192/1024, extreme-layout burst throughput losses grow to **13.48–13.60%**, but
stable decode losses remain **3.42–3.44%**. Mean TTFT rises from 4.94 to
9.63–9.76 seconds, indicating a larger prefill/ramp-up contribution.
Full results and confidence intervals are in `length-results.md`.

Separate CUDA diagnostics passed, but sampled-step host overhead is about
9.5–19.2%, and range splits include launch gaps/collective wait. They cannot
yet verify the half-attention-share prediction or isolate kernel costs.
Raw diagnostic prediction fields must not be promoted to causal measurements.

The installed vLLM communicator source also reveals a discrete path change:
equal per-rank token counts use `ncclAllGather` / `ncclReduceScatter`, while
unequal counts use grouped per-root `ncclBroadcast` / `ncclReduce` calls.
This is consistent with a possible fixed penalty at any nonzero imbalance,
but source inspection alone cannot attribute the earlier ~4% cadence step.
The new independent diagnostics time dispatch/combine inside MoE and record
their actual rank sizes to check this hypothesis. The copied runtime source
is in `evidence/kvcached-lengths-20261017/collective-source.txt`.

## Run and controls

Job **226683**, 2026-10-17 **21:10:08–21:21:57 CST**, `d1n41a18g02`, two
**NVIDIA A800-SXM4-80GB** GPUs, `COMPLETED/0:0`. Exactly one real BF16
Qwen3-30B-A3B-Instruct-2507 tenant, attention TP=1 / DP=2 / EP=2, eager Triton
MoE / allgather-reducescatter; DBO, EPLB, prefix caching, async scheduling,
sleep and CPU offload disabled. Per-rank virtual KV limit remains 2 GiB.

All layouts use the same 16 distinct 128-token prompts and force 256 output
tokens each. Only request-to-DP-rank assignment changes; prompt assignments
rotate between repetitions. Three repetitions, fixed-seed shuffled layouts,
with 8:8 controls at each repetition's beginning and end: 30 measured trials,
480 successful requests and 122,880 output tokens. Nine layout warmups and
three untraced balanced instrumentation controls are excluded from that count.

The primary timing is the median interval between worker execution starts in
a common, scheduler-verified constant-population decode window. This interval
includes host scheduling, launch and synchronization effects; it is **not** a
CUDA-event kernel duration or isolated attention time. For each rank, traces
verify all intended requests execute one decode token per step, without new or
resumed requests. Two ends are trimmed by at least 32 steps; 185–187 verified
steps per rank remain. Zero-request ranks still execute EP dummy passes.
Client TTFT/TPOT/ITL and full-burst throughput are auxiliary observations.

## Per-layout results

Each slowdown uses its own repetition's mean of the start/end 8:8 controls;
the three ratios are averaged. The 95% interval is Student-t with 2 degrees
of freedom, not a confidence interval across independent GPUs. Throughput is
the mean of complete-burst measurements, not stable-window kernel throughput.

| Rank0:Rank1 | I | Stable cadence ms | Slowdown | 95% paired ratio CI | Burst output token/s |
| --- | ---: | ---: | ---: | --- | ---: |
| 8:8 | 0.00 | 64.234 | 1.0000x | normalized reference | 244.741 |
| 10:6 | 0.25 | 66.682 | 1.0381x | 1.0315–1.0448 | 235.787 |
| 6:10 | 0.25 | 66.872 | 1.0411x | 1.0393–1.0428 | 235.391 |
| 12:4 | 0.50 | 66.645 | 1.0375x | 1.0321–1.0430 | 234.477 |
| 4:12 | 0.50 | 66.483 | 1.0350x | 1.0278–1.0422 | 235.659 |
| 14:2 | 0.75 | 66.641 | 1.0375x | 1.0229–1.0520 | 234.776 |
| 2:14 | 0.75 | 66.796 | 1.0399x | 1.0354–1.0443 | 234.441 |
| 16:0 | 1.00 | 67.122 | 1.0450x | 1.0421–1.0478 | 232.900 |
| 0:16 | 1.00 | 66.586 | 1.0366x | 1.0330–1.0402 | 235.322 |

Orientation-averaged slowdown is approximately 3.96%, 3.63%, 3.87% and 4.08%
at I=0.25, 0.5, 0.75 and 1. Burst throughput losses are 3.74%, 3.95%, 4.14%
and 4.34%, respectively. These averages do not remove the need to retain both
orientations: 16:0 and 0:16 differ, and the stable-cadence curve is not strictly
monotonic. The experiment does not establish a causal explanation for each
small difference between layouts.

## Validity checks

- Actual unpadded/padded decode batches match: 8/8, 10/6, 12/4 and 14/2 by
  their respective ranks. At 16:0, real/dummy batches are 16/1, not 16/16.
  Therefore max-rank padding is not the measured mechanism in this eager run.
- Every stable window retains the requested population; per-rank waiting is
  zero and cumulative preemptions do not increase. KV utilization peaks at
  29.7436%, below capacity pressure. Per-rank mapped physical backing peaks at
  672 MiB, reserved backing remains zero and returns to idle before each trial.
- 16:0 / 0:16 idle ranks record 186 dummy passes in their stable windows;
  the real rank retains 185 stable decode steps. The common time-window bounds
  can retain slightly different counts at each rank; the empty rank is not
  stopped or removed from EP.
- Balanced start/end cadence drift is +0.2476%, +0.4132% and +0.2917% for the
  three repetitions. The instrumentation control observes mean balanced TPOT
  of 64.561 ms untraced versus 64.639 ms traced, a +0.1209% difference.
  Controls are sequential, so this is sensitivity evidence, not an isolated
  measurement of tracing overhead.
- All stream chunks contain one output token. Worker traces are buffered in
  memory and written only after each trial; no per-token file writes or
  per-step CUDA synchronization were added for tracing.
- 218 allocation-side regression tests pass in 6.12 s; 25 local client tests
  pass. One known missing-pytest-asyncio configuration warning is nonfatal.

## Implication and limits

**The motivating observation is a measurable but modest nonzero compute cost
for concentrating request/KV homes in this configuration.** Even 16:0 does not
double stable cadence; total MoE token work remains distributed over the same
EP group. Conversely, imbalance is not free. A placement policy should use a
measured cost lookup or bounded prediction rather than request-ratio slowdown.

This result supports evaluating a tradeoff between rank-local KV headroom and
compute cost; it does **not** establish that any placement improves admission,
SLO or throughput. The experiment does not isolate attention from dense/MoE
work and communication. Larger contexts, larger active populations, CUDA
graphs, other communication kernels, and other GPUs require separate curves.
Three repeats on one allocated GPU pair are a motivation result, not broad
statistical characterization. Live KV migration is not implemented here.

## Evidence

All paths below are relative to the outer ParaCloud workspace:

- `evidence/kvcached-motivation-226683/accounting-final.txt`
- `evidence/kvcached-motivation-226683/state/dual-qwen-226683/motivation/results.json`
- `evidence/kvcached-motivation-226683/state/dual-qwen-226683/motivation/summary.csv`
- `evidence/kvcached-motivation-226683/state/dual-qwen-226683/motivation/motivation.svg`
- `evidence/kvcached-motivation-226683/state/dual-qwen-226683/motivation/trials/`
- `evidence/kvcached-motivation-226683/state/dual-qwen-226683/motivation/traces/`
- `evidence/kvcached-motivation-226683/state/dual-qwen-226683/motivation/fixture.json`
- `evidence/kvcached-motivation-226683/state/dual-qwen-226683/hardware-source.txt`

The remote state is `~/kvcached/paracloud/state/dual-qwen-226683/`, with matching
service and job logs. The job has exited and there is no online endpoint.
