# Verified graph-mode attention-DP imbalance motivation

本报告属于 kvcached 研究 `reports/`；执行脚本在上一级，`evidence/` 路径相对于外层 ParaCloud。

## Outcome

CUDA graph is the intended main serving baseline for this motivation; eager
runs are historical controls. Settings, protocol, all results and boundaries
are consolidated in the Chinese `motivation-summary.md`.

The >=30% stable-decode throughput-loss target is verified for **standard
CUDA-graph DP execution**, not for unpadded eager attention alone. All results
use real Qwen3-30B-A3B-Instruct-2507 BF16 weights and requests. No artificial
delay, request failures, prefill-only metric substitution or timing rewrite
is used.

Job **229748** completed `COMPLETED/0:0` on2026-10-18,
20:34:13–20:53:23 CST, `d1n41a14g01`, two A800-SXM4-80GB GPUs. The allocation
passed236 regression tests; local client/control tests:43 passed, with only
the known `asyncio_mode` warning. The job exited and no endpoint remains.

## Configuration and protocol

- Single resident Qwen tenant; attention TP1, DP2, EP2, PP1, BF16 weights.
- FlashAttention2, Triton unquantized MoE, `allgather_reducescatter`.
- FULL_DECODE_ONLY graphs, compilation mode0; torch.compile disabled.
- Capture sizes1/8/16/32/64/128/256; all seven FULL graphs captured successfully.
- Maxlen4096, maxseq256, maxbatchedtokens4096,44GiB virtual KV/rank.
- No DBO/EPLB/prefix caching/async scheduling/sleep/CPU KV offload.
- Input1024/output512 exact tokens; total requests128,256,32 in declared order.
- Three fresh paired repeats per case; balanced-start and balanced-end controls
  bracket both shuffled extreme orientations. Prompts are identical across
  layouts within a case, rotating rank assignments between repeats.
- Primary loss=`1 - 1 / paired_cadence_ratio`. Three repeat ratios, not
  individual steps, form Student-t(df2)95% intervals. The target requires
  the ratio-derived loss lower bound>=30% in **both** hot-rank orientations.
- Worker traces identify constant-population one-token decode, trim32 steps
  per end and retain>=128 steps/rank. Metrics reject waiting/preemption/KV>=90%
  in that window. KV must reclaim to the same idle backing between trials.
- Separate untraced balanced bursts check instrumentation sensitivity; no
  phase hooks or per-step file I/O occur in the measured serving path.

## Results

Cadence is rank-host execute-start spacing in verified stable decode, not an
isolated attention-kernel CUDA duration. Burst throughput additionally includes
prefill/ramp-up and is not the primary target metric.

| Total requests | Layout | Stable cadence(ms) | Paired slowdown | Decode throughput loss |95% loss interval | Burst loss |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
|128|64:64|24.57|1.0000x|reference|reference|reference|
|128|128:0|36.33|1.4785x|**32.37%**|**31.78–32.94%**|33.06%|
|128|0:128|36.25|1.4751x|**32.21%**|**30.71–33.65%**|32.73%|
|256|128:128|32.88|1.0000x|reference|reference|reference|
|256|256:0|53.48|1.6266x|**38.52%**|**38.16–38.88%**|39.33%|
|256|0:256|53.33|1.6222x|**38.35%**|**37.90–38.80%**|39.17%|
|32|16:16|15.44|1.0000x|reference|reference|reference|
|32|32:0|21.57|1.3967x|28.40%|22.10–33.76%|28.44%|
|32|0:32|21.92|1.4196x|29.56%|26.71–32.19%|29.50%|

Both larger cases meet the predeclared criterion; batch32 does not. Preserve
the smaller case instead of displaying only target-reaching results.

## Raw verification

All **36 trials /4992 requests /2,555,904 output tokens** reproduce from the
downloaded traces exactly, including stable-window boundaries and paired
summary ratios. Each request has1024 prompt tokens and512 output token IDs.
The same GPU pair serves all three cases; each worker holds64/128 experts in
all48 layers with TP/PP1 and DP/EP2. No in-window waiting or preemption occurs.

| Total requests | Minimum stable steps/rank | Maximum sampled KV usage | Maximum absolute baseline endpoint drift | Traced/untraced balanced TPOT ratio |
| ---: | ---: | ---: | ---: | ---: |
|128|412|47.4515%|0.23%|1.00005|
|256|379|87.1325%|0.24%|1.00316|
|32|437|17.0656%|0.73%|0.99409|

All pools restore96MiB/rank mapped idle backing before/after every trial;
reserved backing remains disabled. The TPOT controls are sequential and also
include drift; their ratios are sensitivity checks, not an isolated overhead
measurement or proof that skewed layouts have zero client/trace cost.

## Mechanism and limits

The installed vLLM0.26.0 `v1/worker/dp_utils.py` enables DP max-rank padding
when synchronized graph mode is nonzero. Trace evidence verifies:

- Balanced batch128: unpadded64/64, padded64/64.
- Extreme128:0: unpadded128/1(dummy), padded128/128; reversed for0:128.
- Batch256 follows the same128/128 to256/256 pattern.

AgRs obtains communication lengths from forward-context DP metadata. Padding
therefore changes graph/communication/expert work, not merely how independent
attention requests are assigned. This experiment measures the **system cost
of request imbalance under standard graph-mode DP/EP**, which is relevant to
placement, but cannot identify a30–38% isolated attention-kernel share or
prove attention alone obeys linear batch scaling.

An empty rank still participates in model/EP execution. Its ordinary dummy
decode metadata sets the first sequence length to1, the remaining padded
sequence lengths to0, and all KV slot mappings to-1 to skip writes. FULL replay
launches attention, but does not replicate the hot rank's real long-context KV
attention or allocate dummy request KV. Projections/router process padded
shapes and AgRs uses padded communication lengths; exact attention launch/grid
costs and per-kernel contributions remain unprofiled. Installed-source excerpts
are archived in `evidence/kvcached-batch-20261018/graph-empty-rank-source.txt`.

The separate eager sweep's fresh batch256 confirmations lose8.16%/6.00%; its
batch128/context2048 confirmations lose5.14%/2.61%, below30%. Eager and graph
curves must not be pooled. Different allocations and maxlen also prevent
claiming the absolute cadence difference as a paired causal graph speedup.
The results are consistent with substantial eager launch/control overhead
masking imbalance, but do not prove HBM saturation or memory-bound execution.
See `batch-results.md` and `length-results.md` for prior negative evidence.

For the placement idea, use configuration-dependent measured DP costs alongside
rank-local KV headroom. The next mechanism checks are a same-node eager/graph
comparison, intermediate imbalance levels with declared capture-size rounding,
and low-perturbation kernel-active/launch-gap/expert-occupancy measurements.
Request migration and placement gains remain unimplemented/unmeasured.

## Evidence and rerun

Local:`evidence/kvcached-graph-229748/`, including `all-cases-verification.txt`,
all client/trial/trace JSON, CSV/SVG, source/hardware hashes, GPU timeline,
worker records, service logs, and final accounting. Remote:
`~/kvcached/paracloud/state/dual-qwen-229748/` through the run-volume symlink.
Operations and quota migration:`evidence/kvcached-batch-20261018/`.

```bash
ssh paracloud 'root="$HOME/kvcached/paracloud"; sbatch --output="$root/logs/graph-%j.out" --error="$root/logs/graph-%j.err" "$HOME/kvcached/kvcached/experiments/dp_opt/motivation-graph.sbatch"'
```

First inspect the recorded job marker/accounting to avoid duplicate live jobs.
All inference/imports/tests run on Slurm compute nodes; login nodes perform
only scheduler queries, file reads and synchronization. Unrelated223200 was
not changed. No commit/push was made in this experiment.
