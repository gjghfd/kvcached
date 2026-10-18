# Batch / KV imbalance follow-up

本报告属于 kvcached 研究 `reports/`；执行脚本在上一级，`evidence/` 路径相对于外层 ParaCloud。

## Completed evidence from job229639

On 2026-10-18, job229639 ran on `d1n41a22g03`, two A800-SXM4-80GB GPUs,
19:00:12–19:30:56 CST. It passed 232 allocation regression tests, then failed
with `EDQUOT` while flushing the eighth screening case. Seven complete cases
remain valid screening evidence; neither the partial eighth case nor the
overall job is marked successful. No three-repeat confirmations ran there.

The service uses Qwen3-30B-A3B-Instruct-2507 BF16, attention TP1/DP2/EP2,
eager Triton MoE / allgather_reducescatter, no DBO/EPLB/APC/async scheduling,
44 GiB virtual KV per rank, maxlen33792, 256 sequence slots and a 4096-token
scheduler budget. Each screen fixes total requests and prompt/output lengths,
then runs balanced-start, both shuffled extreme orientations, balanced-end.
Output length is 512 throughout. Primary loss is `1 - 1 / paired_slowdown`;
burst throughput includes prefill and must not replace stable decode.

| Requests | Input tokens | Rank0-hot decode loss | Rank1-hot decode loss | Maximum burst loss |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 1024 | 4.01% | 3.32% | 4.40% |
| 64 | 1024 | 2.82% | 4.88% | 6.87% |
| 256 | 1024 | 7.55% | 8.35% | 17.54% |
| 16 | 16384 | 3.09% | 4.06% | 17.20% |
| 32 | 8192 | 3.47% | 5.22% | 16.98% |
| 64 | 4096 | 3.20% | 4.63% | 15.39% |
| 128 | 2048 | 4.90% | 5.94% | 16.09% |

These are single-repeat screening values, not confidence-confirmed estimates.
The largest batch shows a larger penalty, but not the requested 30%. A larger
batch alone does not establish HBM saturation or prove a memory-bound kernel.
Kernel-active time, launch gaps, expert occupancy and hardware counters have
not yet been causally isolated.

Independent raw-data rechecks cover **28 complete trials / 2,304 requests /
1,179,648 output tokens**. Exact prompt/output lengths and stream token counts
match; each rank retains at least 379 stable steps; actual batches are eager,
unpadded and without DBO; empty ranks execute dummy batches. No stable-window
waiting or preemption appears. Maximum sampled KV usage is 85.9973%, below
the declared 90% limit. Balanced endpoint drift magnitude stays below 0.63%
in these seven cases. Every trial's original idle/reclamation checks passed.

Local raw archive: `evidence/kvcached-batch-229639/`. Recheck output:
`verification-analysis.txt` there. Remote raw path remains
`~/kvcached/paracloud/state/dual-qwen-229639/` through a compatibility symlink.

## Quota fix and continuation

The shared `/data` filesystem's free capacity was misleading: the home subtree
has a separate **1,073,741,824-byte** Ceph quota and used 1,071,833,623 bytes
at inspection. The upgraded run subtree has a **590,558,003,200-byte** quota
and used approximately 109 GB, so neither model deletion nor reducing the
experiment evidence is necessary.

At 19:49:53 CST, kvcached `state` and `logs` were copied to
`~/run/kvcached-paracloud/{state,logs}`, checked file-by-file with SHA256,
then the old copies were removed and old paths replaced by symlinks. Weights,
environments and experimental evidence remain intact. Home usage fell to
534,688,915 bytes. Migration manifests remain under the run data root.

Resume job **229747** started at 19:53:16 CST on `d1n41a15g01`, two A800s;
236 allocation tests passed. Verified matching screens from229639 are reused
only for candidate selection, with source paths retained. The missing screens
and both top candidates' n=3 paired confirmations run fresh in229747; no
cross-node pooling of confirmation ratios is allowed. Local client/control
suite:43 passed. Job229747 completed `COMPLETED/0:0` at20:33:43 CST,
40m27s elapsed. All ten screens and two fresh confirmations completed.

| Fresh confirmation | Layout | Decode loss | Ratio-derived 95% loss interval | Burst loss |
| --- | --- | ---: | ---: | ---: |
| 256 requests,1024 input | 256:0 | 8.16% | 6.77–9.51% | 16.96% |
| 256 requests,1024 input | 0:256 | 6.00% | 2.84–8.96% | 15.88% |
| 128 requests,2048 input | 128:0 | 5.14% | 1.45–8.56% | 15.09% |
| 128 requests,2048 input | 0:128 | 2.61% | 1.19–3.98% | 13.37% |

The >=30% eager target is **not achieved**. These intervals use three paired
repeats and Student-t(df2), not individual steps as independent observations.
The completed 256-request confirmation's raw windows/summaries reproduce
exactly; minimum stable steps379, maximum KV85.9207%, no stable waiting or
preemption,96MiB idle backing restored before/after every trial. Traced versus
untraced balanced TPOT ratio1.0061 is a sequential sensitivity control that
also includes time drift, not causal proof of measurement overhead.
Raw archive:`evidence/kvcached-batch-229747/`. All five fresh cases reproduce
exactly:36 trials /5280 requests /2,703,360 output tokens, no stable waiting or
preemption, all pools reclaim96MiB/rank. Full second-case KV peak73.0732%,
minimum380 stable steps/rank; its sequential trace-control TPOT ratio1.00733.

The three newly measured screens are also retained: batch32/context1024 loses
4.48%/1.76%, batch128/context1024 loses5.42%/1.92%, and batch8/context32768
loses5.49%/0.40% (rank0-hot/rank1-hot). They do not change the selected
candidates or the no30%-eager conclusion.

## Independent graph control

Job **229748** started at20:34:13 CST on `d1n41a14g01`, two A800s, after229747.
It completed `COMPLETED/0:0` at20:53:23 CST and passed236 regression tests, using standard
FULL_DECODE_ONLY graphs, compilation mode0, and exact capture sizes
1/8/16/32/64/128/256. It tests batches128/256/32 at1024/512, each with three
paired repeats and both hot-rank orientations. The stable-window validator
requires actual FULL execution and the declared max-rank padding.

Installed vLLM's `v1/worker/dp_utils.py` pads all DP ranks to their maximum
token count whenever synchronized graph mode is nonzero. Thus an extreme
layout can increase padded work on both ranks compared with balanced B/2:B/2.
This is a backend-dependent padding/synchronization cost, not a pure
unpadded-attention loss. Eager and graph curves must remain separate; graph
capture support is verified: all seven FULL decode graphs captured, while
torch.compile remains disabled. The completed batch128 case has12 paired
trials with actual FULL batches, balanced64/64 padded tokens versus extreme
128/128 padded tokens; the empty rank has one unpadded dummy token.

| Graph batch128 layout | Stable cadence | Decode loss | Ratio-derived95% loss interval | Burst loss |
| --- | ---: | ---: | ---: | ---: |
| 64:64 | approximately24.57ms | reference | reference | reference |
| 128:0 | 36.33ms | 32.37% | 31.78–32.94% | 33.06% |
| 0:128 | 36.25ms | 32.21% | 30.71–33.65% | 32.73% |

Both graph orientations exceed the declared30% lower95% loss-bound criterion
in this completed case. Independent raw rechecks reproduce every stable
window and paired ratio exactly:12 trials /1536 requests /786432 output tokens,
at least412 stable steps/rank, KV peak47.4515%, no stable waiting/preemption,
and96MiB/rank backing reclaimed before/after each trial. Maximum baseline
endpoint drift0.23%; sequential traced/untraced TPOT ratio1.00005.
Evidence:`evidence/kvcached-graph-229748/batch128-raw-verification.txt`.
Batch256 also confirms38.52%/38.35% loss, with95% lower bounds38.16%/37.90%.
Batch32 yields28.40%/29.56% and does not meet the criterion. All36 graph trials
were raw-verified; the completed report is `graph-results.md`. This is evidence for graph-mode
DP-imbalance cost, not for a30% pure attention-kernel share. Absolute eager
versus graph speedups across different allocations are not paired causal
measurements. A same-node execution-mode comparison and kernel-active/launch
diagnosis remain separate follow-ups.
