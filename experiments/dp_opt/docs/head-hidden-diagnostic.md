# Common-prefix head/hidden attribution

The FP32-head six-layout control276677 still fails full greedy equality. Every
corank Conv1 first differs at zero-based53 (20632 vs12822), while Conv0 first
differs at442 (279 vs582). Both balanced repeats and all Code outputs match.
Actual head output is FP32 and weights remain BF16. This does not establish
whether hidden vectors changed upstream or head projection accumulated error.

Repeat the exact original full P/O/arrivals and six mirror-order layouts. Keep
FP32 head output, BF16 weights, two A800s, DP2/EP2/attentionTP1, default sparse
graphs/FULL_DECODE_ONLY/compile0,128K,24GiB virtual pools,10240MiB/GPU budget,
utilization.98 and32-step control. Do not truncate outputs, force tokens, enable
batch invariance, change reduction flags or relax the original greedy check.

Observe predeclared positions52/53/54 for P149 and441/442/443 for P74. Record the
complete target prefix, BF16 hidden vector, two BF16 head weight rows and their
original FP32 logits. Guard synchronous non-speculative one-request/one-logits
row mapping, unit head scale and absent bias/softcap. Record actual PyTorch
BF16 reduction flags without modifying them. Four-rank weight rows must match.
GPU-to-CPU snapshot copies synchronize at these events, so this is an
observational correctness diagnostic; its timings are ineligible for benefit.

For identical full prefixes, let d_r be the original GPU logit difference for
the two candidates and s_r the offline CPU float64/fsum projection
`sum_i h_ri * (w_ai - w_bi)`. Reconstruct exactly within declared floating
tolerance:

`d_actual - d_reference = (s_actual - s_reference)`
`                       + ((d_actual - s_actual) - (d_reference - s_reference))`.

The first term measures changed hidden-vector contribution; the second measures
head projection's residual contribution relative to the same stored inputs.
Validate raw candidate logits against actual sampler logprob differences when
both candidates are reported, and validate stored prefixes against full raw
request outputs. If the target is no longer the first divergence, disclose
that limitation rather than treating a later split history as a common prefix.
If prefixes differ, report no decomposition. No particular kernel root cause
is proven by a changed final hidden vector alone; a subsequent layer/attention/
expert control is required. Identical hidden vectors plus changed head outputs
instead motivates a projection/reduction control.

Independent code review on November2 found two attribution-validator gaps in
the initial, unexecuted observer: it ignored captured neighbor positions when
the first divergence shifted, and it did not bind candidate ordering or worker
identity. These gaps are not established causes of the earlier276677 failure.
The observer now uses the actual first divergence when it lies in the declared
center±1 window. A divergence outside that window has explicitly unavailable
attribution, with no center-position substitution or automatic next submission.
Balanced equal outputs are labeled comparisons, not first-divergence evidence.

Weight files must cover the exact four service/rank identities, declare BF16,
and use the fixed candidate ordering. A selected snapshot must match the raw
request's logical ID (or installed eight-hex internal-ID suffix), model/service,
initial DP rank, role, repeat, layout and explicit logits row. Its prompt must
match the SHA-bound frozen prompt bank, not merely have the right length.
Sampler candidate contrasts are reported verified only when both candidates
are present. Fixed-pair projection observations cannot explain an actual top1
flip unless the first differing output tokens are that pair and both sampler
contrasts are covered. Missing coverage or different decision candidates is
explicitly reported as unavailable first-top1 attribution.

The installed runner still computes logits for partial re-prefill chunks whose
samples are discarded. A target output length alone therefore does not identify
a valid full-prefix prediction. The hook now reads the prepared CPU discard
mask and optimistic sequence length, records discarded target observations
separately, and captures only an accepted row that has consumed the exact
prompt-plus-output prefix. Unmasked incomplete forwards fail; repeated accepted
snapshots are not silently overwritten. These guards address a conditional
compatibility risk, not an observed276677 preemption cause.

Actual reduction flag values await GPU observation; mock-test
values do not establish installed defaults or Triton expert behavior.

Preserve the original strict error and all full outputs. Successful diagnostic
completion means SHA and observation reconstruction, not model correctness,
migration correctness or algorithm benefit. No migration or benefit phase
follows automatically. One allocation, protected sync,300s local and remote
query-only monitoring, full terminal raw/log SHA. Local wall-clock gaps must be
recorded even when monotonic intervals appear300s; the remote watcher provides
independently audited fallback coverage, not an unverified continuity promise.

## Independent debug/performance admission

The November2 user instruction makes this an independent DEBUG line, not a
gate on the performance campaign. This observer retains full-token validation
and never supplies benefit timings. A diagnostic failure must not stop or
cancel the separately owned performance allocation. Any performance-line
correctness classification or acceptance policy belongs to that line, not to
this observer.

By default, submission and inherited protected-sync/input-reuse guards still
require the user's whole queue to be empty. To coexist, pass the real
`--concurrent-performance-job JOB_ID` explicitly. Every guard then requires:

- `id -un` agrees with `$USER`;
- the user's queue has exactly one allocation, the specified numeric ID;
- both `squeue` and `sacct -X` identify that user and exactly the same job name;
- the name is `qwen-public-perf` or `qwen-public-perf-SUFFIX`;
- both records are nonterminal; missing, duplicate, array, unrelated or failed
  query records are rejected.

This is not a blanket queue bypass. In particular an existing debug allocation
or any unknown active job prevents creating another debug allocation. The
performance ID is part of the saved campaign identity: recovery must repeat
the same explicit argument, including when that performance job later ends.
An already recorded debug `job` resumes its own query/archive path without
creating another allocation or rechecking that the performance job is active.
If the performance job ends before the initial sync/submission guard, admission
fails closed; reconcile the unsubmitted intent rather than automatically
retrying with a new ID. An ambiguous submission ledger remains a manual
reconciliation condition, never permission to resubmit. A local advisory lock
prevents two coordinator processes from owning the same campaign root.

Only the six declared new observer files may be synced. The inherited147
sources, protected vLLM patch, fork/main/origin, old SHA backup, atomic promote,
and frozen-input before/after SHA checks remain unchanged. The single-threaded
debug process temporarily adapts only the inherited exact empty-queue test;
its module callables and file list are restored even on failure. Before
integration, confirm the independent performance manifest excludes these six
files and that its evidence/upload roots are distinct. Do not share a Python
process with the performance coordinator or start competing campaigns from
different roots: the local lock is per-root, and unrelated external Slurm
submissions can still race queue observations.

Safe local-only entry points (no SSH, sync or submission):

```sh
PYTHONPATH=kvcached/experiments/dp_opt python3 \
  kvcached/experiments/dp_opt/head_hidden_campaign.py \
  --root evidence/NEW_DEBUG_ROOT \
  --parent evidence/public-fp32-head-diagnostic-20261101 \
  --concurrent-performance-job ACTUAL_PERFORMANCE_JOB_ID --prepare-only
```

Use `--preflight-only` instead to prepare and run local CPU/actual-isolated
import checks. Use a fresh root when selecting a different concurrent ID; an
offline placeholder is not a verified Slurm identity. Neither mode records
the candidate job as GPU-verified. The integration owner, not this preparation
task, later launches the same command without a preparation flag in a detached
session with its existing `caffeinate` lifecycle. The saved debug ID resumes
local `poll_job` at300s and its remote query-only watcher at300s; neither polls,
signals or cancels the performance ID. Record actual wall and monotonic gaps
and terminal raw/log SHA, rather than assuming a background PID proves
continuous monitoring.

This bounded preparation performs no sync, allocation, commit/push or Aegaeon
operation. CPU mock Torch/reduction values and shell scheduler fixtures do
not establish GPU defaults, numerical attribution or measured performance.
