# Static same-rank, no-copy control

The first native phase of job274385 fails its complete serial greedy comparison.
NoMove and every Code output match, while both long background outputs differ in
all three migration policies. Four device-bytewise copy echoes pass, but even
unmoved stayers diverge. This is neither a proven copy bug nor a numerical cause.
The failed timings are ineligible for setting selection or benefit claims.

Reuse every original token and arrival from that first native phase: background
P74/O16557 at0s, P149/O16314 at1s, Code P101626/O64 at90s,900s offered window.
Predeclare six no_move trials: balanced, corank0, corank1, corank1, corank0,
balanced. Two background requests start on the same attention rank in each
corank trial; no KV is copied and no pause is invoked. Admission only filters
the already budget/allocator-feasible candidate set, never enlarging it. Code
uses unchanged ordinary direct placement and all requests must drain.

Preserve two A800s, independent models, DP2/EP2/attentionTP1, BF16/128K,
24GiB virtual pool per model/rank,10240MiB/GPU budget, utilization.98, default
sparse graphs/FULL_DECODE_ONLY/compile0 and32-step boundaries. Use the same
WorkawareVerificationWorker as the failed four-policy native phase. Production
scheduler, source validation and graph configuration remain unmodified.

Request top2 logprobs for full serial references and every output. Compare only
the first divergence with an identical preceding history, including both common
candidate margins where available. Same-layout mirror-order repetitions detect
repeatability and instrumentation issues; balanced controls detect whether the
probe itself reproduces the old accepted baseline. Same-rank divergence without
copy implicates placement/batch/prefill-related execution, not a proven particular
kernel cause. If no-copy controls pass, inspect migration/runner metadata next.

The probe catches only the original complete serial/paired-output errors after
all six trials drain, retaining their error and running summary. A separate
diagnostic result never converts the underlying trace to a passing benefit. Do
not enable batch invariance, change precision/graphs, truncate requests, accept
posthoc token tolerances or evaluate already-diverged histories as common-prefix
logits. Terminal SHA and independent source/logprob reconstruction are required.

The144-case cost source of274385 independently reconstructs its saved six-profile
fit. It can be considered for later reuse only with the same measured configuration
and independently verified source SHA; a correctness patch that changes kernels
or their cost invalidates blind calibration reuse. This diagnostic runs no new
cost campaign and no workload-rate or held-out tuning.
