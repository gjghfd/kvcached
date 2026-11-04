# Long-output AzureConv return probe: status and results

Current closure, rechecked2026-11-01: **job268245 COMPLETED at2026-10-30
17:53:58 on two A800s**. All585 raw files,36 requests and112 calibration cases
close across three seeds. The final three-seed table is in "Three-seed closure"
below; earlier two-seed sections are interim history. Conv8192→24576 is threefold,
not fourfold. Workaware Conv avg TPOT improves22.42% vs random, but regresses18.77%
vs no_move; Code avg TTFT improves72.59% vs no_move through capacity admission.
This is a closed-N2 mechanism result, not final Poisson/tail evidence. The user
now explicitly skips fourfold and continues original-response public datasets.

Input-count clarification: the frozen Conv bank has574,063 distinct length-pair
marks with823,814 total row weights. The older protocol's574,063-request wording
confuses marks with represented rows; sampling and all source hashes are unchanged.
The next open-loop plan and its rate priors are in
`../docs/long-conv-open-loop-plan.md`, not applied to this active probe.

## Two closed seeds: retained paired and pooled results

Second seed20261031 improves Conv avg TPOT19.64% vs random and20.92% vs
headroom; Code TTFT improves72.66% vs no_move. Its control transaction starts
0.099491s after Code completion and resumes3.051246s after release, including
0.123055s copying2,706,898,944 bitwise-verified bytes. Transaction walltime
2.951755s is not freeze time; aggregate active blocking upper is0.484058
request-seconds. Matched pre-return anchors save169.536672 request-seconds,
with both Conv spans included and none excluded. Random retains120 stable N2
samples concentrated in one rank after Code release; no balanced N2 observation
occurs before natural completion.

Pooled empirical TTFT below uses2 Code/4 Conv/6 combined requests per policy;
it is not an average of per-seed tail percentiles. All times are seconds.

| Policy | Code avg/P95/P99 TTFT | Conv avg/P95/P99 TTFT | Combined avg/P95/P99 TTFT | Conv avg TPOT(ms) |
|---|---|---|---|---:|
| no_move |205.080/207.269/207.464|0.4118/0.8281/0.9012|68.635/206.296/207.269|12.1231|
| random_no_return |55.578/56.390/56.462|0.2058/0.2584/0.2669|18.663/56.029/56.390|18.2801|
| kv_only_headroom |55.430/56.323/56.402|0.2451/0.3058/0.3084|18.640/55.926/56.323|18.3291|
| kv_only_workaware |55.603/56.619/56.710|0.2483/0.3424/0.3570|18.700/56.168/56.619|14.3020|

Pooled Conv TPOT improves21.76% vs random and21.97% vs headroom, but worsens
17.97% vs no_move. Workaware Code/combined avg TTFT improves72.89/72.75%
vs no_move, from capacity admission. Against random, Code avg worsens0.045%,
combined avg worsens0.196%, and Conv avg/P95/P99 worsens20.62/32.52/33.75%.
The second seed's large initial no_move Conv TTFT changes the pooled contrast;
**all TTFT still precedes return**, so even a favorable pooled Conv TTFT must
not be credited to return. No small-sample tail significance is claimed.

Both seeds and negative metrics are preserved. The paired Conv TPOT gains are
23.77/19.64% vs random, not a claim of24% in every run. Evidence:
`two-seed-metrics-summary.json` and the two independent phase-analysis artifacts.
The third seed, terminal SHA archive and final summary remain required.

## First closed seed: return benefit with old-request costs

Seed20261030 has two Conv and one Code request per policy. TTFT is measured
from arrival; all values are seconds. Code avg/P95/P99 are identical because
there is only one Code observation. Conv and combined tails are small-sample
empirical quantiles, not statistical tail significance.

| Policy | Code avg/P95/P99 TTFT | Conv avg/P95/P99 TTFT | Combined avg/P95/P99 TTFT | Conv avg TPOT(ms) |
|---|---|---|---|---:|
| no_move |202.648/202.648/202.648|0.2670/0.3062/0.3097|67.727/182.414/198.601|12.0461|
| random_no_return |54.676/54.676/54.676|0.2307/0.2652/0.2683|18.379/49.236/53.588|18.8085|
| kv_only_headroom |54.439/54.439/54.439|0.2979/0.3080/0.3089|18.345/49.026/53.356|18.6188|
| kv_only_workaware |54.474/54.474/54.474|0.2998/0.3545/0.3594|18.358/49.063/53.392|14.3380|

- Workaware improves full-request Conv avg TPOT by23.77% vs random and22.99%
  vs headroom. All1200s offered windows and paired requests are retained.
- Against no_move, Code TTFT falls73.12%; combined avg/P95/P99 falls
  72.89/73.10/73.12%. This is capacity-admission benefit, **not return benefit**.
- Against no_move, Conv avg TPOT worsens19.03%; Conv avg/P95/P99 TTFT worsens
  12.30/15.78/16.05%. Against random, Conv TTFT worsens29.95/33.69/33.98%.
  These costs remain part of the result, not excluded warmup or outliers.
- Every request's TTFT occurs before return. The small Code TTFT differences
  among migration policies cannot be attributed to moving KV back.
- Makespan is not fixed-offered-rate throughput. One seed does not establish
  robust benefits over all populations, rates or future arrivals.

### Actual return and pause-inclusive matched work

Code completes185.148092s after origin. Return starts185.264253s (0.116160s
after release); resume completes189.461264s (4.313172s after release). The
control transaction lasts4.197011s, including0.128849s of bitwise-verified
`dp_echo` copying for2,695,888,896 bytes. The4.20s transaction is **not** the
GPU-freeze duration: reported aggregate active blocking upper is0.454715
request-seconds. Migration still pauses inference; no overlap is implemented.

With the same token-ordinal anchors before return and no deletion of planning,
pause, copy, resume or subsequent controls, the two remaining Conv spans save
108.710713/108.699733s, totaling217.410446 request-seconds vs random. Sustained
observed-prefix repayment begins about9.415s after the pre-return anchors.
This is aligned-work evidence, not a randomized causal estimate.

Random remains concentrated in one attention rank through all135 stable N2
population observations after its Code release; the first Conv finishes about
277s later. Workaware first observes1:1 about4.357s after its Code release.
The return therefore precedes natural recovery in this seed.

Evidence under `evidence/long-conv-return-probe-20261030/`:
`first-seed-metrics-summary.json`,
`pause-inclusive-phase-analysis/probe-20261030-independent-analysis.json`, and
that phase's immutable168-file raw archive and before/after manifests.
The running GPU policy is the **old point-optimal workaware** with the new
long-context calibration, not the separately validated robust-J planner.

## Frozen setting

| Item | Setting |
|---|---|
| Background | Full weighted AzureConv input bank; synthetic output8192→24576 |
| Concurrency | Two Conv arrivals at0/1s; no new background births in this probe |
| Code | Real SWE-smith>=100K input,64 output; arrival120s |
| Paired inputs | Seeds20261030/20261031/20261032; retain all three |
| Policies | no_move/random_no_return/kv_only_headroom/kv_only_workaware |
| GPU/layout | TwoA800, two independent Qwen tenants, each DP2/EP2/attentionTP1 |
| KV/graph | 10240MiB/GPU budget,24GiB virtual model/rank pool,128K/BF16/default sparse |
| Migration | Real dp_echo, pause32, no inference overlap; no algorithm/guard change |
| Calibration first |112 long-context cases, both tenants/N2/N3, four lengths/two repeats/all splits |

The new standalone N2/N3 profile does not overwrite the old fits. Full protocol,
capacity arithmetic and eventual open-loop rate calibration are in
`../docs/long-conv-return-probe.md`. All capacity/rate estimates remain priors
until GPU validation. Four policies×three identical paired traces run in one
allocation after calibration; failures and negative outcomes are retained.

The frozen first-seed order is random_no_return→no_move→kv_only_headroom→
kv_only_workaware, not the table's canonical ordering. Its four policies are now
independently verified; each retains the declared1200s window. `no_move` uses the same
full-reservation admission controller and budget, not unmodified native vLLM's
dynamic-growing KV scheduler; capacity waiting here is not whole-card CUDA OOM.

## Independently verified long-context calibration

The closed calibration directory was packed at13:25:06 after its13:03 exit0
marker and transition to a different phase. All59 regular files have identical
before/after manifests; the downloaded bundle and extracted files match SHA,
and the local verifier reconstructs the fit from112 cases and full token records.
Evidence: `closed-calibration-independent-review.json` and
`closed-phase-archives/long-context-calibration/` under the recovery root.
This is calibration with one active tenant and the other resident but idle,
not a jointly active-tenant performance model or a live return result.

| Tenant/population | graph8 penalty(ms) | attention(s/token) | held-out residual(ms) |
|---|---:|---:|---:|
| Code/N2 |5.09303|1.27826e-7|1.41493|
| Code/N3 |0|2.16761e-8|2.55240|
| Conv/N2 |4.88660|1.40854e-7|1.30953|
| Conv/N3 |0|1.62211e-8|1.32494|

Each population has16 paired contrasts, two repeats, four held-out input lengths
and both fully imbalanced layouts. The measured upper envelope is33791 context
tokens per request. The residual is an empirical worst-held-out allowance,
not a statistical confidence interval. No old coefficient or guard is replaced.

For two remaining Conv requests with contexts `K0,K1`, moving from2:0 to1:1
changes the calibrated default-sparse cost by
`delta_t = 0.00488660 + 1.40854e-7 * min(K0,K1)` seconds per decode step.
The existing two-sided allowance is `2*delta = 0.00261906` seconds per step.
At representative contexts8000/20000, the point deltas are6.01343/7.70367ms;
after that same allowance,3.39437/5.08461ms remain. Ten thousand overlapping
decode steps for two requests would thus offer67.887/101.692 request-seconds
**before** subtracting actual planning/pause/copy and headroom-risk differences.
Those are illustrative calibrated-model calculations, not measured policy gains;
actual remaining-work segments and earlier natural completion still govern.

The N2 opportunity includes the sparse graph transition(max rank count1→2
maps to graph1→8), not attention time alone. N3 already uses graph8 when balanced;
its context contribution is much smaller than the retained empirical allowance.
Longer output alone therefore does not establish a general N3 return benefit.
The active probe tests whether the N2 opportunity survives real copying and
whether return precedes natural recovery, without changing its frozen algorithm.

Raw stable-cadence averages, retaining all calibrated Conv lengths and repeats:

| Conv population | Input tokens | Balanced(ms) | Fully skewed(ms) | Decode-rate loss |
|---|---:|---:|---:|---:|
| N2 |8192|11.275|17.620|36.01%|
| N2 |16384|11.861|19.275|38.47%|
| N2 |24576|12.344|21.072|41.42%|
| N2 |31744|12.685|22.291|43.09%|
| N3 |8192|17.770|18.278|2.78%|
| N3 |16384|19.373|19.911|2.70%|
| N3 |24576|20.719|21.567|3.93%|
| N3 |31744|23.088|23.074|-0.06%|

Here rate loss is `1-balanced_cadence/skewed_cadence` at fixed population,
not relative whole-trace throughput or a return-policy speedup. N2 has two
balanced and four skewed samples per length; N3 has four of each. Input length
is not the stable context length: all samples use output-token window1536..2560.
The full two-tenant table and source SHA are in
`closed-calibration-measured-layouts.json`. The small negative N3 result is
retained rather than filtered. These measurements support checking the N2
mechanism; they do not claim pure attention-kernel overhead or net KV-copy gain.

## Execution evidence

Previous265356 ended10:06:57 with0:0; its761 raw files, two Slurm logs and2094
formal requests passed full independent verification. Low Code rate still gave
one return267.991s after Code completion versus random43.769s natural recovery;
Conv full-request TPOT regressed.4925%, so it did not establish timely return
gain. See `workaware-early-return-results.md`, not this new run, for those results.

The new submission passed468 CPU regressions and93-file protected synchronization
with old-SHA backups, atomic installation and new-SHA/patch checks. All85 frozen
runtime/test files remain unchanged. One allocation was submitted10:42:16; no
duplicates, commits, pushes or Aegaeon inference were performed. Local coordinator
85760/caffeinate85761 and remote query-only1813672/ln01 monitor every300s. Two
remote300s rounds are directly verified; this is not a claim about future or
uninterrupted whole-run local monitoring.

The main poll log records10:42/10:47/10:52/10:57 rounds with SSH0 and approximately
300s spacing. Local85760/85761 remain PPID1. Runtime stays frozen while independent
offline diagnostics inspect old return gates; candidates are not installed into
this running experiment.

By11:42 the main polls still have approximately300s spacing/SSH0. All85 frozen
files are rechecked against both local and remote SHA. The new local robust
candidate/follow-up passes540 combined CPU tests but is not deployed; see
`workaware-robust-candidate-results.md`. Follow-up1599/caffeinate1600 independently
waits for full terminal verification and then replays all three probes locally.
Remote heartbeat11:37:24→11:42:25 is fresh. A host-unaware PID check failed when
the alias reached another login node; started marker namesln01, diagnosis reached
ln02, and the next heartbeat review reachedln01. This is not an observed monitor
outage and no duplicate watcher was started.

Recovery/evidence: `evidence/long-conv-return-probe-20261030/`, including frozen
inputs, regression logs, preserved initial wrapper/collection errors, sync
manifest and coordinator/poll logs. On completion the coordinator SHA-archives
raw and independently verifies calibration, inputs, tokens, ownership, budgets
and graphs before writing `independent-verification.json`.

Primary checks are Code release→return versus paired baseline natural recovery,
planning/pause/copy repayment, Conv avgTPOT, and separate Code/Conv/combined
avg/P95/P99 TTFT. Supplement with same-ID/same-ordinal128-token blocks after32
settlement tokens. TTFT before return is not return gain; three Code requests
do not establish Code P95/P99 significance. No GPU results are filled in yet,
and the overall research objective remains incomplete.

## Pause-inclusive analysis continuation

At13:51:49 a new local observer23960/caffeinate23961 starts reading the existing
main poll every300s. It adds no Slurm queries or GPU submissions and does not
modify any of the85 active runtime/test or six prior candidate/follow-up files.
It archives only closed prior seeds and immediately performs full independent
trace verification and pause-inclusive same-ordinal repayment analysis. The
last seed waits for the main campaign's complete terminal SHA archive. Five
new local analysis/test files have their own frozen manifest.

The new diagnostic includes planning/pause/copy/resume in matched elapsed spans,
instead of treating the old post-settlement steady window as net gain. An
independent agent audit catches one real simultaneous-Code double-count error;
its failing log is preserved and the paired-window merge fixes it. All67 current
targeted tests pass. No result is claimed before complete raw verification.
See `../docs/return-repayment-validation.md` for the exact metric and gates.

New evidence: `phase-analysis-local.json`, `phase-analysis-source.sha256`,
`phase-analysis.log`, and `pause-inclusive-phase-analysis/observer-state.json`
under the recovery root. This observer is separate from the frozen robust
candidate counterfactual follow-up, and neither deploys a new live algorithm.

Final full CPU regression passes607 tests in154.27s; targeted67 tests pass.
The initial20-test command completes its tests but its zsh exit wrapper fails
because `status` is read-only; later wrappers use `result_code`. An attempted
outer-workspace `git diff --check` fails because ParaCloud itself is not a Git
repository; the kvcached check succeeds. These command-scope errors do not
change GPU source or substitute for the independently retained audit failure.
# Three-seed closure: 2026-10-30

Job268245 completed at17:53:58 CST on two A800/d1n41a21g03. The complete
585-file archive and both job logs pass SHA verification; all three seeds,
36 trace requests and112 calibration cases pass the original independent
verification. `three-seed-metrics-summary.json` independently rebuilds the
following pooled metrics from the complete archive. These are still the old
point-optimal GPU policy, not the newly prepared robust controller.

TTFT seconds, avg/P95/P99; Conv TPOT milliseconds. Per policy there are only
3 Code,6 Conv and9 combined samples: empirical quantiles, not P99 significance.

| Policy | Code TTFT | Conv TTFT | Combined TTFT | Conv avg TPOT |
|---|---|---|---|---:|
|no_move|204.145/207.026/207.415|0.3733/0.7741/0.8904|68.297/205.567/207.123|12.0876|
|random_no_return|56.205/57.361/57.440|0.2183/0.2883/0.2935|18.881/57.067/57.381|18.5047|
|kv_only_headroom|55.745/56.417/56.421|0.2536/0.3373/0.3448|18.751/56.403/56.418|18.5521|
|kv_only_workaware|55.958/56.726/56.731|0.2603/0.3601/0.3605|18.826/56.707/56.727|14.3569|

-Workaware Conv avg TPOT improves22.42% versus random and22.61% versus
  headroom, but regresses18.77% versus no_move. Preserve this old-Conv cost.
-Code avg TTFT improves72.59% and combined avg72.43% versus no_move. This is
  capacity admission, not return: every initial TTFT precedes the return.
-Versus random, workaware Code avg/P95/P99 TTFT improves only0.44/1.11/1.23%;
  combined improves0.29/0.63/1.14%. Conv TTFT regresses19.24/24.89/22.83%.
  These small-sample contrasts cannot demonstrate a return TTFT benefit.
-Code avg TPOT is159.83ms for workaware versus16.44ms for no_move and157.14ms
  for random. The draining baseline serves Code largely without active Conv;
  TTFT improvements must not hide this mixed-active decode cost.
-The1200s offered windows remain identical; no pooled makespan or fixed-rate
  throughput gain is inferred. `no_move` is the same full-reservation custom
  admission controller, not unmodified native vLLM and not whole-card CUDA OOM.

Third-seed robust offline replay verifies all four before/after-pause ×1/8s
plans, with zero newly guard-passing opportunities. Across three seeds12 such
CPU replays verify; they are not new robust GPU performance measurements.
The pause-inclusive phase observer closes at18:04:39. Seed1/2 retain217.410/
169.537 request-seconds saved across four included Conv spans. Seed3 excludes
both Conv spans because no common pre-return token ordinal exists after both
policies' Code releases: workaware returns before the slower random Code
release. Its empty diagnostic sum is not a measured zero benefit. Do not delay
the runtime return or relax the matching rule to manufacture a diagnostic.
Native routing and zero-tenant Poisson support pass
staged regression1485 tests/5 skips, with one independently reproduced unrelated
wait_forecast failure explicitly deselected. Do not call this an unconditional
all-suite pass. Both old follow-ups close before promotion.13 files are then
promoted with104 original source files preserved, and117 canonical-path focused
tests pass. The first patch-format mismatch makes no canonical change; its
failed patch and zero-collected test log remain beside the successful retry.

The closed N2 mechanism now has a repeatable Conv TPOT return benefit, but does
not finish the final Poisson/fairness/tail-TTFT research experiment. See
`../docs/workaware-robust-native-protocol.md` for the next validation, prepared
in the isolated staging copy and not yet deployed at this record's creation.
