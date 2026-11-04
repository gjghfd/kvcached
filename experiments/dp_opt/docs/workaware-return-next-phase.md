# Next-phase gates after job257484

## Update after verified259091 and active260757 (2026-10-28)

Final checkpoint:260757 is COMPLETED/0:0 at19:07:47;493rawfiles and864formal
requests independently verify. The primary-window workaware-vs-headroom Conv
P95/P99 regress6.429%/3.096%; unique return benefit is not established. Exact
compiler reuse is promoted locally only after terminal, with198targeted tests
and full verifier passing; current GPU results still use the old frozen compiler.
No new calibration/trace job is submitted. Follow the explicit
[phase-calibration-plan](workaware-phase-calibration-plan.md), retaining all
six-policy negative and startup results, rather than blind positive-result tuning.

The14:13 completed-policy subset checkpoint independently verifies132files and
288formal requests for workaware/random, while fair and the other policies are
still executing. ConvP99 improves20.806% descriptively, but every interpolation
endpoint precedes any migration; it is not causal evidence for return or rank
selection. The four Conv requests active at return instead show3.563% worse
full-requestTPOT. Keep these as negative/mechanism limits, and wait for complete
six-policy comparison before promoting the isolated compiler or changing rates.

Control calibration has a concrete new scope issue: random's seven copies take
.421970s, workaware's three take.189467s, but total measured pause calls are
.844273s versus6.537361s. Most workaware admission/return boundary waiting is
in the idle Code tenant, before Conv reaches its own default32 boundary. Calibrate
incoming/host handshake versus genuinely blocked old decode independently. Any
future affected-tenant transaction optimization must retain both attention DP
ranks' EP/default32 execution for a participating model, globally close admission,
prove unaffected tenants cannot allocate beyond frozen full reservations, preserve
copy-peak/owner/graph checks, and compare the shared optimization in baselines too;
do not simply skip an idle attention rank or call a shared data-plane fix an ILP win.

The isolated path-neutral diagnostic also corrects interpretation of an existing
reporting helper: `control_metrics` excludes random greedy actions from its staged
breakdown. Overall totals are verified; zero random pause/copy subtotals from that
helper are not measurements. Patch it only after active source freezing ends.

The pressure follow-up now confirms the admission mechanism: second Code
queue636.361s under no_move versus30.941..33.508s under migration, measured
pressure1.1963. It does **not** establish a unique workaware-return win:
headroom has better Code and Conv tailTTFT, while workaware improves Conv
request-average TPOT by only.72% in the full arrival window. The same frozen
trace is now running with random_no_return in260757; do not revise its active
algorithm, trace, calibration or source manifests.

After complete six-policy archival, separate three contributions before revising
the general algorithm:

1. **Admission rank selection**: an available target requires seven moves while
   the other requires one. Random-versus-ILP gains here are admission-choice gains,
   not evidence that performance-aware return is effective.
2. **Population-domain coverage**: workaware defers returns at N7/N6/N5 and
   executes only at N4. Profile these missing populations with all feasible
   candidate count partitions, mature context coverage, independent repeats and
   held-out errors; reject A800 cases outside actual capacity. Preserve existing
   domains until verified new calibration exists, rather than extrapolating.
3. **Tail protection and real migration cost**: all-control workaware15.149s
   exceeds headroom13.967s despite fewer moves. Charge transaction activation,
   staged copies and uncertainty once at their actual scopes; minimizing remaining
   request-seconds does not by itself protect pending Conv requests' TTFT. Check
   pending first-token work before an elective return and revalidate at the
   pause boundary; any tail budget or arrival-risk prior must be declared before
   the next paired trace, use only current/past arrivals and be reported separately
   from decode-service improvements. Do not inspect future frozen arrivals.

An independent source-owner reconstruction enumerates511 bounded subsets per
target in the first blocked arrival and finds no final-memory-feasible subset:
minimum excess224MiB/32MiB. This is under conservative independent rounding of
each destination request, not proof that the real allocator cannot reuse mapped
holes. Investigate destination block/page selection in a dedicated paused,
rollback-safe GPU probe. If the real allocator can safely use that space, change
the shared placement certificate and verification for **all migration policies**,
not only workaware; do not attribute a shared capacity-accounting fix to a new
   performance algorithm or skip the source-owner/copy-peak checks.

The tail timing audit also requires a **joint-active phase calibration** before
claiming to optimize these ConvP99 samples. Both headroom/workaware P99 endpoints
precede their first actual return and overlap Code residence; no Conv TTFT
interval overlaps either successful return. A later return cannot causally fix
an earlier first token. Preserve stage-specific Code prefill/decode versus
Conv-only measurements and include admission-choice/observer overhead plus
paired run variability; do not change the existing sparse graph or use future
Code marks in a decision. Tail protection remains useful but is not already
proven to explain this particular P99 regression.

The primary acceptance checks remain Conv/Code avg/P95/P99 TTFT and Conv avgTPOT,
with combined metrics, Code TPOT and all-control cost as adverse checks. Retain
every negative outcome. Two Code marks give descriptive tails only; eventual
incremental improvements require paired repeat/seed confirmation.

The260757 N9 auxiliary result adds two separate gates. Its compiler spends
1/9s per scenario and misses incumbents in three unconstrained cases. Exact
source-owner release and identical-membership graph-selector reuse fixes that
locally without changing the objective, while preserving separate attention
segments. The candidate is isolated under `.workaware-stage/solver-reuse`, with
47 tests and exhaustive raw-snapshot optima; do not promote it over the active
allocation. Also distinguish `no_verified_incumbent` fallback from a certified
unprofitable optimum in diagnostics.

After that exact optimization, the old uncertainty gate still rejects the
best N9 four-move candidate: gain126.390 versus bound448.792 request-seconds.
Before a subsequent GPU benefit experiment, collect independent calibration
repeats and inspect population/layout/context-specific held-out errors, including
the missing5/6/7 populations. A more precise bound must be justified by held-out
evidence under a predeclared general rule, not reduced until this final trace wins.
Preserve raw calibration contrasts and conditional-domain failures; an ILP CPU
speedup or a lower surrogate objective alone does not demonstrate net GPU gain.

The verified result is in `../reports/workaware-return-results.md`. This document
plans a follow-up; it does not claim a submitted job, deployed new cost model or
completed benefit goal. Preserve the original trace and every adverse metric.

## Separate the two experimental questions

1. Placement pressure: does a real Poisson Code arrival find enough aggregate
   KV capacity but insufficient capacity on both attention homes, with substantial
   natural-release waiting in the no-move baselines?
2. Recovery cost: after admission, does a calibrated performance-aware return
   improve Conv decode/TTFT enough to pay real pause/copy cost and future admission
   risk, compared specifically with `kv_only_headroom`?

The current trace answers the second with only one late return and modest effects;
it does not establish sustained blockage for the first. Do not conflate a capacity
method's inherited gain with a new recovery policy's unique gain.

## Placement-pressure pilot before freezing new rates

Keep two A800s, both independent Qwen tenants, BF16/128K, default sparse graphs,
pause32,24GiB virtual pools,10GiB controlled budgets and true100K Code inputs.
Keep the old.01956673845194241 Conv RPS as a control; do not change its old results.

Measure Conv reservation-to-completion residence under open-loop arrivals. For
target mean populations4 and5, the first theoretical rates are
`lambda_conv = target_population / E[T_resident_conv]`. These are pilot candidates,
not frozen final rates: cadence changes with population and must be remeasured.
An approximate two-model budget ratio or a single background snapshot is not a
steady-state admission-opportunity proof.

Prespecify a small candidate bracket from those residence measurements, then test
population stability, queue drift, per-rank actual mapped pages, reservation bytes
and allocator probes. Reject overload or cases whose aggregate space cannot fit
the long request. Choose final rates from these stability/capacity criteria, not
from the most favorable measured TTFT. Retain every tested rate and failed guard.

With comparable reservation-to-first-token cost `P`, a50% Code TTFT reduction
requires `W_baseline >= P + 2*W_treatment`. For P≈55s and treatment queue≈1s,
baseline queue needs roughly57s, not this trace's9.36s. More generally use
`W_baseline >= 2*(W_treatment + P_treatment) - P_baseline`. This is a theoretical
feasibility check, not a rule for selecting/removing requests from the trace.

Freeze Poisson seeds, both rates, lengths, budget and evaluation window before
policy execution. Keep all actual Code marks, including arrivals that find an
empty rank; no forced favorable arrival times or synchronized bootstrap presented
as a final trace. Report all five policies, full and primary windows, Conv/Code/
combined avg/P95/P99 TTFT and Conv request-average TPOT. Keep Code TPOT and complete
control cost as adverse-metric checks. Two Code samples do not validate statistical
tails or the fairness threshold, which is30s and was not reached in this trace.

## Cost-model validation before changing the serving ILP

`analyze_workaware_partition_cost.py` independently rebuilds all raw calibration
contrasts and compares graph-only shape cost with actual-attention-count shape
cost plus linear context work. Its output is diagnostic only. Overall held-out
RMSE improves, but ConvN3/N9 worsen; N8/N9 lack some count levels, and populations
5/6/7 are absent. Do not lower error bounds or silently interpolate these states.

Collect independent confirmation repeats beyond the two old repeats, including
the proposed counts and relevant mature contexts. Before each case, prove its
page-aligned physical footprint fits the A800 budget. A balanced long-context case
may fit where an all-hot homogeneous case cannot; do not truncate a failing input
or borrow another population's context/peak envelope. Retain source/context/peak
coverage and explicitly reject candidates outside the measured domain.

Choose the model rule before inspecting final trace metrics. Require held-out
identifiability, complete candidate-count coverage and per-population worst-error
reporting. All direct/return candidates still need physical ownership verification,
full rollback safety and complete-copy cost. Another active tenant still requires
joint-active calibration; no single-tenant profile is silently promoted to that role.

The current remaining-work forecast includes completions, not new births. Test an
arrival-aware horizon using only past arrival counts and frozen rate priors; never
use future trace marks. Also evaluate a per-request queue/TTFT protection constraint,
because minimizing remaining request-seconds alone does not protect TTFT quantiles.
These are proposed revisions, not already validated GPU algorithms.

## Operational gate

Before the next allocation, fix the global archive barrier using a shared watcher
exit marker rather than login-host-local `pgrep`, and preserve immutable manifests
for every failed attempt. Keep unsafe-link/path rejection and exact raw-file SHA
checks. Do not hide the first257484 manifest-overwrite limitation.

The4-ULP verifier fix is installed locally after terminal archival/audit, not synced
over an active job. Future execution requires explicit old-SHA backup, atomic sync,
new-SHA checks and protected vLLM/Aegaeon diff checks, still on gjghfd/main with no
unauthorized commit/push. Use one allocation request and resume300s monitoring to
terminal, followed by complete SHA archival and independent verification.
