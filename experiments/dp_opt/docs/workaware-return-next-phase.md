# Next-phase gates after job257484

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
