# Stratos full-distribution workload search

Prepared2026-10-30 UTC. This replaces the unsubmitted OpenThoughts3 final-grid
continuation, not the running270947 reference/calibration or its raw evidence.

## Workload and trace

The long-resident tenant serves original Bespoke-Stratos public reasoning lengths,
not manually extended AzureConv. Use the full P<=8192 response distribution;
all16710 source responses qualify. Freeze256 length-independent hash-selected
original prompt/response pairs before GPU outcomes. Do not select only O>=16K,
inflate outputs, suppress short responses, or reroll draws. This tenant role is
called Conv in the metrics, but the workload is reasoning, not real human chat.
See `../reports/public-stratos-profile.md` for lengths and HTTP snapshot limits.

The second tenant retains SWE-smith original100K+ inputs and64-token output
replay. Two independent Qwen3-30B-A3B tenants share two A800s, eachDP2/EP2/
attentionTP1. Preserve128K, BF16,24GiB/model/rank virtual pools,10240MiB/GPU
controlled budget, allocator.98, default sparse graph/FULL_DECODE_ONLY/compile0,
pause32 and dp_echo/32MiB copy chunks. Virtual pool size is not physical KV usage.

Both streams are fixed Poisson arrivals, sampled from the frozen original bank.
Held-out means a predeclared independent arrival/draw seed over the same bank,
not a held-out source-data corpus. Preserve all offered, empty and negative cases.

## Rate prior, not a measured final rate

For request i let O_i be its original output tokens and
b_i=96*ceil((P_i+O_i)/1024) MiB be its conservative page reservation.
For the256-bank, E[O]=4630.6875 and E[b*O]=4076769.375 MiB*tokens;
its residence-weighted E[b*O]/E[O]=880.3810 MiB.

For budget B, largest single-Code pool b_Code, target local pressure rho and
decode step t, the same preliminary sizing relation is

`lambda_Conv = 2*(rho*B - b_Code) / (t*E[b*O])`.

With B=10240MiB, illustrative b_Code=9600MiB and rho=1.2:

|Assumed step|Conv rate prior|
|---|---:|
|15ms|0.087913RPS|
|20ms|0.065935RPS|
|30ms|0.043956RPS|

These are **assumed-cost reservation priors**, not new measured Stratos service
times or an assertion of actual mapped pressure. They imply roughly6.10 active
requests under the simplified model. NativeN2/N3/N4 cost/service measurements
alone are not full calibration of a higher steady-state population. Default
graph buckets, growth, natural releases, queueing and migration change the
effective operating point. Before formal evaluation, use actual Code physical
pool and no_move Conv TPOT to freeze three rate trials for rho1.1/1.2/1.3, then
independently verify coverage, actual per-GPU physical pressure and queue evidence.
Reject unqualified trials; do not relax guards or extend outputs to get a pass.

## Safe automatic continuation

After270947's independently archived completion-metadata failure, the unsubmitted
v2 chain is stopped. The diagnosed recovery uses `public_source_recovery.py` and
`evidence/public-stratos-recovery-20261030/`. Its prerequisite is the failed
parent's complete SHA archive, explicit matching metadata failure and exited
coordinator, not a fabricated success. Completed269991 is solely the environment
bootstrap. Four label/watchdog promotions have exact backups, and the minimal
metadata compatibility fix is regression tested before the child's protected
remote sync and sole Stratos allocation request. The original bank is unchanged.
The statements about the earlier parent-success/v2 chain below are historical;
that waiting chain cannot resume after the failed parent.

Before interpreting final gains, inspect actual Conv populations and rejected
return proposals. The current cost model covers N2/N3/N4, with a strict context
and attention envelope. Unsupported current populations stop the forecast;
they are not silently costed as N4. The independently reconstructed parent
profile shows a default-sparse N2 capture-boundary opportunity, whereas the
N3/N4 single-step attention gain can lie within the two-residual uncertainty
charge. This is not proof that mixed-workload returns repay their real costs.

If rates selected for pressure mostly operate above the calibrated population,
collect additional feasible population/context/layout pairs before claiming
that the algorithm's attention model was fairly evaluated there. If the best
qualified setting is at the lower grid boundary, extend adjacent lower rates in
a separately frozen iteration, retaining genuine baseline queue witnesses.
Keep bank, original response lengths, metric/SLO definition and within-round
budgets common to every policy. New Conv arrivals, not only Code arrivals or
Conv completions, can end a count-imbalance window; include that recovery in
the comparison with random migration and in any general wait-versus-return rule.

The watchdog-only v2 recovery supersedes the unsubmitted original continuation.
`public_source_continuation_v2.py` stages four files under
`.public-source-stage-r2`: the two tenant-label fixes and the two batch scripts.
Only after the active parent has an independent SHA closure and its coordinator
has exited may these files be promoted with exact old/new SHA backups.
The active parent and its original staged continuation remain immutable.

Use a common request watchdog of7200s for all calibration strategies and18000s
for all exploration/held-out strategies. The previous1800s included admission
queueing and could abort the very baseline wait being measured. These limits
are operational failure bounds, not changed SLOs: the deadline remains120s.
No request lengths, offered arrivals, budget, graph or algorithm parameter is
changed by this recovery. A watchdog expiry retains unfinished requests and
fails the case; it is never a benefit result.

The current calibration limit remains18h; the future population-calibration/
grid/held-out allocation is80h, superseding the unsubmitted60h configuration.
Four maximum exploration settings times
seven policies times3600s plus seven10800s held-out runs have a49h offered-window
floor, leaving31h for the320-case population supplement, startup, full
serial-reference generations and drain.
This is not a guarantee that every case fits. Observe actual drain/reference
costs; a TimeLimit requires raw archival and a separately frozen recovery, not
dropping requests or weakening correctness checks. During the2026-10-30 UTC
recovery the A800 partition reports `MaxTime=UNLIMITED`, but longer allocations are
not guaranteed to start promptly.

`public_source_continuation.py` runs in a detached local session, only reading
the original270947 parent's local state every300s while waiting. No additional
SSH/GPU request occurs then. It freezes all109 parent runtime SHA, the staged
label fixes and Stratos input manifest, and checks raw snapshot provenance.

Only after successful Slurm accounting, complete SHA archive, matching independent
analysis and the old main process exiting may it back up/atomically promote
`public_workload_search.py` and `public_poisson_followup.py`. Changes only derive
tenant/report names from the verified public source; no ILP, response length,
CUDA graph, budget or fairness algorithm changes. Current running source is not
overwritten. Existing local edits or unknown partial backups block promotion.

The Stratos calibration uses one18h A800 allocation for per-active-tenant cost calibration,
two-seed N2/N3/N4 native four-policy comparisons, then three3600s no_move
open-loop rate trials with600s warmup and Code.001RPS. Current source-independent
microprofile uses synthetic inputs/O3072 solely for cost measurement, not benefit.
SHA/independent closure and a physical-pressure/queue-qualified rate are required.

`workaware_pilot.py` first waits for both engines to empty, then submits only
`engines[case['model']]`. Both tenants' weights/KV managers are resident, but only
one tenant has live requests in each microprofile case. This is **not concurrent
two-active-model joint calibration**. It matches the common post-Code release
phase with Code idle, but does not calibrate Code-prefill/Conv-decode interference
or every overlapping release. Native mixed traces and the Poisson comparisons
measure those costs end-to-end; they do not retroactively make the microprofile
joint. Treat the cost model as an empirical surrogate, retain adverse results,
and require further coupled measurements before claiming general joint optimality.

Then one80h A800 allocation first collects the independently frozen N5..9
supplement and anchors/merges it with the parent's N2..4 raw calibration. Both
models and all rank splits have two-repeat/two-context coverage, conservatively
screened at9984MiB maximum mapped including pinned pools. Original public
response lengths, uncertainty guards, graph configuration and all benefit
metrics remain unchanged. This is population-domain expansion, not an algorithm
retune using measured policy gains. Profiles outside N2..9/context coverage
remain unavailable. See `public-poisson-search.md` for exact plan and SHA checks.
After verification, the same allocation evaluates at most two qualified Conv rates, each
with Code.0005/.001RPS. Seven policies: no_move, own_model, kv_only, no_move_fair,
random_no_return, kv_only_headroom, kv_only_workaware. Exploration is3600s,
warmup600s, fixed seed2026103104. Selection uses equally grouped log avg/P95/P99
TTFT improvement versus no_move plus Conv average TPOT improvement versus random;
eligibility requires nonempty measured tenants and positive Conv TPOT gain versus
random. Freeze the chosen setting, then run10800s held-out seed2026103105 over
all seven policies. Selection logic remains in `public-poisson-search.md`.

All policies receive identical requests, budget and model/graph settings. no_move
is this prototype's full-reservation admission, not unmodified dynamically growing
native vLLM. Include pause, planning, copy, prefill and queue time. Report Code,
Conv and combined avg/P95/P99 TTFT and Conv avg TPOT, with sample counts. Sparse
Code arrivals make P99 descriptive, not statistically significant. Initial TTFT
benefit is admission, not causal proof of return; retain pause-inclusive return
repayment diagnostics and their exclusions. Best means within this finite grid,
not a global optimum or a guaranteed50% gain.

Failure or no qualified rate stops further submission with diagnosis state and
raw evidence. The ongoing monitoring thread must diagnose and revise a general
cost/rate model before any explicit retry; never duplicate allocations, reroll
requests, or automatically resubmit a failed stage. Scientific benefit/conclusion
is not marked complete simply because the finite-grid scripts exit successfully.
