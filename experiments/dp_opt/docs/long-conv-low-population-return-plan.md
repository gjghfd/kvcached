# Candidate: long Conv, low population, unchanged KV pressure

Status2026-10-29: **offline candidate, not selected, synchronized or GPU-tested**.
Update2026-10-30: the user selected a [closed N2 long-output mechanism probe](long-conv-return-probe.md)
first. The open-loop rate/generalized profile plan in this document remains a
follow-up, not the active or already measured configuration.
Keep265356 and all its frozen comparisons running. Consider this next setting
only after its complete SHA archive and independent verification show the
remaining return/steady-decode problem. Preserve all negative results.

## Why this tests the design more directly

Current Conv output8192 needs roughly4–6 resident requests for pressure1.2.
Most such layouts remain on default sparse graph8. Reducing Code RPS alone
lengthens the quiet interval but does not slow Conv's natural rebalancing or
remove its small-population uncertainty/domain gates.

The candidate keeps Code real100K input/output64 and its`.00025RPS`Poisson
stream, but extends synthetic Conv output to **24576**, using the same complete
fixed Azure input bank. It lowers Conv RPS to target roughly **two** long-lived
requests, not lower KV pressure. Default sparse captures, both ranks' EP,
BF16/128K,10240MiB/GPU budget, allocator.98, virtual pools and pause32 remain
unchanged. This exposes the real default sparse transition`1:1 -> 0:2 -> 1:1`
(graph1 -> graph8 -> graph1); it is not a pure kernel-attention experiment.

The complete frozen Conv bank has574063 rows, input2048..7999. All10 Code marks
have100084..101675 input tokens. No favorable requests or seed are selected.
Full-source arithmetic assumes the observed uniform16-token blocks,64 blocks
per96MiB page and one null block. It remains **arithmetic, not allocator proof**:

| Quantity | Fresh-page estimate |
|---|---:|
| Conv input + new output |26624..32575 tokens|
| Mean one-Conv mapped pool |2744.791MiB|
| Maximum Code mapped pool |9600MiB|
| One Conv + Code /10240MiB |1.190625..1.2375, mean1.205546|
| Code rank after evacuating Conv, including idle Conv pool |9696MiB|
| Other rank with any two Conv, including idle Code pool |at most6240MiB|
| Other rank with any three Conv, including idle Code pool |at most9312MiB|

Thus with two balanced Conv requests, **neither** rank can fit Code, but an
evacuation has a large arithmetic margin on the other rank for every input in
the fixed bank. Four or more Conv are not guaranteed evacuable; preserve them
and their genuine queues rather than deleting requests or treating aggregate
space as an indivisible-placement certificate. Verify actual source-page owners,
full reservation/mapped bytes, destination preparation and copy peak on GPU.

## Why performance-aware return can differ from headroom

After Code completes with Conv`0:2`, the current hard-headroom baseline can
already fit another Code on the empty rank. Its`headroom_scenarios`preserves
that existing safe rank, imposing final capacity roughly`B - b_Code =640MiB`.
Even the shortest new Conv is too large to move there. This is a structural
hard constraint, not random target direction or a high-Code-rate effect.

Workaware instead prices that future capacity loss with its online Poisson
hazard and measured cold-migration cost. At a low Code rate it may accept
`1:1`because a long remaining Conv decode horizon repays real migration/pause
cost before the next Code is likely to arrive. Births and completions can still
naturally rebalance random/headroom; measure their time rather than claiming
indefinite imbalance. The algorithm and cost terms remain workload-generic.

## Rate is a prior, not a frozen result

Use the joint residency/KV moment, not independent average size and lifetime:

`lambda_Conv = 2 * (1.2*B - b_Code - idle_Conv) / E[b_Conv_increment * S_Conv]`.

On the complete bank, a legacy global-surrogate extrapolation gives346.125s
mean Conv service,917111.787MiB*s joint moment and **.00565253RPS** as a starting
prior. Those coefficients were not calibrated for this new long-context N2
domain, and a full-reservation shared pool is not exactly independent request
pages. Therefore do not freeze this number from CPU arithmetic. First measure
paired length/service and actual mapped exposure, then use an open-loop pilot
with a warmup lasting multiple measured service times. Freeze a rate from the
same predeclared stable1.2 physical-pressure rule, keeping all requests/timeouts.
No claim of steady state or benefit follows from this prior.

## Required calibration and execution gates

The old N2/N3 profile context envelope24575 is too short. A candidate plan uses
both tenants, two independent repeats, all N2/N3 rank splits and prompts
`8192,16384,24576,31744`, with existing3072 output and1536..2560 sample window.
The existing CPU plan screener accepts **112 cases**, maximum9984MiB per rank
including pinned pools. This does not prove GPU fit. Its sampled upper context
can cover the new maximum32575-token Conv without cutting the horizon.

After the active job is terminal and its full archive is verified:

1. Freeze a versioned calibration plan; recheck longest Code and representative
   long Conv prefill/full-reservation/copy with all four default graph ranks.
2. Run the declared calibration and independent held-out repeat/context/layout
   verification. Preserve old raw and unaffected profiles; overlapping N2/N3
   profiles require explicit merged-raw/refit provenance, not a silent overwrite.
3. Measure long-Conv service/KV exposure, calibrate open-loop RPS and freeze an
   independent trace/config with unchanged Code marks and paired input sampling.
4. Run no-move, timeout-fair no-move, random no-return, KV-only, hard-headroom
   and workaware on the same trace/config, preserving actual execution order.
5. Archive SHA and independently verify all requests/ownership/budgets/graphs.
   Compare Conv/Code/combined avg/P95/P99 TTFT and Conv full-request avgTPOT;
   retain control/queue/prefill/copy costs and adverse results.
6. Supplement primary metrics with release-to-return, baseline natural recovery
   and the predeclared128-token/32-settlement same-ordinal decode diagnostic.
   Few Code arrivals and one trace cannot statistically confirm Code P99 or
   causal/generalized return benefit; repeat/expand if the mechanism is sound.

No capture changes, forced idle-EP skipping, silent length truncation, revised
SLO120s, source overwrite during265356 or new allocation request is authorized
by this candidate. Existing user permission to iterate still requires terminal
verification and protected synchronization before the next actual execution.

Evidence:`evidence/workaware-anchored-replay-20261029/next-low-population-arithmetic-v2-full-bank.json`.
The earlier299-sample arithmetic is retained separately, not overwritten.
