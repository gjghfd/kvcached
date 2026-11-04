# From the closed return probe to a Poisson experiment

2026-10-30. Planning only: job268245 and all its serving sources remain frozen.
Wait for all three seeds, terminal raw SHA verification and independent checks
before changing live routing, syncing or requesting another allocation.

## Input and measurement corrections

Use the exact frozen AzureConv length bank and real SWE-smith >=100K/O64 bank.
The AzureConv file contains **574,063 distinct length-pair marks representing
823,814 weighted source rows**, not574,063 weighted requests. Retain the weights;
this clarification changes no sampled request or input SHA. Conv prompt content
is synthetic with Azure-sampled lengths; O24576 is also synthetic. Keep the
original paired output length and source fields for every request.

First-seed no_move Conv submitted-to-completed mean is296.262s. Admitted Code
service is31.349s under no_move after natural drain, but63.357–63.789s with Conv
active under the migration policies. Do not use the idle/draining baseline
service as the jointly active Code residency estimate. These measurements are
from a closed N2 cohort, **not steady open-loop service calibration**.

Evidence: `open-loop-rate-prior-calculation-r2.json` under the current recovery
root. It rechecks the frozen input hashes and first seed's complete raw SHA.
The initial calculation is retained; r2 corrects bank-count terminology and
Code duty estimation, rather than hiding the first estimate.

## Rate priors, not a frozen workload

Conservative fresh-page means are2726.333MiB per long Conv and9513.600MiB per
Code. With B=10240MiB and192MiB/rank pinned-pool allowance, a toy N2-service
estimate for mean single-rank pressure1.2 is

```
lambda_Conv = 2 * (1.2*B - pinned - E[b_Code]) / (E[b_Conv] * S_Conv)
            = 0.00639438 requests/s
```

This assumes KV/service independence and load-independent N2 service. Neither
assumption is verified: sparse graph thresholds, births, prefill interference,
waiting and migration change service and occupancy. Fresh-page arithmetic is
not the actual allocator/copy peak. The existing0.00565253RPS remains a prior
only; do not replace it in the running closed probe or describe either number
as measured Poisson RPS.

For the next calibration, predeclare Conv0.00565/0.0064/0.0072RPS and
Code0.00025/0.0005/0.001RPS candidates. Choose a rate from measured exposure,
queue stability, supported population/context domains and roughly1.2 pressure,
not from whichever policy happens to have the best tail. Keep all negative and
zero-Code realizations. A separate frozen evaluation must follow calibration.

| Code RPS | Expected Code/1200s | Expected Code/7200s | Approx active-Conv duty prior |
|---|---:|---:|---:|
|0.00025|0.3|1.8|1.59%|
|0.0005|0.6|3.6|3.19%|
|0.001|1.2|7.2|6.38%|

At0.00025RPS a1200s Poisson trace has74.08% probability of no Code request.
The existing mixed-trace generator rejects missing tenants; adapt the next
protocol to retain zero-Code seeds rather than rerolling or inserting a forced
Code arrival. Never call a forced arrival a Poisson draw. Even7200s at0.001RPS
has only7.2 expected Code requests per seed: report counts and empirical tail
quantiles, not established P99 significance. Sparse Code arrivals leave long
return-amortization opportunities but require longer evaluation for Code tails.

## Algorithm and performance domains

Use capacity-aware headroom admission plus the robust release interface
minimizing calibrated remaining-work cost + unchanged differential uncertainty.
Before pause charge future activation/copy; after pause log, but do not again
subtract, already paid blocking/planning. Keep actual control costs in TTFT,
TPOT and pause-inclusive repayment. Freeze and log the optimizer choice so old
point-policy evidence is never silently reinterpreted as robust-policy evidence.

The new long-context profiles cover N2/N3 with the other tenant resident idle.
They are not mixed-active or arbitrary-N calibration. A Poisson trace can enter
unsupported populations: retain conservative no-return decisions and count
domain rejections. Do not extend a global maximum to every population.
Fully concentrated N4 at32K exceeds the10GiB/rank budget, so future feasible-
layout calibration needs an explicit protocol and independent coverage checks;
blindly enumerating all long-context splits or raising calibration budgets
would not validate the original serving setting.

Future Conv births may naturally relieve concentration earlier than a closed-
cohort drain forecast. Record birth/release/return timing and matched-work
repayment in the open-loop experiment; closed N2 gains do not prove the forecast
is accurate under births. No empirical guard is reduced to force return.

## Evaluation gate

- Retain no_move, no_move_fair(tau30), random_no_return, headroom and workaware
  comparisons under identical traces, budgets, sparse graph and model settings.
- Keep two A800, independent DP2/EP2/attentionTP1 tenants, BF16/128K/24GiB virtual
  pools, allocator.98,10GiB controlled budget, pause32 and real dp_echo copying.
- Independently prove global space/local insufficiency and physical copy peaks
  for each migration admission; distinguish queueing from whole-card CUDA OOM.
- Report Conv/Code/combined avg/P95/P99 TTFT and Conv avg TPOT, all failures,
  population-domain rejections, fairness events and predeclared offered windows.
- Drain every offered request; do not claim makespan is fixed-rate throughput,
  delete difficult requests or hide old-Conv regressions.

The three-seed closed mechanism result and robust CPU replays remain preliminary
evidence. The final Poisson research-benefit goal is not completed by this plan.
