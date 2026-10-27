# Work-aware return: verified five-policy trace

## Scope and completion

Job257484 uses two A800-SXM4-80GB GPUs on d1n41a28g03, allocated
2026-10-27 12:46:08–20:27:46+08:00, COMPLETED/0:0. All five policies finish the
identical trace and drain all98 requests each:490 formal requests and3,932,800
output tokens. Independent terminal verification passes453 raw-file SHA256 hashes,
both job-log hashes, paired output tokens, selected complete serial references,
four-rank sparse graphs, ownership, mapped-byte budgets and work-aware ILP/ledger
replay. This is one paired trace, not multi-seed statistical confirmation.

The frozen parent verifier rejects the already diagnosed Linux/Mac1-ULP cost
roundoff after archival. The isolated verifier accepts at most4 ULPs for finite
cost floats while preserving strict schemas, integers/ranks, boolean gate decisions,
tokens, ownership and gain thresholds; complete independent verification succeeds.
Neither serving nor audit sources change during execution or verification.

## Frozen setting

- Two independent Qwen3-30B-A3B tenants share the checkpoint, not weights/KV/EP groups.
  Each tenant has DP2/EP2/attention-TP1, BF16, max_model_len131072.
- Default sparse captures1,8,16,32,64,128,256, FULL_DECODE_ONLY, compile0, pause32.
- 24GiB virtual KV per model/rank;10GiB controlled mapped-byte budget per GPU,
  allocator utilization.98. This is not a whole-card OOM experiment.
- Code: real SWE-smith inputs101371/101670 tokens,64 output tokens,.0005RPS.
- Conv: paired Azure inputs2056–7633 tokens, synthetic8192 output tokens,
  .01956673845194241RPS. Synthetic outputs are not Azure's original output lengths.
- Poisson trace4800s, seed20261024;98 marks,96 Conv/2 Code; declared main arrival
  window600≤t<4800 contains84 Conv/2 Code. Code arrivals2463.442/2663.143s.
- `no_move` is the same full-reservation custom baseline, not unmodified native vLLM.
  Fair baseline freezes the existing queue threshold; no requests are removed.
- The new policy keeps KV-only admission and adds performance-aware post-release
  return. It does not implement performance-aware admission or joint-active calibration.

## Main window:600–4800s

TTFT is arrival-to-first-token, in seconds. TPOT is the arithmetic mean of per-request
`(completed-first_token)/(output_tokens-1)`, not an aggregate token-weighted ratio.
Two Code observations make P95/P99 descriptive interpolation only.

### Code, n2

| Policy | avg TTFT | P95 TTFT | P99 TTFT |
|---|---:|---:|---:|
| no_move |59.4694|63.4909|63.8484|
| no_move_fair |58.6369|61.9502|62.2447|
| kv_only |55.5375|55.7803|55.8018|
| kv_only_headroom |55.6005|55.7733|55.7887|
| kv_only_workaware |55.5615|55.7526|55.7695|

### Conv, n84

| Policy | avg TTFT | P95 TTFT | P99 TTFT | avg TPOT, ms |
|---|---:|---:|---:|---:|
| no_move |.3669|.5296|1.1427|17.8378|
| no_move_fair |.3739|.5200|1.0954|17.8037|
| kv_only |.3825|.5335|1.3357|17.3089|
| kv_only_headroom |.3744|.5325|1.1974|17.2739|
| kv_only_workaware |.3623|.4981|1.1692|17.0269|

### Combined, n86

| Policy | avg TTFT | P95 TTFT | P99 TTFT |
|---|---:|---:|---:|
| no_move |1.7414|.7441|56.3416|
| no_move_fair |1.7288|.8982|56.0599|
| kv_only |1.6652|.8060|55.3487|
| kv_only_headroom |1.6587|.7646|55.4661|
| kv_only_workaware |1.6460|.9326|55.4130|

Compared with headroom, Conv avg/P95/P99 TTFT decrease3.23/6.45/2.36%, and Conv
avg TPOT decreases1.43%. Code TTFT changes only.07/.04/.03%; no distinct Code
benefit is established. Combined avg/P99 TTFT decrease.77/.10%, but combined P95
**increases21.97%**. Conv P99 also increases2.32% vs no_move and6.73% vs fair.
Code avg TPOT increases12.25% vs headroom (137.8944→154.7815ms); keep that adverse
secondary metric despite the primary focus on Conv TPOT.

These are descriptive measurements, not a50% improvement or a statistically proven
unique recovery gain. The research benefit goal remains incomplete.

## Full window:0–4800s

Code values are unchanged. Report full-window results rather than selecting the
more favorable window. All timing includes complete drain.

| Policy | Conv avg/P95/P99 TTFT, s | Conv avg TPOT, ms | Combined avg/P95/P99 TTFT, s |
|---|---|---:|---|
| no_move |.3780/.5918/1.1656|17.6449|1.5840/.8482/55.2692|
| no_move_fair |.3700/.5110/1.0899|17.6024|1.5591/.7295/55.1764|
| kv_only |.3785/.5299/1.2801|17.1885|1.5042/.6082/55.2840|
| kv_only_headroom |.3741/.5432/1.1420|17.1120|1.5012/.6048/55.4200|
| kv_only_workaware |.3630/.5055/1.1067|16.8967|1.4895/.6032/55.3620|

## Why the expected large benefit is absent

The first Code sees balanced Conv1:1, mapped1.21875/1.3125GiB and reserved
1066.5/1183.5MiB. There is a local-capacity admission opportunity, but no_move only
waits9.3625s before natural release; new policy waits.9361s. Both then spend about
54.6–54.8s from reservation to first token. Removing all that first baseline queue
would improve its TTFT by at most14.64%, even before migration costs.

The second Code sees naturally packed Conv4:0 in both waiting baselines, or0:4 in
the migrating policies, with an already empty opposite GPU. It therefore has no
capacity problem: no_move queue.1989s, new policy.2520s. An average pressure ratio
near1.1 does not establish persistent local blockage at actual Code arrival times.
This trace does not reproduce two long-queued Code requests or starvation; it must
not be advertised as the intended strong admission-benefit setting.
Neither Code queue reaches the30s fairness threshold, so this trace also does not
establish a fairness-mechanism benefit.

Headroom makes29 return checks, no return; work-aware makes16 checks, one actual
return at t3166.359, population2, with3.8256s total control and gate39.22 predicted
request-s vs14.56 held-out uncertainty. That return is roughly503s after the second
Code arrives. It cannot explain either Code TTFT or Conv tails during the earlier
Code prefills. The next actual arrival is41.734s after return, while the forecast
horizon is122.464s; future births are not included in the remaining-population cost
forecast. That arrival is posthoc diagnosis only, never input to the live controller.

Several population3/4 candidates are rejected legitimately: at t2740.285,
gain33.47request-s is below45.21 uncertainty; at t2877.508,21.84 is below81.76.
Three checks defer outside the measured domain. Do not lower those bounds or call
solver-without-incumbent physical infeasibility. First examine whether the same
sparse graph contains distinct actual-attention-batch costs, using every held-out
repeat; a model correction must explain raw contrasts, not merely admit more moves.

## Monitoring, archive limitations and evidence

The earlier clamshell sleep gap remains documented; remote query continuity covers
it. One main SSH query times out at17:54:09 (exit124), then17:59:09 recovers.
No inference job is cancelled/re-submitted for either event.

First terminal archive fails only `remote-query.log` SHA because it changes between
manifest and tar creation; the first tar/error log and extracted rejected bundle
remain. The retry succeeds for all453 raw files and both logs. The original first
fixed-name manifest is overwritten by the existing retry before diagnosis; do not
claim preservation of its expected query-log digest. The second manifest is saved
separately. Login-host-local `pgrep` is not a global watcher-completion barrier.

Evidence root: `evidence/workaware-return-20261026/campaign-v2/`:

- `archive-257484.json`, `raw-257484.sha256`, both raw bundles and job logs.
- `independent-postterminal-trace-verification.json`, `independent-audit-state.json`.
- `postterminal-primary-metrics.json`, `postterminal-window-primary-metrics.json`.
- `postterminal-*-TTFT-phase-attribution.json`, `postterminal-return-control-diagnostic.json`.
- `postterminal-Code-arrival-pressure.json`, `postterminal-Code-queue-and-return-details.json`.
- `postterminal-return-horizon-diagnostic.json`, `terminal-archive-query-log-race-diagnostic.json`.

No commit/push, Aegaeon inference, trace mutation or adverse-result removal occurs.

## Post-terminal candidate diagnosis, not a deployed algorithm

After the main monitor and independent auditor exit, all53/82 source hashes are
checked unchanged once more. Only then is the4-ULP verifier fix installed locally;
historical raw/source manifests remain untouched. This is not a serving-policy change.

An offline alternative uses actual maximum attention request count as a categorical
shape feature in addition to linear peak-context work:

`delta T = gamma_N(max actual attention count) + beta_N * delta peak-context tokens`.

All192 raw-verified contrasts and every held-out repeat are evaluated, without using
trace TTFT to fit coefficients. Overall held-out RMSE decreases.6512→.5487ms,
15.75%; ConvN4 worst residual decreases1.8042→1.2517ms. However ConvN3 held-out
RMSE increases.6261→.7067ms and ConvN9 worst residual increases3.8300→3.8447ms.
N8/N9 also lack some candidate count levels; populations5/6/7 remain unmeasured.
The alternative is therefore **not installed or claimed as a GPU benefit**. Retain
the adverse folds and obtain independent, complete-domain validation before serving.

Next-phase gates are in `docs/workaware-return-next-phase.md`: separate placement
pressure from cost-model validation, preserve sparse graphs and baselines, freeze
rates only from stable residency/allocator measurements, and never tune to a desired
positive quantile or retrospectively pick Code arrivals.

The candidate comparison now has a reproducible entry point,
`analyze_workaware_partition_cost.py`, with strict measured-count/identifiability/
held-out tests. It rebuilds the archived raw calibration independently and produces
the same192-contrast diagnostics. Installed verifier and analysis regression:
77 passed, with the existing unknown `asyncio_mode` configuration warning retained.
The first local patch/test invocation used the wrong working directory and attempted
to assign zsh's read-only `status`; that failure log is retained. Corrected invocation
passes, and installed N9 plus full five-policy raw verification pass separately.
The expanded work-aware suite passes135 tests in57.50s with the same existing
warning. A transient metadata-check syntax error also remains recorded; corrected
metadata checks confirm only the post-terminal verifier changes among the53 old
sources, and all82 isolated sources plus protected vLLM remain unchanged.

Both terminal tar files have the same SHA256. The first retained extraction passes
the final stable453-file manifest too; this does not recover the original overwritten
manifest or retrospectively make the first attempt successful. Details:
`postterminal-bundle-digests.json`, `postterminal-local-source-inventory.json`.
