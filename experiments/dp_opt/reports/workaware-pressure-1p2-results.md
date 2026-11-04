# Conv pressure 1.2 follow-up

## Frozen setting (2026-10-27)

This is a new follow-up, not a rewrite of completed job257484. Its baseline had
only9.3625s/.1989s Code queue waits and measured1.107351624 local physical KV
pressure. The user authorizes increasing Conv pressure back to1.2.

Measured Conv background1739.2806284373632MiB/GPU, idle Conv96MiB, Code complete
pool9600MiB, controlled budget10240MiB imply target Conv2688MiB/GPU:
`lambda_new=.01956673845194241*(2688-96)/(1739.2806284373632-96)`.
Frozen candidate is **.030863253171591748RPS**, up57.733%; Code remains.0005RPS.
This predicts1.2, but requires actual GPU measurement rather than assuming it.

Both original Code marks and all sampled Conv mark identities/lengths are
preserved. New trace has142Conv/2Code per policy,121Conv/2Code in600..4800s,
seed20261024,4800s total. TraceSHA256:
`ab6b0d3264ad1c034359f852b47715b9a5a1557c9ec854a13ee4ad8ba0f5a83a`.

Keep twoA800, dual Qwen DP2/EP2/attentionTP1, BF16/128K, default sparse graph,
pause32,24GiB virtual pools,10GiB controlled GPU budget and current profile ILP.
All five policies drain identical offered work. Reuse the verified calibration;
skip repeated N2/N9 mechanism phases only. Code tails remain descriptive n2.

Protocol: `../docs/workaware-pressure-1p2-protocol.md`.
Evidence: `evidence/workaware-pressure-1p2-20261027/`, with `prepared.json`,
`frozen/protocol.json`, `frozen/trace.json` and the fresh submission ledger.
195 integrated archive/watcher/pressure/profile tests pass; existing
asyncio_mode warning is retained. Shared expected/started/exit markers replace
login-host-local pgrep; each archive attempt preserves its own accounting and
manifest, including failed downloads. Canonical manifests only publish after
complete verification, with unsafe-link rejection and fullSHA checks unchanged.
The first combined patch failed README context verification without applying;
the corrected focused patch and subsequent regression pass.

## Verified outcome (2026-10-28)

Job **259091 completed at 2026-10-28 05:43:30**, Slurm `COMPLETED/0:0`,
elapsed7:02:47. The third archive attempt finishes full SHA verification at
approximately09:48; the first two SSH255 downloads remain intact for diagnosis.
The183-file raw manifest, independent trace verifier and shared watcher exit
barrier pass. All **five policies ×144 requests =720 requests** complete with
paired token references, four-rank sparse graphs, owner/copy/budget checks.
This completes this trace experiment, **not the overall research benefit goal**.

Code-absent, time-weighted mean Conv physical KV is2650.369219MiB/GPU;
the complete Code pool is9600MiB, controlled budget10240MiB/GPU.
Measured no_move local pressure is **1.196325119**, inside the declared1.2±.02;
aggregate ratio is.732262619. These means are not an instantaneous packing
or actual allocator feasibility certificate.

### Full arrival window: 142Conv +2Code per policy

TTFT values are **avg/P95/P99 seconds**. Conv TPOT is the mean of per-request
TPOT, in milliseconds. Quantiles use the same interpolation for every policy.

| Policy | Code TTFT | Conv TTFT | Conv avgTPOT (ms) | Combined TTFT |
|---|---|---|---:|---|
| no_move |373.1826/659.4469/684.8926|.376883/.548675/.870928|18.5303|5.554740/.621365/31.846188|
| no_move_fair |106.9281/153.7554/157.9178|.669160/.578558/9.549241|18.1808|2.144979/.737218/43.072844|
| kv_only |71.4478/86.2080/87.5200|.403942/.604413/1.245069|18.2820|1.390662/.653554/33.163957|
| kv_only_headroom |70.3363/84.1945/85.4263|.379214/.550875/.973968|18.2331|1.350840/.658839/32.213142|
| kv_only_workaware |71.7474/86.8740/88.2186|.366179/.603125/1.065557|18.1010|1.357585/.742838/31.842297|

### Predeclared primary window: arrivals600..4800s, 121Conv +2Code

| Policy | Code TTFT | Conv TTFT | Conv avgTPOT (ms) | Combined TTFT |
|---|---|---|---:|---|
| no_move |373.1826/659.4469/684.8926|.377934/.548957/.936560|18.6916|6.439798/.624498/43.208160|
| no_move_fair |106.9281/153.7554/157.9178|.720801/.735384/12.567668|18.2823|2.447750/.880786/48.847828|
| kv_only |71.4478/86.2080/87.5200|.410620/.607729/1.319623|18.4396|1.565696/.697066/43.851323|
| kv_only_headroom |70.3363/84.1945/85.4263|.383225/.565682/.993922|18.3536|1.520673/.744126/43.311460|
| kv_only_workaware |71.7474/86.8740/88.2186|.370894/.642021/1.136852|18.2407|1.531488/.770748/43.122619|

### Interpretation and adverse outcomes

- Workaware versus independent no_move reduces Code avg/P95/P99 TTFT by
  **80.77%/86.83%/87.12%**. This is a capacity-admission benefit, not proof of
  a unique performance-aware return benefit: headroom has lower Code TTFT.
- Full-window workaware versus headroom improves Conv avgTTFT3.44% and
  avgTPOT.72%, but **worsens Conv P95/P99 by9.48%/9.40% and combinedP95 by12.75%**.
  In the primary window Conv P95/P99 worsen13.50%/14.38%; do not hide them.
- Code request-average TPOT is124.613/142.775/167.973/162.844/165.973ms in
  table order; workaware worsens it33.19% versus no_move and1.92% versus headroom.
- Combined full-window P99 barely changes versus no_move although Code P99
  improves markedly. With144 requests and only2Code, the interpolated P99
  excludes the largest Code sample. Report tenant-separated tails; twoCode
  samples do not establish a robust P99 or multi-seed significance.

### Per-arrival capacity and control diagnosis

Code marks remain2463.4417805s and2663.1432232s. The first request direct-fits
under all five policies (queue approximately.06–.16s, TTFT approximately55s).
The second request provides the actual contention witness:

| Policy | Second Code queue (s) | Second Code TTFT (s) |
|---|---:|---:|
| no_move |636.3614|691.2540|
| no_move_fair |103.7896|158.9584|
| kv_only |32.9412|87.8480|
| kv_only_headroom |30.9407|85.7343|
| kv_only_workaware |33.5084|88.5548|

All three migration policies initially find both target ranks infeasible:
initial physicalGiB `[8.8125,1.3125]`, new request9.375GiB, Conv counts`[8,1]`,
nine movable requests, pinnedGiB`[.1875,.1875]`. There are no nonmovable requests
in that snapshot. After one request completes the initial pool is
`[7.875,1.3125]GiB`, counts`[7,1]`; target0 has a seven-move certificate and
target1 a one-move certificate. Workaware selects target1; its successful
transaction pays.1506s preplan,2.9231s pause,3.0967s end-to-end, one move.

Independent all-subset reconstruction on2026-10-28 enumerates511 subsets
of the nine movable requests under max-moves8, for each target and each migration
policy. **Zero subsets satisfy even final memory** in the initial snapshot.
The best final layouts exceed budget by224MiB for target0 and32MiB for target1:
`[9.5625,10.21875]GiB` and`[10.03125,9.5625]GiB` respectively. After completion,
256 subsets produce one final-feasible plan per target, with seven versus one
moves. This verifies infeasibility under the frozen rounded-destination-memory
model, **not actual allocator infeasibility**; an actual mapped-slot/placement
probe is still needed before claiming existing mapped slack can remove the32MiB
deficit. The generated audit is `independent-placement-memory-analysis-20261028.json`
and includes raw decision SHA values and per-subset aggregate accounting.

The same audit finds workaware return deferrals at Conv populations7,6,5,
all because no population-specific calibration exists. Only the population4
case executes its two-move return. Preserve these domain guards: completing
N5/N6/N7 calibration is more justified than silently borrowing the N4/N8 slope.

All-control totals from the independent verifier include ordinary admission
control and unsuccessful checks, not just successful returns:

| Policy | All moved requests | KV copied (MiB) | All control incl. preplan (s) | Return attempts/transactions/moves | Return control incl. preplan (s) |
|---|---:|---:|---:|---|---:|
| no_move |0|0|6.4621|0/0/0|0|
| no_move_fair |0|0|7.1278|0/0/0|0|
| kv_only |1|1161|12.8091|0/0/0|0|
| kv_only_headroom |4|4219.5|13.9667|9/1/3|4.6314|
| kv_only_workaware |3|3570|15.1491|4/1/2|4.9239|

Fewer moved requests do not imply less control time: workaware has more solver
and pause cost than headroom. The request-seconds remaining-work objective alone
does not enforce a Conv tail-TTFT constraint.

An independent timing-overlap audit further limits attribution. Workaware's
actual return runs at trace2826.901..2831.691s; headroom's at2995.727..2999.797s.
**No Conv arrival-to-first-token interval overlaps either actual return.**
Both full/primary ConvP99 interpolation endpoints occur before the first actual
return and overlap Code residence. Workaware's six largest Conv TTFT values are
also before its return, at Code-overlap marks2499..2753s. Thus this trace does not
support blaming its ConvP99 difference directly on the later return pause or
claiming that making that return faster fixes those already-completed first
tokens. Joint Code-prefill/decode contention, admission effects and per-run
variability need separate measurements; temporal overlap is not causal proof.
ConvP95 endpoints include post-return requests, so this observation does not
exclude later layout effects on P95 or Conv decodeTPOT. See
`independent-tail-timing-analysis-20261028.json`, which includes source SHA,
Code residency/return windows and both-window quantile endpoint records.

**Do not diagnose the initial approximately30s wait as a proven retry bug.**
Aggregate headroom alone does not prove safe migration under shared-page
rounding, pinned owners and destination-copy peak constraints. Independently
reconstruct those constraints before relaxing a safety guard. Also separate
rank/admission-choice gains from return gains: a random rank may require seven
moves where KV-only requires one.

### Evidence and monitoring limits

The evidence root contains `primary-metrics.json`,
`evaluation-window-primary-metrics.json`, `pressure-results.json`,
`all-request-comparisons.json`, `independent-trace-verification.json`,
`raw-259091.sha256` and `live-trace-259091/paired-trace/`.
Primary-metric group keys are lowercase `code/conv/combined`; workaware values
are the treatment of each paired comparison, not a separate baseline entry.

The local watcher has eight wall-clock gaps over600s, maximum4940.769811s;
the continuation also has a04:03→06:06 gap. Keep the raw timing evidence and
failed archives. PID/caffeinate presence does not establish uninterrupted300s
monitoring, and the cause of these gaps has not been independently confirmed.
Shared remote query logs and the matching terminal exit marker are archived
separately; do not conflate their continuity with local continuity.

The exact paired trace is reused in job260757 for the new random-no-return
baseline; see [random-baseline-results](random-baseline-results.md).

## Historical execution log

Job **259091** submitted2026-10-27 22:35:46, twoA800/gpu_a800. The fresh
22:36:40 query confirms PENDING/Priority; it then starts **22:40:43** on
`d1n41a28g03`.22:42:36 confirms **RUNNING/paired-trace**, profile verification
complete and startup tests ongoing. Before load the two A800-SXM4-80GB devices
have81150/81153MiB free. This is not yet measured1.2 or an admission/benefit result.
67-file oldSHA/backup/atomic sync/newSHA and fork/main/protected patches pass.
Local independent coordinator60377/PPID1+caffeinate60378 continues; remote
started marker showsln01/1891752, independently read fromln02. A bounded agent
adds three300s observations without controlling the persistent main watcher.
Continue300s
queries to terminal, fullSHA archive and independent correctness/ownership/budget
verification. Report actual pressure, Code queue/prefill and both-tenant/combined
avg/P95/P99TTFT plus Conv avgTPOT, including adverse metrics. Neither a predicted
ratio nor a queued job completes the research benefit goal.

An additional offline constant-service diagnostic uses34 prior Code-absent
Conv requests (mean residence130.837s). At the unchanged Code marks it predicts
Conv1:0 then3:2; it ignores service feedback, Code execution and real allocator
pages. It therefore warns that a mean pressure target does not guarantee every
Code request queues. Preserve both marks and test actual allocation/wait; do not
select a favorable Code arrival time. Raw diagnostic is
`fixed-service-pressure-diagnostic.json` under the new evidence root.

Local Bash3.2 rejects the empty-array/nounset skip syntax; the actual cluster
Bash5.1 was independently checked and successfully enters paired-trace.
This local legacy-shell diagnostic is not a GPU experiment failure; the active
wrapper remains frozen on the verified compute shell.

22:51 parent review: main poll22:50:52 remains RUNNING/SSH0; independent agent
has saved three successful SSH rounds22:40:19,22:45:19,22:50:19, with300.0178/
300.0112s intervals. The agent's subsequent model response returns openai_error;
retain that failure rather than treating it as a GPU error. The bounded agent
is closed; persistent coordinator60377+caffeinate60378 are still PPID1, the
63 source SHA values remain unchanged, and remote query-only monitoring continues.
No cancellation/resubmission. `monitor-parent-review-20261027-2251.json` records
the independent raw review; actual KV pressure and paired benefits are still pending.

Latest poll22:55:52 (SSH0,300.0107s interval): first **no_move_fair** has
elapsed258.688s,12offered/4completed/144,0moves. This is actual formal trace
execution but still before either Code arrival; it cannot yet establish pressure
or benefits. Independent parent verification decodes the three remote log
snapshots, checks every SHA and preserved prefix,300s timestamps, unchanged
shared watcher identity and both local command/start-time identities across
ln01→ln02→ln01 queries. See `independent-remote-query-continuity.json`.
Persistent main and remote monitoring continue to terminal; no new submission,
commit/push or Aegaeon inference. Review `pressure-results.json` plus full/main
primary metrics after complete archival and verification, rather than assuming
the first-order target or selecting a favorable Code arrival.
