# Random no-return baseline: implementation and prospective comparison

Recorded 2026-10-28, Asia/Shanghai. **All six formal policies and N2/N9 auxiliary
fixtures are complete and independently verified. Capacity migration beats the
non-migrating baselines, but a unique workaware-return benefit is not established.**
Predecessor259091 completed05:43:30 and its complete SHA archive,
independent five-policy verification and watcher exit barrier pass by09:48.
The tested candidate has now been promoted to this user fork's `main` worktree
using the official apply_patch executable, followed by73-file protected sync.
These are uncommitted changes; HEAD remains9af69c2, with no new commit/push or
Aegaeon execution. The isolated tested copy is retained, not the active version.

## New baseline

`random_no_return` uses normal direct admission. When a new request cannot fit,
it can move requests belonging to either model, computes bounded one-way greedy
physical-owner/copy-peak certificates, and uniformly draws one certified DP rank.
The greedy order is largest exclusive source release, then smaller destination
memory and request ID. It uses **no ILP, attention objective or proactive return**.
Reproducible serving-seed/repeat/request-ID draws and unchanged choices across
before/after-pause revalidation are audited. Unsafe actions queue, never discard.
See [the complete protocol](../docs/random-baseline-protocol.md).

## Complete six-policy result (2026-10-28 evening)

260757 ends **19:07:47 Asia/Shanghai, COMPLETED/0:0**, after9:17:38 on the
same twoA800. The coordinator finishes archival/analysis by19:15. Parent fresh
SSH/accounting at19:24:17 confirms terminal status and an empty user queue.
**493 raw files plus both Slurm logs** independently pass SHA verification;
the separate remote Bash terminal packer has the identical493-file manifest.
All132 early formal-subset files are unchanged in the final archive.

The full verifier independently passes **6×144=864 formal requests**, plus the
9N2 and30N9 auxiliary requests. Formal token reference is now **no_move**, not
the partial random reference. All configured trials, exact arrivals/prompts/
output tokens, all four graph ranks, physical ownership/budgets, random seeded
choice/no elective return and workaware decisions are checked. The cached last
progress remains144offered/143completed; do not replace that raw checkpoint with
144/144. Complete per-request records and summaries establish completion.

### Primary predeclared600..4800s window

121Conv/2Code per policy; TTFT seconds and request-averageConvTPOT milliseconds.

| Policy | Code avg/P95/P99 TTFT | Conv avg/P95/P99 TTFT | Conv avgTPOT | Combined avg/P95/P99 TTFT (n123) |
|---|---|---|---:|---|
| no_move |374.012702/661.128537/686.649944|.415476/.818492/1.194050|18.878626|6.490228/.957763/43.433034|
| no_move_fair |185.330439/302.829826/313.274216|7.876850/64.808290/146.096519|18.666455|10.762275/91.449535/151.187898|
| random_no_return |69.777668/83.261319/84.459866|.411465/.634281/1.260645|19.019043|1.539371/.842487/43.758040|
| kv_only |69.655494/82.930951/84.110992|.410798/.648099/1.091325|18.410030|1.536728/.717670/43.813586|
| kv_only_headroom |71.286724/86.252888/87.583214|.406340/.628153/.922383|18.427423|1.558867/.677028/43.563841|
| kv_only_workaware |71.050235/85.597075/86.890127|.401959/.668536/.950938|18.428595|1.550711/.764808/43.589165|

Workaware versus no_move reduces Code avg/P95/P99TTFT **81.003%/87.053%/87.346%**,
Conv avg/P95/P99TTFT3.253%/18.321%/20.360% and ConvavgTPOT2.384%.
Versus fair, Code avgTTFT improves61.663%, avoiding fair's large Conv waiting.
These establish the capacity-placement benefit, not its uniquely performance-aware
component. Code remains two samples and interpolated tails are descriptive.

Against random, workaware's CodeavgTTFT is1.824% worse; ConvavgTPOT is3.105%
better, ConvP95TTFT5.401% worse andP99TTFT24.567% better. As the earlier timing
audit shows, theP99 endpoints precede all migrations; do not attribute that
difference to return. Against headroom, ConvavgTPOT is essentially unchanged
(18.428595vs18.427423ms), ConvP95/P99TTFT regress6.429%/3.096% and combined
P95 regresses12.966%. **Workaware's unique tail/return goal remains unmet.**
The combinedP99 also excludes/interpolates away the worst Code wait at this
sample mixture: no_move43.433034s versus workaware43.589165s. It is not evidence
that the longest Code wait improves by only that amount, or robust mixed-tail SLO.

### Full0..4800s arrival-window cross-check

142Conv/2Code per policy; Code distributions are identical to the primary window.

| Policy | Conv avg/P95/P99 TTFT | Conv avgTPOT | Combined avg/P95/P99 TTFT (n144) |
|---|---|---:|---|
| no_move |4.618468/2.379863/102.110069|18.774837|9.748943/54.611667/116.279800|
| no_move_fair |6.762227/41.853749/145.680931|18.487586|9.242341/63.303381/149.923704|
| random_no_return |.400004/.581370/1.169403|18.794237|1.363583/.766904/33.221964|
| kv_only |.403675/.626105/1.048388|18.272504|1.365505/.660292/33.226340|
| kv_only_headroom |.398566/.587659/.911468|18.280274|1.383124/.659902/32.974294|
| kv_only_workaware |.394456/.572986/.926097|18.303054|1.375787/.735979/32.804792|

No_move has an early startup admission transient: ConvIDs2/3/4 wait roughly
113/119/87s at arrivals102/103/135s, before any Code arrival. All six cases start
with empty counts and mapped.1875GiB/GPU, so retained KV ownership is not shown
as its cause. The present admitted-decision archive does not fully diagnose those
rejected early attempts. Do not attribute that full-window ConvP99 advantage to
cross-model Code migration; retain both windows and prioritize the predeclared
primary window rather than choosing the larger full-window improvement.

### Actual capacity and cost witnesses

Measured no_move local pressure is **1.198629529**, inside1.2±.02. At the verified
second-Code migration snapshot, all four migration policies have controlled free
space **2.125/8.6875GiB**, total10.8125GiB, while new allocation requires9.28125GiB
on either DP rank. **Aggregate enough, neither rank enough** is witnessed by
actual frozen owner/page inventories and the shared allocator increment. Real
migration/admission then succeeds. This is controlled mapped-budget pressure,
not proof of whole-card CUDA OOM.

Second Code full-reservation queue/TTFT seconds: no_move637.871239/693.030296;
fair102.133938/315.885313; random29.210241/84.759502;
kv_only29.828367/84.406002; headroom33.288455/87.915796;
workaware32.667110/87.213390. Fair's reservation→first-token time213.751s also
regresses, not just its queue. Use full `reserved_at-arrived_at` for queue; the
older `arrival_waits` helper used submission time and omitted the new random
policy. Its old diagnostic is preserved separately, and the final parent
`parent-full-reservation-pressure-results.json` corrects both definitions.

All moves/total control including preplan: no_move0/10.375839s;
fair0/7.870235s; random7/12.424311s; kv_only1/11.564168s;
headroom4/14.835254s; workaware3/17.439742s. Headroom returns3requests once;
workaware returns2 once. Reduced copy count does not imply lower end-to-end cost.

### Posthoc return opportunity and transfer diagnosis

Recorded2026-10-28 evening from the same immutable260757 archive; no new GPU
experiment or runtime fix is claimed. Reproducible input hashes, matched cohorts,
raw population/cache witnesses and transfer stages are saved in
`evidence/workaware-random-baseline-20261028/return-opportunity-analysis-20261028.json`.

The hypothesis that random naturally rebalances shortly after Code release is
**not supported**. Its second Code completes at2758.379s, but no subsequent
stable multi-request population sample has Conv on both ranks. At the next Conv
arrival2833.800s, the actual population has no Code and GPU0 mapped.1875GiB;
the direct-admission snapshot still caches Code9.375GiB on rank0. Consequently
it excludes that empty rank and admits Conv to the already-hot rank1. KV-only
has the analogous stale other-tenant upper bound with the opposite orientation.

`PlacementController.direct_admission_epoch` refreshes the arriving tenant and
refreshes all tenants only when no cached candidate fits. `live_trace.complete`
does not invalidate the other-tenant admission cache on completion. A safe
reservation upper bound therefore becomes stale for count-balanced placement.
The successful workaware return invalidates the cache and refreshes all tenants;
its subsequent Conv arrivals can use both ranks. This is a shared runtime/cache
issue, not intentional random-baseline behavior. **Current differences conflate
return, observation refresh and rank orientation.** Preserve the measured data
and verified capacity witnesses, but do not claim isolated return benefit.
All policies need completion/release cache invalidation with admission-generation
synchronization before a prospective return or rate comparison.

Successful workaware return starts2827.146s and resumes2831.672s,65.035s after
its Code completes2762.110s. MissingN5/N6/N7 calibration delays it untilN4.
A matched57-request posthoc arrival cohort2832..4800s has ConvavgTPOT18.8203ms
random,17.6216msKV-only and17.4275msworkaware. These descriptive7.40% versus
random and1.10% versusKV-only differences are **not** a predeclared primary
result or a controlled return estimate; cache freshness and rank orientation
are not isolated. The four Conv requests active during return still have worse
whole-requestTPOT under workaware, and the reported P99 endpoints precede it.

The actual tested transport is `dp_echo`, not the default `host_file` fallback.
It gathers logical KV blocks on the source GPU, uses bounded existing DP
`all_gatherv`, scatters on the destination, echoes destination bytes to verify
them, then commits/reroutes and releases the source. Both tenants reach their
default32 DP pause boundary; no migration/inference overlap is implemented.
Workaware admission's one migration takes.069676s total/.052756s peer RPC;
random's seven take.421970s/.296710s respectively. These are different transfer
scopes, not pure NVLink bandwidth measurements. Return has3.429767s pause-call
waiting, including2.712s for idle Code while Conv can still decode. Its active
Conv boundary-to-confirmed-resume interval is.583996s, not3.43s of global idle.

The user accepts paused inference during migration; retain this protocol and
do not make asynchronous overlap a prerequisite. Lowering Code from.0005 to
**.00025RPS** is a prospective candidate, not a demonstrated gain. At86s useful
horizon, Poisson next-Code probability changes4.21%→2.13%. The actual online
return rate is.0005859 and its85.994s probability4.91%; its winning scenario
already leaves Code headroom unprotected. Rate alone therefore does not resolve
missing calibration or control costs. Single-event saved-work opportunity is
`s * (1 - exp(-lambda * W)) / lambda`, where `s` is saved request-seconds per
second and `W` is limited by remaining useful work and genuine natural repair,
not the full mean Code gap. Subtract measured blocking/risk/uncertainty without
double-counting transfer cost already included in blocking.

Halving the rate scales this seed's Code marks2463.442/2663.143s to
4926.884/5326.286s: a4800s replay would have **no Code**, not a better return
policy. A new prospectively frozen trace needs longer duration and enough
paired seeds/events, with unchanged Conv rate/lengths/budget and per-model
metrics.9600s retains these two marks but still cannot establish CodeP95/P99.
First fix cache freshness across policies and calibrate earlier return; then
test reduced-rate traces without interpreting mixed-metric dilution as gain.

### Terminal monitoring and post-run source work

The agent's terminal audit verifies509monitoring raw files and preserves61earlier
files;88sidecar queries include oneSSH124timeout. Local wall gaps2195/2309/398s
and reader gap2407s remain recorded. The final complete remote-query log contains
112polls, maxgap **430s at14:25:15→14:32:25**, not perfect300s continuity. Full
log resolves the saved-tail600s omission without erasing the real430s gap.
All coordinator/sidecar/caffeinate processes exit naturally; the read-only agent
is closed after its audit. Retained worker-socket warnings are not silently cleaned.

After confirmed terminal archival only, exact graph-selector/source-owner release
reuse is promoted locally to`ilp_reference.py`/`workaware_ilp.py`. It preserves
every objective coefficient, independent attention segment, ownership/pinned-pool
constraint and solver limit; identical-member binaries are shared rather than
repeated. The earlier raw N9 CPU optimum audit remains applicable, but the old
uncertainty gate still rejects its candidate. Also fix random control subtotals
and full-reservation wait definitions, with **117+81=198disjoint targeted tests
passing**, and re-run the complete formal verifier successfully after the changes.
These changes occur after260757; **no GPU result is attributed to them**. They
are not GPU-synchronized or published; HEAD remains9af69c2 ongjghfd/main.

Next verification gates are [phase-calibration-plan](../docs/workaware-phase-calibration-plan.md).
Do not blindly retune this already-correct pressure or lower uncertainty to force
a positive result. Raw/formal metrics, integrity, pressure, capacity witnesses,
monitor audit and post-run tests remain beneath the campaign evidence root.

## Formal completed-policy subset (14:13 checkpoint)

At2026-10-28 **14:12:44**, the independent existing sidecar observes260757
RUNNING/no_move_fair,66offered/61completed/144. Workaware finished its case
around12:14 and random around13:37. Their immutable completed subtrees plus
launch/calibration sources are archived separately: **132 files**, identical
remote-before/after manifests, bundle SHA and every downloaded-file SHA verified.
An isolated explicit `completed_policies` verifier passes both policies' **288
requests**, all four graph ranks, frozen prompt/token references, ownership,
physical budgets, migration peaks, control totals, seeded random revalidation
and workaware return checks. The unchanged full order/config remains six policies;
no success exit marker or aggregate `passed` summary is created or substituted.
The partial verifier reports `allocation_success_verified=false`, and does not
replace terminal accounting, the complete job archive or no_move token comparison.
It is isolated under `.workaware-stage/partial-verification`;14new tests and77
existing compatibility tests pass. The68 active local runtime hashes are unchanged.

Full arrival window0..4800s, TTFT seconds, request-averageTPOT milliseconds:

| Policy | Code avg/P95/P99 TTFT (n2) | Conv avg/P95/P99 TTFT (n142) | Conv avgTPOT | Combined avg/P95/P99 TTFT (n144) |
|---|---|---|---:|---|
| random_no_return |69.777668/83.261319/84.459866|.400004/.581370/1.169403|18.794237|1.363583/.766904/33.221964|
| kv_only_workaware |71.050235/85.597075/86.890127|.394456/.572986/.926097|18.303054|1.375787/.735979/32.804792|

Workaware versus random: Conv avg/P95/P99TTFT reductions **1.387%/1.442%/20.806%**,
Conv avgTPOT reduction **2.614%**, but Code avg/P95/P99TTFT regressions
**1.824%/2.805%/2.877%**. Combined averageTTFT regresses.895%; itsP95/P99 improve
4.033%/1.256%. These are single-pair descriptive values, not a demonstrated50%
gain or stable tail improvement. Code's two samples do not support robustP99.

The predeclared600..4800s window has121Conv/2Code:

| Policy | Conv avg/P95/P99 TTFT | Conv avgTPOT | Combined avg/P95/P99 TTFT (n123) |
|---|---|---:|---|
| random_no_return |.411465/.634281/1.260645|19.019043|1.539371/.842487/43.758040|
| kv_only_workaware |.401959/.668536/.950938|18.428595|1.550711/.764808/43.589165|

WindowConvP95 regresses despite betterP99; retain this adverse result.

### Timing and mechanism limits

Random chooses target0 and migrates seven Conv requests; workaware chooses target1
and migrates one, then later returns two. However **both policies' ConvP99 endpoints
occur around2509..2527s, before the second Code arrival2663.143s and before any
migration**. Workaware's successful return is2827.146..2831.672s; no ConvTTFT
interval overlaps it. The reported20.806%P99 difference therefore cannot be caused
by either that admission choice or the later return. It motivates paired repetition
and joint Code-prefill/Conv calibration, not a causal return-benefit claim.

The four Conv requests active at return have paired full-request avgTPOT
**24.086402ms random versus24.944664ms workaware (+3.563%)**. This cohort is
posthoc/descriptive, not an isolated counterfactual, but also does not establish
net recovery benefit. Workaware again defers N7/N6/N5 and returns only atN4.

Path-neutral accounting finds total control including preplan **12.424311s random
versus17.439742s workaware**. Their pause calls are.844273s versus6.537361s;
actual migration time.421970s versus.189467s. Fewer copied bytes does not offset
these measured pause costs. For the successful admission, the idle Code tenant's
default32-boundary armed→reached wait is about.677s random versus2.958s workaware,
while the Conv boundary is about.085s in each. The additional workaware return
also pays an idle-Code boundary wait about2.712s. Preserve default32 and sparse;
next calibration must separate incoming waiting, idle-tenant handshake and genuinely
blocked old-request work rather than charging all costs as copied-byte costs.

The existing `verify_trace.control_metrics` helper omits `blocked_random_greedy`
from its staged-only breakdown, although independent overall totals are correct.
The isolated diagnostic explicitly counts all paths: random really has a pause
and seven copies, not zero. Do not quote the helper's zero random subtotal as a
measurement. No active runtime helper is patched during260757.

Evidence: `evidence/workaware-random-baseline-20261028/early-formal-pair-20261028-1406/`,
including `independent-subset-analysis.json`, `independent-window-control-timing.json`,
closure snapshots, bundle integrity and full132-file SHA manifests. The first
SFTP brace-path download fails; the successful explicit archive-directory download
and original remote artifacts are retained. Continue the same allocation through
all six policies, then perform full independent verification and comparison.

## Active experiment:260757

Job260757 is the sole new allocation, submitted2026-10-28 **09:49:52**,
and starts **09:50:09 on d1n41a24g01**, twoA800. The fresh09:58:47 SSH query
shows **RUNNING/recovery-2**, not a completed inference or benefit experiment.
First validate N2/N9 auxiliary recovery fixtures. Then compare no_move,
no_move_fair,random_no_return,kv_only,kv_only_headroom,kv_only_workaware on the exact
same4800-second trace,142Conv/2Code,Conv.030863253171591748RPS/Code.0005RPS and
unchanged lengths/budgets/sparse graph/pause32. No second allocation is requested
while the predecessor was active; no favorable Code arrivals are selected.
Keep BF16/131072,24GiB/model/rank virtual pool,10240MiB/GPU controlled budget,
allocator.98, sparse`[1,8,16,32,64,128,256]`, FULL_DECODE_ONLY, compile0 and
pause32, with the independently rebuilt population calibration. Runtime sources
are frozen while260757 executes. All phases continue within this20-hour allocation.

### Verified N2 auxiliary result (10:26)

The completed recovery-2 subtree and its remote verifier are separately archived:
**147 files**, remote-before/after manifests identical, all downloadedSHA verified.
An independent local `verify_trace.py` run passes `raw_trace_verified` for all
**three policies ×3 requests =9 requests**, including random-choice revalidation,
token references, ownership/copy/budget checks and sparse graphs. The whole
allocation remains RUNNING;10:20:06 main poll shows **recovery-9**. Neither N9
nor the formal six-policy trace is completed at this checkpoint.

This is a closed-population N2 fixture, not a Poisson trace or tail significance.
Its two Conv requests arrive near0s; one100K Code request arrives20s. TTFT is
avg/P95/P99 seconds; Conv TPOT is per-request average milliseconds.

| Policy | Code TTFT (n1) | Conv TTFT (n2) | Conv avgTPOT (ms) | All moves | All control incl. preplan (s) |
|---|---|---|---:|---:|---:|
| random_no_return |55.04163/55.04163/55.04163|.15940/.20622/.21038|21.88958|1|1.1915|
| kv_only_headroom |55.11199/55.11199/55.11199|.18355/.23287/.23726|21.87562|1|1.3340|
| kv_only_workaware |55.02802/55.02802/55.02802|.18890/.24431/.24923|18.77911|2|4.8041|

Workaware executes one elective return, costing3.6721s including preplan;
random has no proactive return and headroom executes none. Conv averageTPOT
improves **14.21% versus random**, with adverse Conv TTFT values retained.
These Conv first tokens precede Code arrival and all migration/return actions,
so their TTFT differences cannot be caused by the later return. This fixture
validates the requested baseline and a decode-recovery mechanism, **not the
target tailTTFT benefit or a50% improvement**. CodeTPOT is156.496/151.135/138.225ms
in table order; Code completes before return, so its differences are not return gains.

Evidence: `early-recovery-2-20261028-1023/` beneath the campaign evidence root,
with `archive-integrity.json`, before/after manifests, `independent-verification.json`,
`paired-comparisons.json`, `primary-metrics.json` and full raw subtree. This partial
archive does not replace complete terminal archival and verification.

### Verified N9 auxiliary result (10:53)

The N9 subtree independently passes the same147-file before/after SHA archive
and local full verifier, **three policies ×10 requests =30 requests**. This
also exercises multiple one-way migrations under the requested random policy.

| Policy | Code TTFT (n1, s) | Conv avg/P95/P99 TTFT (n9, s) | Conv avgTPOT (ms) | All moves | All control incl. preplan (s) |
|---|---:|---|---:|---:|---:|
| random_no_return |55.50779|.54373/.94039/.97552|27.69241|5|2.7397|
| kv_only_headroom |55.42471|.57166/.96273/.99793|27.96057|4|3.8297|
| kv_only_workaware |54.20081|.55573/.94705/.98204|27.56477|4|3.4892|

**Neither headroom nor workaware executes a return in N9.** Workaware's.46%
ConvTPOT difference versus random is therefore not evidence of a return benefit.
All initial Conv first tokens again precede Code arrival. Preserve these results
and all seven return checks, rather than reporting this fixture as a tail win.
Evidence: `early-recovery-9-20261028-1048/` beneath the campaign evidence root.

### Isolated exact ILP compiler optimization (11:07)

N9's first return has52 remaining-work segments and nine headroom scenarios.
The old1s overall budget assigns approximately.111s per scenario. In three
unconstrained/unprotected-Code scenarios the archived planner reports
`no_verified_incumbent`; selecting the unchanged reference is logged as
`no_profitable_rebalance`, which **does not prove an optimized return was unprofitable**.

A candidate is implemented only in `../.workaware-stage/solver-reuse/`, not active
main/remote sources. It shares graph count selectors when model/population/current
request membership is identical, sums their original graph objective coefficients,
and retains every segment's independent attention weights/variable. Physical
pages share release variables only with identical rank/pool/owners/pinned scope;
all page bytes and future-destination-owner constraints are retained. This is
exact structural deduplication, not shorter forecast, a changed performance fit,
relaxed capacity guard or larger deadline.

47 CPU tests pass, including existing exhaustive ILP/remaining-work tests plus
pool/pin/byte/count-selector checks. Initial test-construction failures and the
first add-file patch formatting failure are retained, not called passing runs.
On the same raw N9 snapshot and same per-scenario deadline:

- Old compiler, three local runs: total solver wall.773/.458/.468s; best objective
  159.772 request-seconds, no moves, three scenarios without incumbents each run.
- Reused compiler: .321/.040/.040s; best objective33.382, four moves, no missing
  incumbents. All three runs ×nine scenarios match independent all-subset replay
  under max-moves4 for the frozen one-way candidate domain (163 subsets/scenario).
  First-run import/startup cost is included; these are **local CPU**, not GPU gains.
- Critically, the unchanged pre-pause gate **still rejects** that candidate:
  predicted net gain126.390 request-seconds, uncertainty448.792. The optimization
  exposes solver-limited search but does not make current calibration sufficient
  or establish a GPU return/tail benefit. Do not lower that bound just to pass.

Evidence includes `solver-reuse-regression-final.log`, legacy/optimized probeJSON
and logs, source SHA, exhaustive proofs and `solver-reuse-prepause-gate.json`.
All68 active runtime SHA values remain unchanged. The formal six-policy trace
starts with actual **kv_only_workaware**, confirmed by10:54:57/10:59:57 polls;
initial configured policy order was not evidence of actual execution order.
It remains RUNNING, with the primary coordinator, remote watcher and independent
full-trace sidecarPID31213/caffeinate31214 querying every300s. The sidecar starts
10:51:02,62s later than its intended10:50 first tick; retain actual timing.

12:13 operational checkpoint: local main/sidecar both encounter approximately
2195s wall gaps (monotonic approximately300s); archived shared remote queries
cover11:21..11:57 at300–301s intervals. Do not claim uninterrupted local monitoring.
An independent shell-only remote terminal packer starts12:08:08,1069958/ln01;
two new scripts are SHA-checked and atomically uploaded to a separate versioned
operational directory, not active runtime. It preserves before/after raw manifests,
compressed CRC/bundleSHA and failure evidence after the original watcher exit
barrier. Details belong to outer `docs/cluster-operations.md`. All49 isolated
solver/archive tests pass; bundle-ready will not replace independent GPU checks.

Report Conv/Code/combined avg/P95/P99TTFT and Conv request-averageTPOT, together
with adverse CodeTPOT, all control/copy cost, actual capacity waits and return
deferrals. Distinguish capacity-admission gains from performance-aware return gains.
TwoCode samples do not support statistically robustP99; positive single-trace
results still require paired repeat/seed confirmation. Do not claim50% or an ILP
benefit before measured net outcomes. If return loses, inspect population-domain
guards and paid migration-vs-inference cost before a general algorithm revision.

## Evidence and continuation

Outer evidence root: `evidence/workaware-random-baseline-20261028/`.
`campaign-state.json` records the verified predecessor and260757 submission ledger;
`promotion.patch`/manifest freeze the candidate and protect concurrent main edits.
The continuation queries the old persistent state every300s without duplicate SSH
polling. The new coordinatorPID1923/PPID1 with caffeinate1924 and remote
query-only watcher2208552 now maintain local/remote300s queries, terminal all-file SHA
archival and independent random-choice/tokens/graphs/owner/budget/metric checks.
`PAUSE` in either campaign blocks automatic continuation. Previous failure stops
promotion/submission for diagnosis; uncertainty never triggers duplicate jobs.

Targeted checks:131passed/1skip across controller/revalidation/precheck/trace,
plus19passed for the standalone random planner and continuation. These overlap;
do not sum them as a distinct regression total. Reused GPU calibration independently
rebuilds under the backward-compatible runtime verifier. Full regression status is
recorded in the evidence logs; no GPU result is inferred from CPU tests.
Final correct-cwd regression: **1085passed/5skipped/one existing failure** in
`test_dp_opt_wait_forecast.py::test_future_births_can_pin_shared_pages_after_current_owner_completes`
(expected1s, actual120s); WAIT_AWARE=0 and that unrelated failure is not patched.
The initial outer-cwd shutdown collection error and failed archive attempts are
retained. No claim of an entirely passing suite or uninterrupted local monitoring.
