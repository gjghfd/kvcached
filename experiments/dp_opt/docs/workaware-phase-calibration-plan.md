# Workaware next calibration after complete260757

2026-10-29 14:59 replay checkpoint: unique265356 starts14:37:42 on twoA800,
`d1n41a21g03`, and is RUNNING in the4800s`cache-fixed-original`trace. The reused
252-case raw source and rebuilt anchored composite pass independent Slurm
checks; local closed-phase verification passes95 calibration files and140
composite files with full SHA manifests.85 protected-sync files,444 deployment
tests and50 compute-launch tests pass; all78 runtime/test hashes remain frozen.
The automatic4800s pair then9600s six-policy protocol is unchanged. Closed-phase
archive/verification and terminal return diagnostics are running every300s;
partial trace progress is not a return gain. Recovery root is
`evidence/workaware-anchored-replay-20261029/`. Do not overwrite active sources.

2026-10-29 update:262969 fails after252/252 GPU cases at03:17:44 because the
standalone supplemental global design is rank2 (all sparse graph8), not because
of GPU capacity. Preserve its original71-file SHA-verified raw archive. The derived
supplement explicitly anchors independently verified old coefficients and keeps
all10 legacy population profiles unchanged; only its new population contrasts and
conservative residuals are evaluated. The composite verifies all16 profiles.
The next uniqueA800 replay reuses frozen successful GPU-phase source hashes and
rebuilds/verifies this composite inside Slurm, then runs both declared trace phases.
No new calibration GPU cases or benefit are claimed by reuse. Full details and
monitoring gaps are in`../reports/workaware-early-return-results.md`.

Historical checkpoint2026-10-28, Asia/Shanghai, now superseded by the failed
terminal archive and replay above. Deployment435 CPU
tests pass;83 files pass protected old-SHA/backup/atomic/new-SHA synchronization.
Unique job262969 starts21:45:43 on twoA800/d1n41a14g01. At21:51:53 it is RUNNING
in calibration-n567, with41 compute-launch tests passed and both models initialized.
Recovery SSH at22:24:59 confirms25/252 completed calibration cases, still RUNNING;
all76 frozen runtime/test hashes remain unchanged. Main and independent read-only
monitors show recent300s rounds, not proof of a gap-free complete campaign.
The shared ln01 remote heartbeat at22:33:15 advances to33/252; an ln02 host-local
PID lookup does not determine ln01 watcher liveness. The local-only return
diagnostic observer waits for full archived/verified phases before analyzing
natural recovery and matched post-return cohorts; see
`../reports/workaware-early-return-results.md`. It does not alter this allocation.
The22:43 main-query round advances to42/252, SSH0/actual gap299.999s; all frozen
runtime hashes remain unchanged in the22:42:54 review. No full phase has yet passed.
Calibration and benefit traces have not yet passed. The complete prior six-policy result confirms capacity gain,
not a unique performance-aware return win. Do not replace its negative results.

## Freeze the right comparison

Preserve twoA800, independent Qwen3-30B-A3B tenants, BF16/128K, DP2/EP2/attentionTP1,
24GiB/model/rank virtual pools,10240MiB/GPU controlled budget, allocator.98,
default sparse graphs/FULL_DECODE_ONLY/compile0 and pause32. Current measured
pressure1.198629529 is already within1.2±.02; no rate increase is justified by
the baseline failing to queue. Use the same4800s trace for the first post-calibration
paired comparison, with every old baseline and all adverse metrics retained.

The first-two-policy20.8% full-window ConvP99 difference is pre-migration and
therefore not a return effect. Primary-window headroom has better ConvP95/P99
than workaware. Optimize the stages that actually precede the affected first
tokens, not a later return that cannot change those tokens retrospectively.

## Shared release-observation prerequisite

Implemented locally: completed streams notify the shared controller before
request-journal I/O. Release dirties only the released tenant, advances global
and tenant epochs, and requires a fresh other-tenant observation before using
its capacity for count placement. Same-tenant completion preserves the two-rank
local fast path, avoiding unrelated Code-prefill RPCs. Full and partial snapshots
that race a release are revalidated under the controller lock. All policies use
this common callback;12 release-specific and16 phase-specific tests are covered
by the passing435-test deployment suite. No GPU success is inferred from these.

The2026-10-28 posthoc raw audit finds other-tenant Code reservations remain
cached after completion in rank-local direct admission. Random and KV-only can
continue filling a single Conv rank despite the other GPU being almost empty;
workaware return also refreshes this cache, confounding isolated return benefit.
See `reports/random-baseline-results.md` and the separately hashed
`return-opportunity-analysis-20261028.json` evidence. Uniformly invalidate or
refresh completion/release observations for **all policies**, including the
admission-generation guards for concurrent snapshots. Verify releases without
requiring another Code arrival or total direct-admission failure, retain physical
copy-peak/owner correctness and regressions, then replay the old fixed trace.
Do not intentionally retain stale baseline observations to amplify return gain.

## Missing population coverage

The old verified profiles coverN2/N3/N4/N8/N9, notN5/N6/N7. Supplement the
missing populations with every count partition0..N, both tenants, independent
repeats, balanced controls and fixed mature-context windows. Validate held-out
repeat **and layout/context** contrasts. The worst held-out residual is an
empirical guard, not a confidence guarantee; do not lower it because a candidate
fails the net-profit test. Raw prompt/token/ownership/graph/physical-budget data
must independently rebuild any new or composite calibration; old fit hashes and
old profile verification must continue to work unchanged.

Capacity-screen candidate prompts before execution. With16-token blocks,
1572864bytes/block and96MiB pages, OUTPUT3072, conservative independent request
rounding plus both pools' pinned96MiB floors gives these largest all-hot controls:

| Population | Candidate prompt | Conservative mapped MiB |
|---|---:|---:|
|5|16384|9312|
|6|14336|9984|
|7|11264|9600|

These are arithmetic screening bounds, not GPU passes. The initially considered
N7 prompt12288 requires10272MiB once the calibrated pool's pinned floor is
included and is rejected before any GPU submission; the11264 control instead
has640MiB controlled margin. Real allocator/free-capacity checks can still reject it.
Retain that failure rather than truncate requests, enlarge a policy's budget or
turn off CUDA graphs. Include shorter4096/8192 controls and explicit domain bounds;
no calibration envelope may imply that untested joint layouts fit. Longer mixed
N7 contexts need separately declared heterogeneous controls that truly fit,
not extrapolation from the smaller uniform-context sample.

## Separate real control scopes

The successful second admission copies only.0697s under workaware but pays
3.1076s of pause-call waiting; most is the idle Code tenant's2.958s boundary
wait while Conv still decodes. Return similarly pays2.712s idle-Code waiting.
Measure incoming/host handshake and active-old blocking separately, using actual
armed/reached/resume receipts and fixed workload windows:

`C_control = C_incoming_wait + sum_m(n_m * B_m) + C_pending_first_token + C_future_risk`.

`B_m` is the observed interval that the active tenant is really blocked, not
the elapsed time waiting for an unrelated idle tenant. Add only verified causal
planning/observer blocking; preserve all host cost in measured TTFT even when
it does not belong in old-decode request-seconds. Refit pause/copy/activation
costs from independent repeats, then revalidate before and after default32 pause.
Do not remove uncertainty or count sunk costs twice to obtain an accepted return.

Any affected-tenant transaction fast path requires globally closed admission,
proof that unaffected engines cannot grow beyond full reservations, allocator
copy-peak/owner safety and both DP ranks' EP/default32 execution in each
participating model. Apply any shared data-plane optimization to all applicable
baselines. It is not implemented by simply skipping an empty attention rank.

## Joint-active tail calibration

Profile Conv-only, Code100K prefill overlap, short Code decode overlap and recovery
as separate phases. Use the fixed real Code prompts and current/past arrival
history only. Compare mature Conv placements at a matched active population
without changing graph captures; prove actual prefills and copy/rollback safety.
Current admission for workaware is still KV-only; the performance ILP runs on
release. A future arrival-aware performance planner must explicitly evaluate
incoming prefill/old decode/pending first-token work before admission, even where
KV direct-fit exists, instead of claiming the present release-only ILP already
protects first-Code Conv tails. Such a policy requires new calibration and its
own separately frozen paired treatment, not relabeling260757.

Code prefilling takes roughly54.6..55.2s under the ordinary migration strategies.
For this unchanged trace, removing all queue alone can improve Code average
TTFT versus random69.78s by only about22%, not50%. A50% claim against a migration
baseline requires measured prefill improvement or a different prospectively
declared workload, not selected marks, hidden baseline delay or ignored token work.
The already-observed81% versus no_move is the separate capacity benefit.

## Acceptance and execution order

1. Fix and regress shared release observations, then independently calibrate
   missing populations and joint-active/control scopes.
2. Replay raw snapshots with the exact compiler optimization; report timeout/no
   incumbent separately from certified no-profit, retaining the same objectives.
3. Freeze a general cost model/decision rule and both pre/post-pause guards.
4. Protected-sync only after no active source is in use; submit one allocation,
   check every300s through complete SHA archival and independent verification.
5. Replay identical baseline/treatment traces; report Conv/Code/combined
   avg/P95/P99TTFT, ConvavgTPOT, CodeTPOT and all control/copy costs. Retain
   startup separately and use predeclared600..4800s as primary, not a favorable
   posthoc window. Require paired repeat/seed confirmation for unique gains.

This is a next-phase plan, not a completed calibration, new GPU deployment,
production algorithm or fulfilled research-benefit target.

The implemented pipeline freezes252 cases covering N5/N6/N7, both tenants,
two independent repeats, all0..N rank counts and three context lengths. New
profiles validate held-out repeat, context and directional-layout contrasts;
the worst residual remains the guard. A relocatable SHA-verified composite
preserves the original N2/N3/N4/N8/N9 profiles without overwriting them. Runtime
manifestv3 is explicit and the old v1/v2 formats remain accepted unchanged.
After calibration, the same allocation runs the old4800s random/workaware pair,
then the six-policy9600s reduced-Code trace. This latter trace freezes2Code/
299Conv, seed20261024,600s warmup, synthetic Conv output8192 and actual Code
inputs>=100K. It is an exploratory paired comparison, not statistical P99 proof.
Restoration artifacts are in `evidence/workaware-early-return-20261028/`.

## Reduced Code rate after valid return calibration

The user explicitly accepts pausing inference during KV migration. Keep the
default32 pause protocol and existing verified GPU `dp_echo` transport; overlap
is optional future work, not a condition for this experiment. The next declared
rate candidate is Code.00025RPS versus.0005, keeping Conv.030863253171591748RPS,
real request lengths,10240MiB budgets and sparse graphs unchanged. Recheck actual
no_move overlap pressure rather than assuming that historical1.2 remains exact.

At an86s useful return horizon the nominal next-Code probability falls only
4.21%→2.13%, and the current successful return already selects unprotected Code
headroom. Require earlier calibrated return and measured net-work benefit; do
not force return, weaken uncertainty or promise rate reduction alone will win.
Freeze newly generated Poisson traces/config/rate priors together: merely editing
the rate field of the old frozen trace is not a rate experiment. Halving this
seed's rate from the origin puts both Code marks past4800s;9600s retains two,
but longer/multiple-seed evidence is necessary for tail claims. Use paired
baseline/treatment arrivals within each rate, declared windows, enough events,
per-Code/per-Conv TTFT and ConvTPOT; distinguish reduced mixed-workload Code
frequency from genuine per-request return benefit. This rate candidate is not
yet GPU-validated.

Execution ledger: the lower trace is now generated/frozen and submitted; GPU
calibration must pass before it runs. All policies use an1800s client timeout
(prior run1600s), with the same unchanged120s SLO and no timeout treated as a
completed request.32h is the allocation limit, not an estimated completion time.
The main coordinator and independent query-only sidecar each have multiple
observed300s rounds; the remote shared-start marker identifies3358022 onln01.
End-to-end monitoring continuity and all performance conclusions still require
terminal audit. No commit/push, protected Aegaeon change or new GPU gain is claimed.
