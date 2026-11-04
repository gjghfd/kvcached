# Public-source verifier failure and independent quality recovery

## Parallel status and independent N2 review (2026-11-02 14:39 +08)

Performance N3 job280104 starts at14:34:04 on two A800/d1n41a13g02.
DEBUG279537 remains FAILED/1:0 at14:33:53; after six trial completion
records its compute verifier rejects duplicate logical reference identities.
The original four-point serial sampler selects indices `[0,1,2,2]` for
three trace requests. Static verification already permits repeatable serial
outputs, whereas the first-layer verifier incorrectly requires one logical
identity per serial execution. Raw SHA archival and independent review
precede any DEBUG-only replacement epoch; the failed parent cannot qualify
the QKV mirror. The performance runtime and host bridge remain frozen.

Independent review in
`evidence/public-performance-followup-20261102-r5/native-N2-seed2-review/`
recomputes all metrics and validates the raw/log/source, installation,
bridge return and subsequent strict receipts; nine CPU tests pass.
Request-pooled two-seed N2 workaware Conv TPOT is17.358ms,9.564% better
than random but28.230% worse than no_move. Code TTFT improves56.127%
versus no_move. Seeds have different cohorts, with only two Code and four
Conv requests per policy; pooled Type7 tails are descriptive, not formal
Poisson/held-out evidence. The first seed's original CANCELLED job state
is retained. Second-seed return runs at156.781–158.216s, exposing old
Conv to at most1.1309 request-seconds of stopped execution. Every first
token precedes this return, so its TTFT benefit is not caused by return.

## Second-seed N2 complete and verified (2026-11-02)

Job279520 runs on two A800 at12:40:53–13:53:19 and remains COMPLETED/0:0.
484 raw files, both logs,216 executed sources and the complete phase pass
SHA and strict verification. The same-coordinator archive bridge performs
real alias recovery, saves an independent installation receipt and retains
the original closed error log. Its returned receipt intentionally does not
claim downstream verification; the separate
`stages/native-2026103102-N2/independent-performance-verification.json`
provides that completed gate. Coordinator10069 automatically continues with
the unique next N3 allocation280104; DEBUG279537 continues independently.

TTFT values are seconds, Conv TPOT milliseconds. Each policy has one Code
and two Conv requests with all25,129 output tokens completed. Code's
avg/P95/P99 coincide because n=1. Combined and Conv percentiles are Type7
descriptors, not statistical tail or held-out evidence.

| Policy | Code TTFT | Conv avg/P95/P99 TTFT | Combined avg/P95/P99 TTFT | Conv avg TPOT |
| --- | ---: | --- | --- | ---: |
| no_move | 119.562 | .151/.167/.169 | 39.955/107.623/117.174 | 15.023 |
| random_no_return | 57.450 | .141/.157/.158 | 19.244/51.720/56.304 | 20.071 |
| kv_only_headroom | 56.536 | .131/.147/.148 | 18.933/50.897/55.408 | 20.334 |
| kv_only_workaware | 56.199 | .127/.146/.147 | 18.818/50.594/55.078 | 18.134 |

Workaware lowers Code TTFT52.996% versus no_move and Conv TPOT9.651%
versus random_no_return (10.819% versus headroom). However, Conv TPOT is
20.710% worse than no_move. The first-seed N2's9.468% improvement over
random is corroborated qualitatively, not a formal significance claim.
Initial Conv first tokens precede Code arrival and return; Code's TTFT gain
comes before return and is not attributed to the later migration back.
The current no_move remains the same full-reservation custom scheduler,
not unmodified native vLLM or a claim of whole-card OOM.

Both jobs actually start far earlier than Slurm's earlier evening ETA.
Local clamshell sleep at12:48:50, intervening DarkWakes and full Wake at
13:45:25 cause real wall-clock gaps. The13:10 health notification is acted
on only at13:48; remote query-only logs continue at300s intervals and the
GPU workloads remain independent of local sleep. The original pmset and
`remote-wall-gap-continuity-20261102-1350.log` preserve that distinction.
First-layer GPU attribution, full Poisson/grid/held-out and the overall
research conclusion are not yet complete.

## Return-gate rejection review: not a pause-wall bug (2026-11-02)

The independent sidecar in
`evidence/public-pause-cost-opportunity-20261102/` checks318 ordered candidates
from frozen N3/N4 rejected snapshots. It preserves actual raw/log/source SHA,
all copy/headroom/risk constraints and every negative margin. These are CPU
counterfactuals, not additional GPU trials or proof of waiting optimality.

Actual old-request blocking already uses per-rank reached→resume receipts,
not the whole `pause_call_s`. The prediction uses a common0.475s activation
proxy plus copy and future-capacity risk. Replacing only activation by any
nonnegative tenant-specific value does not make any of these three snapshots
accept. N3/population3 and N4/population3 best splits have performance gains
28.589/19.720 request-s but uncertainty118.549/34.239 request-s. N4/population4
has gain16.287 request-s versus copy+risk16.754 even before activation,
minimum gain and uncertainty89.012 request-s are charged.

The return uncertainty uses held-out paired layout-contrast residuals, not
the separate absolute-cadence residual. Its coefficient2 bounds two predicted
sides; final gating does not arithmetically subtract uncertainty twice.
The existing trials do not establish shared-EP noise covariance or justify
halving that bound. No current fit, uncertainty, risk or algorithm is changed
to manufacture a positive decision. The sidecar's `fit-uncertainty-note.md`
and `review.md` retain exact source references, all ordered candidates and
the distinction between actual stop exposure and new-request lock waiting.
Future tighter fits or tenant-specific pricing require NEW real calibration,
with all negative costs and the original strict downstream checks retained.

## Queued continuation and source acquisition limits (2026-11-02 12:07+08)

Both existing jobs remain pending; neither is resubmitted. Intermittent
Slurm `squeue` socket timeouts occur at11:30/40/50/12:00. A successful SSH
exit or an empty failed queue response is not evidence of an empty queue.
Accounting remains pending; DEBUG's producer succeeds again at12:04:16.
Bounded resource inspection does not establish healthy free A800 capacity.
Slurm's11:54 estimates of22:54:14 and23:11:02 are estimates, not guarantees.
All original errors and monitoring wall times remain archived.

The NEW optional QKV mirror campaign binds an explicit prepare-only kernel
activity policy, immutable worker/probe SHA and bounded eager-shadow trace
retention. Default is disabled; outside flags are not inherited. Main review
passes284 CPU tests. A genuinely successful first-layer parent with complete
matrices is still required; no mirror prepare or GPU attribution is claimed.

Fixed-revision Stratos parquet acquisition remains unverified after local
TLS resets, remote DNS failure and a mirror redirect back to the inaccessible
official URL. Raw official viewer snapshots are not relabelled as verified
parquet or Git revision. A separately declared API-snapshot source may be
evaluated only with its own full-row/content/token evidence and explicit
provenance limits; it does not change the current frozen OpenThoughts3 trace.
Evidence is in `evidence/public-stratos-fixed-revision-20261102/` and the
NEW mirror kernel handoffs. Performance, numerical debug and held-out
benefit conclusions remain unfinished.

## Same-coordinator archive bridge installed (2026-11-02 11:18+08)

Main-thread integrated regression passes282 tests with the existing
`asyncio_mode` configuration warning. Every216 performance and228 DEBUG
source and both r2 host files retain their frozen SHA. The old coordinator
99679 is retired only after exact UID/start/argv and absence of in-flight
children are checked; its exit and original campaign lock release are
verified. NEW coordinator10069/caffeinate10070 has a three-file host-only
manifest in addition to the original two-file r2 manifest. Actual argv,
process start and PPID1 are recorded in `host-archive-bridge-r3-handoff.json`.
The first real poll at11:17:16 and `host-archive-bridge-*/loaded-source.json`
are verified. Job279520 is neither cancelled nor resubmitted. DEBUG279537
and its immutable authorization chain remain unchanged.

The bridge executes recovery synchronously in the owning coordinator only
after the original archiver has exited and its unique closed error log is
saved. Only the exact progress-pointer error on COMPLETED/0:0 enables this
recovery; all other bounded retries and strict downstream checks remain.
No independent follow process is launched. The new real network archive
path is not yet exercised by a terminal job, so CPU validation is not
reported as end-to-end network success. Both jobs remain PENDING/Priority
at11:18:07. Remote watcher identities/PPID1 are verified, including DEBUG's
11:09:53→11:14:53 interval; queries continue every300s.

## N4 full phase verified; both chains resumed (2026-11-02 11:10+08)

The NEW r2 host epoch passes the complete strict N4 phase gate, not merely
raw-only metric recomputation. The receipt is
`evidence/public-performance-followup-20261102-r5/stages/native-2026103101-N4/independent-performance-verification-host-f5f730a273ec.json`
(SHA256 `47512295bcf43f6e0153158437bf109416351e2e22c6635e514696e192a39d4f`).
All request lengths, arrivals, ownership, budgets, copy and metric checks
remain enabled; the old certificate failure and original COMPLETED/0:0
are retained. Greedy equivalence remains unresolved. Full phase validation
does not turn this five-request native case into statistical tail or held-out
evidence, nor remove its negative Conv TPOT comparisons reported below.

Performance coordinator99679 resumes at10:29:08. Job279520 is the original
second-seed N2 stage; independent first-layer DEBUG279537/r12 is bound to
coordinator1546. Both are PENDING/Priority at11:09:53. The216/228 GPU closures
and the two r2 host sources are frozen. A NEW same-coordinator archive bridge
is reviewed separately to avoid competing external recovery/retry processes.
The GPU allocation and frozen trace are not cancelled or reissued for this
host-only handoff. The original native→rate→grid→held-out chain continues.

Local clamshell sleep from10:54:47 to11:01:49 causes a real wall-clock
monitoring gap despite caffeinate and monotonic intervals. The performance
remote query-only log contains10:56:44 and11:01:44 observations. DEBUG's
unique remote query-only watcher1972698 starts at11:09:53 after frozen-source,
job-owner and marker checks; its first poll is verified, not yet a claim of
multi-interval continuity. Evidence is the r5
`remote-query-gap-and-debug-install-20261102-1110.log` and original pmset log.
Performance and numerical debug remain unfinished and proceed independently.

## N4 complete raw, remaining host gate (2026-11-02)

485 raw files, both logs and216 executed source files pass independent SHA
verification. The original job remains COMPLETED/0:0. The full phase gate
then stops on `Retained snapshot changed physical capacities, ownership,
uncertainty or risk`. Independent frozen-source reconstruction finds exactly
eight one-ULP differences in arrival probability/cold-risk fields; all
Fraction geometry and rank proofs agree. The old portability adapter only
catches the other certificate error message, so it does not reconcile these
guard primitives. `certificate-diff-N4.json` preserves this failure and every
delta; no actual capacity/ownership mismatch is asserted from that message.

Performance coordinator75553 exits. NEW host recovery hardens caller job/
stage, both log inventories, bounded probability math, per-verification risk
scope and independent loaded source hashes before reusing N4. The216 GPU
closure, trace, seed and original data remain frozen. DEBUG r11 stages safely
but finds no fresh active performance window within360s: it submits no job.
Its failure is retained; a later epoch must bind the replacement performance
identity rather than reusing stale authorization.

Raw-only N4 metrics, not full phase/held-out validation: workaware Code TTFT
243.256→55.958s versus no_move (76.996% reduction); Conv TPOT22.762ms versus
random22.706ms (0.247% worse) and no_move18.509ms (22.982% worse). Workaware
returns at350.132→353.029s, after population4→3→2; every first token is earlier.
Combined avg/P95/P99 TTFT is11.301/44.805/53.728s versus no_move
48.760/194.643/233.534s, but these are five-request Type7 descriptors, not
statistical tail-SLO evidence. Complete tables and negative costs are in the
independent `native-N4-review` evidence. N2's9.47% Conv TPOT improvement over
random remains the stronger small-sample return result; N4 is not selected
away or called a monotonic population benefit.

The archive recovery safety sidecar additionally identifies failure-retention,
partial-install and receipt-path limitations in NEW follow automation. These
are repaired separately before launching it; the first follow launch exits
because75553 is already gone, and is not reported as a healthy watcher.

## N4 terminal and spawn-entry recovery (2026-11-02 10:05+08)

Performance279153 completes at09:53:22 with Slurm COMPLETED/0:0; all four
policies finish, but independent verification is still pending. Its original
download contains a progress symlink under canonical `/data/run01/...`,
whereas the frozen archiver accepts only the `$HOME/run/...` anchor. The
original bundle and manifest remain untouched. A NEW host-only recovery
requires actual owner/home/accounting, samefile/readlink and target SHA,
unchanged216 source hashes and strict regular-file/archive/log verification.
Only a temporary parser representation normalizes that one pointer; no
symlink is extracted and no active runtime or trace changes.

DEBUG279359 fails at09:44:56 after347 startup tests and worker imports. The
generated entry called `compute_main()` without a main guard, so vLLM spawn
reran the diagnostic and attempted an exclusive copy of the already-existing
`head_hidden_probe.py`. All262 raw files and both logs are SHA archived;
FAILED remains. This is an entry/startup failure, not numerical attribution.
The NEW helper adds the guard; actual `runpy.run_path` with `__mp_main__`
and parent-entry regression pass within200 targeted CPU tests. The shell
wrapper's subsequent zsh read-only `status` assignment failure is retained
separately, not called a successful wrapper exit. New r11 is prepared locally
with228 files; no resubmission or completed first-layer capture is claimed.

Performance and DEBUG stay independent. Supplementary300s queries resume in
`parallel-progress-observer-r4.jsonl`. N4 metrics wait for complete raw/log
SHA verification; public Poisson/held-out and P74/operator/quality debug remain
unfinished.

## Separate input/output fit recovery (2026-11-02 09:23+08)

DEBUG279275 passes347 startup tests, actual worker imports and all canonical
test suites, then fails before model loading at `live_trace.run`:
`shutil.copy2(args.fit, args.output / "placement-fit.json")`. The DEBUG launch
had set the input fit to that same output file. This is SameFileError, not a
CUDA, numerical or source-alias failure. All259 raw files and both logs are
SHA archived under r9; original FAILED remains. The permission recovery works.

The NEW first-layer helper now gives both CLI and source inventory the same
`DP_OPT_UPSTREAM_PLACEMENT_FIT`, pointing to the immutable upload's fit; the
output archive receives identical bytes at a distinct path. The CPU regression
actually invokes copy2 on these paths and checks digests, in addition to launch
template and mode checks;200 tests pass. Performance216 frozen sources and
279153 are unchanged. New r10/228 changes only helper/test SHA and is launched
with88062/caffeinate88063; it waits for the fresh performance poll to authorize
one DEBUG retry. The mirror sidecar receives this same input/output constraint.
No model-level first-layer capture is claimed yet; all failures and partial
local preparations remain available rather than reclassified.

## Private-copy permission recovery (2026-11-02 08:49+08)

The independent host-recovery safety review is in
`evidence/public-performance-followup-20261102-r5/host-recovery-review.md` with
28 CPU checks. It confirms no triggered blocker for this specific immutable,
trusted N3/serial-verifier recovery, but does not approve arbitrary allocations
or numerical domains. A future NEW host epoch must bind the returned alias
receipt to the caller job/stage, explicitly check probability bounds and numeric
domains, require both log hashes, scope risk context to one call and bind loaded
host source/input identities. Conditional adversarial fixtures are retained;
they are not described as safety passes. Neither active host source is changed.

Performance279153 receives twoA800/d1n41a12g02 at08:37:06 and is actually
RUNNING in the08:41/08:46 polls. Both tenants load BF16/128K with default sparse
graphs. Reference generation precedes trace progress; an absent progress file
does not certify completion. The performance coordinator remains75553, with no
pause, source replacement or cancellation for DEBUG.

Independent DEBUG279164 fails before model execution. All248 raw files and
both logs are SHA archived, original FAILED retained. Its initial346 tests and
actual worker imports pass; the canonical startup suite then has392 passes and
two PermissionErrors. The immutable upload uses444 file modes, `cp -a` preserves
them in private runtime, and `archive_amortized_runtime` uses `shutil.copy2`, so
the source-tamper test fixtures inherit444 and cannot be modified by the actual
Linux user. This is not an alias mismatch or model divergence.

The minimal change is in the NEW DEBUG sbatch only: grant owner-write permission
to its own execution-copy files after copying, never to the immutable upload,
shared checkout or protected patches. A mode-policy evidence record declares the
scope; full source SHA checks still require identical bytes. The new regression
checks both mode bits and unchanged source bytes, and mutates only a fixture.
The200-test local regression passes; its initial missing-shutil error is retained.
No frozen performance source/test changes. The unsubmitted r8 local preparation
is retired; r9 freezes228 files and changes only the debug sbatch and its test,
with explicit old/new SHA provenance. New83178/caffeinate83179 independently
waits for a fresh performance poll before held-submit/registration/release.
First-layer GPU capture, P74 coverage and model-quality repair remain incomplete.

## Strict N3 reuse and parallel continuation (2026-11-02 08:04+08)

278898 drained all four N3 policies, inner exit0, then FAILED/1:0 at
07:30:09 during the outer source-identity postcheck. Its482 raw files and
both job logs are SHA archived. The allocation remains FAILED, including its
original outer exit1; no original raw file, source inventory or job state changes.

The launch recorded `/data/home/<user>/run/...` while runtime provenance
resolved the same files to `/data/run01/<user>/...`. The independent host-only
`performance_path_recovery.py` checks actual owner/home/accounting/A800/job name,
every remote readlink and digest, the complete216-file runtime, preserved source
bytes and an exact receipt bound to the raw manifest. Canonical inventories exist
only in temporary verifier inputs, including workaware sources and later copied
native cases. All nonterminal trace, physical-budget, ownership, arrival, token
length and plan checks remain required; original FAILED status is retained.

After fixing this identity mismatch, strict local certificate replay exposed a
second issue: a Linux/macOS numerical risk primitive differed by one ULP
(`0.8037672345747149` versus `0.8037672345747148`). Independent sidecar evidence
is `evidence/public-performance-followup-20261102-r5/native-N3-review/certificate-diff.json`.
The host-only `risk_certificate_portability.py` independently reconstructs the
Poisson probability and cold-cost risk with Decimal80 and bounds each scalar to
four ULPs of that mathematical value. Only these risk primitives are reconciled;
all Fraction KV geometry proofs and exact nonnumeric certificate fields are
unchanged. High-precision constant-risk dominance must also hold. Real changes,
missing proofs, corrupted source bytes and unrelated verifier failures remain
fatal. Audit attempts r1/r2 and their errors are preserved, not hidden.

Full N3 verification passes in `path-recovery-audit-r3.log` and
`stages/native-2026103101-N3/independent-performance-verification-path-recovery.json`.
The new recovery/portability tests pass25 tests; the initial recovery-plus-followup
regression passes141. Source SHA and actual framework-process argv are recorded
in the recovery coordinator identity. This is host verification portability, not
a change to the216 frozen GPU sources or model numerical behavior.

The detached performance coordinator75553/caffeinate75554 resumes the original
predeclared chain without rerunning N3. Next native N4 job279153 is PENDING at
the08:01:19 SSH-success poll, with independent300s local/remote monitoring.
Remaining native cases, rate calibration, original policy grid and heldout remain
automatic; none are claimed complete. The gap after the old coordinator exited
is recorded as an interruption, not continuous monitoring.

DEBUG279051 passed341 compute startup tests but failed on a bare worker import
from HOME, where a namespace directory shadowed the editable package. A pure
stdlib independent reproduction and shell diagnosis confirm the cwd issue.
The new debug sbatch changes cwd to its own evidence directory before imports.
IDs remain int32/int64; positions remain int64. Its228-file r5/r6/r7 runtime is
byte-identical to the468-passed isolated startup payload. An additional199 local
tests pass after a preserved outer-CWD collection failure was rerun with explicit
PYTHONPATH. DEBUG r5 submitted nothing because the old performance PID had died;
r6 submitted nothing because its safety check raced the new performance submission.
New r7 is safely staged and waits for the next fresh performance poll before the
held-submit/SHA-registry/release chain. No first-layer GPU result yet; P74,
upstream kernel causality and quality repair are still unresolved.

### N3 descriptive metrics

OpenThoughts3 closed original responses, three Conv requests plus one SWE-smith
Code request; one seed/trial, not Poisson or heldout. TTFT values are seconds;
Conv TPOT is milliseconds. Code avg/P95/P99 are the same single request.

| Policy | Code avg TTFT | Conv avg/P95/P99 TTFT | Combined avg/P95/P99 TTFT | Conv avg TPOT |
|---|---:|---:|---:|---:|
| no_move |241.402|0.114/0.137/0.139|60.436/205.213/234.164|18.419|
| random_no_return |56.172|0.121/0.152/0.155|14.134/47.770/54.492|22.303|
| kv_only_headroom |55.667|0.114/0.142/0.145|14.003/47.339/54.002|22.066|
| kv_only_workaware |56.332|0.113/0.137/0.140|14.168/47.903/54.646|21.951|

Workaware reduces this Code TTFT by76.7% versus no_move, but is slightly slower
than headroom/random for Code. Conv TPOT improves only1.6% versus random and
remains19.2% worse than no_move. Combined tails are dominated by one Code
request and do not establish tail-SLO generalization. N2's larger9.5% Conv TPOT
advantage over random is population-dependent, not a universal slope. Raw metrics
and return timing are maintained by the independent N3 sidecar; no sample/SLO/
policy is changed to force gains. Performance and model-debug goals remain open.

The independent N3 return timeline now explains the small incremental gain:
workaware returns only at340.815→344.665s, after a Conv naturally completes and
the population falls from three to two. N2 returned immediately after Code
completion. The measured N3 balanced/concentrated profile has zero nonnegative
attention slope and no fitted graph-cost contrast; its uncertainty is not evidence
of genuinely zero attention work. Once the population reaches N2, the sparse
capture contrast makes a profitable return certifiable, but much of the lifetime
has already elapsed. Longer output alone cannot fix this population/uncertainty
effect. Lower Code duty cycle and a stationary population near two are hypotheses
to test with the predeclared rate calibration, not reasons to relax uncertainty
guards or retroactively pick the favorable native sample. All first-token times
precede the N3 late return; the TTFT improvement comes from admission placement,
not from the return itself. See `native-N3-review/return-recovery.json` and
`native-N3-review/review.md` under the r5 performance evidence root.

## First-layer deployment failure remains independent (2026-11-02 07:15+08)

The first-layer DEBUG279011 starts07:05:02 on twoA800/d1n41a16g01 and
FAILED/1:0 at07:05:08, before worker imports/model loading. Its339 compute
startup tests have334 passes and5 failures, although the exact228-file local
isolated runtime passed461 including additional canonical tests. All five
failures are the local-coordinator poll-window fixture: it assumed a freshly
written filesystem mtime was strictly after a nearly simultaneous wall-clock
header. The failed ordering is not evidence of model numerical behavior.
The fixture now models a real ordered poll with a two-second-old header and
one-second-old completion, and adds a coarse-mtime regression; the real
owner/name/ID, SHA, fresh-poll and60s authorization guards are not relaxed.
Failure messages include all clock values for further diagnosis.

231 raw files and both job logs pass terminal SHA archival. The old verifier
then incorrectly attempts successful-run frozen-input checks even though
startup failed before those inputs were installed. New failure handling checks
the real terminal state after raw/log/runtime SHA, records missing inputs as
missing, and never claims model/source-probe validation passed. The actual
FAILED state, old immutable runtime and all logs remain unchanged.341 local
integration tests pass, including the new negative archival test; the new
independent r4 runtime is undergoing isolated validation before any retry.

Performance278898 continues throughout without cancellation, source changes
or DEBUG snapshot overhead. Its random and no_move trials have drained; the
four-policy native N3 case is not yet independently complete. All new DEBUG
attempts still require explicit append-only ID/name/owner/A800 authorization.
The initial r1 preparation is retained after correcting macOS's framework
launcher-versus-actual-argv provenance, with the original launch fully preserved
and PID/start/arguments verified; no performance process is restarted/signalled.
The staged but unsubmitted r2 preparation is also retained after disabling
pytest cache writes in the exact immutable runtime. Neither is a GPU result.

Evidence: `evidence/public-upstream-first-layer-diagnostic-20261102/parallel-run-r3/`;
the CPU-only retry preparation is `parallel-run-r4/` under the same evidence root.

## Actual startup failure and isolated recovery (2026-11-02 06:14+08)

The previous live status is historical:278772 FAILED/1:0 at05:14:13 after
13s, before model loading or request execution.249 raw files and both job
logs independently pass terminal SHA archival. The failure is deployment
closure, not exhausted GPU capacity. The canonical shell launches tests and
dynamic analysis modules beyond the original import graph; a later isolated
run also exposed the omitted `docs/ilp-example.json` fixture. Its488 passed,
5 skipped,4 failed and25 errors remain recorded, not retrospectively passed.

The recovery explicitly freezes the canonical shell paths, dynamic Python
imports, required JSON fixtures and original pytest configuration. The
protected native vLLM patch is copied only after checking its protected SHA,
never added to the deployment write set.216 actual runtime files pass SHA
verification;171 integration tests and122 previously failing critical startup
tests pass in a real isolated copy. The local pytest configuration warning
is preserved; full torch-dependent canonical startup still runs on compute.

The new source epoch is `evidence/public-performance-followup-20261102-r5/`.
Coordinator44343/caffeinate44344 are observed PPID1 across calls. Protected
explicit-file synchronization completes before278898 is submitted and
starts06:14:31 on twoA800/d1n41a15g01;06:14:33 observes RUNNING. This is not
yet a completed native case. The unchanged five missing native cases,
measured rate calibration, seven-policy exploration and held-out stages
continue with300s queries and terminal SHA/independent verification.
No frozen experiment-matrix fixture is edited during this epoch; this report
and operational guidance hold live status until the matrix can be safely updated.

## Head versus upstream evidence (2026-11-02 05:33+08 terminal)

278529 COMPLETED at05:33:37;169 raw files, both job logs, all four workers'
identical BF16 candidate weights and42 snapshots independently verify.
Balanced repeats preserve exact target hidden bytes and tokens. In all four
P149/corank observations, the actual first divergence is53; the GPU candidate
contrast change is0.044666290283203125, the common-prefix hidden contribution
0.04467148979892954, and the head residual−0.000005199515726417303. Three
P74 observations first diverge at442 with hidden contribution0.1641628497163765
versus GPU change0.16417694091796875. Thus approximately99.99% of those
observed contrast changes originates upstream of the final head projection.

One P74/corank1 repeat instead first diverges at139, outside the old442±1
capture window. Only7/8 observed first divergences are attributed; full
root-cause coverage and quality equivalence remain false. Zero copy and zero
case pause exclude migration from these cases, but the decomposition alone
does not identify attention, MoE or a specific kernel. The next independent
first-layer probe targets unchanged P149/53 and52–54 snapshots; P74/139/442
are explicitly outside its first scope. Its GPU capture/replay and private
runner mappings still require real validation. Debug does not block or add
snapshot overhead to the separate performance runtime.

Evidence: `evidence/public-head-hidden-diagnostic-20261102/parallel-run/independent-head-hidden-verification.json`.

## First native N2 return timeline (2026-11-02 06:40+08 analysis)

The independently complete original N2 case now has a descriptive raw return
alignment, using the authorized greedy-observation adapter rather than weakening
any other paired invariant. The unadapted historical analysis first rejects
different output token IDs; that is not a new inference failure or proof of
quality equivalence. No current GPU runtime or trace changes for this analysis.

Workaware Code client completion is153.684032s relative to trace origin. The
return starts0.000935s later, resumes at156.358661s, and the first balanced
Conv-only observation follows completion by2.704448s. Its reported pause call
is2.005824s; the move copies1,618,477,056 bytes in0.078458s with full device
bytewise echo verification. The integrity-receipt SHA is explicitly not a
KV-content digest. The baseline has72 stable Conv-only samples after its Code
completion, all on one attention rank; workaware has59, only one still
single-rank. Maximum observed sample gaps are2.217s and2.649s respectively.

Thus this case does not suffer a late-return gate: it actually returns almost
immediately after Code release, before natural baseline rebalance. The remaining
Conv TPOT penalty versus no_move must not be attributed solely to late return;
Code prefill/decode overlap and shared compute/control costs require phase
alignment. There are zero new Conv arrivals in the post-return cohort, so this
mechanism case cannot establish improved post-return Conv TTFT or population
tail significance. Greedy outputs differ and sampling remains descriptive.

Evidence: `evidence/public-performance-parallel-20261102/native-N2-return-recovery-descriptive.json`.

## Verified checkpoint and bounded recovery (2026-11-02 05:14+08)

278514 retains its actual CANCELLED terminal state. The original archiver
rejects its dangling next-case progress link because Slurm cancellation left
no root exit-status, rather than the assumed nonzero marker. Two failed
archive attempts and their bundles remain preserved. After verifying argv
and absence of children, only the owned archive-only coordinators86377/14944
are retired. A new explicit owner/name/A800/CANCELLED-authorized wrapper
changes only the missing progress-marker gate, creates no raw exit marker,
and preserves path/member checks and before/bundle/after SHA verification.
396 raw files and both job logs independently verify; the first complete
native case retains its own exit0 and full non-greedy invariants.

The recovery source epoch is now `evidence/public-performance-followup-20261102-r2/`,
with coordinator21790/caffeinate21791 observed PPID1. The seven-file
protected sync verifies all additions (including three missing imports).
278772 starts05:14:00 on twoA800/d1n41a16g01, independently of DEBUG278529.
The first poll05:14:04 is SSH0; remote query-only3567440 is bound by shared
markers.169 final CPU tests and168-file exact-copy imports pass. The driver
schedules five remaining native cases sequentially, then real rates and
split seven-policy exploration/held-out stages continue. No runtime is
overwritten and no result is published. The top coordinator phase currently
retains its prior waiting label; `stages` is authoritative for the active job.

First-seed N2 descriptive results, seconds for TTFT and milliseconds for Conv
TPOT; each policy has one Code request and two Conv requests:

| Policy | Code avg TTFT | Conv avg/P95/P99 TTFT | Combined avg/P95/P99 TTFT | Conv avg TPOT |
|---|---:|---|---|---:|
| no_move | 135.666 | 0.109/0.121/0.122 | 45.295/122.112/132.955 | 12.051 |
| random_no_return | 57.102 | 0.113/0.130/0.131 | 19.109/51.405/55.962 | 18.317 |
| kv_only_headroom | 55.859 | 0.109/0.123/0.124 | 18.692/50.286/54.745 | 18.358 |
| kv_only_workaware | 55.778 | 0.111/0.125/0.126 | 18.667/50.213/54.665 | 16.583 |

Workaware lowers Code TTFT58.9% versus no_move and Conv TPOT9.5% versus
random, but Conv TPOT remains37.6% worse than no_move. Code TPOT also grows
from16.824ms to125.492ms; TTFT gain is not an across-the-board latency or
decode-speed improvement. These tiny-sample quantiles are descriptive;
Code P95/P99 equal its sole sample. No final Poisson/held-out benefit or
quality equivalence is inferred. Full group metrics are preserved in
`evidence/public-performance-parallel-20261102/native-N2-independent-metrics.json`.

## Performance/debug decoupling (2026-11-02 user override)

At04:52:33+08,278514 is scientifically checkpointed after the first complete
native-N2 four-policy case passes its compute-side verifier. Original Slurm
CANCELLED status is retained; remaining predefined cases are not selected
by gains. Its original coordinator continues canonical terminal SHA archival.
The independent follow-up14944/caffeinate14945 starts05:00+08, is observed
PPID1 across calls, and waits for that archive and coordinator exit before
reusing only independently verified complete cases. It rescues five missing
native cases one allocation each, then freezes rates and runs bounded
calibration/seven-policy exploration/held-out stages. Per-policy walltime is
declared8h before launch; native/rate stays4h30, with unchanged trace/drain.
Evidence is `evidence/public-performance-followup-20261102/`.
Deployment additionally includes three previously absent remote imports;
all153 existing frozen sources remain unchanged. The168-file exact-copy
imports and161 final main-thread tests pass. Explicit future DEBUG IDs use
an append-only SHA authorization ledger, with at most one active DEBUG;
unknown third allocations remain rejected. No new Poisson/held-out benefit
or final numerical attribution is claimed at this point.

At04:01+08 the native walltime defect is identified before timeout:
six cases times four policies times900s requires at least21600s, excluding
setup and serial references; job278514 has16200s and ends08:02:49.
An attempted8h30 extension is denied by Slurm. Preserve the running frozen
runtime and all trace windows. A case-boundary checkpoint plus new bounded
native-rescue allocations will finish every predetermined missing case;
partial cases are not performance comparisons and the original terminal
state is retained. This recovery decision does not depend on observed gains.
The debug job278529 independently started03:51:20 on twoA800/d1n41a15g01.

Live observation at2026-11-02 03:36+08: performance278514 started03:32:49 on
twoA800/d1n41a24g01 and runs native-2026103101-N2; independent debug278529
is PENDING/Priority. Local coordinators86377/86800 and bound caffeinate
processes86379/86801 are PPID1 across calls; remote query-only watchers
3737416/3788181 were independently observed. The performance polls03:31/03:36
are SSH0 with299.996s gap; debug first poll03:35 is SSH0. These observations
do not establish future uninterrupted coverage or any completed GPU benefit.
Actual debug deployment tests are251 passed/one skip plus98 focused tests;
performance deployment220 passed/one skip. Existing runtime remains frozen.
An independent follow-up driver is being prepared because native completion
currently freezes rate traces but does not submit calibration or held-out jobs.
Seven-policy3600s exploration and10800s held-out phases must be split across
bounded allocations rather than packed into a single4h30m job.

The user explicitly requires numerical debugging and performance experiments
to proceed independently. Complete greedy equality is now an observation, not
a performance admission or selection gate. Preserve full output differences
and qualify quality-equivalence claims separately. All request length/arrival,
ownership, mapped-budget, migration/copy and timing-accounting checks remain
mandatory; other failures still abort. Historical FAILED jobs are not relabeled.

New opt-in performance modules leave all147 old sources unchanged. They replace
only the two declared post-drain greedy RuntimeError gates and the verifier's
greedy comparison with explicit observations. The inherited structural verifier
continues checking the real raw records. Perf uses the original BF16 default
head and default sparse graphs, without hidden/head snapshots; it can reuse
the independently rebuilt original144-case default-head calibration. The debug
line retains its separate FP32-head attribution and does not feed timings into
the performance comparison.

The completed four-policy274385 native-N2 cases are independently reanalyzed
with their original terminal status, all source/solver evidence, full output
lengths and non-greedy invariants preserved. These are small mechanism cases,
not a final public Poisson/held-out benefit result. Raw performance/reanalysis
is in `evidence/public-performance-parallel-20261102/`.

The resumed performance job targets the same frozen public native traces,
two seeds and populations2/3/4, then freezes rates from measured residency.
Next stages remain real open-loop rate calibration, seven-policy Poisson
comparison and held-out evaluation. The separate debug launcher now allows
one explicitly verified `qwen-public-perf` job; unknown jobs remain rejected.
Both lines retain independent300s monitoring and terminal SHA verification.

## Independent code review (2026-11-02, Asia/Shanghai)

The user-requested read-only subagent review is complete. Its original-version
findings and installed-source evidence are preserved at
`evidence/public-head-hidden-diagnostic-20261102/code-review/review.md` and
`code-review/findings.json`. The review did not submit a job, edit runtime,
run imports/tests on the cluster, or claim a proven numerical root cause.
No new GPU allocation, sync, publication or Aegaeon execution occurs here.

The most useful causal evidence is not migration: all six276677 cases have zero
copy and zero case pause. Both balanced repeats match every serial token;
all four corank repeats diverge for Conv at zero-based53/442, with Code equal.
In r1 the Conv1/Conv0 first differences occur at trace offsets2.068/7.394s,
before Code's90s arrival. These timestamps establish chronology only; failed
timings remain ineligible for performance or setting selection.

Capture records show Conv decode padding1 in balanced versus8 in corank;
Conv1 prefill has149 versus150 total scheduled tokens. This changes the
upstream batch/graph execution shape. It does not prove which attention,
MoE/router, QKV or normalization kernel contributes the error, nor that an
actual Triton tile changed. Actual FP32 head output does not eliminate the
BF16 upstream or BF16-operand head GEMM as candidates. EP arithmetic/padding
and actual KV/input-state errors are also not completely excluded. Installed
Torch BF16 reduction getters cover cuBLAS, not Triton MoE behavior.

The reviewer finds no mandatory reference-prefix, phase-path, synchronous
request/logits-row or output-token-update blocker under the declared static
configuration. It instead identifies four issues in the new, not-yet-executed
attribution observer: unbound candidate/worker/request provenance (P1), ignoring
captured neighboring first differences (P2), silently missing sampler contrast
coverage (P2), and partial re-prefill/discarded-logits collisions (conditional
P2, not observed in276677). They are not causes established for the parent job.

Main-thread local corrections bind ordered weights and snapshots to the exact
worker/request/layout/row and frozen prompt bank; select the actual first
difference within the captured window; report absent decision-candidate or
sampler coverage; and skip discarded partial re-prefill while checking the
actual forward consumed the full prefix. Incomplete first-top1 attribution
cannot be reported complete or trigger automatic resubmission. Original
review artifacts remain unchanged and do not independently review these fixes.

Final local validation:53 new focused tests;304 related tests pass/one skip,
with the existing unknown-asyncio-option warning retained. Exact-copy imports
from153 frozen-source files, Python AST and shell syntax pass. All147 parent
sources remain unchanged;131 raw SHA and the parent static analysis were
independently rechecked. This is local diagnostic validation, not a new GPU
correctness pass, migration pass or algorithm benefit. Logs and source hashes
are in `evidence/public-head-hidden-diagnostic-20261102/`.

Next: protected sync and one unchanged two-A800 common-prefix head/hidden
observation, keeping BF16/default sparse/full original lengths/strict checks.
Use the actual first shared-prefix projection to separate upstream hidden
change from head residual, then choose a layer/kernel diagnostic. Do not
reuse failed timings or claim a root cause from the decomposition identity.
The public seven-policy and held-out benefit objective remains incomplete.

## FP32-head remains insufficient (2026-11-02 02:47+08)

Job276677 ran17:37:56–19:22:36 on November1 and ended FAILED/1:0. After a fresh
November2 02:44 SSH query found an empty user queue, only the owned stale local
coordinator was stopped; the same existing job was archived, not resubmitted.
All131 raw files and two job logs have canonical verified SHA, all147 sources
are unchanged, and independent reconstruction matches the saved full analysis.

Four workers prove BF16 model/head weights and actual CUDA FP32 logits. Both
balanced repeats match all fresh same-setting serial outputs. Every corank
repeat diverges for both background requests; all Code outputs match. Conv1
first differs at zero-based53: serial20632 versus actual12822, with top-two
margin0.024356842 reversing to0.020309448. Conv0 first differs at442 in all four
corank trials. Old/new serial outputs themselves differ and are explicitly
separate from same-configuration correctness acceptance.

Thus changing only logits output dtype does not solve this reproduction. The
original strict RuntimeError remains. The conditional migration stage did not
run, and failed timings remain ineligible for benefit or setting selection.
The next diagnostic records common-prefix head inputs, two candidate weight
rows and original raw logits. Offline high-precision dot reconstruction can
separate changed hidden vectors from head projection error without assuming
which kernel caused either. No numerical root cause is proven yet.

The primary18:09→21:22 wall gap is11540s and the later SSH call failed. The
independent18:07→19:10 wall gap is3754s despite approximately300s monotonic
elapsed time. These are monitoring failures/gaps, not continuous five-minute
coverage; no unsupported attribution to user/device behavior is made. Old logs
are preserved and an independent audit includes the remote query-only watcher.

## Observed monitor continuity (2026-11-01 16:55+08)

Job276677 is still PENDING/Priority. Six primary queries through16:54:49 all
return SSH0; measured wall intervals are300.0026–300.0289s. Five independent
queries through16:52:50 all return SSH0, with monotonic intervals299.9926–
300.0111s. All147 source hashes and producer identities remain consistent.
Primary61604, task-bound caffeinate61605 and independent62423 remain under
PPID1 across commands. The agent thread has handed off and closed; the detached
observer still runs until22:32:50 or terminal archival, without stopping the
primary or remote watchers. Observed continuity is not a promise of future
gap-free operation. No GPU result, correctness pass or profitability exists yet.

## Active explicit head-precision control (2026-11-01 16:31+08)

The sole two-A800 job276677 is PENDING/Priority at the first primary poll
16:29:49 (SSH0). Six source files passed old-SHA backup, atomic replacement
and new-SHA verification; all four immutable inputs passed before/after SHA.
Deployment251 CPU tests passed/one skipped, and the147-file actual isolated
import closure passed. The original141 runtime sources remain unchanged.
The explicit head-precision setting and conditional migration gate are in
[the FP32-head protocol](../docs/fp32-head-diagnostic.md). No GPU correctness or
profitability result is available yet. The old unrelated wait_forecast failure
is not repaired or reclassified; initial local test-path/shell mistakes are
preserved in the preceding diagnostic evidence, not counted as passing tests.

The detached primary61604 and task-bound caffeinate61605 were independently
seen under PPID1. Remote query-only2635253 and a separate query-only agent
monitor continue at300s; only actually observed rounds establish continuity.
Runtime remains frozen. Terminal raw/log SHA plus independent precision/full
output/ownership/budget/graph reconstruction precedes any next-stage decision.
Evidence: `evidence/public-fp32-head-diagnostic-20261101/`. No publication is
authorized; the public seven-policy/held-out objective remains incomplete.

## Terminal zero-copy diagnosis (2026-11-01 16:17+08)

Job276017 ran on two A800s from13:04:09 to14:46:40 and ended COMPLETED/0:0.
All112 raw files and two job logs have verified SHA manifests. Independent
reconstruction reproduces the saved six-layout analysis; all141 frozen sources
remain unchanged. The underlying complete-greedy RuntimeError is retained.
Diagnostic completion is neither a strict correctness pass nor algorithm benefit.

Both balanced repeats match every serial token. All four corank trials diverge
for both background requests but match Code. Conv0 first differs at token130 or
139 (zero-based); Conv1 first differs at252. At Conv0's first shared prefix,
the serial top-two margin is0.25 while actual corank candidates tie. This is
evidence for investigating numerical rounding, not proof of a particular kernel
bug. Zero copies and zero pauses exclude migration as a necessary cause of this
reproduction; they do not prove migration correctness.

The next isolated control uses the installed vLLM's `hf_overrides` setting
`head_dtype=float32`, retaining BF16 model/head weights and default sparse graphs.
All serial references are regenerated under the same new setting. Effective
configuration and four worker head/weight dtypes must be independently checked.
Only a full-token static pass permits same-allocation real migration controls;
all controls use the same head setting. Old default-head costs cannot establish
new-head profitability without verification/recalibration. Failed timings remain
ineligible for selection. No batch-invariant flag or relaxed comparison is used.

Primary and independent monitors have terminal audits (28 scheduled and two
immediate independent rounds, SSH0). A fresh16:17:30 SSH query finds no user jobs.
Evidence: `evidence/public-static-batch-diagnostic-20261101/`; the installed
logits source is saved there. The public seven-policy/held-out goal is unfinished.
Older PENDING records below are historical, not current state.

## Verified monitor and resource state (2026-11-01 12:44+08)

Job276017 remains PENDING/Priority. Primary intervals are300.001/299.985s at the
12:34/12:39 polls, both SSH0. Independent observer19934/PPID1 records12:35/12:40
rounds with299.951s monotonic interval and no frozen-source/producer differences.
This verifies the observed rounds, not future gap-free monitoring. Main19117/
19119/caffeinate19118 and remote1132067 continue through terminal archival. The
independent observer's16:00 bound does not stop either primary watcher; the remote
watcher defaults to1201 polls, approximately100h, unless the job terminates.

The12:41:53 resource snapshot shows zero free GPUs in the public A800/H100/H200
queues. The site's summary lists16 `hp_a800` GPUs, but12:42:27 actual Slurm lookup
returns `Partition hp_a800 not found`; it is not an available queue for this
account. Do not switch QOS or duplicate submission from that summary. Slurm's
November1 21:38 estimate is not an allocation promise. Remain on the sole A800
job with unchanged full original responses,100K Code and default sparse graphs.

Read-only installed-backend snippets are archived in
`evidence/public-static-batch-diagnostic-20261101/installed-fa2-builder-path.txt`.
They show the separate FA3 AOT-only path; a presumed FA3 graph-split setting is
not evidence of an FA2 repair. No installed vLLM source changes or inference on
the login node occur. Kernel/copy root cause remains unproven pending the no-copy
controls. No new public GPU benefit or held-out result is available.

## Sole static control submitted (2026-11-01 12:30+08)

The sole replacement diagnostic276017 requests two A800s. Its first primary poll
at12:29:31 observes PENDING/Priority/SSH0; no new GPU output is inferred. Eight
files pass protected old-SHA backup/atomic replacement/new-SHA checks in a16,425
byte bundle; four original inputs pass double-SHA reuse. All141 runtime sources
are frozen and their actual isolated import closure passes.

The final focused33 tests pass;299 related tests pass/five skip; the coordinator
preflight passes200 tests/one skip. An additional all-research regression reports
2090 pass/19 skip/one existing `wait_forecast` test failure, which also reproduces
alone with eight other tests passing. This unmodified wait-forecast module is
not fixed here or used by the zero-copy no_move diagnostic; its original failed
log remains. The all-research suite is not claimed passing. Existing pytest
`asyncio_mode` warnings and earlier local setup failures remain recorded.

Main19117/caffeinate19118 are genuinely PPID1, with remote query-only1132067 and
an independent read-only monitoring agent. Only initial observations are currently
available; actual300s intervals must be demonstrated, not assumed from PIDs.
Continue through the unchanged six static-layout trials and terminal SHA,
independent full-token/logprob/ownership/physical-budget/four-rank graph checks.
Do not overwrite these141 sources or automatically duplicate submission. Evidence:
`evidence/public-static-batch-diagnostic-20261101/`. This is a root-cause control,
not public algorithm benefit, and the final seven-policy/held-out target remains
unfinished. No commit/push or Aegaeon execution occurs.

## Full native failure and next control (2026-11-01 12:21+08)

Job274385 ends FAILED/1:0 at05:22:53 after03:47:20 on two A800s. All265 raw files
and both job logs independently pass SHA verification;133 runtime/176 outer sources
remain unchanged. Old producers have exited and the12:15:56 query shows no user
allocation. The144-case cost campaign independently reconstructs all six profiles
and exactly reproduces `public-context-fit.json`. This calibration is reusable
only if subsequent correctness work preserves the measured configuration/cost.

The first `native-2026103101-N2` drains all four policies and all12 requests, then
fails `Selected trace request differs from its full serial greedy reference`.
NoMove matches all three full references; Code matches in all four policies.
Both long background requests diverge in every migration policy:

| Policy | Conv1 first difference, zero-based | Conv0 first difference, zero-based |
| --- | ---: | ---: |
| random_no_return |8077|8281|
| kv_only_headroom |8077|8388|
| kv_only_workaware |8077|8281|

The unmoved stayer diverges too. All four migration hops retain device-bytewise
echo evidence, but that is not a full inference-correctness certificate. Workaware
performs one admission hop and one return; both divergences occur before its
return at trace+155.737s. Neither a copy defect nor a numerical/kernel cause is
proven. The observed backend is vLLM0.26.0/g568afb3a1, FLASH_ATTN/FlashAttention2,
so a presumed FA3-only explanation would not match this execution.

The remaining five native cases, rate search, seven policies and held-out do not
run. Failed timings remain diagnostic-only, ineligible for benefit/configuration
selection. The earlier successful threefold mechanism result remains separate;
the user has canceled further fourfold response experiments.

Next is [a static same-rank, no-copy control](../docs/static-batch-diagnostic.md):
the identical original public responses and >=100K Code, original arrivals and
900s window, six mirror-order balanced/corank0/corank1 trials, zero pause/copy,
and full greedy outputs/top2 logprobs. The failed native worker, BF16 and default
sparse graphs are preserved. Strict production checks stay unchanged; an allowed
strict-output error is retained in a separately labeled diagnostic, never a
passing benefit. The29-test focused regression passes; full regression and
protected installation precede the sole A800 allocation.

Independent artifacts are in `evidence/public-efficient-optimizer-chain-20261101-r2/`:
`independent-cost-verification-20261101.json` and
`independent-native-divergence-20261101.json`. Local initial cwd/zsh/PYTHONPATH
setup failures remain recorded, not counted as passing tests. No commit/push or
Aegaeon execution occurs; final public-data benefit is still unfinished. Older
RUNNING sections below are historical observations, not current job status.

## Live calibration progress (2026-11-01 01:54+08)

The01:50:55 primary poll is RUNNING/SSH0 on the same sole274385 allocation,
with5/144 cost cases completed in the runtime progress file. Both model engines
have loaded checkpoints and default seven sparse captures; kvcached pools are
ready. This is partial GPU calibration, not terminal independent verification,
native migration benefit or final public trace results. The detached coordinator
continues all six native phases and measured-rate gates, then conditionally the
separate seven-policy/held-out allocation. Main/remote polling remains300s;
all133 runtime/176 outer hashes stay frozen. No source overwrite or publication.

## A800 execution started (2026-11-01 01:45+08)

Job274385 starts01:35:33 on d1n41a26g03 with two A800-SXM4-80GB GPUs. The
01:40:55 primary poll is RUNNING/SSH0, loading checkpoint in
`public-context-calibration`. Compute input verification preserves the original
manifest and six immutable traces. The134-test launch group and193-test nested
group are observed passing, with the existing warning; these are not144 completed
GPU cost cases or native/algorithm benefit. Both GPUs show81,153MiB free before
load. Main29625/caffeinate29626, remote4050940 and local-only30625 remain genuinely
alive. Seven sidecar rounds reach01:38:07; first/second interval is300.135655s.
The primary's latest interval is300.044799s. Current133 runtime/176 outer sources
stay frozen. Continue the same allocation to strict native and rate verification,
then conditionally seven policies/held-out; no new public benefit is claimed.

## Sole replacement submitted (2026-11-01 01:12+08)

Full deployment regression passes995 tests, retaining the existing pytest warning.
Protected synchronization verifies144 source files, old-SHA backups and atomic
replacement with a374,100-byte bundle; four immutable273918 input files pass
double SHA reuse. Sole job274385 submits at01:10:41 for two A800s. Both primary
01:10:55 and independent01:11:46 observations are PENDING/Priority; no GPU result
is inferred. Main29625/caffeinate29626 and remote query-only4050940 are genuinely
PPID1. Current133 runtime/176 outer hashes are frozen;125 prior control sources
and public inputs remain unchanged. Continue300s polling through native/rate gates
and, if qualified, seven-policy/held-out evaluation. No public benefit or optimal
setting is yet complete; preserve274238 and all older failures.

## Compute-entry recovery (2026-11-01 01:04+08)

Job274238 fails1:0 at00:42:30 with elapsed zero, before inference. All four raw
files and both job-log SHA hashes independently match; stdout is empty. The
compute launch helper eagerly imports undeployed local `public_poisson_followup`.
This is an entry/deployment failure, not a numerical, capacity or solver result.
The original archive and167-source frozen snapshot are retained. A stdlib-only
`efficient_public_launch.py` now generates/verifies immutable launch scripts and
loads only role-specific analysis when requested. Explicit verifier/runtime
prerequisites remove reliance on old remote modules. Twenty-nine tests pass,
including isolated directories containing only each actual sync set; original
cwd/patch failures remain recorded separately. New root:
`evidence/public-efficient-optimizer-chain-20261101-r2/`.

The user explicitly skips fourfold synthetic Conv. Resume immutable original
OpenThoughts3 responses with >=100K SWE-smith, unchanged robust objective,1s
solver budget, BF16/128K/default sparse,32-step control and10,240MiB budget.
Full regression and protected synchronization precede a sole two-A800 allocation;
native correctness and measured rate gates precede seven policies and held-out
traces. No new public GPU benefit has passed. Older status sections are historical.

## Current execution and gated continuation (2026-11-01 00:20+08)

Job273918 completes2026-10-31 23:59:34 with0:0 on two A800/d1n41a18g02. All163 raw
files pass canonical SHA archival. Compute and local independent reconstructions
agree on14 original-length same-shape controls, eight state transactions, eight
migration hops, four sparse graph ranks, and two strict plain no_move repeats
with all six complete greedy request outputs. Four paused roundtrips also retain
their complete original-length tokens. Coordinator5222/caffeinate5223 exit after
closure; the local-only observer5738 records its terminal stop at00:08:22. This
proves the declared controls, not mixed-batch migration or algorithm benefit.
Earlier numerical/transport failures and clock/monitor-gap evidence remain intact.

A separate opt-in continuation is prepared in `efficient_public_continuation.py`
and two wrapper sbatches, leaving canonical source/launches unchanged. Twenty-one
new tests and55 related regressions pass, including verifier restoration,
strict parent closure, prompt-bank SHA reuse, exact generated-launch reconstruction
and no evaluation on unqualified native pressure. The pure robust solver and
original token/economic/physical checks remain unchanged; only the previously
certified equivalent solver expression is selected for robust phases. Point
calibration/baselines use the original entry. Compute and local analysis both
verify executed launch bytes and SHA, rather than trusting a manifest alone.

The gated coordinator9257/caffeinate9258 clears all parent gates and submits the
sole continuation274238 after987 CPU regressions pass. Protected synchronization
verifies135 files with a355,076-byte bundle; four immutable public input files are
reused from273918 after source/destination SHA checks. The00:16:54 primary poll is
PENDING/Priority/SSH0, with299.989s between its first two polls; remote query-only
watcher3125763 also runs. Current124 runtime and outer167 source hashes are frozen.
No new GPU inference or benefit has passed. The old local-only handoff waiter10044
miscounts the seven-field controls object instead of its14-entry trials list and
exits00:11:39. That sidecar error remains preserved, not classified as a GPU fault.
Evidence-only repair launches the sole replacement22693/PPID1 at00:15:18; its
second00:20:18 round measures300.033s with no new alerts and genuinely observes
274238/search. The416.05s parent-observer-to-handoff gap is retained, not claimed
gap-free. The sidecar stops at03:30 or terminal closure, without stopping primary
300s monitoring. Receipt is
`evidence/public-efficient-optimizer-chain-20261031/local-monitor-subagent/continue-search-20261101T0013/quick-handoff-confirmation-20261101T002052.json`.
The finite six-native/rate search precedes pressure-qualified seven-policy/held-out
evaluation, one allocation at a time. Any failure preserves diagnosis instead of
blind duplicate submissions. Evidence is `evidence/public-efficient-optimizer-chain-20261031/`.
Partition MaxTime UNLIMITED is not a guarantee of every account/QOS constraint. See the
[declared live gates and tail-sample limits](../docs/equivalent-solver-live-protocol.md).

### Closed no_move capacity witness

An additional independent reconstruction rechecks all163 raw SHA entries and
reads stable population observations plus the unchanged empty Code-pool geometry.
Both repeats have one Conv request on each rank when Code arrives. Each GPU maps
1,728MiB against the identical10,240MiB controlled KV budget. Code needs6,356 blocks;
its existing mapped pool has63 free slots and64 slots per96MiB page, so at least99
new pages/9,504MiB are necessary. Each rank therefore needs at least11,232MiB,
or1.096875 times its budget, while aggregate free space17,024MiB exceeds the
9,534MiB incoming logical allocation. This is a physical-page lower bound and
budget infeasibility, not a whole-GPU OOM claim or a new migration measurement.

|Repeat|Code queue / TTFT (s)|Conv avg TTFT (s)|Conv avg TPOT (ms)|
|---|---|---|---|
|0|96.8575 / 136.0533|0.10918|12.06950|
|1|96.7427 / 136.3984|0.11218|12.09131|

Code eventually enters after natural Conv completion in this finite population.
There is only one Code/two Conv per repeat: this is not a sustained Poisson trace,
not stable P95/P99 evidence and not proof of50% algorithm improvement. `no_move`
remains the same full-reservation custom scheduler, not unmodified incremental
native vLLM. Derived evidence is
`evidence/public-roundtrip-state-control-20261031-r2/independent-no-move-capacity-review-20261101.json`.

## Allocation recovery (2026-10-31 22:23+08)

The11-source protected update succeeds with an18,450-byte bundle;13 immutable
frozen files are reused after source/destination SHA checks. The sole retry is
job273918, observed PENDING/Priority by the22:21:39 primary poll and an independent
22:22:00 read-only ledger inspection. No GPU correctness or benefit is inferred
from submission. Coordinator5222 and task-bound caffeinate5223 are independently
PPID1; remote query-only2742712 accompanies300s primary polling. Local-only5738
has actual22:18:22/22:23:22 rounds300.022s apart, stable producer identities and
no new alerts. Its first5593 launch fails declaration validation without network
access and exits; that failed receipt and the startup gap are preserved rather
than claiming fault-free continuity. The corrected sidecar is bounded to
November1 03:30 or terminal SHA plus independent verification.

The continued53-file CPU regression passes1004 tests; two supplementary independent
solver/trace-verifier files pass18. Both retain the existing pytest config warning.
All125 active runtime sources remain frozen; the equivalent solver is still
local-only. Continue the same allocation through14 strict state controls and two
plain no_move repeats, canonical terminal SHA archive and full local reconstruction.
Only then resume equivalent-solver native/rate/seven-policy/held-out experiments.
The best public setting and research objective remain incomplete.

## Recovery preparation (2026-10-31 22:17+08)

Job273701 fails1:0 at21:30:24 before the first noop pauses inference: its new
launch omitted explicit `DP_OPT_KV_TRANSPORT=dp_echo`, so the exact-bytewise
roundtrip guard rejects the default `host_file`. All44 regular raw files are
SHA-archived and independently diagnosed. Four completed serial cases preserve
full greedy tokens across ranks (Code101626+64, Conv74+16557); the remaining
Conv149+16314 controls, migrations and plain no_move were not executed. This
is not a token mismatch or CUDA OOM and does not establish migration correctness.
Two legacy archives reject the old three-component relative progress pointer;
a constrained local recovery extracts only declared old control paths. Future
launches instead use canonical absolute `pilot/progress.json` and
`plain-no-move/progress.json` pointers, without changing the canonical extractor.
The primary SSH connection fails21:33:43 and recovers21:38:43; that gap is retained.

The sole R2 root is `evidence/public-roundtrip-state-control-20261031-r2/`.
Its first uploader times out after240s with a partial1,044,480-byte bundle;
remote inspection confirms no source installation, submission ledger or GPU
job. The old producer exits and the failed attempt remains preserved. Recovery
checks every remote273701 frozen input against the unchanged local SHA, uploads
only11 declared sources and trace, then copies the verified immutable parent
inputs locally on the login node and rechecks source/destination SHA. A scoped
legacy SCP wrapper is optional transport recovery, not proof of a protocol bug.
Six new staging tests plus control/archive tests pass52 cases with one existing
pytest warning; outer-CWD and missing-PYTHONPATH collection failures are retained.
All125 R2 frozen runtime hashes remain unchanged. The state is prepared at this
checkpoint, with no new producer/job yet confirmed. Explicit dp_echo and correct
progress paths are fixed;14 strict controls, then two plain no_move repeats,
must pass SHA and independent full-token/state verification before new solver
deployment or the public seven-strategy/held-out benefit search. No publication
or Aegaeon execution is authorized; no best public setting is claimed.

## Historical continuation (2026-10-31 21:20+08)

Job272931 completed0:0 at18:00:06 on two A800/d1n41a28g02. All63 regular raw
files passed terminal SHA reconstruction; the independent diagnostic reproduces
its archived summary. Both no_move repetitions have zero full-serial token
differences and zero paired-repeat differences. The original272106 failure is
not reclassified. Because this diagnostic adds `logprobs=2`, non-reproduction
does not establish the original numerical root cause or migration correctness.
The candidate-contrast analyzer reports `null`, not a measured zero margin.
Its local-only observer ends normally after15 rounds with300.005826–300.041662s
wall gaps; the sidecar handoff gap is300.017855s. All evidence remains in the
separate numerics root, including `independent-numerics-verification.json` and
`local-monitor-subagent/continue-20261031T1650/completion-audit.json`.

The next [strict state-control allocation](../docs/roundtrip-state-control.md)
retains original public Conv74+16557 and149+16314 tokens, Code101626+64, BF16,
128K/default sparse/default32,24GiB virtual pools and10GiB/GPU controlled budget.
It runs14 same-shape serial/noop/paused-roundtrip controls and independently
verifies eight paused transactions and all complete outputs. Only if these pass
does the same allocation run two plain native no_move traces without logprobs.
Neither phase is a benefit result or arbitrary mixed-batch migration proof.
The new coordinator95287/caffeinate95288 starts21:04:00+08 with independent
sessions and confirmed PPID1. Ten new files completed protected synchronization;
the sole new job273701 is PENDING/Priority at21:08:43+08, with query-only remote
watcher2074883. Its124-file runtime is frozen; no control result is yet claimed.
The21:18:43 primary poll now observes RUNNING, with allocation start21:18:19
on two A800/d1n41a14g01, in the same-shape control phase. Local-only sidecar96441
has two real observations at21:11/21:16 and an independently verified PPID1
identity; it is bounded to November1 09:00 or verified terminal, while the primary
coordinator continues independently every300s. Both124 current and136 original
broad frozen sources remain byte-identical at21:18. No control/native result is
claimed from allocation or model initialization.
The sole new evidence root is
`evidence/public-roundtrip-state-control-20261031/`. New32 CPU control tests,
109 continued solver/diagnostic tests and shell/compile checks pass. A preflight
outer-directory redirect error ran no tests and is preserved with its correction.

The generic equivalent solver improvements now combine concentrated uncertainty
grouping and exact nonbinding-scenario pruning:219 indicators become1 and9
scenarios become3. Three offline replays under the original shared1s budget
find the gate-accepted144.8849747547 return versus147.4890608996 unchanged,
preserving139.1645156242 uncertainty request-s. This addresses incumbent quality
without changing the risk/gain objective; it is not GPU benefit. The opt-in
adapter remains local-only, outside the new state-control deployment. See
[equivalence and evidence](../docs/grouped-robust-uncertainty.md). Numerical,
state-continuity and strict native checks precede the renewed seven-strategy
public Poisson/held-out experiment; the best valid setting is not yet obtained.

The21:10 local integration review also finds that the unchanged legacy verifier
expects all nine scenarios. A local-only independent rational certificate
verifier now reconstructs every original scenario, complete group coverage,
nonbinding upper bounds, minimum constant risk and unchanged copy/ownership/
uncertainty parameters, then delegates the original economic/copy/gate replay
with isolated function globals rather than mutating the canonical verifier.
The opt-in adapter installs/restores this controller verifier together with
the equivalent planner. A separate trace verifier retains all original full
serial/paired-token checks and SHA-checks five executed solver sources. Twenty
tests pass, including planner/MILP/pruner-disabled before/after-pause replay;
an initial after-pause test omitted controller accounting and is corrected with
its failed log preserved. These changes are not synchronized to job273701.

The complete local scoped regression now passes478 tests with one pre-existing
asyncio_mode warning. On the actual preserved failed-native D3, three full typed
controller calls through the unchanged runtime dispatch, under the original1s
budget, all produce one gate-accepted return at144.8849747547 robust request-s.
The independent verifier reconstructs all9 original scenarios and certifies3
retained scenarios before unchanged economics/physical replay. CPU walltimes
are.488236/.200581/.178665s. This tests integration beyond a standalone ILP
snapshot, not GPU service benefit. A first local call bypassed the ordinary
dispatch and incorrectly passed its optimizer flag to the typed adapter; it
was rejected before solving, then corrected using the unchanged dispatch and
verifier setting projection. The retained evidence is
`actual-D3-efficient-controller-integration.json` in the new control root.

## Historical diagnostic and solver investigation (2026-10-31 17:22+08)

Job272931 starts17:17:59 on two A800/d1n41a28g02; the17:21:09 primary poll
observes RUNNING/loading, not a completed numerical or benefit result. The
earlier21:14 scheduling estimate was not a promise. The aggregate hp_a800
availability is not usable: actual partition lookup still returns not found.
There is only one allocation, with unchanged runtime and300s monitoring.

The independent headroom audit rules out the proposed explanation that Code
prefill latency was directly included in the cold sample: it is population times
measured migration-admission control. The archived D3 has a free Code-sized
slot, but that alone does not prove returning Conv is unprofitable. Subsequent
exact CPU enumeration finds a feasible one-Conv return with robust objective
144.884975 vs the unchanged147.489061 request-s; it passes the original gate.
The nine scenarios divide a1s solver budget into roughly.111s each. Local
time-limit sweeps reproduce rejected no-move incumbents with71.851081 of
auxiliary objective slack; .5s finds a verified move. This is candidate-quality/
solver-budget evidence, not GPU benefit or proof of an incorrect headroom action.

A new opt-in [exact uncertainty grouping](../docs/grouped-robust-uncertainty.md)
keeps the original objective and bounds while reducing219 uncertainty variables
to1 for this concentrated-layout snapshot. Its26 oracle tests pass; .2s yields
the same verified optimum in three trials, while.111s still does not find a move.
No canonical/deployed solver was changed. Evidence: the new diagnostic root's
`headroom-code-audit/` including original audit, exact enumeration, time-limit
sweeps and grouped replay. Numerical and same-shape migration correctness
remain gates before new GPU benefit measurements; do not retune held-out data.

## C1 terminal token failure and separate diagnostic (2026-10-31 16:51+08)

Job272106 ended FAILED/1:0 at16:19:43 on two A800/d1n41a22g03. All257 regular
terminal raw files were archived and independently SHA-rechecked. The144 cost
cases passed, but the first native four-policy phase fails the unchanged full
serial-greedy token check. No later native, rate, seven-policy grid or held-out
phase ran. Its timings are not accepted benefit data and do not select a setting.

The no_move Conv0 already diverges at zero-based token139/arrival offset3.278602s,
before Code arrives at90s, with no migration. Conv1 and the Code64 outputs match
their full serial references. Each migration policy records one admission move
and zero returns; recorded copies are bitwise-verified, which alone does not
prove complete migration correctness. Workaware release gates include two
`no_profitable_rebalance` and two `outside_verified_workaware_calibration` decisions.
Neither a numerical root cause nor a KV-migration fault is established yet.

Evidence: `evidence/public-quality-verifier-recovery-20261031-r2/failed-native-token-audit.json`
and `evidence/public-openthoughts3-quality-workload-20261031-r2/live-trace-272106/`.

The only new allocation is job272931, two A800/one hour, observed PENDING/Priority
at16:46:09+08. It repeats the identical no_move trace twice with top2 token
logprobs, preserving all original lengths, BF16/128K, default sparse graphs and
budgets. Instrumented timing is diagnostic, never a benefit comparison. The
unchanged strict runner errors remain archived; no tolerance acceptance or
batch-invariant mode is enabled. Its coordinator61238/caffeinate61239 and remote
query-only1659146 monitor every300s, then archive SHA and independently rebuild
the common-prefix comparison. See [diagnostic protocol](../docs/batch-numerics-diagnostic.md)
and `evidence/public-batch-numerics-probe-20261031/`. Numerical evidence and a
strong migration-correctness control must precede restarting the benefit chain.

## Local next-control preparation (2026-10-31 17:03+08)

The17:01:09+08 primary poll still observes job272931 PENDING/Priority; the actual
three successive query gaps are299.99779/299.935425/299.992135s. Local preparation
adds a common-candidate logprob contrast audit and a separate14-case same-shape
pause/noop/roundtrip control, not synchronized or submitted. It checks bytewise
serialized state, both exact KV hops, unique ownership and complete same-source
greedy outputs with no intermediate inference. This does not establish arbitrary
mixed-batch correctness or excuse the original divergence. Protocol:
[same-shape roundtrip](../docs/roundtrip-state-control.md). The combined99 CPU
tests pass with one existing asyncio_mode warning; shell syntax/compile checks
pass. Initial local path/redirect errors were corrected and recorded, not GPU
failures. All114 current diagnostic and136 original frozen source hashes remain
unchanged (`continued-frozen-source-review.json` in the new diagnostic root).

## C0 terminal diagnosis (2026-10-31, Asia/Shanghai)

Job271324 used two A800 on d1n41a11g02, started05:49:48 and failed10:07:14
with Slurm exit1:0. A fresh11:25:22 query found the user queue empty. The144
single-active cost cases and first native phase `native-2026103101-N2` completed;
the native phase itself has exit0 and all four policies drained. The outer job
failed in `verify_trace.py`: `verify_prompt_bank` assumed every real prompt was
a SWE-smith trajectory and indexed `entry["traj_id"]` for a public Stratos prompt.
This is a verifier schema failure, not evidence of CUDA OOM or an inference crash.
The job remains FAILED, not a successful calibration/bootstrap.

Evidence: `evidence/public-quality-verifier-recovery-20261031/` and the unchanged
`evidence/public-stratos-workload-20261030/`. All255 terminal raw files were
SHA-rechecked. Closed-cost62 files match both remote before/after manifests and
are already present in the complete terminal archive. The103925760-byte partial
cost tar and SSH255 error are retained; no redundant large download is needed.
The local cost fit reproduces within the existing rel1e-8/abs1e-12 tolerance;
exact float equality is false and is not claimed. No raw result was overwritten.

## Verified first native phase, not a positive benefit result

Fixed seed2026103101/N2, original Stratos Conv O5165/4856 and one original
SWE-smith Code P101626/O64 at90s. Each policy completed Conv2/Code1; all policies
made zero migrations. Code P95/P99 equal its single observed TTFT and have no
tail-statistical significance. Conv/combined tails here are likewise descriptive.

| Policy | Conv TTFT avg/P95/P99 (s) | Conv avg TPOT (ms) | Code TTFT avg=P95=P99 (s) | Combined TTFT avg/P95/P99 (s) |
|---|---|---:|---:|---|
| no_move | .1110/.1319/.1337 | 12.2386 | 25.9106 | 8.7109/23.3329/25.3950 |
| random_no_return | .3913/.4018/.4028 | 19.0865 | 34.3769 | 11.7198/30.9795/33.6974 |
| kv_only_headroom | .3707/.4781/.4876 | 13.4569 | 26.1962 | 8.9792/23.6256/25.6821 |
| kv_only_workaware | .1463/.1903/.1942 | 14.5602 | 25.6129 | 8.6352/23.0712/25.1046 |

Under no_move both Conv requests finish by63.16s, before Code arrives at90s.
There is no capacity blocking or return opportunity in that trial. Differences
between zero-move policies cannot be attributed to ILP or return benefit. The
144-case fit is cost-only, not evidence of joint-tenant performance gains.

## Explicit independent C1 recovery

The inactive V2 future133 SHA set was verified unchanged, backed up and retired
explicitly after all C0/recovery/fallback owners exited. The original V2 manifest,
quality declaration and C0 input/result identities remain untouched. The schema
fix retains mandatory SWE trajectory matching and adds mandatory public dataset,
revision, shard/row/boundary/response/source-SHA matching and original O equality.
It rejects missing provenance or response extensions, rather than fabricating
trajectory IDs or weakening the source checks.

Recovery uses the already-predeclared OpenThoughts3 quality cohort and unchanged
C1 calibration/evaluation roots from `quality-followup.json`. This is an explicit
diagnosed-failure recovery, not the no-qualified-rate fallback, and does not label
C0 successful. The previous genuinely completed environment bootstrap269991 is
used only for environment/protected-patch identity. C1 remeasures its own144 costs,
N2/N3/N4 native phases and three3600s open-loop rates; it does not reuse Stratos
service estimates. The predeclared public bank256 has original Omean13210.926,
not artificially extended responses; fixed C1 N2 responses are16557/16314 and
12980/12085 for the two declared seeds. Code draws remain unchanged.

Only a fully SHA/independently closed, pressure/queue-qualified C1 rate permits
N5..9 supplementary cost calibration, seven-policy finite-grid evaluation and the
fixed held-out seed. All negative/unqualified outcomes remain. No best setting,
C1 GPU gain or final research completion is claimed at preparation time.

## Local startup timeout and R2 identity (12:04+08)

Final unchanged-source regression passed945/default (13 optional skips) and958/
isolated (zero skips),136 frozen files/50 test files. The first C1 local startup
regression subsequently stopped before synchronization or Slurm submission:
857 passed, one shared-marker shell subprocess timed out at10s. A19-minute local
observer wall gap versus300s monotonic gap and a system hibernate/wake observation
are preserved; this is not a GPU result. The exact three parameter cases re-pass
in1.35s without changing/skipping tests or extending their timeout.

All R1 identities/states/logs remain. Independent R2 recovery/calibration/evaluation
and declaration roots are under `evidence/*20261031-r2/`; only operational root
paths change in the new declaration. All13 frozen input/trace/config files are
byte-identical to the first prepared C1, as are136 source SHA values. The original
quality declaration/V2/V3/C0 stay untouched. R2 launcher23975, task-bound
`caffeinate -is`23976 start12:04:28; this prevents idle/AC system sleep but cannot
promise continuity through manual suspension. No GPU submission is claimed by
this launch receipt. A new local-only observer replaces exactly identified22009;
old logs and1140.148s wall/300.010s monotonic interval remain evidence.

R2 startup858 tests pass;120 remote files pass old-SHA backup/atomic replacement/
new-SHA checks. Unique job272106 submits12:12:14, starts12:12:31 on d1n41a22g03
with two A800; fresh12:13:58 confirms RUNNING/public-context-calibration. The
remote query-only2695959 marker is ln02/04:12:21Z. The local observer has actual
12:04:28→12:09:28 wall299.99999/monotonic300.00246s evidence; this does not imply
uninterrupted monitoring of any other interval. Current allocation/calibration is
not a C1 benefit or best-setting result; all qualification/held-out gates remain.

### Capacity opportunity, estimate rather than an allocation proof

With the previously verified96MiB/1024-token geometry, fixed native seed2026103101
has Code101626+64→9600MiB. N2 Conv74+16557 and149+16314 each require17 independent
pages→1632MiB: balanced local demand with Code11232MiB exceeds the declared10240MiB
KV budget, while aggregate12864MiB is below20480MiB. N4's alternating balanced
Conv totals3072/2880MiB; either Code home needs12672/12480MiB (1.2375/1.21875×budget),
but aggregate15552MiB remains below20480MiB. The unchanged N2/N4 draws therefore
target approximately1.1/1.2 local pressure without modifying original responses.

These are independent whole-page estimates, not actual allocator/fragmentation
or whole-card OOM witnesses; pinned/cached overhead and mature-copy bytes remain
to be checked in the native records. Cost-only calibration has started producing
real completed cases, not stalled initialization. A further causal check is whether
return placements separate the longest remaining Conv requests across GPUs so
later population transitions remain balanced. If workaware/headroom choose the
same plans, do not claim a unique attention-cost or future-phase benefit.

## Continuation audit, 2026-10-31 13:09+08

The same two-A800 job272106 remains RUNNING on d1n41a22g03. Primary poll
13:07:21.122589 reports public-context-calibration40/144 with SSH exit0;
wall299.714s and monotonic300.006s since the previous poll. This is still
cost-only, not native admission, return repayment or the best-setting result.
The existing coordinator/child/idle-sleep guard/local-only observer retain their
process identities. A bounded independent local-only six-observation monitor
adds no SSH producer and does not restart or overwrite the active runtime.

Independent local SHA review rechecks136 frozen source files, all13 frozen
input/config/trace files and three input payloads: zero changes and matching
input-manifest identity. Evidence is
`evidence/public-quality-verifier-recovery-20261031-r2/continued-independent-review-20261031T1310.json`
(actual recorded timestamp13:08:54+08). Existing C0/R1 failures and monitoring
gaps stay in their original archives.

### Sparse Code samples and unchanged selection

An independent arrival-only replay of the already-frozen future seeds/rates
finds these sample counts, not GPU results or a guarantee of qualification:

| Stage | Code RPS | Offered Code | Measurement Code, after600s |
|---|---:|---:|---:|
| Exploration3600s, seed2026103104 | .0005 | 2 | 1 |
| Exploration3600s, seed2026103104 | .001 | 3 | 2 |
| Held-out10800s, seed2026103105 | .0005 | 4 | 4 |
| Held-out10800s, seed2026103105 | .001 | 14 | 14 |

Code uses an independent arrival RNG, so its marks/times do not depend on the
eventual qualified Conv rate. P95/P99 with these counts are empirical
descriptions, not population-tail significance. Arrival-only held-out inspection
must not retune the algorithm, score, rate grid or seed; held-out outcomes remain
unobserved and selection stays exclusively on exploration metrics. No artificial
Conv response extension, seed reroll, SLO change or dropped request is introduced.

After cost calibration, inspect each original native trace for real local
allocation failure despite sufficient aggregate headroom, matured-copy bytes,
return start/resume relative to Code release and random-baseline natural recovery,
remaining Conv output and pause-inclusive aligned work. Only independently closed,
capacity/queue-qualified measured rate calibration enables the seven-policy grid
and fixed held-out. Better Code admission alone or initial TTFT before return must
not be called return benefit. Preserve zero-move and negative phases alongside
Conv/Code/combined avg/P95/P99 TTFT and Conv mean TPOT.

### Bounded source-generation audit, 13:19+08

The exact source-generation cap remains unconfirmed. The archived primary paper
`arxiv:2506.04178v1` describes32768 **evaluation** output tokens; that is not an
annotation limit. The official repository's fixed commit
`e86cf2dbbe69ae178aec302be33a3454a7183457` contains related C1 Code/Science
annotation settings with max_tokens32000, but those ablation configs are not
proven row-level provenance for our fixed final-release shards. The paper's16K
training-trace description likewise does not establish each row's finish reason.
Do not claim natural EOS or infer an exact source cap from these observations.

All three complete YAML sources match their Git blob SHA and size; the timed-out
1369/2097-byte Math download remains, and an independent blob-verified retry is
stored separately. Primary sources, network attempts, SHA manifest and qualified
conclusions are in `evidence/public-source-generation-audit-20261031/independent-source-audit.json`.
No source selection, original response length, serving cap or active runtime
changes because of this inconclusive audit.

## Independently closed C1 cost profile, 15:19+08

All144 cost cases exit0 at15:05; the primary15:07 poll enters
native-2026103101-N2 in the same two-A800 allocation. The closed cost directory
has59 regular raw files; the immutable job-root public-context-fit and CLI
population-profile output add two separately archived sidecars. Both archives'
before/after/bundle/extracted SHA checks pass,61 files in total. Raw reconstruction
rebuilds all six tenant/population profiles and agrees with the served fit under
the unchanged rel1e-8/abs1e-12 comparison; exact floating-point equality is false.
Source-fit SHA is `8faa5ade4d1de667b7d8c2cb66e34a0fc4f6b9d0f5ead4106097b8d261f96d75`.
Evidence:
`evidence/public-quality-verifier-recovery-20261031-r2/closed-cost-review/independent-verification-r3.json`.

These are single-active-tenant costs with the other model resident but idle,
synthetic O3072 and declared context/layout/repeat coverage. They are not public
request gains, simultaneously active-tenant predictions or isolated CUDA attention
kernel times. At equal16384-token contexts, the fitted concentrated-minus-balanced
cadence differences are:

| Conv population | Balanced→concentrated | Graph delta(ms) | KV-work delta(ms) | Total delta(ms) | Worst held-out contrast residual(ms) |
|---|---|---:|---:|---:|---:|
| 2 | 1:1→2:0 | 5.348 | 3.612 | 8.960 | 3.756 |
| 3 | 2:1→3:0 | 0 | 0 | 0 | 10.218 |
| 4 | 2:2→4:0 | 0 | 4.681 | 4.681 | 8.637 |

N2 mixes sparse capture quantization with fitted KV-work imbalance. N3/N4 fits
remain noisy; N3's nonnegative fitted slope is zero, not proof that true attention
cost is absent. Do not pool these populations into one universal linear slope or
relax the robust uncertainty guard to force returns. The planner must account for
future context/population segments and actual pause/planning/copy costs; a single
current-context delta is not cumulative repayment. Native traces will test whether
returns are timely and distinguish actual gains from natural recovery.

Three local review failures remain: zsh's read-only status variable prevented the
first wrapper exit receipt; the first checker assumed fit.json inside the cost
directory; its retry compared the CLI population-profile subobject to the entire
fit. Correct source/CLI inspection resolves these without modifying or redownloading
the original cost raw archive, changing runtime, or resubmitting GPU work. The
final third review passes. They are local audit errors, not GPU failures.

The second local-only monitor finishes19 rounds13:35:21→15:05:22 with actual
5400.960s wall/5400.741s monotonic,63→141 cost progress, matching process identities,
no new fault/freshness loss and all four historical C0 alerts retained. Its exit0
does not stop the original300s producer/observer/continuation or imply the native,
rate, finite-grid or held-out work is complete.
