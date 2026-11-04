# Robust controller integration and deployment

2026-10-30 19:02 deployment update: the streamed bash-s/stdin protected install
passes108 files and all protected checks. The sole native robust job269567
starts18:55:09 on two A800/d1n41a28g01; first native seed is running. This is
new native controller execution, not yet an independently verified result or
final Poisson benefit.1488 CPU tests pass with5 skips and the independently
reproduced pre-existing wait-forecast failure deselected, not fixed. Both
declared seeds retain the old trace, budget, sparse graphs and pause32.
Main66872 and remote4081320/ln01 query every300s. Local-only report consumer
70830 has16 targeted tests and two separately frozen sources; it does not
query SSH or submit GPUs. It reports only after full SHA and independent
native verification, preserving every request and repayment exclusion.
Recovery root: `evidence/workaware-robust-native-20261030-r2/`. No publication,
Aegaeon restart or change to the108 frozen GPU files is authorized.

Deployment reconciliation: the first108-file protected installation exceeds
SSH's remote single-argument limit before execution. No sbatch ledger or backup
is created; all108 remote old SHAs and the empty queue are independently checked.
The failed native coordinator exits before submitting any GPU job. Stream only
the guarded installation script through `ssh paracloud bash -s` stdin, retaining
all old/new SHA, backup and atomic-replace checks.30 targeted tests pass; a new
complete regression precedes the one bounded retry. Keep the failed root and
do not mistake this control-plane error for an inference or KV capacity result.

2026-10-30 closure update:268245 completes with585 raw files/36 trace requests
independently verified. Both old follow-ups close before13 files are promoted
to canonical main, preserving104 original files and all old manifests.117
canonical focused tests pass; the promoted independent verifier revalidates
all three historical point-policy phases without changing raw evidence. Full
canonical regression and protected deployment still precede the new native
GPU allocation. The initial promotion patch-header error makes no source
changes and is retained, not counted as a passing test.

2026-10-30 17:35 update: native routing, controller accounting, independent
verification, CLI/source-archive optimizer markers and the sbatch flag are
implemented only in the isolated `robust-live-staging/kvcached` evidence copy.
The canonical85-file runtime remains unchanged while268245 and both existing
follow-ups run.106 focused tests pass. The expanded all-dp_opt run records1464
passes,5 skips and1 pre-existing `wait_forecast` failure; an unchanged canonical
run reproduces the same failure. Do not report a clean full-suite pass or fix
unrelated forecast behavior. Zero-arrival tenant support is also being tested
in that copy, not deployed. Promotion still requires the complete archive and
independent analyses described here.

2026-10-30. Job268245 still runs the original point-policy85-file runtime.
The new adapter and tests are local only. Wait for the complete GPU archive,
three independently verified seeds and both existing follow-ups before changing
canonical routing or synchronizing any serving file.

## Preserve both planner and controller clocks

`workaware_robust_adapter.py` wraps the independently checked robust planner
without duplicating its large scenario snapshots. It preserves original scope,
planning time and gate in `controller_adapter.raw_planner_accounting`; its own
scope is explicitly a controller plan, not GPU performance evidence. The
optimizer/schema marker prevents an old point-policy record from being silently
interpreted as a robust-policy record.

Before pause, planning time remains a CPU observation, not paused inference.
After pause, `account_paused_controller_planning()` replaces the accounting
clock with the controller's full awaited planning duration, including executor
queue time. It records `active_population * controller_elapsed` and all measured
pre-solve blocking. It **does not** change gain, uncertainty, minimum gain or
continuation acceptance: those costs are already paid. An empty population has
zero active blocking even if control work takes time.

The independent adapter verifier reconstructs the original raw plan, invokes
the solver-independent robust verifier, then checks outer accounting and the
unchanged economic gate. Caller-reported time is not independent clock proof;
the actual live receipt/solve timestamps and all-request metrics remain required.

## Executed local validation

-27 targeted tests pass, including real-solver synthetic plans, raw/outer clock
  tampering, invalid times, schema identity, empty populations and unchanged
  continuation profit after large paid planning costs.
-The complete frozen-plus-local regression passes841 tests in154.97s with the
  unchanged asyncio configuration warning.
-Both first/second seed's real SHA-verified observations independently replay
  before/after pause with1s/8s shared budgets:8/8 typed plans verify. Timing is
  new local CPU timing, not the old GPU controller's timing or new GPU benefit.
-The eight replays derive pre-solve blocking from original boundary receipts and
  `blocking_sampled_at`, rather than reusing a gate total already including the
  old planner. All85 original runtime/test source hashes remain unchanged.

Evidence under the current recovery root:
`robust-controller-adapter-source.sha256` freezes12 local files without replacing
any active manifest; `robust-adapter-real-state-validation.json`, targeted/full
regression logs and validation log retain the executed results.

## Minimal future hooks

1. Add an explicit point/robust optimizer choice to the serving CLI/config and
   release report, defaulting to point for historical compatibility.
2. Dispatch only workaware release plans to the typed adapter when robust is
   selected; do not change headroom, admission, graph, model or KV configuration.
3. After pause, account the full controller awaited duration without changing
   future-copy economics. Preserve raw planner time separately.
4. Select the matching independent verifier from the recorded optimizer. Check
   pre-solve blocking against real four-rank receipts and planning against actual
   controller timing; reject missing/mismatched robust markers.
5. Revalidate historical point phases, run focused/full CPU regressions, perform
   protected explicit-file sync and execute the next single allocation. Preserve
   every old result and source hash; do not overwrite an active version.

Native-controller GPU verification, zero-tenant Poisson handling, open-loop
service/rate calibration and final paired trace evaluation are still pending.
The adapter's CPU correctness alone does not complete the research-benefit goal.
