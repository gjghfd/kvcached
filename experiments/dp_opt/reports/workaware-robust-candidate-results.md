# Robust return optimizer: local validation, not GPU benefit

2026-10-30. Job268245 remains active on two A800s with its85 runtime/test files
unchanged. The candidate is local and not serving GPU requests. It does not
change thresholds, sparse captures, model length, budget or the current trace.
See [algorithm and execution gate](../docs/workaware-robust-candidate.md).

## New controller interface: real-state CPU validation

The separate robust release planner and independent verifier pass185 targeted
tests (87 planner/98 verifier). Final full regression after the last sunk-cost
fix passes792 tests in154.36s with the unchanged asyncio configuration warning.
The earlier694-pass run predates that fix and is not the final validation of
these four frozen files. All85 frozen runtime/test hashes and all local source
manifests remain unchanged.

The first GPU seed's SHA-verified committed-return observation is replayed
before/after pause with1s and8s shared total solver budgets. All four plans
independently verify and select the same Conv move as the old point planner.
CPU walltime is0.788/0.460s before and0.458/0.459s after pause; this is not a
GPU control-cost or speedup calibration. The independent verifier reconstructs
all9 headroom scenarios and grants1/8s total, but does not claim global optimality
or independently prove actual solver wall deadlines.

Before pause the original gate retains gain206.381805/error71.028861/min1
request-seconds. After pause it retains209.181573/70.740765/min1; measured paid
planning/blocking is logged, not deducted again from continuation profit.
The empirical guard, future copy charge and positive-base contract are unchanged.

Evidence: `robust-planner-interface-source.sha256`,
`robust-planner-real-state-replay.json`, `robust-planner-real-state-replay-r3.log`,
and the final regression logs under the recovery root. The failed original
replay's missing feasibility-only override and r2's absent optional `reason`
field are preserved; r3 uses explicit override, `get('reason')`, and the raw
phase's own placement-fit.json. These are local command failures, not GPU faults.
No frozen runtime is modified. The first-seed GPU benefit belongs to the old
point-optimal policy, not this new interface.

The new standalone `replay_workaware_robust_phase.py` first verifies frozen
interface/runtime hashes, complete raw SHA and the original full trace. It
replays every planned before/after-pause state, preserves unplanned releases,
independently checks each incumbent and rechecks source/raw SHA before exclusive
output. Planner input mutation cannot change the verifier's original inventory.
Its22 tests pass; the complete frozen-plus-local regression now passes814 tests
in155.42s with the unchanged configuration warning. A new ten-file combined
manifest freezes this tool without replacing any active source manifest.

First-seed phase verification closes4/4 replays; the old lower-Code-rate phase
closes38/38 (18 before-pause plus one after-pause, each with1/8s total budgets).
Neither phase yields a new guard-passing opportunity. These are conditional
CPU replays, not a new policy trajectory or additional GPU benefit. Evidence:
`robust-phase-first-seed-replay.json`, `robust-phase-old-low-code-replay.json`,
`robust-phase-replay-source.sha256`, targeted/full regression and replay logs.

Next open-loop rate and calibration requirements are documented in
`../docs/long-conv-open-loop-plan.md`; no candidate rate is frozen or run yet.

Second-seed original phase also passes4/4 complete independently verified
counterfactual replays with1/8s shared budgets; no new guard-passing opportunity
is found. Its GPU Conv TPOT benefit belongs to the unchanged point policy, not
the robust interface. Evidence: `robust-phase-second-seed-replay.json` and log.

The new typed controller adapter preserves original planner accounting and
adds actual caller planning/executor time after pause without charging paid
costs twice. Its27 tests and841-test full regression pass (154.97s, unchanged
configuration warning). Both real seeds' before/after observations with1/8s
shared budgets pass8/8 independent controller-format replays; these are new CPU
times, not GPU controller timing. Pre-solve blocking is reconstructed from
the original boundary receipts, not a total already including the old solver.
The12-file `robust-controller-adapter-source.sha256` freezes the new adapter
alongside prior sources, without replacing those manifests. Native routing is
still unchanged. See `../docs/workaware-robust-live-integration.md` for hooks,
evidence and the required terminal execution gate.

## Archived trace bottlenecks

Source: fully independently SHA-verified265356, lower-Code-rate workaware phase.

| Release decision | Count |
|---|---:|
| No profitable return |17|
| Outside measured work domain |12|
| Admission has priority |1|
| Committed return |1|

Of18 planned decisions,3 had a positive point-estimate move rejected by the error
guard,14 had no positive point-estimate move, and1 passed. The first Code release
had gain6.598 vs uncertainty78.202 plus minimum1 request-second. One concrete
domain rejection had Conv counts4:5 and max context12808 tokens against that
population profile's measured maximum10239; a larger global-fit maximum does not
authorize this population-specific extrapolation.

The sole accepted return remains267.991s after the second Code, versus random's
43.769s natural recovery. The old full-request Conv TPOT regression remains
negative evidence. Lengthening output alone is not a fix when per-step predicted
loss is below the retained empirical error bound.

## Candidate and independent checks

- Optimize calibrated inference cost plus existing differential error allowance
  inside the ILP, rather than filtering only its point-optimal winner.
- Preserve serial copy peaks, physical page owners, pinned pages, destination-pool
  reuse, headroom and migration activation/budgets.
- Keep a cheaper verified no-move fallback when the solver times out.
- Skip solving when a conservative improvement bound proves no-move optimal
  within the empirical model. Include numerical slack; do not skip required moves.

Final candidate tests:66 passed. Existing frozen tests plus those tests:534 passed.
Follow-up and candidate tests together:72 passed; all frozen tests plus these:
540 passed. The configured pytest asyncio
option warning is unchanged. Initial5 exploratory failures and fixes remain in
logs: decimal-memory fixtures and a default MILP test-harness gap, not GPU results.
Ruff is not installed locally; no formatter/dependency was added. Tracked
`git diff --check` passes.

All18 planned states were independently enumerated over9 headroom scenarios,
under the same4-move limit and unchanged error guard. Final robust objectives
match the exhaustive reference18/18. The internal solver/certificate proves all
scenarios for13/18 states;5 retain time-limit statuses even though the separately
enumerated optimum matches. There is still only1 guard-passing state: **no new
timely return opportunity** in this old trace.

Final one-pass local planning wall median is0.060640s, maximum3.287509s. This is
a local diagnostic, not paired control-cost calibration or a GPU TTFT/TPOT
improvement. Do not use it as a serving speedup percentage.

## Continuation and evidence

GPU setting remains AzureConv synthetic O24576/closed N2, Code>=100K/O64, three
seeds/four policies, and112 long-context calibration cases first. Latest progress
belongs to [current GPU report](long-conv-return-probe-results.md), not CPU tests.

Main85760/caffeinate85761 and remote query-only1813672 continue300s Slurm/log
checks. Local follow-up1599/caffeinate1600 starts11:31:06; it checks only the main
campaign state every300s, with no extra SSH. After the complete GPU archive and
independent verification, it rechecks raw SHA and runs frozen candidate gate/ILP/
order replays for all three probes. PAUSE, failed verification or source changes
stop it without modifying runtime or submitting a GPU job.

The follow-up's11:31:06 and11:36:06 local rounds are independently observed with
300.008576s spacing;1599/1600 remain PPID1. This is not a claim about whole-run
future coverage. The main Slurm watcher remains responsible for actual queue
queries and terminal archives.

Evidence: `evidence/long-conv-return-probe-20261030/`:

- `robust-candidate-source.sha256`, `robust-followup-source.sha256`.
- `old-return-robust-comparison-final.json`, plus preserved exploratory outputs.
- `robust-candidate-tests-final.log`, `robust-candidate-full-regression-final.log`,
  `robust-followup-tests.log`, and failed/intermediate logs.
- `robust-followup-local.json`, `robust-followup-status.json`, `robust-followup.log`.
- Future `robust-followup-comparison.json` is counterfactual analysis, not a new
  live-policy experiment.

Next live integration waits for268245's complete result and source-freeze release.
The benefit goal remains incomplete. No commit/push, new allocation or Aegaeon
inference occurred.
