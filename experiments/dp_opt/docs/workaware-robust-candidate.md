# Robust return-placement candidate

## Planner interface and independent verification

`workaware_robust_rebalance.py` supplies the release-only controller interface;
`verify_workaware_robust_rebalance.py` independently reconstructs inventory,
physical copy order, costs, uncertainty, references, scenario selection and the
unchanged acceptance gate without calling the planner, solver or MILP code.
Both remain local, not installed in job268245.

The interface compares J inside and across all soft-headroom scenarios and
shares one total solver budget, preserving verified no-move fallbacks. It
requires raw-verified population-specific profiles; legacy global-only error
guards are rejected rather than quietly reduced. The positive-base calibration
contract remains unchanged. A legacy N128 structural placement fit can use the
explicit feasibility-only override; its static performance tables are then
zeroed. Remaining-work performance **still** obeys the measured population and
context domain; this is not performance extrapolation.

Before pause, future pause activation and copying remain charged. After pause,
the already paid planning/blocking is logged but not deducted again from
continuation profit; future copy cost and empirical uncertainty still apply.
Full-request metrics and repayment analysis retain all paid costs. A caller's
planning-time upper bound only constrains reported local time; it does not prove
actual GPU walltime, solver deadlines or global optimality.

The four new planner/verifier files and tests have a separate frozen manifest
`robust-planner-interface-source.sha256`. Final targeted validation passes
87 planner plus98 verifier tests. The first seed's same verified return state,
before and after pause with1s and8s shared budgets, independently replays4/4
with the same accepted move. This is CPU counterfactual validation, not additional
GPU benefit. Final full regression and evidence are tracked in the result report.

2026-10-30. Local candidate only; job268245 keeps its85 frozen runtime/test
files. Nothing here is installed into that running A800 experiment. No new
allocation, commit or push is authorized by this document.

## What the old evidence says

The fully SHA-verified265356 low-Code-rate phase contains31 release decisions:
17 no-profitable,12 outside the measured work domain,1 admission-priority and
1 committed return. Among18 decisions with plans,3 fail the uncertainty guard,
14 have no positive point-estimate move and1 passes. The first Code release
produces6.598 request-seconds of predicted net benefit against78.202 of empirical
error allowance plus the unchanged1 request-second minimum. This is not merely
a lack of time to copy KV. The one committed return is267.991s after the second
Code, later than random's43.769s natural recovery.

Independent enumeration of all9 recorded headroom scenarios for each of these18
states finds no additional guard-passing opportunity. Changing the optimizer
alone must not be advertised as fixing this old trace.

## Objective: put the existing error allowance inside the ILP

For segment s, use the existing calibrated cost

```
C_s(A) = graph_table_s[n0(A)] + attention_coeff_s * abs(D_s(A))
D_s(A) = sum_j (1 - 2 * rank_j(A)) * context_weight_js
C(A)   = sum_s C_s(A) + risk(A) + activation * 1[moves] + sum_j move_cost_j
U(A)   = sum_s uncertainty_s * 1[segment features differ from current placement]
J(A)   = C(A) + U(A)
```

Costs, including the error allowance, are in request-seconds. Keep the calibrated
graph/context coefficients, empirical worst-heldout allowance and minimum gain
unchanged. The allowance is not a statistical confidence interval or a guarantee
about unseen GPU workloads.

Old selection minimizes C first and rejects the winning plan if its error is too
large. It can miss a smaller, safer gain. New selection minimizes J across the
same headroom/rank scenarios; accept only a nonempty move with
`reference_C - winner_J > min_gain`. An exhaustive synthetic regression exhibits
this distinction: the point winner has gain59/error101 and fails, while a short
request's alternative has gain19/error1 and passes. This is a solver-correctness
witness, not a GPU benchmark result.

Binary equivalence selectors represent unchanged count/graph cost and signed
attention difference equal to either the original or its negation. Rank-symmetric
swaps preserve zero differential uncertainty. Different context work or newly
introduced count-cost tables cannot silently receive zero uncertainty. Reuse the
remaining-work ILP constraints; add uncertainty binaries to its objective.

`workaware_robust_ilp.py` implements a standalone release-only solver. It includes
serial copy peaks, physical page-owner release, pinned/unknown-owner pages,
destination-pool reuse, final headroom and activation/migration budgets. A
separate `replay_candidate` rechecks the resulting order and objective. Time-limit
incumbents cannot replace a cheaper independently verified no-move fallback.
Solver time limits and bounds remain explicit; feasible is not optimal.

## Safe no-move fastpath

When the unchanged placement satisfies the selected final headroom constraint,
bound every segment's robust improvement by

```
B_s = max(equivalent_feature_numerical_slack_s, C_s(current) - min(graph_table_s) - U_s)
B   = sum_s B_s - activation - min(move_cost_j)
```

Attention cost is nonnegative, so `min(graph_table_s)` relaxes its best possible
cost. A changed segment pays U_s; an equivalent segment saves at most its explicit
floating-point slack. If B<=0, no nonempty move beats the current placement under
this empirical model, and the ILP is skipped. If no-move violates the scenario's
headroom constraint, do not use this shortcut. Each scenario has constant risk;
cross-scenario comparison still includes its risk separately.

This fastpath reduces avoidable control work without reducing error guards or
using future trace events. It does not invent an attention speedup.

## Why the long-output N2 probe is still needed

With two comparable Conv requests and remaining decode H, the approximate return
margin is

```
margin ~= 2 * H * (layout_cadence_difference - 2 * heldout_contrast_error)
          - pause_copy_cost - minimum_gain
```

Lengthening output amortizes fixed pause/copy cost and postpones natural drain.
But if per-step loss is below the empirical error guard, increasing H scales both
terms and cannot solve that problem. N4–7 typically share sparse graph8; N2 has
a genuine graph8→graph1 transition between concentrated and balanced layouts.
The current job measures new N2/N3 long-context coefficients rather than assuming
the old short-context fit remains valid.

The candidate retains the old current-cohort drain forecast and past-count/rate
priors. Future births may erase imbalance before predicted drain; this limitation
is not solved by robust selection. Neither a synthetic witness nor a shorter
local solver time establishes Conv TPOT or tail-TTFT benefit.

## Validation and next execution gate

New tests compare the MILP against exhaustive assignments and independent serial
copy-order enumeration, including required headroom, shared/pinned pages, reused
pools, rank symmetry, uncertainty, numerical slack and time-limit fallback.
`diagnose_return_gates.py` streams old decisions, checks raw gate arithmetic,
records source SHA and exposes domain/guard reasons without dumping full pages.

Evidence and final candidate-source hashes are in
`evidence/long-conv-return-probe-20261030/`. Exploratory failures and intermediate
solver outcomes are preserved; final evidence uses the frozen candidate hashes.

After268245 is terminal and fully independently verified: compare Code release
to return versus natural recovery, paid pause/copy, Conv avgTPOT and separate plus
combined avg/P95/P99 TTFT. Only then patch the release-planner integration, run
regressions, protected-sync and execute a paired candidate-versus-old comparison.
Do not change runtime while any current phase is active. The eventual Poisson
trace also retains no_move, threshold-fair and random-no-return baselines; the
closed three-Code-request probe cannot establish Code P99 significance.

The independently running local `robust_return_followup.py` waits for the main
campaign's complete archive/verification and then performs the above candidate
replays. It checks state every300s without additional SSH, pins candidate source
hashes, and stops on PAUSE, source changes or diagnosis. It does not integrate
the candidate into live inference or submit another GPU allocation. See
`../reports/workaware-robust-candidate-results.md` for executed CPU validation.

Keep both source manifests and all six named files frozen until that follow-up
closes. Further candidates use new versioned paths instead of replacing the
waiting follow-up's imported modules or updating its expected hashes.
