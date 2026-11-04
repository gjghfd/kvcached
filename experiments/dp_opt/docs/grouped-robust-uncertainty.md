# Exact concentrated-layout uncertainty grouping

Prepared local candidate only. Do not install into the frozen272931 runtime or
treat the failed native trace's timings as benefit data. This optimization keeps
the original ILP objective, uncertainty coefficients and independent serial-copy
replay; it is not a lower uncertainty threshold or a stronger empirical fit.

For a segment whose requests all start on one rank, positive attention weights
make the absolute work difference equal to total work only if no request moves
or all active requests move. Partial movement changes that feature. If the
endpoint graph costs are symmetric, those same two endpoint counts preserve
the graph feature too. Require weights safely outside the existing feature
equality tolerance; otherwise retain the existing mixed-layout formulation.

Thus the original changed-feature indicator is exactly
`z = 1 - y[0] - y[N]`, where `y[k]` is the existing final rank0-count one-hot
variable. Segments with the same active request/source set already share these
count variables; they can also share one `z` and add their unchanged uncertainty
coefficients to its objective. The equivalence does not depend on lengths or
arrival rates. Completion events with different active sets get separate groups;
mixed initial layouts, asymmetric endpoint costs and tiny weights fall back to
the original constraints.

The preserved native D3 has75 forecast segments. The existing nine headroom
scenarios each receive about1/9 of a one-second solver budget. Local independent
time-limit sweeps reproduce rejected no-move incumbents with71.851081 request-s
of auxiliary objective slack at.08/.111/.2s; a.5s incumbent and5s optimum find a
verified return layout. Exact enumeration gives144.884975 vs reference147.489061
robust request-seconds, and passes the original gain/uncertainty/min-gain gate.
This identifies a solver-budget/candidate-quality problem, not proof that the
original headroom preference itself was wrong or that GPU net benefit exists.

Evidence is in `evidence/public-batch-numerics-probe-20261031/headroom-code-audit/`.
The new grouped functions are opt-in and have not replaced canonical/deployed
solver functions. Validate against exhaustive CPU copy-peak/ownership replay,
then measure the identical snapshot under the same bounded solver budget. Only
after terminal numerical and migration-correctness barriers may a fresh runtime
version use this formulation for new GPU benefit experiments.

## Exact scenario pruning and the original shared budget

`prune_headroom_scenarios.py` bounds final usage with exact rational arithmetic:
`upper[r] = initial_gib[r] + sum(memory_gib of movable requests inbound to r)`.
It intentionally does not subtract any release. A final-capacity constraint at
or above this upper bound cannot bind and can be normalized to physical
capacity. Only otherwise identical snapshots are merged, retaining the smallest
constant headroom risk. Request choices, source ownership, serial copy peaks,
forecast costs and uncertainty remain unchanged; this is not dropping requests
or approximate pruning of potentially binding constraints.

For the preserved D3, the nine scenarios reduce to three equivalence groups:
`[0,1,2] -> 1`, `[3,4,5] -> 4`, `[6,7,8] -> 7`. Combined with219-to1 uncertainty
grouping, the same total1s solver budget gives three optimal scenario solutions
and a gate-accepted return in all three offline repetitions. Total solve times
are.428/.144/.143s; each retained scenario receives about.333s. The chosen robust
objective remains144.8849747547 versus147.4890608996 unchanged, including the
original139.1645156242 uncertainty request-s. No gain threshold or empirical
uncertainty bound is lowered. Evidence: `combined-one-second-replay.json` in the
audit directory. This is CPU candidate evidence, not measured GPU return benefit.

`solver_efficiency.py` scopes the pure-planner substitutions to an explicit,
non-nestable context and restores them on exit or exception. The independent
`efficient_workaware_trace.py` entry requires `--workaware-optimizer robust`
and archives five executed candidate sources plus their SHA. Canonical planners
and existing campaign/runner files are unchanged. This entry is local-only;
the next A800 job tests state continuity and a plain native baseline, not this
optimizer. October31 continued focused regression:109 tests pass, including the
new initial-count/source consistency condition; full new phase validation is
separate. Correctness and fresh GPU measurements still gate deployment.

The subsequent local integration review adds a fifth archived source,
`verify_solver_efficiency.py`, and the independent `verify_efficient_trace.py`
CLI. The legacy verifier correctly rejects a shorter uncertified scenario bank.
The new verifier instead reconstructs original headroom scenarios independently,
verifies exact rational upper bounds, complete one-time original membership,
minimum constant risk and every unchanged snapshot field, then executes the
original cost/copy/gate verifier against certified scenarios using isolated
function globals. It invokes neither planner, MILP nor the pruning function.
The adapter installs/restores this pure controller verifier only in the opt-in
context. The trace CLI retains original strict complete serial and paired token
checks. Twenty focused tests pass, including tampered proof/source rejection,
before/after-pause accounting and restoration on errors. An initial after-pause
test missing caller accounting failed and was corrected; its log is retained.
This closes a local integration gap, not the GPU correctness or benefit gates.

## Interpreting the robust objective, not GPU speedup

The October31 offline D3 component audit independently sums the already verified
plan: unchanged placement has147.489061 request-s of centered performance
penalty, split99.061680 graph and48.427381 attention. The return candidate has
0.134378 performance,0.258971 weighted copy,3.548673 migration activation and
1.778437 headroom risk, totaling5.720459 point request-s. Its unchanged worst-case
contrast uncertainty adds139.164516, yielding144.884975 robust request-s.

Thus predicted point gain is141.768602, but gain after uncertainty is only
2.604086 request-s, above the unchanged1 request-s gate. The small relative
robust-objective difference is **not** a predicted1.8% GPU speedup: uncertainty
is a conservative bound, and these centered penalties are not total inference
latency. Neither the large point gain nor the conservative positive gate proves
measured benefit. The arithmetic receipt is
`evidence/public-efficient-optimizer-chain-20261031/offline-D3-objective-components.json`.
The original failed-native timing remains excluded from setting selection.
