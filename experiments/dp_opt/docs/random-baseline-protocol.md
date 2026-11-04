# Random feasible-DP migration without release-time return

Protocol frozen prospectively on 2026-10-28, Asia/Shanghai. The active five-policy
job259091 remains immutable and finishes its complete paired trace before this
candidate is promoted, protected-synchronized or submitted. No second allocation
request is created while it is queued/running; no commit/push is authorized.

## Baseline semantics

`random_no_return` retains the same direct-admission count balance and full-KV
reservation as the existing scheduler. Only a new request blocked by rank-local
capacity triggers migration. Both tenants may donate requests. For each DP target,
a bounded one-way greedy planner considers the same movable inventory and limits
as the other strategies: largest exclusively releasable source pages first, then
smaller destination allocation and request ID. It stops once admission is safe.
Pinned/shared source owners, conservative destination pages and copy-before-release
peaks are checked with the common physical-owner replay and real allocator/echo
checks. Greedy failure is not proof of actual allocator or ILP infeasibility.

Uniformly draw from targets with a greedy safety certificate. Do not minimize
attention cost, migration cost or an ILP objective to pick the rank. No MILP solve
is called. Derive each draw from SHA256 of serving seed/repeat/request ID, not
wall-clock time, scheduling order or future arrivals. Persist both feasible targets
and the chosen target; before/after-pause revalidation cannot silently reoptimize
the choice. Failed safety checks queue rather than discard the request.

There is no release-triggered or performance-triggered return. A later capacity-
blocked arrival may still trigger another necessary migration, even in the opposite
direction; this does not create a proactive return or forbid useful future admission.
The baseline is not deliberately slowed and uses identical pause32, transport,
request eligibility, candidate/move limits and safety checks.

## Prospective comparison

First run the existing closed-population N2/N9 fixtures with random_no_return,
kv_only_headroom and kv_only_workaware to validate real baseline migration and
absence of return. They are auxiliary mechanism checks, not Poisson trace benefit.
Then replay all six strategies on exactly the frozen2590914800-second trace:
no_move, no_move_fair, random_no_return, kv_only, kv_only_headroom, kv_only_workaware.
Conv.030863253171591748RPS and Code.0005RPS,142Conv/2Code marks, arrivals and all
prompt/output lengths are unchanged. Preserve twoA800/BF16/128K/24GiB virtual pools/
10240MiB controlled mapped budget perGPU/allocator.98/default sparse graph/pause32.
Reuse the independently verified population-profile calibration; do not deploy an
uncalibrated cost model, truncate100K input or move Code arrival times to manufacture
an opportunity. This comparison adds a baseline; it does not claim1.2 was achieved.

Report Conv, Code and combined avg/P95/P99TTFT, Conv request-averageTPOT, adverse
CodeTPOT, both full and600..4800windows. Keep actual Code waits, physical pressure,
aggregate-enough/local-blocked witnesses, all pauses/copies and every failed attempt.
Decompose workaware vs random into admission changes, release-time return, saved
decode work and paid control; any positive result from identical admission actions
must not be attributed to an admission ILP improvement. TwoCode marks only support
descriptive interpolated tails. A single paired trace or lucky random draw cannot
establish robustP99 or general50% benefit; retain negative results and require
paired repeat/seed confirmation before a general claim.

Archive all raw files with SHA256, independently validate tokens/four graph ranks/
owner transitions/budgets/random draws/no return and recompute metrics. Inspect
domain-guard deferrals and actual pause/copy-vs-decode tradeoffs before changing the
general algorithm. No policy-specific rates, shortened requests, hidden baseline
delay, selected Code arrivals or discarded requests are allowed. Continue300s
queries through terminal/archive; failed jobs are diagnosed before any new request.
