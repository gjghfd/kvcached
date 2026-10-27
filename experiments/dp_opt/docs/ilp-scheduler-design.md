# Rank-enumerated, cross-model KV-headroom ILP

The cost-amortized `kv_only_amortized` recovery variant adds an activation cost,
past-service migration budget, and displacement-release event gate. Its measured
motivation, equations and limitations are in `amortized-rebalance-protocol.md`.

Release-triggered rebalancing now has a separate no-arrival mode and
`kv_only_rebalance` variant. Its objective, fixed-cost gate, fresh post-boundary
solve and validation limits are in [release-rebalance-protocol.md](release-rebalance-protocol.md).
This is a local implementation, not yet a new GPU benefit result.

Status (2026-10-23): the CPU algebra reference and native vLLM+kvcached live
prototype both exist. Long-KV admission/migration benefit has been measured with
limitations; the first 32K Azure-Poisson four-policy trace has a verified negative
result. The 100K SWE-smith+AzureConv A800 four-policy trace is complete: the
performance-aware ILP has a negative net result, while capacity-first placement
has measured tail/SLO tradeoffs rather than across-the-board improvement.
The user accepts capacity-first as a primary technique candidate and requests
both independent and timeout-fair no-migration baselines; their protocol is in
`queue-baselines-protocol.md`, with no new fair-baseline GPU result yet. Current evidence belongs to
`../reports/benefit-long-kv-results.md`, `../reports/trace-benefit-results.md` and
`../reports/agentic-trace-results.md`; the formulation below remains a surrogate.
Main serving baseline is CUDA graph. Job 229748 supplies endpoint calibration;
jobs 230346/231467 reject the single affine approximation on sparse graphs.
Job231467's complete dense-capture ablation also fails the declared pointwise
error threshold. Protocol is in `curve-protocol.md`; detailed results belong
to `../reports/curve-results.md`.

Per the 2026-10-19 user decision, subsequent scheduler calibration and placement
experiments retain sparse captures `1,8,16,32,64,128,256`. Dense results are
historical ablation evidence only; do not use their costs for the sparse runtime.
The concrete sparse algorithm and cross-model tradeoff are in
[sparse-scheduler.md](sparse-scheduler.md).

## 1. Performance relationship and its measured limits

For two attention-DP ranks of model m, at a fixed post-admission decode
population N_m = n_m0 + n_m1 > 0, define

    d_m = |n_m0 - n_m1|,       I_m = d_m / N_m.

The initial endpoint-calibrated affine **cadence surrogate** was

    T_hat_m(I_m; N_m) = T0_m(N_m) * (1 + alpha_m(N_m) * I_m).
    alpha_N = mean_orientation(mean_repeat(T_extreme / T_balanced_paired)) - 1.

Each repeat uses the mean of its balanced-start/end cadence as its reference.
The two hot-rank orientations have equal weight. This is an interpolation
assumption between observed endpoints, not measured intermediate linearity.

For fixed N one output token/request/step, G = N/T, so

    G_hat / G0 = 1 / (1 + alpha * I).
    loss_hat  = alpha * I / (1 + alpha * I).

Consequently cadence excess and throughput loss cannot both be affine in I.
An alternative direct-loss proxy `loss_hat = beta*I`, with
`beta=alpha/(1+alpha)`, matches endpoints but makes different intermediate
predictions. Do not optimize one while interpreting it as the other.

| N | T0 (ms) | alpha | T0*alpha (ms) | Extreme loss implied by mean ratio |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 15.4413 | 0.408135 | 6.3021 | 28.9841% |
| 128 | 24.5714 | 0.476812 | 11.7160 | 32.2866% |
| 256 | 32.8764 | 0.624374 | 20.5272 | 38.4378% |

Values are recomputed from archived graph results, not pooled with eager.
The mean-ratio loss in this table is not the average of each repeat's loss.
For N=128, the provisional formula in milliseconds is
`T_hat = 24.5714 + 11.7160*I`; for N=256 it is
`T_hat = 32.8764 + 20.5272*I`. There is no supported universal slope.

The sparse capture sizes are 1/8/16/32/64/128/256. For N=128, a layout 80:48
rounds both ranks to graph size 128, just like 128:0; 64:64 uses graph size 64.
Thus the serving curve may be a staircase, with remaining variation inside
a capture bin. An empty rank still runs dummy forward/EP, not an idle GPU.
The supplement measures I=0/.25/.5/.75/1 in both orientations under the sparse
main configuration and a separately labelled dense-capture ablation.

The raw-verified sparse curves do **not** accept that primary affine
surrogate: maximum mean slowdown residual exceeds the declared 0.05
threshold in both runs. Their endpoint alpha remains close to job 229748,
but endpoint agreement is not intermediate linearity. For this configuration use
the layout/capture-aware objective in section 7, not an asserted linear loss.
`R=1+a*I+b*(p/p0-1)` is a post-hoc descriptive alternative; the current231467
sparse profile has a=0.305118,b=0.184904,T0=24.698862ms,p0=64. It is not causal
attention attribution, independent primary-test validation, or a calibrated
concurrent two-model cost.

Do not use a single-model coefficient as a validated concurrent A/B cost.
Concurrent services share GPU compute/bandwidth; aligned and opposite hot
ranks can differ even at identical I_A/I_B. Profile concurrent orientations,
N, context lengths, graph bins and device type before claiming predictive
placement quality. This design's separable objective is a first surrogate.

## 2. Admission event and enumeration

The opt-in implementation is documented in `live-placement.md`; job233056
verifies small GPU migration/admission, not cross-tenant handoff or speedup.
It uses the conservative source-page owner
variant, not unrestricted exact packing or a production HTTP scheduler.

Models A and B stay resident across the same two GPUs; expert shards, EP
groups, and attention-TP=1 stay fixed. A new request j for A arrives. Enumerate
its eligible **attention/KV-home rank** g in {0,1}, not which EP ranks to turn
off. For each g, solve a separate mixed-integer linear program over existing
requests of both models; select the best feasible candidate globally.

Use one consistent snapshot/epoch across candidates. It includes request
ownership, current active decode population, current KV pages and references,
per-pool mapped/free blocks, physical headroom, reservations, graph profile,
eligible destinations, and migration bandwidth/pause estimates. A request
whose KV is resident but not decoding contributes memory but zero decode load.
The count-calibrated objective below assumes all modeled requests are active
decode requests and no completions during the short optimization horizon.

Reserve the new request's prompt KV plus a declared growth budget and safety
margin. Existing requests also need growth reservations. Oracle output length,
maximum-token budget and estimated length are distinct experiment settings.
N_A includes the arriving request; N_B is unchanged. Coefficients for each
candidate's post-admission N are fixed during optimization.

If any no-migration candidate fits, evaluate its placement cost too. Zero
migration is allowed, not mandatory: voluntary rebalancing must repay its
copy/pause cost over the same horizon. Otherwise queue/retry if every candidate
is infeasible; do not fake capacity or drop B silently.

## 3. Assignment and imbalance constraints

Existing request i has model m(i), current home s_i and eligible ranks E_i.
Pinned/in-flight/cooldown/nonmigratable requests have E_i={s_i}. Define binaries
`x_ir` for its final home, and impose

    sum_r x_ir = 1;       x_ir = 0 for r not in E_i.
    z_i = 1 - x_i,s_i.
    n_mr = sum_{i:m(i)=m} x_ir + 1[m=A and r=g].
    d_m >= n_m0 - n_m1;   d_m >= n_m1 - n_m0;   0 <= d_m <= N_m.

With a positive imbalance coefficient in the objective, d_m is the absolute
count difference at optimum. For general R>2 a max-load surplus can be
linearized with `h_m >= n_mr` and `I_m=(R*h_m-N_m)/((R-1)*N_m)`, but the
two-rank calibration is not evidence for that extension.

## 4. Dimensionally defined objective and SLOs

Let H_m be the forecast number of decode steps for model m after admission,
w_m a per-request delay weight, and P_ir a measured/predicted migration cost
in the same weighted request-seconds (affected requests * pause, plus declared
admission/copy penalties). Dropping placement-independent balanced constants,
minimize

    J_g = sum_m c_m*d_m + sum_i sum_{r != s_i} P_ir*x_ir
    c_m = w_m * H_m * T0_m(N_m) * alpha_m(N_m).

This is the special affine-cost form, not the default claim after its sparse
fit rejection. With layout-cost table C_mv (section7), the current reference
instead minimizes `sum_m sum_v C_mv*h_mv + sum_i,r P_ir*x_ir`, where
`C_mv=w_m*H_m*N_m*(T_profile_m(v,N_m-v)-T0_m)`. Baseline constants still cancel
between candidate target ranks; no nonlinear products of decision variables
are introduced. Capture-aware costs and per-request memory remain separate.

This follows from `sum_m w_m*H_m*N_m*(T_hat_m-T0_m)`; the N_m denominator in
I_m cancels. It minimizes predicted aggregate decode delay, not raw bytes or
an unweighted sum of unlike percentages. H_m must be explicit: minimizing
one-step slowdown may choose a different plan from amortizing a migration
over hundreds of tokens. Estimate P using allocated topology/bandwidth; a
constant-per-byte transfer term does not establish actual migration latency.

If all tenants are paused for each serial copy, P must charge every affected
tenant/request plus j's admission waiting, not just the migrating request.
If unaffected tenants continue, account for copy/compute interference instead.
These are protocol-specific coefficients, not a free arbitrary tradeoff.

Require predicted `T_hat_m <= TPOT_budget_m` and per-request pause/admission
budgets, a move limit, cooldown and a B fairness constraint where applicable.
An average-cadence fit is not a P99 SLO certificate; use calibrated upper
envelopes or validate actual tails separately. Use copy bytes/move count as
secondary tie-breakers; record all weights and solver gaps. If transitional
layouts can serve for a long time, include their costs or freeze tenants and
charge the pause. The final-state objective alone does not price them.

## 5. Actual KV capacity, not logical token bytes

Physical GPU k can host attention rank r of both models via mapping pi(m,r).
Let C_k be its usable KV budget after fixed weights, graph/workspace peaks,
other non-KV allocations and an explicit guard. Let M0_k be currently mapped
physical KV backing and reservations. Sum all co-resident pools on GPU k;
do not compare A and B independently against the whole GPU budget.

For the current homogeneous Qwen BF16 setup, logical KV is 96 KiB/token and
a 16-token composite block is 1.5 MiB. A physical page bundle is accounted as
`page_size * num_layers * num_kv_buffers` by
`kvcached/kv_cache_manager.py::_memory_limit_page_bundle_bytes`; the observed
unified K/V setup uses 96 MiB backing bundles. These units are not equal.
Use `_page_capacity` and actual block ranges at page boundaries rather than
assuming every page has the same floor-divided block count.

For each mapped source page bundle p, let O_p be its request owners, a_p its
actual physical bytes and b_p a fixed non-releasable-reference flag (pinned
prefix/cache reference, nonmovable block, etc.). Let y_p mean it is released:

    y_p <= 1-b_p;         y_p <= z_i for each i in O_p;
    y_p >= sum_{i in O_p} z_i - |O_p| + 1 - b_p.

This gives credit only when all owners leave and no pinned references remain.
Empty pages can be immediately trimmed if the allocator confirms them safe.
The flag also needs stream/lifecycle safety: moving every request does not
make an asynchronous in-flight unmap instantly reusable.

At each destination pool (m,r), imported requests plus reserved growth/new j
need B_mr blocks. Let F_mr be currently usable free blocks in retained mapped
pages, excluding reserved/in-flight and pages chosen for release. Allocate
new page bundles using integer q_mr (or per-page binaries with exact capacities).
For equal-sized canonical pages of capacity K_mr and bundle bytes a_mr:

    B_mr <= F_mr + K_mr*q_mr;      q_mr nonnegative integer.
    M0_k - sum_{p on k} a_p*y_p + sum_{(m,r):pi(m,r)=k} a_mr*q_mr <= C_k.

If retained-page free capacity is variable, linearize F from page occupancy
and y; never simultaneously credit a released page and reuse its free blocks.
A safe initial implementation may ignore newly freed blocks inside retained
pages and use only free blocks reserved in pages never released, but it is
conservative and not the unrestricted physical optimum. Exact block packing
requires block-to-page assignments/capacities and shared-prefix ownership.
Destination allocations from different process pools are not interchangeable
unless the physical allocator/capacity-transfer protocol actually supports it.

## 6. Executable migration order: copy before source release

A final placement can satisfy capacity and still be impossible to execute.
Include at most L serial migration stages. Binary u_irt means i moves from
s_i to r != s_i at stage t; define cumulative movement U_ir,t. Then

    sum_t u_irt = x_ir for r != s_i;
    sum_{i,r != s_i} u_irt <= 1;
    U_ir,t = sum_{h<=t} u_irh.

Each request moves at most once; L bounds the permitted move count. Stage
labels may have gaps, with no assumed latency for empty stages. Define
`y_p,t` by the page-release constraints using `sum_r U_ir,t` in place of z_i.
Let Q_mr,t count cumulative newly allocated destination bundles through t,
be nondecreasing, and meet cumulative imported-block reservations using U_ir,t.
No destination page reclamation within this short transaction is assumed.

During the copy at stage t, source pages freed by **earlier** stages can be
used, but pages freed by the current move cannot:

    M0_k - sum_{p on k} a_p*y_p,t-1
          + sum_{(m,r):pi(m,r)=k} a_mr*Q_mr,t <= C_k.

Set y_p,0=0 before the first move, or include only acknowledged initial trims.
At the terminal admission stage, destination demand adds j and all remaining
growth reservations, and credit uses y_p,L. Reserve copy metadata/workspaces
and scratch pages too. A destination's logical free block quota and source's
safe actual unmap are separate constraints/acknowledgments.

The simpler request-private-page algebra reference replaces source-page
credits with each moved request's independent memory size. For a move at t,
destination size is charged for stages <=t, source size credited only <t.
That checks serial feasibility but is not a physical page-sharing solver.

For an unrestricted block-packing version (APC disabled), enumerate virtual
page slots p of each model/rank pool and candidate destination slots for each
imported composite block b. Binary a_bp,t assigns b to a slot when its request
moves at t; that placement persists until the end of this one-move transaction.
For every block b of i and destination r, require
`sum_{p in pool(m(i),r)} a_bp,t = u_irt`, with no assignment to i's source pool.
Original block occupancy of p
before copy t is the fixed original count minus blocks belonging to requests
moved in earlier stages. Let l_p,t indicate the post-stage mapped state and
h_p,t indicate the copy-peak mapped state. Impose, per slot and stage:

    original_blocks_remaining_before_t
      + sum_{b,h<=t} a_bp,h <= capacity_p * h_p,t;
    original_blocks_remaining_after_t
      + sum_{b,h<=t} a_bp,h <= capacity_p * l_p,t;
    h_p,t >= l_p,t-1;     h_p,t >= l_p,t;
    sum_{p on GPU k} bytes_p * h_p,t + other_reservations_k,t <= C_k.

For b, destination eligibility forbids placing its new copy in any block
still containing that request's own source at the current copy stage. Initial
l_p,0 is the actual mapped snapshot; fixed references are additional occupied
blocks. The count-capacity abstraction assumes no extra block-ID contiguity
requirement. If block identity, page-boundary groups or graph-address safety
restrict copies, assign a_bp,t to actual destination block slots instead,
enforcing at most one live/source/import occupant per slot during each peak.
All page bundles/virtual slots, exact capacities, KV buffer groups and
block-copy compatibility must be represented. Count reserved growth/new-j
blocks in a terminal admission stage with no source copy. A physical pool
allowing only grouped allocations adds its group equalities. This explicit
model allows reuse of blocks freed by earlier moves inside retained pages and
keeps a page mapped if imports use it; unlike forced source-page release, it
does not simultaneously credit that page's release. Mapping/unmapping takes
place only at acknowledged stage boundaries. Costs and state still assume
frozen growth/no completions over the transaction. It is much larger than the
conservative aggregate reference; candidate pruning changes its optimality
scope and must be reported.

Final-only ILP plus one post-hoc order check is not an exact executable
optimizer: if the best final plan deadlocks, an executable alternative may
exist. Use staged constraints, or add infeasible-plan/order cuts and re-solve.
Do not claim optimality after greedily rejecting one plan. Offload/scratch
buffering is a separate allowed move type with explicit capacity/cost, not a
silent escape from a deadlock constraint.

## 7. Capture-bin or joint-cost fallback remains MILP-compatible

If the affine fit fails, use a calibrated graph-bin/piecewise table instead
of forcing linearity. For DP=2, each possible final n_m0=v fixes n_m1=N_m-v,
I and the actual graph bin. Define binaries h_mv:

    sum_v h_mv=1;      n_m0=sum_v v*h_mv;
    T_hat_m=sum_v T_profile_m(v,N_m-v)*h_mv.

This is exact table lookup within the measured/profiled domain and still a
linear objective/constraints; costs for unmeasured counts need labelled
interpolation or a conservative envelope. To model A/B orientation coupling,
binary h_vw fixes both models' rank-0 populations and selects a jointly
profiled delay cost J_profile(v,w). It avoids a false separability assumption
but needs many more measurements. GPU mapping and context bins remain fixed
or must also index the table. A small fit residual is not proof of causality.

A compact, **post-hoc descriptive** alternative is
`R=1+a*I+b*(p/p0-1)`, where p is rounded max-rank graph size and p0 is the
balanced graph size. I is linearized via d, and p selected by capture-bin
binaries, so the surrogate stays MILP-compatible. This represents a linear
within-shape load term plus capture-step penalty, not an isolated attention
or communication-time decomposition. Dense captures making p/p0-1=I render
a/b collinear; they cannot identify those components independently. The
supplement's predeclared primary test remains `R=1+alpha*I`; do not replace a
failed primary fit with this post-hoc fit and claim the original test passed.

## 8. Control loop and optimality boundary

    snapshot <- consistent epoch, physical pages, budgets, profiles
    for g in eligible attention ranks of A:
        solve assignment + staged-capacity MILP for A and B
        retain feasible plan, incumbent, bound, status and MIP gap
    select minimum-cost feasible candidate
    reserve pages/headroom and validate snapshot epoch
    at legal decode boundary: copy -> verify -> switch request/KV ownership
    wait for safe source release/unmap acknowledgment after each move
    transfer usable physical capacity; admit j; maintain cooldown/reservation
    observe TTFT/TPOT/goodput and replan on stale state or forecast error

Every request's home update must include engine scheduling state, token
history/positions, KV block tables and ownership, cancellation/completion and
DP/EP synchronization; rewriting an HTTP rank header moves only future
requests, not live KV. Fixed graph-address compatibility must also be checked.
Freeze or conservatively reserve growth while solving/executing. Inference
continues changing state unless the controller establishes such a boundary.

Finite solver time with a feasible incumbent is usable, but not an optimal
solution. With proven-optimal solves for every eligible g, enumeration finds
the optimum **of this finite model**: calibrated cost, move eligibility,
page packing approximation, horizon, safety budget and copy protocol. It is
not a guarantee of best real goodput, future arrivals, or all possible offload
and multihop plans. Start with two ranks and a bounded movable set; greedy
cross-model headroom is a baseline, not the requested primary algorithm.

## 9. Synthetic checks and next execution gates

CPU-only verification in outer `evidence/ilp-design-20261018/` independently
compares the reference MILP with exhaustive assignment/order enumeration.
It uses request-private memory, arbitrary synthetic sizes and transfer costs;
it neither runs Qwen nor measures a scheduler speedup.

The maintained standalone reference is `../ilp_reference.py`, with input
`ilp-example.json` and tests `tests/test_dp_opt_ilp_reference.py`. It uses
optional SciPy/HiGHS, does not modify serving dependencies, and outputs every
target's objective, incumbent/dual bound/gap, selected order and memory peaks.
It rejects nonintegral or independently invalid incumbents. Run from repo root:

```bash
python3 experiments/dp_opt/ilp_reference.py experiments/dp_opt/docs/ilp-example.json --output <result.json>
```

An optional `performance_cost_tables_s` field supplies two complete arrays:
model m's array entry v is its predicted performance cost for final rank-zero
population v (length N_m+1, including the new A request). Table-selection
binaries replace the affine d objective while the serial copy/capacity model
stays unchanged. Costs have the same weighted request-seconds units as move
costs. Reject incomplete/nonfinite tables; orientation-dependent tables are
allowed. This is a solver over provided costs, not direct measurement of every
count or proof that single-model costs add under concurrent execution.
Alternatively, `joint_performance_costs_s[v][w]` provides complete joint costs
for final A/B rank-zero populations v/w. Selection binaries use only each
candidate's reachable count ranges, preserving every feasible layout. Separate
and joint tables are mutually exclusive. Joint costs are supplied inputs, not
automatically derived concurrent-model calibration.
Optional `admission_cost_s[g]` adds a target-dependent constant in the same
units to the candidate objective and lower bound. Returned plans split
performance, migration and admission costs; stage-dependent prefill or pause
interference requires a richer measured model.
`../calibrated_cost.py` constructs explicitly labelled fixed-N surrogates
from raw-verified fits; it refuses the failed primary affine fit, and never
labels interpolated counts or concurrent tenants as measured.
Its `--snapshot` option attaches one fit to both models only if post-admission
populations and sparse capture sizes match. `attach_sparse_costs` accepts
model-specific fits/horizons; runtime context and device compatibility still
need caller validation. The original affine example is historical algebra;
use `sparse-admission-example.json` for the current baseline.

- A counts 64:63, B 64:64; usable KV caps 11/11 GiB, occupied 8/8 GiB;
  new A needs 6 GiB. Movable A0/A1 use 1 GiB on their respective ranks,
  B0/B1 use 2 GiB. All other requests/background are pinned.
- Enumerating g=0 chooses A0+B0 -> rank1, final memory 11:11 and d_A=0,d_B=2;
  g=1 chooses A1+B1 -> rank0, d_A=2,d_B=2. With c_A=c_B=0.011715952 s and
  synthetic migration cost `2*(0.001+GiB/50)` per move, objectives are
  0.147431904 s and 0.170863809 s; g=0 is best. Either order is feasible here.
- Chain case: cap 8/8, occupied 7/7, new 2 GiB on rank0; outgoing A is 2 GiB,
  incoming B is 1 GiB. Only B rank1->0 then A rank0->1 can execute.
- Deadlock: cap 20/20, occupied 19/19, new 2 GiB; outgoing A is 3 GiB,
  incoming B is 2 GiB. Final-only placement fits, but neither copy can start;
  the staged MILP rejects it. All 28 small owner/pinned truth patterns check
  the exact page-release Boolean, separately from the private-memory solver.

Next gates: profile concurrent
orientations -> implement safe ownership migration/capacity handoff -> run
two-model arrivals with fragmentation, migration overhead and SLO-aware
goodput. The current design is not permission to claim those gates completed.

## 2026-10-26 measured tail boundary

The six-policy A800 trace 251533 now passes complete ownership, physical
capacity, staged-copy, token and past-only history verification. Headroom
restoration does not establish additional tail benefit: combined mean/P99
TTFT improve only 0.424%/0.358% over KV-only, while Conv P95 and all-request
completion SLO regress. See `../reports/headroom-tail-status.md` for every
policy, both tenants and the combined metrics, not just incoming Code.

Current Code reservation-to-first-token time is about 54.5 seconds; removing
its 1.42-second KV-only queue alone cannot deliver a 50% reduction. The worst
Conv arrival is optimistic whole-request bin-packing infeasible despite
aggregate logical bytes fitting, so a placement-only ILP cannot admit it.
The next `tail-next-stage-design.md` separates measured mixed-stage costs,
allocator-verified packing opportunities and tenant-selective transactions.
That extension is planned, not implemented or GPU-validated; the existing
safe transport still uses global pause, not true copy/inference overlap.
