# ILP calibration and execution status

Updated 2026-10-20, Asia/Shanghai. Formulas/constraints live in
`../docs/ilp-scheduler-design.md`; measured settings/results in `curve-results.md`.

## Execution checkpoint

Kvcached job **231467** completed **COMPLETED/0:0**, 2026-10-19
08:49:20–09:18:14 CST on two A800-SXM4-80GB GPUs at `d1n41a12g03`.
Both sparse/main and dense/capture-ablation profiles passed 30 trials each,
with all raw windows, request lengths, paired summaries and reclamation
independently verified. Allocation and services ended; no new job remains.

Earlier job230346 was cancelled for sleep after its complete sparse profile;
its incomplete dense startup is not a result. Aegaeon job223200 was separately
cancelled on explicit user instruction at00:24:11 and is not restarted. Only
kvcached research is resumed. Monitoring has stopped after completion.

## Supported performance model

The original endpoint surrogate `T=T0*(1+alpha*I)` implies throughput loss
`alpha*I/(1+alpha*I)`, not linear loss. The completed sparse curves and dense
control **all reject** the primary affine approximation under the frozen
pointwise residual criterion. Do not reuse its endpoint slope as a validated
universal relation.

Use a calibrated layout/capture table with binary selection. For the current
sparse profile, the post-hoc descriptive surrogate is
`R=1+0.305118*I+0.184904*(p/64-1)` and `T0=24.698862ms`.
It models a capture-size jump plus a within-bin slope, not kernel-time shares;
details, residuals and cross-run check live only in `curve-results.md`.
Unmeasured counts are model predictions and concurrent A/B coefficients are
not calibrated. Dense cannot independently identify the shape/load slopes.

## Maintained CPU reference

`../ilp_reference.py` enumerates the arriving A request's attention-DP home,
then solves a serial staged assignment MILP over movable A/B requests for each
candidate. It enforces copy-before-source-release peaks and terminal admission
capacity. It supports an affine example or complete orientation-dependent
`performance_cost_tables_s`; a nonlinear supplied cost table is selected by
binary variables, so objective and constraints remain linear.

`../calibrated_cost.py` constructs labelled fixed-N cost surrogates from
raw-verified fits, in weighted request-seconds over an explicit horizon. It
refuses rejected affine fits, unidentifiable shape fits and invalid numbers.
The builder-to-solver path is tested. SciPy/HiGHS is optional for these CPU
references, not added to serving dependencies.

The 2026-10-19 sparse-only algorithm is in `../docs/sparse-scheduler.md`.
The reference now accepts joint A/B layout costs, distinguishing aligned and
opposite hot ranks, plus target-dependent admission cost. Returned costs are
split into performance/migration/admission, with constant-adjusted lower
bounds and gaps. Joint calibration itself is not measured. Sparse snapshot
attachment rejects dense captures and post-admission population mismatch;
table mode no longer requires the rejected affine coefficient.
`../docs/sparse-admission-example.json` demonstrates moving B to preserve
A's balanced graph while making admission feasible, under labelled synthetic
private memory and single-model cost assumptions.

Tests independently enumerate all small assignments/orders, including both
target ranks, required reverse-chain migration, a final-state-feasible copy
deadlock, no-move admission, move limits, invalid inputs and timeout status.
Also checked: 24 seeded affine cases and12 seeded nonlinear-table cases for
both ranks. Time-limited incumbents never become proven global optima; no
incumbent at timeout is not proof of infeasibility.

The planner supports request-private memory and optional source-page ownership
release with conservative destination-page charges, not exact shared-page packing.
The experimental opt-in ownership/KV transaction passed job233056's small
GPU migration and ILP-admission checks; see `live-placement-status.md`.
Job233056 only released A pages; subsequent cross-tenant capacity and end-to-end
benefit comparisons are recorded in `allocator-pressure-status.md` and
`benefit-status.md`. Single-model calibration still does not establish a joint
performance model or general placement optimality.
Outer `evidence/ilp-design-20261018/`
contains synthetic solver/exhaustive results and28 page-owner Boolean cases;
they are not Qwen migration measurements.

## Validation and evidence

Local targeted client/fitter/raw-verifier/reference/builder suite: **96 tests
pass**, known `asyncio_mode` warning only. Job231467 ran241 existing regressions
before each service. The GPU supplement verifies **60 trials / 7680 requests /
3,932,160 output tokens**; profiles and negative fit results remain separate.

Outer `evidence/kvcached-curve-231467/` owns the full raw archive, logs,
accounting, SHA256, fits and frozen cross-run prediction check. The new verifier
also reproduces historical N128's12 trials/1536 requests/786432 output tokens.
Its peak KV statistic is stable-window-only, not all-phase sampled maximum.
Outer `evidence/sparse-ilp-design-20261019/` holds the new calibrated synthetic
snapshot, both target plans, own-model/no-move controls and regression log.
New checks include12 seeded joint-table instances against exhaustive orders,
admission constants/time-limited bounds, horizon amortization and sparse/N
calibration guards. They do not establish live serving benefit.

```bash
python3 experiments/dp_opt/fit_imbalance.py <results.json> --output <fit.json>
python3 experiments/dp_opt/calibrated_cost.py <sparse-fit.json> --output <cost-table.json>
python3 experiments/dp_opt/ilp_reference.py experiments/dp_opt/docs/ilp-example.json --output <plan.json>
```

## Remaining gates

1. Profile concurrent A/B orientations and context dependence.
2. Extend the verified early-decode transaction to mature KV, demonstrate
   B-to-A physical capacity handoff and validate block-page packing.
3. Run arrival/headroom experiments against no-move, independent-model and
   global greedy baselines, including copy cost, TTFT/TPOT and goodput.

No commit/push is performed for this design/supplement; the prior requested
publication remains separate. No live scheduler benefit is claimed.
