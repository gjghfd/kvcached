# Intermediate attention-DP imbalance curves

Updated 2026-10-19. Single-model CUDA-graph calibration only; no live migration
or concurrent-model placement benefit is measured here.

## Completed sparse profile, interrupted allocation 230346

Job 230346 ran on two A800-SXM4-80GB GPUs at `d1n41a13g02`. User sleep-pause
cancelled the allocation at 2026-10-19 00:22:08 CST, after the sparse profile
had passed all 30 measured trials and exited 0. Dense startup was interrupted;
do not label the whole allocation COMPLETED or use incomplete dense trials.

Qwen3-30B-A3B-Instruct-2507 BF16, DP2/EP2/attentionTP1, FA2, Triton MoE,
AgRs, FULL_DECODE_ONLY with compilation mode0. Capture sizes
1/8/16/32/64/128/256; N128,input1024/output512,44GiB virtualKV/rank,maxlen4096,
maxseq256,maxbatched4096; no DBO/EPLB/APC/async. Three paired repeats bracket
eight shuffled unbalanced layouts by balanced-start/end. Fixed stable decode
windows trim32 steps/end and require at least128 steps. Full protocol is
`../docs/curve-protocol.md`.

The following averages give equal weight to both hot-rank orientation mean
paired ratios R; loss is `1-1/R`. Balanced cadence is 25.204972 ms.

| I | Layouts | Per-rank replay size | Paired R | Decode throughput loss |
| ---: | --- | ---: | ---: | ---: |
| 0 | 64:64 | 64 | 1.000000 | reference |
| .25 | 80:48 / 48:80 | 128 | 1.256866 | 20.44% |
| .50 | 96:32 / 32:96 | 128 | 1.308701 | 23.59% |
| .75 | 112:16 / 16:112 | 128 | 1.444245 | 30.76% |
| 1 | 128:0 / 0:128 | 128 | 1.476630 | 32.28% |

Primary fit `R=1+alpha*I`: alpha=0.548470, repeat-slope t95 CI
[0.543878,0.553061]; RMSE across repeat/layout ratios=0.074343,
descriptive R²=0.345266, maximum layout-mean absolute residual=0.124465.
This **fails** the predeclared 0.05 approximation threshold. Endpoint-only
alpha=0.476630 reproduces old endpoint calibration but underpredicts I=.25
by up to0.142424 in slowdown. Hence endpoint agreement does not justify a
linear intermediate performance relation.

Post-hoc descriptive shape-aware fit:

    R_hat = 1 + 0.317934*I + 0.172902*(p/p0 - 1).
    T_hat = T0*R_hat;    loss_hat = 1 - 1/R_hat.

p0=64; p is the rounded max-rank graph size. Maximum mean residual0.035668,
RMSE of layout means0.021974. This model explicitly includes the graph-size
jump; it does not pass the failed primary test retroactively. It is not a
causal attention/communication-time decomposition, a P99 guarantee, or
validated outside N128/context/capture/hardware. Unmeasured intermediate counts
are predictions; one empty attention rank still participates in dummy/EP.

Raw verification: 30 trials,3840 requests,1,966,080 exact output tokens;
minimum412 stable steps/rank, maximum stable KV use45.0544%; no stable
waiting/preemption, all traces/paired summaries reproduce, every trial
restores96MiB/rank idle mapped backing. `sparse.tar.gz` SHA256:
`7a113d0986a8a31ea3c03a6bb86a344707a8238223ef08d7c9c0f40df713fb87`.
Raw/fits live in outer `evidence/kvcached-curve-230346/`.

## Same-allocation sparse/dense repetition 231467

Submitted 2026-10-19 08:48:50 CST, started08:49:20 on `d1n41a12g03`, two A800
SXM4-80GB GPUs, finished09:18:14 **COMPLETED/0:0**. Both profiles passed all
30 trials; their raw requests/traces independently reproduce every window and
paired summary. Dense adds80/96/112 captures, intentionally changing graph
shapes; all other frozen settings/protocol are unchanged.

| I | Layouts | Sparse replay p | Sparse R / loss | Dense replay p | Dense R / loss |
| ---: | --- | ---: | --- | ---: | --- |
| 0 | 64:64 | 64 | 1 / reference | 64 | 1 / reference |
| .25 | 80:48 / 48:80 | 128 | 1.268443 / 21.16% | 80 | 1.193161 / 16.19% |
| .50 | 96:32 / 32:96 | 128 | 1.308273 / 23.56% | 96 | 1.252848 / 20.18% |
| .75 | 112:16 / 16:112 | 128 | 1.450341 / 31.05% | 112 | 1.414749 / 29.32% |
| 1 | 128:0 / 0:128 | 128 | 1.475352 / 32.22% | 128 | 1.474744 / 32.19% |

Ratios average both orientation paired means, never step-level pseudorepeats.
Baseline cadence: sparse24.698862ms, dense24.743569ms. The two configurations
are sequential in one allocation, not simultaneous randomized A/B controls;
same-device repetition removes a device confound but does not remove all
temporal drift. Similar endpoints but different intermediate ratios show
that I alone is insufficient for the tested main serving configuration.

| Primary fit R=1+alpha*I | Sparse | Dense |
| --- | ---: | ---: |
| alpha | .551656 | .512278 |
| Three-repeat slope t95 CI | [.523343,.579969] | [.508573,.515983] |
| RMSE across repeat/layout ratios | .080172 | .041336 |
| Descriptive R² | .196808 | .871033 |
| Max layout-mean absolute slowdown residual | .133071 | .073912 |
| Predeclared <=.05 residual check | **fails** | **fails** |

Dense is closer to affine but still fails; a high R² or narrow fitted-slope
CI does not override the pointwise threshold or validate the wrong model.
Dense makes p/p0-1=I on these sampled layouts, so separate shape/load slopes
a,b are collinear and cannot be identified from this profile.

Sparse's current post-hoc descriptive relationship is

    R_hat = 1 + 0.305118*I + 0.184904*(p/64 - 1).
    T_hat_ms = 24.698862*R_hat;       loss_hat = 1 - 1/R_hat.

Its max mean residual is0.038036 and RMSE of means0.024885. For this sparse
profile at fixed N128, p=64 at balance and p=128 at every other rank count;
the model has a step plus a within-bin slope. Predictions immediately next to
balance have not been measured and must not be presented as observations.
The irregular change between I=.5/.75 remains unexplained kernel scaling,
not identified attention share or HBM saturation.

As an additional repeat check, coefficients frozen from230346's sparse fit
predict231467's sparse paired ratios (using new balanced normalization) with
max mean residual0.040426. This is a cross-allocation check of the post-hoc
surrogate, not a predeclared second primary test or validation of concurrent
two-model costs. All orientation ratios/CIs and fit residuals are preserved
in `sparse-fit.json`, `dense-fit.json`, and `sparse-cross-run-validation.json`.

Validation totals: **60 trials / 7680 requests / 3,932,160 output tokens**.
Min stable steps per rank411(sparse)/412(dense), max stable KV45.0544%/45.2409%,
no waiting/preemption, all idle backing restored96MiB/rank. The allocation
passed241 regressions before each service; FA2 and7/10 FULL graph captures are
recorded. Capture memory0.18/0.24GiB is startup-reported graph memory, not total
GPU peak. Services exited and the two-GPU allocation was released.

Archive: outer `evidence/kvcached-curve-231467/results.tar.gz`, SHA256
`6d0848ff93bad47324ebbf3b8eb4104859955bcc4885cfe40933a9097652ac42`
matches remote. It includes both complete profiles, hardware/source hashes,
client/trial/trace JSON, GPU timelines, service/allocation logs and accounting.

## Scheduling consequence

Use a calibrated layout/capture table selected with binary variables, or an
explicitly labelled shape-aware surrogate, instead of assuming one global
linear loss coefficient. The rank-enumerated ILP reference now accepts
complete orientation-dependent layout-cost arrays with the same serial
copy-peak constraints. Joint A/B contention calibration and physical-page
migration remain required; these single-model coefficients are provisional
for a synthetic scheduling example only.

For dense or other profiles without an accepted parametric fit, use explicit
layout profiles/interpolation with stated uncertainty, not the sparse a,b.
The standalone builder intentionally refuses failed affine fits and separately
unidentifiable shape fits. It does not automatically extrapolate across N,
contexts, model concurrency, devices or capture configurations.
