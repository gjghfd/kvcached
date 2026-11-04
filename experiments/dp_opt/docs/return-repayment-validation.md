# Pause-inclusive return repayment validation

This supplements, not replaces, full-request Conv TPOT and separate Conv/Code/
combined avg/P95/P99 TTFT. The old steady token-window diagnostic deliberately
excludes pause/settlement; it cannot alone establish that moving KV repays its
cost. No runtime, ILP, graph, trace or uncertainty threshold changes here.

## Matched work including control cost

For each Code-absent interval, identify the first committed Conv return. In the
treatment, choose the last common recorded Conv token ordinal delivered before
the return decision's `started_at`. Require that ordinal to fall after Code
completion in **both** workload clocks. Compare elapsed time from that ordinal
to later identical output ordinals, without removing planning, pause, copy,
resume or settlement time. Bound both spans by the next Code arrival or declared
trace end under their respective clocks.

For anchor ordinal `a` and prefix ordinal `n`, report
`saved(n) = (t_baseline(n)-t_baseline(a)) - (t_treatment(n)-t_treatment(a))`.
Report the first nonnegative observed prefix and the first prefix that remains
nonnegative through the remaining **observed** samples. A temporary positive
prefix is not sustained repayment; sparse cadence is not interpolated.
Retain the maximum observed debt, final duration difference, matched samples,
and every excluded span. Identical paired release/end windows from simultaneous
Code completions merge into one window with all associated Code IDs, preventing
double counting.

This is aligned-work evidence, not a randomized causal estimate: other request
populations and pre-anchor progress can differ. Subsequent controls in the
window also remain included. All primary whole-request metrics remain necessary.
TTFT that precedes a return cannot be attributed to that return.

## Independent execution gate

`analyze_return_repayment.py` requires a closed exit0 phase. Its CLI first
verifies the full raw-file SHA manifest and runs `verify_trace.py`, including
tokens, physical KV ownership, budgets, profiles and graph checks. It refuses
existing output files or outputs inside immutable archived raw. Module-level
`analyze()` is only for callers that have already performed that verification.

`long_conv_phase_analysis.py` reads the **existing main campaign's local poll**
every300s. It never queries Slurm, submits a GPU allocation or modifies runtime.
Only a phase before the observed current seed is eligible for early archiving;
the remote shell separately requires exit0, no symlinks and identical pre/post
SHA manifests around the compressed snapshot. Downloads and extracted files
are independently SHA-verified before trace verification and analysis. The
last seed uses the main campaign's complete verified terminal archive.

The observer computes all three baseline comparisons, recovery timing, steady
token windows and pause-inclusive repayment per closed seed. It preserves
failed attempts, retries transport errors after300s and stops on PAUSE, source
changes or failed verification. No failure triggers automatic GPU resubmission.

## Current deployment and evidence

Job268245 keeps its85 runtime/test files and six prior candidate/follow-up
files unchanged. The new five local analysis/test files are frozen in
`evidence/long-conv-return-probe-20261030/phase-analysis-source.sha256` while the
observer is active. Its local launch identity is in `phase-analysis-local.json`;
state and per-seed results live in `pause-inclusive-phase-analysis/`.

Independent audit initially finds17 passes and one genuine double-count failure;
the failure is retained in `return-repayment-independent-audit-r1.log`. The
paired-window merge fixes it without weakening tests or guards. Current67
targeted tests pass; the final full CPU regression passes607 in154.27s. Actual
GPU repayment results remain separately recorded, not inferred from CPU tests.

Seed20261030 is now independently closed:168raw SHA/12requests/four policies.
Matched pre-return token anchors retain all control time; the two remaining
Conv spans save217.410446 request-seconds vs random, with sustained observed
prefix repayment about9.415s after the anchors. Code and Conv TTFT both precede
return, so none of their TTFT differences is evidence of return benefit.
Full Conv TPOT and its degradation vs no_move remain reported alongside the
improvement vs random/headroom in `../reports/long-conv-return-probe-results.md`.
This is a first-seed closed mechanism probe, not final Poisson or tail significance.
