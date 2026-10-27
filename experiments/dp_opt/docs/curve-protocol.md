# CUDA-graph intermediate-imbalance supplement

Pre-measurement protocol, 2026-10-18. Endpoint measurements from job 229748
calibrate a surrogate but cannot establish a linear imbalance relation.

Policy update, 2026-10-19: future runs use only the existing sparse capture
configuration. `motivation-curve.sbatch` no longer starts the dense service.
The two-profile protocol below describes completed jobs, not future defaults;
their results and raw evidence remain unchanged.

One allocation runs two sequential, independently started single-model services
on the same pair of allocated GPUs. Each uses Qwen3-30B-A3B-Instruct-2507 BF16,
DP=2/EP=2/attention-TP=1, AgRs, FA2, Triton MoE, FULL_DECODE_ONLY and compilation
mode 0. Other settings match graph job 229748: max length 4096, max sequences
256, max batched tokens 4096, 44 GiB virtual KV/rank, no DBO/EPLB/APC/async.

- Global population 128; input 1024/output 512 tokens.
- Layouts 64:64, 80:48, 96:32, 112:16, 128:0, and both hot-rank orientations.
- Three paired repeats: balanced-start, shuffled eight nonbalanced layouts,
  balanced-end; 30 measured trials per configuration.
- Preserve all validity, trimming, fixed-population, reclamation and untraced
  overhead checks from `motivation.py`. Observed graph padding must equal the
  first declared capture size no smaller than the maximum rank population.
- **Sparse/main:** capture sizes 1,8,16,32,64,128,256. All nonzero sampled
  imbalances round to graph size 128. Do not label this unpadded attention.
- **Dense/mechanism ablation:** 1,8,16,32,64,80,96,112,128,256. This intentionally
  changes capture configuration; it is not a replacement for the main baseline.
- Primary response: per-repeat cadence divided by that repeat's mean balanced
  endpoint cadence. Fit `R=1+alpha*I` with intercept fixed at one. Report each
  orientation, residuals, maximum absolute error, RMSE, and descriptive R².
  Compare endpoint-only calibration with the full-curve fit. Throughput loss is
  `1-1/R`, not interchangeable with cadence excess `R-1`.
- Predeclared usefulness check: <=5 percentage-point absolute slowdown residual
  at every measured layout. This is an approximation threshold, not proof of
  linearity, and three repeats do not justify a high-resolution response model.

No phase hooks or migration. This supplement does not calibrate concurrent
two-model performance, heterogeneous KV lengths, or a live placement benefit.
Keep profiles separate; results and raw hashes go into a new immutable archive.
