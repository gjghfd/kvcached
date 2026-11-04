# Default-backend greedy-token divergence diagnostic

Job272106 ends FAILED/1:0 at2026-10-31 16:19:43+08 after all144 cost cases and
the first native four-policy requests drain. The fatal check compares complete
greedy token sequences with serial references. Its257-file terminal SHA archive
is preserved; it is not an accepted benefit result.

The unchanged no_move baseline itself first differs from its serial Conv0
reference at zero-based token139. All Code64 outputs match their serial references.
NoMove has zero KV migrations, so the baseline divergence cannot be caused by a
KV transfer. This does not exonerate every migration or prove a numerical cause.
Archive/measure the complete first-divergence prefix before changing validation.

## Next GPU diagnostic

Use the identical frozen first native trace/bank, including original Conv
responses16557/16314, true Code101626+64 and the900s offered window. Run two
no_move repetitions, not a new favorable seed or shortened output. Preserve
two A800s, independent model residency, DP2/EP2/attentionTP1, BF16/128K,
24GiB virtual model/rank pools,10240MiB/GPU controlled budget, allocator.98,
default sparse graph captures/FULL_DECODE_ONLY/compile0 and pause32.

`batch_numerics_probe.py` adds requested top2 per-token logprobs to the existing
stream interface. This extra instrumentation is diagnostic only and its timing
must not substitute for an algorithm-benefit measurement. No batch-invariant
kernel mode is enabled and no migration policy runs.

At each first divergence, the preceding token IDs are identical. Report the
selected token, common top-candidate logprob differences and top2 margins for
that shared prefix. Also report repeat-to-repeat no_move differences. Missing
logprobs, dropped/shortened outputs, any migration or unrelated runtime error
fails the probe. Do not impose a posthoc tolerance or call a small margin a proof
that migration is correct.

The existing strict trace runner/validator is not weakened. The probe retains
its original error/summary, catches only its exact serial/paired-token failure
after every original request drains, then emits a separately named diagnostic
result. A COMPLETED diagnostic allocation means the probe ran, not that the
underlying native trace passed or that a public-data benefit is established.

Only after terminal SHA and independent logprob/source reconstruction should
we choose a broadly valid correctness treatment. Keep all old failures and
default-kernel results; do not use the failed performance numbers to select
a best setting, retune the held-out trace or relax robust return guards.
