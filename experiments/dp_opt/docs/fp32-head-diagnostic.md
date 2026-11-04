# FP32 logits-head correctness control

The six zero-copy controls of276017 reproduce background greedy divergence
solely by changing attention-DP co-residency. Conv0's first shared-prefix top-two
margin changes from0.25 to a tie. This suggests investigating head rounding,
but is not a proved numerical root cause and does not validate migration.

Use the installed vLLM's explicit `hf_overrides={"head_dtype":"float32"}`.
CUDA `torch.mm` emits FP32 logits from BF16 hidden states and BF16 head weights;
the model does not become FP32. Record the installed implementation SHA and
four workers' actual model, weight and effective head dtypes. Reject quantized
heads, different parallel layouts, changed graphs and batch-invariant flags.
This is a disclosed numerical setting change, not an unchanged-default claim.

Reuse the exact frozen trace: background P74/O16557 at0s and P149/O16314 at1s,
Code P101626/O64 at90s,900s window. Run the original six mirror-order static
no_move layouts with no pause/copy. Regenerate all full serial outputs under
the new head setting. Keep BF16/128K,DP2/EP2/attentionTP1,24GiB virtual pools,
10240MiB/GPU budget,utilization.98,32-step control and default sparse graphs.
Do not substitute short outputs, change arrivals or relax the greedy comparison.

If any static output differs, save the full diagnostic and original strict
error, end the allocation without migration/benefit phases and diagnose the
first common prefix. If every static token matches, continue in the same
allocation with one unchanged-trace no_move/random_no_return/kv_only_headroom
control. Apply the same head setting to all serial references and policies.
Independently verify complete outputs, actual copies, ownership, physical
budget and four-rank graphs. This small control is not a final benefit trace.

Preserve the old default-head outputs. Compare old/new serial outputs only as
a separate numerical observation; new-head layout consistency is checked
against new-head serial outputs. Workaware's144 default-head measurements need
revalidation/recalibration before new-head profitability claims. Correctness
must pass before resuming public-source rate selection, seven strategies and
held-out experiments. One A800 allocation, protected SHA sync,300s monitoring,
terminal raw/log SHA and independent reconstruction; no publication authorized.
