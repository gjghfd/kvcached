# Same-shape migration correctness control

This is a prepared diagnostic, not a submitted allocation or an accepted benefit
result. Complete job272931's terminal SHA/logprob reconstruction first. Keep its
runtime frozen; do not submit another allocation while it is pending/running.

The unchanged first native frozen bank provides two original OpenThoughts3 Conv
responses and one true100K SWE-smith Code prompt. Both independent Qwen tenants
remain resident on the same two A800; only one request is active in each control.
Keep BF16/128K,24GiB virtual pools,10240MiB/GPU budget,allocator.98,EP2/DP2,
attentionTP1 and default sparse/FULL_DECODE_ONLY/compile0/pause32. Do not change
response lengths, enable batch-invariant kernels, or relax token validation.

For each Conv prompt and each starting DP rank, run three complete controls:

1. Uninterrupted single-request full greedy reference.
2. The same request paused at a mature default32 boundary, with no migration.
3. The same request paused at that boundary, moved to the other rank and moved
   back before any inference resumes.

Observe the predeclared mature boundary at at least1024 generated tokens; if
the original response ends before it is safely paused, fail rather than extend
the response. Each hop must use exact full-device bytewise echo, preserve unique
ownership and the serialized request bytes/reservation length, and satisfy the
controlled per-GPU budget including copy peak. Physical block IDs may change.
Any failure keeps the engines paused until the ordinary runner shutdown.

Compare complete output IDs with the same-source uninterrupted reference, with
no tolerance or discarded prefix. Both Code ranks additionally run the unchanged
100K+64 serial request; short Code decode is not extended to force a migration.
Cross-rank serial differences are separately reported, not silently treated as
a same-source roundtrip failure or as proof of rank independence.

The14 controls test state continuity with the same final active layout. They
cannot establish arbitrary mixed-batch migration correctness, eliminate the
original no_move batch divergence, or establish a throughput/TTFT benefit.
The exact request/KV evidence and full sequences are required for independent
terminal reconstruction before a subsequent correctness treatment is chosen.

The local implementation uses the existing pilot branch through an independent
adapter and sbatch, without modifying `live_trace.py`, scheduler, controller or
the currently deployed diagnostic. A new allocation requires protected sync,
source/input SHA checks and one-allocation/300s monitoring barriers.

The October31 continuation uses one three-hour A800 allocation. Only after all
14 controls and an independent full-phase reconstruction pass does the same
allocation run two unchanged native `no_move` traces without the diagnostic
`logprobs=2` instrumentation. Serial and paired full-token checks remain strict.
The state-control and plain-trace evidence roots are separate. Both phases must
retain the original trace/prompt-bank SHA. Terminal local reconstruction checks
every complete sequence, four-rank graph/topology, eight paused transactions,
serialized state, unique page backing and actual copy-peak budgets. No native
failure is reclassified, and neither phase is an algorithm-benefit experiment.

`roundtrip_state_campaign.py` requires the SHA-verified272931 parent, protects
the114-file deployed runtime, synchronizes only the new control/verifier files,
records a submission intent and ledger before its sole allocation, monitors
every300s and archives terminal raw before local verification. Failures stop
automatic submission. Equivalent solver optimizations are not installed here.

The first273701 attempt failed before the first pause: the new sbatch omitted
`DP_OPT_KV_TRANSPORT`, so the unchanged controller selected `host_file`. Four
complete serial controls exist, but no paused transaction or plain trace ran.
Its44 raw files are SHA-archived and independently diagnosed; the original
failed job and two rejected legacy archive bundles remain preserved.

The explicit transport retry keeps all14 controls and lengths, declares
`DP_OPT_KV_TRANSPORT=dp_echo`/32MiB chunks, and rejects a wrong transport before
engine initialization. Progress now follows the existing canonical archive
contract: state controls live in `pilot/` with an absolute root pointer to
`pilot/progress.json`; the later plain phase uses its absolute two-component
`plain-no-move/progress.json` pointer. The legacy archive implementation remains
unchanged. The local-only `archive_roundtrip_job.py` recovers the old relative
nested pointer using a two-path allowlist, never extracting symlinks or changing
raw data; it is not deployed as the future serving/archive path. A retry requires
the immutable273701 diagnosis and a distinct evidence root/submission ledger.
