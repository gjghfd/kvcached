# Conv pressure target 1.2 (2026-10-27)

The user authorizes raising Conv RPS if the no-move baseline does not queue.
Completed job257484 has Code waits9.3625s and0.1989s; the second arrival finds
an empty attention home. Preserve that result and both original Code marks.

## Physical definition and frozen first-order rate

`rho=(mean Conv mapped MiB per GPU + single Code pool MiB)/controlled GPU KV budget MiB`.
The denominator is10240MiB, not80GB physical device capacity or24GiB virtual
capacity. Deduct the idle Code pool96MiB from each total mapped observation
before measuring Conv. Code's complete mapped pool is9600MiB; its incremental
allocation from an empty96MiB pool is9504MiB. The definition adds the complete
Code pool to Conv, so it does not count its idle page twice.

Job257484 no_move uses600..4800s, Code absent and stable admission epochs,
left-hold time weighting:4074.158097s coverage, Conv1739.280628MiB/GPU,
rho1.107351624. Fixed Conv idle pool is96MiB.

Target Conv=`1.2*10240-9600=2688MiB/GPU`. Preserve the fixed page:

`lambda_new=lambda_old*(2688-96)/(1739.2806284373632-96)`

`lambda_old=.01956673845194241` yields
**lambda_new=.030863253171591748RPS** (+57.733%). This is not simply1.2/1.1
times the old RPS. Service-time feedback and discrete packing require real
GPU measurement; do not claim actual rho1.2 from the rate formula alone.
The nominal aggregate ratio at the target is.7359375, but an average is not
a per-arrival packing or actual allocator feasibility proof.

## Paired experiment and decision rules

Only Conv rate changes. Rebuild from the same seed20261024 and original input
banks; independent Code.0005RPS remains byte-identical at2463.4417805002327
and2663.1432231654926s. Conv's original sampled mark prefix is unchanged,
arrival times scale by old/new rate, and additional arrivals use the same
Poisson stream. No synchronized bootstrap, selected arrival times, truncation,
discarded requests or replacement of the old trace.

Keep twoA800, two independent Qwen3-30B-A3B tenants with DP2/EP2/attentionTP1,
BF16/131072/24GiB virtual per pool/10240MiB controlled per GPU/allocator.98,
sparse `[1,8,16,32,64,128,256]`,FULL_DECODE_ONLY,compile0,pause32,4096 batched
tokens. Code true100K+64 and Azure-input synthetic8192-output Conv unchanged.
Reuse the already verified280-case calibration and current population-profile
ILP; do not deploy the diagnostic actual-attention-count model. Skip repeated
N2/N9 mechanism phases, not the original five serving policies.

One allocation runs no_move,no_move_fair,kv_only,kv_only_headroom,
kv_only_workaware on identical4800s traces; all offered requests must drain.
Fair threshold30s and request timeout1600s remain unchanged.
The no_move baseline is the same full-reservation custom scheduler, not an
unmodified native-vLLM dynamic-growth scheduler. No policy discards blocked work.
Report full and
600..4800 windows: Conv/Code/combined avg/P95/P99TTFT, Conv request-average
TPOT, adverse CodeTPOT, queue/prefill/control/copy costs. Two Code samples
only permit descriptive tails, not statistically reliable P99 claims.

Prespecified descriptive target tolerance is1.2±.02. If measured pressure
misses it, retain every policy/result, diagnose service feedback and recalibrate
from physical pressure, not favorable TTFT. If it hits but Code still finds an
empty rank, report that finite Poisson opportunity limitation, rather than
moving Code arrival times. Show each actual arrival's per-rank capacity and
pending-head global-enough/local-blocked witnesses; aggregate mean is not proof.

300s local and remote queries continue to terminal. New shared watcher exit
barrier and immutable per-attempt manifests precede fullSHA archival and
independent token/graph/ownership/budget/ledger verification. No commit/push or
Aegaeon inference. Evidence: `evidence/workaware-pressure-1p2-20261027/`.
