# Long-output AzureConv: first return-mechanism probe

2026-10-30, Asia/Shanghai. User explicitly selected **lengthening AzureConv
outputs first**, not substituting AzureCode. Current265356 must finish and its
full raw SHA/independent verification must pass before synchronization or a new
allocation. Its negative results and original settings remain intact.

## Changes and scope

- Keep the full574063-request weighted AzureConv input bank and independent real
  >=100K SWE-smith prompt bank; do not cherry-pick short prompts or a lucky seed.
- Extend **synthetic** Conv output from8192 to **24576** tokens. Preserve original
  paired Azure output lengths in each request. This is not an unmodified Azure
  generation-length trace. Native AzureCode O mean22.69/max5000 would be shorter
  than the current stress output, despite its longer maximum than native Conv.
- First test a closed population of **two** Conv requests, arriving at0/1s, with
  one Code arrival at120s. No background births occur during the probe. This is
  an explicitly declared mechanism control, **not the final Poisson trace**.
- Use three predeclared independent input-draw seeds20261030/20261031/20261032;
  replay identical requests under no_move, random_no_return, kv_only_headroom and
  kv_only_workaware. The existing runner records the actual shuffled order.
- Retain twoA800, independent DP2/EP2/attentionTP1 tenants, BF16/128K,24GiB virtual
  model/rank pools,10240MiB/GPU controlled budget, allocator.98, sparse defaults,
  FULL_DECODE_ONLY, compile0, pause32, real dp_echo round-trip copying and no
  migration/inference overlap. Timeout1800s/SLO120s are unchanged.

## Capacity and calibration

Full-bank prior arithmetic puts one long Conv+Code at1.190625..1.2375 times the
controlled per-GPU KV budget, mean about1.2055. Both Conv requests can coexist on
the other rank after evacuation, at most6240MiB, while Code+idle pools require
at most9696MiB. These are fresh-page estimates, **not allocator/copy-peak proof**.
Actual allocated/mapped ownership, aggregate/local capacity and copy peaks must
be checked from GPU raw before claiming this mechanism happened.

Before probing, run112 default-graph calibration cases: both resident tenants,
N2/N3, all rank splits, two repeats, prompts8192/16384/24576/31744, calibration
output3072 and stable window1536..2560. Independent raw-token/ownership/graph and
held-out repeat/context/layout verification is required. This creates a **new
standalone N2/N3 fit** in its own evidence directory: no old coefficients or
population profiles are overwritten, merged silently or reclassified as new
GPU measurements. It is sufficient for this N2 mechanism probe, not a replacement
for the full-population production calibration. Costs and uncertainty guards
remain unchanged; failure to admit/return is retained, not forced away.

## What would count as useful evidence

After Code completes, workaware must return Conv while the paired no-return
baseline is still imbalanced, repay measured planning/pause/copy costs, and
improve remaining same-ID/same-ordinal decode. Record release→return and baseline
natural recovery/completion, moved bytes and control costs. Report full-request
Conv avgTPOT and Conv/Code/combined avg/P95/P99 TTFT as well; TTFT before return
cannot be credited to return. Three Code requests do not establish Code-tail
significance, and this quiet closed-population fixture does not establish final
Poisson workload benefit. Preserve all three seeds and every completed/failed
request, including negative outcomes.

If the mechanism is sound, move to the full-bank open-loop trace. The starting
Conv rate prior is.00565253RPS rather than.03086325; it is **not frozen from CPU**.
Measure the joint KV/residency moment and physical background exposure, targeting
roughly1.2 single-rank pressure and about two resident Conv. Keep low Code
.00025RPS and all offered arrivals; validate natural births and completions rather
than asserting permanent imbalance. This follow-up is separate from this probe.

## Execution and recovery

`long_conv_return_campaign.py` waits for265356's full verified terminal archive,
runs local regression, performs old-SHA backup/atomic synchronization/new-SHA
checks, and submits exactly one twoA800 allocation. The allocation executes
long-context calibration then all paired probes. Existing remote and local
query-only polling uses300s; terminal raw SHA is archived and independently
verified before analysis. Any failure stops resubmission for diagnosis. Fork
remainsgjghfd/main; no commit/push or Aegaeon inference is authorized.

Recovery root: `evidence/long-conv-return-probe-20261030/`. Results are not available
until GPU execution and independent verification finish. The total research
goal remains incomplete even if this first mechanism probe passes.
