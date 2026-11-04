# Original Stratos response profile

Observed 2026-10-30 UTC. This report records an original public-data profile and
a queued continuation plan, not new Stratos GPU benefit results.

## Original public data, not extended AzureConv

The official dataset-viewer export of Bespoke-Stratos-17k contains 16710 rows.
All 168 pages have `partial=false` and empty `truncated_cells`. An independent
verifier checks their SHA, ordering and every original message/system string
against the canonical export. The complete original system prompt is retained.
All source responses and all 256 bank responses have balanced thought markers.

This is synthetic long reasoning, not ordinary human conversation. Balanced
markers do not establish natural EOS or semantic correctness. Runtime
`ignore_eos` replays the original teacher-response token length; it does not
claim the current Qwen generates the original answer or naturally stops there.
No response padding, extension, repetition or benefit-selected length tail is
used. The full public distribution, including short responses, remains visible.

|Scope|Responses|Mean input|Mean original output|Output P50/P95/P99|Maximum output|
|---|---:|---:|---:|---|---:|
|Full viewer source|16710|480.73|5072.84|3269/15785/21574|44829|
|Length-independent hash bank|256|462.59|4630.69|3187/13372/20171|24646|

The finite bank is not the full population and slightly underrepresents its
mean and tail. It is selected by source/row/boundary/response SHA, independently
of GPU outcomes. Do not reroll it to improve the result. All 256 exported prompts,
original response lengths and response SHA were rebuilt from the raw source.
The Code stream retains exactly the prior SWE-smith100K/O64 inputs.

For comparison, the earlier OpenThoughts3 two-shard audit finds unclosed thought
markers in13480/20000 responses and36/64 frozen bank entries. These markers
alone do not prove truncation, but Stratos supplies stronger structural-completion
evidence. LongWriter/WildChat distributions remain in `public-response-profile.md`.

## Provenance limits and evidence

Direct fixed-revision Stratos/OpenThoughts2 parquet downloads fail locally with
TLS reset and remotely with DNS failure. The viewer serves complete Stratos
content but ignores a requested Git revision. Therefore this is an official
HTTP **content snapshot**, not a verified upstream Git/LFS file. OpenThoughts2's
viewer reports a partial corpus and is excluded from this full-source export.
The first fast viewer download hit429; its failure remains archived. The retry
uses2-second spacing and conservative backoff respecting `Retry-After`.

Canonical source SHA:
`611374982bd7115e3743d5d0a0802396d796ebe2fe57742df371361faefdcd10`.
Raw-page manifest identity:
`519cbbe6ef9c81ba8ba1dace0875f67253b81493cc7a4263d6c6bf620f357e96`.
Stratos256/Code prompt-bank SHA:
`0073cacf9505ccb28c03de781999435826211cc68e3f3f8eb80490ccc988b198`.

Outer-workspace evidence:
- `evidence/public-response-profile-20261031/stratos-viewer-snapshot/manifest.json`
- `evidence/public-response-profile-20261031/stratos-viewer-snapshot/profile-256/summary.json`
- `evidence/public-response-profile-20261031/stratos-viewer-snapshot/independent-original-cell-verification.json`
- `evidence/public-stratos-workload-20261030/independent-marker-audit.json`

## Execution status

Checkpoint2026-10-31 06:34:54+08: sole recovery job271324 runs on two A800s at
d1n41a11g02, with28/144 per-active-tenant calibration cases completed. It starts
at05:49:48+08. There is no completed Stratos native/grid benefit or best setting.
Recovery63939/caffeinate63940 and child63943 remain alive; independent observer
65227 records300-second local-only snapshots without alarms. Both active109-file
manifests still match every SHA. Only not-yet-started Poisson source files change.

The future final allocation adds320 feasible N5..9 cost cases before policy
comparison, then independently anchors/merges exact N2..9 profiles. This addresses
the mismatch between the roughly6-request workload prior and an N2..4 forecast.
No public output is lengthened. Outside-population/context forecasts still stop,
and the unchanged residual guard can still reject a return. The80h future limit
includes the extra measurement and unchanged finite grid/fixed held-out windows;
it does not establish success, a global optimum, or a promised50% improvement.
Protocol: `../docs/public-poisson-search.md`; raw future-boundary audit:
`evidence/public-stratos-recovery-20261030/future-population-frozen-boundary-check.json`.

Independent source review confirms additional limitations: Code/Conv jointly
active observations have no verified joint forecast, active100K Code fails the
global envelope before the N1-invariance branch, and admission-busy returns
remain release/completion driven. N5..9 expansion only addresses supported
post-Code-release Conv populations. Return max_moves4 is sufficient for the
N9 capture16→8 transition; it is not a reason to raise the bound in this round.
Source verification now explicitly rejects missing/wrong supplement anchors and
changed parent coefficients. The first full future-file regression passes856
tests; the strengthened anchor/adjacent regression then passes55 tests. Existing
`asyncio_mode` warnings remain; a missing field in the initial synthetic test
fixture was corrected and is not counted as an initial pass. No GPU benefit
claim follows from either CPU result.

The real two-layer sbatch/CLI contract adds11 CPU tests: actual parser/engine
arguments preserve cost `point/no_move/pilot`, unset prior fit, then bind the
verified composite for seven-policy `robust/pilot0`. They check default sparse,
BF16/128K, allocator.98 and pause32, and prevent robust launch after bundle or
independent-verifier failure. No inference engine is imported in this CPU test.

### Independent public candidate

An independent16-page OpenR1 candidate sample completes locally:1600 original
responses, Omean5960.80/P9514113/P9916793, no sampled O>=24K. Its resident-weighted
reservation862.26MiB is below this bank's880.38MiB, so longer mean output alone
does not imply fewer resident requests at fixed pressure. It does not reroll the
current bank or rate freeze. See `public-openr1-candidate-profile.md` for sample,
finish-flag and rate-prior limits; this is not a GPU dataset-benefit comparison.

### Parent failure and minimal metadata recovery

Job270947 fails at2026-10-30 21:28:28 UTC (2026-10-31 05:28:28+08) in the first
native `random_no_return` run. It records1/3 completed requests and one move,
then Conv completion raises `KeyError('source_internal_row')`: public marks use
`source_row`, not the legacy Azure/SWE field. This is not CUDA OOM. All215 raw
files and the shared-watcher terminal barrier are independently SHA checked.
No complete native comparison or algorithm benefit is reported from this job.
The independently rebuilt144-case cost phase remains separate valid evidence.

The stopped v2 continuation never submitted Stratos. Exact old sources are
backed up. Completion now copies the mark's existing `source_*` and original
response provenance without inventing an internal row; independent public
verification checks those fields. Real asynchronous two-tenant public/legacy
completion tests, reference/empty trace tests and public-source safety tests
pass:176 tests total, with the existing `asyncio_mode` warning.

`public_source_recovery.py` requires the complete failed archive and exited
parent before starting a new Stratos calibration. It retains270947 as FAILED
and uses genuinely completed269991 only as the environment bootstrap. The
Stratos256 bank and original output lengths, Code100K/O64, budget, graphs and
SLO are unchanged. Four staged label/watchdog files have protected SHA backups
and promotion; GPU execution still requires the child's protected remote sync.
Recovery evidence: `evidence/public-stratos-recovery-20261030/`.
No commit/push, Aegaeon run, new Stratos result or best setting is asserted here.

### Independently checked parent cost phase

At2026-10-30 21:21 UTC,61 files from the completed144-case parent cost phase
were copied into a separate stage snapshot, checked against every remote SHA,
and independently rebuilt locally. The rebuilt fit matches both compute-side
fit outputs. This is a single-active-tenant cost-phase check, not complete parent
closure, Stratos execution or a benefit result. The active parent now runs
`native-2026103101-N2`; two GPUs report81153MiB free before its new model load,
so the earlier socket-cleanup warning does not establish retained GPU weights.

Tenant0 is Code and tenant1 is Conv. An initial local derived diagnostic
accidentally labeled model0's contrast table as Conv; it is preserved, and the
corrected model1 record is authoritative. Raw SHA and rebuilt fits are unchanged.
Evidence: `evidence/public-source-continuation-20261030-r2/cost-stage-270947-snapshot/independent-verification-corrected.json`.

For the Conv tenant at16000 tokens/request, the fitted fully skewed versus
count-balanced single-step contrasts are:

|Active Conv requests|Sparse-graph component|Attention component|Total contrast|Two-residual-bound positive part|
|---|---:|---:|---:|---:|
|2|5.471ms|1.519ms|6.991ms|3.475ms|
|3|0ms|0.695ms|0.695ms|0ms|
|4|0ms|1.898ms|1.898ms|0ms|

The last column normalizes the existing changed-segment two-bound guard before
pause/copy/control charges; it is not measured TPOT improvement or an ILP return
guarantee. N2 crosses the default capture boundary from1 to8; most of that
contrast is padding, not pure attention time. No graph configuration is changed.
For N3/N4, fitted gains can be smaller than conservative uncertainty charges.
Additionally, the current forecast stops when its first population is outside
the calibrated N2/N3/N4 set or context/attention envelope. It does not silently
extrapolate a measured N4 profile to N6.

This is a concrete reason to check return decisions and actual population
histograms, rather than simply raise Conv rate to force queueing. Lower Code
rate alone cannot resolve unsupported Conv populations; frequent new Conv
arrivals can also naturally restore a count-balanced layout before a migration
repays its cost. The existing fixed search remains unchanged. If its winner
lies on a rate boundary, or proposals are mainly outside-domain/uncertainty
rejects, the next independently frozen iteration should first extend measured
cost coverage or explore adjacent rates with the same original bank and queue
guard. Never weaken uncertainty solely to obtain a positive result.

Recovery checkpoint2026-10-30 21:03 UTC: the parent's actual20:58:40 query records
131/144 cost cases, `RUNNING`; all109 active runtime SHA remain unchanged.
There are still no Stratos GPU benefit results or measured best settings.
The unsubmitted original source continuation was intentionally stopped and its
state archived. Its v2 replacement57894/caffeinate57895 runs locally in
`evidence/public-source-continuation-20261030-r2/`, waiting for independent parent
SHA closure and coordinator exit before four-file protected promotion.
The future request watchdogs become7200s/18000s uniformly across strategies;
the120s SLO, frozen original dataset lengths, arrivals and GPU configuration do
not change. Sixteen safety tests and23 staged CLI tests pass, with the existing
`asyncio_mode` warning. See the protocol for allocation/reference/drain limits.
No commit/push or Aegaeon execution occurs.

At19:58:40 UTC, existing job270947 completes73/144 cost cases on two A800s.
It retains its original frozen inputs and is not interrupted or relabeled.
At19:57:59 UTC, the old OpenThoughts3 final-grid coordinator40127 was intentionally
stopped before any allocation submission (`job=null`); its state is archived.
Source continuation49595/caffeinate49596 waits for full270947 SHA archival,
independent closure and the old main process exiting. It then promotes only
source-label/report fixes, runs Stratos native/rate calibration and the seven-policy
finite grid plus fixed held-out trace. See `../docs/stratos-workload-search.md`.

Current source/provenance/continuation regression:87 focused CPU tests pass;
23 staged source-label/actual-CLI tests pass. One pre-existing `asyncio_mode`
warning remains. Early verifier-fixture and staged missing-sbatch failures are
recorded separately and are not counted as passes. No new Stratos inference or
best-setting claim has passed yet; no commit/push or Aegaeon inference occurs.

## Calibration scope correction

Code inspection at20:37 UTC confirms that `workaware_pilot.py` waits for both
engines to empty and activates only `case['model']` in each case. Both models
remain resident, but the cost microprofile is per-active-tenant, not simultaneous
two-model joint calibration. This is appropriate to investigate post-Code
return with Code idle; it is not an interference model for Code-prefill/Conv-decode.
Mixed native and Poisson traces retain real overlap costs in measured metrics.
Do not claim mature joint calibration or optimality across arbitrary overlap
from these per-tenant coefficients.
