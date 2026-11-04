# OpenR1 original-response candidate sample

Observed2026-10-31+08. This public mathematical-reasoning candidate is not
human conversation, a complete corpus profile, or a GPU benefit experiment.
It does not replace Stratos256 in the active271324 campaign.

## Original data and independent verification

The official viewer reports93733 default/train rows for OpenR1-Math-220k.
One fixed SHA-ranked100-row page per16 population strata yields1600 original
assistant responses, selected independently of response lengths/GPU outcomes.
All raw pages are complete/untruncated. Original message strings, response SHA,
source flags and page-local→global row mapping are independently reconstructed.
Raw payload:88102723 bytes; canonical strings:28463678 bytes. No response is
padded, repeated, extended, filtered or redrawn.

HTTP sample identity:
`viewer-sample:53fa0a93e23b819ff79f775b9bb23aeeb37e6fe01c77355172dc02d8edc62e36`.
Upstream Git/LFS revision is not verified. Stratified pages are not the full
population or a simple independent-row sample; quantiles are not confidence intervals.

|Original Qwen-tokenized variable|Count|Mean|P50|P95|P99|Maximum|
|---|---:|---:|---:|---:|---:|---:|
|Input P|1600|103.8094|87|208|378|1019|
|Output O|1600|5960.8031|4913|14113|16793|18813|

Every assistant response exactly matches a recorded source generation.1350
rows lack finish reasons; matched responses in the other250 have249 stop/one
length reasons. Four responses have a matching generation marked incomplete,
and four lack a closing thought marker. Duplicate matching generation strings
can carry different flags, so flag counts need not partition responses.
These annotations do not verify natural EOS, correctness or current-Qwen output
length. The sample has no O>=24K response; do not manufacture such a tail.

## Reservation and rate prior

For `b=96*ceil((P+O)/1024)` MiB, `E[bO]=5139760.2 MiB*tokens` and
`E[bO]/E[O]=862.2597 MiB`. B10240, Code9600 and pressure1.2 imply the simplified
Conv population5376/862.2597=6.2348, not measured mapped pressure or feasibility.

|Assumed decode step|Conv rate prior|
|---|---:|
|15ms|0.06973088RPS|
|20ms|0.05229816RPS|
|30ms|0.03486544RPS|

Stratos256 has Omean4630.6875 and resident-weighted b880.381MiB. OpenR1's mean
response is longer and its rate prior lower, but shorter inputs/narrower tail
leave slightly **smaller** resident-weighted KV per request. Both operate near
six requests in this prior. This alone does not create a low-population,
long-lived capture-boundary opportunity or fix readiness/context deferrals.
Keep the current bank unchanged; only a separately frozen same-budget GPU
comparison can establish a better dataset/setting.

## Evidence and execution

Outer-workspace evidence:
- `evidence/public-openr1-candidate-20261031/sample/manifest.json`
- `evidence/public-openr1-candidate-20261031/profile/independent-review-summary.json`
- Raw/canonical pages, response/token/flag audits and403-file SHA manifest.

Worker76730 completes/exits at06:59:57+08. No GPU/SSH/submission/commit/push or
Aegaeon operation occurs. Initial HF metadata curl35/TLS failure is preserved;
the successful official-viewer16-page worker has no network failure.
Existing isolated Python passes27 new/60 including adjacent tests. Main-agent
default-Python revalidation passes59/skips one tokenizer-dependent test; it is
not60 passes. No global environment changes. Full original sample profiling
and independent retokenization execute in the preserved isolated environment.
