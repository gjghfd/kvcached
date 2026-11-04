# Public response completeness audit

Observation: 2026-10-30 19:23 UTC. The following is a source-integrity audit,
not a GPU benefit result or a new response-length selection rule.

## Current public candidate

The existing OpenThoughts3 workload preserves original assistant text lengths.
It does not extend AzureConv outputs. However, the original public reasoning
responses should not be described as uncapped, natural-EOS, completed human
conversation. Auditing both fixed source shards reveals:

|Scope|Responses|Closed thought markers|Unclosed thought markers|
|---|---:|---:|---:|
|Two original shards|20000|6520|13480|
|Frozen, length-independent bank|64|28|36|

All 64 frozen bank responses were independently located using the original
dataset/revision/source path/row/assistant boundary, and their original response
SHA matched the raw parquet. The raw file sizes and SHA match the official
source manifest. The first exploratory inline audit confused the original source
path with the local destination basename and reconstructed zero bank entries;
that attempt is not verification. The strict standalone audit fixes that identity
mapping and rejects any missing bank entry or changed response.

Unclosed markers are a warning, not proof of generation truncation: these source
files do not provide authoritative finish reasons. Closed markers are likewise
not proof of semantic completion, correctness, or natural EOS. Do not change the
frozen bank to remove inconvenient responses after seeing GPU outcomes.

Evidence in the outer ParaCloud workspace:
`evidence/public-response-profile-20261031/openthoughts3-marker-audit.json`.
Reproduce with `audit_public_response_completion.py`, specifying the archived raw
root and frozen input bank. The audit has 13 passing focused CPU tests.

## Execution and next dataset checks

The only allocation, 270947, remains on two A800 GPUs. At 19:23:39 UTC its
300-second coordinator query reported 37/144 completed joint-cost cases;
earlier queries reported 26, 31, then 37. The startup-only-log concern is resolved
by real case-progress files. This phase uses synthetic calibration inputs, not
the public benefit trace. No public GPU benefit or best setting has yet passed.

Scope correction2026-10-30 20:37 UTC: the earlier wording "joint-cost cases"
does not mean both tenants are active simultaneously. The code waits for empty
engines and submits requests only to the selected tenant. Both weights/KV
managers remain resident; concurrent interference is measured only by the mixed
native/Poisson comparisons, not by this microprofile.

The future seven-policy Poisson coordinator has no allocation and waits for full
parent SHA archival and independent validation. The declared bounded search and
fixed held-out seed remain in `../docs/public-poisson-search.md`. Neither live
runtime nor frozen input files were modified for this audit.

Prefer an independently verified public source with long original responses and
stronger completion provenance if obtainable. LongWriter/WildChat measurements
remain in `public-response-profile.md`; do not represent a rare long-writing tail
as their full distribution. Supplementary Stratos/OpenThoughts2 downloads remain
unverified: local HF TLS reset, mirror redirects to the failing host, and cluster
login DNS resolution fails. The official dataset-viewer metadata endpoint is
reachable locally; a viewer export would need full-cell/no-truncation checks and
an explicit source/provenance manifest before it could replace the current bank.
Do not claim an unverified supplemental length distribution or silently change
the current frozen campaign.
