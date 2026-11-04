# Original public response screening

2026-10-30 UTC. Evidence:
`evidence/public-response-profile-20261031/profile-r2/summary.json` in the
outer ParaCloud workspace; fixed official metadata, cards, download manifest,
LFS SHA checks and original lengths are retained beside it. No new GPU result.

|Source|Profiled assistant responses|Mean P|Mean O|O P50/P95/P99|Maximum O|
|---|---:|---:|---:|---|---:|
|LongWriter-6k, full source|6000|291.1|4584.1|3950/9310/14645|28436|
|OpenThoughts3, two fixed shards|20000|89.8|15593.6|16699/16754/16765|16779|
|WildChat,10000 conversations|24502|3321.8|432.2|397/958/1478|4055|

WildChat preserves real multiround conversation histories; no demographic fields
are exported. The sample is not all WildChat. LongWriter is synthetic long
writing, including AgentWrite planning/segmented generation; it is not verified
single-call human conversation. Only33/6000 original LongWriter responses have
P<=8192/O>=16384 and only4 have O>=24576. These tails are not a typical corpus.

OpenThoughts3 is publicly released QwQ-32B-annotated reasoning, not human chat.
14372/20000 responses meet P<=8192/O16384..32768 (71.86%). Its concentration near
16700 tokens may reflect the source generation cap; an uncapped/natural-EOS
claim is not established. Use its full short-input distribution, not a
benefit-selected long tail, as the first long-resident tenant candidate.

The official Qwen chat template/tokenizer measures original assistant content
without truncation or padding. Runtime length replay can ignore EOS; original
dataset length is not a prediction of this Qwen instance's natural answer.
Bank exports use source/row/boundary/response SHA selection independent of
performance. Raw reconstruction and live Transformers verification are further
gates. Source changes, artificial output overrides and synthetic prompt fallback
must fail validation.

Supplementary OpenThoughts2/Stratos metadata is archived, but current TLS reset
prevents verified additional source downloads. Do not report their distributions.
The first profile failed on missing assistant tool_calls metadata in WildChat;
its raw failure is preserved. Retrying only missing tool_calls=[] defaults does
not change message text, and profile-r2 passed. A mistaken user-site package
installation was completely rolled back by exact newly-added package inventory;
the successful profiling uses an independently verified temporary venv.

See `docs/public-response-workload.md` for the bounded GPU search and held-out
evaluation gates. Artificial AzureConv24576 results remain historical evidence;
their old rate and service time do not calibrate this new public workload.

First public deployment270883 failed before inference because its no_move-only
calibration phase inherited robust optimizer, which requires workaware in the
policy list.26 raw files and out/err are SHA archived. The early failed progress
link target never existed; the original failed archive is preserved, and an
explicit failed-job/nonzero-exit guard records absence without fabricating a
target. A policy-specific optimizer fix retains robust for native comparisons
and point for pure calibration; actual CLI tests cover all three phase forms.
The retry preserves the identical original bank and all request traces. This is
not a dataset capacity failure or a measured negative public-workload benefit.
