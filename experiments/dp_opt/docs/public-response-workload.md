# Public original-response workload search

Prepared 2026-10-30 (UTC). This supersedes automatic continuation of
the AzureConv-output24576 campaign, not its preserved historical evidence.

## Source selection

The Conv *tenant role* needs long residency, but must not be presented as ordinary
human chat if it serves synthetic reasoning or long-writing data. Original
assistant text determines output length under the fixed official Qwen tokenizer;
no padding, response extension, prompt repetition, or benefit-dependent sampling.
Length replay uses ignore_eos and is not a claim that this Qwen model naturally
produces the source response or reaches its original EOS at that length.

The archived first screening covers full LongWriter-6k, two hash-selected
OpenThoughts3 shards, and 10000 length-independent hash-selected WildChat
conversations from one hash-selected shard. Sources are fixed to official repo
revisions and LFS SHA/size. LongWriter is AgentWrite-generated writing, WildChat
is real conversational data, and OpenThoughts3 is QwQ-annotated reasoning. Do not
call the last two substitutes equivalent workloads. Possible generation caps
and source curation must be reported, not inferred to be absent.

WildChat's observed maximum4055 output tokens is too short for the desired
long-resident mechanism. LongWriter has only4 original outputs>=24576 in6000
examples, so that tail cannot represent its typical workload. The initial
candidate is the full P<=8192 OpenThoughts3 response distribution, with an
outcome-independent bounded hash bank; do not select only the longest responses.
Supplementary OpenThoughts2/Stratos screening is optional if verified public
downloads succeed. Network failures do not license fabricated lengths.

## Serving and search gates

Keep two independent Qwen3-30B-A3B tenants on two A800, DP2/EP2/attentionTP1,
BF16/128K,24GiB/model/rank virtual pools,10240MiB/GPU controlled budget,
allocator.98, default sparse FULL_DECODE_ONLY graphs, compile0 and pause32.
Keep real SWE-smith>=100K Code prompts/O64. The prompt bank must reconstruct
public messages, input token IDs and original output length independently from
raw source files. Every mark has prompt_key; the synthetic fallback is forbidden.

1. Service/capacity probe: native original public requests, actual idle/active
   mapped pages, greedy references and default graphs. Supplement cost profiles
   for any new population/context outside the old N2/N3 calibration.
2. Bounded mechanism search: public N2/N3/N4, Code arrival90s, paired fixed seeds,
   no_move/random_no_return/kv_only_headroom/robust kv_only_workaware. This closed
   cohort tests return versus natural recovery, not final Poisson tail benefit.
3. Open-loop calibration: calculate priors using the bank's E[KV*service] and
   measured service, then test local pressure targets1.1/1.2/1.3. Natural warmup,
   full drain, all zero/empty draws preserved. Select by declared capacity/queue
   guardrails, not observed algorithm improvement.
4. Freeze a finite rate grid and seeds before outcome-dependent exploration.
   Compare seven policies, including own_model, KV-only and queue fairness.
   Retain all settings and costs; optimize Conv/Code/combined avg/P95/P99 TTFT
   subject to Conv TPOT and no dropped requests. Confirm the selected candidate
   using a separately predeclared held-out seed. "Best" is only within this grid;
   few Code samples cannot establish P99 significance or a global optimum.

For OpenThoughts3's typical original O~16700, an independent request-page upper
bound is~1632MiB. Code9600 plus N3's mean background2448 gives1.1766 times the
10240MiB budget; N4 gives1.2563. These are arithmetic priors, not actual allocator
admission witnesses. Prompt/output correlation and page sharing matter. N4 is
outside the historical joint fit and requires new measured cost coverage.

All planning/pause/copy/resume/prefill/queue time enters metrics. Initial closed
cohort TTFT predates return; quantify return using same-token pause-inclusive
windows and complete Conv TPOT. Code TTFT improvements due to evacuation are
distinct from return benefit. Keep negative and excluded repayment windows.

## Current status

Old calibration269991 completed0:0;264 raw files and75 requests independently
verified. Ratios1.302601/1.325843/1.348774 yield no qualifying candidate under
the old <=1.3 guardrail. Old synthetic final-trace/log-alias producers are paused,
with no new allocation. New public profiling is CPU evidence only. The source
bank, GPU calibration and best-setting/held-out benefits remain to be executed.
