# TP2 live samples and frontend receipt records

This local-only collector supplies data, not a duration model. It does not change
the host-residual negative result or pass any prediction/capacity gate.

## Recorded boundaries

`CausalCollection` explicitly starts a bounded, case-scoped asyncio sampler. Each
round concurrently requests the existing scheduler/rank/host transport and every
batch-link endpoint. Source observation and receipt timestamps stay separate.
The round includes cohorts at collection start and completion; its `as_of` is
the completion boundary. Receipts arriving after that boundary cannot rewrite
the stored cohort. Transport completeness does not establish epoch agreement,
freshness, common-batch alignment or predictor availability.

The optional `engine.stream(..., receipt_observer=collector)` hook records the
time each cumulative `RequestOutput` reaches the consuming frontend coroutine.
It leaves SamplingParams, routing, prompt/output lengths and the existing
exact-length check unchanged. `min_tokens`, `max_tokens`, `ignore_eos` and future
output lengths are not added to observation features. Arrival registration is
separate and must happen when a request enters the shared waiting queue, before
submission; unregistered submissions are explicitly unavailable. Case request
IDs must be unique, including across tenants.

The inspected deployment source defaults to CUMULATIVE output. Its
RequestOutputCollector can combine multiple outputs before `generate()` yields.
Consequently counts may jump or repeat, and a zero-token output is not a first
token. The recorder retains those receipts without inventing intermediate token
times. These are frontend-consumption timestamps, **not** engine output-queue
enqueue, scheduler update, detokenizer production or GPU completion timestamps.
There is no model-output `observed_at`: a delayed EOS cannot be retrospectively
assigned to its actual generation time. Do not feed these receipts to an API
requiring that missing timestamp by copying `received_at` into it.

Raw finish/stop reasons and stream return/error/cancellation are separate events.
An early terminal output remains recorded even if the legacy exact-length check
then raises. Invalid identity/count/output-kind data does not fabricate a label.
Registered-but-unsubmitted and unfinished requests remain in the report. Closing
the observation window does not cancel inference requests; later callbacks do
not rewrite the censored state at that cutoff.

## Bounds and lifecycle

- Period, deadline, case/clock identity, event/encoded-byte and request limits are
  required arguments, not tuned defaults. Linux clock identity is checked locally.
- There is at most one collection round at a time. Slow rounds skip and count
  overdue ticks; there is no catch-up burst. Timeout/cancellation drains client
  collection tasks and retains results that already arrived from another source.
- Encoded events are retained in memory under the declared bounds. Overflow
  increments drop counters and leaves sequence gaps; terminal request state and
  aggregate error counters are retained separately. Byte limits describe retained
  UTF-8 JSON, not total Python RSS or transient serialization memory.
- `await stop()` closes this sampler only and returns a report. It does not start
  an external monitor, restart an old observer or delete PAUSE. Persist the report
  in the new case's immutable evidence directory. No source has complete batch
  event history: latest-only worker/engine snapshots can overwrite unseen calls.
- `retention_lossless` only means no attempted records were dropped. It is not
  sampling completeness, prediction eligibility or proof that all events occurred
  inside a query's support. Missing endpoints, skipped ticks, invalid records and
  open requests remain explicit. Every report leaves prediction acceptance false.

## Remaining deployment work

This API has an actual optional stream callsite, but no GPU pilot invokes it yet.
A new pilot still needs the complete frontend/worker/config/serializer manifest,
common reference K, controller-consumption target definition, fixed cadence and
deadline, encoded-memory/CPU overhead checks, and bounded-loss acceptance rules.
Arrival registration must be wired into that pilot, not reconstructed from future
submission records. Fresh training and independent validation must retain every
missing, failed, EOS and tail case. Old r1-r4 data cannot become holdout. Receipt
counts alone cannot initialize unobserved device progress or calibrated coupled
phase response; symmetric scoring and unchanged capacity safeguards remain pending.
