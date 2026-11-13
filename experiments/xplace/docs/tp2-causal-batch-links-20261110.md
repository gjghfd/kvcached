# TP2 causal batch links (local implementation)

The new optional observer transports an explicit engine epoch, PID, DP identity,
sequence, RPC name and batch fingerprint through the existing executor RPC. It
does not match batches by shape, timestamps or independently counted worker calls.
No attributes are added to `SchedulerOutput`. The inspected deployment executor
supports callable RPCs through cloudpickle; its message queue pickles the method,
arguments, keywords and reply-rank tuple. The envelope uses those existing fields.

## Observation boundaries

- Only `XPLACE_LIVE_OBSERVATION=1` together with `XPLACE_BATCH_LINK=1` enables it.
  The existing live snapshot remains unchanged. Separate private `batch-links/`
  sockets use the existing bounded transport, producer identity and clock checks.
- The engine observer requires the native synchronous single-batch `step_fn`,
  native executor methods, PP1 and no batch queue. It wraps `step_fn` and
  `collective_rpc`, **not** either scheduler accounting method. Futures, reply
  ranks, aggregation, timeout and the original scheduler/output objects retain
  their normal paths. No extra RPC, barrier, CUDA call or pause is introduced.
- Execute and optional sample carry the same sequence and membership fingerprint.
  Dummy work has its own sequence and no request membership. Worker entry and
  return refer to the explicit RPC, including calls through WorkerWrapperBase;
  nested host scopes remain separately observed by the existing observer.
- Normal native step return provides an **upper bound** on that step's scheduler
  `update_from_output` return. It is not the exact scheduler timestamp: abort
  processing and iteration-detail attachment remain in the original path.
  Exceptions never produce this successful-return bound. It is also before the
  later engine output-queue enqueue and controller receipt, neither of which is
  observed by this observer.
- Features are causal counters read at execute dispatch: scheduled tokens,
  prompt tokens, already-accounted computed tokens and received output tokens.
  Accounted computed tokens include scheduler in-flight accounting. They are
  **not** device progress. DP step/wave/cadence values are local observations,
  not proof of common-rank alignment, identical batches or drained collectives.

Each endpoint retains only its current/latest operation plus revision, completion
and failure counters. Fast overwritten operations are not retained; a poll is
not an event stream. Missing samples, gaps and incomplete peer sets must stay
missing in any future collection protocol. There is no GPU-idle field other than
`unknown`, no remaining-fraction estimate and no prediction gate acceptance.
Invalid order, origin changes, missing features or unsupported execution paths
make association unavailable rather than fabricate a link. Runtime observation
failure does not substitute different inference work.

## Scope and next gate

This is local CPU implementation and pinned-source review, not an installed GPU
pilot or a calibrated predictor. Actual host/worker/config/serializer manifests,
independent training and holdout collection, K, controller receipt observations,
polling overhead/deadline and missing-data coverage still need predeclaration.
The r4 host-residual negative result is unchanged and cannot initialize GPU
progress. Neither symmetric scoring nor capacity/migration policy is wired here.
The existing 15%/30% gates, reserves, negative results and topology order remain.
