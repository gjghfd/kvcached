# Conv pressure 1.2 follow-up

## Frozen setting (2026-10-27)

This is a new follow-up, not a rewrite of completed job257484. Its baseline had
only9.3625s/.1989s Code queue waits and measured1.107351624 local physical KV
pressure. The user authorizes increasing Conv pressure back to1.2.

Measured Conv background1739.2806284373632MiB/GPU, idle Conv96MiB, Code complete
pool9600MiB, controlled budget10240MiB imply target Conv2688MiB/GPU:
`lambda_new=.01956673845194241*(2688-96)/(1739.2806284373632-96)`.
Frozen candidate is **.030863253171591748RPS**, up57.733%; Code remains.0005RPS.
This predicts1.2, but requires actual GPU measurement rather than assuming it.

Both original Code marks and all sampled Conv mark identities/lengths are
preserved. New trace has142Conv/2Code per policy,121Conv/2Code in600..4800s,
seed20261024,4800s total. TraceSHA256:
`ab6b0d3264ad1c034359f852b47715b9a5a1557c9ec854a13ee4ad8ba0f5a83a`.

Keep twoA800, dual Qwen DP2/EP2/attentionTP1, BF16/128K, default sparse graph,
pause32,24GiB virtual pools,10GiB controlled GPU budget and current profile ILP.
All five policies drain identical offered work. Reuse the verified calibration;
skip repeated N2/N9 mechanism phases only. Code tails remain descriptive n2.

Protocol: `../docs/workaware-pressure-1p2-protocol.md`.
Evidence: `evidence/workaware-pressure-1p2-20261027/`, with `prepared.json`,
`frozen/protocol.json`, `frozen/trace.json` and the fresh submission ledger.
195 integrated archive/watcher/pressure/profile tests pass; existing
asyncio_mode warning is retained. Shared expected/started/exit markers replace
login-host-local pgrep; each archive attempt preserves its own accounting and
manifest, including failed downloads. Canonical manifests only publish after
complete verification, with unsafe-link rejection and fullSHA checks unchanged.
The first combined patch failed README context verification without applying;
the corrected focused patch and subsequent regression pass.

## Status

Job **259091** submitted2026-10-27 22:35:46, twoA800/gpu_a800. The fresh
22:36:40 query confirms PENDING/Priority; it then starts **22:40:43** on
`d1n41a28g03`.22:42:36 confirms **RUNNING/paired-trace**, profile verification
complete and startup tests ongoing. Before load the two A800-SXM4-80GB devices
have81150/81153MiB free. This is not yet measured1.2 or an admission/benefit result.
67-file oldSHA/backup/atomic sync/newSHA and fork/main/protected patches pass.
Local independent coordinator60377/PPID1+caffeinate60378 continues; remote
started marker showsln01/1891752, independently read fromln02. A bounded agent
adds three300s observations without controlling the persistent main watcher.
Continue300s
queries to terminal, fullSHA archive and independent correctness/ownership/budget
verification. Report actual pressure, Code queue/prefill and both-tenant/combined
avg/P95/P99TTFT plus Conv avgTPOT, including adverse metrics. Neither a predicted
ratio nor a queued job completes the research benefit goal.

An additional offline constant-service diagnostic uses34 prior Code-absent
Conv requests (mean residence130.837s). At the unchanged Code marks it predicts
Conv1:0 then3:2; it ignores service feedback, Code execution and real allocator
pages. It therefore warns that a mean pressure target does not guarantee every
Code request queues. Preserve both marks and test actual allocation/wait; do not
select a favorable Code arrival time. Raw diagnostic is
`fixed-service-pressure-diagnostic.json` under the new evidence root.

Local Bash3.2 rejects the empty-array/nounset skip syntax; the actual cluster
Bash5.1 was independently checked and successfully enters paired-trace.
This local legacy-shell diagnostic is not a GPU experiment failure; the active
wrapper remains frozen on the verified compute shell.

22:51 parent review: main poll22:50:52 remains RUNNING/SSH0; independent agent
has saved three successful SSH rounds22:40:19,22:45:19,22:50:19, with300.0178/
300.0112s intervals. The agent's subsequent model response returns openai_error;
retain that failure rather than treating it as a GPU error. The bounded agent
is closed; persistent coordinator60377+caffeinate60378 are still PPID1, the
63 source SHA values remain unchanged, and remote query-only monitoring continues.
No cancellation/resubmission. `monitor-parent-review-20261027-2251.json` records
the independent raw review; actual KV pressure and paired benefits are still pending.

Latest poll22:55:52 (SSH0,300.0107s interval): first **no_move_fair** has
elapsed258.688s,12offered/4completed/144,0moves. This is actual formal trace
execution but still before either Code arrival; it cannot yet establish pressure
or benefits. Independent parent verification decodes the three remote log
snapshots, checks every SHA and preserved prefix,300s timestamps, unchanged
shared watcher identity and both local command/start-time identities across
ln01→ln02→ln01 queries. See `independent-remote-query-continuity.json`.
Persistent main and remote monitoring continue to terminal; no new submission,
commit/push or Aegaeon inference. Review `pressure-results.json` plus full/main
primary metrics after complete archival and verification, rather than assuming
the first-order target or selecting a favorable Code arrival.
