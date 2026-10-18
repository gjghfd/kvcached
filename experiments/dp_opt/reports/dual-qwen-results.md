# Dual-Qwen resident / independent-client baseline

These completed runs verify native vLLM+kvcached residency, correctness and
independent-client inference. They do not measure request migration, cross-model
physical capacity handoff or placement gains. Both tenants reuse one real BF16
Qwen3-30B-A3B-Instruct-2507 checkpoint, without GPU weight deduplication.

## Resident inference: job 225965

- All four workers passed topology and expert-coverage checks; each holds
  32,073,216,000 parameter bytes and 1,365 virtual KV blocks (21,840 tokens).
- Four serial arithmetic requests returned `2 + 2 = 4.`; eight concurrent
  arithmetic requests matched their same-model/same-rank serial token IDs.
- Sixteen requests, four per model/rank, each used exactly 128 prompt tokens
  and generated exactly 256 tokens: **4,096 output tokens / 80.829464 s =
  50.674591 output tokens/s** aggregate. This is a startup/concurrency baseline,
  not a steady-state comparison or optimized throughput result.
- Each KV pool's reported mapped footprint grew from **96 MiB idle/null-page
  backing to 192 MiB**, then returned to 96 MiB. **85 of 162 samples** observed
  all four pools above their idle baseline; reserved backing remained zero.
  This verifies simultaneous demand-driven KV residency and subsequent page
  reclamation, not cross-model reuse after live request migration.
- Sampled peak total GPU memory was **66,997 MiB**. Model/DP IPC namespaces were
  distinct, with both tenants' rank0 on `cuda:0` and rank1 on `cuda:1`.

Remote evidence: `~/kvcached/paracloud/state/dual-qwen-225965/benchmark.json`
and matching `logs/dual-qwen-225965-instance-*.log`.
Local evidence: `ParaCloud/evidence/kvcached-dual-qwen-225965/`.
The job exits after verification: **there is no online endpoint now**.

Build job 225943 passed extension import and four namespace tests. Earlier
225933 failed to import `libc10.so`, fixed by adding Torch's library directory;
225937 loaded weights but failed because `ninja` was absent from PATH, fixed by
including the inherited environment's `bin` directory. Failure logs are kept.
The benchmark source subsequently tightened the residency check to require
growth above idle backing; the saved 225965 samples independently satisfy it.
Final regression job **225997** completed with `COMPLETED/0:0` on
`d1n41a18g02`, 2026-10-17 16:08:41–16:08:53 CST: **197 tests passed in 9.64 s**.
The `validate.sbatch` suite covers DP namespaces, benchmark routing and counts,
TP KV-management world size, async unmap ordering and EngineCore shutdown.
Pytest emitted one configuration warning for `asyncio_mode` because
pytest-asyncio is not installed; no tests failed. Logs and job accounting are
archived alongside the benchmark evidence.

## Independent per-model clients: job 226073

Slurm job **226073** ran on **2026-10-17 17:25:47–17:33:21 CST**,
`d1n41a13g02`, two **A800-SXM4-80GB** GPUs, and finished **COMPLETED/0:0**.
The job was submitted at 16:32:06 and initially waited for resources; its
30-minute limit / 15-minute minimum backfill did not require duplicate GPU jobs.
Both tenants retained DP=2/EP=2/attention TP=1 on the same GPU pair.

- Two independent client PIDs, 1505880 and 1505881, completed three synchronized
  rounds. Each tenant passed **48/48 measured requests**, with zero failures,
  24 requests per DP rank and **12,288 output tokens**; 128 input / 256 output
  tokens per request. Correctness checks and warmup are excluded.
- Per-model throughput uses each model's summed active round spans:

| Tenant | Output tokens/s | Mean latency (s) | Mean TTFT (s) | Mean TPOT (ms) |
| --- | ---: | ---: | ---: | ---: |
| instance-1 | 48.143137 | 84.721580 | 0.689078 | 329.538872 |
| instance-2 | 48.086863 | 84.820773 | 0.689231 | 329.927320 |

- Aggregate throughput is **24,576 output tokens / 255.547087 s = 96.170143
  output tokens/s**, using the common cross-model round spans, not the sum of
  per-model throughput denominators. Individual rounds achieved 96.517035,
  96.373679 and 95.624506 output tokens/s.
- Model request windows overlapped for 84.566496, 84.996159 and 85.666699 s.
  All four KV pools grew above idle backing in 168, 169 and 170 samples in
  the respective measured rounds. Each pool mapped **96 → peak 384 → 96 MiB**.
  This verifies concurrent demand-driven residency/reclamation, not migration.
- Stream chunks contained at most one output token; TTFT/TPOT are still
  client-observed delivery timings, not instrumented GPU kernel/step timings.
- The expanded integration suite passed **206 tests in 5.98 s**, with one
  `asyncio_mode` configuration warning and no failures, inside the allocation.

Full client JSON/logs, worker evidence, memory timeline and final job accounting
are archived in `ParaCloud/evidence/kvcached-independent-226073/` and remotely
under `~/kvcached/paracloud/state/dual-qwen-226073/`. This remains a closed-loop
burst baseline, not a steady-state SLO claim or DP-imbalance optimization.
**The job has exited and no endpoint remains online.**
