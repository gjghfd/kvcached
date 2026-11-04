# Long-output open-loop calibration gate

2026-10-30: next stage after the complete native robust pair and its local
result consumer have independently closed. No synchronization or allocation
while the native GPU sources or result verification are active.

One allocation, two A800, unchanged independent DP2/EP2/attentionTP1 tenants,
BF16/128K/24GiB virtual pools, allocator.98,10240MiB/GPU, default sparse graphs,
pause32 and real dp_echo. Reuse the independently verified long-context fit;
do not reinterpret this single-policy calibration as robust return benefit.
The no_move-only CLI must retain the historical point/default optimizer flag;
the existing live CLI rejects robust when no workaware policy is configured.
This flag executes no release optimizer under no_move and does not change
the still-running native robust pair. The prelaunch conflict is diagnosed
before any calibration sync/allocation; old114 sources and inputs remain saved.

- Single full-reservation `no_move` policy, three Conv rates.00565/.0064/.0072.
- Code.001RPS for calibration coverage, **not** a change to the running native
  pair or a frozen final evaluation rate. Future lower Code rates remain priors.
- Seed2026103001 is an integer RNG seed, not an experiment date. Common random
  numbers preserve the complete Code stream across the three Conv rates.
-3600s offered window with600s natural warmup; keep and drain warmup requests.
  Conv inputs retain the complete weighted original bank, synthetic O24576
  explicitly marked. Code retains real >=100K paired input/O64 and prompt SHA.
- Keep zero-Code or all-zero draws. No seed reroll, forced arrival, bootstrap,
  mark filtering or output truncation. Capacity failure remains a failure.

Predeclare the candidate screen: Code-absent stable observations cover at least
70% of the3000s measurement window; physical single-rank mean pressure is1.1–1.3
against the measured single-Code pool; both halves have70% population coverage
and second-minus-first mean capacity queue is at most.5 requests. Choose the
nearest1.2 pressure, then lower Conv rate; never choose by policy TTFT benefit.
This short-run screen is **not** a proof of stationarity or a packability
certificate. Missing idle/isolated-Code witnesses, multi-Code ambiguity or
no qualifying rate are retained explicitly; do not silently extrapolate.

The local coordinator only reads the native campaign/report every300s while
waiting. After its SHA+report barrier, protect-sync114 explicit files and request
the one new A800 allocation. Terminal failure archives raw SHA before stopping
for diagnosis; success independently verifies every rate, all offered requests,
model/graph/ownership/allocator evidence and records queue/service/pressure.

Separate review and a new frozen multi-policy trace follow calibration. Keep
ordinary/fair/random/headroom/workaware, Conv/Code/combined TTFT avg/P95/P99 and
Conv TPOT, all pause/copy costs and domain rejections. Final Poisson benefit,
Code-tail significance and the total research goal remain incomplete.
