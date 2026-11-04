# Native robust release validation

Prepared2026-10-30; no native GPU result exists at preparation time. Wait for
job268245's complete SHA archive, three independently verified seeds and both
existing read-only analyses before promoting or synchronizing any serving file.

## Frozen validation

- Reuse the verified long-context calibration and original closed N2 probes for
  seeds20261030/20261031. Keep all original request IDs, paired lengths, manual
  Conv output24576, Conv arrivals0/1s, real100K Code arrival120s/O64 and1200s
  offered windows. This is not a Poisson experiment and not a frozen final rate.
- One allocation of two A800; independent Qwen3-30B-A3B tenants each DP2/EP2/TP1,
  BF16/128K,24GiB/model/rank virtual pool,10GiB/GPU controlled KV budget,
  allocator.98, default sparse FULL_DECODE_ONLY graphs and consensus pause32.
- Each seed compares random_no_return with kv_only_workaware using the explicit
  robust optimizer. Real dp_echo copy,32MiB chunks and paused inference remain;
  no overlap or altered graph captures. Preserve serial greedy-token references.
- Minimize remaining-work point cost plus the original differential empirical
  uncertainty. Do not lower uncertainty/minimum-gain guards. Account full
  after-pause controller await time separately from raw planner time, but do not
  subtract already paid costs again from continuation profit.
- Deferral is conservative for unverified mixed-active/population calibration;
  record it instead of pretending the N2/N3 resident-idle fit covers all states.

## Independent checks

CLI/config, runtime-control, archived source inventory, release report and typed
adapter must agree on robust. Missing or point markers fail validation. Rebuild
plans independently, verify real boundary receipts, full controller clock,
unchanged request/token ownership, physical source pages and copy-before-release
peak budget. Preserve failures and no-return results; never resubmit blindly.

Compare Code/Conv/combined avg/P95/P99 TTFT and Conv avg TPOT including all
requests and control costs. Report pause-inclusive same-token repayment and all
exclusions separately. Initial TTFT predates return in this fixture and cannot
measure return's TTFT benefit. Two Code samples do not establish P99 significance.
The random baseline is a simple migration baseline, not an unmoved native vLLM
allocator. Historical no_move uses the same full-reservation custom admission.

## Next gate

Native correctness and a closed-cohort return benefit do not finish the research
goal. Next measure open-loop service/KV/queue stability and birth-induced natural
recovery, retaining zero-Code and fully empty Poisson draws. Then freeze a
separate evaluation with no_move, fairness, random, capacity/headroom and robust
workaware comparisons. Candidate rates remain priors, not measured settings.
