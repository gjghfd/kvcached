# Native robust release results

Job269567: two-seed closed N2 validation, not final Poisson.
TTFT seconds, avg/P95/P99; Conv avg TPOT milliseconds. Report empirical sample counts,
not P99 significance. Initial TTFT precedes return and cannot establish return TTFT benefit.

| Policy | Code TTFT | Conv TTFT | Combined TTFT | Conv avg TPOT |
|---|---|---|---|---:|
|random_no_return|55.8160/56.7876/56.8740|0.2292/0.2920/0.2951|18.7582/56.3558/56.7876|17.8188|
|kv_only_workaware|55.5110/56.3017/56.3720|0.2310/0.2750/0.2773|18.6577/55.9503/56.3017|14.4359|

Samples per policy: Code=2, Conv=4, combined=6

All offered requests and pause/planning/copy cost remain included. Negative reductions are regressions.
The random baseline migrates without return; no_move/fairness are not part of this new native pair.
Do not infer improvement over the old point optimizer when plans coincide. Preserve all exclusions
in the independently verified pause-inclusive repayment diagnostics, including unavailable anchors.

Workaware reductions versus random (%; negative means regression):

| Group | Avg TTFT | P95 TTFT | P99 TTFT | Avg TPOT |
|---|---:|---:|---:|---:|
|Code|0.546|0.856|0.883|-4.529|
|Conv|-0.767|5.830|6.045|18.985|
|combined|0.536|0.719|0.856|-0.087|

Full per-seed return counts, pause-inclusive repayment and excluded requests:
`native-pooled-metrics.json` / `phase_return_diagnostics` in this job's evidence root.

Next: measured open-loop service/KV/queue-stability calibration, then separately frozen Poisson
evaluation with ordinary/fair/random/headroom/workaware baselines. Rates remain priors and zero-Code
realizations remain valid. Do not reroll seeds, relax uncertainty or declare the research goal complete.
