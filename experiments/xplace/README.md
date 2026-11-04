# xplace：4 卡跨模型 KV placement campaign

## 0. 用户 guideline（2026-11-02）

1. 由 Claude 接管 Codex 的进度。数值一致性 debug 不重要，只在等待性能实验时顺手做。
2. 依据最初 motivation 完成后续实验，可用 4 张 GPU（每个 tenant DP=4/EP=4）。
3. A/B 效果都要有，并有消融：
   - A：迁移降低 Code TTFT；
   - B：cost-aware placement 降低 Conv 的性能下降。
4. Conv 下降相对同 trace 的 `no_move` 控制在 20% 以内。主指标是 Conv avg TPOT，同时报告 P95/P99 TPOT 和 TTFT。
5. 复用 Codex 的数据集：
   - Code：SWE-smith ≥100K 的 10 条 prompt，输出 64；
   - Conv：OpenThoughts3 quality cohort bank256，保持原始输出长度；
   - 不用 4x resp Conv。
6. 真实实验尽量少，多用 log 分析和 simulator；最终结果必须来自真实实验。
7. Codex 的文档已备份为 `*_codex`，本目录是新的 campaign 入口。
8. 每个实验 ≤1h（每轮约 20 分钟），多个实验打包进同一个 Slurm 作业顺序运行（11-03/11-04）。
9. 分时（Code 驻留期间暂停 Conv）作为新增策略维度（11-03 同意）。

## 1. 目标与成功标准

- **A**：迁移类策略（`mig_random`、`ca_*`）的 Code TTFT（avg/P90/P95）低于 `no_move` 和 `no_move_fair`。
- **B**：
  - `ca_full` 的 Conv avg TPOT 相对 `no_move` 退化 ≤20%，且低于 `mig_random`；
  - 消融中 `ca_admit`（只做 cost-aware 准入）和 `ca_return`（只做 cost-aware 回迁）各自都有贡献。
- **正确性**：
  - 每次迁移都通过 bytewise echo；
  - 输出长度精确，token 不丢不重；
  - greedy 不一致只作观察记录。

## 2. 目录

| 路径 | 内容 |
|---|---|
| `runtime/` | 上传到集群的运行时。<br>`live_scheduler.py`/`worker.py`/`peer_transfer.py` 由 dp_opt 泛化为 N rank；模块名必须保留，kvcached 补丁按名字 import。<br>`controller.py`（事务与迁移）、`planner.py`（准入/回迁）、`cost_model.py`（步时模型）、`trace_runner.py`（策略驱动）、`calibrate.py`（Job 1 校准）、`fit.py`、`main.py`（入口） |
| `xplace.sbatch` | 4×A800 作业脚本：SHA 校验 → GPU 检查 → import 自检和单测 → `main.py` |
| `sim/` | 离线 simulator（与 live 共用 `planner.py`）、live 拟合/验证/预测/扫描（见 §3.1）、DP=2 验证（`validate_dp2.py`） |
| `tools/` | `make_trace.py`（冻结 trace，Code 可 Poisson 或固定间隔）、`analyze.py`（指标，live 和 sim 共用）、`report_main.py`（主报告）、`decompose.py`（按 Code 状态拆分 Conv TPOT）、`check_tm.py`（分时/切片检查：holds、Code prefill、持有期间 Conv 是否出 token）、`vs_forecast.py`（实测对预注册预测）、`final_tables.py` / `combine_runs.py`（多作业汇总）、`xplace_ctl.py`（build/upload/submit/status/fetch，支持 `--dependency`/`--exclude`） |
| `tests/` | 本地和作业启动时运行的 CPU 测试。假 vLLM 引擎（含 chunked prefill、keep-pause）会端到端跑完各阶段；N-rank peer transfer 用多线程模拟 all_gatherv 测试 |
| `docs/protocol.md` | 实验协议：§6 短作业 campaign，§7 v2，§8 v3，§9 V4，§10 v3.1，§11 V6 公平基线，§12 v3.2 |
| `reports/results-short.md` | 结果与结论 |
| `reports/progress.md` | 进度记录（作业、状态、下一步） |

## 3. 方法要点

- **每步代价**：vLLM 开 CUDA graph 时把各 DP rank 补齐到 max token 数，所以每个 tenant 的 decode 每步约为 `base + graph[bucket(max_r n_r)] + c·max_r ctx_r`。另一个 tenant 活跃时还要加跨 tenant 干扰项。
- **准入（Code 放不下时）**：
  - `mig_random`：随机可行目标，按 KV 最大优先选要移走的请求，目的地随机。
  - `ca`：枚举目标和 3 种挑选顺序，目的地用贪心避免跨档位，按"剩余工作 × 每步代价 + 暂停/复制代价"选最优方案。
  - 只对 KV ≥ 4 GiB 的请求（Code）走迁移准入。
- **回迁**（`ca_return`/`ca_full`）：在 Code 完成时触发，之后每 10s 检查一次。把各 rank 的 Conv 计数平整到某一水平，收益要超过迁移代价、headroom 风险和 margin 才执行。
- **分时**（`*_tm`）：Code 被准入后，Conv tenant 在下一个 32 步边界暂停，最后一个驻留的 Code 完成后恢复（`controller.set_hold`）。原因是两个 EP tenant 同卡并行时都慢 10 倍以上，而暂停期间 vLLM DP engine 完全空闲。
- **v2 控制面**（`*_v2`）：
  - Conv 准入只实时查询 Conv tenant，Code tenant 用缓存视图；Code 准入前后各递增一次 `code_epoch`，保证缓存只会高估 Code 占用；
  - 分时下先让 Conv 停到边界，再准入 Code（`hold_first`）。
  - 公平基线 `no_move_v2` / `mig_random_v2`：策略逻辑与 `no_move` / `mig_random` 相同，只加上快路径，使最终对比的控制面版本一致（V6）。
- **v3 分片分时**（`ca_full_tm_s4/s5`，`slice=(S, R)`）：Conv 持有满 S 秒且仍有 Code 在 prefill 时，用 vLLM 原生 keep-pause 暂停 Code，让 Conv 跑 R 秒，再停 Conv、恢复 Code。
- **v3.1 按需切片**（`ca_full_tm_slo`，`slice=(T_wait, R, S_max)`）：只在某个持有中的 Conv 等首 token 超过 T_wait，或持有达到 S_max 时切片；先暂停 Code 再检查它是否仍在 prefill。
- **数据面**：边界暂停（每 32 步，空闲 tenant 先停）→ export → 目的地 prepare → 全部 DP rank 参与 `all_gatherv`（source 复制、destination echo、bytewise 校验）→ commit → 源端释放 → 恢复。
- **诊断**：节点探针（`--probe-max-ms`，慢节点直接中止）；事件循环滞后监测、看门狗抓阻塞时的调用栈、GC 计时（`loop_lag.jsonl`、summary 中的 `loop_lag` / `gc_*`）。
- **v3.2 日志写入**：`trace_runner.Journal` 由后台线程写 jsonl 和 `progress.json`，事件循环只入队。它修复了在共享文件系统上同步写文件造成的 0.6–3.6s 事件循环卡顿。本地已验证，尚未上真机；V6 及之前的真机结果都来自修复前的代码。

## 3.1 simulator（`sim/`）

- `fit_live.py`：从一次 live trace run 拟合真值步时与干扰参数，当前为 `truth-pilot-nm-v2.json`（pilot `no_move`）。
- `validate_live.py`：多个噪声副本对比 live 与 sim，含机制指标（无 Code 时段处于 G16 档的比例）。
- `predict.py`：对冻结 trace 做预注册预测（每个策略独立噪声副本，给出对比差值的分布）。
- `sweep_live.py`：在非最终 seed 上扫描区间。
- `live_samples.py`：从 live cadence 中提取逐秒步时和布局特征。
- simulator 支持分时、先停 Conv、周期切片与按需切片。切片时序已按真机修正：触发后 Code 先多跑一步再暂停，放开 Conv 前有 0.3s DP 共识。
- planner belief 始终用冻结的 `step-params-dp4-v1.json`，与 live 一致。

## 4. 执行

```bash
python3 tools/make_trace.py --conv-rate R --code-rate r --duration-s T --seed S --output evidence/xplace-inputs/<trace>.json
python3 tools/xplace_ctl.py build <tag> --inputs <prompt-bank.json> <trace.json> <planner-params.json>
python3 tools/xplace_ctl.py upload <tag>
python3 tools/xplace_ctl.py submit <tag> --time HH:MM:SS --env XPLACE_PHASES=... XPLACE_POLICIES=... XPLACE_TRACE=... XPLACE_BUDGET_CAP_GIB=...
python3 tools/xplace_ctl.py status <tag> <job>
python3 tools/xplace_ctl.py fetch <tag> <job>
```

- 远端目录：`~/run/kvcached-paracloud/xplace/<tag>/{upload,job-<id>,logs}`，upload 只读、不可变。
- 本地证据：`evidence/xplace-<tag>/`。
- 本地测试：`/tmp/xplace-venv/bin/python -m pytest -q -p no:cacheprovider tests/`（需要 torch 的测试在无 torch 时自动跳过）。
- 报告：`python3 tools/report_main.py <job>/out/trace --end-s <duration>`（A/B/消融表、配对 bootstrap、按 Code 状态拆分 Conv TPOT）。

## 5. 当前结论（2026-11-04，V6 之后）

- **最终推荐 `ca_full_tm_slo`**：对公平基线 `no_move_v2`（V6，seed 704，同机配对），Code TTFT −75%，Conv TPOT −21%，Conv TTFT 无显著差异，Conv/Code SLO 都是 1.00。
- **A 成立**：两个 seed 上，迁移让 Code TTFT 降 45–59%（同机配对）。
- **B 要分两层**：
  - 加上分时等调度后，Conv TPOT 比公平基线好 21%；
  - placement 本身在无 Code 时段有效（步时 −3%），但不分时下两次独立配对中，端到端都比随机迁移差（+2.4%、+7.1%）。
- **不分时的迁移**：Conv TPOT 比公平基线差 15%，CI 上界 23%。
- **样本**：单节点、两个 seed、每轮 4 个 Code。

完整结论与表格见 [reports/results-short.md](reports/results-short.md)，协议见 [docs/protocol.md](docs/protocol.md)，状态见 [reports/progress.md](reports/progress.md)。
