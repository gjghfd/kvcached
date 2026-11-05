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
10. **不能假设知道输出长度**（11-04）：按当前长度准入，当前长度就放不进的长请求不得进入；KV 随长度增长不够时触发 vLLM retract；placement 给请求预留增长空间，但不能用完整输出长度。所有 baseline 和我们的策略都在这个设定下重跑。V1–V7 用的是"输入 + 输出"全量预留（相当于预知输出长度），只作历史参考。
11. 每次启动作业后用 Monitor 监视到结束，结束后记录、总结结果，再启动下一个任务（11-04）。

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
| `sim/` | 离线 simulator（与 live 共用 `planner.py`）、live 拟合/验证/预测/扫描（见 §3.1）、RQ4 扫描（`sweep_rq4.py` 负载 × KV 预算，`sweep_slicing.py` 切片参数）、DP=2 验证（`validate_dp2.py`） |
| `tools/` | `make_trace.py`（冻结 trace，Code 可 Poisson 或固定间隔）、`analyze.py`（指标，live 和 sim 共用）、`report_main.py`（主报告）、`decompose.py`（按 Code 状态拆分 Conv TPOT）、`check_tm.py`（分时/切片检查：holds、Code prefill、持有期间 Conv 是否出 token）、`vs_forecast.py`（实测对预注册预测）、`final_tables.py` / `combine_runs.py`（多作业汇总）、`master_table.py`（论文表 2：多作业、多 seed 的 R1–R14 与单组件对照）、`xplace_ctl.py`（build/upload/submit/status/fetch，支持 `--dependency`/`--exclude`） |
| `tests/` | 本地和作业启动时运行的 CPU 测试。假 vLLM 引擎（含 chunked prefill、keep-pause）会端到端跑完各阶段；N-rank peer transfer 用多线程模拟 all_gatherv 测试 |
| `docs/protocol.md` | 实验协议：§6 短作业 campaign，§7 v2，§8 v3，§9 V4，§10 v3.1，§11 V6 公平基线，§12 v3.2，§13 V7 kvcached 原生基线；当前长度设定：§14 V8，§15 V9/V10（页共享、截断、空闲块修复），§16 V11 复现 seed，§17 V12 补齐 seed 914，§18 V13 RQ4 锚点 |
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
- **kvcached 原生基线**（`kvcached_native`，V7）：两个 vLLM 同时推理，请求到达即交给 vLLM（不指定 DP rank），调度器切换到原生准入（按步分配 KV，不够时抢占重算，`placement_set_native`）。控制器不准入、不迁移、不暂停。为了容量公平，trial 前用 `controller.apply_ballast` 在每卡占掉多余显存，使 kvcached 可映射的共享 KV 与受控预算相同（按 `KVCACHED_GPU_UTILIZATION` 的余量公式计算），trial 后释放。记录抢占次数、每卡最大映射和每 60s 的可映射容量。
- **v3 分片分时**（`ca_full_tm_s4/s5`，`slice=(S, R)`）：Conv 持有满 S 秒且仍有 Code 在 prefill 时，用 vLLM 原生 keep-pause 暂停 Code，让 Conv 跑 R 秒，再停 Conv、恢复 Code。
- **v3.1 按需切片**（`ca_full_tm_slo`，`slice=(T_wait, R, S_max)`）：只在某个持有中的 Conv 等首 token 超过 T_wait，或持有达到 S_max 时切片；先暂停 Code 再检查它是否仍在 prefill。
- **数据面**：边界暂停（每 32 步，空闲 tenant 先停）→ export → 目的地 prepare → 全部 DP rank 参与 `all_gatherv`（source 复制、destination echo、bytewise 校验）→ commit → 源端释放 → 恢复。
- **诊断**：节点探针（`--probe-max-ms`，慢节点直接中止）；事件循环滞后监测、看门狗抓阻塞时的调用栈、GC 计时（`loop_lag.jsonl`、summary 中的 `loop_lag` / `gc_*`）。
- **v3.2 日志写入**：`trace_runner.Journal` 由后台线程写 jsonl 和 `progress.json`，事件循环只入队。它修复了在共享文件系统上同步写文件造成的 0.6–3.6s 事件循环卡顿。V7 真机验证有效（4 轮 0 次卡顿）；V6 及之前的真机结果都来自修复前的代码。

### 3.2 当前长度准入（v4，V8 起）

- **准入**：受控策略只预留 prompt + 256 token（`--kv-mode prompt --reserve-extra-tokens 256`），之后按步增长；KV 不够时由 vLLM 抢占重算（`live_scheduler._preempt_request` 计数）。`kvcached_native` 策略自己把调度器切到 vLLM 原生准入（`kv_mode` 为 native）。planner 和 simulator 都不读输出长度。
- **增长余量**：`planner.conv_growth_tokens` 按"decode 速率 × 时间窗口"估计 Conv 在 Code 驻留期间的增长，并额外留 1 个页组（`slack_pages`）。速率按调度方式取：分时下为 0，切片时只计切片窗口，并行时按受干扰的速率。`*_nh` 策略不留余量，用作消融。
- **容量对齐**：每轮开始前用占位把每卡的物理容量压到 170 个 96 MiB 页组（15.94 GiB），控制器预算同步设为这个值；轮次结束后释放。
- **页共享**：按步增长时，kvcached 把同一 rank 上各 Conv 的块交错放进同一批页里。只迁走少数 Conv 腾不出整页，准入往往要迁走目标 rank 上的大部分 Conv（V10 中位 9 个，最多 13 个），所以 `max_moves` 提到 24（`planner-params-v3`）。planner 按快照中的页归属，只计入所有者全部被迁走的页。
- **物理空闲块**：kvcached 报告的 `free_blocks` 是物理空闲。planner 判断 Code 能否放进时，计入计划迁出后会释放的块（v4.3）；不可行时把每个目标 rank 的诊断写进 decisions。
- **控制器侧排队**：原生 vLLM DP 下，被 KV 挡住的请求会让整个 DP 组持续执行 dummy batch，拖慢共址的另一个模型。受控策略在控制器里排队，放得下才交给 vLLM。
- **截断**：一轮到截止时间（trace + 1000s）时记录部分进度（`incomplete.json`，未出首 token 的 TTFT 记下界），abort 后继续下一轮。
- **有 Code 驻留时不回迁**，避免在 Code prefill 期间做慢迁移。

## 3.1 simulator（`sim/`）

- `fit_live.py`：从一次 live trace run 拟合真值步时与干扰参数，当前为 `truth-pilot-nm-v2.json`（pilot `no_move`）。
- `validate_live.py`：多个噪声副本对比 live 与 sim，含机制指标（无 Code 时段处于 G16 档的比例）。
- `predict.py`：对冻结 trace 做预注册预测（每个策略独立噪声副本，给出对比差值的分布）。含随机策略时加 `--vary-admission-seed`，让副本同时改变准入随机种子，否则随机迁移的区间偏窄（V13 §18.6）；`--workers N` 多进程，结果不变。
- `sweep_live.py`：在非最终 seed 上扫描区间。
- `live_samples.py`：从 live cadence 中提取逐秒步时和布局特征。
- simulator 支持分时、先停 Conv、周期切片与按需切片。切片时序已按真机修正：触发后 Code 先多跑一步再暂停，放开 Conv 前有 0.3s DP 共识。
- 当前长度准入下，KV 按页模拟 kvcached 的分配器（`PagePool`，`page_model="blocks"`，11-05 起为默认）。
  - 单块增长进最满的页，多块分配找能放下的最小页；页内块全部释放才取消映射；需要新页而 GPU 已满时，抢占本租户最新的请求。
  - 快照中的页归属与真机一致。旧的"高水位"规则（`--page-model high_water`）用于重现 V8–V12 的预测。
  - 本地测试：`sim/test_page_pool.py`，不随作业上传。
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

- 一个作业跑多个 trace / KV 上限：`XPLACE_SEGMENTS=<segments.json>`（`main.py --segments`，文件放进 inputs），内容为 `{"segments": [{"label", "trace", "budget_cap_gib", "policies"}]}`，按顺序运行，输出到 `trace/<label>/<policy>`；所有 segment 的 trace 和策略名在第一轮之前检查。
- 远端目录：`~/run/kvcached-paracloud/xplace/<tag>/{upload,job-<id>,logs}`，upload 只读、不可变。
- 本地证据：`evidence/xplace-<tag>/`。
- 本地测试：`/tmp/xplace-venv/bin/python -m pytest -q -p no:cacheprovider tests/`（需要 torch 的测试在无 torch 时自动跳过）。
- 报告：`python3 tools/report_main.py <job>/out/trace --end-s <duration>`（A/B/消融表、配对 bootstrap、按 Code 状态拆分 Conv TPOT）。
- 自写分析脚本注意：`migrations.json` 里的 `request_id` 是 vLLM 内部 ID，形如 `<控制器 ID>-<8 位后缀>`，与 `requests.json` 对齐时要先去掉后缀。

## 5. 当前结论（2026-11-05，V10–V13 之后，当前长度准入）

单节点、两个 seed，三个作业：seed 911 为 V10（14 个策略）；seed 914 为 V11（8 个）和 V12（9 个，其中 3 个与 V11 重复）。各作业内配对，Code n=4，Conv n=58 / 54。两个 seed 的总表 R1–R14 见 results-short §12.8。V13 在两个偏离工作点的位置（λ=0.14、16 GiB；24 GiB）做了真机锚点，四列总表见 §12.9。

- **最终推荐 `ca_full_tm_slo`**：
  - 对控制器基线 `no_move_v2`：Code TTFT −88.5% / −92.9%，Conv TPOT −4.6% / +2.2%（都在 20% 内），Conv SLO 1.00，0 次抢占；
  - 对 kvcached 原生：原生下 2 / 3 个 Code 到 1900s 都没开始（截断，TTFT 记下界），Code TTFT 至少降 95%，Conv TPOT −90%。
- **问题**：按当前长度，后到的 Code 在每张 rank 上都放不进。
  - 原生 vLLM DP 下，被挡住的请求让 Code 实例持续执行 dummy batch，把 Conv 拖到 300–330ms/token；
  - 控制器侧排队可以避免这一点，但被挡的 Code 仍要在控制器里等 378–542s。
- **A（迁移）**：迁走目标 rank 上的大部分 Conv 后，Code 排队降到约 1s。Code TTFT 在并行下 −76% / −87%，分时下 −87% / −89%。
  - 并行下 Conv TPOT +39% / +47%，超出 20%；
  - 分时把它降回来：在不迁移、随机、代价感知三种 placement 上，Conv TPOT 在 seed 911 上为 −15%、−24%、−30%，在 seed 914 上为 −8%、−24%、−29%。
- **B（代价感知 placement）**：
  - 分时下稳定：−11.6% / −10.4%（两个 trace）；分时下同一策略两次运行只差 1.8%。
  - 不分时下小且随运行变化：三次独立配对 −4.3%、−4.6%、−2.0%，方向一致；随机迁移自身两次运行差 4.4%，与 B 同量级。请求级 bootstrap CI 不含轮次间波动。
  - 来源是持续控制失衡：有代价感知回迁的 21 轮（V10–V13），无 Code 时段各 rank Conv 数之差都在 1.3–2.5；迁移但不回迁的 9 轮（随机，或只做代价感知准入）为 2.8–7.0，取决于这一轮迁到了哪里。B 的大小就是随机迁移那一轮多留下的失衡，所以准入与回迁各自的份额在各轮之间不稳定（V12 判定 1、2 因此不成立）。
  - 抢占（V10–V13）：完整设计（回迁 + 增长余量）16 轮共 0 次；去掉余量的 5 轮中只有 λ=0.14 不分时那一轮有 2 次；不回迁的 9 轮中 5 轮出现抢占，共 22 次，最满 rank 的 Conv 映射达到 15.66–15.84 GiB（上限 15.94 GiB）。
  - 来源不在 CUDA graph 档位：每 rank 超过 8 个 Conv，G16 不可避免；在最满 rank 的 batch 和 attention 长度。
- **负结果**：
  - 增长余量（准入时为 Conv 增长多留空间）在 λ=0.12（两个 seed、两种调度）下没有作用：四次对比都没有抢占，Conv TPOT 差别 ≤3.3%、方向不一；为增长留空间主要靠持续回迁。只有在更重的不分时负载下（λ=0.14，V13）余量起了作用：不留余量 2 次抢占、留余量 0 次，代价是 Conv TPOT +1.0%（n=1）。分时下 Conv 被持有、不增长，余量不需要；
  - 公平门不能解决单卡放不下，Conv TTFT 75.5s / 144.5s；
  - 按页挑选迁移对象不比按 KV 挑选省迁移（模拟），代价感知准入没有作用是页共享造成的结构性结果。
- **RQ4（模拟 + 真机锚点）**：碎片和原生的崩溃都随 KV 预算增大而消失，分时的收益不变；按需切片优于所有周期切片（results-short §12.7）。
  - 24 GiB：Code 全部直接准入，原生与受控基线相同，本文策略的收益全部来自分时（Code TTFT −52.5%，Conv TPOT −26.1%）；本文策略在 16 GiB 下的性能与 24 GiB 下几乎相同。
  - λ=0.14：受控基线 4 个 Code 全部被挡；本文策略 Code TTFT −95.3%，Conv TPOT +0.5%；B 在第三个 trace 上复现（分时 −4.9%，不分时 −6.1%）。
  - 块级页模型的样本外误差：V12 上 Code TTFT 3.1%、Conv TPOT 1.8%；V13 上 3.6%、2.6%（受控基线在 λ=0.14 下偏差最大，阈值效应）。
- **全量预留下的旧结论（V1–V7）只作历史参考**：当时 Conv 预留的输出空间约为实际的两倍，与原生相比迁移没有 A 收益。

完整结论与表格见 [reports/results-short.md](reports/results-short.md) §12，协议见 [docs/protocol.md](docs/protocol.md) §14–§18，状态见 [reports/progress.md](reports/progress.md)。
