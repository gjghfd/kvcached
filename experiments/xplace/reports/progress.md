# xplace 进度记录

只记录作业、状态、关键数字和下一步。结论写进 README 和结果报告。

## 2026-11-02 接管

- **停掉 Codex 的链**：Codex 的链已自然结束，DP=2 两个 seed 的 N2/N3/N4 共 6 个 case 完成（最后两个作业是 280104、280582）。我在 evidence 各 root 放了 PAUSE 文件，并停掉了 observer 23973。
- **文档**：Codex 的入口文档已备份为 `*_codex`，新建了 `CLAUDE.md`、研究总结和运维文档。
- **干扰复算**（N4 原始 cadence）：Code 驻留期间 Conv 每 token 176–186ms，驻留外 19.3ms。Conv 对 no_move 的 +22.6% 退化中，约 18 个点来自跨 tenant 干扰。

## 2026-11-03 xplace 代码与 simulator

- **运行时**：泛化为 N rank，所有 DP rank 一起参与 `all_gatherv`。新写了 planner、trace runner、校准、拟合和入口。
- **本地测试**：15 个全部通过，包括假引擎端到端跑完全部阶段、4-rank 多线程 peer transfer、planner。
- **simulator 的 DP=2 验证**（未调参，直接用 DP=2 拟合先验）：

  | 场景 | 指标 | 结果 |
  |---|---|---|
  | 6 个 native case | 迁移策略 Code TTFT | 模拟与真实相差 ≤3% |
  | 6 个 native case | Conv TPOT | 模拟系统性偏低 5–10% |
  | 6 个 native case | no_move Code TTFT | 模拟偏低 2–17% |
  | Poisson 259091/260757 | Conv TPOT | 模拟偏低 5–6% |
  | Poisson 259091/260757 | no_move Code TTFT | 不可比：真实值受旧控制器 stale-cache bug 影响，Code 完成后的空 rank 仍被视为满载 |

- **4 卡初步扫描**（用 DP=2 先验外推，待 Job 1 校准）：
  - 40 GiB 自然预算：在利用率 0.8–0.9 时 Conv 也频繁全局排队，A 效果被稀释，B 效果约 ±2%。
  - 16 GiB 上限、Conv 0.08 RPS、Code 0.004 RPS（利用率约 0.63）：
    - A：Code TTFT 降 40–60%；
    - B：Conv 退化 `mig_random` +5%、`ca_full` +2%。
- **Job 1 = 281989**（tag `j1-calib-20261103`，4×A800，时限 4:30）：00:21 提交，此时 A800 空闲为 0，在排队。
  - 阶段依次为 dataplane → grid（23 个 case）→ interference（Conv 每 rank 0/2/8/16）→ fit → pilot trace。
  - pilot：`pilot-b16-c008-k004-s101`，16 GiB 上限，策略 no_move、mig_random、ca_full。

## 2026-11-03 Job 1 = 281989：只跑了 dataplane（提交方式 bug）

- **原因**：sbatch 的 `--export` 按逗号拆分，`XPLACE_PHASES` 被截成 `dataplane`，`XPLACE_POLICIES` 被截成 `no_move`。作业 00:46–00:51 以 exit 0 结束。
  - 修复：提交时把值里的逗号编码成 `+`，sbatch 中再还原。
- **这次运行已确认的内容**（证据在 `evidence/xplace-j1-calib-20261103/job-281989`）：
  - DP=4 两个 tenant 正常启动。每个 worker 权重 16.43 GiB，CUDA graph 2s 捕获完成，拓扑校验通过（EP 分片互补，同 rank 共卡）。
  - 自然 KV 预算为每卡 [34.25, 36.75, 36.5, 36.75] GiB（启动后空闲 38–40.6 GiB，减 3 GiB 余量）。
  - **4 卡数据面通过**：8 次跨 rank 迁移（0→1、0→3、2→1、3→0 及迁回）全部通过逐字节 echo，请求按精确长度完成。每次迁移 0.6 GiB 耗时 36–80ms，其中 peer copy 25–64ms。
  - 事务耗时 3.4–4.2s，主要来自空闲的 Code tenant 到达暂停边界（2.9–3.5s）；忙碌的 Conv tenant 只要 0.25–0.47s。
- **据此改进：只暂停有请求被移动的 tenant。**
  - 回迁只暂停 Conv；准入时暂停哪些 tenant 由预规划的 moves 决定。
  - 未暂停的 tenant 不会新增 KV，而且准入受锁串行化，所以它的快照对容量判断是保守的。
  - 额外好处：一个 Code 正在 prefill 时，新到的 Code 也能触发迁移，不再被整体推迟。
  - planner 的准入只移动另一个 tenant 的请求（移 B 给 A 腾空间）。
  - simulator 已同步上述语义；15 个测试通过；DP=2 验证结果不变。
- **Job 1b = 282105**（tag `j1b-calib-20261103`）01:00 提交。阶段与计划相同，dataplane 的迁回步骤会走"只暂停 Conv"的新路径。

## 2026-11-03 Job 1b = 282105：校准完成，pilot 进行中

- **启动与 dataplane**：
  - 自然预算为每卡 [34.25, 36.75, 36.75, 36.75] GiB。
  - dataplane 再次通过。只暂停 Conv 的新路径（迁回事务）总耗时 0.72s，暂停等待 0.55s；两个 tenant 都暂停时是 3.9s。
- **grid（23 个 case 全部测到）**：作业内的拟合把 padding 效应混进了 attention 项，相对误差 7.2%。改进后的拟合相对误差 1.8%（RMSE 0.37ms），结果如下，写入 `evidence/xplace-inputs/step-params-dp4-v1.json`：
  - 有 padding 时的基础耗时：G1 10.3ms、G8 14.9ms、G16 16.4ms、G32 19.6ms；
  - 所有 rank 恰好满档时再快 3.0/4.1/4.5ms；
  - attention 斜率 6.1e-8 s/token（按最大 rank 的 context 计）。
- **跨 tenant 干扰**（Code 在 rank 0，Conv 均衡分布；4 卡比 2 卡严重得多）：

  | 每 rank Conv 数 | Code prefill 期间 Conv 每 token | Code decode 期间 Conv 每 token | Code prefill 时长 | Code TPOT |
  |---|---|---|---|---|
  | 2 | 57ms | 293ms | 44s | 249ms |
  | 8 | 49ms | 205ms | 44s | 192ms |
  | 16 | 176ms | 293ms | 61s | 250ms |
  | 无 Conv | — | — | 24.8s | 16.4ms |

- **simulator 用 4 卡参数后的结论**：
  - pilot 区间（16 GiB / Conv 0.08 / Code 0.004）几乎不会阻塞，因为 4 卡上 Conv 更快、寿命更短。
  - A 效果只出现在接近 Conv 饱和的窄带。例如 36 GiB / Conv 0.18 / Code 0.002：Code TTFT 平均值 124s → 62–77s，P95 326s → 63s。
  - **B 效果（Conv TPOT）在所有区间都 ≤2%**，`mig_random` 和 `mig_bestfit` 也差不多。迁移造成的失衡发生在 Code 驻留期间，被跨 tenant 干扰掩盖。
  - Conv TPOT 相对 no_move 在 ±5% 以内。真正的代价是 Conv TTFT 尾部：Code 提前准入挤占了 KV。
- **新增 `mig_bestfit` 基线**：迁移次数最少，目的地 best-fit，即只看 KV 的装箱器。
- **Job 1c = 282306**（tag `j1c-ilayout-20261103`）：设为在 282105 结束后才开始。
  - 要回答的问题：干扰是否主要来自 Code 所在 GPU 上的 Conv？
  - 方法：均衡布局与"隔离"布局（Code 所在卡上没有 Conv、总量相同）对比，每 rank 4/8/12/16 个 Conv，各重复两次。
  - 如果隔离能显著降低干扰，cost-aware 准入就可以多搬一些 Conv 把 Code 所在的卡清空，从而得到大得多的 B 效果；否则 B 只能维持在约 2% 的量级，需要如实报告。

## 2026-11-03 pilot（Job 1b）与 Job 1c 结果；Job 2 已提交

- **Job 1b = 282105**：COMPLETED 0:0（01:32–03:21）。pilot 区间为 16 GiB 上限、seed 101、1500s，评估窗口 [300, 1500)，窗口内 6 个 Code。

  | 策略 | Code TTFT 平均 / P95 | Conv TPOT | Conv TTFT P99 | 控制开销 |
  |---|---|---|---|---|
  | no_move | 148s / 198s | 30.9ms | 20.6s | — |
  | `mig_random` | 58s / 75s | 29.6ms（−4.1%） | 31.9s | 3 次准入事务，11 次迁移 |
  | `ca_full` | 60s / 76s | 28.1ms（−9.1% vs no_move，−5.2% vs `mig_random`） | 28.0s | 6 次准入 + 7 次回迁，30 次迁移，控制 27.5s，暂停 451 request-s |

  - 迁移类策略的 Conv TPOT 优于 no_move。原因是 no_move 下 Code 排队后成批进入，多个 Code 同时 decode 时 Code TPOT 达 0.9–1.05s，干扰最重；迁移让 Code 随到随进，在时间上错开。
  - **simulator 未通过检验**。按 Code 状态拆分 no_move：
    - 无 Code 的时间占 64%，Conv 每 token 22ms；
    - Code prefill 占 24%，Conv 每 token 150–177ms；
    - Code decode 占 6%，Conv 每 token 约 280ms。

    模型对 prefill 阶段干扰低估了约 4 倍。
- **Job 1c = 282306**：COMPLETED。Code 在 rank 0，对比均衡布局与"隔离"布局（Code 所在卡上没有 Conv、总量相同），各重复两次。
  - Code prefill 期间 Conv 每 token：均衡 124/75/125/216ms，隔离 200/221/198/243ms（对应每 rank 4/8/12/16 个）。
  - 隔离下 Code TPOT 也更差。
  - **结论**：隔离会加重干扰，干扰随 Conv 的集中程度上升。cost-aware 应该少挪、均匀分散，这正是现有 planner 的做法。
  - 两次重复之间噪声较大，例如均衡 4 个时两次分别是 190ms 和 58ms。
- **Job 2 = 283370**（tag `j2-main-20261103`）：10:33 提交，时限 12 小时。
  - 配置按 protocol §4.2 冻结：trace 为 seed 202、5300s，388 个 Conv、24 个 Code，其中 23 个落在评估窗口内；16 GiB 上限；6 个策略。
  - 预计每个策略约 95 分钟，总计约 10 小时。

## 下一步（11-03 上午版，已被下文取代）

1. Job 1 结束后用 `xplace_ctl.py fetch` 取回，核对 SHA，然后运行：

   ```
   python3 sim/job1_report.py evidence/xplace-j1b-calib-20261103/job-282105 --trace evidence/xplace-inputs/pilot-b16-c008-k004-s101.json
   ```

   重点看：dataplane 在 4 个 rank 间是否通过、自然预算、步时拟合残差、干扰幅度、pilot 的 simulator 误差。
2. 用 DP=4 参数重新扫描区间，按 protocol §4 冻结 Job 2。

## 等待 Job 1 期间的 simulator 观察（DP=2 先验，结论待 DP=4 校准后复核）

- **可用区间很窄**：利用率约 0.65 时 Code 基本能直接放下；约 0.9 时 Conv 自身也开始全局排队，TTFT P99 达上百秒，系统不稳定。
- **迁移会把排队从 Code 转给 Conv**：Code 提前占用 9.4 GiB，Conv 的 TTFT 尾部变差。16 GiB 上限下 Code 约占总预算的 15%，这个效应最明显；预算越大越弱。
- **高 batch 段的步时决定区间**：G≥32 时线性外推下 Conv TPOT 达 45–56ms，会拉长寿命、加剧过载。这一段必须用 Job 1 的网格实测。
- **Job 2 时长估计**：每个策略需要 ≥15–20 个 Code，Code 0.004 RPS × 4200s，加 drain 约 85 分钟。6 个策略约 9 小时，申请约 11 小时。

## 2026-11-03 下午：simulator 重新校准；Job 2 撤销，改为分时 + ≤1h 短作业

- **接手时 Job 2 = 283370 一直 PENDING**（Priority），Slurm 估计 11-04 17:22 开始。
- **pilot 的 B 效果来自哪里**（`tools/decompose.py`，按 Code 状态拆分 Conv avg TPOT）：
  - `ca_full` 比 `mig_random` 低 1.54ms，其中 1.53ms 来自无 Code 时段；
  - 无 Code 时段的步时：G8 档约 21ms，G16 档约 24.5ms。`ca_full` 在 G16 档的时间占 17%，`mig_random` 占 57%；
  - Code prefill/decode 时段的贡献两者相同。
- **干扰实测**（pilot）：
  - Code prefill 期间 Conv +225ms/步，decode 期间 +269ms/步；
  - Code 与 Conv 并行时 decode 225ms/token（单独 16ms），prefill 59s/100K（单独 25s）；
  - 每个 Code episode 的干扰强度差异很大（单个 Code 段均值 95–389ms）。Job 1b/1c 中有两种模式：Conv 50–85ms 同时 Code prefill 44–53s，或 Conv 175–260ms 同时 Code prefill 58–65s。
- **simulator v2**：
  - `sim/fit_live.py` 用 pilot `no_move` 拟合真值参数（`sim/truth-pilot-nm-v2.json`）：无 Code 步时 = 10.8ms + G8 6.5 / G16 9.7ms + 5.8e-8·最大 context，RMSE 0.8ms；`mig_random`/`ca_full` 作为留出验证；
  - 迁移平时 0.08s，Code prefill 期间 3.6s；
  - 每个 Code episode 的干扰强度加噪声（CV 0.37）；
  - planner belief 仍用冻结的 v1。
  - `sim/validate_live.py` 用 30 个噪声副本检验：Conv TPOT 与实测相差 0.2–3.3%，实测值落在分布内；`mig_random` 的 Code TTFT 比分布下界低 3%；Conv TTFT 不吻合（控制面延迟没有建模）。
- **Job 2 预测**（`sim/predict.py`，seed 202）：A −62%；B −2.1%，[−4.6, +0.2]，P(<0)=0.93；`ca_admit` ≈ 0。
  - 区间扫描（seed 303/404，`sim/sweep_live.py`）：B 在 −2.2% 到 +0.7% 之间；Code 速率 ≥0.006 时过载。
  - 在 seed 303/404 上把 planner belief 换成 v2，结果不变。
- **分时**：
  - vLLM DP 暂停后设置 `ignore_start_dp_wave`，`add_request` 只在 `UNPAUSED` 时启动 wave，所以暂停期间 engine 不跑 dummy batch；
  - simulator 预测（seed 101/303/404）：Conv TPOT −23~27%，`ca_full_tm` 的 Code TTFT −80%；代价是每个 Code 让 Conv 停顿约 26s。
- **用户决定**：
  1. 加入分时；
  2. 每个作业 ≤1h，撤销 Job 2，改为短作业逐个提交。
- **283370 已 scancel**（CANCELLED，运行时间 0）。
- **实现**：
  - `controller.set_hold`、`run_transaction(hold_after=...)`（持有期间不重复暂停、不提前恢复）；
  - `trace_runner` 新增 `*_tm` 策略，统计驻留中的 Code 数，记录 `holds.json`；
  - 新增 2 个 fake engine 测试，fake 对重复暂停会报错。17 个测试全部通过；fake 中 Code 运行期间 Conv 的重叠步数从 64 降到 0–1。
- **短 trace**：`short-b16-c008-p180-s701.json`，Conv Poisson 0.08 共 1100s，Code 在 300/480/660/840s，见 protocol §6。每个 trial 约 1450–1600s，simulator 对 pilot 墙钟时间的估计误差约 1%。
- **预测**：`evidence/xplace-inputs/forecast-short-s701-v2.json`，见 protocol §6.4。
- **S1 = 283685**（tag `s1-tm-20261103`）13:19 提交，`--time=01:00:00`，策略 `ca_full_tm`, `ca_full`。PENDING，Slurm 估计 11-04 21:35，该估计不可靠。

## 下一步

1. S1 结束后 `xplace_ctl.py fetch s1-tm-20261103 283685`，然后运行：

   ```
   python3 tools/report_main.py evidence/xplace-s1-tm-20261103/job-283685/out/trace --end-s 1100
   ```

   检查：每次迁移 `bitwise_verified`、输出长度精确、holds 次数等于 Code 数，并与预测对照。
2. 提交 S2（`no_move_tm`, `no_move`），之后 S3（`mig_random`, `ca_full`）。上传复用同一个 tag，只改 `XPLACE_POLICIES`。
3. **可选优化（本 campaign 不改，以保证 S1–S4 的分时实现一致）**：直接准入路径改为"先暂停 Conv 再准入 Code"。现在是先准入，Conv 要走到下一个 32 步边界才暂停，Code prefill 期间约有 3.8s 的并行；改后预计 Code TTFT 再降约 2s，Conv TPOT 改善 1–2%。

## 2026-11-03 14:30 S1 失败（慢节点）→ S1b

- **S1 = 283685**（d1n41a23g03，13:50–14:24）：`ca_full_tm` trial 在 1800s drain 超时时还剩 8 个 Conv，job 按 fail-fast 退出（exit 1）。已归档到 `evidence/xplace-s1-tm-20261103/`，SHA `135b6ec9…`。
  - **分时在真机上工作正常**：第一个 Code 经迁移准入，控制 1.7s；Conv 持有暂停 31.9s；Code prefill 28.1s（pilot 并行时约 59s），TTFT 29.9s（pilot 约 60s）。
  - **失败原因是节点**：Conv 步时在空载时 39–45ms（pilot 11ms），中段 56–74ms（pilot 24.5ms）。活跃 Conv 涨到 51 个，KV 占满，Code 2、3 排队约 10 分钟。节点 FreeMem 34 GB，CPULoad 16。
- **对策**：
  1. 换成更稳健的 trace（Conv 0.07 RPS，1000s，seed 702）；
  2. 新增节点探针，超过 16.5ms 中止；
  3. 排除 d1n41a23g03；
  4. drain 超时 800s。

  见 protocol §6.6。新增 1 个测试，共 18 个通过。
- **S1b = 283933**（tag `s1b-tm-20261103`）14:34 提交，策略 `ca_full_tm`, `ca_full`。
- **S2 = 283987**（`no_move_tm`, `no_move`）14:48 提交，依赖 `afterany:283933`，S1b 结束后才会开始，始终只有一个 allocation。
- 监视：一个持续运行的 campaign 监视覆盖所有作业（`/tmp/xplace_jobs.txt` 清单，每 90s 查一次状态、节点、探针和 trial 进度）。
- 合并分析：`tools/combine_runs.py` 用软链接把多个作业的 trial 合到同一个目录，然后用 `report_main.py` 和 `vs_forecast.py` 一起出表。每个作业的探针值写在表旁，用来说明跨作业比较的可比性。

## 2026-11-03 晚：S1b 运行；S2–S5 依赖链

- **S1b = 283933**：排队 14:34–21:10（gpu_a800 只剩 1 张空闲卡；hp_a800 我们的组无权限）。21:10 在 d1n41a15g02 开始（FreeMem 148 GB，CPULoad 10）。
  - **探针 12.6ms**，pilot 节点为 12.7ms，与 pilot 同速；
  - `ca_full_tm` trial 21:14 开始。
- **依赖链**（都用 afterany，执行上始终只有一个 allocation；同一 upload `s1b-tm-20261103`、同一 trace）：
  - S2 = 283987（`no_move_tm`, `no_move`）
  - S3 = 285331（`ca_full`, `no_move`）
  - S4 = 285332（`mig_random`, `ca_full`）
  - S5 = 285335（`ca_admit`, `ca_return`）
- 提交时踩到一个坑：zsh 不会对未加引号的 `$VAR` 做分词，env 串被当成一个参数，sbatch 报 "Unable to open file"，没有产生作业。改用 bash 数组后正常。
- **S1b 的 `ca_full_tm` trial**（21:14–21:35，passed，墙钟 1260s）：
  - 77/77 输出长度精确；2 次迁移都通过逐字节校验；
  - 4 次持有暂停，每次 23.8–25.8s，持有期间 Conv 一个 token 都没出；
  - Code prefill 24.6–25.2s（单独 24.8s），Code 3 为 30.6s：它是直接准入的，Conv 走到暂停边界前与它并行了 6.95s；Code decode 16.5–16.7ms/token（单独 16.4ms，并行 225ms）；
  - 与预测对比：Code TTFT 26.8s（预测 27.3 [26.4, 28.7]），Conv TPOT 22.2ms（预测 22.2 [22.0, 22.4]），Conv TTFT 3.7s 偏高（预测 2.8 [2.2, 3.4]）；
  - 代价：Conv 最大 token 间隔 p50 为 25.1s。
  - 后续可选优化："先停 Conv 再准入 Code"，可以去掉直接准入时最多约 7s 的并行。本 campaign 不改。
- **S1b 完成**（21:10–21:56，exit 0，两个 trial 都 passed）。已归档到 `evidence/xplace-s1b-tm-20261103/job-283933`，SHA `bccd910f…`；报告 `report.json`。
  - `ca_full`：Code TTFT 59.9s；Conv TPOT 31.0/40.6/44.4ms；Conv TTFT 1.56/8.9s（avg/p99）；10 次迁移，全部逐字节校验通过。
  - `ca_full_tm`：Code TTFT 26.8s；Conv TPOT 22.3/25.4/25.8ms；Conv TTFT 3.73/28.9s；Conv 最大 token 间隔 p50 25.1s。
  - 分时效果（同机配对）：Code TTFT −55.2% [−59.4, −50.7]（n=4），Conv TPOT −28.2% [−31.4, −24.8]（n=49）。
  - Conv TPOT 拆分：`ca_full` 中干扰贡献 10.4ms；`ca_full_tm` 中干扰贡献 0.15ms、持有停顿 2.1ms。
  - 两个 trial 的 Code TTFT 和 Conv TPOT 都落在预注册预测区间内。
- **S2 完成**（21:58–22:42，同一节点 d1n41a15g02，探针 12.7ms，exit 0）。已归档，SHA `5b151697…`。两个 trial 都是 77/77 输出长度精确。
  - `no_move`：Code TTFT 113.6s（预测 110.5 [96.5, 128.0]）；Conv TPOT 32.2/47.0/51.7ms（预测 30.7 [29.3, 32.3]）；Conv TTFT 1.81s。
  - `no_move_tm`：Code TTFT 53.4s（预测 29.1，偏离：Code 1 排队 93s，live 中的 KV 碎片比 simulator 严重）；Conv TPOT 22.5ms；Conv TTFT 4.64s。
  - 不迁移时的分时效果（同机配对）：Code TTFT −53.0% [−70.3, −44.5]；Conv TPOT −30.3% [−34.0, −26.5]。
  - **S1b 与 S2 的跨作业比较**（同一节点、探针 12.6/12.7ms）：
    - A（`ca_full` 对 `no_move`）：Code TTFT −47.3% [−68.8, −8.9]，Conv TPOT −3.8% [−7.0, +0.2]；
    - 分时下的 A（`ca_full_tm` 对 `no_move_tm`）：Code TTFT −49.8% [−73.2, +4.8]，Conv TPOT −0.9%；
    - 组合（`ca_full_tm` 对 `no_move`）：Code TTFT −76.4% [−85.7, −58.6]，Conv TPOT −30.9% [−34.8, −26.9]。
- **S6 = 285674**（`no_move_fair`, `mig_random_tm`），依赖 `afterany:285335`。用途：A 对公平基线的比较；分时下 cost-aware 与随机迁移的比较。
- S3 = 285331 在 22:43 PENDING（Priority）。上一个节点释放后没有分给我们。
- **S3 完成**（23:16–00:01，d1n41a15g02，探针 12.6ms，exit 0）。已归档，SHA `74789e95…`。两个 trial 都是 77/77 输出长度精确，9 次迁移全部逐字节校验通过。
  - `ca_full`：Code TTFT 57.9s，Conv TPOT 30.3ms；
  - `no_move`：Code TTFT 104.5s，Conv TPOT 30.6ms；
  - A 同机配对：Code TTFT −44.6% [−69.4, −7.2]，Conv TPOT −1.2% [−4.1, +2.4]。
  - 重复性：`ca_full` 两次为 59.9/57.9s、31.0/30.3ms；`no_move` 两次为 113.6/104.5s、32.2/30.6ms。Conv TPOT 跨运行相差 2–5%。
- **S4 完成**（00:01–00:48，d1n41a15g02，探针 12.6ms，exit 0）。已归档，SHA `9e1f7903…`。两个 trial 都是 77/77 输出长度精确，16 次迁移全部逐字节校验通过。
  - `mig_random`：Code TTFT 52.9s，Conv TPOT 30.1ms。两项都优于预测区间（61.3 [57.4, 64.5]、32.8 [31.6, 34.0]）。
  - `ca_full`：Code TTFT 59.5s，Conv TPOT 30.9ms。
  - **B 的同机配对**：`ca_full` 对 `mig_random`，Conv TPOT +2.4% [+1.2, +3.8]，Code TTFT +12.6%。方向与预测相反。
    - 拆分：无 Code 时段 `ca_full` 好 0.38ms（步时 22.2 对 23.1ms，G16 占比 0.31 对 0.54），说明机制本身生效；
    - Code prefill 期间 `ca_full` 多 1.07ms：`mig_random` 这次 4 个 Code 中有 3 个落在轻干扰模式，prefill 平均 52.0s，其余 trial 为 57.5–60.3s。
  - **干扰模式分析**（S1b–S4 加 pilot，40 个单 Code prefill episode）：
    - 每 rank 最大 Conv 数 ≥10 时，轻干扰模式比例为 8/12；8–9 时为 3/17；<8 时为 2/11；
    - Code 所在 rank 上的 Conv 数与干扰无关（r=−0.03）；
    - 按策略：`no_move` 2/14、`mig_random` 5/9、`ca_full` 6/17。
    - 这只是相关关系，有 Code 到达时刻等混杂；Job 1c 的均衡布局实验给出过相反趋势。作为假设报告。
  - **合并重复看 Conv TPOT**：`no_move` 31.4ms（2 次），`mig_random` 30.1ms（1 次），`ca_full` 30.7ms（3 次）。迁移类策略都没有让 Conv 退化。
- **分时下 B 是否可测**（simulator，非最终 seed 801–803，各 3 个噪声副本）：`ca_full_tm` 对 `mig_random_tm` 的 Conv TPOT 在 λ=0.08/0.09/0.10/0.11/0.12 时分别为 −2.4/−1.9/−1.4/−1.3/−1.4%。λ 升高后两者都进入 G16 档，差距反而缩小；`no_move_tm` 的 Conv TPOT 最低（没有迁移暂停，布局也均衡）。结论：placement-only 的 B 在所有区间 ≤2.5%，不再为 B 追加作业。
- **S5 = 285335** 00:48 在 d1n41a15g02 开始，探针 12.6ms，先跑 `ca_admit`。
- **规则更新（用户 11-04 01:2x）**：以后把多个 ≤1h 的实验打包进同一个 Slurm 作业，避免作业之间被别人的任务卡住。实现方式：`XPLACE_POLICIES` 列出所有实验的 trial，`--time` 取各实验之和加启动余量。S6 已在依赖链上，不再改动。
- **S5 完成**（00:48–01:32，d1n41a15g02，探针 12.6ms，exit 0）。已归档，SHA `748ea633…`。两个 trial 都是 77/77 输出长度精确，16 次迁移全部逐字节校验通过。
  - `ca_admit`：Code TTFT 55.9s；Conv TPOT 29.5ms；无 Code 时段步时 22.0ms，G16 占比 0.28。
  - `ca_return`：Code TTFT 51.7s；Conv TPOT 27.9ms，是所有非分时 trial 中最低的；无 Code 时段步时 21.5ms，G16 占比 0.11。
  - 两者都低于预测区间：这两轮的干扰大多是轻模式。
  - 无 Code 时段对 Conv TPOT 的贡献：`no_move` 21.10、`mig_random` 20.85、`ca_full` 20.47–20.54、`ca_admit` 20.09、`ca_return` 19.65ms。
- **每轮压到 20 分钟**（用户 01:4x）：把 seed 702 trace 截到 850s，即 `short-b16-c007-p180-s702-d850.json`，SHA `56ea1ec8…`，与 1000s trace 的前 850s 完全相同。
  - 60 个 Conv，Code 仍在 300/480/660/840s；评估窗口 [300, 850)，窗口内 36 个 Conv。
  - simulator 对 1000s 各轮墙钟的估计误差 ≤0.5%；估计 850s 每轮 18.4–19.9 分钟，最长 20.4 分钟。
  - 用户决定 S6 照旧用 1000s trace，保持与 S1b–S5 可比；850s 只用于以后的实验。
- **S6 = 285674** 01:34 在 d1n41a15g02 开始，探针 12.6ms，先跑 `no_move_fair`。

## 2026-11-04 算法优化轮 v2

- **用户要求**（01:5x）：S6 之后继续分析并优化算法。
- **分析结论**（详见 protocol §7.1）：
  - Conv TTFT 的主要来源是控制面阻塞：在 Code prefill 期间到达的 Conv 排队 2.2–3.2s，无 Code 时 0.15–0.20s；
  - 分时下直接准入的 Code 会与 Conv 并行 3.7–7s；
  - placement 已接近最优：`ca_full` 的可避免 G16 占比为 0.07–0.13。
- **v2 实现**：
  - `controller.py`：缓存 Code 视图并用 `code_epoch` 防止低估；`hold_locked`；`try_direct_admission(before_admit=...)`；
  - `trace_runner.py`：新增 `*_v2` 策略，Code 完成时作废缓存，监视按 epoch 刷新；
  - simulator：支持 `hold_first`，并修正"Code 等待期间被误算为干扰"；
  - 测试共 22 个通过。
- **预测**：`forecast-v2-d850-s702.json`。`ca_full_tm_v2` 对 `ca_full_tm`：Code TTFT −7.1% [−9.5, −4.2]，Conv TPOT −1.8%。
- **V2 = 286379**（tag `v2-bundle-20261104`，850s trace，5 轮：`ca_full_tm_v2`, `ca_full_tm`, `ca_full_v2`, `ca_full`, `no_move`，`--time 2:15:00`）。02:0x 提交，依赖 `afterany:285674`。
- **S6 完成**（01:34–02:19，d1n41a15g02，探针 12.6ms，exit 0）。已归档，SHA `1094373f…`。
  - `no_move_fair`：Code TTFT 96.0s，Conv TPOT 31.8ms，Conv TTFT p99 79s（公平门挡住了 Conv 准入）；
  - `mig_random_tm`：Code TTFT 26.2s，Conv TPOT 22.26ms，与 `ca_full_tm` 无差别。
- **S1b–S6 汇总**：`evidence/xplace-s1b-tm-20261103/final-tables.md`；结论写入 `reports/results-short.md`。
  - 12 个 trial 全部输出长度精确，55 次迁移全部逐字节校验通过。
  - 消融中 `ca_return` 最好（无 Code 时段 G16 占比 0.11），但它的 Code prefill 偏短（50.9s），部分来自干扰模式的运气。
- **V2 = 286379** 02:19 在 d1n41a15g02 开始。
- **V2 第一轮 `ca_full_tm_v2`**（02:22–02:41，passed）：
  - 先停 Conv 生效：直接准入的 Code，Conv 的暂停不晚于 Code 提交，到达边界 0.08–0.44s；4 个 Code 的 prefill 都是 24.0–25.1s；
  - 快路径：60 次 Conv 准入中 59 次命中缓存；持有期间到达的 Conv 排队 0.13s（v1 为 0.75–0.95s）；
  - Code TTFT 25.1s（预测 25.2s），Conv TPOT 22.7ms（预测 22.4ms）；64/64 输出长度精确，3 次迁移全部逐字节校验通过。
- **v3 分片分时**：
  - simulator 选定 S=5、R=1：预计 Conv TTFT p99 23.5 → 3.2s，Conv SLO 0.78 → 1.00，Code TTFT 25 → 30s；
  - 实现：`controller.pause_native`、`release_locked`，`trace_runner` 中的 slicer；
  - fake 增加 chunked prefill；测试共 23 个通过。
- **V3 = 286584**（tag `v3-sliced-20261104`，3 轮：`ca_full_tm_s5`, `ca_full_tm_v2`, `ca_full_v2`，`--time 1:15:00`）。02:5x 提交，依赖 `afterany:286379`。设计与预测见 protocol §8。
- **V2 中间结果**（同机配对，850s trace）：
  - `ca_full_tm_v2` 对 `ca_full_tm`：Code TTFT −5.3% [−11.0, −0.7]，Conv TPOT −0.5% [−0.7, −0.3]，Conv TTFT −6.9% [−11.1, −1.9]；
  - `ca_full_v2` 对 `ca_full`：Conv TTFT −51% [−63.8, −30.9]（avg 0.97 → 0.47s，p99 3.64 → 1.55s），Code TTFT +1.2%（不显著），Conv TPOT +3.9% [+1.7, +6.4]；
    - 拆分：无 Code 时段两者相同（19.56 对 19.59ms），差距全部在 Code prefill 期间（106 对 130ms/token），属于干扰模式噪声；
    - Code prefill 期间到达的 Conv 排队 0.33s（v1 为 2.2–3.2s）；60 次 Conv 准入全部命中缓存；单卡最大映射 15.94 GiB，未超过 16 GiB；
  - 一次未解释的 3.3s 控制面延迟：`ca_full_v2` 第一次准入事务中，Conv 在 0.095s 就已停到边界，但结果晚 3.3s 才回到控制器，engine 端回调是即时的，问题在客户端。12 次事务中只有这 1 次。
- **V3 改为 v3b**：给 trace_runner 加事件循环滞后监测（`loop_lag`）。旧 V3 = 286584 尚未开始，已撤销；新 **V3b = 286826**（tag `v3b-sliced-20261104`，策略相同），依赖 `afterany:286379`。
- **V2 完成**（02:19–03:58，d1n41a15g02，探针 12.6ms，exit 0）。已归档到 `evidence/xplace-v2-bundle-20261104/job-286379`，SHA `81f32729…`；报告 `report.json`。5 轮都在同一作业内，可直接配对：
  - `no_move` 111.6s / 34.0ms / Conv TTFT 1.36s；`ca_full` 56.1 / 30.5 / 0.97；`ca_full_v2` 56.8 / 31.7 / 0.47；`ca_full_tm` 26.5 / 22.8 / 4.39；`ca_full_tm_v2` 25.1 / 22.7 / 4.09。
  - A：`ca_full` 对 `no_move`，Code TTFT −49.7% [−70.1, −4.9]，Conv TPOT −10.3% [−15.3, −4.5]。
  - 最优：`ca_full_tm_v2` 对 `no_move`，Code TTFT −77.5% [−85.7, −62.8]，Conv TPOT −33.2% [−37.1, −28.9]，Conv TTFT +202%。
  - v2 对 v1：分时下 Code TTFT −5.3%、Conv TTFT −6.9%；不分时下 Conv TTFT −51%。
- **V3b = 286826** 03:58 在 d1n41a15g02 开始。
- **V3b 第一轮 `ca_full_tm_s5`**（04:01–04:19，passed）：分片分时在真机上工作正常。
  - 12 次切片（每个 Code 3 次），16 段持有，每段 3.8–8.2s；持有期间 Conv 一个 token 都没出；
  - Code TTFT 29.9s（预测 30.3s）；Conv TPOT 23.6ms（预测 22.7ms）；Conv TTFT 1.25/4.99/6.04s（avg/p95/p99；预测 p99 3.2s）；Conv SLO 0.94；最长停顿 8.2s；
  - 单次切片：Code 暂停 1.0–1.9s（越往后的 chunk 越慢），Conv 窗口 1.26–1.47s，每次切片 Code 停 2.3–3.3s；
  - 事件循环最大滞后 0.12s，没有卡顿；
  - 停顿超过设定 5s 的原因：slicer 先查 Code tenant（要等它当前步结束），再暂停 Code（又等一步）。可改为先暂停再检查，并提前触发。
- **simulator 切片模型按真机时序修正**：S=5 上与 live 的差距在 3–12% 以内。按 S 的权衡曲线选 S=4（Conv SLO 1.00，Code TTFT 31.6s）。
- **V4 = 286921**（tag `v4-final-20261104`，新 seed 704，850s，4 轮：`ca_full_tm_s4`, `ca_full_tm_v2`, `ca_full_v2`, `no_move`，`--time 1:45:00`）。04:34 提交，依赖 `afterany:286826`。预测与判定见 protocol §9。
- **V3b 完成**（03:58–04:57，d1n41a15g02，探针 12.6ms，exit 0）。已归档到 `evidence/xplace-v3b-sliced-20261104/job-286826`，SHA `527f39ab…`。3 轮都 passed，64/64 输出长度精确。
  - `ca_full_tm_s5` 29.9s / 23.6ms / Conv TTFT 1.25（p99 6.04）/ SLO 0.94；`ca_full_tm_v2` 25.4 / 22.7 / 4.17（24.6）/ 0.78；`ca_full_v2` 55.8 / 31.2 / 0.56（1.98）/ 1.00。
  - 分片对全程分时：Conv TTFT −70.1% [−77.9, −53.4]；Code TTFT +18.1%；Conv TPOT +4.0%。
  - 分片对不分时：Code TTFT −46.3%，Conv TPOT −24.3%。
  - 可复现：`ca_full_tm_v2` 在 V2/V3b 两次为 25.1/25.4s、22.72/22.70ms。
  - 事件循环最大滞后 ≤0.52s；V2 中那次 3.3s 延迟没有复现。
- **V4 = 286921** 04:57 在 d1n41a15g02 开始。
- **v3.1 按需切片**：只在某个持有中的 Conv 等首 token 超过 2s、或持有达到 15s 时切片；改为先暂停 Code 再检查。simulator 预测相对 S=4：Code TTFT −11%，Conv SLO 同为 1.00。测试共 24 个通过。
- **V5 = 286971**（tag `v5-slo-20261104`，seed 704，2 轮：`ca_full_tm_slo`, `ca_full_tm_s4`，`--time 0:55:00`）。05:03 提交，依赖 `afterany:286921`。见 protocol §10。
- **V4 第一轮 `ca_full_tm_s4`**（seed 704，05:00–05:20，passed）：
  - Code TTFT 30.7s（预测 31.7s），Conv TPOT 23.35ms（预测 23.3ms），Conv TTFT 0.72/4.05/6.52s（avg/p95/p99；预测 p99 5.4s），Conv SLO 0.95，15 次切片；
  - 61/61 输出长度精确，6 次迁移全部逐字节校验通过；
  - 事件循环卡顿 2 次：t=84.2s 1.0s、t=852.5s 3.6s，与 V2 那次 3.3s 延迟同类。t=84 时还没有 Code，所以与 Code 无关；simulator 中 planner 单次最多 1.2ms，排除了规划耗时；具体原因未知。
- **诊断**：trace_runner 增加看门狗线程（心跳超过 0.5s 不更新时抓主线程调用栈）和 GC 计时。自测中故意阻塞 0.8s 能被准确定位。旧 V5 = 286971 尚未开始，已撤销；新 **V5b = 287011**（tag `v5b-slo-20261104`，策略不变），依赖 `afterany:286921`。
- **V4 完成**（04:57–06:19，d1n41a15g02，探针 12.6ms，exit 0）。已归档到 `evidence/xplace-v4-final-20261104/job-286921`，SHA `e5add7df…`。4 轮都 passed，61/61 输出长度精确。
  - `no_move` 131.8s / 27.2ms；`ca_full_v2` 54.0 / 30.3；`ca_full_tm_v2` 24.9 / 23.2（Conv TTFT p99 24.7s，SLO 0.88）；`ca_full_tm_s4` 30.7 / 23.4（p99 6.52s，SLO 0.95）。
  - 对 `no_move`：
    - A（`ca_full_v2`）：Code TTFT −59.1% [−76.2, −6.8]，Conv TPOT +11.3% [+5.6, +18.0]，在 20% 内；
    - `ca_full_tm_v2`：−81.1%，−15.0%；
    - `ca_full_tm_s4`：−76.7%，−14.3%。
  - Conv 退化的拆分：`ca_full_v2` 中 Code 驻留时间占比 0.38，`no_move` 为 0.25（Code 排队时 Conv 不受干扰）。
  - 与 seed 702 一致：A −49 对 −59%；分时对不分时 Code TTFT −56 对 −54%，Conv TPOT −28 对 −24%。
  - 事件循环卡顿只出现在分时类轮次：`s4` 2 次（最长 3.6s），`tm_v2` 2 次（最长 0.56s），非分时为 0。
- **V5b = 287011** 06:19 在 d1n41a15g02 开始。
- **V5b 完成**（06:19–07:01，d1n41a15g02，探针 12.6ms，exit 0）。已归档到 `evidence/xplace-v5b-slo-20261104/job-287011`，SHA `ed9e4d0a…`。两轮都 passed，61/61 输出长度精确。
  - `ca_full_tm_slo`：Code TTFT 27.8s；Conv TPOT 23.39ms；Conv TTFT 0.67/3.21/4.10s（avg/p95/p99）；Conv SLO 1.00；7 次切片（5 次由首 token 等待触发，2 次由持有上限触发）。
  - `ca_full_tm_s4`：31.3s / 23.45ms / 0.85/4.30/6.59s / SLO 0.95；16 次切片。V4 中同一策略为 30.7s / 23.35ms / p99 6.52s，可复现。
  - v3.1 对 v3 同机配对：Code TTFT −11.2% [−13.3, −9.1]（预测 −11.3%），Conv TPOT −0.3%，Conv SLO 0.95 → 1.00。
  - v3.1 对 V4 的 `no_move`（同节点、同 trace、跨作业）：Code TTFT −78.9% [−87.0, −49.6]，Conv TPOT −14.1% [−18.3, −9.8]，Conv TTFT 无显著差异。
  - 两轮都没有事件循环卡顿（最大滞后 0.24s），看门狗没有捕获到调用栈；卡顿原因仍是开放问题。
- **campaign 结束**：队列已空，监视已停止。最终推荐策略为 `ca_full_tm_slo`；结论见 `reports/results-short.md`。

## 2026-11-04 上午：公平基线作业 V6

- **用户问题**：基线是否用最新代码重跑过。答复：没有。`no_move` 最后一次在 V4，且一直是 v1 控制面（没有快路径）；消融基线只在 S4–S6 跑过（v1）；`no_move_v2` 从未运行。最终对比因此混用了控制面版本，主要影响 Conv TTFT 的比较。
- **用户选择 6 轮方案，含消融基线**。新增策略 `mig_random_v2`，测试共 24 个通过；预测见 `forecast-v6-d850-s704.json`；协议见 §11。
- **V6 = 287445**（tag `v6-baselines-20261104`，seed 704，6 轮：`no_move_v2`, `ca_full_tm_slo`, `ca_full_v2`, `no_move`, `ca_full_tm_v2`, `mig_random_v2`，`--time 2:30:00`）。09:38 提交，立即在 d1n41a15g02 开始。监视脚本改为兼容 bash 3.2，全部作业结束后自动退出。
- **事件循环卡顿的根因已找到**：V6 看门狗抓到 5 次卡顿（0.6–1.1s），调用栈全部停在监视循环的 `save(root / "progress.json")`，即 `open()` / `write_text` / `replace`。输出目录在共享文件系统（`/data/run01`）上，元数据操作偶发卡顿，而写操作直接在 asyncio 事件循环里执行。另有一次 0.52s 的 gen-2 GC，贡献较小。与策略无关。
  - **修复 v3.2**：Journal 改为后台写线程，jsonl 与 progress 都只在事件循环里入队。本地模拟每次写盘阻塞 0.8s 时，事件循环最大滞后从 0.8s 降到 0.0017s。测试共 24 个通过。V6 仍按原代码跑完，以保持一致。
- **V6 前 4 轮**（09:42–11:04，全部 passed，61/61）：
  - `no_move_v2` 110.7s / 29.7ms / Conv TTFT 0.78s（p99 5.53s）/ Code SLO 0.75；
  - `ca_full_tm_slo` 27.7s / 23.6ms / 0.72s（p99 4.02s）/ SLO 1.00 / 1.00；
  - `ca_full_v2` 61.3s / 34.3ms / 0.60s；
  - `no_move` 已完成，待分析。
  - 同机配对 `ca_full_tm_slo` 对 `ca_full_v2`：Code TTFT −54.8%，Conv TPOT −31.2%。
- 监视脚本的 ssh 查询在 09:42–11:03 之间连续失败，事件延迟到 11:03 才补上；作业本身正常。
- **V6 完成**（09:38–11:43，d1n41a15g02，探针 12.7ms，exit 0）。已归档到 `evidence/xplace-v6-baselines-20261104/job-287445`，SHA `ed9b331e…`；报告 `report.md` / `report.json`。6 轮都 passed，61/61 输出长度精确。
  - `no_move` 107.1s / 31.9ms / Conv TTFT 1.64s（p99 14.3s）；`no_move_v2` 110.7 / 29.7 / 0.78（5.53）；`mig_random_v2` 57.9 / 32.1 / 0.57；`ca_full_v2` 61.3 / 34.3 / 0.60；`ca_full_tm_v2` 24.9 / 23.1 / 2.21（24.6）；`ca_full_tm_slo` 27.7 / 23.6 / 0.72（4.02）。
  - 快路径对基线：`no_move_v2` 对 `no_move`，Conv TTFT −52.6% [−61.8, −34.5]，Conv TPOT −6.7%。
  - **最终策略对公平基线**：`ca_full_tm_slo` 对 `no_move_v2`，Code TTFT −75.0% [−84.3, −50.6]，Conv TPOT −20.6% [−24.4, −16.9]，Conv TTFT −7.1%（不显著）。
  - 不分时的迁移：`ca_full_v2` 对 `no_move_v2`，Conv TPOT +15.4% [+8.7, +23.0]；`mig_random_v2` +7.8%。
  - **placement B**：`ca_full_v2` 对 `mig_random_v2`，Conv TPOT +7.1% [+5.5, +9.0]。与 S4（+2.4%）方向相同。无 Code 时段步时 −3%，但 Code 驻留期间干扰更重（prefill 60.7 对 57.2s）。
  - 6 次事件循环卡顿全部来自同步写 progress.json（v3.2 已修复）。

## 2026-11-04 下午：kvcached 原生基线 V7

- **用户要求**：加一个 kvcached 方式的基线（两个 vLLM 同时推理，不加协调）。
- **差异**：现有 `no_move*` 经过控制器（P+O 全量预留、控制器排队、按请求数路由、16 GiB 由准入实现）；原生 vLLM 按步分配、显存不够时抢占重算、按 `waiting × 4 + running` 路由，kvcached 按 GPU 实际剩余显存映射（留 2% 余量）。
- **V6 日志估计**：4 个 Code 到达时，按 Conv 实际 KV，放得下的 rank 为 4/4、4/4、4/4、2–4/4；按全量预留为 4/4、1–2/4、0/4、0–1/4。迁移的 A 可能主要是相对全量预留。
- **实现 `kvcached_native`**：
  - 请求直接交给 vLLM（`data_parallel_rank=None`），调度器切原生准入（`placement_set_native`）；
  - trial 前按 kvcached 余量公式占位，每卡共享容量 85.5 个页组（取整 15.94 GiB），trial 后释放；
  - 记录抢占（`_preempt_request` 计数和每请求 `num_preemptions`）、最大映射、每 60s 容量；rank 从快照反查；原生请求 `reserved_at` = 提交时刻。
  - 测试共 27 个通过（新增 3 个）；`report_main`、`check_tm` 能处理原生输出。
- **预注册**：protocol §13。
- **V7 = 288709**（tag `v7-native-20261104`，manifest `21bb2a6d…`，tarball `e4a05588…`，16:50 上传核对）。seed 704，4 轮：`ca_full_tm_slo`, `kvcached_native`, `ca_full_v2`, `no_move_v2`，`--time 2:00:00`，探针 16.5ms，排除 d1n41a23g03。16:5x 提交，排队中（Priority）。监视已启动。
- **V7 开始**：17:23 在 d1n41a18g02 开始（首次使用的节点；CPULoad 8.2，FreeMem 173 GB）。探针 12.8ms，通过（之前的节点 12.6–12.7ms）。17:28 第一轮 `ca_full_tm_slo` 开始。
- **V7 第一轮 `ca_full_tm_slo`**：17:28–17:46，passed。原生轮 17:46 完成占位：四卡容量 16.006/16.006/16.008/15.984 GiB，即 170 个 96 MiB 页组，合 15.94 GiB（protocol §13.4 更正了页组大小）。
- **V7 第二轮 `kvcached_native`**：17:46–18:07，passed，61/61 输出长度精确。占位释放后容量恢复到 35.8–38.3 GiB。
  - Code TTFT 54.1s（p99 60.1s），Conv TPOT 31.28ms（p99 39.6ms），Conv TTFT 0.25s（p99 1.10s），SLO 都是 1.00。
  - 抢占 0 次；kvcached 分配未命中警告 0 条。
  - 4 个 Code 都是到达即进入：提交延迟不超过 2ms，KV 在 2.7–5.6s 内开始增长。DP 负载均衡把它们依次分到 rank 0–3。
  - 到达时 Code 所在卡的已映射 KV 为 1.41、3.38、6.19、6.66 GiB。后两个 Code 的峰值为 15.84、15.75 GiB，距上限 15.94 GiB 只剩 0.1–0.2 GiB。
  - 每卡最大映射 [10.78, 12.47, 15.84, 15.75] GiB，没有超过 170 个页组。
  - 可映射容量每 60s 采样一次，最低为 [15.95, 15.58, 15.58, 15.42] GiB：激活内存占用了至多约 0.5 GiB 的 KV 空间。受控轮的激活不计入 16 GiB 预算，所以原生轮在这一点上略吃亏，但仍然没有抢占。
  - 无 Code 时段步时 21.57ms，16 档占比 0.32，其中可避免的只有 0.02：vLLM 原生的 DP 负载均衡已经把 Conv 数拉得很平。Code 驻留时间占比 0.39。
  - 对照预注册：Code TTFT 54.1s，略低于预测区间 55–70s 的下沿；Conv TPOT 31.28ms，在 31–36ms 内。
- **判定 1 满足**：`kvcached_native` → `ca_full_tm_slo`，Code TTFT −48.7% [−51.4, −46.4]，Conv TPOT −24.6% [−26.6, −22.7]，Conv TTFT +166% [+64, +320]（0.25s → 0.66s，SLO 都是 1.00）。
- **`ca_full_tm_slo` 跨节点复现**：27.7s / 23.60ms，与 V6 相同。v3.2 首次上真机，事件循环最大滞后 0.012s，0 次卡顿（之前同类轮次为 0.6–3.6s）。
- `tools/report_main.py` 的基线列表加入 `kvcached_native`。
- **V7 第三轮 `ca_full_v2`**：18:07–18:27，passed，61/61。16 次迁移、6 个事务，暂停 46 request·s；事件循环最大滞后 0.002s。
  - Code TTFT 54.3s，Conv TPOT 30.88ms，Conv TTFT 0.51s（p99 2.12s）。
  - **判定 2（迁移对原生准入）**：`kvcached_native` → `ca_full_v2`，Code TTFT +0.5% [−2.9, +4.9]，在 ±15% 内。按预注册写成：迁移相对原生准入没有 A 收益，A 只相对全量预留成立。
    - Conv TPOT −1.3% [−2.3, −0.6]；Conv TTFT +106% [+68, +160]（0.25s → 0.51s），这部分来自控制器准入路径和迁移暂停。
    - 无 Code 时段两者几乎相同：步时 21.44 对 21.57ms，可避免的 16 档占比都是 0.02。
  - 分时效果复现：`ca_full_v2` → `ca_full_tm_slo`，Code TTFT −48.9% [−50.2, −47.7]，Conv TPOT −23.6% [−25.4, −21.8]（V6 为 −54.8% / −31.2%）。
- **V7 第四轮 `no_move_v2`**：18:27–18:47，passed。Code TTFT 134.8s（4 个 Code 依次为 49.8、50.7、279.5、159.4s；后两个在控制器里排队 212.8s、110.2s），Code SLO 0.50。Conv TPOT 27.73ms，Conv TTFT 0.37s（p99 1.36s）。
- **V7 完成**（17:23–18:47，d1n41a18g02，探针 12.8ms，exit 0；启动测试 27 个通过；清单 SHA 核对通过）。
  - 已归档到 `evidence/xplace-v7-native-20261104/job-288709`，tarball SHA `ba0535f5…`；报告 `report.md` / `report.json`。
  - 4 轮都 passed，244/244 输出长度精确；23 次迁移全部逐字节校验通过；4 轮都没有抢占。
  - 4 轮事件循环最大滞后 0.002–0.211s，0 次卡顿：v3.2 在真机上有效。
  - **判定 3（全量预留的代价）**：`no_move_v2` → `kvcached_native`，Code TTFT −59.9% [−75.6, −0.7]，Conv TPOT +12.8% [+6.0, +19.9]。方向成立；Conv TPOT 在预测的 5–20% 内，Code TTFT 降幅大于预测的 30–50%。原生的 Conv TPOT 更高，是因为 Code 驻留时间占比更高（0.39 对 0.24），Conv 与 Code 并行的时间更长。
  - **判定 4（正确性）**：满足。原生轮最大映射 [10.78, 12.47, 15.84, 15.75] GiB，没有超过 170 个页组。
  - **最终策略对 `no_move_v2`**（V7）：Code TTFT −79.4% [−87.3, −46.8]，Conv TPOT −14.9% [−19.3, −10.2]，Conv TTFT +76.9% [+2.3, +173.8]（0.37 → 0.66s）。V6 中 Conv TTFT 为 −7.1%（不显著）。两次差异来自基线：V6 的 `no_move_v2` 有一次 6.8s 的 Conv 准入等待（Conv TTFT 0.78s），V7 为 0.37s。该轮最大滞后 0.37s，不是事件循环卡顿，原因未定，按轮次间波动报告。
  - 累计正确性：12 个作业、36 个 trial、2412 个请求输出长度全部精确；182 次迁移全部逐字节校验通过。
  - 监视已自动退出。

## 2026-11-04 晚：当前长度准入（不使用输出长度）

- **用户要求**（11-04）：
  - serving 系统不能假设已知输出长度，准入按当前长度；
  - 实验负载要设计成按当前长度长请求也进不来；
  - 请求变长、KV 不够时触发 vLLM 的 retract（抢占重算）；
  - placement 给请求留增长空间，但不能假设已知完整输出长度；
  - 新设定下重跑所有 baseline 和我们的系统。
- **事故**：查 vLLM 版本时在登录节点 ln01 上执行了 `python -c "import vllm"`，违反"登录节点禁止 import"的规定，运行约 1 分钟。已结束进程 3736874 及其父 shell 3736788，并核实无残留。之后只读 `_version.py`：vLLM 0.26.0，`scheduler_reserve_full_isl` 默认 True，即原生准入要求当前序列放得下。
- **代码（v4，当前长度设定）**：
  - 调度器三种模式：`full`（旧，prompt + max_tokens，max_tokens 是回放的真实长度，属于 oracle）、`prompt`（只预留 prompt + 固定 extra，之后按步增长，不够时由 vLLM 抢占）、`native`。新增抢占和重算 token 计数；
  - planner 去掉输出长度：剩余长度视为未知；挑选顺序改为 kv_desc / kv_asc，回迁按 context；
  - 增长余量 `conv_growth_tokens`：取"relief 15s 单独速率"与"长请求驻留窗口内的 Conv 增长"中的较大者。驻留窗口按分时、并行、切片三种情况分别计算；
  - 余量作用于 Code 的目标 rank 和迁移目的地，迁移后准入时再按余量复查；
  - 事务允许存在等待中或被抢占的请求；
  - 每轮都用占位把物理容量对齐到预算；
  - 测试 35 个通过。
- **simulator**：加入当前长度准入、原生负载均衡、retract 和重算。旧模式 5 个策略的输出与改写前逐位一致；原生模式在 V7 trace 上复现了 Code 到达时各 rank 的 KV（误差约 0.5 GiB 以内）、Code 落点和 0 次抢占。
- **负载选择**（非最终 seed 801–820）：
  - Conv 0.07 RPS（V7 负载）时，Code 按当前长度总能放进；
  - 14 GiB 上限或更多 Code 会让不分时的策略整体过载（Conv 被拖慢后积压，出现大量抢占）；
  - 选定：16 GiB、Conv 0.12 RPS、Code 在 420/540/660/780s、时长 900s。原生被挡比例约 0.4，`ca_full` 和分时策略 0 次抢占，每轮不超过约 25 分钟。
- **seed 规则**：从 901 起取第一个同时满足 Conv 总数在 108±5% 内、且 Code 到达前 400s 内的 Conv 数之和 ≥ 192 的 seed，结果为 911（105 个 Conv，前置 211）。
  - 冻结 `cur-b16-c012-p120-s911-d900.json`（SHA `ec488df7…`）和 `planner-params-v2.json`（SHA `00bcf6f7…`）。
- **预测**：`forecast-v8-cur-s911.json`（每个策略 20 个副本），见 protocol §14.6。对原生：`ca_full_tm_slo` 的 Code TTFT −79%、Conv TPOT −25%；`ca_full_v2` 的 Code TTFT −59%。
- **V8 = 289203**：
  - tag `v8-current-20261104`，manifest `dbf175fc…`，tarball `e9f7501e…`，21:18 上传并核对；
  - 12 轮，`--time 5:30:00`；
  - `XPLACE_KV_MODE=prompt`，extra 256，上限 16 GiB，drain 超时 1000s，探针 16.5ms，排除 d1n41a23g03；
  - 21:2x 提交，排队中（gpu_a800 当时 0 张空闲卡），监视已启动。
- **push**：用户同意 push 之前的两个 commit。查看远端发现 origin/main 已是 8859d68，两个 commit 都已在远端，无需再 push。
- **增长余量敏感性**（simulator，seed 801–810）：
  - 不留余量时，`ca_full_tm_slo` 抢占 1.7 次（最多 17），`ca_full_v2` 0.3 次；
  - v2 余量下两者都是 0；
  - 余量加倍反而多迁移、多抢占；
  - 随机迁移在各档都是 2.4–2.9 次；
  - Code TTFT 各档相差不超过 0.5s。
- **V8b = 289278**：
  - 余量消融，protocol §14.8；tag `v8b-headroom-20261104`，manifest `bee4847c…`，tarball `4302dd3e…`，21:48 上传；
  - 2 轮 `ca_full_v2_nh`、`ca_full_tm_slo_nh`，`--time 1:00:00`，依赖 `afterany:289203`；
  - 预测：`ca_full_v2_nh` 抢占 23.4 [3.9, 39.4] 次，`ca_full_v2` 为 0；分时这一对都是 0；
  - 与 V8 一起由同一个监视跟踪。测试 36 个通过。
- Slurm 估计 V8 于 11-05 18:47 开始（此类估计一般偏保守）；`--test-only` 对不同时限都给出 2082 年，没有参考价值，所以没有改时限。
- **V8 开始**：22:22 在 d1n41a28g03 开始（首次使用的节点；CPULoad 18.1，FreeMem 992 GB）。是否继续取决于探针（阈值 16.5ms）。
- **V8 探针 12.6ms**，通过（与 d1n41a15g02 同速）。第一轮 `ca_full_tm_slo` 占位后四卡容量 15.98 GiB（170 个页组），22:26 开始。
- **V8 第一轮中途检查**（`ca_full_tm_slo`，拷回本地分析）：
  - 设计项生效：Code 预留 101,931 = 输入 101,675 + 256（未用输出长度）；增长余量 0.55–0.83 GiB；切片由首 token 等待触发；0 次抢占。
  - 420s、540s 两个 Code 直接准入，TTFT 31.8s、30.9s，与 simulator 一致。
  - **660s 的 Code 迁移规划连续 104 次不可行**，0 次迁移；880s 时 660s、780s 两个 Code 仍在排队。660s 时各 rank Conv 占用 6.75–9.47 GiB、每 rank 13–14 个 Conv；Code 需 9.375 GiB + 约 1.0 GiB 余量，各 rank 的 Conv 需降到 5.6 GiB 以下。
  - **原因：页共享**。按当前长度逐步增长时，kvcached 给单个块挑"最满但有空位"的页，同一 rank 上的 Conv 块交错在同一批 2 MiB 页里，空位被别的 Conv 填上，页很少整页空出。Conv 映射量因此停在历史高点（rank 3 的 Conv 从 13 降到 11，映射一直是 10.12 GiB）；Code 属于另一个模型，用不了 Conv 页里的空位。只有把一张 rank 上的 Conv 全部迁走才能释放页，而 `max_moves`=8 小于每 rank 的 Conv 数。
  - kvcached 对全空的页立即释放（`defer_physical_release` 只推迟到本步结束），已排除"延迟释放"的解释。
  - simulator 假设每个请求独占自己的页（且旧设定下全量预留是一次分配，确实基本独占），所以没有预测到。实测进度与模拟一致（各时刻完成数相差 1–2，Conv TPOT 21.0 对 20.3ms），偏差只在页占用。
- **处理**：
  - 撤销 V8b（289278，未开始，测的也是迁移策略）；
  - V8 跑完第 2 轮（原生）和第 3 轮（`no_move_v2`）这两个不依赖迁移的有效基线后撤销；
  - 同时修设计：simulator 加页共享模型，planner 允许清空整张 rank，重新预测、预注册后提交 V9。
- **修复 v4.1**（本地，测试 37 个通过）：
  - **simulator 页共享模型**：Conv 页由该 rank 上所有持有块的 Conv 共有，映射按历史高点，只有 Conv 全部离开时才归零；Code 的大块分配仍按独占页。旧模式 5 个策略逐位一致。
    - V7 原生轮佐证：rank 2 的 Conv 从 8 降到 2，映射一直是 7.41 GiB。
    - 重现 V8 第一轮：660s 各 rank [6.66, 7.59, 8.72, 9.75] GiB，实测 [6.75, 7.69, 7.97, 9.47]；`max_moves`=8 时 660s 的 Code 852s 才进来。
  - **清空整张 rank**：`planner-params-v3.json`（SHA `32a5a49e…`）把 `max_moves` 从 8 提到 24。模拟中 660s 的 Code 在 661s 迁走 13 个 Conv 后进入，TTFT 27.7s。
  - **有 Code 驻留时不回迁**：模拟发现，清空一张 rank 后各 rank 的 Conv 数为 [17, 18, 16, 0]，回迁检查在 Code prefill 期间发起回迁；迁移与 prefill 争带宽（实测每次约 3.6s），事务持锁约 25s，新 Conv 无法准入（TTFT 27–37s）。改为有 Code 驻留时跳过回迁，Code 完成时那次回迁照常。修复后 `ca_full_tm_slo` 的 Conv TTFT p99 从 31s 降到 4.2s。
  - **seed 911 新预测**（页共享，v3 参数）：
    - `kvcached_native` Code TTFT 488s（4 个中 3 个被挡约 10 分钟），Conv TPOT 29.0ms；
    - `no_move_v2` 280s；
    - `ca_full_v2` 60.7s，Conv 39.5ms，0 次抢占；
    - `mig_random_v2` 60.7s，Conv 42.0ms，4 次抢占；
    - `ca_full_tm_slo` 30.3s，Conv 29.3ms，Conv TTFT 0.73s（p99 4.2s），0 次抢占。
- **V8 第一轮** `ca_full_tm_slo` 于 22:48 通过（数据待归档分析），22:48 原生轮开始。
- **V8 第一轮结果**（`ca_full_tm_slo`，`max_moves`=8，22:26–22:48，passed，109/109 输出长度精确，0 次抢占）：
  - 420s、540s 的 Code 直接准入，TTFT 31.8s、30.9s；
  - 660s、780s 的 Code 迁移规划 172 次全部不可行，最终在 Conv 排空后直接准入，TTFT 401s、313s；
  - 只有 4 次回迁移动。
  - 作为"不允许清空整张 rank"的消融证据（跨作业）。
- **V9 预注册**：protocol §15，预测 `forecast-v9-cur-s911.json`（SHA `2d04059d…`）。
- **V9 = 289477**：
  - tag `v9-evacuate-20261104`，manifest `09e4c317…`，tarball `d5c3b8a8…`，23:10 上传；
  - 14 轮，`--time 6:30:00`，依赖 `afterany:289203`；
  - V8 的 `no_move_v2` 一轮跑完后由后台任务自动撤销 V8；同一个监视跟踪 V8、V9。
- **V8 结束（FAILED）**：原生轮（22:48–23:19）在 trace 900s + drain 1000s 内没跑完，抛 TimeoutError，作业按快速失败退出。已归档到 `evidence/xplace-v8-current-20261104/job-289203`，SHA `f81ccb45…`。
  - 第一轮 `ca_full_tm_slo`（max_moves 8）passed，结果见上。
  - **原生轮的现象**：
    - 420s、540s 的 Code TTFT 60.4s、59.7s；
    - 660s（rank 2）、780s（rank 3）的 Code 到 1900s 都没开始执行（Code 映射一直是 0.1 GiB），等待超过 1240s、1120s；
    - 到 1900s 只完成 38/109 个请求，986s 到 1872s 间只完成 6 个；
    - 全程 0 次抢占。
  - **原因（vLLM 源码证实，`DPEngineCoreProc.run_busy_loop`）**：
    - 本地有未完成的请求（包括调度不上而排队的请求）或 wave 仍在运行时，引擎每轮执行 `execute_dummy_batch()`；每 32 步的全局 all-reduce 中只要有一个 rank 还有排队请求，wave 就不停；
    - 于是一个因 KV 不足排在 vLLM 内部的 Code，让 Code 实例 4 个 rank 一直空跑 EP 前向，与 Conv 抢 GPU；
    - Conv 被拖到约 350ms/token（rank 0 在 986–1747s 映射增长 3.9 GiB，约 20 个 Conv，每个约 2.8 token/s），完成变慢、KV 不释放，Code 更进不来，形成正反馈。
  - 受控策略不会出现这种情况：Code 在控制器侧排队，放得下才提交给 vLLM，Code 实例排队期间空闲。可作为论文动机：原生 DP 下 KV 阻塞的长请求会让本实例空转并拖垮共址实例。
  - simulator 没有建模"vLLM 内部排队的请求会让整个 DP 组空转"，所以只预测了 474s。
- **撤销 V9（289477，尚未开始）**：它的第二轮是原生轮，会同样超时，导致后面 12 轮无法运行。
- **修复 v4.2**（本地，测试 39 个通过）：
  - **截断（censoring）**：一轮到达截止时间（trace + drain）但没有出错时，不再抛 TimeoutError 让整个作业失败，而是：
    - 记录未完成请求的部分进度（`stream` 新增实时 progress，写入 `incomplete.json`）；没出首 token 的请求，TTFT 记为截止时间减到达时间（下界）；
    - 取消这些请求（vLLM `AsyncLLM.generate` 在 CancelledError 时会 abort），等引擎排空后进入下一轮；
    - summary 状态记为 `censored`；
    - `complete()`、`wait_capacity()` 不再把主动取消记为错误（V8 summary 里那串 CancelledError 就是这么来的）。
  - `report_main` 读入截断记录，表中增加截断列。截断记录作为下界参与配对比较，会低估原生的 Code TTFT，结论方向因此偏保守。
  - **simulator 增加 DP 空转干扰**：某个实例只剩 vLLM 内部排队的请求（KV 不足的到达或被抢占的请求）时，仍按 decode 强度干扰另一个实例。旧模式 5 个策略逐位一致。
    - 在 seed 911 上重现 V8 原生轮：1900s 时完成 33/109（实测 38）；2 个 Code 一直没开始；1872s 时 Conv 映射 [11.9, 10.5, 9.8, 11.2] GiB（实测 [12.9, 10.9, 10.1, 10.1]）；Conv TPOT 约 285ms（实测约 350ms）；
    - 全部跑完需 5736s，因此原生轮必须截断。
- **V9b = 289572**：
  - tag `v9b-censor-20261104`，manifest `a22d9c26…`，tarball `be451e02…`，23:45 上传；
  - 14 轮，原生放最后并在 1900s 截断，`--time 6:30:00`；
  - 预测 `forecast-v9b-cur-s911.json`（SHA `ad4913cd…`），协议 §15.6；
  - 排队中，监视已启动。

## 2026-11-05：V9b 运行

- **V9b 开始**：00:31 在 d1n41a28g03 开始（与 V8 同一节点，V8 探针 12.6ms），可与 V8 第一轮做同节点跨作业比较。
- **V9b 失败（启动单测）**：00:32 在启动阶段退出（运行 2.4 分钟，exit 1）。新加的测试 `test_report_includes_censored_records_as_lower_bounds` 要 import `tools/report_main.py`，而 upload 包里只有 `runtime/`、`tests/`、`inputs/`，本地完整仓库里能通过，作业里失败。没有跑任何一轮。日志已归档，SHA `0a12b753…`。
  - **修复**：该测试在没有分析工具时跳过。
  - **流程补充**：每次上传前，在 staging 的 upload 目录里单独跑测试（复现作业的启动单测），再上传。
- **V9c = 289718**：
  - tag `v9c-censor-20261105`，manifest `8fbc9fb8…`，tarball `75e2d90d…`，00:40 上传；
  - 与 V9b 只差这个测试文件，运行时代码相同，预注册沿用 §15.6；
  - 只含 upload 内容的布局下测试 38 通过、1 跳过；
  - 14 轮，`--time 6:30:00`；排队中，监视已启动。
- **V9c 开始**：01:00 在 d1n41a22g02 开始（首次使用的节点；CPULoad 6.3，FreeMem 516 GB）。
- **V9c 启动通过**：启动单测 38 通过、1 跳过；探针 12.7ms，通过；第一轮 `ca_full_tm_slo` 占位后四卡容量 15.98 GiB（170 个页组），约 01:04 开始。
- **V9c 第一轮中途检查**（01:17）：`max_moves` 24 下，660s 的 Code 仍有 49 次规划判定不可行，0 次迁移。
  - **根因（比页共享更早挡住）**：planner 的 `_select_for_target` 在考虑任何迁移前先检查 Code 池的 `free_blocks` 是否不少于 Code 所需块数，不满足就直接返回不可行，没有计入迁走 Conv 后释放的页。
  - kvcached 的 `get_num_free_blocks()` 等于 `available_size()`，即物理空闲（`cudaMemGetInfo` − 2% 余量）加池内空位。没有占位时物理空闲约 35 GiB，这个检查从不起作用；占位后它成为真正的约束。
  - 655s 的 rank 0：已用 6.75 GiB，物理空闲 9.23 GiB，约 6,300 块 + 63 个空位，而 Code 需要 6,371 块，只差几个块。
  - `plan_usage` 对到达请求也有同样的问题。
  - simulator 的 `free_blocks` 固定为 10^7，所以没暴露。V8 第一轮的不可行同样首先是这个原因，页共享是另一个真实存在、叠加在上面的约束。
- **处理**：V9c 跑完第二轮 `no_move_v2`（有效基线）后由后台任务自动撤销；修 planner，simulator 改为报告真实的 `free_blocks`。
- **修复 v4.3**（测试 41 个通过；只含 upload 内容的布局下 40 通过、1 跳过；旧模式逐位一致）：
  - planner 计入迁出后释放的块，不可行时记录诊断；
  - 按步增长的轮次，预算设为占位后的实际可映射容量（15.9375 GiB）；
  - simulator 和 fake 都报告物理空闲块。修复前的 planner 加上真实的 `free_blocks`，在 seed 911 上重现了真机的 0 次迁移准入；修复后清空恢复，模拟中 661–662s 迁 13–15 个、781–782s 迁 12–13 个。
- **V10 预测**：`forecast-v10-cur-s911.json`，SHA `ad4913cd…`，与 V9b 预测逐字节相同；协议 §15.7。
- **V10 = 289839**：
  - tag `v10-freeblocks-20261105`，manifest `07718d96…`，tarball `e1d82991…`，01:34 上传；
  - 14 轮（原生最后，1900s 截断），`--time 6:30:00`，依赖 `afterany:289718`（V9c 的 `no_move_v2` 轮结束后由后台任务撤销 V9c）；
  - 监视 b9nlejg4q 同时跟踪 V9c 和 V10。
- **V9c 结束**（01:00–01:49，d1n41a22g02，探针 12.7ms）：第二轮后由后台任务撤销，第三轮 `ca_full_v2` 中途被杀，作废。已归档到 `evidence/xplace-v9c-censor-20261105/job-289718`，SHA `e468d579…`。
  - **第一轮 `ca_full_tm_slo`**（迁移被物理空闲块检查挡住）：passed，109/109，0 次抢占，0 次迁移准入。
    - 4 个 Code 的 TTFT 为 31.3、35.2、456.3、368.3s（均值 222.8s）；Conv TPOT 26.4ms，Conv TTFT 0.67s（p99 3.6s），Conv SLO 1.00。
    - 修复前的 planner 加真实 `free_blocks` 的模拟为 [30.3, 31.2, 486.5, 394.8]s，与实测吻合。
    - 实际上等于"只分时"：与 `no_move_tm_v2` 的预测 233s / 25.7ms 一致。
  - **第二轮 `no_move_v2`**：passed，109/109，0 次抢占。
    - 4 个 Code 的 TTFT 为 64.4、67.2、540.2、508.2s（均值 295.0s）；Conv TPOT 29.81ms，Conv TTFT 0.70s。
    - 预测 279.7 [273.6, 285.8]s、29.6ms：Code TTFT 高 5%（略超区间），Conv TPOT 吻合。
  - 两轮无 Code 时段的 G16 占比 0.94–1.00，可避免的为 0：Conv 0.12 RPS 下每 rank 超过 8 个 Conv，16 档不可避免，placement 在这方面没有可优化的空间。
  - V8 第一轮（d1n41a28g03）与 V9c 第一轮（d1n41a22g02）是"迁移被挡"的两次跨节点重复：660s 的 Code TTFT 401/456s，780s 的 313/368s。
- **V9c 撤销后 V10（289839）随即进入调度**。
- **复现 seed**：按 §14.4 的规则从 912 起，选出 914（113 个 Conv，前置 Conv 218），冻结 `cur-b16-c012-p120-s914-d900.json`（SHA `8a777986…`）。
- **V11 预注册**：protocol §16，预测 `forecast-v11-cur-s914.json`（SHA `7278947c…`）：
  - `ca_full_tm_slo` 对 `no_move_v2`：Code TTFT −94%，Conv TPOT +10.7%；
  - 分时下 `mig_random_tm_v2` → `ca_full_tm_v2`：Conv TPOT −9.7%，抢占 4 → 0（随机迁移的目的地没留余量）。
- **V11 = 289851**：
  - tag `v11-replicate-20261105`，manifest `0d5d9bd9…`，tarball `62d4bc21…`，01:59 上传；与 V10 只差 trace；
  - 8 轮，`--time 4:00:00`，依赖 `afterany:289839`；
  - 只含 upload 内容的布局下测试 40 通过、1 跳过；
  - 已加入监视 b9nlejg4q 的作业列表。
- **V10 开始**：03:04 在 d1n41a28g03 开始（与 V8 同一节点）。
- **V10 启动通过**：启动单测 40 通过、1 跳过；探针 12.7ms，通过；第一轮 `ca_full_tm_slo` 占位后四卡容量 15.98 GiB，03:08 开始。
- **V10 第一轮中途检查**（03:21，`ca_full_tm_slo`，到 760s）：**在真机上迁走目标 rank 上的大部分 Conv 后准入成功**。
  - 660.9s：一次准入事务把 rank 3 上的 10 个 Conv 迁到其他 rank，10/10 逐字节校验通过，单次迁移 0.03–0.13s，暂停 0.18s，事务 0.83s；
  - 迁移后各 rank 已用 [9.84, 9.84, 9.94, 6.19] GiB，准入时目标 rank 的余量为 0.25 GiB；
  - 660s 的 Code 走迁移准入，排队 0.9s，TTFT 27.0s（V8 同节点上不能清空时为 401s）；
  - Code 完成后 691–774s 回迁 4 次共 9 个，全部校验通过；全程 0 次抢占。
  - 420s、540s 的 Code 直接准入，TTFT 31.8s、30.4s。
- **V10 第一轮完成**（`ca_full_tm_slo`，03:08–03:29，passed，109/109 输出长度精确，0 次抢占，30 次迁移全部逐字节校验通过）：
  - 4 个 Code 的 TTFT 为 31.8、30.4、27.0、30.0s（均值 29.8s，预测 30.3s）。660s 的 Code 迁走 10 个，780s 的 Code 迁走 2 个，余量 0.69 GiB。
  - Conv TPOT 29.47ms（预测 29.3ms），Conv TTFT 0.86s（p99 4.3s，预测 0.7 / 4.2s），Conv/Code SLO 都是 1.00。
  - 同节点跨作业对比：V8 第一轮不能清空时为 31.8 / 30.9 / 401.1 / 312.8s。
  - 无 Code 时段的 G16 占比 0.93，可避免的为 0。
- 03:29 第二轮 `no_move_v2` 开始。
- **V10 第二轮 `no_move_v2`**（03:29–03:52，passed，109/109，0 次抢占）：
  - 4 个 Code 的 TTFT 为 62.8、60.6、476.7、438.5s（均值 259.6s；预测 279.7 [273.6, 285.8]s，V9c 为 295.0s）。660s、780s 的 Code 排队 415s、378s；Code SLO 0.50。
  - Conv TPOT 30.88ms（预测 29.6，V9c 29.8），Conv TTFT 0.60s。
  - **同一作业配对，`no_move_v2` → `ca_full_tm_slo`**：Code TTFT −88.5% [−93.8, −49.5]，Conv TPOT −4.6% [−8.9, −0.3]，Conv TTFT +43%（0.60 → 0.86s，不显著，都在 SLO 内）。预测为 −89.2% / −1.9%。
- 03:52 第三轮 `ca_full_v2` 开始。
- **V10 第三轮 `ca_full_v2`**（迁移，不分时，03:52–04:12，passed，109/109，0 次抢占，58 次迁移全部逐字节校验通过）：
  - 4 个 Code 的 TTFT 为 64.4、64.5、65.1、63.7s（均值 64.4s，预测 61.3s）；660s、780s 的 Code 分别迁走 9、13 个后准入。
  - Conv TPOT 41.0ms（预测 39.4），p99 59.4ms；Conv SLO 0.88。
  - `no_move_v2` → `ca_full_v2`：Code TTFT −75.2% [−85.9, +4.5]（n=4，CI 跨 0），Conv TPOT +32.8% [+25.4, +40.3]，超出 20% 上限（预测 +33%）。
    - 原因：Code 驻留时间占比 0.65（`no_move_v2` 为 0.31），Code prefill 期间对 Conv TPOT 的贡献 11.4ms（4.4ms）；
    - 另有 18 次回迁事务，每次暂停 Conv 0.2–1.1s。
  - `ca_full_v2` → `ca_full_tm_slo`（分时）：Code TTFT −53.7% [−57.1, −51.2]，Conv TPOT −28.1% [−30.0, −26.4]。
- 04:12 第四轮 `mig_random_v2` 开始。
- **V10 第四轮 `mig_random_v2`**（04:12–04:34，passed，109/109，0 次抢占）：
  - 4 个 Code 的 TTFT 为 55.4、60.6、69.3、64.3s（均值 62.4s，预测 60.6s）。只有 660s 的 Code 走了迁移（迁 9 个），780s 的 Code 直接放进。
  - Conv TPOT 42.82ms（预测 41.7），Conv SLO 0.88。
  - **placement B**（`mig_random_v2` → `ca_full_v2`，同一作业）：Conv TPOT −4.3% [−5.1, −3.5]，Code TTFT +3.3% [−3.6, +11.5]（n=4）。预测 −6.2% [−11.0, −1.1]。
    - 这是 B 第一次在端到端配对中表现为显著改善（S4、V6 为 +2.4%、+7.1%），但幅度小；
    - 来源不在 CUDA graph 档位（两者无 Code 时段的 G16 占比都是 1.00，可避免的为 0），而在无 Code 时段的 Conv 贡献：26.6 对 28.6ms。无 Code 时段各 rank 的 Conv 数差：代价感知 2.9，随机 3.5；
    - 预测中随机迁移有抢占（3.3 次），实测为 0。
  - `no_move_v2` → `mig_random_v2`：Code TTFT −76.0% [−85.4, −6.0]，Conv TPOT +38.6%。
- 04:34 第五轮 `ca_full_tm_v2` 开始。
- **V10 第五轮 `ca_full_tm_v2`**（全程分时，04:34–04:55，passed，109/109，0 次抢占，35 次迁移）：
  - 4 个 Code 的 TTFT 为 25.4、24.8、26.4、26.1s（均值 25.7s，预测 26.1s）；660s、780s 的 Code 走迁移准入。
  - Conv TPOT 28.89ms（预测 29.0），Conv TTFT 3.38s（p99 24.4s），Conv SLO 0.79。
  - `ca_full_v2` → `ca_full_tm_v2`（分时）：Code TTFT −60.2% [−61.0, −59.3]，Conv TPOT −29.5% [−31.4, −27.6]，Conv TTFT +334%。
  - `ca_full_tm_v2` → `ca_full_tm_slo`（按需切片）：Code TTFT +16.1% [+7.3, +24.0]，Conv TPOT +2.0%，Conv TTFT −74.6% [−82.5, −57.6]，Conv SLO 0.79 → 1.00（预测 +16.3% / +0.9%）。
- 04:55 第六轮 `no_move_tm_v2` 开始。
- **V10 第六轮 `no_move_tm_v2`**（只分时，04:55–05:16，passed，109/109，0 次抢占）：
  - 4 个 Code 的 TTFT 为 25.3、25.0、413.7、321.6s（均值 196.4s，预测 233s）；660s、780s 的 Code 排队 389s、297s。
  - Conv TPOT 26.16ms（预测 25.7），Conv TTFT 2.62s（p99 23.9s），Conv SLO 0.84。
  - `no_move_v2` → `no_move_tm_v2`（不迁移时的分时）：Code TTFT −24.4% [−59.2, −15.2]，Conv TPOT −15.3% [−17.2, −13.1]（预测 −16.7% / −14.0%）。
  - `no_move_tm_v2` → `ca_full_tm_v2`（分时下的迁移）：Code TTFT −86.9% [−92.9, −0.1]，Conv TPOT +10.4% [+7.5, +13.3]（预测 −88.8% / +13.1%）。Conv 变慢是因为 Code 进得更早，Conv 被暂停的时间更长。
- 05:16 第七轮 `mig_random_tm_v2` 开始（3×2 组合的最后一格）。
- **V10 第七轮 `mig_random_tm_v2`**（随机迁移 + 分时，05:16–05:36，passed，109/109，0 次抢占，22 次迁移）：
  - 4 个 Code 的 TTFT 为 25.5、25.2、25.6、26.5s（均值 25.7s，预测 25.9s）。
  - Conv TPOT 32.70ms（预测 30.6），Conv SLO 0.78。
  - **分时下的 placement B**（`mig_random_tm_v2` → `ca_full_tm_v2`，同一作业）：Conv TPOT −11.6% [−12.4, −10.9]，Code TTFT −0.1%，Conv TTFT −3.1% [−5.2, −1.3]（预测 −5.1%）。
    - 来源：从一张 rank 迁走大部分 Conv 后，随机迁移不回迁，Conv 分布长期严重失衡（无 Code 时段各 rank Conv 数差平均 7.2、最大 15；最后一个 Code 结束后为 [4, 19, 19, 15]）。
    - 代价感知回迁把差压到平均 2.1（最后为 [15, 14, 13, 9]），最满 rank 的 Conv 数从 13.9 降到 10.0。每步代价由最满的 rank 决定，所以无 Code 时段的贡献从 28.5ms 降到 24.7ms。
    - 两者 G16 占比都是 0.90，可避免的为 0，收益不来自 CUDA graph 档位，而来自最满 rank 的 batch 和 attention 长度。
  - 不分时下 B 为 −4.3%（第四轮），分时下为 −11.6%。分时下 780s 的 Code 也走迁移准入（不分时的随机迁移中它直接准入），准入迁走的 Conv 更多（随机迁移 22 个对 9 个），之后的回迁更重要。
  - `mig_random_v2` → `mig_random_tm_v2`：Code TTFT −58.8%，Conv TPOT −23.6%。
- **3×2 组合齐全**（placement {不迁移, 随机, 代价感知} × {并行, 分时}，同一作业，d1n41a28g03）。
- 05:36 第八轮 `ca_admit_v2` 开始。
- **V10 第八轮 `ca_admit_v2`**（只做代价感知准入，不回迁，05:36–05:59，passed，109/109，0 次抢占）：
  - 4 个 Code 的 TTFT 为 61.6、55.4、65.1、65.7s（均值 62.0s，预测 60.9s）；迁移 9 次，与随机迁移相同（660s 的 Code 迁走 9 个，780s 的直接准入）。
  - Conv TPOT 42.49ms（预测 41.6）。
  - `mig_random_v2` → `ca_admit_v2`（只把准入换成代价感知）：Conv TPOT −0.8% [−1.0, −0.5]，Code TTFT −0.7%（预测 +0.3%）。准入时的代价感知几乎没有作用：两者选的是同一张 rank（rank 1，15 个 Conv），都迁 9 个，只有目的地不同。页共享下要迁走目标 rank 上的大部分 Conv，迁哪些请求基本已经定了。
  - `ca_admit_v2` → `ca_full_v2`（加回迁）：Conv TPOT −3.5% [−4.5, −2.7]。不分时下 B 的 −4.3% 主要来自回迁。
- 05:59 第九轮 `ca_return_v2` 开始。
- **V10 第九轮 `ca_return_v2`**（随机准入 + 代价感知回迁，05:59–06:19，passed，109/109，0 次抢占，48 次迁移）：
  - 4 个 Code 的 TTFT 为 60.3、59.4、70.5、65.2s（均值 63.9s，预测 61.9s）；Conv TPOT 40.96ms（预测 39.7）。
  - `mig_random_v2` → `ca_return_v2`（只加代价感知回迁）：Conv TPOT −4.3% [−5.1, −3.6]，与 `mig_random_v2` → `ca_full_v2` 的 −4.3% 相同（预测 −5.7%）。
  - `ca_return_v2` → `ca_full_v2`（再加代价感知准入）：Conv TPOT +0.1% [−0.0, +0.2]，Code TTFT +0.9%，没有差别。
  - **单组件消融结论**（不分时，同一作业）：placement 的 B 全部来自代价感知回迁（−4.3%）；代价感知准入在单独加（−0.8%）和在回迁之上加（+0.1%）时都几乎为零。原因是页共享下准入要迁走目标 rank 上的大部分 Conv，挑选空间很小。
- 06:19 第十轮 `ca_full_v2_nh` 开始。
- **V10 第十轮 `ca_full_v2_nh`**（不留增长余量，不分时，06:19–06:42，passed，109/109，**0 次抢占**，29 次迁移）：
  - 4 个 Code 的 TTFT 为 58.0、60.2、64.7、63.1s（均值 61.5s）；Conv TPOT 39.69ms，Conv SLO 0.91。
  - **余量消融**（`ca_full_v2_nh` → `ca_full_v2`）：Conv TPOT +3.3% [+3.1, +3.5]，Code TTFT +4.8% [+0.9, +9.1]。**留余量反而略差**，且两者都没有抢占；预测为 Conv TPOT −1.0%、抢占 0.1 对 0。
    - 原因：留余量的准入要多迁（660s 迁 9 对 8，780s 迁 13 对 3），回迁也更多（18 次事务、36 个 对 11 次、18 个），暂停 177 对 96 request·s；
    - 不留余量时无 Code 时段更均衡（各 rank Conv 数差 1.7 对 2.7，最满 rank 13.6 对 15.0）。
    - 这个负载下，余量保护的抢占风险在不留余量时也没有发生：页共享使 Conv 增长先填已映射的空位，实际增长比"速率 × 窗口"小。
  - 判定（§14.8）："`ca_full_v2_nh` 抢占多于 `ca_full_v2`"不成立，如实报告：在这个负载下增长余量是不必要的保守，代价是多迁移约 2 倍。
- 06:42 第十一轮 `ca_full_tm_slo_nh` 开始。
- **V10 第十一轮 `ca_full_tm_slo_nh`**（分时 + 按需切片，不留余量，06:42–07:01，passed，109/109，0 次抢占，38 次迁移）：
  - 4 个 Code 的 TTFT 为 31.8、30.3、27.2、29.9s（均值 29.8s，与 `ca_full_tm_slo` 相同）；Conv TPOT 29.71ms，Conv SLO 0.98。
  - **分时下的余量消融**（`ca_full_tm_slo_nh` → `ca_full_tm_slo`）：Conv TPOT −0.8% [−0.9, −0.8]，Code TTFT 0.0%，Conv TTFT −5.4% [−12.0, −0.1]（预测：完全相同）。分时下留余量让 780s 的准入少迁 4 个（2 对 6），回迁少 4 个，效果小但方向有利。
  - 余量在两种调度下的效果方向相反：不分时 +3.3%（不利），分时 −0.8%（略有利）。都很小，都没有抢占。
- 07:01 第十二轮 `ca_full_tm_s4` 开始。
- **V10 第十二轮 `ca_full_tm_s4`**（周期切片 S=4，07:01–07:22，passed，109/109，0 次抢占，16 次切片）：
  - Code TTFT 31.8、32.7、33.5、31.9s（均值 32.5s，预测 34.0s）；Conv TPOT 28.96ms，Conv TTFT 0.95s（p99 5.3s），Conv SLO 0.98，最长停顿中位数 7.7s。
  - **按需对周期**（`ca_full_tm_s4` → `ca_full_tm_slo`）：Code TTFT −8.2% [−16.2, −1.7]，Conv TPOT +1.8%，Conv TTFT −9.6%（不显著），SLO 0.98 → 1.00；最长停顿中位数 16.8 对 7.7s（预测 −10.8% / −0.7%，旧设定 V5b 为 −11.2%）。
- 07:22 第十三轮 `no_move_fair_v2` 开始，之后是原生轮（截断于 1900s）。
- **V10 第十三轮 `no_move_fair_v2`**（公平门，07:22–07:45，passed，109/109，0 次抢占）：
  - 4 个 Code 的 TTFT 为 61.0、66.8、407.1、365.8s（均值 225.2s，预测 223.3s）。
  - Conv TTFT 75.5s（p99 301s，预测 73.9s），Conv SLO 0.71：公平门为等待的 Code 挡住了 Conv 的准入。
  - `no_move_v2` → `no_move_fair_v2`：Code TTFT −13.3% [−15.9, +3.6]，Conv TPOT −2.3%，Conv TTFT +125 倍。说明"在控制器里让 Code 优先"不能解决单卡放不下的问题，只会把 Conv 饿死。
- 07:45 第十四轮 `kvcached_native` 开始（1900s 截断，约 08:18 结束）。
- **V10 第十四轮 `kvcached_native`**（07:45–08:17，按设计在 1900s 截断，status censored）：
  - 完成 33/109（2 个 Code、31 个 Conv），截断 76 个（74 个 Conv 已出首 token 未完成，2 个 Code 未开始）。
  - 4 个 Code 的 TTFT 为 65.9、68.1s，以及 ≥1239、≥1119s（下界），截断均值 623.0s。780s 的 Code 被抢占 1 次后再没进入。
  - 986s 之后只完成 4 个 Conv，最后一个在 1749s；Conv TPOT 304ms（部分记录）。全程 0 次迁移，控制器事件循环最大卡顿 2ms。
  - 与 §15.6 的截断预测（约 33/109、下界约 1240/1120s、均值 ≥620s、停滞期约 280ms/token）吻合，与 V8 原生轮（38/109）是同一现象的跨作业重复。
- **V10 结束**（08:17，exit 0）：已归档到 `evidence/xplace-v10-freeblocks-20261105/job-289839`（SHA `7f70d54d…`），manifest 校验通过，启动单测 40 通过、1 跳过；报告见同目录 `report.md`、`report.json`（窗口 420–900s）。
  - `tools/decompose.py` 在原生轮（窗口内没有完成的 Conv）除零失败，已加 `n == 0` 保护，原生轮的分解记为不可用。
- **V10 判定**（§15.5–§15.7）：
  1. 问题存在：原生 2 个 Code 的 TTFT（含下界）> 70s。**成立。**
  2. 清空整张 rank 有效（判定名沿用 §15.5；实际没有一次准入清空整张 rank，见 10:05 的更正）：**基本成立。**
     - 所有迁移策略里，660s 的 Code 都走迁移准入，在控制器排队 0.7–1.5s（不迁移的策略为 297–415s）；
     - 780s 的 Code 在 `mig_random_v2`、`ca_admit_v2` 中直接准入（排队 0.1–0.3s）：这两个策略不分时也不回迁，为 660s 清空的 rank 仍有空间；其余迁移策略走迁移准入，排队 0.6–1.8s；
     - 不分时 ≤70s：只有 `ca_return_v2` 660s 的 Code 为 70.5s，略超；分时 ≤40s 全部满足（25–34s）。
  3. 整体：**成立。**
     - 原生 → `ca_full_tm_slo`：Code TTFT −95.2%，Conv TPOT −90.3%（原生截断，偏保守），Conv SLO 1.00；
     - `no_move_v2` → `ca_full_tm_slo`：Code TTFT −88.5% [−93.8, −49.5]，Conv TPOT −4.6% [−8.7, −0.6]。
  4. retract：`ca_full_tm_slo` 0 次抢占，原生 1 次。**成立。**
  5. 正确性：13 个受控轮全部 109/109、输出长度精确；原生轮完成的 33 个也精确；所有迁移逐字节校验通过。**成立。**
  - placement B 与余量消融：如实报告。不分时 −4.3%、分时 −11.6%，都来自代价感知回迁；余量在不分时下略有害（+3.3%），在分时下略有利（−0.8%），两者都没有抢占。
- 08:19 V11（289851）在同一节点 d1n41a28g03 开始；探针 12.6ms 通过，第一轮 `ca_full_tm_slo` 占位后四卡容量 15.98 GiB，08:22 开始。
- **V11 第一轮 `ca_full_tm_slo`**（seed 914，08:22–08:41，passed，117/117 输出长度精确，0 次抢占，32 次迁移全部逐字节校验通过）：
  - 4 个 Code 的 TTFT 为 30.0、30.6、27.3、34.6s（均值 30.6s，预测 29.7s）。420s 的 Code 直接准入（排队 0.4s），其余 3 个走迁移准入，排队 0.6–1.4s。
  - 660s 的准入只迁走 rank 1 上 14 个 Conv 中的 7 个：Conv 映射从 8.25 降到 6.09 GiB，已经够用。与 V10 相同，迁走部分 Conv 就能腾出足够的页。
  - Conv TPOT 29.3ms（预测 30.2ms），Conv TTFT 0.93s（p99 3.9s）；持有 15 次共 105s，切片 11 次。
- 08:41 第二轮 `no_move_v2` 开始。
- **V11 第二轮 `no_move_v2`**（08:41–09:03，passed，117/117，0 次抢占）：
  - 4 个 Code 的 TTFT 为 57.4、595.4、556.9、520.9s（均值 432.6s，预测 469s）。540、660、780s 的 Code 在控制器排队 542、494、456s；Code SLO 0.25。
  - Conv TPOT 28.62ms（预测 27.3ms），Conv TTFT 0.36s。
  - 与第一轮对比（`no_move_v2` → `ca_full_tm_slo`，均值）：Code TTFT −92.9%，Conv TPOT +2.2%（预测 −93.7% / +10.7%）。§16 判定 3 成立，CI 待作业结束后用 `report_main` 计算。
- **V11 第三轮 `ca_full_v2`**（09:03–09:24，passed，117/117，0 次抢占，52 次迁移全部逐字节校验通过）：
  - 4 个 Code 的 TTFT 为 57.3、57.5、69.2、61.4s（均值 61.3s，预测 61.4s）；660s、780s 的 Code 走迁移准入。
  - Conv TPOT 40.08ms（预测 39.4ms），p99 60.4ms；Conv SLO 0.91。
- **V11 第四轮 `mig_random_v2`**（09:24–09:46，passed，117/117，21 次迁移全部校验通过，**10 次抢占**，预测 5.8 次）：
  - 4 个 Code 的 TTFT 为 48.8、52.4、64.7、56.6s（均值 55.6s，预测 60.2s）；420、540、660s 的 Code 走迁移准入，780s 的直接准入。
  - Conv TPOT 42.03ms（预测 41.1ms），Conv SLO 0.91。
  - **受控策略第一次在真机上出现抢占**：
    - 10 次都发生在 trace 结束后的排空阶段（980–1241s），涉及 6 个 Conv，其中 5 个在评估窗口内；被抢占的 Conv 最长停顿 79–81s。
    - 661.5s 的准入把 rank 2 上的 13 个 Conv 迁到 rank 0、1、3，随机迁移之后不回迁。
    - 被抢占的 6 个 Conv 中：
      - 3 个是这次从 rank 2 迁到 rank 0、1 的。这两张 rank 上被抢的只有迁来的请求，与 vLLM 先抢占运行队列中最后加入的请求一致（迁入的请求在提交时加入目的地队列）；
      - 另外 3 个是 889–898s 直接准入到 rank 2 的新 Conv。Code 离开后 rank 2 最空，新 Conv 都落到这里，rank 2 的 Conv 从 3 个涨到 17 个。
    - 先后达到容量的是：rank 0 在 980–995s（17 个 Conv，映射 15.4 GiB），rank 1 在 1090–1121s，rank 2 在 1161–1241s。
  - **placement B**（`mig_random_v2` → `ca_full_v2`，均值）：Conv TPOT −4.6%（预测 −4.2%，V10 −4.3%），抢占 10 → 0。
    - 准入时的增长余量只覆盖 15s 或一个 Code 驻留期；之后的增长要靠回迁维持均衡。不回迁时，余量不能防止抢占。
    - 是增长余量、代价感知准入还是回迁起的作用，需要 seed 914 上的 `ca_return_v2`、`ca_admit_v2`、`ca_full_v2_nh` 才能分开。
- **更正（10:05）**：前面多处写的"清空整张 rank"不准确。
  - V10 的 18 次准入迁移，在目标 rank 的 12–17 个 Conv 中迁走 2–13 个（中位 9 个），没有一次清空整张 rank；其中 11 次超过 8 个，即 V8 的 `max_moves` 上限。V11 第一轮在 14 个中迁走 7 个。
  - planner 按快照中的页归属，只计入所有者全部被迁走的页（`released_gib`）。所以页共享的实际影响是要迁走目标 rank 上的大部分 Conv，而不是全部。
  - `ca_admit_v2` 与 `mig_random_v2` 选的是同一张 rank（rank 1，15 个 Conv），都迁 9 个，只有目的地不同。这才是代价感知准入没有作用的直接原因。
  - simulator 的页共享模型是"高水位"：只要一张 rank 上还有 Conv，它的页就不减少。所以在模拟中，每次准入都要迁走目标 rank 上的全部 Conv（V10 trace 上 13–17 个），实测为 2–13 个。
  - 已在 results-short、README、CLAUDE.md、论文结构中改正；V10 各轮记录中的相应措辞也已改正。
- **V11 第五轮 `ca_full_tm_v2`**（09:47–10:10，passed，117/117，0 次抢占，50 次迁移全部校验通过）：
  - 4 个 Code 的 TTFT 为 25.1、25.0、25.8、25.7s（均值 25.4s，预测 26.1s）；660s、780s 的 Code 走迁移准入。
  - Conv TPOT 28.65ms（预测 29.7ms），Conv TTFT 3.78s（p99 25.6s），Conv SLO 0.80；持有 4 次共 103s。
- 10:10 第六轮 `no_move_tm_v2` 开始。
- **监视器卡住**：b9nlejg4q 在 08:41 之后没有输出。ssh 没有超时，挂住后整个循环停住。已换成 bz7rpbvkv（`/tmp/xplace_monitor_v7.sh`，用 perl `alarm` 给 ssh 加 90s 硬超时，并开 ServerAlive）。
- **V12 预注册**：protocol §17，预测 `forecast-v12-cur-s914.json`（SHA `58c8f407…`）。
  - `sim/predict.py` 加了 `--workers`，多进程运行，噪声种子不变，结果与单进程逐位一致。
  - 本地测试 41 个通过；复用 V11 的 upload（manifest 校验通过，tarball `62d4bc21…`）。
- **V12 = 290422**（10:29 提交）：
  - tag `v11-replicate-20261105`（与 V11 同一 upload，只改 `XPLACE_POLICIES`），9 轮；
  - `--time 4:15:00`，依赖 `afterany:289851`，排除 d1n41a23g03；
  - 已加入监视 bz7rpbvkv 的作业列表。
- **simulator 块级页模型**（10:30–10:55，protocol §17.5）：
  - 新增 `PagePool`，逐页模拟 kvcached 分配器：单块增长放进最满的页，多块分配找能放下的最小页、否则用最空的页，页内块全部释放后取消映射；需要新页且 GPU 已满时，抢占本租户在该 rank 上最新的请求。
  - `page_model` 默认改为 `blocks`；`--page-model high_water` 逐位重现 V12 的预注册预测。新增 `sim/test_page_pool.py`（3 个测试，只在本地跑）。
  - 对 V10 + V11 前 6 轮共 19 轮的验证：
    - 迁移数大多与真机相同或接近（`mig_random_v2` 9/9、`ca_return_v2` 48/48、`mig_random_tm_v2` 22/22）；
    - `mig_random_v2` 的抢占在 seed 911 / 914 上为 0.0 / 11.5，真机为 0 / 10，高水位模型为 4.5 / 6.2；
    - Code TTFT 绝对误差中位数 1.6%（高水位 2.6%），Conv TPOT 误差不变（约 2%）。
  - 仍有偏差：V11 `no_move_tm_v2` 的 Code TTFT 预测 342s，真机 236s（高水位 376s）。原因是阈值效应：540s 时真机 rank 2 刚好放得下 Code（已用 5.81 GiB），模拟中同样的 Conv 多映射 4 页而被挡。
  - 用块级模型重做了 V12 预测（`forecast-v12b-blocks-s914.json`，SHA `a8155f7f…`），在 V12 开始前写进 §17.5，V12 同时用作新模型的样本外检验。
- **V11 第六轮 `no_move_tm_v2`**（10:10–10:31，passed，117/117，0 次抢占）：
  - 4 个 Code 的 TTFT 为 25.0、25.1、492.2、399.6s（均值 235.5s，**预测 376s**）；660s、780s 的 Code 排队 468s、375s，540s 的 Code 直接准入。
  - Conv TPOT 26.27ms（预测 26.1ms），Conv TTFT 2.89s（p99 25.7s），Conv SLO 0.85。
  - 预测偏高 60%。块级页模型把偏差降到 +45%（见下面的页模型验证），所以高水位模型只解释了一部分。其余来自阈值效应：540s 的 Code 在真机上刚好放下，在模拟中被挡。
- 10:31 第七轮 `mig_random_tm_v2` 开始，之后是原生轮（1900s 截断）。
- **V11 第七轮 `mig_random_tm_v2`**（10:31–10:52，passed，117/117，12 次迁移全部校验通过，**3 次抢占**，预测 4.0 次）：
  - 4 个 Code 的 TTFT 为 25.0、25.2、25.8、26.1s（均值 25.5s，预测 25.8s）；只有 660s 的 Code 走迁移准入。
  - Conv TPOT 31.98ms（预测 32.9ms），Conv SLO 0.80。
  - **分时下的 placement B**（`mig_random_tm_v2` → `ca_full_tm_v2`，均值）：Conv TPOT −10.4%（V10 −11.6%，预测 −9.7%），抢占 3 → 0。在第二个 seed 上复现。
- 10:52 第八轮 `kvcached_native` 开始（1900s 截断）。
- **RQ4 模拟**（11:00 前后，块级页模型，结果见 results-short §12.7）：
  - 负载 × KV 预算：`sim/sweep_rq4.py`，λ ∈ {0.08, 0.10, 0.12, 0.14} × {16, 20, 24} GiB，seed 801–806 各 2 个副本，共 1440 次，用时 3.4 分钟；数据 `evidence/xplace-sim/rq4-sweep-blocks-20261105.json`（SHA `ce9f1feb…`）。
    - 碎片和原生的崩溃都随 KV 预算增大而消失；分时的收益不变；本文策略的 Code TTFT 在所有点都是 28–30s。
    - 同一 sweep 先用高水位模型跑过一次（67 分钟，原生不截断时排空很慢），只用于对比，不作为结果。
  - 切片参数：`sim/sweep_slicing.py`，两个真机 trace，每个 3 个副本；数据 `evidence/xplace-sim/rq4-slicing-20261105.json`（SHA `6782f96c…`）。按需 T_wait=2 优于所有周期切片，与真机的 3 个锚点吻合到 1s 以内。
  - `tools/report_main.py` 的总表增加抢占次数一列（含截断记录）；已重新生成 V10 的 `report.md` / `report.json`。
- **bootstrap CI 不可复现，已修复**（11:10）：
  - `report_main.paired` 用集合 `a.keys() & b.keys()` 取配对请求。字符串的哈希每个进程不同，集合的遍历顺序随之变化，固定种子的 bootstrap 抽到的配对也就不同。结果是 CI 端点每次运行都会变：Conv TPOT 差 0.1–0.6 个百分点，本来就很宽的 Conv TTFT CI 最多差约 24 个百分点；点估计不受影响。
  - 改为按请求 ID 排序后，不同 `PYTHONHASHSEED` 下 CI 完全相同。
  - 已重新生成 V10 报告，results-short §12.4 的单组件表与论文结构 §5.4 的 CI 已换成可复现的值，结论都不变（例如 `no_move_v2` → `ca_full_tm_slo` 的 Conv TPOT 为 −4.6% [−8.9, −0.4]）。
  - progress.md 中更早的 CI 保留当时的输出，不再改写。
- **V11 第八轮 `kvcached_native`**（10:52–11:24，按设计在 1900s 截断）：
  - 完成 37/117；540、660、780s 的 3 个 Code 都没开始，TTFT 下界为 1359、1239、1119s；
  - Conv TPOT 330ms（部分记录），0 次抢占。预测为 3 个 Code 未开始、6.8 次抢占。
- **V11 结束**（11:24，exit 0，用时 3h06m）：
  - 已归档到 `evidence/xplace-v11-replicate-20261105/job-289851`（SHA `ff076c0c…`）。第一次 scp 被对端重置（Connection reset by peer），重试成功。
  - manifest 校验通过（`sha256sum -c --quiet` 无输出），启动单测 40 通过、1 跳过；报告 `report.md` / `report.json` 用修复后的 `report_main` 生成（窗口 420–900s）。
  - 正确性：共完成 856 个请求，输出长度全部精确；167 次迁移全部逐字节校验通过。
- **V11 判定**（§16）：全部成立。
  1. 原生 3 个 Code 的 TTFT > 70s；
  2. 迁移策略中 660s 及之后的 Code，不分时 ≤70s、分时 ≤40s（最慢 69.2s）。两个随机迁移策略中 780s 的 Code 直接准入；
  3. `ca_full_tm_slo` 对 `no_move_v2`：Code TTFT −92.9% [−95.0, −83.9]，Conv TPOT +2.2% [−5.2, +9.1]；
  4. 正确性。
  - 预注册预测：受控策略的 Code TTFT 偏差 −2.9%~+8.4%，`no_move_tm_v2` +60%（阈值效应）；Conv TPOT 偏差 −4.7%~+3.7%。
  - 两个 seed 的对比表见 results-short §12.6。
- **V12（290422）11:27 开始**，同一节点 d1n41a28g03。
- **按页挑选迁移对象的模拟检验（负结果）**：
  - 问题：代价感知准入与随机迁移效果相同，是不是因为按 KV 降序 / 升序挑选太粗？
  - 方法：原型是贪心挑选，优先选"迁走后它所在的页最先全部空出"的 Conv（按所在页剩余所有者数加权）。在 8 个 trace（911、914、801–806，λ=0.12，16 GiB，块级页模型）的 123 次准入可行性检查中，比较腾出空间所需的最少迁移数。
  - 结果：按 KV 挑选的较优者平均 9.41 个，贪心 9.57 个；贪心更少 2 次、相同 107 次、更多 14 次。
  - 结论：页共享下要迁走的 Conv 数基本由页的共享程度决定，换挑选方法腾不出更多页。代价感知准入没有作用是结构性的，不是排序方法的问题；不做真机实验。原型脚本 `/tmp/page_greedy_probe.py`，未进 runtime。
- **V12 启动通过**：启动单测通过，探针 12.6ms；11:32 第一轮 `ca_return_v2` 开始，占位后四卡容量 15.98 GiB。
- **论文表 2 的生成脚本** `tools/master_table.py`：
  - 输入多个作业（`--job SEED=LABEL=PATH`），输出 R1–R14 每个 seed 的指标（多次运行取均值），以及 18 个单组件对照的配对 CI；
  - 同一作业里有两个策略时用作业内配对，否则在同一 trace 上跨作业配对并标注；
  - 已用 V10、V11 试跑，数值与各自报告一致。V12 结束后加入 V12 即可得到完整的两 seed 表。
- **V12 第一轮 `ca_return_v2`**（随机准入 + 代价感知回迁，11:32–11:53，passed，117/117，46 次迁移全部校验通过，0 次抢占）：
  - 4 个 Code 的 TTFT 为 57.1、61.3、59.0、66.4s（均值 61.0s；高水位预测 58.9s，块级 58.8s）；540s、780s 的 Code 走迁移准入。
  - Conv TPOT 40.07ms（预测 38.9 / 39.2ms），Conv SLO 0.89；迁移 46 次（预测 59 / 50 次）。
  - 与 V11 的 `mig_random_v2`（42.03ms，10 次抢占）跨作业相比 −4.7%；作业内配对待第二轮。
- 11:53 第二轮 `mig_random_v2` 开始。
- **V12 第二轮 `mig_random_v2`**（11:53–约 12:15，passed，117/117，10 次迁移全部校验通过，2 次抢占）：
  - 4 个 Code 的 TTFT 为 53.6、52.1、58.4、66.1s（均值 57.6s）；420s、540s 的 Code 走迁移准入，660s、780s 的直接准入（V11 中 420/540/660s 迁移、780s 直接）。
  - Conv TPOT 40.19ms，2 次抢占都在排空阶段（1084s、1094s）。
  - 预测：高水位 40.9ms / 5.8 次，块级 42.4ms / 11.9 次。块级 p5–p95 为 41.3–43.1ms、5.9–16.1 次，这一轮低于 p5。
  - **同一策略两次运行差别很大**：V11 → V12（同 trace、同代码、同节点）Conv TPOT −4.4% [−5.8, −3.3]，抢占 10 → 2，迁移 21 → 10。请求级 bootstrap CI 不含 0，说明它不包含轮次间波动。
  - **判定 1 不成立**：作业内 `mig_random_v2` → `ca_return_v2` 的 Conv TPOT 为 −0.3% [−0.7, +0.1]（CI 上界 > 0，预测 −5.0% / −7.6%），Code TTFT +5.9% [+0.7, +13.2]。
    - 这一轮随机迁移只迁了 10 个，造成的失衡小，回迁没有可改善的空间；
    - 不分时下 B（约 −4%）与同一策略的轮次间波动（4.4%）同量级，单轮配对不足以确认。
- 12:16 前后第三轮 `ca_full_v2_nh` 开始。监视器 12:15–12:33 之间的轮询没有拿到结果（ssh 超时后跳过），没有漏掉状态。
- **V12 第三轮 `ca_full_v2_nh`**（约 12:16–12:36，passed，117/117，37 次迁移全部校验通过，0 次抢占）：
  - 4 个 Code 的 TTFT 为 53.9、53.8、67.4、58.3s（均值 58.3s，预测 61.3 / 61.0s）；只有 660s 的 Code 走迁移准入。
  - Conv TPOT 38.91ms（预测 39.3 / 39.2ms），Conv SLO 0.94；抢占 0（预测 1.4 / 2.4 次）。
  - `mig_random_v2` → `ca_full_v2_nh`（作业内）：Conv TPOT −3.2% [−3.5, −3.0]。
- 12:36 第四轮 `ca_full_v2` 开始。
- **V12 第四轮 `ca_full_v2`**（12:36–12:58，passed，117/117，32 次迁移全部校验通过，0 次抢占）：
  - 4 个 Code 的 TTFT 为 61.7、62.9、63.0、56.0s（均值 60.9s，预测 60.8 / 60.6s）；660s、780s 的 Code 走迁移准入。
  - Conv TPOT 39.40ms（预测 39.1 / 39.1ms），Conv SLO 0.93；迁移 32 次（预测 67 / 54 次）。
  - 作业内配对（Conv TPOT）：
    - `mig_random_v2` → `ca_full_v2`：−2.0% [−2.5, −1.5]（V10 −4.3%，V11 −4.6%）；
    - `ca_return_v2` → `ca_full_v2`：−1.7% [−1.8, −1.5]（V10 +0.1%）；
    - `ca_full_v2_nh` → `ca_full_v2`：+1.3% [+1.0, +1.5]（V10 +3.3%），余量仍略有害。
  - **轮次间波动**：`ca_full_v2` 在 V11、V12 之间差 −1.7% [−1.9, −1.5]，`mig_random_v2` 差 −4.4%。随机迁移的结果取决于随机选中的目的地，波动更大。
  - **不分时下的 B**：`mig_random_v2` → `ca_full_v2` 在三次独立配对中都为负（−4.3%、−4.6%、−2.0%），每次 CI 都不含 0，但幅度随随机迁移那一轮造成的失衡而变。B 在准入和回迁之间怎么分配也不稳定：V10 全部来自回迁；V12 中只加回迁为 −0.3%，在回迁之上再加代价感知准入为 −1.7%。
- 12:58 第五轮 `ca_admit_v2` 开始。
- **B 的大小由随机迁移那一轮留下的失衡决定**（V10/V11/V12 的 population 采样，t ≥ 420s、无 Code 驻留、Conv 未被持有的时段）：

  | 运行 | 策略 | 迁移 | 抢占 | 各 rank Conv 数之差（均值） | 最满 rank 的 Conv 数（均值） | Conv TPOT |
  |---|---|---|---|---|---|---|
  | V10 | `mig_random_v2` | 9 | 0 | 4.65 | 13.31 | 42.82ms |
  | V10 | `ca_admit_v2` | 9 | 0 | 5.82 | 14.05 | 42.49ms |
  | V10 | `ca_return_v2` | 48 | 0 | 1.97 | 12.06 | 40.96ms |
  | V10 | `ca_full_v2` | 58 | 0 | 2.44 | 12.35 | 41.00ms |
  | V11 | `mig_random_v2` | 21 | 10 | 6.95 | 14.65 | 42.03ms |
  | V11 | `ca_full_v2` | 52 | 0 | 2.15 | 12.65 | 40.08ms |
  | V12 | `mig_random_v2` | 10 | 2 | 2.79 | 13.31 | 40.19ms |
  | V12 | `ca_return_v2` | 46 | 0 | 1.97 | 12.74 | 40.07ms |
  | V12 | `ca_full_v2` | 32 | 0 | 2.28 | 13.10 | 39.40ms |
  | V10 | `mig_random_tm_v2` | 22 | 0 | 6.63 | 12.26 | 32.70ms |
  | V10 | `ca_full_tm_v2` | 35 | 0 | 1.97 | 9.29 | 28.89ms |
  | V11 | `mig_random_tm_v2` | 12 | 3 | 5.92 | 12.80 | 31.98ms |
  | V11 | `ca_full_tm_v2` | 50 | 0 | 1.67 | 9.75 | 28.65ms |

  - 带代价感知回迁的策略在所有运行中都把 Conv 数之差压在 1.6–2.4；随机迁移留下的差在 2.8–7.0 之间，取决于这一轮迁了多少、迁到了哪里。只做代价感知准入（`ca_admit_v2`）不回迁，差为 5.8，与随机相同。
  - 不分时下 B 与随机迁移多留下的失衡同向：V12 只多 0.5，B 为 −2.0%；V10 多 2.2，B 为 −4.3%；V11 多 4.8，B 为 −4.6%，另有 10 次抢占。
  - 分时下随机迁移多留下的失衡为 4.7 / 4.3，B 为 −11.6% / −10.4%。分时下 Conv 的大部分 TPOT 发生在无 Code 时段（V10 `ca_full_tm_v2`：24.7 / 28.9ms），失衡的影响更大；不分时下这部分只占约 65%（`ca_full_v2`：26.6 / 41.0ms），其余是与 Code 并行时的干扰。
  - 结论：代价感知 placement 的价值是始终把失衡控制住，B 的大小取决于不这样做时会失衡多少。所以 B 的方向稳定，幅度随 trace 和随机性变化；抢占只出现在失衡严重的随机迁移轮次中。
- **运行时新增 segments**（为 RQ4 的真机锚点做准备，13:05）：
  - 背景：一个作业只能跑一个 trace、一个 KV 上限。不同的负载和预算要分成多个作业，违反"多个实验打包进同一个 allocation"的规则。
  - `main.py --segments <json>`（sbatch 中为 `XPLACE_SEGMENTS`）：按顺序运行多个 {label, trace, budget_cap_gib, policies}，输出到 `trace/<label>/<policy>`，标记为 `trace:<label>/<policy>`，占位文件为 `ballast-<label>-<policy>.json`。
  - 不给 segments 时行为不变（输出、标记、文件名都与以前相同）。
  - 所有 segment 的 trace 校验和策略名在第一轮开始前检查完，写错会在启动时失败，而不是几小时后。
  - 新增 2 个 fake 引擎测试：两个 segment 用不同 trace 和上限，检查输出布局与各自的占位；后面 segment 的策略名写错时，在任何一轮开始前就失败。`test_main_fake.py` 6 个测试通过。
- **RQ4 锚点的模拟评估**（候选 V13，每点 6 个副本，块级页模型）：
  - seed 911 trace、24 GiB：原生 60.1s / 40.0ms，`no_move_v2` 60.1s / 39.3ms，`ca_full_tm_slo` 29.6s / 29.0ms。KV 充足时没有碎片，收益全部来自分时。
  - λ=0.14、16 GiB：按 §14.4 的规则（Conv 数在 126 的 ±5% 内、前置 Conv ≥ 224）从 915 起选出 seed 919（120 个 Conv，前置 227），尚未冻结。
    - 原生 6 个副本都没在 trace + 1000s 内跑完；`no_move_v2` 546s / 33.8ms；
    - `ca_full_v2` 59.3s / 40.1ms / 抢占 1.7；`mig_random_v2` 59.0s / 42.6ms / 抢占 13；
    - `ca_full_tm_slo` 29.4s / 29.6ms / 0 次抢占。
  - 等 V12 第六轮（`ca_full_tm_slo` 的第二次运行）给出分时下的轮次间波动后，再决定 V13 是否加入分时下的 placement 配对。
- **simulator 低估不分时下的 Conv TTFT**（模拟约 0.25s，真机 0.6–0.9s）：
  - 拆开真机的 Conv TTFT（窗口内 n=54）：
    - 迁移策略在控制器排队 0.39–0.52s（p90 0.8–1.2s），提交后到首 token 0.35–0.45s（p90 约 1.1s）；
    - `no_move_v2` 分别为 0.23s 和 0.20s。
  - ~~推测是迁移与回迁事务暂停 Conv 实例~~（19:55 更正：V13 24 GiB 的 `no_move_v2` 没有任何迁移，Conv TTFT 同样是 0.74s）。实际原因是与 Code prefill 并行：Code prefill 期间到达的 Conv，首 token 要 0.8–1.2s（控制器准入的 RPC 要等 Conv 引擎当前步结束，并行时每步慢很多，首个 decode 步也慢），其余时段到达的只要 0.3–0.4s。Conv TTFT 的均值随"Code prefill 期间到达的 Conv 占比"变化（24 GiB 26/58、V10 18/58、V11 8/54）。simulator 对并行期间的这段控制面延迟计入得少。
  - 影响：Conv TTFT 远在 5s 的 SLO 内，不影响结论；作为 simulator 的已知局限记录，分时下两者一致。

- **V12 第五轮 `ca_admit_v2`**（只做代价感知准入，12:58–13:19，passed，117/117，14 次迁移全部校验通过，**5 次抢占**）：
  - 4 个 Code 的 TTFT 为 55.5、58.2、61.4、57.9s（均值 58.3s，预测 60.9 / 60.8s）；540s、660s 的 Code 走迁移准入。
  - Conv TPOT 41.86ms（预测 40.8 / 41.6ms）；抢占 5 次（预测 2.0 / 5.7 次，块级模型准确），都在排空阶段（1156–1226s），最满的 rank 0 有 15–17 个 Conv、映射 15.66 GiB。
  - 无 Code 时段各 rank Conv 数之差 6.43、最满 rank 15.1 个（V10 的 `ca_admit_v2` 为 5.82）。
  - 作业内配对（Conv TPOT）：
    - `mig_random_v2` → `ca_admit_v2`：**+4.2% [+3.3, +5.3]**，判定 2（变化在 ±2% 内或 CI 含 0）不成立；
    - `ca_admit_v2` → `ca_full_v2`（加回迁）：−5.9% [−7.2, −4.8]。
  - 解释：不回迁时结果取决于腾空后的分布碰巧是否均衡。只做代价感知准入的两次运行失衡都大（5.8、6.4），与随机迁移的典型情况相同（4.6–7.0）；这一作业的随机迁移碰巧只留下 2.8。跨作业与 V11 的随机迁移相比为 −0.4%。代价感知准入本身不改善均衡，均衡来自回迁，与 V10 的结论一致。
  - 注意：population 里的抢占计数是引擎生命周期内的累计值，跨轮次延续（这一轮起点为第二轮留下的 2）。每轮的抢占数要按请求记录求和，`report_main` 与各轮小结都这样做。
- 13:19 第六轮 `ca_full_tm_slo` 开始（与 V11 第一轮相同的策略和 trace，用于估计分时下的轮次间波动）。
- **V12 第六轮 `ca_full_tm_slo`**（13:19–13:40，passed，117/117，39 次迁移全部校验通过，0 次抢占；持有 15 次共 106s）：
  - 4 个 Code 的 TTFT 为 30.0、30.6、28.0、34.2s（均值 30.70s；块级预测 30.70s，高水位 29.73s）。
  - Conv TPOT 29.77ms（预测 30.19 / 30.23ms），Conv TTFT 0.94s（p99 3.93s），SLO 1.00 / 1.00；迁移 39 次（预测 66 / 51 次）。
  - **分时下的轮次间波动小**：与 V11 第一轮相比，Conv TPOT +1.8% [+1.5, +2.2]，Code TTFT +0.3% [−1.0, +2.0]，无 Code 时段 Conv 数之差 1.49 对 1.91。不分时下，`ca_full_v2` 为 1.7%，`mig_random_v2` 为 4.4%。
  - **V13 的决定**（规则在本轮结果出来前写定：波动超过 3% 才加重复轮）：
    - 波动 1.8% < 3%，V13 不加重复轮，只做 RQ4 的两个锚点；
    - λ=0.14 锚点本身包含分时下的 `mig_random_tm_v2` / `ca_full_tm_v2` 配对，作为分时下 B 的第三个独立 trace。
- **V12 第七轮 `ca_full_tm_slo_nh`**（13:40–14:00，passed，117/117，31 次迁移全部校验通过，0 次抢占）：
  - 4 个 Code 的 TTFT 为 30.1、30.5、27.3、34.1s（均值 30.51s，块级预测 30.55s）；Conv TPOT 29.43ms（预测 29.80 / 29.89ms），SLO 1.00 / 1.00。
  - **分时下的余量消融**（作业内，`ca_full_tm_slo_nh` → `ca_full_tm_slo`）：Conv TPOT +1.1% [+0.9, +1.3]，Code TTFT +0.6% [−0.3, +2.0]，两者都没有抢占。块级预测为 +1.1%。V10 为 −0.8%。在 seed 914 上，分时下的余量同样不起作用。
- 14:01 第八轮 `ca_full_tm_s4` 开始。
- **V13 预注册**（protocol §18）：
  - 新 trace `cur-b16-c014-p120-s919-d900.json`（λ=0.14，seed 919，SHA `81c3d0c0…`）；
  - segments `segments-v13-anchors.json`（SHA `2d6b20c1…`），共 10 轮：λ=0.14 的分时与整体对比 4 轮、24 GiB 锚点 3 轮（含原生）、λ=0.14 的不分时三轮（含余量消融）；
  - 预测 `forecast-v13-s919-l14.json`（SHA `c50f489e…`，在 6 策略版本之后追加了 `ca_full_v2_nh`，原 6 个策略的预测逐位不变）和 `forecast-v13-s911-c24.json`（SHA `9cb6f3d4…`）。
  - 分时策略的预测区间是退化的（噪声模型只扰动并行时的干扰），轮次间波动以真机的 1.8% 为准。
  - `sim/predict.py` 的 `summarize` 改为在所有副本都没有值时返回 NaN（原生在 λ=0.14 下所有副本都超时，无 Code 时段指标为空，曾导致预测脚本崩溃）。
- **V13 构建与提交**：
  - tag `v13-anchors-20261105`，manifest `4f6b6339…`，tarball `18ed0576…`，14:01 上传，远端 manifest 校验通过；
  - 与 V11/V12 的 upload 相比，只有 `runtime/main.py`（segments）和 `tests/test_main_fake.py` 不同；
  - 本地测试 46 个通过（含 `sim/test_page_pool.py` 的 3 个）；只含 upload 内容的布局下 42 通过、1 跳过；在 bundle 内检查了 segments 的 trace 路径与 prompt bank 校验和。
- **V13 = 291000**（14:02 提交）：`--time 4:45:00`，依赖 `afterany:290422`，排除 d1n41a23g03；已加入监视 bz7rpbvkv 的作业列表。未设 `XPLACE_POLICIES`，`main.py` 记录的默认 `--policies` 在给了 segments 时不使用。
- **V12 第八轮 `ca_full_tm_s4`**（周期切片 S=4，14:01–14:21，passed，117/117，30 次迁移全部校验通过，0 次抢占；持有 20 次、切片 16 次）：
  - 4 个 Code 的 TTFT 为 32.1、32.8、33.3、31.8s（均值 32.5s，预测 33.4 / 33.7s）；Conv TPOT 29.40ms（预测 30.3 / 30.1ms）。
  - Conv TTFT 1.43s（p99 6.38s），Conv SLO 0.87；最长停顿中位数 7.8s（按需切片 15.7s）。
  - 切片模拟（results-short §12.7，seed 914，S=4）为 33.7s / p99 6.0s / SLO 0.91，与真机相符。
  - **按需对周期**（作业内，`ca_full_tm_s4` → `ca_full_tm_slo`）：Code TTFT −5.5% [−13.6, +3.9]，Conv TTFT −34.1% [−44.6, −20.4]，Conv TPOT +1.2%，Conv SLO 0.87 → 1.00。V10 为 Code TTFT −8.2% [−16.2, −1.7]。
  - 判定 5（Code TTFT 变化 < 0）：点估计成立，n=4，CI 含 0。
- 14:21 第九轮 `no_move_fair_v2` 开始（最后一轮）。
- **抢占与回迁的关系**（V10–V12 所有发生过迁移的受控轮次，14:25 汇总）：

  | 轮次 | 轮数 | 有抢占的轮数 | 抢占合计 | 无 Code 时段各 rank Conv 数之差 | 单 rank Conv 映射峰值 |
  |---|---|---|---|---|---|
  | 有代价感知回迁（`ca_return_v2`、`ca_full_*`） | 16 | 0 | 0 | 1.5–2.4 | 11.1–15.1 GiB |
  | 迁移但不回迁（`mig_random_v2`、`mig_random_tm_v2`、`ca_admit_v2`） | 7 | 4 | 20 | 2.8–7.0 | 14.6–15.8 GiB |

  - 上限为 15.94 GiB。不回迁时，最满 rank 的 Conv 映射都接近或达到上限；4 个有抢占的轮次：V11 随机 10 次、V12 随机 2 次、V12 只做代价感知准入 5 次都在 trace 结束后的排空阶段；V11 随机分时的 3 次在 830–875s（14:50 核对后更正，原写作全部在排空阶段）。
  - 有回迁的 16 轮中，峰值最高的是 V11 的 `ca_full_v2`（15.09 GiB），仍低于上限。
  - 准入时留的一次性增长余量在两个 seed 上都没有作用（留与不留都是 0 次抢占）；起作用的是持续的代价感知回迁。这回答了"placement 要为请求增长留空间，但不能用完整输出长度"：做法是迁移后持续回迁、把最满的 rank 压下来，而不是在准入时多留余量。
- **V12 第九轮 `no_move_fair_v2`**（14:21–14:45，passed，117/117，0 次抢占）：
  - 4 个 Code 的 TTFT 为 95.3、412.6、370.7、325.8s（均值 301.1s，预测 284 / 280s）。420s 的 Code 也排队 44.6s：这一轮 420s 时没有一张 rank 放得下它（V11 的 `no_move_v2` 中它直接准入），属于轮次间差异。
  - Conv TTFT 144.5s（p99 320s，预测 136 / 134s），Conv SLO 0.41，Conv TPOT 31.15ms（预测 31.7 / 31.3ms）。
- **V12 结束**（14:45，exit 0，用时 3h18m）：
  - 已归档到 `evidence/xplace-v11-replicate-20261105/job-290422`（与 V11 同一 tag，SHA `7f4e07be…`），一次成功；manifest 校验通过，启动单测 40 通过、1 跳过；报告 `report.md` / `report.json`（窗口 420–900s）。
  - 正确性：9 轮共完成 1053 个请求，输出长度全部精确；239 次迁移全部逐字节校验通过；事件循环最大卡顿 0.28s，0 次超过阈值。
- **V12 判定**（§17.4）：
  1. 回迁带来 B（`mig_random_v2` → `ca_return_v2` 的 CI 上界 < 0）：**不成立**，−0.3% [−0.7, +0.1]。
  2. 代价感知准入约为 0（±2% 内或 CI 含 0）：**不成立**，+4.2% [+3.3, +5.3]，比这一轮碰巧较均衡的随机迁移更差。
  3. 抢占：`ca_return_v2`、`ca_full_v2` 都是 0 次，**成立**；`mig_random_v2` 2 次（V11 为 10 次，预测 5.8 / 11.9 次）。
  4. 增长余量：`ca_full_v2_nh` 与 `ca_full_v2` 都没有抢占，**维持负结果**；Conv TPOT 不分时 +1.3%、分时 +1.1%（留余量略差）。
  5. 切片：Code TTFT −5.5% [−13.6, +3.9]，**点估计成立**，CI 含 0。
  6. 公平门：Conv TTFT 144.5s，是 V11 `no_move_v2`（0.36s）的 401 倍，**成立**。
  7. 正确性：**成立**。
  - 轮次间波动（V11 → V12，同 trace、同节点、同代码，Conv TPOT）：`mig_random_v2` −4.4%，`ca_full_v2` −1.7%，`ca_full_tm_slo` +1.8%。
  - 判定 1、2 不成立的原因：这一作业的随机迁移碰巧只造成很小的失衡（无 Code 时段 Conv 数之差 2.79，其他随机迁移轮次为 4.6–7.0），回迁没有可改善的空间；而只做准入的那一轮失衡 6.43。B 在准入和回迁之间如何分配，取决于各轮最终留下多少失衡，单轮配对不足以分开。
- **样本外的预测误差**（V12 全部 9 轮，两个预测都在作业开始前写定）：

  | 误差 | 高水位 | 块级 |
  |---|---|---|
  | Code TTFT（均值 / 最大） | 3.4% / 5.7% | 3.1% / 6.9% |
  | Conv TPOT（均值 / 最大） | 1.9% / 3.2% | 1.8% / 5.5% |
  | 迁移数（平均绝对误差） | 20.3 | 8.4 |
  | 抢占（平均绝对误差） | 0.9 | 1.5 |

  块级模型的抢占误差主要来自这一作业的随机迁移轮次（预测 11.9 次，实测 2 次，低于其 p5）；只做准入那一轮预测 5.7 次，实测 5 次。
- **seed 914 的总表已齐**（R1–R14）：`tools/master_table.py`，V10 + V11 + V12，见 results-short §12.8。
- **14:58 V13（291000）排队**：gpu_a800 分区 152 张卡都在用，V13 因 Priority 排队，Slurm 估计 23:59 开始（按其他作业的时限估算，通常偏晚）。设计不需要改，监视 bz7rpbvkv 继续跟踪。
- **文档**：results-short 顶部新增"当前设定总览"（V10–V12 两个 seed 的核心对比），原 V7 总览改为历史总览；README、CLAUDE.md、论文结构与记忆已按 V12 更新；修正了"抢占都在排空阶段"的说法（V11 分时随机迁移的 3 次在 830–875s）。
- **V13 开始**（17:40，节点 d1n41a28g02，与 V10–V12 的 d1n41a28g03 不同）：启动中，探针结果待出。如果探针超过 16.5ms，作业会按 fail-fast 退出，需要排除这个节点后重新提交。
- **V13 启动通过**：探针 12.61ms（d1n41a28g03 上为 12.6ms），节点速度相同；第一段第一轮 `s919-l14/ca_full_tm_slo` 17:46 开始，占位后四卡容量 15.98 GiB，与 V10–V12 相同。
- **V13 第一轮 `s919-l14/ca_full_tm_slo`**（λ=0.14，17:46–18:06，passed，124/124，50 次迁移全部校验通过，0 次抢占；持有 12 次共 106s）：
  - 4 个 Code 的 TTFT 为 31.8、26.8、30.9、27.8s（均值 29.32s，预测 29.42s）；420、660、780s 的 Code 走迁移准入，540s 的直接准入。
  - Conv TPOT 29.71ms（预测 29.55ms），Conv TTFT 0.71s（p99 4.56s），Conv SLO 0.98；迁移 50 次（预测 45 次）。
- 18:06 第二轮 `s919-l14/no_move_v2` 开始。
- **V13 第二轮 `s919-l14/no_move_v2`**（λ=0.14，18:06–18:29，passed，124/124，0 次抢占）：
  - **4 个 Code 全部被挡**：在控制器排队 625、585、548、507s，trace 结束后才陆续进入（1045–1287s）；TTFT 均值 629.6s（预测 551s [540, 560]，偏高 14%），Code SLO 0.00。λ=0.14 下连 420s 的第一个 Code 都放不进。
  - Conv TPOT 29.55ms（预测 33.85ms [32.9, 34.5]，在区间外），Conv TTFT 0.24s，Conv SLO 0.98。
  - **整体对比**（作业内，`no_move_v2` → `ca_full_tm_slo`）：Code TTFT −95.3% [−95.7, −95.0]（预测 −94.7%），Conv TPOT +0.5% [−9.1, +10.3]（预测 −12.7%），Conv TTFT 0.24 → 0.71s。判定 3 成立（4 个 Code >70s，−95.3%，Conv 退化在 20% 内）。
  - **预测偏差的原因**：live 窗口内 97% 的 Conv token 在没有 Code 驻留时产生；simulator 让被挡的 Code 早约 70s 进入（979–1197s 对 1045–1287s），窗口内 Conv 有 22% 的 decode 时间与 Code 并行（live 13%），所以基线的 Conv TPOT 偏高 15%。属于 V11 中见过的阈值效应（某张 rank 何时首次降到能放下 Code），不是步时模型的问题。
  - 解读：这个负载下 `no_move_v2` 实际上把全部 Code 推到 trace 之后，Conv 在窗口内几乎不受干扰；本文策略在 Code 全部按时服务的同时，Conv TPOT 与之持平。
- 18:29 第三轮 `s919-l14/mig_random_tm_v2` 开始。
- **V13 第三轮 `s919-l14/mig_random_tm_v2`**（λ=0.14，18:29–18:50，passed，124/124，20 次迁移全部校验通过，0 次抢占；持有 4 次共 103s）：
  - 4 个 Code 的 TTFT 为 26.2、24.9、25.6、25.1s（均值 25.44s，预测 26.10s）；420、540、660s 的 Code 走迁移准入。
  - Conv TPOT 30.65ms（预测 31.30ms），Conv TTFT 2.16s（p99 22.7s），Conv SLO 0.88；迁移 20 次（预测 22 次）。
  - 无 Code 时段各 rank Conv 数之差 4.63、最满 rank 12.5 个，Conv 映射峰值 14.62 GiB；同一作业的 `ca_full_tm_slo` 为 2.15 / 11.25 个 / 14.06 GiB。按失衡机制，下一轮 `ca_full_tm_v2` 应低于本轮（预测 −6.0%）。
- 18:50 第四轮 `s919-l14/ca_full_tm_v2` 开始。
- **V13 第四轮 `s919-l14/ca_full_tm_v2`**（λ=0.14，18:50–19:10，passed，124/124，44 次迁移全部校验通过，0 次抢占）：
  - 4 个 Code 的 TTFT 为 25.4、25.0、25.1、25.6s（均值 25.29s，预测 25.94s）；Conv TPOT 29.14ms（预测 29.42ms），Conv SLO 0.88；迁移 44 次（预测 43 次）。
  - **判定 4 成立**：`mig_random_tm_v2` → `ca_full_tm_v2` 的 Conv TPOT −4.9% [−5.4, −4.6]（预测 −6.0%），Code TTFT −0.6% [−2.4, +1.3]。第三个独立 trace 上方向相同、CI 不含 0。
  - 按需切片（`ca_full_tm_v2` → `ca_full_tm_slo`）：Code TTFT +16.0% [+7.7, +24.2]，Conv TTFT −66.6%，与 V10（+16.1%）、V11（+20.4%）一致。
- **分时下的 B 与随机迁移多留下的失衡成正比**（三个 trace，无 Code 时段、Conv 未被持有的采样）：

  | trace | 随机：Conv 数之差 / 最满 rank | 代价感知：Conv 数之差 / 最满 rank | 多出的差 | 多出的最满 rank | 分时下的 B | B / 多出的差 |
  |---|---|---|---|---|---|---|
  | seed 911（V10） | 6.63 / 12.26 | 1.97 / 9.29 | 4.66 | 2.97 | −11.6% | −2.49% |
  | seed 914（V11） | 5.92 / 12.80 | 1.67 / 9.75 | 4.25 | 3.05 | −10.4% | −2.45% |
  | seed 919，λ=0.14（V13） | 4.63 / 12.50 | 2.47 / 11.13 | 2.16 | 1.38 | −4.9% | −2.27% |

  - 每多出 1 个单位的 Conv 数之差，Conv TPOT 约高 2.3–2.5%；按最满 rank 算，每多 1 个 Conv 约高 3.4–3.9%。n=3，作为观察记录。
  - seed 919 上 B 小，是因为这一轮随机迁移碰巧只留下较小的失衡，不是 B 在更重负载下变弱。代价感知的失衡在三个 trace 上都是 1.7–2.5。
  - 不分时下这个关系不成立（三次配对为 −1.95、−0.96、−3.9% / 单位），因为不分时下只有约 65% 的 Conv TPOT 发生在无 Code 时段，其余由并行干扰决定。
- 19:10 第五轮 `s911-c24/ca_full_tm_slo`（24 GiB 段）开始。
- **V13 第五轮 `s911-c24/ca_full_tm_slo`**（seed 911，24 GiB，19:10–19:30，passed，109/109，12 次迁移全部校验通过，0 次抢占）：
  - 占位后四卡容量 24.05 GiB（目标 256 个页组），轮后释放到 35.8–38.3 GiB，与 16 GiB 段相同。
  - **4 个 Code 全部直接准入**（排队 0.3–0.6s），没有一次迁移准入；12 次迁移都是回迁（9 次事务）。同一 trace 在 16 GiB 下（V10）有 30 次迁移，660s、780s 的 Code 走迁移准入。单 rank 总映射峰值 17.9 GiB，远低于 24 GiB：KV 充足时没有碎片。
  - Code TTFT 29.03s（预测 29.57s），Conv TPOT 29.12ms（预测 28.97ms），Conv TTFT 0.77s（p99 4.18s），SLO 1.00 / 1.00。
- 19:30 第六轮 `s911-c24/no_move_v2` 开始，之后是原生轮（24 GiB 下预计不截断）。
- **B 与失衡关系的模拟检验**（19:35，`sim/b_vs_imbalance.py`，数据 `evidence/xplace-sim/b-vs-imbalance-20261105.json`，SHA `9fd84614…`）：
  - 设置：非最终 seed 801–812 的 trace（λ=0.12、16 GiB、块级页模型），随机迁移每个 trace 用 5 个随机种子（准入的随机种子与噪声都变），代价感知各跑 1 次，共 60 个配对。
  - 分时：B 与"随机迁移多留下的 Conv 数之差"的相关系数 r = −0.79，斜率 −1.9%/单位（过原点 −2.15%/单位）；真机三个 trace 为 −2.3~−2.5%/单位。60 个配对中代价感知全部更好（B 在 −11.9% 到 −0.2% 之间）。
  - 不分时：r = −0.69，斜率 −1.75%/单位；60 个配对全部为负（−11.3% 到 −0.4%）。
  - 抢占：不分时下随机迁移 60 轮中 46 轮抢占、共 643 次（平均 10.7 次/轮），代价感知 12 轮共 12 次（1.0 次/轮）；分时下随机迁移 8 轮共 56 次，代价感知 0 次。
  - 结论：模拟支持真机上的机制——代价感知 placement 的收益取决于随机迁移会造成多大的失衡；方向在所有配对中都一致，幅度随运气变化。可作为论文图 7(b)：真机的点叠加在模拟的点云上。
- **V13 第六轮 `s911-c24/no_move_v2`**（seed 911，24 GiB，19:30–19:52，passed，109/109，0 次抢占）：
  - 4 个 Code 全部直接准入（排队 0.2s），TTFT 为 61.5、64.3、61.1、57.6s（均值 61.12s，预测 59.64s）；单 rank 总映射峰值 16.97 GiB。判定 1 中受控基线的部分成立（4 个 Code 都 ≤70s），原生部分待第七轮。
  - Conv TPOT 39.40ms（预测 39.62ms），Conv TTFT 0.74s（p99 2.12s），Conv SLO 0.91。
  - **判定 2 成立**：`no_move_v2` → `ca_full_tm_slo` 的 Code TTFT −52.5% [−54.7, −50.5]，Conv TPOT −26.1% [−27.9, −24.3]（预测 −50.3% / −26.9%）。KV 充足时没有碎片，收益全部来自分时。
  - **同一 trace 跨预算对比**（V10 16 GiB → V13 24 GiB，跨作业、不同节点，探针相同）：
    - `ca_full_tm_slo`：Code TTFT −2.6%，Conv TPOT −1.2% [−1.4, −1.0]。本文策略在 16 GiB 下已达到它在 24 GiB 下的性能，迁移抵消了少 1/3 KV 的影响；
    - `no_move_v2`：Code TTFT −76.5%（不再被挡），但 Conv TPOT +27.6% [+20.6, +34.3]：Code 能及时进入后与 Conv 并行，Conv 受干扰更多。
- 19:52 第七轮 `s911-c24/kvcached_native` 开始。
- **V13 第七轮 `s911-c24/kvcached_native`**（seed 911，24 GiB，19:52–20:13，passed，109/109，不截断，0 次抢占）：
  - 4 个 Code 由 vLLM DP 负载均衡分到 rank 0–3，到达即进入，TTFT 为 61.2、52.1、67.6、58.3s（均值 59.82s，预测 60.54s），全部 ≤70s。**判定 1 成立**（受控基线与原生的 Code 都 ≤70s，原生不截断）。单 rank 总映射峰值 18.94 GiB。
  - Conv TPOT 40.34ms（预测 40.00ms），Conv TTFT 0.36s（p99 1.41s），Conv SLO 0.91。
  - **KV 充足时原生与控制器基线相同**（作业内，`kvcached_native` → `no_move_v2`）：Code TTFT +2.2% [−7.3, +16.6]，Conv TPOT −2.3% [−2.9, −1.8]。同一 trace 在 16 GiB 下（V10），原生的 2 个 Code 到 1900s 都没开始、Conv 约 300ms/token。原生的崩溃只发生在 KV 不够时（被挡的请求引起 DP 空转）。
  - **原生 → 本文策略**：Code TTFT −51.5% [−57.8, −44.6]，Conv TPOT −27.8% [−29.3, −26.4]（预测 −51.0% / −27.5%）。
  - **控制器准入的代价**（如实记录）：Conv TTFT 原生 0.36s，`no_move_v2` 0.74s（+104%），本文策略 0.77s（+112%）。多出的约 0.4s 来自控制器侧准入，主要是 Code prefill 期间到达的 Conv（见上文 Conv TTFT 的拆分）；都远在 5s 的 SLO 内。
- 20:13 第八轮 `s919-l14-corun/mig_random_v2` 开始（λ=0.14 不分时三轮，判定 5、6）。
- **RQ4 锚点与模拟曲线的对照**（20:20，V13 前七轮；"该 trace 预测"为 §18.4 的预注册预测，"扫描均值"为 §12.7 中同一负载、同一预算下 seed 801–806 的 12 次模拟）：

  | 点 | 策略 | Code TTFT：真机 / 该 trace 预测 / 扫描均值 | Conv TPOT：真机 / 该 trace 预测 / 扫描均值 |
  |---|---|---|---|
  | λ=0.14，16 GiB | `ca_full_tm_slo` | 29.3 / 29.4 / 29.9s | 29.71 / 29.55 / 31.34ms |
  | λ=0.14，16 GiB | `ca_full_tm_v2` | 25.3 / 25.9 / 25.9s | 29.14 / 29.42 / 30.95ms |
  | λ=0.14，16 GiB | `mig_random_tm_v2` | 25.4 / 26.1 / 26.0s | 30.65 / 31.30 / 33.65ms |
  | λ=0.14，16 GiB | `no_move_v2` | 629.6 / 551.4 / 594.4s | 29.55 / 33.85 / 28.60ms |
  | λ=0.12，24 GiB | `ca_full_tm_slo` | 29.0 / 29.6 / 29.1s | 29.12 / 28.97 / 28.52ms |
  | λ=0.12，24 GiB | `kvcached_native` | 59.8 / 60.5 / 59.1s | 40.34 / 40.00 / 37.24ms |
  | λ=0.12，24 GiB | `no_move_v2` | 61.1 / 59.6 / 60.9s | 39.40 / 39.62 / 37.81ms |

  - 有迁移或分时的策略，Code TTFT 与两种模拟都相差 ≤0.6s；Conv TPOT 与该 trace 预测相差 ≤2%。
  - 偏差最大的是 λ=0.14 下 `no_move_v2` 的 Conv TPOT：它取决于被挡的 Code 何时进入（阈值效应），各 trace 之间差别大；真机落在两种模拟之间。
  - 结论：图 10(b)(c) 的模拟曲线在两个偏离工作点的位置都有真机锚点，可以直接叠加。
- **V13 第八轮 `s919-l14-corun/mig_random_v2`**（λ=0.14，不分时，20:13–20:35，passed，124/124，21 次迁移全部校验通过，**2 次抢占**）：
  - 4 个 Code 的 TTFT 为 62.1、59.0、63.7、59.4s（均值 61.03s，预测 60.71s）；420s、660s 的 Code 走迁移准入。
  - Conv TPOT 41.98ms（预测 42.94ms [42.05, 44.89]，略低于 p5），Conv TTFT 0.79s，Conv SLO 0.86。
  - 失衡严重：无 Code 时段 Conv 数之差 6.43、最满 rank 16.15 个，Conv 映射峰值 15.84 GiB（上限 15.94）。
  - 抢占 2 次，在 890s 和 965s，两个被抢占的 Conv 都在窗口内，最长停顿只有 2.6–2.7s（重算很快，V11 中为 79–81s）。
  - **抢占数远低于预测**（17.4 [8.9, 28.7]）。原因是预测方法：`sim/predict.py` 的副本只改变干扰与 prefill 噪声，随机迁移的准入随机种子固定，所以随机迁移的抢占区间被低估。
    - 模拟中同一 trace、换 5 个随机种子，随机迁移的抢占数差别很大（λ=0.12：trace 804 为 0/1/2/14/18，811 为 0/1/6/6/22；60 次中 22 次 ≤2）；
    - 真机 seed 914 上同一策略两次运行为 10 次和 2 次。
    - 以后对随机策略做预测时应同时改变准入随机种子。
- 20:35 第九轮 `s919-l14-corun/ca_full_v2` 开始。
- **预测区间的修正**（20:45，事后分析，不改变 §18.4 的预注册预测）：
  - `sim/predict.py` 新增 `--vary-admission-seed`：每个副本同时改变准入随机种子（副本 0 保持默认种子）。不加这个选项时输出不变，已核对能逐位重现 `forecast-v13-s911-c24.json`。
  - 用与预注册相同的策略顺序（噪声种子相同，只改准入种子）重做 λ=0.14 的预测：
    - 代价感知策略完全不变（不使用随机数），说明这个选项只影响随机迁移的选择；
    - `mig_random_v2` 的抢占均值仍为 17.4，但 90% 区间从 [8.9, 28.7] 变为 [1.0, 34.2]，真机的 2 次落在区间内；
    - `mig_random_tm_v2` 的 Conv TPOT 区间从退化的 31.30ms 变为 [30.80, 32.81]（真机 30.65ms）；
    - 分时下的 B 从退化的 −6.0% 变为 −6.8% [−10.3, −4.5]，真机 −4.9% 在区间内。
  - 结论：预注册预测中随机策略的区间偏窄，是因为没有改变准入随机种子；分时下代价感知策略的区间仍是退化的（模型中既没有并行干扰噪声，也不用随机数），真机上同一策略两次运行差 1.8%。以后的预测对随机策略默认加 `--vary-admission-seed`。
- **V13 第九轮 `s919-l14-corun/ca_full_v2`**（λ=0.14，不分时，20:35–20:57，passed，124/124，47 次迁移全部校验通过，**0 次抢占**）：
  - 4 个 Code 的 TTFT 为 60.0、56.5、51.8、64.0s（均值 58.09s，预测 61.11s）；420、660、780s 的 Code 走迁移准入。
  - Conv TPOT 39.44ms（预测 40.10ms），Conv TTFT 0.68s，Conv SLO 0.92；迁移 47 次（预测 56 次）。
  - **判定 5 成立**：抢占 0 次，随机迁移 2 次。无 Code 时段 Conv 数之差 2.30（随机 6.43），Conv 映射峰值 15.09 GiB（随机 15.84 GiB）。
  - **不分时下的 B**（作业内，`mig_random_v2` → `ca_full_v2`）：Conv TPOT −6.1% [−6.5, −5.6]（预测 −6.6%），Code TTFT −4.8% [−14.9, +4.8]，Conv TTFT −14.0% [−31.9, +6.4]。
    - 四次独立配对（三个 trace）：−4.3%、−4.6%、−2.0%、−6.1%，全部为负、CI 都不含 0；
    - 这一轮随机迁移多留下的 Conv 数之差为 4.13，B / 多出的差 = −1.48%/单位，与不分时下其他三次（−0.96 ~ −3.9%/单位）及模拟（−1.75%/单位）一致。
- 20:57 第十轮 `s919-l14-corun/ca_full_v2_nh` 开始（最后一轮，判定 6）。
- **V13 第十轮 `s919-l14-corun/ca_full_v2_nh`**（λ=0.14，不分时，不留增长余量，20:57–21:17，passed，124/124，27 次迁移全部校验通过，**2 次抢占**）：
  - 4 个 Code 的 TTFT 为 55.3、58.9、57.3、58.0s（均值 57.37s，预测 61.43s）；540s、780s 的 Code 直接准入。
  - Conv TPOT 39.05ms（预测 39.89ms），Conv TTFT 0.57s，Conv SLO 0.92；无 Code 时段 Conv 数之差 1.35，Conv 映射峰值 13.59 GiB。
  - **抢占的位置**：838s、849s，rank 3。780s 的 Code 不留余量、直接准入到 rank 3（上面有 19 个 Conv）；在约 58s 的并行 prefill 期间这些 Conv 继续增长，rank 3 的总映射达到 15.75 GiB（Code 9.28 + Conv 6.47，上限 15.94），vLLM 抢占了最新的两个 Conv（最长停顿 5.6s、20.3s）。
    - 同一 trace 上留余量的 `ca_full_v2`，780s 的 Code 走迁移准入，先迁走了一部分 Conv，0 次抢占。
    - 这正是余量要防的情形：不分时下 Code 驻留期间 Conv 继续增长。分时下 Conv 被持有、不增长，所以分时下余量没有作用（V10、V12）。
- **V13 结束**（21:17，exit 0，用时 3h36m；监视器 21:41 才报告，期间的轮询因 ssh 超时跳过）：
  - 已归档到 `evidence/xplace-v13-anchors-20261105/job-291000`（SHA `95418cb1…`），一次成功；manifest 校验通过，启动单测 42 通过、1 跳过（含新增的 2 个 segments 测试）。
  - 报告：λ=0.14 两段合并为一个目录做作业内配对（`report-s919-l14.md/json`），24 GiB 段单独报告（`report-s911-c24.md/json`），总表 `master-table-v10-v13.md`。
  - 正确性：10 轮共完成 1195 个请求，输出长度全部精确；221 次迁移全部逐字节校验通过。
- **V13 判定**（§18.5，`/tmp/v13_verdict.py`）：
  1. KV 充足时没有碎片：**成立**。24 GiB 下受控基线与原生的 8 个 Code 都 ≤70s（52.1–67.6s），原生不截断。
  2. KV 充足时收益来自分时：**成立**。Code TTFT −52.5% [−54.7, −50.5]，Conv TPOT −26.1% [−27.9, −24.3]。
  3. 更重负载下的整体：**成立**。基线 4 个 Code 全部被挡；Code TTFT −95.3% [−95.7, −95.0]，Conv TPOT +0.5% [−9.1, +10.3]。
  4. 分时下的 B（第三个 trace）：**成立**，−4.9% [−5.4, −4.6]。
  5. 抢占：**成立**，`ca_full_v2` 0 次，`mig_random_v2` 2 次。
  6. 增长余量：**"更重负载下余量防止了抢占"**。不留余量 2 次、留余量 0 次；代价是 Conv TPOT +1.0% [+0.7, +1.4]（留余量时多迁移，47 对 27 次）。n=1，但抢占的位置与机制都对得上。
  7. 正确性：**成立**。
- **样本外预测误差**（V13 全部 10 轮，块级页模型，作业开始前写定）：Code TTFT 平均 3.6%（中位 2.5%，最大 12.4%），Conv TPOT 平均 2.6%（中位 1.3%，最大 14.5%），迁移数平均绝对误差 4.2，抢占 1.65。最大误差都来自 λ=0.14 的 `no_move_v2`（阈值效应）；除去它，Code TTFT ≤7.1%（最大为 `ca_full_v2_nh`，预测 61.4s、真机 57.4s）、Conv TPOT ≤2.3%。抢占误差主要来自随机迁移（预测 17.4，真机 2，见 §18.6）。
- **下一步的判断**（22:05）：预注册的实验矩阵已完成（两个 seed 的 R1–R14 + 两个 RQ4 锚点 + 机制的模拟检验），不再提交新的 GPU 作业。
  - 唯一还能想到的优化杠杆是更紧的回迁（B 与失衡成正比）。但代价感知策略无 Code 时段的 Conv 数之差已是 1.3–2.5，不迁移的轮次自然就有 1.7–2.9（按请求数路由加上请求随机结束），已接近下限。降到最好的 1.3 左右，按 −2.4%/单位估计收益约 1%，小于分时策略的轮次间波动（1.8%），还要多付迁移暂停。不值得再做真机实验。
  - 待用户决定：commit/push V7–V13 的代码与文档；是否做离线 MILP 最优性差距评估；是否在当前长度设定下补控制面 v1 的对照；是否开始画论文图与写正文。
