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
