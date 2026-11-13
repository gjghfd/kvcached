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

## 2026-11-06 最终方案的消融（V14）与论文图

- **commit**：用户 11-06 要求 commit。kvcached `275b437`（"feat(xplace): rerun the campaign under current-length admission (V8-V13)"），27 个文件，未 push；未跟踪的 `kvcached/kvcached/tests/test_dp_opt_public_population_calibration.py` 不是本 campaign 的文件（10-31，Codex 时期），没有提交。提交前在 research-summary 中补了当前长度准入的结论，本地测试 46 个通过。
- **V14 预注册**（protocol §19）：最终方案 `ca_full_tm_slo` 在分时 + 按需切片下的 5 个单组件变体（去掉迁移、随机 placement、去掉回迁、去掉代价感知准入、v1 控制面），两个 seed，12 轮。
  - 新增策略只改 `POLICIES`；新增 5 个参数化 fake 引擎测试；本地 51 个测试通过，只含 upload 的布局下 47 通过、1 跳过。
  - 预测 `forecast-v14-ablation-s911.json`（SHA `029b98a0…`）、`-s914.json`（SHA `9f4ab8d1…`），用 `--vary-admission-seed`。
- **V14 = 292505**（01:17 提交）：tag `v14-ablation-20261106`，manifest `7e686028…`，tarball `fce8c3ad…`，远端校验通过；`--time 5:30:00`，排除 d1n41a23g03；监视 bakgppk22。
- **排队**（01:45）：gpu_a800 0 张空卡，调度器估计 22:35 开始，期间画论文图。
- **论文图**（`figures/`，`make_all.py` 一次生成 10 张，输出 `Paper/figures/` 的 PDF/SVG/PNG，约 8s）：
  - 数据图 8 张：图 1 碎片、图 2 DP 填充、图 3 时间线、图 6 帕累托、图 7 机制、图 8 逐项消融、图 9 CDF、图 10 影响因素；示意图 2 张：图 4 架构、图 5 迁移事务与分时时序。
  - 图 8 的 V14 五行暂用预注册预测（浅色、无误差棒）占位；V14 归档后在 `data.py` 的 `RUNS`/`JOBS` 登记即自动换成真机配对。
  - 图 10：原生在 V10/V11 被截断，真机点画成带上箭头的下界；同一 λ 的两个 seed 左右错开。
  - 图 5 的迁移统计从 V10 归档计算：306 次迁移全部逐字节校验通过，每次复制中位 0.05s。
- **V14 分析预备**：`master_table.py` 加 R15–R19（`no_move_tm_slo`、`mig_random_tm_slo`、`ca_admit_tm_slo`、`ca_return_tm_slo`、`ca_full_tm_slo_v1`）及其对 R12 的单组件对照；`figures/data.py` 已登记 V14 的 4 个 segment 路径（归档前不存在，图 8 自动退回预测）。§19.4 判定脚本用 V10/V11 的替身数据空跑通过；本地测试 51 个通过。
- **V14 运行**（03:23 开始，d1n41a28g02，探针 12.6ms，占位后容量 15.98 GiB/卡，与 V10–V13 相同）。segment a（两个 seed 各 3 轮）全部 passed，0 次长度错误，迁移全部校验通过：
  - `ca_full_tm_slo`：911 为 29.7s / 28.7ms，914 为 30.6s / 29.7ms，都 0 次抢占、SLO 1.00/1.00；与 V10、V11、V12 的同名轮次差 ≤2.5%。
  - 随机 placement（`mig_random_tm_slo` → 最终）：Conv TPOT 911 −10.2% [−11.1, −9.3]、914 −10.8% [−11.8, −9.9]（预测 −4.3%、−7.0%）；Code TTFT 不显著。随机迁移在 914 上 10 次抢占（预测 1.6 [0, 6]）：780s 的随机准入把 rank 0 的 13 个 Conv 迁到 rank 1–3，之后不回迁；rank 2 的 Conv 映射在 835s 达到 15.84 GiB（上限 15.94），rank 1 有 21 个 Conv，迁过去的 5 个 Conv 共被抢占 10 次。无 Code 时段各 rank Conv 数之差：随机 6.4 / 5.7，最终 1.7 / 2.0，约 −2.2、−2.9%/单位。
  - 去掉迁移（`no_move_tm_slo` → 最终）：Code TTFT 911 −83.9% [−91.5, −6.8]、914 −94.8% [−95.6, −93.5]；Conv TPOT +8.1% [+5.2, +10.8]、+16.7% [+11.5, +21.0]。914 上 4 个 Code 全部排到 1114s 之后（预测 345s，实测 589s，+71%）：阈值效应，420s 时最空 rank 的 Conv 映射为 6.0 GiB，同作业的另两个策略为 5.3、5.6 GiB 并直接准入，V11 的 `no_move_tm_v2` 为 5.5 GiB 并直接准入。所以这一行的 Conv TPOT 对照的是一个窗口内没有服务任何 Code 的变体；20% 约束的基线仍是 `no_move_v2`。
- **V14 segment b，seed 911**（与 s911-a 的 `ca_full_tm_slo` 配对，同一作业）：
  - 去掉回迁（`ca_admit_tm_slo` → 最终）：Conv TPOT −3.5% [−3.7, −3.2]（预测 −3.0%），Code TTFT −0.2%。只有 660s 的 Code 要迁移（10 个），780s 的 Code 直接进入上次腾空的 rank。
  - 去掉代价感知准入（`ca_return_tm_slo` → 最终）：Conv TPOT −3.3% [−3.5, −3.1]（预测 −0.4%），Code TTFT +1.0%。
  - 单个组件（相对随机迁移、不回迁）：只做代价感知准入 −7.0% [−7.6, −6.3]，只做代价感知回迁 −7.2% [−8.0, −6.3]，两者都做 −10.2%。两者部分替代：都在缩小准入后留下的失衡（无 Code 时段 Conv 数之差：随机 6.4，只准入 3.8，两者 1.7）。
  - 代价感知准入的新机制：配合回迁时会复用上一个 Code 腾空的 rank（回迁计入下一个 Code 的空间风险，没有把这张 rank 填满），第二次准入只迁 1–5 个 Conv（最终方案 6 轮：V10 2、V11 3、V12 4、V13 5、V14 1 / 2）；随机准入换一张 rank，第二次迁移准入迁 11–13 个。V10 的"代价感知准入 ≈ 0"只适用于第一次准入。
  - v1 控制面（`ca_full_tm_slo_v1` → 最终）：Code TTFT +1.3% [−0.3, +3.0]（预测 −4.4%，不成立），Conv TPOT −1.7% [−1.8, −1.6]，Conv TTFT +0.9%（不显著）；Conv 准入排队均值 0.28 → 0.17s、p99 1.08 → 0.41s。分时下 v1 只让 Code 与 Conv 在下一个边界前重叠约 1s，所以 `hold_first` 的作用小。v1 的迁移 48 次（最终 23 次）：723、774s 的两次回迁把 rank 3 填回，780s 的 Code 只能换 rank 2 再迁 10 个，属于回迁阈值在小扰动下翻转，不归因于控制面。
- **V14 segment b，seed 914**：去掉回迁 −6.4% [−7.1, −5.9]（预测 −4.8%）；去掉代价感知准入 −1.4% [−1.6, −1.2]（预测 +0.2%）；只做代价感知准入 −4.7%、只做回迁 −9.6%（相对随机迁移）。v1 控制面：Code TTFT +0.9%（不显著），但 Conv SLO 0.91、TTFT p99 7.03s（v2 1.00、3.73s）：420s 的 Code 准入进行中到达的 3 个 Conv 被挡 4.6–5.1s（v2 0.04–0.22s），807–808s 到达的 2 个 Conv 准入晚、错过最后一次切片。
- **V14 结束**（07:36 COMPLETED，exit 0，12 轮 passed）：归档 `evidence/xplace-v14-ablation-20261106/job-292505.tar.gz`（SHA `e86d24b6…`，远端与本地一致），slurm 日志在 `logs/`。报告 `report-s911.md`、`report-s914.md`，判定 `verdict-v14.txt`，总表 `master-table-v10-v14.md`（R1–R19）。
  - §19.4：1–3、6、7 成立；4 为小而一致的收益；5 的 Code TTFT 预测不成立，差别在 Conv TTFT 尾部（protocol §19.5）。
  - 正确性：1356 个请求输出长度精确，310 次迁移全部逐字节校验通过。
  - 预测误差：Code TTFT 平均 8.1%（除去阈值效应的一轮 2.4%），Conv TPOT 1.8%；低估 B、随机迁移的抢占和代价感知准入。
  - 文档：results-short §12.10 与总览、protocol §19.5、README §5、CLAUDE.md、research-summary、Paper/structure.md（R15–R19、§4.1/§4.2/§4.4、§5.4、图 8）已更新；图 7(b) 加入 V14 的两个真机点，图 8 全部换成真机配对。
- **下一步**：没有待跑的真机实验。待用户决定：是否 commit / push 本轮改动（V14 策略与测试、protocol §19、作图脚本、文档）；离线 MILP 最优性差距评估；图 4、图 5 示意图是否用作图软件重画。

## 2026-11-06 上午：不留余量的受控基线（V15）

- **问题**（用户 11-06）：受控基线的 Code 直接准入也要留 Conv 增长余量 + 1 个页组（0.7–0.9 GiB/rank），其中"单独速率 15s"的下限在分时下也生效，而分时期间 Conv 不增长。用户的反事实：8 个不迁移基线轮中 4 轮的第一个被挡 Code 不留余量就能进（V11 seed 914 的 540s Code、V14 seed 914 的 420s Code、V9c seed 911 的 660s Code、V13 λ=0.14 的 420s Code）。复核：V11 在 540.7s 最空 rank 空闲 9.94 GiB、需要 9.19 GiB、13 个 Conv；V14 在 420.2s 为 9.84 对 9.28 GiB。模拟（seed 914）：`no_move_v2` 471 → 296s，`no_move_tm_slo` 345 → 235s，与用户的数字一致。
- **改动**：
  - 新基线 `no_move_v2_nh`、`no_move_tm_slo_nh`（不留余量：原生准入语义 + 控制器侧排队 + 按负载选 rank）；
  - 余量规则 v4.4（`planner-params-v4.json`，`relief_scope: "destinations"`，SHA `6b22fefe…`）：15s 下限只用于迁移与回迁的目的地，Code 自己的 rank 只留驻留期间的增长加 1 个页组；
  - 新增 5 个测试，本地 56 个通过，只含 upload 的布局下 52 通过、1 跳过。
- **预注册**（protocol §20）：预测 `forecast-v15-nh-s914/s911/s919-l14.json`。v4 下留余量与不留余量的基线几乎相同，最终方案 v3 → v4 只少迁移（914：51 → 40 次），指标变化 ≤0.7%。对 `no_move_v2_nh`：Code TTFT −89.7% / −89.2% / −94.7%，Conv TPOT −0.6% / −1.3% / −12.4%；对 `no_move_tm_slo_nh`：−87.0% / −85.0%，Conv TPOT +10.9% / +11.2%。
- **V15 = 293986**（10:20 提交，立即开始，d1n41a21g03）：tag `v15-nh-baselines-20261106`，manifest `e2d39500…`，tarball `65b4a276…`；8 轮，3 个 segment（s914、s911、s919-l14），`--time 4:30:00`；监视 bd40uudc1。
- **293986 探针失败**（10:27）：d1n41a21g03 的探针 18.8ms（上限 16.5ms），启动 fail-fast 中止，未跑任何轮次；归档 `job-293986.tar.gz`（SHA `8864543b…`）。用同一 upload 重新提交 **V15 = 294007**（10:28），排除 d1n41a23g03、d1n41a21g03；监视 bob0n0iqr。
- **RQ4 扫描（planner v4）**：`rq4-sweep-v4-20261106.json`，`no_move_v2_nh`、`ca_full_v2`、`ca_full_tm_slo`，432 次。16 GiB 下不留余量的基线 Code TTFT 中位数：λ=0.10 163 → 102s，0.12 332 → 236s，0.14 613 → 387s（旧规则留余量 → 不留余量），代价是抢占（0.12 平均 3.1 次，0.14 为 16.1 次）；20/24 GiB 下没有差别。最终方案 v3 → v4 在所有点上变化 ≤0.4s、≤0.3ms。`sweep_rq4.py` 加了 `--planner`、`--policies`（新策略加在 POLICIES 末尾，旧扫描的噪声种子不变）。图 10 改用这份扫描，基线换成不留余量的版本。
- **V15 运行**（294007，10:29 开始，d1n41a26g03，探针 12.7ms，占位后容量 15.98 GiB/卡）。segment s914 的 3 轮全部 passed，0 次抢占，迁移全部校验通过：
  - `ca_full_tm_slo`（v4）：30.4s / 29.97ms（预测 30.6s / 29.97ms），48 次迁移；540s 的 Code 在新规则下直接准入（Code rank 的余量 0.27–0.30 GiB，旧规则下需要迁移）。与 V11、V12、V14 的旧规则轮次（跨作业）：Code TTFT −0.6~−0.9%，Conv TPOT +0.7~+2.5%，在轮次间波动之内。
  - `no_move_v2_nh`：288.3s / 29.68ms（预测 295.8s / 30.14ms）。反事实兑现：540s 的 Code 0.3s 准入（V11 留余量时排队 542s）；660s、780s 的 Code 仍被挡 479s、441s。最终方案对它：Code TTFT −89.5% [−94.1, −46.1]，Conv TPOT +1.0% [−2.7, +4.4]（预测 −89.7% / −0.6%）。V11 `no_move_v2` → 本轮（跨作业）：Code −33.4%，Conv TPOT +3.7%。
  - `no_move_tm_slo_nh`：241.1s / 26.68ms（预测 235.2s / 27.02ms）。反事实兑现：420s 的 Code 0.9s 准入（V14 留余量时排队 694s）；660s、780s 被挡 469s、383s。最终方案对它：Code TTFT −87.4% [−93.2, −0.2]，Conv TPOT +12.3% [+9.7, +14.6]（预测 −87.0% / +10.9%）。V14 的"去掉迁移"（−94.8% / +16.7%）中多出的部分来自余量：V14 `no_move_tm_slo` → 本轮（跨作业）Code −59.0%。
- **V15 segment s911**（3 轮 passed，0 次抢占）：
  - `ca_full_tm_slo`（v4）：29.8s / 30.10ms（预测 29.9s / 29.29ms），40 次迁移。与 V10、V14（旧规则，跨作业）：Code TTFT +0.1% / +0.5%，Conv TPOT +2.1% / +4.7%；无 Code 时段失衡 2.12（V10 1.58、V14 1.67）。660s 的 Code 迁到 rank 0（旧规则两轮都是 rank 3），780s 的准入迁 6 个（旧规则 1–2 个），回迁 24 次（旧规则 12–18 次）。新规则在这个 trace 上可能有约 2–5% 的 Conv TPOT 代价（n=1，与轮次间波动同量级；模拟预测无差别）。
  - `no_move_v2_nh`：161.2s / 37.37ms（预测 277.0s / 29.67ms）。660s 的 Code 0.2s 直接准入（V10 留余量时排队 415s，用户的反事实与模拟都认为 seed 911 不受影响），只有 780s 的被挡 395s。3 个 Code 在窗口内与 Conv 并行，Conv TPOT 比 V10 的留余量基线高 21.0%（跨作业），Code TTFT 低 37.9%。最终方案对它：Code TTFT −81.5% [−91.6, −50.3]，Conv TPOT −19.4% [−21.5, −17.5]。
  - `no_move_tm_slo_nh`：204.1s / 26.52ms（预测 199.4s / 26.33ms）；660s、780s 的 Code 被挡 393s、308s。最终方案对它：Code TTFT −85.4% [−92.4, −0.5]，Conv TPOT +13.5% [+10.1, +16.7]（预测 −85.0% / +11.2%）。与 V14 的留余量版本（跨作业）：Code +11.0%、Conv TPOT −0.3%，余量在这个 trace 的分时下没有作用。
- **V15 segment s919-l14**：`ca_full_tm_slo`（v4）29.0s / 30.12ms（预测 29.0s / 29.66ms），53 次迁移，Conv SLO 0.98（TTFT p99 4.96s）；与 V13（旧规则，跨作业）Code −1.2%、Conv TPOT +1.4%。`no_move_v2_nh` 632.9s / 26.74ms，4 个 Code 被挡 452–659s（余量在 λ=0.14 上没有作用；比 V13 留余量基线的 Conv TPOT 低 9.5%，两轮都没有 Code 在窗口内运行，属于轮次间差别）。最终方案对它：Code TTFT −95.4% [−95.8, −94.9]，Conv TPOT +12.6% [+4.8, +20.3]。
- **V15 结束**（13:25 COMPLETED，exit 0，8 轮 passed）：归档 `evidence/xplace-v15-nh-baselines-20261106/job-294007.tar.gz`（SHA `11323f01…`）。报告 `report-s914/s911/s919-l14.md`，判定 `verdict-v15.txt`，总表 `master-table-v10-v15.md`（R1–R21）。
  - §20.5：1、2、5、6 成立；3 在 seed 914 上兑现，seed 911 的并行基线上 660s 的 Code 也直接准入（未预计），λ=0.14 上余量没有作用；4：v4.4 对最终方案 Code TTFT −1.2%~+0.5%，Conv TPOT +0.7%~+4.7%（6 次跨作业比较全为正），迁移多 3–17 次。
  - 预测误差：最终方案 Code TTFT 0.3%、Conv TPOT 1.5%；不留余量的并行基线 seed 911 −41.8%、λ=0.14 +14.9%（阈值效应）。
  - 图：图 6 加 R20、R21，20% 线改为相对 R20，R2、R15 空心；图 8 的"不迁移"改为 R21；图 10 改用 v4 扫描与 V15 锚点。
  - 文档：results-short §12.11 与总览（主基线改为 R20）、protocol §20.6、README、CLAUDE.md、research-summary、Paper/structure.md 已更新。
- **下一步**：没有待跑的真机实验。待用户决定：最终方案是否保留余量规则 v4.4（更合理，但 Conv TPOT 可能多约 2%，n=1）；commit / push 本轮改动。

## 2026-11-06 下午：通用版算法（`*_g`）

- **要求**（用户 11-06）：算法要通用，不区分 Code 和 Conv，例如请求放不下时不看它属于哪类请求；之后可能加入新的数据集。设计重写为 `Paper/design.md`（通用版），然后实现。
- **实现**（本地，未上真机）：
  - 每个受控策略都有通用版"原名 + `_g`"；旧版策略不变。
  - 涉及 `runtime/planner.py`、`cost_model.py`、`controller.py`、`trace_runner.py`，以及 `sim/simulator.py`。
  - 新参数文件 `evidence/xplace-inputs/planner-params-v5.json`：v4 加上 `tm_grace_s` 2、`history_s` 900、`rate_prior_s` 500。
- **验证**：
  - 原有 53 个测试通过；新增 `tests/test_general.py`，13 个通过，其中包括两个实例角色互换；`sim/test_general_sim.py` 3 个通过。
  - simulator 中 21 个旧策略 × V15 的 3 个 trace，输出 SHA-256 与改动前一致（脚本与改动前的备份在 `/tmp/xplace-general/`，未归档）。
- **模拟对比**（V15 的三个 trace，预测设定，每策略 5 个噪声副本；通用版对旧版）：
  - R12：Code TTFT −0.1%~−2.4%，Conv TPOT −0.1%~−0.7%，Conv TTFT −13.4%~+2.2%，迁移多 4–10 次；
  - R20：完全相同；
  - R21：Code TTFT −0.1%~−0.2%；
  - R16：Conv TPOT −1.0%~+1.2%；
  - R7（并行）：Conv TTFT p99 上升（seed 914：0.57 → 1.81 s），因为短请求也留余量；
  - 迁移准入只由 Code 触发，没有迁移过 Code；每个 Code 各开启一次分时，分时只落在 Code 所在的实例。
- **下一步**：论文改用通用版，需要用一个打包作业在两个 seed 上重跑 R12、R20、R21、R16 的通用版。待用户决定；未 commit。

## 2026-11-06 晚：通用版的真机实验 V16

- **预注册**：protocol §21。预测文件 `forecast-v16-general-s914/s911/s919-l14.json`（SHA `7e9f2fb0…`、`97549914…`、`a5e7c781…`），segments 文件 `segments-v16-general.json`（`75058df0…`），参数 `planner-params-v5.json`（`64dc0297…`）。
- **提交前修正**：`controller.view_exact` 的精确条件改为"剩余 prefill 按单独运行速度 + 空余预留 token 按 decode 上界"，新增单测。本地 66 个测试通过、1 个跳过（只含 upload 的布局）。
- **V16 = 295488**：
  - 17:41 提交，tag `v16-general-20261106`，manifest `726bf4cf…`，tarball `02f7c685…`；
  - 20 轮，7 个 segment（s914-a、s911-a、s914-b、s911-b、s914-c、s911-c、s919-l14）；
  - `--time 9:00:00`，排除 d1n41a23g03、d1n41a21g03；
  - 提交时 gpu_a800 没有空闲卡，需要排队。
- **295488 开始运行**（19:40，排队约 2 h，节点 d1n41a24g01）。监视任务 b1m1lodg3。
- 探针 12.7 ms，通过；19:5x 开始第一轮 `s914-a/ca_full_tm_slo_g`。
- **第一轮 `s914-a/ca_full_tm_slo_g`**（passed，117/117）：
  - Code TTFT 30.8 s，Conv TPOT 29.9 ms，Conv TTFT 0.94 s（p99 3.66 s）；预测 29.8 s / 29.7 ms；V15 的旧版为 30.4 s / 29.97 ms；
  - 4 个 episode，都在 Code 实例上，每个 Code 一个，持续 28–35 s；
  - 迁移准入 3 次，都由 Code 触发，540 s 的 Code 直接准入；迁移的 39 个请求都是 Conv；
  - 0 次抢占；快路径缓存命中 112 次、实时查询 5 次；事件循环没有卡顿。
- **segment s914-a 完成**（3 轮 passed，0 次抢占）：
  - R20g 275.4 s / 30.98 ms（预测 295.8 s / 30.1 ms）；R21g 192.9 s / 27.61 ms（预测 235.0 s / 27.0 ms）；
  - R20g → R12g：Code TTFT −88.8% [−93.7, −45.7]，Conv TPOT −3.5% [−7.2, −0.2]；
  - R21g → R12g：Code TTFT −84.0% [−91.2, +1.1]，Conv TPOT +8.3% [+4.6, +11.7]；
  - seed 914 上判定 1、2 成立。R21g 的 CI 上界越过 0，只有 4 个 Code。
- s911-a：R12g passed（109/109，33 次迁移，0 次抢占）；R20g passed（109/109），但 Conv 实例有 3 次抢占（预测 0；V15 的旧版 R20 为 0）。不留余量的并行基线在 Code 与 Conv 并行时可能抢占，作业结束后核对抢占发生的时间与 rank。
- **segment s911-a 完成**：
  - R12g 30.8 s / 29.45 ms，Conv TTFT 0.85 s（p99 3.69 s），SLO 1.00/1.00；
  - R20g 167.7 s / 36.34 ms（V15 旧版 161.2 s / 37.37 ms）；R21g 228.0 s / 26.12 ms（V15 204.1 s / 26.52 ms）；
  - R20g → R12g：Code −81.6% [−92.2, −45.7]，Conv TPOT −19.0% [−20.7, −17.2]（V15 −81.5% / −19.4%）；
  - R21g → R12g：Code −86.5% [−93.3, −0.0]，Conv TPOT +12.7%（V15 −85.4% / +13.5%）；
  - seed 911 上判定 1、2 成立。
  - R20g 的 3 次抢占：660 s 的 Code 直接准入 rank 0，该 rank 的映射到 9.38 + 6.56 = 15.94 GiB，即容量上限；随后在 672、687、721 s，rank 0 上的 Conv 增长使 vLLM 依次抢占该 rank 最新的 Conv。这是不留余量基线的预期行为。
- **segment s914-b 完成**（3 轮 passed，0 次抢占），与 s914-a 的 R12g 配对：
  - R16g → R12g：Conv TPOT −8.6% [−9.3, −8.0]（预测 −8.0%），seed 914 上判定 3 成立；
  - R17g（去掉回迁）→ R12g：−8.2% [−8.7, −7.7]（预测 −5.7%）；
  - R18g（去掉代价感知准入）→ R12g：+1.1% [+1.0, +1.2]（预测 −1.3%）；
  - 从随机迁移出发，只加回迁 −9.6%，只加代价感知准入 −0.4%。B 几乎全部来自回迁，与 V14 旧版一致（去掉回迁 +6.4%，去掉代价感知准入 +1.4%）。
  - 准入迁移数：R12g 为 2、8、5，另有回迁 24 次；R17g 为 4、10、19；R18g 为 8、13。
- **segment s911-b 完成**（3 轮 passed，0 次抢占），与 s911-a 的 R12g 配对：
  - R16g → R12g：Conv TPOT −4.6% [−4.9, −4.3]（预测 −4.6%），判定 3 在两个 seed 上都成立；
  - R17g → R12g：−3.7% [−4.4, −3.1]（预测 −3.6%）；R18g → R12g：−0.4% [−0.5, −0.3]（预测 −0.5%）；
  - 从随机迁移出发，只加代价感知准入 −0.9%，只加回迁 −4.2%；
  - R12g 两次准入的迁移数为 10、1：第二次复用上一个 Code 腾空的 rank；随机准入为 10、11（V14 中的机制）。
- **segment s914-c 完成**（3 轮 passed，0 次抢占），与 s914-a 的 R12g 配对：
  - R14g（去掉余量）→ R12g：Code +0.4%，Conv TPOT +1.3%。余量没有作用，与 V10–V15 一致；
  - R10g（去掉切片）→ R12g：Code +23.1% [+12.7, +32.4]，Conv TPOT +2.0%，Conv TTFT −74.8%（p99 25.6 → 3.7 s，Conv SLO 0.81 → 1.00）。判定 4b 成立，Code 的代价接近 25% 的上限（预测 +15%）；
  - R7g（去掉分时）→ R12g：Code −45.3% [−52.2, −40.5]，Conv TPOT −17.5% [−18.8, −16.3]，Conv TTFT +43%（0.65 → 0.94 s，在 SLO 内）。判定 4a 成立。
- **segment s911-c 完成**（3 轮 passed，0 次抢占），与 s911-a 的 R12g 配对：
  - R14g → R12g：Code +3.3%，Conv TPOT +0.2%，余量没有作用；
  - R10g → R12g：Code +22.7% [+12.3, +33.4]，Conv TPOT +3.0%，Conv TTFT −74.3%（p99 24.0 → 3.7 s，SLO 0.79 → 1.00）；
  - R7g → R12g：Code −53.5% [−58.2, −48.7]，Conv TPOT −29.3% [−31.1, −27.5]；
  - 判定 4 在两个 seed 上都成立。
- **V16 结束**（02:4x COMPLETED，exit 0，20 轮全部 passed）：
  - 归档 `evidence/xplace-v16-general-20261106/job-295488.tar.gz`（SHA `3b77f0a5…`，远端与本地一致）；
  - 报告 `report-s914/s911/s919-l14.md`：每个 seed 的 segment 合并为 `out/by-seed/` 后作业内配对；
  - 总表 `master-table-v16-general.md`（`master_table.py --general`，R1 原生取 V10/V11 作跨作业参照）；
  - 判定 `verdict-v16.txt`，通用性核对 `check-general.jsonl`（`tools/check_general.py`）。
  - protocol §21.4 的 7 条判定全部成立，结果见 protocol §21.5、results-short §12.12 与总览。
  - λ=0.14：R12g 28.8 s / 29.52 ms，R20g 610.7 s / 28.72 ms；R20g → R12g：Code −95.3%，Conv TPOT +2.8%。
  - 文档：results-short（通用版总览、§12.12）、protocol §21.5、README §5.0、CLAUDE.md、Paper/design.md §10、Paper/structure.md 已更新。
- **下一步**：
  - 论文图表改用 V16 的通用版数据（图 6、图 8、表 2）；
  - 是否补跑 R19g（v1 控制面）、R11g（周期切片）和原生 R1，待用户决定；
  - commit / push 待用户指示。

## 2026-11-08：图2校准验收、TP2修复与当前会话监控

- 用户明确要求修复后重跑并监控，后续由当前session处理，不再另起模型调用。保留历史PAUSE，不恢复旧observer。
- 核对原作业：300123于11-07 19:13:05 COMPLETED；300260于19:17:20 FAILED。旧事件event-0001于20:41:29定位到当前thread，但模型路由连续503并turn.failed，未分析或补跑。
- 两份raw已取回并核对SHA：300123为`2cfc42341c2f1584c82bcf597a40dc68b340935416857847bdff4c7ee892db60`；300260为`048ff611537bfe42bdf31efcd8a871e339f13c5db529545e45c1c44832da8b4e`。
- 300123验收通过：startup/job passed，DP4/TP1/EP4、两个实例四卡共址；60次有效测量、30点、4672请求逐请求重算通过。BS32/64/128最大失衡布局相对均衡布局分别+39.7%/+59.7%/+91.7%；每点两轮，同一作业，不代表跨节点稳定性。分析为`Paper/draft/imbalance-dense-analysis.json`。图2(b)保留双曲线，更新BS32/64的9/10点及min–max范围。
- 300260首个根因是`runtime/live_scheduler.py:123`拒绝TP2，不是最终的engine初始化包装异常。修复后TP2校准选CalibrationScheduler；迁移事务仍被拒绝，迁移调度器仍限TP1。新测试实例化实际调度器，避免仅验证参数却漏掉初始化约束。
- CPU验证：本地组合112 passed/1 skipped；独立工具与分析测试28 passed；冻结包96 passed/2 skipped（本机无torch的peer测试、未上传的report工具测试）。冻结包调度器/拓扑专项24 passed。后续前台监控状态转换补充回归后，followup工具23 passed。
- 新包`fig2-ep4-tp2-20261108-r1`：tar SHA `d88815cc518a9e0aeb0b9c4a4bf1c0b6a2df9bfe99753363826e5a1559b15124`，manifest SHA `3ac6ff0301d0cfdaf564a6e9782b33354db0a4364dfd10d56bee05bc330dea18`。修复指纹只改变engine与scheduler运行逻辑，扫描参数未变，父作业300260。
- 09:22:36提交303011，单节点四卡、1小时，提交时队列为空；已核对RUNNING，节点d1n41a26g03。没有修改远端共享安装，没有重复提交。
- 当前会话通过`claim-current/finish-current/monitor-current`处理，状态`enabled=false`表示后台模型dispatcher停用，并非前台监控停止。监控进程作为本回合工具运行；终态返回后在同一回合分析。若取消本回合或关闭IDE，它不能自动重新打开对话。

### 303011终态与r2预算修复

- 303011于10:13:30结束，FAILED 1:0，耗时50m47s；启动测试99 passed/1 skipped，DP2/TP2/EP4、四卡共址、逐TP worker身份检查通过。69次测量全部measured、零抢占，但计划72次，结果严格判为incomplete。
- 原始归档SHA `f28f9f53f2d1376a2e9007b0554532f961a8892ced76c9ddbbb508119e31170f`，10:15:19取回并核对。终态直接返回当前session；没有额外模型调用或后台exec resume。
- 原因：扫描上限2800秒，启动下一case前要求保留120秒。运行2706.802秒后无法满足预留，剩余`bs128_d0_r2`、`bs32_d1_r2`、`bs32_d12_r2`未执行；对应前两轮用时约39.2、23.3、26.8秒，合计约89秒。不是TP2初始化故障，也不是Slurm超时。
- r2修改内部扫描预算至3100秒，保留1小时硬限及全部72次、三轮、同一seed/order/输入输出长度；增加stop_reason、remaining_cases与预算字段。mock时钟测试复现旧预算截断，并验证新预算完成相同计划。65项相关CPU回归通过，冻结包测试继续执行。
- 旧69次作为不完整原始记录保留，不混入r2重复数；补跑重做完整计划。用户要求继续循环，但正负结果均保留，不通过删点、降低验收标准或挑选轮次“跑出效果”。
- r2冻结包98 passed/2 skipped；tag `fig2-ep4-tp2-20261108-r2`，tar SHA `ffd4c10ada787a8128e9c765d69addd8eec6881b2473a08b945b9f05ce581b0f`，manifest SHA `6fc246d2348a272797c693e79f36b78f32635bad8f38249a34acb7520f3489fc`。10:22:07远端校验通过，10:22:21提交303174，父作业303011，4卡/1小时；提交时队列为空，无并发allocation。
- 10:28:38查询303174为PENDING/Priority，Slurm预计17:44:16启动（仅调度估计，可变化）。不取消重投、不额外提交抢占资源；当前session继续监控。
- 10:45按用户要求通过当前thread的goal工具设立持续目标（active、无token预算），继续既有前台监控和受控补跑。不是新开CLI模型session；仅在72次完整测量、原始证据和论文记录验收后完成目标。交接状态为`evidence/xplace-codex-followup-20261107/current-handoff.md`。
- 10:47核对303174仍为PENDING/Priority，原监控正常，无重复提交。排队期间补上独立拓扑对比图的raw重算一致性检查及曲线/范围带测试；纸稿环境2项unittest通过。测试只写临时合成数据，未生成未验收TP2图片。
- 11:33:48作业303174实际启动，节点d1n41a21g03；11:36查询为RUNNING、启动测试进行中。未调整队列优先级，之前的14:25/17:44只是估计。新节点不同于r1，待实际测量和完整性验收，不能假定性能与r1相同。

### 303174终态与r3部署修复

- 303174于12:29:51结束，FAILED 1:0，56m03s；远端启动测试101 passed/1 skipped，启动拓扑passed。仍是69/72次全部measured、零抢占，stop_reason=scan_budget_exhausted，scan wall=3010.263s，余下case与r1相同。
- 归档于12:32:13取回并核对SHA `a74f07b8af4770d49c213d8e5f537d3f39efb174f15eb81477625a0ea77fc94a`。r1/r2的完整计划和已执行case集合完全相同；测量wall合计2684.17→2979.84秒，匹配case wall比值中位数1.0893，decode比值中位数1.0844。未执行的三点不能用前轮数据拼齐。
- r3采用部署修复：排除本次d1n41a21g03，保持扫描3100秒、case120秒、4卡/1h、全部72次及测量运行时不变；只额外在原始job记录SLURM_JOB_NODELIST。这里排除依据是本次时限不足，不是根据曲线方向选择结果，也不声称节点永久故障。
- r3新tag `fig2-ep4-tp2-20261108-r3`，manifest SHA `17a5dc7f08f52c59f14b46d022f110c85da3fa69727d0cf95e3fbd7bbe404ff3`，tar SHA `644d4f6ac0168583a995233c016eb55d6cafb72201d3ccc6848c9e92f78b6d3a`。已确认扫描配置和所有测量runtime文件与r2相同，部署排除进入独立修复指纹，冻结包测试进行中。
- r3冻结验证通过，12:41:00远端SHA核对通过；12:41:17提交303594，排除d1n41a21g03，父作业303174，单节点四卡/1小时，队列中无其他本用户作业。event-0004记录补跑关系，当前session继续前台监控；未用额外CLI模型续接。

### 303594完整通过与独立拓扑图

- 303594于12:41:26在d1n41a22g02启动，13:33:51以COMPLETED/0:0结束，52分25秒。当前session的前台监控收到终态并立即进入event-0005验收，没有外部模型续接或重复提交。
- 13:34:43归档取回，SHA `c2aee26ca9847b0fcd7bc7c87596801bdf69ffb58160c8e3954ef9ad85ca9d9a`，全部149个归档文件与解包内容一致。startup/imbalance/job全部passed；101项启动测试通过、1项跳过；8份worker验证DP2/TP2/EP4和两实例四卡共址。
- 扫描2803.956秒，72/72次全部measured、零抢占、无缺失；逐请求重算验证24点×3轮、5664请求，最短共同decode窗口13.933秒。分析为`Paper/draft/imbalance-ep4-tp2-analysis.json`。未混入两个69次的不完整父作业。
- BS32/64/128均衡→最大失衡的步时为15.826→20.104、18.290→28.812、24.905→47.568 ms，即+27.0%/+57.5%/+91.0%。最大失衡不一定最慢：BS32峰值在归一化失衡1.5，BS64也有末端回落；全部点及范围保留。
- 新增独立图7 `imbalance_topologies`，三个batch、两个拓扑独立面板、归一化极差；图2(b)仍为DP4 BS32/BS64双曲线。绘图前重算raw一致性，2项绘图测试通过。正文注明不同节点/运行时、作业内min–max和非单调结果，不宣称TP2迁移已验证。
- DOCX及PDF已重建，13页、7图、4表、2算法、8公式、12参考文献；完整验证器通过并核对两套校准归档、拓扑、全量请求及图像一致性。目视检查图7和预览第12页无裁切。65项相关CPU回归通过，只有既存asyncio_mode配置警告。
- event-0005记录303594为verified，验收收据为`evidence/xplace-fig2-ep4-tp2-20261108-r3/acceptance-303594.json`。本轮校准目标已达到，不再为了更大的增幅继续提交；历史失败和非单调点全部保留，旧PAUSE及observer停用状态不变。

## 2026-11-09：TP2 拟合和统一并发准备

- 当前session正常完成续接后，00:07:39、00:15:50、00:19:59、00:25:08 CST查询305641均为PENDING/Priority，无其他本用户作业、无启动日志；最后两次间隔约300秒。00:07的预计开始时间为18:15:52，仅调度快照，不取消重投。没有新allocation，旧PAUSE及observer不变。
- 新增 `tools/fit_measured_cost.py`，全量grid+imbalance相对NNLS、单调graph增量、五角色输入与原始interference SHA打包、失败候选/报告保留；加载器重算干扰窗口、真实prompt归一化、phase/bucket和fallback残差以及Code solo参数迁移。305641未启动，**未生成实际TP2参数，也未获得新GPU性能结果**。
- 新工具逐请求复核303594的72点、5664请求，全部布局/长度/共同decode窗口匹配。合成夹具另验证打包可重现及拒绝缺失源、错误拓扑、伪窗口、高残差；不把夹具结果当作实验结果。
- 新增 `--max-inflight MODEL0 MODEL1`，默认0/0不限。所有策略统一中心waiting queue、保持原arrival及TTFT，覆盖general/fast/native；不因上限阻塞触发碎片迁移，不因被上限阻塞的队列头永久禁用周期均衡。workload、峰值、截断和队列头阻塞时间有记录，实际数值留待容量pilot冻结。
- 首轮完整回归166 passed/1 failed/1 skipped：新增asyncio.run测试暴露旧cached-view同步单测在Python3.9无活动loop创建Lock的顺序依赖。改为在async scenario中构建controller，不改变runtime锁；组合回归21 passed，最终完整回归167 passed/1 skipped（194.40秒）。既存asyncio_mode配置警告和protocol.md:953尾空白保留，不修无关文件。
- 本轮新代码均未进入305641冻结包，local tar/manifest SHA复核与原提交一致。验收与代码SHA保存在 `evidence/ep4-tp2-ablation-20261108/local-preparation-20261109.json`；完整测试输出为同目录 `local-regression-20261109.log`。下一步仅在305641结束并通过逐分片/原始数据验收后生成真实代价，独立pilot后再进入正式配对，不启动混合拓扑。

### 305641 当前会话排队监控（00:30–01:03 CST）

- 00:30:54首查305641仍PENDING/Priority，预计18:15:52启动（可变调度估计）。随后当前session前台按300秒间隔探测六次，最后01:02:42仍同状态；没有其他本用户作业，没有startup-tests/main/phases/exit-status日志。
- 原始探测记录：`evidence/ep4-tp2-ablation-20261108/queue-monitor-20261108T163154Z.jsonl`，6/6 SSH成功，SHA `31ccc8a4dd62a940c141f426137a89f8c0f5e283743b502f236c36dd6b34f66e`。本轮不改runtime、不重复测试、不提交新allocation，不因排队重投；旧PAUSE及observer状态不变。
- 前台监控正常结束，更新并发布当前campaign，保持active和全部GPU门槛未通过。下一次续接先查实时状态；RUNNING读取阶段日志，终态归档并验收，不把本次会话结束当作实验完成。

### 305641 当前会话排队监控（01:06–01:37 CST）

- 01:06:06首查后按300秒间隔监控，七次查询至01:36:41均为PENDING/Priority，无其他本用户作业、无启动日志。01:06的Slurm预计开始时间变为11月9日09:21:50，较此前18:15:52提前；仍只是可变估计，不取消/重投。
- 原始记录：`evidence/ep4-tp2-ablation-20261108/queue-monitor-20261108T170601Z.jsonl`，7/7 SSH成功，SHA `f560a337cc59536ef37a48e124aec03617b311b1c676ef3bf4336701bfba3260`。只读监控未修改runtime或冻结包、重复测试、提交新allocation或启动混合拓扑；没有新GPU结果。
- 本轮前台命令已结束；更新并发布campaign，保持active，继续等待实际状态变化后验收或诊断。旧PAUSE和observer不变，历史记录保留。

### 305641 当前会话排队监控（01:41–02:12 CST）

- 01:41:11至02:11:46按300秒间隔共查询七次，均PENDING/Priority；无其他本用户作业、无startup-tests/main/phases/exit-status日志。01:41的Slurm预估仍为11月9日09:21:50，仅作调度快照，不据此取消/重投。
- 原始记录：`evidence/ep4-tp2-ablation-20261108/queue-monitor-20261108T174105Z.jsonl`，7/7 SSH成功，SHA `f7c03459f4fdee11fbfb56615c18d13f99400e361fe220bd85de49b1ced7c379`。没有修改runtime/冻结包、重复测试或新增allocation，没有新GPU结果，不进入混合拓扑。
- 前台监控正常结束，检查点保持active并发布至当前session的续接活动目录；全部GPU门槛仍未通过，历史记录、PAUSE和旧observer状态不变。

### 305641 当前会话排队监控（02:16–02:47 CST）

- 02:16:16至02:46:55按300秒间隔共查询七次，均PENDING/Priority；无其他本用户作业、无startup-tests/main/phases/exit-status日志。02:16的Slurm预估为11月9日22:42:25，比上次09:21:50推迟；只作可变调度快照，不据此取消/重投。
- 原始记录：`evidence/ep4-tp2-ablation-20261108/queue-monitor-20261108T181611Z.jsonl`，7/7 SSH成功，SHA `a262f56caec42266d1bc114d26bec8b1058816cb24021770ee2d52e5d2e60a2a`。没有修改runtime/冻结包、重复测试或新增allocation，没有新GPU结果，不进入混合拓扑。
- 已接收并结束本轮唯一前台监控，检查点保持active；发布至当前session的续接活动目录。全部GPU门槛仍未通过，历史记录、PAUSE和旧observer状态不变。

### 305641 当前会话排队监控（02:51–03:22 CST）

- 02:51:22至03:21:56按300秒间隔共查询七次，均PENDING/Priority；无其他本用户作业，startup-tests.log/main.log/out/phases.json/exit-status每次均明确记录MISSING。02:51的Slurm预估为11月9日19:59:34，比上次22:42:25提前；只作可变调度快照，不据此取消/重投。
- 原始记录：`evidence/ep4-tp2-ablation-20261108/queue-monitor-20261108T185117Z.jsonl`，7/7 SSH成功，SHA `7c50348fdb812b2ad69038141dcfd36d4946190e246fadc319c74cde880b679f`。没有修改runtime/冻结包、重复测试或新增allocation，没有新GPU结果，不进入混合拓扑。
- 本轮唯一前台监控正常结束；更新并发布campaign，保持active。TP2逐分片迁移、联合实测拟合及端到端配对仍待真实输出验收，所有历史记录、PAUSE和旧observer状态不变。

### 305641启动测试失败与续接错误态修复（2026-11-09 03:26–09:44 CST）

- 当前session七次前台检查中，03:57发现305641已于03:54:37启动；03:58:44以FAILED/1:0结束，startup tests为111 passed、2 failed、1 skipped。两个peer-transfer测试的vLLM替身缺少新增的get_tp_group；本地缺torch导致此前整组被跳过。未执行dataplane、grid或interference，不能计作GPU机制或性能结果。
- 03:59:23会话因openai_error/bad_response_status_code异常结束。launchd没有停止，但owner为systemError，旧snapshot_guard只允许idle；04:00至09:37连续68次owner_not_idle，未尝试续接，直到用户09:39手动恢复。问题不是Slurm仍在排队，而是监控错误地拒绝了已终止的错误态。
- 现仅允许failed rollout对应的systemError续接，保留active、待审批、身份、人工新turn、STOP和去重检查。先复现失败，再补回归；首轮47通过/1失败来自新增完成态测试缺campaign夹具，修复夹具后48通过。部署副本与源码SHA均为9e9dc3da3585876ff506f57ae020d02e9db1c201a6cbba72cb87e8e3ae1e4fac；只读实时探针确认当前active会话仍受保护，未为验证而制造新错误或插入重复turn。
- 305641于09:42归档，SHA 297e04de3d701ac2ff3907e12b84dd3a1cf43a894f4d184e83c4d2b252de752f，63个文件逐一核对；09:44:21重查用户队列为空。实验包未修改、未提交重跑；下一步修复torch测试覆盖并在新staging验证后新tag补跑。事故和失败报告分别为`evidence/session-watchdog-20261108/system-error-incident-20261109.json`与`evidence/xplace-ep4-tp2-cost-gate-20261108-r0/failure-305641.json`。

### 305641诊断修复后提交306991（2026-11-09 09:47–10:01 CST）

- 本地补齐torch 2.8.0，复现同样的2 failed/1 passed；修复get_tp_group替身，补DP2/TP2两个lane的bytewise payload、身份、源不变和目的区外不变测试，负例改为特定ValueError。直接import torch，禁止缺依赖时静默跳过该模块；未绕过启动门。pip实际装入本地user site，环境路径和与远端torch 2.11.0+cu129的差别记录在修复报告中。
- 专项14 passed；完整172 passed无跳过（190.27秒）；独立新staging 159 passed/1 skipped（186.00秒），仅跳过未上传的分析工具测试。新tag `ep4-tp2-cost-gate-20261109-r1`的manifest SHA为1b5384bd7d38f48e9a23d1f24c07866c7e48e23cf388f90ec7e7e7e29855f72a，tar SHA为a4136026c5f53d8d06c56a185b23d38c996f79e3691e020b9e8f6732019ea7da；32项manifest核对通过。原包及失败证据保留。
- 新包同时包含此前已完成的实测代价/统一并发/消融代码，相对旧包新增5文件、修改7文件；prompt bank SHA保持3a38e81a5699270758952ecfec83b2cc699ca016fe9f3e587f145829165dd3b1。执行参数仍为相同seed20261108、DP2/TP2/EP4、prompt准入与dataplane/grid/interference，不运行trace、不变更数据集或筛选种子。
- 09:59:58远端SHA验证上传完成；10:00:31确认用户队列为空，10:00:36提交306991，四卡/单节点/1h。10:01:24为PENDING/Priority，无启动日志；Slurm暂估11月10日23:56:43，仅为可变快照。当前session已启动每300秒只读监控，先发布新job检查点以防会话异常丢失作业状态。修复证据：`evidence/ep4-tp2-ablation-20261108/startup-repair-20261109.json`。

### 异常自动续接已送达、306991仍排队（2026-11-09 10:41 CST）

- 10:27:11、10:32:13、10:37:14三次failed-turn续接均被当前owner接受，try_count均为1并返回新turn ID；后续模型请求仍遇stream disconnected。这是上游传输继续失败，不是此前owner_not_idle导致的不发送。实际送达与生成成功分开记录，未改模型连接配置，也未重复提交作业；证据为`evidence/session-watchdog-20261108/error-recovery-verification-20261109.json`。
- 已接收并确认原前台监控37759正常结束：10:01:24至10:32:01七次查询均PENDING/Priority，SHA f51a3dc643ee7ddd568b72762df03ab4fe155afea18108fcb9d6fd451710176b。10:41:12额外实时查询仍PENDING，无其他本用户作业、无四项启动/阶段日志；Slurm暂估11月10日13:05:30，仅为可变估计。
- 本轮不重新测试、不改冻结包、不新增allocation。清除过期active_monitor引用，发布最新检查点；GPU迁移/代价/端到端门槛仍未通过，campaign保持active。

### 306991当前会话排队监控（2026-11-09 10:48–10:58 CST）

- 10:48:02、10:53:08、10:58:14按300秒间隔共探测三次，均PENDING/Priority；无其他本用户作业，startup-tests/main/phases/exit-status均MISSING。10:48的Slurm预估仍为11月10日13:05:30，仅为可变估计，不据此取消重投。
- 三次SSH全部成功，日志`evidence/ep4-tp2-ablation-20261108/job-306991-monitor-20261109T024754Z.jsonl`的SHA为54292f4efe9c35028ca1566e448f7506801513c746a06f5e96e56f7831822bfe。前台监控32296正常结束，无遗留命令、无重复提交、无代码修改或重复测试；更新并发布检查点，campaign仍active，实验门槛不变。

### 306991当前会话排队监控（2026-11-09 11:02–11:13 CST）

- 11:02:55、11:08:01、11:13:07按300秒间隔共探测三次，squeue/sacct均PENDING/Priority；无其他本用户作业，startup-tests/main/phases/exit-status均MISSING。11:02的Slurm暂估改为11月9日13:51:48，仅为可变估计，不据此取消重投。
- 已接回前台监控11851并正常结束，三次SSH全部成功，无遗留命令。原始日志`evidence/ep4-tp2-ablation-20261108/job-306991-monitor-20261109T030248Z.jsonl`的SHA为90bf5bc98bbf1f252c17e875d12062fcc6e9072522b0e407bfa8d4d293c3e2be。清除过期active_monitor并更新检查点；未重复提交、修改冻结包、重跑测试或启动trace/混合拓扑，campaign保持active，全部真实实验门槛仍未通过。

### 306991当前会话排队监控（2026-11-09 11:18–11:28 CST）

- 先核实无遗留命令，再于11:18:11、11:23:30、11:28:37按300秒间隔只读查询，squeue/sacct均PENDING/Priority；用户队列仅306991，四项startup-tests/main/phases/exit-status均MISSING。11:18的Slurm暂估为11月9日13:51:17，仅为可变估计，不据此取消重投。
- 前台监控95567正常结束，三次SSH全部成功，日志`evidence/ep4-tp2-ablation-20261108/job-306991-monitor-20261109T031803Z.jsonl`的SHA为d57621edcb6ab2c1f576b0613176e95daf8c042920aa4288cb8dc4a9715947ff。中途已登记并发布监控句柄，结束后清除过期active_monitor并再次更新检查点。未提交新作业、改冻结包、重复测试或启动trace/混合拓扑；campaign保持active，全部实验门槛仍未通过。

### 306991当前会话排队监控（2026-11-09 11:33–12:04 CST）

- 先确认无遗留前台命令，再以300秒间隔完成七次只读检查，观察时间为11:33:24、11:38:30、11:43:37、11:48:46、11:53:53、11:58:59、12:04:05；squeue/sacct均PENDING/Priority，用户队列仅306991，startup-tests/main/phases/exit-status均MISSING。11:33的Slurm暂估为11月9日13:44:46，仅为可变快照。
- 七次SSH全部成功，前台监控33166正常结束。日志`evidence/ep4-tp2-ablation-20261108/job-306991-monitor-20261109T033315Z.jsonl`的SHA为828af3ebb3142e4c8ff71b66c1dcc7aea96aa45f4b40ba202d289afcfce0e570；中途已发布监控句柄，结束后核验原始记录、清除过期active_monitor并更新检查点。未重复提交、取消重投、修改冻结包或重跑测试，未启动trace/混合拓扑；campaign保持active，全部实验门槛仍未通过。

### 306991真机完成、干扰来源拒绝（2026-11-09 12:07–12:33 CST）

- d1n41a11g03于12:07:23启动、12:27:39以COMPLETED/0:0结束，20分16秒。启动159 passed/1 analysis-tools skip（212.08秒），peer-transfer全执行。12:28队列为空；监控54396十次记录SHA为f84f2503f87be156bfb9062229ab49c2bbadeb647bfe8b8fba401e85c532a7d6。tar SHA为6abee43ac6ac573d22881e9325e3c75cc45b3a2a1fa29859f68b057ba528dc2b，115文件逐一比对，31个运行源文件匹配冻结manifest。
- dataplane两事务八次迁移、两端各两个TP lane复制/回读通过，四请求各2500输出，总5196742656字节，迁移中位30.085ms、最大65.291ms。runner安装/清理由成功路径强制核验全部ack，但单个ack未持久化，验收注明范围。grid23/23、526请求、零抢占，最短共同窗口11.717秒；原始布局、长度、cadence和步时均复算匹配。
- 干扰被拒绝：bal8的Code decode为2.962475秒，低于原3秒门槛；bal16窗口6.584532秒，但32-token cadence让20/32个Conv只有一个窗口内点，旧摘要仅取12个有效请求。全部Conv实际还会继续43秒以上，并未提前完成。Code真实prompt101167/输出64，Conv校准输出3000精确。完整拟合保留measured-cost-r1/fit-report.json，无可用step-params。权威报告acceptance-306991-r2.json；首稿误用需要solo+active的helper单行调用，已标analysis_valid=false并保留，不把分析错误当作四例GPU失败。
- 全部23+72=95点decode另作诊断，相对RMSE7.7589%、最大相对误差20.9665%，不是完整代价或泛化证明。三条coordinator乱序统计警告和job-passed后shutdown的TCPStore/NCCL、force-kill警告均保留，未因退出0忽略来源问题。
- 修复校准采样和固定重复契约，支持独立干扰补测连接原95点。305641/306991冻结包及失败数据不改；新的三轮累计窗口规则明文预注册，旧schema不追认，正式误差与性能门槛不放宽。

### 干扰采样修复后提交307623（2026-11-09 14:09 CST）

- 已实现TP2专用stride1、固定三轮/12例、全请求覆盖及零抢占检查、计划和验证报告；新schema2明确采用逐布局三轮累计decode窗口≥3秒，逐例before/prefill仍≥3秒。原Code64和Conv3000输出不变，TP1/正式trace默认采样不变；实测306991旧数据重新调用旧契约仍拒绝。跨作业来源用schema2及全部SHA/job绑定，复用全部95个decode点。
- 专项47 passed（8.19秒），完整runtime+fitter187 passed无跳过（196.69秒）；本地仅有既存asyncio_mode配置警告。独立新staging173 passed/1 analysis-tools skip（190.09秒），peer-transfer全执行。源码与协议SHA、相对r1差异和测试日志见interference-repair-20261109.json；新包只改三个运行/测试文件，prompt bank SHA保持3a38e81a5699270758952ecfec83b2cc699ca016fe9f3e587f145829165dd3b1。
- 新tag ep4-tp2-interference-20261109-r2含32个文件，manifest SHA为25cad4dc6b1e2f0736b58d71588783ae46adfe34dbdf7d45c00983ea85950e6a、tar SHA为d39ec6220b62c2db15df8a2fbd20c1874fd7a24bd44d39490f15610ab4ef3a9d。14:08:19 SSH stdin传输并核远端SHA后原子发布，14:09:08确认用户队列为空，14:09:14提交307623。仅dataplane/interference，四卡/1h/原seed及拓扑，不重扫grid、不运行trace/混合拓扑。
- 14:09:49为PENDING/Priority、无启动日志，Slurm暂估11月11日08:56:44，仅为可变估计。已启动当前session前台监控62783并发布新作业检查点；修复尚待GPU验证，全部完整实验门槛仍未通过。
- 14:09:49、14:14:56、14:20:02三次检查均PENDING/Priority，用户队列仅307623，无启动/阶段/退出日志；前台62783正常结束，无遗留测试或传输命令。日志job-307623-monitor-20261109T060941Z.jsonl的SHA为6cf4c5aafd54e8607519ff1d3cac1f77410a8a514165a9a1c9a680c097a4976b。最终检查点清除过期active_monitor并发布，campaign保持active，不运行正式trace或混合拓扑。

### 307623只读等待续接（2026-11-09 15:31 CST）

- 本轮14:25:54、14:30:59、14:36:07、14:41:15、14:46:23、14:51:30、15:30:20七次SSH均成功；squeue/sacct均PENDING/Priority，用户队列仅307623，startup-tests/main/phases/exit-status均不存在。14:25查询Slurm暂估仍为11月11日08:56:44，仅为可变快照，不取消重投。
- 计划每次查询后等待300秒，前五个实际间隔为304.729–308.569秒，最后间隔2329.572秒（14:51:30至15:30:20，约38分50秒）。时间空档原因未确认，原始时间戳保留，不声称全程连续五分钟观测；末次实时SSH恢复并成功。
- 当前session前台监控68198/PID9787正常结束，进程复查无遗留实验命令。七条原始记录逐条验证，日志job-307623-monitor-20261109T062548Z.jsonl的SHA为a30ec1a477f7d882ae007a5da78b4851bda7ef50310e1c673c1fada93c33dad1；检查点记录实际间隔并清除active_monitor。
- 未修改冻结包、重复提交或重跑测试，没有新GPU输出可验收。继续等待307623的dataplane/interference补测；保留306991拒绝结论和全部负结果，不运行正式消融或混合拓扑，campaign保持active。

### 307623只读等待续接（2026-11-09 16:07 CST）

- 15:37:15、15:42:17、15:47:19、15:52:20、15:57:22、16:02:26、16:07:27七次SSH成功，squeue/sacct均PENDING/Priority；用户队列仅307623，startup-tests/main/phases/exit-status均不存在。以上时间为本地接收UTC转北京时间，原始日志同时保留远端时钟。
- 七次观察跨度1811.944秒，实际间隔301.264–303.260秒。本轮没有上轮的长空档，不追溯填补上轮14:51:30–15:30:20的缺失观测。15:37 Slurm暂估为11月11日03:28:19，仅为可变快照，不取消重投。
- 前台监控36572/PID17240正常结束，进程复查无遗留实验命令。原始日志job-307623-monitor-20261109T073713Z.jsonl七条记录已逐条核验，SHA为910fc53ab883596208a0698f2be791e60c71e055c4b34b676c7924cc85cda523；检查点更新并清除active_monitor。
- 未重复提交、修改冻结包或重跑测试，没有新GPU结果可验收。实测代价门仍未通过，继续等待307623；保留失败和负结果，不运行正式消融或混合拓扑，campaign保持active。

### 307623只读等待续接（2026-11-09 16:42 CST）

- 16:12:24、16:17:26、16:22:29、16:27:31、16:32:34、16:37:38、16:42:40七次SSH成功，squeue/sacct均PENDING/Priority；用户队列仅307623，startup-tests/main/phases/exit-status均不存在。时间为本地接收UTC转北京时间，原始记录同时保留远端时钟。
- 七次观察跨度1816.456秒，实际间隔301.886–303.538秒。16:12 Slurm暂估11月9日20:02:07、候选节点d1n41a26g03，仅为可变快照，并非已经分配资源或预留，不取消重投。
- 前台监控36214/PID22371正常结束，进程复查无遗留实验命令。七条原始记录已逐条核验，日志job-307623-monitor-20261109T081223Z.jsonl的SHA为872b1f7f163d0ac25b96961f2f67db20b82594b50adf47411b79606bbce4b1c0；检查点更新并清除active_monitor。
- 未重复提交、修改冻结包或重跑测试，没有新GPU结果可验收；保持306991干扰拒绝结论和全部负结果，不运行正式消融或混合拓扑，campaign保持active，等待307623的dataplane/interference补测。

### 307623只读等待续接（2026-11-09 17:17 CST）

- 16:47:22、16:52:25、16:57:27、17:02:29、17:07:31、17:12:33、17:17:35七次SSH成功，squeue/sacct均PENDING/Priority；用户队列仅307623，startup-tests/main/phases/exit-status均不存在。时间为本地接收UTC转北京时间，原始记录同时保留远端时钟。
- 七次观察跨度1813.306秒，实际间隔301.815–302.673秒。16:47 Slurm暂估11月9日20:28:11、候选节点d1n41a13g02，仅为可变快照，不是分配或预留，不取消重投。
- 前台监控82958/PID27877正常结束，进程复查无遗留实验命令。七条原始记录已逐条核验，日志job-307623-monitor-20261109T084718Z.jsonl的SHA为c8c03aeefc0beffeb6e3734d481414e232b7acb3907868751b5f851351aa1f28；检查点更新并清除active_monitor。
- 未重复提交、修改冻结包或重跑测试，没有新GPU结果可验收；保留失败、负结果及此前观测空档，实测代价门仍未通过，不运行正式消融或混合拓扑，campaign保持active。

### 307623只读等待续接（2026-11-09 17:52 CST）

- 17:22:33、17:27:36、17:32:37、17:37:38、17:42:40、17:47:42、17:52:43七次SSH成功，squeue/sacct均PENDING/Priority；用户队列仅307623，startup-tests/main/phases/exit-status均不存在。时间为本地接收UTC转北京时间，原始记录同时保留远端时钟。
- 七次观察跨度1810.265秒，实际间隔301.181–303.096秒。17:22 Slurm暂估11月10日12:49:29、候选节点d1n41a28g03，仅为可变快照，不是分配或预留，不取消重投。
- 前台监控18343/PID33618正常结束，进程复查无遗留实验命令。七条原始记录已逐条核验，日志job-307623-monitor-20261109T092229Z.jsonl的SHA为eccdc69272702928cbceb03df21562c082ff8af37e76ec331a5f36b86d5c7f06；检查点更新并清除active_monitor。
- 未重复提交、修改冻结包或重跑测试，未发现新GPU输出；继续等待干扰补测，保留全部失败和负结果，不运行正式消融或混合拓扑，campaign保持active。

### 模型403异常续接与307623等待（2026-11-09 18:33 CST）

- 17:57:12模型请求返回403/上游余额不足，原始rollout的task_complete带错误；失败turn 01a1bfbd-69f0-7932-b709-9b4dcfb67d27共五条记录、没有工具调用，不是Slurm作业失败。watchdog于18:02单次accepted送入同一thread的新turn，本轮已实际执行本地进程核查和SSH，不把消息送达本身等同于生成成功。
- 原始错误事件、rollout逐行SHA、单次attempt/check和恢复后首个probe保存在evidence/session-watchdog-20261108/upstream-403-20261109T095712Z.json，SHA为8953c60a5e9636475c8077af6548f9f2a9f432e0631bdec50534006587e78dd2。五条原始记录及check/probe来源SHA均复核一致；未重置余额、付款或修改上游连接配置，也不声称上游余额问题已永久解决。
- 18:02:44、18:07:45、18:12:46、18:17:47、18:22:55、18:27:57、18:33:02七次SSH成功，squeue/sacct均PENDING/Priority；用户队列仅307623，startup-tests/main/phases/exit-status均不存在。观察跨度1818.078秒，实际间隔301.076–307.804秒。
- 18:02 Slurm暂估11月9日22:08:17、候选节点d1n41a16g01，仅为可变快照，不是分配或预留。前台78985/PID39273正常结束，进程复查无遗留实验命令；日志job-307623-monitor-20261109T100243Z.jsonl的SHA为bf6496a43b9dc7c8d85489d6818dfc26b35a2ff44a00fca21824a17bb8c0db73，七条记录已逐条核验。
- 检查点清除active_monitor并记录会话中断；未重复提交、修改冻结包或重跑测试，没有新GPU结果可验收，不运行正式消融或混合拓扑，campaign保持active。

### 307623等待及检查点长度整理（2026-11-09 19:08 CST）

- 18:38:06、18:43:08、18:48:09、18:53:12、18:58:17、19:03:18、19:08:19七次SSH成功，squeue/sacct均PENDING/Priority，用户队列仅307623；startup-tests/main/phases/exit-status均不存在。观察跨度1813.472秒，实际间隔301.150–305.191秒。18:38 Slurm暂估仍为11月9日22:08:17、候选节点d1n41a16g01，不代表分配或预留。
- 首次发布含active_monitor的检查点因16141字符超过既有16000上限被拒绝，已发布版本没有被覆盖。整理预检曾得到15186字符，未满足本轮预留空间的15000内部目标，未写文件；随后归档完整原文、为20份历史监控/probe建立SHA索引、精简六条历史摘要，降至14684字符后发布成功，未修改校验器或消息预算。
- 整理前完整快照campaign-before-context-compaction-20261109T104105Z.json的SHA为c690f48a8f3da45ac3f12358ecb574e13af20d02dd4955ab714cbfa92dfe2d2e；historical-monitor-index-20261109T104105Z.json的SHA为3462514764bb56ddc08d2bafa0979cdea64491b80d80bfe95cb69ae8112588e4。完整快照、索引及20个原始文件均核验一致；没有删除任何原始记录、失败或负结果，也未改变作业配置、约束及门槛。快照中的旧active_monitor只作历史记录，不可用来恢复命令。
- 前台60839/PID44512正常结束，进程复查无遗留实验命令；日志job-307623-monitor-20261109T103755Z.jsonl的SHA为b93204de7e37d2db91b4767bc57076601c310ae77d9debc6849284439e7995c5，七条记录已逐条核验。最终检查点清除active_monitor，继续保持active。
- 未重复提交、修改冻结包或重跑测试，没有新GPU结果可验收；实测代价门仍未通过，不运行正式消融或混合拓扑。

### 307623等待续接（2026-11-09 19:43 CST）

- 19:12:53、19:17:54、19:22:57、19:27:58、19:33:01、19:38:02、19:43:03七次SSH成功，squeue/sacct均PENDING/Priority，用户队列仅307623；startup-tests/main/phases/exit-status均不存在。观察跨度1810.658秒，实际间隔301.120–303.387秒；本轮首次观察与上轮末次观察相隔273.369秒，不将间隔内状态视为连续实测。
- 19:12 Slurm暂估11月9日21:51:03、候选节点d1n41a16g01，仅为可变快照，不是分配或预留。最后一次本地观察时间为19:43:03.921，远端date为19:43:04，按原始时间保留。
- 接回现有前台90455/PID49400至正常结束，未另开监控；随后进程复查无遗留实验命令。日志evidence/ep4-tp2-ablation-20261108/job-307623-monitor-20261109T111252Z.jsonl共七条记录，逐条核验查询返回码、唯一排队作业、账务状态及缺失输出，SHA为2a1c9d7fd95dfd18c7a5ef9add41e3a800a090ae1d2c2aba524bf99a427523c9。
- 检查点更新观察时间及last_monitor、清除active_monitor，保持active与全部未通过门槛。未重复提交、修改冻结包、重跑测试或删除PAUSE；没有新GPU结果可验收，不运行正式消融或混合拓扑。
- 本轮首次本地检查点补丁因标准diff行号不兼容apply_patch被拒绝，源检查点未变；组合命令未fail-fast，随后重复发布了旧的11:12:53版本。已改为apply_patch支持的上下文格式并启用fail-fast，成功更新后重新发布，未涉及Slurm或实验包。

### 307623终态验收与联合代价负结果（2026-11-09 20:38 CST）

- 排队监控22356/PID54162共五次：19:48:17至20:03:25前四次PENDING/Priority，第五次20:08:26确认RUNNING，实际20:05:11启动。跨度1209.801秒、间隔301.109–304.585秒，状态变化后正常结束，SHA为7ae8e191adc38d0739cc88a0da8fa3b61aa0e0d9ed5dd3d01292fd62253ce146。
- 运行监控58212/PID56862共18次：20:09:26至20:26:56，跨度1049.920秒、间隔61.052–64.277秒；另两次只读查看阶段。Slurm于20:26:07 COMPLETED/0:0，d1n41a16g01/单节点4 A800/20分56秒。20:32:18再查用户队列为空，无遗留实验命令；运行监控SHA为44730e8a9175afcb76da0a5a9d7d159be33b9bc9fa5799a30aed03975be30b7e。完整原始命令、间隔与终态见terminal-review-307623.json，不将查询间隔视为连续采样。
- fetch49489正常结束，归档SHA为c4272105c65011082178aa6c459f857b548964fdbe7909c2ced40e0337419556。100文件逐一核验，31个运行源文件匹配32文件冻结manifest（sbatch未复制入job）。启动173 passed/1 analysis-only skip、219.13秒，peer-transfer测试实际执行。
- 新增本地accept-307623.py并对真实归档运行通过：八次双TP lane迁移/四条2500-token流/5196742656字节，中位27.913ms、最大66.052ms；12例/168请求/零抢占/固定计划/全请求窗口通过，decode每请求最少53点，活跃布局累计29.891/12.313/22.695秒。报告acceptance-307623.json，不把短上下文payload验证扩大为端到端结论。
- 联合拟合50238使用303594全部72点、306991全部23点及307623全部12例，decode为7.7589%/20.9665%，干扰54次预测比较为55.5374%/158.8382%，被15%/30%门槛拒绝。measured-cost-r2完整保留来源、candidate和fit-report，没有step-params.json，不回退prior。
- 新增diagnose-cost-r2.py，重算残差与fitter一致。27个活跃观测的phase-blind/phase-bucket非负加性RMSE下界为41.1157%/36.7605%；精确layout/phase分表仍23.5614%、最坏误差下界45.9771%。逐点残差、不可行区间、源SHA完整保存；仅针对指定函数族，不是所有在线可观测模型的不可能性证明。本轮没有实施runtime模型修复或提交新作业。
- job phase于20:25:55 passed后，20:26:02出现TCPStore/NCCL关闭及进程管理器清理警告；原日志保留，无Python traceback，全部流完整，退出0。不宣称清理无缺陷，也不据此重投。本地shasum因locale失败，hashlib已核当前/冻结calibrate与measured_cost源码一致。
- 最终检查点首次长度预检16223超16000，未写文件或发布；下一条长stdin脚本编码失败也未执行写入。改为短命令并完整保存campaign-before-results-compaction-20261109T123722Z.json，SHA为e3ad9f89d3e027ab6e6a28745bedbaaddd5b7ed6c93f053001638c514c05a60b；旧历史摘要压缩为十条，原始失败/证据/门槛未删改。历史快照中的active_monitor已结束，不可恢复。最终检查点14083字符，20:37:47成功发布。
- 检查点清除active_monitor，阶段为tp2_interference_model_repair_required，保持active与全部门槛false。冻结包、runtime、fitter、门槛、Code64/Conv原长度和seed均未改；没有重复测试、原包重投、正式消融或混合拓扑。下一步先离线分析在线可观测特征与重复差异，再决定有依据的修复和独立pilot。

### 307623时序探索、评分调用审计与断连恢复（2026-11-09，23:16复查）

- 20:43及20:56只读检查均无用户作业；23:16:05再次squeue/sacct确认队列为空、307623 COMPLETED/0:0。20:56至23:16约140分钟没有连续观测；本地进程检查无遗留实验命令。watchdog报告22:57:00.813会话stream断连、23:12续接，不能把它解释为Slurm失败或未经证实的余额问题。未改上游、账户、PAUSE或旧observer配置。恢复时检查点仍停在20:37:47，但全部新增分析文件已落盘，故先核验恢复，不重做不可覆盖的诊断或重复提交。
- 新增causal-analysis-plan-307623.json与analyze-cadence-307623.py，分析全部12例/168请求和27个活跃标量目标。所有可见上下文、年龄、lag1s/3s特征均有明确的arrival/首token截止；首token时才可得的信息不冒充准入时已知。仅执行事先固定的布局均值与布局+上下文+lag1s ridge，整repeat留一、三fold，无追加模型或换窗口。结果causal-cadence-diagnostic-307623.json保留全部case、逐请求窗口、长gap、联合活动bins和逐目标预测。
- 布局均值覆盖27/27目标，relative RMSE/max为44.7771%/165.6571%；ridge仅9个Conv-prefill预测可用，为56.3971%/147.4857%。bal8-r1/16-r1的48个Conv请求在Code首token前1秒各只有一个点，导致严格fold规则将18个decode预测记录为missing；不能声称56.3971%覆盖全部27目标，也不能借用截止后约1.8–3.2ms的callback。独立复核脚本audit-causal-features-307623.py检查原始SHA、168请求和全部目标，输出causal-cadence-review-307623.json。
- 首次复核错误假设stride1必有64个Code callback，AssertionError后没有产出通过报告；保留原稿audit-causal-features-307623-r0.py及causal-review-audit-failure-307623.json。实际有合并callback，Code点数58–64，但全保留63个decode token增量；修正只核真实点数/累计增量/合并计数，原数据、原探索模型和结果未改，复核成功。
- 9例活跃背景Code平均cadence步时48.09–168.65ms，间隔中位39.13–44.03ms；固定100ms阈值下长间隔占时23.26%–82.18%。7例250ms bin输出速率相关≥0.97，另2例0.2506/0.5708完整保留；每个长Code间隔内Conv平均每请求最多推进1.5token。现象支持共同输出停顿而非明确交替高速推进，但host cadence不能确定GPU/EP、engine/IPC、前端/GIL或主机调度原因。没有删除长gap或用间隔中位数替换目标平均。
- audit-cost-callpath-307623.py做三组CPU夹具及三个显式干扰API阳性对照。layout_cost、ca准入、周期评分分别记录2/10/20次decode调用，全部other_active=False；大幅改变干扰参数后三组完整planner结果均不变，显式True/prefill/decode对照则改变。原评分函数和compact state未传播其他tenant状态/显式phase，故仅修参数文件不足。修正文档中fallback已用于当前完整策略的过度表述；未删任何验收分支，没有执行分时算法。
- 冻结tp2-interference-observability-pilot-20261109.md：独立机制观测3轮×4布局×off/on=24例/336请求，原seed23/Code64/Conv3000不变；有界前端heartbeat、engine/scheduler及全部worker lane时间戳，零热路径共享盘写入/逐步RPC/主动CUDA同步。尚未实现hook、staging或新tag，不冒充独立预测pilot。下一步核对部署vLLM源码并实现有界观测和测试，之后才提交机制补测。
- 恢复后的证据核验确认失败fit报告/candidate和原始证据SHA未变，无step-params.json；冻结tar SHA仍d39ec6220b62c2db15df8a2fbd20c1874fd7a24bd44d39490f15610ab4ef3a9d。一次发布前只读预检误期待31份runtime/tests，实际tar为17 runtime+13 tests=30，另有prompt-bank/sbatch/manifest；预检在写入前停止，核清后全部30文件逐字节匹配。没有重新跑整套测试或改变runtime/fitter。
- 检查点23:18:46更新为tp2_interference_observability_preparation，保留active和全部门槛false；15313字符。完整旧检查点、时序/调用审计、脚本、协议和恢复事件进入causal-analysis-checkpoint-20261109T151846Z.json的SHA索引，避免丢失证据或超过消息长度。后续先实现观测与真实评分接入，仍禁止原包盲重跑、正式消融和混合拓扑。


## 2026-11-09 23:57 CST：TP2有界观测实现与r3冻结（尚未上传提交）

- 23:54:43实查用户队列为空，307623保持COMPLETED/0:0；r3远端目录不存在，同名accounting及本地submission均为空，没有恢复旧链或占用新allocation。原始采集通过/干扰模型失败结论未变。
- 实现默认关闭的frontend async/thread heartbeat、engine step_fn、scheduler、executor及全部worker lane host计时（含dummy）；100000条/进程/例，有界且溢出、缺lane、schema错误拒绝。off恢复原方法，保留异常/返回/Future身份；无热路径共享盘写入、逐步RPC或主动CUDA同步。Future响应观察时间不冒充GPU完成；账面上下文不冒充已计算KV。
- main/sbatch仅允许dataplane,observability，固定24例/336请求、原seed23/Code64/Conv3000；两条件分别验schema2，全保留配对开销。五份部署vLLM源码SHA加载模型前核验；3300秒软截止包含启动测试，1h硬限。请求失败/超时保留partial并清理。
- 专项45通过；最终完整232通过、202.09秒、无跳过；隔离staging218通过/1 analysis-only skip、197.71秒，peer-transfer实际执行。旧误路径pytest退出4未执行测试、初版229通过均保留。没有将CPU测试冒充GPU验证。
- 新tag ep4-tp2-observability-20261109-r3冻结39文件，相对r2仅live_scheduler/main/worker/sbatch四文件改动、新增七文件；prompt bank及旧r2包/失败结果核SHA不变。manifest 6f67e7230f21c47e1464ccf5c657d279e649658190aef0236c5d46caaf793650；tar c8295330ef845e59e5c5885b6419dfd2da5611d94b6db67bfd40ce60954eca9b；plan e6871075484b61974a7fdc8cbf4d1ab65bfeef6bff38b918db2dec027d7e6838。不重建或改已有r3包。
- 实现/测试/差异/部署/旧结果SHA索引见observability-implementation-record-20261109.json。23:56第一次checkpoint patch使用带行号的diff头被apply_patch拒绝；三份新增快照/索引/记录已落盘，campaign/progress未变，随后误发布的是旧检查点而非新进展。核验落盘SHA后仅修正patch格式完成更新，不覆盖冻结实验包或重跑测试。
- 既有campaign完整快照及历史evidence索引保留以满足16000字符发布上限。当前仍active，全部门槛false，尚未上传或提交。非分时评分other_active缺口、估计器和独立预测验证尚待后续，不运行正式消融或混合拓扑。


## 2026-11-10 00:04:24 CST: r3 mechanism job 309147 submitted

- Atomic upload completed Nov9 23:59:44 CST; remote39 files plus tar/manifest/plan/prompt SHA verified. Fresh Nov10 00:00:26 empty user queue, same-name accounting, tag job/log and absent upload lock. Single submit returned00:00:32 (Slurm00:00:33). Upload/submit commands ended; intent/preflight/result/submission retained.
- Both modelsDP2/TP2/EP4 share four A800, one node/1h; dataplane,observability only; prompt KV/seed20261108/drain600/TP migration; excluded d1n41a21g03. No frozen package modification, trace, time-slicing or mixed topology.
- First remote observation00:01:15 PENDING/Priority, sole user job, all startup/phase/exit logs absent. Estimate Nov10 10:58 on d1n41a28g03 is not a reservation. Foreground65402/PID74651 checks300s, at most seven pending probes, returns on state change.
- Checkpoint length checks and one stdin encoding error stopped before writes; full prior snapshot retained and repeated historical summaries condensed to meet16000-character budget. Active/all gates false. CPU tests and mechanism data do not replace estimator/scoring repair or independent prediction validation.


## 2026-11-10 00:32:56 CST: 309147 queued after bounded monitoring

- Seven successful SSH checks from00:01:07 to00:31:53 CST found309147 PENDING/Priority and the only user job. All startup/main/phase/observability-progress/status/exit logs remain absent. No new GPU evidence or fetch.
- Actual intervals: 308.109490, 312.681750, 304.945574, 308.497658, 305.385410, 306.736851 seconds; elapsed1846.357156s. Monitor65402/PID74651 ended normally; ps confirms no leftover upload/submit/monitor command. Raw log SHA 39e6fc3b48e1f396f62379026400337fa48b02e3e122c41d6ca9ada598310b46.
- All39 frozen files and manifest/tar verified unchanged, no runnable measured-cost parameters. Review: observability-monitor-review-20261109T163256Z.json. Next session must freshly verify commands/Slurm before monitoring; do not restart old monitor, reupload or resubmit. All gates remain false and campaign active; no formal ablation or mixed topology.


## 2026-11-10 00:40:42 CST: continue309147 read-only wait

- Local process check found no leftover experiment command. Published checkpoint, last monitor/review and preparation SHA verified. Live SSH at00:38:51 CST found309147 PENDING/Priority, sole user job; startup/main/phase/observability-progress/status/exit logs missing.
- New foreground63691/PID80245 starts00:39:42 CST using the existing bounded read-only probe implementation, writing a NEW timestamped log. This does not restore an old observer/process. Seven pending checks300s apart, returning on state change. No upload/submit/fetch/test or runtime change.
- Full previous campaign preserved in campaign-before-monitor-continuation-20261109T164042Z.json with SHA; old historical compaction/pre-submission terminal-review fields consolidated behind that snapshot to keep context below16000 characters. Corrected stale current-tense findings: r3 hooks/staging/job now exist, but the old fit failure and unresolved scoring propagation still stand. All original evidence and gates remain unchanged.


## 2026-11-10 01:11:21 CST:309147 remains queued, bounded monitor complete

- Initial00:38:51 live review plus seven successful SSH monitor probes00:39:42-01:10:17 CST found309147 PENDING/Priority, sole user job; startup/main/phases/observability progress/status/exit files absent. No new GPU result. First monitor estimate05:18 on d1n41a28g03 is mutable, not a reservation.
- Actual intervals 309.393137, 305.118234, 304.876077, 306.094347, 305.048414, 304.693525 seconds; elapsed1835.224144s. Foreground63691/PID80245 ended normally; ps confirms no leftover experiment command. Log SHA 6a22ec64e69572fe7bd4d8dae21eaa34c493ef7909b7aff55a8b9d8f1acf7b7b. Previous monitor path/SHA retained in the new review.
- All39 manifest files plus tar verified unchanged; no runnable measured-cost params, no code edit/test rerun/upload/submit/fetch. Final review observability-monitor-review-20261109T171121Z.json. Campaign active/all gates false, no formal ablation/mixed topology. Current active_monitor removed; next turn must freshly check processes/Slurm before monitoring.


## 2026-11-10 01:23:40 CST: causal scoring-state preparation while309147 queues

- Initial live check01:13:47 found309147 PENDING/Priority, sole user job with missing logs. New foreground81050/PID85373 starts01:15:26, seven bounded300s checks; r3 frozen package unchanged.
- Local scheduler snapshot/summary gains logical phase, in-flight count, pause schema; planner retains causal fields in scoring_observations. Candidate helper copies state, projects moves/known-prompt arrival, refuses missing/future/stale state instead of assuming solo. integrated_cost gains explicit bucket forwarding with default behavior unchanged. No estimator, horizon scoring activation, rollout parameter or newtag.
- Important additional blocker: measured_cost.interference_residuals passes the Conv-layout bucket to BOTH tenant tables, so the existing index is not universally the other tenant bucket. Current layout_cost also treats arrivals as decode throughout the horizon; blindly passing a current prefill phase is incorrect. These are documented in tp2-scoring-state-contract-20261110.md, not fixed by inventing a new predictor.
- Initial targeted47 passed; added edge cases afterwards. Non-timesliced regression tool20922/log scoring-state-nontimesliced-regression-20261109T172304Z.log is running. It excludes mixed time-slicing files except explicit non-timesliced nodes. No formal ablation/mixed topology; gate remains failed.


## 2026-11-10 01:28:16 CST: scoring-state local validation complete, not deployed

- Non-timesliced regression231 passed/0 skipped in75.81s; isolated runtime/tests218 passed/0 skipped in72.38s. Initial47-pass targeted run retained; final regression includes added edge cases. Mixed general/runtime_fake files were not executed wholesale: only explicit non-timesliced nodes ran. Local fitter tests ran; isolated source contains runtime/tests only. No time-sliced algorithm execution.
- Isolated36 source files match current worktree SHA, unchanged during tests; temporary directory removed. Seven changed/new files (four existing runtime/test edits, two new runtime/test files, one new documentation file) recorded in scoring-state-local-files-20261110.json and patch SHA 3e8e997aa5fe9824147fd0d7fcfaf27e6a4cde9df494f98147a65acd1e5b937f. Test processes20922/93268 ended.
- Frozenr3 tar/manifest/all39 files reverified unchanged. Worktree calibrate/measured_cost/trace_runner/controller/main still matchr3, all old failed report/candidate/raw evidence hashes preserved, no step-params.json. No new tag/upload/submission.
- Explicit status local_contract_verified_not_wired_not_deployed: causal descriptors are preparation, not a repaired horizon estimator. Instantaneous prefill cannot persist throughout horizon; old calibration bucket is Conv layout on both tables. Full scoring still does not consume descriptor, independent prediction gate remains failed. Record scoring-state-preparation-20261110.json contains complete SHA/test/exclusion/blocker details. Monitoring309147 continues separately.


## 2026-11-10 01:46:53 CST:309147 pending; local preparation preserved

- Initial01:13:47 review and seven successful monitor probes01:15:26-01:45:57 CST all found309147 PENDING/Priority, sole user job, no startup/main/phases/observability-progress/status/exit logs. Estimate04:04 from01:15 is not reserved.
- Monitor81050/PID85373 ended normally. Actual intervals 304.671539, 305.326659, 305.362529, 304.867733, 305.339642, 305.228158 seconds; elapsed1830.796687s. Log SHA d7174fe838ef4a18f388ad1f5e359e2a76bab43f0049a8b57f89bc1f9eb8fd89. Local test20922/93268 also ended; ps confirms no leftover experiment/test command.
- Frozen r3 manifest/tar/all39 files unchanged. Seven local state-contract worktree files match the separately recorded tested version; old failed candidate/report/raw SHA remain intact. No newtag/upload/submit/fetch or runnable parameter. CPU231/isolated218 pass is preparation only; full interference scoring and independent prediction gate remain unresolved.
- Final monitor review observability-monitor-review-20261109T174653Z.json links prior monitoring and local preparation. Campaign staysactive/all gatesfalse, active_monitor removed. Next turn freshly checks live state before monitoring or acceptance.


## 2026-11-10 02:03:57 CST: offline r3 artifact auditor prepared while309147 queues

- Added only tools/audit_observability.py, its synthetic tests, and tp2-observability-audit-20261110.md. No runtime, frozen upload, parameter or GPU action. Final local47 passed/0 skipped in9.49s; isolated47 passed/0 skipped in9.52s against36 copied r3 runtime/test/tool source files. Earlier36/45/46 and isolated46 results retained.
- Checks external archive/manifest/plan SHA, all8 TP2 moves, fixed24-case336-request ordered matrix, separate12-case conditions,13-process telemetry, heartbeat/count/source/idle boundaries and all12 uncorrected overhead pairs. Synthetic missing/corrupt/negative fixtures are rejected. Main output never passes cost, independent prediction, ablation or mixed gates; no real309147 artifact audited yet.
- Frozenr3 tar/manifest/39 files, prior seven scoring-state preparation files and six old negative-result/source hashes reverified unchanged. Full scoring remains unwired and no runnable cost parameters exist. Existing unrelated protocol.md:953 trailing whitespace was not changed; three new files have no trailing whitespace.
- Current bounded monitor19905/PID90420 continues; latest02:00:40 PENDING/Priority, sole user job, no startup/phase/progress/exit logs. Estimate06:44 was observed01:50 and is not a reservation. Preparation record observability-auditor-preparation-20261110.json preserves source/test/diff SHA.


## 2026-11-10 02:22:27 CST:309147 remains pending; auditor preparation preserved

- Seven successful read-only probes01:50:28-02:21:02 all found309147 PENDING/Priority, sole user job, no startup/phase/progress/exit outputs. Initial01:48:48 review agrees. Estimate06:44 observed01:50 is mutable, not a reservation. No new GPU results or fetch.
- Foreground19905/PID90420 ended normally; actual intervals 304.982695, 306.606103, 306.400806, 305.388664, 305.340071, 305.06907 seconds, elapsed1833.787987s. Monitor SHA 29e6855ffcd10e10f10c77b0cff4f8ade41888b4a137e42282060dbc03146b15. Process check02:21:41 confirms no leftover experiment/test commands.
- Final auditor source/tests/doc and all test-log SHA reverified. Local47 and isolated47 tests pass; isolated9.52s uses copies of frozen r3 runtime/tests. Frozen39 files/tar, prior scoring-state files and all six recorded old negative/source hashes remain unchanged. No actual r3 artifact audited; no runnable parameters, full scoring repair or gate promotion.
- Review observability-monitor-review-20261109T182227Z.json records complete interval/probe/source checks. Removed active_monitor, campaign remainsactive/all gatesfalse; publish follows. Next continuation must freshly verify queue/output before monitoring or one-time terminal archive/audit.


## 2026-11-10 02:57:24 CST:309147 started; running monitor attached

- Live02:24:14 check found sole309147 PENDING/Priority, no logs. Seven queue-monitor probes02:24:47-02:55:21 returned sixPENDING thenRUNNING. Slurm start02:54:59 on d1n41a28g03, hard1h; mutable04:04 estimate was not a reservation. All queries succeeded.
- Queue monitor76861/PID95279 ended normally; source logSHA b3f51180b9784f795d920fa247f15aca7789ace44ed001acf6fc05b5852e72d0. New current-session read-only monitor54159/PID99663 begins02:56:28,60s interval with65-probe/3900s bound and immediate terminal/query-error exit. No parallel allocation, upload, submit or fetch.
- Startup tests/dataplane/observability still awaiting acceptance. No experiment code changes or old regression reruns; local47+isolated47 auditor remains prepared, not a GPU result. Snapshot/review observability-queue-to-running-20261109T185724Z.json preserves transition; campaign staysactive/all gatesfalse.


## 2026-11-10 03:43:16 CST:309147 archived, artifact audit and mechanism review complete

- Slurm02:54:59-03:30:17,COMPLETED/0:0,d1n41a28g03;03:40:09 queue empty. Running monitor54159/PID99663 ended after32 successful probes; all actual intervals retained. Single fetch29053 completed03:32,archiveSHA 75596520ebf249c53fbc7b67bbede492f673e8e80f3a9b59bccc3573b3d0c31b. Initial checkpoint updater incorrectly assumed fetch still running; assertion wrote no changes and no duplicate fetch followed. Raw cleanup warnings after job passed remain archived.
- Prepared offline auditor45335 passed without changes:249 archive files,39 frozen manifest entries,8 double-lane moves,24 fixed cases/336 requests,off/on12 each,zero preemptions and full13-process telemetry per on-case. Total1144386 timing events; median/max move22.897/53.855ms. This is not independent prediction or a complete measured-cost gate.
- All12 paired differences retained:CodeTPOT -55.99% to+139.45%;Conv9 applicable pairs -9.18% to+12.60%,3 solo N/A. Cannot separate instrument effect from run-time variability. Source reports and complete table are in tp2-observability-results-20261110.md.
- Descriptive analysis12519 completed using pre-existing100ms threshold;off92/on84 long callback gaps. No on gap overlaps100ms heartbeat delay; typical wait overlaps worker sample/dummy host calls. Median min-lane long-call coverage97.51%,but4 zero and6 below50% exceptions retained. Independent raw recalculation223752 worker calls agrees; no CUDA-completion evidence or exact GPU/collective root cause. No predictor, fit or runnable params.
- Archive249 files,all39 frozen entries,prior state/auditor source hashes and all old negative-result hashes reverified after analysis. No new allocation,code deployment or unchanged regression rerun. Result index observability-results-checkpoint-20261109T194316Z.json; campaign remainsactive/all gatesfalse. Next read deployed sampling/dummy sources before freezing causal prediction and full-scoring contracts.

## 2026-11-10: sampling/dummy source review and discriminating-scope protocol

- Live03:46:18 CST squeue was empty;309147 remainedCOMPLETED/0:0. Existing archive and exit-status present, no repeated fetch or submission. Three shell-only source reads captured24 deployed files; tar/extracted hashes verified, seven match r3 historical SHA anchors and17 are post-job-only snapshots. No remote Python/import/test/compile.
- Static paths locate an existing transfer-event wait inside sample bookkeeping/output materialization and DP coordination inside dummy/real execution. Resolved GPU-vs-CPU coordination is a source/config inference, not a separately recorded r3 flag. DP wave finish consensus every32 steps means logical idle cannot certify device quiescence. Exact internal time shares and GPU/collective causes remain unmeasured.
- New documentation freezes only the next mechanism discrimination scope: four nested host ranges, original24-case336-request matrix, no added CUDA sync/step RPC/hot-path disk, same100ms descriptive threshold and all exceptions retained. No runtime implementation, estimator or new tag yet. Static replay projection across all96 old worker records peaks at36294/100000 events; this is not measured overhead or a bound on future execution.
- Source review script completed once;39 frozen r3 entries/tar unchanged. First source transfer completed before a local zsh reserved-variable error; payload reused after fullSHA verification without repeated transfer. Local locale-dependent shasum failure replaced by hashlib/LC_ALL=C checks. Reports and incidents preserved in sampling-source-review-20261110.json.
- Full interference scoring remains unwired;95 decode points/12 failed r2 cases and15%/30% gates unchanged. Independent prediction and long-context/capacity pilots must precede paired ablation; no mixed topology or time-slicing. See tp2-sampling-wait-review-20261110.md.
- First checkpoint publication exceeded the existing16000-character cap at18250. Complete unpublished checkpoint was preserved under campaign-before-publish-compaction-20261109T195956Z.json; only historical job/ended-monitor metadata was compacted to14046 characters. Publication retry succeeded and active copy matched; incident sampling-publication-compaction-20261109T195956Z.json retained. No watchdog limit, raw evidence or completion gate changed.

## 2026-11-10 04:39 CST: nested timing r4 frozen and submitted once as309773

- Four nested host scopes now capture runner sample, bookkeeping, to_list and the actual GPUModelRunner module's DP-coordinate binding. Default-off,100000 events/process/case, thread/parent identity, exception and restoration checks; no added CUDA sync, tensor materialization, per-step RPC or hot-path disk. Runtime/estimator/scoring behavior otherwise remains frozenr3; local scoring-state preparation is not included.
- Tests:173 targeted,359 full non-timesliced,220 isolated runtime/tests,95 isolated auditor; all zero skips. Earlier77-pass intermediate result retained. New startup freezes24 explicit node selectors and clears XPLACE_NESTED_TIMING for pytest. Oldr3 whole-directory startup did include time-sliced fixtures; prior logs and historical counts are not relabeled. No time-sliced fixture runs in these new selections.
- Immutable r4 has44 entries:4 modified,6 added,1 old plan filename removed versusr3. Workload fields remain identical:24 ordered cases/336 requests,original prompt/lengths/seed/off-on schedule,one node/fourA800/onehour. Manifest0690b742c85b6899390fbb6a1546c4e3c20a2227dd52069892d0414ee04a3825; tar91efd5efe01398f078de20933ecfd5e8b76f736c7f53144937b994e734255466. Source/patch/test receipt:nested-timing-implementation-record-20261110.json.
- First upload transferred and SHA-verified, then555 extracted directory rename failed. Read-only permission diagnostic preserved; owner-write on only that temporary directory allowed rename,then555 restored and44 entries/tar/plan verified. No retransfer, code-byte change or test rerun. Failed transport and repair receipts remain separate; no GPU failure occurred.
- Fresh04:31:59 empty queue/no same-name accounting plus44 frozen and24 deployed source checks preceded one guarded sbatch309773. Remote .submit-once/submitted-job-id prevent blind retries.04:33 and04:38 observations arePENDING/Priority,sole user job,no startup/phase/progress/exit logs. Foreground77185/PID14185 performs at most7 read-only probes300s apart. Slurm08:41 estimate observed04:33 is mutable,not a reservation.
- No new GPU results or gate promotion. All95 decode points/12 failedr2 interference cases and15%/30% gates remain; nested host times do not prove CUDA completion. Full-scoring/valid-estimator/independent-prediction/capacity/paired-ablation gates still precede mixed topology. Checkpoint published and active copy verified before and after submission.

## 2026-11-10 05:20 CST: offline nested analysis prepared while309773 queues

- Initial05:09 process check found no leftover experiment command. Fresh bounded monitor98713/PID19228 follows309773; first three observations remainPENDING/Priority,sole user job,no startup/phase/progress/exit logs. No upload,submission,fetch or real result analysis.08:41 Slurm estimate is mutable,not reserved.
- Added only tools/analyze_nested_timing.py,its synthetic tests,and tp2-nested-timing-analysis-20261110.md. Local33 passed/0 skipped in4.49s; isolated33 passed/0 skipped in4.88s against49 copied files from frozenr4 plus analysis tools. Earlier31-pass intermediate log retained. No unchanged runtime regression or time-sliced fixture run.
- Analyzer uses externally recorded artifact-audit SHA and archived/frozen source SHA; does not rerun immutable reports. All24 cases/336 requests and12 uncorrected pairs required. Fixed100ms original callback gaps retain merged-token increments; each parent uses direct-child interval union and clipped exclusive remainder. Cross-thread overlap is explicit,not summed as wall time. Missing/corrupt/overflow/lane/scope cases retain failed slots; successful subsets never pass the report. Existing output cannot be overwritten.
- Zero coverage and no-long-gap cases remain;outside observed host scopes meansunknown,notidle. Later heartbeat events are retrospective,not online features. Tests are synthetic,not GPU acceptance,root-cause proof or independent prediction. All cost/ablation/mixed gates remainfalse.
- All44 frozenr4 entries/tar,39 frozenr3 entries/tar,prior scoring-state/negative evidence and existing auditor SHA unchanged. Preparation/source/test/diff receipt:nested-analysis-preparation-20261110.json. Full pre-analysis checkpoint retained in campaign-before-nested-analysis-20261110.json; new checkpoint published with identical active copy.

## 2026-11-10 07:40 CST: live observation transport implemented locally, not deployed

- Read-only preflight07:26:55 and final07:39:52 CST:309773 COMPLETED/0:0,user queueempty,existing r4 archive SHA unchanged. No allocation,upload,submit,refetch or archived analysis rerun.
- Added default-off per-process Unix-socket sampling at worker/engine initialization and explicit controller collection. Worker execute/sample/dummy methods update bounded in-memory host state; a separate thread samples while the host call is blocked. Actual engine.engines_running belongs to rank,not lane;4 rank-wave samples and8 lane-host samples keep separate observation/receipt times. Linux boot/time-namespace clock identity is verified independently at producer and consumer. No step RPC,CUDA sync or hot-path disk added.
- New raw transport kinds are rejected by the prior prediction validator; no relabeling or fake wave propagation. Scheduler receipt times are captured individually,not at gather completion. New API is not called by admission/scoring. GIL/scheduling stalls,missing lanes and timeouts remain unavailable;device activity unknown,forecast unavailable. Runtime observation overhead and Linux/GPU deployment remain unvalidated.
- Final targeted52 and isolated52 passed;13 directly affected scheduler/dataplane/capacity regressions passed,all0skips. Initial41passed/1failed fixture receipt retained:Python3.9 required constructing controller asyncio.Lock inside its async scenario;no production lock change. Intermediate52-pass receipt retained. No time-sliced nodes executed.
- Historical all-unchanged verification exited1 because live_scheduler intentionally gained two hook lines. Independent exact-delta review confirmed that change;120 other prior SHA and9 PAUSE files unchanged,plus9 checkpoint references and r4 result archive. Original guard result retained;tests not rerun to hide it. Existing3 runtime edits/new3 module-test-doc files recorded against pre-edit snapshots;frozenr4 unchanged.
- Evidence:live-producer-review-20261110.json and live-producer-verification-20261110.json;design:docs/tp2-live-observation-transport-20261110.md. Next:rank/lane causal join and same-reference horizon/estimator contract,not a blind mechanism rerun. Full scoring,independent prediction,long-context capacity,paired ablation and mixed-topology gates all remain pending.

## 2026-11-10 07:56 CST: causal rank/lane join validated locally, estimator pending

- Fresh read-only07:45:21/07:55:38 CST checks show309773 COMPLETED/0:0,user queueempty,r4 archive SHA unchanged. No allocation,upload,submit,fetch or archived report rerun. All test/SSH commands and12-process CPU test children ended.
- Added causal_observation.causal_joint_state deriving logical observed/candidate features from the received scheduler snapshots themselves,not a separate cached state. Keeps4 scheduler receipts,4 rank-wave samples and8 worker-lane samples at their own times;requires complete unique identity,per-record clock,freshness,all-source skew and collection-return causality. Missing/late/stale/failed state stays unavailable,not solo.
- Only existing runtime delta is startup producer instance token and PID propagation through live_observation socket/scheduler metadata. Scheduler and rank-wave must match both;different endpoints require unique producer identities. Default remains off,no new stepRPC,CUDA sync or hot-path disk. New API returns bounded-skew samples,not an atomic snapshot,cost or horizon;no admission/scoring callsite added.
- New targeted80 and isolated80 tests passed,plus4 directly affected metadata/worker/engine binding regressions,all0skips. End-to-end uses12 distinct local CPU processes,actual socket/control IPC and IDs/timestamps with simulated workers/schedulers and an explicit test clock;not Linux-clock or GPU acceptance.
- Rechecked121 prior source/evidence items against the last accepted actual hashes,11 unchanged previous local source/doc files,10 checkpoint references and9 PAUSE files;all unchanged. Exact live_observation metadata-only delta independently verified. Prior fixture failure/old expected-SHA guard result remain preserved,not relabeled. r1-r4 and all95decode/12failedr2 cases unchanged.
- Evidence:causal-join-review-20261110.json and causal-join-isolated-regression-20261110.json;contract:docs/tp2-causal-observation-join-20261110.md. Next freeze same-reference work horizon,prefill/inflight coupled transitions and deployable estimator before full symmetric scoring and independent15%/30% prediction/capacity tests. No formal ablation,mixed topology,old observer or PAUSE deletion.

## 2026-11-10 08:12 CST: common reference work implemented locally, duration predictor pending

- Read-only08:01:54/08:11:12 CST checks:309773 COMPLETED/0:0,user queueempty,r4 archive SHA unchanged. No allocation,upload,submit,fetch or archived report rerun.
- Added reference_work.py,its tests,and tp2-reference-work-contract-20261110.md only. Fixed additional-output milestones are relative to received scheduler counters and identical for baseline/candidates,including a causally received external arrival even when baseline leaves it waiting. Assignments cannot change work/source IDs;own/peer indices are symmetric. No online policy callsite or existing runtime/capacity change.
- Verified archived scheduler increments computed and in-flight together;return subtracts in-flight. Input target P+O+K-1 splits disjoint in-flight/unissued intervals into prompt,known-output input and future decode. Final prefill already produces first output;recomputed known outputs are not new output. Five CPU stub transitions execute the archived schedule method and actual return-subtraction AST;not full vLLM/GPU execution. Zero input work still requires an output receipt.
- Offline outcome audit retains every member,missing/earlyEOS/max_tokens/timeout/abort/preemption/error. No successful-subset objective;fast requests continue influencing the coupled horizon after their own milestone. Missing fast-peer tail observations and delayed early-EOS reports cannot count as valid closed-cohort coverage. Eligibility is not precision validation;prediction_gate_passed remainsfalse.
- Local73 and isolated73 passed,zero skips. Local plugin-disabled pytest retains one existing asyncio_mode configuration warning;isolated no warning. No unchanged regression,time-sliced node or old report run. Checked76 pre-edit files,121 historical items,11 checkpoint references and9 PAUSE files unchanged. Prior fixture/SHA guard failures and all GPU negatives preserved.
- Evidence:reference-work-review-20261110.json and reference-work-preparation-20261110.json. No deployable K/model/GPU tag frozen. Config/source capability binding,as-of residual advancement,coupled duration prediction,full scoring,independent15%/30% prediction and long-context/capacity tests remain required before TP2 paired ablation or mixed topology.

## 2026-11-10 08:28 CST: accounting capability binding implemented locally, no GPU deployment

- Fresh read-only08:15:16/08:27:27 CST checks:309773 COMPLETED/0:0,user queueempty,r4 archive SHA unchanged. No allocation,upload,submit,fetch or historical report rerun.
- Default-off engine metadata now records actual scheduler configuration and six startup-resolved source roles. Missing config is not defaulted to supported. Scheduler snapshot and independent wave keep their own observation/receipt times;both capability records must match a predeclared manifest/SHA before producing a bound reference. Source IO is startup-only;sample-time checks cover config/rank/method identity. No stepRPC,CUDA sync or hot-path disk.
- Source review additionally guards actual accounting-method module/qualname/code-name/file and runtime code-object identity,not merely class file SHA. Final local84/isolated84/affected-regression5 passed,zero skips;initial82/82/5 snapshots retained. Local pytest configuration warnings retained. CPU fake engines/request classes and real local sockets are not Linux/vLLM/GPU acceptance;no actual deployment manifest,K or time model frozen.
- Exact existing delta is only live_observation metadata wiring.77 other before-state files,121 historical items,12 checkpoint references and9 PAUSE files unchanged. Existing causal join/reference ledger/scoring/capacity behavior unchanged. Future scorer must use the bound entrypoint,not mistake legacy unbound ledger availability for deployment acceptance.
- Review guard exited1 because its wrapper required git diff --no-index --check to return0;clean changed files return1 with no diagnostics. All hashes/exact delta had passed. Original failed review retained;independent clean/trailing-whitespace probes verify the corrected exit-code interpretation. No source or test rerun for this wrapper repair. Evidence:capability-review-20261110.json plus capability-review-guard-repair-20261110.json.
- Preparation:capability-preparation-20261110.json;contract:docs/tp2-accounting-capability-binding-20261110.md. Next implement as-of residual/coupled duration prediction and a deployable training/holdout protocol,not another input-validator-only claim of prediction. All prior negative results,15%/30% thresholds and capacity reserves remain. No formal ablation,mixed topology,time-sliced algorithm,old observer or PAUSE deletion.

## 2026-11-10 08:49 CST: conditional coupled-duration kernel, not a calibrated predictor

- Read-only08:37:03/08:47:31 CST checks:309773 COMPLETED/0:0,user queueempty,r4 archive SHA unchanged. No allocation,upload,submit,fetch or archived report rerun.
- New coupled_duration advances both tenants numerically on one clock. Directional service responses reprice only remaining batch fractions at peer phase transitions;prompt/recompute/decode,return,controller receipt cadence,residual activity,migration drain/transfer and deferred arrival are separate. Fast peers continue after their own milestone;incomplete cohorts retain every row and a null total. No existing runtime,scoring,capacity or producer edits.
- Local54/isolated54 passed,zero skips;both retain the existing plugin-disabled asyncio_mode warning. First40-pass snapshot and intermediate47pass/2fail snapshot retained. New scale tests exposed a wrong expected22 versus analytic21.75 and an actual floating-cadence defect:ULP-sized errors could add a whole sampling period. Corrected the analytic expectation for all unchanged scales and stabilized only near-integer phase ticks,never delivering before modeled completion. Failure/repair evidence:duration-r1-failure-diagnosis-20261110.json. No GPU result was involved.
- Additional tests cover simultaneous completion,final-event/time-boundary censoring,token conservation,own/peer indices,rank/tenant permutation,source hash mutation,missing residual/support and migration ordering. No unchanged historical test suite or time-sliced node run. All coefficients and residual seeds are synthetic CPU fixtures,not estimated GPU parameters.
- The single-batch rank-barrier model is conditional,not vLLM emulation. Archived engine core has a separate batch_queue path;existing accounting capability does not prove actual pipeline depth,queue order,DP throttling or worker behavior. Resident waiting states reject. A deployable residual initializer,identified response surface,full scoring/capacity integration and frozen training/independent-holdout/overhead protocol remain required.
- Evidence:duration-review-20261110.json,duration-preparation-20261110.json;contract:docs/tp2-coupled-duration-kernel-20261110.md. Preserve all r1-r4 exploratory/negative cases,old guard failures,15%/30% gates and capacity reserves. No new GPU tag,formal ablation,mixed topology,old observer or PAUSE deletion.

## 2026-11-10 09:09 CST: host-return residual fit, negative exploratory result

- Read-only08:55:20/09:07:33 CST checks:309773 COMPLETED/0:0,user queueempty,r4 archive SHA unchanged. No allocation,submit,upload or refetch. One new read-only source excerpt shows current max_concurrent_batches returns PP size under synchronous scheduling;PP1 implies1 for that source,not a new GPU runtime attestation. Core SHA still matches the archived interface.
- New host_residual fits conditional host-return duration,with separate failed returns/right censoring. Query conditions on survival at sample age,not survival through unobserved receipt lag;restricted means are not full means or device progress. No default prior,unobserved-tail extrapolation or missing-group fallback. Runtime remains detached from live policy and the coupled kernel.
- Fixed protocol preceded the new fit:all24 r4 cases,12 timing-off missing,100ms virtual grid,H2seconds,minimum20 calls/3cases,strict earlier-case-only training. All223752 observed outer calls returned normally;all50750 query rows retained. Coverage34421/50750=67.8246%;4283 unseen-group and12046 support failures. On available labeled queries,MAE0.210211s,WAPE63.9640%. The zeros/failed/censored target counts are0 here,not evidence those cases are generally absent.
- This is an unfavorable exploratory result,not a15%/30% joint prediction gate measurement. All r4 was already viewed;virtual zero-lag queries are not actual live samples. Same-process dependence and omitted workload/peer information remain. Do not tune H,minimum support or select cases to relabel this as successful;do not feed host means into GPU/batch remaining_fraction.
- Local32/isolated32/runtime tests and8 analysis tests passed,zero skips. Independent sorted-duration prefix-sum oracle checked all50750 query labels/support decisions and34421 available means;maximum mean discrepancy1.37e-14s. No training-code import in that audit. No unchanged historical test/report rerun;this is a new target/fit over the preserved archive.
- Evidence:residual-exploration-protocol-20261110.json,host-residual-r4-exploration-r0/report.json,residual-empirical-audit-20261110.json,residual-preparation-20261110.json. Model/episodes/query rows and all prior failures remain. Next needs deployable batch/host/scheduler-return causal associations and an independent training/validation pilot;no full scoring,formal ablation,mixed topology,old observer or PAUSE deletion.
- Checkpoint writer stopped before changing campaign:16020-character context exceeded the unchanged16000 limit by20. Source/result/tests/review/preparation had passed and remain unchanged. A separate publication-only repair compacts repeated prose without rerunning fit/tests/review,raising limits or deleting evidence;original writer and failure receipt retained.

### 2026-11-10 — Explicit batch/RPC association, local only

- Read-only checks at09:15:32/09:28:47 CST found the user queue empty and309773 COMPLETED/0:0;the archived r4 SHA is unchanged. No GPU allocation,submission,upload,refetch or repeat of an old result analysis.
- New default-off batch_link carries engine epoch/PID,DP identity,sequence and causal batch fingerprint in the existing callable RPC arguments. No SchedulerOutput mutation,extra RPC or scheduler-accounting override. Execute/sample share explicit identity;dummy work has its own identity. Independent bounded sockets retain latest operation and completion/failure counters,not complete event history.
- Native synchronous step return is only an upper bound on scheduler update return,before engine output-queue enqueue. Worker RPC return,inner host scopes,controller receipt and GPU completion remain distinct. Missing samples/overwritten operations must remain missing. Accounted computed tokens are not device progress;DP cadence/wave observations do not prove common-rank alignment.
- Local44/isolated44/new tests and54 affected live-observer regressions passed,zero skips;initial36-pass source snapshot retained. Six paired CPU AST executions exercised actual pinned engine step/cadence/process loop,executor RPC/FutureWrapper,worker proxy and MQ serialization with fake queues,workers and scheduler across local-buffer/overflow/remote-socket branches. Outputs match enabled/disabled;no actual vLLM/GPU execution is claimed.
- Local cloudpickle import initially failed because the CPU environment lacks it. The failure is retained in batch-link-cloudpickle-20261110/receipt.json;read-only deployment cloudpickle3.1.2 source supplied the isolated dependency path without remote import or package installation. Existing local asyncio_mode warnings and tar locale warnings remain;no test failure or parameter tuning was concealed.
- No new GPU tag,training/holdout split,K,complete deployment manifest or calibrated response is frozen. The host-residual negative result,all r1-r4 exploratory/failed points,15%/30% gates and capacity reserves remain unchanged. Next needs actual linked live-sample collection and controller-receipt labels,then identifiable residual/phase training and independent validation before full symmetric scoring and TP2 ablations. Mixed topology still waits for TP2 success.

### 2026-11-10 — Bounded live collection and frontend receipt hooks, local only

- Initial read-only SSH timed out after90seconds;the original wrapper did not retain partial stdout,so that attempt established no job state. The failure and narrower keepalive retry are retained. Queue/accounting reads at09:37:28 and09:50:55 CST succeeded:queueempty,309773 COMPLETED/0:0. Cause of the initial stall remains unisolated;later checks omit remote archive hashing. No allocation,submission,upload or archive refetch.
- New CausalCollection runs an explicitly owned case-scoped sampler:existing scheduler/rank/host observations and12 batch-link endpoints concurrently,with original source/receipt times and start/completion cohorts. No overlapping rounds or catch-up bursts. Deadlines/cancellation drain client tasks and retain already completed sources. Encoded event bounds,counted gaps,invalid/missing endpoints and open requests remain visible;latest-only producers still are not complete event history.
- engine.stream has an optional receipt hook. Arrivals must be registered at shared-queue entry;unregistered or noncumulative streams are unavailable. Counts,terminal reasons and frontend-consumption timestamps are recorded without adding future output lengths or modifying SamplingParams,routing,prompts,output lengths or the exact-length check. Zero/repeated/coalesced counts are retained. Early stop followed by legacy length-check failure records both;post-cutoff callbacks cannot rewrite censored state.
- The pinned frontend source defaults to cumulative output but coalesces queued RequestOutputs before generate yields. Receipt time is not scheduler return,output enqueue,detokenizer production or GPU completion. No source-generation timestamp is invented for delayed EOS;these raw receipts must not be fed into old outcome APIs by copying received_at into observed_at.
- Local33/isolated33 and2 affected non-timesliced stream-caller regressions passed,zero skips. Initial28pass/1fail and29pass receipts remain. The initial failure was an overlong macOS pytest socket path;only the fixture moved to a short private /tmp directory,not the portable100-byte bound. Three native frontend AST CPU cases confirm coalescing,early-stop/length failure and internal abort on cancellation using fake engine submission/payloads,not vLLM import/GPU inference.
- Source-audit helper initially omitted AsyncFunctionDef,producing an empty-class error before any native case ran. Original script/log and separate repair retained;runtime/native source/assertions were unchanged for that repair. Existing local asyncio_mode warnings remain. No old fit or unchanged historical tests/reports rerun.
- This is a collection interface with an actual optional stream callsite,not a deployed pilot or calibrated joint predictor. Next wire a fresh calibration pilot and freeze full frontend/worker/config/serializer manifest,K,receipt targets,cadence/deadline,overhead/loss rules and training/independent-validation split. Old r1-r4 remain exploratory. No full scoring,ablation,mixed topology,capacity reserve reduction,old observer or PAUSE deletion.

### 2026-11-10 — Causal training pilot wired to the shared queue, local only

- Read-only checks at09:58:02/10:15:01 CST found the user queue empty and309773 COMPLETED/0:0. No allocation,submission,upload,archive refetch or unchanged result-analysis rerun;SSH timeout wrappers retain partial output.
- Added dataplane,causal and off/producers/collect startup modes. Only no_move executes;arrival registration is at the existing central-queue entry and the original stream receives the observer. No altered prompts/output lengths,scoring,route selection or time-sliced policy. Main rejects mismatched causal reserve/concurrency/drain/seed/cache/context flags instead of silently logging different settings.
- Frozen actual-bank collection plan has4training cases/44requests,Conv4/16/4/16 plus one Code per case,original Conv output lengths,K16,100ms cadence,80ms deadline,240s trace window,256-token admission extra and shared limits1/16. All cutoffs/missing/coalesced/failed outcomes remain;independent future runs require a frozen estimator and use reserved disjoint prompt keys. These are not newly measured GPU results.
- Immutable trace-cutoff cohort survives event-buffer overflow. Collector shutdown avoids repeated report decoding;outer cancellation also cleans up completion/failure waiters. Case reports are persisted after closing collection and stream cleanup. Off/producers/collect must restart engines to measure full instrumentation overhead;producer-only comparison is not full overhead.
- Final local71/isolated71 passed,zero skips. Affected regression52 passed at r2;later r3 changes only the causal-specific CLI guard plus six tests,so unchanged regression paths were not rerun. Initial59pass/2fail came from one-lane CPU fixture replies under a TP2 controller;two explicit synthetic lane replies repair the fixture without weakening production acknowledgment checks. All59/61/65/71-stage receipts retained.
- Independently extracted local bundle validates the actual bank/plan in all3modes,4cases/44requests,and bash syntax without importing vLLM/torch. It is explicitly not submission-ready:full actual runtime-source/compute attestation,non-timesliced sbatch startup selection,and overhead/loss acceptance rules remain pending. Existing generic sbatch whole-directory startup must not be used for this pilot.
- No trained joint predictor,full symmetric scoring,accepted measured-cost/capacity gate,TP2 ablation or mixed topology. Preserve all earlier negative results,15%/30% gates,original PAUSE files and historical checkpoints. Next complete deployment review and a bounded fresh training/overhead allocation,not another metadata-only substitute or an unchanged r4 rerun.

### 2026-11-10 — Fresh causal calibration submitted; capture-aligned128/256 planned

- Submitted310540 at10:48:55 CST after two empty-queue checks. New immutable r5 package uploaded once and verified;10:51:00 read-only review foundPENDING/Priority,not an allocated GPU. No old result refetch or observer restart.
- off→collect restarts both engines in one≤1h allocation;reverse-order block is independently predeclared but not yet submitted. Same4cases44requests percondition,unchanged real prompts/outputs,K16,100ms/80ms,central queue,1/16concurrency and256reserve. Driver shares3300sdeadline and records exit/timeout/process-group CPU/RSS;failure prevents nextmode.
- Deployment binding covers runtime and2275packagePython sources,10vLLM/KVnative binaries,2launchers,4versions and23actualworkerconfig fields. Actual Linuxclock/native/eight-worker attestation is required oncompute. Full-system native identity is not claimed;torch is version-bound. Success rechecks sources after shutdown.
- Local148/isolated148/startup533 pass,zero skips. Startup-r0 retained532pass/1fail:direct clock probe identified this DarwinPython3.9 environment's process-relative defaultmonotonic;only the12-processCPUfixture switches to systemCLOCK_MONOTONIC. Production freshness limits unchanged. Local timestamp-parsing publication error is separately recorded and repaired,not a GPU failure or rerun.
- User128/256 fixed-total capture design includes93layouts/3repeats and all reachable capture boundaries±1,with six retained blocks. Pre-GPU capacity arithmetic shows old4096context BS256 endpoint needs60GiB/81920blocks perlane vs archived38.75GiB/65536blocks. Revised ALL layouts/BOTH totals to2048syntheticcontext/1024output (36GiB ideal endpoint),preserving original unsubmitted configs and realConv output lengths. Actual page/free-block capacity checks remain authoritative;no performance-based point/seed selection.
- Frozen coverage/overhead protocol is separate from unchanged15%/30% prediction acceptance. All gates remainfalse:training collection is not accepted prediction or full system gain. Next monitor310540,retain/fetch terminal artifacts once,diagnose failures before newtag,then reverse-order overhead and causal fit/independent validation. No TP2formal ablation or mixed topology yet.

### 2026-11-10 — Offline causal labels and full-pair overhead analysis ready;310540 pending

- Read-only reviews at11:18:32/11:23:29/11:30:30 CST find310540 PENDING/Priority,the only userjob;no allocation,upload,submit,cancel orfetch thisround. No newGPU result orclaimed systemgain. Frozenr5 package/runtime and9originalPAUSE remainunchanged.
- New offline analyzer emits every100msquery atactualsample completion,keepsallcausallyarrivedunfinishedrequests includingcentralqueue,K16frontendfirst-receiptlabels,jointmaxonlyforallreached,andexplicit EOS/error/cancel/censor/missing rows. Cohorts reconcile againstevents;sourcePID/clock/epoch/lane ties toworkerattestations. No deviceprogress,remainingfraction orfutureoutputoracle.
- Frozenr5 directCPU probe separates normalpost-closeemitrejections fromwithin-windowloss onlywhenallsequence/byte/attemptcounterandstop invariantsprove it. Rawdropped/retention_lossless stayunchanged. Missingstop orconservationfailsreject;windowzero-loss/invalid and90%/10% coveragegatesunchanged.
- Twoorderblocks/all8pairs/allplannedrequests required;immutableincompletebeatscleanupcompletion,240scountsusecadenceintervals,worstupperratio decides unchanged5%median/15%worstoverhead. Actualsource/runtime533startup,conditionorder,workerGPU/PID/clock,workloadbudgetandprocessusagechecked. CLIexit0isanalysiscompletion,notacceptance;Slurm/archiveverification remainsseparate.
- Final local88/isolated88 passed,zero skips;full/missingterminalfixtures,CLIandindependentlinearfirst-hit/maxoracleincluded. Preserve r1 41pass/16fail(validationgaps),r3 76pass/3fail(invalidatomicreceiptfixture),isolatedr5collectionerror(missingtransitivefixtures),andearlierASTerror;allseparatelydiagnosedandrepaired. Oneexistinglocalpytestasyncio_modewarning;isolatedhasnone.
- Analysisprotocolr0andr1,source/testsnapshots,diffsandreceiptsretained. No realr5analysis,predictorfit,independentvalidation,capacitygate,scoringorformalablation yet. Onterminalfetchonce/diagnose,thenpredeclaredreverseblock.128/256capacity-r1 explicitconfigsremainunsubmitted;do not regeneratefromold4096defaults ormistakecapture_bucketforactualreplay.

### 2026-11-10 — Capture-aligned scan hooks and failure retention ready locally

- Successful read-only checks at11:45:59/11:57:36 CST found310540 PENDING/Priority,only userjob,no allocation. Later12:45/12:46 queries timed out;12:47:40 independent-connection retry reported SSH backend handshake EOF. Current live state is therefore unverified,not assumed pending or terminal. Retain all transport receipts;no upload,submit,cancel orfetch.
- Actual optional main/worker/scan callsites now collect per-lane returned padding decisions and host CUDAGraph.replay enter/return/exception events. Full4lane/source/PID/GPU/epoch/clock evidence required; bounded20k-events/16MiB perlane/case,raw loss counters and incomplete RPC replies retained. No added CUDA synchronization and no claim of device completion,request batch progress or phase duration.
- Scan persists all planned requests before execution,immutable cutoff before cleanup,late cleanup results separately,and warmup/case errors,timeouts,cancellation,budget skips and all not-run points. Graph failures stop later points even when the performance window is also invalid. Existing plans cannot be overwritten;no automatic retry or seed reselection.
- Local91/isolated91 pass,zero skips;9 native-source CPU AST pairs preserve padding/replay results or exceptions and existing call order. Not GPU/SHM/payload validation. Keep initial65pass/14fail(Event fixture outsidePython3.9 loop),79/90-pass intermediate snapshots,and native audit identity-comparison failure/repair. Local existing asyncio_mode warning remains;isolated none.
- Review verified82 before files,70 frozenr5 files plus archive SHA,and9 originalPAUSE. Six derived configs add only observe_graph_execution=true to explicitcapacity-r1 inputs:93points,3repeats,2048context/1024syntheticoutput,total128/256 and capture neighbors unchanged. Keep old4096 plans and all negative data. Same-total/repeat balanced normalization;missing baseline stays missing.
- Full graph-overhead off/collect protocol and scan deployment manifest remain pending;no new scan tag or GPU result. Freeze those before scan submission. The mandatory r5 collect→off follow-up must reuse its original frozen runtime,not this changed runtime. Terminal diagnosis/fetch/causal analysis precedes follow-up;independent15%/30% prediction,capacity/full scoring and TP2paired ablations remain unmet. No mixed topology,time-slicing,old observer orPAUSE deletion.
- Evidence:scan-observation-preparation-20261110.json,scan-observation-review-20261110.json,scan-observation-native-audit-20261110.json,scan-graph-design-20261110-r0/design.json;implementation guide:docs/tp2-scan-graph-observation-20261110.md.

### 2026-11-10 — Fixed two-order graph-overhead job wired and locally frozen;SSH unavailable

- Read-only retries at12:52:36,13:04:43 and13:12:13 CST all returned SSH backend handshake EOF,including two gateway IPs. No new Slurm state or output was obtained. 310540 remains last-observed PENDING at11:57:36 CST,not currently confirmed pending or terminal. No submit,upload,cancel,fetch,credential change or second allocation.
- New scan_job drives independent off/collect main processes in fixed off→collect and collect→off blocks. Four predeclared synthetic layouts per condition:128/256 balanced/extreme,2048 context/1024 output,768 measured requests each,3072 across16 case outcomes/8 pairs. Common CLI/source/physical-budget rules and fixed150s case deadlines;shared3300s allocation soft deadline,1500s minimum to start a condition,≤1h job.
- Real sbatch/main/worker path reuses complete source/configuration attestation,including8workers and post-shutdown verification. Other observation flags cleared symmetrically;off has no graph hook,collect buffers are active only within case windows. Existing process-group cleanup and1s CPU/RSS sampling apply to both. Return status is collected_not_overhead_accepted,never a gate pass.
- Protocol freezes steady-step,all-request mean TTFT and frontend-completion TPOT;all8pairs/full coverage required,each metric median≤5% and worst≤15%. Startup/CPU/wall reported separately. This four-layout pilot does not validate overhead for all93capture-neighbor points. No K16/joint prediction,policy baseline or formal-ablation claim;no real Conv output or reserve change.
- Local130/isolated130 and complete local upload startup238 pass,zero skips;Bash syntax and actual extracted dispatch branches covered without Slurm/GPU. First126pass/1fail found a real launcher conflict guard relying on errexit;explicit diagnostic/exit2 now rejects four conflicting inputs. Failed source bundle and127/130-pass intermediate receipts retained. Existing local asyncio_mode warning unchanged,isolated none.
- Frozen unsubmitted bundle has78hashed payload files plus manifest;tar audit verifies all79 files. Review preserves86 beforefiles,70 r5 frozen files/archive,and9 originalPAUSE. Both r5 causal modes and its mandatory reverse block must use ORIGINAL r5 runtime,not this new main/sbatch. Old negative fits and all93 curve points remain unchanged.
- Evidence:scan-job-preparation-20261110.json,scan-job-review-20261110.json,scan-job-bundle-20261110-r0/build.json;guide:docs/tp2-scan-overhead-job-20261110.md. Next recover310540 and follow terminal diagnosis/reverse-causal protocol;freeze the graph cross-block artifact analyzer before any graph pilot submission. All completion gates remainfalse.

### 2026-11-10 — Graph-overhead raw analyzer frozen;310921 remains pending

- Read-only squeue/sacct/scontrol all succeeded at14:03:58 and14:14:45 CST:310921 is the only user job,PENDING/Priority,no allocation. Between those observations the14:10 query had squeue/scontrol timeouts with sacct stillPENDING;the partial failure remains recorded. No submit,upload,cancel,fetch or additional allocation this round. Start estimates are not promises.
- Context carried forward:310540 failed startup before inference;its archive and diagnosis are retained.310921 is the already-submitted repaired collect→off block1. After its terminal diagnosis,the same repaired-runtime off→collect block0 is still mandatory. No new GPU inference/performance result is available.
- New offline tool audits both fixed graph-overhead blocks,all16 measured cases/3072 requests/8 pairs and4 warmups. Recomputes all-request steady-step/TTFT/frontend-completion TPOT;requires complete coverage,all three median≤5%/worst≤15%,zero preemption,source/config/238startup identities,8worker/allEPshard/KVnamespace checks,shared deadline and proper cleanup. Invalid/missing/censored rows and original file hashes stay visible.
- Graph evidence uses archived frontend/worker Linux clock rather than this analysis host;checks four lanes,identities,source,window and zero loss. Host replay is not GPU completion. CPU/RSS remain sampled statistics. Four balanced/extreme layouts do not stand in for the93capture-neighbor scan. RealConv lengths,reserves,128/256 totals and3repeats remain unchanged.
- Local118/isolated118 pass,zero skips,including independent constant-rate metrics,actual measure_layout CPU artifacts,complete/missing CLI archives,wrong source/EP/shards/clock/PID/lane,cleanup results and exact threshold boundaries. First98pass/2fail exposed a remaining analysis-host clock call andbinary5%rounding;both repaired with original snapshots/logs/diff retained. Intermediate100/117pass receipts retained. Existing local asyncio_mode warning unchanged;isolated none.
- Analysis protocol freezes all runtime plus the two analyzer dependencies and exact238startup IDs against the original graph deployment manifest.71 r5/71 r6/79 graph frozen files and9 originalPAUSE files verified unchanged. No GPU bundle rebuilt. Evidence:scan-analysis-resume-review-20261110.json,scan-analysis-resume-r0-diagnosis-20261110.json,scan-overhead-analysis-20261110-r0/protocol.json. Guide:docs/tp2-scan-overhead-analysis-20261110.md. Old negative results and all independent prediction/full scoring/paired ablation/mixed gates remain unchanged and false.

### 2026-11-10 — Queued r6 package reverified;terminal analysis source fixed without reruns

- Full read-only observations at14:23:15 and14:33:23 CST both show310921 as the sole user job,PENDING/Priority,no allocation. The current session waited300s in the foreground between checks;no old observer or automatic experiment chain was started. Future scheduler estimates remain nonbinding. No submit,upload,fetch,cancel,GPU result,test rerun or performance-analysis rerun this round.
- At14:27:23 CST,login-node readback verified all70 r6 payload files via MANIFEST.sha256,its manifest SHA,the original upload.tar.gz SHA,the existing .submit-once lock and submission.stdout=310921. Only reads/SHA were used;no remote file was changed and no test/import/inference ran on the login node. This verifies queued artifacts,not the future compute-node native environment.
- Preserved38 files byte-identical to the original88-pass isolated causal-analysis receipt in a permanent local directory,including the original analysis protocol. A terminal-readiness record binds the exact533 unique repaired-package startup IDs and r6 deployment hashes. No analysis was rerun. The historical analysis/tests fixtures are not the repaired GPU startup payload and must never be uploaded or used to construct replacementblock0.
- Continue to retrieve310921 exactlyonce only after terminal accounting,then audit raw/archive/source/startup/8worker/clock/dataplane/cleanup and both4case44request modes. Use the permanent analysis/tools/analyze_causal_calibration.py snapshot with block1 and a NEW output directory. Mandatory repaired-runtime off→collect block0 follows terminal diagnosis regardless of observed performance;failed310540 remains separately archived. Graph overhead and full93point128/256 scans remain unsubmitted,with all thresholds,reserves,Conv lengths,negative results and9PAUSE unchanged.
- Evidence:causal-terminal-readiness-20261110-r0/readiness.json,remote-payload-readback.json,bounded-wait.json and deploy-monitor-310921-20261110T063322Z.json. All completion gates remain false;campaign remains active.

### 2026-11-10 — r6 measurement repairs; CPU candidate frozen, no GPU submission

- One read-only squeue/sacct query at20:12:31CST:queue empty,310921 COMPLETED/0:0. No upload,submission,fetch,old performance-analysis rerun or new reminder. The earlier r6 rejection and all88 request identities remain unchanged.
- Fixed causal paired budget at[38.75,40.25]GiB from per-group minima of bothr6 capacity measurements,keeping3GiB margin/256-token reserve. Each condition preserves natural-budget evidence and fails before warmup if insufficient;no clipping. Kept the origin-anchored240s window repair.
- Predeclared512MiB/1M-event bounded retention. Capacity-only streaming inspection reconciles allfour old collection files without rerunning metrics. CPU stressr1:403526 attempts,old128MiB drops302493,new512MiB retainsall534786549bytes,0invalid/drop,contiguous sequences andstop;over-new-cap rejection also passes. Actualminimumsample/receipt130555/511bytes satisfies25%margin. Firststresscounts retained butits25%minimumclaim rejected duepadding-header sizing;diagnosis/protocolr1 added beforestronger test. Encoded-byte bound is not a totalRSS bound;RPC/coverage/overhead remainGPUunverified.
- Local154 regressions pass. Initial11 regressions fail on old implementation as expected. Isolated candidate544startup tests pass,0skip,alloriginal533identities retained. Failed local harness attempts (macOS Linux-clock helper,patch hunk format,missingcloudpickle andwronghash-field preflight) remain archived;dependency repair uses only manifest-verified archivedcloudpickle3.1.2 outsideupload,notdownloads orstubs. Source andstartup assertion selection are not loosened.
- Frozen candidate-r1/upload derives from exactr6 with only4runtime+2test+4input/manifest changes;no graph/worktree bundle contamination. Originalr6block0 is superseded-not-completed by revisedboth-orders requirement;do not blindlyrerun ormixrevisions. Candidate is notsubmittable until separate newanalysis binds544identities,contractSHA,newmanifest andfixedbudget evidence. Old533analysis andnegative report stayfrozen;allgatesfalse. Guide:docs/tp2-causal-measurement-revision-20261110.md. Evidence:causal-budget-retention-repair-20261110-r0/candidate-freeze.json andretention-stress-result-r1.json.

### 2026-11-10 — Revised analysis bound; single r7 upload in progress, no submission

- One squeue/sacct query at21:48:50CST:queue empty,310921 COMPLETED/0:0. No oldGPU result refetch or analysis rerun. Newanalysis local133/isolated133 pass,0skip;69compute payload files andexisting544CPU startup identities bind to the frozen candidate. Original544startup tests andretention stress were not rerun.
- Four newend-to-end counterexamples expose oldfalse acceptance:equal butundeclared paired budgets,duplicateJUnit identities withsamecount,missingbudget record,foreign memoryGPU. Allfour reject afterrepair;all8pairs remainrecorded. Exactstartup identities,individualfailure/skip checks,compute payloadset/SHA andfixedbudget/worker/startup/workload bindings added. Original thresholds/cohort/negative results unchanged.
- Naturalbudget is still anattestedruntime record,not independently recomputed because mapped-KV summaries are notexported;capacity/longcontext gates remainfalse. Initialfixture exclusive-write misuse andposttest read-onlybinding-directory permission error arediagnosed/retained;binding finished fromsuccessfulreceipt withouttestrerun orcandidate changes.
- r7 block0 off-to-collect localstage copies thefrozen70-file candidate exactly;manifest85ddfee...946,archive32b58dd...41c5,analysisprotocolb41f070...38e4a. Oneupload command started;at22:08:34CST itstillawaits SSHtransfer completion,localstdin reached1647110bytes butnoremoteverificationreceipt. No sbatch executed. Preserve toolsession78646/PID69675,inspectreceipt/process before anynext action;neverlaunch anotherupload. Remote partialname andcommand are retained in the turncheckpoint.
- Afterupload verification only,run freshqueue guard andsubmitnewtag once;then terminaldiagnosis beforeidenticalsource reverseblock1,regardlessperformance. Allgatesfalse/active,9PAUSE/graphbundle unchanged. Guide:docs/tp2-causal-revision-analysis-20261110.md. Evidence:causal-revision-analysis-20261110-r0/analysis-freeze.json andinput-binding-verification.json.

### 2026-11-10 — r7上传完成，312336单次提交、等待运行

- 22:10:54CST原上传命令成功返回；远端SHA验证及原子发布完成，没有第二次上传。此前22:08:34的在途记录是历史快照，不代表仍有传输。
- 22:12:20CST提交保护确认310921已终态、用户队列为空，随后仅一次sbatch提交312336；提交后队列仅312336，PENDING/Priority，尚无启动或GPU验收结果。单节点4A800、1h、block0 off→collect，冻结的70文件包与新版分析协议不变。
- 本轮一个常规查询，另有必要的提交前空队列保护及提交后回执；不短轮询。下次目标检查22:42:20CST；PENDING没有可信结束时间，未设置新时限提醒。原533/544启动测试及容量压力测试未重跑，旧GPU结果未取回或重分析。
- 新分析器本地133、隔离133通过；这是CPU验收器验证，不是544项GPU启动验证。终态后先诊断，再同源执行反向顺序，不按性能选择。全部效果门槛仍为false。
- 回执：`evidence/xplace-staging/ep4-tp2-causal-calibration-20261110-r7/submission-312336.json`；本轮完整检查点历史与查询计数见`causal-revision-analysis-20261110-r0/cycle.json`。

### 2026-11-11 — 人工恢复检查：r7容量拒绝，自动续接IPC失联

- 09:08:58CST一次查询确认队列为空；312336于04:56:41开始、05:20:16 FAILED/1:0结束。scontrol记录已过期，错误原样保留。随后一次终态归档保护、一次取回，不重复提交。
- 归档SHA `6b3ef25706f97e9e7b0901642e2ec60800fc068f20f68db24c878a348b093e56`；70上传文件/69计算文件/544启动身份通过。off四例240秒，44请求中16完成、28截尾；collect共享DP rank0自然预算38.5低于固定38.75 GiB，预热前拒绝，44个计划请求未执行。
- 新冻结分析器只分析本次新结果：八个开销配对全部缺失/不可用，退出0不代表通过；未放宽预算、margin、reserve或采集门槛，未原包补跑。
- 01:18至08:51的16次launchd检查均在IPC快照阶段超时；09:10只读探测仍失败。01:07的IPC重置/router选举与失联相关，但底层路由原因未完全证明。定时器仍加载，自动续接未恢复，需重连原会话后验证；未杀IDE进程或绕过owner保护。
- 诊断见`docs/r7-terminal-and-watchdog-gap-20261111.md`；原始失败、日志、窗口及所有缺失配对均保留。全部门槛false、campaign保持active。

### 2026-11-11 — 用户要求恢复原会话自动续接

- 09:39只读probe仍超时；连接表确认80840的监控路由与其连接到93695旧路由的会话客户端隔离。09:47原ParaCloud窗口私有desktop bridge的ping、PID、inode、窗口注册匹配。
- 新增一次性恢复工具：等待本轮正常结束后，仅通过原窗口内置接口重载一次；STOP、人工新turn、检查点/绑定变化及请求过期均中止。恢复后必须通过原同thread/workspace/resumed/idle/请求检查，才发布结果并唤醒现有watchdog，不绕过去重，不直接发turn。
- 本地91、隔离91通过；初次6个错误为模拟Unix socket临时路径过长，修复及原日志保留。原watchdog发送保护未改，deadline活动副本仍待未来arm前部署，未设提醒。
- 一次性程序须在本轮结束后才能操作；准备/等待回执不代表已恢复，实际结果看`watchdog-recovery-20261111-r0/result.json`与后续发送记录。无本轮集群查询、上传、取回或提交，r7失败和所有门槛false不变。
- r0后台启动因不能读取桌面源码SHA而退出1，未重载；错误保留。r1把运行绑定、输出和检查点限定在已有私有活动目录，并新增路径边界回归；本地96、隔离96通过。成功后只更新活动检查点，下一原会话turn核验后同步回仓库；实际结果路径改为活动目录`recovery-20261111-r1/result.json`。不修改系统权限，不重启失败r0。

### 2026-11-11 — 新会话接管及r8容量修订

- 用户在新thread `01a1c86d-44da-7eb0-934b-269bbd15d887`明确要求接续实验与后台监控。原thread恢复于10:02:48CST重连超时；其失败记录和检查点保留。停止旧watchdog绑定，在独立私有目录绑定当前thread，不恢复旧observer、不删除PAUSE。当前IPC的thread/workspace/resumed/active及0 pending request实测通过；新launchd首次tick正确跳过active turn。监控/提醒本地72、隔离72通过，已部署修复过的deadline源；自动发送仍待本轮结束后的真实验收。
- 10:32:36CST一次squeue/sacct确认队列为空、312336 FAILED/1:0。不重复取回或分析r7。当前新检查点为`evidence/ep4-tp2-ablation-20261111/campaign.json`，新监控配置为`evidence/session-watchdog-20261111/config.json`。
- 修复measure_budget不保留mapped-KV计算输入的问题；现在记录原始summaries、逐rank算术和顺序采样时间。分析器独立重算，篡改或缺失则拒绝。用r6/r7四份原始free-memory快照、忽略非负mapped所得下界最小值`[38.5,40.25]`GiB；最大302MiB波动向上取整为512MiB，两个rank统一额外保留，冻结共同预算`[38.0,39.75]`。3GiB margin、256 reserve、原始数据/输出、240s窗口及各门槛不变；不是未来显存保证。
- 本地初次定向210通过、1旧预算夹具常量失败，修复并保留日志。最终启动本地549/隔离549、分析本地141/隔离141均通过，0skip。冻结时本地JUnit有`experiments.xplace.`前缀，已按明确前缀映射核对549项，GPU合同仍保留精确`tests.*`身份，不重跑测试、不放宽验收。全部544旧启动身份保留。
- r8新70文件包只改2runtime+2test+4input/manifest，排除worktree图扫描代码；分析快照单独冻结并绑定69计算文件及549身份。单次upload于10:43开始，10:44:57仍在原SSH传输中，工具26264/PID38605，尚无上传完成回执或sbatch。禁止第二次上传。所有科学门仍false；详见`docs/tp2-capacity-record-revision-20261111.md`及新campaign。

### 2026-11-11 — r8上传成功，Slurm队列RPC超时后安全封住旧提交

- 10:46:32CST原upload完成远端SHA核验及原子发布，无第二次上传。随后单次提交命令在120秒超时，输出停在`PRE_SUBMISSION_QUEUE`，没有返回job ID。10:49:29只读恢复确认原提交锁及回执尚不存在，有界30秒squeue仍退出124；不能将超时解释为空队列。
- 10:51:11–12在共享远端原子取得原`.submit-once`锁并写专用hold令牌。取得成功证明旧脚本尚未越过其mkdir，未来即使队列查询迟到返回也不能提交；没有执行本轮sbatch，没有GPU作业号。hold和旧失败回执不删除。
- 恢复入口`deploy-r8.py recover-submit`已准备但尚未调用：核验hold及原提交未发生，使用独立共享防重锁、30秒Slurm查询与独立本地/远端回执；guard失败明确记录未执行sbatch。两项mock控制检查及生成脚本bash语法通过，不是实际提交验收。下一次不得早于11:19:29CST，不重复上传或测试。
- 当前thread后台监控已重新对齐30分钟周期，10:54:16的真实launchd tick正确跳过active turn，下一次计划11:24:16CST。当前源码/活动副本、检查点副本及189个已记录PAUSE的SHA一致；无在途部署命令。自动发送须待本轮结束后真实验证，不能提前标成功。
- campaign保持active、全部科学门false。操作入口：`docs/codex-takeover-20261111.md`；最终监控验收：`evidence/session-watchdog-20261111/final-verification.json`。后续按新检查点执行有界提交恢复、终态验收和同源反向块，不恢复旧thread或旧observer。

### 2026-11-11 — 自动续接实测成功，r8恢复提交313797

- 11:24:17CST当前thread收到watchdog续接消息，accepted/try_count=1；上轮`01a1c86d-9507-7123-a751-f624b1d2388e`续接为`01a1c8a2-8813-78b3-bf5c-5b564b03a51c`。原始state/checks与标记核验保存于`evidence/session-watchdog-20261111/automatic-delivery-20261111-r0.json`，不是将active skip当作送达。
- 本地确认无残留部署进程及恢复回执后，仅调用一次`deploy-r8.py recover-submit`。11:26:38–40CST核验冻结包/分析SHA、189个PAUSE、旧hold、312336 FAILED/1:0及实时空队列，再以独立共享锁执行一次sbatch，返回313797；SSH退出0，无重复上传、测试或提交。
- 11:26:40CST提交后squeue仅有313797，PENDING/Priority、StartTime=N/A、时限1h。计划仍为双DP2/TP2/EP4模型、同节点4A800、block0 off→collect、共同预算`[38.0,39.75]`GiB，8case/88请求、549精确启动测试合同；GPU结果尚未产生，不能据提交成功声称测量或性能通过。
- 原`.submit-once/controller-timeout-hold`与新`.submit-after-timeout-once`保持，禁止重放任何提交入口；提交回执为`evidence/xplace-staging/ep4-tp2-causal-calibration-20261111-r8/submission-313797.json`，上轮完整检查点另存`campaign-before-313797-submission.json`。
- 仅重新对齐当前thread监控；11:28:05CST真实launchd tick正确跳过active turn，下一次11:58:05CST，不早于11:56:40查询Slurm。排队无可信结束时间，未设结束提醒；确认真实RUNNING后再按开始时间+1h设置。所有科学门仍false，终态先验收/诊断，再准备同源反向块，不恢复旧observer或分时链。

### 2026-11-11 — 313797半小时复查，仍排队

- 11:58:05CST第二次自动续接送达当前thread，accepted/try_count=1。本地无残留部署或取回命令；11:59:27CST单次有界squeue/sacct/scontrol均退出0，仅313797 PENDING/Priority，Elapsed=0、无分配节点，job目录及日志未创建。原hold、新恢复锁与313797/exit0提交回执均核验存在。
- 调度预测StartTime=13:11:14、EndTime=14:11:14CST，但实际Start/End为Unknown；不是可信运行结束时点，不设置提醒。本轮不重复提交、不上传、不取回或分析，不重跑已通过测试。
- 原始查询：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T0358-r0.json`。上轮完整检查点保留于`campaign-before-313797-20261111T0358.json`，仅压缩活动摘要以满足16000字符限制；失败与负结果不删。当前thread监控重新对齐至12:30:15CST，不早于12:29:27查询；campaign保持active、全部科学门false，旧observer与PAUSE不动。

### 2026-11-11 — 313797第三次续接检查，调度预测延后

- 12:30:15CST第三次自动续接送达同thread，accepted/try_count=1；本地无残留实验命令。12:31:04CST单次有界squeue/sacct/scontrol均退出0，仅313797 PENDING/Priority、Elapsed=0、无节点；运行目录及日志未创建，原hold/恢复锁和313797提交回执仍在。
- 预测开始由13:11:14推迟为17:38:48CST，预测结束18:38:48；实际Start/End仍Unknown，不设置结束提醒。本轮无重复提交、重传、重测、取回或分析，所有科学门仍false。
- 查询原文：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T0430-r0.json`；旧检查点完整保存于`campaign-before-313797-20261111T0430.json`，活动摘要仅替换最新查询指针，不删历史。当前thread下一监控13:01:43CST，不早于13:01:04查询；保持active，旧observer与189个PAUSE不动。

### 2026-11-11 — 313797第四次续接检查，仍未分配节点

- 13:01:43CST第四次自动续接送达同thread，accepted/try_count=1；无残留实验命令。13:02:36CST单次有界squeue/sacct/scontrol全部退出0，队列仅313797 PENDING/Priority、Elapsed=0、无分配节点，工作目录及日志尚未创建；原hold、新恢复锁和提交回执均在。
- 排队预测仍为17:38:48开始、18:38:48结束CST，真实Start/End仍Unknown；不据排期设置提醒。本轮无新提交、重传、重测、取回或分析，无新GPU结果，全部科学门false。
- 原始查询：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T0501-r0.json`；完整前轮快照：`campaign-before-313797-20261111T0501.json`。当前thread下一监控13:33:02CST，不早于13:32:36查询，保持active；旧observer及189个PAUSE不动。

### 2026-11-11 — 313797第五次续接检查，预测排期跨日

- 13:33:02CST第五次自动续接送达同thread，accepted/try_count=1；本地无在途实验命令。13:33:56CST单次有界squeue/sacct/scontrol均退出0，仅313797 PENDING/Priority、Elapsed=0、无节点，工作目录及日志尚未创建；原hold、恢复锁与提交回执均在。
- 调度预测由11月11日17:38:48推迟至11月12日03:02:15开始、04:02:15结束（均CST）；真实Start/End仍Unknown，不设置结束提醒，不因排期延后撤销重投。本轮无新提交、重传、重测、取回或分析，无新GPU结果，全部科学门false。
- 原始查询：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T0533-r0.json`；完整前轮快照：`campaign-before-313797-20261111T0533.json`。当前thread下次检查为11月11日14:04:17CST，不早于14:03:56查询；保持active，不恢复旧observer，189个PAUSE保持。

### 2026-11-11 — 313797第六次续接检查，仍排队

- 14:04:17CST第六次自动续接送达同thread，accepted/try_count=1；本地无在途实验命令。14:05:11CST单次有界squeue/sacct/scontrol全部退出0，队列仅313797 PENDING/Priority、Elapsed=0、无节点，工作目录及日志未创建；原hold、恢复锁及提交回执均存在。
- 调度预测仍为11月12日03:02:15开始、04:02:15结束（CST），真实Start/End仍Unknown；未据此设置提醒或重新提交。本轮无重传、重测、取回或分析，无新GPU结果，全部科学门false。
- 原始查询：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T0604-r0.json`；完整前轮快照：`campaign-before-313797-20261111T0604.json`。当前thread下次检查11月11日14:35:37CST，不早于14:35:11查询；保持active，不恢复旧observer，189个PAUSE保持。

### 2026-11-11 — 313797第七次续接检查，仍排队

- 14:35:37CST第七次自动续接送达同thread，accepted/try_count=1；本地无在途实验命令。14:36:19CST单次有界squeue/sacct/scontrol全部退出0，仅313797 PENDING/Priority、Elapsed=0、无节点，工作目录与日志尚未创建；原hold、恢复锁与提交回执均在。
- 调度预测从11月12日03:02:15调整为02:29:16开始、03:29:16结束（CST）；真实Start/End仍Unknown，不据此设置提醒或重新提交。本轮无重传、重测、取回或分析，无新GPU结果，全部科学门false。
- 原始查询：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T0635-r0.json`；完整前轮快照：`campaign-before-313797-20261111T0635.json`。当前thread下次检查11月11日15:06:40CST，不早于15:06:19查询；保持active，旧observer与189个PAUSE不动。

### 2026-11-11 — 313797第八次续接检查，仍排队

- 15:06:40CST第八次自动续接送达同thread，accepted/try_count=1；本地无在途实验命令。15:07:09CST单次有界squeue/sacct/scontrol全部退出0，仅313797 PENDING/Priority、Elapsed=0、无节点，工作目录及日志未创建；原hold、恢复锁及提交回执均存在。
- 排期预测仍为11月12日02:29:16开始、03:29:16结束（CST），实际Start/End仍Unknown；不设置结束提醒或重复提交。本轮无重传、重测、取回或分析，无新GPU结果，全部科学门false。
- 原始查询：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T0706-r0.json`；完整前轮快照：`campaign-before-313797-20261111T0706.json`。当前thread下次检查11月11日15:37:27CST，不早于15:37:09查询；保持active，旧observer与189个PAUSE不动。

### 2026-11-11 — 313797第九次续接检查，仍排队

- 15:37:28CST第九次自动续接送达同thread，accepted/try_count=1；本地无在途实验命令。15:38:06–07CST单次有界squeue/sacct/scontrol全部退出0，仅313797 PENDING/Priority、Elapsed=0、无节点，工作目录与日志未创建；原hold、恢复锁和提交回执均存在。
- 调度预测仍为11月12日02:29:16开始、03:29:16结束（CST），真实Start/End仍Unknown；不设置结束提醒或重复提交。本轮无重传、重测、取回或分析，无新GPU结果，全部科学门false。
- 原始查询：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T0737-r0.json`；完整前轮快照：`campaign-before-313797-20261111T0737.json`。当前thread下次检查11月11日16:08:25CST，不早于16:08:07查询；保持active，旧observer与189个PAUSE不动。

### 2026-11-11 — 313797第十次续接检查，预测排期进一步延后

- 16:08:25CST第十次自动续接送达同thread，accepted/try_count=1；本地无在途实验命令。16:09:17CST单次有界squeue/sacct/scontrol全部退出0，仅313797 PENDING/Priority、Elapsed=0、无节点，工作目录及日志未创建；原hold、恢复锁与提交回执均存在。
- 调度预测从11月12日02:29:16推迟至11月13日11:05:45开始、12:05:45结束（CST）；真实Start/End仍Unknown，不设置结束提醒、不修改资源或重复提交。本轮无重传、重测、取回或分析，无新GPU结果，全部科学门false。
- 原始查询：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T0808-r0.json`；完整前轮快照：`campaign-before-313797-20261111T0808.json`。当前thread下次检查11月11日16:39:40CST，不早于16:39:17查询；保持active，旧observer与189个PAUSE不动。

### 2026-11-11 — 313797第十一次续接检查，仍排队

- 16:39:41CST第十一次自动续接送达同thread，accepted/try_count=1；本地无在途实验命令。16:40:19–20CST单次有界squeue/sacct/scontrol全部退出0，仅313797 PENDING/Priority、Elapsed=0、无节点，工作目录及日志未创建；原hold、恢复锁与提交回执均在。
- 调度预测从11月13日11:05:45推迟至当日14:17:34开始、15:17:34结束（CST）；真实Start/End仍Unknown，不设置结束提醒、不修改资源或重复提交。本轮无重传、重测、取回或分析，无新GPU结果，全部科学门false。
- 原始查询：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T0839-r0.json`；完整前轮快照：`campaign-before-313797-20261111T0839.json`。当前thread下次检查11月11日17:10:39CST，不早于17:10:20查询；保持active，旧observer与189个PAUSE不动。

### 2026-11-11 — 313797第十二次续接检查，仍排队

- 17:10:39CST第十二次自动续接送达同thread，accepted/try_count=1；本地无在途实验命令。17:11:34CST单次有界squeue/sacct/scontrol全部退出0，仅313797 PENDING/Priority、Elapsed=0、无节点，工作目录及日志未创建；原hold、恢复锁与提交回执均在。
- 调度预测从11月13日14:17:34提前至当日02:19:57开始、03:19:57结束（CST）；真实Start/End仍Unknown，不设置结束提醒、不修改资源或重复提交。本轮无重传、重测、取回或分析，无新GPU结果，全部科学门false。
- 原始查询：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T0910-r0.json`；完整前轮快照：`campaign-before-313797-20261111T0910.json`。当前thread下次检查11月11日17:41:57CST，不早于17:41:34查询；保持active，旧observer与189个PAUSE不动。

### 2026-11-11 — 313797第十三次续接检查，仍排队

- 17:41:57CST第十三次自动续接送达同thread，accepted/try_count=1；本地无在途实验命令。17:42:32CST单次有界squeue/sacct/scontrol全部退出0，仅313797 PENDING/Priority、Elapsed=0、无节点，工作目录及日志未创建；原hold、恢复锁与提交回执均在。
- 调度预测从11月13日02:19:57提前至11月12日00:10:00开始、01:10:00结束（CST）；SchedNodeList=d1n41a23g03仅为预测，实际NodeList为空、Start/End为Unknown。不设置结束提醒、不修改资源或重复提交，无重传、重测、取回或分析，无新GPU结果，全部科学门false。
- 原始查询：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T0941-r0.json`；完整前轮快照：`campaign-before-313797-20261111T0941.json`。当前thread下次检查11月11日18:12:49CST，不早于18:12:32查询；保持active，旧observer与189个PAUSE不动。

### 2026-11-11 — 313797第十四次续接检查，仍排队

- 18:12:49CST第十四次自动续接送达同thread，accepted/try_count=1；本地无在途实验命令。18:13:36CST单次有界squeue/sacct/scontrol全部退出0，仅313797 PENDING/Priority、Elapsed=0、无节点，工作目录及日志未创建；原hold、恢复锁与提交回执均在。
- 调度预测从11月12日00:10:00推迟至11月12日01:02:13开始、02:02:13结束（CST）；SchedNodeList=d1n41a23g03仅为预测，实际NodeList为空、Start/End为Unknown。不设置结束提醒、不修改资源或重复提交，无重传、重测、取回或分析，无新GPU结果，全部科学门false。
- 原始查询：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T1012-r0.json`；完整前轮快照：`campaign-before-313797-20261111T1012.json`。当前thread下次检查11月11日18:43:50CST，不早于18:43:36查询；保持active，旧observer与189个PAUSE不动。
- 本地中文strftime格式返回空字符串造成上轮摘要漏日期；改用显式月/日加ASCII时刻，已依据上轮原始Slurm记录补正为11月12日00:10:00。原错误文本与复现保留于`evidence/ep4-tp2-ablation-20261111/date-formatting-diagnosis-20261111-r0.json`，结构化字段和原始实验结果未变。

### 2026-11-11 — 313797第十五次续接检查，预测排期延后

- 18:43:51CST第十五次自动续接送达同thread，accepted/try_count=1；本地无在途实验命令。18:44:23CST单次有界squeue/sacct/scontrol全部退出0，仅313797 PENDING/Priority、Elapsed=0、无节点，工作目录及日志未创建；原hold、恢复锁与提交回执均在。
- 调度预测从11月12日01:02:13推迟至11月13日04:02:36开始、05:02:36结束（CST）；本次未提供SchedNodeList，实际NodeList为空、Start/End为Unknown。不设置结束提醒、不修改资源或重复提交，无重传、重测、取回或分析，无新GPU结果，全部科学门false。
- 原始查询：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T1043-r0.json`；完整前轮快照：`campaign-before-313797-20261111T1043.json`。当前thread下次检查11月11日19:14:35CST，不早于19:14:23查询；保持active，旧observer与189个PAUSE不动。

### 2026-11-11 — 313797第十六次续接检查，仍排队

- 19:14:36CST第十六次自动续接送达同thread，accepted/try_count=1；本地无在途实验命令。19:15:24CST单次有界squeue/sacct/scontrol全部退出0，仅313797 PENDING/Priority、Elapsed=0、无节点，工作目录及日志未创建；原hold、恢复锁与提交回执均在。
- 调度预测从11月13日04:02:36提前至11月12日18:58:50开始、19:58:50结束（CST）；SchedNodeList=d1n41a16g01仅预测，实际NodeList空、Start/End仍Unknown。不设置结束提醒、不修改资源或重复提交，无重传、重测、取回或分析，无新GPU结果，全部科学门false。
- 原始查询：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T1114-r0.json`；完整前轮快照：`campaign-before-313797-20261111T1114.json`。当前thread下次检查11月11日19:45:39CST，不早于19:45:24查询；保持active，旧observer与189个PAUSE不动。

### 2026-11-11 — 313797第十七次续接检查，仍排队

- 19:45:39CST第十七次自动续接送达同thread，accepted/try_count=1；本地无在途实验命令。19:46:34CST单次有界squeue/sacct/scontrol全部退出0，仅313797 PENDING/Priority、Elapsed=0、无节点，工作目录及日志未创建；原hold、恢复锁与提交回执均在。
- 调度预测从11月12日18:58:50推迟至11月13日04:02:36开始、05:02:36结束（CST）；本次未提供SchedNodeList，实际NodeList空、Start/End仍Unknown。不设置结束提醒、不修改资源或重复提交，无重传、重测、取回或分析，无新GPU结果，全部科学门false。
- 原始查询：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T1145-r0.json`；完整前轮快照：`campaign-before-313797-20261111T1145.json`。当前thread下次检查11月11日20:16:52CST，不早于20:16:34查询；保持active，旧observer与189个PAUSE不动。

### 2026-11-11 — 313797第十八次续接检查，仍排队

- 20:16:52CST第十八次自动续接送达同thread，accepted/try_count=1；本地无在途实验命令。20:17:46CST单次有界squeue/sacct/scontrol全部退出0，仅313797 PENDING/Priority、Elapsed=0、无节点，工作目录及日志未创建；原hold、恢复锁与提交回执均在。
- 调度预测仍为11月13日04:02:36开始、05:02:36结束（CST），较上次未变；本次未提供SchedNodeList，实际NodeList空、Start/End仍Unknown。不设置结束提醒、不修改资源或重复提交，无重传、重测、取回或分析，无新GPU结果，全部科学门false。
- 原始查询：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T1216-r0.json`；完整前轮快照：`campaign-before-313797-20261111T1216.json`。当前thread下次检查11月11日20:48:09CST，不早于20:47:46查询；保持active，旧observer与189个PAUSE不动。

### 2026-11-11 — 313797第十九次续接检查，仍排队

- 20:48:10CST第十九次自动续接送达同thread，accepted/try_count=1；本地无在途实验命令。20:48:51CST单次有界squeue/sacct/scontrol全部退出0，仅313797 PENDING/Priority、Elapsed=0、无节点，工作目录及日志未创建；原hold、恢复锁与提交回执均在。
- 调度预测从11月13日04:02:36提前至11月11日22:02:29开始、23:02:29结束（CST）；SchedNodeList=d1n41a16g01仅预测，实际NodeList空、Start/End仍Unknown。不设置结束提醒、不修改资源或重复提交，无重传、重测、取回或分析，无新GPU结果，全部科学门false。
- 原始查询：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T1248-r0.json`；完整前轮快照：`campaign-before-313797-20261111T1248.json`。当前thread下次检查11月11日21:19:07CST，不早于21:18:51查询；保持active，旧observer与189个PAUSE不动。

### 2026-11-11 — 313797第二十次续接检查，SSH超时、实时状态未知

- 21:19:07CST第二十次自动续接送达同thread，accepted/try_count=1；本地无残留实验命令。唯一一次有界SSH查询于21:20:23CST以exit255超时结束，stdout为空，未取得Slurm或日志数据；实时状态unknown，不视为空队列。
- 最近成功观测仍为20:48:51CST的313797 PENDING/Priority；该次22:02:29/23:02:29预测未在本轮复核，不设置结束提醒，不重试或重复提交。无重传、重测、取回或分析，无新GPU验收，科学门全false。
- 失败原文：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T1319-r0.json`；完整前轮快照：`campaign-before-313797-20261111T1319.json`。下一监控11月11日21:51:26CST，不早于21:50:23查询；current_state_known=false，成功读取后更新。保持active，189个PAUSE哈希不变，旧observer不动。

### 2026-11-11 — 313797第二十一次续接检查，恢复读取、仍排队

- 21:51:26CST第二十一次自动续接送达同thread，accepted/try_count=1；本地无在途实验命令。21:52:35CST唯一一次有界squeue/sacct/scontrol均退出0，仅313797 PENDING/Priority、Elapsed=0、无节点，工作目录及日志未创建；原hold、恢复锁及提交回执仍在。
- current_state_known恢复true，前轮21:20:23CST的SSH exit255失败原文保留。调度预测推迟至11月12日00:51:21开始、01:51:21结束；预测节点d1n41a12g02不代表实际分配，Start/End仍Unknown，不设结束提醒或重复提交。无重传、重测、取回、分析或新GPU结果，科学门全false。
- 原始查询：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T1351-r0.json`；前轮完整快照：`campaign-before-313797-20261111T1351.json`。下一监控11月11日22:23:37CST，不早于22:22:35查询；保持active，189个PAUSE哈希不变，旧observer不动。

### 2026-11-11 — 313797第二十二次续接检查，仍排队

- 22:23:38CST第二十二次自动续接送达同thread，accepted/try_count=1；本地无在途实验命令。22:24:21CST唯一一次有界squeue/sacct/scontrol均退出0，仅313797 PENDING/Priority、Elapsed=0、无节点，工作目录及日志未创建；原hold、恢复锁及提交回执仍在。
- 调度预测由11月12日00:51:21推迟至00:54:58开始、01:54:58结束；预测节点d1n41a12g02不代表实际分配，Start/End仍Unknown，不设结束提醒或重复提交。此前SSH失败原文保留，无重传、重测、取回、分析或新GPU结果，科学门全false。
- 原始查询：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T1423-r0.json`；前轮完整快照：`campaign-before-313797-20261111T1423.json`。下一监控11月11日22:54:48CST，不早于22:54:21查询；保持active，189个PAUSE哈希不变，旧observer不动。

### 2026-11-11 — 313797第二十三次续接检查，仍排队

- 22:54:49CST第二十三次自动续接送达同thread，accepted/try_count=1；本地无在途实验命令。22:55:43CST唯一一次有界squeue/sacct/scontrol均退出0，仅313797 PENDING/Priority、Elapsed=0、无节点，工作目录及日志未创建；原hold、恢复锁及提交回执仍在。
- 调度预测从11月12日00:54:58推迟至11月12日01:11:40开始、02:11:40结束；SchedNodeList=d1n41a12g02仅预测，实际Start/End仍Unknown，不设置结束提醒。此前SSH失败原文保留；无重传、重测、取回、分析或新GPU结果，科学门全false。
- 原始查询：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T1454-r0.json`；前轮完整快照：`campaign-before-313797-20261111T1454.json`。下一监控11月11日23:26:14CST，不早于23:25:43查询；保持active，189个PAUSE哈希不变，旧observer不动。

### 2026-11-11 — 313797第二十四次续接检查，仍排队

- 23:26:15CST第二十四次自动续接送达同thread，accepted/try_count=1；本地无在途实验命令。23:26:57CST唯一一次有界squeue/sacct/scontrol均退出0，仅313797 PENDING/Priority、Elapsed=0、无节点，工作目录及日志未创建；原hold、恢复锁及提交回执仍在。
- 调度预测从11月12日01:11:40推迟至11月12日01:25:35开始、02:25:35结束；SchedNodeList=d1n41a12g02仅预测，实际Start/End仍Unknown，不设置结束提醒。此前SSH失败原文保留；无重传、重测、取回、分析或新GPU结果，科学门全false。
- 原始查询：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T1526-r0.json`；前轮完整快照：`campaign-before-313797-20261111T1526.json`。下一监控11月11日23:57:21CST，不早于23:56:57查询；保持active，189个PAUSE哈希不变，旧observer不动。

### 2026-11-11 — 313797第二十五次续接检查，仍排队

- 23:57:21CST第二十五次自动续接送达同thread，accepted/try_count=1；本地无在途实验命令。23:58:03CST唯一一次有界squeue/sacct/scontrol均退出0，仅313797 PENDING/Priority、Elapsed=0、无节点，工作目录及日志未创建；原hold、恢复锁及提交回执仍在。
- 调度预测从11月12日01:25:35提前至11月12日00:34:53开始、01:34:53结束；SchedNodeList=d1n41a12g02仅预测，实际Start/End仍Unknown，不设置结束提醒。此前SSH失败原文保留；无重传、重测、取回、分析或新GPU结果，科学门全false。
- 原始查询：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T1557-r0.json`；前轮完整快照：`campaign-before-313797-20261111T1557.json`。下一监控11月12日00:28:28CST，不早于11月12日00:28:03查询；保持active，189个PAUSE哈希不变，旧observer不动。

### 2026-11-12 — 313797第二十六次续接检查，仍排队

- 00:28:28CST第二十六次自动续接送达同thread，accepted/try_count=1；本地无在途实验命令。00:29:08CST唯一一次有界squeue/sacct/scontrol均退出0，仅313797 PENDING/Priority、Elapsed=0、无节点，工作目录及日志未创建；原hold、恢复锁及提交回执仍在。
- 调度预测从11月12日00:34:53推迟至11月12日00:40:44开始、01:40:44结束；SchedNodeList=d1n41a12g02仅预测，实际Start/End仍Unknown，不设置结束提醒。此前SSH失败原文保留；无重传、重测、取回、分析或新GPU结果，科学门全false。
- 原始查询：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T1628-r0.json`；前轮完整快照：`campaign-before-313797-20261111T1628.json`。下一监控11月12日00:59:34CST，不早于11月12日00:59:09查询；保持active，189个PAUSE哈希不变，旧observer不动。

### 2026-11-12 — 313797第二十七次续接检查，已运行、启动日志549通过

- 00:59:34CST第二十七次自动续接送达同thread，accepted/try_count=1；本地无在途实验命令。01:00:17CST单次有界squeue/sacct/scontrol均退出0，仅313797 RUNNING，实际00:35:15CST开始、节点d1n41a12g02、单节点4GPU，Elapsed=00:25:02；hold及提交回执完整。
- 启动日志549 passed/105.15s仅作日志观察，XML精确身份、8worker、TP分片与容量算术待终态验收；嵌套日志未读取，不因顶层out/run.log缺失判定卡死。未重投、重传、重测、取回或分析，科学门全false。
- 实际StartTime+1h与scontrol EndTime一致，结束提醒目标11月12日01:35:15CST；下次例行监控11月12日01:32:42CST，不早于11月12日01:30:18查询。原始查询：`evidence/ep4-tp2-ablation-20261111/status-313797-20261111T1659-r0.json`；前轮快照：`campaign-before-313797-20261111T1659.json`。保持active，189个PAUSE不变，旧observer不动。

### 2026-11-12 — 第二十八次续接：r8 科学拒绝，预注册反向块 315554 排队

- 01:35:27CST 唯一例行查询确认 313797 已于 01:17:01 COMPLETED/0:0，耗时 41m46s；squeue 成功且为空，sacct 明确终态，scontrol 已清理作业的错误原文保留。01:35:15 提醒因当前 turn 活跃正确 skip，未重启。
- 单次取回 156,717,387 字节/397 成员，SHA 验证；70 上传/69 计算文件、549 精确启动身份、两模式各8 worker/完整EP4TP2分片/8次迁移及原始容量算术通过。共同预算 [38,39.75]GiB，自然预算均 [38.75,40.5]GiB；88 请求全保留，28完成、60截尾。
- 修复两个仅限分析的缺陷：入口误拒绝整数修订3，产物审计误拒绝正常 startup started→passed。红灯及 r0/r1 结果保留；本地/隔离各155通过。r2 性能、覆盖和四个查询文件与 r1 相同，GPU包和科学阈值未变。
- **科学拒绝**：四case采样覆盖80.61/84.32/80.74/86.78%，均低于90%；Code TTFT采集开销最差+17.77%，超15%上限。无采集中事件丢失；1616次80ms超时与3次停止取消保留，具体慢组件尚不能从组合返回定位。不能按成功子集训练/验收，不启动正式消融或混合。
- 按预注册补齐 collect→off，不替换本块也不试图消除超限。新tag `ep4-tp2-causal-calibration-20261112-r8b1` GPU包逐字节相同，仅tag/block变化；6项隔离提交保护、2项防重放检查通过。02:04:01CST 新鲜空队列及313797终态保护后单次提交 **315554**；02:04:04队列仅315554 PENDING/Priority，无实际开始时间，故不设结束提醒。
- 下一检查 **11月12日02:36:15CST**，不早于02:34:04；本轮不再查询集群。189个PAUSE不变，旧observer未动，当前campaign保持active并发布。详细结果与分析修复：`docs/r8-terminal-and-analysis-entrypoint-20261112.md`；终态审计：`evidence/xplace-ep4-tp2-causal-calibration-20261111-r8/terminal-audit-313797-r2.json`；提交回执：`evidence/xplace-staging/ep4-tp2-causal-calibration-20261112-r8b1/submission-315554.json`。


### 2026-11-12 — 第67次续接：r9本地候选预检，未提交

- 22:24:39CST单次只读查询：队列为空，315554仍COMPLETED/0:0；scontrol清理错误保留。189个PAUSE及旧冻结文件不变，无重复归档或配对分析。
- 原CPU子进程超时未复现：诊断571通过，新候选无插件本地571/上传571通过；原570通过、1失败及5s时限不变，根因未证实。
- 通用容量压力4个c16超512MiB，最大572.70MiB；冻结源码3/9阶段路径细化为最大483.30MiB，未改上限，不保证未来GPU产出容量。
- r9候选绑定73上传/72计算文件、571启动身份；9项身份门及真实XML通过。原样classname前缀比较失败保留，上传契约直接精确匹配。HOLD保留，无打包、部署、提交；还缺逐轮诊断验收器与独立分析绑定。
- 16769/16178字符候选被大小门拒绝，均发生在写入/发布前；保留全量快照后压缩重复摘要。全部科学门false；下次本地22:51:08CST，集群不得早于22:54:40。详情：`docs/tp2-component-preflight-20261112.md`。

### 2026-11-12 — 第68次续接：组件验收和分析冻结，未提交

- 22:56:36CST单次只读查询确认空队列，315554仍COMPLETED/0:0；scontrol清理错误保留，无重复取回或旧结果重分析。
- 逐轮28组件验收新增52项测试通过；本地/独立分析各207通过，保留原155项。两项夹具权限失败与修复保留，运行时安全门和80ms截止不变。
- 新分析精确绑定72计算文件、571启动身份及组件协议。73清单文件加清单自身的74成员归档逐SHA通过；HOLD保留，仅本地打包，无部署、提交或提醒。
- 原CPU超时根因、容量超限压力估算及全部r8科学负结果保留；新一次性部署/提交保护仍待实现测试。下一本地续接23:21:09CST，集群不得早于23:26:38。详情：`docs/tp2-component-audit-20261112.md`。
- 新检查点辅助脚本误索引不存在的可选配置键，已仅修正读取；16469字符候选随后被大小门拒绝，两次均未写入或发布。保留失败与全量前轮快照，压缩重复字段后15684字符通过；工作区/活动目录发布字节与语义SHA一致。

### 2026-11-12 — 第69次续接：启动保护通过，部署部分传输失败

- 一次性部署/提交保护55项CPU测试通过，绑定原571启动与207/207分析回归；HOLD和189个PAUSE保留。单次新包传输触发240s上限，TERM10s后KILL，未调用sbatch。
- 23:34:40CST唯一只读核验：队列空，313797/315554均COMPLETED/0:0；新tag归档仅940032/1655236字节，SHA恰为本地同长度前缀，尚缺715204字节。没有upload目录、成功标记或提交锁；原写入者所在登录节点未知，不能凭ln02进程列表证明全部退出。
- 失败归档、意图、SSH输出和锁保留，无自动重传。旧路径隔离候选14项本地测试通过，含延迟旧写入者；未远端执行，完整恢复及新提交绑定仍待实现。
- 原CPU失败、容量压力超限及全部科学负结果保留，无新GPU allocation。下一本地续接23:51:10CST；集群查询不得早于11月13日00:04:41。详情：`docs/tp2-component-launch-20261112.md`。


### 2026-11-13 — 第70次续接：隔离恢复后单次提交 r9 组件诊断

- r1恢复器125项组合测试通过；初始97通过/16失败及109通过/4失败保留，修复了AND列表失效检查、私有只读目录rename权限和Bash3.2条件退出问题，未改GPU包。
- 00:00:40CST严格隔离器拒绝远端变化；00:01:20只读确认旧部署已于23:51:20写入精确`1\n`退出码，原940032字节前缀不变。拒绝正确保留，无自动重试。r2新增精确失败标记与快照绑定，136项本地测试通过。
- 00:03:58隔离旧路径，保留quarantine、只读前缀和失败码快照；新upload-r2完整接收1655236字节并逐SHA通过，传输耗时230.28s，随后74成员及只读目录发布验证通过。原网络根因仍未知；900s仅是传输上限，不改CPU5s、80ms/100ms、64KiB/512MiB或allocation1h。
- 00:09:20CST唯一Slurm守卫确认队列为空、313797/315554均COMPLETED/0:0，单次提交 **318110**（单节点4A800、1h、r9诊断pilot）。仅知SUBMITTED_UNCONFIRMED，未做提交后查询；571项启动与207项分析绑定不等于新GPU启动已通过，实际Start未知，不设结束提醒。
- 189个PAUSE、HOLD、全部旧失败和r8负结果保留，科学门槛仍全部false。下一本地续接00:21:12CST，集群查询不得早于00:39:21。恢复协议：`docs/tp2-component-recovery-20261113.md`；执行回执：`evidence/ep4-tp2-ablation-20261111/component-recovery-20261113-r2/execution-r2.json`。


### 2026-11-13 — 第71次续接：查询时间门内完成终态交接预检

- 当前仍早于00:39:21CST查询下限，本轮无集群查询、无提交、无真实归档/下载/分析；318110保留SUBMITTED_UNCONFIRMED，实际Start未知，不设置结束提醒。
- 新终态交接保护71项定向、278项组合测试通过（含冻结207项分析测试）。合成571身份XML只用于本地验证，不是真实GPU启动通过。失败/缺失原始产物允许保留，但不进入成功科学验收。
- 首次62通过/1失败及失败tar保留：macOS扩展属性AppleDouble越过根目录白名单，只有合成夹具禁用copyfile元数据，未放宽白名单。补齐流式成员上限、路径别名、非终态/队列冲突、启动身份、异修订缓存导入等保护。
- 单次远端归档/下载执行层尚未实现，不能重放旧fetch。189个PAUSE及全部冻结产物不变，科学门槛仍false。下次自动续接00:51:13CST，可按时间门查询318110一次。详见`docs/tp2-terminal-handoff-20261113.md`。


### 2026-11-13 — 第72次续接：318110排队，尚未启动

- 00:53:53CST单次查询：squeue唯一可见作业318110为PENDING/Priority，sacct实际Start/End均Unknown，scontrol确认单节点4A800、1h。远端job目录和日志尚未创建，提交锁与r2传输/发布成功回执仍在；未提交新作业。
- scontrol的17:41启动、18:41结束均为排队估计，不作为实际Start或可信结束提醒；本轮未设置提醒，未归档、下载或分析。下一集群查询不得早于01:23:53.753185CST（对外取整01:23:54），自动续接预计01:21:13，若早到仅核验本地。
- 第72次自动续接仅投递一次；189个PAUSE、HOLD和冻结产物通过本地完整性核验。沿用71/278项终态保护，未重复运行测试；单次终态取回执行层仍待实现。真实571项GPU启动验收未发生，所有科学门槛及原负结果不变。查询证据：`evidence/ep4-tp2-ablation-20261111/status-318110-20261112T1651-r0.json`。


### 2026-11-13 — 第73次续接：恢复发布后响应断线，未重复远端操作

- 第72轮响应流在publish成功后断开；本轮核验工作区、活动目录和续接携带SHA完全一致。原终检文件未写入，缺失事实及断线错误保留，现时核验不回填旧时间，也不将响应断线当作集群或GPU作业失败。
- 本轮检查早于01:23:53.753185CST时间门，无集群查询、提交、取回、分析或测试重跑。318110仅沿用00:53:53的PENDING/Priority观测；实际Start未知，不根据排队Start/End设置提醒。
- 第73次续接投递一次，189个PAUSE和冻结产物核验不变，全部科学门槛仍false。下一自动续接01:51:14CST，集群查询不得早于01:23:54；不前台等待或短轮询。恢复证据：`evidence/ep4-tp2-ablation-20261111/interrupted-publication-reconciliation-20261112T1721.json`。


### 2026-11-13 — 第74次续接：318110仍因优先级排队

- 01:51:42CST单次squeue/sacct/scontrol查询均成功：唯一可见作业318110为PENDING/Priority，实际Start/End均Unknown，工作目录和日志未创建。提交锁和r2成功回执可见，未提交新作业或重复取回。
- 排队估计改为13:50启动、14:50结束；不是实际启动或可信结束时间，未设提醒。下一查询不得早于02:21:42.226573CST（对外取整02:21:43），自动续接预计02:21:15，若早到仅核验本地。
- 第74次续接仅投递一次；189个PAUSE、HOLD、终态与恢复冻结产物核验不变。本轮未运行新测试、归档、下载或分析；全部科学门槛与负结果不变，单次终态取回执行层仍待实现。查询证据：`evidence/ep4-tp2-ablation-20261111/status-318110-20261112T1751-r0.json`。


### 2026-11-13 — 第75次续接：318110实际启动，注册单次时限提醒

- 02:21:50CST单次查询确认318110是唯一可见RUNNING作业；squeue/sacct/scontrol一致给出实际Start=02:20:41、节点d1n41a11g02、单节点4A800、1h。startup-tests.log仅有部分进度，不能据此认定571项启动通过，尚无正式推理结果。
- 已注册并核验当前会话单次本地提醒：实际Start+1h=03:20:41CST。新LaunchAgent仅运行既有提醒脚本、KeepAlive=false、runs=1且waiting回执存在；不访问集群，不恢复旧observer，不越过查询下限。时限到达不是成功保证。
- 下次集群查询不得早于02:51:51.194262CST（对外取整02:51:52），自动续接预计02:51:16，早到仅核验本地。本轮无新增提交、归档、下载、分析或本地测试；第75次投递仅一次，189个PAUSE和冻结产物不变，所有科学门槛仍false。查询证据：`evidence/ep4-tp2-ablation-20261111/status-318110-20261112T1821-r0.json`；提醒回执：`evidence/session-watchdog-20261111/deadline-armed-318110-20261112T192041Z.json`。


### 2026-11-13 — 第76次续接：318110运行中，启动通过仅作日志观测

- 02:52:08CST单次查询确认318110仍是唯一RUNNING作业，已运行31:27，实际Start/节点/1h时限不变。初次本地时间门检查尚差0.253秒，未访问集群；实际SSH开始02:52:03，未早于02:51:51.194262下限。
- 日志报告571项测试105.90s全部通过，并报告571项精确身份校验；76903字节XML已列出，但未下载独立验收。collect初始化日志已出现。off收尾的socket未确认退出、TCPStore断开和force-kill警告保留，不据此认定终态失败或生产端根因。
- 原03:20:41CST单次提醒仍waiting、runs=1，没有重新注册；下一集群查询不得早于03:22:09.245751CST（对外取整03:22:10）。提醒和当前预计03:21:17普通续接均在时间门内，只能本地核验，不覆盖30分钟约束。
- 第76次续接投递一次，189个PAUSE/HOLD/冻结产物不变。本轮未新增提交、归档、下载、分析或本地测试；全部科学门槛仍false，终态单次取回执行层待实现。查询证据：`evidence/ep4-tp2-ablation-20261111/status-318110-20261112T1851-r0.json`。

### 2026-11-13 — 第77次续接：r9单次取回，诊断有效但科学门槛拒绝

- 03:22:16CST单次查询确认318110 COMPLETED/0:0，实际02:20:41—03:02:29，41:48；队列为空，scontrol已清除但sacct终态完整。03:20:41提醒投递一次后退出，未重设；无新allocation。
- 执行层36项与既有278项合计314项通过并冻结。单次归档/下载171349418字节，SHA256 `9a26482e699f21d29a5fc8f1d328acf977e2b5ee25901f75723eba91373dee61`；安全解包413成员。冻结分析独立核验571项真实启动身份、72计算文件、73清单；88请求=19完成+69截断，9457轮各28组件，无无效轮。
- 科学拒绝：活跃完整率70.51–76.91%低于90%；仅4/8配对，反序块缺失。单块描述性中位损失为Conv吞吐倒数+8.28%、token间隔+9.41%、CodeTTFT+7.65%，均高于5%。活跃丢弃0但post-close拒绝63，不能称全程无损；所有科学门槛仍false，不跨r8/r9配对。
- 2531不完整轮中2026轮仅scheduler缺失；7048次scheduler取消停在rpc_started。仅定位RPC等待路径，尚未证明生产端瓶颈或TTFT因果。下一步本地核对成功/取消时间线与train-c4-r1 tick2410汇总边界，不原包重跑。
- 第77次续接单次投递；189个PAUSE、HOLD、锁、失败与负结果保留。最终检查点两次大小校验失败均未发布，第二次16068字符；失败单独记录后以SHA绑定完整快照精简索引，不删科学证据。下次普通续接做本地诊断，不常规重查已结束作业；集群查询下限03:52:16.690909CST仍保留。
- 结果：`kvcached/experiments/xplace/docs/tp2-r9-component-results-20261113.md`；执行回执：`evidence/ep4-tp2-ablation-20261111/terminal-execution-20261113-r0/execution-r0.json`。

### 2026-11-13 — 第78次续接：全量RPC时间绑定，截止取消解释完成

- 本轮只读既有终态与冻结原始产物，未访问集群、重取、重跑原验收或占用allocation。第78次续接单次投递，189个PAUSE/HOLD/锁及原失败、负结果保留。
- 新的独立本地诊断保留全部37828个scheduler调用：30780完成、7048取消；30718个成功快照可绑定同钟时间，62个完成快照因64KiB部分结果上限省略，不填值。取消调用中7042个所在轮次超时、6个为外部取消；成功子集区间不是全部调用延迟或生产端根因。
- train-c4-r1 tick2410为trace cutoff后的父任务取消，完成时距deadline仍有67.43ms。冻结收集器的合成测试复现28组件均及时但父轮取消的路径；17项新测试全部通过。该轮仍记不完整，不放宽覆盖率或修改原结果。
- 诊断代码、全量调用表、异常轮及报告已冻结。未修改运行代码；下一步先做有界派发/生产进入退出/返回/事件循环相关诊断及取消保留测试，再冻结新修订，不原包盲重跑、不仅为补反序块申请新作业。
- 全部科学门槛仍false；r9仅4/8配对、覆盖率70.51–76.91%、三个中位损失>5%的拒绝不变。下一普通续接预计04:21:18CST，仍是本地工作，不常规轮询已结束作业。
- 检查点首次16063字符超限且未发布，失败记录保留；把五个历史准备字段移入SHA绑定的完整快照后留出续接余量，不删除原文件或科学证据。
- 结果：`kvcached/experiments/xplace/docs/tp2-r9-rpc-timeline-20261113.md`；冻结：`evidence/ep4-tp2-ablation-20261111/rpc-diagnosis-20261113-r0/result-freeze-r0.json`。

### 2026-11-13 — 第79次续接：有界RPC诊断候选609项隔离测试通过

- 新建独立本地候选，未修改r9上传包、原分析或主runtime。75个候选文件中7个有差异，新增默认关闭的关联ID、生产端64条有界历史、紧凑时间元数据及deadline回调探针；socket4096字节、64KiB/512MiB、28组件/16阶段、80ms/100ms不变。导出省略/淘汰明确记数，不声称无损或生产端因果。
- 保留并修复复制只读权限及本地校验误含未复制MANIFEST的问题。首次针对性测试31通过/2夹具失败，定位到Python3.9事件循环创建时机及目录0700要求；修复后33通过，扩展取消/错误身份/实际socket边界后38通过，未修改实验门槛。
- 首次隔离全目录收集因缺少cloudpickle和误含tools-only分析测试中止，0个测试正文执行；原目录与2个收集错误保留。修复为冻结571项非分时启动清单+38项新诊断，并显式复制/SHA绑定既有本地cloudpickle依赖后，609项在93.95s全部通过，精确身份、源码和隔离副本SHA一致。CPU子进程5s未放宽，历史超时根因仍未知。
- 本地候选及100项产物已冻结，新HOLD保留，未部署或提交。最终生产端尾部持久化、跨样本缺口/重启/迟到关联、schema2分析器以及新启动/部署清单尚未完成；复制的571项及r9清单不能用于新提交。下一步在新r1中推进，不能修改冻结r0。
- 第79次续接单次投递；189个PAUSE和旧HOLD/锁/失败/负结果保持。本轮没有集群查询、重取、原分析重跑或allocation。r8/r9原拒绝不变，四项科学门槛仍false；下一普通续接预计04:51:19CST，仍是本地准备。
- 候选说明：`kvcached/experiments/xplace/docs/tp2-rpc-instrumentation-20261113.md`；冻结：`evidence/ep4-tp2-ablation-20261111/rpc-instrumentation-20261113-r0/freeze-r0.json`。

### 2026-11-13 — 第80次续接：r1收尾采集与独立关联校验通过

- 新r1保留r0/r9和主runtime不变。在原trace请求清理之后一次性读取4路engine socket并保存尾部；不新增utility RPC，不改变原cutoff、请求状态和完整率。80ms协作式timeout、4096/2048字节线缆/导出界限不变，尾部≤16KiB且与事件共用原512MiB payload预算；缺失、取消和超限均保留，不自动重试，也不声称生产端已排空。
- 运行侧50项针对性测试通过；621项隔离回归（609原身份+12新测试）在95.10s通过。独立schema2关联分析另有58项隔离测试通过，包括真实本地socket、取消后开始的生产记录、重复/缺口/重启/冲突、未触发回调以及字节/身份对抗校验。精确身份和源文件SHA均核验；本地通过不等同于GPU验证。
- 独立分析保留所有客户状态、未匹配生产记录和缺失序号，不把取消重标完成，不推断未观察等于未执行，不改原完整率。核心指标/完整产物校验的schema2整合、新启动/部署清单与一次性提交契约仍未完成，HOLD和不可部署状态保持。
- 初始本地校验误读规范化状态回执为stdout传输回执，KeyError在任何写入/复制/网络前中止；修复为核验原SHA绑定transport后继续，失败回执保留。第80次续接单次投递；189个PAUSE、旧HOLD/锁及历史失败、负结果保留。本轮无集群查询、取回、旧分析重跑或新allocation。
- 全部科学门槛仍false；r8/r9原拒绝不变，生产端瓶颈和TTFT因果仍未证明。下一步是新的完整验收/部署契约本地整合，不盲重跑或启动混合拓扑。
- 说明：`kvcached/experiments/xplace/docs/tp2-rpc-tail-validation-20261113.md`；修订：`evidence/ep4-tp2-ablation-20261111/rpc-instrumentation-20261113-r1`。

### 2026-11-13 — 第81次续接：schema2完整验收和新本地契约通过

- 新副本整合schema2关联、跨case核对与原完整产物/指标验收；核心causal分析器与冻结r9逐字节相同。off子进程强制RPC关闭、collect需显式opt-in，实际mode记录进入验收。取消、截断、原完整率和负结果不重标，不重算旧GPU结果。
- 633项运行侧隔离回归、276项独立分析、最终70项契约/实际启动程序对抗测试通过；精确身份及源码SHA核验。78项计算payload、78行MANIFEST、含MANIFEST共79个候选文件，启动脚本单独绑定；新本地CPU XML夹具通过完整输入绑定，不是GPU证据。
- 保留只读复制权限失败、依赖字段KeyError、误复制弱启动检查及旧元数据的问题记录。弱启动校验在部署前修复：恢复精确身份/计数和失败/跳过检查，防优化移除断言；旧版本及前67项测试回执保留，旧r0/r1/r9冻结和主runtime不变。
- 本地契约要求HOLD、1节点4A800/1h、30分钟下限、新鲜空队列和两级锁，并始终返回不授权提交；远端transport尚未实现。HOLD和不可部署状态保持；下一步仅在新修订推进执行器及部署preflight，不直接翻转冻结字段或原包重跑。
- 第81次单次投递，189个PAUSE、全部失败/负结果及锁保留。无集群查询、取回、旧分析重跑或allocation。r8/r9拒绝不变，生产端瓶颈及TTFT因果仍未证明，四项门槛均false；预计05:51:21CST继续本地工作。
- 说明：`kvcached/experiments/xplace/docs/tp2-rpc-acceptance-contract-20261113.md`；新准备：`evidence/ep4-tp2-ablation-20261111/rpc-acceptance-20261113-r0`。

### 2026-11-13 — 第82次续接：r10受保护传输冻结并一次部署，未提交

- 新执行器保留父166项冻结产物和主runtime。80上传文件/79行MANIFEST，其中78个计算文件与父输入逐字节相同；归档1,662,343字节、SHA `f29d37bfd4dabc7dd4d2a2618d21ea49fcc8124760cba5b645cd6ee9be512169`，原633启动身份不变。
- 本地首次59通过/3失败，定位到shell未因失败的独立正则条件退出；改为显式case拒绝异常作业号。原失败、源码差异及格式错误留证。65项修复后及65项镜像隔离测试通过，包含双进程真实mkdir竞争、全局/单tag锁、歧义提交、partial、SHA和授权对抗；Slurm均为本地替身。
- 106项源码/输入在真实动作前冻结。06:05:39 CST一次只读预检成功：318110 COMPLETED/0:0、队列为空、新tag/全局占用标记不存在、两项依赖SHA匹配。06:07:02 CST起一次新包传输/发布通过远端SHA和文件数核验；新HOLD保留，无重复部署或旧包重放。
- 尚无submit授权、意图或新allocation；下一Slurm查询/提交检查不得早于06:35:39 CST。预计06:21:22续接只做本地检查，不提前查询，不把排队估计当结束时间，也不设置虚构的结束提醒。
- 第82次单次投递、189个PAUSE和全部失败/负结果保留。四项科学门槛仍false，生产端瓶颈与TTFT因果不作结论；真实新GPU启动、容量、覆盖/开销和严格配对仍待验证。
- 说明：`kvcached/experiments/xplace/docs/tp2-rpc-transport-20261113.md`；执行器：`evidence/ep4-tp2-ablation-20261111/rpc-transport-20261113-r0`。

### 2026-11-13 — 第83次续接：提交链路45项离线核验通过

- 只读核验106项源码冻结、119项结果冻结、80项上传文件及前序部署回执；r10仍为已部署未提交，没有新增allocation，未重复预检或部署。
- 新增45项完整execute链路测试全部通过，独立收集与XML精确身份一致。覆盖授权与查询下限、前序回执绑定、非终态拒绝、成功派发、歧义留证和重复执行阻断；所有传输进程为本地替身，未修改冻结执行器或主runtime，不把模拟作业号当真实提交。
- 06:26:47 CST真实本地时钟核验仍被06:35:39查询下限拒绝。本轮不访问集群、不创建真实submit授权、不等待到下限；下一普通续接预计06:51:23.350348 CST，届时仍需当前campaign绑定的短时授权与锁内新鲜队列检查，不能沿用06:05:39空队列。
- 第83次单次投递；189个PAUSE及HOLD、锁、失败和负结果保留。四项科学门槛仍false，GPU启动/容量/覆盖/开销及生产端或TTFT因果仍未验证。
- 说明：`kvcached/experiments/xplace/docs/tp2-rpc-submit-chain-20261113.md`；证据：`evidence/ep4-tp2-ablation-20261111/rpc-submit-chain-20261113-r0/validation-r0.json`。
- 发布前本地完整性检查首次把主runtime误比为r9诊断候选，断言失败且尚未执行publish。错误与两个SHA已留存；改按冻结transport/base-r0.json的全量主runtime指纹验证，全部匹配，无源码修复或远端动作。该最终校验失败不改变45项链路测试结果或科学门槛。
- 随后收尾检查的ps表头解析和无关lch-dev-2 SSH误报也在publish前失败，三次原失败及修订均保留；改用无表头PID字段并精确识别paracloud实验进程，未触碰无关连接。修正后一次publish成功，36项主runtime指纹不变；查询下限及科学门槛未放宽。

### 2026-11-13 — 第84次续接：r10一次受保护提交318902

- 06:52:55 CST，ln02同一全局及单tag锁内确认318110 COMPLETED/0:0、squeue为空，随后唯一一次sbatch返回318902且exit=0。新短时授权绑定当前campaign、冻结源码和部署回执；没有重放预检/部署，未立即查询新作业。
- 请求单节点4张A800、时限01:00:00，block0 off→collect，仅验证RPC诊断与容量/收集开销；80项上传文件和78项计算payload未改变。633项是冻结启动测试的预期身份数，不是已通过的GPU结果。实际启动、节点及运行状态未知，不能用提交时间加1h构造结束提醒。
- 下一查询不得早于07:22:55.968226 CST；普通续接预计07:21:24.363370 CST，若提前到达只做本地核验。到期执行一次新只读状态查询；若仍未结束，记录下一30分钟下限并结束，不短轮询。
- 第84次单次投递、189个PAUSE及HOLD、全局/tag/本地单次锁、失败和负结果保留。四项科学门槛仍false，不与r8/r9配对，不启动混合拓扑；三个上一轮收尾校验失败仍留存。本轮36项主runtime指纹一致。
- 提交回执：`evidence/ep4-tp2-ablation-20261111/rpc-submit-20261113-r0/submission-318902-r0.json`；原传输/授权/单次意图都已SHA绑定冻结。检查点将四个历史上下文字段移至SHA绑定的完整提交前快照，保留全部原证据。

### 2026-11-13 — 第85次续接：318902仍PENDING/Priority

- 07:23:21 CST，一次新的只读squeue/sacct/scontrol查询一致确认318902 PENDING、原因Priority；用户队列只有该作业，未分配节点或资源，实际Start/End均Unknown，无启动日志或GPU测试结果。全局占用标记仍指向r10/318902。
- 调度器给出的11月15日11:33:48启动、12:33:48结束均为排队估计，不作实际启动或可信结束时间，未设置结束提醒。没有取消、重排或重提作业。
- 下一查询下限07:53:21.624457 CST；普通续接预计07:51:25.454256 CST，若提前到达只做本地检查。等待期间不短轮询、不新占allocation；本轮只更新状态和检查点。
- 第85次单次投递，189个PAUSE、所有HOLD/锁、失败及负结果保留；36项主runtime指纹一致，四项科学门槛仍false。
- 查询与归一化回执：`evidence/ep4-tp2-ablation-20261111/rpc-status-20261112T2321-r0/status-318902-r0.json`；已冻结原始查询输出，无原包重放或旧结果重分析。

### 2026-11-13 — 第86次续接：318902仍排队，调度估计变化不视为启动

- 07:53:27 CST到期后的一次新只读查询确认318902仍PENDING/Priority，用户队列仅此作业；实际Start/End仍Unknown、AllocTRES为空、NodeList为空，无作业输出或GPU启动证据。
- 排队估计变为11月13日17:14启动、18:14结束，SchedNodeList为d1n41a13g02；它们均不是实际启动、结束或已分配节点。保留上一轮11月15日估计，不设置结束提醒，不取消或重提作业。
- 下一查询不得早于08:23:27.843859 CST；普通续接预计08:21:26.457555 CST，提前到达只做本地检查。本轮没有短轮询、部署、提交、旧结果重分析或新增allocation。
- 第86次单次投递、189个PAUSE与所有HOLD/锁/失败/负结果保留，36项主runtime指纹一致，四项科学门槛仍false。
- 冻结查询回执：`evidence/ep4-tp2-ablation-20261111/rpc-status-20261112T2351-r0/status-318902-r0.json`。

### 2026-11-13 — 第87次续接：318902仍PENDING，无GPU启动证据

- 08:23:32 CST，一次到期只读查询确认318902仍PENDING/Priority，用户队列仅此作业。实际Start/End仍Unknown，AllocTRES为空、NodeList为空，无启动日志或GPU结果；未重复提交或部署。
- 11月13日17:14启动、18:14结束及d1n41a13g02仍是调度估计，不视为实际启动、结束或节点分配，未设置结束提醒。此前所有估计与负结果保留。
- 下一查询下限08:53:33.106019 CST；普通续接预计08:51:27.644299 CST，提前到达仅作本地核验。本轮在下限前仅一次本地定时等待，无循环轮询或提前集群查询。
- 第87次单次投递，189个PAUSE、全部HOLD/锁和失败记录保留；36项主runtime指纹一致，四项科学门槛仍false。
- 冻结查询回执：`evidence/ep4-tp2-ablation-20261111/rpc-status-20261113T0021-r0/status-318902-r0.json`。

### 2026-11-13 — 第88次续接：318902仍排队，保留调度估计波动

- 08:53:38 CST一次到期只读查询确认318902仍PENDING/Priority，用户队列仅此作业；Start/End实际值仍Unknown，未分配节点或资源，无启动输出或GPU结果。
- 排队估计回到11月15日11:33:48启动、12:33:48结束，SchedNodeList不再给出；上一轮11月13日17:14及预计节点记录保留。全部是调度估计，不设置结束提醒，不取消或重提作业。
- 下一查询下限09:23:38.375729 CST；普通续接预计09:21:28.380616 CST，提前到达仅做本地检查。本轮仅一次到期查询，无短轮询、部署、提交或新增allocation。
- 第88次单次投递，189个PAUSE、HOLD/全局及单次锁、失败与负结果均保留；36项主runtime指纹不变，四项科学门槛仍false。
- 冻结查询回执：`evidence/ep4-tp2-ablation-20261111/rpc-status-20261113T0051-r0/status-318902-r0.json`。

### 2026-11-13 — 第89次续接：318902仍PENDING，按30分钟间隔检查

- 09:23:44 CST一次到期只读查询确认318902仍PENDING/Priority，用户队列仅此作业；Start/End实际值仍Unknown，无已分配节点/资源、启动日志或GPU结果。
- 11月15日11:33:48启动、12:33:48结束仍仅为排队估计，不设置结束提醒。未重复提交、部署、归档/取回、重分析或新增allocation。
- 下一查询下限09:53:45.096546 CST；普通续接预计09:51:29 CST，提前到达仅做本地检查；本轮下限前仅作一次本地等待，无短轮询。
- 第89次单次投递，189个PAUSE、所有HOLD/锁和失败/负结果保留，36项主runtime指纹一致，四项科学门槛仍false。
- 冻结查询回执：`evidence/ep4-tp2-ablation-20261111/rpc-status-20261113T0121-r0/status-318902-r0.json`。

### 2026-11-13 — 第90次续接：318902仍排队，无GPU启动证据

- 09:53:52 CST一次到期只读查询确认318902仍PENDING/Priority，用户队列仅此作业；Start/End实际值Unknown，无已分配节点/资源、启动日志或GPU结果。
- 11月15日11:33:48启动、12:33:48结束仍是排队估计，不设置结束提醒。未重复提交、部署、归档/取回、重分析或新增allocation。
- 下一查询下限10:23:52.000000 CST；普通续接预计10:21:30 CST，提前到达仅做本地检查。本轮下限前仅作一次本地等待，无短轮询。
- 第90次单次投递，189个PAUSE、全部HOLD/锁及失败/负结果保留，36项主runtime指纹一致，四项科学门槛仍false；未启动混合拓扑实验。
- 冻结查询回执：`evidence/ep4-tp2-ablation-20261111/rpc-status-20261113T0151-r0/status-318902-r0.json`。

### 2026-11-13 — 第91次续接：318902仍排队，核对Code长度口径

- 10:24:00 CST单次到期只读查询确认318902仍PENDING/Priority，用户队列仅此作业；实际Start/End Unknown，无已分配节点/资源、启动日志或GPU结果。11月15日排队时间仍仅为估计，未设置结束提醒。
- 下一查询下限10:54:00.000000 CST；普通续接预计10:51:31 CST，提前到达仅做本地检查。本轮未重提、部署或增加allocation，189个PAUSE、全部HOLD/锁及负结果保留，四项科学门槛仍false。
- 用户询问Code平均与最大长度：完整10条SWE-smith池输入平均100923.6、最大101675 tokens；当前r10配置4条输入平均100758.5、最大101675 tokens；输出配置64，不是已完成GPU生成统计。逐条核对prompt_tokens与token_ids数量相等，未改变负载。
- 长度证据：`evidence/ep4-tp2-ablation-20261111/rpc-status-20261113T0221-r0/code-length-statistics-r0.json`；原始查询和统计均纳入本轮SHA冻结。
