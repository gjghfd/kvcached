# xplace 实验协议（草案，Job 2 前冻结）

## 0. 用户 guideline（2026-11-02）

1. A/B 效果都要有，并有消融：
   - A：迁移降低 Code TTFT；
   - B：cost-aware placement 降低 Conv 的性能下降。
2. Conv 相对同 trace 的 `no_move` 退化 ≤20%。主指标是 Conv avg TPOT，同时报告 P95/P99 TPOT 和 TTFT。
3. 复用数据集：Code 用 SWE-smith ≥100K，输出 64；Conv 用 OpenThoughts3 bank256，保持原始长度；不用 4x。
4. 真实实验尽量少，最终结果必须来自真实实验。
5. 每个 Slurm 作业 ≤1h（11-03），长矩阵拆成短作业逐个提交。
6. 分时（Code 驻留期间暂停 Conv）作为新增策略维度（11-03 同意）。

## 1. 系统设置（固定）

- **硬件与模型**：
  - 4×A800-SXM4-80GB，单节点；
  - 两个 Qwen3-30B-A3B-Instruct-2507 tenant（model0=Code，model1=Conv），共用同 4 卡；
  - 每个 tenant DP=4/EP=4/TP=1，BF16。
- **vLLM 配置**：vLLM 0.26.0 + kvcached 0.1.6；默认 sparse FULL_DECODE_ONLY CUDA graph，capture 档位 1,8,16,32,64,128,256；max_num_batched_tokens 4096；prefix caching 关闭。
- **KV 预留**：按 P+O 全量预留；每个 tenant 每个 rank 有 48 GiB 虚拟池。物理预算由 controller 在软件层强制。
- **KV 预算**：Job 1 实测自然预算（启动后空闲 + 已映射 − 3 GiB 余量）。Job 2 的预算按 §4 的规则确定并在此写明；若使用受控上限，结果中必须标注。

## 2. 工作负载

- **Code**：SWE-smith 10 条 ≥100K 输入（100,084–101,675 token），输出 64；每条 KV 约 9.375 GiB。
- **Conv**：OpenThoughts3 quality cohort bank256。P 均值 85.5；O 均值 13,211（4,827–16,760）；按原始长度 `ignore_eos` 回放。
- **到达**：两路独立 Poisson，`tools/make_trace.py`，有放回均匀抽样。
- **评估窗口**：剔除开头 warmup，只统计到达时间在 `[warmup, duration)` 内的请求；所有请求都跑到完成（drain）。

## 3. 策略

| 策略 | 准入（Code 放不下时） | 回迁 |
|---|---|---|
| `no_move` | 排队等待，Conv 正常准入 | 无 |
| `no_move_fair` | 排队；队首等待超过 30s 后阻止另一 tenant 的新准入，直到队首完成预留 | 无 |
| `mig_random` | cost-unaware：随机可行目标 + KV 最大优先 + 随机可行目的地 | 无 |
| `ca_admit` | cost-aware 准入 | 无 |
| `ca_return` | 与 `mig_random` 相同（同一请求用同一随机种子） | cost-aware 回迁 |
| `ca_full` | cost-aware 准入 | cost-aware 回迁 |

所有策略对能直接放下的请求都按请求数最少路由（相同计数时选占用最少的 rank）。只有 KV ≥ 4 GiB 的请求（Code）会走迁移准入。

## 4. 实验区间的选择规则（防止事后挑选）

1. Job 1 拟合 DP=4 的步时模型和干扰参数，并用 Job 1 的 pilot trace 检验 simulator（Code TTFT 和 Conv TPOT 的误差目标在 ±15% 以内）。
2. 在 simulator 中扫描预算（自然预算或受控上限）、Conv 速率和 Code 速率，**只用非最终 seed**。选择满足以下条件的区间：
   - `no_move` 中 Code 因单卡碎片排队（"总空闲 ≥ Code 需求、每卡都放不下"的时间占比有意义）；
   - Conv 全局排队有限（Conv TTFT P99 < 5s）；
   - 每个策略有 ≥15 个 Code。
3. planner 参数（horizon、margin 等）也只在非最终 seed 上调。
4. 冻结最终 trace（新 seed）、预算、参数，并把以上全部写入本节后再提交 Job 2。

### 4.1 实际执行情况（2026-11-03，偏离上述规则的地方如实记录）

- **simulator 在 pilot 区间未通过检验。** pilot 是 Job 1b，seed 101，16 GiB 上限，Conv 0.08、Code 0.004 RPS，1500s。在 no_move 上：
  - Conv TPOT 实测 30.9ms，模拟 21.3ms；
  - Code TTFT 实测 148s，模拟 45s。

  原因是实测干扰远强于校准所得：Code prefill 期间 Conv 每 token 约 150–177ms，而模型对每 rank ≤8 个 Conv 时取的是 37ms。另外多个 Code 并发时 Code TPOT 达 0.9–1.05s。因此第 2 步的 simulator 扫描不能作为选择区间的依据。
- **改为直接沿用 pilot 区间。** 实测 pilot 已同时出现 A 效果和 B 的苗头：
  - Code TTFT：no_move 148s → `mig_random` 58s / `ca_full` 60s；
  - Conv TPOT：no_move 30.9ms → `mig_random` 29.6ms / `ca_full` 28.1ms。
- **held-out**：最终 trace 改用新 seed 202，并加长时长。没有在 seed 202 上调任何参数或区间。
- **Job 1c 的结论**：把 Code 所在卡上的 Conv 全部移走会使干扰更大，所以 planner 不采用"隔离"策略。

## 4.2 冻结配置（Job 2；2026-11-03 13:1x 在开始运行前撤销，seed 202 上没有任何真实数据）

- **预算**：每卡 16 GiB 受控上限，结果中必须标注。同机实测自然预算为 34.25–36.75 GiB/卡。设上限是为了在可控负载下再现单卡碎片。
- **trace**：`evidence/xplace-inputs/main-b16-c008-k004-s202.json`
  - SHA256 `2c3bb8dd537ac1a431eb27003efd9145411665fe962018c7e4df0d938d5bccbb`；
  - Conv 0.08 RPS、Code 0.004 RPS、时长 5300s；
  - 共 388 个 Conv、24 个 Code，其中 23 个 Code 落在评估窗口 `[300, 5300)` 内。
- **策略与执行顺序**：`no_move`, `ca_full`, `mig_random`, `ca_admit`, `no_move_fair`, `ca_return`。每个策略都从空系统开始，跑完全部请求（drain）。
- **参数**：
  - planner 用 `planner-params-v1.json`（`DEFAULT_PARAMS`）；
  - 步时模型用 `step-params-dp4-v1.json`（Job 1b 改进拟合）；
  - fairness τ=30s，迁移准入门槛 4 GiB。
- **SLO**（用于 goodput）：
  - Code：TTFT ≤ 120s；
  - Conv：TTFT ≤ 5s 且 TPOT ≤ 50ms。

## 5. 指标

- **Code**：TTFT 的 avg/P50/P90/P95/P99；排队时间。
- **Conv**：
  - 每请求 TPOT 的 avg/P95/P99；
  - token 加权 TPOT；
  - TTFT 的 avg/P95/P99。
- **合并**：TTFT 的 avg/P95/P99。
- **SLO goodput**：阈值在冻结时写明（暂定 Conv TTFT ≤ 2s 且 TPOT ≤ 2×`no_move` 中位数；Code TTFT ≤ 120s）。
- **控制代价**：迁移次数、迁移字节、事务次数、暂停的 request-seconds。
- **报告方式**：每个策略相对 `no_move`、相对 `mig_random` 的变化，并给出样本数；不把首 token 之后才发生的回迁算作 TTFT 收益。

## 6. 短作业 campaign（2026-11-03 冻结，替代 Job 2）

### 6.1 为什么改

- **simulator 已用 pilot 重新校准并通过检验**（`sim/fit_live.py` 用 pilot `no_move` 拟合，`mig_random`/`ca_full` 留出）：
  - Conv TPOT avg 误差 0.2–3.3%；30 个噪声副本的分布覆盖了三个策略的实测值；
  - 迁移策略的 Code TTFT 偏高 5–8%；
  - Conv TTFT 不吻合：simulator 没有模拟控制面延迟，见 §6.5。
- **按校准后的 simulator，B（placement 本身）在所有扫描区间都 ≤2%**（seed 303/404，Conv 0.05–0.08 × Code 0.004–0.008）。Job 2 的预测为 `ca_full` 对 `mig_random` −2.1%，90% 区间 [−4.6, +0.2]；消融中 `ca_admit` ≈ 0。
- **主因是跨 tenant 干扰**：Code 驻留期间两个 tenant 都慢 10 倍以上，placement 无法改变这一点。vLLM DP 的 `PAUSED_ALL` 暂停期间 engine 完全空闲，所以"Code 驻留期间暂停 Conv"（分时）可以消除干扰。
- **用户要求每个作业 ≤1h**，因此撤销 12h 的 Job 2，改为 ≤1h 的短作业逐个提交。

### 6.2 新增策略：分时（`*_tm`）

- Code 被准入（直接准入或迁移准入）后，Conv tenant 在下一个 32 步边界暂停；最后一个驻留的 Code 完成后恢复。
- 迁移准入时，事务结束后 Conv 直接转为持有暂停，不经过恢复再暂停。
- 持有期间到达的 Conv 照常预留 KV，但要等恢复后才开始生成。持有期间若有新请求尚未开始 decode，迁移事务推迟。
- 策略：`ca_full_tm`（cost-aware 准入 + 回迁 + 分时）、`no_move_tm`、`mig_random_tm`。

### 6.3 冻结 trace 与作业矩阵

- **trace**：`evidence/xplace-inputs/short-b16-c008-p180-s701.json`，SHA256 `68225407de01cb33ee7a7d756b1fd71fa085a248f5f73dade47f61aae8ab7245`。
  - Conv 按 Poisson 0.08 RPS 到达，持续 1100s，共 92 个；
  - Code 固定在 300/480/660/840s 到达（4 个），间隔 180s，避免短窗口里出现 Poisson 突发；
  - 评估窗口 [300, 1100)，所有请求都跑到完成；
  - seed 选取规则（事先写定）：从 701 起，取第一个 Conv 实际数量落在 88±5% 内的 seed，结果是 701。
- **其余配置与 §4.2 相同**：每卡 16 GiB 受控上限，`planner-params-v1.json`，planner belief 为 `step-params-dp4-v1.json`，迁移门槛 4 GiB。
- **作业**（每个约 50–53 分钟，`--time=01:00:00`，drain 超时 700s，同一个 upload `s1-tm-20261103`）：

  | 作业 | 策略（按运行顺序） | 主要回答的问题 |
  |---|---|---|
  | S1 | `ca_full_tm`, `ca_full` | 分时对 Conv TPOT 和 Code TTFT 的影响（同机配对） |
  | S2 | `no_move_tm`, `no_move` | A 的基线；不迁移时分时的效果 |
  | S3 | `mig_random`, `ca_full` | placement B；`ca_full` 的重复，用来估计 run-to-run 噪声 |
  | S4+ | `ca_admit`, `ca_return`, `no_move_fair`, `mig_random_tm` | 消融（视前面结果决定） |

### 6.4 预注册预测与判定

预测文件：`evidence/xplace-inputs/forecast-short-s701-v2.json`，每个策略 20 个噪声副本，在提交 S1 之前生成。

| 对比 | Code TTFT 预测 | Conv TPOT 预测 |
|---|---|---|
| `ca_full` 对 `no_move`（A） | −74% | −1.9% [−5.3, +1.1] |
| `ca_full_tm` 对 `ca_full`（分时） | −58% | −30% [−31, −28] |
| `no_move_tm` 对 `no_move` | −36% | −28% |
| `ca_full_tm` 对 `no_move_tm`（分时下的 A） | −83% | −5.2% |
| `ca_full` 对 `mig_random`（placement B） | +1.7% | −1.2% [−5.0, +2.9] |

- **判定**：
  - A 成立：迁移类策略的 Code TTFT 低于对应的不迁移基线；
  - 分时成立：`*_tm` 的 Conv TPOT avg 低于同准入策略的非分时版本，且 Code TTFT 不变差；
  - Conv 退化 ≤20%：相对同 trace 的 `no_move`。
- **分时的代价必须同时报告**：Conv 最大 token 间隔（P50/P99）和 Conv TTFT。
- placement B 在短作业中预计测不出（噪声 ±4%），如实报告，不据此调参。

### 6.5 已知局限

- **控制面延迟**：controller 的直接准入要持锁向两个 tenant 的 8 个 engine core 发 RPC。Code prefill 期间每个 chunk 步约 2.4s，RPC 要等当前步结束，因此突发到达的 Conv 会排队（pilot 中 Conv TTFT P99 为 20–32s）。所有策略都受影响，本 campaign 不改。
- **噪声估计**：请求级 bootstrap 不包含 run-to-run 噪声（干扰强度存在两种模式，每次 Code episode 随机落入其一）。用 S1/S3 中重复的 `ca_full` 和 simulator 的噪声副本来估计。

### 6.6 修订（14:3x）：S1 在慢节点上失败，改用更稳健的 trace，并加节点探针

- **S1 = 283685 失败**（节点 d1n41a23g03）。`ca_full_tm` trial 在 drain 超时（1800s）时还剩 8 个 Conv 没完成，job 按 fail-fast 退出，`ca_full` 没有运行。证据在 `evidence/xplace-s1-tm-20261103/`。
  - 分时机制本身正常：4 次持有暂停，每次约 30s；持有期间新到的 Conv 已预留 KV、未开始生成，Code 完成后统一恢复。
  - 根因是节点慢：t=0–60s 只有 1–3 个 Conv 时步时已是 39–45ms，pilot 节点约 11ms；中段每 rank 12–14 个 Conv 时 56–74ms，pilot 约 24.5ms。Conv 寿命翻倍后活跃数涨到 51 个，KV 占满，Code 2、3 排队约 10 分钟。
  - 该节点 FreeMem 只有 34 GB（共 1 TB），CPULoad 16；pilot 节点 FreeMem 882 GB。慢在 CPU 侧：分时下 Code prefill 只慢 13%，Code decode 慢 2.3 倍。
- **simulator 敏感性**（所有步时乘以 k）：原 trace（λ=0.08，D=1100）在 k≥1.15 时迁移策略 Code TTFT 翻倍，k=1.3 时进入过载，作业超过 1h。
- **新冻结 trace**：`evidence/xplace-inputs/short-b16-c007-p180-s702.json`，SHA256 `6dae767580ad0742b15bbd51ede43d5719d0ad14bb71f78f0af3c43aa9086d95`。
  - Conv 按 Poisson 0.07 RPS 到达，持续 1000s，共 73 个；Code 仍在 300/480/660/840s；
  - seed 规则不变：从 701 起取第一个 Conv 数落在 70±5% 内的，结果为 702。
  - simulator 估计 k=1.0/1.2/1.4 时最坏作业时长为 47/52/57 分钟。
- **节点探针**（`--probe-max-ms 16.5`）：启动后跑每 rank 8 个 Conv（1K context）、输出 600 token，测稳态步时。pilot 节点上该值为 12.7ms，超过 1.3 倍即中止。S1b 同时 `--exclude d1n41a23g03`。
- **新预测**：`evidence/xplace-inputs/forecast-short-s702-v2.json`，20 个副本，提交前生成。

  | 对比 | Code TTFT | Conv TPOT |
  |---|---|---|
  | `ca_full` 对 `no_move`（A） | −45% | +1.0% |
  | `mig_random` 对 `no_move` | −44% | +6.9%，P(>0)=0.95 |
  | `ca_full` 对 `mig_random`（B） | −1.3% | −5.5%，[−11.8, +0.6]，P(<0)=0.93 |
  | `ca_admit` / `ca_return` 对 `mig_random` | — | −4.4% / −4.0% |
  | `ca_full_tm` 对 `ca_full`（分时） | −55% | −28% |
  | `no_move_tm` 对 `no_move` | −74% | −28% |
  | `ca_full_tm` 对 `no_move_tm`（分时下的 A） | −6%（k=1 时弱，k=1.2 时为 −74%） | ≈0 |

- **作业矩阵不变**（S1b–S5），upload tag 改为 `s1b-tm-20261103`。

### 6.7 作业矩阵调整（14:5x）：关键对比都放进同一作业

- **原因**：节点之间的速度差异可达数倍。即使有 1.3 倍阈值的探针，跨作业比较仍可能被节点差异翻转。simulator 估计：节点慢 1.3 倍时 `ca_full` 的 Code TTFT 约 140s，而快节点上 `no_move` 约 110s。
- **调整后的作业**：

  | 作业 | 策略 | 用途 |
  |---|---|---|
  | S1b | `ca_full_tm`, `ca_full` | 分时 |
  | S2 | `no_move_tm`, `no_move` | 分时（不迁移） |
  | S3 | `no_move`, `ca_full` | A（同机配对） |
  | S4 | `mig_random`, `ca_full` | placement B |
  | S5 | `ca_admit`, `ca_return` | 消融，与 S4 跨作业比较时注明探针值 |

- `ca_full` 在 S1b/S3/S4 中各跑一次，用这三次的离散程度估计跨节点和 run-to-run 噪声。

### 6.8 以后的实验格式（11-04 用户要求）

- **每轮 ≤20 分钟**：一轮 = 一个策略完整回放一次 trace 并跑完。默认 trace 改为 `short-b16-c007-p180-s702-d850.json`（SHA `56ea1ec8…`），即 1000s trace 的前 850s，并发参数不变；评估窗口 [300, 850)。
- **多轮打包进同一个 Slurm 作业**：`XPLACE_POLICIES` 列出各轮，`--time` 约为轮数 × 21 分钟加 5 分钟。启动时只做一次节点探针；作业运行期间持续监视。
- S1b–S6 仍用 1000s trace，彼此可比；以后的 850s 数据只与 850s 数据比较。

## 7. 算法优化轮（v2，11-04 用户要求"继续分析并优化算法"）

### 7.1 依据（来自 S1b–S5 的 live 分析）

- **控制面阻塞**：Conv 准入要实时查询两个 tenant 的 8 个 engine core。Code prefill 期间，Code tenant 每步 1–2.4s，RPC 要等当前步结束。
  - Conv 排队时间（到达到预留完成）：无 Code 时 0.15–0.20s，Code prefill 期间 2.2–3.2s（p90 3.2–8.7s）；
  - 窗口内约 45% 的 Conv 在 Code prefill 期间到达。
- **分时下直接准入时的并行**：Code 先开始 prefill，Conv 走到 32 步边界才停，期间并行 3.7–7s，Code prefill 从 25s 拉长到 28.5–30.6s（8 个分时 Code 中有 3 个）。
- **placement 已接近最优**：`ca_full` 的可避免 G16 占比只有 0.07–0.13（`mig_random` 为 0.41），剩余的 G16 时间来自 Conv 总数超过 32，不是 placement 能改善的。

### 7.2 v2 改动（只改控制面，不改 planner）

1. **Conv 准入快路径**（`fast`）：
   - Conv 准入时只实时查询 Conv tenant，Code tenant 用缓存视图；
   - Code KV 只会在持锁准入 Code（直接准入、迁移准入、重试预留）时增加，前后各递增一次 `code_epoch` 并更新或作废缓存；
   - Code 完成时作废缓存；监视每 5s 刷新一次，若期间 epoch 变化就丢弃；
   - 缓存只会高估 Code 占用，不会造成超额准入；每 5s 的容量检查仍保留。
2. **先停 Conv 再准入 Code**（`hold_first`，仅分时策略）：直接准入 Code 时，在锁内先让 Conv 停到边界，再提交 Code。
- 新策略：`ca_full_v2`（fast）、`ca_full_tm_v2`（fast + hold_first），另有 `no_move_v2`、`no_move_tm_v2`。
- 测试：新增 4 个（快路径查询 Code tenant 的次数 125 → 33 并守住预算；紧预算加迁移；先停 Conv 时重叠步数为 0；epoch 拒绝过期刷新），共 22 个通过。

### 7.3 验证作业（打包，同一节点配对）

- **trace**：`short-b16-c007-p180-s702-d850.json`（850s，每轮约 20 分钟）。
- **5 轮**：`ca_full_tm_v2`, `ca_full_tm`, `ca_full_v2`, `ca_full`, `no_move`。`--time 2:15:00`，drain 超时 800s，探针 16.5ms。
- **预测**：`evidence/xplace-inputs/forecast-v2-d850-s702.json`，20 个副本。simulator 不含控制面延迟，修正了先停 Conv 时 Code 等待期间被误算为干扰的问题。

  | 对比 | Code TTFT | Conv TPOT |
  |---|---|---|
  | `ca_full_tm_v2` 对 `ca_full_tm` | −7.1% [−9.5, −4.2] | −1.8% [−2.7, −1.0] |
  | `ca_full_tm_v2` 对 `no_move` | −77% | −32% |
  | `ca_full_v2` 对 `ca_full` | ≈0（simulator 不建模） | ≈0 |

- **快路径的 live 估算**（来自 S1b–S5 的排队数据）：非分时策略的 Conv TTFT 均值约降 1s（约 −60%），在 Code prefill 期间到达的 Conv 的排队时间从约 2.5s 降到约 0.3s。
- **判定**：
  - v2 对 v1：Code TTFT 不变差，Conv TTFT 下降，Conv TPOT 不变差；
  - 正确性：输出长度精确、迁移逐字节校验通过、全程没有 MemoryError。

## 8. 算法优化轮（v3：分片分时）

### 8.1 依据

- 全程分时（`*_tm`）在 Code TTFT 和 Conv TPOT 上最好，但 Conv 中途会停约 25s。Conv TTFT p99 约 28s，Conv SLO 达成率（TTFT ≤5s 且 TPOT ≤50ms）只有 0.78–0.80，不分时为 0.86–0.96。
- simulator 扫描（850s trace，6 个副本）：

  | 方案 | Code TTFT | Conv TPOT | Conv TTFT avg / p99 | Conv SLO | 最长停顿 |
  |---|---|---|---|---|---|
  | 全程分时 | 25.2s | 22.4ms | 3.9 / 23.5s | 0.78 | 26s |
  | 分片 S=10 | 28.0s | — | — / 9.8s | 0.86 | — |
  | 分片 S=5 | 30.3s | 22.7ms | 0.6 / 3.2s | 1.00 | 6s |
  | 分片 S=3 | 34.1s | — | — / 2.6s | — | 4s |

  选择 S=5、R=1。

### 8.2 v3 改动

- 新策略 `ca_full_tm_s5`：在 `ca_full_tm_v2` 基础上增加 slicer。
- 触发条件：Conv 持有满 S=5s，且仍有 Code 在 prefill、Conv 有请求。
- 每次切片：
  1. 持锁用 vLLM 原生 keep-pause 暂停 Code tenant（`pause_scheduler_async(mode="keep", clear_cache=False)`，等当前步完成并达成 DP 共识后返回，KV 保留）；
  2. 释放 Conv（持有记录写入 holds）；
  3. 释放锁，等 R=1s；
  4. 持锁让 Conv 停到边界，然后恢复 Code。
- Code 进入 decode 后不再切片，decode 只有约 1s。
- 安全性：
  - 暂停期间 `add_request` 仍会把请求加入调度器并完成预留，只是不调度。所以窗口内到达的 Code 准入不会死锁；之前分时持有期间的 Conv 准入已验证过这一点。
  - 窗口内的迁移事务或新 Code 准入若提前让 Conv 停下，后面的再暂停就是空操作。
- 测试：fake engine 增加 chunked prefill；重叠计数只统计 Code 真正在跑的步。新增分片测试：4 次切片，每次持有约 0.03s（S=0.03），重叠为 0，输出长度精确。共 23 个通过。

### 8.3 验证作业 V3

- **trace**：与 V2 相同的 850s trace。
- **3 轮同机配对**：`ca_full_tm_s5`, `ca_full_tm_v2`, `ca_full_v2`。`--time 1:15:00`，探针 16.5ms。
- **预测**：`evidence/xplace-inputs/forecast-v3-d850-s702.json`，20 个副本：

  | 对比 | Code TTFT | Conv TPOT | Conv TTFT avg |
  |---|---|---|---|
  | `ca_full_tm_s5` 对 `ca_full_tm_v2` | +20% | +1.5% | −83%（3.9 → 0.65s；p99 23.5 → 3.2s） |
  | `ca_full_tm_s5` 对 `ca_full_v2` | −50% | −33% | — |

- simulator 没有计入每次切片约 0.3s 的 DP 共识空转，live 中 Code TTFT 可能再多 1–2s。
- **判定**：
  - `ca_full_tm_s5` 的 Conv TTFT p99 ≤ 8s，且 Conv SLO 达成率 ≥ `ca_full_v2` 的 −0.05；
  - Conv TPOT 比 `ca_full_v2` 低 ≥20%；
  - Code TTFT 比 `ca_full_v2` 低 ≥30%；
  - 正确性同前。

## 9. 最终验证作业 V4（新 seed，4 个最终工作点）

### 9.1 依据

- V3b 第一轮 `ca_full_tm_s5` 实测：Code TTFT 29.9s，Conv TPOT 23.6ms，Conv TTFT p99 6.04s，SLO 0.94，最长停顿 8.2s。
- simulator 按真机时序修正切片模型（触发后 Code 先多跑一步，再暂停；放开 Conv 前有 0.3s DP 共识）。修正后在 S=5 上与 live 的差距：Code TTFT 3%、Conv TTFT p99 4%、SLO 0.02、停顿 1s。
- S 的权衡曲线（seed 702）：
  - S=3：33.2s / p99 3.6s / SLO 1.00
  - S=4：31.6s / 4.1s / 1.00
  - S=5：30.9s / 5.8s / 0.92
  - S=6：30.1s / 7.1s / 0.86
  - S=8：28.5s / 8.6s / 0.83
  - 全程分时：25.2s / 23.5s / 0.78

  选 S=4：让 Conv SLO 达到 1.00、Code TTFT 代价最小的点。

### 9.2 设计

- **trace**：`short-b16-c007-p180-s704-d850.json`（SHA `fc3592ef…`），按同一规则选出 seed 704（57 个 Conv，Code 仍在 300/480/660/840s）。seed 704 上没有调过任何参数。
- **4 轮同机配对**：`ca_full_tm_s4`, `ca_full_tm_v2`, `ca_full_v2`, `no_move`。`--time 1:45:00`，探针 16.5ms。
- **预测**：`evidence/xplace-inputs/forecast-v4-d850-s704.json`，20 个副本：

  | 策略 | Code TTFT | Conv TPOT | Conv TTFT avg / p99 |
  |---|---|---|---|
  | `ca_full_tm_s4` | 31.7s | 23.3ms | 0.5 / 5.4s |
  | `ca_full_tm_v2` | 25.4s | 22.8ms | 2.1 / 24.9s |
  | `ca_full_v2` | 59.3s | 32.3ms | — |
  | `no_move` | 114.3s | 30.2ms | — |

  - `ca_full_tm_s4` 对 `no_move`：Code TTFT −72%，Conv TPOT −22.5%；
  - `ca_full_v2` 对 `no_move`：Conv TPOT +7.3%，P(>0)=0.96。
- **判定**：
  - A 在新 seed 上成立：迁移策略的 Code TTFT 低于 `no_move`；
  - Conv 相对 `no_move` 退化 ≤20%；
  - `ca_full_tm_s4` 的 Conv SLO ≥0.95，Conv TTFT p99 ≤6s；
  - 正确性同前。

## 10. 算法优化轮 v3.1：按需切片（V5）

### 10.1 依据

- 周期切片的开销正比于切片次数（每次让 Code 多花 2.3–3.3s）。但真正需要窗口的是还没出首 token 的新 Conv：它们决定 Conv TTFT 和 SLO。已在运行的 Conv 停多久，只通过停顿总时长影响平均 TPOT。
- simulator 对比（seed 702 / 704，4 个副本）：按需切片（T_wait=2s，S_max=15s）与周期 S=4 的 Conv SLO 同为 1.00，但 Code TTFT 低 2.7–3.5s（28.9 对 31.6s，28.2 对 31.7s），Conv TPOT 略低（22.75 对 22.96，23.20 对 23.33ms），持有段数 11–13 对 20。代价是已在运行的 Conv 最长停顿约 17s，周期切片约 6s。

### 10.2 v3.1 改动

- 新策略 `ca_full_tm_slo`，`slice=(T_wait=2, R=1, S_max=15)`。
- 触发条件：持有期间某个已预留但 `output_tokens == 0` 的 Conv，按 `reserved_at` 已等待 ≥ T_wait；或持有达到 S_max。
- 改为先 keep-pause Code、再检查 Code 是否仍在 prefill。Code 暂停后查询是即时的，Conv 少等一个 Code 步。若所有 Code 都已在 decode，就立即恢复 Code，取消这次切片。
- 周期切片 `ca_full_tm_s4/s5` 的行为不变。
- 测试：新增按需切片测试（触发原因全部为 `ttft`，没有重叠，输出长度精确），共 24 个通过。

### 10.3 验证作业 V5

- **trace**：seed 704 的 850s trace，与 V4 相同。
- **2 轮同机配对**：`ca_full_tm_slo`, `ca_full_tm_s4`。`--time 0:55:00`。
- **预测**：`evidence/xplace-inputs/forecast-v5-d850-s704.json`，20 个副本：
  - `ca_full_tm_slo`：Code TTFT 28.2s，Conv TPOT 23.2ms，Conv TTFT 0.44s（p99 4.0s）；
  - 对 `ca_full_tm_s4`：Code TTFT −11.3%，Conv TPOT −0.6%；
  - 对 `no_move`（V4）：Code TTFT −76%，Conv TPOT −23%。
- **判定**：`ca_full_tm_slo` 对 `ca_full_tm_s4` 的 Code TTFT 更低，Conv SLO 不低于后者 −0.02，Conv TPOT 不更差。

## 11. 公平基线作业 V6（最新代码，11-04 用户要求）

### 11.1 依据

- 之前的基线没有用最新代码重跑：
  - `no_move` 最后一次在 V4（v4 upload），且一直是 v1 控制面，没有快路径；
  - `mig_random`、`ca_admit`、`ca_return`、`no_move_fair` 只在 S4–S6 跑过（v1，seed 702，1000s）；
  - `no_move_v2` 从未运行。
- 因此"`ca_full_tm_slo` 对 `no_move`"混用了控制面版本。这主要影响 Conv TTFT，因为快路径只改变 Conv 的准入延迟；Code TTFT 和 Conv TPOT 基本不受影响。
- 用户选择 6 轮方案，消融基线也用最新代码。

### 11.2 设计

- **代码**：新 tag `v6-baselines-20261104`，包含 v3.1 和诊断（看门狗、GC 计时）。新增 `mig_random_v2`（随机迁移 + 快路径）。测试共 24 个通过。
- **trace**：seed 704 的 850s trace，与 V4/V5b 相同。
- **6 轮同机配对，按此顺序**：`no_move_v2`, `ca_full_tm_slo`, `ca_full_v2`, `no_move`, `ca_full_tm_v2`, `mig_random_v2`。新策略放在最后，出错时不影响前面的轮次。`--time 2:30:00`，探针 16.5ms。
- **预测**：`evidence/xplace-inputs/forecast-v6-d850-s704.json`，20 个副本。simulator 不含控制面，所以 `*_v2` 与对应 v1 相同：

  | 策略 | Code TTFT | Conv TPOT | Conv TTFT avg |
  |---|---|---|---|
  | `no_move_v2` | 118.1s | 30.2ms | — |
  | `ca_full_tm_slo` | 28.2s | 23.2ms | 0.4s |
  | `ca_full_v2` | 59.3s | 32.3ms | — |
  | `no_move` | 114.3s | 30.2ms | — |
  | `ca_full_tm_v2` | 25.4s | 22.8ms | 2.1s |
  | `mig_random_v2` | 60.3s | 33.3ms | — |

- **主要对比与判定**：
  - 最终策略对公平基线 `ca_full_tm_slo` → `no_move_v2`：Code TTFT 显著更低，Conv TPOT 不更差，Conv 退化 ≤20%；
  - 控制面对基线的影响 `no_move` → `no_move_v2`：Conv TTFT 预计下降，Code TTFT 和 Conv TPOT 不变；
  - placement B 在 v2 控制面下 `mig_random_v2` → `ca_full_v2`：预测 Conv TPOT −3%，干扰噪声下可能测不出，如实报告；
  - 正确性同前。

## 12. v3.2：事件循环卡顿的修复

- **证据**：V6 看门狗捕获的 5 次卡顿（0.6–1.1s），调用栈都在 `monitor()` 中同步写 `progress.json` 的 `open()` / `replace()`。输出在共享文件系统上，元数据延迟偶发尖峰，之前在 V2、V4 看到的 3.3/3.6s 很可能也是同一原因。
- **修复**：`Journal` 改为后台写线程（队列），jsonl 行与 progress 快照都在线程里落盘；trial 结束时 `close()` 等队列写完。写入错误记入 summary 的 `journal_errors`，不中断 trial。
- **验证**：本地模拟每次写盘阻塞 0.8s，事件循环最大滞后为 0.0017s，所有测试通过。真机验证可以放进下一个打包作业。
