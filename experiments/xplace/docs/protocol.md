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

## 13. kvcached 原生基线 V7（11-04 用户要求）

### 13.1 依据

- 用户要求加一个"kvcached 方式"的基线：两个 vLLM 同时推理，不加任何协调。
- 现有基线（`no_move*`）都经过控制器，与原生部署有三处不同：
  - 控制器按 P+O 全量预留后才准入，放不下就在控制器里排队；原生 vLLM 按步分配 KV，只要当前 chunk 放得下就运行，显存不够时抢占并重算；
  - 控制器按请求数路由；原生按 `waiting × 4 + running` 选 DP rank，不看显存；
  - 16 GiB 上限由控制器的准入实现；原生 kvcached 从 GPU 实际剩余显存中按需映射页，保留 `1 − KVCACHED_GPU_UTILIZATION`（作业中为 2%）的余量。
- V6 日志估计：在 4 个 Code 到达时刻，按 Conv **实际** KV 计算，Code 所在 rank 放得下的比例为 4/4、4/4、4/4、2–4/4；按 **P+O 全量预留** 计算为 4/4、1–2/4、0/4、0–1/4。
- 推论：原生方式不迁移也可能让 Code 到达即进入，迁移的 A 收益主要是相对全量预留而言；原生的代价可能体现为抢占和重算。

### 13.2 设计

- **新策略 `kvcached_native`**：
  - 请求到达即交给 vLLM（`data_parallel_rank=None`），控制器不准入、不排队、不迁移、不暂停；
  - 调度器切到原生准入（`placement_set_native`）：不做 P+O 预留，抢占为 vLLM 原生的 recompute；trial 结束后切回；
  - 同样的容量：trial 前在每张卡上占掉多余显存（两个 tenant 的 worker 各一半），使 kvcached 可映射的 KV（已映射 + 剩余 − 2% 总显存）为 85.5 个 192 MiB 页组，取整后 85 组，即 15.94 GiB，与受控预算下观测到的单卡最大映射相同；trial 后释放并核对容量恢复；
  - 原生请求的 `reserved_at` 记为交给 vLLM 的时刻，vLLM 内部的排队和抢占都计入 prefill 区间和 TTFT；
  - 记录每个请求的抢占次数、每个 tenant 的抢占总数、每卡最大映射、每 60s 的可映射容量；rank 从快照反查。
- **测试**：新增 3 个（原生提交与 rank 反查、占位精度与释放、`run_phases` 中占位包住原生轮），共 27 个通过。
- **作业 V7**：tag `v7-native-20261104`，seed 704 的 850s trace，4 轮同机配对，按此顺序：`ca_full_tm_slo`, `kvcached_native`, `ca_full_v2`, `no_move_v2`。`--time 2:00:00`，探针 16.5ms。
  - 原生轮放第二：让 100K 的 Code prefill 先在受控轮里跑过一次，分配器缓存到位后，原生轮只靠 2% 的余量容纳激活；
  - 本作业同时是 v3.2（后台写 progress）第一次上真机。

### 13.3 预测（提交前写定）

依据是 V6 中最接近的对照，即 Code 到达即进入、与 Conv 并行的 `mig_random_v2`（57.9s / 32.1ms），加上 13.1 的日志估计：

| 策略 | Code TTFT | Conv TPOT |
|---|---|---|
| `kvcached_native` | 55–70s | 31–36ms |
| `ca_full_tm_slo` | 约 28s | 约 23.6ms |
| `ca_full_v2` | 约 61s | 约 34ms |
| `no_move_v2` | 约 111s | 约 29.7ms |

- 抢占：Conv 0–20 次，集中在 660/840s 的 Code 附近；Code 被分到放不下的 rank 时可能自我抢占。
- **判定**：
  1. 本系统对 kvcached 原生部署，`kvcached_native` → `ca_full_tm_slo`：Code TTFT 降 ≥40%，Conv TPOT 降 ≥15%；
  2. 迁移对原生准入，`kvcached_native` → `ca_full_v2`：若 Code TTFT 差异在 ±15% 内，则迁移相对原生准入没有 A 收益，A 只相对全量预留成立，如实报告；
  3. 全量预留的代价，`no_move_v2` → `kvcached_native`：Code TTFT 降 30–50%，Conv TPOT 高 5–20%；
  4. 正确性：输出长度精确，抢占不改变输出长度；原生轮的容量记录全程不超过 85 个页组。

### 13.4 执行记录

- **页组大小更正**：§13.2 把页组写成 192 MiB。实际是 96 MiB：每页 2 MiB × 48 层，K、V 在同一个 buffer 内。占位目标为 170.5 个页组（15.984 GiB），实测四卡容量为 16.006/16.006/16.008/15.984 GiB，取整后都是 170 个页组，即 15.94 GiB。共享容量与 §13.2 一致，只是描述有误。
- **节点**：d1n41a18g02，探针 12.8ms，自然容量 35.6–38.3 GiB/卡。
- **判定结果**（V7 = 288709，结果见 `reports/results-short.md` §11）：
  1. 满足：`kvcached_native` → `ca_full_tm_slo`，Code TTFT −48.7% [−51.4, −46.4]，Conv TPOT −24.6% [−26.4, −22.7]；
  2. 落在"无 A 收益"一侧：`kvcached_native` → `ca_full_v2`，Code TTFT +0.5% [−2.9, +4.9]。迁移的 A 收益只相对全量预留成立；
  3. 方向成立：`no_move_v2` → `kvcached_native`，Code TTFT −59.9%（大于预测），Conv TPOT +12.8%（在预测内）；
  4. 满足：244/244 输出长度精确，23 次迁移全部通过，原生轮没有抢占，最大映射没有超过 170 个页组。
  - 预测偏差：原生的 Code TTFT 54.1s，略低于预测区间 55–70s；Conv 抢占预测 0–20 次，实际 0 次。

## 14. 当前长度准入：重跑全部策略（V8，11-04 用户要求）

### 14.1 依据

- 用户要求：
  - serving 系统不能假设已知输出长度，准入按当前长度；
  - 实验负载要设计成按当前长度长请求也进不来；
  - KV 不够时触发 vLLM 的 retract（抢占重算）；
  - placement 给请求留增长空间，但不假设已知完整输出长度；
  - 新设定下重跑所有 baseline 和我们的系统。
- 之前所有受控策略（V1–V7）都按 prompt + max_tokens 预留，而 max_tokens 是回放的真实输出长度，相当于 oracle。planner 的剩余长度和挑选顺序也用了它。这些结果只作历史参考。V7 的 `kvcached_native` 不依赖输出长度。

### 14.2 v4 代码改动（"当前长度"设定）

- **准入**：调度器 `prompt` 模式在 add_request 时只预留 prompt + 256 token（固定 extra，不随请求变化），之后每步按需增长。KV 不够时由 vLLM 原生抢占（FCFS，抢占最新请求，重算）。新增抢占次数和重算 token 计数。
- **原生基线** `kvcached_native` 不变：vLLM 默认 `scheduler_reserve_full_isl=True`，即当前序列放得下才开始。
- **planner 不再使用输出长度**：
  - 剩余长度视为未知，所有请求在规划窗口内都计为仍在运行；
  - 挑选顺序改为 `kv_desc` / `kv_asc`，回迁按 context；
  - 到达请求只用 prompt。
- **增长余量**：每个 Conv 的余量为 `conv_growth_tokens × 每 token KV`，取以下两项中的较大者：
  - 单独运行速率下 `relief_horizon_s`=15s 的增长；
  - 长请求驻留期间的增长。驻留时间 = 校准的 prefill 时间 + `decode_window_s`=10s，期间的速率在并行时取干扰下的速率，按需切片时只计切片窗口，全程分时时为 0。

  每个 rank 再加 1 个页组的 slack。余量作用于 Code 的直接准入目标、迁移准入目标、所有迁移目的地和回迁目的地；迁移完成后，准入时再按新快照和余量复查一次。
- **事务**：允许存在等待中或被抢占的请求，只要求被迁移的请求处于 decode。
- **容量**：每一轮都用占位把 kvcached 可映射的共享 KV 对齐到 16 GiB（170 个 96 MiB 页组），轮后释放。控制器预算只作观察，不再强制。
- **新增策略**：`no_move_fair_v2`、`ca_admit_v2`、`ca_return_v2`、`mig_random_tm_v2`，都使用 v2 控制面，用来补齐总表。
- **测试**：35 个通过。新增测试覆盖：
  - planner 不读输出长度；
  - 余量约束；
  - 按当前长度准入时 Code 直接进入，而旧的全量预留需要迁移；
  - 原生和 `prompt` 模式在容量不足时都会触发 retract，且输出长度精确；
  - 有被抢占请求时迁移事务照常完成；
  - 每一轮都会加上占位。
- 计划参数冻结为 `planner-params-v2.json`（SHA `00bcf6f7…`），即 v1 加上 `relief_horizon_s` 15、`decode_window_s` 10、`slack_pages` 1。

### 14.3 simulator 与验证

- 新增当前长度准入、原生按请求数路由（`waiting × 4 + running`）、按页的 retract（抢占该租户在该 rank 上最新的请求）和重算（每 4096 token 一个 chunk，`recompute_s_per_token` 1.25e-4）。
- 旧模式下 5 个策略的输出与改写前逐位一致。
- 原生模式在 V7 trace 上：
  - Code 到达时各 rank 的 KV 与实测相差约 0.5 GiB 以内（660s：[4.41, 4.41, 6.47, 5.44] 对实测 [4.31, 4.59, 6.19, 5.91]）；
  - Code 落点相同，均为 0 次抢占；
  - Conv TPOT 32.8 对实测 31.3ms；
  - Code TTFT 59–60s 对实测 48–60s。

### 14.4 负载选择（只用非最终 seed 801–820）

- Conv 0.07 RPS（V7 负载）时，按当前长度 Code 总能直接放进，迁移没有作用。
- 被挡比例受两方面限制：
  - 每个 rank 的 Conv KV 刚好在门槛附近，一个 Conv 结束就能腾出空间；
  - 受控路由会在 Code 驻留的 rank 上留下空洞。
- 被否决的候选：
  - 14 GiB 上限或每轮更多 Code：不分时的策略整体过载，Code 并行拖慢 Conv 后积压，出现大量抢占；
  - Code 晚于 540s 开始：每轮超过 27 分钟。
- **选定**：16 GiB 上限，Conv 0.12 RPS，Code 在 420/540/660/780s，时长 900s，评估窗口 [420, 900)。seed 801–810 上：
  - 原生被挡比例 0.40（被挡指 Code TTFT > 70s）；
  - 原生 Code TTFT 180s，抢占 1.6 次；
  - `ca_full_v2` 60s、0 次抢占；
  - 分时策略 26–30s、0 次抢占；
  - 墙钟 ≤ 1479s。
- **seed 规则**（只依据 trace，不看任何策略结果）：从 901 起，取第一个同时满足以下两条的 seed：
  - Conv 总数在 108 的 ±5% 以内；
  - 各 Code 到达前 400s 内到达的 Conv 数之和 ≥ 0.12 × 400 × 4 = 192。

  第二条的依据：seed 801–820 上，它与原生被挡比例的相关系数为 0.68。它会把选择偏向"Code 到达时负载较高"的 trace，正是本次要测的场景，结果报告中要注明。
- **结果**：seed 911，105 个 Conv，前置 Conv 211。冻结为 `cur-b16-c012-p120-s911-d900.json`（SHA `ec488df7…`）。

### 14.5 作业 V8

- tag `v8-current-20261104`，`XPLACE_KV_MODE=prompt`，`XPLACE_RESERVE_EXTRA_TOKENS=256`，上限 16 GiB，探针 16.5ms，drain 超时 1000s，排除 d1n41a23g03。
- 12 轮按重要性排序，两两组成一个不超过 1h 的实验：
  1. `ca_full_tm_slo`, `kvcached_native`；
  2. `no_move_v2`, `ca_full_v2`；
  3. `ca_full_tm_v2`, `mig_random_v2`；
  4. `no_move_tm_v2`, `mig_random_tm_v2`；
  5. `ca_admit_v2`, `ca_return_v2`；
  6. `ca_full_tm_s4`, `no_move_fair_v2`。
- 模拟墙钟合计 4.3h，`--time 5:30:00`。

### 14.6 预测（提交前写定，`forecast-v8-cur-s911.json`，每策略 20 个副本）

| 策略 | Code TTFT | Conv TPOT | Conv TTFT | 抢占 | 迁移 |
|---|---|---|---|---|---|
| `ca_full_tm_slo` | 31.1s | 29.2ms | 0.8s | 0 | 11 |
| `kvcached_native` | 166 [120, 195]s | 39.0ms | 0.2s | 1.9 | 0 |
| `no_move_v2` | 168 [120, 207]s | 34.6ms | 0.2s | 0.1 | 0 |
| `ca_full_v2` | 60.7s | 38.9ms | 0.2s | 0 | 18 |
| `ca_full_tm_v2` | 25.7s | 28.9ms | 3.2s | 0 | 20 |
| `mig_random_v2` | 60.2s | 40.0ms | 0.2s | 0 | 4 |
| `no_move_tm_v2` | 57.4s | 28.4ms | 3.1s | 0 | 0 |
| `mig_random_tm_v2` | 25.6s | 29.4ms | 3.2s | 0 | 5 |
| `ca_admit_v2` | 62.2s | 40.2ms | 0.2s | 0 | 5 |
| `ca_return_v2` | 60.9s | 39.1ms | 0.3s | 0 | 19 |
| `ca_full_tm_s4` | 34.1s | 29.3ms | 0.7s | 0 | 12 |
| `no_move_fair_v2` | 130s | 35.1ms | 20.3s | 0 | 0 |

主要对比的预测：
- `kvcached_native` → `ca_full_tm_slo`：Code TTFT −79.0% [−82.9, −73.9]，Conv TPOT −24.9% [−30.5, −16.9]；
- `kvcached_native` → `ca_full_v2`：Code TTFT −59.0% [−71.2, −47.5]，Conv TPOT +0.2%；
- `no_move_v2` → `mig_random_v2`：Code TTFT −62.4%，Conv TPOT +15.8%；
- `mig_random_v2` → `ca_full_v2`：Conv TPOT −2.9% [−8.3, +1.5]，Code TTFT +1.1%，预计测不出；
- `ca_full_v2` → `ca_full_tm_slo`：Code TTFT −48.6%，Conv TPOT −25.0%。

### 14.7 判定

1. **问题存在**：`kvcached_native` 至少 1 个 Code 的 TTFT > 70s，即按当前长度放不进。
2. **A 相对原生成立**：`kvcached_native` → `ca_full_v2`，Code TTFT 显著更低。
3. **整体**：`kvcached_native` → `ca_full_tm_slo`，Code TTFT 降 ≥ 60%，Conv TPOT 降 ≥ 15%，Conv SLO ≥ 0.95。
4. **retract**：迁移和分时策略的抢占次数不多于 `kvcached_native`。
5. **Conv 退化 ≤ 20%**：所有迁移和分时策略的 Conv TPOT 相对 `kvcached_native` 和 `no_move_v2` 都不超过 +20%。
6. **正确性**：输出长度精确；所有迁移逐字节校验通过；抢占重算后输出长度依然精确。
- placement B（`mig_random_v2` → `ca_full_v2`）预计在噪声内，如实报告。

### 14.8 增长余量消融（V8b，排在 V8 之后）

- **依据**（simulator，非最终 seed 801–810，各 10 个 seed）：

  | 余量设置 | `ca_full_tm_slo` 抢占 均值 / 最大 | `ca_full_v2` 抢占 | `mig_random_v2` 抢占 |
  |---|---|---|---|
  | 不留 | 1.7 / 17 | 0.3 | 2.9 |
  | v2（15s、10s、1 页组） | 0 / 0 | 0 | 2.4 |
  | 加倍 | 0 / 0 | 0.5 | 2.9 |

  Code TTFT 在三档之间相差不超过 0.5s。不留余量时，迁移把 Conv 塞进目的 rank，这些 Conv 恢复生成后很快用完空间；随机迁移不检查余量。
- **新增策略**：`ca_full_v2_nh`、`ca_full_tm_slo_nh`，与原策略相同，只把增长余量和 slack 都置零（`no_headroom`）。测试 36 个通过（新增：`_nh` 准入时余量为 0，默认策略大于 0）。
- **seed 911 上的预测**（`forecast-v8b-headroom-s911.json`，每个策略 20 个副本）：
  - `ca_full_v2_nh`：抢占 23.4 [3.9, 39.4] 次，`ca_full_v2` 为 0；Code TTFT 60.6 对 61.2s，Conv TPOT 38.8 对 39.1ms。
  - `ca_full_tm_slo_nh`：抢占 0 次，与 `ca_full_tm_slo` 相同（这个 trace 没有触发挤爆）；Code TTFT 29.8 对 31.1s。
- **作业 V8b**：tag 与 V8 相同（代码变化后需重新 build，tag 为 `v8b-headroom-20261104`），2 轮：`ca_full_v2_nh`、`ca_full_tm_slo_nh`，`--time 1:00:00`，依赖 V8（afterany）。与 V8 中的 `ca_full_v2`、`ca_full_tm_slo` 做跨作业比较，需注明是否在同一节点、探针读数是否相近。
- **判定**：
  - `ca_full_v2_nh` 的抢占次数多于 `ca_full_v2`；
  - Code TTFT 差距在 ±10% 以内（余量不应明显拖慢 Code 准入）；
  - 分时这一对如实报告，预计无差别。

## 15. 页共享下的修正与重跑（V9）

### 15.1 依据（V8 第一轮实测）

- V8 第一轮 `ca_full_tm_slo`（`max_moves`=8）：660s、780s 两个 Code 的迁移规划全部不可行（172 次），等 375s、286s 后才直接准入，TTFT 401s、313s；0 次迁移准入，0 次抢占。
- **原因是页共享**：
  - 按当前长度逐步增长时，kvcached 给单个块挑"最满但有空位"的页，同一 rank 上的 Conv 块交错在同一批 2 MiB 页里；
  - Conv 结束留下的空位会被其他 Conv 的增长填上，页很少整页空出，Conv 映射停在历史高点；
  - Code 属于另一个模型，用不了 Conv 页里的空位；
  - 只迁走部分 Conv 释放不了页，必须清空整张 rank，而 `max_moves`=8 小于每 rank 的 Conv 数（13–14）。
- **佐证**：V7 原生轮 rank 2 的 Conv 从 8 降到 2，映射一直是 7.41 GiB；kvcached 对全空的页立即释放（`kv_cache_manager.free`），排除了延迟释放。
- 旧设定下请求在准入时一次性预留全部长度，基本独占自己的页，所以 V1–V7 的迁移可行；simulator 一直按独占页建模。

### 15.2 改动（v4.1）

- **simulator 页共享模型**：Conv 的物理页按每个 rank 的历史高点计算，由该 rank 上所有持有块的 Conv 共有，只有 Conv 全部离开时才归零；Code 的大块分配仍按独占页；planner 看到同样的页归属。
  - 旧模式 5 个策略的输出逐位不变；
  - 重现 V8 第一轮：660s 各 rank [6.66, 7.59, 8.72, 9.75] GiB，实测 [6.75, 7.69, 7.97, 9.47]；`max_moves`=8 时 Code 也被长时间挡住。
- **允许清空整张 rank**：`planner-params-v3.json`（SHA `32a5a49e…`），`max_moves` 24，其余同 v2。挑选逻辑不变：仍取能腾出空间的最短前缀，在页共享下通常就是该 rank 上的全部可迁 Conv。
- **有 Code 驻留时不回迁**（当前长度设定）：
  - 原因：清空一张 rank 后 Conv 分布严重不均，回迁检查会在 Code prefill 期间发起回迁，迁移与 prefill 争带宽（实测每次约 3.6s），事务持锁约 25s，挡住新 Conv 的准入；
  - Code 完成时那次回迁照常；
  - 模拟中 `ca_full_tm_slo` 的 Conv TTFT p99 因此从 31s 降到 4.2s。
- 测试 37 个通过（新增：有 Code 驻留时不发起回迁；`_nh` 消融的余量为 0）。

### 15.3 作业 V9

- tag `v9-evacuate-20261104`，代码 v4.1；trace 仍为 seed 911（`cur-b16-c012-p120-s911-d900.json`）；`planner-params-v3.json`；`XPLACE_KV_MODE=prompt`，extra 256；上限 16 GiB；探针 16.5ms；drain 超时 1000s；排除 d1n41a23g03。
- 14 轮，按重要性排序：
  1. `ca_full_tm_slo`, `kvcached_native`
  2. `no_move_v2`, `ca_full_v2`
  3. `ca_full_tm_v2`, `mig_random_v2`
  4. `no_move_tm_v2`, `mig_random_tm_v2`
  5. `ca_admit_v2`, `ca_return_v2`
  6. `ca_full_v2_nh`, `ca_full_tm_slo_nh`
  7. `ca_full_tm_s4`, `no_move_fair_v2`
- 模拟墙钟合计 5.0h，`--time 6:30:00`。依赖 V8（`afterany`）：V8 跑完原生、`no_move_v2` 两轮后撤销，其余轮次因页共享问题已无效。
- V8 的原生、`no_move_v2` 两轮作为跨作业重复，用于估计轮次间波动。

### 15.4 预测（`forecast-v9-cur-s911.json`，SHA `2d04059d…`，每策略 20 个副本，页共享模型）

| 策略 | Code TTFT | Conv TPOT | Conv TTFT | 抢占 | 迁移 |
|---|---|---|---|---|---|
| `ca_full_tm_slo` | 30.3s | 29.3ms | 0.7s | 0 | 55 |
| `kvcached_native` | 474 [442, 494]s | 27.9ms | 0.1s | 0.9 | 0 |
| `no_move_v2` | 280 [270, 288]s | 29.9ms | 0.2s | 0 | 0 |
| `ca_full_v2` | 60.6s | 39.2ms | 0.2s | 0 | 42 |
| `ca_full_tm_v2` | 26.1s | 29.0ms | 3.3s | 0 | 54 |
| `mig_random_v2` | 60.2s | 41.8ms | 0.2s | 3.6 | 15 |
| `no_move_tm_v2` | 233s | 25.7ms | 2.4s | 0 | 0 |
| `mig_random_tm_v2` | 25.9s | 30.6ms | 3.3s | 0 | 13 |
| `ca_admit_v2` | 61.8s | 41.9ms | 0.2s | 0.1 | 15 |
| `ca_return_v2` | 61.0s | 39.4ms | 0.2s | 0 | 42 |
| `ca_full_tm_s4` | 34.0s | 29.5ms | 0.7s | 0 | 46 |
| `no_move_fair_v2` | 219s | 29.9ms | 72s | 0 | 0 |
| `ca_full_v2_nh` | 61.8s | 39.6ms | 0.2s | 0.1 | 42 |
| `ca_full_tm_slo_nh` | 30.3s | 29.3ms | 0.7s | 0 | 55 |

主要对比：
- `kvcached_native` → `ca_full_tm_slo`：Code TTFT −93.5% [−93.9, −93.1]，Conv TPOT +5.3% [+0.7, +13.0]。原生让大部分 Code 排队约 10 分钟，Conv 较少受干扰。
- `kvcached_native` → `ca_full_v2`：Code TTFT −87.1%，Conv TPOT +40.8%（不分时，超出 20% 的退化上限）。
- `no_move_v2` → `ca_full_tm_slo`：Code TTFT −89.2%，Conv TPOT −1.9%。
- `mig_random_v2` → `ca_full_v2`：Conv TPOT −6.2% [−11.0, −1.1]，Code TTFT +0.9%。其中 `ca_return_v2` −5.7%，`ca_admit_v2` +0.3%。
- `ca_full_v2_nh` → `ca_full_v2`：差别在噪声内（页共享下 Conv 增长先填空洞，余量作用小），抢占 0.1 对 0。
- `no_move_v2` → `no_move_tm_v2`：Code TTFT −16.7%，Conv TPOT −14.0%。

### 15.5 判定

1. **问题存在**：`kvcached_native` 至少 2 个 Code 的 TTFT > 70s。
2. **清空整张 rank 有效**：所有迁移策略的 660s、780s 两个 Code 走迁移准入。不分时的 Code TTFT ≤ 70s，分时的 ≤ 40s。
3. **整体**：
   - `kvcached_native` → `ca_full_tm_slo`：Code TTFT 降 ≥ 60%，Conv TPOT 退化 ≤ 20%，Conv SLO ≥ 0.95；
   - `no_move_v2` → `ca_full_tm_slo`：Code TTFT 降 ≥ 60%，Conv TPOT 退化 ≤ 20%。
4. **retract**：`ca_full_tm_slo` 的抢占次数不多于 `kvcached_native`。
5. **正确性**：输出长度精确；迁移全部逐字节校验通过。
- placement B（`mig_random_v2` → `ca_full_v2`）与余量消融如实报告。

### 15.6 补充（V8 原生轮失败后，提交 V9 前写定）

- **V8 原生轮**（`kvcached_native`，seed 911）在 1900s（trace 900s + drain 1000s）内没跑完，抛 TimeoutError，作业失败。到 1900s：
  - 只完成 38/109 个请求；
  - 660s、780s 的两个 Code 一直没开始；
  - Conv 约 350ms/token。
- **原因**（vLLM `DPEngineCoreProc.run_busy_loop`）：
  - 本地有未完成请求（包括 KV 不足而排队的请求）时，引擎每轮执行 dummy batch；只要有一个 rank 还有排队请求，DP wave 就不停；
  - 被 KV 挡住的 Code 让 Code 实例的 4 个 rank 一直空跑 EP 前向，干扰 Conv；Conv 变慢后 KV 不释放，Code 更进不来，形成正反馈；
  - 受控策略在控制器侧排队，不受影响。
- **代码 v4.2**：
  - 截止时截断：记录部分进度，没出首 token 的请求 TTFT 记为下界，abort 后继续下一轮，不再让整个作业失败；
  - `report_main` 读入截断记录（作为下界参与配对，会低估原生的 Code TTFT，结论偏保守）；
  - simulator 增加 DP 空转干扰。旧模式逐位不变；重现 V8 原生轮：1900s 时完成 33/109，2 个 Code 未开始。
  - 测试 39 个通过。
- **V9 轮次重排**：原生放在最后，在 1900s 截断。
  1. `ca_full_tm_slo`, `no_move_v2`
  2. `ca_full_v2`, `mig_random_v2`
  3. `ca_full_tm_v2`, `no_move_tm_v2`
  4. `mig_random_tm_v2`, `ca_admit_v2`
  5. `ca_return_v2`, `ca_full_v2_nh`
  6. `ca_full_tm_slo_nh`, `ca_full_tm_s4`
  7. `no_move_fair_v2`, `kvcached_native`

  tag `v9b-censor-20261104`，`--time 6:30:00`，其余设置同 §15.3。
- **预测**（`forecast-v9b-cur-s911.json`）：受控策略与 §15.4 相同。原生方式不截断时需 5748s 跑完，Code TTFT 2527s。在 1900s 截断时预计：
  - 完成约 33/109；
  - 2 个 Code 未开始，TTFT 下界约 1240s、1120s；
  - 4 个 Code 的截断 TTFT 均值不低于约 620s；
  - 停滞期间 Conv 约 280ms/token。
- **判定**：沿用 §15.5。判定 1 改为：`kvcached_native` 至少 2 个 Code 的 TTFT（含下界）> 70s。原生方式的配对对比使用截断下界。

### 15.7 补充：迁移规划被物理空闲块检查挡住（V9c 第一轮，提交 V10 前写定）

- **V9c 第一轮**（`ca_full_tm_slo`，`max_moves` 24，d1n41a22g02）：660s 的 Code 仍有 49 次规划不可行，0 次迁移。
- **根因**：planner 的 `_select_for_target` 在选任何要迁的请求之前，先检查 Code 池的 `free_blocks` 是否不少于 Code 所需块数，不满足就直接返回不可行；`plan_usage` 对到达请求也是同样的检查。
  - kvcached 的 `get_num_free_blocks()` 等于 `available_size()`，即物理空闲（`cudaMemGetInfo` − 2% 余量）加池内空位；
  - 没有占位时物理空闲约 35 GiB，这个检查从不起作用；占位后它成为真正的约束，而它没有计入迁走 Conv 后释放的页；
  - 655s 时的 rank 0：已用 6.75 GiB，空闲约 6,300 块 + 63 个空位，Code 需要 6,371 块，所以在任何迁移之前就被判定不可行；
  - V8 第一轮的不可行，首先也是这个原因。页共享（§15.1）是另一个真实存在、叠加在上面的约束，`max_moves` 24 仍然需要。
  - simulator 的 `free_blocks` 固定为 10^7，所以没暴露。
- **改动 v4.3**：
  - planner：Code 能否放进，用"当前空闲块 + 计划迁出后释放的块"判断；不够时继续增加要迁的请求，而不是直接返回不可行；
  - 不可行时返回每个目标 rank 的诊断（失败阶段、可迁数、所需块和空闲块、全部迁出后可释放的量），写进 decisions；
  - 按步增长的各轮，控制器预算设为占位后的实际可映射容量（170 个页组，15.9375 GiB），不再用 16.0 GiB；
  - simulator：按当前长度准入时报告真实的 `free_blocks`（空位 + 物理空闲页），预算同样对齐；
  - fake 引擎同样报告物理空闲块；
  - 测试 41 个通过（新增：空闲块不够但迁出后够用时必须能规划出迁移；不可行时有诊断）。
  - 旧模式 5 个策略逐位一致。修复前的 planner 加上真实的 `free_blocks`，在 seed 911 上重现了真机：0 次迁移准入，660s、780s 的 Code TTFT 486s、395s（V8 实测 401s、313s）。
- **作业 V10**：tag `v10-freeblocks-20261105`，14 轮顺序与 §15.6 相同（原生最后，1900s 截断），`--time 6:30:00`，其余设置同 §15.3。
- **预测**：`forecast-v10-cur-s911.json`（SHA 前缀见 progress.md），与 §15.4/§15.6 相同，判定同 §15.5/§15.6。
- V9c 的第一轮（`ca_full_tm_slo`，迁移不可行）与 V8 第一轮一起，作为"规划不能腾出空间"情形的两次跨节点重复；第二轮 `no_move_v2` 作为基线的跨作业重复。

## 16. 复现 seed（V11，排在 V10 之后）

- **目的**：总表的核心对比在第二个 trace 上复现，判断 seed 911 上的结论是否依赖具体 trace。
- **seed**：沿用 §14.4 的规则（只看 trace），从 912 起，取第一个满足条件的，结果为 914（113 个 Conv，前置 Conv 218）。冻结为 `cur-b16-c012-p120-s914-d900.json`（SHA `8a777986…`）。seed 914 上没有调过任何参数。
- **代码与参数**：与 V10 相同（v4.3，`planner-params-v3`），只换 trace。tag `v11-replicate-20261105`。
- **8 轮**：
  1. `ca_full_tm_slo`, `no_move_v2`
  2. `ca_full_v2`, `mig_random_v2`
  3. `ca_full_tm_v2`, `no_move_tm_v2`
  4. `mig_random_tm_v2`, `kvcached_native`（原生最后，1900s 截断）

  `--time 4:00:00`（模拟墙钟 3.1h），依赖 V10（`afterany`）。
- **预测**（`forecast-v11-cur-s914.json`，SHA `7278947c…`，每策略 20 个副本）：

  | 策略 | Code TTFT | Conv TPOT | 抢占 |
  |---|---|---|---|
  | `ca_full_tm_slo` | 29.7s | 30.2ms | 0 |
  | `no_move_v2` | 469s | 27.3ms | 0 |
  | `ca_full_v2` | 61.4s | 39.4ms | 0 |
  | `mig_random_v2` | 60.2s | 41.1ms | 5.8 |
  | `ca_full_tm_v2` | 26.1s | 29.7ms | 0 |
  | `no_move_tm_v2` | 376s | 26.1ms | 0 |
  | `mig_random_tm_v2` | 25.8s | 32.9ms | 4.0 |
  | `kvcached_native` | 不截断需 6000s；1900s 截断时 3 个 Code 未开始 | 停滞期约 290ms | 6.8 |

  主要对比：
  - `no_move_v2` → `ca_full_tm_slo`：Code TTFT −93.7%，Conv TPOT +10.7% [+7.6, +13.3]（在 20% 内）；
  - `mig_random_tm_v2` → `ca_full_tm_v2`：Conv TPOT −9.7%，抢占 4 → 0。分时下 placement 的 B 首次被预测为可测出，来源是随机迁移的目的地没留增长余量，引发 retract；
  - `mig_random_v2` → `ca_full_v2`：Conv TPOT −4.2% [−8.9, +0.1]；
  - `ca_full_v2` → `ca_full_tm_slo`：Code TTFT −51.5%，Conv TPOT −23.3%。
- **判定**（与 §15.5 相同，在 seed 914 上复现）：
  1. 原生至少 2 个 Code 的 TTFT（含下界）> 70s；
  2. 迁移策略的 660s 及之后的 Code 都走迁移准入，不分时 ≤ 70s，分时 ≤ 40s；
  3. `ca_full_tm_slo` 对 `no_move_v2`：Code TTFT 降 ≥ 60%，Conv TPOT 退化 ≤ 20%；
  4. 正确性同前。
- 与原生的配对对比使用截断下界（原生停滞期的 Conv TPOT 只计到截止时间，比完整运行乐观）。

## 17. seed 914 补齐总表（V12，排在 V11 之后，提交前写定）

### 17.1 依据

- **V11 前 4 轮**（seed 914）与 V10 一致。`mig_random_v2` 在排空阶段（980–1241s）出现 10 次抢占，`ca_full_v2` 为 0 次。
  - 机制：随机迁移不回迁。Conv 继续增长后，最满的 rank 先后达到容量；被抢的是迁入的请求和迟到的新请求。
  - 要分开增长余量、代价感知准入和回迁各自的作用，需要 seed 914 上的 R5（`ca_admit_v2`）、R6（`ca_return_v2`）、R13（`ca_full_v2_nh`），并与同一作业中的 R4、R7 配对。
- **跨作业比较的噪声**：`no_move_v2` 在 V9c、V10 上的 Conv TPOT 相差 3.6%（不同节点），与不分时下 B 的幅度（约 4%）相当。所以每个新行的父行都放进同一作业。
- **更正**（与 §15.2 的预期不同）：真机上的准入没有一次清空整张 rank。V10 的 18 次准入在目标 rank 的 12–17 个 Conv 中迁走 2–13 个（中位 9 个）。planner 按快照的页归属计算可释放的页；simulator 用"高水位"页模型，每次都迁走目标 rank 上的全部 Conv，所以模拟的准入迁移数偏多。

### 17.2 设计

- **代码与输入**：与 V11 完全相同，复用 tag `v11-replicate-20261105` 的 upload（manifest `0d5d9bd9…`），只改 `XPLACE_POLICIES`。
- **9 轮**（按重要性，两两组成不超过 1h 的实验）：
  1. `ca_return_v2`, `mig_random_v2`（R6 对 R4）
  2. `ca_full_v2_nh`, `ca_full_v2`（R13 对 R7）
  3. `ca_admit_v2`, `ca_full_tm_slo`（R5 对 R4；R12 锚点）
  4. `ca_full_tm_slo_nh`, `ca_full_tm_s4`（R14、R11 对 R12）
  5. `no_move_fair_v2`（R3，对 V11 的 R2 做跨作业比较，效应远大于噪声）
- **重复**：R4、R7、R12 在 V11 中各跑过 1 轮，这次是第二轮，可以估计轮次间波动。
- **作业**：`--time 4:15:00`（9 轮约 3.4h），依赖 `afterany:289851`，排除 d1n41a23g03，其余同 V11。

### 17.3 预测（`forecast-v12-cur-s914.json`，SHA `58c8f407…`，每策略 20 个副本）

| 策略 | Code TTFT | Conv TPOT | Conv TTFT | 抢占 | 迁移 |
|---|---|---|---|---|---|
| `ca_return_v2` | 58.9s | 38.8ms | 0.3s | 0 | 59 |
| `mig_random_v2` | 59.8s | 40.9ms | 0.3s | 5.8 | 14 |
| `ca_full_v2_nh` | 61.3s | 39.3ms | 0.2s | 1.4 | 63 |
| `ca_full_v2` | 60.8s | 39.1ms | 0.2s | 0 | 67 |
| `ca_admit_v2` | 60.9s | 40.8ms | 0.3s | 1.9 | 20 |
| `ca_full_tm_slo` | 29.7s | 30.2ms | 0.9s | 0 | 66 |
| `ca_full_tm_slo_nh` | 29.7s | 29.8ms | 0.9s | 0 | 66 |
| `ca_full_tm_s4` | 33.3s | 30.3ms | 1.2s | 0 | 67 |
| `no_move_fair_v2` | 284s | 31.7ms | 136s | 0 | 0 |

主要对比：
- `mig_random_v2` → `ca_return_v2`：Conv TPOT −5.0% [−10.0, −0.4]，抢占 5.8 → 0；
- `mig_random_v2` → `ca_admit_v2`：Conv TPOT −0.2% [−5.5, +4.5]，抢占 5.8 → 1.9；
- `ca_full_v2_nh` → `ca_full_v2`：Conv TPOT −0.4% [−4.6, +3.1]，抢占 1.4 → 0；
- `ca_full_tm_slo_nh` → `ca_full_tm_slo`：Conv TPOT +1.3%，两者都没有抢占；
- `ca_full_tm_s4` → `ca_full_tm_slo`：Code TTFT −10.8%，Conv TPOT −0.5%；
- `no_move_v2` → `no_move_fair_v2`：Code TTFT −39.4%，Conv TTFT 0.1 → 136s。

### 17.4 判定

1. **回迁带来 B**：`mig_random_v2` → `ca_return_v2` 的 Conv TPOT 配对 CI 上界 < 0。
2. **代价感知准入约为 0**：`mig_random_v2` → `ca_admit_v2` 的 Conv TPOT 变化在 ±2% 内，或 CI 含 0。
3. **抢占**：`ca_return_v2`、`ca_full_v2` 都是 0 次；`mig_random_v2` 的抢占次数如实报告（V11 为 10 次，预测 5.8 次）。
4. **增长余量**：如实报告。若 `ca_full_v2_nh` 有抢占而 `ca_full_v2` 没有，记为"余量在 seed 914 上防止了抢占"；否则维持 V10 的负结果。
5. **切片**：`ca_full_tm_s4` → `ca_full_tm_slo` 的 Code TTFT 变化 < 0。
6. **公平门**：`no_move_fair_v2` 的 Conv TTFT 至少是 V11 `no_move_v2`（0.36s）的 10 倍。
7. **正确性**：输出长度精确，迁移全部逐字节校验通过。
- R4、R7、R12 与 V11 同名轮次的差别作为轮次间波动报告，不设判定。

### 17.5 补充：simulator 改用块级页模型（V12 开始前写定）

- **改动**：`sim/simulator.py` 新增 `PagePool`，逐页模拟 kvcached 的分配器（`KVCacheManager._alloc_impl` / `_free`）：
  - 单块的 decode 增长放进有空位的最满页；
  - 一次分配多块（准入、迁入、重算）时，先找能放下整段的最小页，否则用最空的页，重复直到分完；
  - 所有已映射页都没有空位时才映射新页，页内块全部释放后取消映射；
  - 请求增长先用空位；只有需要新页且 GPU 已满时，才抢占本租户在该 rank 上最新的请求（vLLM FCFS）。
  - 快照中的页归属与真机一致，planner 的 `released_gib` 只计入所有者全部被迁走的页。
  - 新参数 `page_model`，默认改为 `blocks`；`high_water` 为旧规则（一张 rank 上还有 Conv 时页只增不减），V8–V12 的预测都用它。`sim/predict.py` 新增 `--page-model`。
- **回归**：`--page-model high_water` 逐位重现 `forecast-v12-cur-s914.json`。
- **一致性**：每 5s 检查一次，各 rank 池内已分配块数等于该 rank 上请求持有的块数；跑完后所有池为空。
- **对真机验证**：V10 的 13 个受控轮加 V11 的前 6 轮，每轮 10 个噪声副本：

  | 指标 | 真机 | 高水位 | 块级 |
  |---|---|---|---|
  | `mig_random_v2` 的抢占（seed 911 / 914） | 0 / 10 | 4.5 / 6.2 | 0.0 / 11.5 |
  | 迁移数：`mig_random_v2`、`ca_admit_v2`、`ca_return_v2`、`mig_random_tm_v2`（911） | 9、9、48、22 | 15.0、14.6、42.9、13.0 | 9.0、9.0、48.0、22.0 |
  | 迁移数：`ca_full_tm_slo`（911 / 914） | 30 / 32 | 55 / 66 | 31 / 51 |
  | Code TTFT 绝对误差，均值 / 中位数 | | 6.8% / 2.6% | 5.2% / 1.6% |
  | Conv TPOT 绝对误差，均值 / 中位数 | | 2.2% / 1.7% | 2.1% / 1.7% |

  - 仍有的偏差：
    - 不迁移的分时策略在 seed 914 上 Code TTFT 偏高 45%（高水位 60%）。这是阈值效应：
      - 540s 时真机 rank 2 已用 5.81 GiB，加上 Code 和余量正好到上限，Code 直接进入；
      - 模拟中各 rank 的 Conv 数相同，但 rank 2 多映射 4 页（6.19 GiB），Code 被挡到 1056s；
      - 之后两个 Code 的准入时间与真机相近（1082s、1108s 对 1128s、1155s）；
    - 不分时 `ca_full_v2` 在 911 上的迁移数 32.5 对 58；
    - 不分时策略的 Conv TPOT 低估 2–3.6%。
  - V10 的原生轮在 1900s 时完成 33/109、抢占 1 次，与真机相同。
- **块级模型的 V12 预测**：`forecast-v12b-blocks-s914.json`（SHA `a8155f7f…`），作为 V12 对新模型的样本外检验；判定仍按 §17.4。

  | 策略 | Code TTFT | Conv TPOT | 抢占 | 迁移 |
  |---|---|---|---|---|
  | `ca_return_v2` | 58.8s | 39.2ms | 0.2 | 50 |
  | `mig_random_v2` | 60.0s | 42.4ms | 11.9 | 20 |
  | `ca_full_v2_nh` | 61.0s | 39.2ms | 2.4 | 46 |
  | `ca_full_v2` | 60.6s | 39.1ms | 0.1 | 55 |
  | `ca_admit_v2` | 60.8s | 41.5ms | 5.7 | 12 |
  | `ca_full_tm_slo` | 30.7s | 30.2ms | 0 | 51 |
  | `ca_full_tm_slo_nh` | 30.5s | 29.9ms | 0 | 37 |
  | `ca_full_tm_s4` | 33.7s | 30.1ms | 0 | 40 |
  | `no_move_fair_v2` | 280s | 31.3ms | 0 | 0 |

  主要对比：
  - `mig_random_v2` → `ca_return_v2`：Conv TPOT −7.6% [−12.3, −3.3]，抢占 11.9 → 0.2；
  - `mig_random_v2` → `ca_admit_v2`：−2.0% [−7.0, +2.4]，抢占 11.9 → 5.7；
  - `ca_full_v2_nh` → `ca_full_v2`：−0.2% [−4.8, +4.0]，抢占 2.4 → 0.1；
  - `ca_full_tm_s4` → `ca_full_tm_slo`：Code TTFT −8.9%。

## 18. RQ4 的真机锚点（V13，排在 V12 之后，提交前写定）

### 18.1 依据

- RQ4（论文图 10）的负载与 KV 预算曲线来自 simulator（results-short §12.7），需要在真机工作点之外的位置各有一个锚点。
- **两个锚点**：
  - **KV 充足**：seed 911 trace，KV 上限 24 GiB/卡。模拟中碎片消失，原生与受控基线的 Code TTFT 都约为 60s，本文的收益全部来自分时。这一点回答"16 GiB 上限是否是人为制造的"。
  - **更重负载**：Conv 0.14 RPS，16 GiB/卡。模拟中随机迁移抢占约 17 次，代价感知约 0.6 次；不分时的代价感知不留增长余量时为 2.5 次。这里检验"placement 要为增长留空间"。
- **不加重复轮**：规则在 V12 第六轮结果出来前写定（分时下同一策略两次运行的 Conv TPOT 差超过 3% 才加）。实测 V11 → V12 的 `ca_full_tm_slo` 为 +1.8%，所以不加。λ=0.14 的锚点本身包含分时下的随机 / 代价感知配对，作为分时下 B 的第三个独立 trace。

### 18.2 负载

- **seed 规则**：沿用 §14.4 的规则（只看 trace），按 0.14 RPS 缩放：Conv 总数在 0.14 × 900 = 126 的 ±5% 内，各 Code 到达前 400s 内的 Conv 数之和 ≥ 0.14 × 400 × 4 = 224。从 915 起，第一个满足的是 919（120 个 Conv，前置 227）。
- 冻结为 `cur-b16-c014-p120-s919-d900.json`（SHA `81c3d0c0…`）。seed 919 上没有调过任何参数。
- seed 911 的 trace 与 V10 相同（SHA `ec488df7…`）。

### 18.3 设计

- **运行时 segments**（只改了 `main.py`；planner、调度器补丁、trace runner 与 V10–V12 相同）：一个作业按顺序运行多个 {label, trace, KV 上限, 策略}，输出到 `trace/<label>/<policy>`。不给 segments 时行为不变。
- **10 轮，按重要性排序**（`segments-v13-anchors.json`，SHA `2d6b20c1…`）：
  1. `s919-l14`（λ=0.14，16 GiB）：`ca_full_tm_slo`, `no_move_v2`, `mig_random_tm_v2`, `ca_full_tm_v2`；
  2. `s911-c24`（seed 911，24 GiB）：`ca_full_tm_slo`, `no_move_v2`, `kvcached_native`；
  3. `s919-l14-corun`（λ=0.14，16 GiB）：`mig_random_v2`, `ca_full_v2`, `ca_full_v2_nh`。
  
  λ=0.14 拆成两个 segment，只是为了让不分时的三轮排到最后；分析时把两个目录合并，属于同一作业配对。
- 原生方式只在 24 GiB 下运行。λ=0.14、16 GiB 下模拟的原生 20 个副本都没在 trace + 1000s 内跑完，与 V10/V11 的截断结论相同，不再花一轮。
- **作业**：`--time 4:45:00`（模拟墙钟合计约 3.6h，加启动和每轮约 100s 的开销约 3.9h），依赖 `afterany:290422`，排除 d1n41a23g03，探针上限 16.5ms，其余同 V12。

### 18.4 预测（块级页模型，每策略 20 个副本）

- **`forecast-v13-s919-l14.json`**（SHA `c50f489e…`）：

  | 策略 | Code TTFT | Conv TPOT | Conv TTFT | 抢占 [p5, p95] | 迁移 |
  |---|---|---|---|---|---|
  | `ca_full_tm_slo` | 29.4s | 29.6ms | 0.44s | 0 | 45 |
  | `no_move_v2` | 551s | 33.9ms | 0.04s | 0 | 0 |
  | `mig_random_tm_v2` | 26.1s | 31.3ms | 2.0s | 0 | 22 |
  | `ca_full_tm_v2` | 25.9s | 29.4ms | 2.0s | 0 | 43 |
  | `mig_random_v2` | 60.7s | 42.9ms | 0.24s | 17.4 [8.9, 28.7] | 26 |
  | `ca_full_v2` | 61.1s | 40.1ms | 0.23s | 0.6 [0, 3.0] | 56 |
  | `ca_full_v2_nh` | 61.4s | 39.9ms | 0.23s | 2.5 [0, 11.1] | 45 |

  - `no_move_v2` → `ca_full_tm_slo`：Code TTFT −94.7%，Conv TPOT −12.7% [−14.4, −10.3]；
  - `mig_random_tm_v2` → `ca_full_tm_v2`：Conv TPOT −6.0%；
  - `mig_random_v2` → `ca_full_v2`：Conv TPOT −6.6% [−11.6, −2.7]，抢占 17.4 → 0.6；
  - `ca_full_v2_nh` → `ca_full_v2`：Conv TPOT +0.6% [−4.4, +5.6]，抢占 2.5 → 0.6。
- **`forecast-v13-s911-c24.json`**（SHA `9cb6f3d4…`）：

  | 策略 | Code TTFT | Conv TPOT | Conv TTFT | 抢占 | 迁移 |
  |---|---|---|---|---|---|
  | `ca_full_tm_slo` | 29.6s | 29.0ms | 0.71s | 0 | 14 |
  | `no_move_v2` | 59.6s | 39.6ms | 0.22s | 0 | 0 |
  | `kvcached_native` | 60.5s | 40.0ms | 0.21s | 0 | 0 |

  - `no_move_v2` → `ca_full_tm_slo`：Code TTFT −50.3%，Conv TPOT −26.9% [−28.8, −24.9]；
  - `kvcached_native` → `no_move_v2`：Code TTFT −1.3% [−11.2, +9.2]，Conv TPOT −0.9%；
  - `kvcached_native` → `ca_full_tm_slo`：Code TTFT −51.0%，Conv TPOT −27.5%。
- **分时策略的预测区间是退化的**（p5 = p95）：噪声模型只扰动并行时的干扰和 Code prefill，分时下 Conv 被持有，没有噪声。分时策略的轮次间波动以真机为准（V11 → V12：Conv TPOT 1.8%）。

### 18.5 判定

1. **KV 充足时没有碎片**（24 GiB）：`no_move_v2` 与 `kvcached_native` 的 4 个 Code 都在 70s 内出首 token，`kvcached_native` 全部完成、不截断。
2. **KV 充足时收益来自分时**（24 GiB）：`no_move_v2` → `ca_full_tm_slo` 的 Code TTFT 降 ≥ 40%，Conv TPOT 降 ≥ 15%。
3. **更重负载下的整体**（λ=0.14）：`no_move_v2` 至少 2 个 Code 的 TTFT > 70s；`no_move_v2` → `ca_full_tm_slo` 的 Code TTFT 降 ≥ 60%，Conv TPOT 退化 ≤ 20%。
4. **分时下的 B**（λ=0.14）：`mig_random_tm_v2` → `ca_full_tm_v2` 的 Conv TPOT 配对 CI 上界 < 0。这是继 seed 911、914 之后的第三个 trace。
5. **抢占**（λ=0.14，不分时）：`ca_full_v2` 的抢占次数少于 `mig_random_v2`。
6. **增长余量**（λ=0.14，不分时）：如实报告。若 `ca_full_v2_nh` 有抢占而 `ca_full_v2` 没有，记为"更重负载下余量防止了抢占"；否则维持负结果。
7. **正确性**：输出长度精确，迁移全部逐字节校验通过。
- 块级页模型的样本外误差（Code TTFT、Conv TPOT、迁移数、抢占）如实报告。

### 18.6 事后补充：随机策略的预测区间（V13 第八轮之后写，不改变 §18.4 与 §18.5）

- §18.4 的预测只改变干扰与 prefill 噪声，随机迁移的准入随机种子固定，所以随机策略的区间偏窄（`mig_random_v2` 的抢占预测 17.4 [8.9, 28.7]，真机 2 次）。
- `sim/predict.py --vary-admission-seed` 同时改变准入种子；用相同的策略顺序重做后，`mig_random_v2` 的抢占为 17.4 [1.0, 34.2]，分时下的 B 为 −6.8% [−10.3, −4.5]（原为退化的 −6.0%）；代价感知策略的预测不变。
- 判定仍按 §18.5，用原预注册预测计算样本外误差；这里的区间只用于解释偏差。以后对随机策略的预注册预测都加这个选项。

## 19. 最终方案的消融（V14，11-06 用户要求，提交前写定）

### 19.1 依据

- 用户 11-06："补跑最终方案的各消融实验"。
- 对最终方案 `ca_full_tm_slo` 逐个去掉组件。已有、并且都在同一作业内与最终方案配对、覆盖两个 seed 的消融：
  - 去掉分时（`ca_full_v2`）；
  - 去掉切片（`ca_full_tm_v2`）；
  - 按需切片换成周期切片（`ca_full_tm_s4`）；
  - 去掉增长余量（`ca_full_tm_slo_nh`）。
- 缺少的是在最终调度（分时 + 按需切片）下的 placement 消融，以及控制面：
  - **去掉迁移** `no_move_tm_slo`：不迁移、不回迁；
  - **随机 placement** `mig_random_tm_slo`：随机迁移，不回迁；
  - **去掉回迁** `ca_admit_tm_slo`：只做代价感知准入；
  - **去掉代价感知准入** `ca_return_tm_slo`：随机准入 + 代价感知回迁；
  - **v1 控制面** `ca_full_tm_slo_v1`：Conv 准入实时查询 Code 实例，Code 先准入、Conv 后停到边界。这也补上了"当前长度设定下控制面 v1 的对照"。
- 这 5 个变体与最终方案只差一个组件，调度（分时 + 按需切片）不变。

### 19.2 设计

- **代码**：只在 `trace_runner.POLICIES` 与 `simulator.POLICIES` 中新增这 5 个策略，planner、调度器补丁与 V10–V13 相同。新增 fake 引擎测试：5 个策略都能端到端跑完，输出长度精确；有 `hold_first` 的策略没有并行步。
- **12 轮**，4 个 segment（`segments-v14-ablation.json`，SHA `c53a53d1…`），重要的放前面：
  1. `s911-a`：`ca_full_tm_slo`、`mig_random_tm_slo`、`no_move_tm_slo`；
  2. `s914-a`：同上；
  3. `s911-b`：`ca_return_tm_slo`、`ca_admit_tm_slo`、`ca_full_tm_slo_v1`；
  4. `s914-b`：同上。
  
  同一 seed 的两个 segment 合并分析，都与本作业中的 `ca_full_tm_slo` 配对。
- **trace**：seed 911、914，与 V10–V12 相同。
- **其余设置**同 V13：16 GiB、drain 1000s、探针上限 16.5ms、当前长度准入（prompt + 256）。`--time 5:30:00`（模拟墙钟约 4.3h），排除 d1n41a23g03。

### 19.3 预测（块级页模型，每策略 20 个副本，`--vary-admission-seed`）

文件：`forecast-v14-ablation-s911.json`（SHA `029b98a0…`），`forecast-v14-ablation-s914.json`（SHA `9f4ab8d1…`）。

| 策略 | seed 911：Code TTFT / Conv TPOT / 抢占 | seed 914：Code TTFT / Conv TPOT / 抢占 |
|---|---|---|
| `ca_full_tm_slo` | 29.9s / 29.3ms / 0 | 30.7s / 30.2ms / 0 |
| `mig_random_tm_slo` | 29.8s / 30.6ms / 0 | 29.9s / 32.5ms / 1.6 |
| `no_move_tm_slo` | 199.4s / 26.3ms / 0 | 345.0s / 26.8ms / 0 |
| `ca_return_tm_slo` | 29.8s / 29.4ms / 0 | 30.0s / 30.2ms / 0 |
| `ca_admit_tm_slo` | 29.9s / 30.2ms / 0 | 30.7s / 31.8ms / 0 |
| `ca_full_tm_slo_v1` | 31.3s / 29.5ms / 0 | 31.2s / 30.2ms / 0 |

最终方案相对各变体（变体 → `ca_full_tm_slo`）：

| 变体 | seed 911：Code TTFT / Conv TPOT | seed 914：Code TTFT / Conv TPOT |
|---|---|---|
| 去掉迁移 | −85.0% / +11.2% | −91.1% / +12.6% |
| 随机 placement | +0.6% / −4.3% [−6.3, −1.7] | +2.6% / −7.0% [−10.7, −5.0] |
| 去掉回迁 | +0.1% / −3.0% | −0.2% / −4.8% |
| 去掉代价感知准入 | +0.6% / −0.4% [−1.1, −0.1] | +2.4% / +0.2% [−0.5, +0.9] |
| v1 控制面 | −4.4% [−6.6, −2.0] / −0.6% | −1.7% [−3.7, −0.4] / +0.1% |

- 去掉迁移时 Conv TPOT 更低（预测 +11~13%），因为不迁移时 Code 被挡、与 Conv 共存的时间短；这是"让 Code 进得来"的代价，在分时下仍在 20% 内（V10/V11 的 `no_move_v2` → `ca_full_tm_slo` 已测过）。
- 分时策略与代价感知策略的预测区间是退化的（模型中没有并行干扰噪声，也不用随机数）；轮次间波动以真机为准（同一分时策略两次运行差 1.8%）。
- simulator 不模拟 Conv 准入快路径，v1 控制面的预测只反映 `hold_first`。

### 19.4 判定（每个 seed 分别，与本作业中的 `ca_full_tm_slo` 配对）

1. **迁移**：`no_move_tm_slo` → `ca_full_tm_slo` 的 Code TTFT 降 ≥ 60%。
2. **代价感知 placement**：`mig_random_tm_slo` → `ca_full_tm_slo` 的 Conv TPOT 配对 CI 上界 < 0。
3. **回迁**：`ca_admit_tm_slo` → `ca_full_tm_slo` 的 Conv TPOT 配对 CI 上界 < 0。
4. **代价感知准入**（在回迁之上）：如实报告，预计接近 0。
5. **控制面**：如实报告 Code TTFT 与 Conv TTFT 的变化，预计 v1 的 Code TTFT 高 2–4%。
6. **抢占**：`ca_full_tm_slo` 0 次；其他变体的抢占如实报告。
7. **正确性**：输出长度精确，迁移全部逐字节校验通过。
- `ca_full_tm_slo` 与 V10（seed 911）、V11/V12（seed 914）的同名轮次的差别作为轮次间波动报告，不设判定。

### 19.5 结果（V14 = 292505，作业结束后写，不改变 §19.3 与 §19.4）

- 12 轮全部 passed；节点 d1n41a28g02，探针 12.6ms，占位后容量 15.98 GiB/卡。归档 `evidence/xplace-v14-ablation-20261106/job-292505.tar.gz`（SHA `e86d24b6…`）。详见 results-short §12.10。
- 判定（seed 911 / seed 914，变体 → `ca_full_tm_slo`）：
  1. 成立：Code TTFT −83.9% [−91.5, −6.8] / −94.8% [−95.6, −93.5]。
  2. 成立：Conv TPOT −10.2% [−11.1, −9.3] / −10.8% [−11.8, −9.9]。
  3. 成立：Conv TPOT −3.5% [−3.7, −3.2] / −6.4% [−7.1, −5.9]。
  4. Conv TPOT −3.3% [−3.5, −3.1] / −1.4% [−1.6, −1.2]，比预测（≈0）大：配合回迁时代价感知准入复用上一个 Code 腾空的 rank，第二次迁移准入少迁约 10 个 Conv。
  5. Code TTFT +1.3% [−0.3, +3.0] / +0.9% [−2.2, +4.4]，预测（v1 高 2–4%）不成立；Conv TTFT +0.9%（不显著）/ −26.8% [−40.4, −6.5]，seed 914 上 v1 的 Conv SLO 0.91（v2 1.00），来自 Code 准入进行中到达的 Conv 被挡 4.6–5.1s。
  6. 成立：最终方案两个 seed 都 0 次抢占；`mig_random_tm_slo` 在 seed 914 上 10 次（预测 1.6 [0, 6]）。
  7. 成立：1356 个请求输出长度精确，310 次迁移全部逐字节校验通过。
- 轮次间：最终方案对 V10 为 Code TTFT −0.4%、Conv TPOT −2.5%；对 V11、V12 为 0.0% / +1.4%、−0.3% / −0.3%。
- 预测误差：Code TTFT 平均 8.1%（除去 seed 914 的 `no_move_tm_slo` 为 2.4%，该轮为阈值效应，+70.6%），Conv TPOT 平均 1.8%。

## 20. 不留余量的受控基线（V15，11-06 用户要求，提交前写定）

### 20.1 依据

- 用户 11-06 指出：受控基线也加了增长余量。不迁移基线的 Code 直接准入同样要求目标 rank 留出 Conv 增长余量加 1 个页组（`trace_runner.code_headroom`，实测 0.7–0.9 GiB/rank）。余量中"单独速率 15s 增长"这一下限（`planner.conv_growth_tokens`）在分时下也生效，但分时期间 Conv 被暂停，根本不增长。
- 用户用真机日志做的反事实：两种基线只在 Code 准入上不同，第一个被挡的 Code 进来之前轨迹相同。8 个不迁移基线轮中，4 轮的第一个被挡 Code 不留余量就能进：

  | 轮次 | 被挡的 Code | 实际排队 | 不留余量时 |
  |---|---|---|---|
  | V11 `no_move_v2`（seed 914） | 540s 到达 | 542s | +0.7s 可进 |
  | V14 `no_move_tm_slo`（seed 914） | 420s 到达 | 694s | +0.2s 可进 |
  | V9c `no_move_v2`（seed 911） | 660s 到达 | 474s | +4.9s 可进（只富余 1 个页组） |
  | V13 `no_move_v2`（λ=0.14） | 420s 到达 | 625s | +447s 可进 |

  V10 两轮、V11 分时轮、V14 seed 911 轮不受影响，所以 seed 911 上的 A 本身成立。
- 复核（population 快照，容量 170 页组）：V11 在 540.7s 时最空 rank 空闲 9.94 GiB、Code 需要 9.19 GiB、该 rank 有 13 个 Conv；V14 在 420.2s 时为 9.84 GiB 对 9.28 GiB、9 个 Conv。两者都是被余量挡住。
- 模拟（seed 914，块级页模型，旧规则）：去掉余量后 `no_move_v2` 的 Code TTFT 471s → 296s，`no_move_tm_slo` 345s → 235s。
- 影响：V14 中 seed 914 的"去掉迁移"一行（−94.8% / +16.7%）里的"阈值效应"其实是余量造成的；余量消融（R13、R14）只在本文策略上做过，而本文策略被挡就迁移，几乎不付代价。

### 20.2 改动

- **新基线**：`no_move_v2_nh`、`no_move_tm_slo_nh`，即 `no_move_v2`、`no_move_tm_slo` 去掉增长余量。相当于"原生准入语义（按当前长度放得下就准入，KV 不够时由 vLLM 抢占重算）+ 控制器侧排队（没有 DP 空转）+ 按负载选 rank"，作为受控基线替换 R2、R15（总表 R20、R21）。
- **余量规则 v4.4**（`planner-params-v4.json`，`relief_scope: "destinations"`）：单独速率 15s 的下限只用于接收迁移的 rank（迁移准入与回迁的目的地）。Code 自己的 rank 只留其 Conv 在 Code 驻留期间的增长（并行按受干扰的速率，分时只计切片窗口，单纯持有为 0）加 1 个页组：这些 Conv 只在驻留期间增长，Code 完成后它的 KV 就释放了。规则对所有留余量的策略生效，V15 的 `ca_full_tm_slo` 用它；V8–V14 为旧规则（`relief_scope: "all"`）。`planner.py` 的默认值仍是旧规则，以便重现以前的预测。
- 代码：`planner.build_state(..., target_growth_gib)`、`conv_growth_tokens(..., floor)`、`trace_runner` 与 `simulator` 的 Code 余量。新增测试：目标 rank 的余量可以单独设置、目的地仍留增长与页组；`*_nh` 三组策略的 Code 余量为 0；`relief_scope: "destinations"` 下单纯持有时 Code 的 rank 只留 1 个页组；`no_move_tm_slo_nh` 端到端跑完、输出长度精确。

### 20.3 设计

- **8 轮**，3 个 segment（`segments-v15-nh.json`），按重要性排序：
  1. `s914`：`ca_full_tm_slo`、`no_move_v2_nh`、`no_move_tm_slo_nh`（余量影响最大的 trace）；
  2. `s911`：同上；
  3. `s919-l14`（λ=0.14，16 GiB）：`ca_full_tm_slo`、`no_move_v2_nh`。
- 其余设置同 V13、V14：16 GiB、drain 1000s、探针上限 16.5ms、当前长度准入（prompt + 256），`planner-params-v4.json`。`--time 4:30:00`（模拟墙钟约 3.1h），排除 d1n41a23g03。

### 20.4 预测（块级页模型，每策略 20 个副本，`--vary-admission-seed`，planner v4）

文件：`forecast-v15-nh-s914.json`（SHA `2a18f920…`）、`forecast-v15-nh-s911.json`（SHA `7962ae76…`）、`forecast-v15-nh-s919-l14.json`（SHA `25443538…`）。

| 策略 | seed 914：Code TTFT / Conv TPOT / 迁移 | seed 911 | λ=0.14（seed 919） |
|---|---|---|---|
| `ca_full_tm_slo`（v4） | 30.6s / 30.0ms / 40 | 29.9s / 29.3ms / 31 | 29.0s / 29.7ms / 46 |
| `no_move_v2_nh` | 295.8s [277.1, 313.1] / 30.1ms | 277.0s [267.1, 283.8] / 29.7ms | 550.6s [540.2, 561.1] / 33.9ms |
| `no_move_tm_slo_nh` | 235.2s / 27.0ms | 199.4s / 26.3ms | — |
| 参考：`no_move_v2`（v4） | 294.7s / 30.0ms | 279.3s / 29.7ms | 550.8s / 33.9ms |
| 参考：`ca_full_tm_slo`（v3） | 30.7s / 30.2ms / 51 | 29.9s / 29.3ms / 31 | 29.4s / 29.6ms / 45 |

最终方案相对新基线：

| 对照 | seed 914：Code TTFT / Conv TPOT | seed 911 | λ=0.14 |
|---|---|---|---|
| `no_move_v2_nh` → `ca_full_tm_slo` | −89.7% / −0.6% [−2.8, +1.8] | −89.2% / −1.3% [−2.5, −0.0] | −94.7% / −12.4% |
| `no_move_tm_slo_nh` → `ca_full_tm_slo` | −87.0% / +10.9% | −85.0% / +11.2% | — |

- v4 下留余量的基线与不留余量的几乎相同（914：294.7s 对 295.8s；分时 235.2s 对 235.2s）：剩下的驻留期增长加 1 个页组不足以挡住 Code。
- 模拟中 seed 911 与 λ=0.14 上余量都不改变基线（与用户的反事实一致：seed 911 的 V10、V14 轮不受影响；λ=0.14 上真机的反事实为 +447s 可进，模拟预测的受控基线在这个点上偏低 12%，见 §18.5）。
- 最终方案 v3 → v4：914 上迁移 51 → 40 次，Code TTFT 与 Conv TPOT 变化 ≤0.7%；911 上相同。

### 20.5 判定（每个 trace 分别，与本作业中的 `ca_full_tm_slo` 配对）

1. **A 对不留余量的不迁移基线仍成立**：`no_move_tm_slo_nh` → `ca_full_tm_slo` 的 Code TTFT 在 seed 911、914 上都降 ≥ 60%。
2. **对不留余量的受控基线（并行）**：`no_move_v2_nh` → `ca_full_tm_slo` 的 Code TTFT 在三个 trace 上都降 ≥ 60%，Conv TPOT ≤ +20%。
3. **反事实是否兑现**（如实报告，不设判定）：seed 914 上 `no_move_v2_nh` 的 540s Code、`no_move_tm_slo_nh` 的 420s Code 是否几秒内直接准入；各基线的 Code 排队时间与 V10–V14 同 trace 的留余量基线对照（跨作业）。
4. **新余量规则对最终方案**（如实报告）：V15 的 `ca_full_tm_slo` 与 V10–V14 的同名轮次（跨作业，请求级配对），预计差别在轮次间波动（≤2.5%）之内；直接准入与迁移准入的次数。
5. **抢占**：`ca_full_tm_slo` 0 次；不留余量的基线如实报告（并行基线在 Code 驻留期间可能抢占）。
6. **正确性**：输出长度精确，迁移全部逐字节校验通过。

### 20.6 结果（V15 = 294007，作业结束后写，不改变 §20.3–§20.5）

- 第一次提交（293986）在 d1n41a21g03 上探针 18.8ms，启动时中止，未跑轮次（归档 SHA `8864543b…`）。重新提交的 294007 在 d1n41a26g03 上运行（探针 12.7ms，占位后容量 15.98 GiB/卡），8 轮全部 passed。归档 `job-294007.tar.gz`（SHA `11323f01…`）。详见 results-short §12.11。
- 判定：
  1. 成立：`no_move_tm_slo_nh` → `ca_full_tm_slo`，Code TTFT −87.4% [−93.2, −0.2]（914）/ −85.4% [−92.4, −0.5]（911）；Conv TPOT +12.3% / +13.5%。
  2. 成立：`no_move_v2_nh` → `ca_full_tm_slo`，Code TTFT −89.5% / −81.5% / −95.4%（914 / 911 / λ=0.14），Conv TPOT +1.0% / −19.4% / +12.6%（λ=0.14 的 CI 上界 20.3%）。
  3. 反事实：seed 914 上两条都兑现（并行基线 540s 的 Code 0.3s 准入，分时基线 420s 的 Code 0.9s 准入）；seed 911 的并行基线上 660s 的 Code 也直接准入（预计不受影响）；λ=0.14 上余量没有作用（4 个 Code 被挡 452–659s）。
  4. 新余量规则对最终方案：Code TTFT −1.2%~+0.5%，Conv TPOT +0.7%~+4.7%（6 次跨作业比较全为正，均值 +2.1%），迁移多 3–17 次；节点不同、每个 trace 一轮，不能区分规则与轮次间差别。
  5. 成立：所有轮次 0 次抢占。
  6. 成立：926 个请求输出长度精确，141 次迁移全部逐字节校验通过。
- 预测误差：最终方案 Code TTFT 平均 0.3%、Conv TPOT 1.5%；不留余量的并行基线在 seed 911 上 −41.8%（660s 的 Code 真机放下、模拟被挡）、λ=0.14 上 +14.9%，均为阈值效应。

## 21. 通用版算法与逐项消融（V16，11-06 用户要求，提交前写定）

### 21.1 依据

- 用户 11-06：算法要通用，不区分 Code 和 Conv，之后可能加入新的数据集。设计见 `Paper/design.md` §5，实现与本地验证见 §10。
- 通用版的策略为"原名 + `_g`"。V10–V15 的结果都来自按类别区分的旧版，论文改用通用版，需要在真机上重跑主对比和消融。
- 本地验证：
  - 测试：原有 53 个、通用版 14 个通过；
  - simulator 中 21 个旧策略在 V15 的 3 个 trace 上输出逐位不变；
  - 模拟中通用版 R12 与旧版相差 ≤2.4%。
- 提交前的修正：`controller.view_exact` 原来假设 prefill 中的请求按 decode 速率推进，对长 prompt、长输出的请求不安全。现改为"剩余 prefill 按单独运行速度 + 预留的空余 token 按 decode 上界"。本负载的 Code 输出 64 token，在 256 token 的预留之内，修正前后行为相同；新增单测。

### 21.2 设计

- **代码**：tag `v16-general-20261106`。参数：`planner-params-v5.json`（v4 加上 `tm_grace_s` 2、`history_s` 900、`rate_prior_s` 500）、`step-params-dp4-v1.json`。
- **设定**：16 GiB、当前长度准入（prompt + 256）、drain 1000 s、探针上限 16.5 ms，排除 d1n41a23g03、d1n41a21g03。与 V13–V15 相同。
- **20 轮，7 个 segment**（`segments-v16-general.json`，按重要性排序）：

  | segment | 策略 | 作用 |
  |---|---|---|
  | s914-a、s911-a | R12g `ca_full_tm_slo_g`、R20g `no_move_v2_nh_g`、R21g `no_move_tm_slo_nh_g` | 主对比与迁移的消融 |
  | s914-b、s911-b | R16g `mig_random_tm_slo_g`、R17g `ca_admit_tm_slo_g`、R18g `ca_return_tm_slo_g` | placement 的消融：整体、回迁、代价感知准入 |
  | s914-c、s911-c | R14g `ca_full_tm_slo_nh_g`、R10g `ca_full_tm_v2_g`、R7g `ca_full_v2_g` | 余量、切片、分时的消融 |
  | s919-l14（λ=0.14） | R12g、R20g | 更重负载下的锚点 |

- 消融行与同一 seed 的 R12g 在同一作业内配对。
- **不重跑的行**：原生 R1（不受本文算法影响，用 V10/V11 的截断轮作跨作业参照）、R19g（v1 控制面）、R11g（周期切片）。
- **时间**：模拟墙钟合计 7.0 h，`--time 9:00:00`。

### 21.3 预测（块级页模型，每策略 20 个副本，`--vary-admission-seed`）

文件 `forecast-v16-general-s914.json`（SHA `7e9f2fb0…`）、`-s911.json`（`97549914…`）、`-s919-l14.json`（`a5e7c781…`）。

| 策略 | seed 914：Code TTFT / Conv TPOT / Conv TTFT / 迁移 | seed 911 |
|---|---|---|
| R12g | 29.8 s / 29.7 ms / 0.93 s / 46 | 29.9 s / 29.3 ms / 0.71 s / 35 |
| R20g | 295.8 s / 30.1 ms / 0.18 s / 0 | 277.0 s / 29.7 ms / 0.17 s / 0 |
| R21g | 235.0 s / 27.0 ms / 0.58 s / 0 | 199.3 s / 26.3 ms / 0.52 s / 0 |
| R16g | 29.8 s / 32.4 ms / 0.93 s / 20（抢占 1.6） | 29.7 s / 30.7 ms / 0.72 s / 12 |
| R17g | 29.6 s / 31.5 ms / 0.89 s / 21 | 29.9 s / 30.4 ms / 0.71 s / 11 |
| R18g | 30.0 s / 30.1 ms / 0.93 s / 63 | 29.8 s / 29.4 ms / 0.71 s / 47 |
| R14g | 30.7 s / 30.0 ms / 0.91 s / 46 | 29.9 s / 29.3 ms / 0.71 s / 31 |
| R10g | 25.9 s / 29.6 ms / 3.67 s / 43 | 25.9 s / 28.9 ms / 3.26 s / 32 |
| R7g | 60.8 s / 38.9 ms / 0.26 s / 63 | 61.8 s / 39.4 ms / 0.25 s / 31 |

λ=0.14：R12g 29.0 s / 29.6 ms；R20g 550.6 s / 33.9 ms。

对比（→ R12g，seed 914 / 911；λ=0.14 只有第一行）：

| 对比 | Code TTFT | Conv TPOT |
|---|---|---|
| R20g → R12g | −89.9% / −89.2%（λ=0.14：−94.7%） | −1.3% / −1.4%（λ=0.14：−12.5%） |
| R21g → R12g（迁移） | −87.3% / −85.0% | +10.1% / +11.0% |
| R16g → R12g（代价感知 placement） | +0.1% / +0.5% | −8.0% [−9.7, −6.5] / −4.6% [−7.9, −1.8] |
| R17g → R12g（回迁） | +0.9% / −0.2% | −5.7% / −3.6% |
| R18g → R12g（代价感知准入） | −0.4% / +0.4% | −1.3% / −0.5% |
| R14g → R12g（余量） | −2.8% / −0.2% | −0.8% / −0.0% |
| R10g → R12g（切片） | +15.1% / +15.3% | +0.7% / +1.2%；Conv TTFT 3.7 → 0.9 s、3.3 → 0.7 s |
| R7g → R12g（分时） | −50.8% / −51.6% | −23.5% / −25.7% |

- 与旧版 R12（跨作业）：模拟差 −2.4% / −0.2%（Code TTFT），−0.7% / −0.1%（Conv TPOT）。
- 本负载上，通用规则应当只让 Code 触发迁移准入和分时，Conv 不触发。

### 21.4 判定（每个 seed 分别，与同一作业中同一 seed 的 R12g 配对）

1. **主对比**：R20g → R12g，两个 seed 和 λ=0.14 上 Code TTFT 都降 ≥ 60%，Conv TPOT ≤ +20%。
2. **迁移**：R21g → R12g，两个 seed 上 Code TTFT 都降 ≥ 60%。
3. **代价感知 placement（B）**：R16g → R12g，两个 seed 上 Conv TPOT 都 < 0。回迁（R17g）与代价感知准入（R18g）各自的方向如实报告，不设判定。
4. **分时与切片**：
   - R7g → R12g：Code TTFT 降 ≥ 30%，Conv TPOT 降 ≥ 15%；
   - R10g → R12g：Conv TTFT p99 下降，Code TTFT 增加 ≤ 25%。
5. **通用性**（日志核对）：
   - 每个 episode 的独占实例都是 Code 所在的实例，没有 Conv 触发的 episode；
   - 没有 Conv 走迁移准入；
   - 如实报告每个 Code 的 episode 数。
6. **与旧版一致**（跨作业，如实报告）：R12g 与 V10–V15 同 trace 的旧版 R12 相比，Code TTFT、Conv TPOT 的差别是否在轮次间波动（≤5%）之内。
7. **正确性**：
   - 所有轮次 passed，输出长度精确，迁移全部逐字节校验通过；
   - R12g 0 次抢占；
   - 其余策略的抢占如实报告。

### 21.5 结果（V16 = 295488，作业结束后写，不改变 §21.2–§21.4）

- **运行**：
  - d1n41a24g01，探针 12.7 ms；19:40 开始，排队约 2 h，02:4x 结束；20 轮全部 passed，exit 0。
  - 归档 `evidence/xplace-v16-general-20261106/job-295488.tar.gz`（SHA `3b77f0a5…`）。
  - 报告 `report-s914/s911/s919-l14.md`（每个 seed 的 segment 合并为 `out/by-seed/` 后作业内配对），总表 `master-table-v16-general.md`，判定 `verdict-v16.txt`，通用性核对 `check-general.jsonl`。
- **判定**（全部成立）：
  1. R20g → R12g：
     - Code TTFT −88.8% [−93.7, −45.7]（914）、−81.6% [−92.2, −45.7]（911）、−95.3% [−95.6, −95.0]（λ=0.14）；
     - Conv TPOT −3.5% / −19.0% / +2.8%。
  2. R21g → R12g：Code TTFT −84.0% [−91.2, +1.1] / −86.5% [−93.3, −0.0]，Conv TPOT +8.3% / +12.7%。每轮只有 4 个 Code，CI 上界在 0 附近。
  3. R16g → R12g：Conv TPOT −8.6% [−9.3, −8.0] / −4.6% [−4.9, −4.3]（预测 −8.0% / −4.6%）。
     - 回迁（R17g → R12g）：−8.2% / −3.7%；
     - 代价感知准入（R18g → R12g）：+1.1% / −0.4%；
     - B 几乎全部来自回迁。seed 911 上，R12g 的第二次准入复用上一个 Code 腾空的 rank，只迁 1 个；随机准入为 11 个。
  4. 分时与切片：
     - R7g → R12g：Code TTFT −45.3% / −53.5%，Conv TPOT −17.5% / −29.3%；
     - R10g → R12g：Code TTFT +23.1% / +22.7%（≤25%，预测 +15%），Conv TTFT p99 25.6 → 3.7 s、24.0 → 3.7 s，Conv SLO 0.81/0.79 → 1.00。
  5. 通用性：16 个分时轮共 64 个 episode，全部在 Code 实例上，每个 Code 一个；迁移准入只由 Code 触发；526 个被迁请求都是 Conv。
  6. 与旧版 R12（跨作业，9 次比较）：Code TTFT −1.7%~+3.9%，Conv TPOT −2.2%~+2.5%，在 5% 以内。
  7. 正确性：
     - 2282 个请求输出长度精确，526 次迁移全部逐字节校验通过，R12g 0 次抢占；
     - 只有 seed 911 的 R20g 抢占了 3 个 Conv：660 s 的 Code 直接准入 rank 0，该 rank 到达容量上限，之后 Conv 增长触发抢占。这是不留余量基线的预期行为。
- **如实报告**：
  - 余量（R14g → R12g）：Code +0.4% / +3.3%，Conv TPOT +1.3% / +0.2%，没有作用；
  - R18g 在 seed 914 上比 R12g 好 1.1%，与预测（−1.3%）方向相反；
  - R21g、R20g 的 Code TTFT 偏离预测（914：192.9 对 235.0 s、275.4 对 295.8 s），属于阈值附近的轮次间差别。

## 22. 迁移主线稿的独立失衡校准（2026-11-07/08）

本节只补充固定batch机制校准，不运行已从论文删除的执行策略，也不做公共waiting queue消融；不覆盖或改写§21的历史结果。

预注册分别见`imbalance-calibration.md`与`ep4-tp2-calibration.md`，冻结计划和输入随作业归档。DP4/TP1/EP4作业300123完成30点×2轮、4672请求；DP2/TP2/EP4最终作业303594完成24点×3轮、5664请求。均为两个实例共用4卡、只有一个实例接收固定4096输入/1024输出请求，逐点排空、随机顺序、公共decode窗口测量，全部有效且零抢占。三次失败或不完整的TP2前序作业保留，不并入最终重复样本。

DP2/TP2最大失衡布局相对均衡的步时增加27.0%/57.5%/91.0%（BS32/64/128），但BS32/64在末端回落，不宣称曲线单调。图2(b)仍只含DP4的BS32/64，新增图7以独立面板展示两种拓扑和三种batch；范围带是各自作业内的min–max。两个作业节点和运行时适配不同，归一化失衡相同也不等于完整布局相同，不能作为纯TP消融或TP2迁移收益证据。

原始归档SHA、四卡拓扑和逐请求复算结果在对应校准文档及`Paper/draft/validation-report.json`记录。该补充只关闭“TP2失衡机制是否存在”的缺口；通用迁移的强基线、独立重复、受控组件消融及TP2逐shard事务验证仍未完成。
