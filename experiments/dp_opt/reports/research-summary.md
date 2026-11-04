# 跨模型 attention-DP KV placement：研究总结

2026-11-02 由 Claude 接管后重写。Codex 版本的全文（含逐小时状态流水）见 [research-summary_codex.md](research-summary_codex.md)。各分篇报告保留原始设置和负结果，本文只写结论。

## 0. 用户 guideline（2026-11-02）

1. 由 Claude 接管。数值 debug 只在等待性能实验时顺手做。
2. 依据最初 motivation 完成后续实验，可用 4 张 GPU，即每个 tenant DP=4/EP=4。
3. A/B 效果都要有，并有消融：
   - A：迁移降低 Code TTFT；
   - B：cost-aware placement 降低 Conv 的性能下降。
4. Conv 下降相对同 trace 的 `no_move` 控制在 20% 以内。主指标是 Conv avg TPOT，同时报告 P95/P99 TPOT 和 Conv TTFT。
5. 复用数据集：Code 用 SWE-smith ≥100K，输出 64；Conv 用 OpenThoughts3 quality cohort bank256，保持原始输出长度；不用 4x resp、人工延长的 AzureConv 或 Stratos。
6. 真实实验尽量少，多用 log 分析和 simulator，但最终结果必须来自真实实验。
7. Codex 文档备份为 `*_codex`，新文档基于它们重建。

## 1. Idea 与成功标准

两个模型共址、KV 按需分配、attention 用 DP 时，完全均衡不一定最优。A（Code，约 100K 输入的长请求）到达时，如果每张卡都放不下，而总空闲足够，就迁移 B（Conv）在 attention-DP rank 之间的 live 请求和 KV，为 A 腾出 rank-local headroom，即使 B 原本是均衡的。

迁移会让 B 失衡：每步按最大 rank 补齐，会产生 graph padding 和 attention 失衡。因此需要按代价决定：把谁挪到哪里，以及 A 结束后是否回迁。

成功标准（[placement-design.md](../docs/placement-design.md) §10）：
- A 的等待显著下降；
- 计入复制、暂停和同步代价后仍有净收益；
- B 的代价受控；
- cost-aware 策略明显优于简单的 cost-unaware 迁移（与 simple greedy 表现一样则贡献不足）。

## 2. 系统

- **模型与部署：**
  - 两个独立的 Qwen3-30B-A3B tenant（model0=Code，model1=Conv），复用同一 checkpoint，权重、KV、请求和 EP 组各自独立；
  - 每个 tenant attention-DP + EP、attention TP=1、BF16；
  - 默认 sparse FULL_DECODE_ONLY CUDA graph，capture 档位 1,8,16,32,64,128,256；
  - max_num_batched_tokens 4096，max_model_len 131072。
- **KV：** kvcached 让两个 tenant 共享物理 KV 预算，KV 为 96 KiB/token。scheduler 按 P+O 全量预留。所有基线用同一套自定义 scheduler，不是原生 vLLM 的动态增长。
- **迁移：**
  - 在 vLLM 每 32 步一次的 DP all-reduce 边界暂停；
  - 导出 block，在目标 rank 分配；
  - 用 NCCL `all_gatherv` 复制，并做 bytewise echo 校验；
  - 提交 ownership，释放源页，然后恢复。
- **每步代价：** vLLM 在开 CUDA graph 时把各 DP rank 补齐到 max token 数，因此 decode 每步约为 `f(G(max_r n_r)) + c·max_r ctx_r`。

## 3. DP=2 已确认的结论（2×A800）

### 3.1 Motivation（单模型，229748）

固定请求数和长度、只改 DP home 时，失衡造成的吞吐损失：全局 batch128 为 32.4%，batch256 为 38.5%。中等失衡的曲线与 capture 档位相关，不是线性的（[curve-results.md](curve-results.md)）。

### 3.2 准入机制成立，但 cost-aware 没有带来独特收益

- **真实容量见证（254337）：**
  - Code 到达前，两卡 mapped 为 [2400, 1440] MiB，Code 需要 9504 MiB，单卡预算 10240 MiB，所以任何一卡都放不下；
  - 迁走一个 Conv 后，mapped 变为 [3648, 9696] MiB，可以放下；
  - 复制通过 bytewise 校验。
- **Poisson trace 中 Code avg TTFT 相对 no_move 的变化：**
  - 251533：−93.1%
  - 254337：−77.6%
  - 259091：−80.8%
  - 260757：−81.0%
- **随机可行迁移拿到同样的 Code 收益。** 260757 中 workaware 比 random 还差 1.8%。带性能项的准入 ILP 在 237899 中，21/21 个快照与去掉性能项后的动作相同。
- **Poisson 下 Conv TPOT 与 no_move 接近：**
  - 257484：−4.6%
  - 259091：−2.3%
  - 260757：−2.4%（kv_only）、+0.7%（random）
  - 254337：+9.8%
- **代价在 Conv tail TTFT 上：** 251533 中 Conv P99 TTFT 从 1.15s 升到 12.46s。
- **样本极少：** 这些 trace 每个策略只有 2 个 Code。

### 3.3 公开数据 closed native（OpenThoughts3 + SWE-smith，6 个 case）

每个 case 有 1 个 Code，N−1 个 Conv。TTFT 单位为 s，Conv TPOT 单位为 ms，括号内是相对 no_move 的变化。

| case | no_move Code TTFT / Conv TPOT | random | headroom | workaware |
|---|---|---|---|---|
| s1-N2 | 135.7 / 12.05 | 57.1 / 18.32 (+52.0%) | 55.9 / 18.36 (+52.3%) | 55.8 / 16.58 (+37.6%) |
| s1-N3 | 241.4 / 18.42 | 56.2 / 22.30 (+21.1%) | 55.7 / 22.07 (+19.8%) | 56.3 / 21.95 (+19.2%) |
| s1-N4 | 243.3 / 18.51 | 55.9 / 22.71 (+22.7%) | 55.3 / 22.86 (+23.5%) | 56.0 / 22.76 (+23.0%) |
| s2-N2 | 119.6 / 15.02 | 57.4 / 20.07 (+33.6%) | 56.5 / 20.33 (+35.4%) | 56.2 / 18.13 (+20.7%) |
| s2-N3 | 201.2 / 19.41 | 56.2 / 23.03 (+18.6%) | 55.9 / 22.85 (+17.7%) | 55.4 / 22.99 (+18.4%) |
| s2-N4 | 216.8 / 19.30 | 56.4 / 23.51 (+21.8%) | 56.0 / 23.41 (+21.3%) | 56.1 / 23.11 (+19.7%) |

结论：
- 回迁只在 N2 有效，因为它跨越了 graph 档位 1→8。到 N3/N4，robust 门槛中的 uncertainty（118.5/89.0/34.2 request-s）是预测收益的 1.7–5.5 倍，回迁只在人口降回 2 时才触发（[review](../../../../evidence/public-pause-cost-opportunity-20261102/review.md)）。
- closed probe 有结构性偏差：no_move 让 Code 等到 Conv 快结束才运行，所以 Conv 几乎不受干扰，这放大了"Conv 相对 no_move"的退化。

### 3.4 跨 tenant 干扰（2026-11-02 由原始 cadence 复算）

两个 tenant 同时活跃时：

| 指标 | 两个 tenant 同时活跃 | Code 驻留之外 / 单独运行 |
|---|---|---|
| Conv 每 token（N4 三个迁移策略的中位数） | 175–186ms（约 9.5 倍） | 19.3ms |
| Code 100K prefill | 约 54s | 单独约 26s |
| Code TPOT | 约 170ms | 16.7ms |

- N4 中 Conv 对 no_move 的 +22.6% 退化里，约 18 个点来自干扰，约 4 个点来自失衡和整合。
- 推测原因：进程间 GPU time-slicing 叠加每层的 EP collective。
- 影响：Poisson 稳态下 no_move 最终也要付这部分代价，因此 B 效果只能来自失衡和迁移代价。

### 3.5 数值一致性（不再阻断性能实验）

- 零迁移、只改同 rank 布局就能复现 greedy token 分歧（276017）。
- 约 99.99% 的分歧来自上游 hidden，而不是 head（278529）。
- 这是 batched BF16 推理的 batch-invariance 问题，不是迁移引入的。
- 正确性标准改为：
  - 每次迁移都通过 bytewise echo；
  - 输出长度精确，token 不丢不重；
  - greedy 一致性只作观察记录。

## 4. 当前 campaign：4 卡（DP=4/EP=4）

代码、协议和结果在 [kvcached/experiments/xplace/](../../xplace/README.md)，进度见 `xplace/reports/progress.md`。

### 设计

- **策略：**
  - `no_move`
  - `no_move_fair`（τ=30s）
  - `mig_random`：cost-unaware 迁移
  - `ca_admit`：cost-aware 准入
  - `ca_return`：cost-aware 回迁
  - `ca_full`：两者都有
- **方法：** 用 simulator 加 Job 1 的 4 卡校准来选实验区间、调参数（只用非最终 seed）。Job 2 在冻结的 trace 上跑全部策略。
- **成功标准：**
  - A：迁移类策略的 Code TTFT 低于 `no_move` 和 `no_move_fair`；
  - B：`ca_full` 的 Conv avg TPOT 相对 `no_move` 退化 ≤20%，且优于 `mig_random`；
  - 消融中 `ca_admit` 和 `ca_return` 各自都有贡献。

### 4 卡 pilot 之后的结论（2026-11-03）

- **A 成立**（pilot，6 个 Code）：Code TTFT 148s → 58–60s。
- **placement 带来的 B 很小**：pilot 中 `ca_full` 比 `mig_random` 低 5.2%，几乎全部来自无 Code 时段的回迁均衡（G16 档占比 57% → 17%）。按 pilot 校准的 simulator，B 在所有扫描区间都 ≤2%。
- **Conv 退化的主因是跨 tenant 干扰**：Code 驻留期间 Conv 慢约 10 倍，Code 自身也慢约 14 倍，placement 管不到。
- **新方向：分时**（用户同意）。Code 驻留期间暂停 Conv；vLLM DP 暂停期间 engine 完全空闲。simulator 预测 Conv TPOT −25~30%，Code TTFT 再降 58–80%。
- **执行方式改为 ≤1h 短作业**（用户要求），12h 的 Job 2 已撤销。协议见 `xplace/docs/protocol.md` §6。

### 4 卡最终结论（2026-11-04）

详见 [xplace/reports/results-short.md](../../xplace/reports/results-short.md)。10 个作业，26 个 trial，同一节点、同一作业内配对，seed 702 与 704 两个 trace，每轮 4 个约 100K 的 Code。

- **A 成立**：迁移让 Code TTFT 降 45–59%，两个 seed 都复现。
- **B 分两层**：
  - **加上分时等调度**：Conv TPOT 比 `no_move` 好 14–33%，Conv 不仅没退化，反而改善；
  - **placement 本身**：机制成立，cost-aware 回迁把可避免的 G16 占比从 0.41 降到 0.04–0.13；但端到端 ≤2.5%，小于干扰模式噪声（约 ±12%），同机配对中测不出。
- **分时是最大的杠杆**：Code 驻留期间暂停 Conv，可消除跨 tenant 干扰，Conv TPOT −24~30%，Code TTFT −50~56%。
- **三轮算法优化**（每轮都有预注册预测，并在真机上验证）：
  - v2 控制面：Conv TTFT −51%，分时下 Code TTFT 降到约 25s；
  - v3 分片分时：Conv TTFT p99 24 → 6s；
  - v3.1 按需切片：相对 v3，Code TTFT −11%，Conv SLO 1.00。
- **最终推荐 `ca_full_tm_slo`**：对公平基线 `no_move_v2`（V6，最新代码，控制面版本一致，同机配对），Code TTFT −75%，Conv TPOT −21%，Conv TTFT 无显著差异，Conv/Code SLO 都是 1.00。
- **placement 本身的 B 在不分时下为负**：两次独立配对中 `ca_full` 都比 `mig_random` 差（+2.4%、+7.1%）。无 Code 时段的机制有效，但 Code 驻留期间干扰更重。分时下两者无差别。
- **控制器卡顿根因**：在共享文件系统上同步写 progress，v3.2 已改为后台线程写入。
- **正确性**：11 个作业、32 个 trial，2168 个请求输出长度全部精确，159 次迁移全部逐字节校验通过。

## 5. 历史分篇报告导航

| 内容 | 位置 |
|---|---|
| Motivation / graph / curve | [motivation-summary.md](motivation-summary.md)、[graph-results.md](graph-results.md)、[curve-results.md](curve-results.md) |
| 长 KV 准入主场景 | [benefit-long-kv-results.md](benefit-long-kv-results.md) |
| Poisson：headroom / 1.1 压力 / workaware / 1.2 压力 / random | [headroom-tail-status.md](headroom-tail-status.md)、[pressure-ratio-status.md](pressure-ratio-status.md)、[workaware-return-results.md](workaware-return-results.md)、[workaware-pressure-1p2-results.md](workaware-pressure-1p2-results.md)、[random-baseline-results.md](random-baseline-results.md) |
| 长输出回迁 probe | [long-conv-return-probe-results.md](long-conv-return-probe-results.md) |
| 公开数据 native 与数值诊断 | [public-quality-source-recovery-results.md](public-quality-source-recovery-results.md) |
| 原始设计 | [../docs/placement-design.md](../docs/placement-design.md)、[../docs/ilp-scheduler-design.md](../docs/ilp-scheduler-design.md) |
| 机器可读清单（Codex 版） | [../docs/experiment-matrix_codex.json](../docs/experiment-matrix_codex.json) |
