# Motivation 实验总结：CUDA graph 下的 attention-DP 请求不均衡

本报告属于 kvcached 研究 `reports/`；执行脚本在上一级，`evidence/` 路径相对于外层 ParaCloud。

记录日期：2026-10-18（CST，UTC+8）。**主实验使用 CUDA graph；eager 仅作为历史对照。**

## 1. 目的与范围

固定模型、GPU 对、全局请求数及输入输出长度，只改变请求所属的 attention-DP rank，
测量均衡与极端不均衡之间的稳定 decode 性能差异。

研究目标是为后续跨模型 request/KV-home placement 提供动机：联合考虑每个模型内部的
attention-DP 负载、graph padding 成本和 rank-local 物理 KV headroom。
**本实验只有一个模型服务；没有第二个模型的 KV 干扰，也没有实现迁移或 placement。**

## 2. 主实验 setting

| 项目 | 设置 |
| --- | --- |
| 作业与时间 | 229748；2026-10-18 20:34:13–20:53:23 CST；COMPLETED/0:0 |
| 硬件 | 同一节点 `d1n41a14g01`，同一对 NVIDIA A800-SXM4-80GB |
| 模型 | Qwen3-30B-A3B-Instruct-2507，真实 BF16 checkpoint，单个 resident tenant |
| 推理栈 | native vLLM 0.26.0 + kvcached；Torch 2.11 / CUDA 12.9 |
| 并行 | attention TP=1，DP=2，EP=2，PP=1；每 rank 持有每层 128 个 expert 中的 64 个 |
| Kernel / 通信 | FlashAttention2；Triton unquantized MoE；`allgather_reducescatter` |
| CUDA graph | `FULL_DECODE_ONLY`；稳定 decode 实际为 FULL；关闭 torch.compile（compilation mode=0） |
| Capture sizes | 1 / 8 / 16 / 32 / 64 / 128 / 256，全部 capture 成功 |
| Scheduler | max-model-len=4096；max-num-seqs=256；max-num-batched-tokens=4096 |
| KV | 每 rank 44 GiB 虚拟 KV 预算，按需映射物理显存；不代表预先占用 44 GiB 物理显存 |
| 请求 | 每条精确 1024 input / 512 output tokens；固定长度，不因 EOS 提前退出 |
| 全局 batch | 128、256、32，按此顺序在同一服务上依次测试 |
| 布局 | 每个 batch B 测 B/2:B/2、B:0、0:B；布局指 DP0:DP1 的真实请求数 |
| 关闭的功能 | DBO、EPLB、APC、async scheduling、sleep、CPU KV offload；无阶段 profiling hooks |

不同 batch 顺序未随机化；布局比较在各 batch 内配对，不能把跨 batch 的绝对差异完全归因于 batch。
当前 graph 主实验只测均衡和两个极端；**尚无 graph 模式下的中间不均衡度曲线**。

## 3. 请求协议与指标

- 每个 batch 三次重复；每次依次执行“均衡起点 → 随机排序的两个极端 → 均衡终点”，
  seed=42。共 12 个主 trial/batch、36 个主 trial；正确性、warmup 和 untraced controls 不计入主 trial。
- 请求通过并发 barrier 发出，显式指定 DP rank。同一 batch 的各布局复用相同 prompt 集合，
  重复间轮换 prompt 分配；每个 trial 前后等待空闲并检查 KV 物理映射回收。
- 从 worker trace 找出全体请求均为 one-token decode、人口不变的共同窗口；空 rank 的 dummy
  也必须连续参与。裁掉首尾各 32 step，各 rank 至少保留 128 step。
- 主指标 `C`：各非空 rank 的 host execute-start 间隔 p50，取最大值；它是稳定 decode
  cadence，**不是 CUDA attention kernel 时间，也不是纯 GPU step duration**。
- 第 r 次重复的 slowdown `R_r = C_skew,r / mean(C_balanced_start,r, C_balanced_end,r)`。
  报告 `R = mean(R_r)`，固定总 token/step 下的 cadence-derived 吞吐损失 `L = 1 - 1/R`。
  表中 cadence 是各 trial cadence 的均值；loss 由配对比值计算，不直接用四舍五入后的表值相除。
- 三个 repeat ratio 用 Student-t（df=2）构造 95% 区间，再以 `1 - 1/R` 转换为 loss 区间；
  不把同一次 trial 内的数百 step 当作独立重复。
- 目标是 **吞吐损失至少 30%**，等价于 cadence slowdown 至少 1.4286x，而非 latency 增加 30%。
  确认标准：**两个 hot-rank 方向的 95% loss 区间下界都至少 30%**。
- Burst 吞吐、TTFT、客户端 ITL/TPOT 为辅助指标；burst 包含 prefill/ramp-up，不能替代稳定 decode。

## 4. 主结果

| 全局 batch | DP0:DP1 | 稳定 cadence (ms) | 配对 slowdown | Decode 吞吐损失 | 95% loss 区间 | Burst 损失 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 64:64 | 24.57 | 1.0000x | 基准 | — | 基准 |
| 128 | 128:0 | 36.33 | 1.4785x | **32.37%** | **31.78–32.94%** | 33.06% |
| 128 | 0:128 | 36.25 | 1.4751x | **32.21%** | **30.71–33.65%** | 32.73% |
| 256 | 128:128 | 32.88 | 1.0000x | 基准 | — | 基准 |
| 256 | 256:0 | 53.48 | 1.6266x | **38.52%** | **38.16–38.88%** | 39.33% |
| 256 | 0:256 | 53.33 | 1.6222x | **38.35%** | **37.90–38.80%** | 39.17% |
| 32 | 16:16 | 15.44 | 1.0000x | 基准 | — | 基准 |
| 32 | 32:0 | 21.57 | 1.3967x | 28.40% | 22.10–33.76% | 28.44% |
| 32 | 0:32 | 21.92 | 1.4196x | 29.56% | 26.71–32.19% | 29.50% |

**Batch 128 和 256 达到预先声明的确认标准；batch 32 没有达到，保留全部结果。**

## 5. 有效性检查

| 全局 batch | 各 rank 最少稳定 step | 最大采样 KV 使用率 | 均衡首尾最大绝对漂移 | Traced/untraced 均衡 TPOT 比值 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 412 | 47.4515% | 0.23% | 1.00005 |
| 256 | 379 | 87.1325% | 0.24% | 1.00316 |
| 32 | 437 | 17.0656% | 0.73% | 0.99409 |

- 全部 **36 trial / 4992 请求 / 2,555,904 output tokens** 已从原始请求和 trace 独立复算。
- 所有稳定窗口无 waiting、无新增 preemption，人口和 FULL/padding 证据均符合协议；
  KV 使用率均低于拒绝阈值 90%。表中 KV 使用率是服务指标，不是整卡物理显存占用率。
- 每个 trial 前后每 rank 都恢复 96 MiB mapped idle backing；reserved backing=0。
- Worker trace 内存缓冲、trial 结束后落盘；不在主测路径插入阶段 hooks 或逐 step CUDA 同步。
  Untraced controls 是顺序敏感性检查，包含时间漂移，不是零 tracing 开销的因果证明。
- 作业内 236 项回归通过；该轮实现的本地 client/control 测试 43 项通过。

## 6. CUDA graph + 空 attention rank 的机制

Trace 已验证：batch 128 的均衡布局为 unpadded=64/64、padded=64/64；
128:0 为 unpadded=128/1（空 rank dummy）、padded=128/128；0:128 则相反。
Batch 256 对应 128/128 → 256/256。

当前 vLLM 的 DP 协调先同步 graph 模式，再将两 rank 对齐到最大 graph token 数。
空 rank 不能跳过 forward/collective：其 GPU 仍持有 EP expert shard，并参与处理热 rank
真实 token 的 expert 工作。投影/router 按 padded tensor shape 执行，当前 AgRs 通信也使用
padded DP 长度，因此不均衡同时改变真实 attention 分布以及 graph/通信/expert 工作量。

**Padding 不等于复制真实请求的长上下文 attention。** 当前普通 dummy decode metadata 中
第一条 `seq_len=1`，其余 padding `seq_len=0`；dummy 的 `slot_mapping=-1`，跳过 KV 写入。
Attention kernel 随 FULL graph replay，但没有热 rank 的 128 条真实上下文 KV，也不会因
dummy 新增真实请求 KV。固定 launch/grid、零长度处理和各 kernel 的实际耗时尚未 profiling。
已存档安装版本源码摘录：`evidence/kvcached-batch-20261018/graph-empty-rank-source.txt`。

因此确认的是 **CUDA-graph DP/EP serving 下请求不均衡的系统成本**，不是“空 rank 整卡 idle”，
也不能用 32–38% 的损失反推出纯 attention 时间占比或证明 memory-bound/HBM 饱和。
`FULL_DECODE_ONLY` 的 mixed prefill/decode 回退步骤不属于主结果窗口；其他 backend 须另测。

## 7. 历史 eager 对照与边界

| 对照 | Setting | 极端稳定 decode 损失（DP0-hot / DP1-hot） | 说明 |
| --- | --- | --- | --- |
| 初始不均衡曲线，226683 | B=16，128/256；2 GiB KV/rank；eager | 约 4.3% / 3.5% | 五档 I=0/0.25/0.5/0.75/1，均测双方向；非严格单调 |
| 长 token，227012 | B=16；24 GiB KV/rank；maxlen=10240；eager | 8192/1024 时 3.42% / 3.44% | Burst 损失 13.48% / 13.60%，TTFT 4.94→9.63–9.76 s；主要放大 prefill/ramp-up |
| Batch 扫描确认，229747 | B=256，1024/512；44 GiB KV/rank；maxlen=33792；eager | 8.16% / 6.00% | 三次新配对重复，未达 30% |
| 高 KV 扫描确认，229747 | B=128，2048/512；其余同上 | 5.14% / 2.61% | 三次新配对重复，未达 30% |

229639 因独立 home 配额在第八组筛选落盘时失败；只保留七组完成的单 repeat 筛选。
229747 仅复用匹配旧筛选做候选选择，两组确认数据均新测，不将跨 allocation 筛选混入 CI。
Eager 与 graph 使用不同节点、maxlen，不把绝对 cadence 差当作成对的 graph speedup。
此前阶段诊断明显扰动采样 step（约 9.5–19.2%），不作为 attention kernel 占比证据。

## 8. 对 design 的启示

在标准 CUDA-graph DP/EP 配置下，只改变 request/KV home 就能造成 **32–38%** 稳定 decode
吞吐损失，说明 placement 不能只追求 KV 容量均衡。还需建模每个模型内部的 DP 最大 batch、
capture 档位/padding 和真实上下文；只让跨模型整卡总请求数相等，不保证各模型内部均衡。

Graph 中间曲线的 sparse 30 trial 已由 230346 完成，初始单一仿射关系不通过残差标准；
231467 已同 allocation 完成 sparse/dense；两种配置的单一仿射拟合均不通过预声明残差标准，
详见 `curve-results.md`。后续需验证双模型
同时驻留时的 load/KV tradeoff、admission/SLO 和 placement 收益。**这些收益尚未实测。**

## 9. 证据与复现入口

- 主报告：`graph-results.md`；历史报告：`motivation-results.md`、`length-results.md`、`batch-results.md`。
- 原始主实验：外层 `ParaCloud/evidence/kvcached-graph-229748/`，含 `all-cases-verification.txt`、
  `remote-archive.sha256`、`archive-sha256-verified.txt`、trial/trace JSON、CSV/SVG 和最终 accounting。
- 汇总 JSON：上述目录下 `state/dual-qwen-229748/motivation/graph-sweep/results.json`。
- 远端原始数据：`~/run/kvcached-paracloud/state/dual-qwen-229748/`；旧 ops state 路径保留 symlink。
- 执行入口（相对于本报告目录）：`../motivation-graph.sbatch` → `../run-dual-qwen.sh` → `../graph_sweep.py` → `../motivation.py`。

```bash
ssh paracloud 'root="$HOME/kvcached/paracloud"; sbatch --output="$root/logs/graph-%j.out" --error="$root/logs/graph-%j.err" "$HOME/kvcached/kvcached/experiments/dp_opt/motivation-graph.sbatch"'
```

复跑前检查 job marker/accounting，避免重复提交。所有 import、测试和推理只在 Slurm compute
node 执行；登录节点只查询调度、读取或同步文件。此文是既有实验归档，不重新提交作业。
