# KV-only 的 50% 门槛：长驻留 Conv / 长输入 Code 的实验设计分析

记录日期：2026-10-24，Asia/Shanghai。**本页新增结果全部是本地 CPU 条件分析，
不是新 GPU 收益实验。** 247027 已在2026-10-24 12:15:08结束，旧749请求配置未改。
用户现授权本协议新GPU探索；自动17case/8heldout流程已本地实现，待旧终态独立
核验后保护同步提交。本文CPU预测不因授权变为GPU结果；未commit/push/Aegaeon。

实跑原结果的唯一报告仍是 [agentic-trace-results.md](agentic-trace-results.md)。
下一阶段协议见 [long-resident-conv-stress-protocol.md](../docs/long-resident-conv-stress-protocol.md)。

## 1. 先固定“50%”口径

建议新实验的机制主目标为 **长 Code 请求端到端 latency P95 相对 no_move 和
no_move_fair 均下降至少 50%**；两个比较单独成立才算通过。计算式为
`1 - P95(KV-only) / P95(baseline)`。这是延迟减半，即速率倒数意义的 2x；
不是 `baseline/treatment - 1 >= 50%`，后者只需延迟下降 1/3。

这是明确提出的 **Code QoS 目标**，不是把用户的“50%”暗换成全请求收益。
必须同时报告全请求、全部 Conv、baseline 定义的老 Conv 以及 Code 的
queue/TTFT/latency/TPOT P90/P95/P99、mean、120s SLO、失败及完整 drain。
如果主目标指全请求 mean/P95 或吞吐提高 50%，现有机制和下面分析均不能承诺。

245583 只有 4 Code / 749 requests。baseline Code mean latency 76.336s，
全请求 mean 21.221s，因此 Code latency sum 约占 1.92%；即使把所有 Code 的
延迟变为零、Conv 不变，全请求 mean 也最多下降约 1.92%。固定开放环 offered
rate、全部完成的 trace，不能把同样的请求流报告成稳态吞吐提高 50%。

## 2. 当前 trace 为何不足

- 245583 的 Code 平均排队仅 19.582s；准入后的驻留约 57s，其中首 token/prefill
  约 49s。迁移可以省 queue，不能消除真实 100K prefill。
- Conv 原输出 512–1024，mean 662.9；实跑驻留约 19.3s，median TPOT 约 19.76ms。
  普通等待很快获得自然释放，比暂停、RPC 和迁移的优势窗口短。
- 前两条 Code 的到达间隔只有 0.919s。第二条主要遇到另一个 100K Code，而非
  可迁移 Conv 造成的放置碎片。两 Code 的真实总容量竞争不能通过移动 Conv 消失。
- 少数迁移可以帮助 Code，仍可能拖慢已有 Conv；当前实跑已经有此代价，不能删除
  老请求或只统计新请求 queue 来宣称净收益。

提高 Conv RPS 不保证更有效。过高时会使 aggregate 真不足；本研究需要的是
**少量但长驻留的 Conv 分散在两个 rank，总 KV 可合并，释放时间却足够晚。**

## 3. 延迟减半的必要条件

定义 baseline `L0 = W0 + S0`，KV-only `L1 = W1 + M + S1`。这里 M 包含真实
规划、暂停、同步、复制、resume 和准入控制，不仅是 copy kernel 时间：

```text
L1 <= 0.5 * L0
<=> W0 >= 2 * (W1 + M) + 2 * S1 - S0
```

若 S0=S1=57s、W1≈0、M=5s，需要 W0≥67s。
若迁移保留了更多 Conv，Code 执行变慢到 S1=75s，仍有 S0=57s、M=5s，
则需要 W0≥103s；M=10s 时需要 W0≥113s。不能假定混合执行时间不变。
这些是每请求条件，不可简单把各变量的独立 P95 相加作为真实 P95。

## 4. 应修改驻留和容量几何，而非盲目加压

保留两 A800、两个独立 Qwen tenant、DP2/EP2/attention-TP1、BF16、128K、
默认 sparse FULL_DECODE_ONLY 和 32step。Code 输入继续使用已验证真实
SWE-smith whole-history prefixes，输出仍 64，**不靠加长 Code decode**。

优先候选：Conv 输入保留 Azure slice，输出显式合成为 8192；预算 10240MiB/GPU；
Conv .025RPS，Code .0006–.001RPS 作低碰撞机制分析。所有策略使用相同预算。
另保留原 11328MiB/GPU 预算及 12288/16384 输出对照，不能只选择正档。

已冻结 trace 的 Conv 输入为 2049–7764 tokens。1 block=16 tokens=1.5MiB，
输出8192时 full-reservation 约 961.5–1497MiB/request（0.939–1.462GiB）。
Code 新增映射约 9.28GiB，保守规划约 9.375GiB，另有模型/rank pinned pages。
10GiB 下剩余不足以保留任意一条这样的 Conv，因此 Code 目标卡通常必须腾空。
10240MiB 不是 96MiB 页的整数倍；可映射页上限为 10176MiB，不能将剩余64MiB
当成一整页。CPU 几何检查自然使用离散页，GPU 必须重新 probe。

在 20ms/token 和 .2s prefill 的 **假定** 下，Conv service=164.04s。
.025RPS 给出 E[N]=4.101、E[full-reserved KV]=4.580GiB，均值规模允许集中到一张卡，
但不保证各样本均可行。
这不是原 Azure 数据：完整公开 Conv release 的最大输出只有 1500。8192及以上
必须标为 synthetic long-output stress variant，并保留原 paired lengths 对照。

如果固定两边各2条 Conv、残余寿命独立均匀 U(0,S)、Code 需要空 rank，且不再补入
未来 Conv，则等待 W=min(max(rank0 residuals), max(rank1 residuals))：

```text
F_W(w) = 1 - (1 - (w/S)^n0) * (1 - (w/S)^n1),  0 <= w <= S
n0=n1=n: W_p95 = S * (1 - sqrt(0.05))^(1/n)
```

n=2、S=164.04s 给出 W_p95=144.541s；baseline Code P95≈201.541s，
若 treatment service75s+control5s，则为80s，条件下降约60.31%。
该固定人口解析例解释“为什么可能”，不代表真实两个 baseline 的分布。

## 5. 15档 CPU envelope：全部保留

工具 `../stress_trace_envelope.py`，每档10000 offered-state samples，seed20261024。
基于旧 trace 的745 Conv marks和4 Code marks；源 trace SHA为
`6d43e314ddf3b7a64c883dc19bcdefb1210bb655b8c8d48c5af56580d4f1c006`。
基线 Code service57s、迁移控制proxy5s、Conv TPOT proxy20ms。以下是
**safe/supported 且没有其他驻留 Code 条件下**的两条 CPU latency proxy，
不是 no_move/no_move_fair 的实跑或完整开放环队列仿真。

| Conv O | Conv RPS | B MiB/GPU | Code S1(s) | L0 P95(s) | L1 P95(s) | mean下降 | P95下降 | P99下降 | offered超预算/10000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原paired O | .61 | 11328 | 57 | 63.250 | 62.027 | -1.28% | 1.93% | 4.86% | 0 |
| 8192 | .025 | 11328 | 57 | 159.405 | 62.000 | 30.28% | 61.11% | 65.83% | 0 |
| 8192 | .025 | 11328 | 75 | 159.405 | 80.000 | 9.41% | 49.81% | 55.92% | 0 |
| 8192 | .025 | 10240 | 57 | 202.498 | 62.000 | 54.05% | 69.38% | 60.28% | 0 |
| 8192 | .025 | 10240 | 75 | 202.498 | 80.000 | 40.72% | 60.49% | 51.84% | 0 |
| 12288 | .015 | 11328 | 57 | 219.688 | 62.000 | 38.45% | 71.78% | 53.57% | 0 |
| 12288 | .015 | 11328 | 75 | 219.688 | 80.000 | 20.33% | 63.58% | 46.60% | 0 |
| 12288 | .015 | 10240 | 57 | 273.249 | 87.416 | 60.76% | 68.01% | 48.62% | 1 |
| 12288 | .015 | 10240 | 75 | 273.249 | 105.416 | 49.89% | 61.42% | 42.41% | 1 |
| 16384 | .01 | 11328 | 57 | 340.006 | 114.436 | 62.61% | 66.34% | 43.16% | 3 |
| 16384 | .01 | 11328 | 75 | 340.006 | 132.436 | 52.64% | 61.05% | 38.24% | 3 |
| 16384 | .01 | 10240 | 57 | 342.061 | 129.566 | 63.43% | 62.12% | 41.58% | 7 |
| 16384 | .01 | 10240 | 75 | 342.061 | 147.566 | 53.91% | 56.86% | 36.67% | 7 |
| 8192 | .03 | 11328 | 75 | 167.074 | 80.000 | 18.24% | 52.12% | 47.01% | 0 |
| 8192 | .03 | 10240 | 75 | 205.189 | 81.898 | 44.23% | 60.09% | 43.62% | 0 |

主候选8192/.025/10240、S1=75时，9117/10000 local blocked，9009 aggregate空闲
够新增页，8860可立即全source合并，3394条 individual latency减半。
所以 **P95 proxy下降60.49%不等于60.49%的请求都减半**；每请求减半率仅33.94%。
全source packing是一个可行构造，不是ILP最优，也不是当前KV-only必然选择的动作。
更长Conv虽可扩大等待，也更容易 aggregate不够、copy peak不够或延迟自然释放；
16384的P99 proxy反而明显恶化，不推荐直接跳到最长输出。

## 6. 18档敏感性：50%不是保证

共同8192/.025/10240，baseline Code service57s；每格是 control5s / 10s 下
Code latency **条件 P95下降百分比**。完整mean/P99、unsafe计数在JSON，不删负档。

| Conv TPOT proxy | S1=57s | S1=75s | S1=90s |
| --- | ---: | ---: | ---: |
| 15ms | 61.95% / 58.89% | 50.91% / 47.84% | 41.70% / 38.64% |
| 20ms | 69.38% / 66.91% | 60.49% / 58.02% | 53.09% / 50.62% |
| 30ms | 60.20% / 58.44% | 53.87% / 52.11% | 48.60% / 46.84% |

30ms 时 E[N]=6.149、offered超预算3/10000，treatment尾也开始排队；不能将
“Conv越慢越好”或“RPS越大越好”当作规律。15ms/S1=75/M10已有P95未达50%，
S1=90也有失败档。原预算8192/.025/S1=75的49.81%是边界，不应四舍五入宣称达标。

## 7. 真实 trace、SLO与样本量的约束

Code rate .0006RPS、S1=75s，在无排队、无限容量Poisson模型中其他Code驻留概率
proxy为4.40%；.001约7.23%，.002约13.93%，.004约25.92%。真实排队和预算反馈
会使此近似失效，baseline的驻留也不同；**不能按没有其他Code筛选正式结果**。
应保留全部碰撞和aggregate真不足请求，只把分类用于机制解释。

稀疏Code要付出长trace代价：单一trace期望100 Code需要约46.3h(.0006RPS)、
27.8h(.001RPS)或13.9h(.002RPS)，每个策略都要执行。旧4h allocation无法直接
容纳长达此量级的五策略最终实验。可以预声明多个独立seed/固定时长segment；
同seed配对重放不是新增独立Code。小样本probe只能验证机制，不称稳定P95/P99。

8192输出在20ms/token下service164s，**仅执行就超过旧120s SLO**。必须保留120s
全请求/Conv失败数量，不能偷偷改SLO让数字变好。若未来确需长输出任务专用SLO，
须在新协议中单独声明、标明不与旧120s结果同口径；本轮没有新增SLO。
Code稀疏且Conv服务长，优化Code也不能推出全请求mean/P95下降50%。

## 8. 验证、复现与下一动作

```bash
python3 kvcached/experiments/dp_opt/stress_trace_envelope.py \
  --trace evidence/agentic-trace-run-20261021/live-trace-245583/trace.json \
  --output evidence/agentic-trace-run-20261021/stress-trace-envelope-20261024.json \
  --samples 10000 --sensitivity
cd kvcached
python3 -m pytest -q tests/test_dp_opt_stress_trace_envelope.py tests/test_dp_opt_workload_profile.py
```

原始派生结果：顶层`evidence/agentic-trace-run-20261021/`下
`stress-trace-envelope-20261024.json/.log`、`stress-trace-envelope-initial-seven-20261024.json`。
独立代数验证`stress-trace-envelope-verification-20261024.json`：15+18档、762项
checks通过；专项测试17通过，含200例随机block-owner/event逐状态对照；另含
trace/fairness/tail/watcher/rate的相关回归77通过，日志`tests-stress-trace-design-related.txt`。
首次unsafe
fixture太小、未触发超预算的测试失败保留在`tests-stress-trace-envelope-first-failure.txt`；
修正fixture后通过，现有asyncio_mode配置warning保留。不把CPU核验当GPU证书。

模型只给出 offered M/G/infinity人口、count-balanced placement、独立均匀年龄、
96MiB页owner释放以及copy-before-release目标peak。它忽略未来Conv补入、真实
baseline公平门控、迁移后的长期placement反馈、Conv slowdown与混合compute。
原长度对照的Conv service proxy13.46s也不等于实跑19.3s；不是245583复现。

下一步先完成247027原配置复跑并严格归档/配对分析；再按新协议测8192输出的实际
Conv service、混合Code service、全source多请求copy峰值及真实控制成本，才冻结
新RPS/trace。当前没有新GPU pilot或“至少50%”实测结论，整体研究目标未完成。
