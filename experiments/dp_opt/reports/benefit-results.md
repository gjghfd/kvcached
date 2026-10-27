# 跨模型attention-DP/KV调度：原型收益结论

2026-10-21 12:06，Asia/Shanghai。原型收益对照已运行并独立验证，**不等于**
全线上负载、联合性能校准、全请求全局最优或设备OOM评估全部完成。执行状态与
逐job性能表由`benefit-status.md`维护，本页只给可支持的结论和下一步。

2026-10-22状态更新：固定32K Azure长度/Poisson四策略trace237899已运行并独立验证，
ILP净收益为负，详见`trace-benefit-results.md`；100K SWE-smith+AzureConv A800场景仍
校准，正式四策略尚未执行，详见`agentic-trace-results.md`。旧机制/确认结果不替代
open-loop trace，亦不能把固定population吞吐当可持续request rate。

用户随后明确主场景应是**全部长KV、总空间够、当前单卡不够、普通调度排队而重放置
准入**。237068首轮已独立验证该主见证：总空闲576MiB够新A384MiB但任一卡仅288MiB，
迁移一条B释放192MiB后准入。A queue15.76→1.50秒、吞吐+19.18%，但20秒goodput退化；
默认32step六配对237171也已验证336请求：A queue19.64→2.81秒、吞吐配对均值+10.25%，
但一轮略负吞吐、20秒SLO全失败且ILP未优于KV-only。详细正负结果见`benefit-long-kv-results.md`。

## 已完成的端到端链路

同两GPU驻留两个独立Qwen3-30B-A3B-Instruct-2507 tenant，同checkpoint、独立
weights/KV/EP groups，每模型DP2/EP2/attentionTP1/PP1。两个模型独立到达流，经
controller协调准入；不是生产OpenAI HTTP router。所有baseline容量不足时排队，
不丢请求。no_move、own_model、KV-only、ILP在相同case内实跑配对比较。

算法暂停两模型的全DP组，读取page-owner/真实mapped/free-slot状态，枚举新请求rank，
对受限候选解copy-before-release staged ILP，执行真实KV/runner/client迁移，释放源页，
同epoch预留新请求再resume。完整bootstrap→最后completion窗口包含queue、solver、
pause/resume、host-staged复制和prefill。输出逐请求与串行greedy token匹配。

保持sparse captures1,8,16,32,64,128,256、FULL_DECODE_ONLY、compile0。
没有关闭空attention rank的EP执行或改dense。控制实验固定所有模型/策略interval1；
默认32实验另行标记，所有负结果独立保留。性能代价仍是N128单模型capture-aware surrogate外推，
不是测得的双模型联合关系；ILP仅对候选2/model/rank、最多2moves的有限模型最优。

## 核心证据

| job / 场景 | ILP vsno_move吞吐 | ILP vsKV-only吞吐 | 可支持的结论 |
| --- | ---: | ---: | --- |
| 235320 / capacity，256输出，3repeat，默认32 | +48.54% | +6.13% | 容量重叠收益；两策略同动作且cold pause，不归因性能项 |
| 235647 / tradeoff，256输出，3repeat，默认32 | -2.92% | -17.47% | 不同placement但长pause与复制成本使整体负收益 |
| 235712 / tradeoff，256输出，3repeat，interval1 | +23.38% | +4.48% | 初次正均值，小样本区间含零 |
| 236751 / early，256输出，6repeat，interval1 | +23.30% | +3.38% | 正均值但有负repeat，95%t区间[-1.56%,8.31%] |
| 236751 / late，256输出，3repeat，interval1 | +5.99% | +1.64% | 更短收益窗口，小收益且有负repeat |
| 236906 / long，512输出，6repeat，interval1 | **+28.49%** | **+6.68%** | 六次均正，95%描述性t区间[4.12%,9.24%] |
| 237068 / long_kv，512输出，3repeat，interval1 | **+19.18%** | +1.17% | 总够/单卡不够、跨模型准入；KV/ILP同动作，老请求SLO退化 |
| 237171 / long_kv，512输出，6repeat，默认32 | **+10.25%** | **-5.07%** | 默认控制仍能更早准入；一轮vsno_move略负，SLO0/84 |

全部为**各repeat比率平均**，不能用表中跨job绝对值作因果消融。236751与236906
同节点但输入/预算/时段不同，long不是单纯输出长度消融。区间是少量重复描述，不是
多节点/多真实工作负载的泛化证书。七job共2196请求、全部完成与token/ownership验证。

最明确的不同placement收益在236906：KV-only移A54MiB，A2:9触发replay16；ILP移B96MiB，
A3:8/B3:3保持replay8。真实copy分别约0.29/0.50秒，ILP总control约0.62秒仍已计入。
吞吐426.313→454.702tokens/s；平均请求latency20.210→18.931秒，配对latency sum-6.32%。
但A到达TTFT .588→.854秒，配对增加45.28%；算法不是同时优化所有请求延迟。

20秒SLO通过数KV-only36/108→ILP106/108。goodput大增与阈值临近20秒有关，不能
泛称5倍容量收益；posthoc22秒时两者全过，goodput优势回到吞吐量级。主SLO未改。
这段仅指236906辅助场景；237171长KV主场景两种迁移策略20秒SLO均0/84，不能泛化。

## 机制和限制

- sparse graph档位的真实统计符合选择机制，但默认32的负结果证明档位本身不保证收益。
- 更多remaining decode有助于摊销迁移；晚到达只避开82–88个replay16ranksteps，长输出
  避开966–968个；这些是整trial replay统计，含dummy，不是纯attention kernel时间。
- 固定每step DP共识避免本阶段的秒级pause长尾，但有同步开销，所有策略均承担。
  默认32实现仍默认不变；需同硬件严格控制消融才能量化其因果作用。
- 当前预算是两tenant实际mapped-KV总上限，不是整卡OOM/shared physical handle交换；
  fullprompt+maxoutput预留和整reservation host-file复制不是优化后长context数据平面。
- cost需联合layout/context校准；固定.475秒/1GiB/s不是校准。未来到达、context异质性、
  SLO保护、exact packing和更多rank未由本阶段证明。

## 下一研究阶段

1. 先将目标/约束显式加入老请求SLO与迁移保护；长KV准入收益伴随老请求/SLO退化。
2. 同GPU/稀疏graph下测联合A/B layout/context cost及实际pause/copy带宽，导入joint ILP表。
3. 在长prompt主见证后增加多轮独立到达、渐进KV增长与未修改native-vLLM调度对照。
4. 扩大候选/DP rank与实际allocator压力，不把候选最优或受控预算当作真实全局最优。

证据索引在`evidence/benefit-20261020/`；源代码/协议/结果属于用户fork main。
本轮未commit/push、未重启Aegaeon；所有研究job已终态，无遗留allocation请求。
