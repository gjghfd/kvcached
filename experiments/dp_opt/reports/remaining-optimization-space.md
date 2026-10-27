# Code-primary候选的联合优化空间

用户已选择优化2：Code释放后通过ILP回平衡，而非强制历史home或等待新请求补齐。
实现与六策略实测见`release-rebalance-status.md`，算法/安全部署在
`../docs/release-rebalance-protocol.md`。249646已完成并独立验证，释放后回迁正确，
但尚未提高原KV-only净收益；后续优先选择性触发与小人口性能校准，不把实现当收益。

基于247429/case0的终态原始数据，不是新算法实测收益或全局最优结论。
数据来自`evidence/kvonly-stress-campaign-20261024/best-optimization-space-247429-20261025.json`；
相关五份原始文件与原终态SHA一致。原活动248507版本当时不修改，现该旧作业
失败已完整归档，剩余旧网格/heldout由249646新方法实跑目标取代，不当作通过。

## 已确认的瓶颈

1. **Code联合prefill/执行时间有明显差距**：两条相同采样Code的serial reference
   平均TTFT25.342s，混合trace为55.688s，差30.345s。串行参考是无Conv请求时的
   对照，不是带背景负载可达到的硬下界；必须补同prompt、不同Conv人口/布局的
   联合prefill/decode profile，不能将全部差值归于attention DP不均衡。
2. **纯KV复制不是主要TTFT瓶颈**：本次迁移1330642944 bytes，完整迁移事件
   0.088219s。全trace的control+preplan累计6.129s，不等于一条Code的关键路径。
   优化copy有工程价值，但不能解释或消除约55s的Code TTFT。
3. **Conv的23s尖峰是真实aggregate不足**：`agentic-20261024-1-42`排队23.230s。
   首个pending观测两GPU mapped9.5625/9.46875GiB，全局空闲0.96875GiB，小于
   新Conv逻辑需求1.17334GiB；全部该请求pending样本均aggregate不足，Code仍在
   prefill（all_decode=false）。换位置本身不能制造额外容量，不是遗漏了可行迁移。
   这段queue对合并平均TTFT贡献0.430s，但不能删除异常点或声称控制fastpath就能
   回收全部时间。缩短Code占用时间、提前释放或改变经验证的容量管理才可能减少它。
4. **10GiB带有结构性Conv挤占**：按当前full-reservation和96MiB/page、64blocks/page
   几何，两条Code均需100pages=9600MiB；最短采样Conv也需11pages=1056MiB。
   二者合计10656MiB=10.40625GiB，已超过每GPU10240MiB；因此理想无碎片情形
   下也不能在Code所在卡保留任一完整预留的采样Conv。实际共享页、空池、碎片和
   复制峰值仍需真实probe；这个计算不是allocator充分可行性证明。

## 诊断性的剩余空间

下表只回答本case的TTFT距离fair减半还差多少，不替换已冻结Code端到端P95主目标。
2Code不足以确认任何稳健高分位或最终50%收益。

| TTFT指标 | 当前KV-only(s) | 本case fair的一半(s) | 尚需下降(s) |
|---|---:|---:|---:|
| Code平均 | 55.688 | 48.030 | 7.657 |
| Code P95 | 56.067 | 55.653 | 0.413 |
| 合并平均 | 2.891 | 2.411 | 0.480 |
| 合并P95 | 9.199 | 8.850 | 0.350 |

Code平均还需降低当前值的13.75%，合并平均还需16.59%；不是已实现的额外收益。
不能围绕2Code的0.413s差值微调窗口/seed/threshold，制造50%确认。

## 建议的验证顺序

1. **联合profile先行**：复用同prompt与默认sparse/32step，测Code-only、Conv decode
   背景、不同Conv布局和prefill chunk预算的TTFT/TPOT。分别量化Code缩时及老Conv
   的损失，再决定跨tenant的执行协调；不能假设增大或减小chunk都同时改善TTFT和TPOT。
2. **选择性Code释放回迁**：主动无准入ILP已在249646真实执行并独立核验，但新增
   Conv完成触发和单模型N128外推未带来增量净收益。下一轮只针对容量迁移引起的
   displacement，在Code释放后比较不动、自然到达补齐与重平衡成本；Conv完成仅
   重试未完成的恢复任务。保护老Conv剩余完成时间/TPOT与真实copy峰值/default
   sparse边界；不能关闭空rank EP或强制历史home来制造收益。
3. **联合收益门控**：KV-only会将性能cost表归零；现有ILP的默认性能项仍是单模型
   N128 surrogate，wait-aware本轮关闭。候选必须比较等待、立即重排、释放后回迁，
   同时计入Code TTFT、Conv准入tail、老Conv剩余decode、pause/复制成本。先使用
   联合profile校准，不能拿单模型外推或任意权重声称保证联合tail最优。
4. **中间容量区间作为独立机制控制**：后续可预声明10.5–10.75GiB附近的预算，
   争取Code卡保留一条较短Conv，而balanced两条Conv时仍不能准入Code；是否可行
   必须按实际请求/共享页/复制峰值probe。不能只增大预算：已测11.0625GiB的普通
   baseline直接准入，迁移机会消失。同新budget下全部baseline必须一起重跑。

旧17case/heldout未全跑，剩余部分被新方法目标取代；249646完整六策略已终态归档。
除原型释放后回迁已实跑外，以上其他修正仍是下一轮候选，未实施/提交，不改SLO或
选优窗口，未commit/push。联合选择需同时报告Code、Conv、合并TTFT及老Conv
latency/TPOT代价，不以Code-primary最佳等同联合最优。
