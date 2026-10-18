# Qwen 多模型 DPA rank-local KV 腾挪实验设计

## 当前依据与文档边界

本页维护 placement idea，不重复实验 setting/结果。主 serving baseline 使用 CUDA graph，
已完成 motivation 的结论与有效性边界见 `../reports/motivation-summary.md`；eager 留作历史对照。
设计协议见 `motivation-protocol.md`，状态见 `experiment-matrix.json`。
下文按日期保留 early deployment/容量推导，不能将旧“当前”配置当作 graph 主配置。
后续 request/KV ownership 迁移、双模型干扰和 placement 收益仍未实现/测量。
此文的 `evidence/` 路径相对于外层 ParaCloud 工作区，不属于 kvcached Git 仓库。

## 1. 原始研究问题

本设计依据 2026-10-14 的原始聊天，而不是把当前双模型并行 smoke 当作研究目标。

用户提出：多模型 colocation、KV 按需分配、attention 使用 DP 时，完全均衡不一定最优。为了模型 A 的长请求，应当允许迁移模型 B 在不同 DP-attention ranks 上的请求，主动为 A 创造 rank-local KV headroom，即使 B 原先并不存在负载不均衡。

研究对象是正在执行的请求及其 KV ownership；不是更换 API 模型名，不是仅路由新请求，不是迁移专家权重，也不是把长请求改成跨 rank context parallelism。

可检验假设：在共享 KV 池、可达拓扑和每模型 SLO 约束下，跨模型协调能降低长请求的准入等待及增长触发的抢占，收益足以抵偿 KV 迁移、attention 不均衡和 EP 同步开销。这个假设可能失败，不能把“越不均衡越好”作为前提。

2026-10-17 用户指出Aegaeon模型切换会把其他模型KV移到CPU，当前数据面不能自然产生本idea要求的跨模型GPU KV容量竞争。核查本地SimpleScheduler/Worker/BlockManager后确认：不改这条机制，Aegaeon不适合作为原始headroom idea的主要验证后端，只作为参考或独立对照。双原生vLLM服务虽已同时驻留GPU，其静态私有cache也不自动共享容量；迁移B并减少其逻辑blocks不能让A扩大自己的cache。后续必须提供共享物理页池或真实elastic cache容量转移，并先证明B释放的物理容量可被A重新分配。attention-DP不均衡motivation继续保留，但不当作KV收益验证。

## 2. 已验证与待实现的边界

已完成的单模型 motivation 见 `../reports/motivation-summary.md`；双tenant驻留和独立client
baseline见 `../reports/dual-qwen-results.md`。本页不复制性能表、作业进度或正确性统计。

两租户可复用同checkpoint，但请求/KV/SLO/EP组独立，不等于GPU权重去重或跨租户前缀共享。
原生DP/EP部署正确性和KV按需增长/回收已验证；这不证明源rank释放的live请求物理页已被
目标模型复用。跨rank request/KV ownership迁移、容量handoff、跨模型placement收益仍未实现/测量。
Aegaeon Simple模型切换baseline的机制与部署边界仅维护在该checkout的baseline文档。

## 3. Qwen 的容量模型

使用实际已下载的 `Qwen/Qwen3-30B-A3B-Instruct-2507/config.json`：48 层、4 KV heads、head_dim=128、BF16、128 experts、top-8、无 sliding window、配置最大位置数 262144。部署的实际上下文能力和性能仍需重新验证。

在 TP=1、PP=1、普通全量 GQA KV、KV dtype=BF16 下：

`KV bytes/token = 2 × 48 × 4 × 128 × 2 = 98,304 bytes = 96 KiB`。

现有运行日志的每个 16-token composite block 为 1,572,864 bytes，与该推导一致。统计输入和已生成的 KV token，按实际页大小向上取整，不包含权重、激活、workspace、通信缓冲或分配器元数据。

| KV token 总数 | 单请求 KV |
| --- | --- |
| 4,096 | 0.375 GiB |
| 16,384 | 1.5 GiB |
| 32,768 | 3 GiB |
| 65,536 | 6 GiB |
| 131,072 | 12 GiB |
| 262,144 | 24 GiB |

专家 BF16 payload 估计为 `48 × 128 × 3 × 2048 × 768 × 2 = 54 GiB`；其他参数约 2.87 GiB。若 attention TP=1、DPA=P、EP=P，专家均匀分片，无冗余专家，每 rank 单模型权重约为 `2.87 + 54/P GiB`：

| GPU / DPA / EP 数 P | 每 rank 单模型 | 每 rank 两模型 |
| --- | --- | --- |
| 2 | 29.87 GiB | 59.74 GiB |
| 4 | 16.37 GiB | 32.74 GiB |
| 8 | 9.62 GiB | 19.24 GiB |

P=2 的裸权重已由新 EP 作业实测验证为单模型每 rank 29.87051 GiB；两个服务加各自 1 GiB KV 及其他开销后，每卡采样最高 66,675 MiB。P=4/8 仍是布局估算，不是已验证的驻留显存；必须实测扣除 attention workspace、EP 通信、padding、CUDA context、激活峰值和临时 KV 双份后确定可用池容量。H200 上完整双模型能否驻留同样需要实测；不要把裸卡显存视为 KV 预算。

两个模型同 checkpoint 时，CPU 文件 mmap 复用允许，但 GPU 权重去重必须作为独立、明确开关，所有对照使用相同配置；不把模型相同带来的跨模型合批当成迁移收益。第一轮可禁用 GPU 权重去重，使容量更接近独立模型。

## 4. 建议系统路线

2026-10-17 用户决定实现后端正式改为 native vLLM + kvcached，Aegaeon保留独立参考。
本地及远端kvcached现均在用户fork的 `main` 开发，`origin=git@github.com:gjghfd/kvcached.git`，
`upstream`保留原项目，当前基线 `04ccd1d`；Aegaeon仍在`lch/dp_opt`，远端安装在独立venv，
不覆盖Aegaeon的dirty修复或依赖。已补DP-rank namespace/device适配，启动脚本和真实token
benchmark位于 `kvcached/experiments/dp_opt/`。编译作业225943完成并通过4项namespace测试；
双卡225937因ninja PATH缺失失败，修复后225965在两A800上COMPLETED/0:0，16×128输入/256
输出完成（4096token/80.829秒），并通过同rank串行/并发算术一致及四KV池共驻留检查。
池mapped值96→192→96MiB，85/162采样四池高于空闲值。当前只验证并发baseline，
尚未证明迁移B后A复用物理容量；本地证据 `evidence/kvcached-dual-qwen-225965/`。

真实 Qwen 双服务 DPA=2/EP=2 小上下文正确性与并发基线已经由作业 `223641` 建立，下一步做 shared-pool/migration 原型。可以复用已安装 vLLM 的 Qwen kernel 与 parallel config；集群版本显式存在 `data_parallel_size`、`enable_expert_parallel` 和 `all2all_backend`。新硬件或 P=4/8 布局仍需各自真实验证，不能由这次两卡结果推断。

2026-10-17 的候选后端调研：OSDI ’26 **Prism: Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning** 及其开源balloon driver **kvcached** 支持多个活跃模型/live KV同卡驻留，并通过VMM把空闲物理页归还以供其他模型分配；空闲模型也可sleep/evict，因此不是所有模型永久驻留。它提供我们所需的弹性容量机制，但不等于已有Qwen attention-DP迁移实现。Prism本身已有headroom-aware模型placement，研究区别需落在固定EP专家布局下的跨租户attention/KV home调整及其compute代价。详细边界、源码行号和验证顺序在 `kvcached-prism-review.md`；本地 `kvcached/` 当前基线为 `04ccd1da0653d251622122d0695d72bb288fb8ec`，已在用户fork的`main`开发，已通过225965并发baseline；迁移实验仍未实现。

两模型 A/B 都跨同一组物理 GPU，分别持有自己的 DPA/EP 组和执行状态，同一 GPU 上的两个租户从同一个物理 KV 页池分配。两个独立 vLLM 进程的静态私有 KV cache 不是 shared pool；必须提供可跨租户回收、复用的 GPU 页分配层。

先使用 2 GPU 做迁移正确性与机制验证，优先 4 GPU 做论文主体实验，8 GPU 做扩展性。4 卡专家分片后两模型裸权重约 32.74 GiB/rank，比两卡原型更容易容纳长 KV。固定一次对照中的 GPU 型号，不混用 A800/H100/H200 的吞吐曲线；记录实际 NVLink 拓扑。

先做最简安全迁移：在 decode step 边界冻结候选请求，导出实际 live KV 页及请求状态，目标 rank 分配并复制，确认可继续执行后原子切换 ownership，再释放源页。失败则保留源端并回滚目标，不允许丢 token、重复 token、丢 RNG 状态或破坏流式响应。

迁移仅改变 B 请求的 attention/KV home；B 的源 rank 仍须保留并执行 EP 专家 shard、参加必要 collective。不能通过停止源 rank 或清空整个模型来获得虚假的空间和同步收益。

第一版可通过节点共享 pinned CPU cache 做中转，作为正确性原型；记录 D2H/H2D 的实际字节与时间。最终至少比较 NVLink/P2P 实现与 CPU 中转，不将 CPU swap 的成本忽略。请求迁移成功必须证明释放后的物理页可被 A 再分配，而不只减少逻辑 block 计数。

## 5. 实验一：最小反例和负例

用真实 KV 页池建立两 rank 状态。以下数字是可控实验预算示例，不是当前 allocation 的测量值：每 rank 池容量 C=12 GiB，B 各占 8 GiB，可用空间 F=(4,4) GiB。A 新长请求的目标 KV footprint 为 6 GiB，另预留 1 GiB。

- 聚合空闲为 8 GiB，但任何一个 rank 的 4 GiB 都不能单独容纳 A 的 6 GiB。这是跨 rank 容量分散，不是 PagedAttention 的物理连续内存碎片。
- 把 B 的一个约 32K KV-token 请求（3 GiB）从 rank 0 迁到 rank 1，得到 F=(7,1) GiB。
- A 在 rank 0 占 6 GiB，A/B 最终占用为 (11,11) GiB，各剩 1 GiB 增长余量。B 的请求分配更不均衡，但 A 的准入可立即完成。
- B 迁移前完全均衡，因此只基于 B 过载或 imbalance 的触发机制不会主动做该迁移。

这里的 6 GiB 必须明确是当前已需分配 KV，还是已知未来长度的 reservation。先测长 prompt 准入，后测已在生成的 A 请求增长；不能把假定未来 output length 的 oracle 当作线上可知信息。

实验状态必须包含真实、可继续生成的 B 请求。可以在统一 token 边界构建受控快照，但各对照必须从同一状态恢复，不能只为提出的方法暂停背景请求。

同时运行负例：A 本来就能放进某个 rank；全局 KV 真正不足；B 目的 rank 放不下迁移对象；B 的 SLO slack 太小；多个 A 长请求同时到达。这些场景应不迁移、退化为基线或有明确排队，而不是不断腾挪。

## 6. 实验二：主工作负载

使用至少两类场景：

1. 新长 prompt 到达，所有 A 可用 ranks 均缺少准入空间。
2. A 已有长请求持续生成，当前 rank 将接近 KV 上限；迁移 B 的其他请求以保留 A ownership。

生成 A 长请求和 B 短/中请求的混合流量，先从 8K/16K/32K/64K context 与 128/512/2048 output 做小矩阵，再扩展 128K。确认 tokenizer 产生的真实 token 长度和 context+output 上限；synthetic 负载关闭 prefix caching，使用不同内容，避免大量相同前缀让 KV 压力消失。

扫描长请求比例 1%/5%/10%/20%、A:B 流量比例 1:1/1:9/9:1、共享池使用率 50%/70%/85%/95%、平稳与突发到达。使用 open-loop 到达和固定种子，避免闭环客户端把排队或失败自动变成较低负载。每个核心点至少三次独立重复，单独报告 warmup 与 steady-state。

受控容量限制可以帮助快速呈现边界，但必须显式标注 pool cap，并补充实际自然显存预算结果。H200 上空闲较多可能弱化 gap，需要提高活跃 KV token 数；不能人为给基线更小的池。

## 7. 对照与消融

所有策略使用相同权重布局、同一物理页池、请求 trace、迁移数据面、memory cap、prefix 配置和通信 kernel；不要把“基线没有共享池/迁移、我们有”作为主要调度对照。

建议主对照：

- 新请求的 request-count least-loaded，无主动迁移。
- 新请求的 rank-free-KV 最大者路由，无主动迁移。
- 各模型独立的 KV-aware 均衡和迁移；A 控制器不重排 B。
- 全局 KV-aware 腾挪：跨模型 greedy bin packing / virtual pressure，为目标请求创造空间，但不考虑 SLO 与同步代价。这是强对照，而非仅与 round-robin 比。
- 跨模型 cost/SLO-aware headroom 策略。

将 CPU offload 和迁移后的 recompute 作为可用容量处置替代；动态 CP 若未真实实现，则仅讨论其不同解决方向，不伪造对照数值。历史聊天中的 Llumnix/Mell/CrossPool/NanoCP/xLLM 作为相关机制线索，不把“受启发的策略”称为完整复现原论文，更不据旧搜索断言不存在同类工作。

消融包括：跨模型协调开关、主动预留开关、只迁一个请求/greedy 多请求、迁移字节代价、B 的 SLO 约束、EP 同步代价、future length oracle 与估计误差、cooldown/反向迁移限制。

固定 GPU 数与策略后再比较 2/4/8-rank 扩展；不能仅用跨 GPU 型号的结果宣称调度扩展性。

## 8. 最小调度器

候选 A 目标 rank g 的需要释放量：`deficit_g = max(0, current_or_predicted_A_KV + margin - free_KV_g)`。

从 g 上其他模型的可迁移请求中选择集合 S 和合法目的 ranks，满足源端能释放至少 deficit、每个目的 rank 有空间容纳实际页加增长余量，且各模型 SLO/collective 约束仍可接受。

第一版用 greedy：低迁移字节、足够释放空间、目的 rank 可行、B 有足够 slack。对多个候选 g 比较 `estimated_transfer_time + request_pause + attention/EP_penalty + SLO_penalty`，不要先预定一个人为偏好的 rank。

腾出空间后限制 B 的新请求重新填入目标 rank，维持短期 reservation；不能让独立均衡器立刻把 B 迁回。设置 cooldown 和最大迁移次数，并记录反向迁移与请求 starvation。

先用已知 synthetic output length 做 oracle upper bound，再测试只知 prompt 长度/最大输出预算的策略及预测误差。必须区分真实当前 free bytes、reservation 和页复制期间的 transient duplication；全局最终状态能放下并不保证迁移过程能放下。

## 9. 指标和日志

主指标是同时满足 TTFT 与 TPOT SLO 的 completed-request goodput，并分别报告 A、B 和全局结果。不要用 A 更快掩盖 B 被饿死。

记录 A 长请求的准入等待、P50/P95/P99 TTFT、各模型 TPOT/最大 token 间隔、完成率、超时/OOM/抢占/recompute、output token 吞吐及 goodput。流式客户端采集 token 时间，而不是拿非流式 HTTP 总时间冒充 TTFT/TPOT。

数据面记录迁移 live bytes、copy bytes、源/目标 rank、暂停时间、ownership commit、源页实际释放时间、物理内存可复用验证和峰值显存。EP 路径记录各 rank attention 时长、MoE/collective 及最长 step critical path，以检验刻意不均衡的代价。

每 rank 每模型记录 active KV bytes、free bytes、reserved bytes、active requests；同时画 `sum(free_KV)` 与 `max(free_KV)`。对等待请求标记 `max(free_KV) < demand <= sum(free_KV)`，但还要检查权重可达、目的容量、collective 和 transient duplication，不能把该不等式视为必然可解。

正确性使用多种 prompt、长上下文检索和迁移前后续写，并和相同 GPU/同一 DPA+EP 配置的未迁移基线比较。保留页内容/位置与 ownership 检查；跨并行拓扑的浮点差异需要单独解释，不能直接套用当前 TP=1 的短算术 token 对照。

## 10. 成功、失败与下一步

可称为主机制成功：原本 B 已均衡、A 因 rank-local headroom 不足而等待；实际 B 跨 DPA-rank ownership 迁移释放同 rank 物理 KV 页；A 能使用这些页；在计入复制和同步代价后提升 goodput，且 B 的 SLO 受控。

若仅两模型跑通，或仅一个模型吞吐增加，或 simple global greedy 与复杂策略表现一样，则不足以支持所设想的贡献。若 EP straggler/迁移成本抵消全部收益，需要收缩适用区间或重新设计，而不是删掉负例。

执行优先级：真实 DPA+EP 两模型共址正确性 → 共享 KV 页池 → 迁移/ownership/释放正确性 → 两 rank 最小反例与负例 → 四 rank 强对照和消融 → 八 rank 扩展与独立第二 checkpoint。最后一项需另行明确模型与下载授权，当前不新增权重或改动正在排队的作业。

本轮在线查询未返回可引用的检索结果；参数和实现结论来自已下载 checkpoint、已安装 vLLM 源码、Aegaeon 源码及原始实测。正式写论文前需重新核对相关工作的最新机制及实验口径。
