# 异构 Conv/Code Poisson trace：下一阶段设计

2026-10-21。依据 `../reports/mixed-workload-profile.md` 的全量长度分析；
**planned，未执行新的GPU trace**。Code长输入触发、老Conv长驻留调整，不要求Code
长decode。替代此前双Conv压力trace的下一阶段工作负载，不改写237899负结果。

## 1. 工作负载与冻结规则

两个独立Qwen3-30B-A3B-Instruct-2507 tenant，原DP2/EP2/attentionTP1共址；
model0=Code，model1=Conv。checkpoint可复用，权重/KV/EP组仍独立。
维持sparse captures `1,8,16,32,64,128,256`、FULL_DECODE_ONLY、compile0、
默认32step控制、max_model_len32768。1M仅分析尺度，不是此次启动参数。

第一阶段候选：

| 项目 | 预声明设置 |
| --- | --- |
| Code长度 | 完整Azure Code成对(P,O)，条件P>=6000，保留原O |
| Conv长度 | 完整Azure Conv成对(P,O)，条件P<2048且O>=256，保留原O |
| 到达 | 独立Poisson，绝对arrival在trial前冻结，不因完成/迁移时刻重置 |
| rate候选 | Code .1req/s；Conv .5/.75/1.0req/s，中心候选.75 |
| KV预算 | 2112MiB/rank受控总预算，含两model pools，非整卡OOM |
| 比较 | no_move / own_model / KV-only / ILP，保持same input、预算、FIFO |
| synthetic text | 按Azure长度构造，声明非原prompt，固定greedy生成设置 |
| bootstrap | 不做同步长decode屏障，不等见证后再到达；连续stream形成背景 |

先测几个真实P/O组合的prefill、decode和控制成本再定正式rate；上表不是已证实
可持续的GPU速率。保存完整分布作为代表性对照，条件slice只解释机制压力。
正式trace在运行前声明seed、warmup、measurement/drain窗口与SLO，至少3个paired
seed；不根据结果筛seed/移除不利请求/改SLO。代码所用长度、上限和overflow处理
必须写入trace文件，不能只在分析脚本里改变。

CPU近似中心候选约22.55个几何可恢复offered机会/小时，30次期望机会约80分钟；
控制/保守ILP会进一步减少。先短pilot验证安全/速率，再冻结长trace预算，事件
不足的seed仍记录。不要用数分钟trace强行宣称完整算法收益。

## 2. 准入与 wait-vs-migrate 决策

枚举新Code目标DP rank，再在全部模型可迁移请求上做staged ILP；Conv是主要候选，
不是给每模型固定隔离物理预算。保持真实页owner、pinned页、目的pool、源可释放
页和copy-before-release峰值约束。必要时给当前自身模型同样迁移能力。

必须增加明确的 **wait** 动作。对方案x和目标d的净收益估计可写为：

    G(d,x) = w_code * (W_hat_wait - C_control(x) - DeltaT_code(d,x))
             - sum_existing_i w_i * DeltaL_i(d,x)

`C_control` 包含snapshot/ILP/pause/RPC/copy/resume；`DeltaT_code` 包含布局相关
prefill/decode变化，不重复计control。`DeltaL_i` 包含老Conv的暂停和布局性能损失，
评价horizon应由其**剩余驻留**决定，不由Code必须长decode决定。统一到预声明的
latency/weighted-SLO目标，避免将只新Code的收益与所有老Conv的sum混成不一致目标。

若最佳估计G<=0或没有安全方案，等待自然完成，不反复pause重算。先做no-pause
容量/页owner预筛和rate-limit，实际执行前在安全点重核snapshot，超时或陈旧则
回退等待。剩余时间需来自预声明可用信息/在线估计；不能使用基线未来真实完成
时刻做oracle执行决策（离线oracle仅单独标注上界）。

单模型N128的posthoc线性系数不是小batch双模型cost。sparse capture跨档可能
不线性；performance表需要在本负载下校准，暂用surrogate须明确标注。与KV-only
比较same-snapshot计划，并报告相同动作比例；相同动作的运行噪声不证明性能项收益。

## 3. 准入、性能与正确性指标

- 每个blocked事件保存global free、各rank真实mapped/空槽、new增量、单请求是否
  空卡也放不下、owner页、候选及ILP plan。区分global不足与global够/local堵。
- 迁移前后及每copy stage验证峰值；记录实际可释放页，而非仅logical KV sum。
- 分模型统计queue/TTFT/TPOT/latency p50/p95/p99、完成数、输出tokens、drain、
  预声明SLO goodput；额外统计新Code准入加速和被搬老Conv延迟，不只全局平均。
- 拆分snapshot/solver/pause/RPC/copy/resume/prefill，比较自然等待的观测值及估计误差。
  长Code prefill可能阻塞Conv，计入原请求端到端指标，不仅比较KV可行性。
- 同trace完整drain、greedy token paired/reference校验、四worker共驻留与sparse
  graph验证。所有失败和负结果归档，不将丢弃请求/超时当吞吐增益。

推进门槛：先pilot确认global够/local堵事件和跨Conv可执行，后验证wait-aware控制
成本低于有用等待窗口，并保护老Conv；再跑完整四策略paired长trace。不能仅提高
rate让global持续过载，不能以full-reservation no_move冒充原生渐进KV增长baseline。

## 4. 更长输入的独立扩展

用户后续要求agentic 100K输入：`../reports/agentic-100k-dataset.md`已实测SWE-smith
一tool shard，导出10条严格>=100K真实历史prefix。它们是累计history重建的下一轮
输入，不是原始API时序或成功轨迹；用作Code长输入压力，Conv保持Azure长O。
100K约9.16GiB逻辑KV，若占半卡需约18.31GiB总KV预算，同时调整maxlen/pool且
先probe；目前仍32K/4GiB/2112MiB，不能直接运行。闭环agent/跨轮prefix复用另测。

Azure当前release P<8K。需要更大单次Code准入时，先用LongBench v1 RepoBench-P
（32K留512覆盖98.8%），再v2 Code的32–128K/128–256K分层；Conv保持Azure长O。
v2 MCQ参考答案不等于实际生成，固定短输出预算须先验证greedy输出长度/终止行为。

不能仅提高max_model_len：4GiB虚拟pool和2112MiB预算不足完整32K，更长context
需先调整两模型pool/预算并实测graph/activation/复制headroom。超限样本显式报告
unsupported，数据分析不截断，实验采样报告覆盖率。

1M KV单卡约91.553GiB，原双tenant权重+KV约151.294GiB，历史H200也放不下；
DP/EP不分片单请求KV。1M的官方DCA/V0/eager路径不等于当前V1+kvcached+sparse
兼容。需单独实现并验证分片/共址内存/backend，不能为了运行关闭默认CUDA graph。

## 5. 执行与归档边界

当前只CPU；此设计不表示新的GPU执行已发生。下一次实际执行前按外层
`docs/cluster-operations.md` SSH preflight、旧SHA备份、明确文件原子同步；不要覆盖
运行中checkout。只提交一个allocation，A800/H100/H200择实际健康空闲资源，
每300秒查到终态，归档SHA并独立复核再根据结果推进。用户fork main，不未经
新的发布授权commit/push；Aegaeon保持停止。
