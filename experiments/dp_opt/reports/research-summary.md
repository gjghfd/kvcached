# 跨模型 attention-DP / KV placement：算法与实验总览

快照日期：**2026-10-27，Asia/Shanghai**。这是当前研究的统一入口；各轮协议、
完整指标及失败记录仍以对应分篇报告为准。**容量迁移收益已有实测见证；新的性能感知
回迁 ILP 尚未证实稳定、独特的大幅收益。当前 1.2 压力实验仍在运行，研究目标未完成。**

## 1. 科研 idea 与实际实现范围

两个常驻模型共享 GPU 物理 KV 空间，但各自的 attention-DP 调度独立。短输入、长输出
Conv 请求持续占用 KV；稀疏到达的 Code 请求输入约 100K、输出短。即使两卡总空闲足够，
count load balance 也可能让两张卡都无法容纳单条长 Code 请求。通过移动包括另一个
模型在内的请求/KV-home，把零散空闲集中到一张卡，使新请求无需等自然释放。

Code 准入后，Conv 可能被挤到一个 attention rank，产生 graph padding 与 attention
失衡代价。Code 释放后，再判断迁移部分 Conv 回另一个 rank 的收益是否超过真实
暂停、复制和未来长请求再次准入的代价，不强制回到历史 home 或完美均衡。

当前 data plane 是 **native vLLM + kvcached**，不是 Aegaeon 的模型切换。
两独立 Qwen3-30B-A3B tenant 复用 checkpoint 文件，但权重、KV、request state、
EP 组独立；每模型 DP=2、EP=2、attention TP=1。研究只调整 request/KV-home placement。

相关设计：[placement-design](../docs/placement-design.md)、
[最初 staged ILP](../docs/ilp-scheduler-design.md)、
[当前 work-aware 回迁协议](../docs/workaware-return-protocol.md)。

## 2. Motivation 与可使用的性能关系

单模型 A800、默认 sparse CUDA graph 的确认实验（229748）固定请求数和长度，
仅改变 DP home。全局 batch128 的极端失衡吞吐损失为 **32.37% / 32.21%**；
batch256 为 **38.52% / 38.35%**，分别对应两个热 rank 方向。batch32 未通过预声明
30% 确认门槛。它们是稳定 decode cadence-derived loss，不是纯 attention kernel 占比。

空 attention rank 仍参与 EP/collective 和 graph replay；不能说其整卡完全 idle。
sparse capture 阶跃也影响性能。因此，中间不均衡曲线不支持对所有 batch、capture 和
模型使用一个普适线性系数。当前算法在**固定 tenant、population 和实测域**内使用：

`c_m,N = base_m,N + gamma_m,N(G) + beta_m,N * abs(A0-A1)/2`

- `G = sparse_ceiling(max(n0,n1))`，captures 为 `1,8,16,32,64,128,256`。
- `A_d = sum(context_tokens)`，按未来 decode 区间中点计算该 rank 的上下文工作量。
- `gamma` 是同 population 的实测 capture 分类 penalty；`beta` 是非负线性上下文
  失衡系数。比较同批请求的布局时，总上下文不变，公共项抵消。
- 对偶数 N、等长上下文 L，令 `I=abs(n0-n1)/N`，则 attention 失衡项为
  `beta_m,N*N*L*I/2`；同时仍须保留 `gamma` 的阶跃，不能仅拟合一个 `alpha*I`。

例：Conv N2 的 `gamma(8)=5.1994ms`、`beta=1.23197e-7s/token`；
N9 的 `gamma(16)=.9102ms`、`beta=5.78439e-8s/token`，各自均衡档 penalty 为零。
held-out-repeat 最大 contrast 误差分别约1.0263ms、3.8300ms；不是统计置信区间。
这些是客户端 decode cadence surrogate，不是 kernel 分解。

当前人口校准覆盖2/3/4/8/9；缺少5/6/7时保守延期。每个 population 的 context、
candidate peak-attention、capture 域均有 guard。另一 tenant 正在 prefill/decode 时
也延期，因为当前是另一模型 resident-idle 校准，不是完整 joint-active 校准。
新的 actual-attention-count 成本分析仅为离线诊断，**未部署**。

完整证据：[motivation-summary](motivation-summary.md)、[curve-results](curve-results.md)、
[当前校准与回迁结果](workaware-return-results.md)。实现：
`workaware_profile_cost.py`、`verify_workaware_profile.py`、`workaware_cost.py`。

## 3. 当前算法：准入与性能感知回迁分开

### 3.1 新请求准入：KV-only staged ILP

枚举新请求可用 attention-DP home（本布局为0/1），构建物理页 owner 快照，选择
包括跨 tenant 在内的迁移请求。ILP 同时选择 request placement 和复制阶段，约束：

- 每请求最多迁移一次，迁移候选和阶段数受限；不拆分一条请求到多个 attention home。
- 源页只有全部 owner 安全迁出且页可释放时才释放；固定页/pinned backing 保留。
- 目的页按保守 page rounding 计费；复制先于源释放，检查中间峰值与最终每卡预算。
- allocator 源 owner、实际 mapped bytes、复制完整性及提交前 snapshot 都必须复核。

容量优先、尽量少迁移。没有安全方案就排队，不丢请求。当前新 policy 的准入仍是
KV-only；**没有把未来 decode 性能目标直接用于新请求 prefill 准入**。
ILP 在有限候选集和总求解时间内选可行 incumbent；不能宣称整个系统全局最优，
也不能把保守 ILP 无解/超时说成真实 allocator 无解。

实现入口：`live_controller.py`、`live_scheduler.py`、`ilp_reference.py`、
`peer_transfer.py`、`worker.py`。

### 3.2 释放后回迁：work-aware + soft headroom

只使用已到达请求，按剩余输出完成事件和最多128步积分网格分段。枚举每模型的
headroom 状态 `{不保留, DP0, DP1}`，两个模型最多9个场景；每场景用 staged ILP
选择部分回迁。与 placement 有关的 request-second 目标为：

`min sum_m,k steps_k*N_m,k*[gamma_m,N(G)+beta_m,N*abs(A0-A1)/2]`

`    + N_active*F*z + sum_i N_active*(KV_i/BW)*move_i + R_future`

这里 `z` 是暂停 activation，固定成本只计一次；复制成本和未来区间成本均加入目标。
保留 headroom 的场景有最终容量硬约束；不保留的场景支付软风险：

`R_m = (1-exp(-lambda_m*H))*cold_admission_cost_m`

`lambda` 使用过去真实到达和冻结 rate prior 的 Gamma-Poisson 后验；请求大小使用
过去到达的 P95，cold cost 使用实际控制历史与保守 pause/copy 先验。**不读取未来
trace 的到达时刻来做在线决策**。这是未来准入风险代理，不是精确未来等待预测。

只有预测净收益超过实测误差界加1 request-second margin，且重新观测验证仍安全，
才执行。已支付控制成本不重复扣后续决策收益，但所有失败尝试和真实暂停仍计入
端到端指标。当前最多4次回迁、8候选/pool、9场景共1秒规划预算、10秒cooldown；
准入优先。该目标不等价于严格的逐请求 TTFT/P99 SLO 保护。

实现入口：`workaware_rebalance.py`、`workaware_ilp.py`、
`workaware_profile_cost.py`、`live_rebalance.py`。线上名称为 `kv_only_workaware`。

### 3.3 基线与对照名称

| 名称 | 行为 |
|---|---|
| `no_move` | 不迁移，容量不足自然排队；不主动调整另一模型的放置 |
| `no_move_fair` | 不迁移；排队超过30s时限制另一模型新请求准入直到队首能进入 |
| `own_model` | 历史对照：仅允许调整到达模型自己的请求 |
| `kv_only` | 跨模型容量准入迁移，不做释放后回迁 |
| `ilp` | 历史性能项对照，不等同于当前 `kv_only_workaware` |
| `kv_only_headroom` | KV-only 准入，加保留未来请求 headroom 的回迁控制 |
| `kv_only_workaware` | KV-only 准入，加剩余推理工作/实测失衡成本/软 headroom 风险回迁 |

`no_move` 是同 full-reservation 自定义 scheduler，**不是未修改的 native-vLLM
动态增长 baseline**。GPU 与 KV 物理资源仍共享，“不做跨模型调整”不等于硬件互不影响。

## 4. 已完成的容量收益：有效，但有 Conv 代价

### 较强 Code 收益（251533）

六策略同75请求，各自完整 drain；主窗600..2400s为52Conv/2Code。
KV-only 对 no_move：Code avg TTFT **804.888→55.904s**，约降低93.1%；
Code P99 **822.058→57.095s**。同时 Conv avg **.425→.857s**、
Conv P99 **1.148→12.462s** 退化；不能说所有 tenant 同时改善。
headroom 相对 KV-only 没有证实大幅新增 tail 收益。
完整报告：[headroom-tail-status](headroom-tail-status.md)。

### 降载后的配对对照（254337）

六策略各54请求完整结束；主窗为37Conv/2Code。实际压力 **1.129739**，
不是准确1.1；以下同一 trace 内比较，TTFT 秒，TPOT 毫秒：

| 策略 | Code avg/P95/P99 TTFT | Conv avg/P95/P99 TTFT | Conv avg TPOT | 合并 avg/P95/P99 TTFT |
|---|---|---|---:|---|
| no_move |246.366/262.572/264.013|.347/.474/.508|17.592|12.963/23.310/250.688|
| no_move_fair |146.365/162.782/164.241|3.234/31.634/39.146|见分篇|10.574/49.627/150.743|
| kv_only |55.297/56.021/56.085|.487/1.174/2.384|19.311|3.298/7.714/55.490|

KV-only 对 no_move 的 Code avg/P99 降低77.55%/78.76%，合并 avg/P99 降低
74.56%/77.86%；但 Conv TTFT/TPOT、合并 mean latency 不全面变好。
相对 fair，Code avg 降低62.22%、Conv P99 降低93.91%，不能用更慢的 fair
掩盖相对 no_move 的 Conv 代价。

真实容量见证：首 Code 前两卡 mapped 为 `[2400,1440]MiB`，单条 Code 增量
9504MiB，两卡分别不能 fit，而总 free16640MiB 足够。跨模型迁移一个 Conv 后，
mapped 为 `[3648,9696]MiB`，均≤10240MiB；复制 bytewise 验证通过。Code 平均
排队 **191.372→.939s**，约55s的100K prefill 下限仍存在。
完整报告：[pressure-ratio-status](pressure-ratio-status.md)。

这两轮 Code 都只有2样本；P95/P99 是描述性插值，**不是多seed尾部统计确认**。
两轮 workload/rate/request 集合不同，不用跨轮绝对指标给算法做因果归因。

## 5. 最新完成：性能感知回迁（257484）

2026-10-27 12:46:08–20:27:46，两A800，COMPLETED/0:0。
五策略各98请求完整 drain，共490请求/3,932,800输出token；453 rawfile SHA、
两个日志SHA、完整selected serial references、四rank sparse graph、ownership、
physical预算、ILP ledger replay 独立核验通过。

设置：两 Qwen tenant DP2/EP2/attentionTP1，BF16/131072，24GiB/model/rank虚拟
pool，**10GiB/GPU受控mapped预算**，allocator.98，默认 sparse/pause32；
Code真实SWE-smith100K+64输出/.0005RPS，Conv Azure输入2056..7633+
显式合成8192输出/.01956673845194241RPS。4800s/seed20261024，主窗600..4800s。
Conv 长输出不是 Azure 原始输出分布；本实验不是1M serving或整卡OOM。

主窗84Conv/2Code，TTFT 秒，TPOT 为逐请求均值、毫秒：

| 策略 | Code avg/P95/P99 TTFT | Conv avg/P95/P99 TTFT | Conv avg TPOT | 合并 avg/P95/P99 TTFT |
|---|---|---|---:|---|
| no_move |59.4694/63.4909/63.8484|.3669/.5296/1.1427|17.8378|1.7414/.7441/56.3416|
| no_move_fair |58.6369/61.9502/62.2447|.3739/.5200/1.0954|17.8037|1.7288/.8982/56.0599|
| kv_only |55.5375/55.7803/55.8018|.3825/.5335/1.3357|17.3089|1.6652/.8060/55.3487|
| kv_only_headroom |55.6005/55.7733/55.7887|.3744/.5325/1.1974|17.2739|1.6587/.7646/55.4661|
| kv_only_workaware |55.5615/55.7526/55.7695|.3623/.4981/1.1692|17.0269|1.6460/.9326/55.4130|

新方法对 headroom：Conv avg/P95/P99 TTFT 降3.23%/6.45%/2.36%，Conv avgTPOT
降1.43%；Code几乎不变，合并 P95 **退化21.97%**，Code avgTPOT **退化12.25%**。
不是50%收益，也未证实普遍优于KV-only或独特回迁收益。

原因：普通 baseline 两条Code只等9.3625s/.1989s；第二条到达时另一rank已经为空。
reservation到首token约55s，不是持续堵塞场景。唯一真实回迁发生t3166.359，
control3.826s，晚于两条Code，不能解释其此前TTFT。N2机制试验曾测到ConvTPOT
约降低15.39%，但n2且TTFT退化，不能替代上表的固定Poisson trace结果。

完整报告：[workaware-return-results](workaware-return-results.md)；
下一门槛：[workaware-return-next-phase](../docs/workaware-return-next-phase.md)。

## 6. 当前运行：回到1.2压力目标（259091）

用户要求 baseline 不排队时提高Conv压力。旧no_move实测Conv1739.280628MiB/卡、
完整Code pool9600MiB、单卡预算10240MiB，实际rho=1.107351624。

`rho=(mean Conv physical + Code complete pool)/controlled GPU KV budget`

目标Conv=`1.2*10240-9600=2688MiB/卡`，扣固定Conv96MiB反推：

`lambda_new=.01956673845194241*(2688-96)/(1739.2806284373632-96)`

得到 **.030863253171591748RPS（+57.733%）**。仅改Conv rate；Code仍.0005RPS、
两到达时刻2463.4417805/2663.1432232s不变。保持seed、长度、预算、graph、算法，
同Poisson流新增Conv marks；新trace142Conv/2Code/策略，五策略均须完整drain。
复用已验证cadence校准，跳过重复N2/N9机制阶段，不跳过serving基线。

259091于2026-10-27 22:35:46提交，两A800于22:40:43获配；23:10:52主查询仍
RUNNING，首no_move_fair elapsed1156.811s、34offered/33completed/144。
这只是发布前进度快照，**实际1.2±.02、Code排队与收益尚未完成核验**。
服务时间反馈、有限Poisson样本、页离散性使RPS公式不能保证每条Code都排队。
不挑到达时刻/seed、不删请求制造收益；两端每300s查询至终态SHA与独立指标分析。

协议：[workaware-pressure-1p2-protocol](../docs/workaware-pressure-1p2-protocol.md)。
结果/进度：[workaware-pressure-1p2-results](workaware-pressure-1p2-results.md)。

## 7. 文档与证据导航

| 内容 | 权威位置 |
|---|---|
| 当前想法、实现、收益与边界 | 本总览 |
| 完整实验矩阵、job/设置/结果索引 | `../docs/experiment-matrix.json` |
| 最初科研idea与placement要求 | `../docs/placement-design.md` |
| 当前回迁算法、门控与校准公式 | `../docs/workaware-return-protocol.md` |
| 当前实验已完成的全窗/主窗各tenant指标 | `workaware-return-results.md` |
| 当前1.2压力实验 | `workaware-pressure-1p2-results.md` |
| 早期32K Poisson trace负结果与控制开销 | `trace-benefit-results.md` |
| Azure/LongBench长度与真实100K输入来源 | `mixed-workload-profile.md`、`agentic-100k-dataset.md` |
| 所有研究导航与执行入口 | `../README.md` |

原始证据存放在外层ParaCloud的 `evidence/`：`headroom-tail-20261025/`、
`pressure-ratio-20261026/`、`workaware-return-20261026/campaign-v2/`、
`workaware-pressure-1p2-20261027/` 等；大raw、checkpoint、原dataset、运行中的
状态文件不随源码提交。报告保留SHA、job与独立核验范围，不把未公开的大raw当成
仓库内已随附文件。运维在外层 `docs/cluster-operations.md`；不对活动cluster checkout
执行pull/reset或覆盖source。发布代码快照不等于完成最终科研收益目标。

## 8. 本次发布前校验与范围

当前研究源码、测试、协议及结果报告一起提交到用户fork的 `main`。两个隔离试验
stage目录只保留在本地，不提交重复源码；外层checkpoint/raw dataset/evidence也不提交。
正在运行的259091使用冻结source，commit/push不触发远端pull或源码替换。

发布前执行：`python3 -m pytest -q tests/test_dp_opt_*.py tests/test_vllm_tp_world_size.py`。
结果为 **1009 passed / 5 skipped / 1 failed**，保留既有 `asyncio_mode` warning。
唯一失败为 `tests/test_dp_opt_wait_forecast.py::test_future_births_can_pin_shared_pages_after_current_owner_completes`
（预计wait1s，实际返回120s）；2026-10-24/25报告已经记录同一旧forecast边界失败。
本次不修复或隐藏该失败、不改活动source；当前实验 `DP_OPT_WAIT_AWARE=0`。
最新archive/profile/pressure专项与扩大回归195项此前全部通过，但不能把它当全仓库全绿。
本次回归证据在外层 `evidence/publication-20261027/cpu-regression.log`。
