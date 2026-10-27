# 独立排队与超时公平排队基线

2026-10-23 用户明确要求两种不迁移 baseline。本页定义其准入语义和验证门槛；
2026-10-24首轮245583已完成，五策略3745请求与严格公平实际KV门控核验通过；
详细结果只在 `../reports/agentic-trace-results.md`。245085仍是没有公平基线的历史结果。

## 两类基线

| 策略 | 超时公平门控 | 跨模型请求/KV迁移 | KV不足时 |
| --- | --- | --- | --- |
| `no_move` / independent queue | 无 | 无 | 该tenant队首继续等，另一个tenant正常准入 |
| `no_move_fair` / timeout-fair queue | 有 | 无 | 先正常等；超时后冻结另一个tenant的新准入，等自然释放 |
| `kv_only` / capacity-first placement | 当前不启用此门控 | 有 | 枚举目标rank，求低迁移成本方案并安全重放置；不可行则等 |

每个tenant保持独立FIFO，两个独立client的Poisson到达始终继续。公平门控只停
服务端准入，不能暂停client生成、推迟原始到达时间、减少offered requests、
丢弃请求或缩短输出。排队从冻结trace的planned arrival起计，包括门控时间。
“模型间互不影响”指无跨模型准入策略干预，不是隔离共享GPU计算或物理KV预算。
`no_move`仍是同full-reservation受控scheduler，不冒称未经修改原生vLLM baseline。

## 公平门控的精确定义

阈值为 `tau`，CLI `--fairness-queue-threshold-s`。首轮候选默认30秒，是固定
120秒SLO的四分之一，未根据245085尾指标搜索最优值；真实运行前需冻结tau/配置SHA。
阈值不是SLO，也不是模型长请求长度阈值。

1. 无owner时，检查两个FIFO队首和已提交但尚未实际获得KV的队首请求。
2. 若存在 `now-arrived_at>=tau`，选择最早到达的超时队首；跨tenant完全同到达
   时间时按request ID打破平局。不是只优待Code，Conv同样可成为owner。
3. owner锁定期间，禁止其他tenant的新准入。已有运行请求继续prefill/decode，
   没有全局pause，没有KV复制、抢占或EP/模型权重调整。
4. owner只在其完整KV实际reservation已观测后解除，不在HTTP提交、内部排队
   acknowledgment或第一次准入尝试时解除，也不等到请求decode完成。
5. 解除后再次检查超时队首。因此被阻挡的Conv也可能成为下一owner；不在两个
   超时请求间反复切换owner，不同时锁住两个tenant。

公平策略使用一个协调dispatch loop保证门控检查与准入尝试的顺序；独立策略在
rank-local/concurrent模式下保留两个tenant dispatch loops。协调控制本身的
端到端成本不扣除，不以不同控制开销假设来修饰结果。
尚未获KV的内部pending请求保留真实reserved timestamp和观测timestamp；如果
此前已提交的其他tenant请求在门控期仍获得KV，严格独立核验会拒绝该运行，不
把仅“客户端停发”或漏审内部allocator的结果称作公平baseline通过。

普通排队与公平排队都依赖自然释放。若单请求本身大于任一rank可用容量，或
总容量长期不足，公平门控也不保证能完成；超时/未完成须保留为失败，不能丢弃
它们后计算更好的P99。当前100K输入可fit空rank的前提仍须由真实capacity pilot复核。

## 与我们的方法比较什么

独立baseline可能让小请求持续抢占新释放空间；公平baseline通过停止竞争准入
避免这种饥饿，但可能使大量Conv积压，且仍需等原有KV在目标rank自然释放。
capacity-first试图通过迁移主动制造rank-local headroom，减少等待而不是冻结
全部另一tenant准入。收益必须同时考虑Code、全部Conv和baseline定义的老Conv。
仅比不公平baseline获得收益不足以证明比简单公平门控更好。

## 实验入口与冻结设置

`agentic-fair-baselines.sbatch`复用原revalidated→measured→formal→capacity链。
默认五策略 `no_move no_move_fair own_model kv_only ilp`，保留历史对照；live CLI
也支持显式三策略功能验证，独立verifier不再假设一定有ILP结果。

保持原749请求/1200s/seed20261021、Conv .61RPS、Code .004RPS、11328MiB/GPU、
request timeout900s、SLO120s、真实100K token IDs不变。两A800、两独立Qwen tenant，
DP2/EP2/attentionTP1，BF16/128K/24GiB虚拟pool/.98allocator；默认sparse graph/
FULL_DECODE_ONLY/compile0/32step不变。同allocation执行，保存并冻结随机策略顺序。
后续配对重复和更多Code样本需另外预声明协议，不能把旧四策略结果拼成五策略
同allocation对照，也不能根据收益挑tau/seed/rate或删那条退化Code。

首轮tau通过 `DP_OPT_FAIRNESS_QUEUE_THRESHOLD_S=30`冻结；planned tau sensitivity
不是已经执行的实验。入口所需measured control env/SHA与原版本一致，未经保护
同步、旧SHA备份、新SHA/remote fork/main/protected diff复核不得启动。

### 2026-10-24首轮运行前冻结

冻结清单为顶层 `evidence/agentic-trace-run-20261021/fair-baselines-frozen-20261024.json`。
原输入/control/lookup由00:03:05和00:03:42远端实查SHA复核，没有重新生成trace。
五策略输入列表为 `no_move no_move_fair own_model kv_only ilp`，按原seed20261021
随机后的固定顺序为 `no_move_fair no_move own_model kv_only ilp`，一个repeat。
同allocation比较，不使用旧245085结果拼接公平对照；τ固定30秒，不根据执行结果调整。
保持900秒request timeout和原长Code，公平门控违例必须判无效，不放宽独立审计。

站点资源汇总中的 `hp_a800` 并非当前Slurm控制器的实际partition，查询显示不存在；
本轮使用可提交的 `gpu_a800`，不通过不存在的队列或其他GPU改变预声明硬件。

首轮结束后，同配置另一次allocation配对复跑保持此冻结trace/rate/τ/B/SLO/seed/
策略顺序不变，一次trial每job；用于重复性核验，不把同一trace重放当新增独立Code
样本或多seed泛化。不得只保留好的一次，也不据结果修改普通/公平baseline语义。

## 日志与独立核验

`fairness-events.jsonl/json`保存latch/blocked/release，decision保存每次成功提交的
`fairness_authorization`，request同时保存真实 `reserved_at` 和
`reservation_observed_at`。失败同样归档，不将队首提交误报为已获得KV。

`verify_queue_fairness.py`由完整`verify_trace.py`调用，独立重放：冻结tau、完整
请求ID、每tenant FIFO、最老可观测等待队首、门控期准入授权、owner真实reservation、
外模型物理KV分配、零迁移/零pause和事件journal一致性。门控和实际KV安全核验
不等于收益通过。默认归档/五分钟监控/全SHA机制仍执行。

收益分析用`analyze_tail_slo.py --baseline no_move_fair --treatment kv_only`，同时
保留no_move对照，报告两模型及全请求queue/TTFT/latency的P90/P95/P99、固定120s
SLO达标/违约、全部Code逐请求记录、全部Conv与老Conv代价。Code仅4条不能支撑
稳定P99结论。详细结果仍只写`../reports/agentic-trace-results.md`。
