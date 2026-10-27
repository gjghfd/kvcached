# 预规划与默认DP共识边界下的等待成本

2026-10-23 16:06 Asia/Shanghai。公式设计尚非已冻结联合校准门槛；244176在排队，
必须等其mixed人口成本以及原749trace对照，不声称此公式已经给出实测收益。

## 按真实受影响范围计费

令t0为触发预规划时间，tp为CPU快照/ILP结束，tr为第r个attention rank真实停止
decode的默认32step共识边界，tf为完成迁移/新KV准入/所有resume。新请求只算一次
控制延迟，旧请求按各rank真实阻塞人口计权，而非共识等待全乘总人口：

    C_incoming = tf - t0
    C_old_pause_upper = sum_r n_r * max(0, tf - tr)
    C_control_sum_upper = C_incoming + C_old_pause_upper

GPU共享/EP同步意味着rank人口加权并非独立GPU算力之和，而是请求秒上界近似。
client token gap比receipt blocking更长时保留实际观测，不用receipt承诺TPOT。
native keep在t0停请求，然后dummy到共识；boundary在t0..tr仍decode，因此两者不能
用同一个pause_s*population公式。CPU预规划有GIL/调度干扰时仍测client cadence，
不假定asyncio.to_thread零代价。

## 暂停前筛选与暂停后确认

等待W取实际source-page owner的最晚自然完成时间，累积可释放page直到某rank够新
full reservation；两个rank取较早可行等待。它是近期decode rate的外推，非真实
counterfactual，长prefill会改变速率。未知W时保守等待。首先做不暂停的容量必要性
筛选和预规划，再严格复核generation+owner/blocks/pages；stale就放弃，不能应用
旧快照的ILP最优解。

预规划预算参考当前布局而非绝对已有失衡惩罚：

    DeltaPerf_plus = max(0, Perf(final_layout) - Perf(reference_layout))
    PredictedFuture = C_copy_prepare_commit_resume_sum + DeltaPerf_plus
    Proceed only if C_incoming_so_far + C_old_so_far + PredictedFuture < eta * W

eta先保持.8，不因负结果改SLO或挑trace；费用单位均为request-seconds。solver发生在
decode继续时只计incoming延迟和实测干扰，不再乘所有old人口；future迁移依然影响
全部被暂停的old请求，不仅被复制请求。目的prepare/source release/预算RPC/end/
resume与allocator全长度实际reserve都必须计入，不能只用26ms echo collective。

## 长Code prefill干扰不可漏算

原243415 capacity中101675输入Code的TTFT约55s；旧241580正式Code prefill约48–50s。
这些历史值不等于“早准入会增加old总延迟55s×N”，但说明prefill不可只按复制成本
外推：Code提前prefill挤占共享GPU可能使Conv更慢，同时延缓自然释放。必须测mixed
人口old cadence/latency与同749trace no_move/own_model/KV-only/ILP，比较Code减少
排队的收益是否大于所有old请求额外代价。公式还需该联合干扰项：

    NetGain = incoming_wait_saved - old_pause_added - old_prefill_interference_added
              - layout_performance_added - failed_planning_and_stale_cost

单模型N128 capture-shape代理只可标为surrogate；不得把性能项与Code prefill干扰
混为因果attention时间。按真实request记录总latency、TTFT、TPOT及原120sSLO核验；
暂不降低host wait gate或直接启用正式迁移来追正。mixed cost pilot协议见
`admission-cost-protocol.md`，最终trace/rate/预算保持。
