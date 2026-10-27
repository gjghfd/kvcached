# 成本摊销 ILP：平衡 KV 迁移与推理

## 动机与普适事件门控

249646 的 v1 回迁正确，但15个transaction中13次只由普通Conv完成触发。18次回迁
copy合计1.340s，pause26.987s、preplan11.442s；总控制时间KV-only6.340→51.821s，
54请求主窗口mean TTFT Code/Conv/合并均退化。原负结果保留在
`../reports/release-rebalance-status.md`，不能把减少动作本身当性能收益。

新 `kv_only_amortized` 容量准入沿用KV-only。成功迁移准入记录受益请求和被调整模型，
受益请求真实完成后才开放恢复；普通完成只有存在放置债务时才触发尝试。准入优先、
credit不足或非decode导致延迟后，只在下一次真实release重试，不每2秒盲求解。
不要求历史home或完全count平衡，不依赖未来trace和Code/Conv角色；无迁移的direct
redirect尚不产生债务，本版不是涵盖所有偏离的生产调度器。

## 迁移固定开销进入ILP，允许部分恢复

源页owner、copy-before-release瞬时容量约束沿用原ILP。准入先枚举目标DP rank再求解；
恢复不增加人口、无新目标。`z`为至少一个请求迁移的二进制activation变量：

```
min P(layout) + W * F * z + sum(copy_request_seconds[i] * move[i])
subject to W * F * z + sum(copy_request_seconds[i] * move[i]) <= credit
```

`P`是剩余decode horizon的预测性能损失request-seconds；`W`是被暂停驻留请求数；
`F`是固定停顿上界。完整预留KV复制时间乘被暂停请求权重，不只计被移动请求。
固定成本仅计一次，预算有限可部分恢复或不动，而非先求完美均衡再整体拒绝。
执行还要求预测净收益超过冻结的1 request-second margin。

## 过去驻留服务暴露支持的credit

```
gross = sum(completed_at - reserved_at for completed requests)
        + sum(now - reserved_at for active requests)
service = max(0, gross - measured_rebalance_blocking_request_seconds)
credit = max(0, epsilon * service - measured_rebalance_blocking_request_seconds)
epsilon = 0.01
```

这是reservation后驻留服务暴露，**不是GPU kernel compute-time**，包含prefill、API/
调度等待和容量准入停顿。只扣恢复策略实测pause，非全部系统停顿；1%为控制开销
政策，不保证每请求SLO或严格compute-overhead<=1%。未到达请求不能给迁移充值。
成功迁移及暂停后拒绝都扣真实阻塞上界，不可重置预算；四rank边界receipt→resume
完成可独立重建扣账，更新F且不低于prior1.7743364535272121s。真实规划/阻塞/复制
全部计入请求指标。

暂停前计未来固定停顿；安全边界后已支付pause/solve为sunk cost，继续利润只扣剩余
复制，不能再重复惩罚F。但已支付成本仍扣credit，fresh solve新增阻塞也扣预算后
复核，预算不符安全resume。

## 冻结同trace对照

- 两A800、BF16、128K、24GiB/model/rank虚拟pool、10GiB/GPU受控预算、allocator .98。
- 默认sparse `[1,8,16,32,64,128,256]`、FULL_DECODE_ONLY/compile0、32step边界；
  不关闭空attention rank的EP，不修改模型/请求/SLO来追收益。
- 七策略同allocation：no_move、no_move_fair、own_model、KV-only、原准入ILP、v1、v2。
- 原75请求trace SHA `2bd50c7e9555e54aa59260db3f9f959230cbf2052985ec50add88488f642dbfb`；
  Conv Azure输入sample+明确合成8192输出；Code SWE-smith真实约100K输入+64输出。
- Conv .0307909199195414RPS、Code .001RPS、seed20261024、600s自然warmup+1800s主窗。
- 共用最多4回迁、8候选、1s solve limit、10s cooldown和原固定成本/copy带宽。
- 全75drain；主54单独报告Code/Conv/合并mean/P90/P95/P99 TTFT、latency、TPOT、
  120s完成SLO与老Conv代价；2条Code tail只能描述，不能称多seed显著收益。
- 终态raw SHA、token配对、四rank graph、owner/physical mapped/峰值、ILP replay、
  credit/债务历史与冻结runtime源码独立核验；负结果照实归档。

仍使用安全global pause复制，**尚未实现GPU copy/inference真实并行**。性能项仍是
N128单模型sparse surrogate，尚无小batch/context/双模型联合校准。目标和门控通用，
预测系数不能称普适精确或SLO最优；收益须待同trace实测，不以代码测试代替。

## 同trace诊断后的一般化后续设计（尚未实现）

250581已完成的KV-only/v1/v2子树显示：降低控制开销仍未改善mean TTFT；KV-only的
第二条长请求可以直接准入，恢复布局版本却要再次移动请求。后续目标不能只有
count平衡，需要同时计入给下一次准入留下可用KV的价值：

```
min expected_remaining_decode_loss(layout, H)
    + measured_pause_and_copy_cost(layout)
    + sum(arrival_rate[m] * H * expected_admission_delay(layout, m, future_KV_size))
```

所有模型用相同规则。到达率、KV大小分布来自过去可观测到达/外部冻结校准，而非
偷读未来trace；H来自剩余tokens及历史decode速率。未来准入项需预测wait与迁移
两种真实代价，并考虑每GPU共享physical页、模型虚拟pool与碎片；仅总free GiB
不足以声称未来allocator一定可行。恢复可能选择保留非均衡但可准入的布局。

性能项首先要用当前小batch、长KV及是否存在另一模型prefill作联合校准；N128
系数不能证明N4迁移有127request-seconds真实收益。剩余work应在多个预测完成事件
上积分，而非只看首个请求完成前的capture plateau。无当前收益不等于未来无需
恢复，但重试须有状态变化和成本门控，不能又退回每次完成全局pause。

这是下一阶段设计约束，**不是本次v2已实现或已实测通过的新算法**。不应通过固定
某模型名字、强制历史home或调请求/SLO来制造正结果；缺可靠校准时保守保留KV-only。
