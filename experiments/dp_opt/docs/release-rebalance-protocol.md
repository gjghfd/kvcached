# Code释放后的ILP回平衡（优化2）

实现与完整实测见`../reports/release-rebalance-status.md`。2026-10-25的249646六策略
GPU实验已结束并独立验证；回迁正确，但对原KV-only的增量净收益为负。不能把CPU
最优布局、后台监控或旧KV-only准入结果称为新策略额外收益。

## 策略与触发

新增独立策略`kv_only_rebalance`，不改变原`kv_only`和两个排队baseline。
新请求准入仍完全走原容量优先KV-only：仅空间不足时重排，不启用准入性能项。
请求完成后，独立worker查询四rank的真实inventory，通过ILP优化仍驻留的请求。
Code完成是主要触发；Conv完成也可能释放一次复制所需的空间，因此同样通知worker。
完成回调只发通知，不持准入锁等待迁移；事件合并，prefill/inflight或排队请求存在时
延后重试。准入优先，默认重试2s、成功迁移后冷却10s；trial结束必须收拢worker，
迁移异常直接使整个trace失败，不跨trial继续，也不隐藏异常。

“移回去”是恢复当前有利的attention-DP/KV布局，不是强制恢复某个历史request home。
例如Code已结束、Conv为8:0，容量/成本允许时可到4:4；剩余decode短、复制峰值不足
或性能表不支持收益时保持原位。Code仍占据近整卡预算时，ILP不能凭空腾出空间。

## ILP的无准入模式

`ilp_reference.py`增加`new_request_count=0`，同时要求`new_gib=0`、准入成本为0、
`target=None`。没有虚构的新请求；两个模型的最终人口都等于其当前人口。
旧快照不含该字段时仍默认准入一条请求，旧准入snapshot格式保持不变。

无准入时只求解一次现有请求的布局。各rank使用与否由最终人口表中的端点/内部项
选择，仍覆盖单侧及双侧布局；不枚举一个不存在的新请求target。保持原串行迁移
stage、source-page全owner释放、pinned页和destination-pool重用约束，以及每步
copy-before-release的物理预算。保留“不移动”方案，候选与迁移数量限制下的最优
不等于全局allocator packing最优。

## 目标与门控

N_m为真实剩余请求数，H为当前剩余输出token的平均预测窗口。没有新请求时的
单模型sparse surrogate为（N_m=0时P_m=0）：

    I_m = |n_m0 - n_m1| / N_m
    P_m(layout) = H * N_m * T0 * [a * I_m + b * (capture(layout)/capture(balanced) - 1)]

已有独立核验的Conv population lookup可替代对应模型表；当前默认仍是单模型N128
外推，不是双模型联合校准或P99最优保证。回迁不把性能表归零；仅准入保持KV-only。

令L_i为实拷贝KV bytes/GiB，B为冻结的实测copy bandwidth，N为全部驻留请求数：

    J(layout, moves) = sum_m P_m(layout) + N * sum_i L_i / B
    Gain = P(unchanged_layout) - J(best_layout, moves)
    Execute only if Gain > N * F + D_already_blocked + delta

F是一次transaction的固定控制成本，默认复用冻结的`move_fixed_s`，不对每次移动
重复计一次完整pause/resume费用。delta默认1 request-second。暂停前D=0，
暂停后D按实际rank boundary receipt人口计权，再加fresh ILP期间的阻塞成本；
F作为保守未来控制预留。全部单位是request-seconds，不是GPU利用率或吞吐百分比。
CPU规划用真实快照，重新取得准入锁后复核generation和ownership。按默认32step
boundary暂停后重新获取真实人口并重新求解，不能直接应用Code结束前的陈旧解。
fresh解不划算则不迁移并resume。已发生的所有控制时间仍计入trace指标。

## Data-plane与独立核验

复用现有vLLM+kvcached迁移transaction：prepare→真实峰值检查→完整device echo
bitwise核验→commit→source release→路由更新。失败/提交状态不确定时保持paused并
终止，不盲目resume。成功前后核验request ID/tenant/prompt/output/computed state/
block数量不变，以及实际owner、最终rank counts和每卡物理mapped budget。
`verify_rebalance.py`重建snapshot、独立重放有序迁移、参考不动解、成本门控和
default32 receipts；`verify_trace.py`仍要求完整请求、paired tokens、四rank graph
及普通/fair baseline有效性。回迁decision和migrations进入原journal，总control、
preplan、迁移次数和bytes不遗漏；分Code/Conv/合并TTFT及老Conv TPOT/latency报告。

## 下一轮实跑与部署边界

入口`rebalance-trace.sbatch`独立于冻结的旧stress campaign，不重新生成请求。
先复用完整核验247429/case0的固定trace、10GiB/GPU、Conv8192合成长输出/Code100K
输入+64输出、默认BF16/128K/.98/sparse/32step。原五策略加新策略，同trace/预算/
模型配置，先做release功能及token一致性，再做配对收益；后续多个seed才作泛化。
普通及fair基线都重跑，不只对比旧job中的wall time。Conv合成输出不是原Azure输出。

必须显式冻结`DP_OPT_REBALANCE_TRACE_PATH`、`DP_OPT_REBALANCE_TRACE_SHA256`、
`DP_OPT_REBALANCE_BUDGET_MIB`、`DP_OPT_MOVE_FIXED_S`与`DP_OPT_MOVE_BANDWIDTH_GIB_S`。
默认回平衡最多4次移动、每model/rank最多8个候选；可通过`DP_OPT_REBALANCE_*`
独立设置，不能暗改fair threshold、SLO或trace来追50%结果。

本轮接管时不覆盖248507/旧campaign活动runtime，不重复申请allocation，也不创建
会让活动campaign提前退出的PAUSE marker。`rebalance_campaign.py`核验旧作业安全
终态/归档并停止旧本地后续提交后，才做旧SHA备份/原子同步/新SHA复核，提交唯一
A800实验249646；现两作业都已终态归档，无活动allocation。今后重试仍须先核对
queue/提交ledger和正在使用的源码版本，不能覆盖活动代码。持续300s监控，终态
全SHA归档并独立验证；失败先诊断，未获新发布指令不commit/push。
