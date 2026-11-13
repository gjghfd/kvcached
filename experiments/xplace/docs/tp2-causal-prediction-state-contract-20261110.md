# TP2联合因果状态契约v1：仅本地输入边界

## 范围

r4已定位长间隔的输出物化/DP协调host范围，但尚无有效预测器或完整评分。本轮新增 `runtime/prediction_state.py` 及专项测试，**不修改既有runtime调用、不部署producer、不构建新tag、不生成参数、不接入planner评分**。r4冻结包及原始报告保持不变。

`joint_prediction_state`把两个tenant的逻辑布局、真实执行观测、候选动作描述和因果可用时刻分开。它只回答“输入是否满足这个查询契约”，不计算代价，不声称预测验证通过。

## 1. 采样时间不等于可用时间

- 调用者显式传入同一host单调时钟域 `clock_id`、决策时刻 `as_of`、`max_age_s`和`max_skew_s`，没有默认运行阈值。
- 每个scheduler rank必须有真实RPC收讫时刻 `available_at`，每个执行lane必须同时记录 `observed_at`及producer消息收讫时刻 `available_at`。要求 `observed_at <= available_at <= as_of`，并对观测年龄和联合时钟跨度分别检查。
- 只有事件自身时间戳落在过去，不足以证明其在决策前可见。缺失delivery、晚到、旧快照、不同clock或只在stop导出的数据均显式unavailable；不会拿已有的完整rank/lane子集代替全量状态。
- 当前 `build_state` 保留采样时间，但**没有这些receipt元数据**；当前GPU执行侧也没有live execution producer。默认查询因此unavailable。这是尚待接入的实际缺口，不是补了一个字段就完成在线观测。
- `kind=live_scheduler_snapshot/live_execution_snapshot`及clock标识是producer契约，不是对任意JSON的来源证明。离线分析不得把r4的stop记录改标签后当live数据；后续producer及传输路径须记录真实收讫时刻并独立验证。

## 2. 观测执行与候选逻辑分离

`logical_observed`是原快照；`logical_candidate`是迁移/arrival后的假设布局。真实`execution_observed`仍属于动作前的世界，不随候选请求搬到另一rank，也不把候选arrival变成已经运行的GPU任务。

执行侧要求两个tenant所有DP/TP lane唯一且完整，记录当前 `wave_active`、`host_call`（none/execute/sample/dummy）及仍在途host调用的起点；同一DP rank两lane报告的wave标志矛盾时拒绝合并。没有当前调用不代表此前enqueue的GPU工作完成；wave=False也不能覆盖掉已经观测到的dummy调用。

**即使全部输入有效，`device_activity`仍为unknown，`forecast_status`仍为unavailable_not_estimated。** 不输出`other_active=False`或数值cost。scheduler的逻辑idle、事务暂停、host_call=none、wave=False都不单独构成设备静止证明。

新接口显式返回两个方向的own/peer计数与账面上下文，按model和DP rank索引；不调用旧`StepModel.interfere`，不把旧Conv-layout bucket套成通用peer表。候选prefill、decode、waiting分开，输出白名单不含真实剩余输出、max_tokens或未来调用结束时间。所有返回容器与输入隔离。

## 3. horizon仍需审定，不能用当前phase补洞

旧 `planner.horizon_steps` 使用基准solo速率把秒窗口转换成每tenant名义decode工作量，`integrated_cost`计算这段工作量的request-seconds；所有未知完成长度都以大哨兵保留。它**不是固定wall-time窗口中的真实请求存活积分**。

若未知完成条件下把所有请求都假设存活到同一个固定wall-time截止，再计算“请求数×固定时间”，代价会与布局无关。另一方面，把候选arrival的全部已知prompt当成已经完成、直接按decode计分也不正确。因此下一版预测目标必须明确选择固定因果参考工作量的完成成本，或独立定义的进度/等待目标，不能只替换参数或将当前prefill常量铺满horizon。

目前尚未冻结：prefill剩余工作与in-flight账本的解释、阶段转换/预测不确定性、两tenant耦合推进、迁移暂停和旧请求尾部的处理。**本契约不解决这些问题，也不暗含一个有效fallback。** 后续需要在同一基准query上固定baseline/候选的工作预算，避免因候选预计更慢而缩短工作量“获得收益”。设备静止未知、mixed/stale/missing等分支要明确保留或拒绝，不能默默退回已失败的phase-blind常数。

容量保护与目标评分分开：不能利用预计干扰减速缩小现有solo速率增长余量。真实剩余输出长度及正式trace未来结果不得作为工作预算或phase转换的oracle。

## 4. 进入新GPU验证前的剩余工作

1. 明确并实现producer收讫时刻/共同clock和全lane在途/wave观测；本地契约仅能测试数据形状和因果边界，不能证明尚未运行的producer有效或低开销。
2. 基于已查看的r1–r4探索数据审定可部署预测目标、特征、联合索引、horizon转移及missing/stale行为；固定估计器和参数来源。不能把r4 on/off大幅差异视为可忽略开销，不能把现有探索数据重新命名为独立holdout。
3. 再接入准入、周期baseline/候选全评分和日志，做tenant互换、变化响应、不可用分支及容量保护测试。目前该调用路径未改。
4. 独立、可部署测量条件下的预测检查以及长上下文/容量pilot仍在前置门内；15%/30%门、95个decode点和12例r2失败干扰不变。之后才允许配对消融，TP2未通过不得启动混合拓扑。

## 验证范围

专项覆盖真实planner state构造、receipt晚到/缺失、clock/skew边界、所有lane、矛盾wave、空逻辑布局但有dummy、候选与真实执行隔离、tenant互换及未来字段不传播；直接检查准入/周期结果和容量状态未变化。只执行新增测试文件，本地和隔离源副本验证；没有运行分时节点或重跑既有GPU报告。精确命令、次数、source SHA和差异在本轮evidence收据中记录。
