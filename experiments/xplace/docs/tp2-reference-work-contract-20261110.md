# TP2同参考工作量与结果截尾契约（2026-11-10）

## 结论与边界

新增`runtime/reference_work.py`，仅在本地实现固定参考工作量、布局投影与离线结果验收。
**没有持续时间预测器、在线评分调用、GPU部署或通过实测代价门。** 本契约冻结量纲与账本，不冻结运行参数、预测模型或新GPU协议；默认关闭的实时观测通道不变。

## 1. 真实计数语义

审查保存的`sampling-source-20261109T194659Z/v1/core/sched/scheduler.py`：

- `_update_after_schedule`第1246行起同时增加`num_computed_tokens`和`num_in_flight_tokens`。
- `update_from_output`第1652行起扣除对应in-flight；后续才处理生成输出。投机拒绝还会修正computed。
- 第2502行已有`computed - in_flight`作为processed前缀的用法。

因此不能把computed当作“已完成GPU工作”再加一次in-flight。这里以`C-I`为**scheduler已收回结果的账面前缀**，不称GPU完成时刻。该scheduler文件有r3历史SHA锚点；仍须为下一次部署核验实际接口和配置。本模块没有证明GPU新部署，未取得新的远端源码；`v1/request.py`不在这份快照中，不声称检查过。

账本范围限普通decoder、连续前缀、无投机。当前collector没有完整能力/配置证明，输出明确记录`unvalidated_no_runtime_capability_binding`；即使计数恰好符合不等式，也不证明实际运行没有投机、异步占位、KV载入或其他特殊路径。预测器部署前必须补齐来源绑定并拒绝不支持路径。

## 2. 共同参考里程碑

一次决策使用本次三来源因果join，固定所有已观察resident（包括waiting/prefill）以及至多一个已收到的中心队列arrival。arrival必须有真实身份、相同clock及`arrived_at <= available_at <= as_of`，不能从候选是否准入来增删cohort。

设某请求已知prompt长度P、已观察输出O、computed C、in-flight I。对所有请求使用一个显式正整数K，目标为观察到`O+K`个累计输出。K必须在预测/评价前固定，不能由候选布局、真实未来长度或实测正式收益选取。本次不提供K的部署默认值；测试K仅为算术夹具。

K相对于各scheduler样本中的O，不是声称知道精确as-of时的O。样本至决策之间可能继续运行；未来模型必须处理样本年龄和残余状态，不能把这段工作重新从零执行。共同工作ID包含样本、独立时间戳、身份、预算与arrival收讫；布局不能改变它。

普通自回归路径完成该虚拟目标所需输入前缀为`T=P+O+K-1`：最后一个prompt输入产生首输出，因此无已有输出时只有K-1个后续decode输入。分解：

- 已收回账面前缀`R=C-I`；待目标输入量`T-R = I + (T-C)`。
- 在途区间`[R,C)`、未派发区间`[C,T)`完全不重叠。
- 每个区间再按`[0,P)`的prompt、`[P,P+O)`的已知输出输入、`[P+O,T)`的后续decode输入切分。

这同时保留chunked-prefill、最终prefill在途、已有输出后的重算以及在途decode。重算已知输出前缀不算新增输出。`C=P,O=0,I=0,K=1`即使输入账本为零也**不能判目标完成**，必须等真实输出收讫；单靠KV计数不证明采样完成。`I>C`、负数、bool冒充int或`C>P+O`等不能解释的计数拒绝，不夹到零掩盖问题。

## 3. 候选与耦合窗口

`project_reference_layout`只改变assignment，引用相同reference和observation SHA。未准入arrival仍在中心队列、仍有相同目标；准入后才成为对应rank的prefill。两tenant own/peer计数与bucket对称计算。不假装候选migration已完成，不搬迁观测中正在飞行的请求。

这是结构检查，不是容量、movable全部限制、波次排空、传输代价或准入可行性检查。现有容量增长保护不变，绝不从预计干扰缩小reserve。

目标量纲是`sum_i (t_i - as_of)`的request-seconds，其中`t_i`为控制端首次收到达到固定输出目标的进度时刻，不是CUDA时刻。它不是固定wall-time的N乘H，也不是整个窗口沿用当前phase。

未来预测必须在共同时间线上推进两tenant的prefill/已知输入重算/decode及rank调度、在途host返回和采样收讫；持续时间必须依赖当时双方状态。先达到K的请求**不因此变idle或停止干扰**：闭cohort无EOS假设下，它继续服务到最慢请求达到目标。baseline的arrival等待、迁移排空和传输也必须纳入时间；本模块没有生成这些持续时间，因此不能输出placement分数。到里程碑即删除快peer或把逻辑idle当solo均违反本契约。

## 4. 离线验收与截尾

`audit_reference_outcomes`与在线输入分离。输入是带reference ID、clock和真实收讫时刻的进度/终止记录；按每请求收讫顺序验证，不替调用方排序或修补。目标是控制端观察时刻，因此采样cadence、延迟和开销必须在新协议中预注册、对所有候选相同；迟到记录不能倒填为早已可用。

同时保留结果事件的observed_at；它不得晚于收讫或早于上一事件。快请求达到目标之后仍须有覆盖共同结束时刻的观测，**没有后续样本不等于没有EOS**。延迟收到的早期EOS按观测时刻判违例，不能当作窗口外终止。这里的样本还须由后续协议验证完整性和真实生产端来源；SHA只防意外修改，不构成真实性认证。

每个cohort成员都有结果行。目标前EOS、max_tokens、timeout、abort、preemption和error分别保留；缺样本保留missing，不能省略或作零代价。未全达到目标时总目标值为null，不能只累计成功子集。快请求虽已达到目标但在其他请求前终止，也保留闭cohort违例，不能把它作为该假设下的有效预测样本。全目标达到仅获得**可比较资格**，`prediction_gate_passed`始终false：没有预测误差就没有精度通过。

不能通过丢弃截尾例、反复换K/seed、延长Conv真实输出或将r1–r4改名为holdout来满足门槛。独立预测协议必须报告全量覆盖与失败；该局部目标不能替代正式trace的TTFT/TPOT及公平基线主指标。

## 5. 尚未完成

下一步仍须实现配置/源码能力绑定、异步/采样/排空残余处理及真正的耦合持续时间估计器，再接入完整评分。当前rank wave布尔量和lane host栈不含精确batch队列/执行进度，不能凭新账本假装这些信息可知。需要先评估可部署预测粒度与训练协议；拒绝缺失，不回退到已失败常数模型。

r2全部95个decode点、12个失败干扰例、55.5374%/158.8382%误差，r4全部24例12对及3个低覆盖例外继续保留。15%/30%门不变，独立预测与长上下文/容量pilot通过前不做正式消融，TP2配对通过前不做混合拓扑。
