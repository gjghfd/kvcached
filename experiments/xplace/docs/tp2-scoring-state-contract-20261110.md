# TP2评分状态契约准备（2026-11-10）

## 状态与范围

309147使用的r3上传包保持冻结。本改动仅位于本地工作树，未构建新tag、上传、提交或生成可运行代价参数。它补齐因果状态数据与显式bucket传参能力，**尚未把干扰预测接入准入/周期评分**，也不改变15%/30%误差门或任何既有失败结论。

## 已实现的数据链

- scheduler的snapshot/summary增加`phase_schema=1`、`scheduler_paused`及逐请求`phase`/`in_flight_tokens`；沿用已有单调时钟`observed_at`，不新增逐步RPC或GPU同步。
- 非running请求为waiting；running且已计算账面长度不足prompt，或尚无输出token，为prefill；其余为decode。抢占后重算可能重新归为prefill。阶段与movable不同：一个有in-flight token而不可迁移的请求仍可处于decode。
- 这些是调度器逻辑状态，**不是GPU执行完成证明**。computed token账面值可能含计划增量；事务暂停不把逻辑decode伪装成idle，暂停标记单独保留。
- `build_state`将每rank的时钟、暂停状态及必要请求字段复制到`scoring_observations`。不复制max_tokens、真实剩余输出、预留全输出等未来信息；旧compact请求和旧评分数值路径保持不变。
- `candidate_peer_state`只构造候选动作后的瞬时描述。迁移改变候选rank；候选新请求以已知prompt和零已见输出标为hypothetical prefill，不冒充当前已在decode。返回waiting/prefill/decode布局、账面上下文、prompt及decode bucket；两tenant对称。
- 调用者必须显式提供`as_of`和`max_age_s`。未设置运行时默认新鲜度阈值；缺时钟、未来/陈旧时钟、缺阶段、错rank、重复ID或不完整token字段返回unavailable，不能静默当solo或有效预测。无数据不能从movable推断阶段。边界和非法参数有专项测试。
- `integrated_cost`增加可选`other_bucket`并原样传给现有`decode_step`。省略时保持原调用和数值，不根据瞬时descriptor自动选用任何系数。

## 为什么尚不启用完整评分

1. `layout_cost`当前把候选arrival与所有resident按decode模型积分至规划horizon。把当前prefill阶段直接传入并维持整个horizon，会把短期prefill干扰错误延长；仅传True也不能修复已失败的phase-blind模型。阶段转换与聚合规则尚需冻结和独立验证。
2. 现有校准bucket不是通用的“对端bucket”。`measured_cost.interference_residuals`对两个tenant都使用原Conv布局bucket：对Code它是对端规模，对Conv它是自身规模。不能把新API中的peer decode bucket直接套进两侧旧表，更不能把参数键名当成物理语义。新估计器需声明自身/对端布局和阶段的索引契约。
3. 等待、事务暂停、prefill+decode混合、陈旧状态和候选动作描述是不同条件。unavailable分支尚未制定经验证的保守评分；容量增长仍保持solo速率，不因预计干扰而缩小安全余量。

因此，此轮不声称other_active评分缺口已修复。后续须在估计器及状态/时间/索引契约冻结后，接入`layout_cost`、准入和周期基础值/候选日志；验证完整评分确实响应合法干扰变化，保留tenant互换、缺失/陈旧值和容量保护对照，再做独立GPU预测和长上下文/容量pilot。r3机制观测本身不能替代这些门槛。

## 验证范围

专项包括实际scheduler输出、复制隔离、未知/陈旧拒绝、阶段转换、候选迁移与长prompt准入、tenant互换、显式bucket传递及旧评分不变。回归排除两份混合分时测试文件的整文件执行，仅选择其中明确的非分时节点；不执行分时策略。具体命令、完整通过/跳过/排除数和本地隔离验证以evidence日志为准。旧r3包、95个decode点、12例失败干扰及所有原始负结果保持不变。
