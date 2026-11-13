# TP2三来源因果合并：本地验证，不是预测器

## 1. 新接口的范围

`causal_observation.causal_joint_state`直接接收`PlacementController.prediction_observations`的原始返回值，不接受另一个可能来自缓存/候选布局的planner state。逻辑描述符只从本次通过校验的scheduler快照白名单投影构造；baseline与候选引用同一份实际输入。旧`joint_prediction_state`与其每worker波次形状保持历史原样，没有把raw transport简单改名来绕过它。

新接口分别保留4个scheduler收讫、4个engine rank-wave样本和8个worker lane-host样本（双模型DP2/TP2）。每个来源的观测/收讫时间原样保留，不把rank wave复制为lane的同时观测，不把候选请求迁移伪装为已经改变真实host状态。

输出`status=available`只表示身份、来源、因果可用时间、freshness/skew及逻辑字段检查通过。**这仍是不同时间采集的状态，不是原子全局快照、GPU空闲证明或有效代价预测。** `device_activity=unknown`、`forecast_status=unavailable_not_estimated`始终保留；没有cost、other_active或horizon输出，没有自动准入/评分调用点。

## 2. 必须绑定的真实生产端

上轮transport只有socket侧PID，不能把scheduler RPC绑定到同一rank监听器。本轮仅对`live_observation.py`增加启动期随机`producer_id`，连同真实PID附在socket消息与对应scheduler snapshot/summary上。两条通道必须具有相同的PID和32位十六进制实例token；进程重启或采样器换代即使重用PID也不能混合。

所有engine/worker端点的PID和token分别唯一；身份要求两个tenant的每个DP rank和每个TP lane完整、不重不漏，int身份不接受bool。数据内每条clock ID也要匹配调用者指定的共同clock，不能只检查外层标签。

该token是当前私有通道的来源绑定，不是对任意历史JSON的认证。记录仍需由真实在线通道生产，并保留raw、冻结源码与部署证据；禁止事后替换token、kind或时刻以让旧数据通过。旧缺少实例元数据的transport明确不可用，不补造身份。

## 3. 时间与失败语义

- 每条要求`observed_at <= available_at <= as_of`，且`as_of-observed_at <= max_age_s`。
- socket receipt不能晚于execution collection完成；全部receipt不能晚于controller collection完成。controller的`collected_at <= as_of`，防止使用虽已在内部收讫、但整个查询尚未返回的数据。
- 统一skew覆盖全部scheduler、rank-wave和lane-host观测时刻；不能只核对某两类样本或用最新receipt冒充观测时间。age/skew边界允许等号，所有参数显式传入，无默认放宽阈值。
- host调用开始时刻可早于freshness窗口：长期阻塞的调用仍可被当前采样观测；开始时刻不能晚于样本。none要求深度0、无调用起点；活动调用要求1–4层。保留嵌套状态，但不补算GPU执行时间。
- 非法host并发/深度/来源、累计host异常、缺失lane、超时/不完整传输、跨clock、晚到、陈旧或不一致身份全部拒绝。输入raw不改动；调用方仍须保存失败的原始包，不能只保存可用子集。

不同rank的wave可以在采样时不同；即使逻辑idle、wave=False而worker处于sample/dummy，也不以矛盾为由删除观测，更不推导solo。这里只读取`engine.engines_running`的布尔host状态，不保证此前CUDA工作已完成。

## 4. 候选与容量隔离

逻辑描述符复用既有已测试的phase/in-flight白名单，分别计算两个方向的own/peer计数与账面上下文。候选move只改变候选描述符，候选arrival保持prefill且computed/output/in-flight均为0；不使用真实剩余输出、max_tokens、未来完成时间或未来phase。可变输出与输入隔离，排序依据身份而非传输返回顺序。

没有改动`planner.py`、`cost_model.py`、原容量保护或现有准入/周期算法；不从预测干扰缩小增长余量。新接口也不承诺迁移可行性，容量与实际数据面仍由既有保护检查，未来评分集成必须使用同一采集来源。

## 5. 验证与未完成事项

专项测试包含严格身份/来源/时序/缺失/异常/未来字段边界、tenant互换、候选与真实状态隔离。端到端CPU测试启动12个独立进程：4个模拟engine通过控制管道返回scheduler快照、独立socket读取rank wave，8个模拟worker阻塞于sample时由socket返回host状态；使用实际PID、实例token和跨进程monotonic收讫时间。测试共同clock标签是显式替身，**不是目标Linux clock命名空间验证或GPU/vLLM部署验收**。

本轮生产端仅增加初始化元数据，不增加逐step UUID、RPC、CUDA同步或热路径文件写入。默认开关仍关闭，r4冻结包和原始结果未改，无新作业。精确次数、隔离源副本、测试日志与SHA在`causal-join-preparation-20261110.json`记录。

接下来的必要工作是固定baseline/候选同一参考工作预算、prefill/in-flight账本及耦合阶段转换，形成可部署预测器后接入完整对称评分。不能把当前phase铺满horizon，也不能用未知完成条件下的固定wall-time请求数乘时长冒充布局成本。独立预测15%/30%门、长上下文/容量pilot、TP2公平配对消融和混合拓扑门均未通过；r1–r4继续作为已查看探索数据，全部失败和负结果保留。
