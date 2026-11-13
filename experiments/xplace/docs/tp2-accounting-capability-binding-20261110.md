# TP2账本配置与源码绑定（2026-11-10）

## 结论

新增`accounting_capability.py`并扩展默认关闭的engine观测元数据。`prepare_bound_reference_work`在原有因果join和共同工作量之上，要求每rank的scheduler与wave提供一致的配置/源码能力记录，并匹配调用方预声明的manifest及其SHA。

**本轮仅本地实现与CPU验证，没有真实部署manifest、运行参数K、持续时间预测器或GPU结果。** 旧`prepare_reference_work`保留为明确未做部署绑定的账本API；不能把它作为绕过新绑定检查的正式评分入口。所有forecast仍为unavailable。

## 实际生产端读取

`AccountingCapability(engine)`在启用实时观测的engine初始化时读取实际scheduler对象：DP/TP/PP及EP开关、async/chunked-prefill、prefix-cache、block-size、encoder-decoder/diffusion、speculative/KV-transfer/LoRA/connector、KV组数、lookahead、每步采样数，以及运行请求上限、调度token上限和model长度上限。

缺失字段不使用“默认关闭”掩盖。仅支持DP≥2、TP≥1、PP1、EP开启、同步、无APC/投机/connector/LoRA、普通decoder且每步采样1、单KV组、lookahead=0；其他字段显式记录并要求与manifest完全一致。额外的每请求迁移约束、真正的设备EP组、worker代码/设备身份及容量仍须独立验证，不由这些配置推断。

启动期通过当前实际对象/导入解析六个源码角色：vLLM scheduler、Request、engine实现、placement scheduler、live_observation以及能力模块自身。记录module、qualname和文件SHA256。两项核心账本方法`_update_after_schedule`及`update_from_output`必须实际使用已解析的vLLM Scheduler实现，不能以子类/实例覆盖绕过其语义。

额外核对两方法的module、qualname、代码名称和实际来源文件均属于声明的scheduler，避免只给类文件hash却忽略已替换的类方法。采样时同时检查方法对象和启动保存的代码对象身份；不需要每次读取文件。

这是启动时解析到的源码文件及方法身份记录，不是远程认证、字节码等价证明或GPU正确性证明。作业必须保持不可变包，验收须再核查完整接口/worker清单；不会以这些六个角色替代整个部署manifest。不能将先前缺失的`v1/request.py`历史锚点补写成已取得。

## 采样与时序

每次已有scheduler快照/summary及独立rank socket采样重新读取轻量配置字段，检查未偏离启动值、engine/scheduler rank一致、两项账本方法身份未改变。**采样时不重新读取或hash源码、不新增step RPC、CUDA同步、热路径文件写入。** source缺失或配置不支持时仍可返回host观测，但能力字段明确unavailable，不能生成已绑定参考工作量。

记录分别附在原scheduler快照和rank-wave样本中，沿用它们各自真实观测与收讫时间/PID/启动token，不伪造同时性。两条记录不一致就拒绝。worker lane观察方式不变；默认关闭分支不初始化能力采集。4096字节socket上限未提高；真实开销和Linux/GPU环境仍未验证。

## 消费者与失败方式

预声明manifest包括精确kind/schema、六角色源码和两个tenant各自完整配置；其外部SHA必须匹配。生产代码不从当前观测自动创建一个必然相符的manifest，未知源码不能现场“接受为新版本”。实际GPU协议仍未冻结此manifest。

`prepare_bound_reference_work`先验证manifest，再调用已有三来源因果join与共同工作账本，继承所有身份/clock/receipt/freshness/skew/缺lane拒绝。随后检查完整rank能力、严格字段类型、能力内容hash、来源和配置相等、与查询DP/TP一致。bool不能冒充整数；未知额外字段不进入reference，避免把未经审定信息混入预测输入。

缺能力、两个来源不一致、能力unavailable、hash损坏、未审定源码、配置不匹配等均返回unavailable且不提供reference ID。无效调用方manifest抛出ValueError，不能退回无绑定或常数估计。

通过后将四rank能力并入原observation hash，再把manifest SHA并入reference ID；候选仍共享工作量并使用原布局投影，不能移除在途限制或改容量。通过只表示**这些记录与预声明账本配置/源码一致**，不是运行时验收、完整能力证明、预测精度或代价门通过。

## 验证与下一步

专项测试覆盖全部配置缺项/不支持值、源码角色错配、运行中配置/方法/rank变化、source IO失败、manifest损坏、缺lane/晚到/错clock，以及真实本地socket和原消息上限。CPU模拟engine/scheduler与测试clock不代表vLLM或目标Linux实际部署；测试用manifest只属于夹具。

下一步应形成并实现共同时间线上的残余与持续时间模型，明确采样年龄、等待/在途/采样收讫、迁移排空与传输、双tenant阶段耦合；不要继续把状态校验当预测。随后才能冻结新的观测/训练/独立验收协议和manifest/K/cadence/deadline，接入完整对称评分及容量测试。

r1–r4仍为探索数据，全部失败/截尾/负结果保留；15%/30%门及容量reserve不变。独立预测和长上下文/容量pilot通过前不启动正式消融，TP2公平配对通过前不做混合拓扑。
