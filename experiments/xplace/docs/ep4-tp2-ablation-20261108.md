# EP4TP2 消融与混合拓扑实验协议

2026-11-08 用户授权：正常完成也续接当前 session，消息附带结果和待办；先在双模型 EP4TP2 上验证消融与完整系统优势，达到门槛后再运行一个模型 EP4TP1、另一个 EP4TP2。此授权不恢复旧 observer，不删除 PAUSE。

## 阶段与优先级

2026-11-09进展：307623已完成固定三轮干扰补测，八次双TP lane迁移、12/12例与168请求的原始窗口/零抢占校验通过。联合303594与306991的全部95个decode点后，decode相对RMSE为7.7589%，但干扰54次预测比较的相对RMSE为55.5374%、最大相对误差158.8382%，超过不变的15%/30%门槛。`measured-cost-r2`保留失败candidate/report，没有可运行参数。离线诊断显示，phase-blind接口忽略背景规模，bucket8合并bal2与bal8；即使改为精确布局/phase的非负加性常数，同布局重复差异仍令最优相对RMSE下界为23.5614%。详见`measured-tp2-cost-contract.md`及`evidence/ep4-tp2-ablation-20261108/interference-model-diagnostic-307623.json`。下一步先分析可在线观测的机制/上下文特征，不能仅调常数、删重复、放宽门槛或原包重跑。实测代价门仍未通过，不启动正式消融或混合拓扑；306991的旧窗口拒绝结论保留。

2026-11-09时序审计补充：两种固定探索模型均未得到全量通过的预测；Code长间隔期间存在共同输出停顿，尚不能归因GPU调度。另实证发现当前非分时准入/周期评分未传播其他tenant活动状态，干扰参数尚未真正接入这些评分。下一步实现`tp2-interference-observability-pilot-20261109.md`中的默认关闭观测及完整测试，再以独立新tag执行固定机制矩阵；不能仅换参数、改变lag窗口或直接进入正式比较。

1. **TP2 前置真机验证。** 单节点四卡，两个模型均 DP2/TP2/EP4；原始包 `ep4-tp2-cost-gate-20261108-r0` 预定依次运行 `dataplane,grid,interference`，但305641已启动失败，保留冻结包、禁止原包重投；当前以307623验收和实测模型拒绝为最新状态。不运行 trace、不沿用默认代价进行比较。dataplane 必须有往返共八次迁移、两个 TP lane 的复制与回读、全部 runner 安装/清理确认；任一失败立即停止后续阶段。
2. **实测代价与并发 pilot。** 作业 303594 的 72 点刻画 BS32/64/128 的失衡代价；新 grid 补充小 batch、1K/4K/8K/16K 上下文和 context skew，interference 测另一模型长 prefill/decode 干扰。记录原始数据 SHA、拓扑、拟合误差与有效范围；拒绝缺点/无有效窗口/抢占的校准，不把 TP1 参数冒充 TP2 参数。先检查 100K 请求相对每组容量、在途请求数、headroom 分散和 decode 速率，再冻结所有策略共用的并发/到达率。
3. **消融与基线配对。** pilot 与正式运行分开；正式参数冻结后，至少三个预注册 trace seed（911、914、917），配对策略顺序轮换。每作业≤1h，必要时将配对拆为独立、完整且可追溯的运行；不能把不完整试验拼接为通过。
4. **混合拓扑。** 仅在 TP2 迁移、代价、消融和性能门槛通过后进行。预注册主配置为长请求模型 DP4/TP1/EP4，交互模型 DP2/TP2/EP4，同一节点四卡。须先实现逐模型 DP/TP 配置、逻辑 rank 到物理卡映射和跨拓扑共享内存账本；当前统一拓扑代码不能仅改标签复用。交换模型拓扑作为独立验证，不能混作同一重复。
5. 原 EP4TP1 的 BS128/192/256 扩展仍保留为待办，但不挤占 TP2 正确性前置作业；三轮冻结配置和失败上传保持原样记录。

## 严格对照

所有策略共用中心 waiting queue、真实 prompt/原输出长度、current-length 准入、物理预算、并发限制和 trace；不运行分时策略，不使用 oracle，不做中心 waiting queue 消融。

| 对照 | 与完整系统相比唯一变化 | 目的 |
|---|---|---|
| no_move | 不进行任何迁移 | 验证长请求准入改善和 Conv 退化 |
| simple migration | 独立的简单、不考虑代价的迁移基线 | 比较是否优于简单迁移，不将其当作严格单开关消融 |
| full | 饥饿迁移 + 实测失衡代价 + 保可执行的周期均衡 | 本文系统 |
| no_periodic | 仅关闭周期均衡 | 验证主动引入失衡后的恢复作用 |
| no_imbalance_cost | 仅去掉失衡代价项，保持候选集、迁移成本、触发和可行性判定 | 验证实测失衡代价的决策价值 |
| no_starvation_migration | 仅关闭饥饿迁移，保留周期均衡 | 验证长请求解阻塞机制 |

现有 `ca_return_v2_g` 把准入策略改成随机，不是严格的 no_imbalance_cost。现已新增 `ca_no_imbalance_v2_g` 与 `ca_periodic_v2_g`，并使用 `ca_admit_v2_g` 作为 no_periodic；具体语义、两组限制和代价文件门槛见 `measured-tp2-cost-contract.md`。这些新增代码尚未进行端到端真机验证，也不在已冻结的305641包中。所有机制报告触发次数、候选数量、失衡变化、预测与观测代价，不能仅凭吞吐表推断消融生效。

## 性能门槛与失败处理

预注册主目标是受约束的长请求 TTFT，而非要求所有指标同时最小：相同 trace 下，完整系统相对 no_move 的 Code 平均 TTFT 降低至少 20%，Conv 平均 TPOT 退化不超过 20%；同时报告 Code 逐请求及 P95 TTFT、Conv P95/P99 TPOT、Conv TTFT、吞吐、完成率、SLO、抢占、迁移次数/字节/停顿。

进入混合拓扑前，三个配对 seed 都需完整、方向一致，并报告全部原始值与配对差异。完整系统应相对 simple/no_imbalance_cost 改善 Conv 代价而 Code TTFT 不显著牺牲（预注册容忍范围 5%），周期均衡和饥饿迁移消融应有相应机制证据。三个 seed 不足以证明对任意工作负载都最优，论文不作这种声明；进一步负载强度验证需独立列出。

没有消融差异时，先检查机制是否实际触发、GPU/TP lane 预算、失衡模型残差和在线干扰，再修复算法或用容量指标定义新的 pilot。新协议、旧负结果和参数改变原因都保存；不能通过删除失败轮、挑最好 seed、给 baseline 不同资源、缩短 Conv 输出或换用 oracle 来达到门槛。门槛未通过就不声称“已获得最佳性能”，也不自动跳到混合拓扑。

## 当前 session 的持续执行

活动检查点的仓库源为 `evidence/ep4-tp2-ablation-20261108/campaign.json`。每轮结束前写入已知结果、正在运行的 job/tag、观察时间、修复和可执行下一步，然后运行：

```sh
/usr/bin/python3 kvcached/experiments/xplace/tools/session_watchdog.py publish \
  --config evidence/session-watchdog-20261108/config.json \
  --context evidence/ep4-tp2-ablation-20261108/campaign.json
```

监控每 300 秒检查异常和正常 turn 完成，将触发信息、上一轮结论和这个检查点一起发到当前 owner。不直接替 session 提交作业；收到消息后必须核对实时状态再执行。用户取消、STOP、campaign paused/blocked/complete 时不触发。全部证据门槛通过才可将 campaign 标记 complete；每个已通过门槛必须引用证据文件。
