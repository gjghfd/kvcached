# 等价求解器：下一轮实测准入门槛

2026-11-01续接：用户明确跳过4倍输出，沿用已完成的8192→24576三倍机制证据，
继续固定OpenThoughts3原响应+SWE-smith≥100K输入。273918于10-31 23:59:34完整闭合，
163份raw SHA、14强控制及两轮plain no_move完整tokens独立通过。274238于11-01
00:42:30在compute入口导入未同步的本地协调模块时立即失败，零推理，不是GPU收益。
新版本仅隔离`efficient_public_launch.py`及部署依赖，保持下述算法/图/预算/请求。
两种launch都要在只有实际同步文件集合的隔离目录通过生成、分析CLI及verifier导入，
不能依赖远端历史残留源码。旧失败与167冻结源保留于旧evidence及R2源备份。
新证据根`evidence/public-efficient-optimizer-chain-20261101-r2/`；无活动作业后保护
同步、唯一A800，300秒查询至终态。通过native/实测速率门槛才执行七策略及heldout，
输入不人工延长；公开源为长reasoning tenant，不称真人对话或natural EOS。

声明时间：2026-10-31，Asia/Shanghai。当前273918仅运行正确性控制，本文不改变
其冻结代码、请求、预算或提交第二个allocation。等价求解器尚未部署，CPU同预算
找到回迁不等于GPU收益。当前证据根为
`evidence/public-roundtrip-state-control-20261031-r2/`。

## 1. 当前作业必须闭合

- 14个same-shape serial/noop/暂停0→1→0控制完整通过；8次迁移hop的device bytewise
  echo、KV/请求状态、owner、copy peak与四rank sparse graph独立重建。
- 两轮不加logprobs的原始no_move完整greedy输出与serial/paired reference一致。
  不使用新增tolerance、截短响应、删请求或batch-invariant来替代该门槛。
- Slurm COMPLETED/0:0、canonical终态raw SHA、compute/local独立结果一致，才能部署。
  失败先完整归档诊断；273701输运失败与272106数值分歧原证据不改写。

## 2. 只改变求解器的等价表达

下一次无活动作业时保护同步独立入口 `efficient_workaware_trace.py` 及其依赖；
不覆盖canonical planner或当前控制入口。仍使用原robust objective、uncertainty、
gain gate、copy peak与原共享1秒预算。集中布局count变量的精确分组、非binding
headroom场景合并必须经独立rational certificate覆盖全部原场景；不丢弃潜在
binding场景。完整记录执行源码SHA，并使用 `verify_efficient_trace.py`，不能
因为原verifier不认识新证书就跳过原始token/物理检查。

先重放原native六个fixture：seed2026103101/2026103102，Conv人口2/3/4；每条原公开
Conv mark的输入/输出配对不变，Code仍90秒到达，窗口900秒。使用两独立Qwen租户、
两A800、DP2/EP2/attentionTP1、BF16/128K、默认sparse/FULL_DECODE_ONLY/compile0、
pause32与每GPU10240MiB受控预算。比较no_move、random_no_return、kv_only_headroom、
kv_only_workaware；所有策略同trace/资源，完整drain。旧失败timing不用来选择setting。

## 3. 回迁的可证伪判断

每个fixture同时报告：Code是否排队/迁移准入，Conv从哪个rank迁走，Code释放时刻，
首次可行回迁/实际回迁时刻，随机基线恢复均衡的自然完成时刻，真实暂停/复制/规划
成本及Conv客户端TPOT。将收益窗口按真实Code释放和自然恢复定义，不只截取算法
最有利时段。无回迁则区分不可行、经济gate拒绝、限时incumbent质量与控制周期延迟；
有回迁但净指标为负也如实保留。N2每条trace仅一个Code，不把其P99当总体tail证据。

## 4. 公开Poisson收益链不改为人工长输出

当前逻辑Conv租户使用公开OpenThoughts3闭合响应bank的原始长度，Code使用SWE-smith
原始长history；Conv名称是系统租户标签，不意味着AzureConv或对话语义。继续沿用
公开输入SHA/provenance，不人为延长输出。native通过后依据实测service/KV常驻时间
重建已有三档1.1/1.2/1.3压力rate，开放环验证stable、actual mapped预算与baseline
队列；没有排队/合法global-fit local-no-fit见证的case不能声称容量调度收益。

后续沿用 `public_poisson_followup.py` 的有限搜索：Code .0005/.001 RPS、exploration
seed2026103104/3600秒、独立held-out seed2026103105/10800秒及七策略。
选择仍由预声明pressure/coverage/queue门槛与完整比较决定，不因负结果更改SLO120秒。
Conv/Code/combined分别报告avg/P95/P99 TTFT与样本数，Conv报告avg TPOT；保留所有
Poisson零draw、offered请求和完整drain。稀疏Code在3小时held-out的期望样本只有
5.4/10.8个，其经验P99接近最大值，必须显式说明不确定性，不能据此声称总体P99
已有稳定泛化或保证50%提升。后续若需要更多独立seed/更长窗口，另行预声明并完整
保留原有限搜索结果，而不是重复抽样直到得到正结果。
