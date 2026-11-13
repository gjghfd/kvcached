# TP2容量记录与共同预算修订（2026-11-11）

## 原因与预注册

r7的collect在预热前被容量保护拒绝；这不是性能收益或模型拟合结果。旧预算记录只保留
显存快照，没有保留参与计算的mapped-KV summaries，不能独立重算natural budget。
原始失败、44个off请求和缺失的collect条件全部保留，不补标为通过。

本修订读取r6/r7四个条件的原始显存快照，不重新运行旧性能分析。每个共享DP rank取
两个模型、两个TP lane中的最小free_mib，忽略非负mapped-KV得到保守下界：
`floor((free_mib / 1024 - 3) * 4) / 4`。四条件逐rank最小值为`[38.5, 40.25]` GiB。
原始最小空闲的最大条件间差为302 MiB；将其向上取整到0.25 GiB刻度得到512 MiB，
统一作为两个rank额外预留的观测波动余量。因此冻结共同预算为`[38.0, 39.75]` GiB。
没有使用TTFT、吞吐或策略收益挑选预算；不是声称512 MiB能约束未来所有显存波动。

原有3 GiB margin、256-token reserve、原始数据/输出、并发1/16、240秒窗口、
512 MiB/1M事件采集上限及所有完整性/预测/开销门槛不变。每个条件重新启动后仍须独立
满足固定共同预算；不足继续保存记录并拒绝，不能临时clip或改点。预算按共享DP rank
索引，不是模型索引，也不乘TP size。

## 新原始证据与验收

`measure_budget`新增schema、DP/TP尺寸、完整summary快照、逐rank mapped/free/舍入
计算项，以及memory请求、memory返回、summaries返回的单调时钟时间。mapped_bytes
必须是非负整数。计算与原实现相同，非causal路径预算语义不变。

新分析器独立从原始memory和summaries重算两个rank的自然预算，同时检查八个worker
的GPU身份、完整性、时间顺序、预算绑定和计算项一致。缺少输入、篡改mapped/free/
natural、颠倒rank或时间均拒绝；不再把runtime自报natural当作独立验算。
这只验证记录内的算术与来源，不把先后采样冒充原子显存快照或未来容量保证。

## 实验边界

新修订取代r7未完成的两顺序要求，但r7失败仍保留且不算完成。新tag从r7不可变包出发，
只应用本修订明确文件；重新冻结启动测试精确身份、部署manifest和独立分析源。
两个顺序off→collect及collect→off均使用相同源码/输入/预算，失败先诊断，不因性能
选择是否执行反向块。不与r6/r7混配，不执行尚未冻结模型的holdout。

本地和隔离启动测试各549项通过，包含全部544个旧测试身份；分析器本地和隔离各141项
通过，覆盖原始输入缺失和计算项篡改的拒绝路径。70文件包仅改变两份runtime、两份测试
及四份输入/清单，未混入图扫描改动。CPU验收不代表GPU容量、采集或预测门通过。
预算依据和冻结入口见`evidence/ep4-tp2-ablation-20261111/capacity-basis-r8.json`及
`evidence/xplace-staging/ep4-tp2-causal-calibration-20261111-r8/build.json`。
