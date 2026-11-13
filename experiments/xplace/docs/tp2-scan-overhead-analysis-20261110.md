# EP4TP2 图观测开销：离线原始产物验收

## 适用范围

`tools/analyze_scan_overhead.py` 验收预声明的两个顺序 block：off→collect、collect→off。
每个 condition 包含128/256的均衡和极端布局，共16个测量case、3072个请求、8对，
另外逐项验收4个warmup。它不拟合K16预测器，不运行策略，也不验收全部93个capture邻点。
没有GPU结果时，完整合成fixture通过只能证明分析程序的本地行为。

完整输出始终保留两个block的全部计划case、请求分母、原始状态、失败原因、
输入SHA和不可用对。缺失block、失败、预算跳过、截尾、抢占、未运行点不能被当作零开销，
也不能删除后用成功子集通过门槛。`*-cleanup-results.json`只属于失败后的清理记录，
不能覆盖不可变截尾；即使该文件为空或状态摘要误写为measured也不接受。

## 指标与证明

- 从每个计划请求的前端时间、输出计数和stride32收讫cadence重算共同解码稳态步时、
  mean TTFT和前端完成口径mean TPOT。收讫合并、最初零计数和重复最终计数允许存在，
  但逆行计数/时间、丢失最终计数、请求缺失、非法窗口和摘要不一致会拒绝。
- 全部8对均可用才判定三个指标；各指标中位开销≤5%、最坏对≤15%。
  比值通过已序列化数值的十进制运算计算，避免二进制`1.05-1`把恰好5%误判成超限；
  不增添阈值容差，不改变门槛，负开销照常保留。
- 验收源码/测试/输入哈希、238个唯一启动测试身份且零跳过、两条件启动顺序、
  正常进程组退出、3300秒软期限及启动下一条件前至少1500秒余量。
- 验收前后frontend证明、8worker的配置/PID/时钟、4 A800共址、DP2/TP2/EP4、
  互补expert分片和独立KV命名空间、同一模型路径及同物理预算。
- graph start/stop必须使用归档frontend及worker证明的Linux时钟，不能读取分析电脑的
  本地时钟。逐lane验身份、源码、事件序号/字节守恒、零丢失、匹配的host replay调用及窗口覆盖。
  host replay返回不代表GPU完成；整case的调用统计不声称逐请求或逐phase归属。
- 启动/condition wall和采样CPU/RSS单列。采样检查PID/进程组、计数/时间单调及单位一致，
  保留读取失败计数，不把采样最大值或末次CPU计数称作精确峰值/累计资源开销。

分析协议绑定冻结部署manifest、图开销设计、完整238测试身份、全部当前runtime源码和
两个实际分析工具依赖。当前工作树的clock参数修复仅用于离线分析；原r5/r6及graph GPU包不改。
终态Slurm、归档完整性和传输SHA仍必须单独验收，工具exit0只是成功生成报告。

## 验证与失败记录

测试包含完整双block、缺失block、协议/源码/CLI身份、全部计划分母、
截尾/晚到清理、NaN/溢出、图丢失/错lane/clock/PID/来源、EP分片及namespace、
共享期限和5%/15%边界。独立常速oracle直接规定三项指标；另用实际
`imbalance_calibration.measure_layout`产生CPU原始文件，再交给分析器，避免只验证手写摘要。
完整/缺失block和未冻结传递依赖还经过独立CLI子进程验证。

首轮98通过/2失败暴露：start身份仍意外读取分析主机时钟；恰好5%的二进制运算误判。
修复分别统一使用已证明的归档clock、对序列化指标做十进制比值；失败源码、日志、差异均保留。
100/117通过的中间版本与后续最终本地/隔离记录分别保存，不覆写、不当作GPU收益。

冻结入口：`evidence/ep4-tp2-ablation-20261108/scan-overhead-analysis-20261110-r0/protocol.json`。
测试及审阅：`scan-analysis-resume-tests-*/receipt.json`、`scan-analysis-resume-review-20261110.json`。
执行时指定该目录的冻结工具、对应两个终态job root、原graph部署upload目录、
`--manifest-sha256`、`--analysis-protocol`及全新`--output`目录。

仍先完成310921的终态诊断和修复后block0，不能以本地分析器完成为由插队提交graph作业。
真实EP4TP2预测、容量/长上下文、完整评分和同条件基线消融均不由本分析器自动通过。
