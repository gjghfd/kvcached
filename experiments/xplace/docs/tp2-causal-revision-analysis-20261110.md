# EP4TP2 修订测量协议：独立验收绑定

## 验收变化与反例

新分析版本只服务于修订后的 causal 训练采集协议，不重跑或替换 r6 的拒绝报告。
新增四个端到端反例在旧实现上均被误接受：两个条件使用相同但未预声明的预算、
JUnit 重复身份而汇总数不变、缺失 `budget.json`、容量记录来自不同 GPU。
修订后四例均拒绝，仍保留全部八个开销配对记录。

- 将“533 个测试”常量替换为冻结 startup contract 的精确身份集合。核对汇总数、
  每个 testcase、重复/缺失/改名，以及汇总中未体现的 failure/error/skipped。
  修订部署合同包含 544 个身份；测试执行顺序本身不决定是否有效。
- 对计算目录 `runtime/tests/inputs` 的 69 个文件核对精确文件集合和 SHA，拒绝新增、
  缺失、改写或符号链接。第 70 个 payload 文件为 launcher，仍由归档验收单独核对。
- 将固定 `[38.75, 40.25]` GiB、3 GiB margin、可用的 natural budget、8 个内存采样
  lane 和 GPU 身份，与 worker attestation、startup phase 和每个 workload 交叉核对。
  不再把两个条件“彼此相等”当作“符合预声明条件”。
- 保留原 cohort、K=16、窗口、零丢失、active-round coverage、missed-tick、开销与
  独立预测门槛；所有未完成、不可用及失败行保留，退出码不表示门槛通过。

## 验证与来源

本地及隔离目录各 **133 项测试通过、零跳过**。38 个分析源码/fixture 文件另行冻结，
runtime 取自修订候选，分析 fixture 只用于 CPU 验证，绝不能混入 GPU upload。
已有 544 项隔离启动 JUnit 的身份和候选的 69 个计算载荷文件也已由新验收函数绑定；
这是读取已有 CPU 结果，不是再次运行启动测试，更不是 GPU 启动验收。

证据根：`evidence/ep4-tp2-ablation-20261108/causal-revision-analysis-20261110-r0/`。
分析入口为 `analysis/tools/analyze_causal_calibration.py`，配套
`analysis/analysis-protocol.json`；终态时必须先验证 `analysis-freeze.json` 和候选清单，
然后使用新输出目录执行，不能修改旧分析快照或旧报告。

初版新测试 fixture 误用了禁止覆盖的报告写入函数，随后仅修正 fixture 改写方式；
有效的四个行为反例及前后回执均保留。隔离测试通过后，CPU 输入绑定目录继承只读
权限而不能添加 JUnit，已只修复新建目录根权限并完成绑定，没有重跑测试或更改候选。

## 证据边界与后续

natural budget 来自经过源码证明的运行时记录。`measure_budget` 未导出其 mapped-KV
summary，因此分析器不声称从原始内存字段独立重算 natural budget；该限制在报告中
显式标记，独立容量/长上下文门槛依然未通过。

修订 r7 必须先在空队列、无其他 allocation 的保护下提交 block0 `off → collect`。
终态确认、一次性归档和失败诊断之后，才可用同一源包执行 block1 `collect → off`，
不能根据第一轮性能决定是否执行。CPU 验证没有解锁独立预测、正式配对消融或混合拓扑。
