# TP2观测实现说明（2026-11-09）

这是`tp2-interference-observability-pilot-20261109.md`的实现记录，不修改其24例配对、输入、种子、长度或15%/30%代价门槛。GPU验收前不得将本实现或CPU夹具称为真实机制结果，也没有生成可运行代价参数。

## 默认关闭与接口范围

只有显式`--phases dataplane,observability --num-ranks 2 --tensor-parallel-size 2 --enable-tp-migration`允许进入观测。它不能同时运行fit、trace或其他phase；仍须先完成同作业八次迁移校验。普通作业把`XPLACE_TIMING_ALLOWED`置零，engine不安装观测控制入口。worker保留一个关闭时拒绝调用的控制方法，但不覆写任何热路径方法。

`timing_probe.py`在每例timing_on开始时才对实例方法安装wrapper，结束时恢复原实例属性或移除新增覆盖，timing_off不走这些wrapper。部署vLLM的core、worker、scheduler、multiprocess executor及executor base五个源码SHA已通过SSH只读采集；观测作业加载模型前重新校验，版本变化则失败，不猜测私有接口。接口记录保存在`evidence/ep4-tp2-ablation-20261108/deployed-vllm-timing-interfaces-20261109.json`和`deployed-vllm-timing-futures-20261109.json`。

## 时间戳与语义

- 每进程每例最多100000条内存事件，达到上限后只累计丢弃数，不替换历史事件；丢弃、缺序号、错误时间边界、schema错误或缺lane均拒绝telemetry。导出包含pid、thread、case、service、DP/TP rank、worker GPU UUID和源码SHA。
- frontend有20ms async heartbeat和独立线程heartbeat。请求callback复用原stride1 cadence，并记录其所属线程/进程；不再把每个callback重复写进有界事件buffer。原cadence仍为`time.monotonic()`秒，新事件为`monotonic_ns()`纳秒，不能把原浮点时间伪称GPU精确时间戳。
- engine记录实际`step_fn`、scheduler的schedule/update边界、executor提交/返回/Future完成；worker覆盖execute_model、sample_tokens以及DP/EP空载rank的execute_dummy_batch。executor和worker的dummy次数、真实scheduled token总量分别核对，所有八个worker lane都须对应正确的共享四卡映射。
- scheduler的`accounted_context_sum`是当时调度器账面计数，schedule后可能包含计划执行增量，**不是已完成GPU KV状态的证明**。大于1的scheduled token计数也只标为prefill-sized，不能据此排除其他调度模式。
- 部署版本的FutureWrapper在engine读取RPC响应时完成Future，因此ready事件是engine观察到响应的时间，而非worker发送、kernel完成或CUDA event时间。worker方法返回同样只保证host调用返回。本实现不插入CUDA同步，也不修改Future对象、执行结果或原始异常。

## 测量与清理

`timing_calibration.py`按冻结顺序运行24例：第0/2轮同布局off→on，第1轮on→off。两条件的12例分别执行原schema2的完整窗口、输出和零抢占校验，不能合并成六轮或挑选某个条件。原Code prompt固定101167 token，Code输出64、Conv4K synthetic/3000输出及seed23逐轮生成规则不变。

timing_on的控制RPC只在例前/例后执行；先检查所有engine无请求、无batch queue和未完成Future，必要时仅在这些边界等待。测量期间没有新增逐步状态RPC或磁盘导出。原校准在Code开始之前的Conv到达decode检查及3秒settle在两条件都保留。

完整case耗时在请求结束时停止计时，不把随后共享盘写入纳入该数值。请求、telemetry、配对开销和状态在确认idle后导出。每对完整保留Code TTFT/TPOT、Conv全请求平均TPOT及case耗时；不删除慢例、不减去测得开销，不自动将timing_on数据用于拟合。

sbatch入口新增`XPLACE_JOB_STARTED_AT`，3300秒软截止包含Python启动前的测试/环境准备时间；时间从sbatch入口而非Slurm调度时间计量，作业硬上限仍1h。超时或流失败取消子任务、保存完成请求和partial progress、清理观测并保存失败/未完成列表。remote清理失败也单独记录，不能当成正常结束。重复调用已存在的观测输出目录直接拒绝。

## 验证与后续

专项覆盖关闭语义、buffer溢出/并发序号、原方法恢复、异常/返回/Future身份不变、pending Future拒绝结束、部分安装回滚、idle边界、全部lane与GPU映射、时钟/计数/心跳缺失、固定矩阵、版本SHA、完整两条件保留、子任务清理和真实运行中的软超时。完整及隔离staging结果以对应evidence日志为准。

此前一次完整回归命令误写了不存在的`tools/tests/test_fit_measured_cost.py`，pytest退出4且没有执行测试；原日志保留，正确工具测试位置为`tools/test_fit_measured_cost.py`。这不是实验失败或代码测试失败。

本次只增加观测机制，没有修复已发现的非分时评分状态传播缺口，也没有选择新干扰估计器。下一步须先在独立新tag验收观测是否完整、定位共同停顿，再冻结模型修复及独立预测验证。正式消融和混合拓扑仍被现有失败门阻止。
