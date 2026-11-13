# TP2 四范围嵌套计时实现（2026-11-10）

## 范围与解释

本实现落实 `tp2-sampling-wait-review-20261110.md` 的四范围判别协议，不改变估计器、评分、迁移策略或工作负载。`XPLACE_NESTED_TIMING` 默认关闭，只接受0/1，启用仅允许机制观测阶段；off条件仍不安装热路径挂钩。

新增范围为 `runner_sample`、`runner_bookkeeping`、`runner_to_list`、`dp_coordinate`。前两者是外层sample的有序子范围，to_list是bookkeeping的子范围。DP挂钩定位GPUModelRunner方法实际globals所属模块，不是仅替换dp_utils导出。实例已有capture包装不改变查找目标。

所有八个worker记录线程、scope ID、父ID、enter/return/error；完整性校验拒绝缺失、错序、跨线程、错误父级、重复、越界时间、异常或未恢复。每个进程每例仍限制100000事件；溢出拒绝验收，不扩容或静默删点。活动范围禁止stop，重复安装拒绝，部分安装回滚；恢复冲突不覆盖外部绑定，记录错误并失败。

挂钩仅写入内存host时钟；不新增CUDA同步、tensor取值、逐步RPC、热路径文件读写或心跳频率。配置和源码SHA在边界读取。记录解析后的DP/TP/EP ranks、group名称、调度模式、DP同步分支、MoE sequence parallel和AgRs配置；不符合当前DP2/TP2/EP4、PP1同步调度配置时失败。

`runner_to_list`包含原有event等待，不是纯D2H拷贝耗时；DP范围返回不是EP collective的CUDA完成证据。逻辑idle不能解释为设备静止。分析仍须使用子区间并集，不能叠加嵌套耗时；保留100ms长gap阈值、无长gap与低覆盖例外、全部12个on/off差异。此实现和合成测试均不证明GPU根因或独立预测通过。

## 新包与源码门

新机制包以不可变r3 upload为基底，仅覆盖 `runtime/main.py`、`runtime/timing_probe.py`、`runtime/timing_calibration.py`，新增 `runtime/nested_timing.py` 与专项测试，并替换启动测试调用。旧39文件及tar保留。前轮本地 `live_scheduler.py`、`planner.py`、`cost_model.py`、`scoring_state.py` 的状态准备不混入本包。

计划仅新增nested配置；旧24例/336请求顺序、seed23、Code101167/64、Conv4096/3000、stride1、原始prompt bank、三轮与off/on交替均不变。单节点四A800、一个allocation、硬限1小时、sbatch起算3300秒软截止不变。

新启动源码门覆盖前轮24份快照；其中7份有r3历史SHA，另外17份只有作业后来源，不能追认r3当时字节。新作业必须重新逐项匹配24份SHA，不能用旧五项门代替。

## 启动测试修正与验收

旧r3启动脚本对整个tests目录运行pytest，包含混合文件中的分时fixture。保留该事实与原日志，不把旧218通过/1跳过改称全非分时。

新包冻结 `inputs/startup-test-contract.json` 的显式测试node清单、通过数与零跳过要求。混合 `test_general.py`、`test_runtime_fake.py` 仅选择非分时node；排除依赖tools的分析测试。sbatch写出实际 `startup-test-nodes.txt`，执行pytest时移除 `XPLACE_NESTED_TIMING`，避免旧fake Worker意外进入新分支；新专项自行启用开关。

本地完整非分时回归与隔离runtime/tests回归分别记录，不把本地未部署评分状态的通过数当作新包测试数。离线审计根据外部计划SHA定位冻结计划，重算13进程/八worker嵌套校验并核验启动清单。旧r3路径保持兼容，所有不完整/负结果输出仍保留，任何产物验收均不能自动放行四个实验总门。

执行收据、所有失败与修复差异、冻结SHA、测试日志和提交状态记录于 `evidence/ep4-tp2-ablation-20261108/nested-timing-implementation-record-20261110.json`。未得到真实新作业数据前，不声明本实现已通过GPU验证。
