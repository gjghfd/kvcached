# EP4TP2 causal pilot 启动失败与最小修复

## 310540 的实际结果

2026-11-10 13:25:05（北京时间）开始，13:27:25以 `FAILED/1:0` 结束，
节点 `d1n41a23g03`，共2分20秒。启动检查532通过、1失败、零跳过。
唯一失败为 `test_real_twelve_process_transport_binds_scheduler_rank_and_all_lanes`，
停在 `all(event.wait(8) ...)`；该测试耗时10.561秒。
`main.log` 不存在，`out/` 无文件，没有运行模型推理或采集任何因果训练样本。

归档只取回一次，SHA256为
`b0264e0248fe27f0a094eab2fa41f3184fee35eb213b74547b1288be03afc2d5`。
33个runtime、31个测试及5个输入文件逐一与冻结r5一致。
原分析协议照常运行一次：所有8对不可用、覆盖率门和预测门均不通过。
该失败不能用作开销零值，也不能被新作业的成功结果覆盖。

## 原因边界与修复

已知原因是至少一个spawn子进程未满足固定8秒ready假定。
原日志没有各子进程的PID、就绪状态及退出码，无法进一步断言集群上具体是
慢启动、调度延迟或提前退出；没有证据将其归因于CUDA、模型或网络。

本地CPU控制对照只延迟第一个engine进程9秒，原冻结测试在同一行失败；
相同延迟下修复版通过。它证明旧测试对启动时序脆弱，不证明远端延迟恰好为9秒。

只修改 `tests/test_causal_observation.py` 的同步：

- 所有12个进程共用60秒启动期限，而非逐个8秒等待；死亡进程立即失败。
- 报错包含每个进程名称、PID、ready及exitcode；所有lane/身份/时钟断言保留。
- worker sample fixture由协调器释放，90秒安全上限覆盖启动与采集。
  原20秒独立到期可能让先启动的worker在其他进程尚未就绪时退出；未收到释放现在明确失败。
- 保留有界清理、全进程零退出及socket清理断言；不跳过测试，不更改测量期限。

新增本地回归文件验证死亡、单一总期限、非法配置、晚启动和worker生命周期。
它不加入本次GPU启动选择；GPU仍运行原533项，唯一变化是现有fixture内部同步。

## 后续条件不变

r6从**原r5冻结包**派生，只有上述一个测试文件及总包清单改变。
runtime、launcher、数据、配置和部署manifest逐字节相同；不能混入当前工作树的新图观测代码。
部署manifest仍为 `e2395af30aba6eb009928651a0b563582946aeba1f03910f21fbe23d5f554eef`。

按预声明，下一块为 `collect → off`（block1），不是按性能选择顺序。
由于r5的block0完全没有执行，后续仍必须补齐修复后的 `off → collect` 块，
才能计算双顺序开销。K16、100ms/80ms、240秒窗口、原请求长度、reserve256、
15%/30%独立预测门及5%/15%开销门全部保持。

128/256图开销分析器开发在发现终态后暂停，草稿保留于evidence，**没有验收、没有部署**。
原图开销冻结包和全部93点扫描方案保持不变。无正式消融或混合拓扑结果。

证据入口：`evidence/xplace-ep4-tp2-causal-calibration-20261110-r5/`，
`evidence/ep4-tp2-ablation-20261108/causal-startup-*/`，
`evidence/xplace-staging/ep4-tp2-causal-calibration-20261110-r6/repair.patch`。
