# r8 双顺序块验收与负结果（2026-11-12）

## 执行和原始证据

- 反向块 `315554` 于 2026-11-12 20:12:27–20:53:43 CST 执行，`COMPLETED/0:0`，耗时 41m16s，节点 `d1n41a28g03`，单节点四张 A800。
- 21:21:55 CST 的一次查询确认全用户队列为空、sacct 为终态；scontrol 的 `Invalid job id` 原文保留，不覆盖有效 accounting。
- 单次归档和传输：155,596,005 字节、394 个安全成员，SHA256 `7829bacbde9bf08e37ab625cd508f770672d0e9ff6deec5db14a499b9be51998`。
- 两个块的 70 个上传文件、69 个计算文件和 549 个精确启动测试身份均通过。使用原冻结 GPU 包，仅预声明的 tag/block 不同；没有替换负结果或增加种子。
- `315554` 的 collect/off 两模式各 8 个 worker、48 层 EP 专家分片、8 次迁移的源/目的双 TP 分片均通过。自然预算均为 `[38.75, 40.5] GiB`，共同物理预算均为 `[38.0, 39.75] GiB`；3 GiB margin 和 256-token reserve 不变。
- collect/off 的 process 记录均为 exit 0、无模式超时、无外层 forced cleanup。collect 的 `CancelledError`、未回收 gathering future、worker socket 警告全部保留；成功退出不能证明这些警告无害，也不把警告当作 Slurm 失败。

原始目录：`evidence/xplace-ep4-tp2-causal-calibration-20261112-r8b1/`。
单次下载回执：`fetch-315554.json`；源审计：`paired-source-audit-r0.json`；终态审计：`terminal-audit-315554-r0.json`。

## 双块配对结果

采用已有只读 `analysis-r8-artifact-r2`，协议 SHA256 `bd04c4bbeea653b18c68a8df1e01cdac19cb081642923f9011d416ec6be4c045`。
两个块一起分析到新目录 `analysis-paired-r0`，不覆盖父块的 r0/r1/r2 或历史失败。

下表仅为 **采集开关 collect 相对 off 的观测开销**，不是完整系统相对 no_move 的收益。负数也不构成系统优越性证据；区间原样保留。

| 块 | Case | Conv 逆总吞吐开销 | Conv 平均收包间隔开销 | Code TTFT 开销 | 活跃采样覆盖率 |
|---|---|---:|---:|---:|---:|
| 0，off→collect | train-c4-r0 | -3.91% | -3.92% | -16.09% | 80.61% |
| 0，off→collect | train-c16-r0 | +2.72% | +3.40% | **+17.77%** | 84.32% |
| 0，off→collect | train-c4-r1 | -2.72% | -4.98% | -5.89% | 80.74% |
| 0，off→collect | train-c16-r1 | -1.80% | -2.36% | -7.86% | 86.78% |
| 1，collect→off | train-c4-r0 | -0.05% | -0.05% | +3.60% | **78.77%** |
| 1，collect→off | train-c16-r0 | +0.92%～+1.03% | +1.04% | -5.57% | 86.22% |
| 1，collect→off | train-c4-r1 | -1.18% | -2.11% | +4.17% | 82.13% |
| 1，collect→off | train-c16-r1 | -0.33% | -0.60% | -18.29% | 84.74% |

- 8/8 配对可用，无缺失顺序块。全部 16 个条件 case、176 条计划请求记录保留：54 完成、122 截尾。父块为 28/60，反向块为 26/62；不删除截尾请求。
- 八个采集条件的覆盖率为 **78.77%–86.78%**，均低于预注册 90%。采集开放区间无保留事件丢失，关闭后的 76 次拒绝单列保留；不称整个原始流完全无丢失。
- Code TTFT 开销中位数为 -5.73%，但最差值仍为父块的 **+17.77% > 15%**。中位数或反向块的改善不能挽救最差门槛失败。
- 两个 Conv 开销分量通过本采集协议的验收，不代表完整系统相对 no_move 的 Conv 约束或迁移收益已通过。
- 数据面/预算/身份产物审计通过；`complete_overhead_and_collection=false`、`prediction_gate_passed=false`、`independent_validation=false`。四个 campaign 科学门槛仍为 false。

## 诊断边界与下一步

`mechanism-diagnosis-315554-r0.json` 核验父块八个条件的指标、覆盖率、工作负载和查询字节量与原 `analysis-r2` 一致，旧负结果未被联合分析改写。

反向块四个 collect case 共 1,634 个失败采样轮次：1,630 次 `TimeoutError`、4 次 `CancelledError`，均缺少组合 transport 结果。其中 1,629 轮仍有 batch-link 观测，1,341 轮与 Code TTFT 区间重叠。这些是采样轮次，不是请求失败计数；重叠不证明因果关系。

现有 `prediction_observations` 将 scheduler RPC 与 execution endpoint 结果组合返回；任一子任务被外层 80ms 截止取消，可能使整个组合结果缺失。已有成功样本的分组件完成时间不能代表被截尾的失败样本，也不能据此证明哪个组件导致失败。

下一修订先提供有界的组件级完成、超时和取消记录，覆盖 scheduler RPC、engine endpoint、worker endpoint 和 batch-link。需要同时保留失败轮次、部分返回结果、时钟/身份、阶段耗时和字节量；不得用迟到数据补成按时成功的预测输入。

在修改 GPU 包前完成诊断契约与 CPU 回归，特别覆盖部分完成、同一截止时点竞争、取消清理及保留上限。仍保持 80ms 截止、100ms 周期、共同预算、trace、并发、数据集和原输出长度；不能靠放宽阈值、缩短 Conv 输出或选种子过关。任何诊断 pilot 必须与 r8 分开记录，不冒充独立预测验证或正式性能结果。

本次不原包重跑、不提交正式 TP2 消融、不启动混合拓扑。先修复并验证采集机制，再做实测代价接入、独立配对验证和正式消融；仅真实结果通过后推进混合拓扑。
