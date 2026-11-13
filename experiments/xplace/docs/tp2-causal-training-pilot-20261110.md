# TP2 因果校准 pilot：运行入口与采集协议

## 结论

已将在线采集接到真实 `no_move` 共享 waiting queue 和 `engine.stream` 调用点，并增加 `dataplane,causal` 入口。本轮仅完成本地实现和 CPU 验证，**没有提交 GPU 作业，没有拟合或接受联合预测器**。

新入口复用现有准入、排队和清理路径，不启动分时策略，不调用完整评分。TP2 trace 的实测参数门没有放宽；`causal` 是单独的训练采集阶段，不是绕过门槛的正式性能实验。现有 DP2 prior 仅沿用在 no_move 基线的准入增长估计中，不是新的响应预测或实测参数。

## 固定采集设计

实际输入协议：`evidence/ep4-tp2-ablation-20261108/causal-training-plan-20261110-r0.json`。

| 项目 | 固定值 |
| --- | --- |
| 输入 | 原 SWE10 ≥100K prompt / 64 输出；OpenThoughts bank256 原始输出长度 |
| bank SHA256 | `3a38e81a5699270758952ecfec83b2cc699ca016fe9f3e587f145829165dd3b1` |
| 训练 | 4 case，Conv 请求数 4/16/4/16，各加 1 条 Code，共 44 请求 |
| 顺序与请求 | 按 prompt key 字典序确定，不按运行结果或长度挑选 |
| 到达 | Conv 在 trace 0 秒，Code 在 30 秒；沿用 trace 的计划到达时间，登记时刻另存 |
| 拓扑 | 双模型各 DP2/TP2/EP4，共 4 GPU |
| 准入 | `no_move`、current-length、额外 256 token、自然容量减 3 GiB；并发上限 `[1,16]` |
| 窗口 | trace 30 秒 + drain 210 秒；未完成保留截尾，不更改生成上限 |
| 共同目标 | K=16；as-of 时两模型全部已登记且未终止请求，包括仍在中心队列的请求 |
| 采样 | 100 ms 周期、80 ms deadline、最多 200000 事件 / 128 MiB 编码数据 / 64 请求 |

四个 case 的**原始配置输出总数**依次为 61666、211980、49169、220594；这些数是负载元数据，不是预测特征，也不是本轮生成的 token 数。受控 `stream` 仍保持 `min_tokens=max_tokens`、`ignore_eos=True`，不能称为自然 EOS 实验。

目标是首次前端消费累计输出、达到 as-of count+K 的时刻；不是 GPU 生成或设备完成时刻。计划保留无法达到 K、缺失、取消、错误、合并输出及快请求达到 K 后的干扰尾部。当前仅冻结目标定义，尚无将这些记录转换为可部署联合预测的拟合器。

## 接线与截止

- `run_trace(..., receipt_observer=...)` 仅接受 `no_move`。登记发生在共享 waiting queue 的 append 之前；实际登记时间与原 trace 计划到达时间同时可辨认，不向过去填补可观测状态。
- 请求仍经现有 `InFlightLimiter` 和 direct admission。观测不改路由、prompt、生成上限或原始 reserve；登记状态不包含未来输出长度。
- `trace_cutoff` 独立元数据保存当时 cohort，即使有界事件区已满也保留。普通 `requests` 终态可能晚于这个边界；离线截止判定必须使用该元数据，不能用清理后的终态反写 as-of。
- 先关闭采集，再取消请求和清理；随后单独落盘 `collection.json`、usage、case/error/status。setup 中失败时仍保留协议及可取得的部分采集，没有 `trace_cutoff` 表示没有建立正常窗口。
- 关闭采集可不立即解码整个事件区，避免在 trace 清理入口重复构造大报告。报告只在 case 结束后序列化；128 MiB 限制不是 Python RSS 上限。
- 外层取消现在也回收 trace 的完成/失败等待任务；此前这些任务只在正常离开 `asyncio.wait` 时取消。这是本次 allocation deadline 接线需要覆盖的清理路径。

## 条件与独立性

入口区分 `off`、`producers`、`collect`，开关在两个引擎创建前设置。`off` 关闭 producer 和 receipt/polling；`producers` 只运行 producer；`collect` 才运行完整采集。后两者之间的差异**不等于全部插桩开销**。完整开销必须同 allocation、同输入、同物理预算，重启进程做 off/collect 配对；预定顺序为 off→collect 与 collect→off 两个 block。

独立验证候选采用另外四条 Code key 和 Conv 第 129–192 条 key；它们与本轮计划训练请求不交叉。这仅是预留的未来运行划分，不代表原始语料从未看过。执行入口拒绝把 case 改名为 validation；必须先冻结模型、特征/缺失处理和验证协议，再采集新的验证运行。r1–r4 永远是已看探索数据。15% relative-RMSE / 30% 最大相对误差门不变。

**当前不具备安全提交条件：**尚需完整实际 frontend/config/scheduler/executor/worker/model-runner/serializer/source 清单、计算节点逐项校验、GPU 下配置与时钟核验、CPU/RSS/worker 开销及 loss/coverage 验收界限。完整协议中的 overhead/coverage acceptance 明确仍为 pending，不能把本文件称为已经完成的部署冻结。

通用 `xplace.sbatch` 的默认 startup 仍会跑整个测试目录。候选包不能直接提交；新 staging 必须冻结非分时 startup nodes 并替换该默认调用，和源码清单一起验证。每个 allocation 最多 1 小时，进入下一 case 前需留出窗口与清理余量；未完成 case 不补筛选或盲重跑。

## 本地验证

最终本地 71、隔离 71 项通过，零跳过。包括真实 no_move 控制流上的排队请求登记、原始输出保持、收讫、事件满后的截止状态、外部取消及等待任务回收；另以 fake 引擎验证 main 的模式设置、启动/关闭和同作业 dataplane→causal 调用顺序。这些都是 CPU/fake 数据，不证明 GPU/TP SHM 正确或开销可接受。

受影响回归 52 项在 r2 通过；随后 r3 只收紧 causal 专用 CLI 配置一致性，并新增 6 条对应测试，没有重跑未改变的回归路径。CLI 必须显式匹配协议的 reserve、并发、drain、seed、cache 和最大上下文，不能默默记录与实际执行不同的配置。独立归档使用实际 bank 核验三种模式、四例/44 请求，且没有导入 vLLM/torch；它明确标记为不能提交的本地候选包。

首轮 59 通过、2 失败完整保留：两条新夹具把 controller 标为 TP2，却只提供旧 FakeTenant 的一个 memory lane，生产 guard 正确拒绝。只在新夹具补充两个带身份的合成 lane；未放宽生产检查。修复后的 61 通过记录也保留，随后新增入口测试得到 65 通过。

复现脚本：`evidence/ep4-tp2-ablation-20261108/verify-causal-pilot-20261110.py`；每次使用新 label，不覆盖旧收据。实测代价、容量、完整对称评分、TP2 配对消融与混合拓扑门均未通过。
