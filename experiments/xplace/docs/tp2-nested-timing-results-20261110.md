# TP2四范围机制作业309773：真实结果与边界

## 1. 终态与产物

309773（`ep4-tp2-nested-timing-20261110-r4`）于2026-11-10 **06:20:16–06:53:48 CST** 在 `d1n41a28g03` 运行，Slurm `COMPLETED/0:0`。sbatch自身记录06:20:17入口，与Slurm调度开始差1秒，两者不混同。06:55:18实查用户队列为空，已有本地/远端结果包均不存在，随后仅归档取回一次。

- 单节点四A800、双模型DP2/TP2/EP4、1小时上限；没有新allocation或重提交。
- 启动按冻结的24个非分时测试node执行，**220 passed / 0 skipped，78.39s**。
- 归档257文件、冻结44文件及全部计划/SHA通过验收。24例、336请求，off/on各12例168请求，零抢占；原Code 101167-token prompt/64输出、Conv 4096 prompt/3000输出未变。
- 八次全lane迁移共复制5,196,742,656字节，耗时中位21.612ms、最大64.571ms。这是短上下文数据面证据，**不是长上下文容量或实测代价总门通过**。
- 12个on例各13进程、8 worker lane，合计2,036,081事件。96份worker记录最大36,292事件，小于100,000上限，缺失/溢出/未恢复检查通过。事件数最小2的空活动记录也保留，不能说每个worker均执行了四种范围。
- 原始清理日志中的socket退出未确认及force-kill剩余进程警告保留；发生在job phase通过后的清理期，不据此原包补投。

归档SHA：`d09d981cce069dbdfdd9c8d8dca25ba74ec5adceb8aa5a1296e1f8cc6a192881`。
artifact audit SHA：`112db985d23524afc84d1a3c8ada1d8cca33572a052d673918312b2697b6e9e9`。
嵌套分析SHA：`206c37f157a0bcb04bf7a0bc0a7a6525a1927b10695622c12319a9ea1d206bc4`。
独立复核摘要SHA：`cd24e81b84a6099b69a49dbb32d1ee1824e5b3bbf442d4136f3989514a7ba6a3`。

## 2. 长间隔由哪些host范围覆盖

沿用预先固定的100ms原始callback间隔阈值；off共83个，on共103个。不按callback中的token增量归一化筛选，不删除合并callback。全部24例、12对以及无长间隔的6个solo条件保留；on-bal8-r2只有1个长间隔也保留。

以下统计仅针对on的103个长callback间隔，每个角色有两个TP lane，即206个gap/lane观测。数字是各lane对该gap的**exclusive host覆盖比例**中位数及范围，非相互独立样本，不求显著性，不跨lane相加为设备时间。

| 角色 | 主要范围 | 中位覆盖 | 最小–最大 |
|---|---|---:|---:|
| Code DP0（实际请求所在rank） | `runner_to_list` | 98.485% | 83.651–100% |
| Code DP1（dummy rank） | `dp_coordinate` | 99.831% | 0.665–100% |
| Conv DP0 | `runner_to_list` | 98.614% | 83.143–100% |
| Conv DP1 | `runner_to_list` | 98.681% | 81.437–100% |

Code DP0的`runner_sample`和排除子范围后的`runner_bookkeeping`中位覆盖分别仅0.0268%、0.0131%。这将r3的外层长sample进一步定位到**含既有copy/event等待的输出物化范围**，而不是证明sampling kernel慢、纯D2H慢或某个NCCL kernel慢。Code DP1主要停留于DP协调host范围，但其余rank未必同时停留于DP协调；不能把所有lane的长等待合称同一host barrier。

每个长gap取八lane最小外层host覆盖，103个值中位99.794%，范围0.741–99.941%。三个低于50%的例外全部保留：

| on例 | 零基gap index | gap秒数 | 最小lane覆盖 |
|---|---:|---:|---:|
| bal16-r0 | 6 | 0.885854 | 0.741% |
| bal16-r0 | 38 | 0.887756 | 1.025% |
| bal2-r2 | 38 | 0.555282 | 0.957% |

103个on长gap均未与≥100ms的async/thread heartbeat延迟区间重叠。这只是本批后验重叠检查，不排除其他host调度因素，也不是在线可用特征。所分析窗口的跨label重叠为0；分析器仍按子区间并集处理，不能将这一观测变成普遍不重叠假设。范围外时间仍为unknown，不是设备idle。

## 3. 配置与独立复核

全部96份worker配置一致记录：`async_scheduling=False`、`enable_dbo=False`、`enable_eplb=False`、PP1、TP2/DP2、无speculative decoding、`disable_nccl_for_dp_synchronization=False`、`use_sequence_parallel_moe=True`、`all2all_backend=allgather_reducescatter`。实际DP/TP group大小2，EP group大小4；每lane的rank映射保存在原始记录。r4因此补齐了r3仅从源码推断的配置值，但不追认r3的17份作业后源码快照为历史部署锚点。

除既有33项合成测试准备外，本轮没有重跑未修改的测试。新的一次性复核脚本不import原scope parser/analyzer，直接重新配对原始enter/return事件，用sweep-line计算区间并集：核对**96份worker、668,292个范围、96个测量窗口和824个长gap/lane窗口**的inclusive并集、直接子并集及逐调用剩余时间。所有校验通过，原分析输出未重跑或覆盖。

## 4. 全部12个off/on差异：不能视为纯开销

数值为 `100 × (on/off − 1)`，未扣除开销或选择条件。固定轮序已原样执行；这些差异混合了hooks扰动与轮间变化，既不能宣称计时开销可忽略，也不是正式收益。

| pair | 例总时长 | Code prefill | Code TPOT | Conv全请求均值TPOT |
|---|---:|---:|---:|---:|
| bal0-r0 | -1.84% | -2.02% | +0.68% | N/A |
| bal2-r0 | +1.41% | -10.40% | +44.50% | +1.43% |
| bal8-r0 | +2.64% | +16.62% | -39.63% | +2.76% |
| bal16-r0 | +11.13% | +17.06% | +28.83% | +11.43% |
| bal0-r1 | -0.08% | -0.06% | -0.34% | N/A |
| bal2-r1 | +1.30% | -5.44% | +36.52% | +1.26% |
| bal8-r1 | -1.60% | -3.42% | +8.58% | -1.59% |
| bal16-r1 | +5.72% | +5.70% | +31.17% | +5.89% |
| bal0-r2 | +0.27% | +0.21% | +1.03% | N/A |
| bal2-r2 | +7.79% | +5.80% | +55.47% | +7.82% |
| bal8-r2 | -3.32% | -4.50% | -33.38% | -3.32% |
| bal16-r2 | +5.70% | -4.36% | +105.04% | +5.90% |

Code TPOT差异中位 **+18.705%**，范围 **-39.631%至+105.044%**。Conv九个有请求的pair中位 **+2.757%**，范围 **-3.322%至+11.432%**；三个solo条件为N/A，不填零。不能把on数据的误差/性能直接当作timing-off部署的预测验证。

## 5. 对下一步的约束

1. 两类host等待位置已由新范围支持，无需原r4包盲重跑；**GPU执行/collective根因仍未确证**。若必须进一步作kernel级归因，要先独立记录更低层观测协议与成本，不能从本结果直接宣布物理根因。
2. Code DP1有dummy与DP协调的真实活动，再次否定“逻辑idle即可映射other_active=False”。完整评分仍缺因果可用的wave/in-flight状态、有效时限和显式unknown分支。当前stop导出的范围不是已部署的在线特征；事后返回时间、peer未来phase或真实剩余输出长度不得回填预测。
3. 先基于这些已公开探索结果审定预测目标、特征可用时刻、horizon阶段转换、对称索引及missing/stale行为，再实现并测试完整非分时评分。旧Conv-layout bucket不能充当通用peer索引，常量phase不能铺满horizon。本轮没有冻结估计器、生成可运行参数或更改评分代码。
4. r1–r4都已查看，不能作为新的独立holdout。未改动的95个decode点、12例失败干扰和55.5374%/158.8382%误差保留；15%/30%门不变。参数/契约冻结后，仍需独立、可部署测量条件下的预测验证及长上下文/容量pilot，之后才做同条件配对消融。
5. 四个campaign总门均false；没有正式消融、混合拓扑、分时算法、旧observer恢复或PAUSE删除。不得利用本机制作业的12对差异替代论文收益。

## 证据

- `evidence/xplace-ep4-tp2-nested-timing-20261110-r4/fetch-309773.json`
- `evidence/xplace-ep4-tp2-nested-timing-20261110-r4/artifact-audit-309773-v1.json`
- `evidence/ep4-tp2-ablation-20261108/nested-artifact-acceptance-309773.json`
- `evidence/ep4-tp2-ablation-20261108/nested-mechanism-309773-v1.json`
- `evidence/ep4-tp2-ablation-20261108/nested-mechanism-summary-309773-v1.json`
- `evidence/ep4-tp2-ablation-20261108/review-nested-results-309773.py`
- `evidence/ep4-tp2-ablation-20261108/nested-terminal-fetch-preflight-20261109T225518Z.json`

原始报告保留全量24例、103个on长gap、全部lane、12对原始数值以及三个低覆盖例外；本页仅摘录描述性结果。
