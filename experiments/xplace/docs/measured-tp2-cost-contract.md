# TP2 实测代价与消融接口

## 当前状态

2026-11-09补测结果：307623的固定三轮/12例、168请求均通过原始窗口及计划校验，两个模型抢占增量为零，Code101167-token prompt/64输出及Conv3000校准输出不变。所有活跃请求覆盖before/prefill/decode完整窗口，decode窗口内每请求至少53个采样点；bal2/bal8/bal16三轮累计decode时长分别为29.8913/12.3132/22.6952秒。八次双TP lane迁移通过，复制总量5196742656字节，中位27.913ms、最大66.052ms；仍只是四条短上下文流，单runner ack未独立持久化，不替代长上下文或端到端配对。

**原始采集通过不等于代价模型通过。** 全部95个decode点的相对RMSE/最大相对误差仍为7.7589%/20.9665%，12例Code prefill为8.9709%/19.8089%，三例Code solo的跨tenant代价预测检查为10.7601%/13.9087%。干扰在9例活跃背景的27个标量观测上分别检查phase-blind与phase/bucket路径，共54次预测比较（不是54个独立实验），相对RMSE/最大相对误差为55.5374%/158.8382%；两个路径分别为57.6205%/158.8382%与53.3730%/151.9003%，均失败。`measured-cost-r2/fit-report.json`和`step-params.candidate.json`完整保留，**没有**`step-params.json`，不能用于TP2 trace。

诊断定位到现有模型表达不足：布尔fallback忽略背景bucket，phase表又将bal2、bal8合入bucket8。对所有27个活跃观测，保持原solo/before锚点，仅优化非负加性常数时，phase-blind/phase-bucket模型的最小相对RMSE分别为41.1157%/36.7605%；即使离线放宽为精确布局及phase分表，最小相对RMSE仍为23.5614%，单独优化最坏误差的下界仍为45.9771%。因此不能靠换median为相对最小二乘或仅拆bucket修复。同布局重复变化的物理原因尚未确证；这些下界仅适用于上述加性函数族，不代表所有在线可观测模型都不可能。下一步先对齐全部cadence、上下文增长与阶段边界，设计不依赖未来输出或repeat ID的特征/估计器，再预注册修复与独立机制pilot。15%/30%门槛不变，不追加重复直到通过、不删除负例，不因本次schema2追认306991。

验收与可复现诊断：`evidence/xplace-ep4-tp2-interference-20261109-r2/acceptance-307623.json`、`evidence/ep4-tp2-ablation-20261108/interference-model-diagnostic-307623.json`，对应本地脚本`accept-307623.py`、`diagnose-cost-r2.py`。归档100文件逐一核SHA，31个运行源文件匹配冻结manifest。未修改冻结包、拟合器、runtime或预注册门槛，尚未提交下一作业。

2026-11-09：305641启动失败已修复，306991完成真机采集并归档。八次双TP lane迁移、23个grid点/526请求/零抢占通过；与303594的72点合计95点，诊断decode相对RMSE7.7589%、最大相对误差20.9665%。这不是完整代价验收：bal8的Code decode仅2.962475秒，bal16的32-token cadence使20/32个Conv在decode窗口中仅有一个采样点。旧摘要只计算12个有效请求，完整来源校验正确拒绝该数据，未生成可运行参数。权威验收是 `evidence/xplace-ep4-tp2-cost-gate-20261109-r1/acceptance-306991-r2.json`；后缀r2是分析修订，不是新GPU作业。

2026-11-08 新增 `runtime/measured_cost.py`：TP2 trace 在解析 CLI 时、启动引擎前拒绝无来源/错误拓扑/不完整校准/高误差代价文件；进入 runtime 后再次核验。该改动不修改已冻结并排队的作业 305641，其任务仍是 dataplane、grid、interference，不运行 trace。

已有 72 点 TP2 固定 4K 数据不能直接套 TP1 系数。对全部原始点，以每个 rank 的 `count × (4096 + 512)` 作为中点上下文和已记录 layout 输入 StepModel，旧 DP2/TP1 prior 的相对 RMSE 为 42.99%，旧 DP4/TP1 fit 为 50.25%，最大相对误差分别为 77.19% 与 77.73%。所有点均保留，诊断见 `evidence/ep4-tp2-ablation-20261108/cost-transfer-diagnostic.json`。

固定上下文下，max rank count 与 max rank context sum 两列严格共线（比值 4608，秩为 1），不能独立识别每请求开销与每 token attention 开销。新 grid 的不同上下文与后续独立 pilot 因而是必要前置条件，而非为了挑选更好的曲线。

## 代价文件契约

`--step-params` JSON 保留 StepModel 的 `tenants`、`interference`、`code_prefill`，并增加：

- `provenance.schema_version = 1`；`provenance.topology = {num_ranks: 2, tensor_parallel_size: 2, expert_parallel_size: 4}`。
- `provenance.sources` 恰含以下五个角色，每项包含 `role`、同目录文件名 `file`、原始文件的 `sha256`、作业 `job`。所有来源随输入包冻结，不允许引用包外路径。
- `imbalance`：作业303594的 `calibration/imbalance/results.json`；`imbalance_phases`：对应 `out/phases.json`。
- `grid`、`interference`：新 TP2 校准作业的 `conv-grid.json`、`interference.json`；`cost_phases`：对应 `out/phases.json`。复制时使用不同文件名避免覆盖两个 phases 文件。
- 两个 phases 文件必须含通过的 startup/job 和相关校准阶段，真实拓扑须为 DP2/TP2/EP4、4张物理GPU；cost_phases 还须含通过的 dataplane。
- 所有 decode 行都必须 measured、零抢占、有效窗口≥3秒、窗口样本数等于请求数，且行数与通过阶段一致。interference 必须有 solo 和活跃背景，prefill/decode 观测窗口有效。
- `fit.cases`、`fit.relative_rmse`、`fit.max_relative_error` 使用**最终实际模型**对 grid 与 imbalance 所有行重算，不是调整系数前的训练矩阵残差。中点特征规则与诊断一致；上下文不齐的 grid 使用原 contexts 列表。

进入 trace pilot 的预声明数值门槛为相对 RMSE≤15%、最大相对误差≤30%；代码自行重算并与文件报告比较，不信任仅写一个 passed 标记。未达到门槛应分析残差/模型形式并修复，不能静默退回旧 prior，也不能删除不合意的点。

这些是样本内检查，**不是泛化或端到端收益证据**。2026-11-09 增加 `provenance.interference_requests`，每项含 `case`、同目录 `file`、`sha256`、`job`，与原始干扰行一一对应。加载器重算每个原始请求的布局、输出长度、cadence 和 prefill/decode 窗口；活跃背景中每个 Conv 都必须覆盖各窗口，不能只取有结果的子集。Code 的 prefill 按原始 `prompt_tokens` 归一化，而不是假定恰好100K。

`interference_fit` 记录并重新验证三组残差：prefill、干扰增量、同权重/同拓扑的 Conv→Code solo decode 参数迁移。干扰既校验 phase/bucket 分支，也校验模型API提供的 phase-blind fallback，均用上述15%/30%样本内门槛；不能仅通过一个新分支、却留下未经检验的旧 fallback。2026-11-09调用审计纠正此前“现有调用中使用”的表述：当前非分时完整策略的`layout_cost`未传播其他tenant活动状态，不能宣称干扰已接入其准入/周期评分；这是独立待修复缺口，不是删除任何验收分支的理由。Conv 增量以同一 case 的 `conv_before_s` 为锚，期间的上下文增长会混入增量，**不是整个共址执行的独立预测**。Code solo 只有一个长上下文锚点，不是 Code decode grid。更长单请求上下文、在线干扰和跨作业漂移仍必须由独立 pilot 检查。

## 可复现拟合与打包

新增 `tools/fit_measured_cost.py`，在本地读取完整归档，不占GPU、不访问登录节点计算。原始来源验收前不能生成可运行TP2参数，合成夹具不构成实测。306991的首次调用已失败并保留 `measured-cost-r1/fit-report.json`，不在原目录覆盖或盲重跑。以下为修复后补测的命令模板，`NEW_JOB`必须替换为实际作业；补测未完成时不能运行：

```bash
/tmp/xplace-venv/bin/python kvcached/experiments/xplace/tools/fit_measured_cost.py \
  --imbalance-job-dir evidence/xplace-fig2-ep4-tp2-20261108-r3/job-303594 \
  --cost-job-dir evidence/xplace-ep4-tp2-cost-gate-20261109-r1/job-306991 \
  --interference-job-dir evidence/xplace-ep4-tp2-interference-20261109-r2/job-NEW_JOB \
  --output-dir evidence/ep4-tp2-ablation-20261108/measured-cost-r2
```

拟合使用全部 grid+imbalance 点，最小化相对残差，非负最小二乘直接参数化单调 graph 增量，因此不再先拟合、再修改 graph 却沿用旧残差。要求实际测得 bucket1、不同上下文和满列秩；保存 NumPy/SciPy 版本、脚本SHA、设计矩阵秩/条件数、每点有符号残差、两种来源独立残差及测量范围，不删掉非单调点。原始 decode requests 的输出、布局、长度和共同窗口在打包时全部重算并记SHA。

输出目录必须全新。五角色源文件和所有原始 interference requests 按原字节复制；候选参数、失败原因与 `fit-report.json` 保留。只有运行时加载器验收通过才生成 `step-params.json`；失败不得静默回退 prior，也不能用 `step-params.candidate.json` 绕过加载器。进入下一冻结包时必须连同这些来源JSON一起传给 `xplace_ctl.py build --inputs`，不能只复制参数文件。这里的输入验收不能替代 dataplane 全部分片 proof 和真实作业归档验收。

## 所有策略共享的并发上限

`--max-inflight MODEL0 MODEL1` 默认 `0 0`（不限），sbatch 对应 `XPLACE_MAX_INFLIGHT_MODEL0` / `XPLACE_MAX_INFLIGHT_MODEL1`。上限统计已经提交但未完成的请求，包含引擎内等待、prefill、decode和迁移，不包含控制器中心 waiting queue；各模型独立计数，一个模型达到上限不能阻止另一个模型准入。参数统一传给所有策略、所有配对 segment，不提供按策略单独改上限的入口。

general、legacy/fast、native 都在同一中心队列等待上限空位，原 arrival 时间不变，TTFT包含全部等待。达到并发上限时不触发“内存碎片”迁移；仍允许对现有请求进行可行的周期均衡，不能因一个暂不可准入的队列头而把周期机制永久禁用。默认不限时保持原有周期 pending 检查语义。

`workload.json.max_inflight` 记录配置；`summary.json.concurrency` 记录峰值、阻塞检查数及清理后在途数；完成和截断请求记录 `concurrency_head_wait_s`（仅作为队列头时被并发门阻塞的时间，不等于全部排队时间）。当前只验证实现和合成trace，不预选上限数值；实际统一上限、到达率及预算须由容量/机制 pilot 冻结，而不是按正式收益反向选择。

## 固定重复干扰测量修订（2026-11-09，补测前冻结）

这是明确的测量协议修订，不是把2.962秒四舍五入为3秒，也不追认306991的失败行。64-token Code在TP2下可能自然短于3秒；保持输出不变，通过固定数量的独立同条件重复累计观测时间。旧schema 1仍要求每个decode窗口≥3秒；只有携带完整schema 2计划的新采集使用以下规则：

- 布局固定为 `[0,0]`、`[2,2]`、`[8,8]`、`[16,16]`，每轮按此顺序，固定三轮/12例，不按结果增加重复、删轮次或挑种子。沿用校准seed23及逐轮prompt生成规则，Code仍使用原首条SWE prompt/64输出，Conv仍为每请求4K synthetic prompt/3000输出；正式trace的原Conv长度不变。
- 校准Code和Conv的cadence stride均为1；TP1旧校准及正式trace默认stride32不变。保存全部原始请求、计划、每例两个模型的抢占增量；任何抢占拒绝。
- 每例before/prefill窗口仍≥3秒。每个decode窗口必须为正，所有Conv覆盖完整窗口且各有至少两个有效采样点，不取有效子集。每个活跃布局三轮decode窗口总时长必须≥3秒；不足时保留失败，不追加“直到通过”的重复。每例速率仍分别重算和拟合，不用三轮平均掩盖残差。
- `interference-plan.json`的schema/repeats/stride/布局/输出长度/门槛必须与冻结契约一致，原始行记录 `measurement_schema=2`。缺计划、缺轮次、重排、稀疏采样、提前结束或改变长度均拒绝。GPU阶段采集完执行相同来源校验；失败保存 `interference-validation.json`并停止，不能只凭循环结束标passed。
- 补测只运行 `dataplane,interference`，仍为同节点4卡/DP2TP2EP4/≤1h。复用并包含303594全部72点和306991全部23点，不重扫或筛选已通过的decode点。
- 分作业来源用provenance schema 2，在原五角色之外增加 `interference_plan`和 `interference_phases`；同一作业时后者可省略。grid与cost_phases同job，干扰数据/计划/原始请求/对应phases同job；新干扰作业仍须通过同作业八次迁移门。所有文件带SHA随参数冻结。

15%/30%的decode、prefill、干扰及phase-blind fallback残差门槛不变。重复可能暴露更大噪声或模型偏差，不能为通过而放宽。新采样开销、跨作业漂移和在线泛化仍须独立pilot验证；修订不意味着完整系统已获得优势。

## 单开关消融

| 策略 | 饥饿迁移 | 周期均衡 | 失衡目标项 |
|---|---|---|---|
| `ca_full_v2_g` | 开 | 开 | 开 |
| `ca_admit_v2_g` | 开 | 关 | 开 |
| `ca_periodic_v2_g` | 关 | 开 | 开 |
| `ca_no_imbalance_v2_g` | 开 | 开 | 关 |

`no_imbalance` 仅置零 planner 的性能评分/收益项，保留原预测值作为诊断，迁移成本、风险、增长余量、horizon、触发及可行性不变。用于安全余量和执行速度估计的实测模型仍保留；不是把所有性能估计一并删除。

该严格消融限制为两个 attention group：迁出目标 rank 后只有一个目的 rank，目标项开关不会改变候选生成。DP4 的目的 rank 启发式还含性能代价，所以禁止直接将此策略用于 DP4 并声称候选相同。测试验证 DP2 全部候选数量、目标/ordering、原始性能/迁移成本、horizon 和可行性相同；周期候选、风险相同，仅 scored gain 不同。

无失衡目标时周期均衡可能全部被拒绝，因为置零后的收益不能覆盖迁移成本。这是单开关的直接结果，不能再暗中增加另一套收益启发式来“救”消融。日志同时保留原始 gain、scored gain 和开关，便于解释与直接关闭周期均衡的区别。

`workload.json.policy_switches` 记录三个开关；admission alternatives 包含 `performance_cost_s`、`scored_performance_cost_s`、`migration_cost_s`，periodic decisions 包含 `gain_s`、`scored_gain_s`、风险与 net。中心 waiting queue、快路径及 general/type-agnostic 语义不变，不运行分时策略。

## 全量时序探索与调用审计（2026-11-09）

全部12例/168请求及27个活跃标量目标完成因果截止核查。固定留一整轮的精确布局均值模型误差44.7771%/165.6571%；固定布局+上下文+lag1s ridge只覆盖9个Conv-prefill目标，误差56.3971%/147.4857%，其余18个decode目标因两例缺近期速率触发整fold缺失，不能当成全覆盖。没有追加窗口、模型或挑选重复；所有失败仍保留，也不将同job交叉验证当作独立pilot。

Code长间隔期间两个模型输出多为共同停顿，而不是明确的交替高速推进。cadence只能证明主机输出何时可见，现有数据不足以区分前端、engine/IPC、GPU或collective原因。新机制观测方案见`tp2-interference-observability-pilot-20261109.md`：固定24例off/on配对、默认关闭的有界进程内计时、全部lane与零热路径同步写盘；尚未实现或提交，不用于放行现有负结果。

CPU阳性对照确认显式干扰API有效，但非分时`layout_cost`、准入和周期评分在干扰系数大幅扰动后仍完全相同。compact planner state未保留显式phase/prompt，修复须同时覆盖观测、状态传播和评分路径，并独立验证；不能仅换参数文件。详见`evidence/ep4-tp2-ablation-20261108/interference-callpath-audit-307623.json`。本轮未修改runtime/fitter/冻结包，没有可运行实测参数。
