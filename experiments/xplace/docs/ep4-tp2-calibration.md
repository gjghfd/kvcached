# EP4 / TP2 校准（2026-11-07）

## 2026-11-08 修复补跑

原作业300260没有有效TP2测量：引擎实际使用的PlacementScheduler在初始化时拒绝TP2。仅修改拓扑参数不足以使该调度器支持校准；旧CPU测试未实例化它。

修复版仅在TP>1选择CalibrationScheduler，允许TP并沿用原准入和统计方式，保留其他校验，拒绝迁移事务。PlacementScheduler仍限TP1；不能据此声称TP2迁移已实现。新增测试直接实例化两类调度器，覆盖TP2成功、原迁移调度器拒绝TP2及非法模式。

补跑tag为`fig2-ep4-tp2-20261108-r1`，作业303011，配置与扫描点不变。原始失败归档与修复验证记录均保留。当前由同一个Codex会话持有前台监控工具，不启动后台模型续接；提交、验收及后续修复状态记录在`../reports/progress.md`。

303011已完成69/72次测量，全部有效且零抢占，但内部扫描预算2800秒不足：运行2706.80秒后，无法再预留完整的120秒case超时，最后3次未执行。Slurm总耗时50m47s，并未达到1小时硬限。r2把扫描预算延长到3100秒，增加明确的停止原因与缺失case记录，保持其他测量参数和全部72次不变；完整重跑，不把父作业的69次合并进最终三轮。按实测启动约328秒、清理约12秒估算，3100秒扫描上限仍有约160秒余量；这不是对慢节点的完成保证。

r2作业303174在另一节点d1n41a21g03运行，12:29:51结束，仍为69/72、零抢占、incomplete。相同测量集合扫描耗时由2706.80增至3010.26秒（+11.2%）；匹配case的decode步时比值中位数为1.084，wall比值中位数为1.089。r3据本次容量不足排除该节点，保留3100秒预算和全部实验参数；新增原始nodelist记录以核对部署身份。节点速度会随时变化，不将此结果解释为永久硬件故障，也不据性能效果正负选取轮次。两次不完整数据不混入最终三轮。

## 2026-11-08 完整验收结果

最终作业303594（`fig2-ep4-tp2-20261108-r3`）在d1n41a22g02于12:41:26启动、13:33:51结束，Slurm为COMPLETED/0:0，耗时52分25秒，单节点4卡、硬限1小时。扫描2803.956秒，stop_reason=completed；计划72次全部有效、24点×3轮、5664请求、零抢占，无缺失case。

原始归档SHA-256为`c2aee26ca9847b0fcd7bc7c87596801bdf69ffb58160c8e3954ef9ad85ca9d9a`，13:34:43取回并核对全部149个归档文件。startup/imbalance/job均passed；8份worker记录验证两实例在同4张GPU共址、DP2/TP2/EP4及专家分片。启动测试101 passed/1 skipped；逐请求重算的最短共同decode窗口13.933秒。最终三轮全部来自303594，不合并不完整父作业。

| 总batch | 点数 | 均衡步时(ms) | 最大失衡布局步时(ms) | 相对均衡增幅 |
|---|---:|---:|---:|---:|
| 32 | 7 | 15.826 | 20.104 | +27.0% |
| 64 | 8 | 18.290 | 28.812 | +57.5% |
| 128 | 9 | 24.905 | 47.568 | +91.0% |

“最大失衡布局”不是“最大步时”：BS32在归一化失衡1.5处为21.657 ms（+36.8%），随后回落；BS64也从29.021回落到28.812 ms。所有点和三轮范围均保留，不强制单调或挑选有利轮次。极差只能描述布局的一部分，不能仅凭该图断言每个跳变对应某个kernel机制。

图7 `Paper/figures/imbalance_topologies.{pdf,png,svg}`独立展示DP4/TP1与DP2/TP2，BS32/64/128各一条曲线，横轴为`(max−min)/(BS/DP)`，纵轴为绝对步时，阴影为作业内min–max。各自均衡点的slowdown保存在分析JSON。DP4来自另一节点d1n41a26g03，且两个拓扑的布局维度、调度器适配不同，不据绝对步时差宣称TP优劣。

本次记录的软件为vLLM `0.26.0+cu129`、kvcached `0.1.6`、torch `2.11.0+cu129`、CUDA 12.9；源文件SHA保存在原始`runtime-environment.json`。DP4原包未保存同规格的软件指纹，不能假定两个作业软件栈完全相同。

复算与重绘：

```sh
/tmp/xplace-venv/bin/python kvcached/experiments/xplace/tools/analyze_imbalance.py \
  evidence/xplace-fig2-ep4-tp2-20261108-r3/job-303594/out/calibration/imbalance \
  --output Paper/draft/imbalance-ep4-tp2-analysis.json
/tmp/xplace-paper-venv/bin/python kvcached/experiments/xplace/figures/imbalance_topologies.py
```

## 拓扑与边界

当前部署的vLLM为`0.26.0+cu129`。已读远端`vllm/config/parallel.py`与`vllm/distributed/parallel_state.py`：在PP=PCP=1时，EP组包含DP×TP个worker。因此四张物理卡对应：

| 配置 | attention DP | attention TP | EP | 物理GPU数 |
|---|---|---|---|---|
| 现有加密校准 | 4 | 1 | 4 | 4 |
| 本次新增校准 | 2 | 2 | 4 | 4 |

两个模型实例共用这四张卡，不为每个实例再分配四张。请求在一个两卡TP组内执行，迁移方向将是两个DP组之间，而不是在四个独立attention rank之间。不能配置DP4×TP2后仍称为EP4。

这次验证不均衡度对decode步时的影响，不运行迁移、回迁或trace。现有迁移RPC只读取首个worker返回值，TP2仍缺各分片的事务校验；运行入口显式拒绝TP2的这些阶段，避免把微基准误写成完整系统验证。

## 预注册

- 单节点4张A800、两个Qwen3-30B-A3B实例、BF16、DP2/TP2/EP4；仅零基模型索引1有请求，另一个实例仅驻留权重。
- 固定4096输入token与1024输出token、贪心解码、忽略EOS；只用于微基准，不给在线控制器未来输出长度。
- 与DP4扫描使用相同seed和合成prompt生成规则；每个BS的输入集合在各布局与重复间相同。
- 总batch为32、64、128，分别7、8、9个点；24点×3轮=72次测量，共5664个请求，不含暖机。
- 布局为`[BS/2+d, BS/2-d]`，不均衡度为`max−min=2d`，作用单位是DP组而非每个TP worker。每轮随机排列case，重复间轮换两个组的位置。
- BS32的d为0、1、2、4、8、12、16；BS64为0、1、2、4、8、16、24、32；BS128为0、1、2、4、8、16、32、48、64。
- 最大布局为`[128,0]`。预检按每请求块取整、实际每页字节和空闲块数计算，不沿用TP1的每token字节常量；TP组以最紧张的物理卡决定预算。
- 沿用稀疏CUDA graph档位1/8/16/32/64/128/256；请求数跨档位是被测机制的一部分。记录的`capture_bucket`是按配置预测的档位，不能当作逐步kernel trace。
- BS128均衡布局暖机后，逐case排空。每条请求保存cadence、起止时间、初始DP组和输出长度。
- decode统计窗口为所有请求都已生成首token且尚未完成的交集，两端各去掉1秒，至少3秒；每请求需有有效cadence，且无抢占。预填充尾部和完成后的尾部不进入步时。
- 每点报告3轮均值及min–max范围，不称为95%置信区间；仍需跨作业、跨节点重复才可讨论运行稳定性。
- 4卡、硬限1小时，r2扫描最多3100秒（原版2800秒），每case最多120秒。不足时间/空间、窗口无效或抢占均视为不完整，不筛掉后宣布成功。后续补跑依用户授权执行，但必须先诊断、修复和验证。

配置为`configs/imbalance-ep4-tp2-20261107.json`。该文件的DP/TP字段必须与CLI一致，在启动引擎前检查。

## 启动与验收

启动时必须有8份worker记录：2实例×2DP×2TP。检查每实例恰好四个不同GPU、两实例对应worker共址、EP rank完整且每层128个专家恰好分片一次、同TP组KV块数相同、不同实例/DP组IPC独立。TP组内部共享DP-local命名空间，不要求每个TP worker独立命名空间。

`budget.json`保留所有TP worker显存快照；`phases.json`记录实际拓扑。`runtime-environment.json`记录vLLM、kvcached、torch/CUDA版本及并行配置/补丁源码SHA；归档中另有GPU清单与互联拓扑。

完整验收顺序：Slurm正常退出 → startup与job均passed → 实际DP2/TP2/EP4及四卡共址 → 全72次有效 → `analyze_imbalance.py`逐请求重算通过。分析工具只验证测量文件，不能单独替代startup拓扑检查。

## 提交记录

一次性tag为`fig2-ep4-tp2-20261107`，2026-11-07 18:39:10+08冻结25个文件。配置、运行时与测试随包上传；不改远端共享kvcached库。

- tarball SHA-256：`38ad01ec57f1f8c785a447a558eb7eee26cec733005b95f166209b9d7c4aafee`。
- manifest SHA-256：`570b044fab3225f3f2e354b47e1322e361834d4dd12158e4fe3fd8c7ecfd30ce`。
- 本地97项测试通过，包含DP4兼容、DP2/TP2布局、worker身份、专家分片、显存最紧张卡与分析重算；原论文验证仍为13页、6图、4表、2算法。
- 2026-11-07 18:44:41+08上传及远端SHA核验通过；18:45:11+08提交Slurm作业`300260`，时限`01:00:00`，GPU数4。
- 依赖为`afterany:300123`，在已有DP4/TP1作业结束后才可调度，避免两个四卡作业并发；300123已于18:31:18启动，并通过startup检查进入imbalance阶段。
- 导出参数：`XPLACE_PHASES=imbalance`、`XPLACE_NUM_RANKS=2`、`XPLACE_TENSOR_PARALLEL_SIZE=2`、`XPLACE_IMBALANCE_CONFIG=imbalance-ep4-tp2-20261107.json`、`XPLACE_SEED=20261107`。
- 提交回执保存在`evidence/xplace-staging/fig2-ep4-tp2-20261107/`。本条记录的是提交，不是完成或性能结论。
- 18:45查询为`PENDING (Dependency)`，`scontrol`确认`gres/gpu=4`、1小时时限和未满足的`afterany:300123`依赖；同次查询300123为RUNNING。

不恢复历史observer或自动归档链；保留所有已有PAUSE。2026-11-08起按用户授权在当前会话前台监控并诊断补跑。未取得完整有效TP2数据前不把部分点当作完成结果更新论文。

一次性人工检查与完成后的取回命令：

```sh
/tmp/xplace-venv/bin/python kvcached/experiments/xplace/tools/xplace_ctl.py status fig2-ep4-tp2-20261107 300260
/tmp/xplace-venv/bin/python kvcached/experiments/xplace/tools/xplace_ctl.py fetch fig2-ep4-tp2-20261107 300260
/tmp/xplace-venv/bin/python kvcached/experiments/xplace/tools/analyze_imbalance.py \
  evidence/xplace-fig2-ep4-tp2-20261107/job-300260/out/calibration/imbalance \
  --output Paper/draft/imbalance-ep4-tp2-analysis.json
```

先确认终态再执行fetch；运行时不抓取半成品归档。分析输出位于原始证据目录之外。

## 比较与后续

先分别画DP4/TP1和DP2/TP2曲线，以各自均衡点归一化报告decode slowdown。也可辅助使用归一化失衡度`(max−min)/(BS/DP)`；必须同时给出实际布局和DP数。

相同BS、相同横轴极差不代表两种拓扑有相同的attention工作量：DP2使每组基础batch翻倍，TP2又引入attention张量通信并改变每卡KV占用。因此跨拓扑绝对步时不是“仅TP变化”的因果消融，也不能混合成同一条BS曲线。现有DP4作业与新作业不保证在同一节点，比较时需报告节点/软件版本与运行差异。

本次通过只支持失衡机制的拓扑泛化。要声称TP2下XPlace仍改善TTFT且控制Conv损失，必须再做逐shard迁移验证、代价再校准和同trace强对照。
