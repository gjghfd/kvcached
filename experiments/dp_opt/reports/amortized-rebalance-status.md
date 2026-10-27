# 成本摊销 ILP：实现与实测进度

## 最终七策略结果（2026-10-25）

250581在d1n41a22g02两A800运行15:43:52–20:33:56，COMPLETED/0:0；20:41:10主
coordinator完成归档与核验。父agent重算117rawfiles、2份joblog SHA、冻结源码和
完整verifier，结果与coordinator完全一致。七策略各75请求，共525次实际完成、
4187008输出tokens；请求覆盖/配对token/串行reference、四rank默认sparse graph、
owner/physical mapped/受控预算/复制峰值、ILP replay、credit/债务历史与源码SHA
关联均通过。59轮主SSH全成功，最大实际gap300.013s；辅助监督60轮无告警，
主coordinator/caffeinate/监督均自然退出。此轮实验完成，整体50%/多seed目标未完成。

### 设置与评估边界

两独立Qwen3-30B-A3B-Instruct-2507共址两A800，每模型DP2/EP2/attention-TP1；
BF16/128K、24GiB/model/rank虚拟pool、10GiB/GPU受控KV预算、allocator .98；默认
sparse `[1,8,16,32,64,128,256]`、FULL_DECODE_ONLY/compile0、32step边界均保持。
Conv为Azure输入sample+明确合成8192输出，Code为SWE-smith真实约100K输入+64输出。
Conv .0307909199195414RPS、Code .001RPS、seed20261024、600s自然warmup+1800s主窗；
trace SHA `2bd50c7e9555e54aa59260db3f9f959230cbf2052985ec50add88488f642dbfb`。
所有策略同trace、模型/图、预算，均75drain；主窗口54=2Code/52Conv。完成SLO120s、
fair阈值30s、timeout900s均未修改。以下TTFT为arrived→first-token、单位秒，P90/
P95/P99为inclusive type7。**只有2条Code，高分位是描述，不是稳健tail确认**。

### Code TTFT（2请求）

| 策略 | mean | P90 | P95 | P99 |
|---|---:|---:|---:|---:|
| no_move | 801.548 | 815.818 | 817.602 | 819.029 |
| no_move_fair | 95.794 | 109.376 | 111.073 | 112.432 |
| own_model | 799.800 | 814.141 | 815.933 | 817.367 |
| kv_only | 55.590 | 56.532 | 56.649 | 56.744 |
| ilp | 55.812 | 56.488 | 56.572 | 56.640 |
| kv_only_rebalance（v1） | 56.768 | 57.277 | 57.341 | 57.392 |
| kv_only_amortized（v2） | 57.634 | 58.171 | 58.238 | 58.291 |

### Conv TTFT（52请求）

| 策略 | mean | P90 | P95 | P99 |
|---|---:|---:|---:|---:|
| no_move | .410 | .652 | .789 | 1.040 |
| no_move_fair | 1.294 | .990 | 5.467 | 18.225 |
| own_model | .393 | .503 | .695 | .971 |
| kv_only | .841 | .504 | .904 | 12.208 |
| ilp | .853 | .551 | 1.073 | 12.252 |
| kv_only_rebalance（v1） | .898 | .558 | 1.255 | 13.235 |
| kv_only_amortized（v2） | .930 | .520 | 1.299 | 13.954 |

### 合并 TTFT（54请求，不是两tenant均值再平均）

| 策略 | mean | P90 | P95 | P99 |
|---|---:|---:|---:|---:|
| no_move | 30.081 | .775 | 1.040 | 800.478 |
| no_move_fair | 4.794 | 1.191 | 17.365 | 94.775 |
| own_model | 30.001 | .544 | .967 | 798.725 |
| kv_only | 2.869 | .720 | 9.382 | 55.520 |
| ilp | 2.888 | .859 | 9.371 | 55.761 |
| kv_only_rebalance（v1） | 2.968 | 1.100 | 9.929 | 56.730 |
| kv_only_amortized（v2） | 3.030 | .914 | 11.063 | 57.594 |

### Latency、TPOT与完成SLO

主窗口mean；Code/Conv/合并顺序。完整P90/P95/P99 latency与TPOT在父audit JSON，
不以TTFT代替端到端SLO。120s目标对8192输出Conv多数不可达，不能调整SLO追正。

| 策略 | latency（秒） | TPOT（毫秒） | 120s met（Code/Conv/合并） |
|---|---|---|---|
| no_move | 811.393 / 159.807 / 183.939 | 156.276 / 19.460 / 24.527 | 0 / 1 / 1 |
| no_move_fair | 106.873 / 160.300 / 158.322 | 175.860 / 19.412 / 25.207 | 1 / 2 / 3 |
| own_model | 809.090 / 159.388 / 183.451 | 147.456 / 19.411 / 24.153 | 0 / 1 / 1 |
| kv_only | 66.796 / 161.405 / 157.901 | 177.865 / 19.603 / 25.464 | 2 / 0 / 2 |
| ilp | 66.960 / 161.718 / 158.208 | 176.956 / 19.639 / 25.466 | 2 / 0 / 2 |
| v1 | 67.738 / 162.281 / 158.779 | 174.122 / 19.702 / 25.422 | 2 / 0 / 2 |
| v2 | 68.312 / 164.684 / 161.115 | 169.484 / 19.992 / 25.529 | 2 / 0 / 2 |

### 控制成本（全75请求，秒）

| 策略 | 总控制含preplan | pause-call | preplan | 实际migration elapsed |
|---|---:|---:|---:|---:|
| no_move | 4.424 | 0 | 0 | 0 |
| no_move_fair | 5.314 | 0 | 0 | 0 |
| own_model | 3.780 | 0 | 0 | 0 |
| kv_only | 6.860 | 1.809 | .030 | .062 |
| ilp | 6.952 | 1.816 | .031 | .062 |
| v1 | 60.476 | 38.568 | 10.136 | 1.382 |
| v2 | 15.483 | 6.001 | 1.301 | .310 |

v2释放尝试76→3、恢复transactions19→1、恢复moves22→2。总控制相对v1下降74.40%，
pause-call下降84.44%，但**没有净TTFT/latency增量收益**：相对KV-only，Code/Conv/
合并mean TTFT分别退化3.68%/10.57%/5.62%，Conv latency+2.03%。相对普通baseline
Code mean TTFT下降92.81%，相对fair下降39.84%；这是容量迁移已有的收益，不是v2
恢复机制的独特收益，更不是至少50%的双baseline或多seed确认。v2主SLO2/54低于
fair3/54，不能宣称整体SLO全面改善。

### 算法结论与下一门槛

- 通用activation、past-service credit与事件门控确实抑制反复回迁，不依赖模型名字。
- 但first-debt恢复4:0→2:2将Conv搬入原可容纳下一次长请求的rank：KV-only第二条
  Code直接准入queue .069s，而v2要再搬2请求queue2.458s。reservation→first-token
  各策略仍约54.3–55.2s；不能把几秒控制优化当长prefill加速。
- 第二个恢复候选7:4的1秒ILP达到time limit且没有verified incumbent，回退不动后gate
  gain=0并结清；不是证明sparse surrogate所有布局gain=0。预算64.214足够，非ε太小。
  21:35本地3秒重放找到6:5候选，net gain58.270request-s；这仅CPU重放、不是GPU收益。
- 下一步需要小batch/长context/双模型联合成本校准，滚动剩余work与未来准入风险，
  同时保护Conv。后续公式在协议末节，尚未实现；不强制回迁、改rate/SLO追正或复活
  旧grid。当前保留KV-only为已验证参考，不将v2当已盈利的新默认。
- 数据面仍安全global pause，未实现真实GPU copy/inference并行；本轮只验证迁移
  与推理的成本门控。1%为驻留service暴露预算政策，不是严格kernel-overhead保证。

### 完整证据

`evidence/amortized-rebalance-20261025/campaign/`包含117原始文件SHA、原包、两joblog、
Slurm accounting与59轮poll。父独立完整结果
`parent-final-verification-250581.json`和`parent-final-audit-and-window.json`；全75及
主54配对在campaign的`all-request-comparisons.json`、`evaluation-window-comparisons.json`。
下文保留实现/启动/中间查询历史，不覆盖原249646负结果。无commit/push/Aegaeon。

## 唯一新GPU实验

250581于2026-10-25 15:43:52获配d1n41a22g02两A800，15:44:03首次实查RUNNING，
启动阶段尚无新推理/收益结果。25文件保护同步通过，17本地冻结runtime SHA复核；
继续同七策略allocation，每300s主watcher至终态SHA归档、源码/credit/owner/graph/
token/真实性能独立核验。主PID66551+caffeinate66552、remote query-only3800573。
原始恢复目录 `evidence/amortized-rebalance-20261025/campaign/`，runtime冻结，不
commit/push/Aegaeon。

旧v1的74个有before-plan恢复事件本地离线检查，通用债务门控仅留下2个候选，旧
inventory上分别求出2/3移动。其余事件无已完成的容量受益请求债务；这是旧轨迹的
局部算法诊断，不是新counterfactual完整轨迹或GPU收益。离线原始结果保存在
`evidence/amortized-rebalance-20261025/offline-v1-event-diagnostic.json`。

## 实现和提交前证据

### 17:20 已完成v1/v2的独立中间核验

七策略job仍运行，第三个KV-only对照未结束。仅v1/v2两个完整且不再写入的子树
单独SHA归档；下载前后manifest相同，42文件逐一SHA通过。各75请求/598144输出
tokens，配对token/串行reference、四rank graph、ILP replay、credit/债务历史与源码
SHA关联通过。没有伪造全job exit-status或全局passed来调用最终verifier。

| 主54窗口TTFT（秒） | v1 mean/P95 | v2 mean/P95 |
|---|---:|---:|
| Code（2请求） | 56.768 / 57.341 | 57.634 / 58.238 |
| Conv（52请求） | .898 / 1.255 | .930 / 1.299 |
| 合并 | 2.968 / 9.929 | 3.030 / 11.063 |

Conv主窗口mean latency162.281→164.684s、mean TPOT19.702→19.992ms；完成SLO
均2/54，Conv0/52。中间结论：**控制开销减少不等于净性能改善**，上述均为同job
两个完成策略，不拼旧job的KV-only数据或把两条Code高分位当显著收益。

总控制60.476→15.483s（-74.40%），pause-call38.568→6.001s（-84.44%），释放尝试
76→3、回迁transactions19→1、回迁moves22→2。第二个债务恢复在7:4布局回退不动。
2026-10-25 21:35原raw plan复查更正：`proven_infeasible=false`，HiGHS time limit，
没有verified incumbent；此前将gate gain0解释为所有capture8布局无收益不正确。
原table中7:4成本118.068，6:5成本39.356request-s，本地3秒ILP重放找到1move、
总目标59.798、净gain58.270；非新GPU结果，也未改变250581性能/成本实测。
预算64.214足够，并非credit限制拒绝。当前回退会结清epoch，后续事件重试与小batch
联合校准仍必要；不能盲增ε或将time-limit当真实allocator无解。

42文件和中间核验在`evidence/amortized-rebalance-20261025/immutable-trials/`；全部
七策略结束后仍需最终完整SHA/ownership/physical/性能核验及当前KV-only/两基线对照。

### 当前KV-only配对与下一次准入的诊断

第三KV-only完整子树也已下载前后SHA一致、独立检查token/graph/owner/预算/指标，
仍不是全job最终核验。主54窗口KV-only→v2：Code mean TTFT55.590→57.634s，
Conv .841→.930s，合并2.869→3.030s；Conv latency161.405→164.684s。KV-only只有
1次迁移准入，v1/v2均2次。第二条Code KV-only走direct_fastpath，queue .069s；
v2再移动2个Conv请求，queue2.458s。3种策略真实reservation→first-token均约
54.3–55.2s，几秒复制/准入优化不能自行解决长prefill的推理互扰。

这不证明每个差值因回迁而产生，单次配对有动态运行噪声；但第二条请求的准入
路径差异是日志事实，不能只评估回迁后count均衡而忽略下一次长请求headroom。
证据`immutable-kvonly/independent-kvonly-v1-v2.json`与`paired-Code-admission-paths.json`。
下一阶段应先校准小batch/长context/双模型性能，并加入未来准入风险/滚动剩余work，
而非增大epsilon、改RPS/SLO或强制把KV搬回去。具体尚未实现的后续公式见协议末节。

2026-10-25：根据249646负增量结果实现 `kv_only_amortized`，固定停顿一次activation、
迁移/驻留服务credit、真实容量迁移债务的事件触发、fresh continuation排除sunk cost。
不按Code/Conv特判，旧策略保留对照。协议在`../docs/amortized-rebalance-protocol.md`。

164专项CPU测试通过，覆盖控制路径fake迁移→独立replay、账本/债务/源码篡改、预算
下部分平衡、租户互换、blocked FIFO、默认32边界与旧策略兼容。首次核验缩进错误
导致29collection errors，修复后80通过；扩展fixture遗漏reservation_pending导致20
失败，修fixture后164通过；失败日志全部保留，不是GPU通过。一次大patch因不存在
的文档context失败，未应用，随后拆分重做；自己误建空证据目录已仅rmdir清理。

研究广回归691通过/5skip/1既有wait-forecast失败明确deselect；18项新旧coordinator
专项通过。原249646六策略75请求归档由当前verifier再次通过，旧格式兼容；bash
语法、py_compile与git diff whitespace检查通过。不是整个包含torch的全部suite。

15:34:32 SSH实查用户queue空；集群摘要公共A800有16卡可用、4个两卡组合，不是
获配保证。此为提交前历史查询；新job状态以上述15:44实查为准。广回归和保护同步后
启动300秒主watcher/remote query-only，终态自动SHA归档独立核验。
恢复在`evidence/amortized-rebalance-20261025/`；不commit/push/Aegaeon。

credit非kernel compute预算，尚未真实GPU copy/inference并行，N128单模型surrogate
未联合校准；总体50%/多seed收益目标未完成。结束按Code/Conv/合并和成本报告真结果，
不能以减少动作或代码测试代替收益。
