# 在线headroom保护：Tail TTFT进度

## 2026-10-26完整结果：251533

唯一两A800作业251533于2026-10-26 01:55:09 COMPLETED/0:0，GPU执行4:10:26。
六策略各75请求、共450请求/3588864输出tokens完整结束。108rawfiles+2job logs
SHA、20冻结runtime/测试SHA、全请求token配对/四rank sparse graph/ownership/
physical预算/复制峰值/ILP重放/past-only headroom history核验通过。父线程完整
verifier输出与coordinator完全一致，另有subagent独立SHA/请求完整性侧查。

归档前6次SSH255/超时断包均保留，第7次恢复成功，10:18完成coordinator归档核验，
10:21父独立核验、10:25subagent侧查完成。没有重新跑GPU以代替失败归档。此轮
设置保持BF16/128K/真实100K Code/10GiB预算/默认sparse/32step和原75请求trace不变。
Conv仍为Azure输入长度sample、**明确合成8192输出**，不是原Azure输出长度分布。

### 主窗口：平均与尾部TTFT

600–2400秒主窗口54请求（Code2/Conv52），单位秒，type7线性插值高分位：

| 策略 | 请求组 | Mean | P90 | P95 | P99 |
|---|---|---:|---:|---:|---:|
| no_move | Code | 804.888 | 818.904 | 820.656 | 822.058 |
| no_move | Conv | .425 | .641 | .867 | 1.148 |
| no_move | 合并 | 30.220 | .695 | 1.135 | 803.837 |
| no_move_fair | Code | 96.335 | 109.762 | 111.440 | 112.783 |
| no_move_fair | Conv | 1.342 | 1.048 | 5.665 | 18.639 |
| no_move_fair | 合并 | 4.860 | 1.330 | 17.904 | 95.328 |
| own_model | Code | 801.947 | 816.415 | 818.223 | 819.670 |
| own_model | Conv | .435 | .588 | 1.086 | 1.270 |
| own_model | 合并 | 30.120 | .911 | 1.266 | 800.862 |
| kv_only | Code | 55.904 | 56.877 | 56.998 | 57.095 |
| kv_only | Conv | .857 | .542 | .976 | 12.462 |
| kv_only | 合并 | 2.895 | .762 | 9.626 | 55.831 |
| ilp | Code | 55.275 | 55.722 | 55.778 | 55.822 |
| ilp | Conv | .826 | .564 | 1.111 | 11.786 |
| ilp | 合并 | 2.843 | .951 | 8.806 | 55.241 |
| kv_only_headroom | Code | 55.690 | 56.475 | 56.573 | 56.652 |
| kv_only_headroom | Conv | .852 | .502 | 1.047 | 12.299 |
| kv_only_headroom | 合并 | 2.883 | .896 | 9.431 | 55.631 |

**结论：没有证实回迁保护相对KV-only的大幅新增tail收益。** 合并mean/P99仅改善
0.424%/0.358%，Conv P95退化7.36%、合并P90退化17.62%；2Code不足稳健P99或
显著性判断。对普通no_move，Code及合并TTFT大幅改善，但Conv mean约翻倍、P99
1.148→12.299秒，不能说所有模型都获益。对fair，合并P99改善约41.6%，仍不到50%。
新方法不应替代KV-only作为已验证更优的默认策略；也不归因ILP与KV-only单次差异
为性能项独有收益，因为两者都只发生一次容量准入迁移。

### 延迟、SLO与真实控制成本

| 策略 | 主窗Mean latency | 主窗Mean TPOT | 主窗120s完成SLO | 全75完成SLO | Moves | 全控制秒数 |
|---|---:|---:|---:|---:|---:|---:|
| no_move | 185.163 | .024798 | 1/54 | 3/75 | 0 | 4.480 |
| no_move_fair | 160.798 | .025653 | 3/54 | 5/75 | 0 | 4.805 |
| own_model | 183.749 | .024003 | 1/54 | 3/75 | 0 | 3.927 |
| kv_only | 158.376 | .025565 | 2/54 | 4/75 | 1 | 7.650 |
| ilp | 158.223 | .025481 | 2/54 | 4/75 | 1 | 5.321 |
| kv_only_headroom | 158.791 | .025586 | 2/54 | 3/75 | 3 | 10.748 |

控制秒数含direct RPC/ack、preplan与staged操作，不全是GPU停顿。headroom恢复
4次事件尝试，仅1次实际7:3→5:5/2moves；累计恢复control3.091秒、全操作阻塞上界
9.000request-s，KV-only为.565request-s。主窗口Conv mean latency161.883→162.325秒，
完成SLO Conv仍0/52；不能仅报告Code端的改善而漏掉老Conv及warmup SLO退化。

### 为什么tail没有明显改善

1. 当前KV-only Code mean TTFT55.904秒，queue仅1.422秒、reserved→first54.482秒。
   若保持后段路径不变，即使消掉全部queue，mean改善上界也只有2.544%。后段是
   prefill/混合执行/API与调度的观测时间，不是GPU attention kernel时间或严格物理下限。
   no_move/fair/新方法的同段时间也约54.5–54.7秒，继续减少几秒迁移无法制造50%收益。
2. 已用**本轮251533**最近1-41 after-admission snapshot与全部客户端reserved/completed
   时间独立重建最坏Conv 1-42到达时的9个旧驻留请求，并核对全部full-reservation blocks。
   Code9538.5MiB+最小Conv1002MiB>单卡10240；全部Conv含新请求10822.5MiB>另卡。
   虽logical总20361<20480MiB，512种乐观整请求二分布局全部无解。最小等额logical
   budget下界10540.5MiB，尚未包括null/pinned/碎片/复制峰值；不是整卡OOM或native
   渐进KV结论。最近snapshot mapped为9792/9696MiB，也不能当作global physical够的证书。
3. 恢复7:3→5:5发生在Code完成后的release阶段，不能消除此前已经积累的约22秒
   Conv容量等待。N128单模型surrogate仍未证明当前小batch/长context/混合阶段收益。

因此下一阶段必须先区分真正可重放置的局部压力与不可分装箱无解，并测联合执行
临界路径；不再盲调epsilon/solver时间、复活旧grid或只提高预算追正。具体预声明
门槛与通用selective-transaction设计见`docs/tail-next-stage-design.md`，**尚未实现或实跑**。

### 监控、安全与证据

主33次SSH query无SSH失败，但最大**墙钟**gap3169.786秒，不能凭monotonic约300秒
声称整夜无中断。旧监督在00:26因自身1279.973秒gap/poll stale告警退出，记录保留。
remote query-only原始记录52轮，21:44:48→02:00:07，墙钟max301秒、无>360秒gap；
此证据说明远端fallback持续，不能掩盖本地主监控缺口。10:17远端queue空、remote
124342退出；coordinator/caffeinate/监督均自然退出，subagent完成侧查后关闭。

原始文件只在`evidence/headroom-tail-20261025/campaign/live-trace-251533/`；完整结果
派生文件为同根`parent-independent-full-verification.json`、
`parent-all-six-paired-window-comparisons.json`、`parent-main-window-six-policy-metrics.json`、
`parent-full-phase-attribution.json`、`current-251533-Conv-tail-indivisible-placement-witness.json`；
侧查在`supervision/terminal-independent-audit.json`，remote cadence审计单独保留。
本轮正确性与结果归档完成，**显著新增tail收益/多seed研究总目标未完成**，未commit/push/Aegaeon。

### 结束后的监控修复（未重跑GPU）

`stress_campaign.poll_job`以后把墙钟gap记入原`last_poll_actual_gap_s`及新的
`last_poll_wall_clock_gap_s`，另记录`last_poll_monotonic_gap_s`。300秒调度仍用monotonic，
不补发忙轮询、不重复提交；保留原poll-log格式以兼容现有归档审计。两类时钟不同
不能再被单个约300秒值掩盖；这只改后续监控，不重写本轮33条原日志或宣称缺口消失。

新增正常300秒及墙钟1800秒/monotonic300秒的回归，45专项通过；广回归712通过/
5skip/1既有失败明确deselected，既有pytest asyncio_mode warning未改。GPU版本20
冻结文件仍一致，未改变任何本轮模型/调度/请求以重算收益。证据
`wall-clock-monitor-regression.txt`与`postrun-wall-clock-broad-regression.txt`。

## 运行历史：251533

2026-10-25 23:12恢复：唯一251533仍RUNNING，23:09:48主SSH成功/间隔300.011s；
第二项KV-only offered68/completed63/75，第三项headroom未完成。主16981与任务绑定
caffeinate16982、独立只读监督17552均PPID1实查，20冻结runtime SHA一致。辅助agent
连接断开不是GPU失败；关闭agent但保留独立监督daemon，不重复提交。继续同allocation
至终态全SHA/独立核验；暂无本轮净收益结论，报告本地更新不覆盖活动remote源码。

2026-10-25 21:44:43唯一job251533在d1n41a13g02获配两A800-SXM4-80GB；21:45:43
实查RUNNING/启动CPU回归中，两卡加载前free81153MiB。30文件保护同步旧SHA备份/
原子替换/新SHA一致，stage `sync/headroom-20261025-214229`；运行源码冻结。
主coordinator16981/PPID1、任务绑定caffeinate16982实查存活；remote query-only
124342/PPID1实查，均按300秒监控，辅助agent仅本地只读监督，不重复SSH。

21:45时尚未完成新方法推理或收益验证。原trace六策略同预算/图/请求完整drain之后
自动全部raw文件SHA归档、独立future-history/ILP/physical/owner/tokens/graph核验；
Code/Conv/合并mean及P90/P95/P99都计算。2Code仍不是稳健P99确认。
新结果路径`evidence/headroom-tail-20261025/campaign/live-trace-251533/`，
未完成前该结果目录可能尚不存在。禁止覆盖当前remote运行源码，不commit/push。

22:34:48主query已从own_model切换kv_only（第二项，第三项才是headroom）；无error。
这说明第一项执行返回，不代替全job原始SHA/完整请求独立复核。持续同allocation，
不额外提交、不覆盖20个冻结executable/test文件，也不通过局部case提前宣布净收益。

## 23:58前三策略初步结果（整job未终态）

23:54:48主query已进入第四项no_move。前三项不可变子树41rawfiles远端前后SHA
和本地下载SHA一致，75×3请求/1794432输出tokens，所有配对token IDs一致；原trace
及raw summary重算通过。此时尚未做全六策略owner/graph/ILP独立核验，**不是终态
实验结果**。原始子树与初步JSON保存在`evidence/headroom-tail-20261025/early-three-policies/`。

600–2400秒主窗口54请求（Code2/Conv52），单位秒：

| 请求组/策略 | Mean TTFT | P90 | P95 | P99 |
|---|---:|---:|---:|---:|
| Code / KV-only | 55.904 | 56.877 | 56.998 | 57.095 |
| Code / headroom | 55.690 | 56.475 | 56.573 | 56.652 |
| Conv / KV-only | .857 | .542 | .976 | 12.462 |
| Conv / headroom | .852 | .502 | 1.047 | 12.299 |
| 合并 / KV-only | 2.895 | .762 | 9.626 | 55.831 |
| 合并 / headroom | 2.883 | .896 | 9.431 | 55.631 |

合并mean/P99仅改善0.424%/0.358%；Conv P95退化7.36%，合并P90退化17.62%。
最坏Conv `1-42`仍22.744秒，其中queue22.225秒；不能把消除v2余量回归当50%增量
tail收益。headroom只有第三次恢复实际7:3→5:5/2moves，前两次4:0→2:2被guard拒绝；
全策略总moves3 vsKV-only1，pause call累计3.792 vs2.272秒。恢复active-blocking
8.548request-s，合并mean latency退化0.262%、Conv mean latency退化0.273%，120秒
完成SLO同为2/54、Conv同为0/52。以上均描述性一次配对，不宣称统计显著性。

同作业own_model（非用户最终两个普通基线）Code mean801.947→55.690秒，合并
mean30.120→2.883秒，但Conv mean.435→.852秒、P99 1.270→12.299秒，有明确
跨模型代价。不能只挑Code的93%降幅；no_move/fair须等剩余三策略与整job核验。

当前仍只执行251533，保持预算/请求/RPS/SLO/graph不变；完整归档后再判断下一轮，
不能为追50%盲重提相同设置。prefill临界路径与不可分装箱下界是下一步主要机制。

## 2026-10-25本轮设计与验证

23:14:48主query确认已进入第三项`kv_only_headroom`，elapsed42.129秒、offered1/
completed0/75。KV-only执行返回不等于全job核验；两A800继续同allocation，无重复提交。

21:22:35 SSH实查用户queue为空，公共A800摘要29张可用、11种两卡可运行组合；
这是查询时的资源，不承诺提交后马上获配。未提交本轮GPU任务时不声称收益。

250581主窗口最坏Code TTFT 56.767秒（queue2.115+prefill/first54.652），第二条
54.413秒（queue.069+54.345）。最坏Conv `1-42` TTFT22.503（queue22.184+
first.319），其准入1397.845与第二Code完成1397.742相邻。v2该Conv TTFT24.487，
第二Code queue2.458；恢复重新消耗了长KV准入余量。这不是所有tail都由复制造成。

新增`kv_only_headroom`：只用每tenant最近64真实到达长度的P95，枚举未来DP home，
在v2 staged ILP中保持此前安全的下一条请求KV headroom；不改实际copy peak预算。
没收益不pause，债务留到下一次release；允许部分恢复/移换空corridor，不模型特判。
实现/预声明指标与边界见`docs/headroom-tail-protocol.md`。

初轮104专项CPU通过，首次shell收尾`status`是zsh只读变量导致外层exit1，pytest
原日志明确104通过；保留证据，不把shell失败隐去。尚待广回归/旧档兼容/保护同步
和原trace六策略GPU配对验证；未claim任何新GPU通过或50%改善。

广回归705通过/5跳过/1既有失败明确deselected（wait_forecast shared pages未来birth
案例不属于本轮）。CPU过去输入重放证明第一恢复的2moves被headroom guard拒绝，
参考空闲rank证书9.375GiB被保留；这是CPU witness，不是GPU收益。

21:35更正上一轮7:4诊断：原ILP是time-limit无incumbent，而非所有layout预测gain0；
3秒CPU重放找到6:5/1move，净gain58.270request-s。原实测指标不变，已在上一报告
更正。新方法保留债务、用事件重试，同时不通过无限增加求解时间损害TTFT。

新增正向真实copy fake/源码篡改核验的第一次回归203通过/1失败，失败只因expected
error regex不匹配实际更早的safe-rank约束拒绝；已修正断言并保留两份失败日志。
上一轮250581用当前verifier重算75×7请求，输出与原核验JSON完全一致；并非覆盖
旧冻结runtime或重新定义旧基线。最终新测试/广回归仍须全部通过后才同步提交。

证据根`evidence/headroom-tail-20261025/`，前轮独立phase归因在
`previous-250581-phase-attribution.json`；该归因不是反事实/真实kernel profile。

总研究目标仍未完成；100K的约55秒prefill下限和Conv全局容量等待需要下一步真实
联合profiling，不能只靠恢复门控宣称解决，也不盲调ε/RPS/SLO追正结果。

补充对旧250581最坏Conv装箱见证的512种精确logical二分枚举：最小等额单卡预算
下界10540.5MiB，较当前10240增加300.5MiB；最优把1002MiB的Conv放在Code同卡，
另一卡9820.5MiB。10560/10656MiB只有1种logical赋值，11328有10种。此分析忽略
null/pinned/碎片和copy-before-release，只是必要性下界，绝不声称这些预算已真实fit
或提交新实验。下一轮若改变预算必须先真实容量核验、所有基线同预算、预声明，
且不能以重新挑RPS/SLO造收益。证据`posthoc-logical-budget-lower-bound.json`。

## 21:41最终本地验证

710研究CPU测试通过/5skip/1既有失败明确deselect，18个新增headroom测试单独通过；
bash语法、8源码py_compile、git diff --check通过。初次两处文档patch context不匹配
均未产生GPU执行，按实际文件逐项补齐；旧250581的117raw SHA和verifier输出保持。
准备新root唯一A800六策略，source frozen后不覆盖；当前尚无新GPU结果。

## 22:04 Tail临界路径补充（旧250581证据，不是新GPU收益）

同一真实101670-token Code串行submit→first为26.094秒，混合reserved→first54.345秒，
观察比2.083；不能单凭两段时间断言具体kernel/通信因果。KV-only两Code mean TTFT
55.590秒中queue仅1.092秒：即使彻底消除queue，上界也仅改善1.964%。要大幅改善
Code tail，不能仅继续减少迁移/ILP秒数，必须研究prefill/混合执行的临界路径。

最坏Conv `1-42` 并不是一个可以由任意whole-request重放置解决的局部碎片案例。
从其到达前最近一次`1-41`准入后snapshot重建全部9个实际活跃请求，逐一用客户端
reserved/completed时间确认到达时仍驻留，并按真实block几何计算以下乐观必要条件：

- 每卡受控budget10240MiB，Code完整reservation9538.5MiB；最小Conv（含新请求）1002MiB。
- Code+任一Conv至少10540.5MiB>10240，所以Code所在GPU不能放任意完整Conv。
- 全部旧Conv+新Conv需要10822.5MiB>10240，另一GPU也无法容纳全部Conv。
- 总logical reservation20361MiB<20480MiB；因此aggregate够仍不代表DP2整请求装箱有解。

该必要性证明甚至忽略null/pinned页、碎片与copy peak：乐观布局已无解，更真实
allocator/ILP也无法把这条Conv立即准入。不是整卡OOM，也不否认第一Code迁移准入的
机制收益。`global_free_bytes`的mapped统计只能作预筛，不能单独充当allocator无解证明。
本轮全reservation/每请求attention-home不可拆分是证明范围，不外推为渐进native KV
或跨GPU分片KV机制无解。

若要改善这一Conv tail，需要提前完成/释放Code或其他Conv；改变等待准入次序会把
等待转给别的请求，必须一起看两tenant和合并tail/latency，不偷偷调长度/SLO/预算。
新的headroom策略先防止回迁破坏下一Code准入余量，不能宣称解决这个全局装箱瓶颈。

证据：`serial-mixed-prefill-diagnostic.json`及
`Conv-tail-indivisible-placement-witness.json`（均在本轮evidence根）。251533仍原配置
六策略运行，继续300秒监控至终态独立验证；不得改运行源码或把旧证据当本轮收益。
