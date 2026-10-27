# Conv 降载到 1.1 倍 KV 压力

## 最新结论：第二档已完整核验，第三档运行中

2026-10-26 22:48:13最新SSH实查255294仍RUNNING，已进入kv_only，elapsed
1059.832s、24offered/21completed/49。本策略尚未到首Code的1231.721s到达时刻，
不能据此判断第三档Code收益；完整六策略与第三档实际pressure仍待终态核验。

21:55本地核查：第三档主21:52:48查询SSH0、wall299.995s，首own_model
elapsed248.631s/8offered/2completed/49。还未到Code到达与no_move背景实测阶段；
主+remote+独立侧daemon继续每300秒，终态自动SHA/完整trace/pressure与侧audit。

2026-10-26 21:48恢复核查：255294已于21:39:35获配d1n41a18g02两A800，
21:48:04主SSH实查RUNNING/初始化与sparse图捕获，尚无完整trace/pressure结果。
主最近wall300.113s/SSH0；远端query-only2561178/PPID1于21:42:12实查。
侧daemon54575/PPID1已部署，21:37:02→21:42:02 wall299.990s/mono300.002s
实证跨轮存活，7项合成自检通过，真实终态audit尚未执行。subagent已关闭，独立
daemon继续；不把合成自检当真实GPU验证。所有25冻结源仍SHA一致。

第三档255294于2026-10-26 21:32:43提交，21:33:03实查PENDING/Priority，无GPU
执行结果。37文件旧SHA/备份/原子替换/新SHA保护同步通过；local coordinator53796、
任务绑定`caffeinate -i -s -w 53796` PID53797，remote query-only2561178已登记。
33压力专项/230邻接CPU tests通过（既有asyncio_mode warning保留）；新增相同rate
测试首轮暴露tuple排序比较dict的TypeError，改为只按rate排序后通过，首次日志保留。
原20GPU算法文件不变，25新launch源冻结。继续300秒监控和终态SHA/独立核验；
不因排队重提、不改活动runtime、不commit/push/Aegaeon。恢复`bracket-campaign/`
及`bracket-launch.json`；压力1.1仍预测，不能把排队或新rate当收益结果。

### 降载前后，同一种KV-only方法的区别

原251533 Conv.0307909RPS/实测压力1.215368，与已完成第二档254337
Conv.0216355RPS/压力1.129739，都是原算法、同预算/长度/图配置。主窗TTFT秒：

| 请求组 | 原负载 Mean / P99 | 第二档 Mean / P99 |
| --- | ---: | ---: |
| Code | 55.904 / 57.095 | 55.297 / 56.085 |
| Conv | .857 / 12.462 | .487 / 2.384 |
| 合并 | 2.896 / 55.831 | 3.298 / 55.490 |

降载主要改善Conv tail：P99下降约80.9%；Code mean只下降约1.1%，仍受约54秒
100K prefill下限约束。合并mean没有变好，但两轮主窗Conv数量52→37，Code同为2，
Code比例2/54→2/39；不能把不同请求构成的合并均值变化直接当算法退化/改进证据。
两轮Conv主窗mark集合和时刻也不同，不是同trace算法配对因果对照。
第二档此前的Code mean下降77.55%指**同档KV-only相对no_move**，并不是降低
Conv率本身带来的77.55%提升。第三档尚未完成，不能提前将上述结果移用到第三档。

### Baseline自身随降载的变化

同样比较原251533→第二档254337，完整记录两个用户指定baseline，而不只看KV-only：

| 方法 | Code mean TTFT(s) | Conv P99 TTFT(s) | 合并 mean / P99 TTFT(s) |
| --- | ---: | ---: | ---: |
| no_move | 804.888 → 246.366 | 1.148 → .508 | 30.220 / 803.837 → 12.963 / 250.688 |
| no_move_fair | 96.335 → 146.365 | 18.638 → 39.146 | 4.861 / 95.328 → 10.574 / 150.743 |
| kv_only | 55.904 → 55.297 | 12.462 → 2.384 | 2.896 / 55.831 → 3.298 / 55.490 |

普通baseline的Code等待明显减少，因此KV-only的Code mean相对收益约93.1%→77.6%。
fair baseline反而变慢，使相对收益约42.0%→62.2%；后者不是KV-only绝对时间大幅
改善。有限trace中降载不保证fair调度单调变好；两次Code、不同Conv时序/主窗集合
不能支撑稳健泛化或单独归因，不能通过只选更慢fair baseline制造算法优化结论。

同第二档固定39请求比较，KV-only相对no_move合并P99改善77.86%，但Conv
P99 .508→2.384、mean latency144.447→158.660退化，合并SLO120仍6/39。
相对fair则Code mean改善62.22%、Conv P99改善93.91%、合并P99改善63.19%，
合并SLO2/39→6/39。需要分别声明对哪一个baseline的收益，保留双方与各tenant代价。

2026-10-26 21:25：254337于20:15:10 COMPLETED/0:0，两A800/d1n41a15g02，
4:18:33。107raw与2日志SHA、25冻结launch文件、父完整verifier与coordinator一致、
独立侧audit passed；六策略各54请求，共324请求/2556672输出tokens。父独立重算
背景Conv physical为1968.529MiB/卡，`(1968.529+9600)/10240=1.129739`，
aggregate ratio=.665677。**仍未达到预声明1.1±.02，不把1.1297说成1.1。**

主窗600–2400s，每策略39请求（Code2/Conv37）；TTFT秒，线性插值百分位：

| 策略 | 请求组 | Mean | P90 | P95 | P99 |
| --- | --- | ---: | ---: | ---: | ---: |
| no_move | Code | 246.366 | 260.771 | 262.572 | 264.013 |
| no_move | Conv | .347 | .465 | .474 | .508 |
| no_move | 合并 | 12.963 | .474 | 23.310 | 250.688 |
| no_move_fair | Code | 146.365 | 160.958 | 162.782 | 164.241 |
| no_move_fair | Conv | 3.234 | .521 | 31.634 | 39.146 |
| no_move_fair | 合并 | 10.574 | 31.634 | 49.627 | 150.743 |
| own_model | Code | 248.132 | 262.496 | 264.291 | 265.727 |
| own_model | Conv | .582 | .506 | .618 | 5.555 |
| own_model | 合并 | 13.277 | .618 | 30.460 | 252.441 |
| kv_only | Code | 55.297 | 55.940 | 56.021 | 56.085 |
| kv_only | Conv | .487 | .482 | 1.174 | 2.384 |
| kv_only | 合并 | 3.298 | 1.174 | 7.714 | 55.490 |
| ilp | Code | 55.487 | 56.107 | 56.185 | 56.247 |
| ilp | Conv | .428 | .495 | .594 | 1.772 |
| ilp | 合并 | 3.252 | .594 | 7.586 | 55.673 |
| kv_only_headroom | Code | 55.529 | 56.278 | 56.371 | 56.446 |
| kv_only_headroom | Conv | .436 | .527 | .622 | 1.953 |
| kv_only_headroom | 合并 | 3.261 | .622 | 7.798 | 55.753 |

| 策略 | Code mean latency(s) / SLO | Conv mean latency(s) / SLO | 合并 mean latency(s) / SLO |
| --- | ---: | ---: | ---: |
| no_move | 256.008 / 0/2 | 144.447 / 6/37 | 150.168 / 6/39 |
| no_move_fair | 154.690 / 0/2 | 160.445 / 2/37 | 160.150 / 2/39 |
| own_model | 257.599 / 0/2 | 144.152 / 6/37 | 149.969 / 6/39 |
| kv_only | 66.434 / 2/2 | 158.660 / 4/37 | 153.930 / 6/39 |
| ilp | 66.268 / 2/2 | 159.172 / 4/37 | 154.408 / 6/39 |
| kv_only_headroom | 66.478 / 2/2 | 159.503 / 4/37 | 154.732 / 6/39 |

KV-only vs no_move：Code mean TTFT下降77.55%、P99下降78.76%；合并mean下降
74.56%、P99下降77.86%，但P90由.474升至1.174。Conv mean TTFT .347→.487，
P99 .508→2.384，mean latency144.447→158.660（+9.84%），TPOT17.592→19.311ms。
合并mean latency+2.51%，SLO仍6/39，不能称全面提升。相对fair baseline，Code mean
下降62.22%，Conv P99下降93.91%，但这不消除相对独立baseline的Conv代价。
两个Code只提供描述性tail；不是多seed/P99确认，也不证明性能项或headroom独特收益。

真实容量见证：首Code到达，GPU mapped `[2400,1440]MiB`，Code增量9504MiB，
两卡分别不够但总free16640MiB足够。将一个Conv从rank1移至rank0（1.228GiB，
bytewise验证通过），准入后mapped `[3648,9696]MiB`均<=10240。KV-only该次
planning .033s、pause call1.379s、copy .123s、含preplan控制1.621s；整trace
控制5.289s。ILP/headroom同样只有一次跨tenant迁移，无新增placement收益。
Code平均排队191.372→.939s；reserve→首token约54.994→54.358s，当前主要剩余
下限是真实100K prefill，不是还能通过重放置省掉的排队。

进一步核对固定主窗request-s：Code累计latency省379.148s，但37个Conv累计
latency增加525.875s，因此全39请求累计latency增加146.728s。TTFT累计仍省
376.965s，二者不矛盾：已有Conv的decode代价不会反映在它们的首token时间里。
headroom有9次release-rebalance检查、累计控制.205s，全部no_profitable_rebalance，
没有额外搬回。迁移的Conv在1325.844s结束，首Code在约1299s结束、次Code于
1331.572s到达；这是短剩余寿命/短回迁窗口的实测描述，不强制搬回制造收益。
后续通用目标需同时计入老请求剩余decode/SLO和新请求等待，不能只优化新请求
TTFT；若研究prefill/并发调度，需要另行控制实验，不能把相关性当GPU kernel归因。
详见`mixed-latency-tradeoff-254337.json`及原始decision/request记录。

本地38poll、2次SSH失败、2个wall gap>360s/max7202.866s；远端53poll、
15:56:16–20:16:24/max301s/无>360s gap。18:52和20:52系统DNS报错，21:19
主连接恢复后自动取回终态，未重提交。原固定IP尝试因无匹配RSA host key被严格
校验拒绝，未降低校验、未改全局DNS/VPN。父audit首次把首poll `None`转float失败，
完整verifier/指标已成功；单独重试监控提取并保留首次日志，不称全部首次通过。
20:15退出时TCPStore/socket警告保留；所有请求、结果与ownership完整验证通过。

证据：`refine-campaign/`、`parent-full-verification-254337.json`、
`parent-main-window-metrics-254337.json`、`postrun-validation-254337.json`、
`postrun-monitor-audit-254337.json`、`independent-capacity-witnesses-254337.json`、
`refine-supervision/audit.json`，均在`evidence/pressure-ratio-20261026/`。

第三档只按两侧实测物理压力做受限secant插值，不按收益选择：
`lambda=lambda_lo+(1664-K_lo)*(lambda_hi-lambda_lo)/(K_hi-K_lo)`。
两端为253478的1368.667MiB/.01756045RPS和254337的1968.529MiB/.02163550RPS，
候选Conv `.01956673845194241RPS`（原率63.55%）；Code仍.001，49请求/策略，
主窗37（Code2/Conv35），trace SHA
`816b71c72bc840f15f429ded2db2dcf69d13127aa50c0bbb2dff1637a316da87`。
目标仍只是预测；第三档已获配运行，尚无完整实测。首阶比例反馈受有限人口、离散mark/页、
条件采样及service影响，不假定精确线性。插值必须在两端之间、原率上界以下，
且两profile模型/控制/窗口/Code流完全相同；不修改20个原GPU算法文件。

## 历史：首档准备与执行

2026-10-26：用户授权只降低Conv到达率，目标为背景单卡Conv physical KV加一条
Code的完整physical需求约为单卡10GiB受控预算的1.1倍。协议在
`../docs/pressure-ratio-protocol.md`，原始证据在`evidence/pressure-ratio-20261026/`。

初始候选由251533 no_move主窗口实测反推，扣除每模型/rank固定96MiB空闲页；
候选RPS是首阶近似，不是稳态证书或本轮GPU通过。只将Conv Poisson到达时间
等比放慢，Code时刻及100K输入、Conv8192输出、六策略、预算/SLO/sparse/32step均不变。

已冻结候选Conv RPS `.01756044902350957`（原`.0307909199195414`的57.03%），
目标Conv physical `1664MiB=1.625GiB/卡`，Code保持`.001RPS`。新trace每策略44请求
（Code2/Conv42），600–2400秒主窗口32请求（Code2/Conv30），SHA256
`1a1195ea3a5f6f433355647cebeb8f74ba2e2a536c53f599c000f3cb3c8ba0dd`。

独立核验Code完整marks/时刻逐项不变，Conv长度与mark前缀不变、只有到达时间
等比放慢；所有序列仍在128K以内，原20GPU冻结文件完全不变。16新增CPUtests
及220项邻接回归通过，compile/diff与JSON核验通过，尚未提交本轮GPU作业。正式结束后报告真实物理ratio、
Code/Conv/合并tail及avg TTFT和老请求成本，不把降载本身当算法新增收益。

## 历史启动状态（首档）

37文件旧SHA/备份/原子新SHA保护同步通过，stage
`sync/headroom-20261026-111946`。唯一job253478于2026-10-26 11:20:41在
`d1n41a18g02`获配两A800-SXM4-80GB，11:21:37实查RUNNING/compute启动阶段，
加载前两卡free81153MiB。尚无本轮真实背景ratio或完整收益结果。

本地coordinator98995与任务绑定caffeinate98996均PPID1实查存活；remote只读
watcher920304/PPID1实查，每300秒查询。主监控同时记录wall与monotonic gap，
终态自动SHA归档、完整verifier、主窗/全请求配对tail与实际pressure计算；
失败先归档诊断，不自动重复提交。独立只读subagent负责侧监控，不重复SSH。

首次启动前输入目录拼写错误被文件存在性guard拦截，未启动进程/同步/申请；
随后核对原input与frozen-input prompt-bank SHA相同且符合trace后启动。此异常
保留于launch.json，不冒充GPU或测试失败。运行中不覆盖远端源码，不commit/push。

2026-10-26 13:57恢复：13:56:26主SSH实查253478仍RUNNING，已进入第四项no_move，
elapsed1132.714s/24offered/21completed/44；前三项执行阶段结束，但尚未完整归档
独立核验，不提前称收益。主query存在wall gap1164.734s及4869.833s，对应monotonic
仍约300s；侧监督13:19也记录3551.266s wall gap与主poll stale告警，不称本地
连续监控。remote原始只读记录32轮、11:20:43–13:55:47/max302s/无>360s gap，
fallback持续证据在`remote-query-interim-1240.txt`与`monitor-interim-gap-audit.json`。
文件名沿用首次开始诊断时刻，实际下载完成时刻为13:57；根因未用power log证明。
runtime未改，不重提；继续本job至终态SHA/实际ratio/全部tenant tail复核。

## 降载实测预览（不是全job终态）

14:25:16：no_move阶段已结束，独立只读导出5文件（trace/config/population/decisions/
requests），远端导出前后SHA相同、本地SHA通过；44请求完整drain且长度匹配。
Code未驻留主窗1674.352秒left-hold背景mean Conv physical1368.667MiB/卡，
`(1368.667+9600)/10240=1.071159`，aggregate=.607096。候选降得略多，不称已到1.1。
此结果在`no-move-preview/pressure-preview.json`；其余策略及完整ownership/graph/
copy正确性仍待全job终态独立验证。

以同样扣固定96MiB的首阶公式，再从本轮实测背景反推下一Conv候选约.021635RPS；
精确值在`next-rate-candidate.json`。这是比本轮略增、仍低于原.03079092的压力校准，
不是根据收益选择负载。必须先完成253478/全部SHA与tail核验，才能修改本地helper
支持低于原始率上界的双向校准并提交下一唯一allocation；不覆盖当前冻结runtime。

## 首轮完整结果：253478

2026-10-26 15:45:08 COMPLETED/0:0，4:24:27。105raw与2job日志SHA通过，25
launch源码/测试从冻结bundle复核，父完整verifier与coordinator完全一致，侧审计
passed。六策略各44请求，共264请求/2065152输出tokens；预览physical ratio与终态
完全一致。原20GPU runtime未改，无新算法实现/commit/push/Aegaeon。

主窗600–2400秒，每策略32请求（Code2/Conv30）；TTFT mean / P99（秒）：

| 策略 | Code | Conv | 合并 |
| --- | ---: | ---: | ---: |
| no_move | 73.143 / 90.916 | .401 / .885 | 4.947 / 80.035 |
| no_move_fair | 75.074 / 92.684 | .413 / .878 | 5.079 / 81.903 |
| own_model | 73.493 / 91.334 | .392 / .804 | 4.960 / 80.411 |
| kv_only | 56.273 / 56.325 | .369 / .605 | 3.863 / 56.293 |
| ilp | 57.202 / 57.517 | .399 / .960 | 3.949 / 57.325 |
| kv_only_headroom | 57.771 / 58.183 | .373 / .522 | 3.960 / 57.931 |

KV-only vs no_move Code mean改善约23.1%，不是50%；降低压力也使baseline等待
大幅下降，不能把跨率baseline变快隐藏。Conv mean latency142.846→147.605秒
退化约3.3%，完成SLO改善5/30→7/30；合并SLO7/32→9/32，但mean latency
139.029→142.440秒退化约2.5%。headroom并无新增Code收益。两个Code的P99仍只
描述性；详细P90/P95/P99、全部tenant及全44请求配对在parent-main-window-metrics-
253478.json及campaign comparison/attribution，不只挑Code端指标。

## 第二档准备

只在首轮完整核验后修改CPU压力helper，允许基于同model/B/window/prompt-bank/
Code流实测profile，在原Conv率上界以下双向校准。18专项与222邻接tests通过；
第二档Conv `.021635500127908054RPS`（原率70.27%，较首档+23.21%），Code仍.001。
新trace54请求（Code2/Conv52）、主窗39（Code2/Conv37），SHA256
`6e298a9acdb3fa9915234f4716ae59c3c88550192ae7e727382e0814168550bc`。
下一动作唯一A800六策略保护同步/冻结；第二档1.1仍预测，必须重测后再下结论。

2026-10-26 15:58：第二档37文件旧SHA/备份/原子新SHA保护同步通过，唯一job254337
15:56:14提交、15:56:37在d1n41a15g02获配两A800，15:57:50 RUNNING，加载前
free81153MiB/卡。local22734/caffeinate22735 PPID1，remote3872305在15:58:40
PPID1实查。先前单次PID查询没有命中，随后确认同脚本/同job存活，未重启或重提，
根因未确认，不把这一查询疑点当推理错误。独立侧监控在refine-supervision部署中。
runtime冻结，继续300秒主查询和remote fallback，终态自动全SHA/完整verifier/
pressure/所有tenant tail；结果前不声称第二档达到1.1。

首轮监控最终原始审计：local25poll、3次wall gap>360s/max4869.833s；remote54poll
11:20:43–15:45:56/max303s/无>360s。原始文件完整归档，不能因remote持续掩盖
本地主监控缺口。报告/矩阵本地更新不覆盖活动remote runtime，无commit/push/Aegaeon。

第二档独立只读侧监督23294/PPID1于16:07:38实查，16:00:26.945250→16:05:26.951934
两轮wall300.006684s、mono300.005321s，主22734/22735身份匹配/SSH0，无告警。
侧脚本终态自动核验54请求/策略、324总请求/2556672tokens、39请求主窗及phase；
不重跑完整GPU verifier、不计算pressure，仅对父计算结果检查1.1±.02并告知偏离。
证据refine-supervision/；第一档监督12文件SHA不变。16:06:17已进入own_model
elapsed21.930s/0offered/54，继续同allocation，不提前称压力/收益完成。
