# 端到端收益阶段状态

更新于2026-10-21 18:38（Asia/Shanghai）。

## 异构负载下一阶段

本轮完整Azure/LongBench长度profile与CPU ILP已归档，见`mixed-workload-profile.md`。
Code长输入触发，老Conv长驻留搬迁；不需要Code长decode。计划见
`../docs/mixed-workload-trace-design.md`，未执行新GPU trace、未SSH同步或提交作业。
1M只是分析范围，原DP2/EP2/attentionTP1布局不能直接支持；仍维持32K/sparse。
CPU可行不等于净收益，整体收益目标仍未完成。下述job/queue结论是此前查询记录。

## 当前作业

**没有活动研究作业。237899 COMPLETED/0:0，四策略完整trace净收益为负。**
实际15:56:36–16:49:20，两A800；44rawfiles SHA/1544请求/571764输出tokens独立复核
通过，4串行references和四rankFULL graph、ownership、copy峰值及指标复算通过。
ILP queue94.07s vsno_move41.34s，latency113.74vs58.72，20s SLO19vs32/386；
ILP21次规划4次迁移准入，其中2次跨tenant，但same-snapshot KV-only离线ablation
21/21同动作。完整结果/控制拆分/下一步见`trace-benefit-results.md`。
17:00队列空，没有盲目追加GPU任务。首次未压缩证据下载超时，残包保留，压缩重试
44SHA通过。下述RUNNING/未GPU验证为历史记录，不是当前状态，未commit/push。
最终本地研究回归165通过/1跳过；17:16报告/导航/复核工具14文件保护同步通过，
旧SHA/backup/新SHA和gjghfd fork main/Aegaeon保护diff核对完成。

2026-10-21 16:40：237899仍RUNNING，前三策略386请求均已完成，ILP最后进行中。
原始汇总no_move queue41.34s/latency58.72s；own_model64.46/82.84、KV-only63.64/82.56，
不能据此先宣称ILP获益。扩展本地独立verifier16tests通过，逐项复算TPOT/drain/SLO、
拆分direct RPC/blocked pause及迁移成本，四model/rank分别检查FULL graph。
这些本地verifier更新未覆盖正在运行的远端代码；继续300秒监控，终态再归档复核。

2026-10-21 16:01：**237899 RUNNING**，15:56:36获d1n41a16g01两A800。统一队列下
own_model首策略trace168.8秒、offered100/completed85、4迁移，无timeout；继续四策略。
watch实际首次gap312.5秒（SSH/调度延迟）记录，不宣称严格无误差300秒；148CPUtests
通过/1跳过。旧失败终态均已SHA归档。源码不覆盖运行版本，未commit/push。

2026-10-21 15:56：唯一重跑**237899**，15:56:19提交，固定386请求/.60rps不变。
237868 FAILED/1:0，15:41:35–15:51:03，19原始文件SHA完整归档；own_model未完成，
不能比较ILP收益。74请求写出，last population只40completed因monitor被lock堵住。
209决策/170规划/1迁移，control累计265.24秒：每arrival独立cache导致planning storm。
统一dispatcher改两tenant FIFO，只globalfree足够时规划，completion后重新规划；
四文件SHA/backup/protected diff同步，100CPUtests通过/1跳过，重跑不改trace/timeout。
watch_trace_job每300秒查询237899，旧失败与原trace全部保留，未commit/push。

2026-10-21 15:43：**237868 RUNNING**，15:41:23提交、15:41:35获配d1n41a16g01两A800。
首237861 15:38:35–15:38:53在pytest阶段FAILED/1:0，26fail/298pass由测试环境
DP_OPT_LIVE_PLACEMENT泄漏触发mock engine无scheduler，未启动模型。已拆分隔离回归
环境，同时修正新runner worker文件名与KVCACHED_DP_IPC_BASE；两文件旧/新SHA与
backup同步后重试。237868回归已通过，32K模型加载/graph capture中。
首失败out/err与修复sync原始记录保留，不隐瞒失败、不改变trace。每300秒监控新job。

2026-10-21 15:38:13提交唯一作业**237861**，PENDING/Priority，多分区两GPU/100分钟。
25文件旧/新SHA、备份与protected Aegaeon diff同步通过。固定386请求/600秒/四策略
首轮各1次，32K/sparse/default32，direct-fit不pause。watch_trace_job每300秒查询，
原始日志在`evidence/azure-trace-run-20261021/poll-237861.log`；排队不是完成，未commit/push。

用户现授权实跑ILP收益。新增固定Poisson trace/runner，共386请求、600秒、总rate.60、
两tenant192/194，四策略首轮各一次，32K/sparse/default32/2112MiB受控预算不变。
direct-fit共用serialized no-pause路径，只block时进入现有staged ILP；同population
无解不重复pause，completion后重新规划；仍无wait成本候选，不保证净收益。
本地新增trace与directfastpath回归98通过/1跳过；准备安全同步后唯一两GPU作业。
15:24用户queue空、A800资源可用；没有提交、没有commit/push。协议
`../docs/trace-benefit-protocol.md`。下面“只理论/未同步”是此前阶段记录，不是最新授权。

用户明确最终实验应为Azure长度sample/Poisson固定trace，**目前只理论分析、不跑真实
GPU实验**。设计见`../docs/azure-poisson-trace-design.md`；仅本地约30万length行抽样和
Poisson静态数值模型。用户现指定max_model_len32768，两live入口和启动默认统一32K，
默认每model/rank虚拟KV4GiB；sparse captures和step token预算4096不变。32K路径尚未
GPU验证，未SSH同步。复用48片段/304016行的32K长度积分独立归档于
`evidence/azure-trace-32k-theory-20261021/`，旧4K原始证据未覆盖。2112MiB/卡的长
Conversation当前候选总rate.60req/s，更长输出512..1024另列.25；均为未校准假设。
旧.65/.40和静态事件频率只属于4K条件，不外推32K收益。全CSV单请求若超受控单卡
KV容量，须重选物理预算，不能静默过滤。当前global pause+缺wait仍可能抵消准入收益。
当前CPU回归202通过/1跳过，两个launcher的bash语法和Python AST/长度积分/SHA通过。
扩大collection时test_dp_namespace因本机无Torch失败，未安装依赖或修改无关代码，
该项留待allocation验证；已有asyncio_mode配置warning保留。32K证据含测试日志与校验。
旧504长KV请求是机制/确认，不代表最终trace已完成；不提交新Slurm、不commit/push。

无活动研究作业，12:13:02查询用户队列空。默认32step长KV确认**237171 COMPLETED/0:0**，
实际11:24:31–11:39:47，`d1n41a18g03`两A800；184rawfiles/336请求/172032输出tokens独立
复核通过。四策略6配对/seed20261025，长KVfixture/sparse/SLO20不变；A queue19.637→2.809
秒、A TTFT19.959→3.116秒、吞吐配对均值+10.25%vsno_move，有r5-0.39%负轮。
latency sum+17.85%、ILP SLO0/84、vsKV-only吞吐-5.07%，全部保留。详细见主场景报告。
allocation内112+193tests通过。首次SSH下载超时，截断包保留、manifest未变、keepalive
重试后184SHA通过；没有重跑GPU。查询器配置300秒，但11:16→11:54实际有检查缺口，
不能宣称无中断5分钟监控。27文件12:11保护同步通过，12:13远端报告/矩阵SHA确认；
fork main/受保护Aegaeon diff正确，未commit/push。此处12:11/12:13同步为上一阶段记录，
不代表13:33后的Azure理论文档已同步。

长KV主场景作业**237068 COMPLETED/0:0**，实际10:48:29–10:57:16，`d1n41a14g01`两A800。
**100rawfiles SHA/168请求/86016outputtokens**独立验证通过，包括两rank真实容量probe、
rollback ownership、B释放192MiB、copy/admission峰值、全部逐请求串行tokens和指标。
26文件SHA保护同步、每300秒查询到终态均有记录。A queue15.761→1.500秒、A TTFT
16.069→1.770秒、吞吐+19.18%；但mean latency上升、20秒SLO17/42 vs34/42，负结果保留。
详细结果见`benefit-long-kv-results.md`；KV-only/ILP同动作，不宣称独特性能项收益。
最新272CPUtests通过/1跳过（含新增capacity witness正向/破坏测试）。默认32step确认已
完成；同受控预留语义的容量准入原型收益已测得，下一阶段是SLO保护/联合cost校准。
236906已完成、归档并独立复核；前期graph/容量对照见`benefit-results.md`。
用户随后明确主场景是全部长KV、aggregate空间足够但当前单卡不够，普通调度排队，
重放置后准入。已实现`../docs/benefit-long-kv-protocol.md`：A/B各3:3，输入3072/2048，
输出512，新A3584/512、新B1024/512；总logical3984MiB小于4224MiB预算，初始任一卡
不fit新A，B迁移后可全容纳。不能以此前graph收益代替这一主场景。未commit/push。
主场景早期319CPU回归通过/1跳过；offline owner/staged ILP验证A-only无解、KV/ILP一条B迁移
可准入（离线非GPU结果）。执行前明确文件SHA保护同步，统一真实allocator probe成本入账。
长decode317CPU回归通过/1跳过，allocation内104+193tests通过；24文件SHA保护同步，
每300秒监控到终态。协议见`../docs/benefit-long-decode-protocol.md`。
确认阶段CPU回归316通过/1跳过；已知asyncio_mode配置warning。包含shutdown的本地
collection因本机未装Torch失败，该依赖测试留在GPUallocation内，不为此安装或改无关代码。
20文件SHA保护同步完成；allocation内首组103tests通过。首次显式CPU/mem提交被站点
job_submit_lua拒绝且无job，取消显式CPU/mem改用分区GPU默认值后提交唯一236751；
不是两份并行申请。最终归档将保留两次提交记录。

## 已验证：236906长decode收益

实际2026-10-21 **10:00:25–10:12:33**，`d1n41a18g03`两A800-SXM4-80GB，COMPLETED/0:0。
160rawfiles SHA全验证；24trial/**432请求/221184outputtokens**串行token、ownership、
budget/copypeak、interval1、初始decode和独立指标复算通过。不是只提交或功能smoke。

预声明tradeoff_long、seed20261023、四策略各6repeat，18requests/trial，所有输出512。
A2:8/B4:2，A0输入496/64、A1八个64、B0输入80/80/80/512、B1两个80；新A2560/512，
新B64/512晚1秒。初始页A2:5/B3:2，受控aggregate mapped预算672/768MiB；
到达所有bootstrap已输出8–29tokens。不同输入和预算用于保持exclusive-page见证，
不是256→512单因素消融；单模型cost/.475秒fixed迁移项/候选上限均未改。

| 策略 | 吞吐tokens/s均值 | A到达TTFT秒 | 平均latency秒 | TPOT毫秒 | control秒/trial | SLO满足请求/108 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_move | 353.894 | 16.582 | 19.009 | 34.035 | .101 | 102 |
| own_model | 424.907 | .663 | 20.261 | 38.245 | .419 | 25 |
| kv_only | 426.313 | .588 | 20.210 | 38.134 | .409 | 36 |
| ilp | 454.702 | .854 | 18.931 | 35.537 | .622 | 106 |

ILP vsKV-only每repeat吞吐：**+5.75%、+8.93%、+3.36%、+5.56%、+10.07%、+6.39%**；
配对比率均值**+6.68%**，95%描述性t区间**[4.12%,9.24%]**。latency sum降低**6.32%**，
A到达TTFT却**增加45.28%**。vsown_model吞吐+7.04%、latency sum-6.53%；vsno_move
吞吐**+28.49%**、A-TTFT-94.85%、latency sum-0.41%。六repeat全部保留。

每轮ILP移B96MiB，实际494–505ms，A准入后3:8/B3:3；KV/own移A54MiB，实际283–343ms，
A2:9/B4:2。ILP两到达总control617–628ms、solver约8.6ms/trial；复制、pause/resume和
预留等待全部已计入，不以模型假定copy时间代替实际成本。A实际capture：KV-only每轮
966–968ranksteps replay16，ILP零replay16、1018ranksteps replay8。保留所有空rank EP执行。

20秒completion SLO：ILP106/108，KV-only36/108；mean goodput .872/.280req/s。
各repeatgoodput比率平均+496.96%，**不是mean goodput比值**，也不是普遍5倍容量收益；
阈值临近KV-only约20秒latency，导致大幅变化。posthoc敏感性（不替换主20秒SLO）显示
22秒时两者108/108，goodput优势回到约吞吐的6.68%；19秒时56/108与6/108。
ILP r3两请求超20秒、KV-only/own全部不利SLO重复均保留。

此场景支持“在容量可行方案间考虑sparse graph/DP负载能补偿更贵迁移”的原型结论；
不是默认32step共识已有正收益，不保证短decode/晚到达/所有SLO或所有请求TTFT更好。
single-modelN128外推仍未联合校准、候选受限、完整reservation复制与受控mapped预算
均属于限制。原始证据：`evidence/benefit-20261020/live-benefit-236906/`、
`verification-236906.json`、`analysis-236906.json`、`slo-sensitivity-236906.json`。

## 已验证：236751独立确认与晚到达

实际2026-10-21 09:26:36–09:41:02，`d1n41a18g03`两A800，COMPLETED/0:0。
258rawfiles SHA全验证，36trial/**648请求**全部token、ownership、预算、copy峰值、
输出边界、interval1、case独立namespace与指标通过独立本地复核。
early6repeat/seed20261021、late3repeat/seed20261022；四策略、相同18request/预算。

| case / 策略 | 吞吐tokens/s均值 | A到达TTFT秒 | 平均latency秒 | TPOT毫秒 | control秒/trial |
| --- | ---: | ---: | ---: | ---: | ---: |
| early / no_move | 315.274 | 8.018 | 10.441 | 36.415 | .112 |
| early / own_model | 376.766 | .468 | 10.917 | 39.890 | .268 |
| early / kv_only | 376.226 | .464 | 10.955 | 40.014 | .263 |
| early / ilp | 388.585 | .698 | 10.593 | 38.544 | .497 |
| late / no_move | 240.731 | 2.169 | 10.471 | 37.719 | .100 |
| late / own_model | 249.335 | .473 | 10.687 | 39.045 | .274 |
| late / kv_only | 251.216 | .464 | 10.551 | 38.605 | .264 |
| late / ilp | 255.154 | .827 | 10.306 | 37.755 | .503 |

early ILP vsKV-only各repeat吞吐：**-4.00%、+4.46%、+3.46%、+3.55%、+10.69%、+2.09%**，
均值+3.38%，95%描述性t区间[-1.56%,8.31%]。latency sum-3.23%、A-TTFT**+50.69%**。
vsno_move吞吐+23.30%、A-TTFT-91.29%，但latency sum+1.50%；vsown_model吞吐+3.20%。
late vsKV-only：**+5.86%、-1.03%、+0.09%**，均值+1.64%，95%t区间[-7.54%,10.82%]；
latency sum-2.21%、A-TTFT**+78.12%**。vsno_move吞吐+5.99%、latency sum-1.57%。
两case所有请求均满足20秒completion SLO；goodput比率等于完整吞吐比率。

早到达所有bootstrap输出范围8–28，晚到达192–214，没有提前完成/丢请求。两case
ILP都移B72MiB，KV/own都移A30MiB，实际复制A140–177ms、B373–393ms；pause多数
39–44ms，early/no_move有103ms异常保留。late更短remaining horizon仍选B，未自动变
KV-only动作。整reservation复制bytes不随已decode进度增长，不能当作长context缩放实验。
early A replay16在KV-only各repeat为454–456ranksteps，ILP为零；late KV-only仅82–88
ranksteps replay16，ILP为零。记录整个trial包括迁移前步骤；因此late避开大graph的
剩余执行步数确实更少。late图统计中大量replay1含dummy，不能当作有效attention量。

**结论：** 容量释放相对排队已有稳定的端到端收益；性能项相对KV-only均值正但小且
区间含零、存在负repeat，尚不能宣称稳健优势。晚到达收益减小与收益摊销窗口缩短
相符，但两case初始化/时段不同，不作精确因果系数。继续更长输出预声明实验，不删负轮。
证据：`evidence/benefit-20261020/live-benefit-236751/`、`verification-236751.json`、
`verification-236751-early.json`、`verification-236751-late.json`、`analysis-236751-*.json`。

## 已验证：235712固定每step共识

实际2026-10-20 23:15:31–23:21:13，`d1n41a28g03`两A800-SXM4-80GB，
COMPLETED/0:0；88rawfiles SHA验证，12trial/216请求全部token匹配，ownership/预算/
copy峰值/指标复算通过。四worker与runtime-control均确认固定interval1，所有策略相同。

| 策略 | 吞吐tokens/s均值 | A到达TTFT秒 | 平均latency秒 | TPOT毫秒 | control秒/trial | SLO满足请求/54 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_move | 318.983 | 7.962 | 10.297 | 35.997 | .103 | 54 |
| own_model | 370.537 | .461 | 11.139 | 41.023 | .261 | 54 |
| kv_only | 376.962 | .468 | 10.960 | 40.078 | .271 | 54 |
| ilp | 393.489 | .696 | 10.429 | 37.955 | .496 | 54 |

ILP vsKV-only各repeat吞吐+9.21%、+0.72%、+3.52%，配对比率平均**+4.48%**；
request latency sum降低4.76%，SLOgoodput+4.48%。但A到达TTFT**增加48.82%**。
vsown_model吞吐+6.23%、latency sum-6.35%；vsno_move吞吐+23.38%、A-TTFT-91.26%，
但latency sum增加1.31%。所有正负指标保留，不宣称同时优化所有请求。

动作仍然不同：KV-only/own_model移A短请求30MiB；ILP移B长请求72MiB。ILP的A
replay只有1/8，KV-only/own_model出现454–456ranksteps replay16。真实复制：A147–168ms，
B373–383ms；每次pause_call42–45ms，ILP两次准入总control约496ms，solver约7ms/trial。
暂停长尾在此控制作业内未出现，但235647与235712节点不同，不能用跨作业吞吐差作
干净的32→1因果消融。该收益只属于声明的interval1，**不是默认32实现已获收益**。

首次测到性能项不同placement相对KV-only的正收益，但只有3次、小batch、早decode、
单模型cost外推、受限候选与受控mapped-KV预算。下一步独立seed增加重复，并在更晚
decode准入检验收益能否覆盖复制开销。完整证据：`evidence/benefit-20261020/live-benefit-235712/`
与`verification-235712.json`；未修改归档原始数据。

## 已验证：235647 tradeoff负结果

实际2026-10-20 22:49:58–22:56:22，同节点两A800，COMPLETED/0:0。87rawfiles SHA验证、
216请求全部token匹配；真实ownership/copy峰值/指标独立复算。不是geometry功能失败。

| 策略 | 吞吐tokens/s均值 | A到达TTFT秒 | 平均latency秒 | TPOT毫秒 | control秒/trial | SLO满足请求/54 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_move | 238.762 | 10.590 | 13.632 | 47.904 | .863 | 54 |
| own_model | 261.253 | .757 | 15.307 | 56.531 | 1.178 | 54 |
| kv_only | 281.678 | .732 | 13.925 | 51.225 | 1.001 | 54 |
| ilp | 231.803 | 1.564 | 18.659 | 68.887 | 4.435 | 36 |

KV-only/own_model确实移A短请求30MiB，A2:9；ILP移B长请求72MiB，A3:8/B3:3。
实际A捕获histogram：KV-only/own_model每repeat有448–454ranksteps replay16；
ILP全部仅1/8（无16），约508ranksteps replay8。机制预测的档位选择成立，但**性能不成立**。

ILP相对KV-only各repeat吞吐gain为-41.07%、-9.35%、-1.97%，平均-17.47%；
latency sum平均增加35.15%、SLOgoodput平均-37.11%。相对no_move吞吐平均-2.92%，
虽A到达TTFT降低85.22%，不是整体算法正收益。三个repeat全保留，不删除r0。
r0-ilp第二次B到达pause_call8.227秒，r1/r2也比KV-only更慢；solver约4毫秒/准入。
真正B72MiB复制374–390毫秒，高于A30MiB；N128单模型performance外推未能保证联合小batch收益。

原生vLLM的DP pause共识每32steps执行，等待时仍跑dummy模型，可导致长暂停。
下一步固定**所有策略每step共识**，不改变CUDA graph/空rank EP执行，记录额外同步开销，
复跑相同tradeoff协议；默认32原结果保持不变。控制预声明见`../docs/pause-sync-control.md`。
这只隔离控制面等待，不宣称自动修复联合cost；若仍无收益，进一步联合校准。

## 已验证：235320 capacity收益

作业实际2026-10-20 22:24:55–22:31:15运行于`d1n41a22g02`两A800-SXM4-80GB，
COMPLETED/0:0；并非10月22日的旧ETA。74原始文件下载SHA全通过；独立验证器复核
12trial/180请求（全部256outputtokens匹配串行参考）、所有pageowner、预算/copypeak和指标。

| 策略 | 吞吐tokens/s均值 | A到达TTFT秒 | 平均请求latency秒 | TPOT毫秒 | control秒/trial | 总迁移 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_move | 212.119 | 9.207 | 10.052 | 34.879 | .644 | 0 |
| own_model | 216.026 | 9.778 | 10.556 | 36.281 | 1.252 | 0 |
| kv_only | 297.927 | 2.102 | 11.338 | 41.660 | 2.366 | 3 |
| ilp | 311.592 | .806 | 10.259 | 37.500 | .974 | 3 |

ILP相对no_move的**各repeat比率平均**：完整makespan吞吐+48.54%，A到达TTFT-91.24%，
SLO20秒goodput+52.21%，但request latency sum **增加2.17%**。相对own_model：吞吐
+45.43%、A到达TTFT-91.75%、latency sum-2.74%。这不是只报正收益：原decode平均TPOT
相对no_move也更高。migration使新请求早重叠执行，而非独立改善所有原请求decode。

KV-only与ILP每次都移动相同B0独占页owner（30MiB，实测133–173毫秒），释放96MiB后
为A准入。ILP vsKV-only的均值+6.13%不是算法性能项贡献：首个r0-kv_only
pause_call=4.285秒，而其他正常约.2–.7秒；原始数据保留，不能删除首轮当作收益。
求解平均7毫秒/trial（两个准入合计），约5秒异常主要在pause，不是ILP。
后续所有策略统一idle/active pause-resume预热，记录预热且不从已完成数据扣除coldcost。

因此已测到**容量释放相对保留排队的端到端收益**，尚未证明性能项比KV-only更好、
成熟KV、联合校准或全请求最优。下一步同环境执行`../docs/benefit-tradeoff-protocol.md`：
两个容量可行方案的复制bytes/capture8→16权衡，需实际capture证据和四策略对照。
局部修订回归293通过/1跳过，执行前将显式同步；未commit/push。

原始证据：`evidence/benefit-20261020/live-benefit-235320/`、`verification-235320.json`。
终止阶段TCPStore/NCCL警告发生在所有请求完成后，allocation0退出；记录但不改无关shutdown。

## 排队过程记录

- 前一容量功能job235093已完成并独立验证，不能当作性能收益。
- 新job235320于2026-10-20 17:53:32提交，当前PENDING/Priority；18:50起暂估
  2026-10-22 14:54:31开始，替代此前2026-10-20 23:37的估计，均非获配资源。
  18:00至19:49每300秒查询仍pending；
  A800/H100/H200均可，唯一双GPUallocation申请。
- 19:09将同一pending作业时限从45分钟缩短为20分钟，尝试扩大backfill机会；
  没有重提申请、改变测量协议或使用不可见的hp_a800。19:49健康可申请节点全部GPU已分配；
  唯一idle节点不在任何可见partition，不可作为实验资源。
- 已实现两独立模型到达流、保留排队的四实跑基线、3配对repeat、全部15请求token对照、
  原始TTFT/TPOT/SLO和含resume/host-copy的完整control计时。协议为`../docs/benefit-protocol.md`。
- 291本地回归通过/1跳过，14文件旧SHA保护同步完成，顶层持续执行指导也已同步。
  作业内会再次执行真实环境回归；本地不模拟GPU正确性结果。
- 原始证据归外层`evidence/benefit-20261020/`；本地watch-job.py每300秒查询，
  终态后下载完整原始文件/manifest/accounting，独立复算，再依据结果修正/迭代。
- 收益尚未测量；当前代价使用单模型N128外推，候选限制每pool2请求/至多2迁移。
  与KV-only可能同方案，不能预先声称ILP独特收益或联合校准/全请求全局最优。

排队期间进一步加强了本地原始验证器：workerexpert/topology、完整预声明request IDs、
迁移length/destinationownership/copy-peak检查；尚未替换已提交作业使用的源码版本。
