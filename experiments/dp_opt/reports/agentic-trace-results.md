# Azure Conv + SWE-smith 100K：A800准入、尾延迟与成本结果

新一轮“KV-only至少50%”的候选分析见
[stress-trace-design-analysis.md](stress-trace-design-analysis.md)，协议见
[long-resident-conv-stress-protocol.md](../docs/long-resident-conv-stress-protocol.md)。
该分析中的数字仍为本地CPU预测。用户已授权新方案实跑，新增自动17训练case与
8heldout流程；旧247027终态核验已全部完成，749冻结请求配置未改。新长输出GPU
pilot及两预算完整配对结果见前节，尚未完成整个网格及heldout收益确认。

## 新长输出方案：247429终态，248507速率扫描

### 两预算完整结果与监控交接

247429于2026-10-24 19:51:08 COMPLETED/0:0，历时7:12:26；210份raw文件及out/err
已SHA归档，capacity、8K profile、两个case均通过独立verifier。父agent随后重新检查
全部210份文件和日志SHA、重跑完整verifier并与campaign核验逐字段相等；两个预算
使用相同75条请求及测量窗口，五策略各完整drain，共750条完成/5981440输出tokens。
每配置主测量窗口54条（2Code/52Conv），自然warmup21条也完整保留；两Code的尾统计
仍只是描述性数字，不是稳健P95/P99或heldout确认。

| 每GPU受控预算 | Code P95 no_move / fair / KV-only(s) | KV-only下降 vs no_move / fair | 五策略迁移次数 |
|---|---:|---:|---|
| 10240MiB（10GiB） | 828.663 / 122.138 / 67.097 | 91.90% / 45.06% | KV-only1、ILP1，其余0 |
| 11328MiB（11.0625GiB） | 57.620 / 57.107 / 55.430 | 3.80% / 2.94% | 全部0 |

较大预算的普通baseline两Code排队仅0.412/0.369s，fair为0.390/0.387s，KV-only
为0.230/0.339s；本配置没有需要跨模型迁移解决的长期准入阻塞，不能用它声称算法
有效或无效于其他容量区间。当前已观察最佳仍10GiB，但**对fair仅45.06%，未达到
双baseline至少50%门槛**。较大预算是保留的完整负/弱收益配置，不从实验矩阵删除。

较大预算主测量窗口的全体、Conv与老Conv成本如下（6条老Conv依各baseline在Code
arrival时的resident集合定义；三方法Code120s SLO均2/2）：

| 群组 / 对照 | baseline mean→KV-only(s) | baseline P95→KV-only(s) | 120s met baseline→KV-only |
|---|---:|---:|---:|
| 54请求 / no_move | 159.833→160.221 | 222.588→224.657 | 2→2 |
| 54请求 / fair | 167.563→160.221 | 234.416→224.657 | 2→2 |
| 52Conv / no_move | 163.836→164.332 | 223.316→224.799 | 0→0 |
| 52Conv / fair | 171.853→164.332 | 235.826→224.799 | 0→0 |
| 6老Conv / no_move | 212.224→210.276 | 237.248→234.209 | 0→0 |
| 6老Conv / fair | 228.713→210.276 | 251.587→234.209 | 0→0 |

10GiB下老Conv的明显损失见下一节，不能被较大预算无迁移的成本表替代。完整P90/
P95/P99已保存于`independent-terminal-metrics-247429-20261024-2005.json`，不从2条
Code推断稳定高分位收益。完整核验与再次复核在campaign目录
`verification-247429.json`、`independent-terminal-verification-247429-20261024-2002.json`、
`independent-terminal-audit-247429-20261024-2005.json`，原SHA在`raw-247429.sha256`。

只读监督agent实际17:24:54→19:59:53共32轮，间隔300.0005–300.0101s，无告警/
重复SSH；原campaign独占查询、归档与下一stage。248507于19:58:29提交，19:58:38
观测RUNNING；沿用两A800、8K输出/10GiB预算，按预声明倍率0.85/1/1.15扫描Conv
RPS（中心0.0307909199），Code固定0.001RPS。同allocation依次跑三个完整五策略
case，随后仍有12K/16K双预算及heldout；不改活动源码或把自动开跑称最终研究完成。

2026-10-25 00:07:58的只读SSH查询确认248507首0.85倍rate配置case0已exit0，
live分析2条Code的两baseline较小P95下降为-0.5109811%；这是新的弱/负收益配置，
尚待父agent不可变子树下载SHA与独立复核。当前中心rate case1的公平baseline正在
执行，整248507及后续网格/heldout未结束。当前已完整本地独立核验最佳仍247429
case0（双baseline较小改善45.06%），不能因当前低rate完成或中心rate已开跑声称50%。

### Code-primary候选：分模型及合并TTFT

剩余联合优化空间及Conv尖峰根因见
[remaining-optimization-space.md](remaining-optimization-space.md)：尖峰为aggregate不足，
不是漏迁移；当前10GiB不能保留Code+任一完整预留采样Conv，纯复制仅0.088s。

用户要求Code、Conv与合并请求的avg及tail TTFT一起评估。本节针对已完整独立核验
的247429/case0/10GiB候选，不把“预声明Code指标观察最佳”称为双模型联合最优。
TTFT=`first_token_at-arrived_at`，包括排队、准入、控制等待和prefill；不是仅从
提交/准入开始计时，也不是`completed_at-arrived_at`的端到端完成延迟。

统一主窗口arrival在[600,2400)s内：2Code、52Conv、合并54请求。合并平均值可按
请求数加权核对，合并分位数则直接从54条原始TTFT排序、按(n-1)*q线性插值；不能
平均两模型的P95。warmup21条完整保留，full75指标单独保存，不混入本表。单位秒：

| 群组 | 策略 | avg TTFT | P90 | P95 | P99 |
|---|---|---:|---:|---:|---:|
| Code（2） | no_move | 803.913 | 818.138 | 819.916 | 821.339 |
| Code（2） | no_move_fair | 96.061 | 109.613 | 111.307 | 112.662 |
| Code（2） | own_model | 801.567 | 815.905 | 817.697 | 819.131 |
| Code（2） | kv_only | 55.688 | 56.024 | 56.067 | 56.100 |
| Code（2） | ilp | 55.522 | 56.197 | 56.281 | 56.348 |
| Conv（52） | no_move | 0.424 | 0.567 | 0.897 | 1.122 |
| Conv（52） | no_move_fair | 1.314 | 0.913 | 5.363 | 18.518 |
| Conv（52） | own_model | 0.412 | 0.647 | 0.811 | 0.990 |
| Conv（52） | kv_only | 0.860 | 0.507 | 1.110 | 12.289 |
| Conv（52） | ilp | 0.976 | 1.193 | 1.438 | 12.269 |
| 合并（54） | no_move | 30.183 | 0.834 | 1.119 | 802.846 |
| 合并（54） | no_move_fair | 4.823 | 0.974 | 17.699 | 95.044 |
| 合并（54） | own_model | 30.084 | 0.723 | 0.978 | 800.492 |
| 合并（54） | kv_only | 2.891 | 0.999 | 9.199 | 55.662 |
| 合并（54） | ilp | 2.996 | 1.283 | 9.301 | 55.472 |

相对fair，KV-only的Code/Conv/合并平均TTFT分别下降42.03%/34.50%/40.05%；
P95分别下降49.63%/79.30%/48.02%。这与此前Code端到端P95下降45.06%是不同
指标，不能交换口径追50%。相对no_move，Code及合并平均TTFT分别下降93.07%/
90.42%，但Conv平均0.424→0.860s、P95 0.897→1.110s、P99 1.122→12.289s均
退化；合并P95 1.119→9.199s也退化。因此不是所有模型、所有分位都改善的联合收益。

KV-only最大Conv TTFT来自`agentic-20261024-1-42`：23.544s，其中准入排队
23.230s；不能仅凭这一个点将原因归于attention不均衡或纯KV复制。Code只占合并
请求3.70%，本轮普通基线合并P95没有覆盖两条长Code，合并P99才落在Code尾部；
必须同时保留分模型与合并表。2Code的分位数及54请求的P99都是小样本描述，尚无
heldout稳健性结论。原120s SLO继续按端到端完成延迟判定，不在本表改成TTFT SLO。
首token之后的老Conv decode损失不会完整体现在TTFT，下一节的老Conv latency/
TPOT与SLO代价仍必须一起看。

从五策略原始请求独立复算；8份相关输入文件与原终态SHA一致，主比较全部TTFT
字段与既有独立verifier逐项相符。精确数字、full75窗口及各策略最慢Conv在
`evidence/kvonly-stress-campaign-20261024/independent-best-ttft-247429-20261025.json`。
新增诊断不修改活动trace、SLO或预声明Code-primary选优目标；后续报告必须明确
联合代价，不能只凭Code指标或合并平均值宣称联合最优。

### 首10GiB配置完整核验：case0

2026-10-24 16:24:51已确认case0五策略各75请求完整drain，live strict verifier通过；
随后下载该不再写入的完整子树，82文件remote前后及本地SHA一致，独立复核375请求/
每policy598144输出tokens、serial及配对tokens、physical mapped/迁移owner/预算、
四rank sparse/32step、strict公平门控通过。不是父allocation终态归档；同allocation
继续11328MiB/GPU预算对照，随后17训练case和8heldout仍待执行。

主测量窗口arrival>=600s共54请求（2Code/52Conv）；自然warmup21请求仍完整保存、
执行并drain，full75的普通/公平/KV-only mean为172.474/158.306/152.856s，120sSLO
分别3/5/4个完成。完整窗口与旧请求代价独立复算如下：

| 群组 / 对照 | baseline mean→KV-only(s) | baseline P95→KV-only(s) | 120s met baseline→KV-only |
|---|---:|---:|---:|
| 54请求 / no_move | 183.932→157.016 | 272.328→227.575 | 1→2 |
| 54请求 / fair | 163.835→157.016 | 247.163→227.575 | 3→2 |
| 52Conv / no_move | 159.734→160.510 | 252.912→230.005 | 1→0 |
| 52Conv / fair | 166.010→160.510 | 248.836→230.005 | 2→0 |
| 6老Conv / no_move | 134.585→232.113 | 147.861→271.265 | 1→0 |
| 6老Conv / fair | 215.338→232.113 | 288.112→271.265 | 2→0 |

老Conv群组由各baseline在Code arrival时的resident集合定义，不能换成KV-only的
更有利集合。Code P95普通下降91.90%、fair下降45.06%与此前中途2Code计算一致，
因此**本配置未满足两个baseline都至少下降50%**。改善Code以明显老Conv代价换取，
尤其对普通baseline；不能说全SLO、所有请求或成熟算法全部改善。

独立证据：`completed-case0-247429.sha256`、`independent-case0-verification-247429.json`、
`independent-case0-analysis-247429.json`，都在`evidence/kvonly-stress-campaign-20261024/`。

### 首case中途两Code配对结果（不等于完整收益结论）

2026-10-24 14:19:30已读到普通baseline的全部2Code完成记录；与KV-only之前2Code
逐条完整64输出token IDs一致。固定原75请求trace、同10GiB/GPU预算和sparse/32step：

| Code | no_move queue(s) | KV-only queue(s) | no_move latency(s) | KV-only latency(s) |
|---|---:|---:|---:|---:|
| seed20261024-Code0 | 767.233 | 1.338 | 830.395 | 65.127 |
| seed20261024-Code1 | 730.876 | 0.391 | 795.760 | 67.201 |

这2条Code的描述性P95为828.663→67.097s（下降91.90%）；120s SLO为0/2→2/2。
不能当稳定P95/P99、整流量吞吐提升或对公平baseline也通过；五策略完整case、Conv/
老Conv代价和heldout尚未齐全。并未更改900s request timeout，两请求都在cap内完成。

15:32已读取公平baseline的全部2Code，queue58.384/24.079s、latency123.790/
90.742s，P95122.138s、120s SLO1/2。三方法完整64输出tokens逐条相同；KV-only
相对fair仅下降45.06%，**首配置尚未达到双baseline同时>=50%**。不得只选普通
baseline的91.90%声称通过，后续仍按预声明扫描倍率与12K/16K输出、双预算。
这仍是中途2Code描述性结果，fair全75请求、最后ILP和老Conv代价还需完成核验。

同时读到普通baseline不可变行的容量见证：model0零active、model1 counts1:2，
两GPU mapped1.5/2.15625GiB、globalfree17548967936 bytes（约16.34GiB）足以容纳
incoming9971957760 bytes（约9.29GiB），但两rank均local_blocked/all_decode=true。
这证明受控预算下aggregate够而当前placement不能准入，不是整个80GB显存OOM。
该population行与中途请求在`evidence/kvonly-stress-campaign-20261024/interim-*`保存；
终态完整SHA与strict verifier仍会复查，当前不是完整trace归档或最终最优配置。

2026-10-24 12:38:23提交、12:38:42获配d1n41a16g01两A800；92文件SHA保护同步。
保持BF16/128K/24GiB虚拟pool/.98/default sparse/32step。已下载完成且不再写入的
pilot子树，24文件前后remote SHA一致；这不是终态完整trace归档。

独立capacity核验最长101675+64真实Code、四pool分配rollback/四rank graph及
预算通过，当前可选受控KV上限11424MiB/GPU。独立8K joint profile核验完整greedy
tokens、两条Conv真实字节迁移、source清空/target唯一owner、复制峰值与Code
before两rank blocked/after rank0 fit通过。

| Pilot | 实测service(s) | 范围 |
|---|---:|---|
| 4Conv balanced | 136.878 mean | 同代表3949输入、8192输出 |
| 4Conv consolidated+Code | 203.324 mean Conv | 无迁移、0:4，真实Code争用 |
| 4Conv migrated+Code | 200.119 mean Conv | 2:2→0:4，包含真实迁移 |
| Code serial | 26.423 | 无Conv背景 |
| Code consolidated-background | 63.932 | 同时有Conv，不是相同compute环境的反事实 |
| Code migrated-background | 63.505 | after真实迁移，不是正式trace收益 |

完整574063 Conv joint marks（权重823814）与实测balanced service给出rate proxy
`.0307909199195414RPS`，目标4800MiB总背景reservation。每请求mean full-reserved
1138.899MiB；这是有限人口代理，不是开放环稳态证书。move proxy fixed1.774s/
11.641GiB/s只用于本轮控制参数，ILP性能仍历史单模型surrogate，wait-aware关闭。

在任何policy结果前冻结两个预算10240/11328MiB，Code.001RPS、seed20261024、
600s自然warmup+1800s测量，每case75请求（2Code/73Conv），五策略完整drain。
13:08:33首case KV-only offered18/completed12，Code尚未到达，不能称50%已达标。
原始完成pilot快照与独立核验在`evidence/kvonly-stress-campaign-20261024/`。
其后顺序17case扫描及固定最佳实际RPS的8heldout/129Code计划，包含所有负case、
老Conv/整体120sSLO代价；不把synthetic长输出称为原Azure输出。

## 同配置配对复跑：247027，2026-10-24

247027于10:24:26–12:15:08在同d1n41a15g02两A800完整执行，COMPLETED/0:0。
12:33终态处理完成：127rawfiles和out/err全SHA、capacity、3745请求/五策略完整
tokens/owner/物理预算/default sparse/32step、strict公平与两baseline尾独立核验通过。
归档首尝试成功，原始bundle及所有负结果保留于`evidence/agentic-trace-run-20261021/`。

| 策略 | 全请求mean(s) | P95(s) | P99(s) | Code mean(s) | Code P95(s) |
|---|---:|---:|---:|---:|---:|
| no_move | 21.317 | 65.655 | 85.862 | 76.262 | 113.683 |
| no_move_fair | 21.966 | 72.855 | 103.329 | 77.007 | 112.410 |
| KV-only | 21.558 | 64.798 | 85.176 | 73.321 | 106.301 |

Code仍仅4条；P95下降6.49%/5.43%（普通/公平），不是50%或稳定大样本尾结论。
全请求mean对普通baseline退化1.13%，对fair改善1.86%；不选择性只报fair。
245583/247027相同冻结trace/SLO的配对数据由新campaign保存，旧复跑不改RPS/B/τ。
下一阶段只按预声明新长驻留Conv方案扩大等待窗口，不能把本文旧4Code复跑当独立
Poisson到达的大样本，也不能把synthetic8192+输出声称为原Azure输出。

## 首轮五策略结果：245583，2026-10-24

同配置配对复跑247027于10:24:15提交、10:24:26获配同d1n41a15g02两A800，
12:15:08结束，第二轮完整结果与核验见前节。
81文件保护同步通过；本地/远端300s监控，终态自动严格核验及双baseline尾分析。
不改trace/rate/τ/B/SLO/策略顺序，不选择性保留好的一轮。

245583在d1n41a15g02两A800-SXM4-80GB上于00:24:25–02:15:28完整执行，
COMPLETED/0:0。五策略各749请求、494107输出tokens全部drain；127原始文件及
out/err SHA、四pool真实100K capacity、3745请求tokens/serial、selected blocks/
实际迁移owner/bytewise echo、物理预算和四rank默认sparse/32step独立核验通过。
10:17:37另有408项冻结setting与尾延迟独立复算通过，使用stdlib inclusive quantiles。

### 设置保持不变

两独立Qwen3-30B-A3B-Instruct-2507 tenant共checkpoint，每模型DP2/EP2/attentionTP1；
Conv为runtime model1，Code为model0。BF16/maxlen131072/24GiB每model-rank虚拟pool/
allocator.98/11328MiB每GPU受控预算，默认sparse `1,8,16,32,64,128,256`、
FULL_DECODE_ONLY/compile0/32step。冻结1200s/seed20261021/749请求，745AzureConv
(.61RPS)+4真实约101K SWE-smith Code(.004RPS)，request timeout900s/SLO120s。
公平τ=30s；策略顺序为fair/no_move/own_model/KV-only/ILP，一个同allocation trial。
没有根据新节点较快的service改rate、SLO或删除Code；旧245085不是这一轮的baseline。

### 全请求与固定SLO

| 策略 | mean latency(s) | latency P90(s) | P95(s) | P99(s) | 120s达标/749 | moves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_move | 21.221 | 52.090 | 63.096 | 83.320 | 748 | 0 |
| no_move_fair | 21.866 | 49.634 | 71.721 | 101.943 | 748 | 0 |
| KV-only | 21.416 | 51.168 | 63.428 | 84.088 | 749 | 5 |

KV-only vs公平基线：mean -2.0615%、P95 -11.5637%、P99 -17.5151%，但P90 +3.0895%。
vs独立基线：mean +0.9181%、P95 +0.5252%、P99 +0.9219%，仅P90 -1.7709%。
两种baseline各一个违约→KV-only零违约，但被挽救的请求不同：普通基线是Code0-1，
公平基线是Conv1-130。不能称KV-only全面胜过两种baseline，也不能以1个违约的
100%相对下降声称稳定SLO泛化。完整drain吞吐受offered rate限制，不当作吞吐提升。
公平基线这条Conv为120.026s，仅超120s门槛约26ms，尤其不能夸大这一SLO差异。

历史辅助策略本轮也完整实跑：own_model mean20.770s/SLO749/0moves；ILP
mean21.127s/SLO749/0moves。ILP未产生迁移，不能把这次小均值差异解释成性能项收益。

### Code与老Conv的代价

| 群组 | N | no_move P99(s) | fair P99(s) | KV-only P99(s) | SLO no_move/fair/KV-only |
| --- | ---: | ---: | ---: | ---: | --- |
| 全Code | 4 | 120.190 | 117.602 | 113.248 | 3/4、4/4、4/4 |
| 全Conv | 745 | 82.285 | 101.363 | 83.184 | 745/745、744/745、745/745 |

Code平均queue为19.582/19.785/15.947s，mean latency为76.336/75.418/72.539s。
仅4条Code，P95/P99是最大值附近插值，不支持高分位显著性；完整逐请求证据保留：

| Code后缀 | no_move queue(s) | fair queue(s) | KV-only queue(s) | latency no_move/fair/KV-only(s) |
| --- | ---: | ---: | ---: | --- |
| 0-0 | 8.516 | 9.954 | 1.853 | 65.591/64.841/58.708 |
| 0-1 | 64.878 | 64.116 | 57.980 | 121.879/119.234/114.935 |
| 0-2 | 2.440 | 2.465 | 1.863 | 60.481/59.661/57.864 |
| 0-3 | 2.494 | 2.605 | 2.094 | 57.394/57.936/58.649 |

baseline定义的老Conv采用各自Code planned-arrival与reserved/completed，不按
treatment挑选：普通基线群组28条，P99 67.233→86.250s(+28.2852%)、mean +45.8401%；
公平基线群组29条，P99 66.130→86.158s(+30.2845%)、mean +63.9343%。两个不同
baseline群组不混合比较。所有老Conv均满足120s，不代表延迟未受损；Code最后一条
latency也退化，不能只报告提前准入或总体公平P99改善。

### 公平与容量行为核验

公平门控40次activation/release，其中Code owner1次、Conv39次；24次suppressed
attempt只涉及1个跨tenant blocked pair。实际KV reservation解锁、零迁移/零pause、
门控期间外tenant KV reservation为0，严格独立核验通过。不把40次门控都当成Code
被保护的独立机会，也不以公平队列变慢直接归因于一个门控动作。

KV-only有4次staged attempt/3次准入，5条Conv迁移共2.210449GiB，copy elapsed
0.193s；总control含preplan82.689s、pause-call4.040s、preplan1.582s均计入。
Code0-3规划快照为Conv4:4、物理mapped `[2.25,1.875]`GiB，预算11.0625GiB/rank，
保守incoming9.375GiB；迁移一条Conv后规划final `[2.53125,10.96875]`GiB并真实准入。
这是实际迁移/准入与page-aware规划证据，不取代独立两rank allocation/rollback
必要性probe，也不把保守ILP容量判断或受控预算当整卡OOM证书。

### 监控、归档与下一步

远端query-only27次实查、max gap301s，覆盖终态；本地仅6poll、一次SSH失败、
max gap3513.320s，虽caffeinate/PID仍在也不能称本地不中断。首次下载残包8.6MiB
及六次失败记录保留；10:08后恢复，旧/新远端manifest相同，完整241MiB压缩包/
127files/out/err SHA通过，10:15全部核验完成。10:07:54实查queue空。

唯一证据在顶层 `evidence/agentic-trace-run-20261021/`：`live-trace-245583/`、
`verification-245583.json`、`tail-slo-245583.json`、`tail-slo-fair-245583.json`、
`tail-slo-verification-245583.json`。下一先保持原trace/rate/τ/B/SLO同配置配对复跑，
检查普通baseline的收益稳定性与老Conv代价；不因本轮负指标改rate或threshold追正。
更多Code和old-Conv guardrails需另冻结协议。研究总体收益目标未完成，未commit/push。

## 245583提交记录（历史）

集群记录时间2026-10-24 00:07:50，81文件保护同步后提交唯一gpu_a800两卡作业
245583；00:07:52首查PENDING/Priority，不称GPU通过。新实验在同allocation上
比较 `no_move_fair/no_move/own_model/kv_only/ilp`，seed20261021随机顺序已运行前
冻结，公平阈值τ=30秒。保持原749请求/1200秒、Conv.61/Code.004RPS、11328MiB/
GPU、900秒timeout、120秒SLO及128K/BF16/default sparse/32step设置。

本地24712/caffeinate24713和远端query-only595931启动300秒监控至终态。终态先
全SHA归档、capacity/tokens/owner/预算/graph和严格实际KV公平门控独立核验，再
比较KV-only与两种baseline的P90/P95/P99、固定SLO、每条Code及baseline定义老Conv。
若门控期其他tenant内部pending实际获得KV，则fair运行无效，不能放宽审计。
旧245085四策略结果不拼接成新五策略对照；未commit/push，Aegaeon不运行。

## 最新结论：2026-10-23 23:19，Asia/Shanghai

245085 于 20:58:14–22:30:59 在 d1n41a23g03 两 A800 上完整结束，Slurm
COMPLETED/0:0。四策略各 749 请求全部 drain；114 raw 文件及 out/err SHA、真实
100K capacity、2996 请求 tokens/serial、selected 精确 blocks、实际 move 顺序/
owner/bytes/exact echo、物理预算、四 rank sparse graph/32step 和当前 wait/cost
独立核验通过。22:54:02 SSH 实查队列空。结论仍是 ILP 净负，不是研究收益完成。

| 策略 | 全请求平均 queue(s) | 全请求平均 latency(s) | Code 平均 queue(s) | Code 平均 latency(s) | 120s SLO(/749) | 实际迁移 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| own_model | 4.222 | 39.950 | 93.810 | 150.455 | 739 | 0 |
| no_move | 3.705 | 40.067 | 68.674 | 126.518 | 743 | 0 |
| KV-only | 2.525 | 40.285 | 57.973 | 112.801 | 748 | 24 |
| ILP | 3.861 | 41.220 | 110.979 | 167.644 | 737 | 0 |

ILP vs no_move：全请求 latency +2.877%，latency sum +863.332 request-seconds，
SLO 少 6。KV-only 的 24 次跨 tenant 请求迁移共 11.997GiB；Code latency sum
减少 54.870s，但 Conv latency sum 增加 218.332s，全请求 latency +0.545%。
KV-only 的 queue/SLO 改善与平均 latency 退化必须同时记录，不能称全赢。
固定 trace/rate/B/SLO/模型/graph/control/lookup 都未修改；一次固定策略顺序
trial，不证明多 seed 泛化或差异统计显著性。

### KV-only vs 不迁移基线：P90/P95/P99 与固定 SLO

用户要求补看尾延迟。此处重新读取同一 245085 raw，不改变 trace/rate/请求数/
输出/SLO，不是新增 GPU 运行。P90/P95/P99 是经验分位延迟，不能与 SLO 的秒数
阈值混为一谈；继续使用运行前已声明的端到端 latency<=120s。分位数采用
`position=(n-1)*q` 线性插值（Type 7/inclusive）；这些补充指标是用户要求后的
描述性分析，不追认成运行前声明的主指标。

普通基线取 `no_move`：同预算、同 full-reservation 自定义 scheduler，不迁移
活跃请求，不冒称未经修改的原生 vLLM 动态增长 baseline。全部 749 请求都计入。

| 全请求指标(s) | no_move | KV-only | 相对变化 |
| --- | ---: | ---: | ---: |
| latency P90 | 76.053 | 70.693 | -7.05% |
| latency P95 | 95.964 | 78.774 | -17.91% |
| latency P99 | 119.049 | 92.551 | -22.26% |
| queue P90 | 4.083 | 5.233 | +28.18% |
| queue P95 | 29.977 | 15.086 | -49.68% |
| queue P99 | 51.110 | 26.827 | -47.51% |
| TTFT P90 | 4.429 | 5.640 | +27.36% |
| TTFT P95 | 30.449 | 15.217 | -50.02% |
| TTFT P99 | 53.120 | 27.825 | -47.62% |

固定 120s SLO：743/749 (99.20%) → 748/749 (99.87%)，违约 6→1，下降 83.33%。
5 条同 request-id 被挽救（1 Code+4 Conv），没有新增违约；这一轮中 P99 均低于
120s，故不应将“P99是否达标”二值化后再声称只有 KV-only 达标。全请求平均
latency 仍 +0.545%，不能只选下降的尾指标而遗漏其代价。

补充 `own_model` 对照：latency P95 96.922→78.774s、P99 124.130→92.551s，
120s 违约 10→1。该策略也不做跨模型调整，但允许本模型候选重放置，不取代
完全不迁移的 `no_move` 主对照。

#### 分模型与受影响老请求必须单列

| 群组 | N | latency P95 no_move→KV-only(s) | latency P99 no_move→KV-only(s) | 120s达标 no_move→KV-only |
| --- | ---: | ---: | ---: | ---: |
| 全 Conv | 745 | 95.439→78.341 | 112.962→90.771 | 741→745 |
| 全 Code | 4 | 186.378→190.991 | 193.158→205.118 | 2→3 |
| baseline 中 Code 到达时已驻留的 Conv | 66 | 48.449→59.342 | 49.604→88.792 | 66→66 |

老 Conv 群组以 no_move 原始 reserved/completed 和冻结 Code planned-arrival
定义，两个策略使用相同 66 个 request IDs，不按 treatment 结果挑选。其 P99
退化 79.00%；虽然都满足120s，仍需保护老请求，不以总体 tail 改善掩盖代价。
这只是配对结果，不能把所有变化因果归于 KV copy 或 attention 失衡。

Code 仅占 0.53%，全请求 P99 不能替代 Code P99；4 样本的 P95/P99 主要是最大
值附近插值，不支持稳定高分位或显著性结论，因此完整列出全部长请求：

| Code ID后缀 | planned arrival(s) | queue no_move→KV-only(s) | latency no_move→KV-only(s) |
| --- | ---: | ---: | ---: |
| 0-0 | 171.566 | 79.897→36.143 | 138.355→90.918 |
| 0-1 | 172.485 | 137.870→151.619 | 194.853→208.650 |
| 0-2 | 754.829 | 27.237→25.881 | 85.726→80.686 |
| 0-3 | 927.371 | 29.692→18.249 | 87.138→70.948 |

三条改善、一条退化；最早两条只相隔 0.919s，不能假定每次 Code 等待都是
aggregate 足够但 local 不足，下一轮需独立区分总容量不足、局部放置失衡与
控制/前一条 Code 的代价。所有长请求在 no_move 中均最终完成，最大 queue
137.870s，没有无限等待或终态未完成证据。A 持续到达不等于没有自然释放；
“普通基线长期排队”应验证为明确观测窗口内的等待/违约，不宣称永远排队。

当前支持的研究定位：capacity-first 重放置在此固定 trace 改善总体尾延迟与
120s SLO，但不是全部 tenant/老请求/均值都改善。下一步在不改旧 trace 的前提
下做策略顺序配对重复；若需要稳定 Code P95/P99，另预声明更长观测/更多 Code
样本和每 tenant SLO，保留旧结果，不根据尾收益挑 seed/rate 或删异常长请求。

复现：`analyze_tail_slo.py <raw-root> --output <json>`，默认比较 no_move/KV-only；
加 `--baseline own_model` 可复现另一对照。输出在 evidence 下
`tail-slo-245085.json`、`tail-slo-own-model-245085.json`、
`tail-slo-verification-245085.json`。源 SHA 匹配终态 raw 清单，独立使用 stdlib
`statistics.quantiles(method="inclusive")` 等完成 236 项核验，36 项相关 CPU
tests 通过；原 pytest asyncio_mode 配置 warning 保留。未同步/新提交/发布。

### 场景是否成立：均衡 Conv 阻挡长 Code 确实出现

2026-10-23 新baseline准备：用户要求无公平独立排队 `no_move` 与超时停止其他
tenant准入的 `no_move_fair`。默认候选tau30s，可配置，真实KV reservation后解锁，
不暂停老请求或client、不迁移；协议 `../docs/queue-baselines-protocol.md`。
118相关CPUtests通过，新增verifier复核旧245085四策略2996请求通过；首次测试
误把被阻挡Conv自身超时的第二次合法activation视为异常，已修正并保留失败日志。
新版81文件保护同步清单仅本地检查，尚未同步或新GPU申请，不能给245085补造
公平baseline结果。下一新allocation保持原冻结trace/rate/B/SLO，五策略同跑后
独立比较KV-only/no_move/no_move_fair的分模型尾延迟、SLO和老Conv代价。

本 trace 是持续 Poisson Conv .61RPS、Code .004RPS，1200s 共 745 Conv+4 Code。
Conv 输入 2049–7764、输出 512–1024；Code 输入 101167–101626、输出固定 64。
这里报告中的 Conv 对应用户所说模型 A，Code 对应模型 B；runtime model id
恰好相反（Code=0、Conv=1），不要混淆。

原始 KV-only 首次候选见证：Conv count=[10,11]、Code count=[0,0]，每 GPU
受控预算 11.0625GiB，当前所有 pool 的 physical mapped 为 [6.375,6.375]GiB，
单卡空闲各 4.6875GiB，合计 9.375GiB。当前 snapshot 对新 Code 的保守 page-aligned
增量也是 9.375GiB：合计足够，任一卡不足。这个增量是 planner 的保守页预算，
不冒称排除了现有 free slots 复用的真实最小分配量。重放、真实 allocator 和
owner/预算验证均通过，迁移 8 个 Conv 后 counts=[18,3]；planner 两卡 final
上界均 10.96875GiB，真实 after_admission mapped 为 [10.03125,10.875]GiB，
Code pool 此次实际 mapped 增量 9.28125GiB（复用原有 free slots）。这是实际
跨模型准入，不只是 CPU ILP 可行证书。

相同 Code 首请求 queue：no_move 79.897s、KV-only 36.143s、ILP 163.463s。
容量失衡机制已触发；ILP 没有取得其收益不能归因于“没有构造到场景”。
但是 Code 占比仅 4/749，且占一张卡约 84%的受控 KV 预算；为其让位可能把
Conv 从近均衡放到 18:3。B 的收益和大量 A 的成本是真实权衡，不保证总 latency
必降；该 trace 的强失衡见证也不等于所有 B 到达都合计容量足够。

### ILP 未迁移的根因与下一验证

1. 九次详细 boundary 后的候选 cost gate 均拒绝。候选当前可行，但迁移+性能+
   fixed 成本 31.924–63.996 request-seconds，高于 legacy wait 阈值。
   最早一例自然释放 ETA 9.405s、阈值 7.524s、候选完整 gate 成本 45.453；
   实际随后 reserve 还等 92.687s，其间 56 条 Conv 被准入。legacy ETA 不模拟
   未来竞争 Conv 到达及其对共享页的占用，因而这次释放预测过于乐观。
   这些后续信息仅作 retrospective label，不可提供给在线预测，也不证明迁移
   反事实一定净赚 92s；另有预测偏长的样本，不能概括所有 ETA 都偏短。
2. 九次拒绝在 pause 前已经能用当时已知完整候选成本与 legacy wait 判定。
   当前 pause 前只筛 copy lower bound，未计完整 performance delta；实际仍
   支付每次 1.155–3.547s pause call 等成本，零迁移也拖慢老 Conv。ILP control
   含 preplan 累计 342.680s，no_move 291.318s；这是控制墙钟累计，不等同于
   因果 latency sum 增量。
3. 九次详细拒绝的 performance scope 仍是
   `single_model_N128_capture_surrogate_population_extrapolated`，不是这些当前
   混合人口的直接测量。既不能确认迁移代价估计足够准，也不能删除性能项
   或放宽门槛制造收益。需补覆盖实际 counts/graph shape 的双模型联合成本。

理想化两卡容量 C（已扣固定占用）、A 均衡总 KV 为 K_A、B 单请求 KV 为 s_B 时，
主场景区间为 `2*(C-s_B) < K_A <= 2*C-s_B`，同时要求 `s_B <= C`。
等价必要条件是 `max(F0,F1) < s_B <= F0+F1`；可移动性、共享 page owner、
copy-before-release 峰值与碎片还必须用真实 allocator 验证。稳定人口的选 rate
应看 `lambda_A * E[b_A * T_A]`，不能只看到达数量或分别均值的乘积。
容量见证和统计驻留率本身都不保证迁移净收益；还需 B 避免等待的 request-seconds
大于规划/暂停/复制和 A 失衡的总 request-seconds。

下一步先保持同 trace，验证只使用过去完成请求的 arrival-aware wait forecast，
并把完整 candidate cost gate 前置到 pause 前；随后补当前人口联合 cost 校准，
再正式对照。新 forecast 仅本地草稿，最新专项 8 pass/1 fail（fixture 预算问题），
未接 runtime、未同步、未提交，不称预测或 GPU 收益已通过。

证据：`evidence/agentic-trace-run-20261021/` 下 `live-trace-245085/`、
`archive-245085.json`、`capacity-verification-245085.json`、`verification-245085.json`、
`diagnostics-245085.json`、`wait-gate-diagnosis-245085.json`、
`watch-continuity-245085.json`。本地 20 poll 最大间隔 300.418s/SSH 0，remote
20 poll 最大间隔 301s；两个 watcher 已终态退出，未 commit/push/Aegaeon。

### 21:35 运行记录（已被以上终态结论取代）

own_model首策略runtime749完整drain/queue4.222s/latency39.950s/120sSLO739/0moves，
no_move178/749；尚无后三完整或独立net结果。新节点pilot Conv77.56s比旧样本慢，
保持冻结rate与四策略同allocation，不据先行结果改设置追正。compute两phase各
231+24+17+23+193tests通过；local八poll max300.418/SSH0，双300s监控，runtime冻结。

21:08:27 RUNNING/formal trace/own_model11/749；capacity runtime passed，非终态独立
收益核验。local三poll max300.418s/SSH0，remote1253463 PPID1持续实查；原参数冻结，
继续完整四策略与SHA/独立current owner/blocks/wait/cost和net分析，未发布/Aegaeon。

24508520:57:51提交、20:58:14获配d1n41a23g03两A800，capacity running，原749trace
四策略同allocation接续。73文件保护同步SHA通过；513本地tests/5skip、旧244639
2996trace兼容独立核验通过。local98790/98791 PPID1首SSH0、remote1253463每300s；
runtime冻结，终态全SHA/独立重验与net结果。尚无本轮GPU收益，无commit/push/Aegaeon。

### 20:54候选级重验准备（非GPU结果）

实现原选定tenant/rank/精确blocks安全重验，允许无关population churn，pause前及
32boundary后重建当前page owner/copy峰值/准入预算/performance及自然释放wait。
保持原749trace/rate/B/SLO/control/lookup；仅重放原两target候选与actual direct，
不是变化后全局最优。独立verifier核顺序owner释放/成本/当前wait tradeoff，证据包含
原selected精确blocks、两次raw和重验实际成本。协议 `docs/candidate-revalidation-protocol.md`。
完整本地513tests/5skip，shell检查；初次NameError与不足mock均修复、日志保留。

244639的13stale raw离线投影：ILP5/5与KV-only7/8至少一个当前owner容量可行候选。
旧raw未保存原精确blocks，因此不证明旧selected身份全部未变，更不代表GPU收益；
源SHA及独立copy/release重算在 `candidate-replay-244639.json`。准备唯一A800实测重试。

### 244639：查表+预规划原749trace完整负结果

20:16:43 COMPLETED/0:0，两A800/d1n41a13g02。111raw/out/err全SHA首归档、独立capacity/
四策略2996requests全部tokens/serial/owner/physical预算/默认sparse四rank图及32step
preplan receipts和总控制成本通过。冻结lookup从13raw重新构建与SHA一致。20:34:57
queue实查空，Aegaeon保护不变，无commit/push。原trace/rate/预算/SLO未改变。

| policy | mean queue s | mean latency s | Code queue s | Code TTFT s | 120sSLO /749 | moves |
| --- | --- | --- | --- | --- | --- | --- |
| own_model | 1.705 | 21.439 | 19.500 | 68.331 | 748 | 0 |
| no_move | 1.717 | 21.690 | 20.146 | 68.593 | 749 | 0 |
| KV-only | 1.697 | 22.126 | 37.266 | 85.166 | 748 | 1 |
| ILP | 1.498 | 21.964 | 34.492 | 82.281 | 748 | 0 |

ILP vs no_move全请求latency sum **+205.44s（+1.26%）**，Code+56.06s/Conv+149.37s，
SLO少1；queue均值减少12.75%不能抵消服务代价。ILP vs KV-only总latency -0.73%，
也不能称相对no_move净收益。baseline Code到达时resident Conv子集31条latency
-150.55s（-18.18%），但全部745Conv总代价增加，ILP没有真实迁移，不作因果贡献宣称。

KV-only一次550.5MiB exact echo跨Conv迁移，实际共18,240MiB全局空闲足9534MiB
Code，但两rank free9216/9024<actual increment9504；copy peak2.0625/2.71875GiB。
迁移elapsed .07242s，原.07sGiB×.538GiB低估这条完整copy，不能称pilot max是保证。
9计划只有1准入：5次boundary后stale、3次pause前stale；预规划3.862s/暂停8.869s。
ILP5计划全部stale（pause前4、boundary后1），零迁移；preplan1.935s/暂停2.809s。
选定request在这5次当前观测中均仍存在，主要是**未选择的请求完成或新请求准入
使整个population/global generation不一致**，不是ILP求解无解。边界后第一case
Conv6:6→4:6，未选择的两条已完成，本可降低容量压力却整计划丢弃。

修复方向：选定request identity/rank/完整blocks不变，允许其他人口变化；在当前
真实page owners/新population/perf表上确定性重放原两target候选的每次copy-before-
release/最终准入，不能直接应用旧最优解或再暂停内ILP。重验成功只称原候选当前
安全可行，**不是新population全局最优**；选定完成/重映射、未知owner、peak或预算
失败继续拒绝。wait-aware也须当前页释放等待重核，不能盲降gate或改rate追正。

local20poll max300.191s/SSH0，remote20poll max302s覆盖终态；原包195MiB下载较慢
完成后全SHA通过，socket-shutdown WARNING保留。证据`verification-244639.json`、
`diagnostics-244639.json`、`results-244639-compact.json`、`watch-continuity-244639.json`。
**这一轮完整算法收益为负，研究收益目标仍未完成。**

19:59最后ILP开始（19:56:16 RUNNING），前三策略各749完整drain；KV-only runtime
一次550.5MiB迁移、9次planning，control94.668s含preplan98.530s。meanqueue1.697s/
latency22.126s/drain15.119s/SLO748，队列均值微降不等于总代价改善。local15poll
max300.191s/SSH0，等ILP完整和独立逐请求验证，不先宣称净收益或更改参数。

19:43阶段进度：no_move完整749/494107output，runtime queue1.717s、latency21.690s、
drain14.870s、SLO749/749；own_model也已完整，尚待终态独立核验，不比较ILP。
19:41:16 KV-only180/749/moves0，随后ILP；local12轮max300.191s/SSH0，源码冻结。
保持原rate/trace/预算，不能根据新基线背景变轻就临时修改offered rate追正收益。

19:23阶段进度：own_model完整749requests/494107output drain13.762s，runtime mean
queue1.705s、latency21.439s、748/749满足120sSLO、moves0，尚待终态独立配对验证。
19:21:17 no_move229/749，余KV-only/ILP接续；不能据单策略称ILP收益。local八轮
max300.191s/SSH0、remote3280315 PPID1持续。两phase启动compute各215+24+17+23+193
tests通过，源码和固定负载冻结，终态核验2996结果及老Conv代价。

19:08正式进行：24463919:06:15 RUNNING，首own_model440/749completed、moves0，
local五轮max300.000s/SSH0。控制/lookup/rate/trace全部冻结，尚无其他策略完整结果；
继续同allocation，不能提前宣称新控制路径收益或依据首策略重选输入。

18:52正式阶段实查：24463918:46:08获配d1n41a13g02两A800，18:51:17 RUNNING/
formal_trace_running，capacity runtime通过后正在重载双模型。local两轮299.991s/
SSH0，尚无完整trace结果。18:47 PENDING摘要来自18:46:07remote首查，随后18:46:15
local已运行，具体状态时间以raw为准；不把phase名称当全部四策略已执行通过。

18:47原设置唯一替代：66files保护同步a800-20261023-184301/old、497tests/5skip。
24463918:45:46提交，18:46:14 PENDING/Priority；local80866/80867 PPID1首querySSH0，
remote3280315 query-only300s至终态全SHA/独立capacity+2996结果核验。原control/
lookup/749trace/rate/预算/SLO不变，运行源码冻结不覆盖，暂无本次净收益结果。

18:42正式启动失败已诊断：24461918:35:25 FAILED/1:0，前两启动tests通过，新增
measured mock第三组12pass/1fail。子shell继承DP_OPT_CAPACITY_PILOT=1，截取参数块
校准分支读取未定义root，不是allocator/CUDA失败；尚无模型加载或100K执行。
7raw/out/err全SHA首归档，err.partial保留；local两轮max300.102s/SSH0，remote两轮
300s覆盖终态。测试子shell隔离所有继承DP_OPT，仅显式注入case参数；8污染环境
专项/497研究tests5skip通过。lookup/control/749trace/rate/预算/SLO未变，准备唯一
原设置A800替代。启动失败不称capacity通过或净收益完成，无commit/push/Aegaeon。

18:36正式重放已开始：66files保护同步/16冻结输入SHA通过，备份
`a800-20261023-182736/old`。慢SCP残包保留，改集群已归档同SHA raw本地复制，没有
登录节点imports/tests。唯一24461918:34:12提交、18:34:37获配d1n41a26g03两A800，
18:34:42 RUNNING/capacity启动。local79158/79159 PPID1首SSH0、remote3080413每300s，
终态SHA/完整四策略2996结果独立核验；首remote启动缺phase读取exit1保留，随后
local确认运行。runtime冻结、原749trace参数不变，尚无本次GPU净收益结果。

### 244371：完整小人口attention-DP查表

18:19:32 COMPLETED/0:0，两A800/d1n41a18g03；首轮归档46raw/out/err全SHA、独立
capacity及144cases/1664requests通过。每请求serial tokens、两个owner/population
快照及共享256..768窗口（全部prefill结束后、首个completion前）、physical预算和
四rank默认sparse图/32step通过。只Conv active、另Code resident idle，不含联合prefill。

| Conv人口 | 平衡cadence ms | 双方向极端cadence ms | 极端吞吐损失 | affine最大ratio残差 |
| --- | --- | --- | --- | --- |
| 8 | 16.21 | 17.56 / 17.73 | 7.65% / 8.54% | .026 |
| 9 | 17.03 | 19.85 / 19.47 | 14.22% / 12.57% | .057 |
| 10 | 17.16 | 20.91 / 20.52 | 17.95% / 16.37% | .060 |
| 11 | 17.54 | 21.01 / 21.00 | 16.51% / 16.46% | .090 |
| 12 | 16.79 | 20.84 / 20.84 | 19.45% / 19.44% | .055 |
| 16 | 15.72 | 21.58 / 21.49 | 27.13% / 26.83% | .209 |

这里吞吐损失为1-T_balanced/T_layout，不是cadence增长百分比。affine为描述性
固定平衡intercept拟合，只有N8残差<.05，不能整体声称线性。默认sparse阶梯下直接
采用N*horizon*(T_count-min(T_count))非负表，比较增量不受平移影响。每count两重复，
最大重复相对spread 0.16%–5.26%；不是多seed统计泛化或完整长Code共享性能校准。
未测人口及Code仍显式旧N128代理回退，实际结果须披露覆盖范围。

本地15轮query最大300.180s，18:06:26 SSH255握手EOF，18:08:19重查恢复，远端15轮
最大345s覆盖终态；不称监控无中断。shutdown未确认worker socket的WARNING保留，
不全局清理。证据`layout-cost-verification-244371.json`、`results-244371-compact.json`、
`watch-continuity-244371.json`与完整raw。18:25:38 SSH queue实查空，gjghfd/main/
Aegaeon protected diff复核不变，无commit/push。

原244279独立重核验后冻结控制成本：incoming3.06s/preplan.59s/oldfixed.09s/
copy.07sGiB，包含完整move RPC而非只collective；控制config SHA
`dc9eac38f1e75155a893c05923f89f2bb94cc5b4e64ebdbe50bc12c0a3d34263`，lookup SHA
`6d140910091cd1904a0a4d68f0d43472465578870330e5297ce92652ae7e3df3`。规则与范围见
`../docs/measured-control-trace-protocol.md`，493本地tests/5skip通过。下一同749trace/
Conv.61-Code.004/1200s/seed20261021/B11328/SLO120四策略，三迁移策略统一preplan/
boundary32/echo，只有ILP wait-aware；**成本校准完成不等于trace净收益完成**。

17:49运行进度：244371 17:46:33 layout78/144cases/RUNNING；local八轮max299.996s/
SSH0，runtime冻结。新增查表13rawSHA/全部raw重建/共享256..768人口窗口、三策略
prepared参数一致性与总控制成本回归493passed/5skip，未同步/未GPU执行。准入成本
冻结预声明max observed ceil .01：incoming3.06/preplan.59/oldfixed.09/copy.07sGiB；
这不是反事实prefill上界或净收益。下一144case独立曲线/核验、原749trace四策略；
协议`../docs/measured-control-trace-protocol.md`，未改变RPS/seed/预算/SLO。

17:31运行进度：244371 17:26:34两A800/RUNNING，布局校准26/144cases，仍未终态。
local四轮最大299.992s/SSH0持续，runtime冻结。已有485tests/5skip，新增lookup
13原始文件SHA与重新独立构建验证、boundary成本参数传递只本地待回归，未GPU执行。
749trace/rate/seed/预算/SLO均未改变；不能把部分校准或成本优化称正式收益。

17:12下一校准：483研究tests/5skip、独立layout verifier7专项通过，62files保护
同步a800-20261023-170702/old。唯一244371 17:10:52提交、17:11:04获配d1n41a18g03
两A800，17:11:35 capacity启动，compute首212/新增23tests通过。local67064/67065
PPID1首query SSH0、remote1813646 query-only，每300s至终态SHA/独立144cases核验。
仍只一activeConv+另一residentCode idle lookup，未测Code prefill联合布局曲线。

### 244279：混合人口下真实100K准入与预规划成本

16:52:01 COMPLETED/0:0；46raw/out/err全SHA、capacity及6cases/78请求独立核验
通过，完整serial tokens、实际全长度Code reservation、Conv唯一owner、copy峰值/
physical预算与四rank默认graph/32step receipts均通过。仍为成本pilot，非Poisson收益。

| old Conv人口 | moves / copy | native / preplan incoming control s | old max gap native / preplan s | old latency sum delta（preplan-native）s |
| --- | --- | --- | --- | --- |
| 8 | 1 / 432MiB | .8659 / 3.3229 | .8973 / .3061 | +19.3893 |
| 12 | 3 / 1296MiB | .9746 / 1.5497 | 1.0117 / .3647 | -39.3090 |
| 16 | 5 / 2160MiB | 1.2625 / 1.8117 | 1.3068 / .3566 | -30.8854 |

CPU ILP从暂停内.1497/.2897/.5745s移到decode继续时.1609/.3169/.5869s，boundary
暂停内solve严格0；old阻塞上界合计.8837/1.7768/3.2537 request-seconds。native与
preplan各人口只有一次，N8的old latency回退必须保留，不称所有人口改善。
Code TTFT约44.76–49.03s；原Code serial25.28s、Conv serial11.63s总service。
Code prefill时Conv mean gap约.052/.079/.061s，而Code结束后约.017/.020/.021s；
说明shared GPU长prefill干扰确实存在，不能只以复制25ms规划或将历史时间当因果收益。

local5poll最大300.005794s/SSH0；remote5poll16:33:19–16:53:22最大302s覆盖终态。
首次归档成功，双watcher退出；16:56:25 SSH queue实查空。证据
`admission-cost-verification-244279.json`、`results-244279-compact.json`、
`watch-continuity-244279.json`，保留前两启动env失败7raw/out/err完整证据。

关键下一门槛：N128性能surrogate外推罚8/12/16迁移布局14.87/98.49/145.85request-s，
不能称这些小人口成本实测。先独立完整枚举N8/9/10/11/12/16的全部DP counts/双方向/
两重复，Azure3584/1024、256..768稳定decode；协议`docs/layout-cost-protocol.md`。
不改原749trace/rate/seed/B11328/SLO，校准后才更新wait gate，负结果不删。

16:34：所有四批启动tests统一清transport；14继承echo-env专项、475研究tests/5skip
通过。58files保护同步备份a800-20261023-163028/old，唯一244279 16:33:01提交，
16:33:32 PENDING/Priority。local62143/62144 PPID1首SSH0，remote1291631标准log
query-only，每300s至终态全SHA/独立capacity+78请求mixed准入成本核验。
正式749trace/RPS/预算/图未改，未测出本轮成本/收益，不把后台运行称完成。

16:29：244228 16:22:15 FAILED，首修仅第一批启动tests隔离transport，212tests及
新增19通过，但第三peer tensor组仍有host-only mock继承echo（22pass/1fail）。
没有模型加载/100K执行。统一四批tests全部清transport，推理保持echo；14继承
echo-env专项通过，全回归中。首终态归档7raw/out/err SHA、local2poll max300.002885/
SSH0、remote2poll max300s通过。最小修复不完整已保留，下一唯一A800重试同setting。

16:22重试：58files旧/新SHA/备份/保护同步通过，a800-20261023-161859/old；唯一
244228 16:21:20提交、16:21:31在d1n41a16g01两A800获配，16:21:52 capacity启动
tests进行中。local60315/60316 PPID1及首query SSH0，remote1126999已恢复标准
remote-watch.log，300s至终态归档/独立capacity+78请求准入成本核验。不覆盖runtime。

16:18：244176归档第二次成功，raw/out/err全SHA；首SSH255及错误log名原件保留。
472研究tests/5skip及11echo-env专项通过，启动测试清transport而推理保留exact echo。
准备唯一A800替代，继续同6cases/78结果成本校准；尚无mixed准入/净收益结果。

16:13失败诊断：244176 16:06:30获配d1n41a28g01、16:07:07 FAILED/1:0，启动
208tests通过/4mock host迁移失败，没有开始模型加载、100K或mixed成本case。
新launcher导出DP_OPT_KV_TRANSPORT=dp_echo，使host-only fake RPC走peer分支返回bool，
触发TypeError；不是CUDA OOM、allocator不足或算法无解。启动mock测试增加env -u
DP_OPT_KV_TRANSPORT，推理仍保留exact echo；11echo-env专项通过，全回归中。
首终态归档SSH255原件保留、300s重试；16:12:06在线queue确认空，原query watcher
停止避免并发归档。local第二轮300.003779s SSH握手255，remote两轮300s覆盖终态但
错误重定向到remote-watch-launch.log，原记录保留，不称local无异常。

16:04下一实验：471本地研究tests/5skip、21预规划gate/终态专项通过。57files保护
同步，备份a800-20261023-160027/old、旧/新SHA及gjghfd/main/Aegaeon保护通过。首次
显式CPU/mem提交被site Lua拒绝，16:02:27实查queue/accounting无job；按243415默认
提交参数唯一244176于16:02:45成功，16:03:16 PENDING/Priority。不推断Lua具体规则。
先100Kcapacity再6cases人口8/12/16，Azure3584/1024与101670+64Code，比较暂停内
ILP与decode继续预规划/default32边界，成本pilot不是749Poisson正式收益。local57801/
57802实查PPID1，remote866689，300s至终态全SHA/独立核验；不覆盖冻结runtime。

### 243415：默认共识边界保留 decode，减少实际阻塞而非共识等待

12:30:47 COMPLETED/0:0；46raw/out/err全SHA、独立capacity和36cases完整tokens/
serial/owner/physical预算/copy峰值/四rank默认sparse graph通过。每个边界receipt
严格落在32step共识，空attention rank继续EP；native/default32各18同mark/方向配对。

| 成本（秒） | native keep mean / max | default32 boundary mean / max |
| --- | --- | --- |
| incoming control delay | .648602 / 1.010686 | 4.065191 / 5.038848 |
| active tenant blocking upper | .648602 / 1.010686 | .033845 / .047990 |
| client maximum token gap | .632125 / .690455 | .281760 / .338984 |

边界握手期间真实客户端继续生成26–59tokens，native为0。老请求实际阻塞均值降低
94.78%，但新请求等待共识反而增长；不能把pause_s的4秒都乘全人口，也不能忽略
incoming的4秒。client最大gap仍约.28秒，不能只用.034秒receipt阻塞上界承诺TPOT。
本轮仅一个活跃请求，尚未校准两活跃tenant/多个老Conv及100K Code入场后prefill代价。

local5poll最大300.005800s/SSH0；remote5poll 12:15:31–12:35:31最大300s覆盖终态。
首次终态归档成功，watchers退出。15:35:50 SSH复查queue空。聚合证据
`results-243415-compact.json`、`watch-continuity-243415.json`；raw不覆盖。
预规划460tests/5skip只本地，先做混合人口成本/真实Code准入，再冻结wait gate；
保持749trace、rate、budget、SLO，尚无新placement净收益。

12:16进度：boundary成本pilot243415于12:14:52提交、12:15:17在d1n41a22g03获配
两A800；50files保护同步通过，455研究tests/5skip及CLI31通过。先capacity再同36
exact echo配对native keep/default32_boundary，终态自动SHA/独立核验；暂无boundary
GPU结果。local27954/27955、remote942762，每300s，源码冻结不覆盖。

### 243300：完整 GPU echo 正确，剩余瓶颈转移到控制路径

11:33:58提交，11:34:14在d1n41a23g03两A800获配，11:54:30 COMPLETED/0:0。
45raw/out/err全SHA、独立capacity/36cases串行tokens、owner、physical mapped
预算/copy峰值、四rank默认graph通过；每phase compute200+23真实tensor（含一byte
腐败必须失败）+193integration tests通过。本轮完全相同Azure三mark/方向/重复，
比较CPU内容SHA与完整目的KV逐byte GPUecho；receipt不冒称KV内容SHA。

| 实际迁移KV | DP-device + CPU SHA mean s | DP-device + full GPU echo mean s |
| --- | --- | --- |
| 240MiB | 0.842044 | 0.706954 |
| 432MiB | 0.774771 | 0.413278 |
| 768MiB | 1.201867 | 0.362572 |

总体迁移.939560→.494268s（约47.39%下降），但peer RPC本身.516136→.025852s，
说明完整GPUecho确实去掉CPU读回/SHA主成本。端到端仍有destination prepare
均值.369221s/max1.258712s，source export/capacity/release也有长尾；两轮节点
不同，不将243165与本轮的均值直接当严格配对。逆长度趋势是这些控制长尾影响，
不是声明KV更长越便宜的规律。

echo默认pause median.605757s/max4.474623s；CPU SHA pause max1.057799s也保留。
暂停内transaction均值echo为.941163/.716187/.507434s（按三mark），不能只以
26ms复制RPC预估净收益。local6poll最大300.006s/SSH0、remote6poll从11:34:21至
11:59:22最大301s，覆盖终态；终态archive首次成功且双watcher退出。
证据`archive-243300.json`、`transfer-verification-243300.json`、
`results-243300-compact.json`、`watch-continuity-243300.json`。

下一重点：现有ILP在pause内执行，而原生keep在收到命令时停request，再跑空EP
直到默认32step共识。拟改为**ILP在decode继续时预规划，默认32step共识checkpoint
才停止decode**，无请求tenant先握手、活跃tenant后握手；不改capture、不减少32
共识频率、不省略空rank EP。严格复核预规划ownership/blocks/pages，变化时放弃
而非应用过期计划；完整记录incoming控制delay与每tenant实际阻塞，不把二者都
粗暴按全人口乘算。先功能/成本pilot，测到门槛后才冻结wait参数和正式trace。


11:35进度：49files保护同步/备份/新SHA通过，唯一echo成本pilot243300已11:33:58
提交，11:34:14在d1n41a23g03获配两A800，原100Kcapacity进行中。local21643/21644
PPID1首轮实查及remote337129，每300s至终态SHA/独立核验，源码冻结不覆盖。
尚无echo GPU正确性/成本结果，正式trace仍未重跑，不将提交视为收益完成。

### 243165：复制成本通过，尚非 placement 净收益

10:42:32提交，10:58:42在d1n41a15g01两A800获配，11:14:29 COMPLETED/0:0。
45raw/out/err全部SHA，独立capacity与36迁移cases验证通过：完整tokens/serial
reference、唯一owner、physical mapped预算/copy峰值、四rank默认sparse graph。
capacity仍B11328，四pool101675+64真实allocate/rollback与5请求通过。

| 实际迁移KV | host_file mean s | DP-device + CPU全SHA mean s |
| --- | --- | --- |
| 240MiB | 1.233435 | 0.293836 |
| 432MiB | 2.213113 | 0.488250 |
| 768MiB | 3.908578 | 0.812833 |

总体2.451708→0.531639s，约4.6倍复制加速；这是单request成熟KV成本，不是混合
trace净收益。host export gather/D2H .2375s、save .3808s、SHA .6399s；import
load .2052s、H2D .0487s、readback .2349s、SHA .6588s。peer GPU collective仅约
.004–.0105s，完整CPUreadback/SHA约.487–.501s，是主要剩余成本。默认pause median
约.606s，peer有5.995s/8.370s两个异常，原因未证实；不能从复制均值抹掉异常。
退出TCPStore/NCCL warnings保留，job成功且tokens正确，不修无关shutdown问题。

local8poll最大300.017s/SSH0，remote8poll从10:43:06至11:18:07最大301s，覆盖终态；
watchers已退出。证据`archive-243165.json`、`transfer-verification-243165.json`、
`results-243165-compact.json`、`watch-continuity-243165.json`，均在原agentic-trace
evidence目录。

241907的15旧wait状态，14个即使copy=0，固定pause+solve .5s乘active人口仍超过
.8W；唯一余下状态W7.868s/pop12，可承受总control .52456s，固定.5s时copy需
低于.02973s/GiB。因此不能只把copy slope降低就重跑或宣称有净迁移机会。
`wait-cost-affordability-241907.json`是事后代数，不是GPU最优可行见证。

下一同marks/36cases/方向/预算配对比较`dp_allgatherv`与`dp_echo`：从目的实际KV
逐chunk反传到源，GPU逐byte全等、双rank ack后才commit/remove，不采样或概率校验。
echo返回SHA仅是验证receipt，不是KV内容SHA。442本地研究tests/5skip通过（无torch
跳过真实tensor测试），尚未同步/提交；默认正式transport仍host_file，原749trace/
rate/SLO不改。待真实echo成本与固定pause/solve联合门槛，研究净收益目标未完成。

### 241907：wait-aware 四策略完整结果

11:06进度：243165实际10:58:42获配d1n41a15g01两A800；20关闭capacity rawfiles
前后SHA一致、独立100K allocate/rollback、5推理/四rank sparse通过，B11328。
11:05:05进入transfer成本pilot；36配对尚待结果，不把capacity作为placement收益。

10:44进度：48files保护同步/备份/SHA通过，唯一A800243165于10:42:32提交，
10:43:15 PENDING/Priority。先100Kcapacity再36host/DP-device成本pilot，437本地
研究tests/2skip，未修改原trace/RPS或降低wait成本。local3098/3099与remote3678098
每300s、终态独立capacity/transfer；没有新GPU结果，源码冻结不覆盖，不commit/push。

241907于02:34:48 COMPLETED/0:0。最终95raw/out/err全部SHA、capacity、四策略
2996请求tokens/串行reference/ownership/controlled budget/copy峰值/四rank sparse
graph独立通过。多次SSH超时/断连的partial与日志保留，09:29终态恢复第二轮成功，
09:36独立诊断完成；没有把尚缺err的早期raw核验称完整终态归档。

设置仍749请求（745Conv/4Code）、1200s/seed20261021、.61/.004RPS、B11328MiB、
BF16/128K/24GiB虚拟pool/default sparse/pause32，只ILP新增wait-aware。

| 策略 | mean queue s | mean TTFT s | mean latency s | 120s SLO | moves |
| --- | --- | --- | --- | --- | --- |
| own_model | 2.369 | 2.889 | 22.216 | 748/749 | 0 |
| no_move | 1.866 | 2.386 | 21.371 | 748/749 | 0 |
| KV-only | 2.762 | 3.285 | 23.052 | 748/749 | 4 |
| wait-aware ILP | 1.470 | 1.986 | 20.872 | 749/749 | 0 |

ILP queue -21.20%、latency -2.33%，SLO多1，但**零迁移、零solve**，不能证明
placement收益或统计显著性。总latency差-373.190s（Code-7.420/Conv-365.769）；
baseline在Code到达时已驻留的固定28Conv仍多44.969s（+7.00%）。固定offered rate
完整drain407.115→406.951tokens/s，不能称吞吐增益。相对上一轮24.173s staged
负收益，wait合理避免昂贵迁移，但不能把跨run的随机时序波动当ILP新收益。

wait实查69次/69拒绝；15去重journal全部有有效decode剩余量，不是估计全None。
W=.236..7.868s，迁移全active延迟下界11.268..55.570s，全部高于.8W；因此
当前host-staged复制下等待优于迁移的预筛与实际动作一致。估计不是未来oracle，
未证明所有被拒真实最优；小样本不能外推生产化。

local7poll最大gap4682.511s，含522.641/445.174s前两gap，不称连续五分钟。
remote19poll从01:05:13至02:35:21最大302s，独立核验覆盖job终态。
证据`verification-241907.json`、`diagnostics-241907.json`、`archive-241907.json`。

下一先测真实复制成本：已知A800 pair topology NV8，但旧transport为host-file，
不是NVLink优化。增加可选现有DP device allgatherv传KV、小块32MiB、两rank并发，
保留源和目的真实字节SHA读回及owner/预算/commit顺序。先host/peer配对成本pilot，
测gather/collective/scatter/readback/hash/RPC/pause分段，不凭NV8推断端到端速度。
若GPU正确性/成本门槛不达则失败归档；未经实测不降低wait参数、不直接重跑收益。

排队期代数诊断`wait-cost-affordability-241907.json`：15份保存的门槛状态中，14份
即使copy=0，现声明pause+solve=.5s乘active人口也超过.8W；唯一余下状态W7.868s/
population12只允许总control.52456s，fractional复制下界.82617GiB，固定.5s下斜率
需<.02973s/GiB。将copy5降至1/.2/.05s/GiB仍0/15通过。这是已存状态的事后下界
代数，不是新layout可行性/未来oracle或实测GPU收益；说明应同时测固定RPC/控制
成本，不凭NVLink加速就预期同trace必有迁移收益。

## 先前状态：2026-10-23 01:38，Asia/Shanghai

241907 own_model全749完整drain，queue2.369s/latency22.216s/SLO748/749，no_move
01:36:19到178s/112完成，仍待ILP wait实跑和完整独立核验。local实际查询出现
522.641s与445.174s间隔（SSH0），remote七轮01:05:13→01:35:15/max301s独立
实查，不能以remote正常替代本地连续声明；证据`monitor-gap-own-complete-241907.txt`。

241907关闭capacity20files前后SHA一致/独立核验通过，B11424>=frozen11328、18
稳定samples/四pool最长真实allocation rollback/5pilot/四rank图，formal保持B11328。
01:15:10 own_model131s/75完成；三轮300s/SSH0，runtime冻结。仅本地新增在线rate
边界、paused不追加、8点上限和结束清理测试66通过/1skip，未覆盖活动runtime。
等待真实ILP gate/完整四策略，不把capacity或旧离线重放视为收益。

wait-aware唯一重试241907于01:04:39提交，01:05:05在d1n41a16g01两A800开跑，
capacity进行中。41files保护同步通过；local48118/48119和remote1205024每300s，
runtime冻结，同749trace/B11328/原rate/defaultgraph，497tests/1skip。当前只是
获配，不能替代新门槛live验证或收益结论；旧241580完整负结果保留。

01:02下一版已本地实现wait-aware gate：在线最多8点decode速率/.5s采样估剩余，
shared page以最后owner释放时刻预测自然等待W；全active+incoming的pause/solve/
fractional copy乐观损失下界与.8W比，未知/短等待不pause，epoch内再与原ILP
objective和已实际耗时比较。固定cost.35s/.15s/5s-per-GiB来自241580量级，不是
完整joint校准；七条旧迁移截止当时clientcadence离线重放均拒绝，只诊断。497
tests/1skip，准备41files保护同步唯一A800同749trace/B11328/原rate重试，baseline
不改，ILP允许等待，尚无新GPU结果；见协议与`wait-gate-replay-241580.json`。

## 完整四策略结果：241580

**运行与正确性通过，但ILP净收益为负。** 2026-10-22 23:04:25至10-23 00:33:39
两A800/d1n41a11g02；同749请求（4Code/745Conv）/1200s/seed20261021，B11328MiB/
GPU、Conv.61/Code.004 RPS，BF16/128K/24GiB虚拟/default sparse/pause32。94raw及
out/err全SHA、capacity、全2996请求tokens/四串行reference、ownership/mapped/copy
峰值与四rankgraph独立通过。报告`verification-241580.json`、`diagnostics-241580.json`。

| 策略 | mean queue s | mean TTFT s | mean latency s | 120sSLO | staged/moves | staged控制 s |
| --- | --- | --- | --- | --- | --- | --- |
| own_model | 1.712 | 2.225 | 21.295 | 748/749 | 1/0 | 1.205 |
| no_move | 1.970 | 2.485 | 21.521 | 748/749 | 0/0 | 0 |
| KV-only | 2.528 | 3.043 | 22.669 | 748/749 | 5/4 | 13.793 |
| ILP | 3.156 | 3.676 | 23.825 | 748/749 | 5/6 | 24.173 |

ILP相对no_move queue **+60.26%**、TTFT **+47.95%**、latency **+10.70%**；相对
KV-only latency+5.10%。SLO未改善，完整drain吞吐407.042→406.971 tokens/s近似不变，
固定offered rate不能称吞吐增益。ILP总latency多1725.197s，其中745Conv多1719.021s、
4Code多6.176s；baseline在Code到达时已驻留的固定30Conv多778.494s。不能只报Code。

| Code arrival s | no_move queue/latency s | KV-only queue/latency s | ILP queue/latency s |
| --- | --- | --- | --- |
| 171.566 | 8.473 / 66.047 | 6.211 / 62.910 | 11.182 / 68.886 |
| 172.485 | 65.344 / 121.693 | 62.425 / 121.397 | 68.176 / 124.818 |
| 754.829 | .109 / 58.056 | 3.098 / 61.062 | 5.641 / 63.198 |
| 927.367 | 7.925 / 65.352 | 3.213 / 60.615 | 3.308 / 60.421 |

KV-only首Code确有早准入2.262s，但不足抵偿老Conv/整体代价。ILP首pause7.115s，
总pause9.097s/solve1.984s/copy12.547s；KV-only总pause2.132s/solve1.536s/copy9.651s。
ILP有5跨tenant+1自身迁移、KV-only4跨tenant；性能项仍单模型posthoc外推，不是成熟
双模型校准。候选8/moves8内最优不等于等待选项也最优，第一次迁移快于自然等待
并不保证总延迟更优，已有明确反例。

迁后四worker缓存回收/fresh容量使Code真正reserve：KV-only三个Code迁移decision
目标free7231/7103/7103blocks，ILP三个Code目标6591/6783/7167，均不再6340需求下
free5631的pending陷阱；记录全部actual reserved/token正确。新run观测不能证明旧
run缓存是唯一因果。physical-budget witness、logical预测与整卡OOM仍分开。

local21轮最大gap300.148s/全SSH0；remote21轮max301s，终态自动94files SHA/
独立verify/diagnostics完成，watchers退出，00:38:32用户queue空。旧失败241086/
241267/241311全部保留。`results-241580-compact.json`与`watch-continuity-241580.json`
保存小汇总；未commit/push/Aegaeon。

下一不是盲调rate：保持冻结749trace/B/SLO，用真实page-owner释放与在线decode
剩余量估算自然等待，引入实测pause/solve/copy成本下的wait-vs-migrate选择。门槛
需全active成本与新请求收益，允许主动等待；不以GPU功能通过宣称整体正收益目标
完成。本轮只单seed initial comparison，缺多seed/联合cost校准和生产化结论。

### 先前进度记录

前三策略完整drain，最后ILP于00:17:40到280s/132完成/2moves；前KV-only结束前
741/749时4moves，未再出现900s pending超时。仍待四策略全SHA/所有token/owner/
copy峰值与Code/老Conv成本独立核验，暂不宣布净收益。local17轮约300s/SSH0，
runtime冻结，继续同allocation至终态。

23:58:25两baseline own_model/no_move各749（4Code/745Conv）完整，KV-only216
完成（2Code/214Conv）与2真实跨tenant迁移，首两100K不再pending到900s，证据
`post-copy-code-progress-241580.txt`。尚需完整raw/fresh容量与内存核验，ILP未跑，
不将机制通过当净收益；local13轮约300s/SSH0，runtime冻结，继续同allocation。

241580于23:04:25在d1n41a11g02获配两A800；关闭capacity20files前后SHA一致与
独立核验通过，四pool真实101675+64分配rollback/5pilot/四rank sparse图/19稳定
samples，实测B11328、formal同B。23:12:39 own_model47.76s/25完成。证据
`capacity-pilot-241580/`及`capacity-verification-241580.json`。local四轮约300s/
SSH0，双watcher持续，runtime冻结；仍待四策略与迁后actual block/memory、Code
准入和老Conv代价，未commit/push。

22:58更新：39files保护同步通过，唯一transfer-headroom重试 **241580**于22:57:22
提交两A800，22:57:40 PENDING/Priority，local31673/31674与remote3626998每300s。
同749trace/B11328/.61+.004/默认graph，483tests/1skip，runtime冻结。241311两
749baseline及542失败前缀的journal、tokens、记录owner/copy峰值独立通过，两个
baseline及四serial reference token相同，失败前缀token也同；见
`partial-verification-241311.json`。remote16轮/maxgap301s，仍无完整ILP收益。

**241311再次FAILED（22:38:39），本轮四策略收益仍未完成。** 77raw/out/err全SHA
归档、capacity独立通过；own_model/no_move各749完整，KV-only542完成全Conv/0Code，
ILP未开始。证据`formal-failure-241311.json`，没有将失败前缀算作同请求吞吐。

本次并非暂停风暴：仅1次staged/7.195s和2条真实跨tenant迁移。首Code总101435
tokens需6340blocks，迁后target rank0 mapped1.6875GiB、rank1 3.75GiB，保守ILP
final11.0625/4.3125GiB在B内，但**目标实际free_blocks5631<6340**；Code被提交，
0KVblocks/reservation_pending持续到900s timeout，tenant后续Code FIFO也被堵。
末态KV mapped9.75/10.78125GiB、49active/123unfinished，不是CUDA OOM证据。

两个host-staged copy分别523763712/649592832 bytes、2.761/3.391s，copy实际
`index_select→CPU/CPU→GPU/index_select校验`可留下worker PyTorch缓存，抵消
释放的driver free。旧run没有迁前后torch-reserved直接观测，故这只是具体待验证
解释，不断言其是唯一因果。更直接的确定性bug是迁后只验证mapped预算、没核验
实际allocator free，导致不该提交的pending。

下一修复：四策略每个trace起点都同步回收unused worker缓存并保存memory-start；
真实迁移完成后四worker再回收unused、保存free/allocated/reserved，刷新manager
physical capacity TTL，核对new required_blocks<=actual free后才提交。若不足，结束
transaction并resume正确已迁请求、incoming仍外部FIFO等待，决策保留所有copy代价，
不把迁移回滚或隐藏为零成本，不提高预算/timeout。默认图与活KV不清除，新增回收
本身计入staged控制时间；不是物理可行性充分保证或成熟wait-aware最优算法。

145专项/1skip及483全研究tests/1skip通过，测试覆盖迁后actual不足不调用admit、
正确resume/end、四worker同步与fresh capacity。准备同749trace/.61+.004/B11328/
8moves保护同步唯一两A800重试，仍未提交/收益完成/发布。全部旧负结果保留。

no_move也全749完整drain：queue1.735s、latency21.508s、SLO748/749、drain14.335s，
summary/SHA在`no-move-complete-241311.txt`。22:28:17 KV-only已实跑到472s，246
完成/2moves，ILP未开始。各指标仍待终态全SHA/独立token/ownership验证，不能拿
同offered rate下两次迁移当净吞吐或SLO收益。local13轮约300s/SSH0，继续同allocation。

own_model已完整749请求drain：queue1.773s、latency21.375s、120sSLO748/749，
drain13.870s、0moves/3staged、87/90无暂停预筛拒绝。远端summary及SHA保存于
`own-model-complete-241311.txt`，待终态全部tokens/owner/graph独立验证，暂非完整
收益。22:03:17 no_move已接续，local8轮300s/SSH0，继续kv_only/ilp，不覆盖runtime。

21:48:51正式own_model增量journal已有345完成（2Code/343Conv），只2次staged
planning；初期426次无效暂停风暴未在该前缀重现。不从部分请求计算四策略净收益。
local5轮300s/SSH0，remote2218964已PPID1实查，继续同job至全四策略终态。

21:40更新：241311关闭capacity20files前后SHA一致、独立核验通过：四pool真实
101675+64分配rollback/5pilot/四rank sparse图/20稳定内存样本，B11424；formal
仍固定11328/原749trace，21:38:19启动。证据`capacity-pilot-241311/`及
`capacity-verification-241311.json`。三轮local300s/SSH0、remote持续，runtime
冻结；尚无完整四策略收益，继续查Code准入及老Conv代价，不增加预算追正。

21:34更新：241311于21:32:50 d1n41a18g03两A800开跑，capacity进行中；compute
首套190tests通过，local第二轮gap300.007s/SSH0，runtime冻结。新增仅本地离线
逐请求配对Code/allConv/all以及baseline在Code到达时已驻留Conv固定集合分析，
同时报告queue/TTFT/latency/TPOT和SLO差值，Code改善但老Conv/总latency变差仍
明确为负；45专项/480全研究tests+1skip，未同步活动runtime。仍待正式GPU收益。

21:29更新：journal修复38files保护同步通过，唯一重试 **241311**于21:27:55
提交两A800，21:28:17 PENDING/Priority；local20174/20175与remote2218964每300s，
runtime冻结，同749trace/B11328/.61+.004/8moves。等待实际capacity/四策略结果，
477CPUtests/1skip，不把排队当成功，不commit/push/Aegaeon。

21:26更新：241267于21:22:56 FAILED，31raw/out/err全SHA及独立capacity通过，
测得B11424>=frozen11328、四pool101675+64分配rollback/四rank图/5pilot。formal
启动CPU测试188passed/1failed，未开始正式请求：异步线程写requests.jsonl乱序，
与按completion先append的final list顺序不同。不能忽略此failure或把capacity当收益。
修为每个journal独立asyncio锁，保持同journal先后与final一致，不累计重写JSON；
注入首request写延迟仍保序，54专项与477全研究tests/1skip通过。原失败日志保留。
local两轮300.005s/SSH0、remote终态归档完成。下一同749trace/.61+.004/B11328/
8moves/默认图保护同步唯一重试，不重选rate，不跳startup测试，不commit/push。

21:19更新：38files保护同步核验通过，唯一formal **241267**于21:17:50提交，
21:18:19在d1n41a18g03获配两A800，21:18:23 capacity RUNNING。local18677与
绑定18678 PPID1首轮实查、remote query-only2074913，每300s持续；runtime冻结。
同749trace/B11328/原rate/graph，尚无本轮capacity或正式收益结果，不重复申请。

**长校准通过，但首次formal失败；四策略收益尚未完成。** 241086于20:34:03
FAILED/1:0，118rawfiles及out/err全SHA归档通过，独立capacity/calibration通过。
21:06:53用户队列空，无重复allocation。capacity仍四pool101675+64分配rollback/
四rank sparse graph，B11328MiB/GPU、Conv target4347MiB，不是整卡OOM。

| Conv RPS | Conv mean queue s | sample Conv MiB | drain s | monitor最大gap s |
| --- | --- | --- | --- | --- |
| .42 | 1.642 | 3121.09 | 7.03 | 16.83 |
| .51 | 1.238 | 3588.80 | 5.17 | 12.69 |
| **.61** | **1.308** | **4350.77** | **14.00** | **16.01** |
| .70 | 2.096 | 5448.44 | 11.49 | 18.75 |

每档1200s完整drain，稳定队列条件均合格，压力规则选中`.61/.004 RPS`；freeze
1200s/seed20261021 **749请求（4Code/745Conv）**，保留171.57/172.49s双Code突发。
`frozen/trace.json` SHA256为
`6d43e314ddf3b7a64c883dc19bcdefb1210bb655b8c8d48c5af56580d4f1c006`。
校准监测21/27/26/37转态及空窗保留，不凭有限采样称稳态；完整生命周期/control
诊断在`evidence/agentic-trace-run-20261021/calibration-control-241086.json`。

formal顺序为own_model→no_move→kv_only→ilp。首own_model在首Code到达171.566s后
未准入，900s请求timeout触发；**640完成全部为Conv，0Code，426次blocked_staged_ilp，
507.806s累计控制时间，没有迁移/准入**。own_model当时无自身驻留请求，无法移动
foreign Conv，但人口变化触发反复暂停两tenant、solve、resume，拖慢Conv释放并使
Code饥饿。原始失败记录`formal-failure-241086.json`，不能把它当自然no_move排队
或ILP收益结论，也不增加timeout掩盖。global够/local堵最后样本不是CUDA OOM。

重试最小控制修复：在锁外真实observation上重建同page-owner保守候选，pinned页/
非全部owner可选的页不计；可释放页均分给owner，每source取最多max_moves份额之和
作**乐观释放上界**。若该上界仍不够page-rounded new KV，则无需暂停/ILP；own_model
无自身候选自然拒绝。目的仅避免必失败规划，不是实际allocator不可行证明，destination
copy成本未计入，允许的动作仍在冻结epoch内重新验证。每tenant下一规划时间为
`now+max(2s,4*last_planning_elapsed)`；冷却只限制重规划，direct、FIFO、取消和另一
tenant继续服务，全部真实等待计入指标。这不是完整wait-aware最优或生产化scheduler。

离线旧failure snapshot重放首例需3条Conv迁移、后例需6条；原max_moves2无法实现。
formal四策略统一`candidate_limit=8/model/rank,max_moves=8`，own_model仍只能迁自身。
三snapshot在11/16/16候选内求到最优，耗时.488/.261/.297s，仅受限保守ILP可行性，
不是所有请求全集的全局最优、真实GPUcopy或收益；见`migration-feasibility-replay-241086.json`。

下一唯一A800formal launcher复用原trace和rate-decision双SHA、固定B11328，先重新
capacity验证，若新capacity<B则明确失败，不改长度/图/rate/预算/门槛，成功后直接四
策略，不重选校准。25专项及全研究476tests/1skip；初次fixture遗漏source字段4失败
后补齐，旧失败日志保留。完整formal尚未重试/完成，未commit/push。

监控：local19:21→20:06实际gap2708.55s，20:06/20:11 SSH255，20:16恢复；remote
query-only已归档并独立核验24轮18:40:48→20:36:01，maxgap303s。remote连续不能
替代本地连续声明；证据`remote-watch-continuity-241086.json`，旧watcher终态退出。

### 先前运行记录

241086 .42首个1200s校准完整关闭，18files前缀全SHA/独立核验通过；Conv meanqueue
1.642s，sample3121.09MiB欠填，逻辑生命周期时间mean3435.90MiB，2Code容量机会。
monitor21转态/maxgap16.83s完整保留，536sample不当稳态证书。其余三档继续，
不从前缀freeze；证据`calibration-prefix-241086-1913/`及其`-analysis.json`。
每300s持续query正常，运行版冻结，正式四策略尚未执行。

241086关闭capacity20files全SHA及独立verifier通过：四pool101675+64分配rollback/
5pilot/四rank sparse图，B11328MiB/GPU、target4347，20稳定内存样本。证据
`capacity-pilot-241086/`/`capacity-verification-241086.json`。18:51:03已进入1200s
四档长校准，三轮300s query正常，runtime冻结；正式四策略收益仍待完成。

241086已18:44:42在d1n41a21g03获配两A800，18:46:03 capacity RUNNING。两轮
300s query正常，local922/923与remote4130135持续；runtime冻结，新长校准尚未
完成，获配不是正式收益通过。继续同job，容量后独立核验/完整校准合格才formal。

35files保护同步通过，唯一241086于18:40:48提交两A800，18:41:04 PENDING/Priority。
local922/绑定923与remote4130135持续300s，终态自动SHA归档/独立核验；最终
459CPUtests/1skip通过。四档1200s长校准/原阈值与图配置，尚无新GPU结果；完整
校准合格才同allocation四策略，不重复申请/覆盖运行版/commit/push/Aegaeon。

241036于18:28:44 FAILED/1:0，四档488请求完整drain，92raw/out/err全SHA与独立
capacity/calibration通过，freeze仍拒绝，formal未跑。七轮300s查询正常，终态自动
归档成功，18:32:25队列空；不继续依据300s混合压力局部插值找正结果。

| Conv RPS | 完成请求 | Conv meanqueue s | 驻留sample/time MiB | drain s | Code容量机会 |
| --- | --- | --- | --- | --- | --- |
| .33 | 111 | 1.327 | 5441.43 / 5962.75 | 7.57 | 4 |
| .35 | 121 | 4.839 | 3555.32 / 3508.35 | 11.06 | 0 |
| .37 | 124 | 4.483 | 3727.32 / 3774.66 | 7.85 | 0 |
| .39 | 132 | 5.486 | 3727.85 / 3801.24 | 9.40 | 0 |

非单调与真实双Code时序路径一致：.33第二Code等前条结束后229.11s才reservation，
其他三档173–175s就与第一Code重叠。.33压力高、其他欠填，不能把全局rate→KV
假设为线性；sample/time差异也非主要充分解释。300s原pressure100–300s仅含
171.57/172.49s双Code，串行时两prefill覆盖约100s；这会把短窗口压力放大且路径
敏感。600s同seed仍仅2Code，1200s才有4Code（754.83/927.37s又到达），不改seed
或抑制突发。下一各档校准延至 **1200s**，与formal到达窗口一致；queue5s/drain90s/
growth10/pressure.95..1.25阈值保持，窗口比例规则保持。这是显式协议变更，旧
300s结果全部保留，不称稳态证书。

新速率只用已完整验证240982的固定arrival30–120s子集：选择最高已测rate.80，
82Conv全部于137.74s前完成、早于首Code171.57s，未按长度或完成速度筛请求。
精确fullreservation `E[b*S]=7181.851MiB*s`，清洁低干扰估计`4347/E[b*S]=.60528`，
固定乘数.70/.85/1/1.15得 **Conv .42/.51/.61/.70**。该clean估计不代表mixed
可持续率；必须重新完整1200s四档校准，合格才formal。Code.004/seed/runtime/预算
公式/BF16/100K/默认sparse不变，来源/原始所有clean候选/SHA在
`long-window-rate-240982.json`，27专项通过。此前局部压力插值在本轮已被证伪为
可靠选档方式，不重复它或放宽guardrails。


241036关闭capacity20files全SHA与独立verifier通过：四pool最长101675+64实际
分配rollback、5pilot/四rank默认sparse图、B11328MiB/GPU及target4347，19稳定/
0转态内存样本。新节点预算与细化参考一致，有限样本仍非峰值证书或收益。证据
`capacity-pilot-241036/`/`capacity-verification-241036.json`。开放环校准继续，
两轮300s query/本地watcher实查正常，运行版冻结，formal未启动。

最终35files保护同步通过，唯一241036于18:00:58提交，18:01:10已在d1n41a12g02
获配两A800，18:01:16 capacity RUNNING。local95592/绑定95593与remote3587855
持续300s，运行版冻结；新Conv.33/.35/.37/.39、Code.004与全部门槛不变。容量后
完整校准合格才自动freeze正式四策略，尚无新网格GPU结果，不称收益完成。

18:00实现核验：最终458CPUtests/1skip通过，13校准源SHA一致。启动旧档位assert
与新fixture import失败分别保留，更新为核验新网格的测试后全回归通过；初次保护
同步后fixture又修正，提交前再次保护同步。不将失败测试或旧同步冒充最终通过。

240982于17:46:41 FAILED/1:0，四档696请求完整drain；92raw/out/err全SHA及独立
capacity/calibration通过，freeze拒绝，正式四策略未执行。compute启动177/193tests
通过，10轮300s query正常，终态自动归档/核验完成，不以旧progress248/249否认完整
raw249/249 drain。worker退出未确认socket保留warning如实保存，未全局清理。

| Conv RPS | 完成请求 | Conv meanqueue s | 驻留sample/time MiB | drain s | 外部queue增长 | Code容量机会 |
| --- | --- | --- | --- | --- | --- | --- |
| .30 | 104 | 4.357 | 2765.71 / 2778.56 | 10.94 | -.045 | 0 |
| .45 | 147 | 4.426 | 7757.06 / 8334.28 | 8.37 | 7.292 | 1 |
| .60 | 196 | 5.512 | 7712.78 / 8507.04 | 22.65 | 16.756 | 5 |
| .80 | 249 | 3.904 | 7241.88 / 7719.66 | 57.41 | 13.727 | 24 |

保持原queue5s/drain90s/growth10/pressure.95..1.25门槛。并发路径后.30与.45均
queue/drain/growth合格，但前者欠填后者超填；同节点同B与240886 .45 queue8.950
相比本轮4.426下降，但单次运行非严格配对因果/算法净收益。三种修复至此停止盲改
控制，仅按完整no_move校准的局部压力插值细化Conv档位：
`r*=.30+(4347-2765.714)/ (7757.06-2765.714)*(.45-.30)=.34752`，步长`.15/8`
与偏移`[-1,0,1,2]`，四档四舍五入为 **.33/.35/.37/.39**。完整其他候选仍保存，
Code.004/seed/data/runtime/预算公式/guardrails不变，39专项通过。不是假定全局
KV与rate线性、不是从策略收益挑rate；新完整开放环验证合格才冻结1200s四策略。
来源/SHA/候选在`conv-pressure-bracket-240982.json`，诊断在
`calibration-control-analysis-240982.json`。monitor转态3/7/12/9、gap5.88/17.05/
11.78/8.78s均保留；time积分仍非actual allocator release或physical mapped。


仅本地补充请求reservation→client completion逻辑KV积分核验，451CPUtests/1skip
通过（16专项），未同步活动runtime或修改guardrails。对旧240886 .45，同pressure
窗口sample mean5293.20MiB，生命周期时间均值4814.62MiB，差478.58MiB；.60差
-1192.68MiB。空窗/转态确有诊断必要，但逻辑积分终点不是allocator释放，差值不
作纯采样因果结论。旧三轮全raw先独立验证再离线分析，证据
`calibration-residency-analysis-240679/240782/240886.json`（各自独立文件）。

17:39两档关闭前缀30files全SHA及独立verifier通过，无终态伪造：.30/.45 Conv
queue4.357/4.426s，均低于5s；sample驻留2765.71/7757.06MiB却分别欠填/超出原
目标窗口。并发派发确实让Conv可在第二条Code等待期间继续打入，但本轮前缀不能
当总体通过：.45的Conv service均值28.27s，170–245s到达子集62.42s，存在真实
共址计算干扰和容量等待；Code两条先后reservation172.37/228.05s。不把它解释为
纯CPU消除或总体收益。证据`calibration-prefix-240982-1738/`/其`-analysis.json`，
剩余两档继续执行，不改active配置/guardrails，不从前缀freeze。

17:31:38第七轮query正常，240982仍RUNNING，校准.30已关闭，.45进行中；前缀
只有sample驻留尚无完整guardrail结论，不能先称控制修复或收益通过。

240982关闭capacity20files已全SHA归档并独立核验通过：四pool最长101675+64完整
分配/rollback、5pilot与四rank默认sparse graph通过，B11328MiB/GPU、Conv目标
4347MiB，19稳定/0转态内存样本。有限采样不是峰值容量证书或算法收益。初次本地
helper import误址及shell只读变量报错保留，复用原下载包全SHA核验后归档，未动
远端；证据`capacity-pilot-archive-240982.json`/`capacity-verification-240982.json`。

240982已于17:18:37在d1n41a21g03获配两A800，17:23:16 phase为开放环校准。
关闭capacity阶段正独立SHA归档/核验，不能把阶段成功当四策略收益。六轮五分钟
query正常、local87298/绑定87299及remote2854849持续；保持运行版冻结与全部
rate/seed/guardrails。新并发路径尚无完整GPU校准，formal未启动，不重复申请。

17:02执行更新：446CPUtests/1skip、35files保护同步通过，唯一240982于17:01:14
提交两A800，17:01:40 PENDING/Priority。local87298/绑定87299与remote2854849均
PPID1/首轮query实查，300s持续，运行版冻结。保持原rate/seed/guardrail，仅验证
两tenant独立FIFO及锁外RPC预算epoch修复；尚无该路径GPU结果，不称收益完成。

240886于16:47:55 FAILED/1:0，四档696请求完整drain，92raw/out/err全SHA及独立
capacity/calibration通过；freeze压力guardrail拒绝，正式四策略未跑。

| Conv RPS | 完成请求 | Conv meanqueue s | 驻留MiB | drain s | 逻辑Code容量机会 |
| --- | --- | --- | --- | --- | --- |
| .30 | 104 | 4.019 | 2878.13 | 12.26 | 0 |
| .45 | 147 | 8.950 | 5293.20 | 3.78 | 4 |
| .60 | 196 | 8.695 | 5800.81 | 26.30 | 7 |
| .80 | 249 | 14.487 | 8189.94 | 35.29 | 1 |

rank-local/monitor修复减少.45 Conv控制累计33.72→14.58s，但queue未降，不能宣布
总体有效；节点和B也不同，不当严格配对归因。该档monitor9转态/最大gap20.56s；
四档5/9/14/17转态及最长59.50s gap全保留，未知区间不是零KV/queue。独立覆盖诊断
在`calibration-control-analysis-240886.json`，完整结果`calibration-verification-240886.json`。
旧240886 A800容量B11328/target4347实测保持；.45压力合格但queue失败，不能放宽5s。

下一根因候选是cached-blocked全rank检查持全局锁，以及单dispatcher串行等待Code
100K RPC挡住另tenant。保持全部rate/seed/guardrail/预算公式/数据/graph，仅独立
两tenant FIFO dispatcher、锁外observations、短锁内generation复核/最新foreign
保守账本合并再实际提交。安全测试证明Code观察等待时Conv可准入、同tenant不乱序，
foreign增长后重算预算、取消释放锁、失败/迁移失效；446CPUtests/1skip通过。
尚未GPU验证并发修复，准备保护同步唯一A800重试，不commit/push/Aegaeon。

16:26容量更新：240886进入rank-local开放环校准，关闭capacity20files SHA及独立
verifier通过，最长101675+64四pool真实分配rollback、5pilot、四rank默认sparse图。
实测B11328MiB/GPU、Conv目标4347MiB，19稳定/0转态内存样本；有限容量证据非峰值
证书或算法收益。证据`capacity-pilot-240886/`及`capacity-verification-240886.json`。
12轮300s query正常、runtime冻结，新fastpath完整校准/四策略仍待执行与独立核验。

16:22执行更新：240886已16:19:33获配d1n41a21g03两A800，16:21:18 capacity
RUNNING/加载第二tenant权重；compute171/193启动tests通过。local74933/74934与
remote1717423持续实查，11轮300s query正常，runtime冻结。新rank-local/monitor
尚未GPU校准；capacity后独立核验，同rate继续校准与合格四策略，不把获配称收益。

15:57资源更新：240886仍PENDING/Priority，scontrol暂估18:24:59（非获配保证），
抽查三A800节点各8/8分配。保持唯一A800申请/冻结runtime，六轮五分钟SSH正常，
不因等待改GPU/缩输入/图或重复申请。尚无该修复GPU校准，研究收益目标未完成。

15:46监控更新：240886仍PENDING/Priority，15:45:04四轮query成功，local74933/74934
PPID1持续，gap约300.004s。仅本地增加monitor coverage诊断，报告窗口边界/最大
采样空窗和转态数量，不以缺失样本当零KV/queue，440CPUtests/1skip通过；未覆盖
冻结remote。新fastpath尚无GPU结果，继续同job，不把排队视为收益完成。

15:31执行更新：最终437CPUtests/1skip、35files保护同步通过，唯一240886于15:29:42
提交，15:30:04 PENDING/Priority。local74933/绑定74934及remote1717423均PPID1/
首轮query实查，300s持续至终态。保持Code.004/Conv四档/seed/guardrail，runtime冻结；
尚无新fastpath GPU结果或正式四策略收益，不commit/push，不执行Aegaeon。

240782于15:13:10 FAILED/1:0，四档696请求完整drain；89raw/out/err全SHA、独立
capacity/calibration通过，失败是freeze压力guardrail，正式四策略仍未执行。

| Conv RPS | 完成请求 | Conv meanqueue s | 驻留MiB | drain s | 逻辑Code容量机会 |
| --- | --- | --- | --- | --- | --- |
| .30 | 104 | 4.749 | 2880.82 | 10.93 | 0 |
| .45 | 147 | 8.717 | 5165.22 | 5.19 | 3 |
| .60 | 196 | 13.002 | 5848.90 | 26.36 | 6 |
| .80 | 249 | 7.327 | 8273.09 | 35.97 | 5 |

保持Code.004的实测负结果，不继续调rate追正；.45在pressure窗口内但queue>5s。
完整raw`live-trace-240782/`、`calibration-verification-240782.json`、
`calibration-control-analysis-240782.json`及3/4关闭前缀归档均保留。
Conv前170s准入queue约.13s，双Code后170–245s约30.32s；.45 Codeprefill重叠的
9次控制均值2.370s，其余136次.091s，总33.72s。真实global不足与控制等待并存，
不能把全部queue归因CPU，也不能用降offered-rate宣称算法收益。

下一轮同rate/seed/guardrail/budget/graph，只启用保守rank-local admission ledger，
direct before查询arriving tenant两rank，其他tenant为无独立mapped增长下的上界；
blocked才全rank刷新。monitor全rankRPC不持控制锁，前后admission generation一致
才入population，跨epoch完整raw单存transitions。所有策略共同使用，失败/retry/迁移
失效账本，实际copy峰值/真实allocator仍核验。专项56tests与完整435tests/1skip通过，
新增安全补测后保护同步唯一A800重试；未GPU验证新fastpath，不commit/push/Aegaeon。

14:52进度优先：240782于14:45:56获配d1n41a18g02两A800，14:50:54已开放环校准。
capacity关闭阶段21files SHA及独立verifier通过，最长101675+64四poolallocation/
rollback、5pilot及四rank sparse graph；B11424MiB/GPU、Conv目标4539MiB，19稳定
ownership样本与1个转态样本完整保留，转态不作为异时刻free+mapped预算。
证据`capacity-pilot-240782/`、`capacity-verification-240782.json`。
local68219/68220跨命令PPID1/querygap300.006s实查，运行版冻结；Code.004新校准
尚未完整，正式四策略没跑，不能把重复100K容量通过当算法收益完成。

14:46执行更新：428CPUtests/1skip、35files保护同步通过；唯一240782于14:45:32
提交A800，14:45:53 PENDING/Priority。local68219/绑定68220与remote1148689均
PPID1/query实查，300s监控至终态。runtime冻结，未commit/push或运行Aegaeon。
Code.004的新固定seed仍在171.57/172.49s两次相邻到达，不是消除burst；正式1200s
预计4Code，作为有限初测，不能宣称多seed/多rate泛化。完整新校准与四策略尚未执行。

240679于14:29:10 FAILED/1:0，四档704请求完整drain，88raw及out/err全SHA自动归档，
capacity/calibration独立核验通过；失败在freeze压力guardrail，不是engine/allocator崩溃。
只有.30通过queue/drain门槛但驻留欠填，因此不能冻结正式trace；四策略未执行。

| Conv RPS | 完成请求 | Conv meanqueue s | 驻留MiB | drain s | 逻辑Code容量机会 |
| --- | --- | --- | --- | --- | --- |
| .30 | 106 | 3.219 | 3025.80 | 52.08 | 3 |
| .45 | 149 | 6.854 | 4778.48 | 55.66 | 4 |
| .60 | 198 | 18.098 | 12239.01 | 57.76 | 0 |
| .80 | 251 | 25.686 | 14623.25 | 67.70 | 9 |

target-only确认确实减少控制，.45 Conv control57.73s vs旧109.86s，但目标档queue仍
超过5s。.013Code源于H200校准，A800四档全部16Code的精确reservation→completion
驻留均值56.562s，预计人口lambda*S=.735而非原先预声明.25。52.8/53.1s相邻Code
约60s重叠，保留真实双100K容量竞争；不能把所有queue都归因CPU/RPC。

下一轮只按既有`min(.02,.25/E[S_code])`规则重算：raw .004419917，向下millirps取
Code.004，预测人口.22625。保留Conv.30/.45/.60/.80、原seed20261021、300s/1200s、
完整数据/100K/BF16/128K/sparse/32step、A800实测预算公式及queue5s/drain90s/pressure
门槛。不是选择正收益rate、抑制Poisson相邻到达或放宽SLO；独立新校准合格才能formal。
旧16Code与所有失败不改。有限样本Little人口估计不是稳态保证，新rate效果尚未GPU验证。

证据`calibration-verification-240679.json`、`calibration-control-analysis-240679.json`、
`code-residency-rate-240679.json`、`code-residency-rate-trace-counts-240679.json`；都在
`evidence/agentic-trace-run-20261021/`。独立rate工具拒绝非精确时间/零rate，CLI及
launcher回归通过；在用户队列空时保护同步唯一A800重试，不commit/push/Aegaeon。

### 14:20：关闭的两档校准前缀独立复核

240679仍校准第三档，前三阶段不是正式收益。已byte冻结只含完整calibration0/1的
summary，28files SHA归档及source/trace/journal/token数量/ownership/graph/指标核验
通过；不伪造exit marker，partial永不freeze。新离线control analyzer及active-prefix
回归通过，完整研究回归415通过/1skip（active-prefix额外专项19通过），未同步活动源码。

| Conv RPS | 完成请求 | Conv meanqueue s | 驻留MiB | drain s | 逻辑容量机会样本 |
| --- | --- | --- | --- | --- | --- |
| .30 | 106 | 3.219 | 3025.80 | 52.08 | 3 |
| .45 | 149 | 6.854 | 4778.48 | 55.66 | 4 |

相比240575的9.867/9.209s排队，target-only确认改善，但.30欠填、.45仍queue>5s，
不能从前缀选rate或放宽门槛。.45 Conv control累计57.73s，拆为提交前51.52s、后6.21s；
旧同档提交后80%以上控制已减少，但前提交四rank观测仍会等100K kernel边界。与Code
prefill重叠的21条Conv控制均值2.066s，其余124条.116s；不是纯CPU/GPU kernel计时。
52.8/53.1s两Code重叠仍完整保留，Conv到达50–125s queue均值18.74s，说明准入串行
控制与真实容量等待都可能贡献，不能仅凭重叠相关性作因果归因。四档完整后再定下一修复。

证据`calibration-prefix-240679-1419/`与`calibration-prefix-control-240679-1419.json`；
旧完整校准诊断`calibration-control-analysis-240575.json`不改变原raw/guardrail负结果。

240679在d1n41a18g02两A800继续开放环校准；14:09:10实查74/89 offered，未终态。
关闭的capacity阶段20文件SHA归档及独立核验通过：真实最长101675+64 tokens四pool
allocation/rollback、5pilot请求与四rank默认sparse graph。受控预算11424MiB/GPU，
Conv背景目标4539MiB，19个ownership一致的内存样本，0个转态样本；只证明有限采样
容量及100K执行，不是峰值安全证书或算法收益。pilot导出的低rate bracket仍保留为
初始估计；实际校准显式使用.30/.45/.60/.80、Code.013，seed与guardrail不变。

证据`capacity-pilot-240679/`、`capacity-240679.sha256`及
`capacity-verification-240679.json`，均在`evidence/agentic-trace-run-20261021/`。
local60906/60907持续PPID1，两次实际querygap300.00044/300.00079s；不重复申请或
覆盖活动源码。下一步同allocation等待完整校准，合格自动接续四策略，失败先归档诊断。

**13:59运行进度优先**：407CPUtests/1skip、31文件采样一致性保护同步通过，唯一
240679于13:58:48提交，13:59:05在d1n41a18g02获配两A800，13:59:28 capacity
RUNNING。local60906/绑定60907、remote632629均PPID1/query实查，300s监控；
runtime冻结，所有科学设置不变，尚无该作业capacity/正式收益通过。

**13:56终态优先**：240654于13:52:14 FAILED，24raw与out/err全SHA自动归档。
真实5pilot请求（Code101675+64、4Conv659输出）完成，最长四poolprobes已记录，但
最终capacity-setting拒绝、未生成passed summary，不能按完整capacity/graph verifier
称通过，RPC校准未执行。sample18先读free3366MiB，后快照Code完成释放mapped到
192MiB，混合时刻算headroom1934.945MiB；其余最低11318.07MiB。不是观测到真实OOM。

修为memory查询前后两份ownership快照相同才计算free+mapped；转态raw保存在
memory-transitions.json，不作为一致headroom也不删除，所有稳定低free仍计预算。
独立verifier检查before/after相同及mapped几何。原24raw不改、不事后制造passed
summary；下一唯一A800重试保持原rate/guardrail/B公式/default graph并核采样一致性。

**13:48运行进度优先**：405CPUtests/1skip与31文件保护同步通过，唯一240654于
13:47:48提交，13:48:04获配d1n41a18g02两A800，13:48:28 capacity RUNNING。
仅合并提交后目标rank确认，不改.30/.45/.60/.80rates/Code.013/seed或guardrails；
local59349/绑定59350与remote522963 query/PPID1实查，runtime冻结，每300s到终态。
尚无该作业100K容量/正式trace结果，不将再次提交当完成。

**13:44终态优先更新**：240575于13:32:07 FAILED/1:0，四档704请求全部完成，但
freeze guardrail拒绝，正式四策略没跑。88raw与out/err全SHA自动归档；capacity与完整
calibration独立复核通过，terminal状态archived_without_complete_formal_trace，不是收益。

| Conv RPS | 完成请求 | Conv meanqueue s | drain s | Conv驻留MiB | 逻辑Code容量机会 |
| --- | --- | --- | --- | --- | --- |
| .30 | 106 | 9.867 | 60.163 | 2371.20 | 0 |
| .45 | 149 | 9.209 | 50.869 | 4729.70 | 4 |
| .60 | 198 | 20.503 | 61.942 | 11336.45 | 2 |
| .80 | 251 | 31.767 | 64.185 | 15687.42 | 3 |

全部meanConvqueue>5s，所以不能选.45直接运行；该档驻留落入目标窗口不是收益证明。
所有内部reservation_pending计数0，精确scheduler reserved_at完整，原致命reservation
拒绝未复现。seed固定的两个Code在52.8/53.1s近同时到达、混合service约59–63s，
重叠时双卡Code逻辑KV约18.6GiB压缩Conv空间；不是稳定单Code背景保证。
.30/.45的control累计138.36/113.93s，前提交39.81/29.20s、后提交98.55/84.73s
（由记录时序拆分，含kernel/RPC/真实等待，不当纯CPU成本）。Conv无Code区间准入约
.12s，Code期间重复四rank utility RPC等100K prefill kernel步边界，放大串行控制。

下一最小修复只把提交后的reservation确认改为目标rank RPC，并复用该实际snapshot
做budget核验；同控制锁下其他rank无自主growth，其pre-admission mapped为保守上界。
仍全rank before、monitor和迁移观测，保持同rate/seed/queue5s/drain90s/graph/B规则，
不通过改SLO、Code到达或请求子集追正。待CPU回归/保护同步唯一A800重试，尚未GPU
验证该确认优化。证据calibration-verification-240575.json与failure-breakdown保留。

**13:08进度优先更新**：唯一240575在d1n41a22g03两A800执行，容量phase20rawfiles
SHA归档及独立verifier通过，最长101675+64四pool临时分配/rollback、5请求及四rank
默认sparse graph通过。B11328MiB/GPU、Conv目标4347MiB，23有限memorysamples；
13:06:32进入开放环校准，固定Conv.30/.45/.60/.80、Code.013，尚无完整校准/正式
四策略结果。证据`capacity-verification-240575.json`/`capacity-pilot-240575/`，整个job
仍运行，未归档终态，不将此阶段当算法收益。querygap实测300.002s，runtime冻结。

集群实查用户队列空。239658于04:41:59在d1n41a16g01获配两A800-SXM4-80GB，
05:07:54 FAILED/1:0，运行25m55s。**真实100K容量pilot已通过，但正式四策略尚未
执行，不能报告本轮ILP收益。** 后文旧状态保留为历史记录，本节优先。

### A800容量独立核验

两独立BF16 Qwen tenant保持DP2/EP2/attentionTP1、128K总长度上限、24GiB/model/rank
虚拟pool、默认sparse graph及32step pause。四pool最长Code101675+64 token reservation
与rollback、四rank FULL decode graph、5个pilot请求均通过。实测选定受控预算
11424MiB/GPU，最长Code逻辑KV9538.5MiB，Conv背景目标4539MiB；allocator utilization
统一.98。23个有限内存样本支持本次容量验证，不是所有峰值安全保证或算法收益。

证据`evidence/agentic-trace-run-20261021/capacity-verification-239658.json`；
`live-trace-239658/`的80个rawfiles已按`raw-239658.sha256`逐项复核通过。
自动归档在raw下载成功后下载out时SSH超时，`terminal-processing-239658.json`保留失败；
out/err已手动补齐，但不能声称自动终态流程全通过。

### 开放环校准与失败根因

| Conv RPS | 完成/提供请求 | 平均采样Conv reservation | 逻辑global够/local blocked Code样本 |
| --- | --- | --- | --- |
| .08 | 30/30 | 479.51MiB | 0 |
| .12 | 43/43 | 811.39MiB | 0 |
| .18 | 65/65 | 1410.23MiB | 5 |
| .24 | 未完整完成 | 不据截尾结果选rate | 未完成核验 |

前三档背景远低于4539MiB目标，尚不能freeze正式trace。.18的5个监视样本只是逻辑容量
机会，不是实际迁移成功或性能收益。第四档旧`PlacementScheduler.add_request`遇到
full sequence reservation返回None即抛`MemoryError("Full sequence reservation failed")`，
导致EngineCore死亡；这是应排队的allocator拒绝被当作致命错误，不是已观察到CUDA OOM。
初始rate来自单次pilot Conv service67.958s，不能视为已证明的稳态服务率。

本地已添加full reservation失败排队/后续重试、controller实际free-block预检和
reservation_pending识别；388tests通过/1skip，仅mock/CPU验证，尚未同步或compute验证。
下一步先处理准入等待timeout、控制锁及真实reservation完成时间记账，避免内部KV等待
被误报为准入成功/queue收益，再预声明新的开放环rate范围，保护同步唯一A800重试。
保留queue/drain/驻留guardrails，不按策略收益挑rate，不截短100K或关闭默认graph。
监控本地长gap/SSH异常如实保留；当前无活动作业或watcher，未commit/push，Aegaeon停止。

### 12:48后续修复与预声明（尚未compute验证）

full reservation失败留在native waiting，不做partial prefill；保留两原生队列相对顺序。
受控trace仅由controller跨模型预算锁显式重试，避免scheduler自行重试绕过两tenant
总mapped上限。等待释放控制锁，另一模型可继续；同tenant FIFO头等待时不重复generate
或提交后继。decision区分已提交与已预留，queue改为精确scheduler reserved_at-arrived_at，
population/freeze guardrail包含内部capacity_pending，未改变整体900s超时或SLO。
新增CPU/mock并发/FIFO/暂停/driver错误/指标独立复核，完整401tests通过/1skip。
仍待同步和真实compute验证，不能把mock通过当GPU收益。

下一校准Conv .30/.45/.60/.80、Code .013RPS、每档300s，依据前三完整欠填结果预声明，
保持原guardrail、重新实测B和Conv目标，合格才自动freeze1200s四策略。239658归档恢复
已实查80rawfiles及out/err的SHA通过，`archive-239658.json`生成；原自动失败日志保留，
`terminal-processing-239658.json`仍为旧失败记录，不改写成全流程成功。

12:52运维：31文件保护同步成功，唯一两A800作业240567于12:51:15提交，12:51:37
PENDING/Priority。本地51928/绑定caffeinate51929、remote query-only4104383均PPID1
实查并已有首轮query；300s监控，runtime冻结。获配自动重做容量→四档开放环校准→
guardrail冻结→1200s四策略；尚无新GPU结果，不将PENDING或后台monitor称收益完成。

12:54离线独立复核：新增显式`--allow-partial`只验证失败校准的完整前缀，239658前三档
全部source/trace/hash/request/journal/ownership/graph/指标复核通过；3/4，不允许freeze。
Conv meanqueue分别1.151/4.370/2.503s，drain42.458/41.391/56.820s；.18确有5个逻辑
global够/local blocked Code监视样本，但尚无迁移/算法收益。证据
`calibration-prefix-verification-239658.json`，原始文件不改。该verifier/test仅本地更新，
未覆盖活动remote；新作业源码仍为12:50保护同步的冻结版本。

### 12:58：240567初始化资源冲突

240567于12:51:43在d1n41a14g01获配两A800，12:53:05 FAILED/1:0；162 placement
mock/runtime tests与193 integration tests在compute通过。首tenant权重加载时真实
CUDA OOM，raw列GPU0既有PID4174603占69.53GiB、本worker8.83GiB，申请192MiB失败。
不是239658的full-reservation拒绝，也未开始100K或新排队推理，不能声称排队修复GPU
成功。既有PID归属尚未证实，不触碰/杀进程；用户队列12:57:52实查空。

12rawfiles及out/err全SHA自动归档成功，`archive-240567.json`与terminal记录保留。
本地monitor两轮实际gap300.020s，远端亦见终态，旧watcher已退出。下一唯一A800重试
临时避开该节点，启动前记录Slurm/CUDA绑定、UUID/free/compute-apps，两卡各至少
70000MiB free，否则明确资源预检失败；保持BF16/128K/100K/default sparse和原校准。
不通过删进程/缩短请求/改graph消除资源冲突。

13:02运维：403CPUtests/1skip、31文件保护同步通过，唯一替代240575于13:01:18提交，
仅exclude d1n41a14g01。13:01:43在d1n41a22g03获配两A800，13:01:56 capacity pilot
RUNNING；启动前两卡各81153MiB free、compute-apps空，Slurm物理2,3与容器CUDA0,1
映射证据保留，不凭数字不同推断绑定错误。local53329/绑定53330、remote25927均PPID1
query实查，300s继续；冻结源码，不覆盖，尚无本轮100K或四策略收益结果。

2026-10-21 20:54，Asia/Shanghai。协议与RPS计算在
`../docs/agentic-trace-protocol.md`；不是已完成的四策略收益结果。

## 已验证的容量与长度

238436在两H200上实际执行128K/DP2/EP2/attentionTP1，两独立Qwen共址、24GiB/model/rank
虚拟pool、20GiB/GPU受控mapped预算、默认sparse captures/32step pause保持。100K真实
SWE-smith历史token IDs通过运行时模板校验，四pool101371+64临时分配/rollback通过，
目标卡峰值9.46875GiB；四rank真实FULL decode graph。单Code TTFT8.575/service9.168s，
混合有限人口16/32/48下Conv service17.450/24.235/31.093s，Code均值19.011s。

## 首正式trace失败，不能比较ILP收益

根据有限人口拟合冻结Conv1.65/Code.013RPS、1200s/seed20261021、2025请求
（14Code/2011Conv）。作业238436于20:06:50 FAILED/1:0，首own_model请求timeout900s；
其他三策略尚未运行。41原始文件SHA归档完整，终态实际requests记录657条
（Code14全部完成、Conv643），progress最后一次656是较早的监视样本，不是丢请求。
未完成的1368请求保留在trace，不以完成子集评估吞吐/成功率。

658准入决策全部direct_fastpath、没有迁移/ILP尝试。800–1200s平均active2.17，
aggregate mapped仅3.54GiB，远小于40GiB双卡预算。因此长期排队不是KV容量满导致。
原热路径每次准入都在事件循环同步重序列化/覆盖全部历史decision；末文件74.43MiB，
100/300/658条累计文件约21.9/44.1/74.4MiB，记录工作量随请求数近似二次增长。
准入间隔从前100条均值.86s恶化到500–600条2.65s；直接控制总计312.40s。
逐请求流输出/完整快照同样累计重写，加剧事件循环阻塞。完成Conv平均service8.27s
仅是受此准入瓶颈影响且截尾的子集，不可据此声称完整Poisson稳定服务率。

## 修复与预声明重试

所有策略统一改为逐事件JSONL追加（后台线程），终态再写一次兼容JSON数组，仍保留
完整token/cadence/页owner证据。四rank snapshot RPC并发，准入后同一次观测验证容量，
去掉冗余四rank查询，不改变预算、count-LB、ILP候选或graph。新增监视样本记录
分模型logical reservation与pending head global-enough/local-blocked，区分准入控制
瓶颈与物理mapped容量；logical仍不是实际mapped。

唯一H200重试allocation先跑开放环Poisson校准Conv1.65/2.0/2.35RPS、Code.013RPS，
每档300s、seed20261021，不依据策略收益筛rate。目标Conv驻留23GiB，以100–300s
背景/queue和Code容量事件核查，完整drain后再冻结正式1200s四策略。若准入仍瓶颈
或此范围不能达到目标，保留校准并进一步诊断，不能仅加timeout伪造成功。

249本地研究tests通过/1跳过；修复尚待compute验证。gjghfd fork/main，无commit/push，
Aegaeon未启动或修改。原本地监控19:54→20:34有2396.9s实际查询缺口（原因未确定），
不声称连续每5min；本轮增加登录节点仅query的有界remote watcher，仍由主agent分析。

原始证据：`evidence/agentic-trace-run-20261021/`；完整终态包`live-trace-238436/`，
`failure-analysis-238436.json`与`archive-238436.json`保留。总体算法收益目标仍未完成。

20:58运维：21文件保护同步成功，唯一238639于20:57:21提交H200/2GPU/4h，
20:57:54 PENDING/Priority；本地与remote query-only五分钟watcher启动，未获配。

21:18后续：238639于21:12:17仍PENDING时取消，未执行。替代流程改为
1.65/3/4/5Conv、.013Code的四档300s开放环校准，全部10Code episode，无手工
30min空等；预声明稳定性/驻留guardrail自动选rate并冻结1200s四策略，未满足则
保留校准失败。细则只在协议末节，254本地tests通过/1skip，尚无新GPU结果。

21:22运维：24文件旧SHA验证/备份/原子同步/新SHA复核完成，唯一替代238698于
21:19:46提交H200/2GPU/4h，21:22:14仍PENDING/Priority。已启动本地300s watcher及
remote query-only watcher PID1772610；运行版本冻结，不覆盖活动源码。校准成功后
自动freeze并继续四策略，尚无新GPU校准或收益结果。同步证据
`autofreeze-result-20261021-211753.txt`，状态证据`poll-238698.log`，均在上述evidence目录。

21:32本地复核工具：`../verify_agentic_calibration.py`新增完整校准来源/trace重生成、
逐请求/journal/graph/ownership/预算/指标及guardrail独立核验；sampled容量flag不替代
真实allocator/copy probe。专项39tests通过，kvcached cwd研究+integration回归353通过/
1skip，未覆盖活动remote源码。外层cwd首次integration import失败日志保留，正确cwd
重跑通过。这是工具验证，238698尚未产生GPU校准结果。

21:48监控/归档：remote五分钟query继续，本地升级nohup PID9299，终态自动SHA归档、
独立calibration验证；只有完整正式trace才继续四策略验证/诊断。原始包等待remote
query watcher退出后manifest，避免日志追加竞态。最新356CPUtests通过/1skip，工具
仅本地更新、未覆盖活动runtime；238698仍PENDING，无新增GPU结果。

21:49更正：上述本地nohup PID9299未跨CLI命令存活，未完成实际query；恢复为长运行
exec session59048并实查日志。remote PID1772610独立query始终持续，21:48:58仍
PENDING。启动PID不等于守护已就绪，原空日志和查询证据保留；不虚称no-gap本地监控。

22:25状态：超过一小时实际300s查询，238698仍PENDING/Priority；H100/H200无空卡，
两健康H200节点均8/8占用。公共A800仅2空卡、无同节点两卡slot，不改变100K/BF16
配置盲切80GB。remote query PID1772610与local session59048/PID9693均实查持续；
证据`resource-check-238698-2225.txt`与`poll-238698.log`。尚无新GPU校准/四策略结果。

22:54状态：本地Python独立session PID24250/PPID1跨CLI命令及300s轮次存活确认，
22:48/22:53成功query，旧session59048退出，避免重复终态归档。remote PID1772610
持续query（61poll有界约5h）；本地日志`watcher-238698-detached.txt`。作业仍PENDING，
尚无GPU校准或新四策略比较，不将后台执行/工具回归当算法收益完成。

22:59:26终核：238698仍PENDING，remote PID1772610及本地PID24250/PPID1正常。
gjghfd origin/main与Aegaeon受保护diff SHA未变，未commit/push或覆盖活动源码。
后台继续300s监控、终态归档/复核；总体异构100K trace收益目标仍未完成。

23:53用户改A800：238698于23:46:19仍PENDING时取消，无GPU执行。改为A800容量
pilot→动态预算/Conv目标→四档开放环校准→冻结四策略，细则在协议最后一节。
保持100K真实输入/BF16/128K/sparse图，不把整卡OOM等同placement收益。87本地
专项tests通过/1skip，源码尚待保护同步与allocation验证；当前队列资源会变动。

2026-10-22 00:02（Asia/Shanghai）：20文件旧SHA备份/原子同步/新SHA核对通过，
唯一A800作业239082于00:01:55提交，00:02:32仍PENDING/Priority。此时公共A800
空闲0，未开始容量pilot，不引用历史空卡数保证获配。remote query-only PID24597
与本地Popen独立session PID42733/PPID1首轮查询实查，五分钟继续监控；终态自动
SHA归档并独立验证完整校准/trace。最新本地回归364通过/1skip。保持运行版本冻结，
没有commit/push或Aegaeon执行；仍无100K四策略净收益结果。

证据：`a800-sync-output.txt`、`a800-result-20261021-235736.txt`、
`a800-submit-20261022.txt`、`watcher-239082-process.json`与`poll-239082.log`，
均在沿用的`evidence/agentic-trace-run-20261021/`目录。

00:16诊断：排队期间新增独立capacity raw verifier，发现runtime page_mib是96.0，
预算计算返回11136.0并导出环境，下阶段`--kv-budget-mib`的argparse int会拒绝，
因此不能只以先前mock子串断言认为整pipeline可启动。239082于00:15:52仍PENDING
取消，零GPU执行。预算统一返回整数，加强float page/CLI解析回归；31专项通过。
首次测试patch cwd错误曾生成错误嵌套测试文件，已迁到正确tests目录并移除空目录；
失败日志`tests-a800-independent-capacity-wrong-cwd.txt`与原verifier失败日志均保留。
新增核验四pool最长Code实际分配/rollback、四rank graph、完整输入输出、内存最小
headroom与rate/env，不宣称有限样本是peak保证。准备唯一修复后A800重试。

旧H200238698取消证据已归档，1rawfile SHA通过，未开跑故无out/err；
`archive-238698.json`记录缺失原因，不伪造exit marker或GPU结果。

00:17修复回归：研究tests及mock integration TP合计380通过/1skip；额外尝试shutdown
测试在本地无torch环境collection失败，原日志保留，不称其通过；远端compute入口
继续运行其既有真实环境回归。预算整数/独立capacity专项31通过。

00:22运维：22文件旧SHA备份/原子同步/新SHA复核通过，唯一修复后A800239136于
00:20:43提交，首轮PENDING/Priority。本地PID47555/PPID1首轮查询实查与remote
query-only PID268654持续300s监控，终态先capacity独立核验，再校准/完整四策略
核验分析。239082取消包1rawfile SHA已核对，没有GPU执行故out/err不存在，
`archive-239082.json`与`terminal-processing-239082.json`保留。同步记录
`a800-result-20261022-001917.txt`，提交/监视记录`a800-integer-budget-submit-20261022.txt`
与`poll-239136.log`。运行版本冻结，无commit/push/Aegaeon；尚无新A800 GPU结果。

00:27查询：239136仍PENDING/Priority；本地PID47555/PPID1成功间隔300.009s，
remote query-only PID268654/PPID1也跨查询存活。Slurm暂估05:35:36，仅估计非保证；
实际A800 mixed/comp状态不等于可申请两卡空位，不再引用23:20历史idle。保持唯一
A800申请，证据`resource-239136-0026.txt`。CPU几何表`a800-pressure-geometry.json`
只演示B=9984..14304MiB时Conv目标变化，表中20s service假设不是实测A800 RPS。

## 2026-10-22 A800首容量失败与修正

239136于02:06:48在d1n41a22g03两A800-SXM4-80GB执行，02:09:46 FAILED/1:0。
两独立BF16模型、四worker权重/七档sparse graph初始化通过，24GiB虚拟pool各16384
blocks；未开始100K实际prefill。首Code6359-block temporary reservation返回allocated
false，页owner正确rollback，不是CUDA OOM。free12962/13805MiB，mapped192MiB/GPU；
原budget12096MiB忽略C++默认utilization .95的整卡4057.6MiB headroom，真实仅
`63+64*floor((12962-4057.6375)/96)=5951` blocks，不能分配6359。证据
`failure-analysis-239136.json`与20rawfiles SHA通过包`live-trace-239136/`。

新实验所有phase统一显式allocator .98，保留约1623MiB headroom；预算使用
max(1GiB,total*.02)，历史样本推B11520MiB、Conv目标4731MiB，实际新卡重新测。
不减少长度/关闭graph，不把guard修改或OOM称算法收益，四策略同预算。385本地
研究/mocked integration tests通过/1skip；准备保护同步唯一A800重试。

监控异常：本地01:17:11后无query，重启时wall clock发生长跳，原因未确定；不能
宣称本地无gap。远端独立query持续每300s、02:10:51已见FAILED。首次归档SSH255
失败/零字节残包保留，重试20files全SHA通过，`archive-retry-239136.log`记录。

04:43运维：22文件headroom修复保护同步通过，唯一A800重试239658于04:41:32
提交，04:41:59在d1n41a16g01获配两卡，04:42:55 RUNNING/capacity pilot；compute
启动193tests通过。local watcher PID81762，task-scoped caffeinate PID81763，remote
query-only PID3024642；每300s继续。尚未完成100K分配/prefill，不称capacity通过。
同步证据`a800-result-20261022-042325.txt`、提交`a800-allocator-submit-20261022.txt`
及`poll-239658.log`。源码冻结，未commit/push/Aegaeon。
