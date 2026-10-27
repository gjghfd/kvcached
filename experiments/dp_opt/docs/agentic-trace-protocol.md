# Azure Conv + SWE-smith 100K：速率计算与实跑协议

当前2026-10-23：A800冻结749trace的wait-aware四策略已独立完成，0moves/0solve，
不把零迁移下指标改善称placement收益。下一先`transfer-cost-protocol.md`同A800
host/DP-device真实KV成本pilot；不改变原trace/RPS/SLO或据拓扑猜更低迁移成本。
历史H200与旧candidate保留，当前frozen为.61Conv/.004Code、B11328MiB、1200s。

2026-10-21 19:26（Asia/Shanghai）预声明，用户授权扩大长度/容量并执行。
数据来源与100K输入验证见 `../reports/agentic-100k-dataset.md`。

## 设置与假设

两独立Qwen3-30B-A3B-Instruct-2507，共址两卡，每模型DP2/EP2/attentionTP1；
本轮显式 `max_model_len=131072`，每model/rank虚拟pool24GiB，aggregate受控
mapped预算20GiB/GPU。旧32K/4GiB默认保留用于历史复现；本实验使用明确override。
保持sparse captures1/8/16/32/64/128/256、FULL_DECODE_ONLY、compile0、pause32step。
H200优先，实际healthy资源19:13显示e18有两空闲卡，提交前再次核验；不是获配保证。

model0=Code：SWE-smith一个tool shard的10个严格100K–105K真实history prefix，
每请求输出64，完整token IDs冻结，不用重复算术代替。轨迹未解决不影响系统输入
长度重放，但不作成功率评测；不是闭环agent/API实际timing。

model1=Conv：完整Azure配对长度条件P>=2048且O512..1024；823814条、全Conv
3.01719%，E[P]=3948.75/E[O]=658.68，输入明显短于100K Code。保留原O，synthetic
text匹配长度，声明条件压力而非全Azure代表性。这个切片控制驻留数，避免短P切片
为了占满20GiB需要过多请求；不要求Code长decode。

## RPS 先计算，再pilot校准

    b(P,O)=1.5MiB*ceil((P+O)/16)
    S_conv≈.2+.04*O
    E[b_conv*S_conv]=11729.227MiB*s，E[S_conv]=26.547s

Code约9.2GiB，新请求要腾出单卡空间的理想窗口：

    2*(B-b_code) < total resident KV < 2*B-b_code

其中B=20GiB，窗口约21.6–30.8GiB（真实页/已有Code会改变）。取Conv背景目标
23GiB=23552MiB。没有Code时静态率建议：

    lambda_conv≈23552/11729.227≈2.008req/s

较保守未校准首候选1.8req/s，Code.02req/s（平均50s一到达）。假设Codeprefill8s、
decode.04s/token，Code S≈10.56s，平均驻留lambda*S≈.211，不因Code并发把global
容量一直压满。两独立Poisson，不用同步bootstrap/等待见证后再到达。

CPU2000state/case的20GiB预算页owner近似，.02Code+1.8Conv下global够/local堵
514/1834safe（28.03%），一条Conv搬迁且all-decode gate可恢复127/1834（6.92%）；
offered超预算8.3%，可恢复自然等待p50约1.04s。其余速率及假设在
`evidence/agentic-trace-run-20261021/rate-theory.json`，不是GPU队列模型或收益保证。
多Conv搬迁可能有更多机会，但自然释放快/host-staged成本仍可能负收益。

先同allocation跑pilot：验证四pool对100K实际allocate/rollback及单Code完整推理，
然后Conv人口16/32/48各一次，加入一条真实Code。记录prefill、service、graph、
权重和实际页、提出有限人口估计lambda_conv=23552/(b*measuredS)，Code
min(.02,.25/measuredS_code)。这是校准建议，不是证明Poisson稳定服务率。
根据pilot冻结正式trace再执行，所有候选/失败保留，不根据收益筛seed。

## 四策略与判据

pilot通过后在运行前冻结1200s trace（首候选seed20261021：25Code/2194Conv），
四策略no_move/own_model/KV-only/ILP同token IDs、同arrival、预算/图/控制，FIFO，
完整drain，timeout900s，首声明SLO120s另报告分模型p95/p99。无丢请求或输出长度
变化；策略随机顺序，第一轮只初测不是多seed最终收益。真实prefill/copy/pause均
计入端到端；no_move仍是同full-reservation控制而非原生渐进KVbaseline。

保留all-decode gate、globalfree门槛、每request/population一次paused ILP。
现原型非wait-aware，N128性能surrogate未对100K/小batch联合校准；如收益为负先
记录、诊断control与自然等待，不能盲目提高rate或忽略老Conv latency追正。
请求/迁移token正确性、owner/复制峰值、四rank共驻留/graphs必须独立复核。

## 运维与后续

同一个H200 allocation尽量完成pilot与正式trace；提交前明确文件旧SHA备份/原子
同步，运行期间不覆盖checkout。每300s查询至终态，归档全SHA/原请求/负结果后
推进下一步。fork main，不commit/push；不重启Aegaeon，只复用已有公共toolchain。

19:30状态：唯一238436于19:29:43获配d1n41e18g01，pilot进行中；源码/输入SHA
备份原子同步通过。本地研究240tests通过/1跳过，compute内初始两组145/193tests
通过。RPS尚待pilot冻结，不能以已RUNNING声称100K推理或收益通过。

## 19:40 实测校准与正式冻结

238436的pilot已成功退出；18个原始文件独立SHA归档。本机双H200每卡143771MiB，
模型/graph加载后使用72101/71052MiB。四worker均128K/DP2/EP2/TP1，四pool实际
101371+64token临时分配/rollback成功，aggregate目标卡峰值9.46875GiB；这仅验证
空人口单请求，混合人口实际20GiB预算仍由正式trace监控验证。四rank真实FULL decode
graph存在，100K prefill保持正常非decode路径，没有关闭sparse。

单Code TTFT8.575秒、service9.168秒。Conv同mark P3955/O680，在16/32/48有限人口
加入一Code时mean service17.450/24.235/31.093秒；混合Code service
16.223/19.375/21.435秒。pilot同步人口并非稳态Poisson，尤其Code占空比更高，
不能直接称1.741RPS为可持续率。

在正式运行前拟合与冻结（源码`../agentic_rate.py`，3tests通过）：

    S_conv(N,O)≈(10.616508+0.426338*N)*O/680
    b(P,O)=1.5MiB*ceil((P+O)/16)
    E[b*O]=291067.429MiB*tokens，E[O]=658.677
    E[b | resident]≈E[b*O]/E[O]=441.897MiB
    N_target=23552/441.897=53.2975
    S_target=32.2939s，lambda_conv=N_target/S_target=1.65039RPS
    lambda_code<=min(.02,.25/mean(S_code))=.0131503RPS

向下取整后正式 **Conv1.65RPS / Code.013RPS**（平均76.92秒一个Code），1200s、
seed20261021、repeat1、原SLO120s不变。固定Poisson产生14Code/2011Conv、2025总请求；
Code使用10真实episode有放回抽样。代码与原paired Conv源不变，不按收益筛seed。
新trace SHA与源在`evidence/agentic-trace-run-20261021/frozen-verification.json`。

上述假设service与O成比例、未独立校准P成本，53.3人口轻微超出pilot最高48；
同步finite batch与Poisson不同，Code干扰也不同，需用正式实测确认实际驻留量、
全局容量和global够/local堵频率。23GiB只是设计目标，不是已测驻留或收益保证；
若无足够机会须按实际service修正并保留本轮，而非删除失败/筛选结果。

冻结trace实际Code mean逻辑KV9470.786MiB，Conv背景目标每rank11776MiB，因此在
平衡且没有其他Code时每rank差766.786MiB，而global在准入后仍余7937.214MiB。
可行方案可能需要搬约2个均值Conv或1个更大Conv；实际至少约8个96MiB可释放页，
共享owner/pinned页/目的copy峰值使logical求和不足以保证ILP可行。源码仍每model/rank
2候选、最多2迁移，故这是受限方案最优，不是所有request全集全局最优。

19:43:11已核查formal_trace_running；19:48:11首own_model elapsed176.49s，offered298、
completed169、moves0。暂未得到完整比较，不用pilot通过/理论窗口宣称收益。监控器
SSH timeout下继续下一计划轮次，compact输出避免重复dump大token bank，计划间隔300s。

## 20:54 首trace失败后的开放环校准

238436首own_model于20:06:50超时，41rawfiles SHA完整归档；尚无四策略比较。
详细诊断只在`../reports/agentic-trace-results.md`。后期KV明显未满，原累计同步记录
拖慢准入；不得把这条trace当做placement容量收益证据。统一增量JSONL、并发四rank
RPC和复用准入后观测先修复记录/控制瓶颈，所有策略同配置，不改预算/graph/SLO。

重试同allocation先no_move开放环校准：Conv1.65/2.0/2.35、Code.013RPS，每档300s，
seed20261021，所有paired Conv来源保留，真实Code bank，完整drain。选择最接近23GiB
背景且不持续排队发散的一档；这是rate校准，不基于ILP收益选rate。记录100–300s
背景logical reservation、physical mapped、外部queue及Code head容量阻塞，若所测
范围不达目标，保留全部校准再修正。成功后仍冻结1200s/seed20261021/repeat1的四策略，
SLO120s/timeout900s不变，禁止将失败请求丢弃。原Finite-population公式仅保留历史估计。

## 21:18 自动冻结重试，取代三档手工等待

238639于21:12:17仍PENDING时取消确认，无GPU执行，不覆盖活动版本。准备唯一替代
H200作业，校准档位改为Conv **1.65/3/4/5RPS**、Code.013RPS，每档300s/同seed。
理由是pilot Conv在Code前/后client-cadence中位数约10–14ms/token，而Code重叠期间
约25–35ms，原同步人口每轮加入Code的服务时间不是低Code占空比的稳态；三档窄范围
可能始终欠填。cadence仅是客户端描述性分解，不作GPU阶段归因或稳定率证书。

开放环校准统一JSONL记录/完整drain，用全部10条Code marks，而非从旧trace去重的9条。
全部档位测完后`../freeze_agentic_calibration.py`自动选最接近23552MiB背景的稳定档位，
不依据任何ILP结果选择。运行前固定guardrail：Conv mean queue<=5s、drain<=90s、
250–300s外部queue均值减100–150s均值<=10requests；候选平均Conv驻留须位于
目标的.95–1.25，即21.85–28.75GiB。若无满足者，保留全部结果并失败退出，不强行
减请求/改预算/SLO或直接跑欠填正式trace。有限300s仍不是稳定率证书。

满足者直接在同allocation冻结新的1200s/seed20261021、两独立Poisson完整trace与
source/trace SHA，自动继续四策略（own_model/no_move/kv_only/ilp）。取消30min手工上传
等待，避免本地暂停导致已经获配的GPU空等配置。原128K/24GiB虚拟/20GiB预算/
sparse/32step/SLO120s/timeout900s不变。254CPUtests通过/1skip；尚待allocation回归。

归档后先本地运行`verify_agentic_calibration.py <raw>/calibration --inputs <input-dir>
--output <verification.json>`：从全部Code/Conv marks和已声明seed/rates重生成每档trace，
逐请求验证无丢失/时间/长度，核查JSONL与终态数组、四rank拓扑/实际FULL graph、准入页
ownership与mapped预算，独立复算队列/驻留/容量flag及guardrail。校准完整drain但目标
欠填时，原始校准可以验证通过、guardrail仍必须rejected；两者都不是ILP收益结论。
sampled global/local flag仅为控制器采样证据，不等于真实allocator probe或迁移可行证明。

本地watcher可带`--agentic-inputs <input-dir>`在终态自动归档、SHA核查，然后独立核验
calibration；仅正式四策略完整成功时继续`verify_trace.py`和`analyze_trace.py`。失败只
保留证据/错误并停止，不自动改配置或提交GPU。归档等待remote query-only watcher
退出，避免其日志仍追加时manifest/tar不一致。分析工具本地更新不覆盖活动runtime。

## 23:53 用户指定A800，替代H200设置

238698于23:46:19仍PENDING时取消，无GPU执行。新唯一allocation只申请两A800，
仍两个独立BF16Qwen、DP2/EP2/attentionTP1、128K、24GiB/model/rank虚拟pool、默认
sparse captures、FULL_DECODE_ONLY、compile0、pause32step。虚拟pool不是物理预算。

先同allocation跑A800容量pilot：模型/graph初始化后各worker释放**未使用**torch
allocator缓存，不销毁graph或模型；记录cuda free/total/allocated/reserved。四pool
实际allocate/rollback最长101675+64输入；真实Code放rank0，同时4条Conv放rank1，
每秒采样并完整drain。仅此小人口pilot允许明确非均衡rank来检验headroom，不是最终
Poisson trace，不用于声称收益。正式四策略统一仅初始化时释放unused缓存一次。

预声明物理预算B=96MiB*floor(min(14GiB,所有样本free+mapped−1GiBguard)/96MiB)，
须大于最长Code页对齐KV+3页，否则保留失败退出，不截短100K/量化/关闭graph。
这是有限headroom采样加guard，不是所有prefill/迁移峰值安全证书；后续仍核actual
mapped预算与copy-before-release峰值。Conv背景目标
`R=2*(B-b_code_max)+min(.1*b_code_max,768MiB)`，在理论碎片窗口下界上方，避免
大量搬迁需求；pressure为所有策略同一setting，不按ILP结果调整。

根据这4条Conv完整measured service和Azure完整配对E[b*O]估central RPS，四档为
.5/.8/1.2/1.6倍（至少.05RPS），Code仍.013RPS；四档每300s开放环/no_move完整
drain。沿用meanConvqueue<=5s/drain<=90s/外部queue增长<=10以及target .95..1.25
guardrail，成功才freeze1200s四策略。A800速率不沿用H200，成本surrogate仍未联合
校准，负结果与自然等待/老Conv惩罚都保留，不能以OOM或欠填作算法收益。

### 2026-10-22 03:22 allocator headroom修正

239136实际A800 first probe失败：默认`KVCACHED_GPU_UTILIZATION=.95`在C++
`get_avail_physical_pages`从free扣整卡5%，即4057.6MiB；free12962MiB仅5951可用
blocks，不足最长Code6359，虽然24GiB虚拟容量及raw free足够。不是CUDA OOM。
本A800实验显式`.98`覆盖所有阶段/四策略，保留约1623MiB硬guard，不改graph。
预算公式中的1GiB替换为`max(1GiB,total*(1-utilization))`，并把utilization写入config
与独立verifier。旧A800样本因此B11520MiB、Conv目标4731MiB；只是重试前推算，
新allocation仍重新真实probe/prefill/校准，若仍unfit保留失败，不截短/量化。

### 2026-10-22 A800容量通过后的排队与rate重试

239658实际B11424MiB/GPU、Conv目标4539MiB，最长Code101675+64真实四pool分配、
5pilot请求及四rank默认graph通过；四Conv与100K重叠的finite pilot service67.958s
给出的初始.08/.12/.18/.24 bracket不是稳态速率。前三档完整开放环Conv驻留分别
479.51/811.39/1410.23MiB，欠填；第四档allocator拒绝full reservation导致旧scheduler
抛致命MemoryError，不是已观察到CUDA OOM，不以失败/截尾比较收益。

下一唯一A800allocation仍重做真实容量probe，然后用**Conv .30/.45/.60/.80RPS、Code
.013RPS**四档300s开放环校准；固定seed与十条Code输入不变。bracket在策略结果前由
前三完整校准的欠填证据预声明，并写入`calibration-bracket.json`，不覆盖新容量实测B
和目标R。既有queue<=5s/drain<=90s/queue增长<=10/target .95..1.25保持；若不存在合格
候选就拒绝freeze并归档，不放宽判据追正。

full reservation返回None时请求进入native waiting但不允许partial prefill。受控trace
禁止scheduler自行扩大物理mapped预算；controller在同一跨tenant锁内重新检查mapped
增量、实际free blocks并调用单request reservation重试。等待期间不持有控制锁，另一
tenant可继续准入，同tenant在该FIFO头预留成功前不提交下一个请求，不重复generate。
原始decision区分`submitted`/`admitted`/`reservation_pending`，请求增加scheduler
monotonic的`reserved_at`；queue=reserved_at-arrived_at，含引擎内部KV等待，TTFT/
latency不变。population计入capacity_pending，freeze外部+内部未准入队列，避免虚报
排队收益。正常runtime使用精确scheduler时间；快结束未见快照时标记first-token上界。
不增加900s整体请求timeout、不忽略真正driver错误，所有策略同实现/预算/graph。

### 2026-10-22完整校准拒绝后的RPC确认修复

240575四档全部drain，但Conv平均queue均>5s，不能freeze；.45驻留4729.7MiB在目标
窗口不是放宽queue依据。后续保持同.30/.45/.60/.80rates与Code.013/seed不变，先修
提交后重复四rankRPC：full reservation确认只查询目标rank，直接复用actual snapshot
替代再次四rankpost观测。串行控制锁内其他rank仅可完成释放，不可自主mapped增长，
其before值用于保守budget上界，report显式标注mixed-time而非原子实际全局快照。
全rankbefore、monitor与迁移真实观测不删，不改graph/timeout/SLO或数据子集。

### 2026-10-22 free/mapped同一ownership区间采样

240654在Code完成边界先读低free、后读已释放mapped，导致不存在的低free+低mapped
预算。后续每次worker memory查询前后都取四rank ownership，mapped/pages/request
blocks一致才作为capacity sample；不一致的完整前/后和memory raw写transitions并保留，
不拿异时刻相加。稳定区间所有低free样本仍计最小budget，初始/终态paused样本保留；
独立verifier核两边一致及mapped导出，不能用跳过真实稳定峰值假造100K容量。

### 2026-10-22 14:43 A800 Code驻留规则重算

240679完整四档704请求/88raw全SHA与独立验证通过，但.30压力不足、.45/.60/.80
queue>5s，freeze拒绝，正式策略没跑。target-only确认改善不等于门槛通过。.013Code
是H200 service估计；本次16Code精确scheduler reservation→completion均值56.562s，
lambda*S=.735，不满足最初.25人口目标。下一轮只重应用原规则：

    lambda_code=floor_0.001(min(.02,.25/56.562153727))=.004req/s

Conv bracket.30/.45/.60/.80、seed20261021、300s校准/1200s正式、源数据、100K长度、
graph、预算公式及queue5s/drain90s/pressure guardrail不改。新独立校准成功才冻结，
旧结果全部保留，不按policy结果选rate、不抑制Poisson burst。该均值只作有限样本
驻留反馈，不能断言稳态或收益。复现工具`recalibrate_agentic_rates.py`校验完整source/
trace/ownership/graph与精确时间；`analyze_agentic_calibration.py`区分RPC控制/内部
reservation等待，derived控制区间不是CPU或kernel计时。当轮runtime不再增加新优化。

### 2026-10-22 15:28 同rate的rank-local准入与非阻塞监控

240782的Code.004四档696请求完整结束，89raw/out/err全SHA及capacity/calibration
独立验证通过，但freeze仍拒绝：.30欠填，目标档.45 queue8.717s，.60/.80也>5s。
降Code rate不是解决方案；原seed的171.57/172.49s双100K仍造成真实竞争。下一轮
不再改rate/seed/guardrail，修复控制面：

- opt-in `DP_OPT_RANK_LOCAL_ADMISSION=1`，direct before只刷新arriving tenant两rank；
  另tenant取最近串行准入账本的mapped上界，因为完整预留+controlled retry下无独立
  mapped增长。cached blocked时全四rank刷新，不能把保守无解当真实无解。
- 成功后目标rank真实ack更新账本，所有准入/显式retry/迁移先递增generation并失效；
  失败不保留可能过期的低预算。迁移仍pause、实际四rank观察与copy峰值核验。
- 全四rankmonitor RPC在锁外查询，前后短锁检查generation，一致才做population；
  跨admission epoch的完整raw另写monitor-transitions.jsonl，不伪装原子观测，保留
  监视空窗及有限采样限制。不能拿monitor跨模型kernel等待阻止本可准入的Conv。

此改动只是同预算所有策略共同fastpath，并非ILP收益；mock安全/异步测试不替代GPU。
暂保留跨tenant全局串行准入锁，真实capacity block仍可能等待Code步边界，结果必须
如实记录。runtime开关仅本实验，不改变普通serving或capacity pilot。

### 2026-10-22 17:00 独立tenant派发与锁外准入观察

240886的696请求/92raw全SHA及capacity/calibration独立通过，freeze仍拒绝；目标档
.45 Convqueue8.950s，control14.58s（此前33.72s），控制减少未解决队列。锁外monitor
有5/9/14/17转态，最大观察gap18.59/20.56/49.39/59.50s，完整保留，不能当零负载。
direct cached-blocked全rank刷新仍在全局锁内，单个dispatcher按Code头FIFO等待
100K RPC时另tenant不能打入，这是下一轮要验证的控制瓶颈，不是纯GPU因果结论。

保持Code.004/Conv四档/seed/300s/1200s、数据/graph/B公式/guardrail不变，仅启用
`DP_OPT_CONCURRENT_ADMISSION=1`：两个tenant独立FIFO dispatcher，observation RPC
在锁外，同tenant admission lock防重复；短全局锁内核对自己的generation并合并
最新foreign账本，预算验证和实际提交ack仍串行。全rank观察必须全generation稳定，
期间其他模型增长则重读；retry/迁移使有关model generation失效。故不能用锁外旧
mapped授权两个请求同时超配，旧no-pause/graph/真实copy峰值条件保留。此修复只
缩减控制阻塞，不是wait-aware或联合成本成熟算法，真实global不足仍排队。

### 2026-10-22 17:37 离线驻留积分与采样偏差核验

`analyze_agentic_calibration.py`增加完整请求生命周期的逻辑KV积分，报告全到达窗口
及原pressure窗口`[duration/3,duration)`。对每请求，以完整`P+O`按block向上取整，
从scheduler精确reservation到client completion裁剪窗口并积分；同时报告时间均值、
峰值与reservation来源。结束时间是客户端完成而非allocator释放，逻辑KV不等于
物理mapped；不凭此推断实际copy峰值或每卡可分配性。该核验不依赖monitor采样，
不会将RPC空窗当零，不替换已冻结guardrail或改动活动runtime。与原sample mean
的差值只作观测覆盖/口径诊断，不能直接当采样偏差的纯因果估计。

### 2026-10-22 17:58 并发路径完整校准后的局部压力细化

240982四档696请求、92rawfiles全SHA及独立capacity/calibration通过，freeze压力
拒绝。原.30/.45均queue<=5/drain<=90/growth<=10合格，sample驻留2765.714/7757.06
却跨过target4347的.95..1.25窗口。只从完整no_move压力局部插值再测，不放宽门槛：
`r*=r_lo+(target-K_lo)*(r_hi-r_lo)/(K_hi-K_lo)`，步长`(r_hi-r_lo)/8`，偏移
`[-1,0,1,2]`后round.01，得 **Conv .33/.35/.37/.39**；不稳定.60/.80也完整记录。
Code.004/seed20261021/每档300s/正式1200s/B公式/BF16/128K/24GiB虚拟/graph/
pause32/并发runtime不变。拒绝无稳定两侧括号、重复rate、越界/不足精度，来源SHA
在`conv-pressure-bracket-240982.json`。此插值是新增校准网格规则，不是科研性能
线性模型、不承诺新节点/新rate稳态或收益。新容量+完整校准成功才能同allocation
自动freeze四策略，失败先全SHA归档诊断，不从前缀放行。

### 2026-10-22 18:40 非单调压力后的长窗口重新校准

241036四档488请求/92raw SHA及独立核验完整，freeze拒绝；.33 sample5441/queue1.327
而更高.35/.37仅3555/3727MiB，.39queue5.486。第二100K在.33等待至229s才准入，
其他档173–175s准入；300s窗口中串行/重叠路径不同，rate→KV局部插值不可靠。
下一明确将每档开放环 **300→1200s**，与formal长度一致；相同seed1200s有4Code，
保留171.57/172.49s突发，600s仍只有2Code。各阈值queue5s/drain90s/growth10/
pressure.95..1.25及相对窗口比例不变，旧300s负结果保留，不以此声称稳态已证。

新档位以240982完整raw的固定arrival`[duration/10,.4*duration)=[30,120)`集合为
输入，选最高已测且该集合queue<=5s的Conv rate.80，要求集合所有请求完成都早于
首Code，不逐请求按完成快慢或长度筛选；82条全部满足，`E[b*S]=7181.851MiB*s`。
`r*=target/E[b*S]=.60528`，固定乘数.70/.85/1/1.15再round.01，档位
**.42/.51/.61/.70**。clean服务是低干扰起点，不是mixed稳态rate或策略收益，必须
真实1200s四档验证；无满足clean集合则拒绝估算。Code.004/seed20261021/128K/
100K/BF16/24GiB虚拟/B公式/.98/sparse/pause32/并发路径/四策略仍相同。
CLI `--long-window-bracket`保留来源SHA与全部clean候选；launch显式duration1200。

### 2026-10-22 21:15 冻结trace控制修复重放

241086完整四档1200s校准选择Conv.61/Code.004、749请求（4Code/745Conv），
B11328MiB/GPU。原own_model因自身无可移动请求却426次暂停重规划超时；该失败
不是普通scheduler自然等待，也不把640Conv/0Code当四策略比较。

四策略共同先无暂停计算owner-complete source-release乐观上界；own_model只含自身
候选。上界不足则等待，上界允许再进入原冻结transaction并重新验所有物理owner与
copy峰值。owner均分释放份额、top max_moves之和只提供必要条件，不计destination
负担且不能宣布真实allocator无解。每tenant重规划冷却max(2s,4*上次规划实际耗时)，
direct尝试不受限制，FIFO/另一tenant继续，全部等待计入原指标，不放宽timeout/SLO。

旧snapshot首例需3条Conv迁移而max_moves2不够；重试统一8候选/model/rank、8moves，
own_model不跨tenant。这是受限候选ILP，不声称全集最优；沿用历史sparse surrogate，
尚无100K双模型联合性能校准或独特性能项收益。

`agentic-formal.sbatch`只A800，先原100Kcapacity，随后trace/rate-decision双SHA保护
重放241086 frozen input，budget固定11328。新capacity小于此B则失败，不暗抬预算、
截短或重新选择rate。保持BF16/128K/24GiB虚拟/.98/sparse/pause32，1200s和原SLO，
四策略相同设置，收益必须含Code准入、老Conv退化、真实规划/暂停/copy和全部请求。

241267容量独立通过但formal启动CPU test发现并发线程JSONL乱序；新增每journal
异步锁，只序列化同文件写入，不持GPU准入锁、不累计重写历史，不改arrival/FIFO。
注入首完成记录写延迟回归保序；原失败31raw及188通过/1失败保留，不跳startup测试。

完整四策略后离线报告Code、所有Conv、所有请求的相同ID配对queue/TTFT/latency/
TPOT/SLO变化；另固定每baseline在Code frozen arrival时已reservation且尚未client
completion的Conv ID集合，检查其ILP端代价，不能按ILP后完成快慢重新筛选老请求。
该固定集合不是随机因果估计，报告所有Conv和总latency sum才能避免只宣称Code
收益。arrival绝对monotonic origin每策略不同，所有差值先转请求相对arrival。

### 2026-10-22 22:55 迁后实际allocator证据与copy-cache回收

241311 KV-only两Conv迁移但首Code0blocks/pending900s，需6340blocks而迁后target
free5631。仅mapped预算合法不保证allocator可分配；旧run copy缓存因果未直接测。
新四策略每trace起点同样回收unused并记四worker内存，迁后额外回收copy暂存缓存、
记录torch allocated/reserved/free，强制刷新physical容量；actualfree不足不调用admit，
正确保留已迁request、end/resume并等待，所有动作时间继续计入成本。不得清活KV、
关闭默认图、提高budget/timeout或者删除旧失败请求。成功仍需真实reservation/
ownership/copy峰值验证，不把cache回收后读数当峰值保证。

### 2026-10-23 01:02 wait-aware同trace诊断协议

不改变749冻结trace、两A800、B11328、.61/.004、100K/BF16/128K/默认sparse/pause32，
只ILP添加等待动作。每request在线scheduler observation最多8点、相邻至少.5s，
由实际输出进度算r_i，剩余预测R_i=(O_i-o_i)/r_i；未知rate不触发迁移。source页
预期释放时间为所有owner R_i的最大值，pinned/未知owner不计；按释放事件累计，
得到两rank能容纳新page-rounded KV的最早预测时间，取较小者W。这是当前速率
外推，client完结/allocator释放可能有偏差，绝不是用未来completion oracle。

同可迁候选source页释放份额均分owner，按copy_time/release_share排序允许fractional
份额给出乐观copy下界L；不考虑destination负担所以只是拒绝门槛。冻结241580来源
cost为pausefloor.35s、solvefloor.15s、hostcopy5s/GiB，比较
`(N_active+1)*(.35+.15+L) < .8*W`，否则无暂停等自然释放/direct，且每tenant FIFO
保持。全请求pause损失不能只算incoming TTFT。进入冻结epoch后，枚举rank原ILP
layout/copy objective与固定控制损失再对比W，实际已耗pause/solve也计入并从剩余
等待中扣除，不值得就end/resume、不迁。等待是候选动作，不把所有可行解都执行。

新`agentic-wait.sbatch`单A800allocation先capacity、再同原frozen四策略；三baseline
保持旧控制，只有ILP启用等待，worker在线rate观测共有以免单策略额外采样不公平。
新增wait-prechecks增量journal去重同active状态，未知→有估计允许再记录，不能持锁
写大历史；不claim真实全局最优、联合cost成熟或未知rate下无收益机会。

七条旧迁移以当时之前clientcadence重构rate离线均选择等待，不能当GPU收益或
验证新的scheduler观测实现。497CPUtests/1skip后真实同trace验证，将Code/allConv/
固定老Conv与总latency/SLO完整对照，保留所有负结果，不通过改rate或删除请求追正。
