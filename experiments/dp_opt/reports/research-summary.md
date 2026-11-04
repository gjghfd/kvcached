# 跨模型 attention-DP / KV placement：算法与实验总览

**2026-11-02 14:39+08双线续**：N3/280104实际14:34:04在两A800启动；DEBUG
279537于14:33:53 FAILED1:0，末尾firstlayer独验误拒四点serial采样重复最长
Code。先自动SHA归档与独立review，旧host退出后才NEW debug-only修复，不能把
FAILED当mirror成功parent，性能闭包及300s链不动。独立两seed N2 pool WP Conv
TPOT17.358ms（vsrandom改善9.564%，vsno_move退化28.230%）；Code TTFT改善
56.127%，全部first tokens早于回迁，小样本/不同cohort非正式tail/heldout结论。
9CPUtests及完整负项在r5/native-N2-seed2-review，研究与debug目标未完成。

**2026-11-02 N2第二seed实测完成**：279520于12:40:53→13:53:19两A800执行、
COMPLETED0:0，484raw/两log/216源与fullstrict通过；同coord归档桥真实recovery
成功，10069自动唯一N3/280104续接。CodeTTFT119.562→56.199s，降低52.996%；
ConvTPOT20.071→18.134ms，比random改善9.651%，但比no_move15.023ms差20.710%。
Code1/Conv2尾部仅Type7描述，初始TTFT不能归回迁；两seedN2约9.5%相对random
改善重现，仍非正式Poisson/heldout结论。DEBUG279537自12:50:55在A800运行，
firstlayer尚未结束。合盖localgap如实保留，remote300s连续实查，不依赖旧晚间ETA。
完整表与负项在public-quality-source-recovery-results，整体收益/debug目标未完成。

**2026-11-02 回迁门槛独立review**：318有序固定快照候选（非新GPU实测）表明，
不是把pause wall全部误算成老Conv停推；实测已用各rank reached→resume。仅换
非负activation不足以翻转N3/3、N4/4、N4/3；三者paired-contrast uncertainty为
118.549/89.012/34.239 request-s，N4/4连copy+risk已超过性能预测收益。系数2是
两侧预测最坏和，未证明共同EP噪声可消除，不清零/减半U追正。完整证据
`evidence/public-pause-cost-opportunity-20261102/`，说明主拒绝项，不声称等待最优。

**2026-11-02 12:07+08**：两个A800请求仍pending；Slurm queue socket超时与健康容量
未知如实记录，22:54/23:11只是估计，持续300s，不重复请求。NEW mirror显式kernel
opt-in/childenv/SHA/预算主线程284CPU通过，但firstlayer尚未GPU成功，不能prepare
或称数值debug完成。Stratos固定parquet仍网络失败，viewer不冒充Git/parquet；
未来独立API快照source须全行/完整分词核验，当前OpenThoughts3冻结链不变。
正式tail/heldout与debug目标未完成；详见public-quality-source-recovery-results。

**2026-11-02 11:18+08**：同coordinator归档bridge经282CPU回归后安全交接到
10069/caff10070，实际argv/start/PPID1、首poll与loaded-source已核；279520未动，
DEBUG279537不动，11:18:07仍PENDING。216/228+r2两host与NEW三桥源冻结，不跑
外置follow。真实新网络归档仍待终态执行。N4确认Code=0/Conv=1，老Conv实际暂停
约.171s不等于2.677s pause wall，但控制锁阻塞新准入和全部负TPOT仍计入。
持续300s，性能链/数值debug/固定revision候选分析独立推进，不声称目标完成。

**2026-11-02 11:10+08恢复**：N4已通过NEW r2完整strict phase核验，不再只raw复算；
原COMPLETED/旧风险失败均保留，详情与receipt在public-quality-source-recovery-results。
性能99679持续第二seedN2/279520，独立DEBUG1546执行r12/279537，11:09:53均
PENDING/Priority；216/228及r2两host源冻结。后续N3/N4→3600s校准→七策略→10800s
heldout不变，DEBUG成功后才NEW mirror。合盖10:54:47→11:01:49的本地wall-gap如实
保留，remote性能log两次实查连续，DEBUG补query-only1972698首poll通过；继续300s。
N4完整核验不抹掉ConvTPOT退化，也不证明TTFT由回迁改善；小样本非正式tail结论。
NEW同coordinator归档bridge仅host交接，保留现GPUallocation/原trace；整体收益与
数值debug未完成，继续迭代，未commit/push/Aegaeon。

**2026-11-02 N4完整raw更新**：485raw/两log/216源SHA独验，原279153
COMPLETED/0:0；fullphase仍因风险guards8处−1ULP被旧host adapter拦下，原Fraction/
KV几何全同。正在NEW host epoch硬化再完整复用，不重跑N4/修改GPU源；75553已退出，
DEBUG r11无job安全窗口失败保留，后续新r12须绑定新性能PID。N4 workaware Code
TTFT243.256→55.958s（−77.0%vsno_move），combined avg/P95/P99分别
11.301/44.805/53.728s vs48.760/194.643/233.534s；仅5请求Type7描述，非正式tail结论。
ConvTPOT22.762ms比random22.706ms差0.247%，比no_move18.509ms差22.98%；回迁
350.132→353.029s在人口4→3→2后，所有首token此前完成，不能归因TTFT收益于回迁。
N2相对random TPOT改善9.47%仍更强，但未完成第二seed/Poisson/heldout/质量debug。
详情在唯一public-quality-source-recovery-results报告与native-N4-review证据。

**2026-11-02 10:05+08**：N4/279153于09:53:22 COMPLETED/0:0；四策略结束，但
canonical progress symlink使原archiver失败，正做host-only samefile/SHA恢复，不能
提前作完整收益结论。216源/75553不变，300s observer-r4接续。DEBUG279359于
09:44:56 FAILED，262raw两log归档；spawn重入无guard入口已定位，新helper/runpy
回归200过，r11本地准备228文件，尚未提交。firstlayer/P74/质量与formaltrace未完成。
本地68,812完整响应复核仅下一协议候选：Stratos冻结viewer快照276条闭合原O20–28K，
未绑定上游Git revision；不换当前trace或外推旧成本。见long-response-opportunity证据。

**2026-11-02 09:02+08**：性能279153/N4在两A800持续RUNNING，首random策略5/5完成、
其余未完成，不能做完整策略对比。DEBUG279164的248raw已独验归档；只读私有runtime
导致2tamperfixture PermissionError，修NEW sbatch副本mode/200tests过。唯一新DEBUG
279275（r9/228；83178/caff83179）完成hold/owner-SHA注册/release，现PENDING；
75553性能源216不变，300s监控继续。host恢复review28检查无当前触发blocker，但
只保证可信固定N3/串行数值域，扩大protocol须新epoch硬化，不改活动源。正式
Poisson/heldout和第一层/P74/kernel/质量debug仍未完成，无commit/push/Aegaeon。

**2026-11-02 08:16+08**：性能279153/N4、独立first-layer DEBUG279164均PENDING/
Priority；DEBUG r7已完成held提交→owner/SHA授权→release，性能/DEBUG各300s持续
监控，3轮补充SSH0且间隔约300s。没有新GPU结果，未用排队/后台启动代替目标完成。
N3的1ULP风险精确差异由agent独立复现11字段，全部Fraction KV证明相同；严格
Decimal80 host恢复与原FAILED交叉证据完整保留。回迁340.815→344.665s，发生N3
自然缩到N2后，早期全部TTFT不能归因于此回迁。继续原固定trace链，并行数值debug。

**2026-11-02 08:04+08**：278898四策略完整drain/inner0，但outer源码路径后验FAILED
保留；482raw/两logSHA归档。严格owner/readlink/完整source审计及Decimal80风险数值
portability复核通过（仅1ULP风险项，Fraction KV证明不变），未改216 GPU冻结源码。
N3 workaware Code TTFT241.402→56.332s（−76.7%）；Conv TPOT21.951ms比random
22.303ms仅好1.6%，仍比no_move18.419ms差19.2%，不是全面/正式tail收益。详见
`public-quality-source-recovery-results.md`新N3表及原始sidecar证据。性能恢复PID75553，
279153/N4首poll PENDING；300s继续原剩余native→rate→grid→heldout，不重复N3。
独立DEBUG r7/228文件已安全staged，等待fresh性能poll授权；279051启动tests通过后
namespace/cwd失败已定位修复，r5/r6无GPU提交。真正first-layer/P74/kernel与数值修复
尚未完成，性能/debug并行持续，无commit/push/Aegaeon，最终研究目标未完成。

**2026-11-02 07:16+08**：性能278898持续运行，13次300s查询SSH0；N3等待完整四策略
终态独验，不使用未drain片段。原N2回迁在Code结束后0.000935s开始、2.704s观测平衡，
不是晚回迁；其后无新Conv到达，不能声称Conv TTFT因回迁改善。独立DEBUG279011
启动6s失败，231raw/两log归档；334pass/5fail是poll时钟/mtime fixture，不是模型差异。
341本地/463隔离回归与228文件新r4冻结完成，下一新poll授权GPU重试；性能不取消。
真实first-layer capture、P74与具体kernel仍未完成，原N2小样本不替代正式Poisson收益。

**2026-11-02 06:14+08更新**：恢复278772实际启动13s后FAILED，零模型加载/请求；
249raw/两log已SHA归档。补齐真实shell启动闭包/动态分析模块/ILP JSON fixture，
216冻结文件SHA、171集成tests与122隔离critical tests通过。r5独立恢复278898
06:14:31两A800开跑；300s监控/终态独验/自动余五native→rate→七策略→heldout。
DEBUG278529完成，169raw/两log/42快照独验；7/8首分歧共同prefix归因约99.99%
来自上游hidden变化，不是head主因；P74首139未覆盖，具体kernel未证明。
first-layer probe/review并行且未GPU验证，不阻断性能；正式新dataset收益/debug未完成。
最新证据与失败边界见[结果报告](public-quality-source-recovery-results.md)。

**2026-11-02 05:14+08**：首公开原始长响应N2四策略完整独验，Code TTFT
135.666→55.778s（-58.9%vsno_move）；Conv TPOT18.317→16.583ms（-9.5%vsrandom），
仍比no_move12.051ms差37.6%，Code TPOT也变差。每策略1Code/2Conv，tail只能描述，
不称最终Poisson收益。396raw/两log SHA已归档，原278514科学checkpoint仍CANCELLED；
壁限/缺失exit marker归档修复与负代价见[结果报告](public-quality-source-recovery-results.md)。
新独立恢复278772已两A800开跑，sourceepoch-r2自动余五native→rate→拆七策略/
heldout；DEBUG278529独立继续，整体收益/debug目标未完成，不发布/Aegaeon。

**2026-11-02 03:36+08实查**：公开原始长响应性能278514已03:32:49在两A800开跑，
native首case进行中；独立DEBUG278529仍排队。两线300s持久监控与终态SHA核验，
已有runtime冻结，不因数值debug等待/取消性能。下一步独立driver承接rate校准与
拆阶段七策略/heldout；当前尚无本轮完整收益或共同prefix hidden/head归因结论。

**2026-11-02用户新要求：性能/debug并行**。完整greedy一致性不再阻断性能实验或
有限setting搜索；差异保留为质量观察。长度/到达/请求数、KV ownership/真实预算/
copy完整性及所有非greedy错误仍独立检查。性能使用原BF16/default head/default
sparse，无debug CPU快照开销；DEBUG单独FP32 head共同prefix归因，互不取消等待。
旧274385四策略完整native-N2已重新核验性能，原FAILED不重标；新性能两seed/N2/3/4
链在`evidence/public-performance-parallel-20261102/`准备恢复，之后rate/七策略/
heldout。下面历史「strict必须先过」「失败性能一律禁用」不再适用于仅数值分歧的
完整同工作量case；不能以此放宽KV正确性或声称最终研究目标已完成。

**2026-11-02独立代码review**：用户指定的subagent已完成原版只读审查。276677零copy/
零case暂停仍分歧；r1首次分歧约2.07/7.39s，早于Code90s到达。真实Conv decode
padding1→8、prefill149→150指向batch/graph相关路径，但MoE/attention与head GEMM
哪个贡献仍未证明，KV/输入状态也未完全排除。未见原同步req/logits映射必现错误。
新诊断器的candidate/身份、邻位、sampler coverage与partial re-prefill四类风险已
本地修正；53新增tests、304相关tests/1skip及153源隔离导入通过，旧147源不变。
原review在`evidence/public-head-hidden-diagnostic-20261102/code-review/`；结论和
修正状态见[结果报告](public-quality-source-recovery-results.md)。尚未同步或提交新GPU
诊断，不称strict/迁移/收益通过；下一步共同prefix hidden/head归因，研究目标未完成。

**2026-11-02 02:47+08最新结果**：276677已11月1日19:22:36 FAILED/1:0，两A800
执行1h44m40s。131raw/两日志SHA、147源及六布局重建通过。四worker证实FP32 logits
输出和BF16权重，但两balanced完整匹配、四corank均Conv分歧：Conv1首分歧53，
Conv0首分歧442，Code全匹配。仅改变head输出dtype仍不足；原strict错误保留，
迁移阶段没有执行，失败性能不计收益。下一控制保存首次共同prefix的hidden、候选
head权重与原始logits，用离线高精度dot分离hidden与head计算的贡献，尚未证明具体
kernel根因。02:44队列空，未新提交；本地监控大wallgap/SSH失败正独立审计，不
因monotonic约300或caffeinate存在而声称连续。公开收益/heldout目标仍未完成。

**2026-11-01 16:55+08监控更新**：276677仍PENDING/Priority，尚未获配GPU。主六轮
至16:54:49均SSH0、间隔300.0026–300.0289s；独立五轮至16:52:50均SSH0、
monotonic299.9926–300.0111s，147源/主身份一致。detached主与独立observer持续，
agent已交接关闭但observer不停止；独立有界至22:32:50或终态，不停止主/remote。
不把排队或后台运行当结果，不重复申请。正确性→新head成本→公开收益的门槛不变。

**2026-11-01 16:31+08执行状态**：唯一两A800276677正在PENDING/Priority（首主poll
16:29:49，SSH0）。[FP32 logits-head对照](../docs/fp32-head-diagnostic.md)已6源保护
同步，147源隔离导入/冻结、251CPU回归/1skip通过。只改变head输出精度，BF16权重、
default sparse、原公开P/O与到达不变；新serial同配置，四worker强制核验有效dtype及
实际FP32输出。六静态完整tokens通过才同allocation跑三策略真实迁移；失败就保存
原strict错误停止。主61604/caff61605与remote2635253/独立agent继续300s监控，结束
自动SHA与独立重建。当前尚无新GPU结果/收益；后续需重校准新head成本再恢复公开
七策略/heldout，不能使用旧144成本或失败timings宣称新配置收益。证据
`evidence/public-fp32-head-diagnostic-20261101/`，不发布/Aegaeon。

**2026-11-01 16:17+08最新状态**：276017已14:46:40 COMPLETED/0:0，112raw与两job
日志全SHA、六布局与共同prefix独立重建通过；原strict错误仍保留。两balanced重复
完整匹配serial，四个同rank零copy/零pause布局均Conv分歧、Code一致。故布局改变
无需迁移就能复现；具体数值根因尚未证明，失败性能不计收益/不选setting。Conv0
首分歧130/139处serial候选margin .25、实际tie，下一轮只验证内置FP32 logits-head，
BF16权重/default sparse及原trace不变，新配置重建全部serial，四rank核验有效dtype。
完整正确性通过才恢复迁移/公开trace，旧默认head成本需重新验证，研究目标未完成。
16:17:30新SSH队列空；证据`evidence/public-static-batch-diagnostic-20261101/`。
以下12:44 PENDING等记录均为历史状态。

**2026-11-01 12:44+08当前状态**：274385已05:22:53 FAILED/1:0，265raw/两日志
SHA独立复核，144成本校准六profile重建与保存fit一致。首native四策略12请求完整
drain，但三个迁移策略的六Conv与serial完整tokens不一致；no_move三请求及全部Code
一致。未迁移stayer也分歧，四device bytewise echo通过不能当推理正确性证据；该轮
性能不用于选择setting/收益。后续native/rate/七策略/heldout未跑，root cause未证明。
详见[失败诊断与恢复](public-quality-source-recovery-results.md)。

唯一新276017为[静态同rank零复制对照](../docs/static-batch-diagnostic.md)：原P/O/
到达/900s窗口、同worker/BF16/default sparse/128K/B10240，六个镜像布局+完整top2
logprobs，无pause/copy，不是收益实验。33专项、299相关/5skip及200部署/1skip通过；
额外广回归有一未改wait_forecast失败独跑仍复现，不称全套通过、不修无关模块。
8源保护同步/四输入双SHA复用、141实际隔离import及冻结通过。12:44:31仍PENDING/
Priority/SSH0，最近三间隔约300s；独立侧车19934两轮299.951s实查，主19117/19119/
caff19118及remote1132067持续，终态SHA/full-token/logprob/ownership/B/四rank graph
自动独立核验。公共A800空闲0，hp_a800包装空闲16但实际partition not found，不改
QOS/重提；SlurmNov1 21:38仅暂估。证据`evidence/public-static-batch-diagnostic-20261101/`。
用户已取消4倍，三倍机制结论保留；公开数据收益与最终最优setting目标仍未完成。
不commit/push/Aegaeon；先定位正确性再接续收益，以下RUNNING记录仅为历史状态。

**2026-11-01 01:45+08执行更新**：274385已01:35:33在d1n41a26g03获配两A800，
01:40:55 RUNNING/SSH0，正在公开源cost校准/加载checkpoint。compute原输入及六
trace验证、134启动/193嵌套tests实见通过；尚非GPU成本/native/收益通过。主29625/
caff29626、remote4050940与local-only30625持续300s；133runtime/176outer冻结。
原OpenThoughts3原响应/SWE-smith100K，用户取消4倍；同job native/rate过关才七
策略/heldout，root=public-efficient-optimizer-chain-20261101-r2。公开收益未完成。

**2026-11-01 01:12+08最新执行**：新唯一两A800274385于01:10:41提交，01:11:46
独立核查PENDING/Priority，无GPU结果。995CPU回归、144源保护同步及四immutable
输入双SHA复用通过；main29625/caff29626/remote4050940，133runtime/176outer冻结，
300s持续。用户跳过4倍，按原OpenThoughts3/SWE-smith100K native/rate→合格才七
策略/heldout接续；证据`evidence/public-efficient-optimizer-chain-20261101-r2/`。
旧274238零推理入口失败及3倍机制正/负指标保留，公开收益目标未完成，不发布。

**2026-11-01 01:04+08最新状态**：用户跳过4倍Conv，继续公开dataset。三倍机制
268245三seed已闭合，585raw SHA/36requests重新独立复核；Conv TPOT vs随机
改善22.42%，vs no_move退化18.77%；Code avg TTFT vs no_move改善72.59%，不能
当最终Poisson/P99泛化。274238在00:42:30入口立即FAILED，零GPU请求；4raw/两
日志SHA通过。已隔离stdlib计算入口，search/poisson实际同步集合29tests通过，
旧源/失败保留；准备唯一A800新链，证据
`evidence/public-efficient-optimizer-chain-20261101-r2/`。原OpenThoughts3原响应+
SWE-smith100K/原算法图预算，native/rate过关才七策略/heldout；公开收益未完成。

**2026-11-01 00:20+08当前状态**：273918已10-31 23:59:34 COMPLETED/0:0；163raw
canonical SHA及compute/local14强控制/8事务/8迁移hop/四rank graph/两plain6requests
全部完整tokens通过，父producer退出。独立两no_move容量见证：每GPU mapped1728MiB，
Code新增最低9504MiB，对B10240倍率1.096875，总free17024>=logical9534；Code queue
96.86/96.74s、TTFT136.05/136.40s，Conv TTFT.109/.112s、TPOT12.069/12.091ms。
每轮仅1Code/2Conv，证明受控物理预算局部阻塞，不是回迁/Poisson tail或50%收益。
新opt-in接续入口21新tests及55相关通过，不改canonical，独立重建执行launch/solver
证书；9257/caff9258已清除闭合门槛，新唯一274238在135文件保护同步/4输入SHA复用/
987CPUtests通过后提交。00:16:54 PENDING/Priority/SSH0，remote3125763/主300s持续；
当前124runtime/外层167冻结，获配六native/rate→合格七策略/heldout。5738正常终态
退出，旧10044只读监控schema错误保留，新22693于00:15:18/00:20:18两轮接续/300.033s
通过，无新告警；416.05s交接gap不改写为零。侧车到03:30，独立主monitor继续。证据
`evidence/public-efficient-optimizer-chain-20261031/`；容量明细在[当前结果](public-quality-source-recovery-results.md)。
公开最好setting/收益仍未完成，不commit/push/Aegaeon。排队不是新GPU推理或收益通过。

**2026-10-31 22:23+08恢复快照**：11源码18450bytes保护同步、13immutable frozen逐一
SHA复用完成，唯一273918已提交，22:21:39主/22:22:00独立均PENDING/Priority。
main5222/caff5223 PPID1、remote2742712；local-only5738两轮300.022s实查持续。
早期sidecar声明校验失败退出已保留。1004广回归及18独立verifier通过，125源冻结，
尚无GPU控制/native/新收益。保持本作业14强控制→两plain no_move→canonical终态
SHA/独立核验；通过后才部署等价solver→native/rate/七策略/heldout，不在提交处停步。
公开最好setting/总研究目标未完成，不commit/push/Aegaeon。

下一轮部署/正确性/native/Poisson准入及稀疏Code tail样本限制，统一见
[等价求解器实测门槛](../docs/equivalent-solver-live-protocol.md)；此协议不改变活动273918。

**2026-10-31 22:17+08恢复快照**：273701于21:30:24 FAILED/1:0，首noop在pause前因
漏显式dp_echo拒绝，未发生迁移；四serial完整token一致，44raw全SHA及独立诊断封存。
R2首次上传超时但未install/提交，旧producer退出/残包保留；新恢复先后SHA复用远端
immutable frozen，仅传11源+trace。52专项通过，125冻结源未变，目前prepared/无新GPU。
下一步唯一R2的14强控制→两plain no_move→canonical归档/独立验证，再新solver/
native/rate/七策略/heldout；公开收益及最好setting仍未完成，原272106失败不改写。
详见[失败与恢复记录](public-quality-source-recovery-results.md)，证据根
`evidence/public-roundtrip-state-control-20261031-r2/`。不commit/push/Aegaeon。

**2026-10-31 21:20+08历史快照**：唯一273701已21:18:19获配两A800/d1n41a14g01，
21:18:43 RUNNING/强控制启动，尚无14控制或plain native通过。原124/136 source SHA
零变化，主300s监控与local-only96441/PPID1侧车两轮实查持续。478范围CPU回归通过；
实际旧D3同1s预算full typed-controller+独立certifier三轮均接受1回迁，目标值仍
144.8849747547，不是GPU收益，未部署。继续当前控制→两原生no_move→完整SHA/
独立核验，再新optimizer/native/rate/七策略/heldout；总目标与最好公开setting未完成。

**2026-10-31 21:04+08历史快照**：272931于18:00:06 COMPLETED/0:0，63rawfiles全SHA及
独立数值重建通过，两次加logprobs2的no_move均完整token一致；未复现不等于证明根因，
272106原失败仍保留。接续唯一A800强控制：原公开输出长度14-case串行/noop/暂停往返，
通过独立核验后同allocation跑两次不加logprobs的原生no_move；10新文件保护同步完成，
唯一273701于21:08:43 PENDING/Priority，尚无新控制或收益结果。main95287/caff95288
已PPID1/remote2074883；32新CPU控制、109专项/460范围回归通过。
通用等价ILP分组219→1、非binding headroom9→3，同1s预算3/3离线得到原gate接受回迁，
不降风险/阈值，新增独立等价certifier/严格trace入口20tests通过，尚未部署/测GPU。
下一步正确性屏障后新optimizer→公开native/rate→
七策略Poisson/heldout；研究未完成，不commit/push/Aegaeon。详见
[当前恢复结果](public-quality-source-recovery-results.md)和
[强控制协议](../docs/roundtrip-state-control.md)，证据
`evidence/public-roundtrip-state-control-20261031/`。

**16:51+08历史快照**：272106于16:19:43 FAILED/1:0，257rawfiles全SHA归档，144成本
通过但首native完整greedy token校验失败，后续rate/grid/heldout未跑。零迁移no_move的
Conv0在token139/3.278602s就与串行reference分歧，早于Code90s，不直接归咎KV迁移，
也不放宽严格校验；失败timing不用来挑最佳setting。唯一272931两A800独立数值诊断，
16:46:09 PENDING/Priority，两次原no_move仅加logprobs2；每300s监控、终态SHA独立
重建，再做强正确性控制。诊断不是收益通过，公开原输出回迁/最好setting仍未完成。
详见[失败与恢复记录](public-quality-source-recovery-results.md)及
[数值诊断协议](../docs/batch-numerics-diagnostic.md)，最新证据
`evidence/public-batch-numerics-probe-20261031/`。以下时间点为历史快照。

快照日期：**2026-10-31，Asia/Shanghai**。这是当前研究的统一入口；各轮协议、
完整指标及失败记录仍以对应分篇报告为准。**容量迁移收益已有实测见证；长输出N2
三seed机制核验及native robust两seed验证已闭合，最终Poisson泛化未闭合。1.2压力实验及新增随机迁移基线的六策略
对照均已在两A800完成并独立验证；研究目标仍未完成。**

**2026-10-31公开原始响应更新**：271324于10:07:14 FAILED，实际144成本与首native
四策略已完整drain；失败在校验器把公开Conv误当SWE轨迹索引traj_id。255raw SHA及修复后
12请求独立复核通过，但全部0迁移，no_move Conv在Code90s到达前已结束，不能称回迁收益。
完整指标/明确失败恢复见[公开源恢复结果](public-quality-source-recovery-results.md)。新增独立
[OpenThoughts3质量cohort profile](public-quality-response-profile.md)：只按原始
think结构质量选6,515/20,000条，不按长度/性能选；最低原selection-SHA bank256
从原parquet全量重新分词核验，Pmean85.527/Omean13210.926、O P95/P99=16708/16751，
Code10原100K/O64不变。较长原响应提供更合适的长驻留候选，不是人工延长输出，
但不是普通真人Conv或naturalEOS。新[独立接续协议](../docs/public-quality-return-study.md)
原C0失败不改写成功；显式诊断/SHA闭合后独立C1重新实测service/rate→七策略有限grid→固定held-out；
仍无C1 GPU收益或最好setting结论，不commit/push/Aegaeon。

**13:09+08续跑**：唯一272106已在两A800/d1n41a22g03运行；主13:07:21 SSH0确认
成本40/144，尚未完成C1 native或收益对照。136源/13冻结输入配置trace逐SHA零变化，
原300s监控及独立本地侧车继续；不重复提交、不覆盖活动runtime。原自动链依次实测
六native、三档rate，独立闭合及qualified后才七策略有限grid/固定heldout。
固定稀疏Code在exploration measurement仅1/2样本；P99仍只作描述，不据heldout到达
预览调整setting或seed。当前进度与证据见[恢复报告](public-quality-source-recovery-results.md)。

**15:20+08校准闭合**：272106的144成本已exit0，同allocation进入首N2 native。
59raw+2上层fit sidecars共61文件SHA/六profiles独立重建通过，浮点仅既有tolerance一致。
Conv16K N2集中→均衡的拟合cadence差8.960ms含5.348ms sparse-graph项和3.612ms KV-work项；
N3/N4最坏holdout残差高于拟合差，不能把系数当kernel attention占比或直接宣称收益。
仍保持guard和冻结算法；首原生四策/真实容量/迁移回迁及Poisson结论待独立闭合。

**2026-10-30 UTC公开原始输出接续**：用户要求不再人工延长AzureConv。旧synthetic
Poisson/log-alias producers已PAUSE退出，无正式job。269991于17:22:07 UTC完成，
264raw SHA/75请求独立通过，但三个真实pressure1.302601/1.325843/1.348774均不满足
预声明<=1.3，无合格rate；不放宽门槛或复用旧service给新数据集。
[公开原始response筛选](public-response-profile.md)比较WildChat/LongWriter/OpenThoughts3。
真实WildChat输出mean432/max4055偏短，LongWriter>=24k仅4/6000；首候选为
OpenThoughts3的完整短输入分布（公开长推理，**不称真人普通Conv**、源generation cap
尚未核实）。64条outcome-independent hash样本从原parquet独立重建messages/P/O/
token IDs/SHA通过，无padding/截断/人工延长；ignore_eos只重放源长度，不称Qwen自然EOS。
新[有限搜索协议](../docs/public-response-workload.md)先N2/N3/N4/144成本profile与两seed
四策略，再按新实测service冻结三档3600s开放环校准，最后七策略有限grid与held-out。
证据root `evidence/public-workload-search-20261031/`是固定输入ID；当前39专项通过，
完整CPU与保护同步准备中，**尚无公开新GPU结果或最佳setting收益**，无commit/push。

## 1. 科研 idea 与实际实现范围

**21:18完整native结果与校准接续**：269567于20:48:50 COMPLETED/0:0，292rawfiles
SHA及两个seed共12请求独立核验，报告consumer21:00:55闭合。另不调用repo指标函数、
从raw逐请求复算所有pooled字段完全一致。完整指标在
[native robust结果](workaware-robust-native-results.md)：每策略Code2/Conv4，Conv
avgTPOT17.8188→14.4359ms（降低18.985%），avgTTFT .22925→.23101s（退化.767%），
P95/P99 .29200/.29513→.27497/.27729s（降低5.830/6.045%）；Code avgTTFT55.8160→
55.5110s仅降低.546%，combined18.7582→18.6577s降低.536%。Code TPOT退化4.529%。
初始TTFT发生于回迁之前，不能归因于回迁；小样本不是P99显著性、旧point独特收益
或最终Poisson。两个seed各一次真实回迁，释放后.001128/.000781s启动，start→resume
5.021/6.220s，copy transaction .130/.119s，剩余时间不能当纯KV复制；两组偿还共
排除4个缺共同pre-return ordinal的span，空sum0不是实测零收益，不延迟回迁补锚点。
独立crosscheck与成本在native-r2 evidence `recovery-report-crosscheck-20261030.json`。
完整父屏障之后114文件保护同步，唯一开放环校准269991于21:05:02在两A800/
d1n41a24g01执行rate-0，26compute启动tests通过；依次实测三档3600s，按预声明
physical pressure/queue screen而非策略收益选rate。final producer仍等待独立校准，
无正式trace/job。仅为当前日志前缀补同inode别名及未来一次性运维接续（16专项），
不修改冻结算法/配置，不发布。旧时间快照保留为历史，不当成当前活动PID/作业。

**20:10 native首seed独立结果**：269567第一phase138raw SHA/6请求/真实robust
计划验证通过，第二seed仍运行。相对random，Conv avgTPOT17.903→14.444ms（降低19.32%），
Conv avgTTFT .2409→.2682s（退化11.35%），P95/P99 .2904/.2948→.2769/.2777s
仅2个Conv样本；Code TTFT54.736→54.632s基本不变、Code TPOT退化3.62%。回迁在
Code释放后.00113s启动/5.021s后resume，两Conv无共同pre-return ordinal均排除，
不能把空sum0当实测零收益，也不人为延迟回迁补锚点。初始TTFT发生于回迁之前。
单seed固定N2不是整个native/最终Poisson或robust相对旧point的独特收益结论。
证据`evidence/workaware-robust-native-20261030-r2/closed-native-20261030/`。
修复后的cal r2与七策略final producer已19:47启动/跨300s存活，1561CPU回归通过
（5skip/1既有失败deselect）；依次等完整native与cal核验才保护同步/申请，当前
只有269567 allocation，下一流程和全部负结果/代价继续保留。

**19:43接续纠正**：旧本地74881在同步/申请前发现no_move-only不能带robust CLI，
已停止并保留114旧源/全部draw；cal driver改point/default，不改变native robust
或算法，真实CLI合同和新阶段57专项通过，完整回归中。r2复用同3draw，不重抽。
进一步准备[长输出七策略Poisson协议](../docs/long-conv-poisson-evaluation.md)，cal
选rate经独立raw重建才冻结两7200s trace：普通/own_model/KV-only/fair30/random/
headroom/workaware，同两A800请求36h，118源码另外冻结。现在没有新增GPUjob
或新native收益结论，最终trace尚未生成；下方旧PID/运行状态只代表历史时间快照。

**19:19接续准备**：269567首native workaware仍运行，记录2moves而非已核验收益。
新增[长输出开放环校准协议](../docs/long-conv-open-loop-calibration.md)：no_move三档
Conv .00565/.0064/.0072、Code .001、3600s含600s自然warmup，固定同Code Poisson
流，每档Code2/Conv21、22、26；不是最终低Code rate或P99确认。114源码与所有输入
独立冻结，32专项/1536回归通过（5skip/1既有失败deselect）。本地74881只读旧状态，
等native SHA/独立核验/报告闭合才唯一保护同步与申请新两A800；现在没有新增job。
根据单基线实测physical pressure/queue/service而非收益选rate，缺证据或无候选
如实保留；校准之后须另外冻结普通/fair/random/headroom/workaware最终trace。

**2026-10-30 19:02更新**：旧268245完整585raw/36请求/112校准case独立核验；
三seed pooled Conv TPOT相对random/headroom改善22.42/22.61%，但比no_move退化
18.77%。Code avgTTFT改善72.59%来自容量准入；Conv初始TTFT相对random退化
19.24/24.89/22.83%，不是回迁效应。第三seed两Conv因缺共同pre-return ordinal
排除，不能将空sum0算零收益，完整分项在[旧probe报告](long-conv-return-probe-results.md)。
13文件接入main、旧源码/manifest保留，108文件保护同步后唯一269567于18:55:09
在两A800/d1n41a28g01执行native robust两seed对照，不是旧CPU重放或已完成结果。
1488CPU回归通过/5skip/1既有wait_forecast失败deselect；本地与远端300s监控，
新增只读终态consumer待完整SHA和独立核验才落盘汇总/暂停偿还。无commit/push，
下一步开放环service/KV/queue校准后冻结普通/fair/random/headroom/workaware Poisson。
恢复根`evidence/workaware-robust-native-20261030-r2/`；下方时间快照保留为历史记录。

**16:30更新：前两seed回迁收益均通过独立核验，第三seed运行中。** 两phase337raw
SHA/24请求闭合；pooled Conv TPOT workaware14.302ms/random18.280ms/headroom18.329ms，
分别改善21.76/21.97%，但相对no_move12.123ms退化17.97%。Code avgTTFT205.080→
55.603s（改善72.89%）来自准入；相对random55.578s几乎相同，不能归因回迁。Conv
初始TTFT相对random也更差；所有TTFT均在回迁之前。每seed性能/完整avg/P95/P99
及代价只在[当前probe报告](long-conv-return-probe-results.md)，不以两seed代替最终
Poisson或尾统计显著性。robust-J仍只本地，前两seed8/旧阶段38重放通过但没有新
guard-passing机会，814CPU回归通过，原GPU源码冻结不变。

**10月30日首seed回迁机制已有独立实测见证**：268245仍两A800运行第二seed；
首seed168raw SHA/四策略12请求完整核验通过。Workaware Conv avgTPOT14.338ms，
比random18.808ms改善23.77%、比headroom改善22.99%；回迁在Code释放0.116s后
启动、4.313s后resume，同token锚点/暂停计入的剩余span合计省217.410 request-s，
约9.415s达到持续观测偿还。Code TTFT相对no_move改善73.12%来自准入，不是回迁；
Conv TPOT相对no_move仍退化19.03%，Conv初始TTFT也变差。只有1Code/2Conv，
不能声称P99统计显著或最终Poisson完成。完整指标与归因边界见
[长输出probe结果](long-conv-return-probe-results.md)。GPU仍旧point-optimal算法，
新robust-J接口仅本地验证，不能把本seed收益归因尚未部署的新算法。

**10月30日低Code六策略完整phase也没有及时回迁收益**：265356于10:06:57结束，
阶段286raw SHA/1806请求独立通过；Code avgTTFT workaware56.313s/random57.006s，
Conv avgTPOT18.070/17.981ms。唯一回迁晚于第二Code267.991s，随机基线17.89/43.77s
已经自然平衡，不归因小TTFT差异于回迁。10:33:36完整761raw及两Slurm日志SHA、
2094正式请求核验通过，长输出流程已通过此屏障进入提交前回归。全部分项尾指标与监控长gap见
[独立结果](workaware-early-return-results.md)，负结果保留。

**用户选择的下一小步**：先把AzureConv合成输出8192→24576、并发降到两条，做
[长输出回迁机制probe](../docs/long-conv-return-probe.md)，不是先换AzureCode。
265356已完整终态核验；93文件保护同步/468回归通过，新唯一两A800作业268245于
10:42:16提交，10:49:28获配d1n41a21g03两A800，13:22:55为RUNNING/probe-20261030；
112case长context校准已13:03闭合，59raw前后SHA/压缩包SHA及本地独立重建通过。
Conv N2原始稳定decode速率失衡损失36.01–43.09%，N3为-0.06–3.93%，含默认sparse
graph影响，不是纯attention核时间。差分拟合有超过原误差guard的机会，但不是复制后净收益；
85runtime/test及六候选/follow-up冻结不变，三seed四策略继续，首seed闭合而三seed结论未完成，见
[当前长输出probe状态](long-conv-return-probe-results.md)。

新增[计入暂停的回迁偿还独立验证](../docs/return-repayment-validation.md)：本地
23960/23961只读原主poll，每300s发现闭合seed后SHA归档/全trace核验/成本分析，
不查询Slurm或修改GPU算法。67专项/607完整CPU回归通过；审计发现并修复同刻
Code释放重复计账，保留失败日志。五新分析文件单独冻结，首seed净收益与负代价已记录，
三seed仍待完整结果。

2026-10-30本地新增[robust回迁优化候选与独立结果](workaware-robust-candidate-results.md)：
将既有误差界纳入ILP目标，增加no-move优势上界快速否决与限时fallback，不降低guard。
旧18状态/9headroom完整枚举与候选目标18/18一致，但仍无新增及时回迁机会，不能称
GPU收益。72专项/540含旧冻结测试通过；新终态follow-up只在完整GPU SHA与独立
核验后重放三probe，不覆盖活动版本、不自动申请GPU。
新112case长context独立N2/N3校准→三seed四策略N2 probe，不覆盖旧fit、不改活动
算法或降低guard。该closed fixture验证回迁时点与实际代价，不是最终Poisson收益；
后续开放环rate约.00565只是待实测校准prior。没有新的GPU收益结论或发布。

**17:46独立完成的首配对仍为负结果**：265356原4800s random/workaware对照222raw
SHA、288请求全部核验。Conv avg/P95/P99 TTFT退化14.288/6.519/21.034%，avgTPOT仅
改善.288%；Code平均TTFT退化4.866%。唯一回迁在第二Code完成后556.778s，是后来
背景失衡，远晚于random42.58/43.28s自然恢复；1.647%逐token子集不算净/因果收益。
最坏Conv7.123s来自首Code之前6.614s dispatchlag，根因尚未证明，不盲改已有to_thread。
低Code9600s六策略已同allocation自动开始，冻结源码不变；完整本轮收益目标未完成。
见[完整分项指标与证据](workaware-early-return-results.md)。

**2026-10-29 14:59最新进展**：唯一265356于14:37:42在d1n41a21g03获配两A800，
已RUNNING/cache-fixed-original，首random trace完成8/144。85文件保护同步成功，
444本地回归及50compute启动通过，78冻结源SHA未变；复用raw/anchor在Slurm核验，
闭合95文件校准及140文件composite分别SHA/独立核验通过，不是新的收益结果。
原4800s双策略→低Code9600s六策略自动接续，主/remote每300s监控、闭合阶段即归档
分析、终态独立return审计已启动。sidecar首次目录缺失失败及修复后近期三轮持续均
保留。未覆盖runtime、未commit/push/Aegaeon，完整新回迁收益尚未完成。
见[本轮状态与回迁诊断](workaware-early-return-results.md)。

**同作业后续快照17:17**：原配对已切到workaware，完成111/144、记录4次迁移；
配对仍未闭合，不能据迁移计数宣称回迁收益。新增逐token共同Code空窗诊断已本地
回归，旧260757的6.605%只用于工具smoke且仍有cache混淆。另准备
[长Conv/低人口/约1.2压力的后备设计](../docs/long-conv-low-population-return-plan.md)，
全源bankCPU算术及112case容量屏通过，但尚未选用、同步或GPU运行；本轮冻结配置不变。

**2026-10-29提交前诊断**：262969于03:17:44 FAILED/1:0，252GPU校准case已完成，
但N5/6/7全落在graph8，三参数全局设计rank2，未进入任何收益trace。71raw及两
Slurm日志独立SHA归档，原负结果保留。修为显式锚定旧验证系数的可搬迁派生bundle，
旧系数/10旧profile不变，新6profile完整raw/heldout验证通过。下一唯一A800直接
复用这些GPU数据并在Slurm重新核验，接续原两策略与低Code六策略，不重跑252case。
25专项通过，尚无重试收益。实际count模型仅离线repeat诊断改善，未安装/未作GPU
收益结论。本地长wallgap/远端连续与归档保障均记录，见
[失败诊断与复用状态](workaware-early-return-results.md)。无commit/push/Aegaeon。

**历史运行快照（2026-10-28晚，现已失败归档）**：先统一修复各策略的完成/释放缓存观测，保留同tenant
两rank快路径；再补N5/N6/N7全部DP划分与held-out repeat/context/layout校准，保持
旧fit独立可复核。新pipeline接续旧4800s random/workaware对照，再跑Code.00025RPS、
Conv不变的9600s六策略trace（2Code/299Conv）。保留暂停迁移/default32/dp_echo，
不以overlap为前置。不把260757的缓存刷新/rank方向差异算作独特回迁收益。此时
435项CPU部署回归通过、83文件保护同步通过；唯一262969于21:45:43获配两A800，
22:24:59恢复SSH确认校准25/252，新增compute启动41通过；76个冻结文件SHA未变。
主/独立侧监控近期300s持续，未覆盖活动runtime或重复申请。尚无本轮校准/收益完成结论，见
上述next-calibration文档。客户端timeout统一1800s，SLO仍120s，超时不得当完成请求。
22:43主查询推进42/252/SSH0，76冻结文件22:42:54 SHA未变。新增本地只读return审计38项回归通过，
等完整SHA归档/独立核验后自动对齐Code完成、回迁、基线自然恢复及matched Conv
尾延迟；不覆盖活动runtime，也不把回迁前已发生的TTFT差异作为回迁收益，见
[本轮状态与回迁诊断](workaware-early-return-results.md)。

**最新完整结果（260757，2026-10-28 19:07:47结束）**：493rawfiles与两份Slurm
日志SHA通过，6×144=864正式请求及39辅助请求独立核验；实测压力1.198629529。
主窗600..4800s，TTFT秒，Conv TPOT毫秒：

| 策略 | Code avg/P95/P99 TTFT | Conv avg/P95/P99 TTFT | Conv avgTPOT | 合并 avg/P95/P99 TTFT |
|---|---|---|---:|---|
| no_move |374.013/661.129/686.650|.415/.818/1.194|18.879|6.490/.958/43.433|
| no_move_fair |185.330/302.830/313.274|7.877/64.808/146.097|18.666|10.762/91.450/151.188|
| random_no_return |69.778/83.261/84.460|.411/.634/1.261|19.019|1.539/.842/43.758|
| kv_only |69.655/82.931/84.111|.411/.648/1.091|18.410|1.537/.718/43.814|
| kv_only_headroom |71.287/86.253/87.583|.406/.628/.922|18.427|1.559/.677/43.564|
| kv_only_workaware |71.050/85.597/86.890|.402/.669/.951|18.429|1.551/.765/43.589|

Code对no_move的平均TTFT改善81.003%，但random也获得同类容量收益；workaware
对random的Code平均TTFT反而退化1.824%。对headroom的Conv TPOT几乎相同，
P95/P99退化6.429%/3.096%，没有做出独特回迁收益。ConvP99端点早于所有迁移，
不能将其差异归因于回迁；Code只有两条，混合P99还会插值排除最大Code等待。
no_move全窗启动阶段存在早期Conv排队，不能拿该瞬态放大主场景收益。完整/主窗、
容量见证、时序及负结果见 [random-baseline-results](random-baseline-results.md)。

终态后已本地合入精确ILP编译复用及random/full-reservation统计修复，198专项回归与
新verifier复核通过；未经GPU同步/实跑，不把该CPU优化归因于上述结果。下一阶段
优先 [缺失人口、联合phase与真实控制成本校准](../docs/workaware-phase-calibration-plan.md)，
不盲调已达1.2的RPS，也不降低误差门槛强行产生return。没有新commit/push。

2026-10-28 早期检查点：新增随机可行 DP、跨模型迁移但不主动回迁的简单基线；
实现/接续状态见 [random-baseline-results](random-baseline-results.md)，
公平对照见 [random-baseline-protocol](../docs/random-baseline-protocol.md)。
259091已结束并验证，新代码经保护晋升/同步至main；260757于2026-10-28 09:50:09
获配两A800。N2三策略9请求/147文件已SHA归档并独立核验，Conv平均TPOT对random
改善14.21%，但不是tailTTFT收益。N9三策略30请求/147文件也已独立通过，未发生回迁。
14:12:44确认第三策略no_move_fair进行中，66offered/61completed。前两项正式4800s
case共288请求/132文件已前后SHA及显式子集独立核验；未伪造根exit/summary或修改
原六策略配置，完整allocation尚未终态。全窗ConvP99 random1.169403→workaware.926097s
（-20.806%），Conv avgTPOT18.794237→18.303054ms（-2.614%）；Code avgTTFT
69.777668→71.050235s（+1.824%）。但ConvP99端点均早于所有迁移，不能归因于
回迁；回迁时四条老Conv的整请求TPOT反而退化3.563%。保持负结果，不能声称已做出
稳定独特的tail收益。完整及600..4800指标、控制拆分与时序见随机baseline分篇报告。

两个常驻模型共享 GPU 物理 KV 空间，但各自的 attention-DP 调度独立。短输入、长输出
Conv 请求持续占用 KV；稀疏到达的 Code 请求输入约 100K、输出短。即使两卡总空闲足够，
count load balance 也可能让两张卡都无法容纳单条长 Code 请求。通过移动包括另一个
模型在内的请求/KV-home，把零散空闲集中到一张卡，使新请求无需等自然释放。

Code 准入后，Conv 可能被挤到一个 attention rank，产生 graph padding 与 attention
失衡代价。Code 释放后，再判断迁移部分 Conv 回另一个 rank 的收益是否超过真实
暂停、复制和未来长请求再次准入的代价，不强制回到历史 home 或完美均衡。

当前 data plane 是 **native vLLM + kvcached**，不是 Aegaeon 的模型切换。
两独立 Qwen3-30B-A3B tenant 复用 checkpoint 文件，但权重、KV、request state、
EP 组独立；每模型 DP=2、EP=2、attention TP=1。研究只调整 request/KV-home placement。

相关设计：[placement-design](../docs/placement-design.md)、
[最初 staged ILP](../docs/ilp-scheduler-design.md)、
[当前 work-aware 回迁协议](../docs/workaware-return-protocol.md)。

## 2. Motivation 与可使用的性能关系

单模型 A800、默认 sparse CUDA graph 的确认实验（229748）固定请求数和长度，
仅改变 DP home。全局 batch128 的极端失衡吞吐损失为 **32.37% / 32.21%**；
batch256 为 **38.52% / 38.35%**，分别对应两个热 rank 方向。batch32 未通过预声明
30% 确认门槛。它们是稳定 decode cadence-derived loss，不是纯 attention kernel 占比。

空 attention rank 仍参与 EP/collective 和 graph replay；不能说其整卡完全 idle。
sparse capture 阶跃也影响性能。因此，中间不均衡曲线不支持对所有 batch、capture 和
模型使用一个普适线性系数。当前算法在**固定 tenant、population 和实测域**内使用：

`c_m,N = base_m,N + gamma_m,N(G) + beta_m,N * abs(A0-A1)/2`

- `G = sparse_ceiling(max(n0,n1))`，captures 为 `1,8,16,32,64,128,256`。
- `A_d = sum(context_tokens)`，按未来 decode 区间中点计算该 rank 的上下文工作量。
- `gamma` 是同 population 的实测 capture 分类 penalty；`beta` 是非负线性上下文
  失衡系数。比较同批请求的布局时，总上下文不变，公共项抵消。
- 对偶数 N、等长上下文 L，令 `I=abs(n0-n1)/N`，则 attention 失衡项为
  `beta_m,N*N*L*I/2`；同时仍须保留 `gamma` 的阶跃，不能仅拟合一个 `alpha*I`。

例：Conv N2 的 `gamma(8)=5.1994ms`、`beta=1.23197e-7s/token`；
N9 的 `gamma(16)=.9102ms`、`beta=5.78439e-8s/token`，各自均衡档 penalty 为零。
held-out-repeat 最大 contrast 误差分别约1.0263ms、3.8300ms；不是统计置信区间。
这些是客户端 decode cadence surrogate，不是 kernel 分解。

当前人口校准覆盖2/3/4/8/9；缺少5/6/7时保守延期。每个 population 的 context、
candidate peak-attention、capture 域均有 guard。另一 tenant 正在 prefill/decode 时
也延期，因为当前是另一模型 resident-idle 校准，不是完整 joint-active 校准。
新的 actual-attention-count 成本分析仅为离线诊断，**未部署**。

完整证据：[motivation-summary](motivation-summary.md)、[curve-results](curve-results.md)、
[当前校准与回迁结果](workaware-return-results.md)。实现：
`workaware_profile_cost.py`、`verify_workaware_profile.py`、`workaware_cost.py`。

## 3. 当前算法：准入与性能感知回迁分开

### 3.1 新请求准入：KV-only staged ILP

枚举新请求可用 attention-DP home（本布局为0/1），构建物理页 owner 快照，选择
包括跨 tenant 在内的迁移请求。ILP 同时选择 request placement 和复制阶段，约束：

- 每请求最多迁移一次，迁移候选和阶段数受限；不拆分一条请求到多个 attention home。
- 源页只有全部 owner 安全迁出且页可释放时才释放；固定页/pinned backing 保留。
- 目的页按保守 page rounding 计费；复制先于源释放，检查中间峰值与最终每卡预算。
- allocator 源 owner、实际 mapped bytes、复制完整性及提交前 snapshot 都必须复核。

容量优先、尽量少迁移。没有安全方案就排队，不丢请求。当前新 policy 的准入仍是
KV-only；**没有把未来 decode 性能目标直接用于新请求 prefill 准入**。
ILP 在有限候选集和总求解时间内选可行 incumbent；不能宣称整个系统全局最优，
也不能把保守 ILP 无解/超时说成真实 allocator 无解。

实现入口：`live_controller.py`、`live_scheduler.py`、`ilp_reference.py`、
`peer_transfer.py`、`worker.py`。

### 3.2 释放后回迁：work-aware + soft headroom

只使用已到达请求，按剩余输出完成事件和最多128步积分网格分段。枚举每模型的
headroom 状态 `{不保留, DP0, DP1}`，两个模型最多9个场景；每场景用 staged ILP
选择部分回迁。与 placement 有关的 request-second 目标为：

`min sum_m,k steps_k*N_m,k*[gamma_m,N(G)+beta_m,N*abs(A0-A1)/2]`

`    + N_active*F*z + sum_i N_active*(KV_i/BW)*move_i + R_future`

这里 `z` 是暂停 activation，固定成本只计一次；复制成本和未来区间成本均加入目标。
保留 headroom 的场景有最终容量硬约束；不保留的场景支付软风险：

`R_m = (1-exp(-lambda_m*H))*cold_admission_cost_m`

`lambda` 使用过去真实到达和冻结 rate prior 的 Gamma-Poisson 后验；请求大小使用
过去到达的 P95，cold cost 使用实际控制历史与保守 pause/copy 先验。**不读取未来
trace 的到达时刻来做在线决策**。这是未来准入风险代理，不是精确未来等待预测。

只有预测净收益超过实测误差界加1 request-second margin，且重新观测验证仍安全，
才执行。已支付控制成本不重复扣后续决策收益，但所有失败尝试和真实暂停仍计入
端到端指标。当前最多4次回迁、8候选/pool、9场景共1秒规划预算、10秒cooldown；
准入优先。该目标不等价于严格的逐请求 TTFT/P99 SLO 保护。

实现入口：`workaware_rebalance.py`、`workaware_ilp.py`、
`workaware_profile_cost.py`、`live_rebalance.py`。线上名称为 `kv_only_workaware`。

### 3.3 基线与对照名称

| 名称 | 行为 |
|---|---|
| `no_move` | 不迁移，容量不足自然排队；不主动调整另一模型的放置 |
| `no_move_fair` | 不迁移；排队超过30s时限制另一模型新请求准入直到队首能进入 |
| `own_model` | 历史对照：仅允许调整到达模型自己的请求 |
| `kv_only` | 跨模型容量准入迁移，不做释放后回迁 |
| `ilp` | 历史性能项对照，不等同于当前 `kv_only_workaware` |
| `kv_only_headroom` | KV-only 准入，加保留未来请求 headroom 的回迁控制 |
| `kv_only_workaware` | KV-only 准入，加剩余推理工作/实测失衡成本/软 headroom 风险回迁 |

`no_move` 是同 full-reservation 自定义 scheduler，**不是未修改的 native-vLLM
动态增长 baseline**。GPU 与 KV 物理资源仍共享，“不做跨模型调整”不等于硬件互不影响。

## 4. 已完成的容量收益：有效，但有 Conv 代价

### 较强 Code 收益（251533）

六策略同75请求，各自完整 drain；主窗600..2400s为52Conv/2Code。
KV-only 对 no_move：Code avg TTFT **804.888→55.904s**，约降低93.1%；
Code P99 **822.058→57.095s**。同时 Conv avg **.425→.857s**、
Conv P99 **1.148→12.462s** 退化；不能说所有 tenant 同时改善。
headroom 相对 KV-only 没有证实大幅新增 tail 收益。
完整报告：[headroom-tail-status](headroom-tail-status.md)。

### 降载后的配对对照（254337）

六策略各54请求完整结束；主窗为37Conv/2Code。实际压力 **1.129739**，
不是准确1.1；以下同一 trace 内比较，TTFT 秒，TPOT 毫秒：

| 策略 | Code avg/P95/P99 TTFT | Conv avg/P95/P99 TTFT | Conv avg TPOT | 合并 avg/P95/P99 TTFT |
|---|---|---|---:|---|
| no_move |246.366/262.572/264.013|.347/.474/.508|17.592|12.963/23.310/250.688|
| no_move_fair |146.365/162.782/164.241|3.234/31.634/39.146|见分篇|10.574/49.627/150.743|
| kv_only |55.297/56.021/56.085|.487/1.174/2.384|19.311|3.298/7.714/55.490|

KV-only 对 no_move 的 Code avg/P99 降低77.55%/78.76%，合并 avg/P99 降低
74.56%/77.86%；但 Conv TTFT/TPOT、合并 mean latency 不全面变好。
相对 fair，Code avg 降低62.22%、Conv P99 降低93.91%，不能用更慢的 fair
掩盖相对 no_move 的 Conv 代价。

真实容量见证：首 Code 前两卡 mapped 为 `[2400,1440]MiB`，单条 Code 增量
9504MiB，两卡分别不能 fit，而总 free16640MiB 足够。跨模型迁移一个 Conv 后，
mapped 为 `[3648,9696]MiB`，均≤10240MiB；复制 bytewise 验证通过。Code 平均
排队 **191.372→.939s**，约55s的100K prefill 下限仍存在。
完整报告：[pressure-ratio-status](pressure-ratio-status.md)。

这两轮 Code 都只有2样本；P95/P99 是描述性插值，**不是多seed尾部统计确认**。
两轮 workload/rate/request 集合不同，不用跨轮绝对指标给算法做因果归因。

## 5. 最新完成：性能感知回迁（257484）

2026-10-27 12:46:08–20:27:46，两A800，COMPLETED/0:0。
五策略各98请求完整 drain，共490请求/3,932,800输出token；453 rawfile SHA、
两个日志SHA、完整selected serial references、四rank sparse graph、ownership、
physical预算、ILP ledger replay 独立核验通过。

设置：两 Qwen tenant DP2/EP2/attentionTP1，BF16/131072，24GiB/model/rank虚拟
pool，**10GiB/GPU受控mapped预算**，allocator.98，默认 sparse/pause32；
Code真实SWE-smith100K+64输出/.0005RPS，Conv Azure输入2056..7633+
显式合成8192输出/.01956673845194241RPS。4800s/seed20261024，主窗600..4800s。
Conv 长输出不是 Azure 原始输出分布；本实验不是1M serving或整卡OOM。

主窗84Conv/2Code，TTFT 秒，TPOT 为逐请求均值、毫秒：

| 策略 | Code avg/P95/P99 TTFT | Conv avg/P95/P99 TTFT | Conv avg TPOT | 合并 avg/P95/P99 TTFT |
|---|---|---|---:|---|
| no_move |59.4694/63.4909/63.8484|.3669/.5296/1.1427|17.8378|1.7414/.7441/56.3416|
| no_move_fair |58.6369/61.9502/62.2447|.3739/.5200/1.0954|17.8037|1.7288/.8982/56.0599|
| kv_only |55.5375/55.7803/55.8018|.3825/.5335/1.3357|17.3089|1.6652/.8060/55.3487|
| kv_only_headroom |55.6005/55.7733/55.7887|.3744/.5325/1.1974|17.2739|1.6587/.7646/55.4661|
| kv_only_workaware |55.5615/55.7526/55.7695|.3623/.4981/1.1692|17.0269|1.6460/.9326/55.4130|

新方法对 headroom：Conv avg/P95/P99 TTFT 降3.23%/6.45%/2.36%，Conv avgTPOT
降1.43%；Code几乎不变，合并 P95 **退化21.97%**，Code avgTPOT **退化12.25%**。
不是50%收益，也未证实普遍优于KV-only或独特回迁收益。

原因：普通 baseline 两条Code只等9.3625s/.1989s；第二条到达时另一rank已经为空。
reservation到首token约55s，不是持续堵塞场景。唯一真实回迁发生t3166.359，
control3.826s，晚于两条Code，不能解释其此前TTFT。N2机制试验曾测到ConvTPOT
约降低15.39%，但n2且TTFT退化，不能替代上表的固定Poisson trace结果。

完整报告：[workaware-return-results](workaware-return-results.md)；
下一门槛：[workaware-return-next-phase](../docs/workaware-return-next-phase.md)。

## 6. 已验证：1.2压力目标（259091）

用户要求 baseline 不排队时提高Conv压力。旧no_move实测Conv1739.280628MiB/卡、
完整Code pool9600MiB、单卡预算10240MiB，实际rho=1.107351624。

`rho=(mean Conv physical + Code complete pool)/controlled GPU KV budget`

目标Conv=`1.2*10240-9600=2688MiB/卡`，扣固定Conv96MiB反推：

`lambda_new=.01956673845194241*(2688-96)/(1739.2806284373632-96)`

得到 **.030863253171591748RPS（+57.733%）**。仅改Conv rate；Code仍.0005RPS、
两到达时刻2463.4417805/2663.1432232s不变。保持seed、长度、预算、graph、算法，
同Poisson流新增Conv marks；新trace142Conv/2Code/策略，五策略均须完整drain。
复用已验证cadence校准，跳过重复N2/N9机制阶段，不跳过serving基线。

259091于2026-10-28 05:43:30 COMPLETED/0:0；第三次归档全SHA及五策略720请求
独立核验通过。no_move实测Conv2650.369219MiB/卡、**rho1.196325119**，满足1.2±.02；
aggregate ratio.732262619仅为平均量，不是瞬时安全迁移证书。

完整到达窗142Conv/2Code，TTFT秒、Conv逐请求平均TPOT毫秒：

| 策略 | Code avg/P95/P99 TTFT | Conv avg/P95/P99 TTFT | Conv avg TPOT | 合并 avg/P95/P99 TTFT |
|---|---|---|---:|---|
| no_move |373.1826/659.4469/684.8926|.376883/.548675/.870928|18.5303|5.554740/.621365/31.846188|
| no_move_fair |106.9281/153.7554/157.9178|.669160/.578558/9.549241|18.1808|2.144979/.737218/43.072844|
| kv_only |71.4478/86.2080/87.5200|.403942/.604413/1.245069|18.2820|1.390662/.653554/33.163957|
| kv_only_headroom |70.3363/84.1945/85.4263|.379214/.550875/.973968|18.2331|1.350840/.658839/32.213142|
| kv_only_workaware |71.7474/86.8740/88.2186|.366179/.603125/1.065557|18.1010|1.357585/.742838/31.842297|

预声明600..4800s主窗121Conv/2Code的完整表见分篇，不能拿完整窗替换主窗。
workaware对no_move Code avg/P95/P99改善80.77%/86.83%/87.12%，但这是容量准入收益：
headroom的Code更快。对headroom，完整窗Conv avgTTFT/avgTPOT改善3.44%/.72%，
**Conv P95/P99退化9.48%/9.40%、合并P95退化12.75%**；主窗Conv P95/P99也退化。
Code avgTPOT比no_move退化33.19%。两Code样本不足以支持稳定P99；144合并样本
插值P99排除了最大Code等待，因此合并P99几乎不变，必须分tenant一起看。

第一条Code均可direct-fit；第二条no_move queue636.361s、fair103.790s、三迁移策略
30.941..33.508s。最初三者双目标均无安全证书，初始GiB[8.8125,1.3125]、新请求9.375、
Conv[8,1]；一次自然完成后才有target0七迁/target1一迁证书。不能把此前约30秒
等待直接认定为retry bug，也不能用均值aggregate足够证明allocator一定可迁。
独立枚举每target511子集，初始均无final-memory可行解；最小预算超额224MiB/32MiB，
证明当前保守目的页模型无解，不证明真实allocator无解。回迁在N7/N6/N5均因校准缺失
而延期，到N4才执行；三移/3570MiB不代表更少control，workaware总control15.149s，
headroom13.967s。优先补人口域、核对真实目的页增量，再加入TTFT保护，而非盲调系数。

新增random_no_return不主动回迁，只均匀随机选有安全贪心证书的rank；已在260757
以同trace/预算/图/校准与六策略比较。未来结果须分解随机rank选择与性能回迁的贡献。
两次SSH255断包和本地>600s查询缺口保留；不凭PID宣称整夜300s无间断。

协议：[workaware-pressure-1p2-protocol](../docs/workaware-pressure-1p2-protocol.md)。
结果/进度：[workaware-pressure-1p2-results](workaware-pressure-1p2-results.md)。

## 7. 文档与证据导航

| 内容 | 权威位置 |
|---|---|
| 当前想法、实现、收益与边界 | 本总览 |
| 完整实验矩阵、job/设置/结果索引 | `../docs/experiment-matrix.json` |
| 最初科研idea与placement要求 | `../docs/placement-design.md` |
| 当前回迁算法、门控与校准公式 | `../docs/workaware-return-protocol.md` |
| 当前实验已完成的全窗/主窗各tenant指标 | `workaware-return-results.md` |
| 当前1.2压力实验 | `workaware-pressure-1p2-results.md` |
| 早期32K Poisson trace负结果与控制开销 | `trace-benefit-results.md` |
| Azure/LongBench长度与真实100K输入来源 | `mixed-workload-profile.md`、`agentic-100k-dataset.md` |
| 所有研究导航与执行入口 | `../README.md` |

原始证据存放在外层ParaCloud的 `evidence/`：`headroom-tail-20261025/`、
`pressure-ratio-20261026/`、`workaware-return-20261026/campaign-v2/`、
`workaware-pressure-1p2-20261027/` 等；大raw、checkpoint、原dataset、运行中的
状态文件不随源码提交。报告保留SHA、job与独立核验范围，不把未公开的大raw当成
仓库内已随附文件。运维在外层 `docs/cluster-operations.md`；不对活动cluster checkout
执行pull/reset或覆盖source。发布代码快照不等于完成最终科研收益目标。

## 8. 本次发布前校验与范围

当前研究源码、测试、协议及结果报告一起提交到用户fork的 `main`。两个隔离试验
stage目录只保留在本地，不提交重复源码；外层checkpoint/raw dataset/evidence也不提交。
正在运行的259091使用冻结source，commit/push不触发远端pull或源码替换。

发布前执行：`python3 -m pytest -q tests/test_dp_opt_*.py tests/test_vllm_tp_world_size.py`。
结果为 **1009 passed / 5 skipped / 1 failed**，保留既有 `asyncio_mode` warning。
唯一失败为 `tests/test_dp_opt_wait_forecast.py::test_future_births_can_pin_shared_pages_after_current_owner_completes`
（预计wait1s，实际返回120s）；2026-10-24/25报告已经记录同一旧forecast边界失败。
本次不修复或隐藏该失败、不改活动source；当前实验 `DP_OPT_WAIT_AWARE=0`。
最新archive/profile/pressure专项与扩大回归195项此前全部通过，但不能把它当全仓库全绿。
本次回归证据在外层 `evidence/publication-20261027/cpu-regression.log`。
# 2026-10-30 三 seed 闭合更新

268245于17:53:58结束，两A800、585raw及两joblog SHA、36正式请求与112校准case
独立通过。旧point-workaware的Conv avg TPOT比随机迁移改善22.42%、比headroom
改善22.61%，但比no_move退化18.77%。Code avg TTFT比no_move改善72.59%来自容量
准入而不是回迁；Conv avg/P95/P99 TTFT比随机迁移退化19.24/24.89/22.83%。每策略
仅3Code/6Conv，不能声称P99显著。完整分项指标在
[长输出结果](long-conv-return-probe-results.md)，原始重建为three-seed-metrics-summary.json。

第三seed四个robust离线状态重放也通过，三seed共12个、没有新增guard-passing机会；
不冒充新GPU收益。native adapter/routing/独立验证及合法零tenant Poisson支持目前仅
隔离副本，1485CPUtests/5skip通过，另有一项原源码复现的无关wait_forecast失败明确
排除。须等旧phase分析闭合后才promote/保护同步/唯一下一A800验证；最终Poisson、
公平基线及回迁对开放环tail TTFT的收益仍未完成。
