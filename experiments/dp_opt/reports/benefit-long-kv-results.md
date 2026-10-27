# 长KV放置失衡：容量准入收益

2026-10-21 12:06，Asia/Shanghai。协议见`../docs/benefit-long-kv-protocol.md`。
本场景直接检验总KV预算足够、当前任一卡均不能准入、跨模型迁移后准入；不是只利用
CUDA graph档位改变。首轮和默认32step六配对确认均已实跑、归档并独立验证。

## 首轮237068

实际2026-10-21 **10:48:29–10:57:16**，`d1n41a14g01`两A800-SXM4-80GB，COMPLETED/0:0。
三配对重复、四策略、每trial14请求；**100rawfiles SHA、168请求/86016输出tokens**全部
串行greedy token、page ownership、实际mapped预算、copy峰值及指标复算通过。
两独立Qwen3-30B-A3B tenant均DP2/EP2/attentionTP1；相同checkpoint但独立weights/KV。
sparse capture1,8,16,32,64,128,256/FULL_DECODE_ONLY/compile0，统一预热/interval1。

### 容量见证

初始两模型均count-balanced **A3:3/B3:3**，输入3072/2048、输出512；每卡实际mapped
**1824MiB**，aggregate KV预算**2112MiB/卡**。每卡仅288MiB余量，两卡合计576MiB，
新A3584/512完整预留384MiB。全14请求logical预留**3984MiB < 4224MiB**总预算。

每trial两A rank均真实allocation probe，实际各新增384MiB，使该卡mapped2208MiB，
超过2112MiB预算；rollback还原request/block/page。诊断临时超预算独立标记、在四策略
都执行且计入完整makespan，均值约64–66ms；**不是正常serving已越界或设备CUDA OOM**。

每轮ILP/KV-only都移一条B0→B1，完整KV **240MiB**，copy目的增加288MiB、源释放
2个exclusive page bundle/**192MiB**（B源768→576MiB）。真实copy峰值1824:2112MiB。
新A在rank0准入后A4:3/B2:4，mapped2016:2112MiB；独立新B在+1秒到达并准入rank0，
最终A4:3/B3:4、mapped**2112:2112MiB**，全部14请求同时可容纳。A-only完整迁移目的
需384MiB、超过初始288MiB余量，所以不能在copy-before-release规则下启动；实际
own_model三个repeat均未迁移并排队，no_move也排队，直到原请求完成自然释放。

### 端到端结果

| 策略 | A queue秒 | A TTFT秒 | makespan秒 | 吞吐tokens/s | 平均latency秒 | 20秒SLO请求/42 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_move | 15.761 | 16.069 | 26.772 | 267.752 | 19.357 | 34 |
| own_model | 16.512 | 16.834 | 27.090 | 264.654 | 19.871 | 28 |
| kv_only | 1.698 | 1.981 | 22.726 | 315.418 | 20.942 | 5 |
| ilp | **1.500** | **1.770** | **22.463** | **319.111** | 20.669 | 17 |

ILP vsno_move配对吞吐 **+19.30%、+19.32%、+18.92%**，均值**+19.18%**，95%描述性t
区间[18.61%,19.75%]；makespan降低**16.09%**、A queue降低**90.48%**、A TTFT降低
**88.98%**。vsown_model吞吐+20.60%、A queue-90.88%、A TTFT-89.47%。这些是各repeat
比率均值，不是均值之比；小样本区间不代表真实线上负载泛化。

实际B迁移1.263–1.283秒，ILP两到达总control均值1.641秒、solver18.6ms，全部进入
queue/TTFT/完整makespan。pause、resume、prefill、host-file copy未从指标中扣除。

### 负面影响及归因边界

ILP vsno_move **latency sum增加6.78%**、20秒goodput配对均值**下降41.94%**（.424→.253
req/s均值；17/42 vs34/42 SLO通过）。更早准入延长资源竞争，迁移也暂停老请求；准入
和makespan收益不等于所有请求延迟/SLO都更好，不能改SLO或忽略这些负结果。

按role复算：36存量请求平均latency19.037→20.848秒，20秒SLO31/36→13/36；6到达请求
平均latency21.275→19.595秒、SLO3/6→4/6。容量准入优化确实把一部分延迟从新请求转移
给老请求；未来目标应加入老请求SLO保护，不能只按吞吐判断线上部署是否合适。

KV-only与ILP每轮都是B迁移且相同最终layout；两者吞吐差均值+1.17%，95%t区间
[-.57%,2.91%]含零，**不得作为性能项独特收益**。本轮证明跨模型容量重放置，而非
ILP相对任意KV-aware启发式必然更好。不同placement性能项辅助证据仍在236906。

这里是**受控实际mapped-KV预算**而非整卡80GB已满。完整prompt+maxoutput预留/整KV
host-file复制、N128单模型cost外推/固定成本假定、候选2/model/rank和最多2moves仍是
原型限制。默认32step确认结果见下节；不关闭idle-rank EP执行，不改dense graph。

普通基线在本benchmark中指相同自定义full-reservation scheduler的no_move/count选rank，
不是未修改原生vLLM动态KV增长/抢占行为。预留512输出包含未生成KV；此容量见证在
相同预留语义下有效，但不能直接证明只分配已生成tokens的原生vLLM也会阻塞。baseline
在存量完成后自然准入，不宣称无限排队；实际native增长和整卡预算是后续门槛。

原始证据`evidence/benefit-20261020/live-benefit-237068/`、`raw-237068.sha256`、
`verification-237068.json`、`analysis-237068.json`；正常五分钟查询与终态归档均保留。

## 默认32step确认237171

实际2026-10-21 **11:24:31–11:39:47**，`d1n41a18g03`两A800-SXM4-80GB，COMPLETED/0:0。
四策略各6pairedrepeat，seed20261025，其他long_kv输入/输出512/预算/SLO20秒不变。
所有模型和策略均记录DP pause共识32step；执行代码SHA与237068相同，仅测试文件增加
capacity witness测试。不是不同CUDA graph配置：仍为冻结sparse，实际FULL replay只出现1/8。
allocation内112+193测试通过；**184rawfiles SHA、336请求/172032输出tokens**独立复核。

24trial均有两rank真实allocation probe，global足够/local不够及rollback owner验证通过。
KV-only/ILP每轮同样B0→B1移240MiB、源释放192MiB、目的增加288MiB；准入A后
2016:2112MiB、独立B后2112:2112MiB，全部14请求可同时驻留。no_move/own_model均不迁移。

| 策略 | A queue秒 | A TTFT秒 | makespan秒 | 吞吐tokens/s | 平均latency秒 | 20秒SLO请求/84 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_move | 19.637 | 19.959 | 32.272 | 223.716 | 22.726 | 27 |
| own_model | 18.404 | 18.755 | 31.964 | 225.831 | 22.528 | 13 |
| kv_only | 3.653 | 3.968 | 27.644 | 262.452 | 25.368 | 0 |
| ilp | **2.809** | **3.116** | 29.287 | 245.811 | 26.672 | **0** |

ILP vsno_move配对吞吐 **+3.52%、+18.52%、+18.25%、+12.50%、+9.12%、-0.39%**，
均值**+10.25%**，95%描述性t区间[2.16%,18.34%]；makespan配对降低8.92%，A queue
降低**85.94%**、A TTFT降低**84.61%**。保留r5略负吞吐，不声称每repeat都正。
vsown_model吞吐均值+9.90%，区间[-7.16%,26.96%]含零且r0/r3负；A queue降低84.62%。

负面结果更明确：ILP vsno_move latency sum配对**增加17.85%**，72存量请求平均latency
22.208→26.803秒、20秒SLO25/72→0/72；12到达请求平均latency25.834→25.885秒，
SLO2/12→0/12。虽然新A更早被准入，完成SLO仍全部失败；**不能声称默认控制下有
SLO/goodput收益**。no_move goodput均值.141req/s，KV-only与ILP均为0，不计算0/0增益。

KV-only/ILP所有repeat同动作和最终layout；ILP vsKV-only吞吐配对均值**-5.07%**，
95%t区间[-20.88%,10.75%]。本场景不支持性能项独特收益或优于KV-aware启发式。
平均A TTFT之比看似ILP更快，但配对TTFT reduction均值为**-14.08%**，受到KV r0
11.002秒长尾影响；不以均值之比替代配对结果来挑选结论。

实际ILP copy1.218–1.244秒，control均值3.692秒、范围2.659–6.067秒，solver16.7ms。
ILP A pause最长3.980秒；KV r0 A pause9.401秒、no_move r1 A pause8.393秒。这些控制
长尾均计入完整窗口；solver/实际复制长度不是长尾的主要来源。相同最终layout的
策略也有不同pause/到达时刻，因此不能把全部吞吐差归因DP不均衡或cost预测。
跨job节点/时段/种子不同，237068与237171不是严格interval1→32因果消融。

### 归档与监控边界

本地查询器配置300秒，记录11:06、11:11、11:16查询；之后有**11:16:45→11:54:05**
本地检查时间缺口，11:56:51获得终态，未能实际连续每5分钟查询。缺口原因未确定，
不能把配置间隔写成无中断监控。原始poll保留；任务仍实际完整执行、终态由accounting确认。

首次tar下载SSH超时，截断包保留为`.raw-237171-interrupted-20261021-120108.tar`。
12:01恢复时先确认远端manifest/accounting未变，再加SSH keepalive重试；184文件SHA全
通过后才标记归档成功。恢复没有重跑GPU任务或改rawJSON。证据`live-benefit-237171/`、
`raw-237171.sha256`、`verification-237171.json`、`analysis-237171.json`和poll/terminal。

## 当前结论与下一门槛

用户指定的容量准入机制已在两独立模型、全部长prompt、相同受控预算/预留语义下
实测：全局足够但任一卡不够，跨模型释放页使新A提前进入，默认32step也有准入收益。
这完成本阶段原型收益证据，但不完成生产SLO优化、未修改native-vLLM baseline、
渐进KV增长或整卡OOM评估。下一阶段应先加入老请求SLO/迁移保护并联合校准pause、
copy与双模型layout/context；否则更积极准入会恶化老请求。当前不改20秒SLO追正结果。
