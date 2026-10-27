# Azure长度抽样 + 泊松到达：最终trace的理论设计

2026-10-21后续：用户澄清Code长输入触发、Conv长驻留调整，不要求Code长decode。
完整两CSV的后续统计见 `../reports/mixed-workload-profile.md`，下一阶段异构trace见
`mixed-workload-trace-design.md`。本文保留原byte-window理论与双Conv阶段设计，不将
其旧样本统计当全release，也不把旧双Conv结论套到新机制。

2026-10-21。本文源于用户当时指定的**只理论阶段**，保留理论/本地数值分析，不把
它当trace实测收益。用户随后授权GPU初测，执行协议见`trace-benefit-protocol.md`，
首轮237899已完成且净收益为负，实测见`../reports/trace-benefit-results.md`，
当前状态见`../reports/benefit-status.md`。下面未验证/未同步措辞是理论阶段的历史状态。
此前固定population收益是机制实验，
**不是最终trace完成**。实际证据与限制见`../reports/benefit-long-kv-results.md`。

## 0. 当前32K配置：取代下面旧4K速率候选

2026-10-21用户明确后续使用**max_model_len=32768，输入+输出总长度**。两个live
Python入口均支持`--max-model-len`，启动脚本通过`DP_OPT_MAX_MODEL_LEN`传递；默认
统一32768。保持sparse captures1/8/16/32/64/128/256、FULL_DECODE_ONLY/compile0和
max_num_batched_tokens4096；后者是每step token预算，不是单条请求长度上限。
官方checkpoint config的max_position_embeddings为262144，32K是用户选择的实验上限，
不是模型原生最大值；官方配置快照保存在新证据目录。

默认每model/rank虚拟KV容量由2GiB改为4GiB：沿用block16tokens/1.5MiB，单条32768
完整reservation需2048blocks/3072MiB，另有null block/page余量，旧2GiB虚拟容量
无法覆盖。虚拟容量不等于物理预分配，也不自动增加每GPU2112MiB的受控mapped预算。
如果实际sample请求本身KV超过单卡受控容量，跨模型ILP也无法放入；应在实跑前
重新选物理预算，不能静默丢弃或截断。当前小范围样本最大总长度Code9180/Conv8991，
但不能假定完整CSV也全在此范围或32K内。

复用原48片段/304016行、不重新下载，32K CPU重算独立保存在外层
`evidence/azure-trace-32k-theory-20261021/`。Code/Conversation本地样本均100%满足
0<P/O且P+O<=32768；这只是样本覆盖率，不是全数据保证。成对长度与sampling限制不变：

| dataset / 条件 | sample行数 | 平均P | 平均O | 平均reservation MiB |
| --- | ---: | ---: | ---: | ---: |
| Code，支持32K子集 | 152721 | 2519.59 | 22.29 | 239.01 |
| Conversation，支持32K子集 | 151295 | 1649.65 | 106.20 | 165.31 |
| Conversation，P>=2048、256<=O<=512、总长<=32768 | 14154 | 3133.39 | 367.60 | 328.92 |
| Conversation，P>=2048、512<=O<=1024、总长<=32768 | 4737 | 4011.32 | 663.30 | 438.95 |

仍用40ms/token+.2s prefill的**未校准假定**，以及C_eff=3840MiB，主长slice
E[S]=14.904s、E[bS]=4963.63MiB*s；rho=.70-.85对应总rate**.542-.658req/s**。
当前起始候选改为**总.60req/s、每模型.30**，rho约.776、平均active8.94；预声明
sweep`.35,.45,.50,.55,.60,.65,.70`，不是先选出获益case后改范围。更长输出slice
E[S]=26.732s、E[bS]=11987.01MiB*s，对应rate.224-.272，候选**总.25/每模型.125**。
旧`.65/.40`仅属于4K条件分析，不继续用于32K扩大后的相同长度条件。

两个tenant等rate的Code/Conversation支持分布混合E[bS]=738.33MiB*s，memory尺度
rate3.64-4.42req/s；这仍不包含实际prefill/compute饱和，不能直接当可持续速率。
长context改变attention、prefill和迁移字节，既有N128单模型拟合和下面4K静态事件频率
均不能直接外推。当前只修改本地配置、重算长度积分，**32K graph/迁移尚未GPU验证**。

本地可重算命令（输出目录须尚不存在，避免覆盖旧证据）：

    python experiments/dp_opt/azure_trace_lengths.py --reuse-sample ../evidence/azure-trace-theory-20261021 --max-model-len 32768 --output ../evidence/azure-trace-32k-theory-20261021

以下第2节统计、第3节数值速率和第4.2节静态模型均保留为**旧4K条件证据**；
marked Poisson语义、E[bS]公式、wait-vs-migrate和测量原则仍适用，但不称32K结果。

## 1. 数据源和trace语义

来源为Azure官方`AzureLLMInferenceDataset2024.md`和`dataset-llm-2024` release，引用
DynamoLLM（HPCA2025），使用时遵循CC-BY并归因。官方只有`TIMESTAMP`、`ContextTokens`、
`GeneratedTokens`，没有prompt文本/模型ID。应成对sample输入P和输出O，按目标长度
构造合成prompt并强制生成O，不分别独立sample P/O，不把synthetic prompt称作Azure文本。
长度真实不等于内容/专家路由真实，尤其本研究为MoE；多种合成prompt、禁prefix cache
和固定所有策略的输入仍不能消除这个外推边界，不能把重复算术prompt等同Azure内容。

到达采用两个独立marked Poisson stream：`lambda_A=lambda_B=lambda_total/2`，每个
interarrival为`Exponential(lambda_m)`；它们合并仍是Poisson。长度marks独立sample
Azure行，舍弃原TIMESTAMP；所以这是**Azure长度分布驱动的合成Poisson trace**，不是
原始Azure时间trace重放。保存全量绝对arrival时刻/tenant/source row/长度/seed，所有
策略使用相同trace，即使系统暂停/队列拥堵也不能延后未来arrival或按completion打请求。

当前engine上限32768；P+O>32768不能静默截断，须报告排除比例。下面旧4K分析曾按
4096筛选，不再作为当前trace支持限制。Poisson不指定固定同时驻留数，
不能沿用12条bootstrap同步decode边界或为形成容量见证等候再到达。

## 2. 历史4K条件长度抽样结果

证据在外层`evidence/azure-trace-theory-20261021/`，`../azure_trace_lengths.py`下载每份CSV
24个均匀byte-stratum随机位置的256KiB片段，只保留完整行；总约12MiB/304016行。
不是全CSV逐行uniform reservoir，也不把小片段统计称作全数据精确分位数。下载raw
片段/Content-Range/SHA、官方说明/许可证/notebook/release metadata全部保留。

| dataset / 条件 | sample行数 | 平均P | 平均O | O中位数 | 平均reservation MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code，0<P/O且P+O<=4096 | 124502 | 1658.43 | 21.39 | 8 | 158.19 |
| Conversation，同支持条件 | 135984 | 1250.39 | 92.11 | 40 | 126.56 |
| Code，P>=2048、256<=O<=512、P+O<=4096 | 279 | 2792.59 | 346.66 | 325 | 295.03 |
| Conversation，同长请求条件 | 11427 | 2693.72 | 361.74 | 351 | 287.15 |

支持子集约占Code81.52%/Conversation89.88%；长slice占各自全部样本**.183%/7.553%**。
因此Code虽常有长输入，但输出偏短、驻留时间短，不能简单把长P等同长迁移收益窗口。
主容量压力trace优先Conversation长slice，两tenant都从该slice独立sample；须显式标为
长请求条件压力测试而非完整Azure代表性结果。另加Code/Conversation各占一半的支持
分布作为泛化对照，不保证它也有收益，不把Code输出统一改成512来制造驻留。

预声明更长decode敏感性：Conversation `P>=2048、512<=O<=1024、P+O<=4096`，
本地样本1916行、占Conversation全部样本1.266%，平均O612.89。同一service假定下
E[S]=24.716秒、E[bS]=7640.87MiB*s，2112MiB/卡、rho=.70-.85对应总rate
**.352-.427req/s，起始候选.40req/s、每tenant.20**。N10同服务近似的首completion
等待2.247秒，比主slice1.334秒更长；但KV复制量也增加，仍不保证迁移收益。
这只是明确条件分布的敏感性，不与主slice混合、不称完整Azure结果；更极端
`P>=2048、1024<=O<=2048、P+O<=4096`本地仅32行，不宜作为主trace。
数值见`evidence/azure-trace-theory-20261021/longer-decode-sensitivity.json`。

## 3. KV时间积分公式与历史4K速率候选

本配置block16tokens/1.5MiB，64blocks为96MiB composite page。当前full reservation：

    b(P,O) = 1.5 MiB * ceil((P+O)/16)
    S(P,O) ~= t_prefill(P,layout) + O*t_decode(layout,context)
    A(P,O) = b(P,O)*S(P,O)
    E[active requests] ~= sum_m lambda_m*E[S_m]
    E[logical reserved KV] ~= sum_m lambda_m*E[A_m]

这里S是**准入后驻留**时间而非含client queue的latency；Little/Campbell只用于给定
service模型的offered-load估算。b和S通常相关，必须用E[bS]，不能用E[b]E[S]。
实际physical mapped加page/null/fragment overhead W；以`C_eff=2*C_gpu-W`定义：

    rho_KV = sum_m lambda_m*E[b_m*S_m] / C_eff
    lambda_memory_scale = C_eff / E[b*S]       (两tenant同分布时)

rho<1只是均值必要设计尺度，不保证Poisson波动下不排队；C_eff不是物理整卡显存。
若改渐进增长，应以`A=integral_0^S b(t) dt`重新算，不能继续套full reservation结果。

本轮假定t_decode=.040秒/token、prefill=.2秒，**非长trace实测校准**。Conversation
长slice给出E[S]=14.670秒、E[bS]=4235.10MiB*s；size-biased resident平均KV约
288.72MiB，输出负载为lambda*361.74tokens/s。W先取4个pool共384MiB工程余量，非
精确allocator测量，后续需用实际page owner统计替换：

| 每GPU受控KV预算 | C_eff | memory rate scale | rho=.70-.85对应总rate |
| --- | ---: | ---: | ---: |
| 2112MiB（沿用主见证） | 3840MiB | .907req/s | **.635-.771req/s** |
| 1728MiB（另列压力敏感性） | 3072MiB | .725req/s | **.508-.617req/s** |

**优先候选总rate .65req/s，即每模型.325req/s**；2112MiB时rho约.717、offered
active约9.54条、输出约235tokens/s。预先冻结rate sweep `.4,.5,.6,.65,.7,.8`，低负载
对照.4，中间.6-.7，.8只作高压边界。1728MiB敏感性另扫.45-.65，不与2112混算。

之前默认32固定population测得no_move223.72/ILP245.81输出tokens/s。如果非常粗糙地
除长slice均值O=361.74，可得.618/.680req/s，.65恰在两者之间；这仅解释候选，不是
online sustainable capacity证据：前实验O512、同步存量、不同context与完整drain窗口。
新trace真实mu还受prefill、batch、pause/dummy/graph bin影响，不能直接宣布.65会正收益。

service敏感性：.035秒/token+.2秒prefill时E[bS]=3712.89，.05秒/token+.5秒prefill时
5365.67；2112MiB/rho.70-.85对应rate分别.724-.879和.501-.608。**.65是起始候选，
不是跨GPU/时段通用最优**。prefill并发会改变S，必须有不启用迁移的baseline校准。

完整支持分布代入同假定：Code E[bS]=175.68，Conversation776.19MiB*s；1:1混合
rho.70-.85需总rate5.65-6.86req/s。输出均值56.75，约321-389tokens/s，且每次到达都
需要prefill；这可能先compute/prefill饱和而非KV-placement受限，不能用长slice的.65
作为混合trace速率，也不能由这个memory尺度直接推荐6req/s为可稳定服务。

## 4. ILP可行不等于迁移有收益

### 4.1 容量条件

用实际mapped定义F_g=C_g-sum_m mapped_mg，Delta_jg为incoming在pool g新增页：

    sum_g F_g >= b_j,  但所有g都Delta_jg>F_g

是想要的global-enough/local-blocked事件。还需copy-before-release可行：某已有
request i迁至d的实际新增页不超过F_d，源exclusive owner页释放R_is后目标卡足以
准入j，且整个迁移次序峰值不越预算。final packing可行不代表当前序列可行。
已完成主见证复制B240MiB、目的增288MiB、源仅释放192MiB；不能把b_i当释放量。

必须区别两类事件：count-LB首选rank不足、另一rank能直接fit，只需KV-aware routing；
本研究跨模型迁移的强见证要求**所有rank均不能直接fit，但全局空间足够**。
即使忽略页粒度并把384MiB余量对称扣为每卡192MiB，令每卡逻辑容量c=1920MiB、
新请求大小b~=288MiB、存量U=U0+U1，两卡都不足要求`U>2*(c-b)`，全局够要求
`U<=2*c-b`，因此必要的总存量窗口约3264..3552MiB（C_eff的85%..92.5%）。
这不是充分条件，也不是实际page allocator判定；它解释均值rho=.717时为何必须
依靠Poisson高占用尾部才有较多强见证，不能仅通过增大rate保证净收益。

Poisson IID long slice的count-balance常同时让KV接近均衡，而且P+O范围有限；事件
可能稀少。真正长KV大小异质性、存量age及tenant长度分布不同才更可能提高差异。
不能人工强制每隔几条trace出现B exclusive页后称作自然trace。

### 4.2 普通调度的自然释放比同步机制试验快

原试验存量几乎同龄，全部还需约500decode步，新A可以等十几秒。Poisson steady
state的存量age不同，**不能把queue设为一条完整请求的平均S**。以N条同固定S服务
近似，residual为独立U[0,S]，第一条completion期望`S/(N+1)`；S14.67/N10约1.33秒，
而不是14.67秒。当然page可释放性与新请求fit可能需要不止一次completion。

本地`../azure_trace_envelope.py`用length-biased marks/固定S、独立Poisson snapshot/count
balance、contiguous blocks/null page/free-slot/page owner、一次迁移枚举作说明性计算：

| 每卡预算 / 总rate | safe snapshot中global够/local堵 | 一次迁移可解 | 可解事件自然等待均值 | 等待>2.8秒比例 |
| --- | ---: | ---: | ---: | ---: |
| 2112 / .65 | 1.079% | .643% | 1.330秒 | 14.71% |
| 2112 / .70 | 1.592% | 1.007% | 1.143秒 | 8.00% |
| 1728 / .60 | 2.931% | 1.843% | 1.278秒 | 12.05% |

这不是真实finite-memory queue的steady state或ILP性能实验：每设置6000个无约束
M/G/infinity offered snapshots，先条件筛选初始每卡fit；没有queue反馈/future arrivals/
context竞争，也未执行restricted ILP。所有request都枚举且destination用精确free slots，
比当前有限候选更乐观；一迁移又不涵盖多迁移，不能称全算法的严格上下界。
事件count只有几十个，比例仅用于排除“必定有巨大收益”的直觉，不作最终收益预测。
2112MiB/.65的无约束offered snapshot约11.93%已有某卡超预算，尽管均值rho仅.717；
说明均值负载不能排除Poisson尾部。以.643%一次可解事件及14.71%等待>2.8秒粗看，
600秒/.65的390次到达可能只有几个可解事件、不到一个明显超过迁移时间的事件。
这是条件静态模型的样本量尺度提示，不是实际queue里事件数或PASTA收益预测。

当前默认32实测B复制约1.225秒、A admission queue平均2.809秒、global control平均
3.692秒，pause有数秒长尾。上述许多自然释放只等约1秒，**立即移动可能更慢**。
只有纯solver16ms并不代表控制面便宜；别用单模型DP线性cost抵消真实全局暂停。

### 4.3 ILP需要比较排队动作和迁移动作

当前实现枚举rank后只在可准入布局中选最低cost，**没有wait作为优化candidate**，
旧原型的普通direct-fit admission也暂停两个模型；后续live_trace已改为不pause的
串行full-reservation fastpath，但逐请求RPC仍可阻塞。持续trace需要至少：

- 不迁移的direct-fit走轻量容量预留fastpath，不对每条Poisson到达全局pause。
- capacity block才规划；允许等待预计的下一次可释放页事件，completion后重新规划，
  不把first-attempt一次无解等同永远无解，也不无成本重复pause/solver。
- horizon纳入request残余输出、context/KV owner及copy成本，加入老请求SLO/cooldown。
- arrival队列、eligibility（当前要求全部resident均为movable decode）、同epoch并发
  到达/完成与预算预留协调；不能要求一直等到全trace没有prefill而改变arrival process。

此候选负载约9.54条总active、每模型平均4.77条，常见DP counts均落sparse capture8
内；不像236906辅助场景必有A2:9跨16档。故最终trace即使容量准入有收益，也可能
KV-only与ILP同动作。要证明性能项独特收益，需要自然出现不同layout的多个可行
候选并有联合cost依据，不可仅靠N128模型外推或主动制造capture跳变。

既有motivation支持的只是固定N/context条件下、含capture项的描述性模型：

    I_m = abs(n_m0-n_m1) / N_m
    p_m = ceil_to_sparse_capture(max(n_m0,n_m1))
    T_hat_m = T0_m * [1 + a*I_m + beta*(p_m/p0_m - 1)]

单一`1+alpha*I`的预声明线性模型没有通过既有验证，不应当作全trace性能公式。
上述含capture的surrogate也不是长context/低N/双模型联合服务的校准；N=0单独定义
为空模型状态，不令公式除零，不关闭仍需参与EP通信的idle rank。
候选MILP在每个incoming目标rank外层枚举后，对resident request分配x_ig、迁移y_i、
page owner释放和copy阶段峰值设约束；最终每个model的rank0 count用one-hot h_mv
选择，固定N下I和p均是常数表，性能项仍为线性目标。若两tenant互相竞争，应查联合
J(v_A,v_B,context)表而非独立T相加。wait布局与立即准入布局的活跃N不同，因此应在
同一horizon分别评估完整成本后比较，不能把未准入布局的较小N当免费性能优势。
有资格称最优的仅是给定cost/candidate集和可执行copy约束的ILP最优，不是系统真最优。

以单incoming说明：令D_move为完整arrival-to-admission控制路径时间，包含pause共识
等待、solve、copy和resume；若pause持续期间已经包含solve/copy，不重复相加。
TTFT收益的简化条件为

    estimated natural capacity wait W_j > D_move + marginal prefill delay

总latency优化还要覆盖老请求相对wait基线的额外停顿，以及新增竞争的未来decode
损失；以下仅为同horizon的代价近似，不是完整queue的已校准completion模型：

    sum_queued w_j * predicted_wait_saved_j
      > sum_active w_i * extra_stall_i + sum_affected w_i * H_i * DeltaT_i

copy造成的停顿已计入extra_stall，不再重复收取同一复制时间。必须同horizon比较wait/admit两动作的完整
服务成本；两个动作活跃N不同，不能只比较各自balanced-reference以上的imbalance项。
若每条到达都pause tau_direct，额外迁移停顿为tau_extra_move，小占空比下pause duty
近似`lambda*tau_direct + lambda*p_move*tau_extra_move`；迁移时间与direct基本路径
不重复计费。S会进一步增大；.65req/s乘.5秒就约32.5%的暂停时间需求，即便忽略
反馈也很大，足以淹没稀疏容量事件收益。高占空比不能继续用此近似当实测比例。

## 5. 最终trace如何定义“效果”

**不能保证当前ILP在IID Azure-Poisson trace上有显著收益；现有证据不足，理论提示
当前全局pause+强制准入很可能削弱甚至反转总体收益。** 当前32K候选见第0节；
旧4K的.65req/s仅为条件候选，不是新的固定rate。
是否足够出现global够/local堵、以及等待是否比copy长，都必须由后续trace校准验证。
当前主机制确实成立，不等于所有到达都适合重放置或ILP优于KV-only。

若lambda低于两策略可稳定容量，steady-state completed req/s都应约lambda、输出
tokens/s约lambda*E[O]；不能用短drain窗口差异声称恒定throughput gain。这个区间
重点比较mean/p95/p99queue/TTFT、SLO/goodput、老请求停顿、KV利用、迁移数与字节。
要展示持续吞吐/稳定性收益，需要找到`mu_no_move < lambda < mu_ILP`，冻结trace后
证明baseline backlog增长/ILP稳定；mu不能由固定populationmakespan直接替代。

建议最终各设置固定arrival窗口600秒+完全drain（长测试可1800秒），至少3个独立
seed、配对四策略no_move/own_model/KV-only/ILP；不复用特制bootstrap、不中途reset
allocator；把warmup独立且同配置。报告first/end backlog slope、offered/admitted/
completed counts、censoring/drain和全部请求，包括短输出O=1的TPOT定义；前20秒SLO
若沿用须保留0goodput，不调阈值直到正收益。600秒/.60约360请求，若净可获益事件
不足，应报告低effect frequency，而非筛选只保留capacity events的吞吐。

原理论阶段计划先完成source row均匀抽样/固定trace及wait/fastpath设计。用户随后授权
先初测当前原型，固定trace来自已缓存片段而非全CSV均匀抽样，仍有wait成本和联合
校准缺口；执行阶段和正负结果单独记录，不追溯修改这里的旧4K数值证据。
