# 长驻留 Conv / 稀疏 100K Code：KV-only 压力实验预声明

2026-10-24，Asia/Shanghai。状态：**用户已授权实跑；自动探索与独立确认流程已实现，待保护同步提交**。
理论与CPU候选分析只在 [stress-trace-design-analysis.md](../reports/stress-trace-design-analysis.md)。
247027 原冻结配置配对复跑必须先完成；不得覆盖其源码、trace或lookup，不重复申请。

## 问题与目标

验证Code新请求到达时两rank均不能容纳、aggregate物理预算够、迁移其他tenant的
Conv请求后可以准入，且省下的等待超过实际控制/复制/算力干扰代价。

建议机制主指标：**Code end-to-end latency P95相对no_move和no_move_fair分别
下降至少50%**。mean/P90/P99、queue/TTFT/TPOT、120s SLO、全请求与老Conv
代价必须一并报告。Code目标不能概括成全请求或吞吐50%；不把CPU条件proxy当
GPU达标。ILP不必优于KV-only，负结果完整保留。

## 本轮实际执行计划（替代后文早期候选时长）

入口`stress-sweep.sbatch`→`live_trace.py --stress-sweep`，不经回读旧749请求的
agentic-formal入口。每个allocation先真实100K容量/rollback与四rank默认graph，
再测对应长输出：4Conv balanced、4Conv consolidated+Code、4Conv真实迁移+Code。
每个完整输出比对串行tokens，迁移比对字节/源目标owner/复制峰值；不截短Code。
人口4只是有限人口校准，不声称已测人口2/6/8或生产级joint性能lookup。

独立完整Conv marks均值full-reserved KV乘本allocation实测balanced service，
先取4800MiB总背景目标的rate代理；非开放环稳态保证。阶段0：8192输出、10240/
11328MiB双预算、倍率1；阶段1：实测最佳O/B的实际rate×.85/1/1.15；阶段2与3
分别12288、16384输出、两个预算×三倍率。共17个训练case，seed20261024，
Code.001RPS、600s自然warmup+1800s测量，五策略完整drain，策略顺序固定种子
打散。每个stage的全部trace/config在任何策略结果之前冻结；保留全部负case。
stage0/1/2/3墙钟分别16/20/40/40h，gpu_a800分区MaxTime实查UNLIMITED、normal
QoS未显示MaxWall限制；实际Slurm拒绝则停止诊断，不自动重复提交。

选择训练case的目标为Code P95相对普通/公平baseline下降比例的较小值；不挑最好
baseline，不按Code是否碰撞或是否有碎片删请求。正式计时前对所有sampled Code
prompt_key串行暖两rank并检查tokens，prefix caching关闭，真实prefill仍计入每策略。

然后固定训练最佳的O/B/实际Conv RPS/Code RPS，按预声明8个heldout seed
20261025..20261032逐allocation验证；每seed 600s自然warmup+16000s测量、五策略、
32h墙钟。不因收益提前停掉不好的seed。完整汇总原始Code latencies，按seed成组
配对bootstrap2000次，分别报告两baseline以及较小改善的95%区间。>=100个Code、
8个seed且较小改善区间下界>=50%才记确认达标；P99与小seed数局限明确保留。
同时报告所有Conv、baseline定义的老Conv（包括warmup老请求）、整体和120sSLO
代价，不把Code达标当整体吞吐收益。这个搜索仅给声明网格内的观察最佳，非全局最优。

本地`stress_campaign.py`负责等待247027终态SHA/strict核验、保护同步、唯一顺序
allocation、每300s实查、终态SHA归档/独立复核和下一stage；每次提交前核queue空/
gjghfd main。失败或PAUSE marker停止盲重提交，保留明确job和原始失败证据供诊断。
长期证据与恢复状态：`evidence/kvonly-stress-campaign-20261024/campaign-state.json`。
ILP仍用历史单模型performance surrogate；仅move成本由本allocation长输出混合
profile更新，wait-aware关闭、不复用旧1024输出lookup。KV-only是本轮主技术。

## 固定与改变的设置

- 两A800-SXM4-80GB，两个独立Qwen3-30B-A3B tenant；同checkpoint可复用，
  权重、KV pool、request和EP group独立；每tenant DP2/EP2/attention-TP1。
- BF16、max_model_len131072、24GiB/model/rank虚拟pool、allocator utilization .98。
  保持sparse captures `1,8,16,32,64,128,256`、FULL_DECODE_ONLY、compile mode0、
  max_num_batched_tokens4096、控制interval32；不关闭空attention rank的EP执行。
- Code保持真实SWE-smith完整消息边界约100K输入、输出64；验证所有用到的prompt
  与原bank SHA、token IDs及长度一致，不截断/padding或改变Code语义冒充原数据。
- Conv原Azure slice的输入及来源保留，压力版本输出固定8192。
  **必须记录原output_tokens和synthetic_output_tokens，source label明确synthetic**；
  原paired output512–1024作对照，不声称8192来自Azure公开原分布。
- 预算对照11328与10240MiB/GPU；10240按96MiB真实页执行，不将尾数当可分配页。
  所有baseline、KV-only、ILP同预算；不是整卡OOM，原生vLLM动态增长基线仍未实现。
- 保留 no_move/no_move_fair(τ30s)/own_model/kv_only/ilp 五策略；no_move和fair均
  零迁移，fair只阻止其他tenant新KV准入、已有decode和客户端到达继续。
- 第一候选Conv .025RPS；.03及输出12288/.015、16384/.01只作已声明对照，不盲目
  加压追正。速率当前是理论候选，最终取决于独立GPU容量/service校准，结果前冻结。
- 保留120s SLO与900s request timeout；不得删超时请求。若长输出校准已表明900s
  不足，须执行前显式修改协议与所有策略同配置，不从收益结果反向调timeout。

## 阶段A：容量与混合成本校准，不作收益宣称

在唯一后续A800 allocation先检查各model/rank的graph、pinned/mapped pages和
最长Code真实分配/rollback。完整8192/12288/16384 Conv输出的序列上限、真实
prefill与default sparse decode均须通过，不把只改max_model_len当通过。

测Conv人口2/4/6/8的原2:2等balanced布局与全source合并布局，收集实际驻留时间
分布、KV随年龄增长、完成释放、TPOT和后续count调度反馈。Code分别在无Conv、
balanced背景及合并背景测service；测1/2/3/4条Conv合并的preplan、pause、RPC、
copy、resume以及source/destination owner、copy-before-release peak。全source构造
只用于校准可行性，正式策略运行现有KV-only，不强制算法执行CPU的理想动作。

冻结rate前使用实测`lambda_conv * E[reserved_KV * resident_time]`核算背景占用，
目标约4.5–6GiB总Conv KV；同时核验目标rank不能fit而另一rank可接受所有迁移页。
人口队列增长/真实aggregate不足不算目标碎片场景。完整保留所有失败，不在正式
trace里按KV状态筛选到达。service和move成本不足以支撑减半门槛时如实报告，
不通过换SLO、删请求、bootstrap同步人口或关闭graph制造通过。

## 阶段B：短trace机制诊断

低碰撞Code候选 .0006/.001RPS。先使用预声明seed20261024/20261025/20261026、
3600s offered window，另600s自然Conv warmup、Code仍由自己的Poisson流产生。
warmup请求从完整空系统自然到达，必须记录并drain；测量窗口之前发起的老Conv
仍纳入影响分析，不同步bootstrap、不停止另一client帮助Code准入。

全部seed/策略使用同一冻结trace和bank，每个trial完整drain再换策略；策略顺序
执行前用固定seed打散。3600s下Code可能仅数条，不保证发生碎片，不能称稳定
P95/P99或最终50%通过。报告全部trace，包括无机会/碰撞多/收益为负的seed。

Code阻塞独立分类：自身tenant已有Code、aggregate真不足、local placement不足、
all-decode readiness gate、controller成本、prefill/compute争用。分类只能解释，不
删除正式端到端总体指标中的请求。至少保留真实两rank allocation/rollback witness、
global可容纳所需增量、source真实owner页释放、迁移后Code准入与预算核验。

## 阶段C：最终trace的样本量/碰撞权衡

低碰撞和大量Code无法同时靠提高RPS免费实现。建议预声明Code率扫描
`.0006/.001/.002`；.004作高碰撞对照，不能为了减少运行时间偷偷替换低碰撞主档。
trace时长/seed应按原始Poisson到达样本数量、分区time limit和预算在执行前冻结，
不按性能结果停止或延长。拆成多个独立segment，每个segment重新自然warmup并
完整drain；切分造成的新边界和老请求代价明确报告。

稳定Code P95至少应预期获得约100个不同到达事件并跨多个独立seed；这是实验
规划量级，不是统计显著性保证。更少样本则标为descriptive。P99需更多样本，
不以同一4条Code的多次配对重放声称已有大样本。各seed分别报告，汇总按预声明
定义，置信区间按trace/seed配对且注明小seed数限制，不能将相关请求当独立实验。

保持五策略和同trace，先完整保留普通/公平两个baseline。若资源不足分阶段执行，
明确哪些对照尚未完成，不称完整算法收益通过。旧4h launcher不能直接运行长窗口
的五策略；后续单独容量/calibration/正式trace入口需遵守分区限制。
2026-10-24 11:07:35读取gpu_a800分区MaxTime=UNLIMITED，但尚未核QoS/account
时限；阶段B的3600s窗口加warmup/drain五策略也不应塞进旧4h launcher。执行前
核实实际允许时限，申请足够预算或按冻结策略块拆分，不能运行中删case赶time limit。

## SLO、归档和停止条件

8192长输出可能仅service就超过120s，原SLO报告不能丢弃。若需要任务专属新SLO，
另行预声明并独立列报，不替换本协议120s指标。baseline定义的老Conv群组逐条
对照、completion与TPOT退化必须记录；不拿Code收益证明整体公平或Conv无代价。

运行前保护同步指定文件、旧SHA备份/原子替换/新SHA复核，独立新trace/config/
lookup manifest。现有agentic launcher会回读旧冻结749trace和预算；**不能仅export
新RPS/O/B就声称跑了本协议**。新trace输出变换和新长输出joint cost lookup还需
实现、测试和校准，目前没有可称已执行的stress trace。

所有远端访问ssh paracloud，登录节点只读/同步/调度；GPU推理、import、测试和
JIT在allocation。每300s观察至终态，立即SHA归档，再独立核验完整请求、token/
ownership/actual mapped/预算/graph、公平期间零外tenant准入以及所有性能成本。
失败先诊断；不重复allocation、不重启Aegaeon、不自行commit/push。
