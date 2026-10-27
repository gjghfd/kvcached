# Work-aware soft-headroom ILP：2026-10-26预声明

用户要求基于kv_only_headroom迭代新ILP，明确优化attention-DP不均衡和KV回迁。
主指标是Conv/Code的avg、P95、P99 TTFT及Conv avg TPOT；全部请求、所有基线同配置。
本轮先改变回迁目标与门控，准入继续使用原KV-only，便于隔离回迁的增量贡献。
不是修改模型权重/EP组，不改变默认sparse graph，不宣称已实测收益。

## 已发现的限制

- 原headroom硬约束总是保留历史P95长请求的容量，即使Code很少也可能不允许把
  Conv搬到空卡；恢复机制没有机会发挥作用。
- 过去驻留服务1%的credit是绝对执行门槛，可能拒绝未来净收益很大的早期回迁。
- 单一平均剩余token horizon、N128单模型外推忽略短请求先完成、长请求继续失衡。
- sparse captures是1/8/16等：N2的1:1是capture1，2:0是capture8；N9的4:5是8，
  9:0是16。需要同时建模graph阶跃和attention工作，不把graph效果叫纯attention占比。

## 新的通用目标

只用当前已到达请求。按剩余token完成事件及最多128步的积分网格分段。对模型m，
区间k中人口为N，步长为Δs，A_r是该rank请求computed_tokens加区间中点增量之和。
用实际两resident模型、另一tenant idle的稳定decode校准：

`cadence = θ0 + θG*ceil_capture(max(n0,n1)) + θA*max(A0,A1)`。

三项非负回归；N3/N8的同capture不同DP分配用于识别attention项，不只测graph跳点。
这是客户端稳定decode cadence surrogate，不是GPU kernel纯attention分解。必须做
独立raw/ownership/graph/窗口验证后才能加载，不允许N128静默外推或未验证拟合。
当前第一版对另一tenant尚在prefill/decode的恢复保守延期，不能冒称完整联合校准。

损失与目标：

`P = Σ_m,k Δs*N*[θG*(G-G_balanced(N)) + θA*abs(A0-A1)/2]`

`min P + N_active*F*z + Σ_i N_active*(KV_i/bandwidth)*move_i + R_future`。

复用原staged ILP：每请求/复制阶段二进制变量，源页全owner释放、固定页、目的保守
pages、copy-before-release峰值与最终预算约束。增加各未来区间的count one-hot和
context差绝对值变量。允许部分回迁，不强制完美均衡或历史home。实际请求不截断；
超过已测context域的预测只取显式有限horizon，域外不假装有可靠性能收益。

对每模型枚举不保留/保留DP0/保留DP1三种headroom状态（两模型最多9场景）。保留
状态仍是硬最终容量约束，不保留状态支付风险：

`R_m = (1-exp(-lambda_m*H))*cold_admission_cost_m`。

lambda为过去累计到达数/暴露与外部冻结rate先验的Gamma-Poisson后验均值，不能仅
拿两个Code的短间隔当稳定rate，不能读取未来trace。大小只取已到达历史P95。
cold cost用真实过去准入控制成本与保守pause/copy先验；不是精确未来排队预测。
风险是placement objective常数，并在solver/replay中完整计入。

执行条件：净预测收益超过1 request-second margin及拟合残差界；执行前后重新验证。
固定pause activation只计一次。安全边界后已支付成本不重复扣继续利润，但真实
阻塞、失败尝试仍完整记账。新方法不再保证过去service的1%开销cap，必须报告真实
开销；仍保持最多4回迁、8候选/pool、1秒总规划场景预算、10秒cooldown与准入优先。

## 实验序列与指标

2026-10-27 refined return成本（不改变KV-only准入）：对tenant m、当前人口N，
`c_m,N = base_m,N + gamma_m,N(sparse_ceiling(max(n0,n1)))
           + beta_m,N * abs(A0-A1)/2`，`A_d=sum(context_tokens)`。
gamma为实测sparse阈值分类penalty，beta为同population的线性attention系数；
在剩余输出completion事件/最多128step段上按活跃请求数积分为request-second目标。
N2 Conv gamma(1)=0、gamma(8)=5.1994ms，beta=1.23197e-7s/token；
N9 Conv gamma(8)=0、gamma(16)=.9102ms，beta=5.78439e-8s/token。
held-out-repeat最坏contrast误差分别1.0263/3.8300ms，不删除N9噪声或降低门槛。
仅在candidate与reference特征改变的段累计误差；N1 rank-symmetric不计无关误差。
profile各N/context/peak-attention真实域之外延期，包括候选峰值；双active仍延期，
不拿resident-idle校准当joint模型。ILP目标包含真实KV/BW复制、pause activation及
在线Poisson soft headroom risk；候选净改善需大于误差加原min_gain，准入优先。
原256440校准和N2正确性均已SHA独立核验；旧wrapper的no_move KeyError只修
reference选择，所有paired/serial tokens仍严格相等。新版唯一allocation复用完整
cadence源，在compute内独立重建profile→N2/N9→固定五策略trace，不再重校准6h。
offline N2支持回迁并不代表新GPU收益；负结果和少Code tail统计限制保留。

历史隔离接入位置：`.workaware-stage/`（不是当前serving runtime）。入口
`workaware_campaign.py`先等255294全部SHA归档、父pressure验证与独立侧audit通过，
再通过已校验六文件patch安装，CPU回归后重exec以避免加载旧Python模块，保护同步
唯一A800 allocation。allocation内连续cadence→N2/N9机制→4800s五策略trace，
Code .0005/Conv .01956673845194241，seed20261024，主窗口600..4800，所有请求drain。
恢复在`evidence/workaware-return-20261026/campaign-v1/`；失败完整归档后诊断，不自动重提。
这是第一轮回迁增量：新policy `kv_only_workaware`准入仍是KV-only，完成后用新ILP；
不是已完成双active tenant联合成本/性能项准入优化。两Code样本不足P99统计确认。

当前255294继续到终态，25冻结源不修改、不覆盖活动remote、不另起allocation。
先在新增本地模块实现和CPU回归；旧任务SHA归档/完整核验后才接入solver/serving。

下一唯一两A800allocation优先做真实cost/恢复机制pilot：
- 保持BF16/128K、24GiB虚拟pool、10GiB/GPU受控KV、allocator .98、default sparse/32step。
- N2/N3/N4/N8/N9，两个可行长KV context，DP counts方向与顺序随机配对；另一tenant
  resident-idle。所有布局真实容量probe、稳定窗口检查，不支持的case失败保留。
  校准追加小N 16K/22K context，完整output3072、预声明1536..2560稳定窗口，避免
  长prefill污染早期256-token窗口；全部布局实际page预算复核。两个tenant均校准。
  回迁候选峰值attention tokens还须在实测特征域内，不以人口/context矩形域外推。
- 一条真实SWE-smith约100K Code及长输出Conv的准入→Code释放→回迁，保留对照
  kv_only_headroom与新方法；N2与N9分别测sparse阈值，不只选N9的正例。
- 机制fixture允许预声明人口/到达位置，但明确不是最终Poisson trace收益。

随后降低Code nominal rate .001→.0005RPS，Conv rate由实测服务时间/容量校准冻结，
不用负载单调改善作保证，不按TTFT正负挑seed。低Code只扩大两个Code之间的恢复
机会，不必然增大全trace平均损失；同时报告全trace与预声明Code完成后恢复窗口。
已有2400s trace直接将Code时间放慢会没有Code，不能复用短窗假装做P99实验。
正式长窗与配对多seed必须有足够Code样本；少样本P95/P99只描述，不宣称显著性。

至少实跑no_move、no_move_fair、kv_only、kv_only_headroom、新方法，原负结果保留。
统一输入/输出/预算/graph、完整drain，不能丢排队请求或调SLO追正。分别报告两模型
avg/P95/P99 TTFT、Conv request-average TPOT（另报token-weighted）、全窗口/恢复窗口、
SLO/latency、copy/pause/planning及净开销。优化迭代必须依据成本预测误差和通用目标，
而非针对某请求ID或模型名字硬编码。多seed及置信区间未完成前不称科研目标完成。
