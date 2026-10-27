# 双模型端到端收益实验

预声明于2026-10-20；结果产生前冻结本协议。此阶段验证真实排队/规划/复制成本下的
capacity-driven placement收益，不是生产吞吐、成熟KV或全请求全局最优证明。

## 固定配置与请求

- 同allocation两GPU，两个独立Qwen3-30B-A3B-Instruct-2507 tenant；各DP2/EP2/TP1。
  checkpoint复用，但权重、KV namespace、请求与EP组独立，worker/topology保存核查。
- 默认sparse graph `1,8,16,32,64,128,256`，FULL_DECODE_ONLY、compile0、同步decode。
  无APC、speculation、LoRA。每tenant/rank虚拟KV2GiB；不触发整卡OOM。
- bootstrap A3:3/B4:3，13请求，输入64/输出256。全部至少decode8token后设到达边界；
  这是状态触发的受控到达，不是固定open-loop绝对时间trace。
- 独立A流在边界到达1请求，输入2816/输出256；B流在边界+1秒到达1请求，输入64/输出256。
  每策略全部15请求必须完成，无拒绝/静默丢弃；不足空间时保留client queue。
- 边界physical mapped页必须为A1:1/B2:1，各页96MiB；GPU预算固定480/288MiB。
  B0必须存在独占非pinned页。任何trial起始geometry变化直接失败，不调整预算追求正结果。

## 策略与成本

每repeat实跑 `no_move`、`own_model`、`kv_only`、`ilp`；seed42打乱顺序，3个配对repeat。
复用engine但等待request/KV释放，并检查geometry；各rank预热与串行token参考不计测量窗口。

- no_move：只在真实mapped-slot允许时准入，选较少request的rank，不移动。
- own_model：仅移动到达tenant的request，失败排队自然释放。
- kv_only：与ILP同候选/预算/迁移代价但layout性能项置零，direct-fit优先。
- ilp：枚举目的DP，两rank整页owner-credit约束与capture-aware layout表一起优化。
  每pool至多2候选、至多2迁移；因此只对限制候选集合求解，不宣称全请求最优。
  每目的rank solver time limit1秒，预热SciPy/HiGHS。首次无解后不重复迁移搜索，
  后续只查direct-fit；暂停后prefill资格变化安全resume并重试，不丢到达请求。

布局代价外推单模型N128/A800的posthoc surrogate：a=.3051180351、b=.1849037531、
T0=.0246988624167s；horizon取当前剩余输出均值。迁移fixed .475s来自235093H200实测，
带宽假定1GiB/s。双模型小batch/GPU变化未校准，不以预测代价冒充测量结果。
host-staged真实KV复制、runner-state/owner迁移与destination copy peak全部保存。

## 指标与有效性

窗口从bootstrap外部到达到最后一个请求完成；包括client等待、规划、pause/resume、
复制、prefill和decode。报告总output throughput、request latency sum/mean/p99、
TTFT mean/p99和A到达TTFT、平均/p99 TPOT（completion-first-token除以255），
client chunk最大间隔、真实control/solve时间。TPOT是client平均值，不是kernel时间。

预声明统一完成latency SLO20秒，goodput=满足SLO请求数/完整makespan；deadline不得看到
结果后选择。输出逐请求timestamps、cadence/tokenIDs、decisions/前后ownership/预算、
串行参考、配置、sourceSHA、GPU/topology与日志；下载后逐文件SHA256复核。
`verify_benefit.py`独立复算原始指标和配对ILP相对三基线的差值/比率；保留负/中性结果。
仅3repeat描述性比较，不声称统计显著。若ILP与KV-only执行相同placement，容量收益不等于
ILP性能模型的独特收益；后续另设计placement tradeoff，并进行双模型校准。

入口：`DP_OPT_LIVE_MODE=benefit sbatch experiments/dp_opt/live-placement.sbatch`。
