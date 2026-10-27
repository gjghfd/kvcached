# Conv 降载：单卡 KV 压力目标 1.1

## 预声明（2026-10-26）

用户要求降低 Conv 请求量，将背景 Conv 单卡 KV 占用加一条 Code 的需求，由约
1.215 倍单卡预算降至约 1.1。只改变 Conv 的到达率，不修改算法、长度、Code
到达时刻、预算、SLO、CUDA graph 或请求完成要求。此轮是负载敏感性试验，不是
新算法独特收益或多 seed 稳健 tail 的确认。

分母为受控 KV budget B=10240MiB/卡，不是整卡80GB或24GiB虚拟pool。用251533
no_move 的600–2400秒主窗、Code未驻留且准入epoch稳定的样本，按相邻观测间隔
left-hold加权。总mapped扣除同rank实测空闲Code pool 96MiB，避免双计；Conv
固定空闲pool也是96MiB。Code单请求完整physical pool为9600MiB。

目标 Conv 单卡 physical 为 `1.1*10240-9600=1664MiB`。首阶候选为：

`lambda_new=lambda_old*(1664-96)/(mean_Conv_physical-96)`。

减去固定页后，动态背景只假设与到达率成比例；服务时间、离散页、有限窗口和
Code重叠可能改变该关系，因此不能把预测1.1当成GPU实测。完整测量后报告actual
ratio，再决定是否需要下一次明确的校准；不根据收益挑seed或丢弃负结果。

首轮253478完整结束后实测1.071159，偏离1.1±.02范围。第二档预声明约
`.021635500127908054RPS`，从实测背景反推、仍低于原`.0307909199195414`。
校准允许比当前候选略增或略减，但必须低于原始Conv率；原trace Code流、窗口、
模型/预算/输出/prompt-bank一致性必须通过。只在前一job全部终态SHA/独立核验
后更新helper并申请唯一后续allocation，保留首轮负/正结果，不挑收益来确定率。

第二档254337完整核验后实测1.129739，与首档1.071159分居目标两侧。第三档
预声明使用受限secant：`lambda=lambda_lo+(1664-K_lo)*(lambda_hi-lambda_lo)/(K_hi-K_lo)`，
得到`.01956673845194241RPS`。这是按压力校准，不是按TTFT收益筛选；两个profile
的全部非路径runtime controls、模型/预算/窗口/prompt-bank/Code流必须一致，物理
压力随rate递增且严格夹住目标。拒绝外推、非有限值、重复rate及超过原率上界。
仍使用原251533 Conv marks的time dilation，不重新sample seed，不改变算法。
该候选保留49请求/策略、主窗37（Code2/Conv35），尚无第三档实测pressure。

旧 trace 两流独立Poisson。对Conv的原到达时间作 `t_new=t_old*lambda_old/lambda_new`，
保留2400秒以内的mark前缀；这是降低Poisson率的time dilation，不是按结果筛请求。
Code的完整请求、prompt/token IDs/到达时间逐项不变。新trace所有offered请求均drain。

## 固定设置与判据

- 两A800，双独立Qwen3-30B-A3B-Instruct-2507，各DP2/EP2/attentionTP1。
- BF16/131072总长度/24GiB每model-rank虚拟pool/10GiB每GPU受控budget/allocator .98。
- 默认sparse `[1,8,16,32,64,128,256]`、FULL_DECODE_ONLY、compile0、32step、4096 batched tokens。
- Conv仍Azure输入长度sample加显式合成8192输出；Code仍SWE-smith约100K真实输入+64输出。
- Code .001RPS；seed20261024；600秒自然warmup+1800秒主窗，fair阈值30秒、SLO120秒、timeout900秒。
- 同allocation六策略：no_move、no_move_fair、own_model、kv_only、ilp、kv_only_headroom。
- 首先报告no_move Code未驻留背景ratio；1.1±.02只作描述性接近目标范围，不筛选收益。
- 同时报Code/Conv/合并mean/P90/P95/P99 TTFT，latency/TPOT/SLO与copy/pause/control。
- 两Code仍不足以统计确认P99；aggregate够不等于完整请求可装箱，保留不可行见证。

复用原保护同步、单一allocation ledger、300秒SSH及remote只读fallback和终态
SHA/完整verifier。研究原始证据在`evidence/pressure-ratio-20261026/`。不commit/push，
不执行Aegaeon，旧251533报告与输入不覆盖。
