# Tail TTFT：在线准入余量保护

## 预声明范围（2026-10-25）

250581的原始记录揭示两个不同瓶颈：Code reservation→first token约54–55秒；Conv
最坏请求排队22.18秒，准入发生在Code完成后。v2恢复4:0→2:2导致第二Code从
KV-only直接准入变为再次迁移，不能把回平衡控制成本下降当成tail改善。

本轮`kv_only_headroom`先修复恢复对未来准入的退化，不声称能消除100K prefill成本，
也不通过改变trace、输出、SLO、预算或CUDA graph制造收益。原75请求/2400秒trace
及600–2400秒主窗口不变；Code只有2个样本，高分位仍是描述性结果，不是稳健P99。

## 通用算法

准入仍用KV-only；不同tenant没有Code/Conv角色特判。每tenant仅使用真实已经到达
并派发的最近64条请求总长度，取nearest-rank P95：`L_m=sorted(L)[ceil(.95*n)-1]`。
请求输入+声明输出总量的保守新增映射需求为

`D_mr = ceil(ceil(L_m/block_tokens)/blocks_per_page) * page_bytes`。

不利用mapped空槽折扣；因此是充分的physical headroom证书，不是allocator充要条件。
当前未重放置布局若至少一rank满足`M_r+D_mr<=B_r`，回迁不得破坏这种可准入性。
原布局已无法fit的类型不虚构新的证书；缺历史时不预测未来长度。

先为每个受保护tenant枚举未来DP home `h_m∈{0,1}`（最多4个组合），每个组合解
原无新请求的staged ILP，新增最终状态约束

`M'_r <= B_r - max({D_mr : h_m=r}, default=0)`。

不同类型的未来请求是单个独立的可准入机会，不假设同时到来，因此取max而非sum。
复制中间峰值仍受原`B_r`约束，不错误地把预留headroom当复制峰值或初始状态预算。
同一组合仍允许不动、部分恢复或把空闲corridor搬到另一rank，不强制均衡/旧home。

目标保留v2 `P(layout)+W*F*z+Σcopy_request_seconds[i]*move[i]`和ε=.01过去服务credit。
无收益时不pause，保留债务并仅由下一次真实release触发重算；FIFO待准入优先。
暂停后再用冻结owner/physical inventory校验ILP、credit与headroom；安全复制机制不变。
未来RPS、未到达mark、prompt-bank中未到达长度都不用于在线决策。

性能P仍是N128单模型sparse surrogate，未联合校准；本轮硬门控只避免已知未来fit
机会的回归，不证明forecast概率、kernel成本模型或严格每请求TTFT/SLO上界。

## 配对实跑

- 两A800、双独立Qwen3-30B-A3B-Instruct-2507，各DP2/EP2/attentionTP1；BF16/128K。
- 24GiB/model/rank虚拟pool，10GiB/GPU受控KV budget，allocator .98，32step边界。
- sparse `[1,8,16,32,64,128,256]`/FULL_DECODE_ONLY/compile0、4096 batched tokens不改。
- 固定Conv .0307909199195414RPS/8192输出、Code .001RPS/真实约100K输入+64输出。
- seed20261024，600秒自然warmup+1800秒主窗口；75全部drain，不丢未完成样本。
- 六策略：no_move、no_move_fair、own_model、kv_only、ilp、kv_only_headroom。
- fair阈值30秒、完成SLO120秒、timeout900秒；测量全部pause/solve/copy/等待。

主指标为Code、Conv、合并各自mean/P90/P95/P99 TTFT；同时看latency/TPOT/SLO。
先判断v3是否避免v2准入余量回归，再与本作业KV-only作真正配对比较。跨job v2
仅作描述，不以历史最优值代替当轮基线。2Code不足统计显著性；确认要后续多seed。

## 核验与运行安全

新增arrival JSONL/final记录供独立verifier重建截止decision时间的完整bounded history。
检查未来泄漏、遗漏真实到达、所有scenario约束、ordered复制峰值和最小可行score；
保留旧v1/v2档案兼容。运行源码含11个文件SHA归档，campaign另冻结测试/同步清单。
唯一A800 allocation，300秒主SSH查询与remote只读watcher，终态完整SHA归档/独立
复核；失败先诊断不盲重提交。不commit/push，不执行Aegaeon。
