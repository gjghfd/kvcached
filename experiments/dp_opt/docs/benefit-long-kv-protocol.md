# 主场景：长KV，总空间足够但当前单卡无法准入

2026-10-21 10:31预声明，按用户补充明确主收益不是只避开sparse graph档位，而是
count load-balance形成的KV放置失衡使新请求排队，跨模型重放置解除单卡容量约束。

## 工作负载与容量见证

同两A800/H100/H200驻留两独立Qwen3-30B-A3B tenant，每模型DP2/EP2/attentionTP1。
checkpoint相同，权重/KV/请求/EP组独立。保持sparse graph1,8,16,32,64,128,256，
FULL_DECODE_ONLY/compile0、固定所有策略interval1及统一预热。算法cost不变。

- A存量3:3，每请求输入3072/输出512，完整预留224blocks=336MiB。
- B存量3:3，每请求输入2048/输出512，完整预留160blocks=240MiB。
- 所有12存量至少输出8tokens且movable时，新A输入3584/输出512到达（256blocks=384MiB）。
- 独立B流输入1024/输出512于A后1秒到达（96blocks=144MiB）。所有14请求均完成。
- count load-balance：两个模型存量均3:3，所有原请求均长，不能把短64token背景当长KV。
- block16tokens/1.5MiB，page bundle64blocks/96MiB。初始页A11:11/B8:8，
  两GPU各1824MiB；每GPU受控aggregate mapped-KV预算**2112MiB**（22页）。
- 当前每GPU空闲288MiB，而新A在任一A pool实际需要新增384MiB：两rank都不直接可行。
  两GPU总空闲576MiB够新A384MiB，且全14请求logical reservation合计**3984MiB**，
  小于总预算4224MiB；最终含page/null/迁移碎片的physical mapped也可容纳。

存在可行方案：B0一条240MiB迁至B1，copy目的需要3页/288MiB恰好够；源B3→2释放
2页/192MiB。新A在rank0新增4页/384MiB后aggregate为2016/2112MiB，A4:3/B2:4。
随后新B在rank0利用B空闲槽，只新增1页/96MiB，最终2112/2112MiB且A4:3/B3:4。
镜像方案同样可行。A-only的任一完整336MiB迁移在目的实际需要4页/384MiB，
初始仅3页headroom，copy-before-release不能启动；单纯同模型移位置不会释放其他模型页。

这个证明是**受控有效KV预算**下的placement容量实验，不是整卡CUDA OOM或物理handle
跨tenant复用。物理GPU可能还有其他自由显存；不能据此宣称实际80GB整卡已满。
后续真实更大KV压力是另一个门槛。只按logical sum宣称可行也不够，须核验page与峰值。

令`F_g=C_g-sum_m P_mg`为每卡受控预算减去实际mapped页，`Delta_Ag`为新A在该pool
实际新增页需求。本场景满足`sum_g F_g >= KV_A`，但对每个g都有`Delta_Ag > F_g`；
可行的B迁移满足copy目的新增页不超过目的F，且源exclusive页释放后足以覆盖新A。
即不能只看全局free，也不能假定释放等于迁移logical KV bytes；本轮复制240MiB，
目的新增288MiB、源只释放192MiB。ILP须约束迁移中间峰值，不仅最终layout。

这里no_move/own_model与算法共用同一个自定义scheduler，所有请求提前预留
`prompt+max_tokens`，以保证KV增长预算。no_move试任何能fit的rank后才按count选择，
所以甚至比盲目只路由到最少请求rank更强。它**不是未修改vLLM的默认动态分配/抢占
调度实测**；不能由此断言原生vLLM在相同prompt与整卡显存下也一定排队。同样有限
存量请求会完成，baseline最终可以准入；“一直排队”指容量阻塞持续到自然释放，
不是已证明无限等待。此处max_tokens预留包含尚未生成的KV，后续需要渐进增长/实际
设备预算下的独立实验才能外推到生产native调度。

## 对照与有效性

四策略no_move（count-balanced初始放置，试任一rank仍不fit则排队）、own_model、
KV-only、ILP，各3pairedrepeat，shuffle seed20261024；固定20秒completion SLO。
主对照ILP vs普通no_move/own_model的A queue/TTFT/makespan/goodput；KV-only可能同动作，
此时不得将差异解释为性能项贡献。需报告全部请求而非仅新请求或被丢弃后的吞吐。

每trial到达前统一pause/begin，对A两rank作真实临时allocation probe再rollback，保存
actual新增页/预算不fit、还原ownership；该诊断可能临时超过受控预算，必须单独标记，
不能当作正常serving已越过预算。同四策略均计入full origin→completion窗口。
在probe前后原requests/blocks/pages保持不变；恢复后定义正式到达boundary。
正常迁移/admission copy峰值必须在预算内。验证aggregate logical/globalfree、实际probe
两rank不fit、A-only目标需求、B exclusive-owner释放、源→目的copy、准入前后预算。

指标完整包含solver/pause/resume/host-file copy/queue/prefill；新请求到达时显式记录
容量队列，不要求无限排队：有限原请求结束后普通调度会自然准入。所有输出512与对应
prompt串行greedy token比对；每trial14请求、总168请求。负轮与SLO失败保留。

## 默认暂停控制确认（11:02预声明）

首轮237068已验证上述容量见证与准入收益，但采用interval1，且20秒goodput退化。
追加同long_kv fixture的四策略各**6pairedrepeat**，seed**20261025**，所有输出512、
SLO20秒不变；只将所有策略DP pause共识恢复为实现默认的**32step**，保持sparse graph、
输入、预算、warmup、成本模型与候选不变。同allocation完整24trial/**336请求**。
该阶段验证默认控制下准入收益是否仍能覆盖真实暂停/迁移成本；不把新job与237068的
绝对吞吐差当作同步周期的严格因果消融（设备/时段可能不同）。所有负repeat、老请求
latency/SLO退化均保留；20秒SLO不为得到正goodput改阈值，主准入指标是A queue/TTFT。
