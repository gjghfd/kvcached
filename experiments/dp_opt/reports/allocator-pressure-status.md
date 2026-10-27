# Actual allocator 对照与跨模型容量验证

2026-10-20，Asia/Shanghai。作业235093的四case已完成并独立复核，证明受控聚合KV
预算下B释放源页可让A准入；不是整卡OOM或完整算法性能收益。固定两个独立Qwen
tenant，DP2/EP2/attention-TP1，原有sparse graph。

本页只记录235093的功能对照；后续端到端排队基线和placement收益见`benefit-status.md`。

## 执行状态

作业 **235093** 为 **COMPLETED/0:0**，实际运行于 **2026-10-20 15:53:39–16:06:52**，
在 `d1n41e18g01` 使用两张 **H200**。此前暂估次日上午的开始时间不是实际时间。
61个原始文件SHA256已核验；四case输出及迁移ownership/page/预算独立复核通过。

| case | 原请求token对照 | 新A token对照 | 结果 |
| --- | --- | --- | --- |
| packing-no_move | 15/15 | 1/1 | 零迁移准入，rank0实际新增192 MiB |
| pressure-no_move | 13/13 | 未准入 | 两rank实际分配均超受控预算 |
| pressure-own_model | 13/13 | 未准入 | 只迁移A的ILP无解 |
| pressure-ilp | 13/13 | 1/1 | B一条0→1释放96 MiB，A新增288 MiB后准入 |

压力ILP实际复制30 MiB，迁移事务 **475.52 ms**；记录的pause约 **5.173 s**，不含
最后resume。最终GPU mapped480/288 MiB，A4:3/B3:4。实际allocation probe后回滚
恢复原请求block和mapped页，压力no-move峰值576/480 MiB，超过预算480/288 MiB。
基线这一轮未排队提交新请求、case间probe和solver开销也不同，**不能当作收益对照**。
下一轮需要保持相同到达请求，在基线等待自然释放，并把完整等待/规划/迁移开销计入。

本地 **269 passed、1 skipped**（无Torch跳过真实字节roundtrip），已知asyncio_mode
配置warning。新增probe/rollback/几何guard、CPU压力ILP与相关TP/unmap回归通过；
真实Torch、shutdown和GPU功能检查在allocation内通过。九个明确文件已按旧SHA
核对、备份、原子同步，受保护Aegaeon补丁hash不变。
远端证据 `~/run/kvcached-paracloud/state/live-placement-235093/`，日志
`~/run/kvcached-paracloud/logs/live-placement-235093.{out,err}`。

## 验证目的与顺序

同一双卡allocation依次运行四个fresh-engine case：

1. `packing-no_move`：复现233056的A4:3/B5:3，原请求64/256，新A2048/256。
   实际allocator先在暂停epoch临时预留新请求KV，记录peak，free并drain retired页；
   验证原block/page ownership恢复。预计rank0新增192 MiB并能在576/384 MiB预算内
   零迁移准入；rank1不符合预算。选rank0真正提交请求，不经过保守ILP的no_move判断。
2. `pressure-no_move`：A3:3/B4:3，原请求64/256，新A **2816/256**；预算480/288 MiB。
   两rank分别执行实际分配/回滚探测，预计均超受控KV预算，记录拒绝；原请求继续完成。
3. `pressure-own_model`：相同压力布局，probe后只允许ILP移动A，预计无可行方案。
4. `pressure-ilp`：相同压力布局，probe后枚举rank+ILP，预计移动B/rank0独占页owner
   至rank1，释放96 MiB，然后新A准入rank0；记录释放前后和准入前后的mapped容量。

所有原请求和准入成功的新请求均检查准确256token输出，并分别对照64token和
新请求prompt的串行greedy reference。新请求串行reference在原请求提交前执行。

## 为什么新请求改为2816tokens

每page bundle96 MiB/64blocks；原请求20blocks，新请求3072tokens=192blocks=288 MiB。
A3:3使两rank原请求均位于包含null block的首page，不能通过移走A释放该page。
B4:3仅rank0存在一个非pinned、单owner的额外page；现场必须核验，不假定页编号。

初始两GPU mapped KV预计288/192 MiB，额外headroom192/96 MiB。
即便把目标rank全部原A请求移走，新A的192blocks+不可释放null block仍需至少
4pages=384 MiB；保持B不动时两GPU最低需求为576/480 MiB，分别超过480/288预算。
因此**A自身重排/碎片打包不能替代B释放容量**。更短的2048prompt没有这个下界。

预计移动B30 MiB KV时GPU1 copy peak192→288 MiB，释放GPU0的B页后288→192 MiB；
再为新A新增288 MiB，最终GPU mapped480/288 MiB，A4:3/B3:4。
CPU同几何参考中rank0 optimal、rank1 infeasible，own_model/no_move均无解；不是GPU结果。

## 判断边界

- probe使用真实vLLM/kvcached reservation，但受控预算是**聚合KV上限**，不是整卡OOM。
  可以临时映射超过该预算再回滚，用实测bytes决定是否允许准入。
- “B释放后A扩映射”验证跨租户mapped容量可用性，不宣称复用同一个CUDA物理页handle。
- 这是早期decode/全序列预留的容量功能验证；probe也会暂停服务，不能将各case的
  总耗时直接解释为调度性能收益。仍未校准双模型性能成本或优化host-staged迁移。
- 任一几何、epoch、rollback或token对照不符均让实验失败，不把预期当结果。

原始文件和同步记录归外层 `ParaCloud/evidence/allocator-pressure-20261020/`。
入口 `live-placement.sbatch` 默认 `allocator_suite`；旧233056流程可显式用
`DP_OPT_LIVE_MODE=suite`。本次不commit/push或重启Aegaeon。
