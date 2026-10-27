# Live placement 实现与验证结果

本页记录233056的历史功能验证，不能当作收益实验。后续235093容量移交验证见
`allocator-pressure-status.md`；235320/235647/235712及确认阶段的端到端收益和负结果
见`benefit-status.md`，当前状态以该报告为准。

## 运行与 setting

作业 `233056`：**COMPLETED/0:0**，2026-10-19 **21:34:33–21:39:34**，
节点 `d1n41a14g01`，两张 A800-SXM4-80GB。先 migration，后 admission/ILP，
每阶段重新启动两个独立 Qwen3-30B-A3B-Instruct-2507 tenant。旧作业 `232987`
未运行，20:20:24取消并替换；此前暂估的次日上午开始时间不是实际开始时间。
2026-10-20 11:55查询用户队列为空；本次复核未提交新作业或重启 Aegaeon。

- 每模型 DP2/EP2/attention-TP1/PP1；同 checkpoint，独立权重、请求、KV 和 EP 组。
- 四个 worker 的 KV namespace 不同，同 rank 的两模型共址同 GPU；48层专家分片
  在两个 rank 上互补，覆盖每层128个专家。每模型/rank参数32,073,216,000 bytes。
- Sparse captures `1,8,16,32,64,128,256`，FULL_DECODE_ONLY，compilation mode0。
- 初始 A=4:3、B=5:3，共15请求，input64/output256；greedy、ignore_eos、无 APC。
- 实验 Scheduler 为 prompt+max_output 全量预留。原请求20个16-token block，
  logical KV30 MiB；每 physical page bundle96 MiB，每 tenant/rank虚拟KV2 GiB。
- 新 A 请求 input2048/output256，预留144blocks/216 MiB，planner保守收费288 MiB。
  两GPU受控 KV mapped 上限为576/384 MiB，不是整卡显存耗尽/OOM实验。

代码、事务保护和限制见 `../docs/live-placement.md`。CPU planner仅给全部 owner
已移动且非 pinned 的源页释放credit，目标按整页保守计费；不是任意碎片packing最优。

## GPU 功能结果

| 检查 | 强制迁移 | ILP 新请求准入 |
| --- | --- | --- |
| 原请求串行 greedy token 对照 | 15/15，每条256tokens | 15/15，每条256tokens |
| 新请求 | 无 | 完成256tokens；未做不同prompt的串行token对照 |
| A 布局 | 4:3 → 3:4 | 4:3 → 4:4，含新增请求 |
| B 布局 | 5:3 → 4:4 | 5:3 → 4:4 |
| 迁移 | A/B各一条0→1 | 先B、后A，各一条0→1 |
| 实际释放源页 | 两次均0 | B=0；A=96 MiB |
| 两GPU最终 mapped KV | 384/384 MiB | 576/384 MiB |

两阶段分别输出3840/4096tokens，不含各自独立的256token串行reference。
独立复核了原始SHA256、源码版本、请求唯一归属、block不重叠、page owner、
迁移后位置及保留请求的block table；KV payload hash由运行代码校验。
临时payload文件已删除，本次不能独立重新读取GPU数据并复算该hash。

### 容量与 ILP 选择

初始两GPU mapped KV为384/192 MiB，受控headroom各192 MiB；保守新请求收费
288 MiB。枚举结果rank0 optimal、rank1 infeasible，均仅针对当前保守planner。

1. B请求 `live-1-0-1-bca28686` 移到rank1，源pool保持192 MiB；copy peak384/288 MiB。
2. A请求 `live-0-0-3-a27d78b3` 移到rank1，源pool192→96 MiB；copy peak384/384 MiB。
   该请求独占A/rank0的page1，因此真正释放一个96 MiB page bundle。
3. 在同一暂停epoch准入新A至rank0，核验预留和预算后resume；最终A/B均4:4，
   两GPU576/384 MiB，未超过受控上限。

**这次腾出容量的是A自己的页，不是B的页。** B迁移用于surrogate中的负载均衡，
不能把“A/B都有迁移”当作“B释放物理页并让A复用”的跨模型容量证据。
强制迁移阶段也说明：移动30 MiB logical KV不必释放任何96 MiB源物理页。

### 同快照离线策略重放

仅 `ilp` 在GPU上执行；其他策略是本次使用同一保存snapshot重解，非live对照。

| 策略 | 移动 | 最终A/B布局 | objective，预测request-seconds |
| --- | --- | --- | ---: |
| ilp | B一条+A一条 | 4:4 / 4:4 | 0.314000 |
| own_model | 仅A一条 | 4:4 / 5:3 | 0.639308 |
| kv_only | 仅A一条 | 4:4 / 5:3 | 0.157000，仅迁移项 |
| no_move | 当前保守模型无解 | — | — |

只移动A也能满足planner容量约束；ILP额外移动B是因为预测不均衡成本
0.482308高于假定的单次迁移成本0.157。`kv_only`关闭性能项，objective不可与
其他策略直接当作同口径收益比较。N=8外推、成本可加和迁移带宽均未校准；
no_move没有在GPU上实际尝试，不能据此断言原生allocator必定无法分配。

进一步按真实snapshot推算：A/rank0已有两个page，128个槽位中81个被原请求与
null block占用，**还有47个mapped空槽**。新请求144blocks若先复用这47槽，
只需再映射两个page/192 MiB，而非planner收费的288 MiB；零迁移时GPU0
预计384+192=576 MiB，恰好满足现有预算。kvcached `KVCacheManager._alloc_impl`
会先使用avail_pages。本次没有实跑该路径，故这是静态geometry估计；但它说明
**当前场景没有证明真实容量不足或迁移的必要性**。下一组须先实跑原生零迁移分配，
排除保守计费制造的“无解”，再构造需要B释放物理页的压力场景。

## 开销与验证边界

每次复制30 MiB，迁移事务耗时166.1–167.6 ms；是host-staged文件传输加状态/RPC
操作的耗时，不是纯NVLink复制时间。admission记录 `pause_s=0.905132`，从暂停
调用前计到事务结束，**不包含finally里的resume**。原请求最大客户端chunk间隔：
migration1028.54 ms、admission1085.41 ms；不是稳定decode TPOT或kernel时间。

迁移前保存snapshot中，被移动请求只生成1–3tokens、computed64–66tokens，属于
早期decode。复制包含未来预留的未使用尾部，不是长上下文成熟KV压力测试。
小规模验证显式 `allow_uncalibrated=true`；不能沿用N128 motivation系数宣称收益。

下一阶段需要：验证上述零迁移packing估计；构造B释放独占源页并由A复用且对照
确实无法准入的场景；补新prompt的串行正确性对照；
在更长已生成KV和同sparse配置下实跑no_move/own_model/kv_only/ILP，对比
TTFT、TPOT、goodput及完整pause/resume，并校准双模型联合成本和迁移开销。

## 回归与证据

本地已有253 passed、1 skipped；allocation内planner/live tests **53 passed**，
包含真实Torch CPU字节roundtrip，TP/unmap/shutdown回归 **193 passed**。
退出日志有 `Keeping worker sockets: engine exit is unconfirmed` 警告；作业正常结束，
但不据此宣称所有socket已清理，也不做全局清理。

原始28个文件及逐文件hash、stdout/stderr、accounting和独立复核脚本归外层
`ParaCloud/evidence/dp-live-implementation-20261019/`。复核入口：
`python3 evidence/dp-live-implementation-20261019/verify-233056.py`（从外层workspace运行）。
远端原始目录：`~/run/kvcached-paracloud/state/live-placement-233056/`。
代码保持用户fork `gjghfd/kvcached` 的main；本次只更新验证记录，未commit/push。
