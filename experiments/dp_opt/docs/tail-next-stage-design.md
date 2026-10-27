# Tail TTFT下一阶段：真实可行性与分模型事务

2026-10-26由251533完整六策略实测驱动。本页是下一阶段设计与实验门槛，**尚未实现、
未提交新GPU作业、不构成收益证明**。旧75请求/10GiB结果与负面SLO证据不修改。

## 两个不同的优化问题

当前KV-only Code TTFT约55.9秒，queue仅1.42秒；保持后段执行不变时，仅消queue
最多改善2.54%。需要测同一100K Code在另一tenant resident-idle与真正Conv decode
共执行时的reserved→first、GPU attention/expert/通信与空隙。串行26秒对混合54秒
只是已有观测，不足直接证明kernel原因；不要把客户端该段时间写成attention占比。

当前最坏Conv虽logical总空间足够，却整请求二分装箱无解。任何placement-only ILP
都不能直接准入。下一trace不能再把这类tail称为可迁移收益场景或只凭aggregate free
给可行标签；需要逐请求KV、页几何、null/pinned、copy-before-release与实际probe。

## 预声明的下一实验门槛

- 固定两A800、双独立Qwen3-30B、DP2/EP2/attentionTP1、BF16/128K、默认sparse与32step。
- 先联合cost pilot：resident-idle/Conv N4/N8，对称DP布局与单hot布局；真实100K Code
  与长输出Conv。所有布局先物理capacity probe；不能支持的布局如实记录，不降100K/
  关闭graph或偷偷放宽单个策略预算。只在模型idle阶段测出的lookup不冒称联合校准。
- 若需新trace，显式heterogeneous Conv输出（长驻留占主体、部分较小KV），仍从Azure
  输入长度sample；扩展的输出须标synthetic，不能声称原Azure自然输出。少量100K Code
  来自同固定SWE-smith bank。不能定点修改某条最坏请求/固定到达来伪造Poisson收益。
- 设计阶段离线筛查whole-request装箱与实际容量，在线决策仍只用已到达信息。较小
  resident Conv需真实能与Code同卡；不是把logical下界当allocator充分条件。修改预算
  只能作为预声明独立factor且全部基线相同，不把预算收益归因新算法。
- 用实测KV×驻留时间、allocator余量和service速率重新计算RPS，开放环pilot校准后
  freeze trace；不沿用旧RPS再筛seed追50%。避免只剩2Code的P99声明，后续至少扩大
  Code样本并做配对多seed，所有请求drain，timeout需覆盖基线等待但SLO不可追结果修改。
- baseline为no_move/no_move_fair；机制消融为KV-only-global与KV-only-selective，必要时
  增加headroom。全策略同请求/预算/graph。分别报告Code/Conv/合并TTFT mean/P90/P95/P99、
  old-request latency/TPOT/SLO，不能只挑Code或只比较不同trace的历史最优。

## 通用selective transaction设计

不是按Code/Conv模型名硬编码：参与集合`S`是ILP候选中**有请求迁移的tenant集合**。
先枚举新请求attention home rank，仍解原有physical staged ILP；不改两模型权重/
EP组，也不跨GPU切分一个请求。两tenant都有迁移时退回原全局事务。

若只迁移一个tenant，可尝试只暂停该tenant全部DP/EP rank在默认32step安全边界；
另一tenant的prefill/decode继续。前提必须被实现和独立验证：

1. 全局准入lease禁止新请求/新的full KV reservation；非参与tenant已有full-reservation
   blocks保持不变，其mapped上界受控，完成只能释放。动态增长/外部绕过lease/不支持
   的allocator状态一律回退原安全全局路径，不能仅假定另模型KV不增长。
2. ILP所有copy-before-release峰值保守计入非参与tenant的KV上界；复制buffer/workspace
   另核物理guard。目的地pages未提交前不得用于attention，源owner及页pin保证不重用。
3. 冻结后只对`S`重验selected owner、decode state和有序复制计划；非参与tenant可推进
   或完成，但不能迁移、增加reservation或越过mapped上界。无法证明则安全abort/resume。
4. 成本目标按实际暂停的请求集合计`sum(W_m*F_m, m∈S)`，加入copy与真实混合干扰；
   非参与tenant不是“免费”，仍测copy带宽竞争、Code TTFT与老Conv latency。
5. 收尾核验参与tenant人口/token状态守恒，非参与tenant继续执行的token序列与串行
   对照一致；迁移期间四rank ownership/预算连续成立，控制成本与全部请求tail都入账。

这才可能在另模型prefill期间处理一个**原本真可装箱**的容量排队请求；当前251533
最坏Conv无解，单改暂停范围也救不了它。它不等于已有实现的GPU copy/inference并行，
更不允许通过跳过空attention rank的EP工作来伪造原motivation收益。

先完成上述CPU/race回归及真实容量/事务smoke，再同trace全基线收益；没有实测前不
宣称普适SLO最优或50%改善。若预算/真实allocator仍不可行，保留负结果，不盲重提交。
