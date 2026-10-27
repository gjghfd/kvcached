# 32K Azure长度/Poisson trace：首轮真实收益对照

2026-10-21 15:35预声明。用户授权从理论阶段进入真实GPU实验。本轮是当前原型的
初测，不是完整多seed/多rate最终结论，不因结果为负而删除请求或改trace。

237899四策略386请求全部完成并独立复核，净收益为负。结果及下一阶段门槛见
`../reports/trace-benefit-results.md`；本协议保留执行前设置，不回写条件追正结果。

## 固定设置

- 两独立Qwen3-30B-A3B-Instruct-2507，DP2/EP2/attentionTP1，两个GPU共享checkpoint，
  不共享GPU权重/KV/EP组；A800优先可用，H100/H200也允许。
- max_model_len32768，虚拟KV4GiB/model/rank，受控aggregate mapped预算2112MiB/GPU。
  sparse captures1/8/16/32/64/128/256、FULL_DECODE_ONLY/compile0，DP pause共识32step。
- 复用Azure Conversation原48byte-window片段中的成对长度；条件P>=2048、O256..512、
  P+O<=32768，不强制输出512。此来源不是全CSV均匀reservoir，声明为长度sample初测。
- 两独立Poisson流总rate.60req/s、duration600秒、seed20261021，固定386请求，A192/B194。
  所有策略同绝对相对arrival表，所有到达预建async task；不因queue或pause改变到达。
- 四策略no_move/own_model/KV-only/ILP，首轮各1次，seed固定随机顺序；完全drain。
  no_move是同full-reservation自定义scheduler，并非未经修改native-vLLM渐进KV基线。
- 合成prompt固定重复算术token，无prefix cache，生成sample原O；完整配对tokens对照，
  四条不同长度的串行greedy reference额外校验，reference不进入测量窗口。

## 准入及成本

所有策略共用serialized direct-fit路径：在controller锁内读取actual mapped/free slots、
count-LB从可fit rank选择、提交并等待真实full-reservation ack；不global pause。
只有capacity-block的迁移策略才进入既有paused staged ILP。no_move等待capacity；
own_model只移自己的请求，KV-only忽略性能项，ILP使用现有N128单模型capture surrogate。
该surrogate对低N/长context/双模型尚未校准，不能称系统真最优。

以resident id/eligibility fingerprint限制重复规划；completion或population变化后可以
重新规划，不每200ms反复pause同一无解population。当前仍未比较wait action的预测成本，
movable全decode条件仍存在；因此净收益可能为负或ILP与KV-only动作相同。如实报告。
迁移完整暂停/求解/host-staged复制/resume入账，copy-before-release峰值必须不越预算，
迁移KV SHA/ownership与route update沿用已验证原型，不优化成NVLink以追求正结果。

### 15:54修复：首trace失败后的统一队列

237868首策略own_model排队超时，未完成四策略，不能得出ILP收益结论。19原始文件SHA
归档。记录74已完成请求，但最后population sample仅40完成：monitor在控制锁上等待，
不能只用最后progress声称completed=40。209控制决策/170迁移规划，仅1次实际迁移，
累计265.24秒control；每arrival自己的state cache未阻止不同排队请求重复pause，且
许多规划发生在global空间也不足，浪费控制时间。这是runner问题，不隐藏为硬件慢。

当前修复成两tenant FIFO、一个central dispatcher，不改变Poisson arrival或386长度。
队首请求可独立准入，另一tenant队首被堵不阻止它；只global实际free>=incoming
logical bytes且全部decode时规划，一request+resident-id状态最多一次；population
完成/变化后重新规划。该gate有意只测global-enough/local-blocked，可能排除靠压缩
碎片才能满足global容量的布局，不宣称完整allocator最优。四策略共用相同队列规则。
保留旧失败和请求timeout300秒，先重跑同一trace；若仍过载，再预声明低rate对照。

## 判据与证据

报告完整请求queue/TTFT/latency/TPOT、20秒SLO、arrival窗口完成数、backlog时间曲线、
drain时长、迁移数量/字节、control time。完成请求数与trace必须相同；超时为失败，
不能静默丢弃。固定offered rate下不把drain吞吐差直接称稳态吞吐提升。
每2秒记录population/真实mapped，所有trial容量复核、worker共驻留/topology/32K、
captures与配对greedy tokens检查。单seed初测只支持本trace，不作泛化显著性声明。

本地证据`evidence/azure-trace-run-20261021/`；远端
`~/run/kvcached-paracloud/state/live-trace-<job>/`；源码只归本研究目录。执行前明确文件
SHA保护备份同步，唯一allocation；每300秒查询，终态归档全文件SHA并独立复算。
