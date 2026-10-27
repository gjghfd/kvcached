# Attention-DP 不均衡：实验协议与后续场景

此文只负责 measurement protocol 和待测设计。已完成 setting/结果的唯一入口为
`../reports/motivation-summary.md`；机器可读配置/状态为 `experiment-matrix.json`。
主服务基线是 native vLLM+kvcached 的 CUDA graph；旧 eager 曲线是独立对照，不合并。

## 研究问题

固定 expert placement，改变请求/真实 KV 所属的 attention-DP rank，量化不均衡的系统代价。
这为联合 load/KV-headroom placement 提供约束，但单模型实验不验证跨模型 KV 干扰、migration 或 admission/SLO 收益。
两个租户可复用同一真实 Qwen BF16 checkpoint，必须维持独立请求、KV、EP 组和客户端；禁止 dummy 权重、跨租户合批或 prefix 复用。

## 主实验 A：CUDA graph

单模型、两 GPU、DP=2/EP=2/attention TP=1。精确输入/输出 token IDs，显式
`X-data-parallel-rank` 指定 attention home。全局 batch B 保持不变，布局为
B/2:B/2、B:0、0:B；当前主测 B=128/256/32，中间不均衡度尚未执行。

每个配置三个配对 repeat；每轮为均衡起点、随机排序的两个极端、均衡终点（seed=42）。
相同配置各布局复用同一 prompt 集合、轮换 prompt 到 rank 的分配。Barrier 发出闭环并发请求；
不是固定 request-rate arrival 实验。温度0、seed42、精确输出长度、ignore_eos。
Correctness、warmup、untraced controls 与主 trial 计数分离；每轮前后检查 idle 和物理 KV 回收。

`FULL_DECODE_ONLY` 主窗口必须实际为 FULL。Trace 要验证真实请求人口、各步 one-token decode、
空 rank 连续 dummy 参与、unpadded/padded token 数及无 DBO。稳定共同窗口裁首尾各32 step，
每 rank 至少128 step；采样 waiting=0、无新增 preemption、KV使用率<90%。
虚拟 KV reservation 不等于 mapped physical backing；禁止把容量触发排队混入 compute penalty。

主 C 是非空 ranks 中最大的 host execute-start 间隔 p50，不是 CUDA kernel 时间。
每 repeat 的 `R_r=C_skew/mean(C_balanced_start,C_balanced_end)`；报告
`R=mean(R_r)`、`loss=1-1/R`。三个 repeat ratio 的 Student-t(df2)95%区间转换为 loss 区间，
不把 step 当独立重复。目标要求两个 hot-rank 方向的 loss95下界都>=30%。
Burst/TTFT/ITL/TPOT 为辅助指标，禁止用 prefill 主导的 burst 替代稳定 decode 目标。

## 机制证据与 profiling 边界

Graph 模式按 rank 最大 graph batch 对齐；空 rank 跑 dummy forward，仍处理其 EP shard。
Dummy 序列 metadata 和 -1 KV slot 不等价于真实长上下文请求，不因 padding 分配真实请求 KV。
因此 loss 包含 graph/padding/投影/router/通信/expert 变化，不能从结果反推纯 attention 占比。

主路径只内存缓冲 trace、trial 后 flush；不加逐 step 文件 I/O、CUDA同步或阶段 hooks。
Phase diagnostics 要独立于主性能 trial，并报告扰动；顺序 traced/untraced controls 不构成零开销的因果证明。
后续应同节点同配置对比 eager/graph、扫描中间 I 和 capture 档位、用低扰动 GPU kernel profile 分离实际耗时。

## 历史 eager 对照

`../motivation.sbatch` 实验A固定16请求128/256，I=abs(n0-n1)/16，扫描
8:8、10:6/6:10、12:4/4:12、14:2/2:14、16:0/0:16，三次首尾配对。
长度对照和 batch/KV 筛选/确认分别由 `../motivation-lengths.sbatch`、`../motivation-batch.sbatch` 执行。
旧筛选只能供 selection，不能混入新 allocation 的 confirmation CI。完整 setting 和失败边界见各专项 report。

仅在理想线性 attention 模型中，`T_s=A_s+M`、`T_b=A_s/2+M`，
吞吐损失为 `0.5*A_s/T_s`；均衡 attention 占比 f 下为 `f/(1+f)`。
该公式依赖线性 scaling/不变 M，不直接适用于 graph padding，也不是实测 kernel 归因。

## 待测 B：双模型同向/反向布局

两个独立 resident tenant 在同一对 GPU 上，各自DP2/EP2/TP1、独立benchmark进程。
固定每模型请求数和长度，比较两模型都均衡、同向倾斜、反向倾斜，并交换 hot rank。
原始矩阵保留16请求下12:4和16:0候选，执行前需结合主 graph batch 区间重新冻结配置，
不能宣称这些旧候选已测。graph 中间 I 的 230346 sparse 部分完整验证后睡前取消，
231467 在同 allocation 完成 sparse/dense（COMPLETED/0:0）；冻结配置见 `curve-protocol.md`，结果见
`../reports/curve-results.md`。

分别记录每模型及整卡负载、padded batch、真实KV与等待/抢占；先保持KV非瓶颈隔离compute，
再在可证明的物理KV headroom压力下测 tradeoff。两卡跨模型总请求数相等不保证各模型内部均衡。
权重同checkpoint不是跨租户 KV共享；实际物理页释放/复用须有数据面证据。

## 待测 C：等请求数、不同上下文

原计划8短/8长请求、短输入128/长输入512、输出256；比较长度集中于不同rank与每rank混合，
交换方向。原候选仍未执行；正式配置应沿用graph基线、独立预先验证容量并冻结输入构成。
同时报告真实KV字节、最大context、capture/padded batch，不能用请求数代替attention/KV工作量。

## 从 motivation 到 placement

成本函数必须包含模型内部DP最大batch、capture档位、真实context、EP同步与rank-local KV。
跨rank migration需要正确的request ownership commit、live页copy、源页释放和目标可复用容量验证；
先跑 correctness/数据面，再比 KV-only、load-only、联合策略与强 greedy baseline 的 goodput/SLO。
具体系统设计归 `placement-design.md`；rank 枚举与跨模型 ILP 的公式/约束归
`ilp-scheduler-design.md`。当前只完成 CPU 代数 reference，不记录未测迁移收益。
