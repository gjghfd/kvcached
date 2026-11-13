# EP4TP2：图观测开销的双顺序作业

## 当前状态

2026-11-10：实际 launcher/main/worker 路径已接通，协议和本地不可变包已冻结；
**没有上传、没有提交，没有新的 GPU 数据或效果结论**。
310540 的实时状态仍需先恢复只读查询。不得据旧的 PENDING 快照占用第二个 allocation。
该作业不替代 r5 必须完成的 collect→off 复测；后者必须继续使用原冻结的 r5 包。

本实现是固定 batch 的机制开销实验，不是 policy baseline、联合 K16 预测验证、
长上下文验证或正式消融。复用名为 `causal_deployment` 的源码/配置证明字段，
并不表示运行了 causal collection。

## 预声明设计

| 项目 | 固定值 |
| --- | --- |
| 模型与设备 | 两模型 DP2/TP2/EP4，共享单节点 4 A800 |
| 目标 batch 布局 | 128: [64,64]/[128,0]；256: [128,128]/[256,0] |
| 合成请求 | 每请求 2048 context、1024 output；seed 20261110 |
| 顺序 | block0: off→collect；block1: collect→off |
| 重复与方向 | 各 block 一次；block1 旋转 heavy DP rank；同 block 两种模式完全相同 |
| 完整覆盖 | 每模式 4 case/768 请求；总计 16 case、3072 请求、8 对 |
| 扫描 warmup | 每模式另有 256 请求，输出 256；不混入测量 |
| 期限 | 每 case 150s、每 condition scan 900s；作业软期限 3300s、Slurm≤3600s |
| 准入启动保护 | 剩余作业时间不足 1500s 时不启动下一 condition |
| 进程 | 每个 condition 独立重启全部 engines；独立输出与 transfer 路径 |

这是在 93 点曲线之外预声明的四个均衡/极端布局开销 pilot。
**即使通过，也只支持这四个布局的开销结论，不证明所有 capture 邻点都无扰动。**
原 128/256、93 点、3 repeats 的曲线配置及所有历史容量修订保持不变。
完整曲线仍需自己的实际 GPU 数据与适用范围明确的对照。

两个 condition 共享相同模型路径、输入文件、物理 GPU、源码和配置。
CLI 固定 prompt 模式及 256 reserve，不缩 reserve；此合成扫描在 case 内直接启动
固定数量的请求，没有额外 controller inflight 限制，scheduler 的每 lane max_num_seqs 仍为256。
这不是对真实共享 waiting queue 的消融，也没有改变真实 Conv 的输出长度。
预算使用相同的 natural-budget 规则及 3GiB margin；验收还需核对两边实际预算一致。

## 开关与证据

`scan_job.py` 为每个 block 按固定顺序调用独立 `main.py` 子进程。
off 配置不安装 graph hook；collect 配置才安装并在 case 内开启有界 buffer。
两边均关闭 nested timing、live observation、batch links 和旧 capture 计数器，
不允许从父环境继承一个隐藏的观测条件。

main 在 GPU 初始化前、全部 worker 就绪后和成功关闭后执行已有源码/配置核查。
核查包括完整 runtime、vLLM/kvcached/cloudpickle Python 树、原生/launcher 文件、
Torch graph 源码、实际版本和八个 worker 的配置/clock/PID。
包内的所有文件另由 `MANIFEST.sha256` 绑定；启动测试使用显式非分时列表且不得跳过。
当前冻结依赖先前读取的源文件；新的 compute-node 核查还没有发生，变化必须拒绝。

复用已有 `run_child` 的共享 allocation deadline、每秒 process-group CPU/RSS 采样、
日志与有界进程组清理。自然退出失败、残留进程、timeout 或 signal 都不接受。
失败及未运行 condition 写入状态，没有自动补跑或成功子集验收。
收集结束只返回 `collected_not_overhead_accepted`，不自动通过任何门槛。

## 开销判定

必须具备两个顺序 block、全部八对、全部计划请求，以及合法窗口、零抢占、
完整 graph 证据和进程/源码/预算证明。未完成、丢失和失败留在分母中，门槛不通过。

三个预声明主指标为：

- 原扫描的 `steady_step_s`。
- 全请求 `mean(first_token_at - submitted_at)`。
- 全请求 `mean((completed_at - first_token_at)/(output_tokens - 1))`。

最后一个指标是前端流完成口径，不是逐 GPU token 时刻。
每对报告 `collect/off - 1`；每个指标均要求中位开销≤5%、最坏对≤15%，完整覆盖率100%。
启动成本、总 wall time 与 sampled CPU/RSS 单独报告，不冒充主 case 指标或真实峰值。
原独立预测的15%/30%门槛、容量门和正式效果指标没有修改。
跨 block 的离线验收尚待实现并冻结；不能凭子进程 exit0 宣称开销通过。

## 本地验证与产物

- 针对性本地/隔离各130通过；完整隔离包启动选择238通过，均零跳过。
- 真实 CPU 子进程退出、独立日志、固定 deadline、源码/协议/CLI 变更拒绝、
  实际 shell 分派语句和 main 源码核查开关均已测试；没有运行 GPU 或 Slurm。
- 首轮126通过/1失败：本地 shell 未因复合条件守卫而终止，冲突变量仍调用了 driver。
  修复为显式错误分支和 `exit 2`，四种冲突分别验证；原失败包与修复记录保留。
- 中间127/130通过快照保留；本地已有 asyncio_mode warning 未改，隔离无此 warning。
- 本地包：`evidence/ep4-tp2-ablation-20261108/scan-job-bundle-20261110-r0/`。
  `upload/inputs/graph-overhead-protocol.json` 绑定四份配置；`build.json` 记录整个包和238项启动结果。

下一步先恢复310540的只读状态并完成其终态诊断与预声明复测。
本图开销 pilot 只有在当前 allocation 已结束、数据/协议验收路径完整后才能提交；
正式 EP4TP2 成效和混合拓扑仍未满足前置门槛。
