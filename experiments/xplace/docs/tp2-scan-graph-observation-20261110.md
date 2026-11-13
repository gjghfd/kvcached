# EP4TP2：失衡扫描的图调用观测与失败保留

## 状态与边界

2026-11-10：本地实现并验证，**尚未部署 GPU**。不修改已经提交的
`ep4-tp2-causal-calibration-20261110-r5`；其反向顺序复测也必须复用冻结包，
不能把当前开发中的 runtime 混入同条件对照。本文不报告新的容量、性能或预测结果。

目标是让总请求数 128/256 的扫描能够解释 capture 档位邻点，同时保留所有失败与截尾，
而不是通过删点或调整 seed 得到更漂亮的曲线。

## 实际运行路径

- `main.py` 每次运行先将 `XPLACE_GRAPH_EXECUTION_ALLOWED` 置零；只有包含
  imbalance phase 且配置显式 `observe_graph_execution: true` 时才启用。
  原 `DP_OPT_CAPTURE_EVIDENCE=0` 不变，不把旧计数器当作 replay 证明。
- `worker.py` 初始化后安装 `graph_execution.py` 的可选钩子，并通过已有
  `collective_rpc("placement_graph_execution", ...)` 执行 start/stop。
  关闭开关时不替换模型 runner 或 PyTorch 方法。
- 读取 runner `_determine_batch_execution_and_padding` 的**实际返回值**，包括
  DP 协调后的 mode、padded token 数、batch descriptor、microbatch 标志；
  `num_tokens` 同时支持位置和关键字参数。不读取 DP tensor 来新增同步。
- 包装 Python `torch.cuda.CUDAGraph.replay`，记录调用进入、正常返回或异常，
  同时读取该调用的 forward context。原返回值和原异常继续传播。
- 每个被测模型的 DP2×TP2 四个 lane 分别返回 PID、GPU UUID、clock ID、
  独立 worker epoch 与源码 SHA。四份必须齐全且身份一致；不按时间或形状拼接不同 lane。

这里的 replay 事件只证明 **host 调用了该方法**，返回时间不是 GPU 完成时间。
没有添加 CUDA synchronize/event timing；原生路径自身的同步不因此消失。
selection 的 `NONE` 也只表示该决策返回 eager 模式，不是设备空闲证明。
没有 replay 的 lane 可以是有效的零调用记录；不能凭零调用推断 idle、dummy 类型或丢失。
本模块不把 selection/replay 绑定成请求批次进度，也不产生 remaining_fraction。

源码绑定来自留存的 vLLM 0.26.0+cu129 / Torch 2.11.0+cu129 文件。
启用前核验 runner、vLLM graph wrapper、forward context、Torch graph 类四个文件的 SHA。
实际版本或方法所有权变化会拒绝启用/控制，不静默兼容。
源码副本与读取收据在 `evidence/ep4-tp2-ablation-20261108/scan-observation-sources-20261110/`。

## 有界记录与失败语义

每 lane、每 case 上限为 20,000 事件及 16 MiB 编码数据；start/stop 也计入上限。
保留 attempted、retained、dropped、invalid、bytes；丢失不清零、不补插值。
验收要求 start/stop、连续序号、字节/计数守恒、四 lane、源码/身份一致，且
所有 replay_enter 都有唯一正常返回。异常、未配对调用或丢失使观测无效。
这只是观测完整性检查，**不是独立预测或性能门槛**。

控制 RPC 期限 10 秒；超时/取消后保存已完成 lane 回复及缺失 lane，取消本地等待，
有界等待 0.1 秒并记录仍未结束的任务。取消本地等待不代表远端 RPC 已停止。
外部取消继续传播，不能被 graph stop 清理吞掉；失败后不得沿用该 case 标识重试。

扫描请求提交前保存全部计划请求、布局及目标。请求错误或取消发生时，在清理前保存
已完成请求与所有未完成请求的进度深拷贝；清理后才完成的请求写入另一文件，
不能改写原截尾状态。清理等待上限 5 秒，剩余任务与二次取消明确记录。
warmup/case 超时、错误、取消、预算跳过及未运行点都写入持久化状态。
即使性能窗口同时无效，graph 观测失败也会停止后续点，而不是继续使用坏观测。
已有 plan 的输出目录拒绝覆盖；没有自动重试或重新选 seed。

## 128/256 设计不变

容量修订 r1 的六个显式 JSON 仍是输入来源：所有点采用 2048-token 合成 context、
1024-token 合成输出，3 repeats、93 点，包含 capture 档位及 ±1 邻点。
本次派生配置仅增加 `observe_graph_execution: true`，不改顺序、seed、布局、
预算、超时、请求输出或 repeat。旧 4096 配置和容量不足判断完整保留。
不要从仍保留历史默认值的 `capture_aligned_blocks()` 重新覆盖这些配置。

`capture_bucket` 保留为输入布局代理；应另外展示实际 mode/padding/replay 分布。
DP 协调可能让多个不均衡布局采用相同 padding，所以不承诺曲线平滑、单调或产生收益。
每个 total/repeat 只相对自己的均衡点归一化；均衡点失败时相应比值为缺失，
不得换用更有利的 repeat。全部负点、容量跳过、缺失点与未运行点都保留。

完整 GPU 提交清单及观测开销的双顺序 off/collect 对照尚未冻结，也未提交扫描。
观测开销必须在相同请求/物理预算下实测；不能用本地单测或未观测的 r5 数据替代。
在此之前这些派生配置只是本地机制准备，不是可发布的无扰动性能曲线。
独立 15%/30% 预测门、长上下文/容量门及完整评分均仍未通过。

## 验证与保留

- 本地、隔离各 91 项通过，零跳过。覆盖真实 main 开关语句、worker RPC 委托、
  源码变更拒绝、四 lane、位置参数、控制取消、部分回复、记录溢出、不可变截尾和未运行点。
- 9 组原生方法 AST 的 CPU 对照覆盖 DP padding 再分派、eager、模式不匹配、
  缺少 context、Torch replay 返回及异常；原方法结果/异常与调用顺序一致。
  这不验证 GPU launch、SHM/RPC 载荷大小、真实时钟或分布式运行。
- 首轮 65 通过/14 失败及源码保留：Python 3.9 测试夹具在运行循环外创建 Event。
  修复为在 coroutine 内创建，未放宽生产逻辑。中间 79/90 通过版本也保留。
- 原生源码对照的首次失败保留：独立构造的假 tensor 内 lambda 被按对象身份比较；
  改为比较假 tensor 的标量内容，原生返回对象不变另由单测覆盖。
- 本地保留已有 `asyncio_mode` 插件配置 warning；隔离测试无该 warning。
  没有执行分时策略测试，没有重跑旧实验报告或修改 PAUSE。
