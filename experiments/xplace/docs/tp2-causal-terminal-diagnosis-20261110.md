# EP4TP2 因果校准 r6：终态测量完整性失败

## 结论与适用范围

作业 310921 于 2026-11-10 18:36:22–19:17:39（北京时间）运行，Slurm 为
`COMPLETED/0:0`。这是 block1 的 `collect → off`，不是预测或系统收益验收通过。
冻结分析器生成报告成功，但三个开销指标均为 0/8 可用配对，独立预测门槛仍未通过。
block0 未执行，不能使用失败的 r5 替代。

已保留两个条件各四例、各 44 个请求的完成与不可变截尾记录。原始 prompt 与输出长度
全部保持不变。533 个启动测试的身份完整匹配，70 个部署文件及计算节点复制的 69 个
文件哈希匹配；两个条件各验证八个 worker、同四张 A800、每模型 48 层的全部 128 个
expert 分片、八次数据面迁移。两条件的进程清理警告保留，未用退出码代替效果门槛。

## 三项测量缺陷

1. **观测窗口提前约一秒。** 八例 `censored_at_s` 均约 239.00 秒，未达到预注册的
   240 秒。`trace_runner` 将 trace origin 设为当前单调时钟加一秒，却立即开始
   `duration + drain` 超时。长 Conv 在有限窗口内未完成本身是允许的；错误是窗口端点
   不符合协议，不能把现有数据重新标为 240 秒或放宽分析器阈值。
2. **物理预算不相同。** `collect` 的 attention-group 预算为 `[38.75, 40.25]` GiB，
   `off` 为 `[38.75, 40.5]` GiB。每个条件重启后独立测量 natural budget，引入了
   0.25 GiB 差异。后续必须预先确定有容量依据的共同预算、对两个条件验证可用性并在
   不足时拒绝运行；不能事后归一化掉差异，不能缩减 3 GiB margin 或 256-token reserve。
3. **16 并发的采集历史不完整。** 两例达到 128 MiB 字节上限，分别记录 70,130 和
   71,189 次丢弃，缺少终止 stop marker。不能把这些丢弃解释为停止后的正常拒绝。
   `receive` 已更新实时 cohort，但 `emit` 可能丢弃历史事件，因而出现
   `cohort_receipts_differ_from_events`。必须修复有界内存下的完整记录或预注册足够的
   存储容量，并重新验证开销，不得筛掉丢失的点。

两例四并发均保留唯一 stop marker，窗口内事件丢失为零；其 4/1 次拒绝可严格归为
关闭后的事件。该事实不挽救提前截尾和预算不等导致的拒绝。原始 incomplete-round
计数也不能冒充预注册的 active-round transport coverage。

## 本地窗口修复及后续边界

新增 CPU 回归在原实现上复现负的短窗口端点（0.1 秒窗口约在 trace origin 前
0.899 秒结束）。修复将等待时间锚定到 `origin + duration + drain`，保留原始长度和
不可变截尾机制。源码修改只在工作树中，未覆盖 r6 上传包、冻结分析器或原始结果。
针对性 CPU 回归 60 项通过、零跳过；测试回执位于
`evidence/ep4-tp2-ablation-20261108/causal-terminal-diagnosis-20261110-r0/`。
该修复尚未 GPU 验证，也尚未完成整个修复包的启动契约冻结。

预算和采集修复完成前不提交新作业。原 r6 同源 block0 要求仍保留为待处理项，不能
盲目补跑，也不能将修复版与 r6 混作有效配对；任何修订协议都必须重新预声明并完成
两种顺序，不得按已观察到的性能选择顺序或样本。独立 holdout、正式基线/消融和混合
拓扑均尚未解锁。

## 不可变证据

- `evidence/xplace-ep4-tp2-causal-calibration-20261110-r6/fetch-310921.json`
- `evidence/xplace-ep4-tp2-causal-calibration-20261110-r6/terminal-preflight-310921.json`
- `evidence/xplace-ep4-tp2-causal-calibration-20261110-r6/terminal-audit-310921.json`
- `evidence/xplace-ep4-tp2-causal-calibration-20261110-r6/analysis-r0/report.json`
- `evidence/xplace-ep4-tp2-causal-calibration-20261110-r6/failure-diagnosis-310921-r0.json`

原始归档 SHA256：`43bf38b19907f7fa0bd5812f3faefe22faee3e1ea8267cf2368f156808ad2c42`。

后续本地修订见 `tp2-causal-measurement-revision-20261110.md`；该修订不改变本报告的
r6 拒绝结论，也不代表 GPU 验收通过。
