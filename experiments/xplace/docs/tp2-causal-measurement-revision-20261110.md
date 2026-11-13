# EP4TP2 因果测量修订：CPU 验证，不是 GPU 验收

## 固定共同预算

r6 的两个条件分别测得 `[38.75, 40.25]` 和 `[38.75, 40.5]` GiB natural budget。
修订计划按逐 attention-DP group 的容量下界预注册共同预算 `[38.75, 40.25]` GiB，
不是依据 TTFT、吞吐量或策略收益选择预算。3 GiB margin、256-token reserve、原始
prompt/output 长度、并发限制、请求顺序和 240 秒窗口均不变。

每个条件重启后仍独立测量 natural budget，保留原始证据；仅当两个 group 都满足
共同预算时才绑定控制器预算。任何一项不足则保存 `budget.json` 并在预热、数据面和
因果工作负载之前失败，不静默截短预算。非 causal 路径不变。窗口修复沿用上一轮
`origin + duration + drain` 的绝对单调时钟截止点，不重标 r6 的 239 秒记录。

## 有界采集容量及证据边界

采集上限从 128 MiB / 200,000 事件修订为 **512 MiB / 1,000,000 事件**。
仍然有硬上限；超限继续拒绝、计数并导致完整性验收失败。100 ms 周期、80 ms
deadline、零窗口内丢失、90% active-round transport coverage、missed-tick 和开销
门槛全部不变。512 MiB 是编码事件字节上限，不是 Python RSS 或报告序列化峰值上限。

新增容量诊断逐文件读取 r6 的四个采集历史，核对原始计数和保留字节守恒；没有重跑
原性能分析。保留事件的最大 sample 为 104,427 bytes，receipt 为 407 bytes，最大
总尝试数为 183,705。丢弃事件的尺寸未知，不作补齐或损失事件重建。

预声明 CPU 存储压力包络为 2,500 个 padded sample、400,000 个 padded receipt、
1,024 个 padded metadata，加 start/stop，共 403,526 次尝试；sample/receipt
尺寸至少覆盖观测最大值的 1.25 倍。这个测试重复和填充真实事件结构，验证序列化及
存储守恒，不验证重复事件的 cohort 语义、GPU RPC 或训练标签。

- 128 MiB 条件丢弃 302,493 次，序列不连续；即使最后 stop 可保留，也不能忽略前面的缺口。
- 512 MiB 条件保留全部 403,526 次、534,786,549 bytes，零丢失、零无效、序列连续且 stop 完整。
- 单独越过新上限的测试正确拒绝并记损失；不允许无限缓存。
- macOS 测试使用明确的 CPU clock identity 和 emission-time mock，生产 Linux clock 验证没有修改。

压力协议 r1 显式断言实际最小 sample/receipt 尺寸，测得 130,555/511 bytes，均超过
上述 25% 下界。首轮压力测试的计数仍保留，但其填充实现少算头部变化，不能证明所写
的最小 25% 余量；修订原因和两个版本的结果均保留，没有修改生产源码或筛选 GPU 点。

## 冻结候选与未完成事项

本地定向回归 154 项通过；从原 r6 upload 复制并只应用明确修复的隔离候选，完整启动
选择 **544 项通过、零跳过**，包含全部原 533 项以及 11 项新回归。提醒器也验证了
52 ms 早醒后继续等待及不前进时钟不自旋；旧提醒回执不改写，不重新唤醒旧作业。

候选相对 r6 只改变四个 runtime 文件、两个测试文件和四个输入/清单文件，新的
70 文件清单及 544 个精确测试身份已冻结。图扫描改动、历史分析测试 fixture 和
本地 cloudpickle 测试依赖未混入 upload；原 launcher、native/package/configuration
与 prompt bank 保持一致。源码及证据在：

`evidence/ep4-tp2-ablation-20261108/causal-budget-retention-repair-20261110-r0/`

上一轮冻结时，该候选尚不能提交：旧验收器硬编码 533 个启动测试。后续独立新版本已
绑定 manifest、544 个身份、startup contract SHA 和共同预算证据，验证和边界见
`tp2-causal-revision-analysis-20261110.md`。原门槛不变；不得修改或重跑旧 r6 拒绝报告
来获得通过，实际提交仍须满足空队列等保护条件。

r6 原包未执行的 block0 记录保留为“被修订要求取代、未完成”，而非补跑已知无效的
测量协议。修订协议重新要求 block0 `off → collect` 和 block1 `collect → off`，
无论第一个 block 的性能如何均需完成或保留明确失败，不能与 r6 混配；失败先诊断。
模型和特征协议冻结前不得执行 holdout。独立预测、完整评分、正式 TP2 配对消融和
混合拓扑全部仍未解锁。
