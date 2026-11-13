# EP4TP2 因果采集的离线标签与开销分析

此工具消费新校准作业的终态本地归档，不训练预测器，也不宣称 EP4TP2
效果通过。310540 的部署源码、负载、采集周期和门槛均未修改。

## 全 cohort 的 K16 收讫目标

`tools/analyze_causal_calibration.py` 为从 collection origin 到不可变 cutoff
的每个 100ms tick 输出一行。无 sample、采集跨过 cutoff、空 cohort、
源不可用和目标截尾均显式保留。查询时刻使用采集完成的 `as_of`，不把
后来收到的状态回填到 scheduled/start 时刻。

每次查询把 cohort 与此前已记录的全部 arrival、submission、receipt、
terminal 核对，不能遗漏共享队列中尚未提交的请求，也不能把尚未结束的
请求标成结束以缩小目标集合。请求目标是其 as-of 累计收讫数再增加16
的首次前端收讫；合并输出只有一个收讫时刻，不插值。只有全部成员达到
目标，联合时长才取各成员时长的最大值。EOS不足、错误、取消和窗口截尾
保持 null 目标及下界。快请求达到某个查询的里程碑后仍继续运行；后续
查询不停止其干扰或假定未来输出长度。

特征与未来标签分离。仅暴露白名单中的因果字段，来源需满足新鲜度、
clock、PID、epoch、lane 和配置绑定。worker PID 还需对应终态部署证明。
同形状或相近时间不构成同批次；latest-only 快照不是完整事件历史，
host 返回、scheduler 计数与前端收讫都不能直接当作 GPU 进度。

## 有界事件保留

冻结 r5 的 collector 关闭后，正常清理 stream 仍可能增加 emit 拒绝计数。
这不是自动豁免丢失：只有首尾 start/stop、递增 sequence、retained 字节数、
attempt/drop 守恒以及不可变 cutoff 全部核对成功，才单独报告窗口内丢失与
关闭后拒绝。窗口内仍要求零丢失、零 invalid；原始 `dropped` 和
`retention_lossless` 原样保留。缺 stop 或守恒失败不能猜成关闭后拒绝。

采样计数同时核对实际 sample 与 missed-tick 事件，独立检查全部来源，
不只信任 `transport_complete`。覆盖率沿用冻结部署协议的90%，missed
比例上限仍为10%；这不是15%/30%的独立预测误差门。

## 全请求配对开销

保留两个顺序块、每块4个 case、全部8对和所有计划请求。两条件必须
具有相同物理设备、预算、并发、reserve 与 workload 条件，分别重启
engine；缺失条件或请求不能被成功子集均值掩盖。

240s 的 token 数由共同使用的 cadence 给出保守区间，窗口外 count
不能当作边界精确值。不可变 incomplete 行优先于清理时新增的完成行。
每请求 prefix TPOT 沿用冻结公式；没有首 token 的 TTFT 不填成 cutoff。
开销区间按最坏上界判定原有5%中位数、15%最坏对门槛，不更改门槛。
CPU/RSS 仅报告1Hz采样统计，不称精确累计 CPU 或真实峰值 RSS。

终态源码、533项零跳过启动测试、前后端/8 worker 证明、顺序、清理、
dataplane 与实际 workload 另行核对。CLI 退出0只表示分析完成；验收须读
报告中的布尔门。Slurm accounting、归档 SHA 和一次性取回仍需外部验收。

## 使用与验证边界

仅在终态归档成功后运行一次新的输出目录：

```sh
python tools/analyze_causal_calibration.py \
  --job 0=/path/to/terminal/job-310540 \
  --analysis-protocol /path/to/causal-analysis-protocol-20261110-r1.json \
  --output /path/to/new-analysis-directory
```

第二个预声明的 collect→off 块终态后，另建输出目录并添加 `--job 1=...`，
不要覆盖第一个块的负结果。当前测试使用合成 CPU 事件与终态夹具，包含
独立逐事件首次命中/max 核对、全部缺失 case、截尾/取消、错误来源、
非法区间及单块 CLI 落盘。原有 GPU 负结果与所有本地失败记录均保留。
测试通过不等于采集开销通过，更不等于独立预测或正式消融通过。
