# TP2四范围观测的离线分析契约（2026-11-10）

## 当前用途

`tools/analyze_nested_timing.py` 只做归档后的机制描述。它不修改冻结r4，不提交、取回或重跑作业，不拟合估计器，不接入评分，也不放行任何实验总门。准备时309773仍排队，当前测试全部是合成数据，不是真实作业验收或性能结论。

分析沿用原100ms callback-gap阈值，作用于原始callback秒时间差，不除以该次callback携带的token数。原始秒时间戳仅为与host范围对齐而近似换算为纳秒。全部callback间隔、合并callback的可见token增量、24例和全部12个off/on差异均保留；无长间隔例和零覆盖worker也保留。单个callback已含全部输出时没有可观察的callback间隔，时间占比为不可用，而不是推断为无等待。

## 区间计算

- 从通过SHA核对的归档 `runtime/nested_timing.py` 加载范围解析器，重查八个worker的四范围、父子关系、配置及恢复标记。
- 每个父调用的直接子区间先取并集；父调用剩余host区间是父区间减该并集。`runner_to_list`包含在bookkeeping内，不能与bookkeeping直接相加。按测量窗口和每个长callback gap分别裁剪。
- `sum_call_inclusive_ns = sum_call_direct_child_union_ns + sum_call_exclusive_ns` 是逐调用分割恒等式，不是设备时间或wall time。跨线程调用可以重叠。
- `inclusive_union_ns` 和 `exclusive_union_ns` 是同标签区间各自的并集。所有标签exclusive区间的总并集必须等于外层worker范围的并集；不同标签之和若超过该并集，单列 `cross_label_overlap_ns`，不当作更多wall time。
- 未覆盖窗口命名为 `outside_observed_host_scopes_ns`，只代表未被这些挂钩覆盖，**不是idle证明**。host调用返回不是CUDA completion；to_list包含原有同步依赖，不能标成纯拷贝时间。
- 异步/线程heartbeat仅描述既有长延迟区间的重叠，可能使用gap结束后的事件，因此是回顾性信息，不是在线预测特征。合法的提前唤醒且lateness为0不形成延迟区间。

## 来源与不完整结果

先用现有artifact auditor验收归档，再把已记录的audit SHA传给分析器。分析器核对audit字节、job ID和门状态，所有读取的原始文件必须匹配audit中的归档成员SHA；计划与冻结计划一致，执行的范围解析器还必须匹配冻结源码SHA。不会重写或重新运行既有验收报告。

完整结果必须保留24例、336请求和12个原始差异对。个别文件缺失、SHA变化、溢出、lane缺失、scope损坏等形成对应case错误，其余case槽位仍保留，最终状态为failed，不接受成功子集的汇总。若audit本身未通过但仍有可信成员清单，能读取的case仅作带失败标记的部分证据；若来源/计划/解析器无法核实，则记录前置失败而不编造case数据。所有输出门固定为false。

输出路径必须不存在；失败报告也不覆盖原文件。不从缺失数据推出性能结论，不扣除测量开销，不筛选seed、on/off条件或阈值。

## 实际使用

以下只在309773终态、单次归档和artifact audit完成后执行；`RECORDED_AUDIT_SHA256`必须来自已记录的验收收据，不应临时对未知报告计算SHA以绕过来源核对。

```sh
/usr/bin/python3 kvcached/experiments/xplace/tools/analyze_nested_timing.py \
  --evidence evidence/xplace-ep4-tp2-nested-timing-20261110-r4 \
  --audit evidence/xplace-ep4-tp2-nested-timing-20261110-r4/artifact-audit-309773-v1.json \
  --audit-sha256 "$RECORDED_AUDIT_SHA256" --job-id 309773 \
  --output evidence/ep4-tp2-ablation-20261108/nested-mechanism-309773-v1.json
```

测试、源码快照、差异及r4未修改证明记录于 `evidence/ep4-tp2-ablation-20261108/nested-analysis-preparation-20261110.json`。真实GPU机制、有效估计器、完整评分、独立预测、容量pilot和配对消融仍是后续工作。
