# 异构双模型长度 profile：长输入 Code 触发、长驻留 Conv 调整

2026-10-21（Asia/Shanghai）。本轮只做本地 CPU 数据分析、容量近似和离线 ILP
可行性验证，没有提交 GPU 作业，没有同步到集群，也没有改 serving 默认配置。
这不是新的实测吞吐收益；237899 的负结果完整保留在 `trace-benefit-results.md`。

## 1. 用户澄清后的机制

**Code 不必驻留很久。** Code 模型的新请求输入长，prefill 后需要较大的单卡 KV；
Conv 模型的既有请求输出较长、驻留较久，KV 分散在两个 attention-DP rank。
即使两卡合计空闲足够，两边各自空闲也可能都容不下 Code。调度器迁移同属 Conv
的请求到另一个 rank，集中可释放的物理页，让 Code 提前准入。

迁移不改变请求的模型，也不把 Conv 的 KV 交给 Code 解释；释放的是共享 GPU 的
物理显存。Code 使用自己的 pool 映射新页。两模型分别使用独立权重、request/KV
状态和 EP 组，共用 checkpoint 不等于共享权重或 KV。

所以要比较的是 **Code 的自然等待时间 vs 控制/迁移开销及老 Conv 的性能代价**，
不是让 Code 长 decode 来摊销迁移。Conv 总驻留长也不自动保证等待长：许多老请求
的最早完成可能仍很近，必须看剩余时间和真正能释放的页。

## 2. 数据来源与统计口径

原始证据：工作区 `evidence/mixed-workload-profile-20261021/`。每个源文件保存
URL、大小、SHA256 和响应元数据；抓取日期是 2026-10-21，不把 release 的名字
或文件更新时间当作实际请求采集日期。

- Azure：官方 `AzurePublicDataset` 的 `AzureLLMInferenceDataset2024.md` 和
  `dataset-llm-2024` release，读取完整 Code/Conversation 两份 CSV 的每一行。
  共 **44,107,694** 行，无非正输入/输出；不是此前的 byte-window 样本。
- LongBench v1：官方 THUDM 数据包，21 个任务共 **4,750** 条，排除重复 `_e`。
  v2：官方数据文件，6 个 domain 共 **503** 条。
- LongBench 用 Qwen3-30B-A3B-Instruct-2507 正式 tokenizer 实际分词，关闭
  truncation/padding。P 是单 user chat 的 context+question/input/choices；不是
  官方评测完整模板。reference answer tokens **不是实测生成 O 或驻留时间**。
- Azure 只有成对的 P=`ContextTokens`、O=`GeneratedTokens`，没有原始 prompt。
  后续长度重放只能声明为 length-trace + synthetic text，不能称原语义请求重放。
- 分位数是全量 nearest-rank；保留 P/O 联合分布，不能独立重采样破坏相关性。

关键源 SHA256：

| 文件 | SHA256 |
| --- | --- |
| Azure Code CSV | `71de5c55cbc35f8f1ed0b6b7806b4cd1e9764b0058469725a6aac98023a1448f` |
| Azure Conv CSV | `a0cc9b969a9bbf0fd811802cbf4323edd3a209ace791e3799ad4f9207f213941` |
| LongBench v1 ZIP | `cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64` |
| LongBench v2 JSON | `15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2` |

## 3. 完整 Azure 分布

| 数据 | 请求数 | P 均值 / p50 / p90 / p99 / max | O 均值 / p50 / p90 / p99 / max |
| --- | ---: | --- | --- |
| Code | 16,803,695 | 2511.28 / 1930 / 6251 / 7685 / 7743 | 22.69 / 8 / 43 / 271 / 5000 |
| Conv | 27,303,999 | 1631.58 / 928 / 3830 / 6683 / 7999 | 105.51 / 41 / 342 / 694 / 1500 |

独立抽两类请求，`Pr(P_code > P_conv)=62.8182%`，方向符合用户设想，但并非每条
Code 都更长。Code `P>=4096` 占 18.4876%，Conv 占 8.7699%；Conv `O>=256`
占 15.1753%。Code/Conv 的总长度最大值分别 12,691 / 9,158。
**这两份 release 没有 P>=8192 的请求，更没有 32K–1M 输入。**

长度引导的未校准近似：

    b(P,O) = 1.5 MiB * ceil((P+O)/16)
    S(P,O) = 0.2 s + 0.04 s * O
    A(P,O) = b(P,O) * S(P,O)

| 全分布 | E[b] MiB | E[S] 秒 | E[b*S] MiB·秒 |
| --- | ---: | ---: | ---: |
| Code | 238.264 | 1.108 | 284.955 |
| Conv | 163.555 | 4.420 | 1172.260 |

该 proxy 下 Conv 平均驻留是 Code 的 3.991 倍，因而可形成较久的背景占用。
但这**不是 GPU profile**：忽略了长输入 prefill、长 context decode 和双模型
compute contention。不能用平均 O 直接断言某条真实请求一定驻留更久。

### 显式条件压力切片

| 模型 / 条件 | 原数据覆盖 | E[P] | E[O] | E[b] MiB | proxy E[S] 秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code，P>=6000 | 10.6199% | 7368.93 | 23.58 | 693.763 | 1.143 |
| Conv，P<2048 且 O>=256 | 2.68234% | 1611.29 | 381.41 | 187.519 | 15.456 |

Conv 条件保留 732,387 条，Code 保留 1,784,530 条。它们保留原成对 O，
不强制加长 Code 输出。此切片强化了“Code 大 KV 到达 / Conv 长驻留可迁移”，
但只能称 **声明的条件压力测试**，不能称完整 Azure 代表性负载。

## 4. CPU 容量机会近似，不是排队或性能模拟

`mixed_workload_envelope.py` 用两条独立 Poisson 流的 stationary M/G/∞ offered
state：人口均值 `lambda*E[S]`，resident marks 按 S 长度偏置，age 在 `[0,S]`
均匀取样。各模型当前请求数在两 rank 紧凑均分，长度仍随机；不模拟真实 count-LB
历史、排队反馈或迁移后的稳态。

每 model/rank 独立 pool，block16 tokens/1.5MiB，composite page96MiB，page0
pinned。按 owner 释放源页、利用目的 pool 空槽，检查 copy-before-release 峰值
和 Code 准入后新增映射。只迁移一条 Conv，保留当前 all-decode gate。

每 case **50,000** 独立状态，GPU 受控 KV 预算 **2112MiB/rank**。safe 指现有
offered state 未超预算；以下 blocked/recoverable 比例都以 safe 为分母。
同时报告 overbudget，不把条件化结果当真实到达准入概率。

| 分布 | Code rate | Conv rate | global够/local堵 % | 一搬 Conv 可恢复 % | offered超预算 % | 可恢复状态自然等待 p50 秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 完整 Azure | .5 | 1.0 | .091 | .060 | .818 | .634 |
| 完整 Azure | .5 | 1.5 | .366 | .198 | 4.846 | .640 |
| 完整 Azure | .5 | 2.0 | .779 | .368 | 15.252 | .833 |
| 完整 Azure | .5 | 2.5 | 1.314 | .672 | 32.108 | .689 |
| 条件切片 | .1 | .5 | .945 | .684 | .304 | 1.362 |
| 条件切片 | .1 | .75 | 8.358 | 6.521 | 3.948 | 1.499 |
| 条件切片 | .1 | 1.0 | 18.048 | 13.827 | 18.218 | 1.570 |

rate 单位 req/s。完整分布有机制但机会稀少；一味加 rate 会显著增加总容量过载。
条件切片 `.1 Code + .75 Conv` 是折中候选：可恢复状态自然等待均值 1.884s，
p90 4.141s。3,132 个可恢复状态中，等待 >1s / >2s / >3s 的分别
1,986 / 1,180 / 641（63.4% / 37.7% / 20.5%）。这些只比较新 Code 的时间，
**没有计入老 Conv 的变慢，不是净收益概率**；若控制还要 1–3s，机会会被吃掉。

仅用 offered-state 的全部状态比例粗算 `.1*.06264*3600≈22.55` 次几何可恢复
机会/小时，30 次期望机会约 80 分钟；这是未反馈近似，不是 GPU 实验样本量保证。
短 trace 很可能没有足够事件，不能运行数分钟后因无迁移就否定机制。

### 保守 ILP 离线见证

每个 5,000-state case 取最先最多 10 个几何可恢复状态，以现有 candidate 规则
每 Conv/rank 最多 2 条、最多 2 次迁移，枚举 Code 目标 rank 后调用 staged ILP。
目的页按保守整页需求、源页按 owner 检查。所有 performance table 设零，move
cost 只是合成值；本项仅证明可行性，不校准 performance 项或真实迁移成本。

| 分布 / Conv rate | ILP 可行 / 检查状态 | 不搬其他模型的 own_model |
| --- | ---: | --- |
| 完整 / 1, 1.5, 2, 2.5 | 1/2, 8/10, 5/10, 6/10 | 所有对应状态均无解 |
| 条件 / .5, .75, 1 | 7/10, 8/10, 9/10 | 所有对应状态均无解 |

几何可恢复不等于保守 ILP 一定能解；candidate 限制、整页保守需求和目的空槽
credit 的差异都保留，不把 8/10 外推为全量成功率。

`.1/.75` 的 sample0 是具体见证：

- 原 Code counts=0:0，Conv=8:9；mapped=1728:1824MiB。
- 预算2112:2112MiB，合计空闲672MiB；ILP 新 Code 保守需求672MiB，任一卡
  分别只有384/288MiB，直接准入和 own_model 都不可行。
- Conv `1-7` 从rank0搬到rank1，释放源3页=288MiB；Conv变7:10。
- Code准入rank0后，ILP final/peak都是2112:2112MiB，复制先于源释放仍安全。

该状态证明用户要求的跨模型空间调整，不证明端到端收益。几何模型有另一个
可利用目的空槽的一搬方案，不能与上述保守 ILP 的 plan 混为同一执行。

## 5. 更长 Code：LongBench 的实际 Qwen token 长度

| 数据 / 任务 | n | P均值 | p50 | p90 | p99 | max | reference O均值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 LCC | 500 | 3221 | 2284 | 5820 | 15041 | 32629 | 12.55 |
| v1 RepoBench-P | 500 | 10927 | 9182 | 19633 | 33136 | 39263 | 13.74 |
| v2 Code Repository Understanding | 50 | 1071304 | 441530 | 3328277 | 4163879 | 4163879 | 1 |
| v2 Long-dialogue History Understanding | 39 | 75611 | 64646 | 120624 | 125447 | 125447 | 1 |

参考答案不是生成实测。v2 是 MCQ，不能把它的 long-dialogue 样本当成长 decode
在线 Conv 流。因此 **用 Azure Conv 提供驻留长输出，用 LongBench Code 提供长
输入** 更合适。v1 的 SAMSum/QMSum 本身输入也长，不能替代短输入 Conv 的语义。

RepoBench-P 留512输出预算，32K覆盖494/500（98.8%），64K覆盖500/500。
v2 Code 留512输出预算的覆盖如下；留64预算时本 release 各档计数恰好相同。

| 总长度上限 | 覆盖 n/50 | 覆盖率 | 留64输出后子集 E[P] |
| --- | ---: | ---: | ---: |
| 32K=32768 | 3 | 6% | 28741 |
| 64K=65536 | 6 | 12% | 41000 |
| 128K=131072 | 15 | 30% | 72552 |
| 256K=262144 | 19 | 38% | 100027 |
| 1M=1000000 | 30 | 60% | 260636 |

v2 Code 有20条 P>1M（40%）；分析保留，不暗截断、不复制 context 或把 word
metadata 当 token。任务仅50条，重复抽样必须报告重复及单任务覆盖，不能伪装大规模
真实 coding trace。建议分阶段：Azure条件切片 → RepoBench-P → v2的32–128K/
128–256K分层，而不是一开始就把所有请求改成1M。

## 6. 1M 是研究分析范围，不是当前布局可直接运行

官方 config 为48层、4 KV heads、head_dim128。BF16、attentionTP1时：

    bytes_per_token = 2(K,V) * 48 * 4 * 128 * 2 = 98304 = 96 KiB

| 单请求总 tokens | 单卡逻辑 KV GiB |
| --- | ---: |
| 32768 | 3 |
| 65536 | 6 |
| 131072 | 12 |
| 262144 | 24 |
| 1000000 | 91.553 |

EP2 只分专家，DP2 只分请求，都不切单请求 attention KV。当前每model/rank 4GiB
虚拟 pool 最多约43690 tokens；2112MiB预算最多22528 tokens，均尚未扣pinned页
和其他请求。**max_model_len=32K 不等于预算下能服务任意32K请求。** 后续更长输入
必须同时改 pool/预算并先测真实权重、graph/activation和复制峰值。

历史四worker实测每tenant/rank权重32,073,216,000 bytes；两tenant/rank共
59.741GiB。原布局单卡双模型权重+1M KV至少 **151.294GiB**，未计Conv KV、图、
activation。历史H200实测143771MiB=140.401GiB，也不足；A800/H100 80GB更不行。
两卡DP的合计空间不能用于分片这一条1M请求。

归档官方 model card 确有1M扩展，但其路径使用 `config_1m.json`、DCA/MInference；
vLLM示例是 `VLLM_USE_V1=0`、`DUAL_CHUNK_FLASH_ATTN`、TP4、maxlen1010000、
`enforce-eager`、maxseq1，建议约240GB总GPU内存。这不是现有nativeV1+kvcached+
DP2/EP2/attentionTP1+sparse graph原型已验证的路径，不因此覆盖checkpoint config
或关闭用户要求的CUDA graph。1M另需解决KV分片/共址权重与backend兼容。

## 7. 核验、代码与后续入口

本轮代码：`../workload_profile.py`、`../mixed_workload_envelope.py`；测试
`../../../tests/test_dp_opt_workload_profile.py`。复现实验步骤和停止门槛在
`../docs/mixed-workload-trace-design.md`，默认serving仍32K/sparse/32step。

- `verification-azure-sources.json`：13份官方源文件SHA/大小、独立重读两份完整
  Azure CSV及各切片/分位数通过。
- `verification-longbench.json`：5253唯一(source,id)、27任务/domain统计通过；
  原文重新分词11条，包含4,163,879-token最大Code和长度门槛附近样本。
- `verification-ilp-envelope.json`：62状态ILP重解、own_model无解、串行复制/释放/
  final/peak独立重算通过。
- `mixed-envelope-verified-50000.json` 和 `verification-envelope-increments.json`：
  补充新增映射及准入后峰值后，7case同seed重跑计数/等待统计与旧50000结果一致。
  不能把full logical KV直接叠加到mapped量：pool已有可用槽，新增映射才消耗余量。
- 首次新增pool-reuse测试fixture错误（另一rank可直接fit），修正fixture后相邻
  6文件回归 **123 passed**；保留失败日志。既有 `asyncio_mode` 配置warning未改。
- 最终全部 `test_dp_opt_*.py` 研究回归 **228 passed / 1 skipped**，AST、矩阵JSON、
  导航链接和 `git diff --check` 通过；日志 `research-tests-full.txt`。

结论：数据支持用户纠正的机制，保守ILP也有跨模型可行见证；需要先控制迁移开销
并确认事件频率，不能声称已证明异构双模型的GPU净收益，更不能把1M分析当作1M
运行成功。本轮不commit/push、不触碰Aegaeon、不启动GPU；研究总目标仍未完成。
