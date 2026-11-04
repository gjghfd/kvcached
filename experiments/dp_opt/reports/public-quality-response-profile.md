# 公开原始长响应：质量 cohort 的独立 profile

2026-10-31，CPU 数据与源重建结果，**不是 GPU 收益或最好 setting**。
用户要求 Conv 使用公开数据原始 response，而非人工延长 AzureConv。
当前 Stratos 作业输入保持不变；新 C1 是独立候选，不替换历史实验。

## 选择依据

公开源为 `open-thoughts/OpenThoughts3-1.2M` 的两个固定 revision/LFS shard，
revision `61bcf9d4eb38b30295efc2021227a63cc5bb34c8`。完整本地20,000条原响应中，
预先只按恰好一对有序 `<think>`/`</think>` 且 close 后有非空原 final-text 的
结构质量条件选择6,515条（32.575%）。没有按 token 长度或 GPU 结果选择；原文
没有裁剪、拼接、padding 或延长。所有排除行及原因保留，未排除图配置或 KV 不利样本。

在完整 cohort 中取最低原始 selection SHA 的256条作为 C1 bank；逐条从原 parquet
独立重建消息、完整原响应、Qwen prompt IDs/P/O，256条全量重新分词通过。
两源20,000行/6,515 eligible 源与质量条件复核通过；十条 Code marks 字节一致，
Code prompt IDs/全部字段一致，六个固定 native Code draw 与 C0 一致。

|对象|数量|mean P|mean 原 O|O P50|O P95|O P99|O max|E[bO]/E[O] MiB|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|完整两 shard 质量 cohort|6515|86.411|13289.941|13877|16720|16750|16775|1362.803|
|原始 hash bank256|256|85.5273|13210.9258|13689|16708|16751|16760|1360.2159|
|当前 Stratos 原始 bank256|256|462.590|4630.6875|见 Stratos profile|13372|20171|24646|880.381|
|OpenR1 分层 page-hash 候选 sample|1600|103.809|5960.8031|4913|14113|16793|18813|862.260|

这些不同 source/sample 的数字不是同一总体或 GPU 实测比较。长 O 的均值本身不等于
驻留加权 KV 大，也不证明回迁能偿还 pause/copy 成本。OpenR1 比 Stratos 的 mean O
长，但加权 KV 略小，因此 pressure1.2 仍约六个 resident，不明显更适合当前 N2..4。
质量 cohort 相对 **Stratos/OpenR1** 更接近低人口长驻留；相对 **未过滤的两
OpenThoughts3 shards**，其 pressure1.2 人口反而增加14.27%，不能把这两种比较混淆。

## 固定 native draws，无重抽

|seed|N2 原 O|N3 新增 O|N4 新增 O|
|---|---|---:|---:|
|2026103101|16557 / 16314|15134|12741|
|2026103102|12980 / 12085|14988|13616|

这是现有 `make_native` 的固定 seed 抽样；Code 仍在90s到达，O64，完整100K输入。
没有为方便回迁另抽更长的 Conv。以上是冻结输入长度，尚不是实际完成时间或
GPU 回迁次数。较长原 O 提供了比早期 AzureConv 短原响应更合理的验证机会，但仍
需要测自然恢复、回迁时剩余 O、控制成本及所有租户 tail/avg 指标。

## 速率只作理论先验

沿用 B10240MiB/rank、约9600MiB Code reservation，以实际 bank256 的
`E[bO]/E[O]=1360.2159MiB` 估算：

|目标 local 倍率|假设 resident Conv|假设20ms TPOT 下 Conv RPS|
|---|---:|---:|
|1.1|2.4467|.0092600|
|1.2|3.9523|.0149585|

公式是 `N=2*(ratio*B-b_code)/(E[bO]/E[O])`，
`rate=2*(ratio*B-b_code)/(TPOT*E[bO])`。20ms 是假设，非 C1 GPU 数据。
正式 rate 必须由 C1 的真实 service/physical pressure/open-loop queue 校准冻结，
不能直接套用上表或 C0 的 service。后续协议见
[公开质量 cohort 回迁实验](../docs/public-quality-return-study.md)。

## 证据与失败记录

`evidence/public-openthoughts3-quality-cohort-20261031/` 保存完整源/排除 map、
预声明 identity、原 profile 和两 LFS SHA；`bank-profile/` 保存原始 hash256 exports；
`build/` 保存重建、独立 audit、256条 retokenization、native draw、tests 与 manifests。
冻结 C1 inputs 在 `evidence/public-openthoughts3-quality-workload-20261031/frozen/inputs/`。
Input manifest SHA：`5aa7646d745b681976054a316e403fd1ea33fee106f47a05862f0f546b3c0b88`。
Quality identity SHA：`260ec3193e27852bcb18ad67320011fd1b1a9c8f8df78de6b08804545c5da885`。

首 post-build blanket baseline audit 退出1，原 traceback/stdin 缺口保留，没有改写成
全部通过。随后原始 source/bank 专项独立 audit 通过。差异被定位为 **future-only**
`public_poisson_followup.py` 的主线程接续修改；959/960其余 baseline 项及 C0四输入
未变，不回滚该修改或称960全部相等。最终完整未来回归与 SHA 另存于 recovery root，
不把旧48单元测试当完整 future/runtime 通过。

这里的 Conv 是公开长 reasoning tenant，**不是普通真人 conversation**。恰有 think
闭合及 final-text 不等于语义正确、natural EOS，源 generation-cap 疑虑仍保留；两个
shards 不代表全1.2M。Serving 用原始 O 重放不是 Qwen 自然停止长度。当前没有 C1 GPU
收益或最佳 setting；保留当前 C0 和所有失败/负结果，最终以固定 held-out 科学结论为准。
