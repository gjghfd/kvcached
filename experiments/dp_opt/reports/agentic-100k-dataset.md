# Agentic 100K 输入：SWE-smith 轨迹实测

2026-10-21（Asia/Shanghai）。用户要求找 agentic dataset，单次输入约100K tokens。
本轮仅查官方数据、下载一个shard、CPU分词和导出；未启动GPU、未同步remote、
未commit/push、未改当前32K/sparse serving配置。

## 推荐与适配

**优先采用 `SWE-bench/SWE-smith-trajectories` 的 `tool` split。** 它包含coding
agent对软件仓库的多轮操作：读代码、bash、修改文件、运行测试及相应tool输出。
这比把长文档QA称为agentic更贴近本研究的Code模型场景。

这里100K输入是 **累计历史后的一次assistant调用**：保留之前user/assistant/
tool消息，到下一条assistant消息之前截止。不是最初任务描述就有100K，也不是
将整个episode的所有输入输出token相加。不拼接不同任务、不重复文本补长、不在
消息中截断。原始agent是否压缩过历史/实际API是否收到相同100K上下文未知，
因此准确称为 **从公开agent轨迹重建的未压缩history输入**，不声称原API实测。

官方源与固定版本：

- 数据：`https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories`
- revision：`08e109b4a59eaeebf80e4675cd125d42e7ac99a4`
- 实测文件：`data/tool-00000-of-00008.parquet`，248,851,790 bytes。
- SHA256：`ac76e9efe75978f83d4daec98a55604587dd58b0f1509364893b8778cbf5b487`，
  与HF LFS oid一致。完整源文件与card/API/tree快照在工作区
  `evidence/agentic-100k-profile-20261021/`。
- 官方card标注MIT；工具内容中的第三方代码可能有其自身许可，发布原文本时仍须
  尊重来源。没有执行数据中的bash或其它工具，所有内容仅作分词输入。
- README提到5017条训练轨迹，但当前API列出tool/xml/ticks各不同条数；本报告
  不用README数字作全dataset计数，不把不同格式当作互不重复episode。

## Qwen 长度核验

用此前归档的Qwen3-30B-A3B-Instruct-2507 tokenizer及官方chat template，
`add_generation_prompt=True`，无truncation/padding。将text blocks按原顺序用
换行连接，保留assistant tool calls和tool响应；不重复添加`thought`/`action`
元字段。轨迹未提供完整tool schemas，本次`tools=None`；将来添加schema会增加
长度，执行器必须以实际prompt_token_ids重新检查。

完整读取 **一个tool shard的3013条**，不是对全dataset的均匀抽样：

| 指标 | 结果 |
| --- | ---: |
| 最后一轮assistant输入均值 | 27467.72 tokens |
| p50 / p90 / p95 / p99 | 21814 / 53813 / 68903 / 102979 |
| 最后一轮输入最大值 | 193159 tokens |
| 最后一轮输入>=100000 | 32条 |
| 任一完整消息边界有95000–105000输入 | 32条 |
| 任一完整消息边界有100000–105000输入 | **17条** |
| 严格100000–105000导出 | **10条，每episode最多一条** |

选择区间是声明的长输入条件压力子集，不是代表性Code分布。不能因为whole
trajectory长就保证一定存在接近100K的调用边界，大tool响应可能越过整个区间。

严格导出的部分样本（P含chat模板，不含目标assistant回答/未来消息）：

| instance_id | P tokens | prefix消息数 |
| --- | ---: | ---: |
| `getmoto__moto.694ce1f4.pr_6159` | **100084** | 82 |
| `python-hyper__h11.bed0dd4a.func_pm_remove_cond__h6ghpwe2` | **100108** | 80 |
| `pandas-dev__pandas.95280573.pr_57116` | **100129** | 94 |
| `dbader__schedule.82a43db1.lm_rewrite__bjsq7wvi` | 100482 | 158 |
| `encode__starlette.db5063c2.pr_2352` | 100924 | 126 |

可直接读取的本地文件：

`evidence/agentic-100k-profile-20261021/profile-rerun/swesmith-100k-ge-inputs.jsonl`

每行保留source row/traj/instance ID、message_index、normalized messages、
Qwen prompt_tokens和prompt SHA；按100K以上最近边界选择，没有补长到恰好100000。
另有95K–105K最接近100K的10条，`swesmith-100k-inputs.jsonl`；其中部分不足100K，
**用户要求严格>=100K时只用`ge`文件**。

### 有效性限制

- 此shard有1412条`resolved=true`，但严格100K–105K的17条**全部resolved=false**。
  适合系统容量/延迟重放，不用它们论证agent任务成功率或成功轨迹的代表性。
- assistant历史来自原生成模型，不是Qwen自己生成。Qwen真实输出长度/正确性未测；
  `next_assistant_reference_tokens`只是原下一消息的模板token增量，非Qwen测量。
- 单次请求测试可冻结一个真实history prefix；完整agent实验还需重放逐轮增长和
  环境等待，并区分cold prefill与跨轮prefix cache/KV复用。不把独立冷请求Poisson
  直接称为真实agent到达时间trace，当前原型也不支持prefix caching迁移。

## 备选：AgentLongBench

官方代码：`https://github.com/euReKa025/AgentLongBench`；数据
`https://huggingface.co/datasets/ign1s/AgentLongBench`，归档revision
`7c32c2520a777ea51e3a8dcf9d94d524523be12f`。

官方介绍支持32K–4M环境rollout，包含tool-response/env-feedback/final-guess，
可以做可控长上下文agent机制对照。但它是猜实体/侧向思维谜题的**模拟轨迹**，
不是软件工程coding流量。本轮只核验card/项目/论文说明，没有下载其约1GB数据包
或用Qwen实测128K目录，不能把目录标注128K当作已核验100K输入。

论文：`AgentLongBench: A Controllable Long Benchmark For Long-Contexts Agents via
Environment Rollouts`，arXiv2601.20730，初稿2026-01-28，归档页标注v3修订于
2026-01-30。不因名为agentic而把单文档AgenticLU QA与真实工具轨迹混为一类。

## 对当前 KV placement 实验的含义

当前Qwen BF16、48层/4KVheads/head_dim128、attentionTP1：96KiB/token。
输入100000tokens的逻辑KV约 **9.155GiB**；留64输出并按16token blocks取整约
9.161GiB，实际mapped还含page0/96MiB页粒度/空槽复用，必须allocation probe核验。

若希望Code一条请求占**单卡KV总预算一半**，100K对应总预算约 **18.31GiB/卡**；
例如将候选设为20GiB/卡则约46%，不是自动改成当前预算。原2112MiB受控预算、
4GiB/model/rank虚拟pool及max_model_len32768都容不下，三者需同时检查/调整。

后续候选设置：Conv继续Azure短输入/长输出作为常驻背景；Code取该100K真实
agent history，输出先短预算。评估搬Conv腾出连续**容量余量**，不是要求物理
连续KV地址；paged KV不要求整请求物理连续。考虑真实页owner、copy峰值和自然
等待，依然不要求Code长decode。

需要先在H200等真实allocation测双模型权重、graph/activation、100K prefill和
迁移headroom，再预声明KV总预算/rate。DP/EP不切单请求KV，不盲目将单卡标称容量
当可用KV；维持sparse graph，100K在checkpoint原生262144范围内也不等于现有
runtime已验证。此处只是下一步设计，没有新GPU收益结论。

## 复现与核验

代码：`../agentic_profile.py`；可选`pyarrow`/`jinja2`/`tokenizers`只安装在临时
tokenizer venv，不改变项目依赖。初次pyarrow22不支持本机Python3.9，改用21；
初次严格Jinja渲染遇到assistant可选tool_calls缺失，修正为空列表并保留失败日志。

`verification-prefixes.json`从原parquet独立重建20个导出prefix，验证原消息边界、
未来回答排除、官方模板token长度及SHA。源文件SHA/大小、3013唯一traj IDs、
分位数和resolved计数核验通过。最终研究回归234通过/1跳过，既有asyncio warning
不改。数据文件留evidence；只把脚本/协议/报告留kvcached，不commit/push或重启Aegaeon。

严格>=100K的复现参数：

```sh
/tmp/dp-opt-tokenizer-20261021/bin/python \
  kvcached/experiments/dp_opt/agentic_profile.py \
  --source evidence/agentic-100k-profile-20261021/swesmith-tool-00000-of-00008.parquet \
  --tokenizer evidence/mixed-workload-profile-20261021/qwen-tokenizer.json \
  --tokenizer-config evidence/mixed-workload-profile-20261021/qwen-tokenizer-config.json \
  --minimum-tokens 100000 --maximum-tokens 105000 --target-tokens 100000 \
  --output /tmp/agentic-100k-new
```

输出目录须新建，保留所有既有分析结果。
