# 冻结小人口成本后的同trace四策略重放

2026-10-23 17:40 Asia/Shanghai预声明。244371布局校准仍运行，不声称查表或收益完成。
不更换749请求/1200s/seed20261021、Conv .61/Code .004 RPS、B11328MiB/GPU、timeout900/
SLO120。两A800/BF16/128K/24GiB虚拟pool/.98/默认sparse与32step不变。

## 校准输入和冻结

先独立复核244371全部144cases/1664requests、serial greedy tokens、两个稳定窗口
人口/页owner/physical预算、四rankgraph。查表附13原始文件SHA；仅status字符串不能
通过准入。运行前以及终态从全部归档raw重新构建表，与外部冻结SHA和内容逐一比较。
未测人口明确回退N128代理并标注范围，Code prefill干扰不属于查表。

`freeze_measured_control.py`对244279原始6cases/78requests重新独立验证后，仅取
preplan_boundary N8/12/16的实测控制成本。预声明max observed再向上取整0.01秒：

- incoming共识等待：max(pause_call_s)，预计3.06s，不乘old人口。
- preplan：max(preplan_s)，预计.59s；暂停后使用真正已支付成本。
- 非复制transaction固定项：max(control_s-pause_call_s-sum(move_elapsed))，预计.09s。
- 每GiB复制斜率：max(move_elapsed/kv_GiB)，预计.07s/GiB，含完整准备/提交/释放。
- 每请求额外move固定项0：实测完整move已含RPC，不能重复加旧host .475s。

预计值来自成本pilot，不是查表完成或trace净收益结论；最终冻结输出由脚本计算。
只有各三人口一轮数据，max observed不是所有负载上界。保存control-config.json/.env
及SHA，两个文件独立上传；正式启动必须验证SHA、CLI、lookup源SHA、transport一致。

## 公平控制路径

no_move不迁移；own_model/KV-only/ILP共享暂停外CPU预规划、boundary32与exact dp_echo。
仅ILP启用wait-vs-migrate与性能差增量。三迁移策略共用带宽、预算、候选8/最大8moves；
own_model仅本tenant候选、KV-only性能项置零。baseline仍相同full reservation受控
scheduler，不冒称未修改原生vLLM的动态KV增长基线。

检查generation+页owner+blocks，失效计划零迁移保留并计费。CPU预规划不再漏出统计：
保存control_s原事务时间及control_s_including_preplan，独立重算；暂停receipt与old
rank加权阻塞保留。所有TTFT/TPOT/latency仍含实际预规划、共识、prefill、复制与等待。
不要把receipt暂停改善等同净收益；Code早prefill挤占Conv的代价由同trace老请求实测。

## 执行与判读

244639完整净负与全population staleness诊断后，新的同trace候选安全重验实验单独
在 `candidate-revalidation-protocol.md`，旧严格指纹版本与结果保留。

launcher `agentic-measured.sbatch`：先真实100Kcapacity，再加载冻结control env与原
749trace，在同allocation做四策略，固定原政策顺序/一次初测，不筛选seed/改rate。
每300秒双watcher、终态全SHA与2996结果/ownership/graph/预算独立验证。
同时比较Code准入/TTFT、全部及baseline Code到达时已resident Conv代价、总latency/
120秒SLO/完整drain吞吐；负结果保留，不能用恒定offered RPS宣称吞吐增加。
