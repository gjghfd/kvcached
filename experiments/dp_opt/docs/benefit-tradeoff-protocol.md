# 后续placement tradeoff对照（尚未执行）

2026-10-20 22:44执行前冻结：235320已完成，真实warm migration133–173毫秒、
暂停约.2–.7秒，首次pause4.285秒；该异常保留在首轮结果。后续统一控制预热后执行。
tradeoff当前没有GPU测量，不把本文件的预期当作收益。

## 为什么需要第二个场景

235093原始snapshot的限制候选ILP离线回放中，KV-only与ILP都移动同一B请求，
布局A4:3/B3:4。该场景只能比较排队和跨模型容量释放，不能辨别性能项的贡献。
后续需要至少两个容量可行方案，其中便宜复制方案跨越sparse capture边界。

## 同预算的两个可行方案

继续双模型DP2/EP2/TP1、sparse graph、256输出、96MiB/64blocks整页。
bootstrap A2:8/B4:2，共16请求；外部A到达输入2816/输出256，独立B到达输入64/输出256，
全部18请求完成。bootstrap prompt长度按顺序指定：

- A0：752、64；完整reservation分别63、20blocks。null+第一个63blocks填满首page，
  第二个请求独占下一page，逻辑复制30MiB、可真实释放96MiB。
- A1：8请求均输入64，每请求20blocks；null+160blocks需要3pages。
- B0：80、80、80、512；前三请求各21blocks，null+63blocks填满首page，第四请求
  48blocks独占下一page，复制72MiB、可真实释放96MiB。
- B1：2请求均输入80，各21blocks；null+42blocks为1page。

需实测确认页owner，不依赖request创建顺序自动等于allocator次序。若并发提交重排，
应在同rank按准入顺序创建并验证，但不得将不同策略起始布局/请求长度改变为追求收益。
预期mapped A2:3/B2:1pages，两GPU384/384MiB；controlled预算576/480MiB。
新的A请求为192blocks，rank0现有44mappedfree，需新增3pages288MiB，直接准入超cap；
rank1现有31free，需新增288MiB，也超cap。

两个候选均可在GPU0释放96MiB后为A到达映射288MiB：

1. **move A短请求A0→A1**：复制30MiB，目的A1真实空槽可复用（保守ILP仍按96MiB
   计入copy peak）。最终A2:9/B4:2，A最大rank9，使sparse replay从8升16。
2. **move B长请求B0→B1**：复制72MiB，目的B1实际新增96MiB；最终A3:8/B3:3，
   A保持capture8，B仍capture8。最终GPU0/1 mapped576/480MiB。

KV-only的固定暂停成本相同、bytes更少，应倾向A；含capture-aware performance项的ILP
在足够剩余horizon时应倾向B。这是需要实跑验证的假设，不是校准后的性能事实。
own_model只能选A；no_move保留到达排队，待自然释放。后到达B会改变B请求数，但必须
在四策略使用相同到达规则，完整排队/控制/复制成本全部入账。

## 执行门槛

- 先在235320同GPU实际计时判断host-staged固定cost/solverwarmup，保留第一轮正负结果。
- 新配置snapshot须实测核验：mapped页、可释放owner、KV bytes、copy peak与两个方案可行性。
  边界先至少decode8tokens，必要时追加成熟KV窗口；不事后删掉不利migration时间。
- 3配对repeat、seed42、同allocation；完成SLO仍20秒，不根据结果重挑deadline。
- 输出除完整18请求TTFT/TPOT/throughput/goodput外，加A实际capture8/16证据及B表现；
  性能模型从N128外推依旧显式标注，若明显失配则进一步做同硬件双模型校准。
- 主对照是ILP vs KV-only/own_model的实测代价差异；如无收益如实报告。
  如果KV-only/ILP选择同方案，应诊断horizon/权重/可行性，不能强制改动作制造收益。

入口已实现：`DP_OPT_LIVE_MODE=benefit DP_OPT_BENEFIT_SCENARIO=tradeoff sbatch
experiments/dp_opt/live-placement.sbatch`。固定3repeat、18requests、统一idle/active
控制预热，worker按tenant/rank记录实际capturehistogram；每trial仍检查起始owner/geometry。
源码执行前SHA保护同步。CPU几何参考选择KV-only/own_model移动A，ILP移动B，
是合成snapshot可行性验证，不是新GPU结果。迁移fixed权重保持.475秒以免事后调权重；
实际成本全部计入指标并验证，不以该保守假设冒充校准。
