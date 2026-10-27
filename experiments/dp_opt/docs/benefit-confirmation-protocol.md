# 独立重复与晚到达边界确认

2026-10-21 09:20预声明。235712三次ILP vsKV-only吞吐均正，配对均值+4.48%，
但仅+0.72%到+9.21%，并且ILP到达TTFT更差。不能凭三次结果宣称稳健或全场景收益。

单独一个双GPUallocation顺序执行两个独立case；每个case重新初始化两tenant并统一
预热，不能从早到达数据挑repeat。所有四策略在每个case都执行，固定interval1。

| case | 每策略repeat | shuffle seed | 到达条件 |
| --- | ---: | ---: | --- |
| early | 6 | 20261021 | 所有16bootstrap请求movable且已输出至少8tokens |
| late | 3 | 20261022 | 所有16bootstrap请求movable且已输出至少192tokens |

每trial仍18requests，A2:8/B4:2，输入A0[752,64]、A1八个64、B0[80,80,80,512]、
B1两个80；A到达2816、B到达64，B晚1秒；所有输出256，SLO20秒。预算576/480MiB，
sparse graph1,8,16,32,64,128,256、FULL_DECODE_ONLY、compile0，DP2/EP2/attentionTP1。
不改变ILP单模型外推cost、.475秒fixed迁移项、候选上限2/model/rank与最多2moves。

晚到达不是更大reservation：当前实现预留完整prompt+max_output，复制整reservation，
预计A仍30MiB/B仍72MiB。它检验更成熟decode状态、较短剩余收益horizon与自然释放的
权衡，不证明真实长context迁移的bytes缩放。记录实际输出进度和复制bytes，不能用
无保护的墙钟延迟代替所有请求decode边界。若bootstrap提前结束，判实验失败而非改阈值。

所有case完整origin→最后completion计吞吐，包含排队/规划/pause/resume/复制/prefill。
以case内同repeat比率平均比较，不将早晚绝对吞吐直接作因果差；两case分别报告全repeat、
TTFT/TPOT/latency/SLO及迁移开销。预先定义early为确认，late为适用边界；保留负结果。
离线给出paired gain的均值、95%t区间（少样本描述性，不把宽区间当作证明）。

归档包含root suite配置与每case的config/source/hardware/requests/capture/control/exit。
独立verifier检查每case完整四策略、runtime/worker interval一致、初始输出边界、token/
ownership/budgets/copypeak和原始指标。root只在两个case都完成时记录成功，不静默跳过late。
