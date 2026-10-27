# 512输出与迁移成本摊销

2026-10-21 09:55预声明。236751 early/late的ILP vsKV-only吞吐均值+3.38%/+1.64%，
但95%描述性区间含零且都有负repeat。保持算法参数，增加remaining decode窗口，检验
性能placement收益能否更稳定地覆盖迁移成本，不重复挑选有利seed或删掉负轮。

单双GPUallocation，四策略no_move/own_model/KV-only/ILP，各6repeat，shuffle seed20261023。
两个独立tenant均DP2/EP2/attentionTP1，同checkpoint/greedy/ignore_eos/无APC；
sparse captures1,8,16,32,64,128,256、FULL_DECODE_ONLY、compile0、固定interval1及统一预热。

所有输出**512tokens**，到达边界所有16bootstrap movable且至少8outputs。保持A2:8/B4:2，
根据完整reservation重建exclusive-page容量见证：

- A0输入[496,64]，block数[63,36]；A1八个64，各36blocks。
- B0输入[80,80,80,512]，block数[37,37,37,64]；B1两个80，各37blocks。
- 初始页A2:5/B3:2，aggregate mapped480/672MiB；受控预算**672/768MiB**。
- 新A输入**2560**/输出512，共192blocks/288MiB；B输入64/输出512，晚1秒。
- A短请求迁移54MiB，释放A0一页后A2:9/B4:2；A replay16。
- B长请求迁移96MiB，释放B0一页后A3:8/B3:3；A replay8。
- 两动作各释放96MiB并在目标占一页；保持真实allocation no_move两rank均不能直接准入。

这些输入/预算变化是预声明的allocator见证调整，不是纯output256→512单因素消融。
controller仍单模型N128外推、fixed迁移.475秒、假定1GiB/s、候选2/model/rank、最多2moves，
不按本轮实测重新调cost。真实迁移成本、fullreservation尾部、规划/pause/resume/queue入账。

固定completion SLO仍20秒，不因长输出改deadline；可能所有策略goodput下降，照实报告。
全18requests/trial/432requests完成和串行token一致是门槛。初始页/owner、两GPUcopy峰值、
实际capture、runtimeinterval与decode边界需独立验证。六repeat均记录，主指标为case内
paired ILP vsKV-only吞吐/latency/TTFT/SLO、95%描述性t区间；不与256输出跨job绝对吞吐
作因果比较。若ILP仍只有不稳定小收益，最终结论应为性能模型/迁移控制还需联合校准。
