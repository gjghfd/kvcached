# 默认32step下的混合人口准入成本校准

2026-10-23 15:54 Asia/Shanghai预声明。243415只验证单老请求边界阻塞约34ms，
新请求仍等待共识约4秒；不能将其外推为混合trace净收益。

保持两A800、两独立Qwen3-30B-A3B tenant、DP2/EP2/attentionTP1、BF16/128K、
24GiB/model/rank虚拟pool、allocator.98、11328MiB/GPU、sparse默认七档图及32step。
先同allocation复测四pool最长Code实际allocate/rollback、四rank graph和100K推理。
容量不足即失败，不截Code、不关闭图、不增加预算。

然后独立加载双模型，Conv原Azure完整配对P3584/O1024，仍synthetic匹配长度输入。
人口8/12/16，两attention ranks各半，从最后请求也成熟>=32decode开始控制。
新Code为原完整输入trace中最长SWE-smith mark，真正>=100K prompt、输出64。
人口每档比较native_inside（暂停内ILP）与preplan_boundary（decode继续时求解，
默认32step共识后才停止decode），中档反转顺序；共6cases/78请求，另两serial reference。
全部dp_echo完整逐byte校验，四rank EP不跳过，输入输出/预算/capture/pause频率相同。

逐case证明aggregate free足够但两rank actual mapped-slot增量都不够；真实迁移Conv
后分配Code全长度KV，记录copy-before-release峰值、源/目的owner、所有tokens与完整
客户端cadence。预规划必须在pause前及boundary后复核generation、owner/blocks/pages，
改变就拒绝过期plan，不重用旧结果。此pilot旧Conv不在正式Poisson trace中，不能用
同步人口bootstrap称最终吞吐收益；不改原749trace、rate、seed、timeout/SLO。

ILP max_moves8/candidate8/solver limit1s；copy代理每move .01s+logicalGiB/32仅为了
成本pilot选择容量可行移动，不是已经联合校准的净收益门槛；ILP性能项仍单模型
capture-shape surrogate/extrapolation，不冒称双模型最优。pilot不启用wait gate，
因为旧host gate会拒绝，而新mixed控制/prefill代价尚未测得；正式wait参数暂不变。

原始decision记录pause/preplan/solve/migrate/prepare/reserve/end/resume及每rankreceipt，
客户端记录Code TTFT、old Conv token gap/latency sum；必须把100K prefill干扰包括在
收益评估，不能只看26ms复制RPC。独立verifier检查78结果、两serial、唯一owner、
physical mapped预算/完整copy峰值、真实全长度Code reservation和四rank图。

下一依据这些实测冻结公式：incoming规划+边界等待仅计新请求，老人口实际阻塞
按rank人口加权；再加未来copy/commit/resume和非负性能增量，以及prefill联合干扰。
不把preplan snapshot最优当运行中全局最优，不以降低门槛或挑rate/seed制造正结果。
