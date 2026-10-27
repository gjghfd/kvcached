# 真实小人口的attention-DP布局成本

2026-10-23 17:02 Asia/Shanghai预声明。244279完整准入成本6case/78请求通过，
预规划消除暂停内.15/.29/.57秒求解，减少old gap；但N128性能外推对8/12/16
人口迁移分别罚14.87/98.49/145.85 request-seconds，不能当这些小人口实测成本。
下一保持原749trace/RPS/seed/SLO/B11328，独立校准真实人口而非调门槛追正。

同两A800/两residentQwen/DP2/EP2/TP1/BF16/128K/.98/24GiB虚拟pool/默认sparse图与
32step。先原100Kcapacity，再重载双模型；Code模型resident但无业务，仍warm执行
两rank图，Conv使用原AzureP3584/O1024完整配对mark/synthetic匹配长度输入。

人口固定8/9/10/11/12/16；每人口完整枚举n0=0..N，包括双hot-rank方向、两次重复。
每个N/repeat内seed20261023+N*100+repeat打乱count顺序，共144cases/1664请求。
所有请求真实完整1024输出，不扩写，不迁移；客户端256..768tokens稳定decode
估cadence。两端snapshot确认相同population/owner/physical mapped预算，四rank图。
独立核验每请求256窗口起点晚于全部请求first-token、768终点早于首个completion；
不把串行提交期间人口增长或提前drain混入布局曲线。未通过则保留raw，不改窗口追正。
每请求full greedy tokens须同serial；原始cadence与完整配置保存，终态全SHA归档。

输出每N每count实测cadence查表，均值仅两重复，不声称统计泛化。与平衡mean对比
构造request-seconds性能项：N*horizon*(T_count-T_balanced)，优化器需要非负cost时
平移每N表的最小值，比较布局增量不受平移影响。仅覆盖该context/硬件/一active
Conv+另一resident idle场景；Code active/prefill共享干扰不包含，不能冒称已联合
校准所有负载。原input lengths分布不同需标明context外推，未测population不能冒称
查表覆盖。正式trace须保留相同四策略与老Conv实际总代价验证，负结果如实记录。
