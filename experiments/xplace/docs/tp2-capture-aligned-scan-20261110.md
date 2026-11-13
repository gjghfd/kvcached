# EP4TP2 固定总请求数、capture 边界扫描

用户于 2026-11-10 指定总 batch 可取 128/256。这里的总数是一台
tenant 的两个 attention-DP rank 之和；TP lane 不再乘一次请求数。
双模型仍共址四卡，另一个 tenant 空载但保留模型常驻。

`capture_aligned_blocks()` 固定生成六个块：三个 repeat，各包含 BS128
和 BS256；第二个 repeat 反转总 batch 顺序，重 rank 随 repeat 轮换。
每块独立 ≤1h allocation，不并行占用两个 allocation。

每个总数从均衡布局开始，在轻 rank 的 0、1、8、16、32、64、128
可达档位及其 ±1 邻点取样；不超过总数的一半。BS128 有 14 个布局，
BS256 有 17 个，三个 repeat 共 93 个测量点，六次 warmup 另外记录。
例如 BS128 的 (64,64)、(96,32)、(112,16)、(120,8)、(127,1)、
(128,0)，BS256 的 (128,128)、(192,64)、(224,32)、(240,16)、
(248,8)、(255,1)、(256,0)。相邻非对齐点不能删除。

固定总数时，两个非空 rank 不一定能同时落在 capture size 上。
当前最大-rank 档位代理在 BS128 的所有非均衡点均为 128，BS256
均为 256。因此不能为了画“更好曲线”预设平滑、单调或每个点跨档。
`capture_bucket` 仍只是配置推导的代理，不是实际 GPU graph replay 证据；
DP padding、dummy coordination 和重放情况必须与真实记录区别报告。

每个总数/repeat 使用同 seed=20261110、同 synthetic prompt 列表。
初版 context=4096/output=1024 的未提交配置完整保留。提交前容量核算
发现 BS256 极端点每 lane 需60GiB/81920块，超过r4记录的最紧张卡
38.75GiB自然预算和65536块虚拟cache；这是容量约束，不是性能筛点。
新版对两个总数的全部93点统一采用 context=2048、output=1024，极端点
理想payload为36GiB/49152块；实际page分配和空闲块仍以真机检查为准。
可执行配置及差异证据保存在
`evidence/ep4-tp2-ablation-20261108/capture-aligned-capacity-20261110-r1/`。
不删除256端点、邻点或旧版配置，也不压缩reserve。此为固定上下文机制校准，不改 OpenThoughts
原输出，不替代真实 trace 的效果门。单点 timeout=150s，scan 上限3000s；
最大片17点加warmup的预算2700s，启动仍受硬1h限制。缺失、容量拒绝、
抢占、超时及所有负值保留；不能仅发布成功子集，也不能筛 seed。

画图报告实际布局、极差、每个重复与覆盖率，以同总数/同repeat均衡点
归一化；不得把不同总数或拓扑的绝对时延混成一条曲线。
本设计已落地为可执行配置生成器和CPU测试，尚未提交GPU扫描。
先完成当前实测代价采集部署，再按固定块顺序执行，不抢占或重复 allocation。
