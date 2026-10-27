# Sparse CUDA Graph 下的跨模型 KV 准入调度

2026-10-20。算法设计与 CPU 参考实现；opt-in数据面已通过小规模GPU迁移/准入验证，
不是已上线调度器或已验证的跨模型性能收益，结果见 `../reports/live-placement-status.md`。
完整页级约束见 [ilp-scheduler-design.md](ilp-scheduler-design.md)，实测见
[curve-results.md](../reports/curve-results.md)。

## 1. 决策问题

两个 Qwen 模型 A/B 同时驻留在两张 GPU 上，各自 DP=2、EP=2、attention-TP=1。
固定专家布局、模型进程和 EP 组，只改变请求的 attention/KV-home rank。
新 A 请求 j 到来时，枚举其可用 DP rank g；每个 g 解一个 ILP，允许移动 A/B
的既有请求，满足 KV 空间与复制峰值约束，选择预测总代价最小的方案。
不通过关闭空 attention rank 或改变 CUDA Graph 配置制造收益。

固定现有 sparse capture：`1,8,16,32,64,128,256`。Dense 仅保留历史消融。

## 2. Motivation 给出的成本模型

定义 n_mr 为模型 m 在 rank r 的活跃 decode 请求数，N_m=n_m0+n_m1：

    I_m = |n_m0-n_m1| / N_m
    p_m = min {c in sparse_captures : c >= max(n_m0,n_m1)}
    p0_m = capture(ceil(N_m/2))

Job 231467 的 N=128、input=1024/output=512、A800 单模型数据：

| 布局 | replay p | cadence 比例 R | 吞吐损失 1-1/R |
| --- | ---: | ---: | ---: |
| 64:64 | 64 | 1 | 基准 |
| 80:48 | 128 | 1.268443 | 21.16% |
| 96:32 | 128 | 1.308273 | 23.56% |
| 112:16 | 128 | 1.450341 | 31.05% |
| 128:0 | 128 | 1.475352 | 32.22% |

非平衡点平均两种 hot-rank 方向。单一 `R=1+alpha*I` 不通过预声明残差阈值；
不能声称性能损失与不均衡度全局线性。当前 sparse 的事后描述模型是：

    R_hat = 1 + 0.305118*I + 0.184904*(p/64-1)
    T_hat_ms = 24.698862 + 7.536068*I + 4.566912*(p/64-1)
    loss_hat = 1 - 1/R_hat

同一 capture 档位内近似线性，跨档位有跳变；不是 attention/通信时间的因果分解。
其已测布局均值最大 R 残差约 0.038，不能当成其他布局、P99 或双模型误差上界。
65:63、81:47 等未实测点必须标为预测，不能用该模型证明平衡点旁的跳变幅度。

校准范围固定 N、context、GPU、kernel 与 capture。A 从 128 个既有请求增加到 129 个，
需要 N=129 校准，不能套 N=128。当前例子使用 A 原来 127 个、新增后 128 个，B 保持
128 个。`attach_sparse_costs` 拒绝人口数不匹配或 dense 配置；真实 context、硬件与
backend 仍需调用方核对。新请求 prefill 的耗时、KV 与干扰需要单独建模。

## 3. 目标函数：性能与移动使用相同单位

预测 horizon H_m 内每个请求再执行 H_m 个 decode step，权重 w_m。用加权
request-seconds 衡量总延迟，扣除各候选间相同的固定 N_m 平衡常数：

    C_m(v) = w_m*H_m*N_m*[T_hat_m(v,N_m-v)-T0_m]
    J_g = sum_m C_m(n_m0) + sum_i P_i*z_i + Q_g

P_i 包括一次移动使所有受影响请求产生的暂停、复制干扰和准入等待，不能只按 KV
bytes 排序或漏算 B 的暂停。Q_g 为目标 rank 特有的准入/prefill 代价；依赖方案的
等待应放入 P_i 或阶段成本，不能伪装成固定 Q_g。当前参考适用串行冻结、移动代价可加。

代入 capture-aware 模型，性能部分为：

    sum_m w_m*H_m*T0_m*a_m*|n_m0-n_m1|
         + w_m*H_m*N_m*T0_m*b_m*(p_m/p0_m-1)

第一项是档位内不均衡，第二项惩罚更大 replay 档位；不要只优化第一项。
H 决定迁移是否能偿还成本。长度预测不可靠时需要 cooldown、移动预算和重算。
不同模型可以用不同 H/w，但每次决策必须记录它们，不能混用一步耗时与总暂停。

两模型共享计算/带宽，单模型 `C_A+C_B` 目前仅是可检验的可分离假设。
正式双模型校准需采集同向/反向 hot-rank、context、N 与图档位，使用联合成本
`C_AB(v,w)`，不能把单模型系数当成已验证的共址系数。

## 4. ILP 编码

请求 i 的源 rank 为 s_i，允许目的集合 E_i。固定/in-flight/cooldown 请求只允许原位。
x_ir 是最终放置，z_i 表示移动，a_i 为活跃 decode 标志；新请求 g 在当前子问题中
为常数，人口数是在其完成 prefill 后预测：

    sum_r x_ir=1;       x_ir=0 for r not in E_i
    z_i=1-x_i,s_i
    n_mr=sum_{i:model(i)=m} a_i*x_ir + 1[m=A and r=g]

通用模型允许 a_i=0 的 memory-only 请求；当前 CPU 参考仅覆盖 a_i=1 的可移动请求，
其他内存作为固定占用，并不支持移动 idle 请求的负载/内存分离接口。

为每个最终 rank-0 请求数 v 建立二进制 h_mv：

    sum_v h_mv=1;       n_m0=sum_v v*h_mv
    minimize sum_m,v C_m(v)*h_mv + sum_i P_i*z_i + Q_g

C_m(v) 在求解前计算为常数，故原曲线非线性仍然是 ILP。有联合校准时改为 h_vw：

    sum_v,w h_vw=1
    n_A0=sum_v,w v*h_vw;       n_B0=sum_v,w w*h_vw
    minimize sum_v,w C_AB(v,w)*h_vw + sum_i P_i*z_i + Q_g

参考支持 `performance_cost_tables_s` 或 `joint_performance_costs_s`，不能同时提供。
查表模式不需要旧的 `imbalance_cost_s` 仿射系数。
联合表轴为 A/B 的最终 rank-0 人口数，不是 GPU 总请求数；同样 I 的同向/反向负载
可有不同成本。建模时裁到移动集合可触达的区间，不删除任何可行布局。
成本要求非负；若实测参考差值为负，可给所有联合表元素加同一常数，不改变最优方案。
大规模需要稀疏矩阵和候选集压缩；当前 CPU 小例子不是在线求解吞吐测试。

## 5. KV 容量与可执行迁移

C_k 按同一 GPU 汇总 A/B 的所有池，扣除权重、图/workspace 峰值与安全余量。
j 预留 prompt KV、增长预算和 prefill workspace；既有请求增长也需预留。
驻留但非 decode 的请求贡献内存，不应贡献活跃 decode 人口数。

实际释放以物理页 bundle 为单位。源页 p 的 owner 集合 O_p、不可释放标志 b_p：

    y_p <= 1-b_p;       y_p <= z_i for every i in O_p
    y_p >= sum_{i in O_p} z_i - |O_p| + 1-b_p

只有所有 owner 离开、无固定引用且生命周期允许 unmap 才能释放。
移动一个请求不必然释放其逻辑 KV bytes。目的池检查可用 block 和真实页边界；
释放源页和复用其 free block 不能重复计入，跨进程池还需实际容量移交确认。

最多 L 个串行复制阶段 u_irt，每个请求最多移动一次、每阶段最多一次移动。
Q_mr,t 为截至 t 新映射的目的页数，y_p,t 为截至 t 已安全释放的源页：

    M0_k - sum_{p on k} bytes_p*y_p,t-1
          + sum_{pool(m,r) on k} bundle_bytes_mr*Q_mr,t <= C_k

当前复制只用前序已释放的源容量，不能先扣当前源 KV 再申请目的 KV。终点再准入 j。
若两边均缺复制 headroom，即使最终布局有空间也可能死锁，必须判不可行；offload
或二次移动需另行显式建模。页级打包/预留的完整公式见通用设计第 5/6 节。

CPU planner 已补 source-page owner 释放约束和保守目标整页分配；实验数据面已接入
token/block-table/KV copy 及源 unmap，233056原请求GPU token对照通过。不是任意共享页
精确打包，也没有跨租户容量移交或性能收益的实测结果。见 `live-placement.md`。

## 6. 控制流程与最优性

    获取一致 epoch：请求位置、页 owner、增长/安全预留、成本 profile
    枚举 j 的 eligible rank g
        解 A/B 联合放置 + 串行迁移 + 容量 ILP
        保存可行 incumbent、目标、lower bound、gap、顺序
    比较所有可行方案（包括零移动方案）
    校验 epoch 和实际 block/page 预留；过期则重算
    在合法 decode 边界 copy -> 校验 -> 切换所有权 -> 等待源释放确认
    确认容量移交后准入 j；记录 TPOT/TTFT/goodput、pause、bytes 和模型误差

可加入 TPOT 上包络、B 公平约束、最大移动数/暂停、cooldown；均值曲线不保证 P99 SLO。
参考目前支持移动上限和固定目标准入成本，不包含完整 SLO/公平控制器。
不强制移动；全部不可行则排队等待自然释放。超时有 incumbent 不等于最优，
超时无解不证明不可行。所有 g 均证明 optimal/infeasible 才是该有限模型中的最优方案。
在线预算应覆盖全部子问题与事务验证；当前 CLI 的 time-limit 是每个 g 的预算。

## 7. 跨模型取舍合成例子

输入 [sparse-admission-example.json](sparse-admission-example.json)：A=64:63、B=80:48；
每 GPU 的 KV 安全容量 24 GiB、占用 23.90625 GiB。j 需 192 MiB；候选请求各有独占
96 MiB，每次移动 0.01 request-seconds。H=32、w=1，使用上述单模型事后模型。
内存、移动代价为构造值，不是 Qwen 物理页 owner 或实测 pause。

| 方案 | 移动与新请求目标 | 最终 A / B | 预测 J（request-seconds） |
| --- | --- | --- | ---: |
| 不迁移 | 任意目标 | 无法准入 | 不可行 |
| 仅迁移 A | A1: rank1 到 rank0；j 到 rank1 | 65:63 / 80:48 | 45.621388 |
| 跨模型，g=0 | B0: rank0 到 rank1；j 到 rank0 | 65:63 / 79:49 | 45.139080 |
| 跨模型，g=1 | B1: rank1 到 rank0；j 到 rank1 | 64:64 / 81:47 | 26.915315 |

允许 B 略变不均衡换取 A 保持 graph=64；不是始终向更空的 attention rank 移动。
两卡最终与复制峰值均不超过 24 GiB。相邻人口数未测，单模型成本可加未验证；
例子只证明算法能表达取舍与准入，不是实际性能收益。

```bash
python3 experiments/dp_opt/calibrated_cost.py <sparse-fit.json> --horizon-steps 32 --snapshot experiments/dp_opt/docs/sparse-admission-example.json --output <snapshot.json>
python3 experiments/dp_opt/ilp_reference.py <snapshot.json> --output <plan.json>
```

CPU 结果在外层 `evidence/sparse-ilp-design-20261019/`。下一步依次补双模型 sparse
校准、页级迁移数据面、准入实验。主对照保持相同迁移能力：无迁移、仅本模型优化、
跨模型 KV-only greedy、capture-unaware ILP、完整 sparse-aware ILP。
