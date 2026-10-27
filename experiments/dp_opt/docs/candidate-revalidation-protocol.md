# 候选级安全重验：原749请求trace

## 动机与冻结边界

2026-10-23的244639完整四策略净负：ILP vs no_move总latency +1.26%，五次ILP
计划均因全population指纹/generation变化失效，未执行迁移。不能将这个负结果
改写成算法收益，也不能通过改变rate、trace、SLO或关闭默认graph来追正。

保持原1200s/seed20261021/749请求（745Conv+4真实100K Code）、Conv .61/Code .004RPS、
11328MiB/GPU、BF16/128K/24GiB虚拟pool/allocator .98/default sparse/32step，
共享244371查表与244279冻结控制成本。lookup/control/trace的SHA全部不变。

## 安全与算法范围

三迁移策略共同启用 `--revalidate-prepared`；no_move不迁移。先在decode继续期间
枚举两target并求解原bounded ILP，随后在pause前及default32 boundary后，对**原
两个target候选**按原移动顺序确定性重放，不在暂停期间重求ILP。

- 原选定request必须仍在同tenant/rank，精确blocks不变且movable；未选请求完成/
  新准入不直接否决候选。四pool几何、完整page owner与physical budget必须正确。
- 按当前页owner核copy-before-release峰值；所有owner搬走才释放，pinned及任何本次
  destination pool中原页不算释放。核新请求全reservation与迁移后每卡预算。
- 重新计算当前population/horizon、实测Conv lookup及显式N128 fallback、复制成本；
  在目前仍可行的原候选及actual direct fits中选策略对应的低成本动作。
- ILP在pause前重新检查natural-release wait门槛，boundary后再按当前wait和实际已付
  CPU/preplan/rank加权blocking计算是否执行。未知wait拒绝迁移；不拿旧wait继续准入。
- 在冻结事务内检查actual allocator/copy峰值，复制仍exact dp_echo，失败不绕过owner。

该动作是 `revalidated_feasible_not_current_optimum`，**不是变化后全局ILP最优**。
原bounded ILP也只覆盖每model/rank最多8候选/最多8moves。Code prefill联合代价
仍未校准，不宣称查表已包含此反事实。现有同full-reservation baseline不冒称原生
vLLM动态增长调度。

## 证据与独立验证

决策保存原候选selected tenant/rank/精确blocks/几何；两次当前raw observation/
snapshot/cost/wait gate、重验耗时、原plan、当前plan/最终winner及wait tradeoff。
全部重验成本计入实际事务与TTFT/latency，不再漏出preplan。
`verify_candidate_revalidation.py`独立顺序核copy/release/计数/成本，结合raw snapshot
重构及current wait门槛；`verify_trace.py`继续核所有2996输出tokens/serial/owner/
physical budget/四rank sparse graph与default32 receipts。

旧244639离线13个stale决策raw重放：ILP5/5及KV-only7/8至少有一个当前owner容量可行
投影。旧日志未保存原精确blocks，故**不构成原blocks不变证明或GPU收益**；真实
重试才验证selected身份及net latency/SLO。源SHA与投影在
`evidence/agentic-trace-run-20261021/candidate-replay-244639.json`。

## 执行与判读

唯一A800 launcher `agentic-revalidated.sbatch` → measured → capacity → 原四策略trace。
不改变旧launcher默认。每300s双watcher，终态全SHA/独立capacity及trace验证。
报告Code准入/TTFT、全部Conv与baseline Code到达时resident Conv的总latency代价、
全请求latency sum、120sSLO与完整drain。不把CPU投影、queue降低或相对KV-only改善
代替相对no_move净收益；负结果如实保留。
