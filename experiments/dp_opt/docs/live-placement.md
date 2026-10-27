# vLLM + kvcached placement 原型

本页描述可执行实验代码。233056是双模型GPU迁移和ILP准入功能验证，不是收益数据；
后续容量移交与四策略端到端正/负结果见`../reports/allocator-pressure-status.md`和
`../reports/benefit-status.md`。当前接口按集群安装的 vLLM 0.26.0 / MRV1
实现；升级 vLLM 后需要重新验证。

## 代码路径

- `../live_controller.py`：同时暂停两个独立 AsyncLLM 的 DP 组，读取一致 epoch，
  枚举 A 新请求的 rank 0/1，调用 ILP，然后按返回顺序迁移 A/B 请求。
- `../live_scheduler.py`：替换每个 DP rank 的 Scheduler，提供导出、预分配、
  ownership commit/remove、rollback、真实allocator临时预留探测和worker状态安装。
- `../ilp_reference.py`：支持原有 request-private 参考和 `source_pages` 模式；
  后者只有同页全部 owner 已迁移才释放源容量，目标按整页保守计费。
- `../live_benchmark.py`：两个真实 Qwen tenant，各 DP2/EP2/TP1，共用 checkpoint
  文件但不共享权重、KV、请求或 EP group。使用 AsyncLLM 请求，不是 HTTP router。
- `kvcached/integration/vllm/patches.py`：仅 `DP_OPT_LIVE_PLACEMENT=1` 时绑定控制接口，
  普通 serving 不启用该路径。

## 事务与内存约束

使用原生 DP consensus pause/resume，不能单独 resume 一个 EP rank。事务开始
要求所有当前请求处于稳定 decode 且可迁移；仅支持 greedy、文本输入、同步调度、
单 KV group，无 APC、LoRA、结构化输出、speculation、流式输入或 connector。

迁移顺序是：源 Request pickle → 目标完整 KV reservation → 检查两 GPU 上四个
KV pool 的 mapped 峰值 → 源 GPU 到 CPU 文件 → CPU 到目标 GPU → 字节 SHA256
一致 → 安装原 prompt/output/computed-token/block-table 状态 → 目标 commit →
源 ownership 删除及 KV free/unmap → 更新 frontend abort 路由 → 删除源 worker 状态。
保留原 frontend output collector，不能重新生成请求或丢弃已生成 token。

源码删除前的失败回滚目标；一旦调用源删除 RPC，异常结果可能已经释放源 KV，
不再删除目标，保持所有 engine 暂停并结束实验。没有 crash recovery 或线上重试保证。
有 deferred physical release 时，确认 worker batch queue 已空再 drain retired pages。
独占 `/tmp` 目录只用于当前作业，清理不触碰其他任务。

这是 host-staged 文件传输，不是 NVLink 优化；复制整个已预留 logical KV blocks，
包含尚未使用的尾部，`kv_bytes` 是 copy bytes 而非有效历史 token 字节数。
不改变专家 placement，不停止空 attention rank 的 EP 执行，不重新 capture graph。

为了冻结后仍能验证增长，本实验在请求进入 Scheduler 时为
`prompt_tokens + max_tokens` **全序列预分配 KV**。这与普通 vLLM 的动态增长不同，
必须单独标注。planner 使用当前 page owner 和真实 mapped bytes；目标分配按
`ceil(reserved_blocks * block_bytes / page_bytes)` 整页保守计费，不求最优碎片打包，
只支持均匀 page/block geometry。无可行方案时不释放/驱逐 B，也不伪造容量。
若计划也向某个源 pool 入迁请求，新的分配可能复用该 pool 的原有空闲槽并成为
页的新 owner；保守 planner 不再给该 pool 的原始页释放 credit，避免只看初始
owner 错算容量。允许 A/B 各自单向迁移，但可能拒绝实际可行的同 pool 双向方案。

准入预算是同一 GPU 上 A/B KV mapped bytes 的受控上限，不是所有 CUDA 分配的
硬显存限额。迁移前检查保守 headroom，目标分配后检查真实 copy peak，源释放后
再检查；新请求在**同一暂停 epoch 内**提交到选定 rank 并完成 KV reservation，
再次检查预算后 resume，避免规划和新请求分配间的竞态。实验期间不能有未经
controller 协调的其他提交；当前不是透明接管所有 OpenAI HTTP 请求的线上调度器。

## 实验入口

固定 sparse capture `1,8,16,32,64,128,256`，FULL_DECODE_ONLY、compilation mode 0。
先跑 `migration`：A 初始 4:3，B 5:3，input64/output256；A/B 各强制迁移一条 0→1。
校验复制 hash、所有原请求的准确输出长度，以及相同 prompt 的 serial greedy token
对照。数值 batch 差异也可能造成对照失败，不能未经调查归因于 KV 迁移。

之后 `admission`：新增 A input2048/output256，按初始实际占用额外给每 GPU 两个
page-bundle headroom，可选 `ilp / own_model / kv_only / no_move`。233056仅实跑ilp，
其余策略目前只有同snapshot离线重放，不能记为live对照。记录两个候选、
求解状态、迁移顺序、copy peak、ownership 前后位置与请求 cadence。
是否实际需要/允许迁移以冻结 snapshot 和 ILP 为准；不可行不能记为 admission 成功。

```bash
ssh paracloud 'root="$HOME/kvcached/paracloud"; sbatch --output="$root/logs/live-placement-%j.out" --error="$root/logs/live-placement-%j.err" "$HOME/kvcached/kvcached/experiments/dp_opt/live-placement.sbatch"'
```

同一个双卡 allocation 可依次运行多个 mode/policy，别为相同实验重复排队。
sbatch 默认 `allocator_suite`：`packing-no_move`，随后 `pressure-no_move/own_model/ilp`；
每case重新启动两模型。真实probe在pause epoch内分配后释放，并核对原ownership和
mapped恢复；新增请求同时做独立串行token对照。setting和验证门槛见
`../reports/allocator-pressure-status.md`，GPU状态未完成前不宣称capacity handoff。
原233056流程用 `DP_OPT_LIVE_MODE=suite`：先migration，再admission/ILP。
参数由 `DP_OPT_LIVE_MODE/POLICY/POPULATION/INPUT_TOKENS/OUTPUT_TOKENS/ARRIVAL_INPUT_TOKENS/HEADROOM_PAGES/CACHE_BYTES`
及 `DP_OPT_LIVE_SCENARIO` 传入。压力几何固定population8、原64/256、新2816/256，
预算480/288 MiB；packing复现原64/256、新2048/256、预算576/384 MiB。
环境与checkpoint沿用已验证的kvcached部署；不重启Aegaeon。

N128/input1024/output512/A800 的 motivation 系数是单模型 posthoc surrogate。
小规模双模型 smoke 必须显式 `--allow-uncalibrated`；当前脚本用于功能验证，不把
外推系数、假定迁移带宽或 host copy 时长当作已校准的 placement 性能。
summary/config/ready/error、worker topology、snapshot、KV checksum、source hash
和 exit-status 放在 `~/run/kvcached-paracloud/state/live-placement-<job>/`。
