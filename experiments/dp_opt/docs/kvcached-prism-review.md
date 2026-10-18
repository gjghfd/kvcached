# kvcached / Prism：驻留机制与 attention-DP 实验适用性

初始调研状态（2026-10-17）：当时尚未安装到集群或验证 Qwen DP=2/EP=2；后续安装和并发验证已完成，见下面更新。机制分析仍不能替代实际容量转移证据。

后续更新（2026-10-17）：用户已决定使用kvcached实现idea，两端已clone并维护
用户fork的`main`（`origin=git@github.com:gjghfd/kvcached.git`）；旧`lch/dp_opt`保留，
Aegaeon仍在`lch/dp_opt`。隔离安装225943成功，补DP-rank namespace/device后，双Qwen作业225965
在两A800上COMPLETED/0:0；4路串行和8路并发算术一致、16×256输出完成。
四个KV池从96MiB空闲backing增长到192MiB并回收；85/162采样同时高于空闲值。
详细证据与配置在本研究目录的 `../README.md` 和 `../reports/dual-qwen-results.md`。以下适配风险为
调研时记录；当前只验证本次DP=2/EP=2/TP=1配置，不扩展到任意拓扑，也不声称已验证迁移。

本页属于 kvcached 研究的机制调研，不是 Aegaeon 项目文档。
源码/论文结论以记录中的固定版本为准；后续已完成实验只在 `../reports/` 更新。
本页 `evidence/` 路径相对于外层 ParaCloud；源码路径相对于 kvcached checkout。

## 1. 来源与版本

- OSDI ’26 正式论文：**Prism: Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning**，Shan Yu 等，20th USENIX OSDI，2026-07-13 至 2026-07-15。正式标题以 USENIX PDF 为准，部分早期介绍使用 *Prism: Unleashing GPU Sharing for Cost-Efficient Multi-LLM Serving*。
- 论文主页：`https://www.usenix.org/conference/osdi26/presentation/yu-shan`。
- 正式 PDF：`https://www.usenix.org/system/files/osdi26-yu-shan.pdf`。
- arXiv：`2505.04021`，当前页面关联 `v3`。
- 开源 balloon driver：`https://github.com/ovg-project/kvcached`。本地 checkout 为外层工作区的 `kvcached/`，当前基线 `04ccd1da0653d251622122d0695d72bb288fb8ec`，提交日期 2026-10-15；现在直接在用户fork的`main`开发，原项目保留为`upstream`。
- 独立 Prism 研究实现：`https://github.com/Multi-LLM/prism-research`，核对 HEAD `595ec1f170e75a43897a7a2ad58ac5a9820aa2e8` 的 README 和五个相关源码文件；这是另一套基于 SGLang 的实现，不是上述 kvcached plugin 仓库的 controller。
- 正式 PDF、可搜索文本及来源清单保存在外层 `evidence/kvcached-prism-20261017/`。这些文件不属于 Aegaeon 仓库；不得把其位置解释为远端已同步。

## 2. 核心结论：支持多个活跃模型及其 live KV 同时驻留

**是，但不是所有模型永久驻留。** Prism 将空间共享和时间共享结合起来：多个活跃模型可以共同驻留于同一 GPU / model-parallel GPU group，分别保留 live KV；空闲模型可以被驱逐到 CPU，后续再激活。它不要求每当 A 执行一个 token，就将仍活跃的 B 的 live KV 换出。

依据：论文 §3.2–§4 讨论保留 working set 来避免交错请求的换入换出；§6.2 明确讨论 resident models 并发处理请求时的跨模型显存争用；§6.1 的迁移描述包含 resident KV caches。独立研究实现的 `python/sglang/multi_model/scheduling/gpu/resource_manager.py:79` 从可用显存中扣除所有active模型权重，并在 `:96` 汇总所有worker/active模型的KV占用；`python/sglang/srt/model_executor/worker_pool_model_runner.py:397` 在明确deactivate时释放该模型KV池及GPU模型，而非每次其他模型计算都释放。模型的 GPU 驻留与 CUDA kernel 是否同时执行是两个不同问题，本结论不证明 kernel overlap。

在关闭 idle sleep、offload 和 prefix caching 的受控原型中，目标稳态可以是：

```text
GPU0: A weights + B weights + A live KV(rank0) + B live KV(rank0) + workspace
GPU1: A weights + B weights + A live KV(rank1) + B live KV(rank1) + workspace
```

这正是原始跨模型 KV headroom idea 需要的驻留前提。该图是目标配置，不是已经在 kvcached 上跑通的 Qwen 部署。

## 3. 为什么它能跨模型复用容量

论文 §5.1 指出固定大 KV tensor 的问题：即使逻辑 block 空闲，物理 buffer 仍归原模型持有，不能部分释放给其他模型。§5.2 的 kvcached 将虚拟地址与 GPU 物理页分离：

1. 每个 engine 保留自己的虚拟 KV tensor、block table 和模型语义。
2. 有实际 KV 需求时，才为对应虚拟页建立物理 GPU backing。
3. 页面不再被请求或缓存引用后，可以解除映射并释放物理容量。
4. 其他 engine 随后从同一 GPU 的可用物理容量分配 backing，不需要共享 KV 内容或使用相同 checkpoint。

不同模型的 token blocks 放在独立页面，不能把这里的“共享”解释为模型 A 读取模型 B 的 KV，或同 checkpoint 自动去重。也不要求所有独立进程直接持有同一个 Python allocator 对象；关键是最终返回 GPU 驱动可分配的物理容量。

当前源码对应位置（路径相对于本地 `kvcached/` checkout）：

- `kvcached/kv_cache_manager.py:905`：free request blocks；页面完全空闲时进入物理页释放路径。
- `kvcached/kv_cache_manager.py:985`：整页空闲判定；异步模式可以先 retire，后续等待安全边界再释放。
- `csrc/page_allocator.cpp:264`、`csrc/page_allocator.cpp:290`：保留少量空页以加速后续分配，其余页 unmap。
- `csrc/page_allocator.cpp:447`：`trim()` 解除 reserved pages 的映射。
- `csrc/page_allocator.cpp:789`：实际 unmap，必要时通过 TP worker IPC 同步。
- `csrc/page.cpp:44`：GPU physical allocation handle 的 release。

**因此它具备我们缺少的弹性物理容量机制，但不会自动完成 DP request migration。**

## 4. 回收边界：逻辑 block 数不等于可转移物理字节

### 4.1 运行中的 KV 不会因降低 quota 而被任意撤销

`examples/02_memory_control/README.md:33` 明确说明：lower limit 可以驱逐 prefix cache，但运行中请求使用的页不被撤销；只有请求释放相应引用后才能回收。`set_instance_memory_limit` 是 process-local API，预算如何选择、fairness、admission 和跨实例仲裁由外部控制层负责。缩容可能处于 `deferred`，不能将收到预算变更当成容量已经可用。

故降低 B 的 limit 本身不等价于将 B 的 live KV 迁出。原型必须实现请求状态/KV ownership 的安全迁移，再释放源端 blocks，并确认目标 B 继续生成。

### 4.2 物理回收粒度和缓存策略会影响收益

- 一个物理 page 内还有其他 live blocks 或保留 prefix 时，仅迁走部分请求不能回收整页。
- reserved / preallocated 空页可能仍持有物理 backing；`trim()` 或明确的预留页设置才使容量真正返回。不要把所有空页都算成可跨模型使用。
- 异步调度下页 retire 之后还要等待执行队列的安全边界，不能在旧 kernel 可能访问它时解除映射。
- 各模型仍有原始虚拟 reservation 上限和当前预算；A 能使用 GPU 空闲容量，不代表能越过自己的上限。
- 统计需要同时报告 logical live bytes、mapped physical bytes、reserved backing、prefix cache、allocation limits 及 driver free memory，而不是只看 `nvidia-smi` 或 logical free blocks。

额外注意 Qwen 的回收粒度：`benchmarks/bench_idle_footprint/README.md:38` 给出计费关系 `num_layers * page_size * num_kv_buffers`。本地 vLLM 0.26 adapter 对 packed K+V 的 unified pool 使用 `num_kv_buffers=1`（`kvcached/integration/vllm/interfaces.py:631`）；如果 Qwen 的48层同质 GQA 确实走该路径、默认每层2MiB页，则一个全层 page-id 对应96MiB backing，而非2MiB。旧 split K/V 路径的 `num_kv_buffers=2` 则对应192MiB。这是条件推导，不是已实测配置；必须从运行时 geometry 日志确认，默认预留页预算同样按此倍率计算。

## 5. 与我们的 DP=2 / EP=2 / attention TP=1 的关系

### 5.1 论文不是现成的 attention-DP 优化实现

Prism §4 将 model-parallel GPU group 描述为严格调度边界；§6.1 做负载感知的模型 placement/migration，§6.2 做 GPU request arbitration，Appendix A.2.2 讨论 TP placement。它已经提出通过 KV Pressure Ratio 最大化显存 headroom，不能把“共享 KV”或“headroom-aware placement”本身当作我们的新贡献。

待验证的差异应是：**在固定 MoE expert placement / EP group 下，迁移其他租户请求的 attention/KV home，主动允许一定 attention-DP 不均衡来给长请求整理 rank-local headroom，同时量化 attention 与 EP 执行代价。** 调研只建立了这一具体问题与论文方案的机制区别，尚未完成全领域 novelty 排查。

### 5.2 当前开源 adapter 不能直接假设 DP/EP 已适配

当前源码的 vLLM worker 初始化读取 TP rank、TP world size 和 PP rank（`kvcached/integration/vllm/patches.py:2178`），worker socket 路径使用 IPC namespace + PP rank + TP rank（`kvcached/utils.py:214`、`kvcached/utils.py:228`）。没有从这些路径看到 attention-DP rank 专门参与页管理 namespace。

这不是“确定不支持 DP/EP”的结论，但在 TP=1、PP=1、DP=2 下，两 DP worker 的 TP/PP rank 都是0。如果继承相同 IPC namespace，可能出现 socket/页管理冲突；如果过度隔离，又需要确认 EngineCore 的 quota、mapping 和回收指向对应 worker。每模型×DP rank 的 namespace、device 和 controller-owner 映射必须作为适配验证项。

**不能为了让 EP 两卡参与，就把 kvcached 的 TP `world_size` 改成2。** attention-DP 的 KV 应是 rank-local；TP 广播页管理的语义与 EP collective 语义不同。B 的请求迁走后，源 rank 的 expert shard 及必要 EP participation 仍然必须保留。

当前 README 声明 vLLM ≥0.17、测试至0.30，PyTorch ≥2.10；集群已用的 vLLM0.26/Torch2.11落在名义范围。源码仍对0.28之前的 Model Runner V2主动拒绝，需确认使用 V1。版本范围不是 Qwen DP/EP 正确性证明，不就地升级或修改已验证环境；安装/编译/推理均放到 Slurm allocation 的隔离实验环境。

### 5.3 开源 driver 与论文完整 Prism 不可混为一谈

README/论文将 kvcached 定义为 balloon driver；公开 `controller/README.md:1` 提供多模型 router、独立 backend 启动和 idle sleep/wake 示例。这个示例不能自动当作论文 KVPR placement、slack-aware arbitration 或 live DP migration 的完整复现。后续做 baseline 时明确哪些组件来自上游，哪些由本实验实现。

此外已找到独立 `Multi-LLM/prism-research` 仓库：README列出global placement、local admission、elastic memory及worker pool；代码树中实际存在 `python/sglang/multi_model/scheduling/policy/simple_global.py`、`gpu/gpu_scheduler.py` 和 `gpu/resource_manager.py`。因此不能说Prism完整控制面未公开；应把此研究实现作为后续paper baseline核对入口。当前只保存固定提交的README及五个相关源码快照到外层 `evidence/kvcached-prism-20261017/prism-*`，尚未复现其环境、算法或benchmark，也不假设该SGLang fork可以直接套用新版本plugin。部分示例使用 `--load-format dummy`，真实正确性/性能实验不得沿用dummy权重。

## 6. 下一步验证顺序

1. **单GPU弹性容量验证**：在隔离 allocation 中使用两个小模型或两个独立 cache owner，禁用 sleep/APC，确认 A、B 同时保留 live KV；保持 B 的部分请求继续运行，释放另一部分后观察整页 unmap，并让 A 使用新增物理容量。先用请求自然结束隔离 allocator 链路，不宣称已实现迁移。
2. **Qwen DP/EP兼容性**：复用已下载 checkpoint，两服务各 DP=2/EP=2/TP=1，确认四 worker namespace/device、页容量、专家覆盖和并发生成正确；验证显式 home-rank routing 与16:0时的 dummy/collective participation。
3. **真实容量转移链路**：在安全 token 边界将 B 的一个 live request 从rank0迁至rank1；目标生成正确后释放源KV，等待安全回收、必要时trim；证明 A 的rank0随后分配新增 backing，而 B两侧EP组仍正常。
4. **联合实验**：保留现有 attention-DP motivation，将 fixed-private vLLM、kvcached弹性池且无migration、KV-only策略、compute+KV策略区分开。迁移前后的 live请求状态、物理池预算、sleep/APC设置及服务SLO保持一致。

成功标准不只是 B logical blocks减少，而是：`B(rank0) mapped backing下降 → GPU0可分配容量上升 → A(rank0) mapped backing上升并继续完成长请求`。还要计入页粒度碎片、目标端瞬时双份KV、迁移成本和不均衡的step-time代价。

## 7. 当前工作状态

已完成论文阅读、固定版本源码核对、本地 clone 与设计记录。没有运行 kvcached benchmark，没有远端同步/安装，没有新 Slurm 作业；原 attention-DP 扫描仍未实施。

该段为最初调研状态；后续baseline已由225965完成（见开头更新），attention-DP扫描、
live migration和跨模型capacity handoff仍未实施。
