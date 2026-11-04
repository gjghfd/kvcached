# 数值一致性 note（顺手任务，不阻塞性能实验）

## 0. 用户 guideline

数值 debug 不重要，只在等待性能实验时顺手做；不单独占用 GPU 作业。

## 1. Codex 的发现（2026-10-31 至 11-02，DP=2）

- **现象**：迁移类策略中 Conv 的 greedy 输出与单请求串行 reference 不一致（274385）。
- **零迁移也能复现**：
  - 同 rank、零复制、零暂停，只改变 batch 布局（与哪些请求同批、graph padding 1→8），Conv 就会分歧，而 Code 一致（276017）；
  - 首次分歧出现在 token 53–442。
- **与 logits 精度无关**：把 logits head 改成 FP32 输出后仍然分歧（276677）。
- **来源在上游 hidden**：在共同前缀处做 hidden/head 分解，7/8 个首次分歧约 99.99% 来自上游 hidden 的变化，而不是 head（278529）。
- **未定位到具体 kernel**：first-layer / QKV mirror 的定位作业在多次工具类失败后没有得到 GPU 结论。

## 2. 解释

这是 batched BF16 推理的 batch-invariance 问题：batch 大小、CUDA graph padding 或同批请求变化，会改变 GEMM/attention 的归约顺序，产生 ULP 级差异。greedy 解码在 margin 很小的位置会把这种差异放大成 token 分歧。它不是 KV 迁移引入的：迁移只是改变了 batch 组成，而迁移复制本身经过逐字节 echo 校验，结果相等。Codex 的 sbatch 显式 `unset VLLM_BATCH_INVARIANT`，说明该 vLLM 版本有 batch-invariant 模式，但它会改变性能，不适合性能实验。

## 3. xplace 采用的正确性标准

1. **迁移的 KV 逐字节一致**：source 在 bytewise echo 中比较 destination 回传的数据，所有 DP rank 都参与确认。
2. **输出长度精确**：每个请求的输出必须正好等于 `min_tokens=max_tokens=O`，token 不丢、不重复（`engine.stream` 中检查）。
3. **请求归属与容量**：迁移后源页释放；每个 rank 的映射量始终不超过预算（`check_capacity`）。
4. **greedy 一致性只作观察**：每个请求记录 `token_sha256` 和前 16 个 token，可以跨策略比较分歧比例，但不作为通过与否的条件。

## 4. 如果之后要做（低优先级，只在有空闲 GPU 时）

- 用 `VLLM_BATCH_INVARIANT=1` 跑一次小规模 dataplane：迁移前后同一请求的输出应完全一致。这可以把"迁移正确"和"batch 不变性"彻底分开，但只作为正确性旁证，不用于性能数据。
