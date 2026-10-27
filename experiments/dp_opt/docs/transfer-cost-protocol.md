# A800 KV transport cost pilot

2026-10-23 12:13 Asia/Shanghai，先测复制，不重选trace/RPS追正结果。

241580迁移准入成立但净收益负；241907 wait69/69拒绝，0moves/0solve。下一需
实测减少host-file路径的成本是否有机会，不以零迁移下latency -2.33%称placement
收益。保持两独立Qwen tenant/DP2/EP2/attentionTP1、BF16/128K、24GiB虚拟pool、
allocator.98、B11328MiB、默认sparse/FULL_DECODE_ONLY/pause32。

## 预声明配对

先同allocation做原100K capacity pilot，完整输入/四pool真实allocate/rollback和
四rank图；不截Code。capacity通过后单独加载同双模型做transport pilot。它只测
迁移成熟Conv，不当真实Poisson容量收益实验；正式749trace/seed/RPS/SLO不变。

从冻结Azure完整paired切片选距总长度2560/4608/8192且O靠近1024的mark，仍用
原P/O/source_internal_row，不扩写输出。三mark实际P/O为2048/512、3584/1024、
7167/1024，逻辑KV240/432/768MiB。输入为匹配长度synthetic text，与原trace一致。
每mark×3重复×2transport×2模型=36请求，另每模型/mark完整serial reference。
transport顺序随repeat交替，source随repeat/model交替覆盖两方向。

host_file保留原完整CPU文件复制/SHA；dp_allgatherv复用已初始化模型DP device
communicator，两rank RPC并发，同步相同layout/count/source metadata，然后源
有Nblocks/目的0blocks的变长collective。真实KV块按32MiB上限分段gather→collective
→目的scatter，源及目的都CPU读回增量SHA，完整长度/字节SHA一致才commit/remove。
因此不是无校验纯NVLink带宽：保留CPU integrity读回，NV8拓扑不保证端到端收益。
没有新通信组、没有与active EP collective并发、没有关闭idle rank的EP执行。

先达到>=32decode/movable再暂停默认32step；每次保留before/after owner与mapped
budget、迁移copy峰值、request token IDs、四rank real graph。等待/复制/pause/resume
计时完整记录；transport仅单request成本，不外推混合population暂停成本。

## 安全和证据

复制分段限32MiB，拒绝超块/layout mismatch。peer RPC失败/timeout时不能确认CUDA
完成，保留两端mapping、保持暂停并失败退出，不rollback/unmap正在通信的页。正常
两个RPC完成后的SHA不一致可rollback目的。源remove不确定仍保留已提交目的。
默认仍host_file；新peer仅本pilot明确切换，不自动启用正式ILP。

`transfer-trials.json`保存RPC与worker计时：metadata、gather、collective、scatter、
CPUreadback/SHA，host另save/load；controller另记录prepare/commit/remove/容量RPC。
`verify_transfer_pilot.py`独立重建36cases、串行tokens/唯一owner/预算/图/18配对成本。
终态archive完整SHA后自动核验capacity和transfer，不称收益目标完成。

## 下一门槛

243415第三阶段已完成，单老request阻塞mean.648602→.033845s，但incoming等待
mean.648602→4.065191s；不能只以复制/阻塞加速称准入更快。下一混合人口真实100K
准入成本校准见`admission-cost-protocol.md`；749trace/rate/预算/SLO保持，不先降低wait
gate追正结果。完整单请求结果/异常/监控连续性只在唯一agentic结果报告。

### 第二阶段：完整 GPU bytewise echo

243165已完整通过36cases，DP-device+CPU SHA迁移均值2.451708→.531639s，GPU
collective约.004–.0105s，主要剩余成本是完整CPU读回/SHA。不把该复制加速称
trace净收益；保留默认pause5.995s/8.370s异常。下一唯一A800allocation先同100K
capacity，再用完全相同三marks/3重复/两模型/交替方向和transport顺序，共36cases
比较`dp_allgatherv`（完整CPU内容SHA）与`dp_echo`。

echo只接受uint8实际KV backing view，双方geometry及integrity metadata一致才复制。
每<=32MiB chunk先源→目的device collective并scatter；重新gather目的实际KV，
反向collective回源，与源原chunk逐byte `torch.eq(...).all()`；全部views/blocks
完成后双rank确认flag。不是采样、概率hash、CPU内容SHA或纯带宽测试。任一byte
不一致必须失败，通信状态不确定保留两端mapping和暂停，不commit/remove。

返回`device_echo_full_bytewise`与`bitwise_verified=true`。`sha256`仅验证receipt，
`sha256_scope=verified_integrity_receipt_not_KV_content`，禁止标为KV内容SHA。
外部归档SHA仍覆盖全部原始文件；完整serial token IDs、owner、physical mapped
预算/copy峰值与四rank graph独立验证不变。verifier从config独立重建transport
组合，不将旧host/peer cost字段覆盖为新的测量结果。默认正式transport仍host_file。

本地442研究tests/5skip通过；compute启动须跑真实tensor双线程collective测试，
包括单byte腐败必须拒绝，再执行真实GPU36cases。下一wait门槛必须计入默认pause/
solve固定成本及异常，不凭降低copy斜率认定同749trace有净迁移机会。

### 第三阶段：默认 32step 共识边界暂停

243300完整echo正确性/成本通过，peer RPC .516136→.025852s，但完整migration
.939560→.494268s，destination prepare长尾与default pause仍主导，不能用26ms
替代全control。下一`agentic-boundary.sbatch`先100Kcapacity，再同36cases全部
用`dp_echo`，配对`native_keep`与`default32_boundary`；mark/方向/重复/预算不变。

native_keep立即PAUSED_ALL，默认32步共识前跑dummy EP；boundary只先置pending_pause，
继续真实decode，在原生默认32步allreduce确认双rankpending后才PAUSED_ALL。
不增加collective频率、不省略idle EP、不改sparse，batch_queue未空则失败。
控制器按人口先暂停idle tenant，再暂停active tenant；四rank返回armed/reached/
callback/default32step receipt，无共识不得begin。任一异常保留暂停/页证据失败。

完整tokens/serial reference/owner/copy峰值/四rankgraph核验不变。额外保存暂停
起点与resume时间、四receipt及完整client cadence。独立报告incoming control delay
（含等边界）与active tenant blocking上界（共识到resume），并核验最大client chunk
gap，不把decode继续期间的solver/握手都乘全active人口，也不忽略incoming等待。
当前只是单request成本实验；尚未实现暂停外ILP或宣称混合trace收益。455本地研究
tests/5skip通过，新增拒绝过期/提前callback与非32边界的安全检查，compute同样回归。

只有GPU正确性与成本独立通过，才依据实测p95固定cost/每GiB斜率重新冻结wait
参数并评估同749trace是否存在净迁移机会；未知/仍亏则保留负结果，先解决成本。
不降低SLO/timeout、不改Azure输出、不选择有利seed、不给未经联合校准的性能项
独特收益结论。登录节点只sync/read/query；唯一A800allocation，300s双watcher。
