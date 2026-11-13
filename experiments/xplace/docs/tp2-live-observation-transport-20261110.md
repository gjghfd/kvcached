# TP2实时观测生产端：本地实现，未部署

## 1. 解决的缺口与边界

r4确定了长callback间隔内的host等待位置，没有证明GPU根因；其stop-export事件不能回填为当时已收讫的在线特征。上一版`joint_prediction_state`只是输入校验器，没有真实生产端。本轮新增默认关闭的生产端和控制器收讫记录；**不新增预测器、参数、评分路径或GPU作业**。

源码核查发现，存档vLLM的`WorkerProc.worker_busy_loop`逐条取命令并同步执行；DP engine在同一busy loop中交替处理utility输入与推理step。因此通过原worker collective RPC查询“当前阻塞中的host调用”会等该调用返回，不能用这种RPC伪装并发采样。

实现使用每个worker/engine独立的本机Unix socket监听线程。worker仅在`execute_model`、`sample_tokens`、`execute_dummy_batch`的进入/退出更新内存状态；采样、序列化、socket发送在监听线程进行。没有新增逐step RPC、CUDA API/同步或热路径文件写入。此设计**不等于低开销已经验证**：锁、Python/GIL、线程调度和socket读取均可能扰动运行。

## 2. 接入位置与有界资源

- `VerificationWorker.initialize_from_config`在真实DP/TP identity可用后安装wrapper；参数、返回对象和原始异常保持不变。
- `bind_engine`安装独立rank采样，读取实际`engine.engines_running`单一布尔字段；同时给既有scheduler snapshot/summary添加生产端clock ID。
- `PlacementController.prediction_observations(root, timeout_s)`是显式的按需接口，**尚无准入、周期评分或自动轮询调用点**。每次并发采集两个tenant的所有rank scheduler RPC和所有socket，不等待某个阻塞scheduler才开始socket采样。
- 每个进程一个监听线程，单连接处理、backlog=1、4KiB消息上限；host嵌套最多4层。非法并发、栈深或返回顺序会将该worker状态置为unavailable，不把冲突状态当none/solo，也不改变原调用的执行结果。
- socket由调用方指定的短路径、同用户0700私有目录承载；endpoint本身0600，拒绝目录symlink和覆盖既有socket。目录必须按新pilot创建，禁止复用其他作业的root。消息记录真实producer PID。关闭只删除自身inode的socket，不删除实验PAUSE或其他证据。
- 接收端有显式总timeout和4KiB限制；断连、错误clock/identity/时间戳、生产端异常、缺失lane都保留失败记录。取消和异常收束剩余协程，不留下未完成的controller采集任务。生产端异常不会静默杀掉监听线程。

Linux clock ID由boot ID与`/proc/self/ns/time`共同标识`time.monotonic`域，生产端与收讫端各自读取并比对。平台/namespace不支持时拒绝启用，不用随意字符串代替。CPU测试在macOS使用明确的测试clock；这不是目标Linux时钟或真实GPU可用性验证。

## 3. 收讫时间与rank/lane不可混同

scheduler的`available_at`在每个rank自己的RPC返回时采集，早到rank不会被统一改成`gather`完成时刻；socket的`available_at`在完整一行消息到达时采集。`observed_at`来自生产端。全部记录要求`observed_at <= available_at`；未来、非法或缺失时间戳不能成为observed记录。

**wave是engine rank属性，不是worker lane属性。** 本轮核查纠正了上一版抽象契约的一个生产端假设：worker没有可独立读取的`engine.engines_running`。因此原始transport分开保留4条`rank_waves`和8条`workers`（双模型DP2/TP2），各有自己的观测/收讫时刻。不把rank wave复制到两个lane并伪称同一时刻观测。

输出kind刻意使用`live_scheduler_transport_v1`/`live_execution_transport_v1`，**不能直接传给旧`joint_prediction_state`的live snapshot接口**。已有预测校验器对该原始kind明确拒绝；需要下一步单独审定rank wave与lane host的联合freshness/skew及转换契约，且保留各自时间戳。不能只重命名kind、覆盖时间戳或用未来样本补齐缺失lane。

`status=observed`仅说明本次完整传输和基础校验成功，**不是fresh/同步状态、GPU空闲证明或预测可用性**。所有输出保持`device_activity=unknown`，控制器保持`forecast_status=unavailable_not_estimated`。无host调用、wave=False、逻辑idle均不能映射为solo。

监听线程能在释放GIL的host等待期间读取内存；若GPU/扩展调用持有GIL或进程未获调度，它也可能迟到或timeout。此时保留unavailable。scheduler RPC仍可能等待推理边界；并发采集不保证全体样本足够接近，不能用最新socket状态掩盖旧scheduler样本。

## 4. 开关与部署前置

默认不启用。未来独立pilot需要明确设置`XPLACE_LIVE_OBSERVATION=1`、私有本地`XPLACE_LIVE_ROOT`以及现有`DP_OPT_SERVICE=instance-1/instance-2`。重型`XPLACE_TIMING_ALLOWED=1`与新开关互斥，避免将r4 stop-export计时和新通道混为同一部署条件。没有修改r4冻结upload、sbatch、manifest、原始结果或已拒绝的拟合报告。

本轮没有创建/提交新tag，也没有冻结在线采样频率。未来打包必须包含新模块、核验实际vLLM/patch接口SHA与Linux clock/目录权限，并预注册采样deadline、频率和观测扰动检查；不应直接把当前本地测试当作GPU部署验收。

## 5. 验证与下一步

新增测试实际启动本地Unix socket、阻塞worker线程及独立spawn进程，检查阻塞调用尚未返回时的样本、真实PID与收讫顺序、8-lane/4-rank完备和缺失保留、超时/取消/清理、有界状态、方法恢复，以及真实`bind_engine`/`VerificationWorker`调用位置的CPU替身集成。没有运行分时策略。精确测试版本、初次夹具失败、修复、隔离副本和source SHA见本轮evidence收据。

下一步先完成双来源rank/lane因果合并与same-reference工作horizon契约，再实现有效估计器和完整对称评分。r1–r4仍是已查看探索数据，95个decode点与12例失败干扰全部保留；15%/30%独立预测门不变。容量增长保护不得根据预测干扰缩小；独立预测和长上下文容量pilot通过后才进入公平配对消融，TP2通过后才允许混合拓扑。当前所有campaign总门仍为false。
