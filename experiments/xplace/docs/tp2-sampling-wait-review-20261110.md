# TP2采样等待：源码边界与下一次判别协议（2026-11-10）

## 当前结论

309147已完成并归档；24例/336请求及完整telemetry通过的是产物检查，不是独立预测验证。沿其长间隔证据只读检查部署源码后，可以明确两类潜在等待位置，但**尚不能给它们分配实测耗时，也不能宣布GPU/collective根因已确证**。本次没有改runtime、参数、冻结r3包或启动新allocation。

源码快照记录于03:47:04–03:50:36 CST，共24个文件，逐项核对远端SHA、tar内容和本地副本。五个r3计时接口及运行环境记录的`config/parallel.py`、`distributed/parallel_state.py`共七个历史SHA相符；**另外17个文件只有作业后的快照，包括GPUModelRunner、dp_utils和config/vllm.py，不能追认它们为r3当时加载字节的完整证明**。

### 1. sample外层时间不是采样kernel时间

当前源码的同步调度路径为：

`Worker.sample_tokens → GPUModelRunner.sample_tokens → _sample → _bookkeeping_sync → _to_list`。

`_to_list`依次执行pinned CPU buffer的non-blocking copy、`transfer_event.record()`、已有的`transfer_event.synchronize()`，之后才返回Python token列表。相关位置为`gpu_model_runner.py:4497`、`:4660`、`:3668`、`:7757`。

因此，外层sample区间**可包含此前enqueue的forward、采样和copy依赖的等待**，而不是这些工作各自的GPU耗时。短`worker_execute`也不证明该步forward已经在GPU完成。这是由调用顺序支持的解释；r3没有记录上述内层边界，不能据此断言其长sample全部耗在`_to_list`，或把它叫作“纯D2H拷贝慢”。

归档runtime明确设置`async_scheduling=False`、`enable_dbo=False`、`enable_eplb=False`；main.log记录异步调度关闭，初始化配置为PP1、无speculative config。因此不应将此批结果解释成PP异步token广播、异步output-copy线程或EPLB重平衡耗时。`with_gpu_sync_check`的实现控制sync-debug模式，其名字也不代表每次额外执行设备同步。

### 2. dummy不是空操作，DP控制与EP数据通信不同

`Worker.execute_dummy_batch`调用`GPUModelRunner._dummy_run`；dummy和真实执行均进入`_determine_batch_execution_and_padding → coordinate_batch_across_dp`。`dp_utils._run_ar`执行DP all-reduce，随后通过`.item()`等操作取得同步后的graph模式/布局。dummy还会执行模型forward；它并非没有GPU工作。

`disable_nccl_for_dp_synchronization`有CPU/Gloo和device-group两条分支。当前`config/vllm.py:1114`在未显式覆盖且同步调度时将其置False，和归档engine参数相配时应走device-group分支。**这是源码配置推断，r3未单独持久化这个最终布尔值；不能把DP协调直接称为CPU barrier。** 下一次必须记录实际解析值与group标识。

MoE还有独立的sequence-parallel条件。由历史SHA匹配的`config/parallel.py:654`和DP2/TP2/EP开启/AgRs配置，可推断MoE sequence parallel开启；不能用编译pass里的`enable_sp=False`否定它。已选中的AgRs manager在该分支使用EP group，dispatch为all-gatherv、combine为reduce-scatterv；而padding协调使用固定TP lane的DP group。EP4不意味着每个控制collective都由同样四个rank参加。

MoE的naive/internal-kernel分支还取决于quant-method能力。这里仅记录可达调用与manager选择，不把某条分支的存在或Python返回时间当作该collective的实测执行/完成证据。

### 3. 逻辑idle不等于设备静止

历史SHA匹配的`core.py:2045`在本rank未执行真实请求、但本地仍有未完成请求或DP wave仍活跃时可继续dummy；`:2089`每32步才进行一次wave结束共识检查。因此waiting-only、空rank、刚完成最后请求等情形不能仅凭逻辑请求阶段推断“无对端干扰”。

本地未部署的`candidate_peer_state`目前输出的是请求逻辑phase；其`idle`没有engine-wave或device-quiescence含义。这不是已接入评分的修复。后续状态契约必须区分请求phase、当前host方法/已知in-flight、DP wave和未知设备状态，不能把`available + idle`自动映射为`other_active=False`。事务暂停也不能代替全lane静止证明。

## 判别协议v1：范围冻结，尚未实现/提交

这不是新估计器，也不是GPU根因已确证后的算法修复。下一次若补测，仅增加以下四个有界host范围；先完成实现差异、CPU/隔离测试和源码门核验，才可创建新tag。禁止原r3包盲重跑。

| 新范围 | 挂接位置 | 能判别的内容 | 不能声称的内容 |
|---|---|---|---|
| `runner_sample` | runner实例的`_sample` | sampler调用是否覆盖外层长sample | 单个sampling kernel耗时 |
| `runner_bookkeeping` | runner实例的`_bookkeeping_sync` | CPU输出整理范围 | 独立于其子范围的额外耗时 |
| `runner_to_list` | runner实例的`_to_list` | 含已有copy/event等待的输出物化范围 | 纯copy传输耗时或确切阻塞kernel |
| `dp_coordinate` | GPUModelRunner实际查找的`coordinate_batch_across_dp`绑定 | dummy/execute内DP协调范围 | 某个NCCL kernel自身耗时 |

`coordinate_batch_across_dp`已通过`from ... import ...`绑定在runner模块中；仅替换dp_utils模块的同名属性可能漏测，测试必须覆盖真实查找点和完整恢复。不得替换collective实现、删除既有同步、改变CPU/GPU后端或禁用dummy。

采集与验收约束：

- 保留原worker/engine/frontend边界、固定24例/336请求、12个off/on配对、原布局/轮序/seed/prompt/output、13进程及所有lane身份；继续只做机制pilot，不转为正式收益实验。单节点四A800、单allocation、硬时限一小时。
- 新scope默认关闭；只在现有进程内有界buffer写host时钟，父子scope/线程可重建。不增加CUDA同步、逐步RPC、热路径落盘、tensor值提取或前端轮询频率。解析后的调度/DP/EP配置在边界记录，不能每步从GPU回读。
- 每个新scope必须保存enter/return/error及父区间关联；边界失败、缺失、溢出、重复或未恢复都保留并拒绝验收。下一次启动的源码门覆盖上述24份快照，不能借用r3的五份接口SHA为新增挂钩背书。
- 分析沿用100ms callback-gap阈值和全请求窗口，列出每个长gap及全部例外；按父范围的子区间并集计算覆盖及剩余host时间。`runner_to_list`嵌套在bookkeeping中，不能把二者相加；无长gap案例和低覆盖/零覆盖案例不得删除。
- 全部12个off/on差异仍原样报告；不扣除开销、不挑开关条件放行、不把host范围测量冒充CUDA completion。若需要kernel/collective级归因，仍需另行预注册更低层观测，不以本协议直接作肯定结论。

对r3已归档的96份worker记录进行**静态事件数投影**：每次execute/dummy至多新增一次DP范围，每次sample至多新增三个范围，公式为`old_events + 2 × (execute + dummy + 3 × sample)`。固定旧调用序列下最大为36,294，96份均未超过原100,000上限。完整逐记录投影保留；这不是新采集开销的实测值，也不能保证未来运行调用数不变，运行时溢出仍须失败，不能静默丢事件。

## 估计器与正式实验仍未放行

内部等待结果取得前，不把`phase`常量外推到整个预测horizon，不把旧Conv-layout校准bucket当成通用peer bucket，也不把事后长区间长度回填成在线可用特征。只有截止时刻前已完成的样本或已观察到的in-flight起点才具有因果可用性；当前仍未冻结新鲜度、阶段转换、索引和缺失时的有效预测分支。

完整非分时评分仍未接入，没有可运行TP2代价参数。保留全部95个decode点、12例r2失败干扰、55.5374%/158.8382%负结果及15%/30%门。之后依次需要冻结估计器与评分契约、完整评分的对称/容量测试、独立GPU预测、长上下文/容量pilot，再冻结统一并发/速率/预算做配对消融。TP2独立配对未通过前不做混合拓扑，不恢复分时或旧observer。

## 证据导航

- `evidence/ep4-tp2-ablation-20261108/sampling-source-review-20261110.json`：三份source tar/SHA、24文件与七个历史锚点、静态调用位置、96条事件数投影及限制。
- `evidence/ep4-tp2-ablation-20261108/review-sampling-source-20261110.py`：只读本地核查程序，拒绝覆盖既有报告，不import部署包、不拟合参数。
- `evidence/ep4-tp2-ablation-20261108/sampling-source-20261109T194659Z/`、`sampling-communications-20261109T194826Z/`、`sampling-dispatch-20261109T195031Z/`：源码快照及传输收据。
- `tp2-observability-results-20261110.md`、`tp2-scoring-state-contract-20261110.md`：原实测结果和未部署的状态准备；旧结论及SHA快照不覆盖。
