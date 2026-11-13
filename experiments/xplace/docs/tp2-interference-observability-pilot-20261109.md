# TP2干扰观测机制pilot v1（2026-11-09）

## 状态与边界

本协议在307623的聚合负结果和全部cadence探索之后制定，是**下一次执行前冻结的机制观测方案**，不是事后预注册，也不是已完成的独立预测验证。尚未实现观测开关、构建staging、生成manifest或提交作业；没有可运行TP2代价参数。它不能放行实测代价、正式消融或混合拓扑。

307623、失败的`measured-cost-r1/r2`和两种固定探索模型全部保留。模型误差门仍为relative RMSE≤15%、最大relative error≤30%；不能删除长停顿、改变目标窗口或把平均步时替换为间隔中位数。正式数据集、Code64、Conv原输出和同条件要求不变。

## 已知事实与待判别原因

- 12例/168请求已通过原始验收。9例活跃背景中，Code间隔中位数为39.13–44.03ms，但完整decode平均为48.09–168.65ms；按事先固定100ms阈值描述的长间隔占其cadence总时长23.26%–82.18%。7例250ms窗口中的Code/Conv输出速率相关系数≥0.97，另2例仅0.2506、0.5708，不能略去。
- 每个长Code间隔内，Conv平均每请求推进0–1.5 token，并非看到Code停顿时Conv持续高速前进。所有这些时间戳都是前端输出可见时刻，不能据此确定是GPU调度、EP collective、engine进程、IPC、Python/GIL或主机调度引起的共同停顿。
- `bal8-r1`和`bal16-r1`在Code首token之前1秒内，每个Conv请求都只有一个采样点，缺少可计算速率的第二点；两组分别为16和32请求。后一个callback在截止时刻之后约1.8–3.2ms才可见，不能偷偷用于截止时刻预测。stride1仍允许多token合并callback。
- 固定留一整轮交叉验证中，精确布局均值模型覆盖27/27目标，误差44.7771%/165.6571%；布局+可见上下文+lag1s ridge仅能预测9个Conv-prefill目标，误差56.3971%/147.4857%，另18个decode目标按缺失fold规则保留为missing。不能换窗口或追加模型直到通过。同一job三轮的交叉验证不是独立pilot。
- 非分时调用审计另发现：`plan_admission`和`plan_return`经`layout_cost`调用`integrated_cost`时未传其他tenant活动状态，最终`other_active=False`。三组CPU夹具中将干扰参数大幅改变后，完整评分结果不变，而显式干扰API的三种阳性对照明显改变。compact state也没有保留prompt长度及显式phase。因此只修fitter或参数文件不足。

证据：`causal-cadence-diagnostic-307623.json`、`causal-cadence-review-307623.json`、`interference-callpath-audit-307623.json`，均位于`evidence/ep4-tp2-ablation-20261108/`。这不是所有因果模型不可能达到门槛的证明。

## 固定机制矩阵

- 同节点4 A800，两个模型仍DP2/TP2/EP4，最多一个allocation、作业硬时限1h；不取消重投，不运行trace或分时策略。
- 在独立新tag中保留启动测试、拓扑与八次双lane迁移校验，再执行3轮×4布局×2观测条件，共24例。布局固定`[0,0]`、`[2,2]`、`[8,8]`、`[16,16]`；每轮顺序不变。
- 每个布局内配对`timing_off`和`timing_on`。第0/2轮为off→on，第1轮为on→off；不看先验收益改变顺序。两条件使用同一原Code prompt/64输出，以及原seed23、原逐轮生成规则、4K Conv prompt/3000输出。共336个请求；所有条件均用stride1，正常执行负载不做其他修改。
- 每例单独保存原始请求、计划、输出精确长度、窗口、抢占及开关；每条件分别验证完整12例的schema2原始契约。任一条件缺例或抢占，整套机制pilot不完整；不得只保留timing_off或timing_on。不能将两条件的重复拼接为六轮校准来压低残差。
- 307623的12例interference阶段约822秒；24例约1644秒只是排程估计，不是时限保证。硬时限仍3600秒；为导出/清理预留时间，在3300秒记录超时并结束执行，保留未完成列表，不按结果删例或补到通过。

## 必须实现的观测契约

观测代码默认关闭，所有记录先写有界内存，case结束且引擎空闲后导出；禁止在热路径同步写共享文件系统、逐步远端RPC、主动CUDA同步或改变调度策略。上传前必须依据已安装vLLM版本的源码核对hook，不能猜测私有接口。

1. **前端**：独立async heartbeat，固定20ms；记录计划唤醒、实际唤醒、最大延迟，及独立watchdog线程的单调时钟采样。为每次stream callback记录进程/线程、请求ID、累计已见token和本地单调时钟。回调记录可以复用已有cadence，不重复计时或生成新oracle。
2. **engine/scheduler**：本地step边界记录调度前后时间、已计算token、该步实际scheduled token数、prefill/decode/混合batch状态、活跃布局及上下文和。不能从后来的请求完成时间倒填phase。记录scheduler输出交给worker和结果返回的边界，区分无调度、等待执行和输出发送。
3. **worker**：覆盖两个模型全部DP/TP lane，记录已验证执行接口的进入/返回、rank/进程/GPU映射。host返回不自动等于GPU完成，字段必须标注语义；本轮不插入阻塞CUDA计时。若只能定位到worker区间，不把它进一步归因于kernel或NCCL。
4. **对齐与完整性**：同节点使用`monotonic_ns`，启动/结束保存与wall时钟的对应关系及process ID、topology、源码SHA。每进程每例最多100000条事件，计数总写入/导出/丢弃；溢出或缺lane显式使telemetry验收失败，不覆盖旧事件后宣称完整。phase边界和首尾时刻必须覆盖整个请求窗口。
5. **开销**：原始off/on逐对报告完整case耗时、Code TTFT/TPOT、Conv TPOT、全部callback及丢失计数，不通过删除慢例或减去平均开销修饰性能。少量同job配对无法证明开销可忽略；timing_on默认只作定位，不直接用于正式拟合参数或性能比较。

有后台观测活动时仍只由当前session提交/验收。不得恢复旧observer，不删除PAUSE。

## 可证伪的判别与后续修复

- 前端heartbeat出现同窗长延迟而worker持续推进：优先调查前端可见性、GIL或主机调度；不能用输出cadence直接拟合GPU服务时间。
- 前端heartbeat正常、多个engine/worker边界同时延长：进一步定位engine、IPC或执行区间；仅有host时间不能区分GPU排队与collective，保留未确定结论。不能直接“修复”为强制交替运行。
- 同layout/phase但不同当前上下文或历史服务状态能解释差异时，才提出固定的因果估计器。特征只能来自决策时刻之前，缺失/陈旧状态须显式拒绝预测或采用事先验证的保守分支，不能使用目标窗口输出、repeat ID或未来长度。
- 无论观测结果如何，非分时评分都需要显式保留并传播其他tenant的当前phase、布局和状态时间戳。候选预测/基础值/选定分支需持久化；加入tenant互换、solo、prefill/decode、phase转换、缺失/陈旧值和候选状态传播的测试。容量保护使用solo速率的保守意图与性能评分分开，不能因干扰让安全余量随便缩小。
- 真正预测pilot必须在估计器、特征截止、聚合方式和误差门冻结后另行独立执行，完整纳入新预测以及全部95个旧decode点和12例旧干扰，不用机制pilot探索结果冒充独立泛化证明。通过代价与长上下文/容量pilot后才能冻结统一max-inflight/速率/预算并进入配对消融。

下一可执行工作是实现默认关闭且有界的观测hook、相应单元测试与main/calibration显式入口；完成本地和独立staging回归、源码差异和计划SHA后才可生成新tag。现有r2冻结包不可修改，当前协议不表示已有提交就绪包。
