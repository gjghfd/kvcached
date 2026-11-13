# EP4TP2 新实测校准部署

本轮把已接线的 causal training pilot 包装成真实可提交作业，不接入
完整评分。`causal_job.py` 以独立前端/引擎子进程运行 off→collect，第二个
预声明 block 为 collect→off。每个条件使用相同冻结四 case/44 请求，
Code/Conv 原 prompt 与输出、共享 waiting queue、3GiB margin、并发1/16、
reserve256 均不变。两个 block 分别占一个最多1h的四卡 allocation，
只在前一个结束并验收后提交后一个。不是“看收益再选是否重复”。

## 执行与身份

两个条件共享 sbatch 起点及3300s软截止；剩余不足1500s不启动新条件。
子进程返回码、stdout/stderr、未执行条件和取消/超时均落盘。
每秒对同一进程组记录 CPU ticks/RSS，用于两种条件相同的开销核查；
这是采样值，不是准确累计CPU或真正峰值内存。成功退出允许5s自然清理，
仍残留进程则强制清理并拒绝继续条件。Slurm硬时限仍为3600s。

`causal_deployment.py` 绑定完整实验 runtime Python，以及实际部署的
vLLM、kvcached、cloudpickle 共2275个Python文件；绑定10个vLLM/kvcached
原生扩展、common/qwen launcher和四个包版本。torch是版本绑定，不能将其
误称为所有CUDA/驱动/系统共享库的逐字节证明。完整源码在登录节点只读
抓取；真正包路径、native SHA、Linux时钟和八个worker的23项实际配置
必须在compute allocation中验证，缺失/变化即退出。

前端启动前核验native和源码；成功条件在engine shutdown后再次核验。
worker记录完整config repr；并核验DP2/TP2、同步调度、EP、禁DBO/EPLB、
chunked prefill、cache/length、capture sizes等。不同条件不共享engine。
启动测试采用显式非分时node清单，禁止whole-directory测试和skip。

## 冻结协议

部署协议位于 `evidence/ep4-tp2-ablation-20261108/deploy-protocol-20261110-r0/`。
原 `causal-training-plan-20261110-r0.json` 保持不变，K16、100ms采样、
80ms截止、128MiB编码事件预算。所有缺失/覆盖率必须报告；最新值socket
不等于完整历史。时钟/PID/epoch身份不能由请求形状或邻近时间猜测。

采集接受要求：drop/invalid为0、sampler无错误、身份/时钟完整、active
round完整transport至少90%、missed ticks不超过10%。这些是采集门，
不是预测门；不允许通过对缺失数据插值满足门槛。

两种顺序均跑完才评估开销。逐case比较Conv窗口反吞吐、平均已收讫token
间隔与Code TTFT，要求每个主指标所有八对完整可用、中位增加≤5%、最坏
case增加≤15%。截尾保留 `incomplete.json`；无首token不可拿截止时间
代替真实首token。CPU/RSS及掉样另报，不仅计算成功请求子集。

训练数据不能充当独立验证。固定保留的holdout在预测器、特征、模型hash
和完整验证协议冻结前不可执行。15% relative-RMSE/30% max error不变。
本次只采训练与开销数据；不能据此声称EP4TP2系统已有效。

## 本地发现

启动套件r0为532通过/1失败。直接父子进程测量证实本地Python3.9.6/Darwin
默认monotonic在此环境有不同进程起点；fake共同clock标签掩盖了差异。
只修十二进程测试夹具，父子均使用系统CLOCK_MONOTONIC；生产Linux时钟
身份、新鲜度阈值与fail-closed规则不变。原失败、原快照及诊断完整保留。

## 提交

2026-11-10 10:48:55（北京时间）提交310540，tag为
`ep4-tp2-causal-calibration-20261110-r5`，限时1h、单节点四A800。
提交前两次squeue为空；独占提交标记和回执防止重复。10:51:00复查为
PENDING/Priority，没有分配到GPU；调度器给出的预估开始时间不是承诺。
本地148、隔离148、完整非分时启动清单533通过，零跳过；独立解包核查
2308个Python文件、两mode真实bank/plan CLI及bash语法。实际GPU启动、
采集覆盖、开销和预测尚未通过，不能把提交当作实验完成。
