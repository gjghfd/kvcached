# TP2机制观测离线审计（2026-11-10）

## 范围

`tools/audit_observability.py`是在309147排队期间准备的离线审计器，不属于r3上传包，也不修改runtime、实验矩阵、代价估计器或15%/30%误差门。单元测试使用显式合成的24例、336请求和13进程事件夹具；其通过不是GPU结果。

审计只能在终态只读核查、确认没有既有fetch进程或归档之后运行。归档只能取回一次，不能覆盖已有raw、失败报告或审计报告。脚本不访问集群、不提交、不执行推理、不提取归档、不拟合参数、不写`step-params.json`。

## 检查内容

- 逐成员核验fetch归档和本地文件SHA，拒绝链接、路径穿越、重复成员和额外job文件。外部给定的manifest、tar、计划SHA是锚点；39份冻结文件与job副本相符后才从冻结runtime导入校验器。导入禁用bytecode写入，不修改冻结目录。
- 检查终态退出码、启动218通过/1个分析工具skip、五份已部署vLLM接口SHA、DP2/TP2/EP4与四A800、1小时生命周期，以及dataplane/observability/job通过记录。Slurm终态仍须另行核实，不能用文件代替`sacct`。
- 核外迁、回迁两次事务、8次迁移、每端2条TP lane全字节校验和四条2500输出流。此处仍是短上下文数据面证据，不是长上下文迁移/容量验证。
- 按固定顺序保留全部24例，核单调时钟测量区间按计划依次执行且不重叠。两条件各自核12例schema2、168请求、原Code101167/64、Conv4096/3000、请求ID、完整窗口、零抢占；不把两条件合为六轮。缺例、partial、失败文件、格式错误和多余例均保留为失败，继续检查其他例。
- timing_on每例重验13进程、全部worker lane、共享GPU映射、事件边界/序号/计数、零溢出、source SHA、100000上限、start/stop身份一致及idle边界。保存两类heartbeat最大延迟，但不把host/Future时刻冒充GPU完成。
- 重算全请求Conv平均TPOT与完整case耗时，核配对文件与单例源一致；全部12对都保留原值及on/off相对变化。solo的Conv指标保持缺失，不以零填充、不做开销扣除、不据开销筛选条件。
- 校验Code prompt身份、phase entry及3300秒软截止范围、每例时窗、completed/progress/原始校验文件一致性。

即使全部文件检查通过，输出状态也只为`artifact_checks_passed_not_predictive_validation`，实测代价、独立预测、配对消融和混合拓扑四个gate始终false。还须分析真实机制数据，并按既定流程冻结估计器、独立预测和长上下文/容量pilot，不能凭审计器放行正式实验。

## 终态后命令

先核`309147`的`squeue/sacct`、本地命令和既有归档。若已经fetch，不重复调用fetch。仅当确认未fetch且作业终态后，执行既有`xplace_ctl.py fetch`，随后本地运行：

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/xplace-venv/bin/python \
  kvcached/experiments/xplace/tools/audit_observability.py \
  --evidence evidence/xplace-ep4-tp2-observability-20261109-r3 \
  --staging evidence/xplace-staging/ep4-tp2-observability-20261109-r3 \
  --job-id 309147 \
  --manifest-sha256 6f67e7230f21c47e1464ccf5c657d279e649658190aef0236c5d46caaf793650 \
  --tarball-sha256 c8295330ef845e59e5c5885b6419dfd2da5611d94b6db67bfd40ce60954eca9b \
  --plan-sha256 e6871075484b61974a7fdc8cbf4d1ab65bfeef6bff38b918db2dec027d7e6838 \
  --output evidence/xplace-ep4-tp2-observability-20261109-r3/artifact-audit-309147-v1.json
```

退出1会保存失败报告；目标存在则拒绝覆盖。若修复审计器，保留旧报告、修复差异和测试，再用新版本文件名审计同一原始归档，不能把失败报告悄悄替换成通过。

默认真实归档尚未审计；当前状态以campaign和本轮preparation证据为准。后续代码变更不得追改r3冻结包。
