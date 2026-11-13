# 图2(b)的加密校准（2026-11-07）

用户要求增加不均衡度扫描点，并增大总batch。只运行一次独立校准作业，不恢复自动链、不修改PAUSE或历史证据。

## 预注册设置

- 单节点4张A800；复用现有Qwen3-30B-A3B、attention DP=4、EP=4、TP=1和稀疏CUDA graph配置。
- 两个实例加载，但仅实例1执行校准请求；无迁移、无跨实例并行负载。与旧J1b保持相同的权重驻留条件。
- 每请求固定4096个合成输入token、1024个输出token。该固定长度是独立microbenchmark设置，不作为在线调度的输出长度oracle。
- 总batch为32、64、128；各取9、10、11个布局，共30点，每点2轮、共60次测量。
- 布局为`[n+d,n,n,n-d]`，`n=BS/4`，横轴不均衡度为`max−min=2d`。第二轮旋转rank位置；每轮随机排列布局顺序。
- 每个BS使用同一组输入token序列，不随布局变化；最大单rank为64请求，预检完整固定测试长度的容量，不能容纳的点明确记录为跳过。
- 先以BS128均衡布局暖机，暖机不进入统计。每次测试均等其他请求排空。
- 保存逐请求cadence、输出长度和时间；仅在所有请求同时decode的公共窗口取步时，窗口两端各去掉1秒、有效窗口至少3秒；发生抢占或不完整采样的试验不计为有效点。
- 图上显示两轮步时均值和min–max误差棒；不把请求数当作独立重复样本，不将两轮区间称为置信区间。
- 作业硬限1小时，扫描最多2800秒、单case最多120秒；不完整则如实报告，不自动补跑。先保留当前论文图，结果完整核验后再替换。

配置：`configs/imbalance-dense-20261107.json`。运行入口：`--phases imbalance --imbalance-config ...`。

## 提交记录

- tag：`fig2-dense-20261107`；Slurm作业：`300123`。
- 提交时间：2026-11-07 18:08:18 +08:00；4张A800，硬限01:00:00，仅执行`imbalance`阶段。
- 作业包SHA-256：`4bb4d0acfb4f2133c7c3b2340b253ce9bb75f7256fecfe660c3a3e4c7e022b3a`。
- 本地runtime测试76项通过；分析器测试5项通过。测试环境有既存的`asyncio_mode`配置警告。
- 分析器：`tools/analyze_imbalance.py`，核对每条cadence、固定请求数、布局、共同decode窗口、抢占计数及两轮完整性；数据未完整时拒绝更新论文。
- 截至2026-11-07 18:10:22 +08:00，作业为`PENDING (Priority)`。Slurm暂估2026-11-07 20:44:08开始，非保证时间；此时尚无新实测结果，论文图2保持不变。

## 本次作业的手动检查与分析

不启动observer或自动补跑。完成后核验归档，再运行分析器；分析输出与原始证据分开放置。

```sh
/tmp/xplace-venv/bin/python kvcached/experiments/xplace/tools/xplace_ctl.py status fig2-dense-20261107 300123
/tmp/xplace-venv/bin/python kvcached/experiments/xplace/tools/xplace_ctl.py fetch fig2-dense-20261107 300123
/tmp/xplace-venv/bin/python kvcached/experiments/xplace/tools/analyze_imbalance.py \
  evidence/xplace-fig2-dense-20261107/job-300123/out/calibration/imbalance \
  --output Paper/draft/imbalance-dense-analysis.json
```

2026-11-08验收通过：60次测量、30点、4672请求，DP4/TP1/EP4和四卡共址均已核对。完整分析为`Paper/draft/imbalance-dense-analysis.json`。BS32/64/128最大失衡布局相对各自均衡布局的平均步时增加39.7%/59.7%/91.7%，均为同一作业内两轮，不代表跨节点重复。

图2(b)保留用户要求的BS32/BS64双曲线，更新为9/10点及两轮范围；BS128的11点保存在完整分析并在正文报告。不得把J1b旧点混入新实验的重复样本。正文区分4K输入长度与decode期间继续增长的上下文，并注明图2(a)与新图2(b)来自不同校准作业。
