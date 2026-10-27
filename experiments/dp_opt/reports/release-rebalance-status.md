# 优化2：释放后ILP回平衡的实现与实验结果

两A800上的新方法六策略真实实验已结束并独立验证。**回迁正确，但在本冻结trace上
不如原KV-only，优化2的额外净收益未兑现**。本轮实验完成不等于总体50%/多seed收益
目标完成；设置、全部负结果、老Conv代价与原始证据如下。旧rate网格/heldout未执行
部分如实保留为被新目标取代，不称旧全网格通过。

## 六策略最终结果（2026-10-25）

249646于09:49:39获配d1n41a22g02两A800，13:59:45以COMPLETED/0:0结束；
14:04主coordinator发现终态，14:07完成归档与独立核验。父agent再次重算，94rawfiles、
2份job日志、11个冻结runtime文件SHA通过；六策略各75请求，共450次实际完成，
每策略598144输出tokens，默认sparse graph、KV ownership/copy峰值、完整配对输出
及请求覆盖均通过，父verifier结果与coordinator完全一致。52次主SSH查询全部成功，
最大实际间隔300.012s。14:13:50再次实查queue为空，remote query-only watcher与
local coordinator/caffeinate均已自然退出；不新增allocation、不复活旧网格。

### 固定设置

- 两独立Qwen3-30B-A3B-Instruct-2507 tenant共址两A800，每模型DP2/EP2/attention-TP1；checkpoint复用，权重、KV、请求及EP组独立。
- BF16、max_model_len=131072、24GiB/model/rank虚拟pool、10GiB/GPU受控物理KV预算、allocator utilization=.98。虚拟容量/受控预算不等于整卡OOM。
- 默认sparse captures `1,8,16,32,64,128,256`、FULL_DECODE_ONLY/compile mode0、32step边界；不关闭空attention rank的EP执行。
- AzureConv输入长度sample，**输出显式合成8192tokens**；SWE-smith真实约100K输入+64输出。不是原Azure输出长度分布或未压缩history的真实在线agent时序。
- 两独立Poisson流，Conv=.0307909199195414RPS、Code=.001RPS、seed20261024；600s自然warmup+1800s评估，总到达窗口2400s。
- 75请求全部drain；预声明600–2400s主窗口54请求（2Code/52Conv）。fair阈值30s、完成SLO120s、driver timeout900s均未修改。
- 原五策略与新kv_only_rebalance使用同冻结trace、预算、模型、图配置。新策略准入仍KV-only，释放后只做无新人口性能ILP，不强制历史home。
- 固定控制成本1.7743364535272121s、copy bandwidth11.64086282917902GiB/s，回平衡最多4moves、候选8、cooldown10s/retry2s。
- trace SHA256 `2bd50c7e9555e54aa59260db3f9f959230cbf2052985ec50add88488f642dbfb`。

### TTFT：Code、Conv及合并

以下均为预声明54请求窗口，单位秒，inclusive type7线性插值。**Code只有2条**，
其P90/P95/P99仅描述这2个样本，不能当多seed统计或50%稳健收益确认。

#### Code / 2请求

| 策略 | mean | P90 | P95 | P99 |
|---|---:|---:|---:|---:|
| no_move | 802.546 | 816.838 | 818.625 | 820.054 |
| no_move_fair | 97.789 | 111.417 | 113.120 | 114.483 |
| own_model | 802.348 | 816.892 | 818.710 | 820.165 |
| kv_only | 55.469 | 55.893 | 55.947 | 55.989 |
| ilp | 55.852 | 56.697 | 56.803 | 56.888 |
| kv_only_rebalance | 57.100 | 57.992 | 58.103 | 58.192 |

#### Conv / 52请求

| 策略 | mean | P90 | P95 | P99 |
|---|---:|---:|---:|---:|
| no_move | 0.411 | 0.542 | 0.788 | 1.034 |
| no_move_fair | 1.353 | 0.927 | 4.736 | 20.208 |
| own_model | 0.389 | 0.516 | 0.762 | 0.921 |
| kv_only | 0.869 | 0.654 | 1.115 | 12.016 |
| ilp | 0.858 | 0.654 | 0.978 | 12.460 |
| kv_only_rebalance | 0.957 | 0.684 | 1.465 | 13.800 |

#### 合并 / 54请求

| 策略 | mean | P90 | P95 | P99 |
|---|---:|---:|---:|---:|
| no_move | 30.119 | 0.718 | 0.991 | 801.474 |
| no_move_fair | 4.925 | 1.036 | 19.380 | 96.767 |
| own_model | 30.091 | 0.689 | 0.887 | 801.257 |
| kv_only | 2.891 | 0.978 | 9.041 | 55.437 |
| ilp | 2.895 | 0.867 | 9.582 | 55.789 |
| kv_only_rebalance | 3.036 | 1.295 | 10.408 | 57.033 |

### Latency与TPOT

Latency为arrival→completion，TPOT为first→completion除以output−1；不能用TTFT替代
完成时延，不能把Code与Conv输出长度不同引出的合并TPOT均值解释为统一每step成本。

#### Latency(s) — Code / 2

| 策略 | mean | P90 | P95 | P99 |
|---|---:|---:|---:|---:|
| no_move | 810.766 | 825.487 | 827.327 | 828.799 |
| no_move_fair | 108.821 | 122.331 | 124.020 | 125.371 |
| own_model | 811.258 | 825.454 | 827.229 | 828.648 |
| kv_only | 66.591 | 66.738 | 66.757 | 66.771 |
| ilp | 66.869 | 67.426 | 67.495 | 67.551 |
| kv_only_rebalance | 68.133 | 69.193 | 69.326 | 69.431 |

#### Latency(s) — Conv / 52

| 策略 | mean | P90 | P95 | P99 |
|---|---:|---:|---:|---:|
| no_move | 160.050 | 211.782 | 248.451 | 269.022 |
| no_move_fair | 161.382 | 212.463 | 249.572 | 280.807 |
| own_model | 160.157 | 212.924 | 249.492 | 271.453 |
| kv_only | 161.739 | 214.210 | 231.443 | 272.048 |
| ilp | 163.313 | 218.072 | 234.992 | 272.203 |
| kv_only_rebalance | 162.785 | 220.737 | 235.319 | 274.899 |

#### Latency(s) — 合并 / 54

| 策略 | mean | P90 | P95 | P99 |
|---|---:|---:|---:|---:|
| no_move | 184.151 | 233.271 | 268.743 | 809.662 |
| no_move_fair | 159.435 | 211.385 | 247.870 | 280.162 |
| own_model | 184.272 | 234.348 | 271.271 | 810.193 |
| kv_only | 158.215 | 209.509 | 228.895 | 271.914 |
| ilp | 159.741 | 213.357 | 232.431 | 272.060 |
| kv_only_rebalance | 159.280 | 215.917 | 232.805 | 274.764 |

#### TPOT(ms) — Code / 2

| 策略 | mean | P90 | P95 | P99 |
|---|---:|---:|---:|---:|
| no_move | 130.473 | 137.278 | 138.129 | 138.809 |
| no_move_fair | 175.103 | 176.954 | 177.186 | 177.371 |
| own_model | 141.425 | 146.946 | 147.637 | 148.189 |
| kv_only | 176.540 | 180.943 | 181.493 | 181.934 |
| ilp | 174.864 | 179.440 | 180.012 | 180.469 |
| kv_only_rebalance | 175.138 | 177.798 | 178.131 | 178.397 |

#### TPOT(ms) — Conv / 52

| 策略 | mean | P90 | P95 | P99 |
|---|---:|---:|---:|---:|
| no_move | 19.490 | 25.752 | 30.217 | 32.806 |
| no_move_fair | 19.537 | 25.857 | 30.393 | 32.840 |
| own_model | 19.505 | 25.917 | 30.366 | 33.093 |
| kv_only | 19.640 | 26.025 | 28.134 | 33.160 |
| ilp | 19.833 | 26.501 | 28.535 | 33.195 |
| kv_only_rebalance | 19.757 | 26.728 | 28.613 | 33.519 |

#### TPOT(ms) — 合并 / 54

| 策略 | mean | P90 | P95 | P99 |
|---|---:|---:|---:|---:|
| no_move | 23.600 | 28.391 | 32.770 | 129.962 |
| no_move_fair | 25.299 | 29.016 | 32.671 | 174.964 |
| own_model | 24.021 | 28.518 | 33.069 | 141.011 |
| kv_only | 25.451 | 26.788 | 33.045 | 176.210 |
| ilp | 25.575 | 27.229 | 33.072 | 174.521 |
| kv_only_rebalance | 25.512 | 27.296 | 33.405 | 174.939 |

### 120s完成SLO与控制成本

| 策略 | Code满足/2 | Conv满足/52 | 合并满足/54 |
|---|---:|---:|---:|
| no_move | 0 | 1 | 1 |
| no_move_fair | 1 | 2 | 3 |
| own_model | 0 | 1 | 1 |
| kv_only | 2 | 0 | 2 |
| ilp | 2 | 0 | 2 |
| kv_only_rebalance | 2 | 0 | 2 |

下表是完整75请求trial的累计控制墙时，不是某一请求关键路径；子计时可能重叠，
preplan可与decode重叠，不能相加声称实际请求延迟。端到端开销已经进入上面的原始时延。

| 策略 | 总moves | KV GiB | control+preplan(s) | pause call(s) | preplan(s) | migration elapsed(s) |
|---|---:|---:|---:|---:|---:|---:|
| no_move | 0 | 0.000 | 4.059 | 0.000 | 0.000 | 0.000 |
| no_move_fair | 0 | 0.000 | 5.612 | 0.000 | 0.000 | 0.000 |
| own_model | 0 | 0.000 | 3.801 | 0.000 | 0.000 | 0.000 |
| kv_only | 1 | 1.239 | 6.340 | 1.221 | 0.033 | 0.102 |
| ilp | 1 | 1.239 | 7.766 | 2.069 | 0.030 | 0.062 |
| kv_only_rebalance | 21 | 22.008 | 51.821 | 26.987 | 11.442 | 1.340 |

新方法21总moves包含3次容量准入迁移和18次释放后回迁，15个成功回平衡transaction；
78次release decisions全部独立重建通过。其中2个transaction由Code完成触发：Conv
4:0→2:2（2moves）与1:8→4:5（3moves）；其余13个仅由Conv完成触发。成功transaction
人口4–10，9个只有4条Conv。释放后累计控制含preplan为41.507s；总copy事件1.340s
远小于51.821s总控制墙时，不能将额外开销全部归为KV搬运带宽。

### 结论与下一算法修正

1. **优化2实现正确，尚未提高原KV-only净收益**。主窗口新方法相对KV-only：Code/Conv/合并meanTTFT分别增2.94%/10.15%/5.03%，P95增3.85%/31.43%/15.12%；Conv meanlatency增0.65%、meanTPOT增0.60%，完成SLO不变。保留原KV-only为当前默认候选，不因新方法名字或迁移次数宣称性能改善。
2. **容量准入收益仍成立，但不能把它归给回迁**。新方法对普通baseline Code meanTTFT下降92.89%、meanlatency下降91.60%；原KV-only更好。对fair，Code/Conv/合并meanTTFT下降41.61%/29.27%/38.34%，Code P95下降48.64%，合并P95下降46.29%；这些不是回迁相对KV-only的增量收益。
3. **联合tail/老Conv代价必须保留**。普通baseline合并P95TTFT=.991s，新方法10.408s，虽然合并mean/P99显著降低。Code占2/54<5%，合并P95并非Code-only tail。120s完成SLO为普通1/54、fair3/54、KV-only2/54、新方法2/54；新方法Conv仍0/52，不能宣称所有SLO改进。
4. **成本与收益模型是诊断方向，不是已验证因果分解**。13次Conv-only触发占15个transaction，多数人口只有4；N128单模型surrogate对小batch/default sparse的收益外推，以及把平均remaining decode作为固定布局长horizon，可能高估短暂不均衡收益。当前数据不能分离计时波动、自然到达补齐、prefill重叠及每个回迁的反事实收益；没有联合校准，不称ILP最优SLO。
5. **下一轮应做选择性释放触发+小人口校准消融**：记录容量迁移造成的临时displacement，在对应Code释放后触发一次ILP；Conv完成只重试未完成的该恢复任务，而不是每次完成都新开平衡。用真实人口/上下文/default sparse布局表校准，并用自然新到达/完成限制获益horizon；同冻结trace与预算实跑Code-trigger-only、校准版和原KV-only。此处是后续候选，尚未实施或提交，不能声称已改善。

只有本轮新方法真实实验及结论完成；总体多seed/多rate收益、双baseline至少50%和生产化目标
仍未完成。原旧网格/heldout未执行部分是被取代，不当作通过。不改SLO、不删请求、不选择
有利窗口或把极小Code样本P95恰好过线当最终收益。

### 原始证据与复现

均相对于ParaCloud根目录：

- `evidence/kvonly-rebalance/live-trace-249646/`：六策略全部原始请求、决策、owner/预算/图与配置。
- `evidence/kvonly-rebalance/raw-249646.sha256`、`raw-249646-20261025-140449.tar.gz`：94rawfiles清单及完整压缩包；out/err独立SHA在archive记录中。
- `evidence/kvonly-rebalance/verification-249646.json`、`parent-final-verification-249646.json`：完整原始trace独立核验，两次结果一致。
- `evidence/kvonly-rebalance/new-method-evaluation-window-results.json`：预声明54请求的六策略Code/Conv/合并全部均值与P90/P95/P99、SLO及配对得失。
- `evidence/kvonly-rebalance/new-method-tail-ttft-results.json`：完整75请求的普通/fair/KV-only配对分析（包含warmup），不与54窗口混用。
- `evidence/kvonly-rebalance/interim-release-diagnosis-249646.json`：78release decisions重建、15transaction触发/布局/gate、逐token对照与38文件中间SHA。
- `evidence/kvonly-rebalance/accounting-249646.txt`、`poll-249646.log`、`campaign-state.json`：Slurm终态与52次300s查询记录、恢复点。

父agent完整核验可在本地运行：

```bash
python3 kvcached/experiments/dp_opt/verify_trace.py \
  evidence/kvonly-rebalance/live-trace-249646 \
  --prompt-bank evidence/agentic-trace-run-20261021/input/prompt-bank.json \
  --output evidence/kvonly-rebalance/verification-rerun-249646.json
```

无commit/push；main的origin仍为gjghfd fork，Aegaeon不重启、不覆盖保护补丁。

## 历史接管与实施记录
09:44:08经SSH实查：248507已于2026-10-25 05:23:13 FAILED，当前queue为空，旧本地
PID75020已退出。原257rawfiles SHA独立核验通过；case0的64请求和case1的75请求
各原五策略均独立通过。case2/1.15x/own_model部分72完成，900s TimeoutError，
末观测aggregate free8.938GiB<incoming9.315GiB、另一个Code仍在场；不是CUDA OOM。
完整失败保留，诊断在`evidence/kvonly-rebalance/old-job-diagnosis.json`。新coordinator
显式校验该诊断与raw manifest，复用原归档，不把失败case当成功或重复申请旧配置。

## 实现与验证

- `kv_only_rebalance`沿用原KV-only准入；完成事件独立触发无新请求ILP回平衡。
- `new_request_count=0/new_gib=0/target=None`不增加虚构人口，保留不动解及真实
  source owner/pinned/串行copy peak约束；单次固定pause费用与预测收益门控独立。
- 默认32step boundary后重新求解当前人口，KV完整device echo、request/runner/
  routing状态、每卡mapped budget继续独立核验。失败保持paused并传播到整个trace。
- 本地专项196项通过；广回归640通过/5跳过/1明确deselect。既有wait-forecast
  `test_future_births_can_pin_shared_pages_after_current_owner_completes`仍失败，未修改；
  首次广回归包含该失败的日志完整保留，不宣称全suite无错误。
- 实现过程中修复joint-table局部变量覆盖incoming count及completion journal别名；
  null block保护未放宽，测试fixture使用合法非零blocks。两个launcher语法检查通过。
- 247429/case0的75请求×原五策略由新独立verifier重新核验为`raw_trace_verified`，
  source/trace与原归档一致；旧snapshot默认格式未改变，不是新方法实跑。

日志/证据在`evidence/kvonly-rebalance/`，相对于ParaCloud根目录。

## 原始inventory的CPU机会检查（非GPU结果）

固定247429/case0、原预算/原测得固定成本1.774336s与copy bandwidth11.640863GiB/s，
对所有“Code已无请求、Conv不均衡至少2、全部可迁移”的准入前inventory进行重放。
13个eligible快照均被当前surrogate判为有利；包括真实fresh-all-tenant的Conv8:0，
ILP选4次迁移至4:4。该快照平均remaining5928.875tokens，预测gain-after-copy
354.370request-seconds，固定预留14.195request-seconds；这些数字是N128外推，
**不是实际Conv节省时间、净收益或SLO保证**。部分其他快照是cached-upper-bound。
新方法使用fresh四rank观测，最终需GPU A/B。准入后快照含新prefill请求的初次筛选
为0eligible，残记录保留，不能把它解读为释放后没有回平衡机会。

## 新实验固定协议与恢复

### 12:07：两完整trial的独立中间核验（非六策略终态）

kv_only和kv_only_rebalance各75请求、598144输出tokens已完整drain。仅下载这两组
已封存的trial及不可变配置，38rawfiles远端打包前后SHA一致、本地SHA全部通过；
78次release decision由verify_rebalance独立重建，15个成功transaction/18回迁
通过四rank默认32step、无新人口、fresh ILP、源owner、完整copy bytes/hash、峰值
预算及最终routing验证。两策略75请求的逐token输出全部一致。新策略共21moves，
其中3次属于原KV-only容量准入，18次属于释放后回平衡，不混为一类。

预声明600–2400s窗口为54请求（2Code/52Conv），**新方法暂未改善原KV-only**：

| 分组 | mean TTFT KV-only→新方法(s) | P95 TTFT(s) | mean latency(s) | mean TPOT(ms) |
|---|---:|---:|---:|---:|
| Code/2 | 55.469→57.100 | 55.947→58.103 | 66.591→68.133 | 176.540→175.138 |
| Conv/52 | 0.869→0.957 | 1.115→1.465 | 161.739→162.785 | 19.640→19.757 |
| 合并/54 | 2.891→3.036 | 9.041→10.408 | 158.215→159.280 | 25.451→25.512 |

120s完成SLO仍为2/54（Code2/2、Conv0/52）。完整75请求也如实保留，合并meanTTFT
2.178→2.278s、meanlatency154.060→155.209s；窗口不为追正而调整。2Code的高分位
仅描述这2条，不作为稳健P95或多seed结论。

累计control+preplan从6.340s增至51.821s，释放后部分41.507s。15个回平衡transaction
仅2个由Code完成触发（Conv4:0→2:2、1:8→4:5），其余13个由Conv完成触发；
transaction人口4–10，其中9次只有4条Conv。当前性能项仍N128单模型surrogate，
小batch下是否高估回平衡收益需要联合小人口profile/选择性触发对照，不能把预测
gain当作实际节省。这些事实提示控制频率和成本模型是下一诊断方向，不证明某个
未经实跑的修正一定有收益。

原始中间包/诊断：`evidence/kvonly-rebalance/interim-trials-249646.tgz`、
`interim-trials-249646/`、`interim-release-diagnosis-249646.json`。首次12:03导出
误用不存在的runtime-control-env.json失败，0byte残包保留为
`interim-trials-249646-export-failed-20261025-1203.tgz`；12:05改用runtime-control.json
重试通过。第一次诊断脚本在结果写入后仅打印字段名KeyError，修正打印后完整指标
已复算；独立verifier本身通过，不把该打印错误宣称GPU失败或隐去。

12:04:22主coordinator观察no_move RUNNING/9completed、实际gap300.007s；六策略
还未终态，普通/fair/ILP对照及整allocation验证待完成。继续同allocation，不覆盖
活动runtime、不重复提交，不把这份中间表标为整个新方法收益实验完成。

2026-10-25 11:49:34：固定随机顺序为own_model→kv_only→kv_only_rebalance→no_move
→no_move_fair→ilp，当前第三策略kv_only_rebalance RUNNING，elapsed1835.27s、
60offered/54completed/75、18moves。这只是进度计数，尚未独立重放release触发、
fresh无准入ILP、copy峰值、ownership或计算配对收益，不能据此称新方法通过。
local coordinator17187/PPID1与caffeinate17188仍存活，SSH查询实际gap300.003s。
辅助agent在11:21记录首次新策略进入并结束该轮监督，其返回遭API断连；父agent
接手等待。主coordinator持续查询未因此中断，所有原query日志保留。

2026-10-25 10:04:28：249646已于09:49:39获配d1n41a22g02两A800，目前RUNNING。
首策略own_model已进入固定trace，elapsed345.76s、12offered/9completed/75；初始化
与default sparse capture无已知新错误。新方法尚未执行，不能把当前运行当作回迁
正确性或收益通过。local17187/PPID1与caffeinate17188存活，两次实际查询间隔
300.005s；继续既有coordinator/remote query-only watcher，不重复申请或覆盖runtime。

2026-10-25 09:49：16文件保护同步完成，唯一新作业249646首轮PENDING/Priority。
local17187（PPID1）/caffeinate17188与remote query-only2051964负责300s监控，
runtime已经冻结；作业pending不是新方法推理通过或收益完成。新提交ledger、
旧SHA备份/新SHA结果都保留在`evidence/kvonly-rebalance/`对应记录中。

使用247429/case0同一trace（75请求；Conv合成8192输出、Code真实100K输入+64输出），
trace SHA256为`2bd50c7e9555e54aa59260db3f9f959230cbf2052985ec50add88488f642dbfb`。
两A800/BF16/128K/24GiB虚拟pool/10GiB物理受控预算/.98/default sparse/32step，
原五策略加`kv_only_rebalance`，同负载同模型参数，不改30s fairness/120s完成SLO。
逐模型与合并mean/P90/P95/P99 TTFT、latency、TPOT及真实控制成本全部报告，
不能把小样本Code-primary候选称为双模型联合最优或50%确认。

coordinator先核验旧best、首次成功SSH后精确停止旧本地提交进程，保留当前GPU作业，
300s监控至终态，全SHA归档并独立核验旧case，再做旧SHA备份/原子更新/新SHA复核。
main/gjghfd origin、Aegaeon protected diff与vLLM补丁均保护。新sbatch有独立提交
ledger，SSH不确定时先查ledger，不重复申请。新作业完整结束才做独立结果核验。

恢复：`evidence/kvonly-rebalance/campaign-state.json`与`campaign-process.json`；
阶段失败先归档/诊断，禁止盲目再次提交。未经授权不commit/push，也不跑Aegaeon。
