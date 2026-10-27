# 32K Azure-length Poisson trace：首轮真实实验为负

2026-10-21，Asia/Shanghai。作业237899，15:56:36–16:49:20，COMPLETED/0:0。
**当前ILP原型在这条固定trace上没有净收益，且性能项未改变决策。** 这不是所有ILP
算法必然无益的结论，也不能用此前固定population的容量准入收益替代本次负结果。
本轮单seed/单rate、每策略一次；尚未完成多seed/多rate最终泛化实验。

## 1. 实验设置与正确性

- 两个独立Qwen3-30B-A3B-Instruct-2507 tenant共驻两A800-SXM4-80GB，每模型
  DP2/EP2/attentionTP1，共checkpoint但独立权重、KV namespace、请求和EP组。
- native vLLM+kvcached，max_model_len32768（输入+输出），max_num_seqs256，
  max_num_batched_tokens4096；虚拟KV4GiB/model/rank，**aggregate mapped受控预算
  2112MiB/GPU**。该预算不是整卡显存容量；实验没有证明整卡OOM。
- sparse captures1/8/16/32/64/128/256，FULL_DECODE_ONLY/compile0，DP pause共识32step。
  保持无prefix cache、异步调度关闭，空attention rank仍执行EP。
- Azure Conversation成对长度sample：P>=2048、256<=O<=512、P+O<=32768。
  来源为缓存byte-window片段，不是全CSV均匀reservoir；合成重复算术prompt，不是
  Azure原始文本。实际平均P3091.14/O370.31，最大P+O8337；32K是配置，不是每请求32K。
- 两独立Poisson流，名义总rate0.60req/s、每模型0.30，arrival窗口600秒、seed20261021。
  固定386请求，A192/B194，实际该窗口到达数/时长为0.64333req/s；未来arrival不随
  排队或暂停改变。每策略142941输出tokens，全部drain，无删请求或改SLO。
- 四策略相同full-reservation scheduler、tenant FIFO和serialized direct-fit路径，
  实际随机顺序own_model→no_move→kv_only→ilp。no_move按count-LB从可fit卡选择，
  容量不足则等自然释放；**不是未经修改的原生vLLM渐进KV baseline**。
- blocked策略仅在global实际free>=incoming logical KV且resident全decode时规划；
  每model/rank最多2候选、最多2迁移，枚举两个arrival target后求staged ILP。
  destination复制完成后才释放source，host-staged文件传输，非优化NVLink路径。
  现有N128单模型surrogate未在本次低batch/长context/双模型联合校准，没有wait-cost候选。

44原始文件与远端SHA一致，1544请求的长度/固定arrival/完整逐token配对均通过，
另有4条不同长度完整串行greedy reference。四model/rank分别有真实FULL decode证据，
ownership/page inventory、观察时actual mapped预算和copy-before-release峰值通过。
独立复算queue/TTFT/TPOT/latency/SLO/drain/control totals；Azure来源长度和Poisson
arrival用缓存sample与seed重新生成，386条全部一致。

## 2. 完整trace结果

所有指标含真实排队、控制、暂停、复制和prefill；20秒completion SLO预先固定。

| 策略 | 平均queue s | p99 queue s | 平均TTFT s | 平均latency s | 平均TPOT ms | 20s SLO通过 | arrival结束后drain s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_move | 41.34 | 107.20 | 41.69 | 58.72 | 46.03 | 32/386 | 101.29 |
| own_model | 64.46 | 168.49 | 64.80 | 82.84 | 49.00 | 29/386 | 150.43 |
| kv_only | 63.64 | 173.38 | 63.99 | 82.56 | 50.21 | 31/386 | 155.09 |
| ilp | 94.07 | 183.40 | 94.41 | 113.74 | 52.21 | 19/386 | 194.31 |

ILP vs no_move：平均queue **+127.54%**、TTFT+126.46%、latency+93.69%，SLO少13请求；
vs KV-only：queue+47.82%、latency+37.77%，SLO少12请求。固定offered rate和完成数下，
不把complete-drain token/s解释为稳态吞吐；本轮净收益结论以queue/latency/SLO为主。

| 策略 | 600s内完成 | 600s末未完成 / 仍在queue | 全窗口精确peak未完成 / queue | tenant A/B平均queue s |
| --- | ---: | ---: | ---: | ---: |
| no_move | 332 | 54 / 45 | 61 / 51 | 35.05 / 47.57 |
| own_model | 308 | 78 / 71 | 85 / 76 | 44.34 / 84.38 |
| kv_only | 301 | 85 / 76 | 90 / 81 | 84.53 / 42.96 |
| ilp | 290 | 96 / 88 | 103 / 93 | 90.37 / 97.73 |

peak与窗口末状态按完整arrival/submitted/completion时刻重算，不用稀疏population
sample估计。监视器2秒sleep不等于每2秒无间断测量：controller lock使最大sample gap
5.63–8.84秒；arrival task实际dispatch lag最大1.04–1.27秒，固定计划arrival仍作为计时起点。

## 3. 容量机制出现，但没有trace净收益

ILP4次真正迁移准入均有global-enough/local-blocked见证，其中2次迁移其他tenant。

| arrival（trace-20261021-前缀省略） | 卡0/1空闲MiB | 新请求logical MiB | 原布局卡0/1实际新增mapped MiB | 迁移model | 控制 s |
| --- | ---: | ---: | ---: | --- | ---: |
| 0-2 | 480 / 288 | 547.5 | 576 / 576 | A | 1.714 |
| 0-13 | 96 / 384 | 421.5 | 480 / 480 | B | 2.258 |
| 1-83 | 576 / 192 | 706.5 | 672 / 576 | B | 3.368 |
| 1-140 | 384 / 192 | 564.0 | 576 / 480 | A | 1.577 |

各行总空闲足够，但任一卡当前actual incremental allocation都超过headroom。
迁移后实际reserve/admission与KV SHA/copy峰值通过。因此不能说“根本没有容量事件”，
但4/386次准入帮助远不足以证明全trace收益。保守candidate/staging无解也不等于
真实allocator或全部可能重放置方案无解。

## 4. 负结果诊断

| 策略 | staged规划 / 准入 | 实际迁移 / 跨tenant | direct控制 s | staged控制 s | pause调用 s | solve s | 实际copy迁移 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_move | 0 / 0 | 0 / 0 | 218.55 | 0 | 0 | 0 | 0 |
| own_model | 19 / 4 | 4 / 0 | 218.72 | 16.28 | 10.06 | .273 | 5.35 |
| kv_only | 18 / 8 | 7 / 3 | 218.69 | 24.25 | 13.17 | .346 | 9.80 |
| ilp | 21 / 4 | 4 / 2 | 217.52 | 21.75 | 13.88 | .366 | 6.87 |

1. **不是ILP求解太慢。** ILP21次求解总计.366秒；17次无准入的staged控制共12.83秒。
   pause调用包括等rank共识/当前step完成，并非这整个区间都GPU idle。actual全局停止
   区间未单独打点，不能把pause_call或总control全部称作GPU暂停。
2. **direct控制仍重。** 每次成功direct准入约.57秒，包含多次snapshot/capacity RPC和
   full-reservation ack；这里没有global pause。四策略均约218秒，不能拿ILP总239秒
   全当新增暂停成本。应把状态采集/ack改成轻量事件驱动路径，而不是调capture。
3. **性能项本轮没有独特作用。** 对ILP21个真实snapshot保留candidate、预算和copy
   cost，仅将performance tables置零后离线重新求解，**21/21 target+move完全一致**。
   这不是另一个真实latency baseline；跨策略执行产生不同population/释放时序，不能
   因ILP trace更慢就推断性能项导致退化。GPU实跑KV-only对照仍单独保留。
4. **小batch没有跨有利capture边界。** 采样resident总数最多13–14（两模型合计），
   全部真实FULL decode只用了padded1或8。N128的attention不均衡拟合在此没有验证，
   不能据此宣称attention本身时间占比小，也不支持把N128模型直接外推。
5. **缺等待与老请求成本。** 当前可迁移就尝试，不比较自然释放的等待成本，也没有
   queue horizon/SLO保护。准入会改变后续驻留、prefill及page packing，接近容量时
   queue积压会放大。ILP平均resident19.67秒vsno_move17.38秒，真实窗口rate乘
   E[logical reserved MiB*resident seconds]/4224约.998vs.884；这是条件性实测压力
   诊断，不是独立测得稳定capacity。21.75秒staged控制不能单独解释93秒makespan差。

## 5. 下一阶段门槛

先改算法/控制，不继续盲调rate追正结果：

- 从decode cadence、remaining tokens及实际page owners预测下一次可fit自然释放时间，
  将wait与migration一起比较；旧请求增加的暂停/TPOT/SLO损失与新请求queue节省
  用相同request-seconds口径计价，copy与resume使用实测cost。
- 在pause前做保守no-pause可行性预筛，失败population缓存与cooldown；避免有global
  free却在有限候选/复制峰值约束下必无解的反复全局暂停。不能牺牲owner验证。
- 联合校准当前小batch、长context和两模型共驻性能；降低snapshot/ack开销，补充
  原生渐进KV baseline。用户要求的默认sparse/32step/maxlen32768继续保持。
- 修改后先同386请求trace对照，再预声明多seed和rate sweep；报告全部正/负轮、
  相同20秒SLO、完整drain、paired输出与raw SHA，不删除困难请求或同步bootstrap。

本轮请求的“跑一个实验看看是否有收益”已得到完整负结果；整体研究收益目标尚未
解决。没有继续提交无改进的重复GPU作业；17:00用户队列空，不commit/push。

## 6. 证据与复现

协议：`../docs/trace-benefit-protocol.md`。外层`evidence/azure-trace-run-20261021/`中：

- `live-trace-237899/`、`raw-237899.sha256`、`archive-237899.json`、accounting及out/err。
- `verification-237899.json`：所有策略完整指标、控制拆分、tenant指标与配对对照。
- `diagnostics-237899.json`：精确backlog、来源重生成、容量见证及same-snapshot ablation。
- trace SHA256：`26ee056f1bc6587a48532ee02f713469c52081bc371de0457fe5c32247114194`。
- `poll-237899.log`：每300秒sleep，实际查询间隔约302–312秒，终态16:52观察。

237861回归环境泄漏失败、237868planning storm超时失败均保留SHA归档，未冒充成功
trial。首次237899未压缩SSH归档600秒超时，156MiB残包保留；改压缩后18MiB归档成功，
44文件全SHA通过，不重跑GPU、不覆盖raw。

集群启动回归133+193tests通过；本地最终研究回归165通过/1跳过，含18条trace复核/
诊断测试。已有asyncio_mode配置warning保留。17:16已保护同步14份报告/导航/复核工具，
远端旧文件备份和新SHA全部通过，fork main及Aegaeon受保护diff不变，未commit/push。

从ParaCloud根目录复核：

```bash
PYTHONPATH=kvcached/experiments/dp_opt python3 kvcached/experiments/dp_opt/verify_trace.py evidence/azure-trace-run-20261021/live-trace-237899 --output evidence/azure-trace-run-20261021/verification-237899.json
PYTHONPATH=kvcached/experiments/dp_opt python3 kvcached/experiments/dp_opt/analyze_trace.py evidence/azure-trace-run-20261021/live-trace-237899 --sample evidence/azure-trace-theory-20261021 --output evidence/azure-trace-run-20261021/diagnostics-237899.json
```
