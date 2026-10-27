# 固定DP暂停共识间隔控制

2026-10-20 23:10预声明，在235647负结果之后执行，不修改已归档指标或默认sparse graph。

235647确认ILP的A replay保持8，而KV-only出现约450步replay16；但ILP端到端吞吐
反而下降，首个ILP B到达pause_call8.227秒。原生vLLM0.26.0 DP engine
`_has_global_unfinished_reqs`每32步调用`ParallelConfig.sync_dp_state`，pause
会设置pending_pause且继续dummy execution直到所有rank达成共识。快慢与requestdecode
位置、dummy/JIT及同GPU另一模型干扰可能有关；8秒不是ILPsolver（约4毫秒）。

新控制在opt-in live placement环境中固定`DP_OPT_PAUSE_SYNC_INTERVAL=1`，所有模型、
所有rank、四策略全程相同配置，每step调用相同的`sync_dp_state`协议，只有全rank共识
才清pending/忽略stalewave。默认32保持原实现；不得仅本地pending时临时改频率，
否则rankcollective顺序可能不同。保持每rankEP/dummy正常执行，不跳过模型collective。

增加的CPU/Gloo同步开销完整计入所有策略，不将此控制比较宣称默认vLLM性能收益。
三repeat、四策略、同tradeoff18request/预算576:480MiB、输出256、SLO20秒，
其他参数/单模型外推/.475固定cost不变；继续统一control预热。worker/runtime-control记录
间隔，原始capture/ownership/token和所有pause/resume/复制成本保留。

目标是检验长pause是否降低、性能项是否仍误预测；负结果同样记录。若仍无ILP vsKV-only
收益，必须做同硬件双模型联合context/layout校准或改cost，不反复只筛正结果。
CPU测试核对相同collective参数/consensusflags/intervalguard；真实GPU正确性需新job验证。
