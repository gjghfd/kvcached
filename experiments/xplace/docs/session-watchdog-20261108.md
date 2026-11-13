# 当前 Codex 实验会话持续监控（2026-11-08）

## 范围

用户先授权异常续接，又于 2026-11-08 23:28 CST 明确授权正常完成也续接；2026-11-10 改为任务等待/运行时每 1800 秒查询一次。异常或正常 turn 结束后发送带实验信息的“继续”，正常续接仅限显式 active campaign。不是旧实验 observer 的恢复：不导入、不运行 `codex_followup.py`，不使用 `codex exec resume`，不创建模型会话，不直接提交实验、归档或删除任何 `PAUSE`。手动取消、活跃 turn、待处理用户输入和单条工具命令失败均不触发。

每次续接只查询一次；若作业仍等待/运行，更新并发布检查点后结束本轮，不再做短间隔前台等待。若当前 session 从已运行作业的进度或时限得到可信结束时间，应在该时点设置一次性提醒并记录依据；Slurm 排队 `StartTime` 不能作为结束时间，作业时间上限也不能称为保证的实际完成时间。排队时只保留半小时周期；确认运行后，可在已确认的 Slurm 时限处设置保守提醒。提醒只唤回当前 session，不直接提交或取回实验。

一次性时限提醒使用 `session_deadline_reminder.py`，由独立的 `RunAtLoad=true`、无重复间隔、无 `KeepAlive` 的 launchd 任务启动。进程只做一次有界睡眠（最多一小时），到点重新核对 active campaign、同一作业的运行状态及相同提醒时间；已结束、更换作业、暂停、STOP 或过期均不发送。发送复用原会话 guard、共享锁和持久化 claim；不打断活跃 turn，不重试不确定发送，也不改变半小时任务的下一次调度时间。睡眠/系统负载可能导致延后，不能保证秒级唤醒。运行结束后该 launchd 任务不会自行重复；可按检查点中的 label 卸载，不能删除 PAUSE 或发送 claim。

固定目标为 `01a1b49f-9449-7941-9407-da3901aa53a2`，工作目录为 `/Users/mi/Desktop/MyProjects/ParaCloud`。自动消息要求继续前检查现有工具/作业状态；补跑仍须先分析失败并修复，而非原包盲重跑。

## 检测与续接

1. 读取固定 rollout 的最新 turn 生命周期，验证会话 ID 和工作目录。异常与正常完成分别设置启用时间；历史完成/失败后已有新 turn 时不触发。半行写入、未识别状态、用户消息待处理都暂停发送。campaign 缺失、非 active 或证据门槛格式无效时不发送。
2. 通过现有 VS Code 扩展的本地 Unix socket 订阅只读快照，重新发现 owner，不硬编码进程或 owner ID。确认同一 thread、同一目录、已 resumed、idle 且无审批/输入请求。
3. 再读一次 rollout 和停止开关，尽量消除用户手动继续与后台发送之间的竞争。只向该 owner 调用 `thread-follower-start-turn`；使用当前线程原有设置，不另起 app-server、不改模型/权限、不代批审批。
4. 每条消息附带触发类型、结束/检查时间、上轮 turn ID、上轮结论/错误摘要、检查点更新时间、当前阶段、已知作业、已分析结果、证据、约束和下一步。作业状态注明是快照，要求当前 session 先查实时状态，不能直接重复提交。
5. 发送前持久化 claim、消息 ID、触发 turn、检查点和消息 SHA。每个完成/失败 turn 最多一次不确定/已接受的发送；收到明确 `no-client-found` 才允许下个周期重试。新 turn 可以继续恢复，没有总次数限制；全部证据门槛通过且 campaign complete 后停止。
6. 发送超时或断连时不能证明请求是否被接收，因此标记 `uncertain`，不盲重发；后续从 rollout 中的消息标记确认送达。若始终不能确认，保留记录供人工检查。新 turn 活跃时不会追加消息。

检测对象是有明确终态的 turn，不以日志不更新、长时间等待工具或 GPU 作业变慢推断“挂死”。若 UI 在创建 turn 之前失败且没有写入 rollout，或 IDE owner 已关闭/需要手动恢复，则记录或保持等待，不能保证自动恢复。

接口来自本机安装扩展 `26.715.61943` 的实现，不是稳定的公共 API。固定初始化/快照/转发协议版本，并验证扩展关键源码 SHA；版本或协议变化时停止发送，不能猜测新格式。没有为测试而中断真实会话或人工制造计费 turn：真正的转发经过 mock IPC 端到端测试，真实 owner 仅做只读探测。

## 后台部署

- 源码：`experiments/xplace/tools/session_watchdog.py`。
- 配置与部署记录：仓库根目录的 `evidence/session-watchdog-20261108/`。
- **活动运行目录**：`~/Library/Application Support/ParaCloud/session-watchdog/`。源码、配置、plist 部署副本及所有活动日志都在这里。
- launchd label：`local.paracloud.codex-session-watchdog`；当前 `StartInterval=1800`，`RunAtLoad=true`。活动源码、配置和 plist 必须一起部署；状态中的下次检查时间使用配置间隔。每轮进程退出后等待下次定时启动，所以 `state = not running` 配合 `last exit code = 0` 是正常状态。
- 初次从 Desktop 仓库直接运行被 macOS 后台隐私限制拒绝，错误已保存在 evidence。修复是迁出运行文件和日志，不是修改系统隐私授权；后台不需要读取实验代码或写入 Desktop。
- 当前以 `launchctl bootstrap` 加载到当前用户登录域，不依赖 Codex turn 的前台工具进程。退出登录或重启后需重新加载；电脑休眠期间不能承诺准点探测。没有改动 root 所有的 `~/Library/LaunchAgents`，也没有启用 keep-awake。

查看状态与日志：

```sh
runtime="$HOME/Library/Application Support/ParaCloud/session-watchdog"
/usr/bin/python3 "$runtime/session_watchdog.py" status --config "$runtime/config.json"
tail -5 "$runtime/checks.jsonl"
launchctl print "gui/$(id -u)/local.paracloud.codex-session-watchdog"
```

停止（立即关闭发送开关，再卸载该后台任务，不影响 GPU 作业）：

```sh
runtime="$HOME/Library/Application Support/ParaCloud/session-watchdog"
/usr/bin/python3 "$runtime/session_watchdog.py" stop --config "$runtime/config.json"
launchctl bootout "gui/$(id -u)/local.paracloud.codex-session-watchdog"
```

退出登录/重启后重新加载未手动停止的监控：

```sh
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/Application Support/ParaCloud/session-watchdog/local.paracloud.codex-session-watchdog.plist"
```

此命令不删除 `STOP`，更不删除任何 `PAUSE`。活动停止开关在 Application Support 目录，仓库 evidence 是备份，不是活动状态。

## 验证

初版 31 项专项 / 55 项合并回归通过。正常完成扩展后，42 项监控测试、3 项上传测试和 24 项旧 followup 纯回归测试共 69 项通过；没有启动旧 followup。新增覆盖正常完成附带信息、正常/异常实际 Unix socket 转发、campaign 暂停竞争、历史完成隔离、无上下文不发送、未完成门槛不能退出以及拒绝陈旧检查点。pytest 的 `asyncio_mode` 配置警告为既存环境问题。

首次成功后台检查：2026-11-08 22:35:08 CST；第二次为 22:40:08 CST，实测间隔 300.504 秒。两次都发现当前 turn 活跃，未发送任何续接消息。后台只读 IPC 探测也通过，确认能访问原有 owner。周期、部署 SHA 与只读探测结果保存在仓库 `evidence/session-watchdog-20261108/acceptance.json` 及同目录备份。

## 检查点更新

当前实验协议：`experiments/xplace/docs/ep4-tp2-ablation-20261108.md`。仓库检查点源：`evidence/ep4-tp2-ablation-20261108/campaign.json`。每轮必须写入结果/失败诊断/修复、新 job ID、实时状态观察时间、下一步和证据，然后用 `publish` 原子发布到活动目录；监控不会读取受 macOS 后台隐私限制的 Desktop 文件。

```sh
/usr/bin/python3 kvcached/experiments/xplace/tools/session_watchdog.py publish \
  --config evidence/session-watchdog-20261108/config.json \
  --context evidence/ep4-tp2-ablation-20261108/campaign.json
/usr/bin/python3 kvcached/experiments/xplace/tools/session_watchdog.py preview \
  --config evidence/session-watchdog-20261108/config.json
```

`preview` 只展示消息，不向会话发送。必须在 TP2 消融/实测代价门槛通过后才进入混合拓扑；所有门槛附证据且通过，才接受 campaign.status=complete。用户仍可随时用 STOP 或 campaign.status=paused 停止。
