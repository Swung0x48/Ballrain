# L2 小节 4：按球型选择参考线

当前仓库这份描述文件配对的是 `benchmarks/refpaths-v2/L02.npz`。在 Python 训练端指定 `--path-dir benchmarks/refpaths-v2 --branch-dir benchmarks/refpaths-alt/L02-stone`；不要把石线传给 `--path-dir`，否则纸球也会走南环。未配置时不创建分线投影器。当前池协议只分发一条线，拒绝导出双线策略，以免原生客户端悄悄退回 TAS。

`L02.json` 把整关的两份参考线按 SHA256 配对。`main_s` / `branch_s` 各为分岔和并线处的累计弧长，`endpoints` 为共同球心。实际切换取决于**当前可见球型、小节和支线投影区间**，而不是 TAS 帧。几何分岔在石变上游，留出了从路口正常入轨的空间；回纸即选主线，无须等到几何并线点。重生、前缀交接和新进程均重置投影器，不继承上条命的最高进度。

- **空间量**：`local_s` 是活动参考线上的实长；横向距离、切线、切向速度、近远路标、剩余距离、参考速度均在该线上求值。
- **课程量**：把 `local_s` 按 JSON 的对应站点分段线性换算成主线米数；`stations` 的列依次是主线、石线。`path.s` 是本命已挣得的单调主线进度。换线这一拍不付进度，只吸收两次投影的差。此后仅在超过本命历史最高站点时付进度，从换线处的已付值线性追回到下一个共同站点。后退再前进不重付。`path_progress` 用主线总长归一化；奖励、门限使用此 `s`。两座变球器之间只保留首尾站点：179.955 主线米的预算均摊到 707.528 支线米，每个新米得到 0.25434 主线米的进度，不再把大部分奖励提前付在南环之前。整段奖励和纸线相同；不可能既对每个新增米按主线原价付费，又不增加长路总额。反向课程的 frontier 仍是 **TAS 有效帧**，不是支线米数或支线时间。
- **停滞**：支线用 `local_s` 的新高推进计时标记；切线时换表，不跨表减米数。启用速度豁免时，支线阈值取配置与 1 m/s 的较小值；硬超时仍限制没有新进度的运动。主线保留原阈值。
- **时钟**：`ref_elapsed_ms` 从本命起点累计，仅积分新走过区间的活动线 `t_ms`；`ref_remaining_ms` 从最高站点沿当前线估算到飞船。换线重算分段参考和时间预算，不改游戏的 SR。未观测到球的变球动画不推进投影。
- **石球静止起点**：`--rest-starts 14730-14730@branch:1700 --p-rest 1`（`branch:` 后是整条石线的米数）。`drive.npz` 来自 `probe_l2_stone.py` 成功一命的按键回放，不是状态编辑。严格按同一 TAS 前缀回放至对应路标，核对石球和小节，再闭环驱动到目标、制动；不受支撑、速度或位置不符则关闭这次进程，以相同 TAS 帧重新正常开局；`start.branch_rest_failures` 记录本环境累计失败数，`branch_rest_error` 记录原因。区域不可停时不把失败当成课程起点。普通 `lo-hi@x:z:jitter` 语义不变。

复现：用项目 Python 运行 `scripts/build_l2_stone.py` 生成资产与两幅图；`scripts/probe_l2_stone.py D:/repos/Ballrain/runtime x86p2002d 14730` 在独立目录生成控制日志，成功才写 `drive.npz`；将该文件与同版本的分线配对。`scripts/probe_stone_rest.py D:/repos/Ballrain/runtime <build> <copy> <branch-dir> <main-dir>` 测试真实石球静止交接。`scripts/summarize_stone_probe.py` 汇总过盘和速度，失败尝试不参与最快用时计算。

## Learner 下发协议（v1）

`--branch-dir` 只在 learner 启动时读取。训练时包在 `WorkerSpec.env.branch_assets`（远端模板为 `template["env"].branch_assets`），保存时在 `state["branch_assets"]`。没有分线时为 `None`（旧 checkpoint 可没有此键）。有分线时字典如下：

| 键 | 类型和内容 |
|---|---|
| `level` | 整数，关号 |
| `line_npz` | bytes，原样的 `L<nn>.npz` 分线 |
| `desc_json` | bytes，原样的 `L<nn>.json` 配对与站点描述 |
| `drive_npz` | bytes，原样的 `drive.npz` 回放 |
| `sha256` | 字典，以上三个 bytes 键各对应一个小写十六进制 SHA-256 |

`env_config.features.branch_sha256` 是按 `line_npz`、`desc_json`、`drive_npz` 顺序拼接三个 **32 字节原始摘要** 后再取的 SHA-256（小写十六进制）。worker 先验关号、各摘要、总指纹及主线配对，再写入带总指纹的私有临时目录；投影和静止交接读取同一份安装件。`--resume` 默认用 checkpoint 原件，显式指定目录时要求逐字节相同，否则拒绝；老式仅存目录的分线 checkpoint 必须在 learner 显式补给目录。

Python bridge 在 hello 声明 `branch_assets_version: 1`，否则分线任务在发送模板前拒绝。pool/native 接入方须从 `state["branch_assets"]` 打包三份原件，将摘要绑定到策略与任务，验包后才启用分线；任务协议须排除不支持分线的老客户端，不能只加 hello 标记。完成 gateway、`weights.export_state` 和 native 端支持前保留拒绝导出和连接的行为。**主线不随本包发送**：Python worker 仍需提供 `path_dir` 中 `path_sha256` 对应的主线，pool 已有独立的主线安装协议；本次不改变两者的寻址方式。
