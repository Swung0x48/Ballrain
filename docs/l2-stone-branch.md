# L2 小节 4：按球型选择参考线

当前仓库这份描述文件配对的是 `benchmarks/refpaths-v2/L02.npz`。在 Python 训练端指定 `--path-dir benchmarks/refpaths-v2 --branch-dir benchmarks/refpaths-alt/L02-stone`；不要把石线传给 `--path-dir`，否则纸球也会走南环。未配置时不创建分线投影器。当前池协议只分发一条线，拒绝导出双线策略，以免原生客户端悄悄退回 TAS。

`L02.json` 把整关的两份参考线按 SHA256 配对。`main_s` / `branch_s` 各为分岔和并线处的累计弧长，`endpoints` 为共同球心。实际切换取决于**当前可见球型、小节和支线投影区间**，而不是 TAS 帧。几何分岔在石变上游，留出了从路口正常入轨的空间；回纸即选主线，无须等到几何并线点。重生、前缀交接和新进程均重置投影器，不继承上条命的最高进度。

- **空间量**：`local_s` 是活动参考线上的实长；横向距离、切线、切向速度、近远路标、剩余距离、参考速度均在该线上求值。
- **课程量**：把 `local_s` 按 JSON 的对应站点分段线性换算成主线米数；`stations` 的列依次是主线、石线。`path.s` 是本命已挣得的单调主线进度。换线这一拍不付进度，只吸收两次投影的差。此后仅在超过本命历史最高站点时付进度，从换线处的已付值线性追回到下一个共同站点。后退再前进不重付。`path_progress` 用主线总长归一化；奖励、门限使用此 `s`。两座变球器之间只保留首尾站点：179.955 主线米的预算均摊到 707.528 支线米，每个新米得到 0.25434 主线米的进度，不再把大部分奖励提前付在南环之前。整段奖励和纸线相同；不可能既对每个新增米按主线原价付费，又不增加长路总额。反向课程的 frontier 仍是 **TAS 有效帧**，不是支线米数或支线时间。
- **停滞**：支线用 `local_s` 的新高推进计时标记；切线时换表，不跨表减米数。启用速度豁免时，支线阈值取配置与 1 m/s 的较小值；硬超时仍限制没有新进度的运动。主线保留原阈值。
- **时钟**：`ref_elapsed_ms` 从本命起点累计，仅积分新走过区间的活动线 `t_ms`；`ref_remaining_ms` 从最高站点沿当前线估算到飞船。换线重算分段参考和时间预算，不改游戏的 SR。未观测到球的变球动画不推进投影。
- **石球静止起点**：`--rest-starts 14730-14730@branch:1700 --p-rest 1`（`branch:` 后是整条石线的米数）。`drive.npz` 来自 `probe_l2_stone.py` 成功一命的按键回放，不是状态编辑。严格按同一 TAS 前缀回放至对应路标，核对石球和小节，再闭环驱动到目标、制动；不受支撑、速度或位置不符则关闭这次进程，以相同 TAS 帧重新正常开局；`start.branch_rest_failures` 记录本环境累计失败数，`branch_rest_error` 记录原因。区域不可停时不把失败当成课程起点。普通 `lo-hi@x:z:jitter` 语义不变。

复现：用项目 Python 运行 `scripts/build_l2_stone.py` 生成资产与两幅图；`scripts/probe_l2_stone.py <runtime-dir> x86p2002d 14730` 在独立目录生成控制日志，成功才写 `drive.npz`；将该文件与同版本的分线配对。`scripts/probe_stone_rest.py <runtime-dir> <build> <copy> <branch-dir> <main-dir>` 测试真实石球静止交接。`scripts/summarize_stone_probe.py` 汇总过盘和速度，失败尝试不参与最快用时计算。

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

## `L<nn>.json.rules`：原生客户端读取约定（v1）

`rules` 是封闭的 JSON 对象；整项缺省等于 `{}`，缺键使用下表默认值。先验包和主线配对，再校验所有键和值，错误须拒绝任务，不得静默回退。数值须有限、非负，不能用布尔代替；帧数为 `1..2147483647` 的整数。只有标为 `null` 的三项可显式置空，表示继承**本任务**的环境配置，不是用常量覆盖它。规则只用于分线；普通 TAS 静止起点及主线停滞判定不变。

|键|默认值|单位、取值和作用|
|---|---|---|
|`stall_pause_speed`|`1.0`|m/s；启用速度豁免时，有效阈值为本值与 worker `stall_speed` 的较小值，三维速率严格大于它才暂停。硬超时不变。|
|`stall_speed_source`|`"speed+speed_change"`|先作向量和再取范数；另一可选值 `"speed"` 只取原速度。|
|`stall_odometer`|`"local_s"`|停滞判定取活动线物理米数；`"s"` 取主线进度信用。换线时重新设基准，不跨线求差。|
|`rest_stop_speed`|`null`|正数，m/s；继承 `env.rest_speed`（默认 0.5）。驱动与制动以水平速率 `<` 判停，最终交接允许 `<=`。|
|`rest_max_drive_frames` / `rest_max_brake_frames`|各 `null`|追踪、制动各自的游戏帧预算；分别继承 `env.rest_max_frames`（默认 600）。每轮发指令前检查，制动结束另发一帧松键。|
|`rest_control_repeat`|`2`|追踪、制动每次控制指令的重复帧数。|
|`drive_arrival_radius`|`1.0`|水平目标距离严格小于此值（米）且足够慢，结束追踪。|
|`drive_gain` / `drive_speed_cap`|`0.8` / `6.0`|水平目标速度 = 水平位移乘增益（每秒），按范数限速（m/s）。|
|`drive_deadband`|`0.3`|目标与当前水平速度之差的范数严格大于此值（m/s）才按方向键。|
|`rest_require_support`|`true`|布尔；交接时要求现有接触特征 `supported` 为真。`false` 仅解除支撑要求，不解除分线匹配。|
|`rest_position_tolerance`|`2.0`|最终交接允许的最大三维欧氏位置误差（米，含端点）。|
|`rest_fallback`|`"same_tas_frame"`|驱动或交接失败时关闭失败进程，在同一 TAS 帧重开普通起点一次；`"error"` 则关闭并报错，不重开。|

没有单独的“0.25434”参数。`stations` 的每行是 `[main_s, branch_s]`；相邻站间以 `Δmain_s / Δbranch_s` 换算，换线不发放投影跳变的信用。改奖励密度应改站点，而非在客户端存近似斜率。球种、小节、分线起止、轨迹和时间均取自资产；有效观测才推进投影、物理前视走活动线、主线和回放哈希匹配、所需球种和小节匹配均为算法或完整性约束，不是 L2 调参项。回放每 512 帧分块仅为传输，不是驱动帧预算。

桥接 hello 保留 `branch_assets_version: 1`，另宣告 `branch_rules_version: 1`。只有规则解析结果与旧行为**不等价**时才要求这个能力；默认规则包仍可交给已支持分线资产的旧客户端。原生调度端也应在派工前按此检查。新增规则名或枚举值仍须更新读取器；修改支持范围内的值只需重新打包。摘要仍按原始字节计算，JSON 固定 UTF-8/LF；`drive.npz` 中的 `branch_sha256` 固定的是轨迹 NPZ，非描述 JSON。
