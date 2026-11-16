# Agent 模式 IPC 契约（v1）

BallrainIO 的 agent 模式。Python 逐步决定每一帧的 delta 和按键，mod 把每帧的原始状态交给 Python。
状态的含义和读法见 `state-spec.md`。回放模式（M0）不变：agent 模式复用同一套帧注入路径，只是 `GetFrame(index)` 的来源从文件换成 Python。

## 1. 传输方式：共享内存 + 事件

每个游戏实例用一块命名共享内存（文件映射）和两个自动复位的命名事件：

- `<name>.obs`：mod → Python，表示"观测已写好"；
- `<name>.act`：Python → mod，表示"动作已写好"。

采用严格锁步：任一时刻只有一方在写。

选它而不选命名管道，理由如下：

- **延迟**：一次往返只是两次 `SetEvent` 和 `WaitForSingleObject`。命名管道每条消息要经过内核拷贝和帧化，估计约 20–50 µs（管道未实测）。
  **共享内存已实测**（BallrainIO 0.3.0，SR06_1_36_163_132，13205 帧，见 §8.6）：回放 1754 ms；agent 模式 K=1、每帧一条动作、全部段开启时 2568 ms，
  即每帧多约 62 µs，这已包含 mod 读全部状态、写观测、往返和 Python 存根的解析。
  K=1 逐帧决策时，管道可能多出 20–50% 的开销，而共享内存的开销可以忽略。
- **零拷贝**：Python 直接把 `mmap` 包成 numpy 视图来解析定长字段，不需要额外依赖。用到的只有标准库 `mmap`（带 `tagname`）和 `ctypes` 的 `CreateEventW` / `WaitForMultipleObjects`。
- **进程失联检测**：Python 用 `WaitForMultipleObjects` 同时等 obs 事件和游戏进程句柄；mod 等 act 事件时带超时，并检查父进程句柄。任何一方死掉，另一方都能在超时内退出。
- 每关一次的大块静态消息（几何有几 MB）不走共享内存，而是写成文件（§3）。这样共享内存可以定长，也便于离线调试。

命名：`Local\ballrain-<copy>-<pid>-<nonce>`（不含用户信息）。由 Python 创建映射和事件，mod 只打开。8 个实例各用各的名字。

## 2. 共享内存布局（小端）

| 偏移 | 类型 | 字段 |
|---|---|---|
| 0x00 | u32 | magic `'BRSM'`（0x4D535242） |
| 0x04 | u32 | 协议版本 = 1 |
| 0x08 | u32 | 映射总大小（默认 4 MiB） |
| 0x0C | u32 | obs 区偏移（0x1000） |
| 0x10 | u32 | obs 区容量（默认 3 MiB） |
| 0x14 | u32 | act 区偏移 |
| 0x18 | u32 | act 区容量（默认 512 KiB，足够放 6 万帧动作） |
| 0x1C | u32 | 保留 |
| 0x20 | u64 | `mod_seq`：mod 每写完一条 obs 加 1 |
| 0x28 | u64 | `py_seq`：Python 每写完一条 act 加 1 |
| 0x30 | u32 | `mod_state`：0 init，1 hello，2 menu，3 loading，4 running，5 waiting，6 ended，7 error |
| 0x34 | u32 | `end_reason`：0 无，1 LevelFinish，2 GameOver，3 帧数上限，4 Python STOP，5 ResetLevel/ExitLevel，6 超时，7 错误 |
| 0x38 | u32 | mod 的 pid |
| 0x3C | u32 | Python 的 pid |
| 0x40 | u32 | `frames_done`：已注入帧数 |
| 0x44 | u32 | `last_error`（错误码，正文写进 ModLoader.log） |
| 0x48 | u32 | 本进程支持的 obs 版本位图（bit v = 支持 v） |
| 0x4C | char[180] | mod 版本串和 physics_RT sha256 前 16 个 hex（mod 在 hello 时填写，格式 `BallrainIO 0.3.0 physics_RT e72e4afcfa5c33a7`） |

写入顺序：先写载荷，再写 seq（带写屏障），最后 `SetEvent`。读方等到事件后，先确认 seq 递增了 1，再读载荷。

### 2.1 观测消息（obs 区）

```
u32 magic 'OBS1'   u16 version=1   u16 flags     u32 byte_len   u32 frames_done
u32 section_count  u32 reserved
section × section_count:
  u32 tag(fourcc)  u16 sec_version  u16 sec_flags  u32 payload_len  payload (4 字节对齐)
```

`frames_done = n` 表示：本消息描述的是第 0..n−1 帧都执行完之后，在第 n−1 帧 `OnProcess` 时读到的状态。下一条动作从第 n 帧开始。这与 trace 帧号是同一口径。
Python 遇到不认识的 tag 必须跳过，这样新版 mod 增加段时旧 Python 不受影响。
段内字段只追加、不改义；要改义时提升 sec_version。

v1 的段如下（`f32` / `f64` 原样拷贝内存，不做换算）：

| tag | 内容（按顺序） | 频率 |
|---|---|---|
| `TIME` | f64 env current_time、time_of_next_psi、time_of_last_psi；f32 管理器 +0xC8、+0xCC、+0xD0；f32 本帧 delta；f64 SR（mod 按 BML+ 0.3.12 规则累加的 double 和）；f32 `GetSRScore()`；u8 SR 是否在计时；u32 本条覆盖的帧数 | 每条 |
| `BALL` | u32 ckid；u8 类型（0 木，1 纸，2 石，254 其它，255 无）；u8 是否找到 IVP 物体；CK 世界矩阵 12×f32；core：f64×3 pos_last_psi、f64×4 q_last、f64×4 q_next、f64 tpsi、f32×3 speed、f32×3 speed_change、f32×3 rot_speed、f32×3 rot_speed_change、f32×3 rotation_axis、f32 current_speed、f32 abs_omega、u32 core 位域 +0x00、u32 core +0x60；f32 extra_radius | 每条 |
| `FRIC` | u16 n；每项：u32 对方 ckid、u8 自身 status、u8 对方 status、u8 cp_status、u8 pad、f32 last_gap_len、f32 real_friction_factor、f32×3 lt.surf_normal、f64×3 lt.contact_point_ws、u8 lt 是否有效（mod 只检查可读）| 每条 |
| `MIND` | u16 n；每项：u32 对方 ckid、u8 自身 status、u8 对方 status、u16 pad、f32 len_numerator、f32×4 contact_plane（含 hesse）、f32 sum_extra_radius、u32 ledge 标识、u16 tri_index、i8 edge_index、u8 有几个三角形、f32×9 特征三角形（物体坐标）、f32×9 相邻三角形（仅边接触时） | 每条 |
| `BODY` | u16 n；附近运动体（半径由配置给，默认 60）：u32 ckid、u32 core +0x60、f64×3 pos、f64×4 q、f64×4 q_core_f_object、f32×3 shift_core_f_object、f32×3 speed、f32×3 rot_speed、CK 世界矩阵 12×f32 | 每条 |
| `CAMS` | 5 × {u32 ckid，12×f32 世界矩阵}：Cam_OrientRef、Cam_Orient、Cam_Pos、Cam_Target、InGameCam；u32 当前挂接相机的 ckid | 每条 |
| `GLCT` | 当前在用的 PhysicsContinuousContact：u32 BB ckid、u16 输出脉冲位图、3 × {i32 active，i32 number} | 每条 |
| `ARRS` | 自上条消息以来变化的数组单元：u16 n；每项 u32 数组 ckid、u16 行、u16 列、u8 列类型、u8 len、bytes（int/float 4 字节；字符串原文；对象写 ckid；参数写原始数据，最多 64 字节）。监视的数组：Energy、CurrentLevel、IngameParameter、PH、Checkpoints、ResetPoints（第一条消息给全量） | 变化时 |
| `EVTS` | u16 n；每项 u32 帧号、u16 事件码（§2.3）、u16 参数（例如物理化时的序号） | 有事件时 |
| `WTCH` | 被监视对象（分数球、生命球、trafo 等，清单见关卡消息）可见性变化：u16 n × {u32 ckid，u8 可见，f32×3 位置} | 变化时 |
| `PHYS` | 本条覆盖范围内新增的物理化记录序号。记录正文和网格追加写到关卡消息的增量文件（§3） | 有时 |
| `TRAC` | 可选：本条覆盖的每一帧各一条 {u32 帧号，球 CK 世界矩阵 12×f32}，供奖励积分和 trace 用 | 按动作 flags bit1 |
| `SCOR` | HS 读数、命数簿记、跟随中的分数小球数（§8.2） | 每条（非最小观测） |
| `SREN` | SR 终值、尾帧数、LevelFinish 时的 SR 和 HS 快照、跟随中的小球数（§8.2） | LevelFinish 之后的观测和最后一条 |

单条观测一般 1–3 KB（10 个 mindist 约 1.3 KB，5 个接触约 0.3 KB，相机 0.3 KB，运动体每个约 130 B）。

### 2.2 动作消息（act 区）

```
u32 magic 'ACT1'  u16 version=1  u16 flags   u32 py_seq
u32 n_frames      u32 repeat      u32 stop_event_mask(低 32 位) u32 stop_event_mask_hi
u32 obs_mask      u32 reserved
frame × n_frames: f32 delta_ms, u32 keys        // 正好是 legacy .tas 的 8 字节帧
```

- `keys` 是 9 个逻辑位：bit0 上、1 下、2 左、3 右、4 shift、5 space、6 Q、7 Esc、8 Enter。
  mod 按 `Keyboard` 数组的键位映射写原始 0/1，与回放完全相同。
- 每个列出的帧重复 `repeat` 次（≥1），action repeat 就在 mod 里完成。也可以直接列出 K 帧。两种写法都会让每一帧原样进入帧日志。
- `stop_event_mask`：执行过程中只要发生掩码里的事件（例如 BallOff、BallNavActive、LevelFinish），就在**该帧**执行完后提前结束，并返回观测。没执行的帧作废，`frames_done` 反映实际执行到哪里。
  用这个机制可以快进死亡和复活动画，而且快进时的按键仍由 Python 明确给出。
- `flags`：bit0 STOP（结束本进程的 run：写出输出文件，按 `BALLRAIN_EXIT` 决定是否退出游戏），bit1 请求本条返回 `TRAC`，bit2 请求重写关卡消息。
- `obs_mask`：可以关掉不需要的可选段以减少开销。**不能用来改段的语义**。它是"关闭"掩码，0 = 全开：
  bit0 BODY，bit1 MIND，bit2 FRIC，bit3 CAMS，bit4 GLCT，bit5 WTCH，bit6 BALL，bit7 ARRS。TIME、EVTS、PHYS、SCOR 总是有；TRAC 由 flags bit1 打开。
  ARRS / WTCH 关闭期间不更新"上次值"，重新打开后的第一条给出自上次发出以来的全部变化。

### 2.3 事件码

| 码 | 事件 | 码 | 事件 |
|---|---|---|---|
| 1 | PreLoadLevel | 16 | PreCheckpoint |
| 2 | PostLoadLevel | 17 | PostCheckpoint |
| 3 | StartLevel | 18 | ExtraPoint |
| 4 | CamNavActive | 19 | PreLifeUp |
| 5 | CamNavInactive | 20 | PostLifeUp |
| 6 | BallNavActive | 21 | LevelFinish |
| 7 | BallNavInactive | 22 | GameOver |
| 8 | CounterActive | 23 | PreEndLevel |
| 9 | CounterInactive | 24 | PostEndLevel |
| 10 | PauseLevel | 25 | Physicalize |
| 11 | UnpauseLevel | 26 | Unphysicalize |
| 12 | BallOff | 27 | ActiveBall 改变 |
| 13 | Dead | 28 | PreResetLevel |
| 14 | PreSubLife | 29 | PreExitLevel |
| 15 | PostSubLife | 30 | RunEnd（附 end_reason） |

## 3. 关卡消息（文件）

路径由 `BALLRAIN_LEVELMSG` 给出。StartLevel 时写 `<path>`（全量）；之后每次新的物理化记录追加到 `<path>.inc`。
两者都用同一种段格式，magic 为 `'LVL1'`。Python 在收到带 StartLevel 事件的观测后读取全量文件，在收到 `PHYS` 段后读增量。

| tag | 内容 |
|---|---|
| `STRS` | 字符串表（实体名、网格名、组名、数组名），按下标引用 |
| `ENTS` | 全部 3D 实体（派生类也算）：ckid、名字、类型 id、父 ckid、可见、世界矩阵，以及 `Continuous Contact ID`、`Coll Detection ID`、`Floor`（原始字节）三个属性。**只按属性名取** |
| `GRPS` | 全部组及成员 ckid |
| `ARRS` | 全部数据数组的列名、列类型和全部单元（规则同 §2.1） |
| `PHYS` | 物理化记录：序号、帧号、ckid、fixed、friction、elasticity、mass、碰撞组、frozen、coll、calcmc、两个阻尼、碰撞表面名、质心偏移、凸网格 ckid 列表、球的 {中心，半径} 列表、凹网格 ckid 列表；以及 Unphysicalize 记录 |
| `MESH` | 每个（实体，网格）一次：ckid、网格 ckid、类型（凸/凹）、导出时的世界矩阵 16×f32、顶点 f32×3×nv（局部坐标）、三角形 u32×3×nf |
| `IVPO` | 已找到 IVP 物体的实体：ckid、object_type、unmoveable、extra_radius、材质原始 32 字节 |
| `DEPT` | DepthTestCubes 各成员的世界 AABB |
| `WLST` | 每帧被监视对象的清单：名字以 `All_P_Extra`、`All_P_Trafo`、`All_PC_`、`All_PS_`、`All_PE_`、`LifeBalls`、`Trafos`、`P_Extra`、`P_Trafo`、`PC_`、`PR_`、`PS_`、`PE_` 开头或含 Depth（不分大小写）的组中的 3D 实体（与探针相同；SR06 为 150 个） |
| `LTAB` | 本关解析出的表：5 个相机 ckid（顺序同 CAMS），PhysicsContinuousContact 块的 ckid 列表 |

## 4. 时间语义（与回放严格一致）

帧号口径：下标 N 是帧数组（.tas 载荷）的 0 起下标。trace、事件和 `frames_done` 用"已注入帧数"计数，所以下标 N 那一帧执行时发生的事件，事件号是 N+1（见 `state-spec.md` §1）。

1. 下标 N 帧的观测 = 该帧 `OnProcess` 时的状态（物理之后）。`frames_done` = N+1。
2. mod 在第 N 帧的 `OnProcess` 末尾写观测。如果当前动作块已经执行完或提前停止，就设置 obs 事件。
3. **mod 在第 N+1 帧的 Time PreProcess 后回调里阻塞，等待动作**。原因是 delta 也由 Python 决定，必须在 `SetLastDeltaTime` 之前拿到。
   拿到后，Time 回调注入 `delta`，同一帧的 Input PreProcess 后回调写入按键并推进帧号。这与回放中的一帧完全相同。
   阻塞点不能放在 Input：那时本帧的 delta 已经被游戏用掉了。
4. 默认 delta = f32(1000/132) = 7.5757575 ms，每帧恰好一个 PSI。delta 可以逐帧变化：legacy .tas 逐帧存 delta，任意 delta 都能复现。
5. run 从 PreLoadLevel 开始（与回放相同，第 0 帧是 PreLoadLevel 之后的第一个 Time 回调）。
   mod 在 PreLoadLevel 时发出 `frames_done = 0` 的首条观测（只有 TIME 和 EVTS），然后在第 0 帧的 Time 回调里等第一条动作。
   加载期间（约 100 帧，到 StartLevel 为止）也要逐帧给出 delta 和按键：它们会进入 .tas，并决定加载结束的时刻。
   Python 通常用一条 `stop_event_mask = StartLevel` 的动作把这段快进过去。
6. **按键屏蔽是 Python 的选项，不写死在 mod 里**。控制尚未激活时（CamNav / BallNav 之前、死亡动画、变球期间），按键照样会被注入。
   例如前奏里按 shift 会改变导航开始的时刻。所以 Python 必须显式决定这些帧的按键（默认全 0，可配置）。
7. 第 1 关的前奏需要一次脚本化的 Q（bit 6）来跳过教程。SR01 的做法是在下标 140 的帧按 Q，下标 141 按 shift+left；CamNavActive 的事件号为 142，即发生在下标 141 那一帧。
   这是 Python 侧的"前奏策略"（固定帧序列或 TAS 前缀），mod 不特殊处理。

## 5. 回合边界

- **前缀回放后交接**：设置 `BALLRAIN_PREFIX=<frames 文件>`（headerless 8 字节帧）后，mod 先按回放模式逐帧注入前缀，不和 Python 往返。
  可选 `BALLRAIN_PREFIX_EVENTS=1`，对前缀期间的每帧也写观测（obs flags bit1），**仍是锁步**：Python 回一条 `n_frames = 0` 的动作作为确认，mod 才执行下一个前缀帧
  （v1 实现的变更：原文"不等待"会让共享内存被覆写时读到一半，所以改为确认）。确认时若 `n_frames > 0`，就丢弃剩余前缀，立即交接。
  前缀用完后发出第一条普通观测，交给 Python。不设 `PREFIX_EVENTS` 时前缀期间不发观测，前缀期间的事件全部累积在交接时那条观测的 `EVTS` 里；也不发 `frames_done = 0` 的首条观测。
  前缀帧照常进入帧日志。
- **每条命一个 RL 回合**：BallOff 事件就是 RL 回合的边界。同一个进程、同一条连续的 .tas 里可以有多条命。
  BallOff 之后游戏自动复活：264 帧（264 fps）后命数减 1，686 帧后 PostSubLife，再过约 380 帧有 CounterActive / BallNavActive。
  这期间 Python 照样要提供帧，推荐用 `stop_event_mask = BallNavActive` 快进。
- **进程级结束**：
  - GameOver：与回放相同，停在下一个 Time 回调，本帧的观测仍会写出。
  - LevelFinish：停止注入，`end_reason = 1`。
  - 达到 `BALLRAIN_MAX_FRAMES`。
  - Python 发 STOP。
  - 等待超时或父进程消失。
  - ResetLevel / ExitLevel。

  每个 episode 用一个进程（同进程重载在完成关卡后不能逐字节复现）。
- **SR 的最终值**：回放在 LevelFinish 就停止，而 CounterInactive 要在其后才到。agent 模式在 LevelFinish 后继续自动注入帧（delta = 最后一帧的 delta，keys = 0），直到 CounterInactive，最多 `BALLRAIN_SR_TAIL`（默认 32）帧，然后在最后一条观测的 `TIME` 段报告最终 SR。
  **这些尾帧不写进帧日志**：可导出的 .tas 止于 LevelFinish 那一帧，与参考 TASSupport 的停止点一致。
  **尾帧数已实测为 3**：SR06_1_36_163_132（132 fps）和 HS03_4906_1_56_934（240 fps）都是 LevelFinish 在事件号 N，CounterInactive 在 N+3。
  因为发生 CounterInactive 的那一帧不再累加，最终 SR = LevelFinish 帧之后再加 2 帧 delta。SR06 的正式 SR 为 96159 ms（double 和 96159.09），float32 和为 96163.29（与文件名 1:36.163 吻合）；HS03 为 116958 ms，float32 和 116934.38（与文件名 1:56.934 吻合）。
  尾帧不进 trace，`frames_done` 不含尾帧（TIME 段的原始帧计数含尾帧）。
  最后一条观测就建在尾帧结束（CounterInactive）那一帧，所以它的 ARRS / SCOR 是正式 HS 读数用的值（`benchmarks.md` "HS 读数"）。
- **诊断用的 post-finish 窗口**（0.3.1）：`BALLRAIN_POST_FINISH=N` 时，SR 尾帧结束后再自动注入 N 个空闲帧（delta 同上，keys = 0），
  SR 不再累加，`SREN` 的尾帧数不变；这些帧同样不进帧日志和 trace。`BALLRAIN_POST_LOG` 给出逐帧文本日志（LevelFinish 钩子、尾帧、窗口内每帧的
  Energy 第 0 行、跟随中的小球数、全部数据数组的变化、分数球物体的可见性和到球的距离）。只用于测量（`tools/probe/post_finish.py`）；
  训练和门禁不设它。设了它时最后一条观测的数组已经被结算扣过，Python 改用 LevelFinish 钩子的快照（`SREN` 的 `post_frames` > 0 可判别）。
- **帧日志就是可导出的 .tas 载荷**：结束时 mod 写出 `BALLRAIN_FRAMELOG`，内容是全部已注入帧（含前缀），格式为 headerless 8 字节帧。
  Python 用 `ballrain.tas` 封装成 legacy .tas，再按 M0 规则验收：参考 TASSupport 回放 3 次，与多数结果逐字节相同即通过。

## 6. 启动 agent 模式

环境变量（`ballrain.instances` 管理的变量每次启动都会显式设置或清除）：

| 变量 | 值 | 说明 |
|---|---|---|
| `BALLRAIN_MODE` | `agent` | `replay` 是 M0 回放，其它值让 mod 保持不活动 |
| `BALLRAIN_LEVEL` | 1..13 | 主菜单后自动加载的关卡（同回放） |
| `BALLRAIN_SHM` | `Local\ballrain-…` | 共享内存名；事件名为其后加 `.obs` / `.act` |
| `BALLRAIN_PARENT_PID` | Python 的 pid | mod 用来检测 Python 是否存活 |
| `BALLRAIN_TIMEOUT_MS` | 默认 60000 | 等待动作的超时。超时后写出输出文件，`end_reason = 6`，退出 |
| `BALLRAIN_LEVELMSG` | 文件路径 | 关卡消息（§3） |
| `BALLRAIN_PREFIX` | 文件路径（可选） | 交接前回放的前缀帧 |
| `BALLRAIN_PREFIX_EVENTS` | `1`（可选） | 前缀期间也写观测，每帧等 Python 确认（§5） |
| `BALLRAIN_MAX_FRAMES` | 整数（可选） | 帧数上限 |
| `BALLRAIN_BODY_RADIUS` | 默认 60 | BODY 段包含的运动体距离 |
| `BALLRAIN_SR_TAIL` | 默认 32 | LevelFinish 后等 CounterInactive 的最多帧数 |
| `BALLRAIN_POST_FINISH` | 整数（可选，0.3.1，诊断） | SR 尾帧之后再跑的空闲帧数（§5） |
| `BALLRAIN_POST_LOG` | 文件路径（可选，0.3.1，诊断） | post-finish 逐帧文本日志（§5） |
| `BALLRAIN_FRAMELOG` | 文件路径 | 可导出的帧日志（同回放） |
| `BALLRAIN_TRACE` | 文件路径（可选） | 每帧球 trace，格式与回放相同，用于和参考对照 |
| `BALLRAIN_EVENTS` | 文件路径（可选） | 文本事件日志（同回放） |
| `BALLRAIN_SKIP_RENDER` | `1` | 不渲染 |
| `BALLRAIN_EXIT` | `1` | run 结束后退出游戏（一进程一 episode） |
| `BALLRAIN_LOADS` | 不设或 `1` | agent 模式只支持单次加载 |

启动顺序：

1. Python 创建共享内存（写好头部，`mod_state = 0`）和两个事件。
2. 在 `runtime/<copy>/Bin` 下启动 Player.exe（只启动自己的拷贝；每个拷贝同时只跑一个进程；只停止自己启动、并且核对过映像路径的 pid）。
3. 等待 hello：轮询到 `mod_state ≥ 1`（不发事件），版本、DLL 哈希都正确。physics_RT.dll 在 OnLoad 时可能还没加载，所以 hello 可能推迟到主菜单（PostStartMenu）才出现，随后立即变为 2。
4. 收到 `frames_done = 0` 的首条观测后进入锁步。

mod 拒绝运行的情况：physics_RT.dll 的 sha256 不是 2002 版；共享内存的 magic 或版本不符；同时设置了 `BALLRAIN_FRAMES`（与回放互斥）。
这些情况都把 `mod_state` 置为 7，并在 ModLoader.log 里写明原因。

## 7. 一致性约束（不得违反）

- agent 模式下每帧的注入路径必须与回放逐行相同：Time 回调写 delta，Input 回调写按键并推进帧号，加上 `ResetPhysicsTime` 的定时器、BB 删除、键位读取。
  验收方法：同一组帧分别用 agent 模式和回放模式跑，trace 必须逐字节相同。
- 所有状态读取遵守 `state-spec.md` §0：纯内存读，`update=FALSE`，不遍历未知属性，不调用 IVP。
  新增读取必须先用探针在若干记录上证明 trace 不变。
- 观测只追加字段或段。特征、奖励、掩码、坐标系变换全部在 Python 里做。

## 8. v1 实现细节（BallrainIO 0.3.0，权威字节布局）

本节是 mod 实际写出的布局，也是 `ballrain/protocol.py` 解析的布局（逐条在真实游戏上用 `protocol.parse_observation` / `parse_level_message` / `parse_level_stream` 核对过，§8.6）。
上文只给了字段顺序的地方，以本节为准。

### 8.1 通用规则

- 小端，**紧凑排列**：字段之间、数组元素之间都没有填充。矩阵 "12×f32" 是 VxMatrix 第 0–3 行的前 3 列（第 3 行是位置），"16×f32" 是整个 4×4；四元数 x y z w。
- 消息头 24 字节：u32 magic，u16 version，u16 flags，u32 byte_len（含头的整条消息），u32 frames_done，u32 section_count，u32 reserved。
- 段头 12 字节：u32 tag，u16 sec_version，u16 sec_flags，u32 payload_len。payload_len **不含**填充，下一段从 align4 处开始。段内字段只追加：读方要接受比已知长度更长的段和列表项（新版可能在项尾追加字段）。
- 观测 flags：bit0 最后一条（run 已结束），bit1 前缀帧观测（`BALLRAIN_PREFIX_EVENTS`），bit2 有段因放不下被丢弃，bit3 最小观测（只有 TIME 和 EVTS）。
- `frames_done` = 帧日志里的帧数（可导出帧）。SR 尾帧不计入。
- mod 出错时：`mod_state = 7`，`end_reason = 7`，`last_error` 置码（2 环境变量，3 DLL/钩子，4 动作非法，1 共享内存），然后 SetEvent(obs) 但**不**推进 `mod_seq`；若 run 已开始，随后还会照常写出输出文件并发出一条 final 观测。
- 动作：`n_frames = 0` 且无 STOP 时，mod 不执行帧，原样再发一次上一条观测（新的 `mod_seq`）；带 `BALLRAIN_PREFIX_EVENTS` 的前缀期间它是"确认，继续下一帧前缀"。STOP 时忽略本条的帧。
  `stop_event_mask` 只对本块已执行的帧里发生的事件生效（前缀期间不生效）；事件码 c 对应位 c。

### 8.2 每帧段（OBS1）

| 段 | 长度 | 布局（偏移: 类型 字段） |
|---|---|---|
| TIME | 69 | 0 f64 env current_time，8 f64 time_of_next_psi，16 f64 time_of_last_psi，24 f32 管理器 +0xC8，28 f32 +0xCC，32 f32 +0xD0，36 f32 本帧 GetLastDeltaTime，40 f64 SR（double 和），48 f32 GetSRScore()，52 u8 SR 计时中，53 u32 本条覆盖帧数（构造状态时）；**追加**：57 f32 float32 和，61 u32 原始帧计数（含尾帧），65 u32 尾帧数 |
| BALL | 230 | 0 u32 ckid，4 u8 类型（0 木 1 纸 2 石 254 其它 255 无），5 u8 找到 IVP 物体（见 §8.7），6 12×f32 CK 世界矩阵，54 f64×3 pos_last_psi，78 f64×4 q_last，110 f64×4 q_next，142 f64 time_of_last_psi，150 f32×3 speed，162 speed_change，174 rot_speed，186 rot_speed_change，198 rotation_axis，210 f32 current_speed，214 f32 abs_omega，218 u32 core +0x00，222 u32 core +0x60，226 f32 extra_radius |
| FRIC | 2+53n | u16 n（≤32）；项：0 u32 对方 ckid，4 u8 自身 status，5 u8 对方 status，6 u8 cp_status，7 u8 0，8 f32 last_gap_len，12 f32 real_friction_factor，16 f32×3 lt 法向，28 f64×3 lt 接触点，52 u8 lt 可读 |
| MIND | 2+112n | u16 n（≤64）；项：0 u32 对方 ckid，4 u8 自身 status，5 u8 对方 status，6 u16 0，8 f32 len_numerator（间隙），12 f32×4 contact_plane（xyz 是法向；第 4 个 float 即 +0x74 不维护，是无意义的残值），28 f32 sum_extra_radius，32 u32 ledge 地址，36 u16 tri_index（无为 0xFFFF），38 i8 边号，39 u8 三角形个数（0/1/2），40 f32×9 特征三角形（物体坐标），76 f32×9 相邻三角形 |
| BODY | 2+180n | u16 n；项：0 u32 ckid，4 u32 core +0x60，8 f64×3 pos_last_psi，32 f64×4 q_last，64 f64×4 q_core_f_object，96 f32×3 shift_core_f_object，108 f32×3 speed，120 f32×3 rot_speed，132 12×f32 CK 世界矩阵 |
| CAMS | 264 | 5 × {u32 ckid，12×f32}（Cam_OrientRef、Cam_Orient、Cam_Pos、Cam_Target、InGameCam），u32 当前挂接相机 ckid |
| GLCT | 30 | u32 BB ckid（0 = 没有在用的块），u16 输出脉冲位图，i32×6 = {active, number} × 3（石、木、金属） |
| ARRS | 2+Σ | u16 n；项：u32 数组 ckid，u16 行，u16 列，u8 列类型（CK_ARRAYTYPE：1 int，2 float，3 string，4 object，5 parameter），u8 len，len 字节（int/float/object 4 字节；string 原文 ≤255；parameter 原始数据 ≤64） |
| WTCH | 2+17n | u16 n；项：u32 ckid，u8 可见，f32×3 位置。可见性**或位置**变化时发出 |
| SCOR | 56 | 0 u8 有效（0 无 Energy，1 Points/Lifes/LifeBonus 与 LevelBonus 均为 int，2 只有原始值），1 u8×6 Energy 第 0 行列类型，7 u8 AllLevel 第 6 列类型，8 u32×6 Energy 第 0 行原始值（Points、Lifes、StartPoints、StartLifes、Timefactor、LifeBonus），32 u32 LevelBonus（AllLevel[L−1,6]），36 i32 GetHSScore()，40 i32 HS = Points + Lifes×LifeBonus + LevelBonus，44 u32 BallOff 次数，48 u32 最近 BallOff 事件号，52 u32 最近 BallNavActive 事件号；**追加（0.3.1，长度 60）**：56 u16 跟随中的分数小球数（某个实例的 `P_Extra_Point_Ball0` 不可见、它的 `P_Extra_Point_Ball1..6` 仍可见的个数，`state-spec.md` §2.9），58 u16 还有小球在跟随的分数球个数 |
| EVTS | 2+8n | u16 n；项：u32 事件号，u16 事件码，u16 参数（Physicalize / Unphysicalize：记录序号；RunEnd：end_reason） |
| PHYS | 2+4n | u16 n；u32 记录序号（这些记录已追加到 `.inc`） |
| TRAC | 4+52n | u32 n；项：u32 事件号，12×f32 球 CK 世界矩阵。sec_flags bit0 = 缓冲区满、有帧丢失 |
| SREN | 80 | 0 f64 SR，8 f32 float32 和，12 f32 GetSRScore()，16 u32 尾帧数，20 u8 已 LevelFinish，21 u8 LevelFinish 后计时已暂停（CounterInactive/Pause），22 u8 仍在计时，23 u8 0，24 f64 LevelFinish 帧的 SR，32 f32 LevelFinish 帧的 float32 和，36 HS 快照（在 LevelFinish 钩子里读，布局同 SCOR 偏移 0–43，不含跟随中的小球）；**追加（0.3.1，长度 88）**：80 u16 LevelFinish 钩子时跟随中的小球数，82 u16 本条观测时（尾帧结束时）跟随中的小球数，84 u32 post-finish 窗口帧数（§5） |

obs_mask（置位 = 关闭）：bit0 BODY，bit1 MIND，bit2 FRIC，bit3 CAMS，bit4 GLCT，bit5 WTCH，bit6 BALL，bit7 ARRS（与 `ballrain/protocol.py` 的 `OBSMASK_*` 一致；§2.2 中的旧说法以此为准）。

### 8.3 关卡消息（LVL1）

flags：0 = 全量文件（StartLevel 时写，先写 `<path>.tmp` 再原子替换；动作 flags bit2 可要求重写），1 = 增量消息。
`<path>.inc` 从 run 开始就打开，**每一条**物理化记录（包括 StartLevel 之前的地面）都按顺序追加在里面，全量文件里也有一份快照。
每条消息有自己的 STRS；字符串用其下标引用，0xFFFFFFFF = 无。每个段都以 u32 n 开头。

| 段 | 项布局 |
|---|---|
| STRS | u16 len，bytes（ANSI 代码页） |
| ENTS | u32 ckid，u32 名字，u32 class id，u32 父 ckid，u8 可见，12×f32 世界矩阵，i32 Continuous Contact ID（−1 = 没有该属性；float 参数取整），i32 Coll Detection ID（同上），u16 len + Floor 属性原始字节 |
| EMSH | u32 ckid，u32 当前网格 ckid（扩展段） |
| GRPS | u32 ckid，u32 名字，u32 成员数，u32×成员 ckid |
| ARRS | u32 ckid，u32 名字，u16 行，u16 列；每列 {u32 列名，u8 类型，u32 参数类型名（仅 parameter 列，例如 "Matrix"）}；然后逐行逐列 {u8 len，bytes}（编码同 §8.2 ARRS） |
| PHYS | u32 序号，u32 事件号，u32 ckid，u8 是否 Unphysicalize，u8 fixed，f32 friction，f32 elasticity，f32 mass，u32 碰撞组名，u8 frozen，u8 coll，u8 calcmc，f32 线阻尼，f32 角阻尼，u32 碰撞表面名，f32×3 质心偏移，u16 n + u32×n 凸网格，u16 n + {f32×3 中心，f32 半径}×n 球，u16 n + u32×n 凹网格。Unphysicalize 记录的数值为 0、名字为无 |
| MESH | u32 实体 ckid，u32 网格 ckid，u8 类型（0 凸，1 凹），16×f32 导出时世界矩阵，u32 nv，f32×3×nv 局部顶点，u32 nf，u32×3×nf 三角形 |
| IVPO | u32 ckid，u32 object_type，u8 unmoveable，f32 extra_radius，32 字节材质原始数据。每个（实体，IVP 物体）只发一次；StartLevel 时还没有可走的 IVP 物体，所以通常出现在 `.inc` 里（球物理化之后），重新物理化会再发 |
| DEPT | u32 ckid，f32×3 最小角，f32×3 最大角（用网格顶点 × 世界矩阵算出） |
| WLST | u32 ckid，u32 所属组名（第一个匹配的组） |
| LTAB | u32×5 相机 ckid，u32 n + u32×n PhysicsContinuousContact 块 ckid（扩展段） |

### 8.4 events 文件（agent 模式）

格式与回放相同（`<事件号> <名字>`），但写全部事件码的名字（Physicalize、CounterActive 等），另有：
`SR sr=… f32sum=… running=0/1`（每次计时开始/暂停），`HS points=… lifes=… lifebonus=… levelbonus=… hs=… bml=… followers=…`（LevelFinish；`followers` 为 0.3.1 追加），
`Summary frames=… wall_ms=…`（可导出部分结束时，口径同回放），之后是尾帧里的事件，最后 `RunEnd` 和
`SRFinal sr=… sr_ms=… f32sum=… bml=… running=… tail=… inactive=… finished=… sr_at_finish=… end_reason=… post=…`（`post` 为 0.3.1 追加）。
回放模式的 events 文件逐行不变。

### 8.5 结束与输出

- 最后一条观测（flags bit0，`mod_state = 6`）发出时，帧日志、trace、events 和 `.inc` 都已写完并关闭；`BALLRAIN_EXIT=1` 时随后退出游戏。超时、父进程消失和错误无论如何都退出。
- GameOver 和帧数上限：最后一条观测的状态取自最后一帧的 OnProcess，在下一个 Time 回调里关闭输出后发出。ResetLevel / ExitLevel：在当帧 OnProcess 发出。

### 8.6 实测（2026-11-16，runtime/train-00，BallrainIO 0.3.0，最终构建 sha256 9a8f1390…）

存根客户端（会话 scratchpad，不入库）把 M0 的 BallrainIO 帧以不同方式交给 agent 模式，trace 和帧日志都与 `runtime/m0-results-r2/G1/<记录>/bio/` **逐字节相同**：

| 记录 | 方式 | 观测数 | 结束 | SR（0.3.12 规则）/ float32 和 | 其它 |
|---|---|---|---|---|---|
| SR06_1_36_163_132 | K=1：每帧一条动作，全部段 + TRAC | 13206 | LevelFinish，尾帧 3 | 96159 / 96163.29 | |
| SR09_1_29_321_264 | K=1 | 18351 | GameOver（end_reason 2），4 次 BallOff | 53962 / 53966.06 | |
| HS03_4906_1_56_934 | K=1 | 29498 | LevelFinish，尾帧 3 | 116958 / 116934.38 | HS 读数 4906（Points 3406 + 6×200 + 300），4 次暂停 |
| SR11_2_00_382_132 | K=1 | 16776 | LevelFinish，尾帧 3 | 120371 / 120382.98 | 494 条物理化记录 |
| SR06_1_36_163_132 | 块：37–86 帧的块、单帧 × repeat（≤400）、stop mask {StartLevel, BallNavActive, PreCheckpoint}，obs_mask = BODY + CAMS + WTCH（41） | 217 | 同上 | 同上 | 被关闭的三个段确实不出现 |
| SR06_1_36_163_132 | `BALLRAIN_PREFIX` 3000 帧后交接 | 160 | 同上 | 同上 | |
| SR06_1_36_163_132 | `BALLRAIN_PREFIX` + `PREFIX_EVENTS` | 3159 | 同上 | 同上 | |

- 所有观测都被 `ballrain/protocol.py` 的 `parse_observation` 解析，全量关卡文件和 `.inc` 都被 `parse_level_message` / `parse_level_stream` 完整解析（`.inc` 无剩余字节）。
- 所有观测上 `(float)SR == GetSRScore()`。
- 四条 K=1 记录逐帧与探针的 frames.txt 对比：BALL 的 PSI 位姿、四元数、速度，MIND 的个数、对方 ckid、间隙、法向，FRIC 的个数、对方 ckid、间隙、摩擦系数，在所有可比帧上**完全相等**。唯一的差别见 §8.7。
- STOP（end_reason 4）、`BALLRAIN_MAX_FRAMES`（3）、客户端进程消失（6：mod 写出输出后退出）：帧日志和 trace 都是参考的前缀。
- 回放模式用同一构建重跑 SR06、SR09：trace、帧日志、events 与 M0 逐字节相同。
- 未在游戏里走过的路径：等待超时（只试了父进程消失）、ResetLevel/ExitLevel 结束（5）、非法动作（7）、动作 flags bit2 重写关卡消息、非前缀的 `n_frames = 0`、段放不下被丢弃。
- 耗时：K=1 全部段时 SR06 的可导出部分 2568 ms，回放 1754 ms（开发中的构建、同一读取路径、存根只校验长度）；存根额外用 `protocol.parse_observation` 解析每条观测时为 3702 ms。

### 8.7 球的 IVP 物体在 Unphysicalize 之后不存在

`BALL` 的"找到 IVP 物体"在球的 Unphysicalize 到下一次 Physicalize 之间为 0：每次死亡（SR09：13637→14436 等）和**每次变球**（HS03：5201→5800、9740→10339、17738→18337、27590→28189；SR06、SR11 同理）。
变球开始时旧球就被 Unphysicalize 了。探针在这些帧里读到的"冻结的旧球"（state-spec.md §2.8、§3.5 中"变球期间旧球冻结、速度为 0、没有 mindist"）是**已释放的内存**：
它的 client_data 还指着球实体，所以按 client_data 找到了它。mod 现在只在 Physicalize / Unphysicalize 钩子之间缓存球的 IVP 物体，钩子触发即作废（钩子在块执行前运行），
缓存失效后只在管理器 +0x28 的活动物体表里找。变球和复活期间 Python 应当只用 CK 矩阵（插值位姿）或认为球不在物理中。

### 8.8 M1：Python 客户端与真实 mod 联调（2026-11-16，同一构建 9a8f1390…）

`ballrain/client.py` 现在对每条前缀帧观测（flags bit1）回一条 `n_frames = 0` 的确认动作（§5），并把这些观测交给 `AgentSession.on_prefix_obs`；
默认用 direct 方式启动（Popen，cwd = `<copy>/Bin`，与 M0 驱动相同；`SessionConfig.launch` / `EnvConfig.launch` 可改）。
`scripts/m1_gates.py` 的结果（`runtime/m1-results/results.json`）：

- M1-A：Python 回放策略（K=1，全部段 + TRAC + 关卡消息，每条观测都经 `parse_observation` 和 tracker）在 29 条严格 G1 记录、11 个 G2 合成输入、
  SR06 的 K=4 / 前缀 6000 / 前缀 + 前缀观测 3000 / Python 侧停顿（含 2 s 与 5 s）变体和 10 次重复上，trace 与帧日志都与 M0 逐字节相同；
  新构建的回放模式在全部 40 个严格用例上 trace、帧日志、events（除 Summary 行）与 M0 相同。HS13 在游戏里本身不可重复（真实 TASSupport 从第 42749 帧起每次轨迹不同，M0 已排除）。
- M1-B：BallanceEnv 随机策略 11 个进程（8 个拷贝并行；第 1 关 Q 键、34 条命、6 次 GameOver、2 次经 TAS 前缀到 LevelFinish、3 次帧数上限）
  导出的 .tas 各由参考 TASSupport 回放 3 次，3/3 相同且与 agent trace 逐字节相同。

### 8.9 BallrainIO 0.3.1（2026-11-16，构建 sha256 51762f20328c34dd…）

- 新增：SCOR / SREN 的追加字段（跟随中的分数小球，§8.2），LevelFinish 的 events 行 `followers=`，`SRFinal` 的 `post=`，
  诊断用的 post-finish 窗口 `BALLRAIN_POST_FINISH` / `BALLRAIN_POST_LOG`（§5、§6）。读取的只是被监视实体的 `IsVisible()`，不改变执行。
- M1-A 在新构建上重跑：agent 模式 29 条严格 G1 记录 + 11 个 G2 + SR06 变体与 10 次重复，trace 和帧日志与 M0 逐字节相同（54 通过，4 个非严格用例只报告）；
  回放模式 40 个严格用例 trace、帧日志、events 与 M0 相同（HS13 非严格，照旧不可重复）。M1-D 的 36 条记录（含 `runtime/tas-patched` 的
  HS04_6925_2_27_295_esc）在 8 个拷贝上回放，有 M0 结果的记录 trace 和帧日志与 M0 一致，mod 的跟随计数在每一帧上都等于 Python 从 WTCH 复算的值（共 1102524 帧，0 处不同）。
- M1-B 的 HS 用例（`L01_hs_esc_random`：随机策略含 Esc，70 次暂停 / 解除；`L01_hs_collect`：HS01 前缀 6600 帧后用 collect 宏，暂停 240 帧收齐 6 个小球）：
  导出的 .tas（含全部 Esc 和暂停帧）由参考 TASSupport 回放 3 次，3/3 相同且与 agent trace 逐字节相同。
- 结果记录的 `observations` 现在是解析过的观测条数（此前 `save_raw=False` 时恒为 0）。
