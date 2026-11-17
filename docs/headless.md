# 无头后端（headless，x87-soft 引擎，近似环境）

BallanceMMO 的无头引擎（SimTool：开源 CK2 / VxMath / CK2_3D + 真 BuildingBlocks + x87-soft 物理，原生编译）加上 Ballrain 的 agent 模式
（`headless/src/ballrain_agent.cpp`），说的是 BallrainIO agent 模式的同一套线格式（`agent-protocol.md` 的 OBS1 / ACT1 / LVL1），
`ballrain/protocol.py`、`state.py`、`features.py`、`rewards.py`、`env.py` 不需要改就能用。补丁集、源码提交和编译见 `headless/README.md`。

**定位**：用于大批量训练的近似环境。零售版 Player.exe + BallrainIO 仍是评估和导出 .tas 的唯一依据。
无头 run 的元数据有 `backend = "headless"`（`ballrain.recording.RunMeta`），每条观测的 OBS1 头带 `OBS_HEADLESS`（flags bit4），
hello 串里有 `headless`；零售 run 是 `backend = "retail"`。不要把无头 run 当作零售游戏的证据。

## 1. 结构

| 部分 | 位置 |
|---|---|
| 引擎 + agent 模式 | `BallanceMMOSimTool --ballrain-agent --root <游戏目录> [--tas-* ...]`（`headless/build_mac.sh` 编译；帧方案是 SimTool `--tas` 的参数和默认值，其余配置来自 `BALLRAIN_*` 环境变量，§6） |
| 传输 | `ballrain/transport.py`：`SocketChannel`（Python 端）/ `SocketModChannel`（游戏端，测试和 fakegame 用）；共享内存那条路径不变 |
| 启动 | `ballrain/headless.py`：`HeadlessLauncher` / `HeadlessConfig`；`EnvConfig(backend="headless", headless={...})` 自动用它 |
| 会话 | `ballrain/client.py`：launcher 可以提供 `make_channel()` 和 `backend`；hello 检查按 backend 分支（零售：physics_RT 哈希；无头：`headless`） |
| 不依赖 Python 包的驱动 | `headless/tools/replay_stub.py`（只用标准库，回放一个 raw 帧文件，存下全部观测） |
| 门禁 / 吞吐 | `scripts/headless_gate.py`（和 tas-v1 零售数据集逐帧比较）、`scripts/headless_bench.py` |

一个进程一个 run（与零售相同），run 结束或 socket 断开时引擎自己退出。

## 2. 传输：socket（沿用 legacy 原型的约定）

M0 之前的 TCP 原型（`legacy/TCPServer.py`、`legacy/comm_protocol.md`，mod 里删掉的 TCPClient）的约定照用：**Python 是服务端、游戏连上来**，
小端 u32 消息类型头，Ping/Pong 握手，TCP 端口基数 27787（默认 0 = 任意空闲端口，多实例时不冲突）。变化：每条消息加了 u32 长度
（`u32 type, u32 len, payload`），默认用 unix socket（macOS / Linux；Windows 用 TCP，`TCP_NODELAY`），载荷就是 M1 的线格式：

| 类型 | 方向 | 载荷 |
|---|---|---|
| `0x100` PING | Python → 游戏 | 空（连接后立即发） |
| `0x101` PONG | 游戏 → Python | u32 pid，u32 obs 版本位图，u32 mod_state，char[180] 版本串（含 `headless`） |
| `0x102` STATE | 游戏 → Python | u32 mod_state，u32 end_reason，u32 frames_done，u32 last_error |
| `0x103` OBS | 游戏 → Python | u32 mod_state，u32 end_reason，u32 frames_done，u32 last_error，u64 mod_seq，然后一条 OBS1 |
| `0x104` ACT | Python → 游戏 | 一条 ACT1（含 py_seq，语义与 `agent-protocol.md` §2.2 完全相同：repeat、stop mask、STOP、obs_mask、TRAC、重写关卡消息） |

共享内存头里的字段（mod_state、end_reason、frames_done、mod_seq）随每条 OBS 和 STATE 带过来，`SocketChannel.header()` 给出同样的 dict。
socket EOF：游戏已结束时是"结束"（`wait_obs` 返回 None），否则是 `PeerGone`。游戏等动作超时（`BALLRAIN_TIMEOUT_MS`）或 socket 断开时写完输出后退出。
关卡消息（`BALLRAIN_LEVELMSG` 和 `.inc`）、帧日志、events、trace 和零售一样写**本地文件**，所以 Python 必须和引擎在同一台机器上（§9）。

## 3. 帧路径与零售对齐

1. 引擎启动、加载 base.cmo，400 个固定 tick（15.15 ms）到主菜单。
2. 装钩子（§4），做 TASSupport legacy 模式 / BallrainIO 的脚本改动：删掉 `Ball_Explosion_Wood/Paper/Stone` 里第一个 `Set Position`（变球碎片从不物理化；
   用 SimTool 的 `apply_tas_legacy_edit`，`--tas-no-legacy` 关掉）。
3. 给 `Load Level`，再跑 **34 个隐藏的固定 tick**（`--tas-offset`；不进帧日志，期间的事件记在第 0 帧）。之后的第 0 帧与零售的第 0 帧对齐（PreLoadLevel 之后的第一帧）：
   零售在 PreLoadLevel 之前还有一段菜单时间，这 34 个 tick 补的就是它（x87-soft 会话测定，对全部 31 条 TAS 都成立）。
4. 每帧：Python 给的 delta 写进时间管理器，9 个 legacy 键按 `Keyboard` 数组的键位写原始 0/1（与 BallrainIO `SetKeyboardState` 相同），跑一次 `CKContext::Process`。
5. 第 0 帧执行完、它的观测之前做物理时钟复位（TASSupport OnStart 的 1 tick 定时器；`--tas-reset-frame`，默认 1）：IVP 时间 0、下一个 PSI 1/66 s、平滑 delta，
   再加上 Player.exe 在那一刻的 IVP 状态：`next_movement_check = 4`（`--tas-movement-check`）、IVP 随机种子 3560118427（= 75^21，`--tas-seed`）（x87-soft 会话测得，3 关 / 多次运行相同）。
6. 加载关卡时**不**重置 x87 精度（`headless_engine::set_reset_precision_on_load(false)`，`--tas-reset-precision` 恢复 BMMO 的逐关重置）：
   2002 进程只在第一个凸包（启动时 Init Ballpieces 建的）之前是 24 位，之后一直 64 位。
7. 确定性：CRT `rand()` 固定种子 1（CKStartUp 原来按 `time()` 播种）；球-球接触对默认按 CK id 排序（产品的 `IVP_Environment::bmmo_ball_order_key`；零售按堆地址，逐进程会变，§3.1）。
   物理管理器的平滑 delta（m = (m·3 + delta)/4）按 2002 DLL 的 x87 方式算（`CKIpionManager::SetDeltaTime`）：帧时间不变时与 float 写法相同，
   一变就不同——课程交接（TAS 的 3.79 ms → agent 的 7.58 ms）之后 5–6 帧就会分开，这一项修好之前所有交接都偏离零售。
   同一构建、同一组帧，轨迹逐位相同（§8）。
8. SR、尾帧、GameOver / LevelFinish / 帧数上限 / STOP / 超时的结束方式、前缀回放（含 `BALLRAIN_PREFIX_EVENTS` 的确认）都按 `Agent.cpp` 移植：
   SR = BML+ 0.3.12 规则的 double 和（本帧事件之后加本帧 delta），LevelFinish 后跑尾帧到 CounterInactive（最多 `BALLRAIN_SR_TAIL`），尾帧不进帧日志。

第 1 关的 Q 和其它前奏策略照旧在 Python 侧（`env._intro`）；无头 L1 从关卡加载起第 527 帧拿到控制，与零售实测相同。

2–6 的参数就是 SimTool `--tas` 的参数（BallanceMMO x87-soft 产品提交 `ba5ba279`），默认值是实测的 Player.exe 状态；agent 自己不再设一份
（`HeadlessConfig` 只在与默认值不同时传参数）。hx1c 构建的 `BALLRAIN_HL_NMC / SEED / LEGACY / KEEP_PRECISION / HIDDEN_TICKS / ORIG_RANDOM / BOOT_TICKS`
变量被忽略（设了会在 headless.log 里警告一行）。

### 3.1 球-球接触顺序（`BALLRAIN_HL_BALL_ORDER`，`HeadlessConfig.ball_order`）

两个球接触时 IVP 把哪个当第一个，决定接触法线的符号和冲量 / 摩擦求解里两个核的角色。2002 DLL 按两个 CK3dEntity 的堆地址排，地址顺序每个 Player.exe 进程都不同：
第 1 关在关卡加载时池里的 P_Ball_Stone_MF001–004 互相接触，那一刻的顺序决定这个进程整条 run（零售 50 个新进程的实测：`runtime/analysis/ball_order/summary.json`）。
无头引擎按一个键排（小的 = 低地址 = 第一个），模式：

| 值 | 含义 |
|---|---|
| `ckid`（默认） | 产品的 CK id 键，不动它（门禁用的就是它） |
| `reverse` | CK id 反过来（玩家球也跟着反，零售从不这样） |
| `perm:<seed>` | 池里的道具球（名字 `P_Ball_*_MF*`）在它们自己的 CK id 位置之间按种子置换；其余的球（含玩家球）位置不变 |
| `table:<名字=名次,...>` | 列出的实体在它们 `ckid` 下占的那些位置之间按名次（不能重复）重排；其余的球（Ball_Stone、Ball_Paper、纸球池等）位置不变 |
| `retail:<seed>` | 按零售实测分布抽一个进程的顺序。第 1 关：Ball_Wood 最低；然后 P_Ball_Stone_MF / P_Ball_Wood_MF，Stone_MF 在前的概率 0.66；然后 MF001–004 的均匀随机排列；像 `table` 一样只在这 7 个球自己的 `ckid` 位置之间重排，其余的球位置不变。其它关没测过：用 `ckid`，headless.log 里警告一行 |

`perm` / `retail` 不带种子时由 `HeadlessLauncher` 给每个进程抽一个新种子（32 位）。实际跑的值记进元数据 `RunMeta.ball_order`（`extra_env` 里的
`BALLRAIN_HL_BALL_ORDER` 优先于 `ball_order`；第 1 关以外的 `retail:<seed>` 记成 `ckid`）；引擎在 headless.log 里写一行 `ball order <值>: <球按键排的顺序>`。抽法是 splitmix64 + Fisher–Yates，
`ballrain.headless.retail_draw_l1` / `perm_draw` 和引擎抽得一样。非法的值是错误（引擎报 ERR_ENV 退出），不会悄悄退回 `ckid`。

实现：非 `ckid` 模式下 agent 把自己的键函数写进物理环境（`CKIpionManager::Reset` 会重建环境并装回产品的键，所以每个 tick 之前和每个 Physicalize 块之前都重新写一次），
排名表在第一次用到时按名字从关卡实体建（之后出现新的同类实体会重建并警告）。

## 4. 事件

BML+ 0.3.12 的事件钩子（`EventHook.cpp` / `ScriptHelper.cpp`）按同样的图位置、同样的顺序重建：在 `Event_handler`、`Gameplay_Ingame`（CamNav / BallNav On/Off）、
`Gameplay_Energy`、`Gameplay_Events` 里插入直通的钩子块（Nop 块换成自己的函数），Physicalize 用包装原型函数的方式（BML+ `HookPhysicalize`）。
`ActiveBallChanged` 取 `Gameplay_Ingame` 的 `ActiveBall` 局部参数（BallrainIO 相同）。事件号口径同 `state-spec.md` §1。

没有重建的：`Highscore` 那处图修改和 PostEndLevel（都在 run 结束之后），PreStartMenu / PostStartMenu / ExitGame / Next Level 只插了直通块、不出事件。
`BALLRAIN_POST_FINISH` 诊断窗口不支持。

## 5. 段和字段：哪些可用、哪些近似

字段读法：CK 侧（矩阵、数组、组、属性、网格、可见性）与 BallrainIO 相同的 CK API；IVP 侧不再用 2002 DLL 的偏移，而是按开源 IVP 的成员名读
（`IVP_Core`、`IVP_Real_Object`、`IVP_Mindist`、`IVP_Contact_Point`，后者通过 `IVP_BMMO_State` 访问器）。全部只读。

| 段 / 字段 | 无头后端 |
|---|---|
| TIME | 全部（env 时间来自物理桥的 `get_clock_state`；管理器三个值是 `m_DeltaTime` / `m_PhysicsDeltaTime` / `m_PhysicsTimeFactor`）。**第 0 帧**的观测建在隐藏 tick 之后，env 时间等与零售不同 |
| BALL | 全部；`core_flags` 只有 physical_unmoveable 两位（bit 2–3），`core_state` 只有 movement_state（低字节）——其余位是 2002 二进制位域，没有对应 |
| FRIC | 对方、状态、cp_status、间隙、摩擦系数可用；`lt_normal` / `lt_point` / `lt_valid` **不可用**（零售读的是事务内存，这里填 0，`lt_valid = 0`） |
| MIND | 全部，除 `ledge`（零售是地址，这里 0）；`plane[3]`（+0x74）零售也是不维护的残值，不可比 |
| BODY | 全部（遍历全部物理化实体，跳过不可动的核，与零售遍历整个簇等价）；`core_state` 只有 movement_state。项的顺序与零售不同（Python 按距离排序） |
| CAMS、ARRS、WTCH、EVTS、PHYS、TRAC | 全部 |
| GLCT | 块 id 和输出脉冲可用；三组 {active, number} 计数**不可用**（填 0，在 2002 指针后面） |
| SCOR / SREN | 全部；`bml_hs` 是计算值 Points + Lifes × LifeBonus（与 BML+ `GetHSScore()` 在全部 tas-v1 记录上相同）；Energy 第 4 列（Timefactor）是参数单元，原始值是指针，不可比 |
| HASH（新增，最后一条观测） | 运行指纹：每个导出帧的（帧号，球 CK 矩阵）的 FNV-1a 64；同一构建同一组帧必然相同。写进元数据 `trace_hash` 和 events 的 `TraceHash` 行。零售没有这个段 |
| LVL1 | 全部段；IVPO 的材质 32 字节**不可用**（0）；Floor 属性的原始字节表示不同（零售是子参数 id 列表，这里是串），Python 不用 |
| CK id | 由本引擎分配，与零售不同；门禁按实体 / 组 / 数组名字映射 |

有近似字段的段在段头 `sec_flags` 置 bit15（`protocol.SECF_APPROX`）：BALL、FRIC、MIND、BODY、GLCT、SCOR、SREN、IVPO。

## 6. 环境变量（引擎读）

BallrainIO agent 模式的变量照用：`BALLRAIN_LEVEL`、`BALLRAIN_LEVELMSG`、`BALLRAIN_FRAMELOG`、`BALLRAIN_TRACE`、`BALLRAIN_EVENTS`、`BALLRAIN_PREFIX`、
`BALLRAIN_PREFIX_EVENTS`、`BALLRAIN_MAX_FRAMES`、`BALLRAIN_TIMEOUT_MS`、`BALLRAIN_SR_TAIL`、`BALLRAIN_BODY_RADIUS`；新增 `BALLRAIN_SOCK`（`unix:<路径>` | `tcp:<主机>:<端口>`）。
无头设置：`BALLRAIN_HL_BALL_ORDER`（§3.1，默认 `ckid`）、`BALLRAIN_HL_CRT_SEED`（1）、`BALLRAIN_HL_VERBOSE`。帧方案是 SimTool 的参数（`HeadlessConfig` 字段 → 参数）：
`boot_ticks` → `--boot-ticks`（400）、`hidden_ticks` → `--tas-offset`（34）、`reset_nmc` → `--tas-movement-check`（4）、`reset_seed` → `--tas-seed`（3560118427）、
`reset_frame` → `--tas-reset-frame`（1）、`legacy_edit=False` → `--tas-no-legacy`、`reset_precision` → `--tas-reset-precision`、`orig_random` → `--tas-orig-random`。
`BALLRAIN_SKIP_RENDER` / `BALLRAIN_EXIT` / `BALLRAIN_LOADS` 被忽略（不渲染、一进程一 run）。

Python 侧：`BALLRAIN_HL_BIN`（SimTool 路径）、`BALLRAIN_HL_ROOT`（游戏目录）、`BALLRAIN_HL_SOCKET`（unix | tcp）是 `HeadlessConfig` 的默认值。

## 7. 门禁（`scripts/headless_gate.py`）

把一条 TAS 用无头 agent 模式按"每个动作 1 帧、全部段"回放（与 `dataset.replay_record` 相同），再与 tas-v1 的零售回放逐帧比较：
全部 OBS1 段逐字段（CK id 按名字映射；§5 的不可用字段单列为"approximated"）、关卡消息（全量 + 增量）、两边各自用 GameTracker 按真实会话的方式喂出来的 f5 特征向量、
事件、帧日志、SREN（SR double 和、float32 和、尾帧、HS 快照）。

### 7.1 结果（2026-11-16，M6；报告 JSON 在 `runtime/headless-results/gate/`）

先在 hx1 构建（hx1c 之前的同内容游戏数学 + 球序）上做了全量比较；hx1c 构建上 7 条记录各 3 次的轨迹哈希与之完全相同，
SR01_1_14_457_264、HS01_4736_1_32_258、SR06 的逐字段比较复跑结果见 §7.2。

每条记录跑 5 次（全部段，K = 1），轨迹哈希 5 次全同：

| 记录 | 球轨迹（与 tas-v1 trace） | LevelFinish / 官方 SR | 帧日志、SREN（SR double 和、float32 和、尾帧 3、HS 快照） | 事件 | f5 特征（逐位） |
|---|---|---|---|---|---|
| SR01_1_14_457_264 | 20483/20483 | 20587 / 74446 ms | 相同 | 相同 | **20485/20485 帧相同** |
| SR01_1_15_797_240 | 18944/18944 | 19048 / 75795 ms | 相同 | 相同 | 526 帧只差 `follower_near` |
| SR01_1_16_726_240 | 19337/19337 | 19441 / 76724 ms | 相同 | 相同 | 512 帧只差 `follower_near` |
| HS01_4736_1_32_258 | 23326/23326 | 23430 / 92187 ms | 相同（HS 4736） | ExtraPoint 时刻在 8497–8561 之间有 4 处不同 | `follower_near`；14 帧 `points` / `followers` |
| HS01_4738_1_31_317 | 23146/23146 | 23250 / 91324 ms | 相同（HS 4738） | ExtraPoint 从 8383 起有不同 | `follower_near`；94 帧 `points` / `followers` |
| HS01_4740_1_30_344 | 25268/25268 | 25372 / 90329 ms | 相同（HS 4740） | ExtraPoint 从 9005 起有不同 | `follower_near`；59 帧 `points` / `followers` |
| SR06_1_36_163_132 | 13096/13096 | 13205 / 96159 ms | 相同 | 相同 | `follower_near`；2 帧 `normal_cam[2]` |

逐字段（§5 的近似字段之外）在全部记录上相同的：TIME（第 1 帧的 `mgr_step_s` 除外）、BALL 全部（含 PSI 位姿、四元数、速度、速度增量）、FRIC、MIND
（SR06 有 16 帧 `plane` 差 1e-14）、BODY、GLCT 脉冲、SCOR（HS 记录里分数小球计分时刻附近除外）、EVTS（同上）、PHYS、SREN；关卡消息的物理化记录（每帧按多重集比较：
同一帧内 Physicalize 的调用顺序与零售不同，记录序号因此对应到别的实体）、网格（顶点、三角形、导出矩阵）、DepthTestCubes、监视清单、IVPO 实体、组、玩法数组
（Energy、CurrentLevel、IngameParameter、PH、Checkpoints、ResetPoints、AllLevel、Keyboard）。

仍然不同（都不影响球的物理；交给 x87-soft 会话）：

- **相机矩阵**：Cam_Orient / Cam_Pos / InGameCam 最大差约 7e-4，SR01_1_14_457_264 的 Cam_OrientRef 从第 6344 帧起差到 3e-5；f5 只用它们的偏航角，SR01_1_14_457_264 上特征仍逐位相同。
- **被监视的装饰物体位置（WTCH）**：P_Extra_Life_Sphere / SilverBall（从 3600–3900 帧起）、分数球的框和跟随的分数小球（最多差 4–17 单位）。不是 CRT `rand()`
  （换种子不变），是这些动画脚本的数学。后果：`follower_near` 特征，HS 记录里分数小球到达的帧（ExtraPoint 事件、Points 在几十帧内不同；最终 HS 相同）。
- 加载期间：第 1–2 帧挂接的相机、第 9–12 帧附近一个数组单元的瞬时值（HS 记录里 ARRS 的不同还包括上面的计分）、第 1 帧的 `mgr_step_s`（隐藏 tick 留下的物理步长）。
- 第 0 帧的观测建在隐藏 tick 之后（§3），TIME 不可比，单独报告。
- 关卡消息里 52 个实体的 StartLevel 矩阵不同：全是变球碎片 `Ball_*_piece*`（legacy 模式下从不物理化）；菜单类数组（Logic_Scripts 等，单元里是脚本 id）不可映射。

训练用的特征配置（`train_m2.py --world-actions`：`FeatureConfig(hs_block=False, camera="world")`，196 维）：SR01_1_14_457_264、SR01_1_15_797_240、SR01_1_16_726_240
**全部帧逐位相同**（20485 / 18946 / 19339 帧），SR06 只有 3 帧 `normal_cam[2]` 不同（`follower_near` 在 HS 块里，SR 训练不用）。
同一构建下 TraceHash 与动作块大小和 obs_mask 无关（SR01_1_14_457_264：K = 1 全部段、K = 4、K = 1000 且关掉全部可选段，哈希都是 `3bf0b490e7cf7acb`）。
7 条数据集分别录在 train-00 / 01 / 02 / 05 / 07 上，复位状态（4 / 75^21）在这些拷贝上都给出逐位相同的轨迹，可以看作它们的 Player.exe 复位状态一致的证据。

逐次运行的确定性：球序修好之后 SR01_1_14_457_264 和 SR06 各 12 次、上表 7 条各 5 次，轨迹全同（球序修好之前 SR01 约 25% 的运行在第 7490 / 19307 帧分开）。

### 7.2 离线路线（off-route）：随机输入的零售参照

hx1c 构建（`hx1c-96c0cc01-ce6afc34`）上用 agent 模式回放，与零售参照（参考 TASSupport 回放，3 次相同）逐帧比较球轨迹：

- M1-B 的 13 个用例（L1–L4、L6、L9、L11：随机、抖动、Esc 暂停 / collect、死亡、GameOver、TAS 前缀到 LevelFinish）：**13/13 全部帧相同**；
- M2 的 3 个导出（SR01 前缀 + agent 尾段，课程交接）：**3/3 相同**。同样的 3 个在 hx1c 之前的构建上于交接后 5–6 帧分开（15544、12337、20555）。
- 第 5、7、8、10、12、13 关的新参照（`scripts/offroute_refs.py`，零售 train-12..15，参考 TASSupport 3 次，`runtime/offroute-refs/L<nn>/<case>/`）：
  每关 3 个（random：死亡到 GameOver；hs_esc_random：Esc 暂停 / 解除 + 死亡；jitter：8000 帧上限），共 18 个，合计 61 次死亡、599 次暂停、12 次 GameOver。
  每个都是参考 TASSupport 3 次回放逐字节相同且等于 BallrainIO agent 轨迹；hx1c 无头 agent 模式回放 **18/18 全部帧相同**。

### 7.3 产品提交构建（2026-11-16，M6 arm64，`x87s-ba5ba279-f7d25565-e98aa446`；之前的 `-e9e7ac69` 只差 table / retail 的排位规则，门禁结果相同）

BallanceMMO x87-soft 产品提交 `ba5ba279`（Ballanced `b70ab01` 等，`headless/README.md`）+ 新的 0002 + agent（帧方案用 SimTool 的 `--tas` 参数，默认 `ckid` 球序），
与 hx1c 构建（`hx1c-96c0cc01-ce6afc34`）同样的输入逐个比较（`replay_stub`，K = 1，全部段）：

| 用例 | 次数 | 与零售参照 | TraceHash / trace / events / framelog / level.msg / SR / 结束方式与 hx1c |
|---|---|---|---|
| 7 条门禁记录（SR01 ×3、HS01 ×3、SR06） | 各 3 次 | 7/7 全部帧相同（同 §7.1 的帧数） | 21/21 逐字节相同，3 次之间也相同 |
| M1 / M2 离线路线（§7.2，13 + 3） | 1 | 16/16 全部帧相同 | 16/16 相同 |
| `runtime/offroute-refs`（第 5、7、8、10、12、13 关） | 1 | 18/18 全部帧相同 | 18/18 相同 |

TraceHash 都与 §7.1 / hx1c 相同（例如 SR01_1_14_457_264 `3bf0b490e7cf7acb`、SR06 `1a9f686e68ac71ff`）。`obs.bin` 和 `level.msg.inc` 不逐字节相同，
但 hx1c 构建自己的 3 次 SR06 之间也不同：SCOR / SREN 的 Energy 第 4 列是指针（§5），IVPO 记录按 IVP 对象地址去重（地址复用时落到别的增量消息里）。
`headless_gate.py` 在 7 条记录上复跑（报告在 `runtime/headless-results/gate-x87s-ba5ba279/`）：除特征版本号（现在是 f6）之外与 §7.1 的报告完全相同，
特征不同的帧数也相同（SR01_1_14_457_264 20485/20485 帧逐位相同）。

球序模式（§3.1）在 r3 的一条 L1 抽查 run（SR01 前缀 8907 帧 + agent 尾段，headless 轨迹与零售回放在第 17862 帧分开）上：`ckid`、`retail:12`、`perm:2195`
和按 `ckid` 顺序写的 `table` 逐字节重现 headless 轨迹；把 P_Ball_Stone_MF004 放在 MF002 前面的 `table`、`retail:0`、`perm:1319` 逐字节重现零售回放轨迹；
`reverse` 两个都不是（第 7386 行就分开，玩家球的顺序也反了）。所以这条 run 的分歧就是零售那个进程里 MF004 的地址低于 MF002。

## 8. 确定性

同一构建、同一组帧，每次运行的轨迹必须逐位相同。球序修好（hx1c）之前不是：SR01_1_14_457_264 在约 25% 的运行里于第 7490 帧（Ball_Wood 碰 P_Ball_Wood_MF，
接触对按 CK3dEntity 的堆地址排序）或第 19307 帧分开，Mac 和 Windows 构建都有（x87-soft 会话定位，修法是按 CK id 排序；零售 Player.exe 同样按地址排，
偶尔分叉的来源可能就是这里）。检测手段：元数据的 `trace_hash`（§5 HASH）——同一 .tas 在同一构建上重放，哈希必须相同。

## 9. Python 在哪里跑：整个训练器放在 Mac 上

- 引擎写本地文件（关卡消息、帧日志），每步一次往返；PC 和 Mac 之间的局域网往返（约 0.3–1 ms）比无头引擎跑 4 帧还慢，所以 Python 必须在引擎旁边。
- Mac 上的 Python（uv 管理的 CPython 3.12、PyPI 的 torch 2.15 CPU）：PPO 一次更新（4096 步 × 4 轮，minibatch 1024，4 线程）0.10 s，
  每步推理 0.057 ms（1 线程）。采集 4096 步要 1–3 s（§10），更新只占一小部分。
- 所以选 (b)：整个 `scripts/train_m2.py` 在 Mac 上跑（learner + 每个实例一个 rollout worker），不用把 rollout 运回 PC。
  `ballrain/train/worker.py` 不用改：`WorkerSpec(env=EnvConfig(..., backend="headless", headless={...}), copy="hl-03")`，`BallanceEnv` 按配置自己建 `HeadlessLauncher`。
  `train_m2.py` 需要的只是一个后端开关（`EnvConfig.backend` / `headless`，以及实例名 `headless.headless_copy_names(n)` 代替 `train-xx`）。
- 内存：每个引擎进程约 460–470 MB，每个 worker 的 Python 约 70–170 MB（curriculum 读数据集时更大）；16 GB 上 12 个实例约 8 GB。

## 10. 吞吐

### 10.1 M6（2 Super + 4 P + 6 E 核，16 GB），2026-11-16，默认构建

`scripts/headless_bench.py`：每个实例一个 OS 进程跑 `BallanceEnv(backend="headless")` 60 s，L1，SR，f5 特征，action repeat 4，132 fps；
指标口径同 `scripts/m2_start_states.py throughput`（null = 一直空动作，random = 随机粘滞动作 2–39 步，curr random = SR01_1_14_457_264 最后 3000 帧内的课程起点 + 随机）。
"帧/秒"含前缀回放的帧，"决策帧/秒"不含。最右列是 PC 上零售 Player.exe 的同名配置（`runtime/m2-results/throughput.json`，最多 8 个拷贝）。

| 配置 | 实例 | 决策步/秒 | 帧/秒 | 决策帧/秒 | 命/小时 | 特征 ms/步 | env.step ms | 新进程占时 | 零售 PC（同配置） |
|---|---|---|---|---|---|---|---|---|---|
| null | 1 | 1366 | 5515 | 5462 | 0 | 0.52 | 0.71 | 3% | ×1 482 步/秒 |
| null | 6 | 5892 | 23779 | 23568 | 0 | 0.72 | 0.99 | 3% | ×4 1674 |
| null | 10 | 7263 | 29313 | 29049 | 0 | 0.94 | 1.34 | 2% | ×8 1937 |
| null | 12 | 8423 | 34008 | 33692 | 0 | 0.98 | 1.39 | 3% | |
| random | 1 | 549 | 5826 | 2196 | 24359 | 0.45 | 0.62 | 51% | ×1 60 |
| random | 6 | 2717 | 26070 | 10867 | 101838 | 0.67 | 0.92 | 44% | ×4 202 |
| random | 10 | 3371 | 33895 | 13483 | 136632 | 0.93 | 1.35 | 41% | ×8 305 |
| random | 12 | 3691 | 36740 | 14765 | 147029 | 1.05 | 1.57 | 39% | |
| curr random | 1 | 281 | 18587 | 1118 | 12100 | 0.48 | 0.67 | 75% | ×1 47 |
| curr random | 6 | 1311 | 83731 | 5216 | 54474 | 0.65 | 0.93 | 74% | ×4 154 |
| curr random | 10 | 1739 | 112518 | 6917 | 72794 | 0.96 | 1.42 | 69% | ×8 222 |
| curr random | 12 | 1924 | 123149 | 7651 | 80250 | 1.04 | 1.57 | 69% | |

- 开新进程：启动到 hello 0.18–0.25 s（中位数），加载 + 34 个隐藏 tick + 开局 L1 约 0.1 s，前缀回放约 3 万帧/秒；零售是启动 1.5 s + 加载 1.4 s + 前缀 1.2 万帧/秒。
- 内存：每个引擎进程 RSS 约 460 MB（中位数，各配置相同）；基准的 worker（不带 torch）66–107 MB。真实训练（`train_m2.py`，10 个 worker，带 torch）：
  10 个引擎 4.6 GB + 12 个 Python 进程（learner、10 个 worker、resource tracker）2.6 GB，无 swap。
- 基准期间 Mac 上同时跑过 `uv sync` 和两次短回放（random ×10 / ×12 两档），数字可能略低。
- 扩展：6 → 10 → 12 个实例收益递减（E 核和共享内存带宽），random 12 个实例比 10 个多约 10%，curr random 多约 11%。
- 训练时的建议：**10 个 worker**（留出 2 个核给 learner 和主进程 Python；PPO 更新在 Mac 上每次 0.10 s，每 4096 步一次，占比很小）。
  干跑（`train_mac.sh`，2 个 worker，6 次更新，batch 1024）在 Mac 上端到端通过：rollout、PPO 更新、课程、模型检查点、tensorboard、`runs.csv` 都正常，出现过一次 LevelFinish。
  10 个 worker 的负载运行（课程起点，`--world-actions --stall-terminal --stall-s 4`，2.5 分钟，31 次更新）：1000–1270 决策步/秒，PPO 更新 0.145 s，
  `camera_off_quarter` 与 `invalid_runs` 都是 0，527 个进程，episodes 目录 29 MB（worker 会清理）。
- 进程生命周期：worker 退出（或被杀）时它的引擎因 socket EOF 在几秒内退出；**learner 被 `kill -9` 时 worker 不会退出**（成为孤儿继续开进程，需手动停掉）——
  这是 `ballrain/train/worker.py` 的事（建议 worker 检测父进程消失）。正常停止用 STOP 文件。

## 11. 在 Mac 上训练（`headless/mac/`）

PC 侧（不在被跟踪文件里写主机名 / 用户 / 密钥：`BALLRAIN_MAC=<user>@<host>`，`BALLRAIN_MAC_SSH="-i <key> ..."`）：

```sh
headless/mac/push_repo.sh -r SR01_1_14_457_264 [-c runtime/m2-train/r2/latest.pt]   # 工作树（含未提交的代码）+ 数据集记录 + 模型检查点
headless/mac/pull_run.sh r3 300        # 每 5 分钟把 Mac 上的 runtime/m2-train/r3 拉回 PC 的 runtime/m2-train-mac/r3（TensorBoard 读这里）
```

Mac 侧（`$MAC_DIR` 默认 `~/ballrain`：`repo/` 是推过去的工作树，`hl/x87s-<BallanceMMO 提交>-<0002 哈希>-<agent 哈希>/` 是引擎构建（`hl/ENGINE` 记着当前那个；运行中的训练用启动时的路径，新构建不会覆盖它），`game/` 是游戏目录）：

```sh
headless/mac/setup_mac.sh [src.tar]    # uv python install 3.12 + uv sync；没有引擎时用 src.tar 编译（不改 hl/ENGINE）
headless/mac/setup_mac.sh --activate   # 门禁（§7.3 的比较）通过之后：把新构建写进 hl/ENGINE
headless/mac/train_mac.sh r3 --workers N --world-actions ...   # = scripts/train_m2.py --run r3 --backend headless ...，
                                                                 # caffeinate（先 -u 再 -s -i），从 /bin/sh 启动，nohup
touch repo/runtime/m2-train/r3/STOP    # 停止（learner 自己的停止文件）
```

`train_m2.py --backend headless` 的 `--hl-bin` / `--hl-root` / `--hl-socket` 对应 `EnvConfig.headless` 的 `binary` / `root` / `socket`
（缺省时取 `BALLRAIN_HL_BIN` / `BALLRAIN_HL_ROOT` / `BALLRAIN_HL_SOCKET`，`train_mac.sh` 会设好前两个），实例名 `hl-<offset+i>`。

## 12. 没有做 / 限制

- 在真实引擎上没跑过：`BALLRAIN_PREFIX_EVENTS`（fakegame 测过）、动作 flags bit2 重写关卡消息、非前缀的 `n_frames = 0`、ResetLevel / ExitLevel 结束。
- 远程驱动（PC 上的 Python 驱动 Mac 上的引擎）：传输支持 TCP，但关卡消息 / 帧日志是本地文件，不支持。
- `BALLRAIN_POST_FINISH` 诊断窗口不支持。
- 加载期间（第 1–12 帧左右）有个别观测与零售不同：第 1–2 帧挂接的相机、第 9 帧附近一个数组单元的瞬时值。之后逐位相同（§7.1）。
- 其它关卡的帧 0 对齐（34 个隐藏 tick）由 x87-soft 会话在全部 31 条 TAS 上验证过；本门禁只跑了第 1 关的 6 条和 SR06。

## 13. Windows x86：跑原版 2002 physics_RT.dll 的宿主（2026-11-17）

构建与机制见 `headless/README.md`（补丁 0003、`headless/src/phys2002.*`、`headless/build_win.ps1`）。同一个 agent 模式、同一套线格式，
IVP 段（TIME / BALL / FRIC / MIND / BODY / GLCT / IVPO）按 `BallrainIO/src/Agent.cpp` 读 2002 DLL 的内存，所以 §5 里"不可用"的
FRIC `lt_*`、GLCT 三组计数、BALL / BODY 原始位域、IVPO 材质字节都有了；这些段不再置 `SECF_APPROX`（MIND 除外：`ledge` 是本进程的地址）。
hello 串带 `phys2002 e72e4afcfa5c33a7`。`train_m2.py --backend headless --hl-bin <这个构建> --hl-root <装着零售 physics_RT.dll 的游戏目录>` 不用改代码
（PC 上 2 个 worker、3 次更新的冒烟跑通：`framelog_match` 全真，`handover_mismatches` 0）。

### 13.1 门禁（PC，干净树构建：src.tar + 0002 + 0003 + headless/src）

| 检查 | 结果 |
|---|---|
| 7 条 tas-v1 记录逐字段（`headless_gate.py --phys2002`，K = 1，全部段） | 球轨迹、BALL / FRIC / MIND / BODY / GLCT / TIME（第 1 帧 `mgr_step_s` 之外）/ PHYS / SREN、关卡消息（物理化记录、网格、IVPO、数组、组、DEPT、WLST）全部相同；原来单列的 FRIC lt_*、GLCT 计数、BALL / BODY 原始位域也相同，只有 HS01_4738_1_31_317 的 `lt_point` 107 帧 / `lt_normal` 56 帧（第 16204 帧起，差约 1e-5，两边 `lt_valid` 都是 1，球轨迹不受影响） |
| 仍不同（与 x87-soft 后端相同的 CK 侧来源） | CAMS 矩阵（SR01_1_14_457_264 14293 帧）、WTCH.pos（13872 帧）、加载期间 ARRS 2 帧和 CAMS.attached 2 帧、第 1 帧 `mgr_step_s`；HS01 三条里小分数球附近的 EVTS / SCOR / WTCH（ExtraPoint 时刻，最终 HS 相同）；关卡消息 52 个 `Ball_*_piece*` 的 StartLevel 矩阵 |
| 只按地址 / 残值单列 | MIND `ledge`（地址）、MIND `plane[3]`（IVP 不维护）、Energy 参数单元（指针） |
| f6 特征，训练配置（`hs_block=false, camera=world`） | SR01 ×3、SR06：**全部帧逐位相同**（x87-soft 后端在 SR06 上有 3 帧 `normal_cam[2]` 不同）；HS01 ×3 只有 `points` 14 / 94 / 59 帧 |
| TraceHash | 7 条都与 §7.1 / x87-soft 相同（如 SR01_1_14_457_264 `3bf0b490e7cf7acb`、SR06 `1a9f686e68ac71ff`） |
| 离线路线（轨迹对零售参照） | M1-B 13/13、`runtime/offroute-refs` 18/18（ref1–3 都相同）全部帧相同 |
| M2 交接（`BALLRAIN_PREFIX` = 前缀帧，再逐帧动作） | 3/3 与零售参照和 BallrainIO 的 agent 轨迹都相同；`BallanceEnv`（`start_record` / `start_frame` 15538，world actions，K = 4）前缀部分到第 15538 帧与零售相同 |
| 确定性 | SR01_1_14_457_264、SR06 各 10 次 + 5 次、L01_random_a / L01_hs_esc_random 各 5 次、3 条 M2 各 5 次：TraceHash、轨迹全同；逐字段比较观测（`det_obs`）除 MIND `ledge` / `plane[3]` 外全同；`BallanceEnv` 同种子两次特征逐字节相同 |
| L1 球序模式（r3 抽查 run，SR01 前缀 8907 帧） | 与 §7.3 的 x87-soft 结果完全一样：`ckid`、`retail:12`、`perm:2195`、按 `ckid` 顺序的 `table` 重现 x87-soft 的无头轨迹（TraceHash `fcb350f52ebd365f`，等于那条训练 run 自己的）；MF004 在 MF002 前的 `table`、`retail:0`、`perm:1319` 重现零售回放；`reverse` 两个都不是（7386 行后分开） |
| 同一二进制 `BALLRAIN_HL_PHYS2002=0`（源码物理） | SR01_1_14_457_264 / SR06 的 TraceHash 与 x87-soft 相同 |
| 其余 24 条 tas-v1 记录（L2–L12，SR08 / SR09 / HS09 / HS13 除外），`--phys2002` 逐字段 | **BALL 全部字段 24/24 每帧相同**，帧日志 24/24、SR 和结束方式相同；SR 记录的 EVTS 13 条里 10 条相同。不同的：旋转机关的 BODY（SR07 / SR10 的 `P_Modul_41*`、SR03 的 `Modul17_Dreharme*` 等，同一二进制的源码物理给出一模一样的差别，所以是 CK 侧驱动，两个无头后端共有）；HS 记录的计分 / 小球（11 条里 6 条最终 HS 不同，CK 侧）；FRIC `lt_*` 零星 1–8 帧（HS04 2708 帧，跟着 BODY 走）；CAMS、WTCH、加载期间 ARRS |
| 死亡 / 暂停 / GameOver 路径的观测（缓存失效、簇遍历、实体映射） | 6 个离线路线用例（L01 ×3、L05、L07、L09、L13）同一二进制源码物理 vs 2002：除源码给不出的字段和第 0 帧 TIME 外，所有段逐帧相同；帧日志 31/31 与零售 episode 相同，事件 21/31 相同（其余 9 条只差零售在帧数上限时多写的一行 `RecordEnd`，1 条是 L01_hs_collect 的 ExtraPoint 早 1 帧） |
| TraceHash 与动作块大小 / obs_mask 无关 | SR01_1_14_457_264、SR06：K = 1、K = 4、K = 1000 + obs_mask 255 三种哈希相同 |
| 错的 DLL | `BALLRAIN_HL_PHYS2002` 指向别的文件：引擎在 hello 之前报 sha256 不符并以 STATE（ERROR，last_error 2）退出；Python 侧 `TransportError: game refused to run`，不会退回源码物理 |

### 13.2 吞吐（PC，同时有训练在跑，数字偏低）

- agent 模式，K = 1，全部段，`replay_stub`（SR01_1_14_457_264，单实例）：2002 DLL 3.9–4.3 千帧/秒，同一二进制的源码物理 2.3–2.5 千帧/秒；
  `--tas`（无观测）2.06–2.26 万帧/秒。
- `headless_bench.py`（BallanceEnv，K = 4，30 s）：

| 配置 | 实例 | 2002 DLL 决策步/秒 | 帧/秒 | x86-soft（`bin-x86soft`）决策步/秒 | 帧/秒 |
|---|---|---|---|---|---|
| random | 1 | 297 | 3392 | 149 | 1717 |
| random | 8 | 1703 | 18380 | 1101 | 12247 |
| curr random | 1 | 180 | 12142 | 72 | 4901 |
| curr random | 8 | 1111 | 71014 | 462 | 32738 |

### 13.3 没有解决

- 第 9 关：HS09_6170_2_04_662 在 agent 模式和 `--tas` 下都在第 9725–9738 帧附近崩溃（每次相同，与球序钩子无关），SR09 只在 `--tas` 下崩（agent 模式跑完）。
  崩在 `CKBehavior::Execute`（CKBehavior.cpp:389）读 `m_Context`：外层行为在 "physicalize new Ball"（Ball_Paper）的 Physicalize 执行期间被释放 / 清零。
  怀疑 2002 Physicalize 经 CK2 垫片（`DestroyObject` / `CKDestroyObject` 等）的调用在 Ballanced 里语义不同，没有查实。第 1 关不受影响。
- SR09：`--tas` 与零售 tas-v1 数据逐帧相同（源码物理到零售数据结尾 18349 帧，2002 DLL 到它在 14432 帧崩溃为止），但 **agent 模式**（2002 和源码物理给出完全相同的轨迹）
  在第 12284 帧起球的 CK 矩阵差 1 ulp，之后分开。与观测读取（K = 1000 + obs_mask 255 也一样）、球序钩子、CRT 种子都无关，所以是 agent 模式和 `--tas` 的帧方案 / 钩子差别，
  两个无头后端共有，只在这条（G1 不用的）记录上看到。原因没查。
- HS01_4738_1_31_317 的 FRIC `lt_point` / `lt_normal`（§13.1）：对方是 `P_Modul_01_Pusher`（7 帧）和 `A03_Floor_02` / `A04_Floor_01`，没有解释。
