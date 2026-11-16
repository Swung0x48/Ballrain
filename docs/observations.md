# 观测、奖励与回合（Python 侧）

本文说明 Python 如何把 agent 模式的原始状态（`agent-protocol.md`、`state-spec.md`）变成 RL 的观测和奖励，
以及每个回合怎样录制成可复现的 legacy .tas。代码在 `ballrain/` 下：

| 模块 | 作用 |
|---|---|
| `protocol.py` | 线格式：共享内存头、OBS1 / ACT1 / LVL1 消息、各段的 numpy dtype，解析和编码 |
| `transport.py` | 共享内存 + 两个命名事件（Python 端 `PyChannel`，mod 端 `ModChannel`） |
| `client.py` | `AgentSession`：一个游戏进程 = 一个 run；启动、握手、发动作、读关卡消息、帧日志 |
| `state.py` | `GameTracker`：关卡消息、数据数组、事件累积出的状态（命数、分数、扇区、变球、跟随中的分数小球、暂停、正式 HS 读数…） |
| `geometry.py` | 三角形网格、二维网格索引、最近点、竖直射线、可走面边界 |
| `path.py` | 参考路径（起点 → 检查点 → 飞艇） |
| `features.py` | 特征提取（版本 `f3`） |
| `rewards.py` | SR / HS 奖励 |
| `recording.py` | 每个进程的 .tas + 元数据；BML+ 0.3.12 SR 计时器的 Python 复算 |
| `env.py` / `vecenv.py` | gymnasium 单实例环境；N 个拷贝的向量环境 |
| `dataset.py` | TAS 数据集构建（回放策略，逐帧原始状态） |
| `fakegame.py` | 测试用的模拟 mod：同一套线格式和传输，玩具关卡 |

原则：mod 只给原始数据，**所有特征、奖励、掩码、坐标变换都在 Python 里算**。改观测不需要重编 mod，也不需要重跑保真门禁。

## 1. 坐标与约定

- **CK 矩阵**是行向量约定：`world = [x y z 1] · M`。第 0/1/2 行是物体的 x/y/z 轴，第 3 行是位置。线格式里的 "12×f32" 是第 0–3 行的前三列。
- **IVP 四元数**按 (x, y, z, w) 存，按列向量旋转：`world = R(q) · v + pos`。
  在 SR06 探针数据上，`R(q_last_psi)` 与球的 CK 矩阵转置相差 0.04（CK 矩阵比上一个 PSI 超前半个 PSI，所以本来就不会完全相等），与不转置的比较相差 0.69。
- **运动体**：`world = R(q_core) · (R(q_cfo) · v + shift) + pos`，v 是 CK 网格的局部顶点（`state-spec.md` §2.5）。
  `q_core_f_object` 的约定**未验证**：探针样本里它都是单位四元数。
- **球的位置**一律用 IVP core 的 PSI 位置（BALL 段 `pos`），不用 CK 矩阵：接触、mindist 都是 PSI 时刻的值。
- **相机系**：取 `Cam_OrientRef`（默认，离散的 90° 朝向；可配置为 `Cam_Orient` 或世界系）的 z 轴投影到水平面作为 forward，
  right = (f.z, 0, −f.x)，up = 世界 y。向量 v 在相机系里写成 (v·right, v.y, v·forward)。键盘方向由相机朝向决定，所以方向性特征都放在这个系里。
- **路径系**：参考路径当前线段的水平切向作为 forward，right 同上。

## 2. 球与路面的相对关系

难点是每帧都要知道：球在不在路面上、是否接触、离路边多远、接触法向、接触的是哪种表面，运动体上也一样。做法如下（依据见 `state-spec.md` §3）：

| 问题 | 怎么算 | 来源 |
|---|---|---|
| 是否接触（几何） | `min(MIND.gap) ≤ 0.05` → `geom_contact` | IVP mindist（物理真值） |
| 是否稳定滚动 | FRIC 非空 → `fric_contact`，`n_fric` 是接触数（栏杆上常为 2） | IVP friction synapse |
| 离表面多高 | `min_gap`：最小 mindist 的间隙，没有 mindist 时为 2.0 | IVP mindist |
| 接触法向 | 最小 mindist 的 `contact_plane`（从表面指向球心），转到相机系 → `normal_cam`。**不用** `tmp_contact_info` 的法向（约 20% 的帧是旧值） | IVP mindist |
| 接触点 | 球心 − 法向 × (半径 + 间隙) | 派生 |
| 表面类别 | 接触对象（优先 friction 接触，否则最小 mindist 的对象）的 `Continuous Contact ID`：1 石、2 木、3 金属；0 或无属性记为 other；无接触记为 none。游戏逻辑的 GLCT 只作旁证（它不统计 ID 0 的对象，暂停时读数为 0） | ENTS 属性 |
| 对象在动吗 | 对象的物理化记录不是 fixed → `partner_moving`；与它的相对速度 `rel_vel_partner` = 球速度 − 运动体速度 | PHYS、BODY |
| 是不是栏杆 | 对象属于 `Phys_FloorRails` → `partner_rail` | GRPS |
| 离路边多远（方向） | 射线扇：从接触点沿 `fan_dirs` 个方向（0 = 相机前方，顺时针）每 0.25 一步，在球心上方 1 处向下投射；第一次没有落点、或落点比接触高度低 `fan_drop`（1.0）以上，就是路边 → `edge_fan` | 几何（静态网格 + 当帧位姿的运动体） |
| 离路边多远（最近） | 网格边界法：可走面（\|n_y\| > 0.5）顶点按 1e-3 焊接，只属于一个面的边是边界边；接触点到最近边界边的距离 → `edge_mesh`。栏杆不参与（栏杆上的边界距离没有意义） | 几何 |
| 栏杆在哪 | 球心到最近栏杆三角形的距离减半径 `rail_dist`，以及水平方向 `rail_dir`（相机系） | 几何 |
| 周围地形 | 以球为中心、相机系下 G×G 的竖直射线：落点高度 − 球底高度（裁剪到 ±5，归一化），以及是否有落点 | 几何 |
| 这一帧可信吗 | `valid` = 找到了 IVP 球体，球类型已知，且不在变球中（变球 = 活动球的 Unphysicalize 到新球的 Physicalize，见 §3 `transforming`；暂停中和 LevelFinish 之后仍然有效） | BALL、EVTS、PHYS |

其它规则：

- **stopper 排除**：`Phys_FloorStopper` 组的物体不与玩家球碰撞，几何里一律去掉。
- **纸球**是 315 顶点的多面体，`extra_radius` = 0。接触、间隙、法向只用 mindist；地形特征里的"球底"按半径 2 近似。
- **运动体几何**：BODY 段给出附近运动体的 PSI 位姿，配合 MESH 的局部顶点，当帧变换后参与地形射线（`dynamic_geometry`）。
- **撞击和弹跳**：几何上贴着但没有 friction 接触的帧是撞击、弹跳或落地前一帧，不是错误。两个标志分开给。

**核对**：用探针在原版上导出的网格和 mindist（SR01、SR11 各 2000 帧）检查 `geometry.py`。对每个静态对象，取该对象所有 mindist 的最小值，与 Python 算出的球心到该对象网格的距离减 2 比较：

| 记录 | 样本 | 差值 median | p95 | max |
|---|---|---|---|---|
| SR01 | 2094 | 4e-10 | 5.8e-9 | 0.011 |
| SR11 | 1457 | 2.9e-10 | 5.1e-9 | 2.9e-8 |

路边距离：SR01 前 3000 帧中 1151 个"贴在 Phys_Floors 上"的帧，8 方向射线扇的最小值 50% 小于 0.5、p90 为 4.75；
与网格边界法的相关系数 0.98；"网格边界 < 0.3 而射线扇 ≥ 2"的帧数为 0。这与 `state-spec.md` §3.4 用 16 方向得到的分布一致。

注意要按对象取最小：mindist 是按凸 ledge（凹网格的凸分解块）给的，单个 ledge 的距离大于等于整个对象的距离。
特征计算本身（7×7 高度格 + 8 方向射线扇 + 边界距离 + 栏杆）约 1.5 ms/帧（L1 22351、L11 26966 个静态三角形，构建网格索引 0.03 s）。

## 3. 特征向量（版本 f3）

`FeatureExtractor(FeatureConfig()).compute(tracker)` 返回 `(float32 向量, info)`。`layout()` 给出 (名字, 长度) 列表，`slices()` 给出切片，
`feature_names()` 给出逐元素名。最后整体裁剪到 ±`clip`（10），NaN 置 0。默认配置共 204 维（`hs_block=False` 时 196 维，与 f2 的布局相同）。

| 名字 | 长度 | 含义（归一化） |
|---|---|---|
| valid | 1 | 本帧球状态可信（见上） |
| ball_type | 3 | 木 / 纸 / 石 one-hot |
| vel_cam | 3 | 球速度（speed + speed_change，世界系）转到相机系 / `vel_scale`(20) |
| vel_path | 3 | 球速度在路径系 / 20 |
| speed | 1 | 速度大小 / 20 |
| angvel | 3 | rot_speed + rot_speed_change / 20（**core 坐标系，未验证**） |
| cam_yaw_ref, cam_yaw | 2+2 | `Cam_OrientRef`、`Cam_Orient` 朝向角的 (cos, sin) |
| cam_rotating | 1 | 两者不同，即相机正在转 90° |
| pos_world | 3 | 仅 `absolute_position=True`：世界位置 / 50 |
| geom_contact, fric_contact, n_fric, n_mind, min_gap | 5 | 见 §2（n_fric/4，n_mind/10） |
| normal_cam | 3 | 接触法向（相机系） |
| surface | 5 | none / stone / wood / metal / other |
| partner_moving, partner_rail | 2 | 见 §2 |
| rel_vel_partner | 3 | 与运动对象的相对速度（相机系）/ 20 |
| edge_mesh | 1 | 接触点到可走面边界 / 12（上限 12） |
| edge_fan | 8 | 各方向到路边的距离 / 8（上限 8） |
| rail_dist, rail_dir | 1+2 | 最近栏杆的间隙 / 6、水平方向 |
| heights, height_hit | 49+49 | 7×7、间距 1 的高度格（相对球底，/5，无落点 = −1）和落点标志；顺序为 [right 下标][forward 下标] |
| path_progress | 1 | 参考路径上的进度（0–1） |
| path_remaining | 1 | 剩余弧长 / 1000 |
| path_lateral | 1 | 到路径线段的距离 / 50 |
| path_tangent_cam | 2 | 路径切向在相机系中的 (right, forward) 分量 |
| waypoints | 9 | 接下来 3 个路径节点相对球的位置（相机系）/ 50 |
| bodies | 28 | 最近 4 个运动体（BODY，60 单位内，stopper 除外）：存在标志、相对位置 / 50、相对速度 / 20 |
| lives, points, sr_s, sector | 4 | 命数 / 10、Points / 1000、SR 计时（double 和）/ 100 s、扇区 / 扇区数 |
| transforming, ball_nav | 2 | 正在变球、玩家有控制。变球（f3）= 从活动球（`Ball_*` 实体）的 Unphysicalize（不在 BallOff 之后）到球的下一次 Physicalize，事件参数是物理化记录序号，由关卡消息的 PHYS 记录得到实体（client 先读增量再更新 tracker）。HS03 上正好是 IVP 球体缺失的 4 段（5201–5800、9740–10339、17738–18337、27590–28189，`tests/test_dataset_regression.py`）。f1 用 Activetrafo（`IngameParameter[0,0]`，变球后**不清零**）把第一次变球之后的帧都判成变球；f2 加了"BallNav 不活动"，但暂停期间（HS03 8590–8719 等）和 LevelFinish 之后 BallNav 也不活动，仍然误判；f3 改用事件 |
| depth_margin | 1 | 球心高出正下方死亡盒（DepthTestCubes）顶面的高度 / 50；只作整形用，死亡以 BallOff 事件为准 |
| followers | 1 | 跟随中的分数小球数 / 6（mod 的 SCOR 计数，0.3.1；旧数据从 WTCH 复算：主球不可见、小球可见，`state-spec.md` §2.9）。`hs_block` 控制本行到 since_checkpoint |
| follower_sources | 4 | 各分数球实例跟随中的小球数 / 6，从大到小（WTCH） |
| follower_near | 1 | 最近的跟随小球到球心的距离 / 50（上限 1；没有跟随时为 0） |
| paused | 1 | 暂停中（PauseLevel 到 UnpauseLevel；按键的两帧之后才切换） |
| since_checkpoint | 1 | 距上一次 PreCheckpoint（没有时距第一次 BallNavActive）的帧数 / 10000。还在跟随的小球在下一个检查点作废 |

`FeatureConfig` 的全部字段（网格大小、射线扇方向数、缩放、是否含运动体……）都会写进每个 run 的元数据。
改变已有条目含义时提升 `FEATURE_VERSION`。

**参考路径**（`path.py`）：节点为 `ResetPoints[0]`（起点）、`Checkpoints[k]` 的矩阵位置、`PE_Balloon_Platform`（飞艇平台）。
节点 s−1 → s 是第 s 扇区的线段；投影只在当前扇区 ±1 的线段里找，避免跳到关卡另一处。以后可以换成 TAS 轨迹参考线。

## 4. 奖励（`RewardConfig`）

时间一律取 TIME 段的 SR double 和（BML+ 0.3.12 计时器），不按帧数算。

**SR 模式**（`mode="sr"`），每步：

- 进度 `progress_weight · Δs`（s 为参考路径弧长；后退乘 `backward_scale`）；
- 时间 `−time_weight · ΔIGT(秒)`；
- 势函数整形 `γ·Φ(s') − Φ(s)`，`Φ = shaping_weight · s`，终止状态 Φ = 0；
- 检查点 `checkpoint_bonus`（默认 0）；
- 通关 `finish_bonus`，死亡（BallOff）`−death_penalty`。

**HS 模式**（`mode="hs"`）：

- 分数势 `score_scale · Δ(Points + Lifes × LifeBonus)`：包含时间分的递减、碰到分数球 +100、每个小球追上球时 +20（都在 Points 里，按增量计，不按 ExtraPoint 事件数）、生命球（+200）；在检查点作废的小球不会出现在 Points 里，所以永远不付；
- 通关时加 `score_scale ·`（LevelBonus + 20 × 尾帧结束时仍在跟随的小球数），与正式 HS 读数一致（`benchmarks.md` "HS 读数"）；
- BallOff 时 `−score_scale · LifeBonus`：游戏要在 264 帧后才扣命，那时 RL 回合已经结束，所以在死亡时就记上；
- 可选的势函数整形 `hs_follower_shaping`（默认 0 = 关）：Φ = `hs_follower_shaping · score_scale · 20 · 跟随数`，`γΦ' − Φ`，终止状态 Φ = 0。开始跟随时先付，小球作废时收回，到达时由 Points 的 +20 接上；
- 可选的路径进度整形 `hs_progress_weight`。

这样一个进程内所有回合的 HS 奖励之和 = `score_scale ·`（正式 HS − 开局的 Points + Lifes × LifeBonus），测试里核对了这一点（含 LevelFinish 时还有小球在跟随的情形）。

**暂停不是奖励漏洞**：暂停中游戏不计时、不扣分（游戏规则），HS 奖励为 0；SR 模式的时间项也是 0，所以 SR 模式默认不给 Esc。
为了不让回合停在暂停里：连续暂停帧数达到 `max_pause_frames`（默认 3000）时 env 自己注入一次 Esc 解除暂停（`info["forced_unpause"]`）；
一个回合累计的暂停帧（`info["episode_pause_frames"]`）达到 `max_episode_pause_frames`（默认 20000）后，env 解除当前暂停，并且不再允许暂停
（Esc 只能解除暂停，collect 宏变成空动作），所以策略不能靠反复暂停拖住回合。暂停帧也计入 `max_episode_frames` / `max_process_frames`（默认不设）。

## 5. 回合、进程与录制

- **一个游戏进程 = 一个 run**（同进程重载在完成关卡后不能逐字节复现）。
- **每条命一个 RL 回合**：BallOff 终止回合；下一次 `reset()` 用显式的空闲帧（`idle_keys`，默认 0）快进死亡动画，
  `stop_event_mask = BallNavActive`，同一进程继续。GameOver、LevelFinish、帧数上限、截断、`close()`、带 seed 的 `reset()`、
  或在一条命中途 `reset()` 时，结束进程并启动新进程。
- **开局**：快进到 StartLevel；第 1 关交替按 Q / 空帧直到 CamNavActive（跳过教程，`skip_tutorial`）；再快进到 BallNavActive。
  这些帧都进入帧日志。可以用 `prefix`（`BALLRAIN_PREFIX`，mod 先按回放方式注入）从 TAS 前缀交接。
- **动作**：`Discrete(9)`（无键、四个方向、四个斜向），`camera_actions=True` 时再加 shift+左 / shift+右。每步注入 `action_repeat` 帧，
  delta 固定为 `delta_ms`（默认 f32(1000/132)，每帧一个 PSI），遇 BallOff / LevelFinish / GameOver 提前停止，未执行的帧作废。
- **Esc**（`esc_action`，HS 模式默认开，SR 模式默认关）：一次按下 = 第 1 帧 Esc、之后松开，共 `esc_frames`（默认 4，至少 3）帧。
  游戏在按下的那一帧还跑物理，之后冻结，直到下一次按下（含该帧），再下一帧恢复；PauseLevel / UnpauseLevel 事件在按下后 2 帧到达，
  `paused` 只看这两个事件（HS 记录里有 2/147 次按键没有切换）。没有控制时（不是暂停中，且 BallNav 不活动、死亡、变球、通关）Esc 被屏蔽为空帧（`esc_requires_control`）。
  暂停中方向键按 `mask_keys_while_paused` 换成 `pause_keys`（默认 0）。
- **collect 宏**（`collect_action`，默认关）：有小球在跟随时，Esc → 每 `collect_chunk` 帧检查一次，直到没有跟随的小球或等满 `collect_max_frames` 帧 → Esc；
  否则是一个空动作。`info["collect"]` 给出前后的跟随数和等待帧数。最佳做法（用户）：飞艇和每个检查点之前按 Esc 收集。
- 所有 Esc 和暂停帧都是普通注入帧：进入帧日志和导出的 .tas，参考 TASSupport 回放时暂停也会重现（M1-B 的 HS 用例）。
- **超时**：mod 等动作的超时 `timeout_ms` 默认 30 分钟，避免 PPO 更新期间触发。

**录制**（每个进程一份，`<out_root>/<copy>/<run_id>/`）：

- `<run_id>.tas`：legacy .tas，内容是 mod 写的 `BALLRAIN_FRAMELOG`（实际注入的全部帧，含开局和前缀，止于 LevelFinish 那一帧；SR 尾帧不写入）。
  Python 自己记录发出且被执行的帧，结束时与 mod 的日志逐字节比较，结果写入 `framelog_match`；不一致时记录第一处差异，导出的仍是 mod 的日志。
- `<run_id>.json`：关卡、模式、seed、拷贝、前缀（路径、帧数、sha256）、mod 版本、特征和奖励配置、每条命的边界
  （`start_frame` / `end_frame` 用 frames_done 口径，结束原因、SR、分数）、全部事件、结束原因，以及：
  - `sr_ms`：通关时的正式 SR = mod 的 double 和（含 LevelFinish 之后到 CounterInactive 的尾帧）截断到毫秒；
  - `sr_mod_at_levelfinish`：LevelFinish 那一帧的 double 和（mod 的 SREN 段）；
  - `sr_python_at_levelfinish`：Python 按 BML+ 0.3.12 规则在导出帧和事件上复算的值（StartLevel 清零；CounterActive / UnpauseLevel 开始；
    CounterInactive / PauseLevel 暂停；运行中每帧在该帧事件之后加 `(double)` 的 f32 delta），应与上一项完全相等；
  - `hs`：**正式 HS = 结算画面的分数** = Points + 20 × 仍在跟随的小球 + Lifes × LifeBonus + LevelBonus，取 SR 尾帧结束时（最后一条观测）的值
    （`hs_parts.source = "tail_end"`；`benchmarks.md` "HS 读数"）。`hs_parts` 还记 LevelFinish 钩子的快照（`levelfinish_hook`，不含小球）和钩子时的跟随数，
    两者 + 20 × 跟随数不等于正式值时写进 `notes`。`hs_mod` 是钩子快照。HS 奖励的终止项也用正式读数。
- 验收按 M0 规则在门禁阶段做：参考 TASSupport 回放 3 次，与多数结果逐字节相同即通过。

## 6. TAS 数据集（`dataset.py`、`scripts/build_tas_dataset.py`）

每条 TAS 用"回放策略"（记录自己的帧，每个动作 1 帧）在 agent 模式下跑一遍，保存每帧的原始状态：

- `raw.npz`：`obs_bytes`（全部 OBS1 消息首尾相接）、`obs_offsets`、`frames_done`、`record_frames`、`framelog`，
  以及从原始状态解析出的便利列（球位置、速度、类型、SR、mindist 数和最小间隙 / 法向 / 对象、接触数、运动体数）；
- `level.msg`、`level.msg.inc`、`trace.txt`、`events.txt`；
- `meta.json`：帧日志是否等于记录的前缀、结束原因、SR、HS 等。

原始字节是唯一的真值，解析器修正后可以用 `load_observations()` 重新解析，不需要重跑游戏。这一步要启动游戏，由门禁阶段执行。

## 7. 多实例

`scripts/make_copies.py` 把 `runtime/train-00` 复制为 `train-01..07`（不复制日志和启动脚本），并核对 physics_RT.dll 的 2002 版哈希、
以及全部二进制文件与源拷贝的 sha256（BallrainIO.bmodp 除外，它经常重编）；`--sync-mod` 把新编译的 BallrainIO.bmodp 装进所有拷贝
（正在运行的拷贝会被拒绝）。各拷贝的 mod 不一致时脚本报错退出。
`vecenv.make_vec_env(config, n=8)` 每个子环境用一个拷贝（gymnasium AsyncVectorEnv，spawn）；`make_sb3_vec_env` 给 stable-baselines3。

## 8. 未验证 / 待测

- 角速度的坐标系；`q_core_f_object` 的旋转约定（样本里都是单位四元数）。
- 纸球的几何（多面体 × 每帧姿态）尚未实现，纸球只用 mindist。
- 风扇（P_Modul_18）的风力没有导出，也没有特征。
- 俯瞰（space）时的相机特征。
- 第 1 关跳过教程的 Q 脉冲在 132 fps 下何时生效，只在模拟器里测过。
- 参考路径是检查点折线，弯道处进度只是粗略估计。
- 死亡快进期间（BallOff 到 BallNavActive）吃到的分数小球不计入任何回合的 HS 奖励（只在最终 HS 里）。死亡时仍在跟随的小球会怎样，没有测（游戏机制问题，待问用户）。
- 暂停菜单里方向键会不会移动菜单选项、死亡或变球期间按 Esc 的效果，没有测；env 默认在暂停中屏蔽方向键，在没有控制时屏蔽 Esc。
- 游戏进程崩溃或卡死（PeerGone / Timeout）时当前回合截断，帧日志以 Python 侧为准，下次 `reset()` 重开进程。

## 附录：线格式中 `agent-protocol.md` 没写明的约定

这些约定写在 `ballrain/protocol.py` 的模块说明里，并已与 `BallrainIO/src/Agent.cpp`（逐字段写入的 `Wire` 写法）逐段对照：

- A1 全部小端、**紧凑**：字段之间、数组元素之间都没有填充。
- A2 消息头 24 字节：u32 magic、u16 version、u16 flags、u32 byte_len（含消息头）、u32 frames_done、u32 section_count、u32 reserved。
  OBS1 的 flags：bit0 FINAL（本 run 最后一条）、bit1 PREFIX、bit2 DROPPED（有段放不下被丢弃）、bit3 MINIMAL。
- A3 段头 12 字节；payload_len 不含填充，下一段从 4 字节对齐处开始；读方允许段尾有多余字节（字段只追加），跳过不认识的段。
- A4 列表段以 u16 n 开头（TRAC 是 u32 n）；PHYS 段的元素是 u32 物理化记录序号。
- A5 "12×f32" 矩阵见 §1；四元数 xyzw。
- A6 MIND 元素总是带两个三角形（18×f32），`n_tris` 说明有效个数；没有三角形时 ledge = 0、tri_index = 0xFFFF、edge_index = −1。
- A7 ACT1 头 36 字节；事件码 c 对应停止掩码第 c 位；`obs_mask` 置位表示**关闭**该段：bit0 BODY、1 MIND、2 FRIC、3 CAMS、4 GLCT、5 WTCH（mod 另有 6 BALL、7 ARRS）。
- A8 关卡消息每段以 u32 n 开头；字符串是本消息 STRS 表的下标（0xFFFFFFFF = 无），STRS 可以在任意位置（mod 写在最后）；
  全量文件和 `.inc` 里追加的每条消息各有自己的 STRS。ENTS 的空记录 ckid 为 0，读方丢弃。
- mod 的扩展（读方可选）：TIME 段末尾追加 f32 SR 的 float32 和、u32 原始帧计数（含尾帧）、u32 尾帧数；
  每帧的 `SCOR` 段（HS 快照、BallOff 次数、最近一次 BallOff / BallNavActive 的帧号）；最后一条观测（以及 LevelFinish 之后）的 `SREN` 段
  （最终 SR double 和、float32 和、`GetSRScore()`、尾帧数、是否通关、是否见到 CounterInactive、LevelFinish 时的 SR 和 HS 快照）；
  关卡消息的 `EMSH`（实体当前网格）和 `LTAB`（相机 ckid、接触 BB 列表）。
