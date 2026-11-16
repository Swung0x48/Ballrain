# 游戏状态清单与读取规范

本文列出 RL agent、奖励和分析可能用到的全部游戏状态。每项写明：是什么、为什么需要、在哪里、怎么读、频率、代价，以及读取会不会影响游戏。
IPC 契约见 `agent-protocol.md`。

标注：

- **[实测]**：用临时探针 mod 在 2002 版 physics_RT.dll（sha256 e72e4afc…）上读出并核对过。
- **[源码]**：只来自重建源码（Ballanced）或 BML+ 源码，未在原版上核对。
- **[待测]**：需要以后用探针再确认。

## 0. 原则

1. **mod 只导出原始数据**。每帧导出一个带版本号的原始状态超集，每关额外导出一次静态消息（几何和对象表）。
   所有特征和奖励都在 Python 里算。改观测不需要重编 mod，也不需要重跑保真门禁。
2. **读取不能改变游戏状态**。这一点已在 5 条记录上验证（见 §0.1）。
3. **原始 IVP 偏移只对 2002 版 DLL 有效**。读取前先过 sha256 闸门，这与 BallrainIO 的 `VerifyPhysicsDll` 相同。
   所有原始内存读都包在 SEH（`__try`/`__except`）里，并且只做 `memcpy`：不调用任何 IVP 函数，不写内存，不分配物理对象。

### 0.1 无扰动验证 [实测]

探针 mod 由 BallrainIO 0.2.0 的回放路径（LOADS=1）加上全部只读 dump 组成。它每帧做以下读取：

- 遍历球的 friction / exact / invalid synapse 链表；
- 解码碰撞三角形；
- 遍历 IVP cluster 中的全部物体；
- 比较全部数据数组；
- 读取相机和被监视对象的矩阵，以及声音 BB 的 local/output 参数。

在 SR01_1_14_457_264、SR06_1_36_163_132、SR11_2_00_382_132、HS03_4906_1_56_934、SR09_1_29_321_264 上，
探针输出的 trace 与 M0 BallrainIO 的 trace **逐字节相同**。SR09 含 4 次 BallOff 和 GameOver。

读取 CK 对象时要注意以下几点：

| 风险 | 规则 |
|---|---|
| `CKParameter::GetValue / GetReadDataPtr / GetStringValue` 的默认参数是 `update=TRUE`，会拉动参数运算（parameter operation）求值 | 一律传 `FALSE`。`CKDataArray` 的 PARAMETER 列：`GetElement(r, c)` 存的是该单元的 `CKParameterOut` 指针，BallrainIO 用 `GetReadDataPtr(FALSE)` 取原始数据（最多 64 字节，`GetDataSize()`），不取字符串；探针当时用的是 `GetStringValue(buf, FALSE)`，两种都不拉动运算 |
| 按下标遍历实体的全部属性（`GetAttributeType(i)` + `GetAttributeParameterByIndex(i)`） | **禁止**。在 L11 的 `P_Modul_37_Bridge` 上会让游戏 fast-fail 退出（0xC0000409）。只按已知属性名取 `GetAttributeTypeByName` + `HasAttribute` + `GetAttributeParameter` |
| `GetStringValue(nullptr)` 返回的长度可能不可靠 | 字符串缓冲区开到 64 KiB，或只对 int/float/bool 类型取字符串，其余读原始字节 |
| `CK3dEntity::GetBoundingBox` 可能重算缓存 | 不用它，包围盒用网格顶点自己算 |
| `GetObjectsListByClassID(CKCID_3DENTITY)` 只返回精确类，不含 3DOBJECT | 枚举实体用 `GetObjectListByType(CKCID_3DENTITY, TRUE)` |

代价：探针用文本格式每帧约 8 KB，一关 95–226 MB；回放耗时从 2–4 s 变为 21–43 s，主要花在格式化和全数组比较上。
二进制观测（`agent-protocol.md`）每帧约 1–3 KB，单帧读取开销应远小于 100 µs [待测]。

## 1. 时间语义

| 事实 | 内容 | 状态 |
|---|---|---|
| 帧内顺序 | Time PreProcess（注入 delta）→ Input PreProcess（注入按键）→ 行为脚本 → `ModManager::PostProcess` 先调 `PhysicsPostProcess()`（IVP 模拟），再调各 mod 的 `OnProcess` | 0.3.12 源码 + 实测 |
| 观测点 | `OnProcess` 看到的是本帧物理之后的状态 | 实测 |
| **帧号口径** | 本文有两种编号。"下标 i"指帧数组（.tas 载荷）的 0 起下标。"事件号 / 帧 n"指 trace、事件和探针记录用的编号，等于该帧执行后已注入的帧数，即 n = i + 1。例如下标 140 的帧执行时发生的事件记为事件号 141 | 实测 |
| 物理时钟 | 物理时间 = Σ delta_ms × time factor。time factor = 管理器 +0xD0 = 0.002（加载期间为 0.001）。PSI 间隔固定为 1/66 s（物理时间） | 实测 |
| PSI 频率 | 132 fps（delta = f32(1000/132) = 7.5757575 ms）时每帧恰好 1 个 PSI；264 fps 时每 2 帧 1 个；240 fps 时不整除 | 实测 |
| 两种位姿 | `OnProcess` 时 CK 实体矩阵是 IVP 插值到 env current_time 的位姿；132 fps 时比上一个 PSI 超前半个 PSI。IVP core 的 pos/q 是上一个 PSI 的值。接触、mindist 也都是 PSI 时刻的值 | 实测 |
| 与 IVP 比较时 | 用 PSI 位姿（core），不要用 CK 矩阵。否则插值误差会被当成不一致：运动物体用 CK 矩阵时 median 误差 0.007、p95 0.07；用 core 位姿时为 1e-5 / 0.011 | 实测 |

## 2. 状态清单

列含义：**用途** S = SR，H = HS，C = 控制/观测，Z = 安全/终止；**频率** F = 每帧，L = 每关一次，E = 事件；**读法**中的偏移均指 2002 版 DLL。

### 2.1 球

| 项 | 用途 | 位置与读法 | 频率 | 代价 / 副作用 |
|---|---|---|---|---|
| 当前球实体 | C | `Gameplay_Ingame` 脚本的 local 参数 `ActiveBall`，用 `GetValueObject`（local 无运算，安全）；`CurrentLevel[0,1]` 也有 | F | 极小 / 无 |
| 球类型 | C | 实体名 `Ball_Wood` / `Ball_Paper` / `Ball_Stone`。IVP 类型：木球、石球为 3（IVP_BALL，半径 2），纸球为 2（凸多面体，315 顶点 / 320 面，最大半径 2.115，平均 2.004） | F | 极小 |
| 球物理参数 | C | `Physicalize_GameBall` 数组：木球 friction 0.8 / elasticity 0.2 / mass 1.9 / 线阻尼 0.9 / force 0.43；纸球 0.5 / 0.4 / 0.2 / 1.5 / 0.065；石球 0.5 / 0.1 / 10 / 0.3 / 0.92。也可从 OnPhysicalize 记录取 | L | 无 |
| 世界矩阵（插值） | C | `CK3dEntity::GetWorldMatrix()`，与 trace 相同 | F | 极小 / 无 |
| IVP 实物体 | — | CKIpionManager +0x28 是 `IVP_U_Vector<IVP_Real_Object>`（ushort memsize, ushort n, void** elems）。其中元素 `client_data`（+0xB0）== 球实体的那一个就是球。也可遍历 cluster 链表得到全部物体（§2.5） | F | 小 / 无 |
| core 指针 | — | 实物体 +0xA4 `physical_core` | F | — |
| PSI 位姿 | C | core +0xB8 `pos_world_f_core_last_psi`（3×f64），+0xE8 `q_world_f_core_last_psi`，+0x108 `q_world_f_core_next_psi`（各 4×f64，xyzw），+0x128 `m_world_f_core_last_psi`（3 行 ×4 f64 + vv），+0x68 `time_of_last_psi`（f64） | F | 小 / 无 |
| 线速度 | C | core +0xA4 `speed` + core +0x84 `speed_change`（各 3×f32，世界系，单位/秒）。TASSupport 的 `GetVelocity` 就是这两项之和 | F | 小 |
| 角速度 | C | core +0x94 `rot_speed` + core +0x74 `rot_speed_change`（3×f32，core 坐标系，rad/s）[源码：坐标系] | F | 小 |
| 速度交叉校验 | — | 有限差分：`(pos_k − pos_{k−1}) / (tpsi_k − tpsi_{k−1})` 等于**上一个** PSI 的 `speed`（显式欧拉，median 误差 2e-8）。与本 PSI 的 speed 差 0.18–0.22（这是加速度造成的） | — | 实测 |
| 其它 core 字段 | C | +0x1A8 `rotation_axis_world_space`，+0x1B8 `current_speed`，+0x1BC `abs_omega`，+0x14 `rot_inertia`（.w = 质量），+0x70 `i_delta_time`（=66）；+0x60 一个 u32 位域，低字节是 `movement_state`：1 运动、8 NOT_SIM（冻结）。+0x00 的位 2–3 是 `physical_unmoveable` | F | 小 |
| 半径 | C | 实物体 +0xA0 `extra_radius`：木球、石球为 2.0，纸球为 0 | L | — |

### 2.2 相机 [实测]

相机对象有 `Cam_OrientRef`、`Cam_Orient`、`Cam_Target`、`Cam_Pos`、`Cam_MF`、`InGameCam`（类型为 CKCamera，挂在 `Cam_MF` 下）。当前渲染相机可由 `GetPlayerRenderContext()->GetAttachedCamera()` 得到，即 `InGameCam`。

| 项 | 用途 | 读法 | 频率 |
|---|---|---|---|
| 相机朝向（离散） | C | `Cam_OrientRef` 世界矩阵第 2 行（z 轴）的 yaw，总是 90° 的整数倍 | F |
| 旋转进行中 | C | `Cam_Orient` 的 yaw 平滑插值。SR01 下标 6276 的帧按下 shift+right，之后约 70 帧（264 fps，约 0.27 s）`Cam_Orient` 从 180° 转到 90°（事件号 6277 起），`Cam_OrientRef` 在转完时才跳到 90°（事件号约 6350）。判据：`yaw(Cam_Orient) ≠ yaw(Cam_OrientRef)` | F |
| 相机位置 / 目标 | C | `Cam_Pos`、`Cam_Target`、`InGameCam` 的世界矩阵（每个 12×f32） | F |
| 俯瞰（space） | C | 本次样本没有用到 space。预计表现为 `Cam_Pos` 相对 `Cam_Target` 抬高 [待测] | F |

键盘导航方向由相机朝向决定，所以 Python 应把动作和观测都放到相机坐标系里。

### 2.3 控制状态

| 项 | 用途 | 读法 | 频率 |
|---|---|---|---|
| BallNavActive / Inactive | C | BML 事件 `OnBallNavActive/Inactive` | E |
| CamNavActive / Inactive | C | BML 事件 `OnCamNavActive/Inactive` | E |
| 激活阶段 | C | `CurrentLevel[0,4]`（"Activation Phase?"，bool） | F |
| 教程进行中 | C | `IngameParameter[0,3]`（"Tutorial activ?"）。SR01 在事件号 272 变为 FALSE | F |
| 第 1 关前奏 | C | SR01：下标 140 的帧按 Q（bit 6）跳过教程，下标 141 按 shift+left；CamNavActive 为事件号 142（即下标 141 那一帧），BallNavActive 为事件号 936。其它关卡没有 Q，BallNavActive 约为事件号 515（132 fps） | E |

### 2.4 物理时钟

| 项 | 用途 | 读法 | 频率 |
|---|---|---|---|
| IVP_Environment | — | 管理器 +0xC0 | — |
| current_time / time_of_next_psi / time_of_last_psi | C | env +0x120 / +0x128 / +0x130（f64）。`cur − last` 在 132 fps 时恒为 0.007576 | F |
| 平滑 delta / 物理步长 / time factor | C | 管理器 +0xC8（ms）、+0xCC（= delta × factor，秒）、+0xD0（0.002） | F |
| base_time | — | *(env+0x4) +0x18（f64）。BallrainIO 在开局时会写 0，只读不改 | F |
| IVP_RAND_SEED | — | DLL 基址 +0x685B4（只读） | L |

### 2.5 所有刚体

**发现物体**：从球的实物体出发，沿 +0x08 `next_in_cluster` 和 +0x0C `prev_in_cluster` 遍历 cluster 链表。
跳过 `object_type`（+0x04）== 1 的 cluster 节点。`client_data` 必须在 CK 实体集合内才算有效。L1 共 46 个物体（含静态）[实测]。

**物理化记录**：BML 的 `OnPhysicalize` 在原 BB 执行**之前**触发，这时 IVP 物体还不存在，参数里的网格数组在回调返回后释放。
所以回调里只能拷贝实体 ID、全部参数和网格 ID，在下一帧的 `OnProcess` 再导出网格和查找 IVP 物体 [实测]。
地面在 PreLoadLevel 后约第 20–60 帧物理化（L11 是第 22 帧，早于 StartLevel）。机关在扇区激活时物理化，重置时 Unphysicalize 后再物理化。
SR11 共 290 次 PHYS、306 次 UNPHYS。

| 类别 | 内容 | 读法 / 导出 | 频率 |
|---|---|---|---|
| 静态地面 | 组 `Phys_Floors`、`Phys_FloorRails`（两者都是 friction 0.7 / elasticity 0.3，碰撞组 `Floor`）、`Phys_FloorStopper`（碰撞组 `Ball`，与玩家球同组）。stopper **不与玩家球碰撞** [实测]：
5 条记录里玩家球离 stopper 最近时间隙只有 0.23（SR11 的 `A01_Floor_Stopper`），但始终没有对它的 mindist 或接触；而可碰撞物体在约 1.2 以内就会有 mindist。
几何计算必须排除 stopper。全部作为 concave 网格物理化 | 一次性导出：实体 ID、世界矩阵、网格顶点（局部坐标）和三角形。L1/L3/L6/L11 分别为 22351 / 17501 / 15799 / 27214 个静态三角形。CK 网格乘以 CK 世界矩阵后，与 IVP mindist 距离一致（median 6e-8） | L |
| 地面材质（声音类） | 属性 `Continuous Contact ID`：1 = 石面（`Sound_RollID_01`），2 = 木面（`Sound_RollID_02`），3 = 金属 / 栏杆（`Sound_RollID_03`）。另有属性 `Coll Detection ID`（`Sound_HitID_01..03`，部分机关为 4）和属性 `Floor`（结构体，读原始字节）。`BallSound` 数组按 球 × {onStone, onWood, onMetal} 给出音效名 | 按属性名读 | L |
| 运动体 | P_Modul_*（桥、翻板、推块、秋千、箱子…）、P_Box、P_Ball_*（场景小球）、P_Dome（fixed）、PE_Balloon 平台。物理化时的 friction / mass / 阻尼 / 凸包网格 ID 见 OnPhysicalize（例如 `P_Modul_37_Bridge` 由 3 个凸网格组成，mass 3，startFrozen） | 网格一次性导出（局部坐标）；位姿每帧导出：core +0xB8 / +0xE8 的 PSI 位姿，加上实物体 +0x2C `q_core_f_object`（指向 4×f64）和 +0x30 `shift_core_f_object`（4×f32）。物体到世界的变换：`R_core·(R(q_cfo)·p + shift) + vv`。速度取 core +0xA4 / +0x94 | F（只发球附近，例如 60 单位内） |
| 运动体验证 | 用全部碰撞网格加上述 PSI 变换，球到网格的距离与 mindist 一致：median 1e-5，p95 0.011（SR11，1034 个样本）。直接用 IVP ledge 三角形（§3.2）时 median 1e-9 | — | 实测 |

### 2.6 扇区、检查点、复活点 [实测]

| 项 | 用途 | 读法 | 频率 |
|---|---|---|---|
| 当前扇区 | C S H | `IngameParameter[0,1]`（Activate Sector），`[0,2]`（Deactivate Sector） | F（变化时） |
| 当前复活点 | C Z | `CurrentLevel[0,3]`（CurrentResetpoint，4×4 矩阵）。检查点触发的同一帧会更新 | F |
| 全部复活点 | C | `ResetPoints` 数组（每扇区一个矩阵） | L |
| 检查点 | C S | `Checkpoints` 数组（矩阵 + 对象，例如 `PC_TwoFlames_MF`）；事件 Pre/PostCheckpoint（相隔 8–10 帧） | L + E |
| 起点 / 终点 | C | `PS_FourFlames_*`（起点）、`PE_Balloon_*`（终点飞艇平台，本身是运动体，会与球接触） | L |
| 占位表 | C H | `PH` 数组（行 = 占位：Sector、Type Name、Matrix、Object、Set?）和 `PH_Groups`（类型、数量、激活和重置规则）。扇区激活时，`PH[r,3]` 会改成实际实例名 | L + F |

### 2.7 死亡（BallOff）[实测：SR09，4 次死亡]

| 项 | 用途 | 读法 | 频率 |
|---|---|---|---|
| BallOff | Z | BML 事件 `OnBallOff`。**以此为准** | E |
| 死亡体积 | Z | 组 `DepthTestCubes`（L9 有 `Quader01..04`，不可见盒子），世界包围盒用网格顶点算（例如 L9 的 Quader02：x −129…548，y −232.7…−111.9）。`Gameplay_Ingame` 的 local `maxDepth`（结束时 L1 = −329.4，L9 = −432.7）是 `DepthTest` 组物体（场景小球、箱子）的掉落重置阈值，与玩家球无关 [源码推断] | L |
| 判定 | — | **规则（用户确认）：球与关卡中的死亡盒（`DepthTestCubes`）相交即判死。** 实测：SR09 的 4 次 BallOff 帧（事件号 13373、15015、16711、18348）球心比正下方盒子的顶面高 2.2–3.0（球底高出 0.2–1.0），说明参与相交测试的不是半径 2 的球面本身（具体用哪个包围体待向用户确认，不另做探针）。奖励整形可以用"球离死亡盒顶面的高度"，终止只用事件 | — |
| 死亡后续 | Z S | BallOff 同帧有 CounterInactive（SR 计时暂停）。264 帧后（264 fps 下 1.0 s）`Energy[0,1]` 减 1；686 帧后 PostSubLife；再过约 380 帧有 CounterActive 和 BallNavActive。命数为 0 时，最后一次 BallOff 后第 2 帧 GameOver | E |

### 2.8 变球器（trafo）

| 项 | 用途 | 读法 | 频率 |
|---|---|---|---|
| 变球器对象 | C | 组 `All_P_Trafo_Paper / _Stone / _Wood`。目标球类型 = 组名或对象名（如 `P_Trafo_Paper_MF`）。位置见 `PH` 中 Type 为 `P_Trafo_*` 的行 | L |
| 正在变球 | C | 事件序列（HS03 第一次变球）：5200 BallNavInactive，5201 旧球（`Ball_Wood`）Unphysicalize，同时 `IngameParameter[0,0]`（Activetrafo）变为该 trafo（`P_Trafo_Paper_MF`），5799 ActiveBall 改变，5800 新球 Physicalize，5801 BallNavActive。**变球期间旧球已被 Unphysicalize，没有 IVP 物体**（`agent-protocol.md` §8.7；探针当时读到的"冻结、速度为 0、没有 mindist 的旧球"是已释放的内存）。Python 的判据：从活动球（`Ball_*` 实体）的 Unphysicalize（不在 BallOff 之后）到球的下一次 Physicalize。Activetrafo 在变球结束后**不清零**，暂停和 LevelFinish 之后 BallNav 也不活动，所以两者都不能单独当判据 | E |
| 新球 | C | `ActiveBall` 改变；新球会有一条 OnPhysicalize 记录 | E |

### 2.9 分数球、生命球 [实测：HS03]

| 项 | 用途 | 读法 | 频率 |
|---|---|---|---|
| 位置 | H | `PH` 中 Type 为 `P_Extra_Point` / `P_Extra_Life` 的行（扇区 + 矩阵 + 实例名） | L |
| 分数球被吃 | H | 碰到主球时 `Energy[0,0]` +100，同一帧主球 `P_Extra_Point_Ball0[<nnn>]` 变为不可见，下一帧有 ExtraPoint 事件；随后 6 个小球 `P_Extra_Point_Ball1..6[<nnn>]` 跟着玩家球，追上的那一帧该小球不可见且 `Energy[0,0]` +20，下一帧有一次 ExtraPoint 事件（同一帧到达多个小球时只有一个事件，所以按 Points 的增量计分，不按事件数）。HS03：主球在事件号 2549，6 个小球在 3734–3771。实例是按扇区复用的（HS01 只有模板和 001–003 四个实例）。规则（用户给定）：按 Esc 暂停时小球会飞到球上被收集，计时和扣分都停止 | E + F |
| 跟随中的小球 | H | **判据**：某实例的主球不可见、它的小球仍可见 = 正在跟随（不需要历史）。在 TAS 数据集上：增加总是 6 的倍数（命中）；HS01_4738 上逐帧与按命中历史累计的计数一致，每次减少都在同一帧伴随 Points +20 × 减少数；最后一帧的个数正好等于 HS 文件名分数与 LevelFinish 读数之差 / 20（HS04 除外，它还在检查点丢球）。检查点：PreCheckpoint 之后 2 帧仍在跟随的小球全部消失，不计分（规则：到下一个检查点时还在跟随的小球作废）。LevelFinish 时仍在跟随的小球在之后约 50–340 帧内陆续到达并计分（post-finish 探针，`benchmarks.md` "HS 读数"）。mod 每帧在 SCOR 里给出这个计数（0.3.1），Python 另从 WTCH 复算（按实例的个数、最近小球的距离） | F |
| 生命球被吃 | H | 事件 PreLifeUp，约 140 帧后 PostLifeUp；`Energy[0,1]` +1；`P_Extra_Life_Sphere` 等子对象变为不可见 | E |
| 已收集状态 | H | 按 PH 实例累计事件，加上可见性。死亡后分数球不会再出现，生命球会重新出现（benchmarks.md） | Python 派生 |

### 2.10 分数与计时

| 项 | 用途 | 读法 | 频率 |
|---|---|---|---|
| Energy 行 | H | `Energy[0,0]` Points（时间分计数器，每 500 ms IGT 减 1，含已吃分数球），`[0,1]` Lifes，`[0,2]` StartPoints = 1000，`[0,3]` StartLifes = 3，`[0,4]` Timefactor（500 ms），`[0,5]` LifeBonus = 200。开局时 Lifes = 3 到事件号 104–109 才写入 | F |
| 关卡奖励 | H | `AllLevel[L−1,6]` LevelBonus = 100·L。同一行还有 StartBall、Sky、Music | L |
| HS | H | Points + Lifes × LifeBonus + LevelBonus。读取时机见 benchmarks.md（LevelFinish 之后会结算清零） | Python |
| SR 计时 | S | BML+ 0.3.12 规则：StartLevel 清零；CounterActive / UnpauseLevel 开始；CounterInactive / PauseLevel 暂停；运行中每帧在 PostProcess 加 `(double)GetLastDeltaTime()`，**发生事件的那一帧也算**。已逐帧核对：`(float)` 这个 double 和 == `IBML::GetSRScore()`，在 SR01、HS03 全部帧上完全相等（HS03 有 4 次暂停）。正式值为 `int64` 截断 | F |
| SR 结尾 | S | LevelFinish 后 3 帧 CounterInactive（30 条到达 LevelFinish 的记录全部是 3 帧，132 / 240 / 264 fps）。回放在 LevelFinish 就停了；agent 模式见 `agent-protocol.md` §5 | E |
| 结算 | H | post-finish 探针（HS01_4738、HS03、SR05、HS12、HS04_esc）：CounterInactive 之后 Points 不再变化，只有仍在跟随的小球到达时 +20；LevelFinish 后约 2900–3200 帧（L12 约 5500 帧后 PreEndLevel）开始结算：Points 逐帧扣到 0，然后每约 148 帧扣一条命（PreSubLife / PostSubLife）。结算画面的分数没有存在任何数据数组里（窗口内全部数组的变化只有 Energy、CurrentLevel、IngameParameter、LifeBalls），按 Points + Lifes × LifeBonus + LevelBonus 在结算开始时的值重建 | E + F |

### 2.11 事件（BML 钩子）[实测]

PreLoadLevel、PostLoadLevel、StartLevel、CamNav*、BallNav*、CounterActive/Inactive、Pause/UnpauseLevel、BallOff、Dead、Pre/PostSubLife、Pre/PostCheckpoint、ExtraPoint、Pre/PostLifeUp、LevelFinish、GameOver、Pre/PostEndLevel、PreResetLevel、PreExitLevel，
再加上 OnPhysicalize / OnUnphysicalize。事件的帧号与 trace 帧号是同一口径（Input 回调推进之后的帧号）。

## 3. 球与路面的相对关系

历史上的难点是：每帧都要知道球是否在路面上、是否接触、离路边多远、接触法向是什么、接触的是哪种表面，运动体上也一样。
我们用探针比较了三个来源，结论是：**IVP 内存给出物理真值；几何可以在 Python 里精确复算并补出"离路边多远"；游戏逻辑只用来交叉核对表面类别。**

### 3.1 探针实验

记录：

| 记录 | 关卡 | fps | 有球帧 | 内容 |
|---|---|---|---|---|
| SR01_1_14_457_264 | L1 | 264 | 19653 | 木球，栏杆、木板、箱子 |
| SR06_1_36_163_132 | L6 | 132 | 12690 | 木球 + 纸球，移动桥 `P_Modul_25_Bridge` |
| SR11_2_00_382_132 | L11 | 132 | 16253 | 木球 + 纸球，移动桥、翻板、推块、秋千、箱子 |
| HS03_4906_1_56_934 | L3 | 240 | 28654 | 木、纸、石三种球，翻板 `P_Modul_29_Platte*` |

每帧导出以下数据：

- 球的 friction synapse，每个附带 contact point 的 0xA0 字节，以及 `tmp_contact_info` 的 0xC0 字节；
- exact synapse，每个附带 mindist 的 0x90 字节，以及对方特征三角形（物体坐标）；
- 对方物体的 core 位姿；
- 声音 BB 的数据；
- 球的 IVP core。

几何由 OnPhysicalize 时导出的 CK 网格在 Python 里计算。

### 3.2 来源 (1)：IVP 物理真值（2002 版 DLL 布局，按不变量核实）[实测]

**friction synapse 和 contact point**：处于摩擦（滚动/静止）接触时才有。

- 实物体 +0x28 `friction_synapses` 是链表头。`IVP_Synapse_Friction` 共 0x14 字节：
  +0 next，+4 prev，+8 `l_obj`，+0xC short `contact_point_offset`，+0xE status 低字节，+0x10 edge。
- contact point = synapse + offset（offset 为 −8 或 −0x1C）。两个 synapse 分别在 cp+0x08 和 cp+0x1C，对方的 `l_obj` 就是接触对象。
  对方 status 的取值：0 = 点、1 = 边、2 = 三角面、3 = 球。
- cp +0x44 `real_friction_factor`：球摩擦 × 地面摩擦，例如 0.8 × 0.7 = 0.56。
- cp +0x58 `last_gap_len`：间隙，median 0.0136，范围 0.0098–0.026。
- cp +0x5C `slowly_turn_on_keeper`（short，初值 20）；cp +0x60 `cp_status` 低字节。
- **原版 contact point 里没有 `last_contact_point_ws` / `surf_normal_ws`**：在 0xA0 字节里找不到单位向量。
  法向和接触点在 cp +0x40 `tmp_contact_info` 指向的 `IVP_Impact_Solver_Long_Term` 里：
  +0x00 `surf_normal`（3×f32，单位向量，方向从球指向对方，即几何法向的反方向），+0x20 `contact_point_ws`（3×f64）。
  不过这块内存是事务内存，约 0.6% 的帧法向不是单位向量，约 20% 的接触点离球心不是 2，所以只能作辅助。
- 栏杆上常见 2 个同时存在的接触（两根栏杆）。接触数分布（L1）：0 个 6755 帧，1 个 12322 帧，2 个 546 帧，3–4 个 30 帧。

**exact synapse 和 mindist**：球附近每个凸 ledge 一个，间隙在约 1.2 以内都有，接触前后都有。

- 实物体 +0x20 `exact_synapses`（+0x24 是 invalid）。`IVP_Synapse_Real` 共 0x1C 字节：
  +0 vptr，+4 minlist_index，+8 next，+0xC prev，+0x10 `l_obj`，+0x14 edge，+0x18 short `mindist_offset`，+0x1A short status。
- mindist：synapse[0] 在 +0x18，synapse[1] 在 +0x34，+0x50 `sum_extra_radius`（球为 2.0），+0x54 `len_numerator`，+0x68 `contact_plane`（3×f32，+0x74 为 hesse）。
- 第一个 synapse 在 +0x18，比重建版少 4 字节（重建版的位域多占一个字）。这与"原版 len_numerator 在 +84"的说法一致。
- `len_numerator` = 球心到对方特征的距离 − `sum_extra_radius`，即**表面间隙**。
  在 126135 个有三角形的 mindist 上，与 Python 几何复算的差值 median 6e-9（L1）。
- `contact_plane` 是单位向量，从对方表面指向球心，与几何法向夹角 median 2e-6°、p95 1.5e-5°。
- 对方特征三角形：对方 synapse 的 edge 指针 & ~0xF 得到三角形，三角形头 u32 低 12 位是 tri_index；ledge = 三角形 − 16·(tri_index+1)。
  ledge +0 是 i32 `c_point_offset`，+0xC 是 i16 `n_triangles`。三条边在三角形 +4+4k：低 16 位是顶点下标，位 16–30 是相对的 opposite 下标（有符号）。
  顶点是 16 字节（xyz + pad），在**物体坐标**下。edge 指针的 `(addr & 0xC) >> 2 − 1` 是边号。用 core 位姿变到世界后，与 mindist 长度一致（median 1e-9），静态和运动体都是。
- 同时存在的 mindist 数量：0–10 个以上。没有 mindist 的帧（L1 2392 帧，HS03 7455 帧）是指球附近 ±1.2 内没有物体（在空中、开局前），或者正在变球。

### 3.3 来源 (2)：游戏逻辑 [实测]

- `Sound_Manager` 脚本（挂在 `All_Sound` 上）有两个 `PhysicsContinuousContact` BB：Number Group Output = 3，Time Delay Start/End = 0.3/0.3 s。
  正在使用的那个 BB 的 local `IVP_Handle` 指向 `PhysicsContactData`。原版布局与重建版不同：
  +0 delay start，+4 delay end，+8 指针（指向 count = 3），**+0x10 指向 GroupOutput[3] {i32 active, i32 number}**。
  - `number[k]` = 当前与 `Continuous Contact ID` = k+1 的物体之间的摩擦接触数，**没有延迟**；
  - `active[k]` 是经过 0.3 s 去抖的值。
  - 输出 `contact on k` / `contact off k` 的脉冲在 `OnProcess` 时可以用 `IsOutputActive` 看到。
- `Sound_Manager` 另有 5 个 `PhysicsCollDetection` BB 用于撞击音效：Collision ID 1–4 各一个，另一个不按 ID。`Ball_CollSound_Pieces` 里还有碎片用的同类 BB。
  它们的输出参数是 Entity、Speed (0–1)、Normal、Position，事件式触发，有 0.5–2 s 的休眠。
- 只读这些值不会改变执行：读的是 local/output 参数，`update=FALSE`，没有挂任何回调。

### 3.4 来源 (3)：Python 几何

- 静态三角形 = 物理化时导出的 CK 网格 × CK 世界矩阵。运动体 = 碰撞网格 × PSI 物体位姿（§2.5）。
- 每帧的计算：
  - 球心（PSI 位置）到附近三角形的最近点，间隙 = 距离 − r；
  - 法向 = (球心 − 最近点) / 距离；
  - 最近点落在顶点、边还是面上；
  - 离路边的距离。
- **离路边的距离**有两种算法，它们一致（相关系数 0.88–0.99；"网格边界 < 0.3 而射线扇判断 ≥ 2"的帧数为 0）：
  - 网格边界法：取可走面（|n_y| > 0.5），顶点按 1e-3 焊接，只属于一个可走面的边就是边界边，求接触点到边界边的距离；
  - 射线扇法：沿 16 个方向，每 0.25 一步向下投射，第一次没有落点或落点比接触高度低 1.0 以上，就算到了路边。
- 在地面上接触时，射线扇算出的路边距离：40% 的帧 < 0.5，p90 约 4（L1、L3）。TAS 大量贴边、切角，这个分布是真实情况，不是误差。
  栏杆上的边界距离恒为 0.267，没有意义，应改用栏杆专用特征。

### 3.5 逐帧一致性

间隙阈值 0.05，与 IVP 的摩擦距离同一量级。有摩擦接触的帧里，最小 mindist 最大为 0.047。

| 比较 | SR01 | SR06 | SR11 | HS03 |
|---|---|---|---|---|
| 摩擦接触 vs mindist ≤ 0.05（TT / TF / FT / FF） | 12898 / 0 / 1500 / 5255 | 6102 / 2 / 1003 / 5583 | 5539 / 5 / 1306 / 9403 | 10434 / 15 / 1949 / 16256 |
| 木球、石球：mindist vs 静态几何间隙 ≤ 0.05（TF / FT） | 0 / 0 | 0 / 0 | 0 / 329 | 1 / 598（木）、0 / 598（石） |
| 纸球：mindist vs 球近似几何（TF / FT） | — | 206 / 527 | 287 / 428 | 879 / 796 |
| 摩擦接触 vs 游戏逻辑 number > 0（TF / FT） | 80 / 0 | 104 / 0 | 254 / 0 | 504 / 0 |
| 表面类别：IVP 接触对象的 ID == 游戏逻辑组 | 12898 / 12898 | 6103 / 6104 | 5490 / 5544 | 10313 / 10449 |
| 法向：mindist plane vs 几何（median / p95 / max，度） | 2e-6 / 1.5e-5 / 6.1 | 1e-6 / 1.2e-5 / 9.1 | 0 / 1.1e-5 / 0.001 | 3e-6 / 1.4e-5 / 6.3 |

解读：

- **"FT"（几何或 mindist 说很近，但没有摩擦接触）不是错误**。那是撞击、弹跳和落地前的帧：IVP 这时处在撞击模式，还没建立摩擦接触。
  落地前一帧的 mindist median 为 0.017–0.029。所以"接触"要分两层：
  - 几何接触：间隙 ≤ ε；
  - 摩擦接触：friction synapse 存在，表示稳定滚动。
- 木球、石球的 FT（L11 的 329 帧，L3 的 598 + 598 帧）**全部**发生在变球期间：旧球已被 Unphysicalize，探针按 client_data 找到的是已释放的 IVP 物体（`agent-protocol.md` §8.7），所以没有 mindist（md_n = 0）、速度为 0，而几何说贴着地。
  已逐段核对：L11 是事件号 8195–8523（8194 BallNavInactive，8525 纸球物理化）；L3 有两段，一段从事件号 5201 起（木→纸），一段到 18337 纸球物理化为止（石→纸）。Python 用球的 Unphysicalize / Physicalize 事件屏蔽这些帧（`GameTracker.transforming`，§2.8）。
- 游戏逻辑的 TF 有两类。
  第一类是 `Continuous Contact ID` = 0 的对象（箱子、部分机关、PE_Balloon 平台），游戏逻辑本来就不统计它们。
  第二类是**暂停期间**：Esc 菜单打开时，number 全部为 0，而 IVP 接触仍在。这就是表中 1 / 54 / 136 帧"类别不一致"的来源：HS03 的 23330–23463 正是 PauseLevel 到 UnpauseLevel；SR11 的 13512–13565 是一段 BallNav 中断；另有 1–2 帧落在变球边界。
  所以游戏逻辑覆盖不了全部接触，只能作表面类别的旁证。
- 纸球是凸多面体，用球近似不够。纸球以 mindist 为准；如果要做几何，需要纸球网格 × 每帧姿态（core q）。
- 运动体：用 PSI 位姿时 median 1e-5；用插值后的 CK 矩阵时 median 0.007、p95 0.07。
- mindist 的对方特征是"边"（pst = 1）时，接触点到网格边界的距离 median 0.0005，但 p90 为 0.27：有一部分是凸分解产生的内部边。
  所以 **pst 不能直接当"贴路边"**，路边距离由 Python 几何算。

### 3.6 推荐：mod 导出什么，Python 派生什么

**mod 每帧原始导出**（见 `agent-protocol.md` 的 FRIC、MIND、BODY 段）：

- 全部 friction 接触：对方 ckid，双方 status，`last_gap_len`，`real_friction_factor`，`cp_status`，以及 `tmp_contact_info` 的法向和接触点（标为不可靠）；
- 全部 exact mindist：对方 ckid，双方 status，`len_numerator`，`contact_plane`，`sum_extra_radius`，对方特征三角形（物体坐标，边接触时含相邻三角形）及 ledge 标识；
- 球的 core PSI 位姿和速度；附近运动体的 PSI 物体位姿；
- 游戏逻辑接触组 {active, number} × 3 和输出脉冲。

**每关一次导出**：全部物理化物体的参数和网格（局部坐标 + 世界矩阵）、组成员、属性（`Continuous Contact ID`、`Coll Detection ID`、`Floor`）、DepthTestCubes 包围盒。
运行中新的物理化记录增量推送。

**Python 派生**：

- 几何接触 = min(mindist) ≤ ε；摩擦接触 = FRIC 非空；
- 法向 = 最小 mindist 的 plane；
- 表面类别 = 接触对象的组和属性；
- "即将落地 / 刚离开" = mindist 的变化趋势和滞空帧数；
- 离路边距离、局部高度图、相机系下的射线扇 = 用静态和运动几何计算；
- 运动体上的相对速度 = 球速度 − 运动体速度。

## 4. 待测清单

- 每帧二进制读取的实际耗时。
- 俯瞰（space）时的相机特征。
- BallOff 判定：规则已由用户确认（与死亡盒相交）；参与相交的包围体待问用户。
- 纸球的几何距离（多面体 vs 三角形）。
- 角速度的坐标系。
- 风扇（`P_Modul_18`）的风力：有 `SetPhysicsForce`、`TT Scaleable Proximity` 类 BB，还有 `P_Modul18_ActiveSounds` 数组。纸球受风脚本 `Ball_PaperWind` 影响。力的大小、作用区域、开关状态都还没有清点，需要补探针。
- 游戏逻辑接触 BB 在暂停时的行为：number 归零的确切时机。
