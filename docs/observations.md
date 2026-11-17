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
| `path.py` | 参考路径：TAS 参考线（`benchmarks/refpaths/L<nn>.npz`）或检查点折线；带连续性的投影 |
| `features.py` | 特征提取（版本 `f5`） |
| `rewards.py` | SR / HS 奖励 |
| `recording.py` | 每个进程的 .tas + 元数据；BML+ 0.3.12 SR 计时器的 Python 复算 |
| `env.py` / `vecenv.py` | gymnasium 单实例环境（含 TAS 起点、预启动池）；N 个拷贝的向量环境 |
| `curriculum.py` | TAS 起点（`TasStarts`）与反向课程（`ReverseCurriculum`） |
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
- **路径系**：参考路径在投影点处的水平切向（TAS 参考线取投影点前后各 2 单位弧长的弦）作为 forward，right 同上。

## 2. 球与路面的相对关系

难点是每帧都要知道：球在不在路面上、是否接触、离路边多远、接触法向、接触的是哪种表面，运动体上也一样。做法如下（依据见 `state-spec.md` §3）：

| 问题 | 怎么算 | 来源 |
|---|---|---|
| 是否接触（几何） | `min(MIND.gap) ≤ 0.05` → `geom_contact` | IVP mindist（物理真值） |
| 是否稳定滚动 | FRIC 非空 → `fric_contact`，`n_fric` 是接触数（栏杆上常为 2） | IVP friction synapse |
| 离表面多高 | `min_gap`：最小 mindist 的间隙，没有 mindist 时为 2.0 | IVP mindist |
| 接触法向 | 最小 mindist 的 `contact_plane`（从表面指向球心），转到相机系 → `normal_cam`。**不用** `tmp_contact_info` 的法向（约 20% 的帧是旧值）。纸球的 plane 方向不可靠，按几何定向（f5，见下） | IVP mindist |
| 接触点 | 球心 − 法向 × (半径 + 间隙) | 派生 |
| 表面类别 | 接触对象（优先 friction 接触，否则最小 mindist 的对象）的 `Continuous Contact ID`：1 石、2 木、3 金属；0 或无属性记为 other；无接触记为 none。游戏逻辑的 GLCT 只作旁证（它不统计 ID 0 的对象，暂停时读数为 0） | ENTS 属性 |
| 对象在动吗 | 对象的物理化记录不是 fixed → `partner_moving`；与它的相对速度 `rel_vel_partner` = 球速度 − 运动体速度 | PHYS、BODY |
| 是不是栏杆 | 对象属于 `Phys_FloorRails` → `partner_rail` | GRPS |
| 离路边多远（方向） | 射线扇：从**支撑点**沿 `fan_dirs` 个方向（0 = 相机前方，顺时针）每 0.25 一步，在球心上方 `fan_up`（1）处向下投射；第一次没有落点、或落点比**前一个采样**（第一个采样比参照高度）低 `fan_step_drop`（0.75）以上，就是路边 → `edge_fan`（f4 规则 `fan_rule="step"`）。支撑点（f5）：间隙 ≤ 1 的朝上 mindist（n_y ≥ 0.5）中最近的一个的接触点；没有时（腾空、只有侧面 / 上方接触）取球心正下方的第一个落点；再没有（悬在虚空上）就没有参照高度，只有"没有落点"算路边。见下 | 几何（静态网格 + 当帧位姿的运动体） |
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

**f4 射线扇规则**（`features.fan_edges`，`tools/analysis/edge_fan_rule.py`）。f3 的规则"落点比接触高度低 1 以上"把连续的缓下坡当成路边：
SR01_1_14_457_264 第 12320 帧，方向 7 的落点每 0.25 降 0.08（18°），到 3.25 处累计低于 −1，f3 读成路边（`runtime/viz/frame_12320.png`）。
f4 改为比较相邻采样：坡面每一步只降 tan(坡角) × 0.25，悬崖一步就掉 1 以上。阈值来自数据：SR01 地面接触帧（非栏杆，每 3 帧取 1，3657 帧 × 8 方向 × 32 采样）
相邻采样高差的分布，|dy| ≤ 0.25 的占 99.3%（坡面），≤ −1 的 1481 个（路边、落到下层），中间稀疏：[−0.75, −0.25) 574 个、[−1, −0.75) 35 个，
所以取 0.75（每 0.25 降 0.75 = 72°）。没有落点仍然是路边。

| 记录（地面接触帧） | 规则 | 方向无路边（=8）的比例 | 8 方向都无路边的帧 | 单方向距离中位数 | 最小值与 edge_mesh 的相关 | 误报：最小值 ≤ 1 且 edge_mesh > 3 | 漏报：edge_mesh < 0.3 且最小值 ≥ 2 |
|---|---|---|---|---|---|---|---|
| SR01_1_14_457_264（3657） | f3 | 32.8% | 0.1% | 4.5 | 0.844 | 1.0% | 0.1% |
| | f4 | 45.4% | 0.6% | 7.25 | 0.866 | 0.9% | 0.1% |
| HS01_4740_1_30_344（4576） | f3 | 33.4% | 0.5% | 5.0 | 0.790 | 1.6% | 0.7% |
| | f4 | 45.3% | 0.5% | 7.25 | 0.800 | 1.6% | 0.8% |
| SR06_1_36_163_132（1613） | f3 | 23.2% | 0.1% | 4.75 | 0.784 | 2.1% | 0.4% |
| | f4 | 39.4% | 12.0% | 6.5 | 0.885 | 2.1% | 0.4% |
| SR11_2_00_382_132（1919） | f3 | 17.5% | 0.4% | 3.5 | 0.567 | 7.5% | 0 |
| | f4 | 18.6% | 0.4% | 3.5 | 0.571 | 7.5% | 0 |

（SR06、SR11 两行用定向后的纸球法向重算，见下文"纸球的 mindist 法向"；用原始 plane 时是 f3 22.4% / 0.687 / 6.1%、f4 37.0% / 0.792 / 6.1%，SR11 f3 16.5% / 0.371 / 16.1%、f4 17.5% / 0.375 / 16.1%。）

"最小值 ≤ 0.25"的帧比例两种规则一样（SR01 50.7%，HS01 47%），f4 没有改变它：这些帧第一个采样就没有落点（SR01 地面接触帧 edge_mesh 的中位数是 0.23）。
在这些帧里，球心水平方向离接触点中位数 0.8（SR01 0.84、HS01 0.75），从球心竖直向下、在接触点以下 1 以内打到地面的只有约 1/3（SR01 561/1777，HS01 668/1800），
即多数时候球心已经越过路面边缘、球压在边沿上。射线扇从接触点出发，所以读成 0.25；改成从球心正下方出发会改变这一类帧的读数（待定）。栏杆上的帧（两条栏杆之间是空的）两种规则也一样。
可视化核对：`runtime/viz/viz_features.py --frames 11492,14910,8243,12320`（f4 重新生成）：12320 的方向 7 从 3.25 变成 8，
11492（平地）、14910（贴路边，0.75 / 1 / 1.25 处没有落点）、8243（栏杆）不变。f3 规则保留为 `fan_rule="contact"`。

**f5 射线扇参照**（`tools/analysis/edge_fan_rule.py --references`）。f3 / f4 的参照是**最近的** mindist：间隙 ≤ 1 时用它的接触点和高度，否则用球底高度、扇心在最近接触点或球心。
所以球腾空时（间隙 > 1 或没有 mindist），每个落在地面上的采样都比球底低 0.75 以上，8 个方向都读成 0.25；最近的接触是侧面或上方的物体（墙、推箱、石球 / 木球 / 纸球道具）时，
参照高度是那个接触点的高度（大约在球心高度），扇心在球的侧面，也读成 0.25。f4 的统计只看了地面接触帧，所以没有发现。
f5 改用支撑点（见上表）。最近的接触朝上且间隙 ≤ 1 的帧与 f4 逐位相同（下表"地面支撑"一行的变化数为 0），其余特征逐位不变。各记录每 4 帧（SR06、SR11 每 2 帧）取 1 个有控制的帧：

| 记录 | 类别（帧数） | 变化的帧 | 8 方向都 ≤ 0.25：f4 → f5 | 最小值 ≤ 0.25：f4 → f5 | 误报（最小值 ≤ 1 且 edge_mesh > 3）：f4 → f5 |
|---|---|---|---|---|---|
| SR01_1_14_457_264 | 地面支撑（4184） | 0 | 0 → 0 | 51.3% → 51.3% | 23.0% → 23.0% |
| | 腾空、正下方有路面（220） | 199 | 88.6% → 0 | 90.9% → 4.5% | 66.8% → 17.3% |
| | 腾空、正下方没有路面（377） | 14 | 99.2% → 95.5% | 100% → 100% | 67.1% → 67.1% |
| | 最近接触在侧面 / 上方、非栏杆（56） | 54 | 21.4% → 25.0% | 100% → 41.1% | 42.9% → 12.5% |
| | 运动的接触对象、非栏杆（79） | 30 | 15.2% → 1.3% | 64.6% → 34.2% | 64.6% → 43.0% |
| HS01_4740_1_30_344 | 地面支撑（5144） | 0 | 0 → 0 | 48.1% → 48.1% | 21.6% → 21.6% |
| | 腾空、正下方有路面（309） | 282 | 87.4% → 0 | 91.6% → 7.1% | 61.8% → 13.6% |
| | 最近接触在侧面 / 上方、非栏杆（88） | 83 | 33.0% → 17.0% | 98.9% → 29.5% | 64.8% → 12.5% |
| | 运动的接触对象、非栏杆（119） | 56 | 24.4% → 4.2% | 72.3% → 33.6% | 69.7% → 31.9% |
| SR06_1_36_163_132 | 腾空、正下方有路面（558） | 540 | 93.2% → 0 | 97.0% → 6.3% | 59.3% → 5.7% |
| | 最近接触在侧面 / 上方、非栏杆（100） | 97 | 14.0% → 29.0% | 99.0% → 44.0% | 10.0% → 6.0% |
| SR11_2_00_382_132 | 腾空、正下方有路面（386） | 345 | 88.1% → 0 | 90.9% → 9.8% | 46.9% → 18.9% |
| | 最近接触在侧面 / 上方、非栏杆（207） | 174 | 45.9% → 27.5% | 97.6% → 50.7% | 58.5% → 40.6% |

（SR06、SR11 两行的 f4 列已经用了下面定向后的纸球法向。）剩下的情形：

- 腾空且正下方没有路面（飞越缺口、悬在虚空上；SR11 有 1397 帧）：没有参照，读数 0.25 是"这里没有路"，按设计如此。
- 推着道具球、骑在栏杆上等：支撑点是栏杆或地面，靠近道具球的方向读成 0.25（采样落到球体上再掉下来）。射线扇不区分障碍物和路边。
- 运动对象的"误报"可能主要是比较基准的问题（没有逐帧画图核对）：`edge_mesh` 只用静态网格，球在飞艇平台（`PE_Balloon_Platform` / `Platte*`）这类运动体上时它量的是到静态边界的距离；
  SR01 剩下的 34 个"误报"帧全在飞艇平台（27）和木球旁（7）。
- 可视化：`runtime/viz/f5_extra/frame_13894.png`（腾空，正下方是宽阔路面：f4 是 8 个 0.25，f5 是 [8, 8, 8, 8, 7.5, 6.25, 4.5, 6.25]，edge_mesh 4.26）、
  `frame_19311.png`（侧上方接触石球 P_Ball_Stone_MF001，法向 y = −0.78：f5 取正下方路面，最小 5.75）、`frame_7494.png`（推木球，支撑点在栏杆上）；
  `runtime/viz/frame_{11492,14910,8243,12320}.png` 已用 f5 重新生成，四帧都是地面支撑，读数与 f4 相同。

**纸球的 mindist 法向**（f5）。`contact_plane` 的方向取决于 mindist 里两个 synapse 的顺序。木球和石球的 synapse 是球（`self_status` = 3，IVP_ST_BALL），
plane 总是从对方指向球心；纸球是多面体（synapse 为点 / 边 / 面，`self_status` 0–2），plane 多数是**反的**：与几何核对（球心到对方网格最近点的方向），
SR11 的纸球条目 481 个中 351 个反向、SR06 188 个中 107 个；`self_status` = 3 的条目没有反向的（SR11 1148、SR06 1910 个）。f4 因此在纸球贴地时得到朝下的法向、
球心上方的"接触点"。f5 的 `features.mind_normals` 对 `self_status` ≠ 3 的条目按几何定向（对方是静态物体时用关卡网格，运动体用当帧 BODY 位姿），定向后与几何全部一致（SR11 458/458、SR06 180/180，
不到 0.9 余弦的 23 / 8 个两种都不算）。`normal_cam`、接触点、`edge_mesh` 的起点、支撑点都用定向后的法向。L1 没有纸球，SR01 / HS01 的特征不受影响。

注意要按对象取最小：mindist 是按凸 ledge（凹网格的凸分解块）给的，单个 ledge 的距离大于等于整个对象的距离。
特征计算本身（7×7 高度格 + 8 方向射线扇 + 边界距离 + 栏杆）约 1.5 ms/帧（L1 22351、L11 26966 个静态三角形，构建网格索引 0.03 s）。

**运动体的射线开销**（f5 前的实现）：运动体取 BODY 中心 60 单位内的全部网格（L1 末段约 1000 个三角形），没有网格索引，高度格和射线扇对它们逐对求交，
SR01 第 6000–6999 帧平均 34.8 ms/帧、18000–18999 帧 12.4 ms、20000 帧以后 27 ms。现在先算出这一帧全部射线的 (x, z) 范围，
只保留包围球（网格局部顶点的最大长度 + |shift|）与它相交的网格、(x, z) 包围盒与它相交的三角形（对范围内的射线精确，结果逐位相同）：同样几段降到 1.9 / 2.1 / 2.0 ms，
SR01 全程平均 5.8 → 1.4 ms，HS01 6.7 → 1.5 ms（每 4 帧取 1 帧，`tests/test_features.py` 有逐位相同的用例）。

## 3. 特征向量（版本 f5）

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
| normal_cam | 3 | 接触法向（相机系；f5 起纸球按几何定向） |
| surface | 5 | none / stone / wood / metal / other |
| partner_moving, partner_rail | 2 | 见 §2 |
| rel_vel_partner | 3 | 与运动对象的相对速度（相机系）/ 20 |
| edge_mesh | 1 | 接触点到可走面边界 / 12（上限 12） |
| edge_fan | 8 | 各方向到路边的距离 / 8（上限 8）；f5 从支撑点量起（见 §2），f4 从最近接触点量起 |
| rail_dist, rail_dir | 1+2 | 最近栏杆的间隙 / 6、水平方向 |
| heights, height_hit | 49+49 | 7×7、间距 1 的高度格（相对球底，/5，无落点 = −1）和落点标志；顺序为 [right 下标][forward 下标] |
| path_progress | 1 | 参考路径上的进度 s / 全长（0–1） |
| path_remaining | 1 | 剩余弧长 / 1000 |
| path_lateral | 1 | 球心到参考线的三维距离 / `path_lateral_scale`（10；f3 是 / 50） |
| path_tangent_cam | 2 | 投影点处参考线的水平切向在相机系中的 (right, forward) 分量 |
| waypoints | 9 | 参考线上投影点前方弧长 5、10、15（`waypoint_spacing` × 1..`waypoints`）处的点相对球的位置（相机系）/ 50；过终点后重复终点。f3 是接下来 3 个折线节点 |
| bodies | 28 | 最近 4 个运动体（BODY，60 单位内，stopper 除外）：存在标志、相对位置 / 50、相对速度 / 20 |
| lives, points, sr_s, sector | 4 | 命数 / 10、Points / 1000、SR 计时（double 和）/ 100 s、扇区 / 参考路径的扇区数 |
| transforming, ball_nav | 2 | 正在变球、玩家有控制。变球（f3）= 从活动球（`Ball_*` 实体）的 Unphysicalize（不在 BallOff 之后）到球的下一次 Physicalize，事件参数是物理化记录序号，由关卡消息的 PHYS 记录得到实体（client 先读增量再更新 tracker）。HS03 上正好是 IVP 球体缺失的 4 段（5201–5800、9740–10339、17738–18337、27590–28189，`tests/test_dataset_regression.py`）。f1 用 Activetrafo（`IngameParameter[0,0]`，变球后**不清零**）把第一次变球之后的帧都判成变球；f2 加了"BallNav 不活动"，但暂停期间（HS03 8590–8719 等）和 LevelFinish 之后 BallNav 也不活动，仍然误判；f3 改用事件 |
| depth_margin | 1 | 球心高出正下方死亡盒（DepthTestCubes）顶面的高度 / 50；只作整形用，死亡以 BallOff 事件为准 |
| followers | 1 | 跟随中的分数小球数 / 6（mod 的 SCOR 计数，0.3.1；旧数据从 WTCH 复算：主球不可见、小球可见，`state-spec.md` §2.9）。`hs_block` 控制本行到 since_checkpoint |
| follower_sources | 4 | 各分数球实例跟随中的小球数 / 6，从大到小（WTCH） |
| follower_near | 1 | 最近的跟随小球到球心的距离 / 50（上限 1；没有跟随时为 0） |
| paused | 1 | 暂停中（PauseLevel 到 UnpauseLevel；按键的两帧之后才切换） |
| since_checkpoint | 1 | 距上一次 PreCheckpoint（没有时距第一次 BallNavActive）的帧数 / 10000。还在跟随的小球在下一个检查点作废 |
| far_waypoints | 3 × n | 仅 f6 的 `far_waypoints`（默认空）：参考线上投影点前方弧长 20、30、45（可配）处的点相对球的位置（方向特征的坐标系，含高度）/ 50；放在向量末尾 |
| ref_speed | n | 仅 f6 的 `ref_speed_at`（默认空）：参考记录在弧长 s + 0、15、30（可配）处沿参考线的推进速度（弧长 / 游戏时间，单位 / 秒）/ `vel_scale`；来自参考线资产的 `t_ms`（每个点被参考轨迹经过的游戏时间，`build_refpaths.py` 生成；L01.npz 重建时点、扇区、meta 逐位不变，只加了 `t_ms`）。没有 `t_ms` 时为 0。来自 TAS，通用策略（M3）需要别的来源 |

`FeatureConfig` 的全部字段（网格大小、射线扇方向数、缩放、是否含运动体……）都会写进每个 run 的元数据。
**f6**（`FEATURE_VERSION = "f6"`）：可选的前瞻块 `far_waypoints`、`ref_speed`，默认关闭，此时布局与 f5 完全相同。原因（r3，第 3 扇区）：球以约 21 单位/秒接近 s 1375–1395 的腾空缺口，
射线扇（8 单位）和高度格（±3）只看到约 0.4 s 之后，路点（5/10/15）约 0.7 s，策略无法提前知道"前面有缺口、需要多快"。
新条目放在向量末尾，热启动时把旧策略第一层的对应输入权重补 0、观测统计量补 (均值 0, 方差 1)，切换那一刻策略和 value 不变（`train_m2.py` 的 `--resume` / `--init-from` 自动扩展，Adam 状态重置）。
改变已有条目含义时提升 `FEATURE_VERSION`。

**参考路径**（`path.py`，f4）。两种来源，`FeatureConfig.path_source` 选择：

- `"tas"`（默认）：该关数据集里最快的通关记录（`end_reason` = LevelFinish，正式 SR 最小）的球心轨迹，从第一次有控制（BallNavActive、找到 IVP 球、未暂停）到 LevelFinish；
  没有 IVP 球的帧（变球）直接连过去。先按时间做高斯平滑（σ = 40 ms，去掉弹跳造成的弧长），再按弧长每 0.5 单位重采样。每个点带扇区标签（回放时 tracker 的 `sector`，即运行时投影受约束的那个量），
  扇区沿线不回退。生成脚本 `scripts/build_refpaths.py`，资产 `benchmarks/refpaths/L<nn>.npz`（float32 点、uint8 扇区、`meta`：记录名和 sha256、间距、平滑参数、起点 / 检查点 / 飞艇平台坐标、各扇区弧长范围；不含路径）。
  资产只在它存的起点和检查点坐标与当前关卡一致（1 单位内）时使用，否则（例如 fakegame 的玩具关卡）退回折线。第 9、13 关在 tas-v1 里没有通关记录，没有资产，用折线。
- `"checkpoints"`：f3 的折线，`ResetPoints[0]`（起点）→ `Checkpoints[k]` 的矩阵位置 → 飞艇平台。节点 s−1 → s 是第 s 扇区的线段。

| 关 | 来源记录 | 点数 | 全长 | 扇区 |
|---|---|---|---|---|
| 1 | SR01_1_14_457_264 | 4110 | 2054.3 | 4 |
| 2 | SR02_1_28_992_264 | 4662 | 2330.1 | 5 |
| 3 | SR03_1_36_884_264 | 5377 | 2687.1 | 5 |
| 4 | SR04_2_04_431_264 | 6745 | 3371.4 | 5 |
| 5 | SR05_2_10_850_264 | 5976 | 2984.5 | 5 |
| 6 | SR06_1_29_838_264 | 4467 | 2232.0 | 5 |
| 7 | SR07_2_05_840_264 | 5303 | 2645.7 | 5 |
| 8 | HS08_5721_3_09_448（SR08 在本版本上 GameOver） | 8131 | 4062.0 | 5 |
| 10 | SR10_2_24_432_264 | 5751 | 2871.5 | 5 |
| 11 | SR11_2_00_382_132 | 5865 | 2930.3 | 6 |
| 12 | SR12_2_12_823_240 | 7321 | 3659.2 | 8 |

**投影**（`PathFollower`，每个回合一个状态，`env.reset()` 时清零：新进程、复活、TAS 交接都算回合开始）：
在上一次投影的弧长 s_prev 附近的窗口 [s_prev − 10, s_prev + 30] 里找最近点，窗口限制在当前扇区的弧长范围内（两端各放宽 20）。
后备：没有 s_prev（回合开始）、窗口为空、或整个扇区范围内的最近点比窗口内的最近点近 5 以上（球离开了这段线，例如掉到下层路面）时，用扇区范围内的最近点，
并标记 `jump`（`info["path"]["mode"]` = init / window_empty / reacquire）。`jump` 那一步的 Δs 默认照常付（经过 `hold_unsupported`，见 §4 `pay_jumps`）；`pay_jumps=False` 时按 0 计（只移动参考点，旧行为）。
路线在同一扇区里绕圈或交叉时，窗口保证投影沿着走过的分支前进：参考记录自己的投影与"扇区内最近点"不同的帧数，L5 1013、L7 1459、L10 951、L6 0，
用连续性投影的横向距离最大值仍是 1.2 / 1.4 / 2.6（`scripts/build_refpaths.py --check`；`tests/test_path.py` 有自交的 8 字形用例）。

**横向距离**（单位，`path_lateral` × 10；L1，按扇区 median / max）：

| 记录 | 参考线 | 扇区 1 | 扇区 2 | 扇区 3 | 扇区 4 |
|---|---|---|---|---|---|
| SR01_1_14_457_264 | f3 折线（飞艇节点在原点） | 132 / 208 | 20.5 / 55 | 30.6 / 60.5 | 134 / 391 |
| SR01_1_14_457_264 | f4 折线（飞艇节点修正后） | 132 / 208 | 20.5 / 55 | 30.6 / 60.5 | 52.2 / 106 |
| SR01_1_14_457_264（参考） | f4 TAS 线 | 0.03 / 0.67 | 0.04 / 0.69 | 0.03 / 0.55 | 0.02 / 0.57 |
| SR01_1_15_797_240 | f4 TAS 线 | 0.64 / 15.8 | 0.60 / 4.4 | 0.18 / 4.2 | 0.91 / 11.8 |
| SR01_1_16_726_240 | f4 TAS 线 | 0.97 / 12.4 | 0.69 / 5.4 | 0.49 / 4.4 | 1.16 / 11.7 |
| HS01_4740_1_30_344 | f4 TAS 线 | 0.19 / 46.0 | 1.41 / 33.3 | 3.70 / 44.1 | 0.64 / 10.9 |

HS01 的最大值是绕去吃分数球的岔路。所有 L1 记录在 TAS 线上都没有触发 reacquire，s 不回退，终点 s ≈ 2053.8（全长 2054.3）。

**飞艇平台节点**（f3 的 bug）：`LevelData.entity_position("PE_Balloon_Platform")` 在全部 13 关都是 (0,0,0)。原因：全量关卡消息的 ENTS 是 StartLevel（约第 103 帧）的世界矩阵快照，
那时整个飞艇组（`PE_Balloon_MF` 及子物体）停在原点；平台在约第 22–25 帧第一次 Physicalize 时在关卡终点，`.inc` 里那次 Physicalize 导出的 MESH 带着终点处的世界矩阵，
约第 34–55 帧 Unphysicalize，最后一个扇区开始时再次 Physicalize（不再导出网格）。全量消息里平台的网格也按原点快照重新导出了一份，
所以结果取决于合并顺序。f4 的来源：`LevelData.physical_position(name)` = 与该实体的 Physicalize 记录同一帧导出的碰撞网格（`phys_meshes`，与合并顺序无关）的世界包围盒中心，
没有时退回 ENTS。31 条通关记录（含 HS04_esc）上，它到 LevelFinish 时球心的距离 1.3–2.8（水平 0.1–1.5）；用网格矩阵的位置是 1.0–3.4（`tests/test_path.py`）。

## 4. 奖励（`RewardConfig`）

时间一律取 TIME 段的 SR double 和（BML+ 0.3.12 计时器），不按帧数算。

**SR 模式**（`mode="sr"`），每步：

- 进度 `progress_weight · Δs`（s 为参考路径弧长；后退乘 `backward_scale`；势函数整形项不受影响，`shaping_weight` 默认 0）；
- **投影跳变照常付**（`pay_jumps`，默认开，M2 起）：投影 re-acquire（`jump`）那一步的 Δs 和其它步一样计入（先进 `held`，有支撑时付清，BallOff 时作废）。
  原因（用户）：只要不死、到达下一个检查点 / 飞艇，任何路线都合法，例如从 L1 上层螺旋弯道落到下层路面。旧行为（`pay_jumps=False`）跳变那一步记 0，
  捷径跳过的那段弧长永远不付，等于惩罚捷径；落回前面走过的路段也不扣，再走一遍还能再拿一次进度。现在一个回合付出的进度 = 0.1 ×（s_末 − s_初），
  前跳（落在线上更远处）按跳过的弧长付，后跳照扣。窗口之外的落点靠 re-acquire（整个扇区范围内的最近点比窗口内近 5 以上）找回，投影仍限制在当前扇区 ±20 内。
- 时间 `−time_weight · ΔIGT(秒)`；
- 势函数整形 `γ·Φ(s') − Φ(s)`，`Φ = shaping_weight · s`，终止状态 Φ = 0；
- 检查点 `checkpoint_bonus`（默认 0）；
- 通关 `finish_bonus`，死亡（BallOff）`−death_penalty`；
- **没有支撑时的进度先挂起**（`hold_unsupported`，默认开）：这一步结束时球没有支撑（没有间隙 ≤ 1 的朝上 mindist，即 `info["supported"]` 为假：腾空、只贴着墙）时，
  Δs 记入 `held`（`info["progress_held"]`），不付；下一个有支撑的步（落地）或 LevelFinish 时一次付清，BallOff 时作废（净后退的部分照扣）。
  原因：掉出路面后投影仍会前进，在弯道处每步最多可以跳过窗口的 30 单位弧长（审查者的例子：SR03_1_46_066_240 一次坠落 69 步得了 87 单位进度、单步 +27.9）；
- **死亡退还本扇区的进度**（`death_refund`，默认 1.0）：BallOff 时扣回本回合在当前扇区（回合开始或上一次检查点之后）已经付出的进度。复活在上一个检查点，
  这段路要重跑，所以"冲一段再死"不比原地不动划算。只退本回合付过的部分：课程起点的回合不会因为起点前那段（没付过的）路而多扣。

  旧行为是 `RewardConfig(hold_unsupported=False, death_refund=0)`。离线核对（`tools/analysis/sr_reward_check.py`：数据集回放，每 30.3 ms 游戏时间一步，与 env 的 132 fps × 4 相同）：
  L1 的 6 条记录（SR01 ×3、HS01 ×3，全部通关）和 SR03_1_36_884_264、SR05、SR07 的参考记录，新旧两种配置付出的进度完全相同（都等于终点 s − 起点 s，挂起的进度在落地时付清；
  SR03 纸球段有 1124/3197 步没有支撑）。SR03_1_46_066_240 的 4 次死亡：旧配置付 12.0 / 4.85 / 0.4 / 11.75，新配置先付 4.62 / 1.09 / 0 / 8.03，死亡时全部退回，净 0。
  在 L1 TAS 线上每 5 单位弧长沿切线以坠落速度（水平 34 u/s、竖直 −47 u/s）飞出、直到进入死亡盒（410 个起点）：旧配置付出的进度中位数 18.5、最大 138.8 单位，
  新配置全部是 0，回合总奖励 ≤ −5.15。
  **M2 r4 起 `death_refund` = 0.4**（训练端 `--reward '{"death_refund": 0.4}'`）：退还 1.0 时死亡的代价按 progress_weight 0.1 / 单位折算，
  相当于把进度按 10 单位/秒换成时间，而策略在 L1 上约 25 单位/秒，重跑同样的路实际只花约 2.5 倍少的游戏时间（复活动画不计 SR）。
  例：L1 最后一个扇区的 TAS 捷径（两次在坡上起跳，比绕路快约 3.0 s）在扇区里约 500 单位处，死亡代价 5 + 50 ≈ 55，收益 +3，
  死亡率超过约 5% 就不值得走；按真实 SR 算（重跑约 20 s）界限约 13%。0.4 让死亡代价约等于重跑的时间（5 + 20 ≈ 25，界限约 12%）。
  训练记录里所有 r3 自己到达那里的通关局都走了绕路（610 / 610）。
  注意：`death_refund` 让死亡的代价随本扇区的进度增大，γ = 0.99、每秒 33 步时有效视野只有约 3 秒，过大的死亡代价可能让策略宁可停着不动（截断不扣分）。这是训练时要调的量。
- 死亡与复活之间不计 SR（数据上的观察，审查者发现）：导出验收的 run 里，复活后那条命的 SR 增量正好等于它自己的帧数 × delta，
  例如第 0 条命在第 15742 帧、sr_ms 56852.27 结束，第 1 条命从第 16280 帧到 16570 帧（290 帧 = 2197 ms），结束时 59049.24；
  BallOff 到 BallNavActive 的 538 帧没有计时。本轮导出的 6 次复活都如此。所以死亡在 SR 上的代价是重跑这一段，不是动画时间。没有在游戏机制层面核对（§8）。

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
  这些帧都进入帧日志。可以用 `prefix`（`BALLRAIN_PREFIX`，mod 先按回放方式注入）从 TAS 前缀交接；TAS 起点见 §5.1。
- **动作**：`Discrete(9)`（无键、四个方向、四个斜向），`camera_actions=True` 时再加 shift+左 / shift+右。每步注入 `action_repeat` 帧，
  delta 固定为 `delta_ms`（默认 f32(1000/132)，每帧一个 PSI），遇 BallOff / LevelFinish / GameOver 提前停止，未执行的帧作废。
- **世界方向动作**（`world_actions`，默认关；M2 训练用 `--world-actions`）：9 个动作是世界方向（上 = +z，右 = +x，与 `FeatureConfig(camera="world")` 的方向特征同一个系），
  env 每一步按当前 `Cam_OrientRef` 的 (right, forward)（`features.yaw_basis`）换成方向键：`world_to_camera_keys`，`Cam_OrientRef` 是 90° 的整数倍，分量恰好是 −1 / 0 / 1。
  原因：方向键按相机朝向推球，相机系的特征 + 相机系的动作在另一个相机朝向下看到的是旋转过的关卡。L1 的 SR01_1_14_457_264 在第 141 帧（拿到控制之前）和第 6276 帧各转一次相机，
  `Cam_OrientRef` 的 yaw 依次是 0 →（第 103 帧）−90 →（112）90 →（208）180 →（6344）90；k = 0 的开局只按 Q、动作里没有转相机，整关都是 90。
  所以课程起点在 [936, 6344) 的回合（相机 180）和 k = 0 的回合在同一段路上的相机差 90°。世界系的策略与相机朝向无关；相机 yaw 特征（`cam_yaw_ref`、`cam_yaw`）仍在观测里。
  运行元数据新增 `env_options`（action repeat、camera / world actions、stale-key 重按、delta）。
- **选键**（`world_to_camera_keys`）：按 `Cam_OrientRef` 的实际行向量算每个方向键的推力方向（上 / 下 = ±z 行、右 / 左 = ±x 行，投影到水平面再归一化，`key_push_dirs`；
  两个键是两个独立的力，相加），在 8 种组合里取与目标世界方向余弦最大的一个（并列时单键优先）。静止时 `Cam_OrientRef` 在四分之一圈上（SR01 回放里偏差最大 1.1e-6 rad，
  正常转相机的 250 ms 动画期间也不变，到动画结束那一帧才跳到新朝向，动画的是 `Cam_Orient`），这就是动作的精确旋转；偏离四分之一圈时（转到一半被打断等）取能推出的 8 个方向里最近的一个。
  测试用了加 1.3e-5 噪声的矩阵和每 5° 一个的中间朝向（`tests/test_world_actions.py`）。
- **按键推力模型**（BallanceMMO 会话读 Gameplay_Ingame 脚本 dump 后的结论，与下面的实测一致）：推力 = Force Value（纸 0.065、木 0.43、石 0.92）×
  归一化的 `Cam_OrientRef` 对应行向量，取**最近一次 Create** 时的值。Create 发生在：每次按下（逐帧边沿，没有去抖）；以及每次转相机结束时——`Cam Navigation.Update`
  把 Ball Navigation 关掉再打开，这个 Nop 关掉所有 SetPhysicsForce 控制器并切换 Key Events，而 Key Event 在键按着时被打开会触发 Pressed，于是按着的方向键都按新的
  `Cam_OrientRef` 重新建力（转相机键本身还按着时是否也如此未核对）。两次转相机之间力保持锁定（已有控制器的块上再 Create 不起作用）。斜向是两个独立的力相加。
  死亡、通关、暂停 / 菜单同样经过 Nop 关掉所有力。`Cam_OrientRef` 本身不做动画：`Cam_Orient` 用 250 ms 动画转过去，转完时 `Cam_OrientRef` 被设成 `Cam_Orient`，
  所以 Ref 在动画结束那一帧跳到新朝向（SR01：第 6344 帧）；Ref 只有在"打断转相机"的技巧（或地图自身的基准朝向）下才会偏离四分之一圈，这正是相机守卫检查的量。
- **按键方向锁定的处理**（`repress_stale_keys`，默认跟 `world_actions`，保留开启）：env 记录每个方向键按下时的相机（四分之一圈），某一步要按的键里有
  "按下之后相机变过"或"之后游戏关过力"（复活、解除暂停）的，这一步的第 1 帧先松开这些键、第 2 帧起再按下（action repeat 4 时 1 + 3 帧，`info["repressed"]`）。
  TAS 交接时，前缀最后一帧按着的键按记录里按下那一帧的相机算（`TasStarts.camera_quarters()`，缓存在记录目录的 `camera.npz`）。
  按上面的模型这是保守的：转相机结束和控制恢复时游戏自己会重建按着的键的力，所以重按通常不改变推力，只少推 1 帧；只在 TAS 刚转完相机就交接时触发。
  默认配置下复活快进和暂停期间注入的都是空键（`idle_keys`、`pause_keys` 为 0），控制恢复后的第一次按键本来就是新的按下。
  **实测**（`tools/analysis/m2_key_checks.py`、`runtime/analysis/m2_key_checks.json`，真游戏）：
  - k = 6350 交接（前缀按着下 + 右，分别在第 6338、6342 帧、相机 180 时按下，第 6344 帧相机变成 90）：选按键正好是这两个键的世界动作，
    推力（减去空动作对照的位移）与世界动作的余弦，有重按时每步都是 1.000，不重按时也是 0.987 → 1.000（与旧相机方向的余弦 0.16 → 0.01）。
  - k = 1000（相机 180）按住上 40 帧，按住上的同时 shift + 左转相机，再一直按住 120 帧（A），与转完后松开 1 帧再按（B）、不转相机（C）比较：
    转完之后 50 帧的速度变化 A (−5.11, −6.53, 2.55)、B (−5.21, −6.53, 2.55)、C (0.06, −6.34, −2.31)：一直按着的键在转完后按新相机方向推，A 与 B 只差 B 少推的那 1 帧，
    与"转相机结束时重建按着的键的力"一致。
  - 复活：k = 15000 起推下路面死亡，复活后按住指向参考线切向的世界方向 30 步，球水平移动 8.97（同样输入下空动作 0.16）：复活后的新按键有力。
- **Esc**（`esc_action`，HS 模式默认开，SR 模式默认关）：一次按下 = 第 1 帧 Esc、之后松开，共 `esc_frames`（默认 4，至少 3）帧。
  游戏在按下的那一帧还跑物理，之后冻结，直到下一次按下（含该帧），再下一帧恢复；PauseLevel / UnpauseLevel 事件在按下后 2 帧到达，
  `paused` 只看这两个事件（HS 记录里有 2/147 次按键没有切换）。没有控制时（不是暂停中，且 BallNav 不活动、死亡、变球、通关）Esc 被屏蔽为空帧（`esc_requires_control`）。
  暂停中方向键按 `mask_keys_while_paused` 换成 `pause_keys`（默认 0）。
- **collect 宏**（`collect_action`，默认关）：有小球在跟随时，Esc → 每 `collect_chunk` 帧检查一次，直到没有跟随的小球或等满 `collect_max_frames` 帧 → Esc；
  否则是一个空动作。`info["collect"]` 给出前后的跟随数和等待帧数。最佳做法（用户）：飞艇和每个检查点之前按 Esc 收集。
- **相机守卫**（始终开启，用户规则）：不允许非 90° 的视角（例如转相机被打断造成的偏离朝向）。`Cam_OrientRef` 偏离最近的四分之一圈超过 `camera_quarter_tol`（1e-4 rad；
  零售版静止时的偏差在 1.1e-6 以内）、并且距离上一次 shift + 左 / 右按下已超过 `camera_turn_grace_ms`（400 ms 游戏时间，转相机动画约 250 ms）时，env 立即结束回合，
  `episode_end = "camera_off_quarter"`，按终止处理（不 bootstrap），这一步的奖励改成让整个回合的回报等于 −`death_penalty`（不给这个回合任何奖励），
  进程结束，元数据 `invalid = "camera_off_quarter"`：这个 run 永远不作为结果导出，SR 不计入（训练日志的 `invalid_runs`、`camera_off_quarter` 计数，tensorboard 同名曲线）。
  TAS 起点同样过滤：`TasStarts` 去掉静止时相机偏离四分之一圈的帧（记录的转相机按键之后 400 ms 内除外；SR01_1_14_457_264 一帧也没有去掉）。
  我们的 agent 没有转相机的动作，所以这个守卫在 r2 里不应触发；加入相机动作以后才有意义。
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
- 验收按 M0 规则在门禁阶段做：参考 TASSupport 回放 3 次，与多数结果逐字节相同即通过；不通过时用 `gates.export_diagnosis` 区分偶发分叉和保真问题（§5.1）。

### 5.1 TAS 起点与反向课程（`curriculum.py`）

`EnvConfig(start_record="SR01_1_14_457_264", curriculum={...})`（或固定的 `start_frame=k`）：每个**新进程**从该记录的第 k 帧开始。
env 把记录的前 k 帧写成该 run 目录下的 `prefix.bin`，作为 `BALLRAIN_PREFIX`；mod 按回放方式注入（不往返），第 k 帧交给 Python。

- k 用记录自己的帧计数（264 fps 的记录每游戏秒 264 帧），交接后 env 按 `delta_ms`（132 fps）注入。可选的 k：0（从关卡加载开始，第 1 关照常按脚本化的 Q）
  或数据集回放中"有控制"的帧（BallNav 活动、有 IVP 球、未死亡、未变球、未暂停，且在 LevelFinish 之前；L1 是 936 起）。`TasStarts` 从数据集读出这些帧并缓存在记录目录的 `starts.npz`。
  k < CamNavActive 的情形由 `_intro` 照常处理（还没 StartLevel 就快进，CamNav 未激活就按 Q）；有效 k 都在控制开始之后，不会重复按 Q。
- 交接状态与数据集逐项一致：L1 的 k = 5000、10000、15000、19000 各 3 次，球的位置（逐位相等）、扇区、检查点数、Points、Lifes、SR 的 double 和都等于数据集回放在第 k 帧的值（`scripts/m2_start_states.py measure`）。
- **交接检查**（`start_check`，默认开）：交接时把上面这些量与数据集比较，不一致就停掉这个进程（run 目录里留下 `start_mismatch.json`）并重启，
  最多 `start_retries`（3）次，次数记在 `info["start"]["handover_mismatches"]` 和元数据的 `start_state` 里。原因：同一个前缀并不总是得到同一条轨迹。
  **同样的输入不总是得到同样的轨迹**：
  - BallrainIO：上一轮吞吐测量的 173 个前缀进程（170 个 k > 7818；本轮复测覆盖了原目录，备份在 `runtime/m2-results-bak-review1`）中 1 个分叉：`l1_curr_random_x8/train-07/…-57244-3`，PreCheckpoint 在 7818 而不是 7817，
    它的 `framelog.bin` 前 12741 帧与 `prefix.bin` 逐字节相同、`prefix.bin` 与记录前 18962 帧逐字节相同；之后死亡 4 次、在 12741 帧 GameOver（交接前），被交接检查发现并重启。
    之前报告的"约 460 个中 3 次"里另外 2 次（同样是 7818，相隔 6 s，train-00 和 train-03）来自更早的吞吐测量，那些 run 目录已被后来的测量覆盖，无法复核。
    本轮复测（f5）的 181 个前缀进程（178 个 k > 7818）没有分叉，交接检查 0 次重启。磁盘上可核对的合计：354 个中 1 次。
  - 真 TASSupport 也一样：M0 的 SR01_1_16_726_240 两轮参考回放（`m0-results` 与 `m0-results-r2` 的 `ref_trace.txt`）在第 7111 行（第 7214 帧，球姿态第 5 位小数）分开，
    第二轮的 BallrainIO 轨迹等于第一轮的参考，所以是第二轮参考回放那一次分叉。
  - 交接检查只覆盖前缀。agent 自己玩的帧（k = 0 的整个 run 也是）没有可比的参照，偶尔分叉不会被发现；导出检查会发现（参考回放与 agent 轨迹不同），
    再用 BallrainIO 回放同一个 .tas 区分原因（`gates.export_diagnosis`）：回放等于参考多数 → agent 那一次分叉（`agent_outlier`，重跑即可，.tas 仍按参考的结果判定）；
    回放等于 agent 轨迹 → 系统性差异（`systematic`，保真问题）；参考 3 次没有多数 → `ref_split`。
  原因未查（是否要查待定）。
- SR：回合的 SR / 时间奖励从交接开始（`Reward.begin` 取交接时的 SR）；`info["start"]` 给出 `kind`（tas / level / respawn）、`frame`、交接时的 `sr_ms`、`copy`、`timing`。
  导出的 .tas = 前缀 + agent 帧（mod 的帧日志本来就含前缀），由真 TASSupport 回放验收。元数据新增 `start_state`（记录、sha256、k）和 `timing`（META_VERSION 3）。
- 死亡后照常每条命一个回合：复活在同一进程里从检查点继续（`kind = "respawn"`）；GameOver / LevelFinish 后开新进程，重新抽 k。
- 课程：`ReverseCurriculum` 维护一个前沿（有效帧的下标，初始在最后一个有效帧之前 `start_back`=1500 帧），
  从 [前沿, 末尾] 抽 k（`recent`=0.7 的概率从前沿以上 `band`=1500 帧内抽，其余在整个窗口均匀抽）。训练端用 `curriculum.episode_outcome(info)`
  从每个结束的回合取 (k, 是否 LevelFinish)（向量环境：`curriculum.vector_outcomes(infos)`，gymnasium 的 dict-of-arrays 和 SB3 的 list 都可以），调用 `update(k, success)` / `update_many`；前沿带内最近 `min_episodes`=20 个回合的成功率 ≥ `advance_at`=0.5 时前沿后退 `step`=1000 帧，
  到达第一个控制帧后以 `p_level_start`=0.1 的概率抽 k = 0。状态是普通 dict：训练端持有一个课程对象，用 `venv.call("set_start_state", cu.state_dict())`
  （gymnasium）或 `venv.env_method("set_start_state", ...)`（SB3）推给各个 env。只有 `kind == "tas"` 的回合计入课程。课程起点的回合建议设 `max_episode_frames`。
- **启动开销**（L1，train-00，每个 k 3 次，均值，秒）：

| k | 启动（Popen → mod hello） | 交接（hello → 第一条观测：加载关卡 + 前缀回放） | Python 处理交接观测 | 开局（到控制） | 合计 |
|---|---|---|---|---|---|
| 0 | 1.49 | 1.40 | 0.00 | 0.26（Q、到 BallNavActive，第 527 帧） | 3.15 |
| 5000 | 1.55 | 1.93 | 0.014 | 0 | 3.50 |
| 10000 | 1.56 | 2.23 | 0.015 | 0 | 3.80 |
| 15000 | 1.57 | 2.63 | 0.015 | 0 | 4.22 |
| 19000 | 1.55 | 3.02 | 0.017 | 0 | 4.59 |

  前缀回放约 12200 帧/秒（交接时间对 k 的线性拟合，截距 1.44 s 是加载关卡），所以 k = 19000 只比 k = 0 多 1.6 s；进程启动和加载关卡（约 2.9 s）占大头。
- **导出验收**（`scripts/m2_start_states.py export`，f5 特征和新的 SR 奖励）：3 个课程起点的进程（k = 15538、12332、20550，随机策略）。前两个交接后死亡、复活 3 次直到 GameOver
  （回合类型 tas + 3 × respawn），第三个直接 LevelFinish。导出的 .tas（前缀 + agent 帧）由真 TASSupport 各回放 3 次，再由 BallrainIO 回放 1 次：
  3 个参考轨迹完全相同，且与 agent 模式的轨迹、BallrainIO 回放的轨迹逐字节相同（`kind = pass`），帧日志一致，交接检查 0 次重启。
  （上一轮 k = 19794、15612、19546 也全部通过。）
- **预启动池**（`warm_pool=True`，`warm_copy`）：每个 env 用两个拷贝，当前进程运行时在另一个拷贝上预先启动下一个进程（含前缀回放和开局），
  下一次需要新进程时直接接管。下一个进程的 k 在预启动时抽取，所以比课程状态的更新晚一个进程。`vecenv.make_vec_env` 在 `warm_pool` 时给第 i 个 env 分配
  train-2i 和 train-2i+1（8 个拷贝 = 4 个 env）。吞吐见 §7.1。

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

### 7.1 吞吐（2026-11-16，BallrainIO 51762f20…，f4 特征，action repeat 4，132 fps）

`scripts/m2_start_states.py throughput`：每个实例一个 OS 进程跑 `BallanceEnv` 60 s。null = 一直空动作（命不结束，直到 6 万帧的进程上限），
random = 随机粘滞动作（2–39 步换一次），curr = 从课程抽的 TAS 起点（SR01_1_14_457_264，最后 3000 帧内），warm = 预启动池（每个 env 两个拷贝）。
"拷贝"一列是占用的游戏拷贝数。时间占比是所有实例 wall time 之和的比例。

| 配置 | 拷贝 | 决策步/秒 | 帧/秒 | 命/小时 | 进程数 | CPU 占用 | 新进程（启动 / 交接 / 开局 / 等预启动） | 上个进程收尾 | env.step（游戏+IPC / 特征） |
|---|---|---|---|---|---|---|---|---|---|
| L1 null ×1 | 1 | 482 | 1946 | 0 | 2 | 0.14 | 10%（5 / 4 / 1 / –） | 2% | 88%（32 / 56） |
| L1 null ×4 | 4 | 1674 | 6763 | 0 | 8 | 0.25 | 13% | 1% | 84%（31 / 53） |
| L1 null ×8 | 8 | 2279 | 9248 | 0 | 16 | 0.38 | 19% | 1% | 77%（26 / 51） |
| L1 random ×1 | 1 | 60 | 677 | 2887 | 14 | 0.16 | 86%（32 / 30 / 6 / –） | 18% | 10%（4 / 6） |
| L1 random ×4 | 4 | 202 | 2221 | 9278 | 45 | 0.25 | 86% | 13% | 10% |
| L1 random ×8 | 8 | 306 | 3406 | 14268 | 75 | 0.39 | 85%（41 / 29 / 4 / –） | 12% | 11% |
| L1 curr random ×1 | 1 | 47 | 3061 | 1873 | 9 | 0.14 | 79%（23 / 44 / 0 / –） | 12% | 17%（3 / 15） |
| L1 curr random ×4 | 4 | 149 | 10877 | 6470 | 34 | 0.25 | 79% | 10% | 18% |
| L1 curr random ×8 | 8 | 231 | 16901 | 10018 | 54 | 0.41 | 77%（26 / 41 / 0 / –） | 8% | 20%（3 / 17） |
| L1 curr random warm ×1 | 2 | 59 | 4176 | 2604 | 13 | 0.23 | 63%（等 39%） | 10% | 27%（5 / 22） |
| L1 curr random warm ×4 | 8 | 184 | 13320 | 8171 | 42 | 0.34 | 64%（等 42%） | 6% | 25%（4 / 20） |

（"新进程"包含上个进程的收尾，收尾那一列单列出来：STOP、等进程退出、写 .tas。帧/秒含前缀回放的帧。）

**复测（f5 特征、运动体按射线范围裁剪、新 SR 奖励；同一构建）**。表中"特征 ms/步"= 所有实例的特征时间之和 / 决策步数（多实例时含 CPU 争用）；
"决策帧/秒"不含前缀回放的帧（上表的"帧/秒"含）。`l1_curr_null_x8` 是新加的配置：从课程起点（最后 3000 帧内，L1 末段的运动体附近）开始一直空动作，测那里的特征开销。

| 配置 | 决策步/秒 | 决策帧/秒 | 命/小时 | 进程数 | 特征 ms/步（之前） | CPU 占用 |
|---|---|---|---|---|---|---|
| L1 curr random ×1 | 47.5 | 189 | 2047 | 10 | 1.19（3.11） | 0.21 |
| L1 curr random ×4 | 153.6 | 611 | 6503 | 32 | 1.83（3.64） | 0.42 |
| L1 curr random ×8 | 222.0 | 883 | 9261 | 48 | 2.26（5.51） | 0.62 |
| L1 curr null ×8 | 1390.7 | 5563 | 733 | 16 | 3.01（–） | 0.69 |
| L1 null ×8 | 1936.6 | 7746 | 0 | 16 | 2.14 | 0.52 |
| L1 random ×8 | 305.1 | 1221 | 14160 | 73 | 1.95 | 0.46 |

随机策略的吞吐与之前相同（开新进程仍占 wall time 的 85–88%）；特征时间减半以上。末段一直停在运动体旁的 curr null ×8 平均每个 env 174 步/秒，特征占 wall time 的 53%，
没有出现单 env 30–100 步/秒的瓶颈（离线：SR01 第 6000–6999 / 18000–18999 / 20000+ 帧 34.8 / 12.4 / 27 ms → 1.9 / 2.1 / 2.0 ms，见 §2）。

- 随机策略的命很短（L1 约 1 s），时间几乎都花在开新进程上：启动 Player.exe 约 1.5 s、加载关卡 1.4 s、前缀回放（12200 帧/秒）、上个进程收尾。
  null 策略（命不结束）时，step 里一半以上是 Python 特征（×1 约 1.1 ms/步，×8 约 2 ms/步，射线扇 + 高度格 + 边界距离），游戏和 IPC 约 0.6 ms/步。
- 预启动池让每个 env 槽位快约 25–40%（×1：59 对 47 步/秒，2604 对 1873 命/小时），但每个拷贝更慢：同样 8 个拷贝，4 个预启动 env 是 184 步/秒，
  8 个普通 env 是 231 步/秒。启动时间 L、进程寿命 T 时，普通方式每个拷贝花 L+T，预启动用两个拷贝花 max(L, T)，2·max(L, T) ≥ L+T，所以拷贝数固定时普通方式总是不差。
  默认不开预启动池；拷贝更多（train-08..15）或策略变强（命变长）时再考虑。
- 第 2 关的 M1-E 在本构建上重测（`scripts/m1_gates.py --gates E`）：null ×1/×4/×8 = 416 / 1387 / 1981 步/秒，random ×1/×4/×8 = 54 / 188 / 289 步/秒
  （14845 命/小时），raw client ×8 = 331 步/秒；与 M1 时（早先的构建）的 359 / 1231 / 2006 和 51 / 170 / 280 相当。

## 8. 未验证 / 待测

- 角速度的坐标系；`q_core_f_object` 的旋转约定（样本里都是单位四元数）。
- 纸球的几何（多面体 × 每帧姿态）尚未实现，纸球只用 mindist。
- 风扇（P_Modul_18）的风力没有导出，也没有特征。
- 俯瞰（space）时的相机特征。
- 死亡快进期间（BallOff 到 BallNavActive）吃到的分数小球不计入任何回合的 HS 奖励（只在最终 HS 里）。死亡时仍在跟随的小球（用户）：会跟着球飞下去，但通常跟不上下落速度，所以通常丢掉；没有在数据上核对。
- 暂停菜单里方向键会移动菜单选项，死亡和变球动画期间 Esc 被禁用（用户）。env 默认在暂停中屏蔽方向键（`mask_keys_while_paused`），在没有控制时屏蔽 Esc（`esc_requires_control`），与之一致。
- 第 1 关脚本化的 Q 在 132 fps 下已在真实游戏上生效：从关卡加载开始，第 527 帧拿到控制（`scripts/m2_start_states.py measure`，k = 0）。
- 游戏进程崩溃或卡死（PeerGone / Timeout）时当前回合截断，帧日志以 Python 侧为准，下次 `reset()` 重开进程。
- SR 计时器在死亡动画和复活期间是否一律停止：导出 run 的数据都是如此（§4），没有在游戏机制层面核对，也没有看 BallOff 之前的最后几帧。
- `edge_mesh` 只用静态网格：球在运动体（飞艇平台、移动板）上时它不是到"脚下这块"边界的距离。射线扇用了运动体，两者在那里不一致。
- 纸球法向现在由 Python 按几何定向；mod 若导出球是 mindist 的第几个 synapse，就可以直接定向（需要改 mod 和重跑 M1-A）。
- 同样输入的轨迹分叉（§5.1）：原因未查。
- **计划：保留变球块**（用户）：TAS 回放（legacy 脚本修改）会去掉变球块，因为它们曾让 TAS 不可复现；训练环境也一样（L1 没有变球，M2 不受影响）。
  最终模型要能在正常游戏（有变球块）里工作，所以以后加一个 env 选项（默认关，保持与 TAS 兼容）：不执行 legacy 的 Set Position 删除，用于鲁棒性训练和评估。
  这样的 run 不能导出给 legacy TASSupport。变球之后按着的键是否还有力未核对（死亡 / 暂停会关掉力，见 §5"按键方向锁定"），遇到有变球的关卡时再测。

## 9. M2 训练（`scripts/train_m2.py`、`ballrain/train/`）

- **结构**：learner（本进程，PPO 更新）+ 每个游戏拷贝一个 rollout worker 进程（`ballrain/train/worker.py`），不用锁步的向量环境：
  gymnasium / SB3 的向量环境每一步等所有拷贝，任何一个拷贝开新进程（3–5 s）都会让其它拷贝停下。实测（L1 课程起点，随机粘滞策略，8 个拷贝）：
  AsyncVectorEnv 锁步 80 决策步/秒，8 个独立进程 222 步/秒（§7.1）。每个 worker 自己 step 自己的 `BallanceEnv`（含 reset），每 `chunk` 步把一段轨迹
  （归一化后的观测、原始观测、动作、行为策略的 log-prob 和 value、奖励、terminated / truncated、截断时末观测的 value、段尾 bootstrap value）发给 learner；
  learner 凑够 `batch_steps` 后做 GAE 和 PPO 更新，再把新权重、观测统计量和课程状态广播给 worker，worker 在下一个段边界换上（最多落后一次更新，
  超过 `max_stale` 的段丢弃；PPO 的比值用段里存的行为 log-prob）。
- **网络 / PPO**：策略和 value 各一个 MLP（默认 256-256 tanh），离散 9 动作；观测按运行均值 / 方差归一化（裁剪 ±10，第一批只拟合统计量）；
  GAE（γ 0.997、λ 0.95，132 fps × 4 帧一步时 γ 的视野约 10 s 游戏时间）；奖励乘 `reward_scale`（0.1）；BallOff / LevelFinish / GameOver 是终止（bootstrap 0），
  帧数上限、停滞、崩溃是截断（bootstrap 末观测的 value）。
- **回合上限**（worker 侧截断）：帧数上限 = `cap_mult`（2.5）×（TAS 从回合起点到 LevelFinish 剩下的游戏时间；复活回合从该扇区在 TAS 里的第一帧算）+ 10 s；
  停滞 = 参考线弧长 `stall_s`（10 s 游戏时间）内没有增加 5 单位（复活回合用 `--respawn-stall-s`）。截断结束这个游戏进程（`env.reset()` 在一条命中途会开新进程）。
  `--stall-terminal`（r2 从第 162 次更新起）：停滞按失败处理，终止（不 bootstrap），奖励加上 −(`death_penalty` + `death_refund` × 本扇区已付进度)，与死亡相同（项名 `stall_penalty`）。
  原因：L1 末段（参考线 s ≈ 1600–1700，TAS 在这里随运动体以 vy ≈ +2.8 上升）策略掉回下层后卡住，截断时只损失 −4 的进度加上 bootstrap 的时间价值，
  比死亡（约 −27）便宜得多，卡住成了便宜的失败。
- **按失败率加权的起点**（`CurriculumConfig.weighted` / `--cu-weighted`，r3 从第 4795 次更新起 0.5）：抽完 k = 0 之后，这个比例的起点先在 [前沿, 末尾] 里
  按 `bin_frames`（2000）帧分箱选一个箱，箱 b 的概率 = `bin_floor`（0.05）+ (1 − 箱数 × floor) × fail_b / Σ fail，fail_b 是训练端最近 100 个从该箱起步的
  TAS 回合里没有 LevelFinish 的比例（`set_bin_fail`，每次广播时更新；不足 10 个回合的箱按 1 算），再在箱内均匀抽帧；其余的起点照旧（band / 全窗口均匀）。
  原因：前沿停在 6686 时 70% 的起点都在 800 帧的 band 里，从第 3 扇区 TAS 状态起步的回合只占约 4%，那一段的通关率从 0.45 掉到 0
  （遗忘）；改成 recent 0.4 后 15 分钟仍是 0。已经掌握的箱保留 floor 份额用于复习。
- **停滞惩罚的折扣补偿**（`--stall-compensate`，r3 从第 5715 次更新起）：停滞惩罚在决定停下之后一个停滞窗口（4 s = 132 步）才到，
  按 γ = 0.997 折扣后只剩 0.67 倍，比当场死亡（`death_penalty` + 本扇区进度退还）便宜，所以在前面有难关时策略宁可停在路上
  （任务里点名的"因为 death_refund 而停住"）。补偿后惩罚 = −(death_penalty + 退还) / γ^窗口步数，停下永远不比当场死亡划算。
  证据：r3 前沿停在 6686 时第 3 扇区两段下落之间的平地（s 1340–1370）上停滞 1145 次，s_end 分布在 1340–1344、1353–1359 两段（停在线上，不是卡进坑里）；
  从平地上的 TAS 状态（第 13400–13600 帧）起步的回合 100% 死在 s 1375–1395 的腾空缺口，13800 帧以后（缺口之后）起步的 90% 通关。
- **助跑不算停滞**（`--stall-speed`、`--stall-max-s`，r3 从第 17457 次更新起 3 u/s、15 s）：用户：第 3 扇区的两次跳跃从静止也能完成，但要技巧，
  先往回退一段再冲（助跑），或者故意离开栏杆骑单根栏杆爬高。原来的停滞判定只看参考线弧长（4 s 内没有 5 单位新进度），会把助跑切掉。
  现在球速高于 `stall_speed` 时停滞计时暂停，另设 `stall_max_s` 秒没有新进度的硬上限，防止来回晃。进度奖励本来就可伸缩：退回去再回来，
  后退扣的和前进补的相等，净代价只有时间。
- **静止起点**（`EnvConfig.rest_starts`、`p_rest`，r3：第 13450–13650 帧，即缺口前的大平台，5%）：从 TAS 状态交接后，env 按世界方向逆着水平速度刹车，
  直到慢于 `rest_speed`（0.5，速度按特征的 speed + speed_change）或满 `rest_max_frames`，再交给策略；回合类型 `tas_rest`，不计入课程，
  日志 `rest_cp_sec*`。第 2 检查点之后（第 11760–11900 帧）那一段是下落，刹停水平速度后球直接往下掉，不能当静止起点；
  第 2 检查点处的静止状态由复活回合提供（死在第 3 扇区后从检查点静止起步），复活回合同样用新的停滞规则。
- **子目标窗口**（`--gates`，训练端整形，默认关；r4 从第 25690 次更新起用于 L1 的 shortcut 2）：`{名字: {center, radius, bonus, min_speed, heading, heading_tol}}`，
  每条命里球第一次进入窗口（距中心 ≤ radius，水平速度 ≥ min_speed，速度方向 atan2(vz, vx) 与 heading 相差 ≤ heading_tol）时这一步加 bonus（项名 `gate_bonus`），
  每个窗口每条命最多一次。窗口中心取 TAS 自己在零售回放里的状态（起跳 TAS 第 4945 帧、落地第 5050 帧、落地后向南约 0.5 s 第 5182 帧），宽度参考人类录像的分布；
  奖励几个单位，远小于一次死亡。tensorboard `gate/<名字>_hits`、`_hit_rate`。速度单位是球速度字段（约为位置差分得到的单位/秒的一半）。
- **路线探针**（`--route-probes`）：`{名字: [xmin, xmax, zmin, zmax]}`，每个回合记录球进过哪些框、BallOff 前最后在哪个框（`probes`、`death_probe`），
  tensorboard `route/<名字>_visits`、`_deaths`；L1 最后一个扇区用来区分 TAS 的坡上跳跃和绕路。
- **远程 worker 的看门狗**：桥每 30 s 写 `remote-<tag>/heartbeat.json`；`scripts/remote_supervisor_m2.py` 管理桥进程，进程退出、心跳超过 180 s、
  连接着但 10 分钟没有转发、或每分钟超过 5 次 worker 崩溃时，按 STOP → 等待 → 结束它自己的进程树 → 按路径停掉剩下的 Player.exe → 用当前代码重启。
  learner 端：10 分钟没有远程数据时 train.log 打 `WARNING remote`，tensorboard `remote/alive`、`remote/chunks_per_min`。
  桥代码比 learner 旧时不认识的模板字段被丢弃并警告（需要重启桥）。
- **复活回合**：`--no-respawn`（`EnvConfig.per_life=False`）时 BallOff 结束进程，下一条命是新的课程起点。r2 前期用它（策略接近均匀时复活回合几乎都停在检查点不动，
  占了 2/3 的步数），策略熵降到 1.4 以下后恢复复活回合：验收是从关卡加载开始的整关，死亡后要从检查点静止起步，复活回合正好是这些状态。
  日志 `rsp_cp_sec1..4` = 从各扇区检查点静止起步、到达下一个检查点（或 LevelFinish）的比例。
- **课程成功**（`--success cp`，默认）：一个 TAS 起点的回合（`start.kind == "tas"`，含 k = 0）在这条命里到达下一个检查点（扇区号变大）或 LevelFinish 即成功；
  `--success finish` 只认 LevelFinish。复活回合不计入课程。learner 持有唯一的 `ReverseCurriculum`，用 `update(k, success)` 更新，
  前沿后退的规则见 §5.1（band 内最近 `min_episodes` 个回合成功率 ≥ `advance_at`）；前沿到达第一个控制帧后 k = 0 的概率改为 `--p-level-start`（0.3）。
- **日志**（`runtime/m2-train/<run>/`）：`updates.csv`（每 `log_every` 次更新一行：步数、步/秒、前沿及其扇区、band 成功率、各起点扇区和各 2000 帧起点区间的
  成功率 / 通关率、回合回报、死亡 / 停滞 / 上限次数、按扇区的死亡数、通关进程的 SR、起点分位数、损失、熵、KL、explained variance）、
  `episodes.csv`（每个回合：起点、结束原因、回报和各奖励项、步数、游戏时间、平均前进速度、跳变次数、动作分布）、`runs.csv`（每个游戏进程：结束原因、正式 SR、死亡数）、
  `tb/`（tensorboard，含起点分布直方图）、`train.log`、`ckpt_*.pt` / `latest.pt`（模型、优化器、观测统计量、课程状态、计数器；`--resume` 续训）。
  种子：learner `--seed`，worker i 用 `seed·1000 + i`（worker 重启时加 100 × 重启次数，续训时加 7919 × 已完成的更新数），写在 `config.json`。异步采集本身不可逐位复现。
- **磁盘**：每个进程目录约 1.8 MB（主要是 level.msg）。worker 读完元数据后删掉没有通关的进程目录，通关的只留 `.json`、`.tas`、`events.txt`（`--prune none` 全留）。
- **停止**：在 run 目录下建 `STOP` 文件；learner 通知 worker 关闭游戏进程，之后按路径（`instances.running_in_copy`）停掉所用拷贝里还在运行的 Player.exe。
- **评估与导出**（`scripts/eval_m2.py`）：从关卡加载（k = 0，脚本化的 Q）闭环运行，argmax 或低温采样（每个 run 一个种子），死亡后同进程复活；
  `--export n` 把最快的 n 个通关 run 的 .tas 交给真 TASSupport 回放 3 次、BallrainIO 回放 1 次，用 `gates.export_diagnosis` 判定。
- **导出验收规则**（用户）：允许对接触顺序敏感的导出。真 TASSupport 回放 5 次，取多数；5 次分成 3:2 或更接近时再加 2 次（共 7 次）。
  报告写明每个通过的导出的零售复现率（k/N 与 agent 轨迹逐字节相同）和不同轨迹的个数（`eval_m2.py` 的 `ref_identical_to_agent`、`ref_distinct_traces`）。
- **零售的球-球接触顺序**（`tools/analysis/ball_order_dist.py`，50 个新进程，x87-soft physics_RT.dll 的接触对日志，只在测量期间放进 runtime/ref，之后换回 2002 版并核对 sha256）：
  IVP 按堆地址给球-球接触对排序，场景里池化的球每个进程地址顺序不同。L1 上出现在接触里的 7 个球：Ball_Wood 永远最低（50/50）；
  第 2、3 位永远是 P_Ball_Stone_MF 和 P_Ball_Wood_MF，Stone_MF 在前 33/50；P_Ball_Stone_MF001–004 永远在后四位，24 种排列出现了 21 种，
  对均匀分布的 χ² = 23.9（自由度 23），与均匀随机一致。这就是 spot-check 里零售自身约 8% 的回放分叉的来源（run 282）。
- **接触顺序诊断**（`sim2real_m2.py orders`，在有无头引擎的机器上跑）：同一个 .tas 在无头后端按 `BALLRAIN_HL_BALL_ORDER` = ckid / reverse / perm:<seed>… 各回放一次，
  记录每种顺序的轨迹哈希、结束原因、LevelFinish 帧、检查点数、SR、第一处分叉；轨迹不同 = 顺序敏感，结果（通关 / 检查点 / SR）不同 = 结果敏感；
  tensorboard `eval/order_sensitive_rate`、`eval/outcome_sensitive_rate`。只作诊断，不拒绝任何导出。引擎侧的顺序模式由无头后端实现（接口同上，`HeadlessConfig.ball_order`）。
- **CPU / GPU**：PPO 一次更新（4096 步 × 4 轮，minibatch 1024）CPU 4 线程 0.19 s，cuda:0 / cuda:1 0.11 s（GPU 同时被其它程序占满），
  而采集 4096 步约 9 s 且与更新并行，GPU 端到端最多快 1%：用 CPU。

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
