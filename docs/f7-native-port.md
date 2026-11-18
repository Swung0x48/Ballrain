# f7 reach 块：原生移植要照的定义

f7 = f6 + 末尾一块 `reach`（`FeatureConfig.reach`，`ballrain/reach_features.py` 包装 `ballrain/fastreach.py`）。现在没有部署：
资源池（`pool.weights.export_state`）和原生客户端（`FeatureConfig::from_json`）都拒绝 f7，`tools/export_model.py` 对 f7 检查点不导出 `plain/`。
本文是 C++ 移植要逐项照着做的定义；Python 代码是参考实现，两者冲突时以代码为准并修正本文。当前版本 `fastreach5.2-block1`。

## 1. 块在向量里的位置

- 永远是最后一块，59 个 float（`fastreach.FEATURE_NAMES`）：19 个头部量，然后 8 个方向 × 5 个量。
  r8 的 f6 布局 208 维，f7 是 267 维，reach 块是 [208, 267)。布局由 `features.make_layout` 决定（`layout_size` / `layout_slices`）。
- 头部量顺序：`grounded, doomed, unknown_zone, air_any, air_t_first, air_first_dy, air_near_r, air_near_f, air_near_dy, air_t_near,
  air_near_effort, air_near_can_stop, air_near_clear, sup_n_r, sup_n_y, sup_n_f, side_n_r, side_n_y, side_n_f`。
- 方向顺序（`DIR_NAMES` = `reach.KEY_COMBOS` = `env.ARROWS9[1:]`，方向槽 q 对应动作 q + 1）：
  `up, down, left, right, up_left, up_right, down_left, down_right`；每个方向依次 `edge, brake_margin, gap_deficit, obst, obst_deficit`。
- 没有 `valid`（找到球、类型 0/1/2、不在变球）时整块为 0；位置或速度有非有限值时整块为 0（Python 的 `perceive` 会拒绝这种状态）。

## 2. 输入

- `pos` = BALL `pos`（float64），`vel` = BALL `speed + speed_change`（与 `vel_cam` 同一个量），球种 `tracker.ball_type()`（0 木、1 纸、2 石）。
- 按键象限 `quarter`：`camera="world"`（世界方向动作）时为 0；否则 `round(yaw(Cam_OrientRef 矩阵) / (π/2)) mod 4`（`FeatureExtractor.action_quarter`）。
  8 个方向的世界 (x, z) 单位向量和推力大小由 `reach.key_push(KEY_COMBOS, quarter)` 算出，**包括** `sin/cos` 的舍入（象限 1 时
  `up` = (1, 6.123233995736766e-17)，对角线大小 1.414213562373095）：移植时用同样的表（4 个象限 × 8 行），不要写成精确的 0/1。
  相机右 / 前向（法向量的投影用）是 `key_push(KEY_RIGHT)` / `key_push(KEY_UP)`。
- 运动体：BODY 段里 `body_radius`（`FeatureConfig.body_radius`，60）以内、不在 stopper 组的条目的位置；任意一个离球 **< 8**（`UNKNOWN_R`）时
  `unknown_zone` = 1（在内核之后由包装函数设置）。
- 未知区：`FastReach.zone_boxes(level, exclude_names=BALL_NAME_TYPES)`：`phys_by_ent` 里没有 unphysicalize、非 fixed、不在 stopper 组、
  名字不是 `Ball_Wood/Ball_Paper/Ball_Stone` 的实体，取 `mesh_world_bbox`（有 Physicalize 记录时用它带的网格矩阵），每边外扩 4。
  球心在任一盒内（闭区间）时 `unknown_zone` = 1。每次关卡数据版本变化时重算；它们不进场的键。

## 3. 场（static fields）

### 3.1 几何与规范顺序

- 三角形：`features.level_geometry(tracker)`，即 `LevelData.static_entities()`（物理化、fixed、不在 stopper 组）的全部网格，
  顶点用网格自带的世界矩阵变换（`geometry.ck_to_world`，float64）；轨道三角形是其中在 Phys_FloorRails 组的实体。
- `FastReach.canonical_geometry`：不依赖 CK id 的顺序。
  1. 每个实体（三角形的 owner）的键 = (实体名（没有名字时空串），该实体按 9 个坐标 A.x, A.y, A.z, B.x … C.z 字典序最小的那个三角形)；
     实体按这个键排序（名字按 Unicode 码点比较，再比坐标），得到 rank 0..k−1。
  2. 三角形按 (rank, A.x, A.y, A.z, B.x, B.y, B.z, C.x, C.y, C.z) 字典序排序。
  3. `is_rail` = 三角形的 owner 是轨道实体。
- 零售和无头 SimTool 的 L2 静态实体和网格逐位相同但 CK id 不同；按上面的顺序，零售与 6 份无头关卡消息的键相同。

### 3.2 键和文件名

`FastReach.fields_key`：

```
static = "canon-" + hex(sha256( A(<f8, n×3) ‖ B ‖ C ‖ rank(<i8, n) ‖ is_rail(u1, n) ))     # 全部小端、行优先
h = sha256()
for (…, lo, hi) in level.depth:                       # DepthTestCubes，按关卡数据里的顺序
    if min(hi.x − lo.x, hi.z − lo.z) >= 20:  h ‖= (lo.x, lo.y, lo.z, hi.x, hi.y, hi.z) as <f8
h ‖= b"|"                                              # （没有未知区）
h ‖= b"|fans"
h ‖= fan_obbs(fan_boxes(level)) as <f8 (k×18, 行优先)
h ‖= (20.0, 2.0, 20.0, 66.0, 1.4, 0.5, 1.95, 0.1, -1, 0, -1, 1, 16, 1, 0.1) as <f8
    # death_min_extent, R_BALL, GRAVITY, PSI_HZ, ROLL_INERTIA, UP_FACE, MOUNT_CAP, MOUNT_FREE, FAN_LOCAL_LO, FAN_LOCAL_HI, FAN_FORCE
key = f"{static}:{hex(h)[:16]}:{BUILD_VERSION}"
file = f"L{level_no:02d}-{hex(sha256(key.utf8))[:16]}.npz"
```

`fan_boxes`：`level.by_name` 里名字是 `P_Modul_18_Kollisionsquader` 或它后接纯数字（`…001`）的实体，**按名字排序**，取实体的世界矩阵
（4×3 行向量：x、y、z 轴，位置）。`fan_obbs` 每行 18 个数：中心 `0.5 (lo + hi) @ A + M[3]`（lo = (−1, 0, −1)，hi = (1, 16, 1)，A = M 的前三行），
半长 `0.5 (hi − lo) · |A 的各行|`，单位化的局部轴（3 行），世界 AABB 半长 `|U|ᵀ @ e`。

当前 L2 的两份场：`L02-66e8d101e6cd97d4.npz`、`L02-86886f7d56fff7a5.npz`（第 3–4 小节的机关物理化时静态几何不同）。

### 3.3 文件格式

`np.savez`（不压缩的 zip，每个成员是 `.npy`；原生端可以用已经 vendored 的 miniz 读）。成员（L2 的形状）：

| 成员 | dtype | L2 形状 | 内容 |
|---|---|---|---|
| `info` | `<U…`（0 维 UTF-32 字符串） | () | JSON：`key`、构建时间、`fans`（名字、中心、半长、轴）、`built_by` 等；只有 `key` 是语义的 |
| `tiles` | int32 | (402, 130) | 4×4 单位的瓦片索引（−1 = 空） |
| `tile_ij` | int32 | (5680, 2) | 瓦片坐标 |
| `Hh` / `Hn` | float32 / int8 | (5680, 8, 8, 3) / (…, 3, 2) | 每格最多 3 层可走面高度 / 法向（int8 编码） |
| `P1`, `P2`, `Pd`, `Poff`, `Pdim` | float32, float32, float32, int64, int64 | (5680,4,4,2), (5680,2,2,2), (69856,2), (11,), (10,2) | 高度金字塔 (min, max) |
| `Rr`, `Rl`, `Rn`, `Rf` | float32, float32, int8, uint8 | (5680, 8, 8, 4[, 2]) | 支撑场：每格最多 4 簇的静止高度、下沿、法向、标志 |
| `bidx` / `bricks` | int32 / int16 | (403, 27, 131) / (19379, 9, 9, 9) | 稀疏 SDF：8³ 体素砖的 9³ 角点，定点 1/4096，截断 3 |
| `seg`, `rstart`, `ritems` | float64, int32, int32 | (580, 7), (52261,), (5145,) | 轨道管段（轴端点 + 半径）和按瓦片的 CSR 列表 |
| `boxes` | float64 | (3, 6) | 保留的死亡盒 (lo, hi) |
| `uzones` | float64 | (0, 6) | 场文件里总是空；运行时填未知区 |
| `fans` | float64 | (9, 18) | `fan_obbs`；BUILD_VERSION 5 以前的文件没有这个成员（键也对不上） |
| `meta` | float64 | (8,) | `ox, oz, sx0, sy0, sz0, void_y (= 最低面 − 30), r, BUILD_VERSION` |

每个数组的含义、构建方法和查询时的读法见 `fastreach.py` 模块文档和 `FastReach.build`。读入后 float32 / int8 / int16 的值在内核里转成 double 用，
移植时同样先按 Python 的公式转 double（例如 SDF `int16 / 4096`），不要在 float32 里算。

谁来构建：见 §7 未决问题。

## 4. perceive 的语义

参考实现是 `fastreach._perceive`（numba 编译和纯 Python 两条路径结果逐位相同，`tests/test_fastreach.py`），加上 `FastReach.perceive`
包装里的两步（运动体 → `unknown_zone`；`unknown_zone` = 1 时 `doomed` 清零）。模块文档分段描述了每个量：

- **Air**：`y(n) = y0 + vy S1(n) − g S2(n)`，按键可达集合是边长 2 D(n) 的正方形（D = a_air S2），金字塔深度优先遍历找可达落点；
  `air_any`、最早落点、最省力落点（离无按键飞行最近）、落地后能否停下、飞行路径是否无遮挡。视界 `N_PSI` = 396 PSI（3 游戏秒）。
- **Ground**：8 个方向的路边距离、刹车余量、缺口所需速度差、台阶 / 墙及其所需速度差，支撑面 / 侧面法向（按键坐标系）。
- **Doomed**：腾空且死亡高度之前没有可达落点；接地（不在轨道上）且全刹车路径离开支撑面、刹车离开和沿速度全推起跳都没有可达落点。
  视界内既没有落点也没到死亡高度 = `unknown_zone`，不是 doomed。
- **unknown_zone 优先**：`unknown_zone`（未知区、运动体、视界、风扇）为 1 时 `doomed` 必须为 0。
- **风扇**（P_Modul_18，L2 有 9 个；推力 = 每 PSI 向上 `0.1 / m`：纸 0.5、木 0.0526、石 0.01 < `FAN_MIN_DV` = 0.02 → 石球忽略风扇）：
  1. 无风扇扫描；`_fan_touch` 找可达集合（无风扇高度下的正方形）第一次进入任一推力盒世界 AABB 外 `FAN_R_OUT` = 3.6 的 PSI `ft`。
     没有、或两个无风扇落点都在 `ft` 之前：结果不变（dbg fan 0）。
  2. 否则沿无按键飞行算风扇剖面（`_fan_profile`：PSI n 的球心离推力盒 ≤ `FAN_R_NOM` = 2.85 时 `v[n+1] = λ v[n] − g + dv`，否则不加 dv；
     PSI 0 的推力不加，给定速度已含它）。两个无风扇落点都在剖面第一次被推之前：保留无风扇结果。
  3. 扫描剖面（given 1）：接触测试用剖面实际高度；PSI 窗口用实际高度的前缀最小值和后缀最大值（都是非增的，窗口精确）；
     剖面第一次被推之后，离所有推力柱（推力盒沿 y 无限延伸的水平距离，`_fan_col_dist`）超过 `FAN_R_NOM + rc` 的点只在
     "此刻没被推，且离柱距离 ≤ 上次被推以来无按键飞行的水平位移 + D(n − 上次被推的 PSI)" 时才算。有落点 = fan 1（特征用它）。
  4. 没有：有无风扇落点 = fan 2（恢复无风扇的特征和 collect）。
  5. 都没有：高度带扫描（given 2）：下界 = 无风扇公式，上界 = `_fan_upper`（从 `ft` 起每个 PSI 都被推），带里任一高度能在正方形内接触到的点算可能落点；
     下降段从下界的顶点开始，死亡 = 上界穿过死亡高度。有可能落点或上界不死 = fan 3（`unknown_zone`，doom_reason 5），否则 fan 7（doomed，doom_reason 6）。
  6. 接地且离推力盒 ≤ `FAN_R_OUT`：从不 doomed（fan 4），被推起来的球（dv > g/66，纸）离盒 ≤ `FAN_R_IN` = 1.95 时 `unknown_zone`。
     接地 doomed 测试里的刹车离开 / 全推起跳两次起飞扫描没有落点且能碰到推力盒时做同样的高度带测试：安全 = fan 5 / 6，否则 fan 8 / 9。
  dbg 的 fan 代码不在块里，只用于对照。
- 物理常数（`fastreach.CONSTS`）：g = 20，PSI 66 Hz，滚动惯量 1.4，可走面 n_y ≥ 0.5，上台阶 1.95 / 0.1。球（`reach.BALLS`）：

| 球 | 质量 | 半径 | air λ | air 推力 | ground λ | ground 推力 | 风扇 dv |
|---|---|---|---|---|---|---|---|
| 木 0 | 1.9 | 2.0 | 0.9863636363636363 | 0.22631578947368422 | 0.9898268398268398 | 0.16165413533834588 | 0.052631578947368425 |
| 纸 1 | 0.2 | 2.004 | 0.9772727272727273 | 0.325 | 0.9833333333333333 | 0.23214285714285718 | 0.5 |
| 石 2 | 10.0 | 2.0 | 0.9954545454545455 | 0.092 | 0.9963203463203463 | 0.06571428571428571 | 0（忽略） |

## 5. 缩放（`reach_features.scale`）

`v = raw × SCALE`，然后 NaN → 0、+inf → 1、−inf → −1，最后截断到 [−1, 1]；进入观测时和其余条目一样转 float32（全局 clip 10 不起作用）。
`SCALE`：`air_t_first`、`air_t_near` 为 1/3（秒，视界 3 s）；`air_first_dy`、`air_near_r`、`air_near_f`、`air_near_dy` 和 8 个方向的全部 5 个量为 1/20
（fastreach 自己把长度截断在 ±`L_MAX` = 20、速度差在 ±`CLIP_V` = 20）；其余（标志、法向、`air_near_effort`）为 1。
未接地时方向量是默认值（edge、brake_margin、obst = 20 → 1，两个 deficit = −20 → −1）。

## 6. 版本钉

- `FeatureConfig.reach_version` = `reach_features.reach_version()` = `f"fastreach{BUILD_VERSION}.{QUERY_VERSION}-block{BLOCK_VERSION}"`，
  `reach` 打开时自动填（学习器的配置因此把版本带给每个 worker）；f6 为 None，不进 f6 清单。
- `BUILD_VERSION`（fastreach，现 5）：场的构建或数组变化时提升（进场的键，旧文件自动重建）。`QUERY_VERSION`（fastreach，现 2）：perceive 的任何语义变化。
  `BLOCK_VERSION`（reach_features，现 1）：名字、顺序、缩放、未知区、运动体规则的变化。
- `tests/test_reach_obs.py::test_block_definition_digest_matches_the_block_version` 把名字、方向名、SCALE、N_PSI、L_MAX、CLIP_V 的摘要
  （sha256 前 16 位 `3a67104b827836b8`）和版本串钉在一起，改了其中任何一个而没提升版本会失败。
- worker 端：`ReachBlock(expect_version=…)` 在任何计算之前比较，不同就抛 `ReachVersionError`。原生客户端必须有自己的版本串，
  清单里的 `reach_version` 与之不同时拒绝任务（与 `path_sha256` 一样），不得静默计算。

## 7. 容差与验证

- 目标是逐位一致：内核全是 IEEE double 的加减乘除、`sqrt`、`floor`、`ceil`、比较，与 numba / 纯 Python 的结果逐位相同已经测过。要做到逐位：
  - 运算顺序照抄（包括 `max(…, 0.0)` 的位置、`S1/S2` 的闭式 `(1 − λᵏ)/(1 − λ)/66`）；编译时禁止浮点收缩（GCC/Clang `-ffp-contract=off`，MSVC 不开 `/fp:fast`、
    不开 `/fp:contract`），不用 fast-math。
  - 依赖 libm 的只有：`air_tables` 里的 `λ**k`（numpy `pow`）、刹车停止 PSI 的 `log`、侧面探针的 `cos/sin(q π/4)`、按键方向的 `sin/cos(q π/2)`。
    建议直接嵌入 Python 生成的表（3 种球的 S1/S2、8 个探针方向、4 个象限的方向表），或者在自检里与 Python 的表逐位比较。
  - 场的 float32 / int 数组按 Python 的方式转 double。
- 验证（与 f6 的 `tools/native/golden.py` 同样的做法）：L1、L2 的 TAS 回放每 2 帧一个状态，加上 r8 的追踪记录（`runtime/m2-train-mac/r8/episodes-traced`），
  Python 的块作金标准。要求：连续量缩放后 |Δ| ≤ 1e-9（逐位一致时为 0）；标志量（`grounded`、`doomed`、`unknown_zone`、`air_any`、`air_near_can_stop`、
  `air_near_clear`）在全部状态上相同，若有不同，必须能指出对应的阈值量（落点接触半径、死亡高度、推力距离、`FALL_TOL` 等）离阈值在 1e-9 以内，
  且 L2 TAS 上 `doomed` 必须为 0、L2 坑的死亡必须与 Python 一样全部提前标出（当前追踪集 79/79）。
- 运行时：L2 TAS 上 Python（numba）`perceive` 中位数 47 µs、p99 0.97 ms（reach 块含未知区和运动体 137 µs / 2.1 ms，机器有负载），风扇状态最慢
  （fan 1 中位 0.70 ms、p99 6.3 ms）；场每份约 58 MB，构建约 4 s。

### 7.1 算力池里的验证：可达性块放宽（2026-11-18，用户决定）

fastreach 是近似模型：标志量卡在阈值上翻转，或者落点出现 / 消失引起连续量跳变，本身没有物理意义。所以验证器（`rollout.replay_task`、`server_verify`）
**不要求原生移植逐位一致**，只放宽可达性块（每行观测里 `features.reach` 的 59 个数），其余照旧严格：

- **其余部分严格不变**：观测的其余条目（OBS_TOL）、引擎状态链（逐位）、奖励、flags、结束原因、可选数组、按 `sample_seed` 抽的动作。
- **可达性偏差行**：可达性块超出 OBS_TOL 的行。在这一行上，log-prob、价值和动作抽取用“混合行”核对：验证器自己的 float32 特征，可达性块换成客户端提交的值，
  也就是客户端的策略实际看到的那一行。容差用便宜检查那一档（log-prob 0.05、价值 2%、累积分布边界 0.01），因为提交的可达性值是 float16 副本。
  动作仍必须是混合行上按任务种子抽出来的那个，所以客户端不能借这些行随意选动作。
- **两道上限**（`[verify]`；2026-11-18 按原生移植的实测定下，见本节末尾）：
  - `reach_max_row_frac` = **0.02**：每段数据里偏差行最多占 2%，但**至少允许 1 行**（第 0..T−1 行，最终段再加最后一行；256 步的段最多 5 行；短的最终段（通关、死亡所在的段）至少允许 1 行，否则一次诚实的翻转就会把它拒掉）；
  - `reach_max_logp_shift` = **1.0**（nat）：偏差行上，提交的可达性值让所选动作的 log-prob 变化不能超过这个值，即 |log π(a | 提交的可达性) − log π(a | 验证器的可达性)|。
  超出任一上限：**只拒收这一段**（`replay_mismatch`，日志写明原因），**不标记账号**，同一任务的其他段照常通过。
  只有严格部分不一致时，才走原来的处理：拒收任务、标记为可疑、写提醒。可达性偏差本身永远不会让账号被标记。
- **送进学习器的数据**：通过的段如果有偏差行，验证器写一个修正文件 `chunks/<任务>/<段>.reach.npz`（行号、验证器算的 float32 可达性值；也包括段末那一行，即学习器的自举行 `raw_last`，它只修正、不计入比例）。
  学习器链接送数据时套用它，学习器训练用的是验证器的可达性值（更安全）。behaviour log-prob 仍是客户端提交的，V-trace 会修正这个差别。
- **剩下的风险**：
  - 在两道上限之内，客户端可以在每段最多 2% 的行上改可达性值，让所选动作的概率最多变化 e 倍，对动作有轻微影响；
  - 抽查放行、没有重放的段，偏差行无法发现，也无法修正，学习器用的就是客户端的值；
  - 上限偏紧时会误拒诚实的段（只是丢数据，不伤账号），偏松时操纵空间变大。
- **实测和定值**（原生 0.4.0，`e7cdd19`）：
  - 55 745 个状态（L2 TAS 记录 30 000 个，合成 25 745 个）：用系统 ucrtbase 时，C++ 内核与 numba 逐位相同；用静态运行库时，31 个状态差 ≤ 4e-16，没有标志翻转，没有跳变；
  - r8 的 f7 检查点：policy 20 000 行的特征最大差为 0，动作全部相同。
  - 所以诚实的客户端几乎不会产生偏差行。两道上限保持 0.02 和 1.0 nat，作为异常保护。以后如果出现诚实的 “rejected for reach”，只改配置和本节。
- 被这样拒收的段，玩家在统计里看到的是通用的“重放结果与提交不一致”（`replay_mismatch`，令牌不受影响）。
- 测试：`tests/test_pool_reach_relaxed.py`（假游戏上用一个特征段代替可达性块，扰动验证器自己的行，模拟诚实的翻转；另有一个测试核对真实 f7 布局里的可达性块是最后 59 个数）。

## 8. 未决问题

- **谁为 C++ 客户端构建场：已定（2026-11-18，方案 (a)）。** 网关用 Python 构建场，作为任务素材下发；客户端只做查询，以及算 `canonical_geometry` 和场键（判断当前几何用哪份场：L2 一局里会换两次）。通信格式由算力池会话和原生客户端会话商定：

  **任务**（protocol `TASK.assets`，可选的 `reach`）：最多 8 项，每项对应这一关可能用到的一份场（L2 两份）：

  | 字段 | 类型 | 含义 |
  |---|---|---|
  | `level` | 整数 | 关卡 |
  | `key_sha256` | 64 位小写十六进制 | `sha256(fields_key 的 UTF-8)`（§3.2）：客户端按当前几何算出键，用它找到对应的这一项 |
  | `sha256` | 64 位十六进制 | **未压缩** `.npz` 的 sha256 |
  | `size` | 整数 ≤ 128 MiB | 未压缩的字节数 |
  | `zsha256` | 64 位十六进制 | zstd 压缩文件的 sha256 |
  | `zsize` | 整数 ≤ 32 MiB | 压缩后的字节数 |

  - **下载地址不进任务**，和其他素材一样由 sha 推出：`GET /pool/assets/<zsha256>.zst`（公开、不可变、`Cache-Control: immutable`，可以让 Cloudflare 缓存）。以后要换 CDN，另加一个单独评审的字段。
  - **客户端**：
    1. 核对 `zsize` 和 `zsha256`；
    2. zstd 解压，输出上限就是 `size`（防解压炸弹）；
    3. 核对 `sha256`；
    4. 存为 `<根目录>/cache/fastreach/L<nn>-<key_sha256 前 16 位>.npz`（客户端下载的素材都在 `<根目录>/cache` 下），文件名与 Python 缓存的相同。根目录是 `BALLRAIN_WORK_DIR`，默认 `~/.ballrain`。客户端只写在这个根目录下，不另设路径环境变量；
    5. 读入后要求 `info.key` 等于自己算出的 `fields_key`。

    当前几何的键在 `assets.reach` 里没有对应项时，这个任务无法计算，以 `error` 结束（不算不一致）。
  - **清单**（`weights.export_state`）：保留 `features.reach` 和 `features.reach_version`（§6 的版本钉）；`requires` 写 `"reach_" + reach_version`（`.` 换成 `_`），例如 `reach_fastreach5_2-block1`（`weights.reach_capability()`），最低客户端 0.4.0。
    以后 `QUERY_VERSION` / `BLOCK_VERSION` 提升，旧客户端会被按能力挡下，收到说明为什么要更新的提示，而不是领到任务后才在版本钉检查上失败。
    场的 sha 不进清单：场属于关卡几何，不属于策略，所以在任务里。
  - **网关**：发布 f7 策略时，按策略的开局记录（`level.msg` / `.inc` 和观测）像 `tools/reach_warmup.py` 一样回放，用 `FastReach.fields_for_tracker` 在服务端缓存 `<runtime>/fastreach` 里构建或载入这一关的场（L2 两种几何都会走到）。键末尾的 `BUILD_VERSION` 不是版本钉里那个的文件一律不下发。然后用 zstd 压缩放进素材库，把这些项写进这个策略的每个任务。
    实测：L2 一份场原始 58.5 MB，zstd 9 级后 4.0 MB。场按几何而不是按策略区分，所以策略更新时 sha 不变，客户端只下载一次。
  - **验证器**：用同一个服务端缓存 `<runtime>/fastreach` 重放。重放前核对任务列出的每一项在缓存里的文件 sha256，缺文件或不一致时记 `replay_failed`，之后重试。所以客户端和验证器用的是逐字节相同的场。
- 风扇模型的近似：推力判定用无旋转的 `FAN_R_NOM` 规则（r8 追踪上 96.1 % 一致，精确 OBB 规则 98.8 %）；fan 1 / 2 的特征只沿无按键飞行；
  高度带测试对 doomed 是保守的（带里每个高度都算可能），所以木球在风扇旁有落点在带里时是 unknown 而不是 doomed。
- 只有 L2 有风扇和实测；其他关卡的移动机关只作为未知区，不进场。
- 构建锁只在同一主机的进程之间有效（文件锁 + 心跳，25 s 视为残留）；网络文件系统上的共享缓存没有测过。
