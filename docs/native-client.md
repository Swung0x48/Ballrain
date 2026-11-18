# 原生志愿者客户端（native/，C++17）

志愿者算力池的玩家端程序：不需要 Python，直接驱动无头引擎（x87-soft 物理），自己算观测特征、跑策略、按服务器种子抽动作。
Python 侧（`ballrain/pool`、`tools/pool`）是训练 / 网关 / 校验方。设计背景见 `volunteer-compute.md`、`pool-prototype.md`。

里程碑：**M1** 核心库 + 回合运行器（§1–4）；**M2** 池协议客户端 + 校验改为跨 CPU 容差（§5）；**M3** Windows 测试版：任务格式 3（§5.0）、
第 2 关、控制器（多实例、上传队列、Job Object、自动暂停）、终端界面、浏览器登录、诊断导出、签名更新检查、打包、CI（§7–10）。
玩家看的说明在 `native-client-testers.md`。

## 1. 结构

| 部分 | 位置 | 说明 |
|---|---|---|
| `ballrain_core` | `native/core/` | 纯可移植 C++17：线格式（`wire`，对应 `ballrain/protocol.py`）、游戏状态（`state`，对应 `state.py`）、几何（`geometry.py`）、参考线与投影（`path.py`）、特征（`features.py`，f6）、策略与抽样（`pool/weights.py` 的 NumpyPolicy、`pool/sampling.py`）、JSON（含 Python `json.dumps` 规范输出）、sha256、npz 读取（miniz）、录制流格式 |
| `ballrain_engine` | `native/core/src/session.cpp`、`env.cpp` | 一个引擎进程的会话（`client.py` AgentSession + `headless.py` 启动方式 + `transport.py` SocketChannel 的 Python 一侧）和环境（`env.py` BallanceEnv 中池任务用到的部分：L1 开场 Q、世界方向动作、按住键在相机换向后重按、相机守卫、按命重生） |
| 平台层 | `native/platform/` | 唯一的系统相关代码：本地 TCP、子进程、单调时钟、目录、环境变量。`platform_win.cpp` / `platform_posix.cpp`（macOS / Linux / Android） |
| 池客户端 | `native/core/src/pool_client.cpp`、`task_runner.cpp`、`pool_task.cpp` | 网关协议、任务（`rollout.py` 的移植）、一个任务从缓存到数据段 |
| 控制器 | `native/core/src/controller.cpp` | `client_core.py` 的移植，不含界面代码：配置、令牌、游戏目录检查、登录、会话、实例线程、上传队列、自动暂停、日志（§7） |
| 诊断 / 更新 | `diagnostics.cpp`、`update.cpp` | 脱敏后的诊断 zip；ed25519 签名的发布清单（§8） |
| 程序 | `native/apps/` | `ballrain_pool`（玩家用的终端界面，FTXUI）、`nc_client`（控制器的命令行驱动，端到端测试用）、`nc_task`（离线任务）、`nc_runner`（在线跑回合）、`nc_golden`（离线黄金对比）、`nc_release`（发布签名）；`native/tests/selftest.cpp`（不需要引擎的单元检查） |
| 第三方 | `native/third_party/` | miniz（MIT）、zstd（BSD）、FTXUI 7.0.3（MIT）、orlp/ed25519（zlib），版本和来源见 `third_party/README.md` |
| Python 工具 | `tools/native/golden.py`、`stream_diff.py` | 录 Python 侧的黄金流；比较两条流 |

传输一律用回环 TCP（引擎在所有平台都支持 `tcp:`），客户端是服务端、引擎连上来，消息格式同 `headless.md` §2。
关卡消息（`level.msg` / `.inc`）照旧是引擎写的本地文件，客户端在 AgentSession 读它的同一时刻读（StartLevel 或尚未加载时读全量，有 PHYS 或 StartLevel 时读增量）。
子进程只继承它的三个标准句柄（Windows：`PROC_THREAD_ATTRIBUTE_HANDLE_LIST`，套接字建成不可继承；POSIX：套接字 `FD_CLOEXEC`），
所以同时开多个引擎时，一个引擎不会握着另一个引擎的连接（否则关掉那条连接对方收不到 EOF）。

`headless/src/ballrain_wire.h` 没有复用：它是引擎一侧写 OBS1 / LVL1 段的写入器，客户端要的是读取（照 `protocol.py` 的布局移植），
要写的只有 ACT1（36 字节头 + 帧，没有段结构），用不上它的段封装。

**参考线必须显式给出**：`path_source = "tas"` 而 `path_dir` 没设时直接报 `RouteError`。Python 在这种情况下会用仓库里的
`benchmarks/refpaths`；客户端没有这个默认目录，若悄悄退回检查点折线就是 r6 出过的那种"跑在别的路线上"的错误（`path.py` 文档）。

## 2. 编译

- Windows x64（VS 18 或 2022 的 MSVC，静态运行时）：`native\build_win.cmd [Release|Debug]` → `build\native-Release\`（环境变量
  `BALLRAIN_BUILD_DIR` / `BALLRAIN_CMAKE_ARGS` 换目录、加 `-D` 选项）。
- 预设（`native/CMakePresets.json`）：`cmake --preset windows-x64|linux-x64|macos-arm64|android-arm64 -S native`，输出在 `build/native-<预设名>`。
- Linux / macOS：`cmake --preset linux-x64 -S native && cmake --build build/native-linux-x64`；https 用 libcurl（macOS 系统自带；Linux 装
  `libcurl4-openssl-dev`），找不到时只能连 http 的本机测试网关。
- 版本号只在一处：`native/CMakeLists.txt` 的 `project(... VERSION x.y.z)`，客户端在 hello 里报告它，发布 zip 也用它命名。
- Windows 程序都嵌了 `platform/utf8.manifest`（进程代码页 UTF-8，Windows 10 1903 起），argv、环境变量和窄字符文件接口按 UTF-8 处理。
- Android arm64：`-DCMAKE_TOOLCHAIN_FILE=<ndk>/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=28`。
- 浮点：MSVC `/fp:precise`，GCC / Clang `-ffp-contract=off -fno-fast-math`（不做 FMA 收缩，减少跨 CPU 差异）。
- 测试：`ctest`（selftest）；加 `-DBALLRAIN_GOLDEN_DIR=<录制目录> -DBALLRAIN_MANIFEST=<策略 json> -DBALLRAIN_PATH_DIR=<L01.npz 所在目录>` 后每条录制一个 golden 测试。

## 3. 黄金对比的做法

1. `tools/native/golden.py {policy|random|tas}` 用 Python 侧（`BallanceEnv` / `AgentSession` + 无头引擎）跑一遍，在传输层录下：hello、每条 OBS（含 24 字节 socket 头）、每条发出的 ACT1、
   AgentSession 每次读关卡消息时读到的字节（全量文件 / `.inc` 未读的尾部，没读时记 status 0）、每次 `reset_episode()`、每个特征向量、策略模式下每步的 log-prob / value / u / 动作。
   录制放在 `runtime/` 下（含关卡几何，不进 git；meta.json 里的路径已去掉）。
2. `nc_golden` 把录下的引擎字节原样喂给 C++ 的 tracker / 特征 / 策略，逐行比较特征（|差| ≤ 1e-4 或相对 1e-4）、log-prob（`LOGP_TOL` 1e-3）、value（`VALUE_TOL`），并检查按 u 抽到的动作。
3. `nc_runner` 在线跑同一个种子（同样的 `retail:<seed>` 球序、`BALLRAIN_HL_PHYS2002=0`），写同格式的流；`stream_diff.py` 逐条比较两条流（含最终 TraceHash）。
   `--actions-from <stream>` 让它按另一条流里的动作走（用来在线复现 Python 的随机动作 run：死亡、重生、GameOver）。
   `nc_golden` 对策略流（meta.json `mode` 为 policy）要求按 u 抽到的动作与录制的相同，否则失败；随机 / TAS 流的动作不是抽出来的，不检查。

### 3.1 复现（Windows PC，仓库根目录；`PY` = 仓库的 Python 环境）

```sh
# 自用拷贝（不动训练用的那份）：训练用的引擎、游戏目录、参考线、检查点
mkdir -p runtime/nc/engine runtime/nc/ckpt runtime/refpaths-alt
cp <main checkout>/runtime/hl-win/bin-x86p2002d/BallanceMMOSimTool.exe runtime/nc/engine/
cp -r <main checkout>/runtime/hl-win/game runtime/nc/game
cp <main checkout>/runtime/refpaths-alt/L01.npz runtime/refpaths-alt/
cp <main checkout>/runtime/m2-train-mac/r7/latest.pt runtime/nc/ckpt/r7_latest.pt
# 策略导出（可信的检查点，torch）：runtime/nc/policy/<sha>.json + .npz
$PY -c "from ballrain.pool import weights as W; print(W.export_checkpoint('runtime/nc/ckpt/r7_latest.pt', 'runtime/nc/policy', 'r7')['policy_sha'])"
M=runtime/nc/policy/<sha>.json
$PY tools/native/golden.py policy --manifest $M --out runtime/nc/golden/policy_s1 --seed 1 --ball-seed 12
$PY tools/native/golden.py random --manifest $M --out runtime/nc/golden/random_s2 --seed 2 --max-steps 6000
$PY tools/native/golden.py tas --manifest $M --out runtime/nc/golden/tas_sr01 --record <main checkout>/runtime/datasets/tas-v1/SR01_1_14_457_264
native/build_win.cmd Release
build/native-Release/nc_selftest.exe
build/native-Release/nc_golden.exe --manifest $M --stream runtime/nc/golden/policy_s1/stream.bin --path-dir runtime/refpaths-alt
build/native-Release/nc_runner.exe --manifest $M --engine runtime/nc/engine/BallanceMMOSimTool.exe --root runtime/nc/game \
    --path-dir runtime/refpaths-alt --out runtime/nc/cpp/policy_s1 --seed 1 --ball-seed 12
$PY tools/native/stream_diff.py runtime/nc/golden/policy_s1/stream.bin runtime/nc/cpp/policy_s1/stream.bin
```

## 4. M1 结果（2026-11-17，Windows PC，r7 u82000 的策略导出 `9fec4769…`）

引擎：训练用的 `bin-x86p2002d` 的拷贝，`BALLRAIN_HL_PHYS2002=0`（源码物理，hello 为 `... x87soft engine headless ...`；§13.1 已验证与 x87-soft 后端 TraceHash 相同），游戏目录是 `runtime/hl-win/game` 的拷贝。
特征配置取自策略清单（r7：`camera=world`、`hs_block=false`、`far_waypoints 20,30,45`、`ref_speed_at 0,15,30`、`runtime/refpaths-alt/L01.npz`），208 维。

| 检查 | 结果 |
|---|---|
| 离线特征（`nc_golden`）：策略从出生点（5912 行）、随机粘滞动作 4 次死亡到 GameOver（1092 行）、SR01_1_14_457_264 全关 TAS 按 2 帧一块回放（10295 行） | **17299 行全部逐位相同**（最大差 0）；Windows MSVC、WSL Ubuntu gcc 13、clang 三种构建结果一样 |
| 策略 | log-prob 最大差 1.3e-5、value 最大相对差 9e-7（容差 1e-3）；按 u 抽到的动作 5911/5911 与 Python 相同，没有落在 CDF 边界容差内的情况 |
| 在线 C++ 运行 vs Python 同种子（6 个种子 × 球序种子，策略从出生点到 LevelFinish） | 6/6：ACT1 流逐字节相同（约 5900 条/条）、特征全同、最终 TraceHash 相同 |
| 在线复现 Python 随机动作 run（`--actions-from`，4 次死亡 + 重生 + GameOver） | ACT1 逐字节相同、4 次 reset、特征全同、TraceHash 相同 |
| 速度（单实例，同一引擎，同时有训练在跑） | C++ **680–930 步/秒**（策略推理 0.18 ms/步，其余主要是引擎的 2 帧）；Python（`BallanceEnv` + numpy 策略，同一循环）217–262 步/秒：约 3–4 倍 |
| Android NDK r28 arm64 | core + engine + 程序都能编译（没在设备上跑） |

适用范围：以上全部在同一台 x86-64 机器上（MSVC / gcc / clang 三种构建都是 x86-64），没有在 arm64 上跑过；“逐位相同”只覆盖 r7 的特征配置、第 1 关、木球。
移植了但没有任何黄金数据跑到的分支：纸球（非球面突触）的 MIND 法线定向（`_closest_on`）、非 world 相机系、`hs_block`、检查点折线参考线、
暂停 / Esc（含强制解除暂停）、前缀观测确认、TAS 起点（k > 0，还没移植）。

### 4.1 同一输入两次运行，引擎字节并非完全确定（影响 M2 的哈希链）

Python 同种子跑两次（`policy_s1` / `policy_s1b`），C++ 与 Python 之间也一样：ACT、特征、TraceHash 全同，但
- OBS：**每条 SCOR**、最后一条 SREN 不同——Energy 第 4 列（Timefactor 参数单元）里是指针（`headless.md` §5）；
- `.inc` 增量关卡消息：IVPO 段 1–3 处不同（按 IVP 对象地址去重，`headless.md` §7.3）。

其余段（TIME、BALL、FRIC、MIND、BODY、CAMS、ARRS、EVTS、WTCH、PHYS、HASH，以及全量 level.msg 和其它增量段）逐字节相同。
所以 M2 的哈希链不能直接哈希引擎原始字节：要么把 SCOR / SREN 的 Energy 第 4 列置零、跳过 IVPO（特征不读它们），要么只哈希特征读到的那些段。

后来查到的跨机器 / 跨安装差别（Android 会话在 Windows 引擎和 3 台手机之间逐字段比过；本机用另一个游戏目录复现了后两项）：
- WTCH 的装饰物体位置在 MSVC CRT 和 bionic 之间差 1 ulp（3 台手机之间相同）；
- level.msg 的 GameSettings 第 0 行第 3 列是游戏目录的路径串；
- level.msg 的 GRPS 里 All_Musicfiles 组的成员取决于 Sounds/ 里有哪些文件；
- IVPO 的漏发取决于堆的历史（同一台手机换一份游戏数据也会变）。
链 v3 把这些都去掉了（`pool-prototype.md` §3.2）；其余段（BALL / FRIC / MIND / BODY / GLCT / CAMS / ARRS 数值 / TIME / EVTS）、帧日志、事件和 f6 特征在 Windows 和手机之间逐位相同。

跨引擎构建：同一组动作在另一个构建 `bin-x86soft`（x87-soft 构建，`runtime/hl-win`）上回放，结果与上面完全一样（只有同样的指针字段不同，TraceHash 相同）。
但两个都是 Windows x86 MSVC 构建；**Mac arm64 / Linux 的引擎构建之间这些段是否逐字节相同还没验证**——CK 侧（相机、WTCH 装饰物体）是原生浮点代码，
`headless.md` §7 里零售与无头的差异正出在这些段上。M2 的链设计要先把这一点查清（或者只哈希物理侧的段）。

## 5. M2：池协议客户端，校验不再依赖跨 CPU 逐位一致

### 5.0 任务格式 3（M3，2026-11-18）
池子任务的开局分布改成和学习器一样（`pool-prototype.md` 第 5 节）：任务可以带 `rest`（静止开局：推到目标再刹车，或只刹车）、`life_seed`（每条命的计数随机化和提前重生抽取）、`respawn_frontier`（提前重生规则：任务以 `respawn_skipped` 结束）；网关按检查点的 `rehearse_starts` / `rehearse_behind_frontier` / `rest_starts` 抽开局。原生客户端移植了 `env._drive` / `_brake`（开局类型 `tas_rest`）、`counters.py`（`CounterRand`）和提前重生规则；`task_end` 让没有最终段的任务马上关掉（开局失败不再占一个小时的实例名额）。
- 对照：`tools/native/task_diff.py` 14 个场景全部通过（新增推到目标、交接帧在排除范围里、只刹车、推到 14232 附近的目标、两种提前重生；r7 清单带 `counter_rand`，k > 0 的场景都带每条命的计数值）；`tools/native/e2e_pool.py --env-draws '{"p_rest": 0.8, "p_rehearse": 0.1}'` 经网关发 6 个任务（推到目标 1 个、只刹车 2 个、普通 TAS 开局 3 个，`respawn_frontier` 4100），36 段全部验证通过，伪造段照样查出。
- 第 2 关（r8，`SR02_1_28_992_264`，参考线 `benchmarks/refpaths/L02.npz`）：`nc_golden` 对 SR02 全关 TAS 按 2 帧一块回放（12198 行，木球 6357 行、纸球 5789 行，变球 329 行，接触运动物体 253 行，`bodies` 非零 2419 行）、r8 从出生点（6584 行）、随机粘滞动作（758 行）**全部逐位相同**（最大差 0）；`task_diff.py` 对 r8 跑 10 个场景（关卡开头、四分位 TAS 开局、只刹车、推到目标、r8 自己的提前重生规则和带抽取的变体，含 LevelFinish 和 stall）全部通过；`start_dist.py` 对 r8（`rehearse_behind_frontier`，前沿 17093）两边一致。顺带修了两处只在别的关卡出现的问题：`nc_golden` 的关卡号写死成 1；`L02.npz` 没有 `t_ms`，网关导出和客户端安装参考线时都要求它（现在可选）。注意：没有 `t_ms` 时 `ref_speed` 特征恒为 0，r8 的这 3 维一直是 0（Python 和 C++ 一样）。
- 游戏文件清单：x87-soft 引擎只报 `base.cmo`、`Database.tdb`、`3D Entities/`、`Text/` 四项（不再报 `Bin/VxMath.dll`）。网关只看自己清单里的名称，参考表里多出的 `Bin/VxMath.dll` 不影响；多报一项的客户端会被拒（只有这个分支上没发布过的 0.3.0 构建会这样）。

### 5.1 校验方式改了（Python 侧，链 v3）

**适用范围先说清楚**：本机上原生客户端算出的特征和 Python 逐位相同，所以端到端测试里验证器的容差其实**没有被原生客户端用到**；
容差确实能放过跨 CPU 级别的舍入、又抓得住伪造，证据只有 Python 测试（`test_float_noise_of_another_cpu_is_tolerated` 和伪造测试）。

以前数据段的哈希链覆盖每行的 float32 特征，验证器重放时要逐位复现——这等于要求志愿者的 CPU / 编译器算出的特征和服务端逐位相同。现在（`pool-prototype.md` §3.2、§10.6）：

- **链覆盖引擎原始字节**：每行的引擎摘要 = 上一行之后收到的全部 OBS1 消息和关卡消息读取（规范化：去掉随机器、进程或安装变化的 SCOR、SREN、WTCH、IVPO、GRPS 和字符串单元的内容），
  `h(i) = sha256(h(i-1) || d_i)`（`ballrain/pool/engine_chain.py`，`native/core/src/engine_chain.cpp`）。引擎是确定的，验证器的重放必须逐位复现这条链。
- **验证器自己算特征**，float16 观测在一个 float16 ulp + 1e-4 内即可；log-prob / 价值用验证器自己的特征和原有的 `LOGP_TOL` / `VALUE_TOL` 比；
  动作必须是服务端种子的抽取，或者 `u` 落在累积分布边界的 `ACTION_EPS` 内；奖励、env_rewards 在 1e-4 × max(1, |r|) 内，弧长在 1e-3 × max(1, |arc|) 内；flags、end、sector、帧数、IGT 逐位。
- 数据段格式号 `CHUNK_FORMAT` 改成 3（2 是链 v2：它还包含 WTCH、GRPS 和路径单元，游戏目录路径不同的诚实玩家都会对不上），客户端版本号 0.3.0，网关默认 `min_client_version = "0.3.0"`：
  旧的 0.1.0 / 0.2.0 主控程序在 hello 时就收到 `client_too_old`，而不是之后每段都 `bad_chunk`。正在用的网关如果在 `config.toml` 里自己写了 `min_client_version`，要改成 0.3.0；
  已经存在网关目录里的格式 1 数据段，验证器解不开，会停在 `replay_failed`（不影响账号）。
- Python 主控程序（`client_sampler`）和验证器（`rollout`）同步改了；测试：`tests/test_pool_engine_chain.py`（规范记录、字节布局）、
  `test_pool_verifier.py` 新增“另一种 CPU 的浮点噪声照常通过”和“伪造引擎状态并重算链”两项。池子相关测试 102 项通过、4 项跳过
  （`test_http_layer_refuses_before_reading_and_bounds_connections` 在一次全量运行里因负载超时失败过一次，单独和再跑两遍都通过；没动过那部分代码）。

### 5.2 客户端（`native/`）

| 部分 | 位置 | 对应的 Python |
|---|---|---|
| 引擎摘要 / 链 | `core/src/engine_chain.cpp` | `pool/engine_chain.py` |
| 数据段编码（JSON 头规范输出、raw / zstd 段、float16 就近取偶、sha256） | `core/src/chunk.cpp`，zstd 1.5.7 单文件（`third_party/zstd`，BSD） | `pool/protocol.py encode_chunk` |
| 开局包（校验同 `install_starts_pack`；相机过滤、exclude、`expected`、前缀） | `core/src/starts.cpp` | `pool/task.py`、`curriculum.TasStarts` |
| SR 奖励 | `core/src/reward.cpp` | `rewards.py` |
| env 补齐：TAS 起点（前缀回放、交接检查与重试、按录制恢复按住的键和相机计时）、奖励、相机守卫罚分、引擎摘要 | `core/src/env.cpp`、`session.cpp` | `env.py` |
| 任务运行：worker 的训练奖励（分段奖励、门、帧上限、stall / stall_terminal / stall_compensate）、任务结束判定、数据段组装、按种子抽样 | `core/src/task_runner.cpp` | `pool/rollout.py`（验证器重放用的那份） |
| 网关客户端：HTTP（Windows 用 WinHTTP，http / https；POSIX 用 libcurl，见 §2、§11）、消息、sha 寻址缓存、game_files 哈希、引擎信息 | `core/src/pool_client.cpp`、`platform/` | `pool/client_api.py`、`task.py`、`client_game.py` |
| 命令行客户端：`nc_client login`（login_start / login_poll，存令牌）、`nc_client run`（hello → 领任务 → 下载并校验 → 跑任务 → 每段立即上传，网络错误退避重试；拒收原样打印，`TOKEN_FATAL` 停止） | `apps/nc_client.cpp` | `client_worker.py`、`client_core.py` 的一部分 |

### 5.3 端到端（`tools/native/e2e_pool.py`）

本机回环：模拟论坛 + 网关（真实 r7 策略导出、SR01 开局包、`runtime/refpaths-alt` 参考线）+ Python 验证器（x87-soft 引擎 `bin-x86soft`），
原生客户端走完整的登录流程（浏览器那一步由脚本带着 cookie 模拟，模拟论坛用户 `eligible`）、跑任务、上传；验证器在试用期内重放每一段。

| 组合 | 任务 | 数据段 | 验证 |
|---|---|---|---|
| 客户端 `bin-x86soft`，验证器 `bin-x86soft` | 2 个（LevelFinish 2738 步；max_steps 3000 步），开局 k 来自 r7 的课程（前沿 4100 附近） | 23 段全部接收 | **23/23 通过**，5738 步通过验证 |
| 客户端 `bin-x86p2002d` + `BALLRAIN_HL_PHYS2002=0`，验证器 `bin-x86soft`（两个不同的 x87-soft 构建） | 4 个，都 LevelFinish（1645 / 3877 / 2729 / 1495 步） | 40 段全部接收 | **40/40 通过**，9746 步通过验证 |
| 伪造：把一段原生数据段第 10 步的奖励加 0.25 并重新封装，交给验证器的 `replay_task` | — | — | 报 `replay_reward_mismatch`，第 10 步 |
| 链 v3 之后：客户端用 `bin-x86p2002d` + 另一个游戏目录（另一路径、只有 base.cmo / Database.tdb / 3D Entities / Text / Bin\VxMath.dll，没有 Sounds），验证器用 `bin-x86soft` + 完整目录 | 3 个（LevelFinish ×2、max_steps） | 30 段 | **30/30 通过**（链 v2 下同样的组合每一行都对不上） |

第三次运行（3 个任务、41 段，网关心跳间隔设成 5 秒）：客户端发了 2 次心跳都被接受，41/41 通过。客户端在任务里约 300–990 步/秒（含每段上传）；验证器重放约 200–350 步/秒。

### 5.4 任务结束方式的差分测试（`tools/native/task_diff.py`，`native/apps/nc_task`）

诚实的策略很少走到的分支（一旦和验证器不一致就是 `token_revoked_replay`，吊销的是诚实玩家）：原生客户端按强制动作（`idle` / 随机粘滞动作）或策略抽样跑一个任务，
Python 验证器的 `replay_task` 重放它的数据段；强制动作时只跳过“动作是不是种子抽样”这一项，其余（引擎状态链、观测、log-prob、价值、奖励、flags、end、可选数组）照常比。

| 场景 | 开局 k | 动作 | 结束 | 步数 / 回合 | 结果 |
|---|---|---|---|---|---|
| idle_spawn | 0 | idle | stall（stall_terminal + stall_compensate，罚分 -7.51） | 276 / 1 | 通过 |
| sticky_spawn | 0 | 随机粘滞 | GameOver（3 次重生） | 738 / 4 | 通过 |
| sticky_mid | 9900 | 随机粘滞 | GameOver（TAS 起点，重生是 respawn 类型） | 545 / 4 | 通过 |
| sticky_late | 13500 | 随机粘滞 | GameOver | 694 / 4 | 通过 |
| policy_frames | 9900 | 策略 | max_frames | 450 / 1 | 通过 |
| policy_stop | 13500 | 策略 | stopped（玩家停止） | 77 / 1 | 通过 |
| cap_gates | 9900 | 策略 | cap（另导出一个 cap_mult = 0.05、带两个门的策略版本） | 800 / 1 | 通过，门奖励出现 |
| sticky_cap_gates | 0 | 随机粘滞 | GameOver | 624 / 4 | 通过，门奖励出现 |

没有覆盖的：`error`（任务中途引擎崩溃）、`deadline`、`truncated`（引擎自己的截断、相机守卫）。

```sh
$PY tools/native/e2e_pool.py --dir runtime/nc/e2e --ckpt runtime/nc/ckpt/r7_latest.pt     --record <main checkout>/runtime/datasets/tas-v1/SR01_1_14_457_264     --engine runtime/nc/engine-x86soft/BallanceMMOSimTool.exe --game runtime/nc/game     --client build/native-Release/nc_client.exe --tasks 2 --max-steps 3000
```

另外两项交叉检查：`tools/native/stream_digests.py`（Python）与 `nc_golden --digests`（C++）对同一录制算出的引擎摘要逐行相同（17299 行）；
Python 两次运行、C++ 在线运行、`bin-x86soft` 构建的摘要也都相同。

## 7. M3：控制器和终端界面

### 7.1 控制器（`native/core/src/controller.cpp`）
`ballrain/pool/client_core.py` 的移植，和界面无关（以后的 Android 应用直接用它）：
- **数据目录** `%USERPROFILE%\.ballrain`（macOS / Linux `~/.ballrain`）：客户端写的、不放在程序目录里的东西全在这里（程序目录可以是只读的）。
  `config.json`、`token`（POSIX 0600）、`user.json`（论坛账号，诊断导出时用来替换）、`cache/`（按 sha 存的策略、开局包、参考线）、`queue/`（上传队列）、
  `logs/`（`client.log` 超过 2 MB 轮转，保留 3 份；出错实例最近一次的引擎日志 `engine-NN.log`）、`runs/`、`engine-work/`（见下面的非 ASCII 路径）。同一个数据目录只能开一个客户端（Windows 命名互斥体，POSIX 文件锁）。
- **实例**：最多 8 个线程（默认 min(8, 核数 / 2)，网关的 `max_instances` 优先），每个线程一个引擎子进程，按任务取、跑、切段。
  引擎在 Windows 上低于正常优先级、建成挂起、放进客户端的 Job Object（`KILL_ON_JOB_CLOSE`）再恢复，所以客户端怎么退出（包括崩溃、被结束进程）
  引擎都会跟着结束；POSIX 上 nice 10，引擎在连接断开时自己退出。没有窗口，不会抢焦点。
- **出错重试**：任务出错（引擎起不来、中途死掉）时该实例等 2 秒，之后每次加倍，最多 5 分钟；连续 8 次后停下并给出提示（按“开始”重试）；
  跑满 2 分钟的任务清零计数。引擎在连上之前就退出时立刻判失败（以前要等满 120 秒的 hello 超时）。
- **上传队列**：数据段先原子地写到 `queue/<task_id>/`，后台线程按顺序上传；断网时数据段留在磁盘上，网络恢复或客户端重启后接着传，直到任务期限。
  队列超过 256 MB 时实例暂停取任务。被拒收的任务立即停止并丢弃；`chunk_duplicate`（上一次其实已经到了）当成功处理。
- **`task_end`**（格式 3 新增）：没有最终段就结束的任务（开局失败，或刚切完一段引擎就死了）在数据段都传完后关掉，网关马上空出实例名额，
  不再占到任务期限；客户端被强行结束后，下次启动时上次没跑完的任务也会这样关掉（段数取网关的计数）。
- **自动暂停**：每 5 秒查一次 Player.exe 是否在运行、是否在用电池（可在配置里关）；暂停时正在跑的任务以 `stopped` 结束。
  共用的机器可以设暂停开关：`config.json` 的 `quiet_file`（`nc_client run --quiet-file PATH`），这个文件或目录存在时所有实例暂停，
  删掉后自动继续（实测：出现后 1 秒内暂停，`e2e_pool.py --quiet 5:15` 63 段通过）。
- **登录**：`login_start` 后在系统浏览器打开网关的确认页（论坛的 client secret 只在网关），每 `poll_s` 秒 `login_poll`；成功后令牌写进数据目录。
- **横幅**：网关拒收、吊销、刷新会话失败时，横幅原样显示网关的 `reason.message`（FTXUI 的 `text` / `paragraph` 不解析任何标记）。
  令牌失效类原因会停掉所有实例，直到重新登录。
- **非 ASCII 路径**：无头引擎用代码页路径找文件，Ballance 目录或引擎路径里有中文时（中文用户名、桌面上的游戏）它会找不到 base.cmo。
  这时客户端先试 8.3 短名，没有短名就把引擎目录、以及 x87-soft 版要读的四项游戏文件复制到数据目录下的 `engine-work/`（按引擎 sha256 /
  游戏文件哈希命名，只复制一次），运行目录也放在那里。用户名是中文时数据目录本身也不是纯 ASCII，这时用 `engine-work` 的 8.3 短名（系统盘通常有）；
  连短名都没有时给出中文提示，玩家可以设环境变量 `BALLRAIN_WORK_DIR` 指向一个纯英文的文件夹。
  实测：数据目录、引擎、游戏目录都放在中文路径下，端到端 12/12 段通过。

### 7.2 终端界面（`native/apps/ballrain_pool.cpp`，FTXUI）
只画 `ControllerView`、调用控制器的动作：
- **首次设置**：网关地址（发布版预填）、Ballance 目录（“查找常见位置”）、开发版没有附带引擎时还有引擎路径；“检查游戏文件”按 x87-soft 版的清单逐项给出中文提示。
- **登录页**：“在浏览器中登录”，显示登录代码和地址（浏览器没打开时可以手动复制）。
- **主界面**：开始 / 暂停、实例数 −/＋、上榜开关（网关的状态为准，默认开）、每个实例一行（状态、任务、本任务步数、累计步数、已接收步数、步/秒）、
  待上传段数、网关统计（已接收 / 已验证 / 待验证）、横幅、自动暂停提示、网关通知、新版本提示、最近的日志。
- **导出诊断**：先列出要打包的文件和 `info.json` 的内容，确认后保存到数据目录（§8.2）。
- **启动参数**（`BallrainPool --help`）：`--page setup|login|main|diag` 直接打开某一页；`--start` 启动后立即连接并开始采样（没登录时先进登录页）；
  `--login` 立即开始浏览器登录；`--gateway`、`--ballance-dir`、`--engine`、`--instances`、`--quiet-file`、`--auto-pause on|off` 写进配置并保存
  （检查通过时视为已完成首次设置）；`--data-dir`、`--version`。不认识的选项报错并打印用法。
- **退出**：“退出”按钮、Ctrl-C、Ctrl-Break、关窗口都走同一条正常收尾：任务以 `stopped` 结束、引擎正常停止、剩下的数据段最多再传 5 秒。
  FTXUI 收到 Ctrl-C 时会在退出循环后恢复它找到的信号处理并再次触发信号，所以客户端在 FTXUI 之前装好自己的 SIGINT / SIGTERM 处理（只设退出标志），
  Windows 上另有 `SetConsoleCtrlHandler`（关窗口 / 注销 / 关机时在 Windows 给的约 5 秒里等收尾）。进程被强杀时引擎由 Job Object 结束。
  退出后客户端自己再发一遍关闭鼠标跟踪的序列、丢掉未读的输入、恢复启动时的控制台模式（否则退出后在窗口里点鼠标会把鼠标报告当成文字输进 shell）。
  实测（演示网关上 `--start` 采样中）：Ctrl-C 后 1 秒内界面和两个引擎都退出，两个任务以 `stopped` 关闭、上传队列清空；`taskkill /F` 结束界面进程后
  1 秒内两个引擎都被结束，留下的任务在下次启动时补传并关闭（`client restarted`）。
- 测试用：`--snapshot setup|login|main|diag` 把一页渲染成文本；`--snapshot live:<秒>` 先真实地连接、开始采样，过若干秒再渲染主界面并正常退出。
  `e2e_pool.py --tui` 用它在同一个网关上检查界面（2 个实例 25 秒，完成 14 个任务，主界面显示“采样中”、玩家名和每个实例的计数）。

### 7.3 端到端结果（2026-11-18，本机，r7 策略，x87-soft 引擎）
| 运行 | 结果 |
|---|---|
| 3 个实例 × 8 个任务（`--instances 3`），两遍 | 42 / 41 段全部通过 |
| 格式 3：`--env-draws '{"p_rest": 0.8, "p_rehearse": 0.1}'`，6 个任务（推到目标、只刹车、普通开局） | 36 段全部通过，伪造段照样查出 |
| 跑 12 秒后强行结束客户端（`--kill-after 12`），再重启 | 3 秒后没有残留的引擎；被杀那次的 5 个任务传完并关闭，31 段通过 |
| 引擎环境变量造成开局失败（`BALLRAIN_PREFIX` 指向不存在的文件） | 两个任务都由 `task_end` 关闭（`closed`，原因写进任务） |
| 数据目录、引擎、游戏目录都在中文路径下 | 12 段全部通过 |
| 第 2 关（r8）`task_diff.py` 10 个场景、黄金流 19540 行 | 全部通过 / 逐位相同（§5.0） |
| 实例数等于网关的 `max_instances`（`--instances 3 --max-instances 3`，9 个任务） | 没有 `too_many_instances`（实例先等自己上一个任务传完再取新任务）；33 段通过 |
| 断网：第 6 秒起网关对所有请求回 502 HTML（隧道断开时的样子）20 秒（`--outage 6:20`） | 上传失败 4 次后自动重试，恢复后 55 段全部通过 |
| 任务中途杀掉一个引擎（`--kill-engine-after 6`） | 该任务以 `error` 结束（前缀照常验证），实例退避后继续；33 段通过 |
| 自动暂停：第 5 秒起一个假的 Player.exe 运行 10 秒（`--fake-player 5:10`） | 自动暂停（任务以 `stopped` 结束）并自动恢复；59 段通过 |
| 打包出的 zip：用 zip 里的引擎（含 `ENGINE_VERSION`、应用本地的 VC++ 运行库）跑 4 个任务，再用 zip 里的 `BallrainPool.exe` 实时运行 25 秒 | 16 段通过；界面主页显示“采样中” |
| 诊断导出（`nc_client diag`，离线） | zip 里没有令牌、登录名、家目录、论坛昵称 / 用户名（昵称来自连接时存下的 `user.json`） |

## 8. 打包、发布和更新

### 8.1 Windows 测试版 zip（`tools/native/package_win.py`）
- 用发布配置编译（网关地址、更新地址、发布公钥作为 CMake 缓存变量 `BALLRAIN_DEFAULT_GATEWAY` / `BALLRAIN_UPDATE_URL` / `BALLRAIN_RELEASE_PUBKEY`
  传进去，不进仓库），跑一遍 selftest，组装：`BallrainPool.exe`、`engine/x87soft/windows/`（bin-x86soft 的 `BallanceMMOSimTool.exe`、x86 的
  `msvcp140.dll` 和 `vcruntime140.dll` 放在引擎旁边、`ENGINE_VERSION`）、`使用说明.txt`、`LICENSES/`（收录的库加上 `--license` 给的引擎源码许可证）。
  不含任何游戏文件。zip 里的文件顺序和时间固定，同样的输入得到同样的 zip。脚本会打印引擎的 sha256：网关的 `[engines.x87soft]` 白名单要加上它。
- 实测（占位网关、测试密钥）：zip 6.8 MB；解压后的 `BallrainPool.exe` 认出了随附的引擎，网关地址已预填。
- 没有代码签名：第一次运行时 SmartScreen 会拦（“更多信息” → “仍要运行”），测试者指南里写了。

### 8.2 诊断导出
`info.json`（客户端版本、平台、引擎 sha256 / 版本、游戏文件哈希、实例状态、横幅、统计）、`config.json`、`logs/` 打成 zip。所有内容先替换：
令牌、数据目录、家目录、登录名、论坛用户名 / 昵称 / id、IPv4 地址（selftest 里有例子）。界面在保存前列出文件和 `info.json`。

### 8.3 签名的更新检查（`update.cpp`、`apps/nc_release.cpp`）
- 发布清单 `release.json`（`product`、`version`、`notes`、`assets.<平台>.{url, sha256, size}`）和 `release.json.sig`（ed25519 对清单原始字节的签名，十六进制）
  跟 zip 一起放在 GitHub Release；客户端启动时取 `BALLRAIN_UPDATE_URL` 和 `.sig`，用编译进去的公钥验签，版本更新时在主界面显示下载地址（暂不自动替换）。
  签名不对、格式不对、下载地址不是 https 的清单一律忽略。
- 密钥：`nc_release keygen --out-dir <目录>` 写 `release-signing.key`（32 字节种子，十六进制）和 `release-signing.pub`。**私钥放在仓库外**，建议
  `%USERPROFILE%\.ballrain-release\`，不要提交、不要打印；公钥交给发布构建。发布时：
  `package_win.py ... --pubkey-file <pub> --sign-key <key> --download-url <zip 的 https 地址>` 生成清单并签名；`nc_release verify` 可以复核。
- 还没有生成正式的发布密钥（这一步由维护者在自己的机器上做）。

## 9. CI 和发布前检查

- **GitHub Actions**（`.github/workflows/native-client.yml`）：Windows 上编译、selftest、`tools/native/ci_goldens.py`（fakegame 的合成关卡：Python 经无头 socket 传输录制，
  `nc_golden` 回放，特征逐位相同、log-prob / 价值在容差内、按 u 抽的动作一致）、界面能渲染，产物上传为 artifact；Linux（gcc + libcurl）和 macOS arm64 只检查编译和 selftest。
  本机验证过 Windows 的各步和 Linux 预设的编译 / selftest；工作流本身还没在 GitHub 上跑过（我不推送）。
- **发布前（本机，需要游戏文件）**：真实关卡的黄金流（`golden.py tas/policy/random` + `nc_golden`，第 1、2 关）、`task_diff.py`（第 1 关 14 个场景、第 2 关 10 个）、
  `e2e_pool.py`（`--instances 3`、`--env-draws`、`--kill-after`、`--tui`）、`start_dist.py`（网关和学习器的开局分布）。这些数据含关卡几何，不进 CI。

## 10. 服务端要做的事
- **HTTPS**：网关挂在已有 dashboard 域名的一个路径下（例如 `/pool`），走已有的 cloudflared 隧道，配置 `trust_forwarded = true`；`public_base_url` 是这个 https 地址，
  也就是发布构建的 `BALLRAIN_DEFAULT_GATEWAY`。客户端 WinHTTP 用系统 TLS，跟随重定向。
- **引擎白名单**：`[engines.x87soft]` 加发布 zip 里引擎的 sha256（`package_win.py` 会打印）；验证器用 x87-soft 引擎重放。
- **游戏文件**：x87-soft 版的参考表只需要 `base.cmo`、`Database.tdb`、`3D Entities/`、`Text/`；表里多出的 `Bin/VxMath.dll` 网关会忽略。
- **格式 3**：`min_client_version = "0.3.0"`（默认值已改）；任务字段 `rest` / `life_seed` / `respawn_frontier` 由网关按检查点自动填，学习器那边不用改；
  新的结束原因 `respawn_skipped`；新接口 `POST /pool/task`（`task_end`）。
- **第 2 关**：r8 的 `L02.npz` 没有 `t_ms`，网关导出参考线现在接受它；注意这时 `ref_speed` 特征恒为 0（学习器一侧的事实，不是客户端的问题）。

## 10a. 第 2 关小节 4 的石球分线（`docs/l2-stone-branch.md`，`ballrain/branch.py` 的移植）

- **`core/branch`**：`BranchFollower` 维护两条线（主线 `L02.npz`、石线 `L02-stone/L02.npz`）的投影，按 `L02.json` 的站点把石线米数换成主线米数；
  只有在可见球是石球、在小节 4、并且投影落在石线区间内时用石线。进度 `path.s` 是本命单调的主线进度，换线那一拍不付进度，之后线性追回到下一个共同站点；
  参考时钟只积分新走过区间的活动线 `t_ms`（主线没有 `t_ms` 时用描述文件里的 `main_span_t_ms` 拼出来）。`load_branch` 核对关卡和主线 sha256。
- **特征**：`FeatureConfig.branch_dir`；路径块（路标、切线、剩余距离、参考速度、小节除数）用活动线，`path_progress` 用主线总长归一化；没开时布局和数值都不变。
- **环境**：每步信息带 `branch` / `local_s` / `ref_elapsed_ms` / `ref_remaining_ms`（看不到球时用跟随器上一次的时钟）；石球静止开局（`RestStart.branch_s`）
  回放 `drive.npz` 已验证的按键流到对应站点，核对石球和小节，再闭环开到目标、制动，检查支撑、位置和速度。
  任何一步失败都不算有效交接：关掉这个进程，在同一帧重新做一次普通 TAS 开局（`kind` 为 `tas`，带 `branch_rest_failures` / `branch_rest_target` / `branch_rest_error`，同 `env._prepare`）。
  只有分线自己的检查失败才回退（资源、范围、球掉了、球型 / 小节不对、没有停稳在石线上）；引擎、连接故障照旧报错（任务失败，不上传），免得上传一个校验端会判成不一致的开局。
- **奖励整形**：分段奖励用活动线参考时钟，帧数上限用 `(ref_elapsed + ref_remaining) × cap_mult`，换线时重算（Python 的 `rollout.TaskRunner`、`client_sampler` 也补上了这一步）。
  停滞判定同 `worker.stall_age`：在石线上里程用活动线米数 `local_s`（换线就重新起算），速度用 `speed + speed_change`，静止阈值取 `min(1, stall_speed)`；
  主线上不变。`rollout.TaskRunner` 和 `client_sampler` 改成直接调用 `worker.stall_age`。
- **对照**（主线 `benchmarks/refpaths-v2/L02.npz`，分线 `benchmarks/refpaths-alt/L02-stone`，r8 v2 清单）：`golden.py branch`（SR02 录像到 `drive.npz` 的前缀帧，再接上它的按键流，球经变球器变石球、走南环）：22402 帧、11202 行特征 C++ 与 Python
  **逐位相同**；同一条流不带分线回放时最大差 10，3204 行读到了石线的 `t_ms`，说明确实走在石线上。`task_diff.py --branch-dir`（r8 检查点，分线目录暂时在清单之外给两边）6 个场景全部通过：石球静止开局 + 策略、石球静止开局 + 随机粘滞动作、变球器上游的 TAS 开局 + 策略、小节 3–4 的 TAS 开局 + 随机粘滞动作，以及两个回退场景：目标超出按键流范围、一开始就失败的 `branch_rest_fallback`，`rest_speed` 设成 0、回放和刹车跑完才失败的 `branch_rest_late_fallback`。
  `tools/native/branch_diff.py`（`nc_branch_diff`）把跟随器单独拿出来对照：`tests/test_stone_branch.py` 的几个场景加 40 组随机游走（横向噪声、换球、换小节、倒退、跳跃、重置），每一步的输出逐位相同。
- **分发**（分线是学习器下发的配置，不写死在任何程序里，也没有哪个组件去读自己本地的分线文件）：学习器把三份文件的原始字节放在状态里
  （`state["branch_assets"] = {level, line_npz, desc_json, drive_npz}`）；`weights.export_state` 检查后把它们的 sha256 钉进清单
  `features.branch = {line_sha256, desc_sha256, drive_sha256}`（训练配置里有 `branch_dir` 却没带文件就拒绝导出）；网关 `publish_state`
  打成分线包（`task.export_branch_pack`，三个 uint8 数组，内容相同 sha 就相同），任务带 `assets.branch`；客户端和校验端
  `install_branch_pack` 装到缓存 `branch/<sha>/`，`build_env` / `play_task` 再核对一次三份文件。网关按 `rest_starts` 的
  `lo-hi@branch:s` 条目抽石球静止开局（和环境同样的随机数调用顺序），发 `rest = {"target": null, "branch_s": s}`。
  端到端（`e2e_pool.py --branch-dir`，r8 检查点，`rest_starts` 两个 `@branch:` 条目、`p_rest` 0.8）：网关导出带 `features.branch` 的清单和分线包，
  发出 4 个任务（3 个石球静止开局），原生客户端下载安装分线包跑完，校验端 23 个块全部通过，伪造的块被抓到。
- **旧客户端**：0.3.0 碰到清单里的 `features.branch` 会报“unknown feature config field”而放弃这个任务，不会不带分线悄悄跑出校验不过的数据；
  网关那边怎么避免把这类任务发给旧客户端（版本下限还是能力声明）还在和池会话商定。

## 11. 还没有做 / 限制

- **https 没有对真网关实测过**：Windows 的 WinHTTP https 分支、POSIX 的 libcurl 分支都只编译过（端到端都是本机 http）。Android 仍然只有 http。
- **跨引擎构建的字节一致**只在两个 Windows x86 构建之间验证过；Mac arm64 / Linux 引擎构建之间要先核对（§4.1）。另外容差只能吸收舍入，特征里的离散判定在另一种 CPU 上翻转会被当成不一致（`pool-prototype.md` §8）。
- **更新只提示不替换**；没有代码签名；macOS / Linux 还没有打包脚本和引擎构建（Linux 引擎构建脚本不存在）。
- **引擎依赖的其他组件许可证**（Ballanced、IVP、CKBuildingBlocks 等）要在公开发布前逐个确认并用 `--license` 放进 zip；现在只收了 BallanceMMO 的 BSD-3。
- 令牌文件：POSIX 上写成 0600；Windows 上靠用户目录的权限（同 Python 主控程序）。
- 网关响应只做了轻量的结构检查（类型和必需字段），没有照搬 `protocol.MESSAGES` 的完整 schema。
- 只支持 SR 策略（与网关导出的限制相同）；`route_probes` 只是记账，没有移植。
- `fastreach`、2002 零售版 Player.exe 版本：以后。
