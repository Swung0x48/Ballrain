# 原生志愿者客户端（native/，C++17）

志愿者算力池的玩家端程序：不需要 Python，直接驱动无头引擎（x87-soft 物理），自己算观测特征、跑策略、按服务器种子抽动作。
Python 侧（`ballrain/pool`、`tools/pool`）是训练 / 网关 / 校验方。设计背景见 `volunteer-compute.md`、`pool-prototype.md`。

里程碑：**M1** 核心库 + 回合运行器（§1–4）；**M2** 池协议客户端 + 校验改为跨 CPU 容差（§5）；M3 控制器界面、子进程生命周期、OAuth、单文件发布。

## 1. 结构

| 部分 | 位置 | 说明 |
|---|---|---|
| `ballrain_core` | `native/core/` | 纯可移植 C++17：线格式（`wire`，对应 `ballrain/protocol.py`）、游戏状态（`state`，对应 `state.py`）、几何（`geometry.py`）、参考线与投影（`path.py`）、特征（`features.py`，f6）、策略与抽样（`pool/weights.py` 的 NumpyPolicy、`pool/sampling.py`）、JSON（含 Python `json.dumps` 规范输出）、sha256、npz 读取（miniz）、录制流格式 |
| `ballrain_engine` | `native/core/src/session.cpp`、`env.cpp` | 一个引擎进程的会话（`client.py` AgentSession + `headless.py` 启动方式 + `transport.py` SocketChannel 的 Python 一侧）和环境（`env.py` BallanceEnv 中池任务用到的部分：L1 开场 Q、世界方向动作、按住键在相机换向后重按、相机守卫、按命重生） |
| 平台层 | `native/platform/` | 唯一的系统相关代码：本地 TCP、子进程、单调时钟、目录、环境变量。`platform_win.cpp` / `platform_posix.cpp`（macOS / Linux / Android） |
| 程序 | `native/apps/` | `nc_runner`（在线跑回合）、`nc_golden`（离线黄金对比）；`native/tests/selftest.cpp`（不需要引擎的单元检查） |
| 第三方 | `native/third_party/miniz` | miniz 3.1.0（MIT），读 npz（zip64、deflate） |
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

- Windows x64（VS 18 或 2022 的 MSVC，静态运行时）：`native\build_win.cmd [Release|Debug]` → `build\native-Release\`。
- Linux / macOS：`cmake -S native -B build/native -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build/native`。
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

跨引擎构建：同一组动作在另一个构建 `bin-x86soft`（x87-soft 构建，`runtime/hl-win`）上回放，结果与上面完全一样（只有同样的指针字段不同，TraceHash 相同）。
但两个都是 Windows x86 MSVC 构建；**Mac arm64 / Linux 的引擎构建之间这些段是否逐字节相同还没验证**——CK 侧（相机、WTCH 装饰物体）是原生浮点代码，
`headless.md` §7 里零售与无头的差异正出在这些段上。M2 的链设计要先把这一点查清（或者只哈希物理侧的段）。

## 5. M2：池协议客户端，校验不再依赖跨 CPU 逐位一致

### 5.1 校验方式改了（Python 侧，链 v2）

**适用范围先说清楚**：本机上原生客户端算出的特征和 Python 逐位相同，所以端到端测试里验证器的容差其实**没有被原生客户端用到**；
容差确实能放过跨 CPU 级别的舍入、又抓得住伪造，证据只有 Python 测试（`test_float_noise_of_another_cpu_is_tolerated` 和伪造测试）。

以前数据段的哈希链覆盖每行的 float32 特征，验证器重放时要逐位复现——这等于要求志愿者的 CPU / 编译器算出的特征和服务端逐位相同。现在（`pool-prototype.md` §3.2、§10.6）：

- **链覆盖引擎原始字节**：每行的引擎摘要 = 上一行之后收到的全部 OBS1 消息和关卡消息读取（规范化：去掉带地址的 SCOR、SREN、IVPO），
  `h(i) = sha256(h(i-1) || d_i)`（`ballrain/pool/engine_chain.py`，`native/core/src/engine_chain.cpp`）。引擎是确定的，验证器的重放必须逐位复现这条链。
- **验证器自己算特征**，float16 观测在一个 float16 ulp + 1e-4 内即可；log-prob / 价值用验证器自己的特征和原有的 `LOGP_TOL` / `VALUE_TOL` 比；
  动作必须是服务端种子的抽取，或者 `u` 落在累积分布边界的 `ACTION_EPS` 内；奖励、env_rewards 在 1e-4 × max(1, |r|) 内，弧长在 1e-3 × max(1, |arc|) 内；flags、end、sector、帧数、IGT 逐位。
- 数据段格式号 `CHUNK_FORMAT` 改成 2（旧的段在新链上本来就对不上），客户端版本号改成 0.2.0，网关默认 `min_client_version = "0.2.0"`：
  旧的 0.1.0 主控程序在 hello 时就收到 `client_too_old`，而不是之后每段都 `bad_chunk`。正在用的网关如果在 `config.toml` 里自己写了 `min_client_version`，要改成 0.2.0；
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
| 网关客户端：HTTP（Windows 用 WinHTTP，http / https；POSIX 目前只有 http，见 §6）、消息、sha 寻址缓存、game_files 哈希、引擎信息 | `core/src/pool_client.cpp`、`platform/` | `pool/client_api.py`、`task.py`、`client_game.py` |
| 命令行客户端：`nc_client login`（login_start / login_poll，存令牌）、`nc_client run`（hello → 领任务 → 下载并校验 → 跑任务 → 每段立即上传，网络错误退避重试；拒收原样打印，`TOKEN_FATAL` 停止） | `apps/nc_client.cpp` | `client_worker.py`、`client_core.py` 的一部分 |

### 5.3 端到端（`tools/native/e2e_pool.py`）

本机回环：模拟论坛 + 网关（真实 r7 策略导出、SR01 开局包、`runtime/refpaths-alt` 参考线）+ Python 验证器（x87-soft 引擎 `bin-x86soft`），
原生客户端走完整的登录流程（浏览器那一步由脚本带着 cookie 模拟，模拟论坛用户 `eligible`）、跑任务、上传；验证器在试用期内重放每一段。

| 组合 | 任务 | 数据段 | 验证 |
|---|---|---|---|
| 客户端 `bin-x86soft`，验证器 `bin-x86soft` | 2 个（LevelFinish 2738 步；max_steps 3000 步），开局 k 来自 r7 的课程（前沿 4100 附近） | 23 段全部接收 | **23/23 通过**，5738 步通过验证 |
| 客户端 `bin-x86p2002d` + `BALLRAIN_HL_PHYS2002=0`，验证器 `bin-x86soft`（两个不同的 x87-soft 构建） | 4 个，都 LevelFinish（1645 / 3877 / 2729 / 1495 步） | 40 段全部接收 | **40/40 通过**，9746 步通过验证 |
| 伪造：把一段原生数据段第 10 步的奖励加 0.25 并重新封装，交给验证器的 `replay_task` | — | — | 报 `replay_reward_mismatch`，第 10 步 |

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

## 6. 还没有做 / 限制

- **https 没有实测过**：Windows 的 WinHTTP https 分支没跑过（端到端都是本机 http）；POSIX 的 http 客户端只编译过，没有运行过。
- **POSIX 上没有 https**：macOS / Linux / Android 的 `http_request` 只支持 http（本机测试网关）。真网关在 https 隧道后面，这几个平台要接系统 TLS（macOS Security / Network 框架、Android 的 Java HTTP，Linux 用 libcurl 或 OpenSSL）。
- **跨引擎构建的字节一致**只在两个 Windows x86 构建之间验证过；Mac arm64 / Linux 引擎构建之间要先核对（§4.1）。另外容差只能吸收舍入，特征里的离散判定在另一种 CPU 上翻转会被当成不一致（`pool-prototype.md` §8）。
- **只有一个实例、没有界面、登录要自己打开浏览器**：M3（TUI、多实例、Job Object / 进程组、系统浏览器 + 本机回调、单文件发布）。
- 令牌文件：POSIX 上写成 0600；Windows 上靠用户目录的权限（同 Python 主控程序）。
- 网关响应只做了轻量的结构检查（类型和必需字段），没有照搬 `protocol.MESSAGES` 的完整 schema。
- 上传是同步的（每段在本步结束时直接 POST）；Python 主控程序是后台线程。本机回环上没有影响，慢网络下会拖慢采样。
- 只支持 SR 策略（与网关导出的限制相同）；`route_probes` 只是记账，没有移植。
- `fastreach`、2002 零售版 Player.exe 版本：以后。
