# 原生志愿者客户端（native/，C++17）

志愿者算力池的玩家端程序：不需要 Python，直接驱动无头引擎（x87-soft 物理），自己算观测特征、跑策略、按服务器种子抽动作。
Python 侧（`ballrain/pool`、`tools/pool`）是训练 / 网关 / 校验方。设计背景见 `volunteer-compute.md`、`pool-prototype.md`。

里程碑：**M1**（本文件现在的内容）核心库 + 回合运行器；M2 池协议客户端 + 校验改为跨 CPU 容差；M3 控制器界面、子进程生命周期、OAuth、单文件发布。

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
3. `nc_runner` 在线跑同一个种子（同样的 `retail:<seed>` 球序、`BALLRAIN_HL_PHYS2002=0`），写同格式的流；`stream_diff.py` 逐条比较两条流。
   `--actions-from <stream>` 让它按另一条流里的动作走（用来在线复现 Python 的随机动作 run：死亡、重生、GameOver）。

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

### 4.1 同一输入两次运行，引擎字节并非完全确定（影响 M2 的哈希链）

Python 同种子跑两次（`policy_s1` / `policy_s1b`），C++ 与 Python 之间也一样：ACT、特征、TraceHash 全同，但
- OBS：**每条 SCOR**、最后一条 SREN 不同——Energy 第 4 列（Timefactor 参数单元）里是指针（`headless.md` §5）；
- `.inc` 增量关卡消息：IVPO 段 1–3 处不同（按 IVP 对象地址去重，`headless.md` §7.3）。

其余段（TIME、BALL、FRIC、MIND、BODY、CAMS、ARRS、EVTS、WTCH、PHYS、HASH，以及全量 level.msg 和其它增量段）逐字节相同。
所以 M2 的哈希链不能直接哈希引擎原始字节：要么把 SCOR / SREN 的 Energy 第 4 列置零、跳过 IVPO（特征不读它们），要么只哈希特征读到的那些段。

## 5. 还没有做（M2 / M3）

- TAS 起点任务（k > 0：前缀回放、交接检查、`_init_held` 用录制的相机象限），训练奖励（`rewards.py` + worker 的分段奖励 / 卡住截断），池任务的结束判定（max_steps / max_frames / cap / stall）：M2 与池协议一起做。
- `fastreach`、2002 零售版 Player.exe 版本：以后。
