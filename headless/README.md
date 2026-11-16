# headless：x87-soft 无头引擎上的 agent 模式（近似后端）

BallanceMMO 的无头引擎（SimTool：开源 CK2 / VxMath + 真 BuildingBlocks + x87-soft 物理）加上 `src/ballrain_agent.cpp`，
用与 BallrainIO agent 模式相同的线格式（OBS1 / ACT1 / LVL1）锁步运行。说明、可用字段、门禁结果、吞吐见 `docs/headless.md`。

**近似环境**：只用于大批量训练。评估和导出 .tas 仍以零售版 Player.exe 为准；无头 run 的元数据带 `backend = "headless"`，每条观测带 `OBS_HEADLESS` 标志。

## 源码（固定提交，只读获取）

| 仓库 | 提交 |
|---|---|
| BallanceMMO（分支 x87-soft） | `b640cbc5838a43c6949a66ffc67c15cef8e2ece7` |
| Ballanced | `aae1c9166651e60dc0943d189e8ba71c4c55810c` |
| CKBuildingBlocks（含 physics_RT） | `5d8d6e2672678039a2a21fe7357053f896779403` |
| IVP（source-physics） | `cc5499c4347d627a27f10c70fa50a741d6f442c5` |
| 其余子模块 | 由 BallanceMMO 提交钉住，`mk_src.sh` 写进 `commits.txt` |

`mk_src.sh OUT`（环境变量 `BMMO_REPO` 指向含这些提交的 BallanceMMO 克隆）用 `git archive`（`core.autocrlf=false`，LF）打包成 `OUT/src.tar`，
不改动源仓库。补丁就是针对这些 LF 文件写的。

## 补丁集（按顺序叠加）

| 文件 | 层 | 内容 |
|---|---|---|
| `patches/0001-hx1c-vs-b640cbc5.diff` | 基础 | x87-soft 会话的权威实验补丁集 hx1c，**原样复制**（sha256 `96c0cc0148cd5573fa50f9bfa2eeeb0025fb842277532e0e769cb1396f895b96`；来源 `tas-bisect/hx1c/`，其 README sha256 `e1954aeecc04a4ab…`、base-commits.txt `baf8480e8f5f1603…`）。内容：SimTool `--tas` 回放、legacy 改动 / 复位状态 / `HX1_KEEP_PRECISION` 的环境开关、`headless_engine` 的逐帧 delta 与 Random 开关、游戏侧 x87 数学（VxMath、CK2_3D、TT Set Dynamic Position）、`CKIpionManager::SetDeltaTime` 的 x87 平滑 delta、球-球接触对按 CK id 排序。**实验代码，尚未进入产品**；他们的产品提交落地后改用那些提交 |
| `patches/0002-ballrain-agent-entry.diff` | 基础 | `sim_tool.cpp` 的 `--ballrain-agent` 入口；CMake 把 `sim/ballrain_agent.cpp` 加进 BallanceMMOSim（叠在 0001 之上） |
| `src/ballrain_agent.cpp`、`src/ballrain_wire.h` | 基础 | agent 模式本体，复制到 `BallanceMMOServer/sim/`；legacy 改动、复位状态、`HX1_KEEP_PRECISION`、CK id 排序键都由它自己设好 |

我们自己的部分（0002 + `src/`）只加东西，跟着 hx1c 换基即可。诊断用的补丁（相机注入、力 / 冲量日志等）不在补丁集里。

## 编译（macOS arm64，Apple clang）

```sh
SDL3_DIR=<SDL3 源码树> CMAKE=<cmake> NINJA=<ninja> headless/build_mac.sh OUT/src.tar WORK
# -> WORK/build/BallanceMMOServer/BallanceMMOSimTool
```

SDL3 在 macOS 上需要两处本地修改（见 `build_mac.sh` 头部）。从 `src.tar` 解包、打补丁到编译完成，M6 上约 220 s（6 个并行任务）。
正在被训练使用的二进制不要覆盖：每个构建放进新目录（例如 `~/ballrain/hl/hx1c-<diff 哈希>-<agent 哈希>`），训练重启时切换。

## 运行

- Python：`EnvConfig(backend="headless", headless={"binary": ..., "root": <游戏目录>})`，或设 `BALLRAIN_HL_BIN` / `BALLRAIN_HL_ROOT`。
- 不依赖 Python 包的回放驱动：`python3 headless/tools/replay_stub.py --bin ... --root ... --level N --frames REC.raw --out DIR`，
  输出可交给 `scripts/headless_gate.py` 和零售数据集比较。
- 引擎直接调用：`BallanceMMOSimTool --ballrain-agent --root <游戏目录>`，配置全部来自 `BALLRAIN_*` 环境变量（`docs/headless.md`）。
