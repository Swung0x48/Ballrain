# headless：x87-soft 无头引擎上的 agent 模式（近似后端）

BallanceMMO 的无头引擎（SimTool：开源 CK2 / VxMath + 真 BuildingBlocks + x87-soft 物理）加上 `src/ballrain_agent.cpp`，
用与 BallrainIO agent 模式相同的线格式（OBS1 / ACT1 / LVL1）锁步运行。说明、可用字段、门禁结果、吞吐见 `docs/headless.md`。

**近似环境**：只用于大批量训练。评估和导出 .tas 仍以零售版 Player.exe 为准；无头 run 的元数据带 `backend = "headless"`，每条观测带 `OBS_HEADLESS` 标志。

## 源码（固定提交，只读获取）

x87-soft 会话的**产品提交**（已推送）。之前的实验补丁 hx1c 已经全部进入这些提交，不再叠加。

| 仓库 | 分支 | 提交 |
|---|---|---|
| BallanceMMO | x87-soft | `ba5ba279374bfcad2bd5dfe81e129dda507a2881` |
| Ballanced | bmmo-x87-soft | `b70ab01b750a8985655dc1353b6d99bbf676cbe2` |
| VxMath | bmmo-x87-soft | `1b07da5bcfa809db1f5bdad2267cff6026787dd9` |
| RenderEngine（CKRenderEngine 分叉，Ballanced 的 `.gitmodules` 指向它） | bmmo-x87-soft | `cd6f55a8855a80c50dc0f89accdb4a39f3807122` |
| CKBuildingBlocks（含 physics_RT） | bmmo-x87-soft | `2f1cdbdb5901686b51fd9110392b2d68497b1218` |
| IVP（source-physics） | bmmo-x87-soft | `bbd137ac7678815535d424a61da5e13679c9892e` |
| 其余子模块 | | 由 BallanceMMO 提交钉住，`mk_src.sh` 写进 `commits.txt` |

`mk_src.sh OUT`（环境变量 `BMMO_REPO` 指向含这些提交、子模块已检出的 BallanceMMO 克隆）用 `git archive`（`core.autocrlf=false`，LF）打包成 `OUT/src.tar`，
不改动源仓库、不联网；默认提交下它会核对上表五个物理相关子模块的钉住提交，不符就失败。补丁 0002 就是针对这些 LF 文件写的。

来源记录：产品提交与 hx1c（`tas-bisect/hx1c/`，补丁 sha256 `96c0cc0148cd5573fa50f9bfa2eeeb0025fb842277532e0e769cb1396f895b96`，叠在 BallanceMMO
`b640cbc5` / Ballanced `aae1c91` / CKBuildingBlocks `5d8d6e2` / IVP `cc5499c` 上）逐文件比较过：数学代码相同，只差注释、`#if VX_X87_GAME` 包裹、
环境开关（`HX1_*`）换成 SimTool 参数、全局的球序钩子换成每个环境的 `IVP_Environment::bmmo_ball_order_key`（`CKIpionManager::CreateEnvironment` 装 CK id）。

## 补丁集与我们自己的源码

| 文件 | 内容 |
|---|---|
| `patches/0002-ballrain-agent-entry.diff` | `sim_tool.cpp` 的 `--ballrain-agent` 入口：用 SimTool 自己的 `parse()` 读参数（`--root`、`--level`、`--boot-ticks`、`--tas-*` 及其默认值），把它们和 SimTool 的 `apply_tas_legacy_edit` 交给 agent；CMake 把 `sim/ballrain_agent.*` 加进 BallanceMMOSim |
| `src/ballrain_agent.cpp`、`src/ballrain_agent.h`、`src/ballrain_wire.h` | agent 模式本体，复制到 `BallanceMMOServer/sim/`。帧方案（legacy 改动、复位状态、不在加载关卡时重置精度、34 个隐藏 tick）用 SimTool `--tas` 的参数和默认值，自己不再重复设；另有球-球接触顺序的模式（`BALLRAIN_HL_BALL_ORDER`，`docs/headless.md` §3.1） |

只加东西，不改产品代码。诊断用的补丁（相机注入、力 / 冲量日志等）不在补丁集里。

## 编译（macOS arm64，Apple clang；Windows x64 同样的 CMake 参数）

```sh
SDL3_DIR=<SDL3 源码树> CMAKE=<cmake> NINJA=<ninja> JOBS=<n> headless/build_mac.sh OUT/src.tar WORK
# -> WORK/build/BallanceMMOServer/BallanceMMOSimTool
```

只编 sim（`BMMO_BUILD_SIM_ONLY`）、x87 SOFT（`BMMO_PHYSICS_X87=SOFT`），`VX_X87_GAME` 由产品的 `cmake/BallancedHeadless.cmake` 加上；
agent 在关卡加载后检查环境装了球序键，没有（例如构建里缺了 `VX_X87_GAME`）就报错退出。
SDL3 在 macOS 上需要两处本地修改（见 `build_mac.sh` 头部）。Windows 上顶层 CMake 还要 vcpkg 的工具链和已装好的包（Protobuf、GameNetworkingSockets）。

正在被训练使用的二进制不要覆盖：`setup_mac.sh SRC_TAR` 把每个构建放进新目录 `~/ballrain/hl/x87s-<BallanceMMO 提交前 8 位>-<0002 哈希>-<agent 哈希>`，
门禁通过之后再 `setup_mac.sh --activate` 写 `hl/ENGINE`（训练启动时读它）。

## Windows x86：跑原版 2002 physics_RT.dll 的宿主（补丁 0003）

同一个无头引擎的 32 位 Windows 构建，物理换成**零售原版** `physics_RT.dll`（sha256 `e72e4afc…`，与 BallrainIO 校验的是同一个文件）。
CK 侧（矩阵、相机、脚本、BuildingBlocks）仍是 Ballanced 源码，所以相机 / 装饰物体的那些差异（`docs/headless.md` §7.1）照旧；IVP 侧就是 2002 的二进制。

| 文件 | 内容 |
|---|---|
| `patches/0003-hlwin-phys2002.diff` | 只改产品文件：CMake 加 `sim/phys2002.*`、`sim/Ivp2002.h` 和宏 `BALLRAIN_PHYS2002_HOST`；`headless_engine.cpp` 在注册插件时用 2002 DLL 顶替源码的 physics_RT，在建 CKContext 处把 x87 设成 24 位（Player.exe 的 D3D8 设备留下的状态）；`sim_tool.cpp --tas` 的 2002 时钟复位；`CKIpionManager::GetManager` 在 2002 模式下返回 NULL（那个 GUID 下是另一个类）；`CK3dEntity::GetOrientation` 用 x87 的 Normalize（2002 的 Set Physics Hinge 调它） |
| `src/phys2002.cpp`、`src/phys2002.hpp` | 装载器：先核对 sha256，`LoadLibraryEx(DONT_RESOLVE_DLL_REFERENCES)` 映射，自己填导入表（CK2 / VxMath 名字 → 静态链接的 Ballanced 上的 thiscall 垫片，`VxQuaternion::FromMatrix` → 零售 `Bin/VxMath.dll`，MSVCRT / KERNEL32 → 系统 DLL），再跑它的 DllMain，作为静态插件 "physics_RT" 注册 |
| `src/ballrain_agent.cpp`（`BALLRAIN_PHYS2002_HOST` 下的部分） | agent 模式的 TIME / BALL / FRIC / MIND / BODY / GLCT / IVPO 按 `BallrainIO/src/Agent.cpp` 读 DLL 内存（偏移来自 `BallrainIO/src/Ivp2002.h`，构建时复制过去，不另存一份）；时钟复位写 DLL 内存（TASSupport 的复位 + nmc 4 + 种子 75^21）；不跑 BMMO 的脚本补丁；hello 串是 `BallrainIO-headless 0.1 phys2002 e72e4afcfa5c33a7 engine headless level N` |
| `build_win.ps1` | `powershell -File headless/build_win.ps1 SRC_TAR WORK`：解包 `mk_src.sh` 的 src.tar、打 0002 + 0003、复制 `headless/src` 和 `Ivp2002.h`、x86 配置并编译。路径都来自环境变量（`VCVARS`、`CMAKE`、`NINJA`、`VCPKG_ROOT`、`VCPKG_INSTALLED`（x86-windows）、`SDL3_DIR`、`GTEST_DIR`、`YAMLCPP_DIR`、`PATCH`、`JOBS`，见脚本头部） |

`build_mac.sh` / `setup_mac.sh` 只用 0002（0003 只能在 x86 上编译）。

**对映射后的镜像只改一处**（文件不改，hash 照验）：IVP 里唯一一处球-球接触顺序的比较（RVA `0x164C1`，`pop0->client_data < pop1->client_data`，
即两个 CK3dEntity 的堆地址）改成调用一个键函数，先核对原字节再改。默认键是 CK id（与 x87-soft 产品构建的 `bmmo_ball_order_key` 相同），
`BALLRAIN_HL_BALL_ORDER` 的各模式（`docs/headless.md` §3.1）照样生效。`BALLRAIN_HL_PHYS2002_HOOK=0`（诊断）不改，按本进程堆地址排（只允许 `ckid`）。

选择物理：agent 模式默认用 `<root>/BuildingBlocks/physics_RT.dll`（必须是 2002 那个文件，不对就报错退出，不会悄悄退回源码物理）；
`BALLRAIN_HL_PHYS2002=<路径>` 换文件，`BALLRAIN_HL_PHYS2002=0` 用同一个二进制里的源码 x87-soft 物理。SimTool `--tas` 仍由 `BALLRAIN_PHYS2002=<路径>` 打开。
所以 `train_m2.py --backend headless` 不用改：`--hl-bin` 指向这个构建、`--hl-root` 指向装着零售 physics_RT.dll 的游戏目录即可（Windows 上 socket 自动是 TCP）。
回放驱动 `tools/replay_stub.py` 在没有 AF_UNIX 的 Python 上用 TCP，并把 hello 串写进 `summary.json`；`scripts/headless_gate.py --phys2002` 要求 hello 里有 `phys2002`，
并把 x87-soft 引擎给不出的 IVP 字段（FRIC lt_*、GLCT 计数、BALL / BODY 原始位域）当精确字段比较，只有地址（MIND ledge、Energy 参数单元）和 IVP 不维护的残值（MIND plane w）单列。
门禁结果和吞吐见 `docs/headless.md` §13。

## 运行

- Python：`EnvConfig(backend="headless", headless={"binary": ..., "root": <游戏目录>})`，或设 `BALLRAIN_HL_BIN` / `BALLRAIN_HL_ROOT`；
  `HeadlessConfig` 的帧方案字段只在与 SimTool 默认值不同时才变成参数（`--tas-offset`、`--tas-seed` 等），`ball_order` 是接触顺序。
- 不依赖 Python 包的回放驱动：`python3 headless/tools/replay_stub.py --bin ... --root ... --level N --frames REC.raw --out DIR`，
  输出可交给 `scripts/headless_gate.py` 和零售数据集比较。
- 引擎直接调用：`BallanceMMOSimTool --ballrain-agent --root <游戏目录> [--tas-* ...]`，其余配置来自 `BALLRAIN_*` 环境变量（`docs/headless.md`）。
