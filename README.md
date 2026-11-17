# Ballrain

在原版 Ballance（2004 零售版 Player.exe + 2002 版 physics_RT.dll）上训练强化学习 agent。硬性要求：
轨迹来自原版物理，并且每个回合都能导出成 legacy .tas，由真正的 TASSupport（patched 0.3.12，LegacyMode 1）回放出完全相同的轨迹。

两个独立目标：**SR**（出生到登上飞艇的最短游戏内时间，按 BML+ 0.3.12 的 SR 计时器）和 **HS**（结算分数）。基准见 `docs/benchmarks.md`。

## 目录

| 路径 | 内容 |
|---|---|
| `BallrainIO/` | C++ BML+ 0.3.12 mod。回放模式（M0）复现 TASSupport legacy 回放；agent 模式按 `docs/agent-protocol.md` 与 Python 锁步 |
| `headless/` | 无头后端（近似环境）：BallanceMMO x87-soft 无头引擎上的 agent 模式，同一套线格式，socket 传输；补丁集、源码提交、Mac 编译与训练脚本（`docs/headless.md`） |
| `ballrain/` | Python 包（见下表） |
| `scripts/m0_gates.py` | M0 保真门禁 G0–G5（启动游戏） |
| `scripts/make_copies.py` | 创建 / 校验训练拷贝 `runtime/train-01..07`，同步新编译的 mod |
| `scripts/build_tas_dataset.py` | 用 agent 模式回放全部 TAS，保存逐帧原始状态（启动游戏） |
| `scripts/build_refpaths.py` | 从 TAS 数据集生成各关的 TAS 参考线 `benchmarks/refpaths/L<nn>.npz`（离线） |
| `scripts/m2_start_states.py` | TAS 起点的开销、吞吐、导出验收（启动游戏，结果在 `runtime/m2-results/`） |
| `scripts/train_m2.py` | M2 训练：PPO + 反向课程，L1 SR（启动游戏，输出在 `runtime/m2-train/<run>/`，见 `docs/observations.md` §9） |
| `scripts/eval_m2.py` | M2 评估（从关卡加载闭环运行）与导出验收（真 TASSupport 回放 3 次） |
| `docs/` | `benchmarks.md` 基准，`state-spec.md` 游戏状态清单，`agent-protocol.md` IPC 契约，`observations.md` 观测 / 奖励 / 录制，`headless.md` 无头后端 |
| `tests/` | `uv run pytest`；不启动游戏（`BALLRAIN_LIVE=1` 时才跑 `test_live.py`） |
| `reference/TASSupport/` | 参考 TASSupport 源码 |
| `legacy/` | 早期 TCP 原型（`BallanceEnv.py`、`TCPServer.py`、`Message.py`、`main.py`、`validate_image.py`、`comm_protocol.md`），已不再使用，只作参考 |
| `runtime/` | 游戏拷贝和运行输出（不提交）：`ref`（只装真 TASSupport，作参考）、`train-00..07`（只装 BallrainIO）、`episodes/`、`datasets/` |

`ballrain/` 各模块：

| 模块 | 作用 |
|---|---|
| `tas.py` | legacy .tas 编解码 |
| `trace.py`、`gates.py` | 逐字节轨迹比较、门禁判定 |
| `instances.py` | 游戏拷贝管理与启动（只在 `runtime/<copy>/Bin` 下启动，只停止自己启动并核对过路径的进程），创建训练拷贝 |
| `synth.py` | 合成测试输入 |
| `protocol.py`、`transport.py`、`client.py` | agent 模式：线格式、共享内存 + 事件、单进程会话 |
| `state.py`、`geometry.py`、`path.py`、`features.py` | 状态累积、几何、参考路径（TAS 参考线 / 检查点折线）、特征 |
| `rewards.py`、`recording.py` | 奖励；每个进程的 .tas + 元数据（SR 按 BML+ 0.3.12 规则） |
| `env.py`、`vecenv.py` | gymnasium 环境（每条命一个回合，可从 TAS 第 k 帧起步，可预启动下一个进程）、N 个拷贝的向量环境 |
| `curriculum.py` | TAS 起点与反向课程 |
| `train/` | M2 训练：`ppo.py`（网络、观测归一化、GAE、PPO 更新）、`worker.py`（每个拷贝一个异步 rollout 进程） |
| `dataset.py` | TAS 数据集构建 |
| `fakegame.py` | 模拟 mod（同一套传输和线格式），用于测试和离线开发 |
| `headless.py` | 无头后端的启动器（`EnvConfig(backend="headless")`），见 `docs/headless.md` |

## 环境准备

```bash
uv sync
uv run pytest
```

游戏拷贝放在 `runtime/` 下（可用环境变量 `BALLRAIN_RUNTIME` 改位置）。`runtime/ref` 只装真 TASSupport；`runtime/train-00` 只装 BallrainIO。

编译 mod：`BallrainIO/build-local.bat`（本机脚本，不提交）生成 `BallrainIO/build/BallrainIO.bmodp`，复制到 `runtime/train-00/ModLoader/Mods`。然后：

```bash
uv run python scripts/make_copies.py            # 创建并校验 train-01..07（physics_RT 哈希、二进制 sha256）
uv run python scripts/make_copies.py --sync-mod # 把新编译的 mod 装进全部拷贝
```

## 游戏窗口（默认隐藏）

`instances.launch_direct`（所有脚本、`client.py` / `env.py` 的 `launch="direct"`、`run_ballrain`、`run_reference` 都走它）用
`STARTUPINFO(dwFlags=STARTF_USESHOWWINDOW, wShowWindow=SW_HIDE)` 启动 Player.exe：Player.exe 用 `ShowWindow(SW_SHOW)` 显示主窗口时，
Windows 把这第一次调用换成 SW_HIDE，之后窗口模式下它不会再显示主窗口。没有窗口，也没有任务栏按钮。只靠启动器，不改 mod，
所以 `ref`（只装真 TASSupport）同样适用；`runtime/ref-run.ps1` 用 `Start-Process -WindowStyle Hidden`，效果相同。

- 调试时显示窗口：`BALLRAIN_VISIBLE=1`（环境变量，spawn 出来的 worker 进程会继承；不传给游戏），或 `launch_direct(..., visible=True)`，
  或 `ref-run.ps1 -Visible`。`explorer_cmd` 配方的窗口总是可见。
- 启动画面（splash）：Player.exe 在主窗口之前弹出 `Bin/splash.bmp`（约 0.4 s，会抢前台），启动器管不到它。
  `uv run python scripts/make_copies.py -n 16 --splash off --also ref` 把所有 `train-NN` 和 `--also` 列出的拷贝的 `Bin/splash.bmp` 改名为 `splash.bmp.off`
  （Player.exe 读不到图片就跳过 splash 并继续，返回值不检查），`--splash on` 改回。拷贝里有 Player.exe 在跑时不改任何拷贝。
  新拷贝从 `train-00` 复制，状态跟着走；`verify_copy_matches` 会报出不一致的拷贝。
- 保真（2026-11-16，`runtime/test-nowin` / `test-nowin-ref`，有无 splash 都测了）：BallrainIO 回放 SR06_1_36_163_132、SR01_1_14_457_264、
  HS03_4906_1_56_934 的 trace 和 framelog 与 `runtime/m0-results-r2/G1/<record>/bio/` 逐字节相同（SR06 在不跳过渲染时也相同）；agent 模式（BallanceEnv，SR01 第 3000 帧
  TAS 前缀交接，400 步固定随机动作）可见 / 隐藏 12 次运行的 framelog、trace、.tas、观测完全相同（events.txt 只差 Summary 的 wall_ms）；
  真 TASSupport 回放 SR06、SR01、HS03 与 `ref_trace.txt` 相同；`Bin/Player.ini` 不被改写。启动时间和 steps/s 与可见窗口相同（在噪声内）；
  去掉 splash 启动约快 0.5–0.8 s。
- 限制（抢焦点没有解决）：Player.exe 显示主窗口后立刻 `SetFocus`（在 mod 的 OnLoad 之前约 0.2 s），所以 Windows 允许新进程拿前台时
  （例如前台本来就是另一个 Player，或此刻没有前台窗口），隐藏的主窗口在启动后约 1–2 s 照样成为前台窗口，键盘输入进了一个看不见的窗口，
  直到用户点别的窗口或下一个 Player 启动再抢走。隐藏后只少了两件事：运行期间不会被再次激活（可见窗口会被反复激活），
  以及（关掉 splash 时）每次启动的抢前台从两三次变成一两次。只能在 mod 里补救（OnLoad 时把前台交还），而且管不到 `ref`。
  错误对话框（MessageBox）同样看不见，进程会挂到超时，这时用 `BALLRAIN_VISIBLE=1` 重跑看。
  另一个桌面（`CreateDesktop` + `lpDesktop`）能把一切藏起来，但不可用：不在输入桌面上时 D3D 报告 HAL 设备不可用（`D3DERR_NOTAVAILABLE`），
  Player.exe 以 "No render driver found" 退出。

## 使用

```python
from ballrain.env import BallanceEnv, EnvConfig
from ballrain.vecenv import make_vec_env

env = BallanceEnv(EnvConfig(level=1, mode="sr", action_repeat=4))   # 单实例，runtime/train-00
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(1)               # 1 = 上

venv = make_vec_env(EnvConfig(level=1, mode="hs"), n=8)              # train-00..07，每个拷贝一个子进程

# 从 TAS 轨迹的第 k 帧起步，k 由反向课程抽取（从关卡末尾往前）
cfg = EnvConfig(level=1, mode="sr", action_repeat=4, start_record="SR01_1_14_457_264", curriculum={},
                max_episode_frames=6000)
venv = make_vec_env(cfg, n=8)                                        # warm_pool=True 时每个 env 用两个拷贝
```

课程的更新接口见 `docs/observations.md` §5.1（`ReverseCurriculum.update` + `venv.call("set_start_state", ...)`）。

- 每条命一个 RL 回合；BallOff 结束回合，进程继续到 GameOver / LevelFinish / 上限。每个进程结束时在 `runtime/episodes/<copy>/<run_id>/`
  写出 `<run_id>.tas`（legacy）和 `<run_id>.json`（每条命的边界、SR、HS、帧日志核对）。
- 观测、奖励、录制的细节见 `docs/observations.md`。
- 不启动游戏的开发和测试：`BallanceEnv(cfg, launcher=ballrain.fakegame.FakeLauncher())`。

M0 门禁：`uv run python scripts/m0_gates.py --gates G0`（会启动游戏）。导出的回合按 M0 规则验收：参考 TASSupport 回放 3 次，与多数结果逐字节相同即通过。
