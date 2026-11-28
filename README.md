# Ballrain

在原版 Ballance（2004 零售版 Player.exe + 2002 版 physics_RT.dll）上训练强化学习 agent。硬性要求：
轨迹来自原版物理，并且每个回合都能导出成 legacy .tas，由真正的 TASSupport（patched 0.3.12，LegacyMode 1）回放出完全相同的轨迹。

两个独立目标：**SR**（出生到登上飞船的最短游戏内时间，按 BML+ 0.3.12 的 SR 计时器）和 **HS**（结算分数）。基准见 `docs/benchmarks.md`。

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
| `tools/dashboard/` | 训练进度网页（局域网可访问，自动刷新），见下文“训练进度网页” |
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
| `state.py`、`geometry.py`、`path.py`、`features.py` | 状态累积、几何、参考路径（TAS 参考线 / 盘点折线）、特征 |
| `rewards.py`、`recording.py` | 奖励；每个进程的 .tas + 元数据（SR 按 BML+ 0.3.12 规则） |
| `env.py`、`vecenv.py` | gymnasium 环境（每条命一个回合，可从 TAS 第 k 帧起步，可预启动下一个进程）、N 个拷贝的向量环境 |
| `curriculum.py` | TAS 起点与反向课程 |
| `train/` | M2 训练：`ppo.py`（网络、观测归一化、GAE、PPO 更新）、`worker.py`（每个拷贝一个异步 rollout 进程） |
| `dataset.py` | TAS 数据集构建 |
| `fakegame.py` | 模拟 mod（同一套传输和线格式），用于测试和离线开发 |
| `headless.py` | 无头后端的启动器（`EnvConfig(backend="headless")`），见 `docs/headless.md` |
| `segsearch/` | 段搜索（“自动 TAS”）：TAS 两个门之间在 headless 引擎上搜比 TAS 快的按键，冷启动验证（`python -m ballrain.segsearch`，见 `docs/segsearch.md`） |

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
- 抢焦点（BallrainIO 0.3.2 交还前台；2026-11-16 舰队仍是 0.3.1，等推送）：Player.exe 显示主窗口后立刻 `SetFocus`，这时 mod 还没加载
  （约 0.2 s 后才 OnLoad），Windows 照样让新进程拿到前台（前台是用户窗口时也一样），隐藏窗口成了前台窗口，键盘输入进了看不见的窗口；
  0.3.1 下要一直占到进程退出（每次启动约 4 s）。0.3.2 在 OnLoad 和之后每 0.1 s（OnProcess，任何模式）检查：前台窗口属于本进程**且不可见**时，
  把前台交给 z 序最上面的、用户可能在用的窗口（可见、未最小化、未 cloak、不是 tool / no-activate / 穿透窗口、不是被拥有的窗口、别的进程），
  记一行 `focus (OnLoad): foreground handed to pid ...` 到 ModLoader.log。只调窗口管理 API，不碰游戏对象、输入和时间。
  占有前台的进程可以随时交出前台，所以交还总能成功；之后游戏自己的再次激活拿不到前台（只多一个不改变前台的 EVENT_SYSTEM_FOREGROUND 事件，
  2 ms 轮询 `GetForegroundWindow` 看不到 Player）。可见的游戏窗口（手动启动、`BALLRAIN_VISIBLE=1`、`explorer_cmd`）从不交还。
  - 实测（2026-11-16，3 个拷贝循环重启 SR06 回放 60 s，每 5 ms 轮询 `GetForegroundWindow`）：Player 占前台的时间从 71.3%（每次中位 4.08 s）
    降到 7.4–9.2%（每次中位 0.19 s，最长 0.35 s），每拷贝 10–11 次运行（基线 11）；交还目标都是此前的前台窗口。
    剩下的约 0.2 s/次是 SetFocus 到 mod 加载之间，mod 管不到：按舰队每 300 s 约 243 次启动算，仍约 15%，没有到 0。
    按事件计时的 `fg_watch` 会把上面那个假事件算成 Player 占前台直到下一个事件，新 mod 下读数虚高（约 89%），要用轮询测。
  - 不交还：`BALLRAIN_FOCUS=1`（启动器变量，不传给游戏）或 `launch_direct(..., keep_focus=True)`；启动器随即给游戏设 `BALLRAIN_KEEP_FOCUS=1`
    （可见窗口也设）。
  - 管不到：`ref`（没有 BallrainIO，每次回放仍占前台到退出）。启动器侧试过，不行：启动器 `SetForegroundWindow` 交还几乎总被拒
    （210 次里 208 次失败，占比反而升到 80%），启动器在启动前调 `LockSetForegroundWindow(LSFW_LOCK)`（返回 1）后 Player 照样在 0.9 s 后拿到前台。
    再往下压只能让代码在主窗口出现之前运行（例如额外的 BuildingBlocks 插件 DLL 装线程 CBT 钩子，挡住 HCBT_ACTIVATE），超出 mod 的范围，没做；
    由游戏进程在交还前调 `LockSetForegroundWindow` 没测过，而且会锁住整个桌面的前台切换直到用户按 Alt 或点击，不推荐。
  错误对话框（MessageBox）同样看不见，进程会挂到超时，这时用 `BALLRAIN_VISIBLE=1` 重跑看。
  另一个桌面（`CreateDesktop` + `lpDesktop`）能把一切藏起来，但不可用：不在输入桌面上时 D3D 报告 HAL 设备不可用（`D3DERR_NOTAVAILABLE`），
  Player.exe 以 "No render driver found" 退出。

## 游戏声音（默认静音）

`instances.launch_direct` 启动后把这个 PID 的 Windows 音频会话静音（`ballrain/audio.py`，ctypes 直接调 Core Audio：枚举所有输出设备的
`IAudioSessionManager2` 会话，按 `IAudioSessionControl2::GetProcessId` 匹配自己启动的 PID，`ISimpleAudioVolume::SetMute(TRUE)`）。
会话要等游戏打开音频设备才出现（启动后约 0.8 s），所以后台线程每 0.1 s 查一次，直到找到会话，之后每 2 s 再查一次直到进程退出（会话重建也会被再静音）。
只动自己启动、映像路径核对过是 `<copy>/Bin/Player.exe` 的进程；不改系统设置、设备音量或别的程序。游戏本身完全不变（只在音频引擎里静音），`ref` 也适用。

- 打开声音：`BALLRAIN_AUDIO=1`（启动器变量，worker 继承，不传给游戏）或 `launch_direct(..., audio_on=True)`。
- Windows 按可执行文件路径记住静音：某个拷贝的 Player.exe 被静音过一次，以后从同一路径启动的进程一开始就是静音的（不经过本模块也一样），
  所以 `BALLRAIN_AUDIO=1` 会主动取消静音，而不是不管。要手动恢复：Windows 音量合成器里取消静音，或用 `BALLRAIN_AUDIO=1` 跑一次。
- 验证（2026-11-16，`test-nowin` / `test-nowin-ref`）：静音在会话出现后 0.06–0.4 s 内生效（此前会话处于 inactive，峰值 0）。
  会话自己的峰值表在静音之前测量，静音时照样跳动（约 1.5）；输出设备的峰值表（混音之后）在静音运行中始终为 0，不静音时为 1.0。
  BallrainIO 回放 SR06 / SR01 / HS03 的 trace、framelog 与 `runtime/m0-results-r2/G1/<record>/bio/` 逐字节相同，真 TASSupport 回放 SR06 与 `ref_trace.txt` 相同。
- 游戏自己的音量选项（`Database.tdb` 的 `DB_Options.Volume`，原值 1.0）不起作用：设成 0 后（只改这 4 个字节，trace 照样相同）
  BallrainIO 和 TASSupport 回放的声音和原来一样响，所以拷贝里保持原值。工具留着：`instances.tdb_volume` / `set_tdb_volume`，
  `scripts/make_copies.py --volume V [--also ref]`；`verify_copy_matches` 现在也比较 `.tdb` 的 sha256。

## 训练进度网页

```bash
uv run python tools/dashboard/server.py --host 0.0.0.0 --port 8088   # 或在 Claude 预览里启动 launch.json 的 "dashboard"
```

浏览器打开 `http://<本机 IP>:8088/`（手机同一局域网也可以）。页面按 Ballance 的菜单风格显示当前 run 的状态、课程推进记录、
SR 估计（对比 TAS / 人类纪录 / 精品线 / 上榜线）、课程开局点地图、最近尝试轨迹、历史轨迹抽样和训练曲线，全部是浏览器里可缩放、
可悬停、可开关图层的交互图表；可切换 r3 / r4 / r5。页面每 60 秒自动更新，数据每 10 分钟重算（“立即刷新”全局每 5 分钟最多一次）。
选项、工作方式以及公开 / 不公开哪些内容见 `tools/dashboard/README.md`。

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
