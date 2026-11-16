# Ballrain

在原版 Ballance（2004 零售版 Player.exe + 2002 版 physics_RT.dll）上训练强化学习 agent。硬性要求：
轨迹来自原版物理，并且每个回合都能导出成 legacy .tas，由真正的 TASSupport（patched 0.3.12，LegacyMode 1）回放出完全相同的轨迹。

两个独立目标：**SR**（出生到登上飞艇的最短游戏内时间，按 BML+ 0.3.12 的 SR 计时器）和 **HS**（结算分数）。基准见 `docs/benchmarks.md`。

## 目录

| 路径 | 内容 |
|---|---|
| `BallrainIO/` | C++ BML+ 0.3.12 mod。回放模式（M0）复现 TASSupport legacy 回放；agent 模式按 `docs/agent-protocol.md` 与 Python 锁步 |
| `ballrain/` | Python 包（见下表） |
| `scripts/m0_gates.py` | M0 保真门禁 G0–G5（启动游戏） |
| `scripts/make_copies.py` | 创建 / 校验训练拷贝 `runtime/train-01..07`，同步新编译的 mod |
| `scripts/build_tas_dataset.py` | 用 agent 模式回放全部 TAS，保存逐帧原始状态（启动游戏） |
| `docs/` | `benchmarks.md` 基准，`state-spec.md` 游戏状态清单，`agent-protocol.md` IPC 契约，`observations.md` 观测 / 奖励 / 录制 |
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
| `state.py`、`geometry.py`、`path.py`、`features.py` | 状态累积、几何、参考路径、特征 |
| `rewards.py`、`recording.py` | 奖励；每个进程的 .tas + 元数据（SR 按 BML+ 0.3.12 规则） |
| `env.py`、`vecenv.py` | gymnasium 环境（每条命一个回合）、N 个拷贝的向量环境 |
| `dataset.py` | TAS 数据集构建 |
| `fakegame.py` | 模拟 mod（同一套传输和线格式），用于测试和离线开发 |

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

## 使用

```python
from ballrain.env import BallanceEnv, EnvConfig
from ballrain.vecenv import make_vec_env

env = BallanceEnv(EnvConfig(level=1, mode="sr", action_repeat=4))   # 单实例，runtime/train-00
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(1)               # 1 = 上

venv = make_vec_env(EnvConfig(level=1, mode="hs"), n=8)              # train-00..07，每个拷贝一个子进程
```

- 每条命一个 RL 回合；BallOff 结束回合，进程继续到 GameOver / LevelFinish / 上限。每个进程结束时在 `runtime/episodes/<copy>/<run_id>/`
  写出 `<run_id>.tas`（legacy）和 `<run_id>.json`（每条命的边界、SR、HS、帧日志核对）。
- 观测、奖励、录制的细节见 `docs/observations.md`。
- 不启动游戏的开发和测试：`BallanceEnv(cfg, launcher=ballrain.fakegame.FakeLauncher())`。

M0 门禁：`uv run python scripts/m0_gates.py --gates G0`（会启动游戏）。导出的回合按 M0 规则验收：参考 TASSupport 回放 3 次，与多数结果逐字节相同即通过。
