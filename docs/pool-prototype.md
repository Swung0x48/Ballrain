# 志愿者算力池原型：协议与文件

状态：第 1 步（共享约定）已实现，见 `ballrain/pool/`；第 2b 步主控程序已实现（`tools/pool/controller.py` 和 `ballrain/pool/client_*.py`，第 9 节）；第 2a 步网关和验证器已实现（`tools/pool/gateway.py`、`tools/pool/verifier.py`、`tools/pool/mock_forum.py` 和 `ballrain/pool/server_*.py`、`rollout.py`，第 10 节）；评审修复（第 2c 步）见第 10.9 节。
设计和与用户逐项对过的决定见 `docs/volunteer-compute.md`，尤其是第 11 节。本文是实现层面的约定：接口、消息、数据段格式、原因码、文件布局。

**原型范围：** 志愿者的电脑做训练采样。网关收下的数据段只存盘、计数，**不接入正在运行的训练器**。

## 1. 模块

| 模块 | 内容 | 谁用 |
|---|---|---|
| `ballrain/pool/reasons.py` | 拒收 / 吊销原因码 `Reason`，中文说明 `MESSAGES`，`render()` / `payload()` / `http_status()`，`TOKEN_FATAL` | 网关、主控程序 |
| `ballrain/pool/protocol.py` | 严格 JSON（`loads_strict` / `dumps`）、消息 schema（`MESSAGES`、`validate` / `parse_message` / `make_message` / `encode_message`）、数据段编解码（`encode_chunk` / `decode_chunk`）、引擎状态哈希链（`chain_seed` / `chain_from_digests` / `check_chain_link`；行摘要在 `engine_chain.py`）、验证容差（`OBS_TOL` / `REWARD_TOL` / `ARC_TOL`、`obs_close` / `value_close`）、游戏文件清单（`GAME_FILE_LIST(_2002)`、`tree_digest`） | 全部 |
| `ballrain/pool/weights.py` | 检查点导出成 npz + 清单（`export_checkpoint` / `export_state`，只在服务端、只读学习器自己的检查点），客户端加载（`read_manifest` / `load_policy`），纯 numpy 前向 `NumpyPolicy` | 网关（导出）、客户端、验证器 |
| `ballrain/pool/sampling.py` | 采样任务的确定性动作抽取：`sample_u(sample_seed, t)`、`sample_action`、重放用的 `action_consistent`（第 10.6 节，第 2c 步新增） | 主控程序、验证器 |
| `ballrain/pool/task.py` | 任务 `Task`、本机游戏 `LocalGame`、`hash_game_files`、开局包和参考线（`export_starts_pack` / `install_starts_pack`、`export_path_asset` / `find_path_asset` / `install_path_asset`）、`build_env_config` / `build_env`、`valid_start_frames`、`save_npz_deterministic` | 网关（发任务、导出素材）、客户端、验证器 |
| `ballrain/pool/rollout.py`、`server_*.py`、`dev.py` | 验证器的任务重放（`TaskRunner`、`replay_task`）；网关的配置、sqlite 存储、论坛 OAuth、任务来源、HTTP 应用、验证器；假游戏上的开发 / 测试环境（第 10 节） | 网关、验证器、测试 |

**不用 pickle。** 协议、发给客户端的文件、客户端读取的文件都不经过 pickle：
- JSON 用标准解析器加严格检查；
- 数组用固定 dtype 的扁平二进制（`np.frombuffer`，dtype 由本模块决定，不由发送方决定）；
- npz 一律 `np.load(allow_pickle=False)`，写出时也是 `write_array(allow_pickle=False)`；
- `torch.load` 只出现在 `weights.load_trusted_checkpoint` 里（`export_checkpoint` 和网关的任务来源调用它），只用于学习器自己的检查点。

客户端会导入的模块（`protocol`、`reasons`、`weights`、`task`）不在模块层面导入 torch。测试 `test_pool_sources_are_pickle_free` 和 `test_client_modules_do_not_import_torch` 守着这两条。

## 2. 接口

公网地址在网关本机的配置里（第 6 节）。所有 JSON 请求和响应都是 `application/json; charset=utf-8`，内容是 `protocol.MESSAGES` 里的一种消息：
- 带 `"protocol": 1` 和 `"type"`；
- 不认识的字段、类型不对（布尔值不算整数）、越界、超长、重复键、NaN / Infinity 一律拒收；
- 单条消息上限 64 KiB，嵌套上限 12 层。

**认证：**
- 令牌放在 `Authorization: Bearer <pool token>`；
- 会话 id（`hello_ok.session_id`）放在 `X-Pool-Session`；
- 标“公开”的接口不需要令牌，内容按 sha 命名、不可变，可以让 Cloudflare 缓存（`Cache-Control: public, max-age=31536000, immutable`）。

| 方法 和 路径 | 认证 | 请求 | 响应 |
|---|---|---|---|
| `POST /pool/login` | 无 | `login_start` 或 `login_poll`（按 `type` 区分） | `login_started` / `login_status` |
| `GET /pool/login?login=<login_id>` | 无（浏览器） | — | 302 跳到论坛授权页 |
| `GET /pool/oauth/callback?code=…&state=…` | 无（浏览器） | — | 中文 HTML 页：“登录成功，可以回到主控程序”，或者原因码的说明 |
| `POST /pool/hello` | 令牌 | `hello` | `hello_ok` |
| `GET /pool/task` | 令牌 + 会话 | — | `task` 或 `no_task` |
| `GET /pool/weights/<policy_sha>.json` | 公开 | — | 权重清单（第 4 节） |
| `GET /pool/weights/<policy_sha>.npz` | 公开 | — | 权重数组 |
| `GET /pool/assets/<sha256>.npz` | 公开 | — | 开局包或参考线（第 5 节）。**这是在任务给定的接口清单之外新增的一个接口**：任务要用的开局状态和参考线也按 sha 分发 |
| `POST /pool/chunk` | 令牌 + 会话 | `application/octet-stream`，一个数据段（第 3 节），上限 4 MiB | `chunk_result` |
| `POST /pool/heartbeat` | 令牌 + 会话 | `heartbeat` | `heartbeat_ok`（令牌被吊销时带 `revoked`） |
| `GET /pool/me` | 令牌 | — | `me` |
| `POST /pool/me` | 令牌 | `me_update`（目前只有“上榜”开关） | `me` |
| `GET /pool/leaderboard` | 公开 | — | `leaderboard`（只有上榜玩家的论坛昵称和通过验证的步数） |

**出错时：**
- 响应是 `error` 消息，带 `reason`（第 7 节），HTTP 状态码见 `reasons.http_status`：令牌问题 401 / 403，格式问题 400，限速 429，重复或乱序 409，超大 413；
- `no_task` 用 200；`replay_failed` 用 202（数据先保留，不算玩家的错）；
- 主控程序把 `reason.message` 原样显示给玩家；
- 收到 `TOKEN_FATAL` 里的原因码时，停止所有实例，提示重新登录。

### 2.1 登录（论坛 OAuth，授权码流程）

1. 主控程序 `POST /pool/login`，发 `login_start`，收到 `login_started`：`login_id`、`poll_secret`、`browser_url`（即 `GET /pool/login?login=<login_id>`）、`user_code`（登录代码 `XXXX-XXXX`，主控程序显示）、有效期、轮询间隔。
2. 主控程序打开浏览器访问 `browser_url`。这个地址**不直接跳论坛**，而是一个确认页：显示登录代码，请玩家和主控程序上的核对，并提醒“链接是别人发来的就关掉”。页面上的“继续”链接带一次性随机数，同时设一个 `SameSite=Strict`、HttpOnly 的 cookie（`pool_login`，两者的 sha256 都记在这次登录上）；只有同一个浏览器拿着这个 cookie 点这个链接才继续，否则重新显示确认页。继续之后，网关生成随机的 `state`，和 `login_id` 绑定，然后 302 到 `https://forum.ballance.top/oauth/authorize`，参数 `response_type=code`、`client_id`、`redirect_uri`（精确匹配）、`scope=user.read`、`state`。这样别人把自己发起的登录链接发给玩家，也拿不到玩家的令牌（除非玩家无视代码和提醒，亲手点了继续）。
3. 论坛回调 `GET /pool/oauth/callback`。网关做这几件事：
   1. 校验 `state`（只能用一次，过期就是 `oauth_state_mismatch`）；
   2. 在服务端用 `POST https://forum.ballance.top/oauth/token` 换取论坛令牌；
   3. 用 `GET https://forum.ballance.top/api/user` 读取 `id`、`username`、`nickname`、`avatar_url`、`is_email_confirmed`、`joined_at`、`comment_count`；
   4. **`email` 读到就丢，不保存，也不写日志**；论坛令牌用完就丢；
   5. 检查门槛：注册满 30 天（`account_too_new`）、邮箱已验证（`email_not_verified`）、至少 5 条回复（`too_few_replies`，按 `comment_count` 算）。
4. 主控程序按 `poll_s` 的间隔 `POST /pool/login` 发 `login_poll`，状态是 `pending`、`ok`（带 `token` 和 `user`）、`refused`（带 `reason`）或 `expired`。成功页显示论坛用户名和登录代码。
5. 每个论坛账号只有一个有效令牌：重新登录会发新令牌，旧令牌随即失效（`token_invalid`）。

服务端只保存论坛用户 ID、用户名、昵称和令牌的哈希，不保存邮箱，IP 不写日志。

### 2.2 会话和任务

- `hello` 报告这几项：客户端版本、平台（`windows` / `macos` / `linux`）、引擎（`kind` 为 `2002` 或 `x87soft`，带引擎文件 sha256 和版本）、`game_files`、想跑的实例数。
- `game_files` 的键正好是 `GAME_FILE_LIST`（2002 引擎再加 `BuildingBlocks/physics_RT.dll`），值是 sha256：
  - 目录项（以 `/` 结尾）的值是 `tree_digest`，即每行“文件 sha256 + 空格 + 小写正斜杠相对路径”，按路径排序后整体取 sha256；
  - 缺失的文件记为 64 个 0；
  - 计算方法见 `task.hash_game_files`。
- 不一致时返回 `game_files_mismatch`，指出是哪一项。
- `hello_ok` 返回 `max_instances`（上限 8）、心跳间隔、任务轮询间隔、玩家信息和“上榜”状态。
- 每个实例各自 `GET /pool/task`。任务的全部内容都由服务端决定（第 5 节）。

### 2.3 消息字段（由 `protocol.MESSAGES` 生成）

每条消息还带 `protocol`（= 1）和 `type`（= 消息名）。“否”表示可选字段；不在表里的字段一律拒收。

**`login_start`**（POST /pool/login 请求）

| 字段 | 必需 | 类型 |
|---|---|---|
| `client_version` | 是 | 字符串 1..32 版本号 |

**`login_started`**（POST /pool/login 响应）

| 字段 | 必需 | 类型 |
|---|---|---|
| `login_id` | 是 | 字符串 16..64 `[A-Za-z0-9_-]` |
| `poll_secret` | 是 | 字符串 32..64 `[A-Za-z0-9_-]` |
| `browser_url` | 是 | 字符串 1..500 |
| `user_code` | 是 | 字符串 `XXXX-XXXX`（`[A-Z2-9]`） |
| `expires_in_s` | 是 | 整数 1..3600 |
| `poll_s` | 是 | 整数 1..60 |

**`login_poll`**（POST /pool/login 请求）

| 字段 | 必需 | 类型 |
|---|---|---|
| `login_id` | 是 | 字符串 16..64 `[A-Za-z0-9_-]` |
| `poll_secret` | 是 | 字符串 32..64 `[A-Za-z0-9_-]` |

**`login_status`**（POST /pool/login 响应）

| 字段 | 必需 | 类型 |
|---|---|---|
| `status` | 是 | `pending` \| `ok` \| `refused` \| `expired` |
| `token` | 否 | 字符串 32..128 `[A-Za-z0-9_-]` |
| `user` | 否 | `user` |
| `reason` | 否 | `reason` |

**`hello`**（POST /pool/hello 请求）

| 字段 | 必需 | 类型 |
|---|---|---|
| `client_version` | 是 | 字符串 1..32 版本号 |
| `platform` | 是 | `windows` \| `macos` \| `linux` |
| `engine` | 是 | `engine` |
| `game_files` | 是 | `game_files` |
| `instances` | 是 | 整数 1..64 |

**`hello_ok`**（POST /pool/hello 响应）

| 字段 | 必需 | 类型 |
|---|---|---|
| `session_id` | 是 | 字符串 16..64 `[A-Za-z0-9_-]` |
| `max_instances` | 是 | 整数 1..64 |
| `heartbeat_s` | 是 | 整数 5..3600 |
| `task_poll_s` | 是 | 整数 1..3600 |
| `user` | 是 | `user` |
| `leaderboard` | 是 | 布尔 |
| `server_time` | 是 | 整数 0..1099511627776 |
| `notice` | 否 | 字符串 0..500 |

**`task`**（GET /pool/task 响应）

| 字段 | 必需 | 类型 |
|---|---|---|
| `task` | 是 | `task` |

**`no_task`**（GET /pool/task 响应）

| 字段 | 必需 | 类型 |
|---|---|---|
| `reason` | 是 | `reason` |
| `retry_s` | 是 | 整数 1..3600 |

**`chunk_result`**（POST /pool/chunk 响应）

| 字段 | 必需 | 类型 |
|---|---|---|
| `task_id` | 是 | 字符串 1..64 `[A-Za-z0-9_-]` |
| `seq` | 是 | 整数 0..1000000 |
| `status` | 是 | `accepted` \| `held` \| `rejected` |
| `verify` | 是 | `pending` \| `passed` \| `failed` \| `skipped` |
| `steps` | 是 | 整数 0..4096 |
| `reason` | 否 | `reason` |
| `token_revoked` | 否 | 布尔 |

**`heartbeat`**（POST /pool/heartbeat 请求）

| 字段 | 必需 | 类型 |
|---|---|---|
| `session_id` | 是 | 字符串 16..64 `[A-Za-z0-9_-]` |
| `running` | 是 | 布尔 |
| `instances` | 是 | `instance` 列表（≤ 64） |
| `steps_per_s` | 是 | 数值 0..1e+07 |

**`heartbeat_ok`**（POST /pool/heartbeat 响应）

| 字段 | 必需 | 类型 |
|---|---|---|
| `server_time` | 是 | 整数 0..1099511627776 |
| `max_instances` | 是 | 整数 0..64 |
| `notice` | 否 | 字符串 0..500 |
| `revoked` | 否 | `reason` |

**`me`**（GET / POST /pool/me 响应）

| 字段 | 必需 | 类型 |
|---|---|---|
| `user` | 是 | `user` |
| `leaderboard` | 是 | 布尔 |
| `token_state` | 是 | `active` \| `revoked` |
| `stats` | 是 | `stats` |
| `revoke_reason` | 否 | `reason` |

**`me_update`**（POST /pool/me 请求）

| 字段 | 必需 | 类型 |
|---|---|---|
| `leaderboard` | 是 | 布尔 |

**`leaderboard`**（GET /pool/leaderboard 响应）

| 字段 | 必需 | 类型 |
|---|---|---|
| `updated` | 是 | 整数 0..1099511627776 |
| `entries` | 是 | {rank, nickname, steps_verified} 列表（≤ 200） |

**`error`**（任何接口的出错响应）

| 字段 | 必需 | 类型 |
|---|---|---|
| `reason` | 是 | `reason` |

**共用对象：**

`reason`

| 字段 | 必需 | 类型 |
|---|---|---|
| `code` | 是 | code（37 种） |
| `message` | 是 | 字符串 0..1000 |
| `params` | 是 | 对象 {键: 字符串 / 整数 / 数值 / 布尔}（≤ 16 键） |

`user`

| 字段 | 必需 | 类型 |
|---|---|---|
| `forum_id` | 是 | 整数 1..2^53 |
| `username` | 是 | 字符串 1..64 |
| `nickname` | 是 | 字符串 0..64 |
| `avatar_url` | 否 | 字符串 0..500，可为 null |

`engine`

| 字段 | 必需 | 类型 |
|---|---|---|
| `kind` | 是 | `2002` \| `x87soft` |
| `sha256` | 是 | sha256（64 位小写十六进制） |
| `version` | 是 | 字符串 1..64 版本号 |
| `hello` | 否 | 字符串 0..200 |

`stats`

| 字段 | 必需 | 类型 |
|---|---|---|
| `steps_accepted` | 是 | 整数 0..2^53 |
| `steps_pending` | 是 | 整数 0..2^53 |
| `steps_verified` | 是 | 整数 0..2^53 |
| `chunks_accepted` | 是 | 整数 0..2^53 |
| `chunks_rejected` | 是 | 整数 0..2^53 |
| `last_rejection` | 否 | {code, message, params}，可为 null |

`instance`

| 字段 | 必需 | 类型 |
|---|---|---|
| `slot` | 是 | 整数 0..63 |
| `state` | 是 | `idle` \| `starting` \| `running` \| `paused` \| `error` |
| `steps` | 是 | 整数 0..2^53 |
| `task_id` | 否 | 字符串 1..64 `[A-Za-z0-9_-]`，可为 null |
| `error` | 否 | 字符串 0..300 |

`task`

| 字段 | 必需 | 类型 |
|---|---|---|
| `task_id` | 是 | 字符串 1..64 `[A-Za-z0-9_-]` |
| `kind` | 是 | `sample` \| `eval` |
| `run` | 是 | 字符串 1..64 `[A-Za-z0-9_-]` |
| `policy_sha` | 是 | sha256（64 位小写十六进制） |
| `level` | 是 | 整数 1..13 |
| `record` | 是 | 字符串 1..64 `[A-Za-z0-9_-]` |
| `record_sha256` | 是 | sha256（64 位小写十六进制） |
| `start_frame` | 是 | 整数 0..10000000 |
| `ball_order_seed` | 是 | 整数 0..4294967295 |
| `engine` | 是 | `2002` \| `x87soft` |
| `chunk_steps` | 是 | 整数 1..4096 |
| `max_steps` | 是 | 整数 1..1048576 |
| `max_frames` | 是 | 整数 1..100000000 |
| `deadline_s` | 是 | 整数 1..604800 |
| `issued_at` | 是 | 整数 0..1099511627776 |
| `greedy` | 是 | 布尔 |
| `sample_seed` | 是 | 整数 0..2^53-1（采样任务的动作抽取，见第 10.6 节） |
| `assets` | 是 | {starts, path} |

`game_files`：对象，键正好是 `base.cmo`、`Database.tdb`、`Bin/VxMath.dll`、`3D Entities/`、`Text/`、`BuildingBlocks/physics_RT.dll` 中的名称（x87-soft 引擎不带 physics_RT.dll），值是 sha256。

## 3. 数据段（`POST /pool/chunk` 的请求体）

一个任务就是一个游戏进程：从服务端指定的开局（记录、第 k 帧、球序种子）开始，用指定版本的策略一直玩到下面任一情况：
- 进程结束（`LevelFinish`、`GameOver`、截断）；
- 到达 `max_steps` / `max_frames`；
- 玩家停止。

中途死亡会在同一个进程里重生（按生命分段），所以整个任务可以由“开局 + 种子 + 按键”完整重放。任务按 `chunk_steps` 步切成若干段，编号 `seq` 从 0 开始，按顺序提交。

### 3.1 二进制格式（`protocol.encode_chunk` / `decode_chunk`）

整数一律小端：

```
magic   "BRPC"                   4 字节
format  u16 = 2                  2    （2：哈希链改成引擎状态链 v2，第 3.2 节）
flags   u16 = 0                  2
hlen    u32                      4
header  JSON（UTF-8，规范形式）    hlen 字节，上限 16 KiB
body    各段数组依次拼接            按 header.arrays 的顺序
digest  sha256                   32 字节，覆盖它前面的全部内容
```

**header 的字段（`CHUNK_HEADER`）：**
- `protocol`、`type: "chunk"`、`task_id`、`seq`、`step0`（本段第一步在任务里的序号）、`steps`（1..4096）、`n_obs`、`n_act`、`policy_sha`；
- `chain_prev`：本段第一行之前的哈希链值；
- `final`、`end`：只在最后一段设置，取值为 `LevelFinish` / `GameOver` / `truncated` / `cap` / `stall` / `max_steps` / `max_frames` / `deadline` / `stopped` / `error`；
- 可选的 `stats`（`frames`、`wall_s`、`episodes`）；
- `arrays`：每段 `{name, codec: raw | zstd, nbytes（存储大小）, sha256（解码后的 sha256）}`。

**数组的 dtype 和形状不随数据发送**，由数组名和 `steps` / `n_obs` 决定（`CHUNK_ARRAYS`）。下表中 T = steps：

| 名称 | dtype | 形状 | 必需 | 内容 |
|---|---|---|---|---|
| `actions` | u8 | T | 是 | 动作序号（< n_act），对应清单里的 `actions.keys` |
| `rewards` | f32 | T | 是 | 训练奖励，语义同 `ballrain.train.worker`，参数是清单里的 `rollout` 和 `reward` |
| `logp` | f32 | T | 是 | behaviour log-prob（温度 1），取值在 [-60, 1e-5] |
| `values` | f32 | T+1 | 是 | 每一行观测的价值，最后一行用于自举 |
| `flags` | u8 | T | 是 | bit0 terminated，bit1 truncated；截断只能出现在最后一段的最后一步 |
| `obs` | f16 | (T+1, n_obs) | 是，只能 zstd | 未归一化的原始观测，float16 副本给学习器用；验证器在容差内和自己算的特征比对 |
| `obs_hash` | u8 | (T+1, 32) | 是 | 每一行之后的引擎状态哈希链值（第 3.2 节） |
| `env_rewards` | f32 | T | 否 | `BallanceEnv.step` 本身的奖励 |
| `sector` | u8 | T | 否 | 这一步之后所在的小节 |
| `frames` | u16 | T | 否 | 这一步推进的帧数 |
| `sr_ms` | f64 | T | 否 | 这一步之后的 IGT（毫秒） |
| `arc` | f32 | T | 否 | 这一步之后在参考线上的弧长（NaN 表示没有） |

**解码顺序**（任何一步失败都返回 `bad_chunk` / `chunk_too_large` / `logp_invalid`，不会执行输入里的任何代码）：
1. 总长度；
2. magic 和 format；
3. 末尾的 sha256；
4. header 长度；
5. 严格 JSON；
6. header schema；
7. 段清单：名称已知、不重复、必需段齐全、编码方式允许；
8. 各段大小之和等于 body；
9. 每段解码后的大小必须精确等于预期（zstd 解压时输出上限就是预期大小，防解压炸弹；窗口最大 8 MiB，`window_log_max = 23`），再核对 sha256。raw 段的存储大小必须等于解码大小；zstd 段的存储大小不能超过 `zstd_bound(解码大小) = n + n/128 + 64`（高于 libzstd 的 compressBound），所以不能用空块把一段撑大；
10. 内容：浮点有限、动作小于 n_act、log-prob 在范围内、flags 位合法、最后一段的最后一步必须结束。

### 3.2 引擎状态哈希链（链 v2，2026-11-17 起；原生客户端 M2）

链覆盖的是**引擎发来的原始字节**，不是从它算出来的 float32 特征。原因：志愿者可能用原生 C++ 客户端，或者在另一种 CPU / 编译器上算特征，
float 的舍入会有极小的差别；引擎（x87-soft）本身是确定的，同一组按键给出同样的字节。

- `h(-1) = sha256("ballrain-pool/engine-chain/v2\0" + task_id)`；
- `h(i) = sha256(h(i-1) || d_i)`，`d_i` 是第 i 行的**引擎摘要**：从第 i-1 行形成之后到第 i 行形成为止（第 0 行：从进程启动起，含开局交接和开场）收到的全部引擎消息的规范记录，
  `sha256("ballrain-pool/engine-row/v2\0" + 记录…)`（`ballrain/pool/engine_chain.py`，原生实现 `native/core/src/engine_chain.cpp`）：
  - `"O"` + 一条 OBS1：magic、u16 版本、u16 flags、u32 frames_done、u32 段数，然后每段 u32 tag、u16 版本、u16 flags、u32 长度、内容；**去掉 SCOR、SREN**；
  - `"F"` + level.msg（读到的那一条 LVL1，同样的布局）、`"I"` + 一次读 `.inc` 消费掉的完整增量消息：u32 条数，每条同上，**去掉 IVPO**。
  - 去掉的三段里有地址：SCOR / SREN 的 Energy 第 4 列是参数指针，IVPO 按 IVP 对象地址去重（`headless.md` §5、`native-client.md` §4.1：同一输入跑两次只有这三处不同）。特征不读它们（`hs_block` 关）；HS 策略要先给 SCOR 的跟随计数单独处理。
- 读关卡消息的时刻同 AgentSession（StartLevel 或未加载时读全量，有 PHYS 或 StartLevel 时读增量）；交接检查不通过而重开的进程，摘要从新进程重新开始。

观测是每一步之前的那一行，加上最后一步之后的一行。第 s 段带第 `step0 .. step0+steps` 行（共 steps+1 行），`chain_prev = h(step0-1)`，所以第 s 段的最后一行就是第 s+1 段的第一行（`check_chain_link`）。

验证器重放时链必须逐位复现（引擎确定）；它在特征、log-prob、价值、奖励上**用容差**和提交的值比对（第 10.6 节）。
同一台机器上，原生客户端和 Python 的引擎摘要在 17299 行录制上完全相同，两个 x87-soft 构建（`bin-x86soft`、`bin-x86p2002d` + `BALLRAIN_HL_PHYS2002=0`）也相同；
Mac arm64 / Linux 的引擎构建之间是否逐字节相同还没验证（`native-client.md` §4.1）。

### 3.3 网关收段（原型）

1. 依次检查：令牌、会话、任务归属、期限、`seq` 顺序（`chunk_duplicate` / `chunk_out_of_order`）、`policy_sha` 等于任务指定的版本（`policy_mismatch`）；
2. `decode_chunk(data, n_act=…, n_obs=…)`；
3. 与上一段做 `check_chain_link`（`chain_broken`）；
4. 存盘，响应 `chunk_result`（`status: accepted`，`verify: pending`）。

网关实际还多查几项（第 10.4 节）：第 0 段的 `step0 = 0`、`chain_prev = chain_seed(task_id)`；`step0` 接着已收的步数；非最终段正好 `chunk_steps` 步；总步数不超过 `max_steps`；`n_obs` / `n_act` 与策略一致。

验证器重放（第 10.5 节）：
- 引擎状态链（逐位）、float16 观测（容差内）、flags、`end`、可选数组（sector / frames / sr_ms 逐位，env_rewards / arc 容差内）不一致返回 `replay_mismatch`；
- 奖励超出容差返回 `replay_reward_mismatch`；
- 用 `NumpyPolicy` 重算 log-prob / 价值，不一致返回 `replay_logp_mismatch` / `replay_value_mismatch`；
- 采样任务的动作不是按任务的 `sample_seed` 抽出的，返回 `replay_action_mismatch`。

docs/volunteer-compute.md 第 4 节规定，任何一次不一致都隔离这个令牌的待用数据，并吊销令牌（`token_revoked_replay`）。原型阶段数据不进入训练器。

## 4. 权重（`weights.py`）

服务端用 `export_checkpoint(runtime/m2-train-mac/<run>/ckpt_*.pt, runtime/pool/weights/)` 导出。它只读学习器自己的检查点（本机已有的副本，不连接 Mac），写出两个文件：

- **`<policy_sha>.npz`**：
  - 内容：`pi.{0,2,4}.weight/bias`、`v.{0,2,4}.weight/bias`（float32，torch `nn.Linear` 的布局），以及 `rms_mean`、`rms_var`（float64）；
  - 只有 `.npy` 成员，用 `save_npz_deterministic` 写出：名称排序，zip 时间戳固定，不压缩，所以同样的内容每次导出字节都一样。
- **`<policy_sha>.json`**（清单，格式 `ballrain-pool-weights/1`）：
  - `source`：run、update、steps；
  - `arch`：`mlp_actor_critic`、n_obs、n_act、hidden、activation；
  - `obs_norm`：eps 1e-8、clip 10、count；
  - `actions`：键位、名称、world_actions；
  - `env`：`ENV_FIELDS` 白名单里的 EnvConfig 字段；
  - `engine_options`：HeadlessConfig 里影响模拟的字段，即 SimTool 的 `--tas` 配方；
  - `features`：完整的 FeatureConfig，`path_dir` 置空，`path_sha256` 保留；
  - `reward`：RewardConfig；
  - `rollout`：WorkerSpec 的采样语义，只取 train_m2 实际传给它的参数，其余按默认值；
  - `arrays`：每个数组的 dtype 和形状；
  - 另有 `policy_sha`、`npz_sha256`、`npz_bytes`。
  - 每个值都经过 `assert_public` 检查，不能像本机路径。

**`policy_sha` 是规范化的身份：** sha256 覆盖清单的身份字段（规范 JSON）和各数组（名称、dtype、形状、原始字节）。它不等于 npz 文件的 sha256，文件的 sha256 单独记录，用于下载校验。

**客户端加载：** `load_policy(npz, manifest, expect_sha=task.policy_sha)` 依次检查：
- 文件大小和 sha256；
- 用 `np.load(allow_pickle=False)` 读取，数组名、dtype（只允许 `<f4` / `<f8`）、形状与清单一致，数值有限；
- 重算 `policy_sha`。

**`NumpyPolicy`** 复现 `ppo.normalize` 和 `ActorCritic`：
- `normalize(raw)`、`logits(x)`、`log_probs(x)`、`value(x)`；
- `act(x, rng, greedy, u=None)` 返回（动作，behaviour log-prob，价值）。池子任务传 `u = sampling.sample_u(task.sample_seed, t)`，动作由逆累积分布决定（第 10.6 节）；`rng` 只给不经网关的测试用。随机流和 torch 不同，学习器需要的只是所取动作的 log-prob。

**一致性：**
- 合成网络上，概率最大误差小于 1e-5；
- r6 最新检查点在 SR01 记录的实际观测上（GameTracker + FeatureExtractor 生成），也小于 1e-5（`tests/test_pool_weights.py`）。

r6 当前的权重文件约 0.97 MB。

## 5. 任务与开局素材（`task.py`）

**`Task` 的字段（`protocol.TASK`）：**
- `task_id`；
- `kind`：`sample` 或 `eval`；`eval` 时 `greedy` 为 true；
- `run`、`policy_sha`、`level`；
- `record`、`record_sha256`：开局用的 TAS 记录；
- `start_frame`：k，必须是记录的有效开局帧，并且不在这个 run 的 `exclude_starts` 里（`valid_start_frames`）；0 表示从关卡加载开始；
- `ball_order_seed`：32 位，引擎用 `retail:<seed>`；
- `engine`：`2002` 或 `x87soft`；
- `chunk_steps`、`max_steps`、`max_frames`、`deadline_s`、`issued_at`；
- `assets`：`{starts, path}` 两个 sha256。

**客户端怎么用：**
1. 下载 `GET /pool/assets/<assets.starts>.npz`，用 `install_starts_pack` 校验后整理成 `<cache>/starts/<sha>/<record>/` 记录目录（raw.npz 里只有 record_frames，另有 starts.npz、camera.npz 和精简的 meta.json），`TasStarts` 直接读这个目录；
2. 下载 `GET /pool/assets/<assets.path>.npz`，用 `install_path_asset` 放到 `<cache>/refpaths/<sha>/L<nn>.npz`；
3. 用 `build_env(task, manifest, LocalGame(...), starts_dir, path_dir, out_root)` 得到 BallanceEnv：
   - 配置取清单里的 env / features / reward / engine_options；
   - 开局用任务给的记录目录和第 k 帧；
   - 球序用任务的种子，后端用本机引擎；
   - **关掉所有客户端侧的随机抽取**：没有课程，`p_rest = p_rehearse = 0`；
   - 动作集、观测维度、k 的有效性和清单不一致时，抛出 `TaskMismatch`。

**服务端：**
- `export_starts_pack(数据集记录目录, runtime/pool/assets/)` 导出开局包。数据集自己的 meta.json 含本机路径，不随包分发。
- `find_path_asset(level, manifest.features.path_sha256, [候选目录])` 找到这个 run 实际用的参考线，再用 `export_path_asset` 导出。例如 r6 用的是 `runtime/refpaths-alt/L01.npz`，不是 `benchmarks/refpaths/L01.npz`。

两个包都按文件 sha256 命名，内容不变时重新导出得到同样的 sha。

## 6. 文件布局

| 位置 | 内容 | 进 git |
|---|---|---|
| `ballrain/pool/` | 共享约定（本文） | 是 |
| `tools/pool/` | 网关、验证器、模拟论坛、主控程序的入口 | 是 |
| `tests/test_pool_*.py` | 单元测试 | 是 |
| `runtime/pool/config.toml` | 网关本机配置：`[forum] base_url / client_id / client_secret / redirect_uri`，`[pool] public_base_url / min_account_days = 30 / min_replies = 5 / max_instances = 8`。也可以用环境变量 `BALLRAIN_POOL_FORUM_CLIENT_ID` / `BALLRAIN_POOL_FORUM_CLIENT_SECRET` 设置 | **否** |
| `runtime/pool/weights/<policy_sha>.{npz,json}` | 导出的权重 | 否 |
| `runtime/pool/assets/<sha256>.npz` | 开局包、参考线 | 否 |
| `runtime/pool/chunks/<task_id>/<seq:06d>.brpc` | 收下的数据段，原样保存 | 否 |
| `runtime/pool/state.sqlite3` | 令牌哈希、论坛用户 ID / 用户名 / 昵称 / 头像地址、上榜开关、吊销原因（按论坛账号）、OAuth state 和登录轮询密钥的哈希、会话、任务、数据段的文件名 / 大小 / 验证状态（第 10.3 节） | 否 |
| `runtime/pool/cache/`、`runtime/pool/replay/` | 验证器安装的素材；重放时的 run 目录（重放完即删） | 否 |
| 客户端数据目录 `cache/{weights,starts,refpaths,assets}` | 按 sha 缓存的下载内容 | — |
| 客户端数据目录 `config.json`、`token`、`logs/`、`runs/` | 主控程序的设置、算力池令牌、每个实例的日志、任务进行中的 run 目录（任务结束即删除）（第 9.3 节） | — |

`runtime/` 整个不进 git。已跟踪的文件里不出现密钥、用户名、主机名或 IP。

## 7. 原因码

每个拒收或吊销都带 `{"code", "message", "params"}`。`message` 是中文，由 `reasons.render` 只用列出的参数填入：参数去掉控制字符，每个最长 200 字。主控程序原样显示 `message`。

| code | HTTP | 参数 | 中文说明（主控程序原样显示） |
|---|---|---|---|
| `account_too_new` | 403 | days, min_days | 论坛账号注册未满 {min_days} 天（已注册 {days} 天），暂时不能领取算力池令牌。 |
| `email_not_verified` | 403 | — | 论坛账号的邮箱尚未验证，验证后再登录即可领取令牌。 |
| `too_few_replies` | 403 | min_replies, replies | 论坛账号的回复数不足 {min_replies} 条（当前 {replies} 条），暂时不能领取令牌。 |
| `login_failed` | 401 | detail | 论坛登录失败，请重试。（{detail}） |
| `oauth_state_mismatch` | 401 | — | 登录请求已过期或不匹配，请回到登录页重新开始。 |
| `token_missing` \* | 401 | — | 请求没有带算力池令牌，请先在主控程序里登录。 |
| `token_invalid` \* | 401 | — | 算力池令牌无效，请重新登录。 |
| `token_revoked` \* | 403 | note | 你的算力池令牌已被管理员吊销。说明：{note} |
| `token_revoked_replay` \* | 403 | chunk, step, task_id | 你的算力池令牌已被停用：任务 {task_id} 的第 {chunk} 段数据在第 {step} 步重放结果与提交不一致。这个令牌还没进入训练的数据已全部隔离。 |
| `protocol_unsupported` | 400 | got, supported | 主控程序的协议版本 {got} 不受支持（服务端支持 {supported}），请更新主控程序。 |
| `client_too_old` | 400 | got, min_version | 主控程序版本 {got} 过旧（最低 {min_version}），请更新后再试。 |
| `engine_unknown` | 400 | engine, sha | 无头引擎（{engine}，sha256 {sha}）不是已发布的版本，请重新下载主控程序附带的引擎。 |
| `engine_mismatch` | 400 | got, want | 任务要求 {want} 引擎，而本机运行的是 {got} 引擎。 |
| `game_files_mismatch` | 400 | file, got | 游戏文件校验不通过：{file} 与原版不一致（sha256 {got}）。请检查 Ballance 安装目录。 |
| `too_many_instances` | 400 | max_instances | 同时运行的实例数超过上限 {max_instances}，请在主控程序里调低实例数。 |
| `session_expired` | 409 | — | 会话已失效，主控程序会自动重新连接。（第 2a 步新增：`X-Pool-Session` 缺失、未知、不属于这个令牌，或超过 15 分钟没有活动；主控程序重新发 `hello` 即可） |
| `no_task` | 200 | — | 当前没有可分配的任务，请稍后再试。 |
| `task_unknown` | 404 | task_id | 任务 {task_id} 不存在或已被服务端撤回。 |
| `task_expired` | 410 | deadline_s, task_id | 任务 {task_id} 已超时（期限 {deadline_s} 秒），数据不再接收。 |
| `task_not_yours` | 403 | task_id | 任务 {task_id} 不是分配给这个令牌的。 |
| `chunk_duplicate` | 409 | chunk, task_id | 任务 {task_id} 的第 {chunk} 段数据已经提交过。 |
| `chunk_out_of_order` | 409 | chunk, expected, task_id | 任务 {task_id} 的数据段顺序不对：收到第 {chunk} 段，应为第 {expected} 段。 |
| `stale_policy` | 400 | got, want | 数据使用的策略版本 {got} 已过时（服务端要求 {want}），这段数据未被接收。 |
| `policy_mismatch` | 400 | got, want | 数据使用的策略版本 {got} 与任务指定的 {want} 不一致。 |
| `bad_message` | 400 | detail | 请求格式错误：{detail} |
| `bad_chunk` | 400 | detail | 数据段格式错误：{detail} |
| `chunk_too_large` | 413 | limit, size | 数据段太大（{size} 字节，上限 {limit} 字节）。 |
| `chain_broken` | 400 | chunk, step, task_id | 任务 {task_id} 的第 {chunk} 段观测哈希链在第 {step} 步断开。 |
| `logp_invalid` | 400 | chunk, step, task_id | 任务 {task_id} 的第 {chunk} 段在第 {step} 步的 behaviour log-prob 无效。 |
| `replay_mismatch` | 400 | chunk, step, task_id | 任务 {task_id} 的第 {chunk} 段数据在第 {step} 步重放结果与提交不一致。 |
| `replay_reward_mismatch` | 400 | chunk, step, task_id | 任务 {task_id} 的第 {chunk} 段数据在第 {step} 步重放得到的奖励与提交不一致。 |
| `replay_logp_mismatch` | 400 | chunk, step, task_id | 任务 {task_id} 的第 {chunk} 段数据在第 {step} 步的 log-prob 与策略重算结果不一致。 |
| `replay_value_mismatch` | 400 | chunk, step, task_id | 任务 {task_id} 的第 {chunk} 段数据在第 {step} 步的价值估计与策略重算结果不一致。（第 2a 步新增） |
| `replay_action_mismatch` | 400 | chunk, step, task_id | 任务 {task_id} 的第 {chunk} 段数据在第 {step} 步的动作不是按任务的采样种子从策略抽出的。（第 2c 步新增） |
| `replay_failed` | 202 | task_id | 服务端暂时无法重放任务 {task_id}，数据先保留，稍后再验证（不影响你的令牌）。 |
| `rate_limited` | 429 | retry_s | 请求太频繁，请 {retry_s} 秒后再试。 |
| `server_busy` | 503 | — | 服务端繁忙，请稍后再试。 |
| `internal_error` | 500 | — | 服务端内部错误，请稍后再试。 |

\* 属于 `TOKEN_FATAL`：主控程序收到后停止所有实例，并提示重新登录。参数缺失时显示“?”。

## 8. 已知限制

- **zstd：** 开发环境是 Python 3.14，用标准库的 `compression.zstd`，所以客户端目前需要 Python 3.14。给旧版 Python 准备的 `zstandard` 回退分支（有输出上限的流式读取）没有在本机测试过，因为本机没有装这个包；它也不检查帧后面的多余数据（解码后的 sha256 仍然要对上）。
- **`build_env` 只负责建环境：** `max_steps`、`max_frames`、`deadline_s`、`greedy` 由客户端的采样循环执行。
- **k = 0：** 从关卡加载开始，没有前缀重放。第 1 关的教程跳过（按 Q 直到 CamNavActive）由 env 自己完成，同样的配置重放时会重复同样的操作。
- **跨机器：** 链 v2 不再要求特征逐位相同（第 3.2 节），只要求引擎字节相同；不同引擎构建（Windows x86 / Mac arm64 / Linux）之间的引擎字节还没有核对过。
  容差能吸收的只是舍入；特征里的离散判定（射线擦边、投影重新捕获、stall / cap 的帧数阈值）在另一种 CPU 上翻转时，奖励或结束方式会差出容差，这种情况会被当成不一致，现在只能靠概率很小来接受。
- **奖励语义：** 数据段里的 `rewards` 是 worker 的训练奖励，包括 stall、split、gate 等塑形项，参数取自清单的 `rollout`。客户端的采样循环要和 `ballrain/train/worker.py` 一致，验证器才能在容差内复现（Python 主控程序的 `client_sampler` 和原生客户端的 `native/core/src/task_runner.cpp` 都照 `rollout.TaskRunner` 实现）。
- **只支持 SR：** 原型不支持 Esc / collect 动作（HS 模式），导出时会直接拒绝。

## 9. 主控程序（第 2b 步）

玩家电脑上运行的主控程序：登录、选择游戏目录并校验、设定实例数、开始 / 暂停、显示状态；每个实例是一个采样子进程。自动测试对接的是测试用的桩网关（`tests/test_pool_stub_gateway.py`）；和第 10 节的网关（`gateway.py dev-fake`，模拟论坛）只在本机手动对接过一次（第 9.8 节）。

### 9.1 运行

```sh
uv sync --inexact --extra pool                     # 只为界面装 textual（可选依赖组 pool）；--headless 不需要它
python tools/pool/controller.py                    # 界面（TUI）
python tools/pool/controller.py --headless --start --instances 1 --gateway https://… --ballance-dir … --edition all --engine …
```

- `--edition windows`：Windows 版，2002 headless 宿主，加载玩家自己的 `BuildingBlocks/physics_RT.dll`（必须是 sha256 `e72e4afc…` 那个文件），引擎类型 `2002`。
- `--edition all`：全平台版，x87-soft headless，引擎类型 `x87soft`。Windows 上用 `bin-x86soft` 构建；macOS / Linux 的构建由 `headless/` 的脚本产出，主控程序只需要 `--engine` 给出路径。macOS / Linux 二进制不在原型范围内，这两个平台上主控程序没有实际跑过。
- 引擎文件：默认找主控程序附带的 `engine/<版本>/<平台>/BallanceMMOSimTool[.exe]`；源码目录里没有这个目录，用 `--engine` 指定（本机训练用的是 `runtime/hl-win/bin-x86p2002d` 和 `bin-x86soft`）。
- 网关地址不写进代码：首次运行时填写，或者用 `--gateway` / 环境变量 `BALLRAIN_POOL_GATEWAY`。只接受 https；http 只允许本机地址（测试网关）。
- 其他选项：`--data-dir`、`--status-file`（每秒写一次状态快照 JSON）、`--duration`、`--token-stdin`、`--login`（无界面登录）、`--leaderboard on|off`、`--no-auto-pause`。`--simulate`、`--fake-game` 和 `--dev-hello <dev.json>`（开发网关给的引擎和游戏文件哈希）只给测试和开发用。

### 9.2 界面

- **首次运行**（缺网关地址、令牌，或者游戏目录 / 引擎有问题时显示）：
  - 网关地址；
  - “在浏览器中登录论坛”：发 `login_start`，打开 `browser_url`，按 `poll_s` 轮询 `login_poll`，成功后自动保存令牌；也可以把令牌粘贴进来；
  - Ballance 安装目录、版本、引擎文件，“校验游戏文件”逐项列出问题。
- **主界面**：
  - 开始 / 暂停按钮（空格键）；
  - 实例数（− / + 或 −、+ 键，默认核心数的一半，最多 8，网关给的 `max_instances` 更小时以它为准）；
  - “上榜”开关；
  - 汇总行：玩家昵称、状态、本次贡献的步数和已被接收的步数、累计已接收和已通过验证的步数（`GET /pool/me`，每 60 秒一次）、步/秒；
  - 实例表：状态、当前任务、本任务步数、累计步数、已被接收、步/秒、说明；
  - 日志区。
- **横幅**：网关拒收或吊销时，在主界面顶部显示 `reason.message`，**原样显示**，不按原因码重新生成，也不解释其中的标记（`Static(markup=False)`）。吊销时横幅是红色，并标“令牌已停用”。自动暂停的原因也显示在这里。

本机能判断的只有两类问题，提示里会写明是哪个文件（`client_game.check_game_dir`）：
- 缺文件或文件夹，例如“缺少文件：base.cmo。游戏文件不完整，请检查安装目录。”；文件夹为空也会提示；
- Windows 版的 `physics_RT.dll` 不是 2002 原版：提示本机文件和应有文件的 sha256 前 12 位，并建议换回原版或改用全平台版。

其余文件是否为原版由网关在 `hello` 时核对，不一致时返回 `game_files_mismatch`，主控程序原样显示。本机不内置其他文件的期望哈希。`hello` 只上传哈希，不上传路径。

### 9.3 模块

| 文件 | 内容 |
|---|---|
| `ballrain/pool/client_api.py` | `GatewayClient`：只用标准库 HTTP，响应有大小上限，并按 `protocol.MESSAGES` 校验。<br>出错时抛两种异常：`GatewayRefusal`（`.message` 原样显示，`.fatal` = 原因码属于 `TOKEN_FATAL`）和 `GatewayUnavailable`。<br>`Cache` 按 sha 缓存权重、开局包和参考线，先校验再用 |
| `ballrain/pool/client_game.py` | 两个版本、游戏目录校验（中文提示）、引擎文件检查，以及 `hello.engine`：类型、文件 sha256、版本号。版本号取二进制旁边的 `ENGINE_VERSION`，没有就用 `dev-<sha 前 12 位>` |
| `ballrain/pool/client_lifecycle.py` | 子进程组 `ChildGroup`（Windows 用 Job Object；Linux 用新会话加父进程退出信号）、子进程端的 `ParentWatch`、低优先级，以及检测 Player.exe / 电池的函数 |
| `ballrain/pool/client_sampler.py` | 采样循环 `TaskSampler.run_task`、`ChunkBuilder`（跨段的哈希链）、后台上传 `Uploader`，以及两种动作来源 `PolicyActions` / `ReplayActions` |
| `ballrain/pool/client_worker.py` | 实例子进程：领任务 → 下载权重和素材并校验 → `build_env` → 采样 → 上传 |
| `ballrain/pool/client_core.py` | 不含界面的核心 `Controller`：配置和令牌、校验、会话（hello、心跳、`me`、上榜）、实例管理、自动暂停、横幅、`snapshot()` |
| `tools/pool/controller.py` | 入口：`--headless` 驱动和 Textual 界面（`build_app`） |

**客户端数据目录：**
- 位置：Windows 是 `%LOCALAPPDATA%\BallrainPool`，macOS 是 `~/Library/Application Support/BallrainPool`，Linux 是 `$XDG_DATA_HOME/ballrain-pool`；
- 内容：`config.json`（设置）、`token`（权限 0600）、`cache/`、`logs/instance-NN.log`（实例的 stderr）、`runs/`（任务进行中的 run 目录，任务结束就删除）。

### 9.4 实例进程

每个实例由 `python -m ballrain.pool.client_worker` 启动。它和主控程序之间用 JSON 行通信，不用 pickle，也不用 multiprocessing：
- **主控 → 实例（stdin）：**
  - 第一行是配置：网关、令牌、会话 id、版本、引擎、游戏目录、数据目录；
  - 之后是命令：`run`、`pause`、`stop`，以及重新连接后发的 `token`。
  - 令牌只走 stdin，不出现在命令行里。
- **实例 → 主控（stdout）：** `hello`、`state`、`engine`（引擎 pid）、`progress`、`chunk`（每段的 `chunk_result`）、`refusal`（原因对象 + 是否致命）、`log`。

实例收到配置之前不启动任何东西。领到任务后，它依次做这几件事：
1. 下载权重清单和 npz，`load_policy` 校验 sha；
2. 下载开局包和参考线，`install_*` 校验；
3. `build_env`；
4. 用 `NumpyPolicy` 采样：温度 1，评估任务用 argmax；
5. 每 `chunk_steps` 步编码一段，交给后台线程按顺序上传：网络错误时退避重试；重试后收到 `chunk_duplicate` 就当作已接收；收到拒收就停止这个任务。

领任务和上传由各实例自己做；心跳由主控程序统一发，因为 `heartbeat` 消息本来就带全部实例的状态。

**暂停的含义：**
- 正在跑的任务在下一步结束：最后一步标截断，`end = "stopped"`，最后一段照常交出；
- 引擎随即退出，实例进程留着，等下一次“开始”；
- 关闭主控程序、减少实例数时，实例直接退出，不交最后一段，任务留给网关按期限过期。

**Windows 上的一个坑：** 一个线程在同步读管道（这里是 stdin）时，同一个文件对象上的其他调用都会被挡住，其中就包括 DLL 加载时 C 运行库调用的 `GetFileType(stdin)`。结果是实例卡在第一次 `import numpy`。所以实例先用无缓冲的方式读完配置行，再把管道换到一个复制出来的句柄上读，标准输入改成 NUL（`client_worker._take_stdin`）。

### 9.5 采样循环与奖励

一个任务就是一个游戏进程（第 3 节）。在这个前提下，`TaskSampler.run_task` 按下面的规则处理每一步：

- **训练奖励的语义就是学习器的。** `TaskSampler` 继承 `ballrain.train.worker.Worker`，只为了原样复用它的辅助方法：`_Episode`、帧上限、小节分段奖励、TAS 参考时间、run 清理。每步的处理是从 `Worker.run` 照搬过来的：分段奖励、门、路线探针、`cap` / `stall` 截断、`stall_terminal` 罚分。`WorkerSpec` 的字段取自清单的 `rollout`。全程不用 torch。
- **死亡后还有命：** 用 `env.reset(options={"relaunch": False})` 在同一进程里重生。重生快进时进程如果结束了（GameOver），任务就在这里结束。这一步不会开第二个进程。
- **进程结束就是任务结束：**
  - 任何截断（env 自己的，或 worker 的 `cap` / `stall`）；
  - 任何终止：通关（LevelFinish）、GameOver、`stall_terminal`、视角保护（`camera_off_quarter`）。
- **任务层的上限：** 到达 `max_steps`、`max_frames`（从交接起的游戏帧数）、本地期限（收到任务时起算，`deadline_s` 减去最多 30 秒的余量），或者玩家暂停 / 停止。这时最后一步标截断，`end` 依次是 `max_steps` / `max_frames` / `deadline` / `stopped`。
- **每步结束时、关段之前就判定是否结束。** 所以任何一段的最后一步都可能成为最终段的最后一步，截断标志只会出现在最终段的最后一步。
- **`end` 的取值：** 照上面写；`env` 用其他方式结束进程时一律记成 `truncated`，包括 PeerGone / 超时截断、`max_episode_frames`、`camera_off_quarter`。这一步是终止还是截断，由 flags 区分。`camera_off_quarter` 没有加进 `END_REASONS`，协议没有改动。
- **引擎在任务中途出错：** 已缓存的最后一步标截断，`end = "error"`。如果这一段还没有任何一步，任务直接放弃，留给网关按期限过期。

`ReplayActions` 可以按记录的动作重放同一任务，验证器以后可以直接用这条代码路径。

**对照测试**（`tests/test_pool_sampler.py`，假游戏）：同一串强制动作分别交给 `ballrain.train.worker.Worker` 和 `TaskSampler`，比较训练奖励、终止和截断标志，要求**逐位相同**。测了三种情况：
- 死亡、重生直到 GameOver；
- 静止导致 `stall_terminal`；
- 跨小节（有分段奖励）直到通关。

同一任务按记录的动作重放，哈希链和奖励也逐位相同。

### 9.6 进程生命周期

关闭主控程序时，它启动的所有模拟器和组件都必须一起结束（docs/volunteer-compute.md 11.4）。正常退出、Ctrl+C / Ctrl+Break、关掉终端窗口、崩溃、被强行结束，都是这样。

- **Windows：** 一个 Job Object，设置 `KILL_ON_JOB_CLOSE`，另外设低于正常的优先级、子进程崩溃时不弹错误对话框。
  - 每个实例以挂起状态创建，加入 job 后再恢复（`NtResumeProcess`），所以它启动的任何东西，包括引擎，都在 job 里。
  - job 句柄不可继承。主控进程一结束，系统就关闭这个句柄，job 里的进程全部被终止。
  - venv 的 `python.exe` 只是转发器，所以实例直接用基础解释器加 `__PYVENV_LAUNCHER__` 启动，和 multiprocessing 的做法相同。这样主控程序拿到的 pid 就是真正的实例进程。
  - 关闭控制台窗口时，`SetConsoleCtrlHandler` 先做一次正常停止。
- **Linux：** 实例在新会话里（自己的进程组，引擎也在这个组里），设置 `PR_SET_PDEATHSIG(SIGTERM)`。实例收到 SIGTERM 后杀掉整个进程组。另有父进程监视线程兜底。
- **macOS 及其他：** 实例监视父进程（stdin EOF、`getppid()` 变化），父进程没了就杀掉自己的进程组。
- **不按映像名杀进程。** 主控程序只结束自己启动的进程。检测 Player.exe 只读进程列表，从不碰那个进程。

**测试（本机 Windows）：**
- 无界面的主控程序跑 1 个实例（实例下面挂一个替身引擎子进程），用 TerminateProcess 强杀主控进程，断言 5 秒内实例和替身引擎都不在了；
- `--duration` 正常退出、Ctrl+Break 也各测一次。
- 用真实的 2002 引擎手动做同样的强杀：r6 权重，SR01 第 8587 帧开局，任务跑到一半时强杀主控，实例和引擎在 0.1 秒内都不在了。

### 9.7 自动暂停、上榜、拒收与吊销

- **自动暂停：** 每 5 秒检查一次，Player.exe 在运行（玩家在玩）或使用电池时暂停，条件解除后自动继续。这只影响“正在采样”，不改变玩家按下的“开始”。
  - 检测方法：Windows 用 Toolhelp 进程列表和 `GetSystemPowerStatus`；Linux 读 `/proc` 和 `/sys/class/power_supply`；macOS 用 `ps` 和 `pmset`。都是尽力而为。
  - 两项都可以在 `config.json` 里关掉；`--no-auto-pause` 全部关掉。本机有零售 Player.exe 拷贝在跑训练，所以测试都关掉了自动暂停，或者换成注入的检测函数。
- **上榜：** 以网关的状态为准。连接时读 `hello_ok.leaderboard`（网关默认开），之后每 60 秒从 `GET /pool/me` 同步一次；只有玩家点开关时才发 `POST /pool/me`，界面显示网关返回的值。所以重启主控程序不会把玩家关掉的开关重新打开（有测试）。
- **拒收与吊销：** 下面这些都会显示横幅：
  - `heartbeat_ok.revoked`；
  - `chunk_result.token_revoked`；
  - `chunk_result` 的 `rejected`；
  - 任何接口返回的 `error`；
  - `me.token_state = "revoked"`。
  - 原因码属于 `TOKEN_FATAL` 时，所有实例停止，令牌标记为失效，直到玩家重新登录或粘贴新令牌。
  - 其他拒收只结束对应的任务，横幅保留。

### 9.8 测试

| 文件 | 内容 |
|---|---|
| `tests/test_pool_stub_gateway.py` | 桩网关（`http.server`，按协议收发，存下收到的数据段，并像第 3.3 节那样检查顺序和哈希链），以及它的自测 |
| `tests/test_pool_sampler.py` | 假游戏上的采样：数据段能解码、能接上链，死亡后在同一进程重生，GameOver / 截断 / `stopped` / `max_steps` 的处理，重放逐位相同，与学习器 worker 的奖励对照（三种情况） |
| `tests/test_pool_controller.py` | 游戏目录提示、网关拒收原样显示、`hello` 只带哈希、上榜开关往返、吊销横幅（心跳、`me`、实例上报三条路径）、真实子进程的开始 / 暂停和自动暂停、强杀 / 正常退出 / Ctrl+Break 的生命周期、真实实例进程在假游戏上完整跑一个任务，以及拒收时吊销横幅；另外检查实例进程不导入 torch |
| `tests/test_pool_tui.py` | 用 Textual 的 pilot 无界面驱动：首次运行页列出缺失文件；主界面的开始 / 暂停、上榜开关（只在玩家点击时发一次 POST）、吊销横幅原样显示（不解释标记） |

**本机手动实测**（桩网关，r6 权重，SR01 记录，1 个实例）：
- **全平台版**（`bin-x86soft`，从第 20187 帧开局）：两个任务都通关，各 114 / 120 步；
- **Windows 版**（`bin-x86p2002d`，从第 17587 帧开局）：两个任务都通关，各 1190 / 1210 步，共 10 段，约 250 步/秒；
- `hello` 的 `game_files` 在 Windows 版带 `physics_RT.dll`，全平台版不带；
- 停止后引擎都不在了。

**与第 10 节网关对接**（本机手动一次）：`gateway.py dev-fake`，用本机端口，状态目录放在临时目录；模拟论坛用户 `eligible`；主控程序用 `--fake-game`，`--dev-hello` 读网关的 `dev.json`，1 个实例。
- 登录：浏览器那一步用带 cookie 的 urllib 代替，结果领到令牌；
- `hello` 通过，上榜开关关、开各一次，网关返回的状态正确；
- 1 个实例约 8 秒做完 6 个任务：4 个 GameOver（游戏结束），2 个停滞（`stall`）；共 42 段，没有被拒收；
- 验证器重放逐位通过了已完成的任务，网关的 `me` 显示已通过验证 1409 步。
- 顺带修了一处：登录轮询遇到 `rate_limited` 会按 `retry_s` 等待再试，不再报错。

### 9.9 还没做 / 限制

- **网关：** 自动测试只对接桩网关；和第 10 节网关的对接（模拟论坛的 OAuth、hello、上榜、采样、验证器重放）只在本机手动跑过一次，没有写成测试。真实论坛的 OAuth 没有测过。
- **macOS / Linux：** 这两个平台上的生命周期代码（父进程退出信号、进程组）和自动暂停检测都没有实际运行过；这两个平台的引擎二进制也不在原型范围内。
- **安装包：** 没有绿色包，也没有内嵌 Python；客户端需要 Python 3.14（标准库 zstd，见第 8 节）。启动时的 ed25519 签名更新检查也没有做。
- **测试依赖：** 采样测试依赖假游戏的数据集构建，只能在 Windows 上跑；还依赖 torch，用来像学习器那样导出一个合成策略。
- **任务期限：** 期限按本机收到任务的时刻计算，不用服务端时间。
- **上传失败：** 一段数据重试多次仍然上传不了时，任务后面的数据段不再上传，这个任务留给网关按期限过期。

## 10. 网关与验证器（第 2a 步）

收下的数据段只存盘、计数、验证，**不接入任何训练器**（r6 不受影响）。网关只读 r6 在本机的副本里的检查点，不连接 Mac。

### 10.1 运行

```
python tools/pool/gateway.py serve [--config runtime/pool/config.toml] [--port 8099] [--no-verify]
python tools/pool/verifier.py [--once]            # 验证器单独一个进程（serve 默认在线程里跑它）
python tools/pool/gateway.py dev-fake --dir <临时目录> [--port 8099] [--forum-port 8098]
python tools/pool/gateway.py admin list | revoke --forum-id N --note "说明" | unrevoke --forum-id N
python tools/pool/mock_forum.py --port 8098 --redirect-uri http://127.0.0.1:8099/pool/oauth/callback
```

- 只绑定本机回环地址，纯 HTTP。HTTPS 以后由隧道加，**这一步没有加进隧道**。
- `dev-fake`：假游戏关卡、合成策略、模拟论坛全放在 `--dir` 下，验证器用 FakeLauncher。主控程序需要的引擎 sha256 和 `game_files` 写在 `<dir>/dev.json`。
- 没有新增依赖：`http.server`、`sqlite3`、`urllib`、`tomllib` 都是标准库。

### 10.2 配置（`runtime/pool/config.toml`，不进 git）

键名见 `ballrain/pool/server_config.py` 的说明。补充约定：
- `[pool] public_base_url` 是站点根地址，末尾的 `/pool` 会去掉（接口路径本身以 `/pool/` 开头）；`redirect_uri` 默认是它加 `/pool/oauth/callback`。
- `[policy] ckpt_dir` 默认 `runtime/m2-train-mac/<run>`，`datasets_dir` 默认 `runtime/datasets/tas-v1`，`path_dirs` 默认 `benchmarks/refpaths` 和 `runtime/refpaths-alt`。
- 没有 `[engines]` 一节时，用本机的 `runtime/hl-win/bin-x86p2002d`（2002）和 `bin-x86soft`（x87soft），游戏目录 `runtime/hl-win/game-min`。允许的引擎 sha256 默认就是这两个二进制的 sha256；`game_files` 的参照值是 `task.hash_game_files(game-min)`。
- 论坛应用 ID / 密钥也可以用环境变量 `BALLRAIN_POOL_FORUM_CLIENT_ID` / `BALLRAIN_POOL_FORUM_CLIENT_SECRET`，环境变量优先。

### 10.3 登录与令牌（`server_oauth.py`、`server_app.py`）

- **两种登录方式，结果一样：**
  - 主控程序：`login_start` → 浏览器打开 `browser_url` → `login_poll` 取到令牌（第 2.1 节）。回调页只写“登录成功，可以回到主控程序”，不显示令牌。
  - 只用浏览器：`GET /pool/login` 是带链接的说明页，点 `?go=1` 去论坛；回调页把令牌**只显示这一次**，并说明怎么粘贴到主控程序。
- **state：** 32 字节随机数，只能用一次，10 分钟过期，另外绑定一个浏览器 cookie（`pool_oauth`，HttpOnly、SameSite=Lax、Path=/pool/oauth/callback，https 下加 Secure）。别的浏览器拿到回调地址也用不了。
- **换令牌和读用户都在服务端：** `POST /oauth/token`（授权码、精确的 redirect_uri、client_id、client_secret），再 `GET /api/user`。`ForumClient.user` 只取 `id`、`username`、`nickname`、`avatar_url`、`is_email_confirmed`、`joined_at`、`comment_count`，`email` 在解析时就丢掉；论坛令牌用完即弃。论坛的响应体不进日志，也不进异常信息。
- **门槛**按固定顺序检查：注册天数（`account_too_new`）→ 邮箱（`email_not_verified`）→ 回复数（`too_few_replies`）。论坛返回的字段缺失或格式不对时是 `login_failed`。昵称去掉控制字符、截到 64 字；为空时用用户名。
- **令牌：** `bp1_` 加 32 字节随机数（base64url）。sqlite 里只存 sha256；明文只在内存里放到 `login_poll` 取走为止（最多 10 分钟，网关重启就丢，重新登录即可）。每个论坛账号只有一个有效令牌，重新登录换新令牌，旧令牌返回 `token_invalid`。
- **吊销按论坛账号记录**，带存下来的原因。被吊销的账号重新登录会被拒绝，并显示同一条原因，所以换令牌绕不过去。`GET /pool/me` 和心跳仍然应答（`token_state: revoked` 加 `revoke_reason`；心跳带 `revoked`、`max_instances = 0`），其余接口返回 403 和这条原因。管理员吊销用 `admin revoke`（`token_revoked`，`note` 原样显示）。
- **页面：** 中文，所有插入的文字都 `html.escape`；`Content-Security-Policy: default-src 'none'`、`Cache-Control: no-store`、`Referrer-Policy: no-referrer`（回调地址里的授权码不会经 Referer 泄露）。

### 10.4 接口实现

- **认证顺序：** 令牌（缺失 / 格式错 / 无效 / 被吊销）→ 按账号限速 → 会话（`X-Pool-Session`，必须属于这个令牌、15 分钟内有活动，否则 `session_expired`）。
- **`hello`：** 客户端版本（`client_too_old`）、引擎 kind 和 sha256（`engine_unknown`）、`game_files` 逐项与参照比对（缺项或不同都是 `game_files_mismatch`，指出是哪一项）、实例数上限 8（`too_many_instances`）。
- **`task`：** 每个账号同时最多 8 个未结束的任务，跨所有会话计算（`too_many_instances`）。待验证数据（状态 `held`）超过每账号 512 MiB 或总量 20 GiB 时返回 `no_task`（`server_busy`）；已放行、拒收、隔离的段不算在内。没有发布的策略时返回 `no_task`。
- **`chunk`：** 解码（耗时的部分）在锁外；之后依次检查任务存在（`task_unknown`）、归属（`task_not_yours`）、重复（`chunk_duplicate`）、已结束或过期（`chunk_out_of_order` / `task_expired`）、乱序（`chunk_out_of_order`）、策略版本（`policy_mismatch`）、`n_obs` / `n_act`、`step0` 接续（`chain_broken`）、段长（非最终段正好 `chunk_steps` 步）、`max_steps`、第 0 段的链起点，以及与上一段的 `check_chain_link`（`chain_broken`）。通过后原样写到 `chunks/<task_id>/<seq:06d>.brpc`，记一行，并决定是否抽中验证（第 10.6 节）。同一时间只收一段，所以同一 `seq` 并发提交时只有第一份生效。另外：解码前先查这个账号待验证数据的上限（`server_busy`）；请求体不能超过这个任务尺寸下诚实编码的上限 `protocol.max_chunk_bytes(chunk_steps, n_obs)`（`chunk_too_large`）；记行时在同一个 `BEGIN IMMEDIATE` 事务里重新检查任务仍是 `open`、未过期、`next_seq` 正好是这一段、账号没被吊销（验证器可能刚关了任务或吊销了账号），不满足就按 `task_expired` / `chunk_out_of_order` / `chunk_duplicate` / 存下的吊销原因拒收，并删掉已写的文件。
- **大小和格式：** 先看 `Content-Length` 再读请求体。数据段上限 4 MiB（413 `chunk_too_large`），`hello` / `heartbeat` / `me` 16 KiB，`login` 4 KiB，GET 不收请求体。不接受 chunked（411）；没有 `Content-Length` 的 POST 也是 411；请求目标最长 2048 字节（414）；套接字超时 30 秒；同时最多处理 32 个请求（503 `server_busy`）。带令牌的接口在**读请求体之前**先查令牌（格式、是否存在、失败限速），不通过就回应并关闭连接，所以匿名客户端不能让网关缓存请求体。最多 128 个连接线程（多出的连接直接收到固定的 503 并关闭）；看门狗把单个请求（连同读请求体、写响应）超过 120 秒的连接关掉，防止一个字节一个字节地拖。权重和素材文件按 64 KiB 分块发送。解析不了的请求目标回 400；连接级的异常只记一行，不带地址和回溯。
- **限速**（令牌桶，键只在内存里，取哈希）：
  - `login_start`、登录 / 确认页和回调：每个地址每分钟约 10 次，突发 20（一次登录要用 4 次：开始、确认页、继续、回调）；
  - 主控程序的 `login_poll` 用自己的桶：先按地址（5 次/秒，突发 50），再按 `login_id`（1 次/秒，突发 10），不消耗上一条的额度（主控程序每 2 秒问一次，浏览器的回调不会因此被限速）；
  - 公开接口：每个地址 5 次/秒（突发 60）；
  - 令牌校验失败：每个地址每分钟约 10 次；
  - 带令牌的接口：每个账号 10 次/秒；数据段每个账号 10 段/秒（突发 30）。
  - 超出时返回 429 和 `Retry-After`。地址取自连接；`trust_forwarded = true` 时取 `CF-Connecting-IP`。`public_base_url` 是 https（即在隧道后面）而没有写 `trust_forwarded = true` 时，网关**拒绝启动**：隧道后面所有请求都来自 127.0.0.1，一个人就能用完所有人的额度。
- **日志不记**客户端地址、查询串（回调里的 code / state）、令牌和邮箱。

### 10.5 任务来源（`server_tasks.py`）

- `PolicySource.refresh()` 取 `ckpt_dir` 里最新的可读检查点。最新的一个可能还在拷贝中，读不了就往前试，最多 3 个。检查点用 `weights.load_trusted_checkpoint` 读（`torch.load` 仍然只在 `weights.py` 里），导出权重和两个素材，并取**同一个检查点**的课程状态。之后每 `refresh_s`（默认 900 秒）检查一次；旧版本的权重留在磁盘上，进行中的任务照常验证。
- **开局帧 k 的抽法和训练器相同：** 在去掉 `exclude_starts` 的有效开局帧上建 `ReverseCurriculum`，载入这个检查点的 `frontier`、`history`、`bin_fail`，再调用 `sample`（前沿带、按失败率加权的分箱、到达关卡开头后的 k = 0）。随机数生成器的种子来自 `secrets`。训练器的静止开局、复习开局不用。球序种子每个任务重新抽 32 位。
- `chunk_steps` 取清单的 `rollout.chunk`（r6 是 256）；`max_steps` / `max_frames` / `deadline_s` 取配置（默认 8192 / 60000 / 3600）。

### 10.6 验证器（`server_verify.py`、`rollout.py`）

- **抽查在收段时决定**（`chunks.selected`）：试用期内每段都抽中，之后每段 10%（`secrets`）。试用期**按步数**算：账号待验证和已通过的步数（拒收的不算）达到 `probation_chunks × chunk_steps`（200 × 256）之前都在试用期，所以用一堆 1 步就停的任务刷不掉。客户端不知道哪段被抽中，`chunk_result` 一律是 `verify: pending`。
- **动作抽取是确定的**（`ballrain/pool/sampling.py`）：每个任务带服务端随机生成的 `sample_seed`（53 位）。第 t 步（整个任务从 0 数）的均匀数 `u_t = sample_u(sample_seed, t)`（splitmix64），动作取策略累积分布在 `u_t` 处的逆（和 `NumpyPolicy.act` 的抽法一样）。重放用自己算出的概率检查提交的动作：只有 `u_t` 离某个累积分布边界不到 1e-4（容忍不同 CPU 的舍入）时才放过不一致，否则 `replay_action_mismatch`。这同时补上了两个口子：客户端自己挑动作再如实报 log-prob；把缓存的轨迹换个 `task_id` 重签后交给另一个开局相同的任务（球序在第 1 关只有 48 类，课程到关卡开头后 k 都是 0）。贪心（eval）任务仍然要求 argmax。分布仍是策略本身（温度 1），学习器照样拿到 behaviour log-prob。
- **按任务重放。** 任务结束（收到最终段）或过了期限之后，验证器从任务开局（记录、第 k 帧、球序种子）开始，用提交的动作一直重放到被抽中的最后一段。第 s 段只能和它前面的所有段一起重放。重放到的段要么通过，要么在第一个不一致的段拒收；后面没重放的段记为 `skipped`，按抽查规则放行。
- **“通过验证的步数”**（`me.stats.steps_verified`、排行榜）= 重放通过的段 + 抽查放行的段。被拒收和被隔离的段不算。
- **重放代码：** `rollout.TaskRunner` 在 `task.build_env` 建的环境里逐步执行。奖励塑形直接复用 `ballrain.train.worker` 的 `_Episode` 和 `Worker` 的辅助方法（绑定到一个替身对象上，`worker.py` 没有改动）。任务层的判定与 `client_sampler.TaskSampler.run_task` 逐项一致，包括 `end` 名称、上限先于重生判定、`max_frames` 从交接起计数、`stopped` / `deadline` 在重生之前判定、可选数组的取值。`tests/test_pool_verifier.py` 用主控程序的采样器产生数据段，再由验证器重放，要求全部通过。
- **比较顺序：** 先比步号，最早的一步胜出。同一步内依次比较：
  1. 引擎状态链（逐位，`engine state`）；float16 观测和验证器自己算的特征比，每个分量允许一个 float16 ulp 加 `OBS_TOL`（1e-4）× max(1, |值|）（`observation`）；
  2. 价值、log-prob（容差 1e-3，log-prob 先截到 [-60, 0]，和客户端一样；都用验证器自己算的特征）、贪心任务是否取了 argmax、采样任务的动作是否是 `sample_seed` 的抽取（`u_t` 离累积分布边界不到 1e-4 时放过）；
  3. 执行这一步，再比训练奖励（`REWARD_TOL` 1e-4 × max(1, |r|)）、flags、`end`、可选数组（sector / frames / sr_ms 逐位，env_rewards 同奖励的容差，arc `ARC_TOL` 1e-3 × max(1, |arc|)，两边都是 NaN 算相同）。
- **客户端声明的结束：** `stopped` / `deadline` 只能出现在最终段的最后一步，重放也在那一步停下（不重生）。`error` 表示下一步引擎出错，所以这最后一步照常执行，flags 允许从 0 变成截断。
- **引擎故障不是不一致：** `BallanceEnv.step` 自己捕获 `PeerGone` / `Timeout`，设 `info["error"]` 后正常返回一个截断步。两边都在 `env.step` 之后、`ep.step` 之前检查 `info["error"]`：主控程序丢掉这半步（重生时引擎出错也一样，丢掉死亡那一步），在上一步以 `error` 结束最终段，这一段还没有步时放弃任务（`TaskAborted`，已交的段到期后照常验证）；验证器抛 `EngineFailure`，记 `replay_failed` 以后重试。重生时引擎断开会抛异常而不是返回 `None`（`client.run_until` 把 `PeerGone` 往上抛），所以不会被当成 `GameOver`。
- **不一致时：**
  - 这一段拒收。原因（`replay_mismatch` / `replay_reward_mismatch` / `replay_value_mismatch` / `replay_logp_mismatch` / `replay_action_mismatch`，带任务、段号、步号）存进这一段，也就是 `me.stats.last_rejection`；
  - 这个账号所有没验证过的段（待验证的和抽查放行的）都隔离；
  - 账号吊销，原因 `token_revoked_replay`，参数是同样的任务、段号、步号。主控程序原样显示，例如“任务 … 的第 1 段数据在第 266 步重放结果与提交不一致”。
- **验证器自己的问题**（引擎起不来、交接失败、崩溃、缺素材）是 `replay_failed`：数据保留，按 300 秒 × 次数退避重试；5 次之后这些段停在 `replay_failed`，不算通过，也不算玩家的错。
- 同一时间只有一个引擎进程；重放的 run 目录（`runtime/pool/replay/<task_id>`）重放完就删除。
- 验证器只处理它读到的那一批段（`finish_verify` / `skip_task` 带 `max_seq`）；处理已吊销账号的任务时，把它还在 `held` 的段也隔离。

### 10.7 测试与实测

| 文件 | 内容 |
|---|---|
| `tests/test_pool_gateway.py` | 模拟论坛上的 OAuth：合格账号拿到令牌，三种不合格账号的原因码和参数，门槛顺序，论坛字段格式不对；state 只能用一次、必须带 cookie、拒绝授权、错误的应用密钥；模拟论坛本身要求精确的 redirect_uri、不允许隐式授权、必须带 state、授权码只能用一次；只用浏览器登录时令牌只显示一次；令牌轮换，管理员吊销后各接口的应答，重新登录被拒；`hello` 的各项检查；乱码、超大请求、错误的方法和路径；限速；真实 HTTP 服务器按 `Content-Length` 拒收 5 MiB 的数据段、拒收 chunked、日志里没有地址和查询串；sqlite 和输出里没有邮箱和令牌明文；配置文件和环境变量；`runtime/pool` 不进 git；r6 的策略导出和课程抽样（没有 r6 时跳过） |
| `tests/test_pool_verifier.py` | 假游戏上端到端：采样和重放逐位相同；诚实的数据段被收下并通过验证，`me` 和排行榜的步数；重复、跳段、最终段之后再交、别人的任务、别的策略、链起点不对、链断开、未知任务、8 个任务的上限；**伪造的数据段**（改奖励、改观测、改引擎状态并重算链、改动作、改 log-prob、改价值、改 flags）都被抓到；另一种 CPU 级别的浮点噪声（所有浮点数组都加小扰动）照常通过，原因码和步号正确，账号吊销，其他任务的数据被隔离，原因原样显示，从排行榜上消失；过了试用期后的抽查和放行；验证器引擎故障不吊销；过期任务只验证已到的段；主控程序的采样器和 `rollout` 的采样器（含玩家停止、重生）都能通过验证 |

**本机实测**（r6 最新检查点，课程前沿 4100，一次只开一个引擎）：
- **2002 宿主**（`bin-x86p2002d` + `game-min`）：用主控程序的采样器跑 4 个诚实任务（2 个各 600 步；2 个跑到 `stall` 结束，各 4090 / 3789 步，含一次死亡重生），共 37 段，全部通过；一个 15–16 段的任务重放约 19–21 秒。另一个账号的任务把第 1 段第 10 步的奖励加 0.25，验证器报 `replay_reward_mismatch`（第 1 段第 266 步），账号吊销，后面的段全部隔离。
- **x87-soft**（`bin-x86soft`）：一个诚实任务通关（454 步，2 段），通过；伪造的奖励同样被抓到。

### 10.8 还没做 / 限制

- **没有接真论坛测过。** OAuth 只在模拟论坛上测过。真论坛的 `/api/user` 如果是 JSON:API 包装（`data.attributes`）也能解析，但字段名没有实际核对过。
- **自己挑动作已经查得出（第 10.6 节的 `sample_seed`）**，但只对被重放的段有效；过了试用期，没抽中的段仍然只按规则放行。
- **跨机器**（第 8 节）：链 v2 只要求引擎字节相同；实测都在同一台机器上（原生客户端 + Python 验证器，两个 x87-soft 构建）。
- **任务循环有两份：** `rollout.TaskRunner`（验证器）和 `client_sampler.TaskSampler`（主控程序）实现同一套规则，靠对照测试保持一致，以后应合并成一份。有一处值得注意：死亡那一步的 `arc` 记的是重生之后那一行的弧长。主控程序是这样记录的，验证器照样比较。
- **有选择地放弃任务查不出来：** 玩家可以只交“跑得好”的任务，把其余的停掉或丢掉，这会让结果分布偏向好局。重放只能证明交上来的数据是真的。
- **实例上限按活着的会话计算：** 一个会话超过 max(3 × 心跳间隔, 90) 秒没有动静，它名下没结束的任务就不再占用 8 个实例的名额（主控程序被强杀后重启，最多等这么久）；这些任务仍然到期限才过期，验证器再验证已到的段。
- **`stale_policy` 没有用：** 任务固定用发出时的策略版本，数据也不进训练，所以不按新旧拒收。
- **验证器是单线程**，2002 宿主上重放约 200 步/秒，抽查 10% 时够用。规模上来以后要多开进程，并且要顾及本机的训练。
- **只有回环地址上的纯 HTTP。** 隧道、HTTPS、按 `CF-Connecting-IP` 限速，都要等接入隧道时再核对。
- **磁盘保留没做：** 放行、拒收、隔离的段一直留在 `chunks/` 下（不再占用上限，但占磁盘），接入训练器时再定保留策略。

### 10.9 评审修复（第 2c 步）

评审提出的 19 条都核对过，全部属实并已修复：

| 方面 | 修复 |
|---|---|
| 登录链接被转发 | 确认页 + 登录代码 + 一次性链接和 `SameSite=Strict` cookie（第 2.1 节）；成功页显示账号和代码；主控程序（界面和 `--login`）显示登录代码 |
| 存储被撑满 | zstd 段大小上限、raw 段大小精确、请求体不超过诚实上限、收段时查每账号上限、总上限只算 `held`（第 3.1、10.4 节） |
| 登录轮询限速 | `login_poll` 有自己的两个桶；https 必须 `trust_forwarded = true`（第 10.4 节） |
| 读请求体前不认证 | 先查令牌、连接数上限、单请求 120 秒看门狗、文件分块发送（第 10.4 节） |
| 错误响应本身出错 | `protocol_unsupported` 的 `got` 转成字符串；`handle()` 有固定的兜底响应；解析不了的请求目标回 400；连接异常只记一行 |
| zstd 窗口 | `window_log_max = 23` |
| 引擎崩溃当成截断 | 两边检查 `info["error"]`，重生时出错也一样（第 10.6 节） |
| 轨迹换签重交 | `sample_seed` 确定性抽取（第 10.6 节） |
| 试用期按段数 | 改按步数（第 10.6 节） |
| 期限和吊销的竞态 | `add_chunk` 在事务里重查，失败删文件；验证器只处理快照里的段 |
| 实例无限重启 | 每个槽位记连续失败次数，2 秒起翻倍到 5 分钟，连续 8 次后停止并显示横幅；按“开始”或稳定运行 2 分钟后清零 |
| 登录中退出 | `action_quit` / 退出按钮先 `request_stop()`，`on_unmount` 里 `shutdown()` |
| 崩溃转储 | 实例在任务结束后把 `<Ballance>/Bin/crash-*.dmp`（本次引擎的，或任务开始后新写的）移到 `<数据目录>/logs/`，只留最新 5 个 |
| `session_expired` | 主控程序自动重新 `hello`（最多 30 秒一次），成功后清掉横幅；`too_many_instances` 时调低实例数也会重新连接 |
| 上榜开关连点 | 请求进行中开关禁用 |
| 吊销后改实例数 | 上限为 0 时不再把选择压到 1 |
| 坏的缓存清单 | 清单先核对 `policy_sha`，任何失败都删掉清单和权重；实例连续失败时等待时间翻倍（最多 5 分钟） |
| 演示中发现：相对路径 | `--engine` / `--ballance-dir` 给相对路径时，实例（工作目录是数据目录）找不到引擎；现在保存和下发前都转成绝对路径 |
| POSIX `preexec_fn` | 改用 `start_new_session=True`；`PR_SET_PDEATHSIG` 和父进程核对在实例自己的 `main()` 开头做（`client_lifecycle.bind_to_parent`，没有在 Linux / macOS 上跑过） |

**本机配置要改一处：** `runtime/pool/config.toml` 的 `public_base_url` 是 https，按新规则网关会拒绝启动，需要在 `[pool]` 里加 `trust_forwarded = true`（这个文件不进 git，没有替你改）。
