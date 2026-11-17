# 训练进度网页

```bash
uv run python tools/dashboard/server.py                     # 默认监听 0.0.0.0:8088
uv run python tools/dashboard/server.py --host 127.0.0.1 --tensorboard-url /tensorboard/
```

也可以在 Claude 预览里启动 `.claude/launch.json` 的 `dashboard`。

| 选项（环境变量） | 默认 | 作用 |
|---|---|---|
| `--host`（`BALLRAIN_DASH_HOST`） | `0.0.0.0` | 监听地址 |
| `--port`（`BALLRAIN_DASH_PORT`） | `8088` | 端口 |
| `--tensorboard-url`（`BALLRAIN_DASH_TENSORBOARD_URL`） | 空 | 页面上 TensorBoard 按钮的链接；为空时显示“暂未开放”，不会自动拼本机地址 |
| `--train-machine` / `--eval-machine` | `Mac mini` / `Windows PC` | 页面上显示的机器描述（只写通用型号） |
| `--interval` | 600 | 后台重算间隔（秒） |
| `--min-refresh` | 300 | “立即刷新”的全局限流（秒）：任何人触发后，这段时间内不再重算 |
| `--no-game-assets` | 关 | 不提供游戏贴图 / 菜单字体，页面退回纯色（公开部署时若不想分发游戏素材就打开） |
| `--show-shortcut-line` | 关 | 在课程地图上画鬼躲石变参考线（这条线取自人类录像，公开时保持关闭） |

工作方式：后台线程每 10 分钟读一次训练输出，把地图 / 曲线 / SR 估计算成 JSON（gzip）写到 `runtime/dashboard/`（不提交）；
页面每 60 秒拉一次 `/api/status`，数据版本变了才重新下载对应 JSON，全部图表在浏览器里用 ECharts 绘制（缩放、悬停、图例开关、点击高亮）。
关卡俯视轮廓只在第一次启动时由 TAS 记录重建一次，缓存在 `runtime/dashboard/cache/`。启动时从本机游戏拷贝的 `Textures/`
复制几张贴图（天空、木板、地砖、菜单按钮、菜单字体）到 `runtime/dashboard/assets/`，游戏素材不进 git；找不到时页面退回纯色。
进程自动降低 CPU 优先级，只读训练输出，不启动游戏，不碰训练进程。

## 最新进展报告

页面最上方的“最新进展”来自 `runtime/dashboard/reports.json`（不提交），用下面的命令追加一条（每 3 小时写一次）：

```bash
uv run python tools/dashboard/report.py add --title "标题" --item "要点 1" --item "要点 2" --next "下一步"
uv run python tools/dashboard/report.py list
```

写入前会检查有没有不能公开的内容（路径、主机、IP、用户名、进程号、实例名、本机运维琐事、人类录像数据），有就拒绝写入；
服务端输出前会再查一遍。页面显示最新一条，更早的折叠在下面。

## 精彩回放

页面“最新进展”下面的“精彩回放”来自 `runtime/dashboard/media/`（不提交），由下面的任务每小时更新一次：

```powershell
# 训练机上的无头回合只在训练机上保留通关目录：和 pull_run.sh 一样，在 shell 里设好 BALLRAIN_MAC / BALLRAIN_MAC_SSH
Start-Process -WindowStyle Hidden -FilePath uv -ArgumentList 'run','python','tools/dashboard/highlight.py','--every','3600' `
  -RedirectStandardOutput runtime/video-work/highlight.out -RedirectStandardError runtime/video-work/highlight.err
uv run python tools/dashboard/highlight.py --once [--force]   # 立即做一次（--force：最佳回合没变也重录）
uv run python tools/dashboard/highlight.py --once --force --media runtime/video-work/media-test   # 试跑：只写到草稿目录
uv run python tools/dashboard/highlight.py gate               # 录制版 BallrainIO 与 0.3.2 的轨迹逐字节比对
```

- **选回合**：当前 run 最近 60 分钟（学习器时间）里无死亡通关的回合；先取这些通关回合里最早的开局区间（2000 帧一档）里的，
  按整关 SR 从快到慢，再按开局区间从早到晚接其余通关回合（只按整关 SR 排会偏向接手很晚、几乎全是 TAS 的回合）。逐个尝试，直到有一个在原版游戏里完全复现。
- **重放**：回合自己的 `.tas`（从读关开始的完整输入：TAS 前缀到第 k 帧，之后是智能体的按键）在专用拷贝 `video-00`
  （2002 物理）里用训练环境本身重放：每步把 `.tas` 的按键解码成动作交给环境，环境再编码成按键发出，发出的帧必须和 `.tas`
  逐字节相同。窗口隐藏、从不激活、静音，进程低优先级。
- **核对**：mod 自己的帧日志存在且与 `.tas` 相同、在同一帧 LevelFinish、没有死亡、SR（BML+ 0.3.12 计时，含 SR 尾）精确到毫秒相同；
  有轨迹的回合再比对整条轨迹；录像完整（只启动一次游戏、1024×768、不丢帧、从读关录到飞船之后）。不一致重试一次，再换下一个候选。
  `video-00` 里装的必须是录制版 BallrainIO（不是就先装上，装不上就跳过这一轮）。
- **发布**：工作文件都在 `runtime/video-work/`。只有一整套（普通版、叠加信息版、`highlight.json`）全部做完并核对通过才发布：
  两个视频以带版本号的新文件名放进 `media/`，再一步替换 `highlight.json`（页面和服务端只认 json 里写的文件名），
  然后删掉更早的版本（保留上一套给正在播放的人）。任何一步失败，上一套原样保留，只记日志。
- **录制**：BallrainIO 0.3.3 的 `BALLRAIN_CAPTURE` 在每帧画完后读回游戏自己的后台缓冲（不读屏幕），按游戏时间每 1/60 秒一帧，
  从读关到上飞船后约 6 秒，不剪辑。H.264、1024×768、60 帧、faststart、无声。
- **叠加信息版**：同一次录制，右侧 342 像素的面板（1366×768）：TAS / 智能体控制、第几 tick 与智能体第几步、SR 计时、
  球种与速度、实际按下的键与按键记录、策略对 9 个方向的概率与所选动作、价值估计、本步奖励 / 累计回报与分项、
  观测里的周围地形（7×7 地面高度、8 条到边缘的射线）。数值都由这次重放用训练代码重算（策略取该回合之前最近的模型检查点）。
  进度奖励与累计回报只是数值，可以公开；参考路线的路点和路线几何不画（路线来自人类录像）。价值估计换算回奖励单位
  （除以训练的 reward_scale），和旁边的奖励、回报可比。页面默认播放叠加版，可切换，切换时保持播放位置；
  有人正在播放时，新一小时的回放等暂停后才换上。

## 公开内容与隐私

页面可能放到公网上，所以服务端只输出训练本身：

- **公开**：里程碑与 run 编号（M2、r1…r5）、更新次数 / 步数、课程开局点与推进记录（改写成中文，不含命令行参数）、
  成功率 / 通关率 / 熵 / 每秒步数曲线、智能体轨迹（只有世界坐标、帧号、结果、开始时间）、SR 估计与基准（TAS、人类纪录、精品线、上榜线，
  只有成绩数值，不含玩家名）、从头开始的原版评估结果、训练数据来自哪台机器（Mac mini / 14700KF / H 255 这类通用机型名，由 `--source-labels` 把 worker 上报的通用主机标签映射过去）及其后端（x87-soft headless / 2002 headless / 2002 retail）的步数比例、通用机器描述、
  精彩回放（原版游戏自己画面的录像与叠加信息版；元数据只有 run 编号、更新次数、回合时间、录制时间与 UTC 偏移、整关 SR、
  接手帧 k、TAS 成绩、原始后端标签、是否复现一致、时长）。
  时间按本机时区显示并标明 UTC 偏移。
- **不公开**：文件路径、主机名、IP、用户名、SSH 目标、进程号、回合 run id、游戏拷贝 / 实例名、训练日志里的运维信息
  （警告、断线、重启、worker / 实例数量、窗口 / 静音 / 焦点处理等）、本机 TensorBoard 地址、任何人类录像数据或由它派生的内容
  （鬼躲石变参考线默认不画）、BMMO 公测数据。出错信息先去掉路径再显示。
- **服务端安全**：只响应固定的几个地址（`/`、`/api/status`、`/api/data/<名字>.json`、`/static/` 下的 js / css、`/assets/` 下启动时生成的文件清单、
  `/media/highlight[-overlay]-<版本>.mp4`（只认 `highlight.json` 写的文件名格式；json 换成带版本号的格式后，第一版的固定文件名不再提供；支持 Range，
  无效或不认识的 Range 按整个文件处理；文件读入内存后切片发送，不长期占用文件，内存里最多留 4 个、文件没了就丢掉）），
  文件名用白名单正则匹配，没有目录列表、没有任意文件读取；`POST /api/refresh` 不接受任何参数，并受全局限流。`/tensorboard/` 不由本服务处理。
