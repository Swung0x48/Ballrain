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
| `--show-shortcut-line` | 关 | 在课程地图上画捷径 1 参考线（这条线取自人类录像，公开时保持关闭） |

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

## 公开内容与隐私

页面可能放到公网上，所以服务端只输出训练本身：

- **公开**：里程碑与 run 编号（M2、r1…r5）、更新次数 / 步数、课程开局点与推进记录（改写成中文，不含命令行参数）、
  成功率 / 通关率 / 熵 / 每秒步数曲线、智能体轨迹（只有世界坐标、帧号、结果、开始时间）、SR 估计与基准（TAS、人类纪录、加精线、上榜线，
  只有成绩数值，不含玩家名）、从头开始的原版评估结果、训练数据来自哪种后端（“无头 x87-soft” / “原版 2002”）的比例、通用机器描述。
  时间按本机时区显示并标明 UTC 偏移。
- **不公开**：文件路径、主机名、IP、用户名、SSH 目标、进程号、回合 run id、游戏拷贝 / 实例名、训练日志里的运维信息
  （警告、断线、重启、worker / 实例数量、窗口 / 静音 / 焦点处理等）、本机 TensorBoard 地址、任何人类录像数据或由它派生的内容
  （捷径 1 参考线默认不画）、BMMO 公测数据。出错信息先去掉路径再显示。
- **服务端安全**：只响应固定的几个地址（`/`、`/api/status`、`/api/data/<名字>.json`、`/static/` 下的 js / css、`/assets/` 下启动时生成的文件清单），
  文件名用白名单正则匹配，没有目录列表、没有任意文件读取；`POST /api/refresh` 不接受任何参数，并受全局限流。`/tensorboard/` 不由本服务处理。
