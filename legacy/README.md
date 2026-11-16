# legacy

早期的 TCP 原型：`TCPServer.py` / `Message.py`（端口 27787 上的二进制消息，协议见 `comm_protocol.md`）、
`BallanceEnv.py`（基于深度图的 gymnasium 环境）、`main.py`（SB3 训练入口）、`validate_image.py`。

它们依赖已被替换的旧版 BallrainIO（TCP 客户端），在当前仓库里不能运行，只作参考保留。
新的实现见 `ballrain/`（agent 模式：`docs/agent-protocol.md`、`docs/observations.md`）。
