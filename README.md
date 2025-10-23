# Ballrain

Ballrain 是一个基于 Ballance 游戏的强化学习环境，通过 TCP 通信将 Python 强化学习框架与 C++ 游戏插件连接，用于训练球体导航和关卡完成的智能体。

## 架构

项目分为两个主要部分：

### Python 部分
- **TCPServer.py**: TCP 服务器，处理与游戏 Mod 的通信
- **BallanceEnv.py**: 基于 Gymnasium 的强化学习环境
- **Message.py**: 定义消息类型和数据格式

### C++ 部分 (BallrainIO)
- **TCPClient**: TCP 客户端，连接到 Python 服务器
- **BallrainIO.cpp**: Ballance 游戏插件，处理游戏逻辑和状态传输
- **MessageTypes.h**: 定义 C++ 端的消息类型结构

## 安装和使用

### 环境要求
- Python 3.x, Conda (可以是 Miniconda)
- Ballance 游戏
- BML (Ballance Mod Loader)
- Visual Studio 2022，安装桌面C++工作负载

### 安装步骤

1. 克隆项目:
```bash
git clone <repository_url>
cd Ballrain
```

2. 创建 Conda 虚拟环境并安装 Python 依赖:
```bash
conda env create -f environment.yml
```

3. 编译 C++ 插件 (BallrainIO):

需要事先下载 BMLPlus 0.3.10 开发 Debug 包，解压到 `3rdparty/BMLPlus`。

需要事先克隆 doyagu 的 Virtools SDK 2.1 仓库。

```bash
git clone https://github.com/doyaGu/Virtools-SDK-2.1
```

```bash
cd BallrainIO
cmake . -B build -A Win32 -DVirtoolsSDK_DIR="/path/to/Virtools-SDK-2.1/CMake"
```

打开 `BallrainIO/build` 目录中的 `Ballrain.sln` 编译，或使用命令行编译：

```bash
cmake --build . --config Release
```

4. 安装编译后的 C++ Mod 到 Ballance 游戏目录

### 使用方法

1. 启动 Python 服务器: （注意服务器需先于游戏启动）
```python
python main.py
```

2. 在 Ballance 游戏中加载插件

3. 连接建立后，请手动进入一关。游戏会自动开始接受来自 Python 环境的控制，并开始训练。

## 消息协议

项目使用基于 TCP 的二进制消息协议，支持以下消息类型：

- `BallNavActive` (0): 游戏导航激活
- `BallNavInactive` (1): 游戏导航停用
- `GameState` (2): 游戏状态信息 (60 字节)
- `KbdInput` (3): 键盘输入指令 (4 字节)
- `Tick` (4): 游戏时钟信号
- `ResetInput` (5): 重置游戏
- `BallOff` (6): 掉球信号
- `SceneRep` (7): 场景表示信息 (可变长度)

## 通信流程

1. Python 服务器启动并监听端口 27787
2. C++ 插件连接并进行 ping-pong 握手验证
3. C++ 发送场景表示数据 `SceneRep` 和导航激活信号 `BallNavActive`
4. 进入游戏循环：
   - C++ 发送游戏状态 `GameState` 和时钟信号 `Tick`
   - Python 接收状态并计算动作
   - Python 发送键盘输入指令 `KbdInput`
   - C++ 应用输入并更新游戏状态
