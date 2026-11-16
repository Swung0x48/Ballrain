# Ballrain

Ballrain is a reinforcement learning environment for the game Ballance. It connects a Python reinforcement learning framework to a C++ game plugin over TCP, and is used to train agents for ball navigation and level completion.

## Architecture

The project is split into two main parts:

### Python side
- **TCPServer.py**: TCP server that handles communication with the game mod
- **BallanceEnv.py**: Gymnasium-based reinforcement learning environment
- **Message.py**: Defines message types and data formats

### C++ side (BallrainIO)
- **TCPClient**: TCP client that connects to the Python server
- **BallrainIO.cpp**: Ballance game plugin that handles game logic and state transfer
- **MessageTypes.h**: Defines the message type structures on the C++ side

## Installation and Usage

### Requirements
- Python 3.x, uv
- Ballance game
- A build of the game using a specific modified CKDX11Rasterizer (required for the `DepthImage` message to carry valid depth data)
- BML (Ballance Mod Loader)
- Visual Studio 2022 with the Desktop development with C++ workload

### Installation steps

1. Clone the project:
```bash
git clone <repository_url>
cd Ballrain
```

2. Create the virtual environment and install the Python dependencies with uv:
```bash
uv sync
```

`uv sync` creates a `.venv` directory, resolves the dependencies declared in `pyproject.toml`, records them in `uv.lock`, and installs them into the environment.

3. Build the C++ plugin (BallrainIO):

You need to download the BMLPlus 0.3.10 Debug development package beforehand and extract it to `3rdparty/BMLPlus`.

You also need to clone doyagu's Virtools SDK 2.1 repository beforehand:

```bash
git clone https://github.com/doyaGu/Virtools-SDK-2.1
```

```bash
cd BallrainIO
cmake . -B build -A Win32 -DVirtoolsSDK_DIR="/path/to/Virtools-SDK-2.1/CMake"
```

Open `Ballrain.sln` in the `BallrainIO/build` directory and build it, or build from the command line:

```bash
cmake --build . --config Release
```

4. Install the compiled C++ mod into the Ballance game directory

### Usage

1. Start the Python server: (the server must be started before the game)
```bash
uv run main.py
```

2. Load the plugin in the Ballance game

3. Once the connection is established, manually enter a level. The game then automatically starts accepting control from the Python environment and begins training.

## Message protocol

The project uses a TCP-based binary message protocol that supports the following message types:

- `BallNavActive` (0): game navigation activated
- `BallNavInactive` (1): game navigation deactivated
- `GameState` (2): game state information (60 bytes)
- `KbdInput` (3): keyboard input command (4 bytes)
- `Tick` (4): game clock signal
- `ResetInput` (5): reset the game
- `BallOff` (6): ball fell off the track signal
- `SceneRep` (7): scene representation information (variable length)
- `DepthImage` (8): depth image (320 × 240, 1 byte per pixel)

> The `DepthImage` message is only produced when the game runs with a specific modified build of CKDX11Rasterizer; with a stock rasterizer no valid depth image is available.

## Communication flow

1. The Python server starts and listens on port 27787
2. The C++ plugin connects and performs a ping-pong handshake for validation
3. C++ sends the scene representation `SceneRep` and the navigation activation signal `BallNavActive`
4. The game loop begins:
   - C++ sends the game state `GameState`, the depth image `DepthImage` and the clock signal `Tick`
   - Python receives the state and computes an action
   - Python sends the keyboard input command `KbdInput`
   - C++ applies the input and updates the game state
