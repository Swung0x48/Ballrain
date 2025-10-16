# Ballrain TCP Communication Documentation

## Overview
This document describes the TCP communication protocol between the Python RL environment and the C++ Ballance game plugin. The communication is bidirectional and enables the Python environment to control the game and receive state information for reinforcement learning purposes.

## Architecture
- **Python side**: Runs a TCP server (`TCPServer.py`) that listens for connections from the game
- **C++ side**: Acts as a TCP client (`TCPClient.h/.cpp`) that connects to the Python server
- **Default port**: 27787
- **Protocol**: Little-endian binary messaging system

## Connection Establishment
1. Python environment starts a TCP server on port 27787
2. C++ plugin connects to the server using `TCPClient::Connect("127.0.0.1", 27787)`
3. Handshake occurs:
   - Python server sends "Ping" message
   - C++ client receives "Ping", responds with "Pong"
   - Connection confirmed as valid
4. Communication waits for `BallNavActive` message to begin active gameplay

## Message Types
Messages follow a standard format:
- **Header**: 4-byte message type (little-endian integer)
- **Body**: Variable-length data depending on message type

### Message Format Reference
```python
# Python side (Message.py)
class MsgType(Enum):
    BallNavActive = 0      # No body
    BallNavInactive = 1    # No body
    GameState = 2          # 60-byte body
    KbdInput = 3           # 4-byte body
    Tick = 4               # No body
    ResetInput = 5         # No body
    BallOff = 6            # No body
    SceneRep = 7           # Variable length body
```

```cpp
// C++ side (MessageTypes.h)
enum class MessageType : int {
    BRM_BallNavActive,
    BRM_BallNavInactive,
    BRM_GameState,
    BRM_KbdInput,
    BRM_Tick,
    BRM_ResetInput,
    BRM_BallOff,
    BRM_MsgSceneRep,
    BRM_InvalidType
};
```

## Data Transfer Format and Content

### 1. GameState Message (Type 2)
**Direction**: C++ → Python
**Body size**: 60 bytes (little-endian format)

| Field | Size     | Offset | Type | Description |
|-------|----------|--------|------|-------------|
| ballType | 4 bytes  | 0 | int (little-endian) | Type of ball currently in play |
| position | 12 bytes | 4 | 3x float32 | Ball position (x, y, z) |
| quaternion | 16 bytes | 16 | 4x float32 | Ball rotation (x, y, z, w) |
| currentSector | 4 bytes  | 32 | int (little-endian) | Current sector index |
| nextSectorPosition | 12 bytes | 36 | 3x float32 | Next sector position (x, y, z) |
| lastSectorPosition | 12 bytes | 48 | 3x float32 | Previous sector position (x, y, z) |

### 2. SceneRep Message (Type 7)
**Direction**: C++ → Python
**Variable body size**: Count of AABB boxes followed by 24 bytes per box (6 float32 values)

**Format**:
- 4 bytes: Count (number of floor boxes)
- 24 bytes per box: Axis-Aligned Bounding Box with 6 float32 values (min.x, min.y, min.z, max.x, max.y, max.z)

### 3. KbdInput Message (Type 3)
**Direction**: Python → C++
**Body size**: 4 bytes (key states)

**Format**: 4-bit flags in single byte:
- Bit 0: UP key (0=up, 1=pressed)
- Bit 1: DOWN key (0=up, 1=pressed)
- Bit 2: LEFT key (0=up, 1=pressed)
- Bit 3: RIGHT key (0=up, 1=pressed)

### 4. Tick Message (Type 4)
**Direction**: C++ → Python
**Body size**: 0 bytes
**Purpose**: Signal that game state has been sent and game is waiting for next input

### 5. Control Messages
- **BallNavActive (0)**: Signals that game is ready to receive control commands
- **BallNavInactive (1)**: Signals that game is no longer accepting control
- **ResetInput (5)**: Request from Python to reset the level
- **BallOff (6)**: Signals that the ball has fallen off (game over condition)

## Communication Timing and Flow

### Initialization Flow
1. Python server starts and waits for connection
2. C++ client connects and performs ping-pong handshake
3. C++ sends SceneRep (floor AABB data) followed by BallNavActive
4. Python receives SceneRep and acknowledges BallNavActive

### Game Loop Communication
Each game tick follows this pattern:

1. **C++ sends GameState**: Current ball position, rotation, and sector information
2. **C++ sends Tick**: Signals that game state is available and waiting for input
3. **Python receives messages**: Processes state and determines next action
4. **Python sends KbdInput**: Sends 4-bit control command (up/down/left/right)
5. **C++ receives and applies input**: Updates game state accordingly
6. **Repeat** for next frame

### Level Reset Flow
1. Python sends ResetInput message
2. C++ receives message and calls RestartLevel()
3. C++ sends scene representation again followed by BallNavActive
4. Python receives lingering messages and confirms navigation active

## Key Use Cases

### Reinforcement Learning Environment
- Python RL agent receives state information via GameState messages
- Agent computes action based on current observation
- Action is sent as KbdInput to control the ball
- Reward is calculated based on movement toward next sector

### Scene Understanding
- Floor geometry is communicated via SceneRep message
- Allows Python environment to understand level layout
- Enables more sophisticated decision making based on spatial information

## Error Handling
- If TCP connection is lost, both sides detect and handle appropriately
- Game state synchronization occurs after connection issues
- Python environment can detect game over conditions via BallOff messages