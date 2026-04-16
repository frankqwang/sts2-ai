# Proto Bridge 使用指南

## 概述

Proto Bridge 是 STS2 AI 的 Protobuf 协议桥接层，替代原有的手写二进制协议 (`binary_pipe_client`)。

- 传输层不变（Named Pipe + 长度前缀帧），只替换 state 序列化格式
- Python 端和 C# 端完全独立于旧代码，可以共存
- 切换只需改一行 import，上层代码零修改

## 文件结构

### Python 端

```
networkV2/s0_bridge/
├── __init__.py
├── constants.py                 # 操作码、状态类型等共享常量
├── proto_pipe_client.py         # ProtoPipeClient（替换 BinaryPipeClient）
├── proto_state_converter.py     # GameState proto → 兼容 dict 转换层
└── generated/
    ├── __init__.py
    ├── game_state_pb2.py        # protoc 生成，不要手改
    └── game_state_pb2_grpc.py   # gRPC 备用
```

### C# 端

```
HeadlessSim/
├── ProtoStateBuilder.cs         # 游戏对象 → proto message 映射（新增）
├── BinaryProtocol.cs            # 原有二进制协议（改动：加了 HostProtocol.Proto）
├── Program.cs                   # 改动：加了 --protocol proto 路由
└── obj/.../GameState.cs         # protoc 自动生成，不要手改
```

### Proto 定义

```
proto/game_state.proto           # 唯一的 schema 源文件
```

## 快速上手

### 1. 启动 C# 模拟器（proto 模式）

```bash
dotnet run --project HeadlessSim -- --port 15527 --protocol proto
```

这会创建命名管道 `\\.\pipe\sts2_mcts_proto_15527`。

### 2. Python 端连接

```python
from networkV2.s0_bridge.proto_pipe_client import ProtoPipeClient

client = ProtoPipeClient(port=15527)
client.connect()

# 接口和 BinaryPipeClient 完全一致
state = client.call("reset", {"seed": "TEST1"})
state = client.call("step", {"action": "choose_map_node", "index": 0})
state = client.call("state")

client.close()
```

### 3. 从旧代码迁移

只需替换 import：

```python
# 旧
from env.binary_pipe_client import BinaryPipeClient
client = BinaryPipeClient(port=15527)

# 新
from networkV2.s0_bridge.proto_pipe_client import ProtoPipeClient
client = ProtoPipeClient(port=15527)
```

`call()` 返回的 dict 格式完全一致，上层代码（normalize / 网络输入 / action 选择）零修改。

## 多环境并发

每个 worker 使用独立端口：

```python
num_envs = 8
clients = [ProtoPipeClient(port=15527 + i) for i in range(num_envs)]

for c in clients:
    c.connect()

# 各 worker 独立使用自己的 client
# 单个 ProtoPipeClient 实例不可跨线程共享
```

### 长时间运行（训练循环）

用 `safe_call()` 代替 `call()`，管道断开时自动重连：

```python
# safe_call = call + 自动重连（最多 3 次，可配）
state = client.safe_call("step", {"action": "end_turn"})
```

### 监控

```python
for c in clients:
    print(c.stats)
# {'port': 15527, 'call_count': 1234, 'error_count': 2, 'connected': True}
```

## 重新生成 Protobuf 代码

修改 `proto/game_state.proto` 后：

### Python

```bash
cd STS2AI
python -m grpc_tools.protoc \
  --proto_path=proto \
  --python_out=Python/networkV2/s0_bridge/generated \
  --grpc_python_out=Python/networkV2/s0_bridge/generated \
  proto/game_state.proto
```

### C#

C# 端在 `dotnet build` 时由 `Grpc.Tools` 自动生成，无需手动操作。

## 运行测试

```bash
cd STS2AI/Python
python -m pytest tests/test_proto_state_converter.py -v
```

## 协议细节

### 帧格式（和旧协议一致）

```
请求: [4字节小端长度][1字节opcode][参数bytes]
响应: [4字节小端长度][1字节status][1字节opcode][payload bytes]
```

### 差异点

| | 旧 (binary) | 新 (proto) |
|---|---|---|
| State payload | 手写二进制 | protobuf GameState.ToByteArray() |
| 符号表 | 需要（增量字符串ID缓存）| 不需要 |
| 玩家静态缓存 | 需要（版本号缓存deck/relics）| 不需要 |
| 请求编码 | opcode + 二进制参数 | 完全一致 |
| 非state响应 | 手写二进制 | 完全一致 |
| Pipe 名称 | sts2_mcts_bin_{port} | sts2_mcts_proto_{port} |
| 握手版本 | PROTOCOL_VERSION=12 | PROTO_PROTOCOL_VERSION=1 |

### 新增/删除字段

修改 `proto/game_state.proto` 后重新生成代码即可，两端自动兼容。这是迁移到 protobuf 的核心收益。

## 已知限制

- `card_select` 和 `relic_select` 状态在 proto 里暂未定义独立 message，converter 返回空壳
- `treasure` 状态复用 `CombatRewardsState` 承载
- Python 端 `ProtoPipeClient` 单实例不可跨线程共享
