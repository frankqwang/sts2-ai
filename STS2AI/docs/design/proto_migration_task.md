# 任务：STS2 Bridge 协议从手写二进制迁移到 Protobuf 序列化

## 背景

当前 STS2 AI 的 C# 游戏端和 Python 训练端通过 Windows Named Pipe + 手写二进制协议通信。
协议在 `STS2AI/Python/env/binary_pipe_client.py` 中手写解码，C# 端手写编码。

问题：字段增删需要两端同步改代码，容易不一致；没有 schema 文档，只能靠读代码推断字段格式。

## 方案

**保留 Named Pipe 传输，只替换序列化格式为 Protobuf。**

```
当前:  C# 手写编码 bytes → Named Pipe → Python 手写解码 bytes
改后:  C# protobuf.Serialize() → Named Pipe → Python protobuf.Parse()
```

为什么不用 gRPC：
- Named Pipe 是 Windows 内核级 IPC，通信延迟约 0.01-0.05ms
- gRPC 走 TCP，即使 localhost 也要 0.3-1.0ms，慢 10-20 倍
- 当前通信不是瓶颈（游戏 step 1-10ms 才是），但没必要主动变慢
- 后续如果需要跨机器训练，再在 Protobuf 基础上加 gRPC 传输层

## Proto 文件

已定义在 `STS2AI/proto/game_state.proto`，覆盖所有战斗 + 非战斗状态。

## 需要做的事

### 1. C# 端

- 引入 `Google.Protobuf` NuGet 包
- 从 proto 生成 C# 代码：`protoc --csharp_out=. game_state.proto`
- 找到当前写 Named Pipe 的序列化代码（应该有类似 BridgeWriter / StateSerializer 的类）
- 把手写 binary 编码改为：
  ```csharp
  var state = new GameState { ... };  // 填充 protobuf message
  byte[] bytes = state.ToByteArray();
  // 写入 Named Pipe: [4字节长度][bytes]
  ```
- 保留 4 字节长度前缀的帧格式（和现在一样），只是 payload 变成 protobuf bytes

### 2. Python 端

- `pip install protobuf`（不需要 grpcio）
- 从 proto 生成 Python 代码：`python -m grpc_tools.protoc --python_out=. game_state.proto`
- 新增 `env/proto_pipe_client.py`：
  ```python
  # 读取 Named Pipe: [4字节长度][protobuf bytes]
  raw = pipe.read(length)
  state = GameState()
  state.ParseFromString(raw)
  # 转为 dict（和现有 binary_pipe_client 输出格式一致）
  return protobuf_to_dict(state)
  ```
- `protobuf_to_dict()` 确保输出和现有 `binary_pipe_client._decode_state()` 完全一致
- 上层代码不需要任何改动

### 3. 测试

- 单元测试：C# 填充 GameState → 序列化 → Python 反序列化 → 与现有 binary_pipe_client 输出逐字段对比
- 集成测试：用 proto_pipe_client 替换 binary_pipe_client 跑一个完整 episode
- 现有训练脚本 (train_combat_only.py) 零修改即可运行

### 4. 后续可选：加 gRPC 传输层

如果后续需要跨机器训练：
- C# 端加 gRPC server（复用已有的 proto message，只加 service 接口）
- Python 端加 grpc_client.py（与 proto_pipe_client 接口一致）
- 配置选择用 pipe 还是 gRPC

## 不需要改的

- `networkV2/` 下的所有代码不需要动
- `env/combat_training_env.py` 的 normalize 逻辑不需要动
- 训练代码不需要动
- 现有 `binary_pipe_client.py` 保留（不删），新增 `proto_pipe_client.py` 并行

## 验收标准

1. proto_pipe_client 输出和 binary_pipe_client 输出格式完全一致
2. 现有训练流程零修改就能跑
3. 新增/删除字段只需改 proto 文件 + 重新生成代码
4. 性能不退化（Named Pipe 传输不变）

## 预估工作量

- C# 端 protobuf 序列化替换：2 天
- Python 端 proto_pipe_client：1 天
- 测试 + 联调：1 天
- 总计：约 4 天
