"""s0_bridge — Protobuf 协议桥接层(V2 训练唯一通道)。

2026-04-18 起:手写 binary wire(`env.binary_pipe_client`)已废弃,全部走
`networkV2.s0_bridge.transport.PipeConnection` + `ProtoCodec`。
"""
