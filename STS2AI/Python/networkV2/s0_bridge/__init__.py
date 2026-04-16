"""s0_bridge — Protobuf 协议桥接层。

替代 env.binary_pipe_client 的手写二进制协议，
使用 protobuf 序列化，同时保持输出 dict 格式完全兼容。
"""
