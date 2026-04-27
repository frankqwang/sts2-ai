# STS2AI bridge 启动脚本（PowerShell Wrapper）

`STS2AI/bridge` 是当前 game bridge Python 根目录。旧 Python 根目录和
`networkV2` 主线已不作为默认上下文。

## 当前 wrapper

- `spectate.ps1`
  - 旧 networkV2 可视化 wrapper，保留用于迁移参考
  - 当前 LLM 观战请用 `STS2AI/llm/scripts/spectate_llm.ps1`
  - 录屏/日志落到 `STS2AI/Artifacts/recording/`
- `trainer_common.ps1`
  - 公共 PowerShell 函数（`Resolve-CommandOrPath` 等），被 `spectate.ps1` source

## 当前 bridge 命令速查

```powershell
cd STS2AI/bridge

# bridge 单测
python -m pytest tests -q

# bridge smoke
python scripts/game_bridge_smoke.py --mode combat_build_api

# spectator 控制器
python -m game_bridge.spectate.cli --mode null --backend spectator --transport http_json
```

LLM 训练、观战和评测入口在 `STS2AI/llm` 下。
