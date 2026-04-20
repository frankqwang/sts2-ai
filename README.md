# STS2AI

这是一个围绕《杀戮尖塔 2》做桥接与数据抓取的项目目录。

当前只保留两条主线：

- `STS2AI/Python/game_bridge`：游戏桥接、session、spectate、sim 启动与状态读取。
- `STS2AI/Python/data/skada`：skada 数据抓取、明细整理、覆盖率扫描等脚本。

过去文档里提到的训练主线、`networkV2`、离线 ranking、teacher/diagnostics 等内容都已经过时，不再作为本仓库的默认说明。

## 目录

```text
STS2AI/
├── Docs/                         项目文档
├── ENV/                          C# / 模拟器 / 观战相关工程
├── Python/
│   ├── game_bridge/              bridge 主体
│   ├── data/skada/               skada 抓取与整理脚本
│   ├── scripts/                  辅助脚本
│   └── tests/                    测试
└── scripts/                      仓库级脚本
```

## 快速开始

### Python 依赖

```powershell
pip install -r STS2AI/Python/requirements.txt
```

### Bridge 冒烟

```powershell
python STS2AI/Python/scripts/game_bridge_smoke.py --mode fake_spectate
python STS2AI/Python/scripts/game_bridge_smoke.py --mode full_run_state
```

### HeadlessSim 启动

```powershell
cd STS2AI/Python
python -m game_bridge.sim.cli launch --port 15527
```

### Session 状态检查

```powershell
cd STS2AI/Python
python -m game_bridge.session.cli inspect --kind full_run --auto-launch --use-pipe
```

### Spectate

```powershell
cd STS2AI/Python
python -m game_bridge.spectate.cli --mode manual --auto-launch --use-pipe
```

### Skada 脚本

skada 相关脚本位于 `STS2AI/Python/data/skada`，按具体任务直接运行对应脚本，例如抓取、daemon、明细构建和覆盖率扫描。

## 约定

- 除根目录 `README.md` 外，其它文档统一放在 `STS2AI/Docs`。
- 运行产物、日志、临时输出统一放在 `STS2AI/Artifacts`。
- `STS2AI/Artifacts/zero` 下的直接子目录统一使用 `MM-DD-HH-MM-name` 命名，例如 `04-20-19-30-skada-replay-train`，方便按名称顺序直接定位最新输出目录。
- `zero` 相关的分析、可视化、数据挖掘脚本统一放在 `STS2AI/zero/analysis`，训练结束后默认把图表和摘要输出到当次产物目录下的 `analysis`。
- 关键文件和关键函数要写“说明意图”的短注释，尤其是 loss、采样、晋级、teacher 这类容易演化的逻辑；代码注释默认用中文。当代码语义发生变更时，注释必须在同一次提交里同步更新，避免注释落后于实现。
- PowerShell、bash 等脚本凡是涉及文件输入输出，默认显式使用 UTF-8 读写，避免中文日志、文档和数据文件出现乱码。
- `src` 默认视为只读参考，不直接修改。
