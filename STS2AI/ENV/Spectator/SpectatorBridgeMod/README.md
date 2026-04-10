# STS2 MCP Spectator Mod

Build against the upstream `0991` game assembly:

```powershell
dotnet build STS2AI/ENV/Spectator/SpectatorBridgeMod/sts2_mcp_spectator.csproj -c Debug
```

The output directory contains:

- `sts2_mcp_spectator.dll`
- `sts2_mcp_spectator.json`

Install by copying both files into a single folder under the game's `mods/` directory, for example:

```text
mods/sts2_mcp_spectator/
  sts2_mcp_spectator.dll
  sts2_mcp_spectator.json
```

Optional launch flags:

- `--mcp-port 17140`
- `--mcp-decision-overlay-file D:/path/to/ai_overlay.json`
