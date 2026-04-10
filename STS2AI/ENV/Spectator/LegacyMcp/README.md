# Slay The Spire 2 - MCP Server

A mod for [**Slay the Spire 2**](https://store.steampowered.com/app/2868840/Slay_the_Spire_2/) source builds that exposes game state and actions over localhost HTTP, with an optional MCP bridge for tool-using agents.

Singleplayer and multiplayer support remain available through `/api/v1/*`. Training clients should prefer the new `/api/v2/combat_env/*` endpoints.

> [!warning]
> This mod allows external programs to read and control your game via a localhost API. Use it only with local processes you trust.

## For Players

### Install For The Source Build

This repository is aligned to the **Godot source-build runtime**, not the loose-DLL Steam install flow.

1. Build and install the mod into the Godot runtime's `mods` directory:

```powershell
.\build.ps1 `
  -SourceRepoDir "C:\Users\Administrator\Desktop\Slay the Spire 2" `
  -GodotExe "C:\dev\game\Godot_v4.5.1-stable_mono_win64\Godot_v4.5.1-stable_mono_win64_console.exe" `
  -Install
```

2. This creates:

```text
out/STS2_MCP/STS2_MCP.dll
out/STS2_MCP/STS2_MCP.pck
```

3. And installs:

```text
<godot_exe_dir>\mods\STS2_MCP.dll
<godot_exe_dir>\mods\STS2_MCP.pck
```

4. Make sure mod loading is enabled in the source-build settings save.
5. Launch the game through Godot. For combat training, include `--combat-trainer`.
6. The mod starts an HTTP server on `http://localhost:15526/`.

### Connect To Claude

Requires [Python 3.11+](https://www.python.org/) and [uv](https://docs.astral.sh/uv/).

**Claude Code**: add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "sts2": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/STS2_MCP/mcp", "python", "server.py"]
    }
  }
}
```

**Claude Desktop**: use the same config in `claude_desktop_config.json`.

Full tool reference: [mcp/README.md](./mcp/README.md) | Raw HTTP API: [docs/raw_api.md](./docs/raw_api.md)

## For Developers

### Build

Requires [.NET 9 SDK](https://dotnet.microsoft.com/download/dotnet/9.0) and a Godot Mono executable that can run the source build.

```powershell
$env:STS2_SOURCE_REPO = "C:\Users\Administrator\Desktop\Slay the Spire 2"
$env:STS2_GODOT_EXE = "C:\dev\game\Godot_v4.5.1-stable_mono_win64\Godot_v4.5.1-stable_mono_win64_console.exe"
.\build.ps1 -Install
```

The build script:
- compiles `STS2_MCP.dll` against the source-build assembly output
- packs `mod_manifest.json` into `STS2_MCP.pck`
- optionally copies both files into `<godot_exe_dir>\mods\`

### Training API

`/api/v2/combat_env/*` is the training-focused bridge:
- `GET /api/v2/combat_env/state`
- `POST /api/v2/combat_env/reset`
- `POST /api/v2/combat_env/step`

These endpoints:
- require the game to be launched with `--combat-trainer`
- call the in-game `CombatTrainingEnvService` directly
- do not fall back to the older UI-driven `/api/v1/*` semantics

### Legacy Coverage

The original `/api/v1/singleplayer` and `/api/v1/multiplayer` routes are still present for broader UI-driven automation.

## License

MIT
