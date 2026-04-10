# HeadlessSim Boundaries

`tools/headless-sim` is a standalone full-run simulator host for AI training.

What belongs here:
- Core simulation runtime and DTOs under `src/Core/Simulation`
- Pure logic managers and models needed by `FullRunTrainingEnvService`
- Minimal null-platform save schema types required by serialization
- Stable compatibility stubs for excluded Godot and UI symbols

What does not belong here:
- UI reflection helpers such as `src/Core/Simulation/FullRunUiShim.cs`
- Scene tree, overlay, screen, and rendering-heavy `N*` implementations
- Platform strategies that perform real Steam, cloud, or runtime integration
- Stubs that try to recreate gameplay behavior

Stub policy:
- Stubs exist only to satisfy compilation boundaries
- Stubs may expose type shells, static factories, and trivial properties
- Stubs must not carry real selection, UI, or gameplay logic
- If a new dependency appears, prefer tightening compile inputs before adding a stub
