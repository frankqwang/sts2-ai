# Full Run API Checklist

Use this checklist after rebuilding `STS2MCP` to confirm the MCP contract still matches the docs.

## Menu

- `GET /api/v1/singleplayer` returns `state_type = "menu"` when no run is active.
- `menu.available_actions` contains `select_character`, `set_ascension`, and `start_run`.
- `menu.available_characters` is present and lists unlocked characters.
- `menu.selected_character`, `menu.ascension`, and `menu.max_ascension` are present when character select is visible.

## Game Over

- `GET /api/v1/singleplayer` returns `state_type = "game_over"` when the run ends on the game-over overlay.
- `game_over.kind` is `game_over`.
- `game_over.terminal` is `true`.
- `game_over.outcome` is `death` or `victory`.
- `game_over.buttons` is present and includes visible overlay buttons with indices.
- `game_over.available_actions` is present and mirrors enabled overlay button presses.

## Overlay Press

- `POST /api/v1/singleplayer` supports `{ "action": "overlay_press", "index": 0 }`.
- `overlay_press` uses the visible button index from `game_over.buttons` or `overlay.buttons`.
- `overlay_press` is only needed for structured overlays that are not already handled by a screen-specific action.

## Card Select

- `GET /api/v1/singleplayer` returns `selected_cards` for both `card_select` and `simple_select` flows.
- `card_select.selected_count`, `min_select`, `max_select`, and `remaining_picks` are present.
- `simple_select.selected_count`, `min_select`, `max_select`, and `remaining_picks` are present.
- `can_confirm` and `requires_manual_confirmation` are present so clients can tell when selection is actionable.
- `selected_cards` entries include at least `index`, `id`, `name`, `type`, `cost`, and `is_upgraded`.

## Quick Smoke

1. Build `STS2MCP` with `-p:STS2AssemblyDir=D:\dev\ai-slay-sts2\sts2\.godot\mono\temp\bin\Debug`.
2. Start the game and load the MCP mod.
3. Query `GET /api/v1/singleplayer` at menu and confirm the `menu` block.
4. Reach a card selection screen and confirm `selected_cards` and `remaining_picks` are present.
5. Reach a run-ending `NGameOverScreen` and confirm the `game_over` block.
6. Press a visible game-over button with `overlay_press` and verify the screen advances or the action is rejected with a clear error.

## Regression Rule

- If `game_over` collapses back into generic `overlay`, or if `overlay_press` disappears from the POST contract, treat it as a regression.
- If menu actions disappear from `available_actions`, treat it as a regression.
- If `card_select` or `simple_select` loses `selected_cards` or `remaining_picks`, treat it as a regression.
