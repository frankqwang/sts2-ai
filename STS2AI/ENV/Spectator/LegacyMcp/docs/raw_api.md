# Raw API Reference

These API endpoints are available for direct HTTP requests *without* using the MCP server. For example, you can use `curl` or Postman to interact with the mod directly.

The `v2/combat_env` endpoints are intended for the **Slay the Spire 2 source build** running in `--combat-trainer` mode. They do not fall back to the older UI-driven API.

The mod exposes two endpoints:
- `http://localhost:15526/api/v1/singleplayer` 闂?for singleplayer runs
- `http://localhost:15526/api/v1/multiplayer` 闂?for multiplayer (co-op) runs
- `http://localhost:15526/api/v2/combat_env/*` 闂?for combat training on the source-built game

The endpoints are mutually exclusive: calling the singleplayer endpoint during a multiplayer run (or vice versa) returns HTTP 409.

`/api/v2/combat_env/*` is different:
- it is only intended for the source-built training runtime
- it requires launching the game with `--combat-trainer`
- it does not fall back to `/api/v1/*`

:::note
These endpoints are designed for local use and do not have authentication or security measures, so they should not be exposed publicly - unless you know what you're doing!
:::

## `GET /api/v1/singleplayer`

Query parameters:
| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `format`  | `json`, `markdown` | `json` | Response format |

Returns the current game state. The `state_type` field indicates the screen:
- `monster` / `elite` / `boss` 闂?In combat (full battle state returned)
- `hand_select` 闂?In-combat card selection prompt (exhaust, discard, etc.) with battle state
- `combat_rewards` 闂?Post-combat rewards screen (reward items, proceed button)
- `card_reward` 闂?Card reward selection screen (card choices, skip option)
- `map` 闂?Map navigation screen (full DAG, next options with lookahead, visited path)
- `rest_site` 闂?Rest site (available options: rest, smith, etc.)
- `shop` 闂?Shop (full inventory: cards, relics, potions, card removal with costs)
- `event` 闂?Event or Ancient (options with descriptions, ancient dialogue detection)
- `card_select` 闂?Deck card selection (transform, upgrade, remove, discard) or choose-a-card (potions, effects)
- `relic_select` 闂?Relic choice screen (boss relics, immediate pick + skip)
- `treasure` 闂?Treasure room (chest auto-opens, relic claiming)
- `game_over` - Terminal game-over/victory overlay with structured buttons and outcome
- `overlay` - Catch-all for unhandled overlay screens with structured buttons/actions
- `menu` - No run in progress

### State details

**Battle state includes:**
- Player: HP, block, energy, stars (Regent), gold, character, status, relics, potions, hand (with card details including star costs), pile counts, pile contents, orbs
- Enemies: entity_id, name, HP, block, status, intents with title/label/description
- Keywords on all entities (cards, relics, potions, status)

**Hand select state includes:**
- Mode: `simple_select` (exhaust/discard) or `upgrade_select` (in-combat upgrade)
- Prompt text (e.g., "Select a card to Exhaust.")
- Selectable cards: index, id, name, type, cost, description, upgrade status, keywords
- Already-selected cards: index, id, name, type, cost, upgrade status
- Selection counts: `selected_count`, `min_select`, `max_select`, `remaining_picks`
- Confirmability: `can_confirm`, `requires_manual_confirmation`
- Full battle state is also included for combat context

**Rewards state includes:**
- Player summary: character, HP, gold, potion slot availability
- Reward items: index, type (`gold`, `potion`, `relic`, `card`, `special_card`, `card_removal`), description, and type-specific details (gold amount, potion id/name)
- Proceed button state

**Event state includes:**
- Event metadata: id, name, whether it's an Ancient, dialogue phase status
- Player summary: character, HP, gold
- Options: index, title, description, locked/proceed/chosen status, attached relic (for Ancients), keywords

**Rest site state includes:**
- Player summary: character, HP, gold
- Available options: index, id, name, description, enabled status
- Proceed button state

**Shop state includes:**
- Player summary: character, HP, gold, potion slot availability
- Full inventory by category: cards (with details, cost, on_sale, keywords), relics (with keywords), potions (with keywords), card removal
- Each item: index, cost, stocked status, affordability
- Shop inventory is auto-opened when state is queried

**Map state includes:**
- Player summary: character, HP, gold, potion slot availability
- Current position and visited path
- Next options: index, coordinate, node type, with 1-level lookahead (children types)
- Full map DAG: all nodes with coordinates, types, and edges (children)

**Card select state includes:**
- Screen type: `transform`, `upgrade`, `select`, `simple_select`, `choose`
- Player summary: character, HP, gold
- Prompt text (e.g., "Choose 2 cards to Transform.")
- Cards: index, id, name, type, cost, description, rarity, upgrade status, keywords
- Selected cards: index, id, name, type, cost, upgrade status
- Selection counts: `selected_count`, `min_select`, `max_select`, `remaining_picks`
- Preview state, confirm/cancel button availability
- Confirmability: `can_confirm`, `requires_manual_confirmation`
- For `choose` type (e.g., Colorless Potion): immediate pick on select, skip availability

**Relic select state includes:**
- Prompt text
- Player summary: character, HP, gold
- Relics: index, id, name, description, keywords
- Skip availability

**Card reward state includes:**
- Card choices: index, id, name, type, energy cost, star cost (Regent), description, rarity, upgrade status, keywords
- Skip availability

**Treasure state includes:**
- Player summary: character, HP, gold
- Relics: index, id, name, description, rarity, keywords
- Proceed button state
- Chest is auto-opened when state is queried

**Game over / overlay state includes:**
- `screen_type`: exact overlay class name
- `kind`: normalized overlay kind such as `game_over` or `generic_overlay`
- `terminal`: whether the overlay ends the run
- `outcome`: `death` or `victory` when the overlay is a game-over screen
- `buttons`: all visible overlay buttons with index, text, enabled/visible state, and confirm/cancel hints
- `available_actions`: enabled overlay button presses that clients can enumerate directly
- `primary_text`: best-effort main text extracted from the overlay

**Menu state includes:**
- `is_main_menu_visible`: whether the main menu is visible
- `has_run_save`: whether a save exists that can be resumed
- `can_open_singleplayer`: whether the singleplayer button is currently enabled
- `singleplayer_submenu_visible`: whether the singleplayer submenu is open
- `character_select_visible`: whether character select is open
- `selected_character`: the selected character id when character select is visible
- `ascension` / `max_ascension`: current and maximum ascension values
- `can_start`: whether a run can be started from the current menu state
- `available_actions`: menu actions the client can call directly, currently `select_character`, `set_ascension`, and `start_run`
- `available_characters`: unlocked characters and their menu metadata

## `POST /api/v1/singleplayer`

**Play a card:**
```json
{
  "action": "play_card",
  "card_index": 0,
  "target": "jaw_worm_0"
}
```
- `card_index`: 0-based index in hand (from GET response)
- `target`: entity_id of the target (required for `AnyEnemy` cards, omit for self-targeting/AoE cards)

**Use a potion:**
```json
{
  "action": "use_potion",
  "slot": 0,
  "target": "jaw_worm_0"
}
```
- `slot`: potion slot index (from GET response)
- `target`: entity_id of the target (required for `AnyEnemy` potions, omit otherwise)

**End turn:**
```json
{ "action": "end_turn" }
```

**Select a card from hand during combat selection:**
```json
{ "action": "combat_select_card", "card_index": 0 }
```
- `card_index`: 0-based index of the card in the selectable hand (from GET response)
- Used when a card effect prompts "Select a card to exhaust/discard/etc."

**Confirm in-combat card selection:**
```json
{ "action": "combat_confirm_selection" }
```
- Confirms the current in-combat hand card selection
- Only works when the confirm button is enabled (enough cards selected)

**Claim a reward:**
```json
{ "action": "claim_reward", "index": 0 }
```
- `index`: 0-based index of the reward on the rewards screen (from GET response)
- Gold, potion, and relic rewards are claimed immediately
- Card rewards open the card selection screen (state changes to `card_reward`)

**Select a card reward:**
```json
{ "action": "select_card_reward", "card_index": 1 }
```
- `card_index`: 0-based index of the card to add to the deck (from GET response)

**Skip card reward:**
```json
{ "action": "skip_card_reward" }
```

**Proceed:**
```json
{ "action": "proceed" }
```
- Proceeds from the current screen to the map
- Works from: rewards screen, rest site, shop (auto-closes inventory), treasure room
- Does NOT work for events - use `choose_event_option` with the Proceed option's index
- Does NOT automatically handle game-over overlays

**Press an overlay button:**
```json
{ "action": "overlay_press", "index": 0 }
```
- `index`: 0-based index of the visible overlay button from the `buttons` array in the GET response
- Use this for terminal overlays such as `game_over`, or any other structured overlay that exposes buttons
- Enabled buttons are also listed in `available_actions` for easier client-side enumeration
