using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text.Json;
using System.Threading;

namespace STS2_MCP;

public static partial class McpMod
{
    private static void HandleGetFullRunEnvState(HttpListenerResponse response)
    {
        try
        {
            Dictionary<string, object?> state = RunOnMainThread(BuildVisibleFullRunEnvState).GetAwaiter().GetResult();
            SendJson(response, state);
        }
        catch (Exception ex)
        {
            SendError(response, 500, $"Failed to read full run env state: {ex.Message}");
        }
    }

    private static void HandlePostFullRunEnvReset(HttpListenerRequest request, HttpListenerResponse response)
    {
        try
        {
            Dictionary<string, JsonElement> parsed = NormalizeFullRunEnvPayload(ParseFullRunEnvRequestObject(request, allowEmptyBody: true));
            WaitForFullRunEnvState(
                predicate: static current => IsMenuReadyForFullRunReset(current) || GetStateType(current) != "menu",
                timeoutMs: GetOptionalInt(parsed, "timeout_ms", 20000),
                pollDelayMs: GetOptionalInt(parsed, "poll_delay_ms", 50));
            Dictionary<string, object?> startResult = RunOnMainThread(() => ExecuteStartRun(parsed)).GetAwaiter().GetResult();
            if (IsErrorResult(startResult, out string? resetError))
                throw new InvalidOperationException(resetError ?? "Failed to start run.");

            Dictionary<string, object?> state = WaitForFullRunEnvState(
                predicate: static current =>
                    GetStateType(current) != "menu"
                    && IsSettledFullRunState(current)
                    && IsActionableOrTerminalFullRunState(current),
                timeoutMs: GetOptionalInt(parsed, "timeout_ms", 20000),
                pollDelayMs: GetOptionalInt(parsed, "poll_delay_ms", 50));
            SendJson(response, state);
        }
        catch (JsonException ex)
        {
            SendError(response, 400, $"Invalid JSON: {ex.Message}");
        }
        catch (InvalidOperationException ex)
        {
            SendError(response, 400, ex.Message);
        }
        catch (Exception ex)
        {
            SendError(response, 500, $"Full run env reset failed: {ex.Message}");
        }
    }

    private static void HandlePostFullRunEnvStep(HttpListenerRequest request, HttpListenerResponse response)
    {
        try
        {
            Dictionary<string, JsonElement> parsed = NormalizeFullRunEnvPayload(ParseFullRunEnvRequestObject(request, allowEmptyBody: false));
            if (!parsed.TryGetValue("action", out JsonElement actionElem) || actionElem.ValueKind != JsonValueKind.String)
                throw new InvalidOperationException("Missing 'action' field.");

            Dictionary<string, object?> beforeState = RunOnMainThread(BuildVisibleFullRunEnvState).GetAwaiter().GetResult();
            string action = actionElem.GetString() ?? string.Empty;
            Dictionary<string, object?> state;
            string? stepInfoCode = null;
            bool accepted;
            string? actionError;

            if (string.Equals(action, "wait", StringComparison.OrdinalIgnoreCase))
            {
                try
                {
                    state = WaitForChangedFullRunEnvState(
                        beforeState,
                        timeoutMs: GetOptionalInt(parsed, "timeout_ms", 2000),
                        pollDelayMs: GetOptionalInt(parsed, "poll_delay_ms", 25));
                }
                catch (TimeoutException)
                {
                    state = RunOnMainThread(BuildVisibleFullRunEnvState).GetAwaiter().GetResult();
                    stepInfoCode = "state_change_timeout";
                }

                accepted = true;
                actionError = null;
            }
            else
            {
                Dictionary<string, object?> actionResult =
                    RunOnMainThread(() => ExecuteAction(action, parsed)).GetAwaiter().GetResult();
                accepted = !IsErrorResult(actionResult, out actionError);

                if (accepted)
                {
                    try
                    {
                        state = WaitForChangedFullRunEnvState(
                            beforeState,
                            timeoutMs: GetOptionalInt(parsed, "timeout_ms", 2000),
                            pollDelayMs: GetOptionalInt(parsed, "poll_delay_ms", 25));
                    }
                    catch (TimeoutException)
                    {
                        state = RunOnMainThread(BuildVisibleFullRunEnvState).GetAwaiter().GetResult();
                        stepInfoCode = "state_change_timeout";
                    }
                }
                else
                {
                    state = RunOnMainThread(BuildVisibleFullRunEnvState).GetAwaiter().GetResult();
                }
            }

            SendJson(response, ShapeFullRunEnvStepResult(state, accepted, actionError, stepInfoCode));
        }
        catch (JsonException ex)
        {
            SendError(response, 400, $"Invalid JSON: {ex.Message}");
        }
        catch (InvalidOperationException ex)
        {
            SendError(response, 400, ex.Message);
        }
        catch (Exception ex)
        {
            SendError(response, 500, $"Full run env step failed: {ex.Message}");
        }
    }

    private static void HandleUnsupportedFullRunEnvMutation(HttpListenerResponse response, string endpoint)
    {
        SendError(
            response,
            501,
            $"{endpoint} is unsupported in spectator mode. Use the Sim backend for save/load or branching operations.",
            errorCode: "unsupported_in_spectator_mode");
    }

    private static Dictionary<string, JsonElement> NormalizeFullRunEnvPayload(Dictionary<string, JsonElement> payload)
    {
        if (!payload.ContainsKey("ascension") && payload.TryGetValue("ascension_level", out JsonElement ascensionLevel))
            payload["ascension"] = ascensionLevel;
        return payload;
    }

    private static Dictionary<string, JsonElement> ParseFullRunEnvRequestObject(HttpListenerRequest request, bool allowEmptyBody)
    {
        string body;
        using (var reader = new StreamReader(request.InputStream, request.ContentEncoding))
            body = reader.ReadToEnd();

        if (string.IsNullOrWhiteSpace(body))
        {
            if (allowEmptyBody)
                return new Dictionary<string, JsonElement>();
            throw new JsonException("Request body must be a JSON object.");
        }

        return JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(body)
            ?? throw new JsonException("Request body must be a JSON object.");
    }

    private static Dictionary<string, object?> WaitForFullRunEnvState(
        Func<Dictionary<string, object?>, bool> predicate,
        int timeoutMs,
        int pollDelayMs)
    {
        DateTime deadline = DateTime.UtcNow.AddMilliseconds(Math.Max(100, timeoutMs));
        int delay = Math.Max(10, pollDelayMs);
        Dictionary<string, object?>? lastState = null;

        while (DateTime.UtcNow <= deadline)
        {
            lastState = RunOnMainThread(BuildVisibleFullRunEnvState).GetAwaiter().GetResult();
            if (predicate(lastState))
                return lastState;
            Thread.Sleep(delay);
        }

        throw new TimeoutException("Timed out waiting for full run env state transition.");
    }

    private static Dictionary<string, object?> WaitForChangedFullRunEnvState(
        Dictionary<string, object?> previousState,
        int timeoutMs,
        int pollDelayMs)
    {
        string previousSignature = GetFullRunStateSignature(previousState);
        string previousStateType = GetStateType(previousState);
        DateTime deadline = DateTime.UtcNow.AddMilliseconds(Math.Max(100, timeoutMs));
        int delay = Math.Max(10, pollDelayMs);
        Dictionary<string, object?>? lastChangedState = null;
        string? lastChangedSignature = null;
        int stablePolls = 0;

        while (DateTime.UtcNow <= deadline)
        {
            Dictionary<string, object?> state = RunOnMainThread(BuildVisibleFullRunEnvState).GetAwaiter().GetResult();
            string signature = GetFullRunStateSignature(state);
            if (!string.Equals(signature, previousSignature, StringComparison.Ordinal))
            {
                lastChangedState = state;

                if (string.Equals(signature, lastChangedSignature, StringComparison.Ordinal))
                    stablePolls++;
                else
                {
                    lastChangedSignature = signature;
                    stablePolls = 1;
                }

                if (IsFullRunTerminalState(state, ExtractFullRunOutcome(state)))
                    return state;

                if (IsSettledFullRunState(state) && IsActionableOrTerminalFullRunState(state))
                {
                    if (ShouldReturnImmediatelyForChangedFullRunState(previousState, previousStateType, state))
                        return state;

                    if (stablePolls >= 2)
                        return state;
                }
            }

            Thread.Sleep(delay);
        }

        if (lastChangedState != null)
            return lastChangedState;

        throw new TimeoutException("Timed out waiting for changed full run env state.");
    }

    private static Dictionary<string, object?> BuildVisibleFullRunEnvState()
    {
        Dictionary<string, object?> state = BuildGameState();
        string? outcome = ExtractFullRunOutcome(state);
        state["legal_actions"] = BuildFullRunLegalActions(state);
        state["run_outcome"] = outcome;
        state["terminal"] = state.TryGetValue("terminal", out object? legacyTerminal) && legacyTerminal is true
            ? true
            : IsFullRunTerminalState(state, outcome);
        state["backend_kind"] = "spectator";
        state["coverage_tier"] = "visible";
        state["is_pure_simulator"] = false;
        return state;
    }

    private static Dictionary<string, object?> ShapeFullRunEnvStepResult(
        Dictionary<string, object?> state,
        bool accepted,
        string? error,
        string? stepInfoCode = null)
    {
        string? outcome = ExtractFullRunOutcome(state);
        bool done = IsFullRunTerminalState(state, outcome);
        double reward = 0.0;
        if (done)
            reward = outcome == "victory" || outcome == "win" ? 1.0 : -1.0;

        return new Dictionary<string, object?>
        {
            ["accepted"] = accepted,
            ["error"] = error,
            ["state"] = state,
            ["reward"] = reward,
            ["done"] = done,
            ["info"] = new Dictionary<string, object?>
            {
                ["state_type"] = GetStateType(state),
                ["run_outcome"] = outcome,
                ["step_info_code"] = stepInfoCode
            }
        };
    }

    private static List<Dictionary<string, object?>> BuildFullRunLegalActions(Dictionary<string, object?> state)
    {
        var actions = new List<Dictionary<string, object?>>();
        string stateType = GetStateType(state);

        switch (stateType)
        {
            case "menu":
                AppendMenuLegalActions(actions, state);
                break;
            case "map":
                AppendIndexedLegalActions(actions, state, "map", "next_options", "choose_map_node");
                break;
            case "combat_rewards":
                AppendIndexedLegalActions(actions, state, "rewards", "items", "claim_reward");
                AppendProceedIfEnabled(actions, state, "rewards");
                break;
            case "card_reward":
                AppendCardRewardLegalActions(actions, state);
                break;
            case "rest_site":
                AppendIndexedLegalActions(actions, state, "rest_site", "options", "choose_rest_option", enabledKey: "is_enabled");
                AppendProceedIfEnabled(actions, state, "rest_site");
                break;
            case "event":
                AppendEventLegalActions(actions, state);
                break;
            case "shop":
                AppendShopLegalActions(actions, state);
                break;
            case "card_select":
                AppendCardSelectLegalActions(actions, state);
                break;
            case "relic_select":
                AppendIndexedLegalActions(actions, state, "relic_select", "relics", "select_relic");
                if (TryGetDict(state, "relic_select", out Dictionary<string, object?> relicSelectState))
                    AppendIfTrue(actions, relicSelectState, "can_skip", new Dictionary<string, object?> { ["action"] = "skip_relic_selection" });
                break;
            case "treasure":
                AppendIndexedLegalActions(actions, state, "treasure", "relics", "claim_treasure_relic");
                AppendProceedIfEnabled(actions, state, "treasure");
                break;
            case "monster":
            case "elite":
            case "boss":
                AppendCombatLegalActions(actions, state);
                break;
            case "hand_select":
                AppendHandSelectLegalActions(actions, state);
                break;
            case "overlay":
            case "game_over":
                AppendOverlayLegalActions(actions, state);
                break;
        }

        return actions;
    }

    private static void AppendMenuLegalActions(List<Dictionary<string, object?>> actions, Dictionary<string, object?> state)
    {
        if (!TryGetDict(state, "menu", out Dictionary<string, object?> menu))
            return;

        foreach (Dictionary<string, object?> character in EnumerateDictionaries(menu.TryGetValue("available_characters", out object? rawChars) ? rawChars : null))
        {
            if (GetBool(character, "is_locked"))
                continue;
            if (TryGetString(character, "id", out string characterId))
            {
                actions.Add(new Dictionary<string, object?>
                {
                    ["action"] = "select_character",
                    ["character_id"] = characterId
                });
            }
        }

        if (GetBool(menu, "character_select_visible"))
        {
            int maxAscension = Math.Max(0, GetInt(menu, "max_ascension", 0));
            for (int ascension = 0; ascension <= maxAscension; ascension++)
            {
                actions.Add(new Dictionary<string, object?>
                {
                    ["action"] = "set_ascension",
                    ["ascension"] = ascension
                });
            }
        }

        if (GetBool(menu, "can_start"))
        {
            var startAction = new Dictionary<string, object?>
            {
                ["action"] = "start_run",
                ["ascension"] = GetInt(menu, "ascension", 0)
            };
            if (TryGetString(menu, "selected_character", out string selectedCharacter))
                startAction["character_id"] = selectedCharacter;
            actions.Add(startAction);
        }
    }

    private static void AppendCardRewardLegalActions(List<Dictionary<string, object?>> actions, Dictionary<string, object?> state)
    {
        if (!TryGetDict(state, "card_reward", out Dictionary<string, object?> rewardState))
            return;

        foreach (Dictionary<string, object?> card in EnumerateDictionaries(rewardState.TryGetValue("cards", out object? rawCards) ? rawCards : null))
        {
            var action = new Dictionary<string, object?>
            {
                ["action"] = "select_card_reward",
                ["index"] = GetInt(card, "index", -1),
                ["card_index"] = GetInt(card, "index", -1)
            };
            if (TryGetString(card, "id", out string cardId))
                action["card_id"] = cardId;
            if (TryGetString(card, "type", out string cardType))
                action["card_type"] = cardType;
            if (TryGetString(card, "rarity", out string cardRarity))
                action["card_rarity"] = cardRarity;
            if (TryGetString(card, "cost", out string cardCost))
                action["cost"] = cardCost;
            if (card.TryGetValue("is_upgraded", out object? isUpgraded) && isUpgraded is bool upgraded)
                action["is_upgraded"] = upgraded;
            if (TryGetString(card, "name", out string cardName))
                action["label"] = cardName;
            actions.Add(action);
        }

        AppendIfTrue(actions, rewardState, "can_skip", new Dictionary<string, object?> { ["action"] = "skip_card_reward" });
    }

    private static void AppendEventLegalActions(List<Dictionary<string, object?>> actions, Dictionary<string, object?> state)
    {
        if (!TryGetDict(state, "event", out Dictionary<string, object?> eventState))
            return;

        if (GetBool(eventState, "in_dialogue"))
        {
            actions.Add(new Dictionary<string, object?> { ["action"] = "advance_dialogue" });
            return;
        }

        foreach (Dictionary<string, object?> option in EnumerateDictionaries(eventState.TryGetValue("options", out object? rawOptions) ? rawOptions : null))
        {
            if (GetBool(option, "is_locked") || GetBool(option, "was_chosen"))
                continue;

            actions.Add(new Dictionary<string, object?>
            {
                ["action"] = "choose_event_option",
                ["index"] = GetInt(option, "index", -1)
            });
        }
    }

    private static void AppendShopLegalActions(List<Dictionary<string, object?>> actions, Dictionary<string, object?> state)
    {
        if (!TryGetDict(state, "shop", out Dictionary<string, object?> shopState))
            return;

        foreach (Dictionary<string, object?> item in EnumerateDictionaries(shopState.TryGetValue("items", out object? rawItems) ? rawItems : null))
        {
            if (!GetBool(item, "is_stocked", defaultValue: true))
                continue;
            if (!GetBool(item, "can_afford", defaultValue: true))
                continue;
            actions.Add(new Dictionary<string, object?>
            {
                ["action"] = "shop_purchase",
                ["index"] = GetInt(item, "index", -1)
            });
        }

        actions.Add(new Dictionary<string, object?> { ["action"] = "proceed" });
    }

    private static void AppendCardSelectLegalActions(List<Dictionary<string, object?>> actions, Dictionary<string, object?> state)
    {
        if (!TryGetDict(state, "card_select", out Dictionary<string, object?> selectState))
            return;

        bool previewShowing = GetBool(selectState, "preview_showing");
        int selectedCount = GetInt(selectState, "selected_count", 0);
        int maxSelect = GetInt(selectState, "max_select", -1);
        bool selectionQuotaReached = maxSelect > 0 && selectedCount >= maxSelect;

        if (!previewShowing && !selectionQuotaReached)
        {
            var selectedIndices = new HashSet<int>();
            foreach (Dictionary<string, object?> selCard in EnumerateDictionaries(selectState.TryGetValue("selected_cards", out object? rawSel) ? rawSel : null))
            {
                int idx = GetInt(selCard, "index", -1);
                if (idx >= 0)
                    selectedIndices.Add(idx);
            }

            foreach (Dictionary<string, object?> card in EnumerateDictionaries(selectState.TryGetValue("cards", out object? rawCards) ? rawCards : null))
            {
                int cardIndex = GetInt(card, "index", -1);
                if (cardIndex < 0 || selectedIndices.Contains(cardIndex))
                    continue;
                actions.Add(new Dictionary<string, object?>
                {
                    ["action"] = "select_card",
                    ["index"] = cardIndex
                });
            }
        }

        AppendIfTrue(actions, selectState, "can_confirm", new Dictionary<string, object?> { ["action"] = "confirm_selection" });
        AppendIfTrue(actions, selectState, "can_cancel", new Dictionary<string, object?> { ["action"] = "cancel_selection" });
    }

    private static void AppendCombatLegalActions(List<Dictionary<string, object?>> actions, Dictionary<string, object?> state)
    {
        if (!TryGetDict(state, "battle", out Dictionary<string, object?> battleState))
            return;
        if (!TryGetDict(battleState, "player", out Dictionary<string, object?> playerState)
            && !TryGetDict(state, "player", out playerState))
            return;

        string turn = GetString(battleState, "turn");
        bool isPlayPhase = GetBool(battleState, "is_play_phase", defaultValue: true);
        if (turn != "player" || !isPlayPhase)
            return;

        List<Dictionary<string, object?>> enemies = EnumerateDictionaries(battleState.TryGetValue("enemies", out object? rawEnemies) ? rawEnemies : null)
            .Where(enemy => GetBool(enemy, "is_alive", defaultValue: true))
            .ToList();

        foreach (Dictionary<string, object?> card in EnumerateDictionaries(playerState.TryGetValue("hand", out object? rawHand) ? rawHand : null))
        {
            if (!GetBool(card, "can_play"))
                continue;

            int handIndex = GetInt(card, "index", -1);
            string targetType = GetString(card, "target_type");
            bool requiresTarget = targetType is "enemy" or "anyenemy" or "any_enemy";
            if (requiresTarget)
            {
                foreach (Dictionary<string, object?> enemy in enemies)
                {
                    int targetId = GetInt(enemy, "combat_id", -1);
                    string? target = TryGetString(enemy, "entity_id", out string entityId)
                        ? entityId
                        : (TryGetString(enemy, "id", out string fallbackId) ? fallbackId : null);
                    if (targetId < 0 && string.IsNullOrWhiteSpace(target))
                        continue;

                    var action = new Dictionary<string, object?>
                    {
                        ["action"] = "play_card",
                        ["card_index"] = handIndex
                    };
                    if (targetId >= 0)
                        action["target_id"] = targetId;
                    if (!string.IsNullOrWhiteSpace(target))
                        action["target"] = target;
                    actions.Add(action);
                }
            }
            else
            {
                actions.Add(new Dictionary<string, object?>
                {
                    ["action"] = "play_card",
                    ["card_index"] = handIndex
                });
            }
        }

        foreach (Dictionary<string, object?> potion in EnumerateDictionaries(playerState.TryGetValue("potions", out object? rawPotions) ? rawPotions : null))
        {
            if (!GetBool(potion, "can_use_in_combat", defaultValue: true))
                continue;

            int slot = GetInt(potion, "slot", -1);
            string targetType = GetString(potion, "target_type");
            bool requiresTarget = targetType is "enemy" or "anyenemy" or "any_enemy";
            if (requiresTarget)
            {
                foreach (Dictionary<string, object?> enemy in enemies)
                {
                    int targetId = GetInt(enemy, "combat_id", -1);
                    string? target = TryGetString(enemy, "entity_id", out string entityId)
                        ? entityId
                        : (TryGetString(enemy, "id", out string fallbackId) ? fallbackId : null);
                    if (targetId < 0 && string.IsNullOrWhiteSpace(target))
                        continue;

                    var action = new Dictionary<string, object?>
                    {
                        ["action"] = "use_potion",
                        ["slot"] = slot
                    };
                    if (targetId >= 0)
                        action["target_id"] = targetId;
                    if (!string.IsNullOrWhiteSpace(target))
                        action["target"] = target;
                    actions.Add(action);
                }
            }
            else
            {
                actions.Add(new Dictionary<string, object?>
                {
                    ["action"] = "use_potion",
                    ["slot"] = slot
                });
            }
        }

        actions.Add(new Dictionary<string, object?> { ["action"] = "end_turn" });
    }

    private static void AppendHandSelectLegalActions(List<Dictionary<string, object?>> actions, Dictionary<string, object?> state)
    {
        if (!TryGetDict(state, "hand_select", out Dictionary<string, object?> handSelectState))
            return;

        foreach (Dictionary<string, object?> card in EnumerateDictionaries(handSelectState.TryGetValue("cards", out object? rawCards) ? rawCards : null))
        {
            actions.Add(new Dictionary<string, object?>
            {
                ["action"] = "combat_select_card",
                ["card_index"] = GetInt(card, "index", -1)
            });
        }

        AppendIfTrue(actions, handSelectState, "can_confirm", new Dictionary<string, object?> { ["action"] = "combat_confirm_selection" });
    }

    private static void AppendOverlayLegalActions(List<Dictionary<string, object?>> actions, Dictionary<string, object?> state)
    {
        string containerKey = GetStateType(state) == "game_over" ? "game_over" : "overlay";
        if (!TryGetDict(state, containerKey, out Dictionary<string, object?> overlayState))
            return;

        foreach (Dictionary<string, object?> button in EnumerateDictionaries(overlayState.TryGetValue("buttons", out object? rawButtons) ? rawButtons : null))
        {
            if (!GetBool(button, "is_enabled", defaultValue: true))
                continue;

            actions.Add(new Dictionary<string, object?>
            {
                ["action"] = "overlay_press",
                ["index"] = GetInt(button, "index", -1)
            });
        }
    }

    private static void AppendIndexedLegalActions(
        List<Dictionary<string, object?>> actions,
        Dictionary<string, object?> state,
        string containerKey,
        string collectionKey,
        string actionName,
        string enabledKey = "")
    {
        if (!TryGetDict(state, containerKey, out Dictionary<string, object?> container))
            return;

        foreach (Dictionary<string, object?> item in EnumerateDictionaries(container.TryGetValue(collectionKey, out object? rawItems) ? rawItems : null))
        {
            if (!string.IsNullOrWhiteSpace(enabledKey) && !GetBool(item, enabledKey, defaultValue: true))
                continue;
            actions.Add(new Dictionary<string, object?>
            {
                ["action"] = actionName,
                ["index"] = GetInt(item, "index", -1)
            });
        }
    }

    private static void AppendProceedIfEnabled(List<Dictionary<string, object?>> actions, Dictionary<string, object?> state, string containerKey)
    {
        if (!TryGetDict(state, containerKey, out Dictionary<string, object?> container))
            return;
        AppendIfTrue(actions, container, "can_proceed", new Dictionary<string, object?> { ["action"] = "proceed" });
    }

    private static void AppendIfTrue(List<Dictionary<string, object?>> actions, Dictionary<string, object?> state, string key, Dictionary<string, object?> action)
    {
        if (GetBool(state, key))
            actions.Add(action);
    }

    private static string GetStateType(Dictionary<string, object?> state)
    {
        return state.TryGetValue("state_type", out object? raw)
            ? (raw?.ToString() ?? string.Empty).Trim().ToLowerInvariant()
            : string.Empty;
    }

    private static int GetFullRunAct(Dictionary<string, object?> state)
    {
        return TryGetDict(state, "run", out Dictionary<string, object?> runState) ? GetInt(runState, "act", 0) : 0;
    }

    private static int GetFullRunFloor(Dictionary<string, object?> state)
    {
        return TryGetDict(state, "run", out Dictionary<string, object?> runState) ? GetInt(runState, "floor", 0) : 0;
    }

    private static bool IsSettledFullRunState(Dictionary<string, object?> state)
    {
        string stateType = GetStateType(state);
        return stateType is not "" and not "unknown" and not "menu";
    }

    private static bool IsMenuReadyForFullRunReset(Dictionary<string, object?> state)
    {
        return GetStateType(state) == "menu"
            && TryGetDict(state, "menu", out Dictionary<string, object?> menuState)
            && GetBool(menuState, "is_main_menu_visible", defaultValue: false);
    }

    private static bool ShouldReturnImmediatelyForChangedFullRunState(
        Dictionary<string, object?> previousState,
        string previousStateType,
        Dictionary<string, object?> state)
    {
        string stateType = GetStateType(state);
        if (!string.Equals(stateType, previousStateType, StringComparison.Ordinal))
            return true;

        if (GetFullRunAct(state) != GetFullRunAct(previousState) || GetFullRunFloor(state) != GetFullRunFloor(previousState))
            return true;

        return !RequiresStableFullRunPoll(previousStateType) && !RequiresStableFullRunPoll(stateType);
    }

    private static bool RequiresStableFullRunPoll(string stateType)
    {
        return stateType is "monster" or "elite" or "boss" or "hand_select";
    }

    private static bool IsActionableOrTerminalFullRunState(Dictionary<string, object?> state)
    {
        if (IsFullRunTerminalState(state, ExtractFullRunOutcome(state)))
            return true;

        if (!state.TryGetValue("legal_actions", out object? rawActions) || rawActions is not IEnumerable enumerable || rawActions is string)
            return false;

        foreach (object? item in enumerable)
        {
            if (item is Dictionary<string, object?>)
                return true;
        }

        return false;
    }

    private static string GetFullRunStateSignature(Dictionary<string, object?> state)
    {
        return JsonSerializer.Serialize(state, _jsonOptions);
    }

    private static bool IsFullRunTerminalState(Dictionary<string, object?> state, string? outcome)
    {
        string stateType = GetStateType(state);
        return stateType == "game_over"
            || string.Equals(outcome, "victory", StringComparison.OrdinalIgnoreCase)
            || string.Equals(outcome, "death", StringComparison.OrdinalIgnoreCase)
            || string.Equals(outcome, "win", StringComparison.OrdinalIgnoreCase)
            || string.Equals(outcome, "loss", StringComparison.OrdinalIgnoreCase);
    }

    private static string? ExtractFullRunOutcome(Dictionary<string, object?> state)
    {
        if (TryGetString(state, "run_outcome", out string runOutcome))
            return runOutcome;

        if (TryGetDict(state, "game_over", out Dictionary<string, object?> gameOverState) && TryGetString(gameOverState, "outcome", out string outcome))
            return outcome;

        return null;
    }

    private static bool IsErrorResult(Dictionary<string, object?> result, out string? error)
    {
        error = null;
        if (result.TryGetValue("status", out object? statusRaw)
            && string.Equals(statusRaw?.ToString(), "error", StringComparison.OrdinalIgnoreCase))
        {
            error = result.TryGetValue("error", out object? errorRaw) ? errorRaw?.ToString() : "Unknown error";
            return true;
        }

        return false;
    }

    private static bool TryGetDict(Dictionary<string, object?> state, string key, out Dictionary<string, object?> dict)
    {
        if (state.TryGetValue(key, out object? raw) && raw is Dictionary<string, object?> typed)
        {
            dict = typed;
            return true;
        }

        dict = null!;
        return false;
    }

    private static IEnumerable<Dictionary<string, object?>> EnumerateDictionaries(object? value)
    {
        if (value is not IEnumerable enumerable || value is string)
            yield break;

        foreach (object? item in enumerable)
        {
            if (item is Dictionary<string, object?> typed)
                yield return typed;
        }
    }

    private static bool TryGetString(Dictionary<string, object?> state, string key, out string text)
    {
        if (state.TryGetValue(key, out object? raw) && raw != null)
        {
            text = raw.ToString() ?? string.Empty;
            return !string.IsNullOrWhiteSpace(text);
        }

        text = string.Empty;
        return false;
    }

    private static string GetString(Dictionary<string, object?> state, string key)
    {
        return TryGetString(state, key, out string text) ? text.Trim().ToLowerInvariant() : string.Empty;
    }

    private static int GetInt(Dictionary<string, object?> state, string key, int defaultValue)
    {
        if (!state.TryGetValue(key, out object? raw) || raw == null)
            return defaultValue;
        return raw switch
        {
            int value => value,
            long value => (int)value,
            uint value => (int)value,
            JsonElement elem when elem.ValueKind == JsonValueKind.Number && elem.TryGetInt32(out int value) => value,
            _ when int.TryParse(raw.ToString(), out int parsed) => parsed,
            _ => defaultValue
        };
    }

    private static int GetOptionalInt(Dictionary<string, JsonElement> payload, string key, int defaultValue)
    {
        if (payload.TryGetValue(key, out JsonElement elem) && elem.ValueKind == JsonValueKind.Number)
            return elem.GetInt32();
        return defaultValue;
    }

    private static bool GetBool(Dictionary<string, object?> state, string key, bool defaultValue = false)
    {
        if (!state.TryGetValue(key, out object? raw) || raw == null)
            return defaultValue;
        return raw switch
        {
            bool value => value,
            JsonElement elem when elem.ValueKind == JsonValueKind.True => true,
            JsonElement elem when elem.ValueKind == JsonValueKind.False => false,
            _ when bool.TryParse(raw.ToString(), out bool parsed) => parsed,
            _ => defaultValue
        };
    }
}
