using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Google.Protobuf;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Simulation;
using MegaCrit.Sts2.Core.Training;
using STS2AI.Bridge;
using STS2AI.Bridge.Runtime;

namespace STS2_MCP;

public static partial class McpMod
{
    private static void HandleBridgeRpc(HttpListenerRequest request, HttpListenerResponse response)
    {
        string body;
        using (var reader = new StreamReader(request.InputStream, request.ContentEncoding))
        {
            body = reader.ReadToEnd();
        }

        BridgeRequestEnvelope envelope;
        try
        {
            envelope = JsonParser.Default.Parse<BridgeRequestEnvelope>(body);
        }
        catch (Exception ex)
        {
            response.StatusCode = 400;
            SendBridgeJson(response, BuildBridgeError(BridgeMethod.State, BridgeStatus.ProtocolError, "invalid_json", ex.Message));
            return;
        }

        BridgeResponseEnvelope result;
        try
        {
            result = ProcessBridgeRequest(envelope);
        }
        catch (Exception ex)
        {
            result = BuildBridgeError(
                envelope.Method,
                GetBridgeErrorStatus(ex),
                GetStructuredErrorCode(ex) ?? "internal_error",
                ex.Message);
        }

        SendBridgeJson(response, result);
    }

    private static BridgeResponseEnvelope ProcessBridgeRequest(BridgeRequestEnvelope envelope)
    {
        return BridgeRpcDispatcher.DispatchAsync(new SpectatorBridgeRuntime(), envelope).GetAwaiter().GetResult();
    }

    private static BridgeResponseEnvelope ProcessBridgeReset(BridgeRequestEnvelope envelope, bool forceCombatReset = false)
    {
        Dictionary<string, JsonElement> parsed = BuildResetPayload(envelope, forceCombatReset);
        bool combatReset = HasEncounterReset(parsed);
        Dictionary<string, object?> initialState = RunOnMainThread(BuildVisibleFullRunEnvState).GetAwaiter().GetResult();
        if (!combatReset && GetStateType(initialState) != "menu")
        {
            RunOnMainThread(async () =>
            {
                await MegaCrit.Sts2.Core.Nodes.NGame.Instance.ReturnToMainMenuAfterRun();
                return 0;
            }).GetAwaiter().GetResult();
        }
        if (!combatReset)
        {
            WaitForFullRunEnvState(
                predicate: static current => IsMenuReadyForFullRunReset(current),
                timeoutMs: GetOptionalInt(parsed, "timeout_ms", 20000),
                pollDelayMs: GetOptionalInt(parsed, "poll_delay_ms", 50));
        }

        Dictionary<string, object?> startResult = combatReset
            ? RunOnMainThread(() => ExecuteStartVisibleCombatAsync(parsed)).GetAwaiter().GetResult().GetAwaiter().GetResult()
            : RunOnMainThread(() => ExecuteStartRun(parsed)).GetAwaiter().GetResult();
        if (IsErrorResult(startResult, out string? resetError))
        {
            throw new InvalidOperationException(resetError ?? (combatReset ? "Failed to start combat." : "Failed to start run."));
        }

        WaitForFullRunEnvState(
            predicate: static current =>
                GetStateType(current) != "menu"
                && IsSettledFullRunState(current)
                && IsActionableOrTerminalFullRunState(current),
            timeoutMs: GetOptionalInt(parsed, "timeout_ms", 20000),
            pollDelayMs: GetOptionalInt(parsed, "poll_delay_ms", 50));
        WaitForBridgeGameState(
            predicate: static current => IsBridgeResetStateReady(current),
            timeoutMs: GetOptionalInt(parsed, "timeout_ms", 20000),
            pollDelayMs: GetOptionalInt(parsed, "poll_delay_ms", 50));

        return BuildBridgeStateResponse(envelope.Method);
    }

    private static BridgeResponseEnvelope ProcessBridgeAct(BridgeRequestEnvelope envelope, BridgeMethod responseMethod)
    {
        if (envelope.PayloadCase is not BridgeRequestEnvelope.PayloadOneofCase.Act
            and not BridgeRequestEnvelope.PayloadOneofCase.CombatAct)
        {
            throw new InvalidOperationException("act request missing payload.");
        }

        LegalAction actionPayload = envelope.PayloadCase == BridgeRequestEnvelope.PayloadOneofCase.CombatAct
            ? envelope.CombatAct.Action
            : envelope.Act.Action;
        Dictionary<string, JsonElement> parsed = BuildActionPayload(actionPayload);
        GameState beforeState = RunOnMainThread(BuildBridgeGameState).GetAwaiter().GetResult();
        string action = actionPayload.Action ?? "";
        string? stepInfoCode = null;
        bool accepted;
        string? actionError;

        if (string.Equals(action, "wait", StringComparison.OrdinalIgnoreCase))
        {
            try
            {
                WaitForChangedBridgeGameState(
                    beforeState,
                    timeoutMs: GetOptionalInt(parsed, "timeout_ms", 2000),
                    pollDelayMs: GetOptionalInt(parsed, "poll_delay_ms", 25));
            }
            catch (TimeoutException)
            {
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
                    GameState state = WaitForChangedBridgeGameState(
                        beforeState,
                        timeoutMs: GetOptionalInt(parsed, "timeout_ms", 2000),
                        pollDelayMs: GetOptionalInt(parsed, "poll_delay_ms", 25));

                    if (string.Equals(action, "choose_event_option", StringComparison.OrdinalIgnoreCase)
                        && HasTransientChosenBridgeEventOption(state))
                    {
                        try
                        {
                            WaitForBridgeGameState(
                                predicate: static current =>
                                    !HasTransientChosenBridgeEventOption(current)
                                    && IsBridgeStepStateSettled(current)
                                    && (current.Terminal
                                        || current.LegalActions.Count > 0
                                        || current.Event?.IsFinished == true),
                                timeoutMs: GetOptionalInt(parsed, "timeout_ms", 2000),
                                pollDelayMs: GetOptionalInt(parsed, "poll_delay_ms", 25));
                        }
                        catch (TimeoutException)
                        {
                            stepInfoCode ??= "event_settle_timeout";
                        }
                    }
                }
                catch (TimeoutException)
                {
                    stepInfoCode = "state_change_timeout";
                }
            }
        }

        GameState gameState = RunOnMainThread(BuildBridgeGameState).GetAwaiter().GetResult();
        return new BridgeResponseEnvelope
        {
            Method = responseMethod,
            Status = accepted ? BridgeStatus.Ok : BridgeStatus.RejectedAction,
            Act = new BridgeActPayload
            {
                Accepted = accepted,
                Error = actionError ?? stepInfoCode ?? "",
                State = gameState,
            }
        };
    }

    private static BridgeResponseEnvelope ProcessBridgeSkipCombat()
    {
        Dictionary<string, object?> beforeState = RunOnMainThread(BuildVisibleFullRunEnvState).GetAwaiter().GetResult();
        string stateType = GetStateType(beforeState);
        if (stateType != "monster" && stateType != "elite" && stateType != "boss" && stateType != "combat")
        {
            return new BridgeResponseEnvelope
            {
                Method = BridgeMethod.SkipCombat,
                Status = BridgeStatus.Ok,
                Act = new BridgeActPayload
                {
                    Accepted = true,
                    Error = "not_in_combat",
                    State = RunOnMainThread(BuildBridgeGameState).GetAwaiter().GetResult(),
                }
            };
        }

        RunOnMainThread(async () =>
        {
            await MegaCrit.Sts2.Core.Combat.CombatManager.Instance.EndCombatInternal();
            return 0;
        }).GetAwaiter().GetResult();

        try
        {
            WaitForChangedFullRunEnvState(beforeState, timeoutMs: 5000, pollDelayMs: 50);
        }
        catch (TimeoutException)
        {
        }

        return new BridgeResponseEnvelope
        {
            Method = BridgeMethod.SkipCombat,
            Status = BridgeStatus.Ok,
            Act = new BridgeActPayload
            {
                Accepted = true,
                Error = "combat_skipped",
                State = RunOnMainThread(BuildBridgeGameState).GetAwaiter().GetResult(),
            }
        };
    }

    private static BridgeResponseEnvelope BuildBridgeStateResponse(BridgeMethod method)
    {
        return new BridgeResponseEnvelope
        {
            Method = method,
            Status = BridgeStatus.Ok,
            State = new BridgeStatePayload
            {
                State = RunOnMainThread(BuildBridgeGameState).GetAwaiter().GetResult(),
            }
        };
    }

    private static GameState BuildBridgeGameState()
    {
        CombatTrainingStateSnapshot? combatSnapshot = null;
        if (CombatManager.Instance.IsInProgress || CombatManager.Instance.DebugOnlyGetState() != null)
        {
            combatSnapshot = BridgeCombatSnapshotBuilder.BuildStateSnapshot();
        }

        FullRunSimulationStateSnapshot snapshot = FullRunSimulationStateBuilder.Build(
            RunManager.Instance.DebugOnlyGetState(),
            FullRunSimulationChoiceBridge.Instance,
            isPureSimulator: false,
            backendKind: "spectator",
            coverageTier: "visible",
            forceMapView: false,
            cachedCombatState: combatSnapshot);

        return BridgeGameStateBuilder.FromFullRunSnapshot(snapshot);
    }

    private static string GetBridgeGameStateSignature(GameState state)
    {
        return Convert.ToBase64String(state.ToByteArray());
    }

    private static GameState WaitForBridgeGameState(
        Func<GameState, bool> predicate,
        int timeoutMs,
        int pollDelayMs)
    {
        DateTime deadline = DateTime.UtcNow.AddMilliseconds(Math.Max(100, timeoutMs));
        int delay = Math.Max(10, pollDelayMs);
        GameState? lastState = null;

        while (DateTime.UtcNow <= deadline)
        {
            bool busy = RunOnMainThread(() => IsActionExecutorBusy()).GetAwaiter().GetResult();
            if (busy)
            {
                Thread.Sleep(delay);
                continue;
            }

            GameState state = RunOnMainThread(BuildBridgeGameState).GetAwaiter().GetResult();
            lastState = state;
            if (IsBridgeEventDialogueOnly(state))
            {
                AdvanceDialogueIfNeeded();
                Thread.Sleep(delay);
                continue;
            }

            if (predicate(state))
            {
                return state;
            }

            Thread.Sleep(delay);
        }

        if (lastState != null)
        {
            return lastState;
        }

        throw new TimeoutException("Timed out waiting for bridge GameState.");
    }

    private static GameState WaitForChangedBridgeGameState(
        GameState previousState,
        int timeoutMs,
        int pollDelayMs)
    {
        string previousSignature = GetBridgeGameStateSignature(previousState);
        DateTime deadline = DateTime.UtcNow.AddMilliseconds(Math.Max(100, timeoutMs));
        int delay = Math.Max(10, pollDelayMs);
        GameState? lastChangedState = null;
        string? lastChangedSignature = null;
        int stablePolls = 0;

        while (DateTime.UtcNow <= deadline)
        {
            bool busy = RunOnMainThread(() => IsActionExecutorBusy()).GetAwaiter().GetResult();
            if (busy)
            {
                Thread.Sleep(delay);
                continue;
            }

            GameState state = RunOnMainThread(BuildBridgeGameState).GetAwaiter().GetResult();
            string signature = GetBridgeGameStateSignature(state);
            if (!string.Equals(signature, previousSignature, StringComparison.Ordinal))
            {
                lastChangedState = state;

                if (IsBridgeEventDialogueOnly(state))
                {
                    AdvanceDialogueIfNeeded();
                    Thread.Sleep(delay);
                    continue;
                }
                if (HasTransientChosenBridgeEventOption(state))
                {
                    Thread.Sleep(delay);
                    continue;
                }

                if (string.Equals(signature, lastChangedSignature, StringComparison.Ordinal))
                {
                    stablePolls++;
                }
                else
                {
                    lastChangedSignature = signature;
                    stablePolls = 1;
                }

                if (state.Terminal)
                {
                    return state;
                }

                bool settled = !string.IsNullOrEmpty(state.StateType)
                    && state.StateType != "unknown"
                    && state.StateType != "menu"
                    && state.StateType != "loading";
                bool actionable = state.Terminal || state.LegalActions.Count > 0;

                if (settled && actionable && stablePolls >= 2 && !IsBridgeEventDialogueOnly(state))
                {
                    return state;
                }
            }

            Thread.Sleep(delay);
        }

        if (lastChangedState != null)
        {
            return lastChangedState;
        }

        throw new TimeoutException("Timed out waiting for changed bridge GameState.");
    }

    private static bool IsBridgeStepStateSettled(GameState state)
    {
        if (IsActionExecutorBusy())
        {
            return false;
        }
        if (IsBridgeEventDialogueOnly(state))
        {
            return false;
        }
        if (HasTransientChosenBridgeEventOption(state))
        {
            return false;
        }
        if (!IsBridgeCombatLike(state.StateType))
        {
            return true;
        }
        return !IsCombatPresentationBusy() && HasBridgeCombatInputSnapshotReady(state);
    }

    private static bool IsBridgeResetStateReady(GameState state)
    {
        if (state.Terminal)
        {
            return true;
        }
        if (string.IsNullOrEmpty(state.StateType)
            || state.StateType == "unknown"
            || state.StateType == "menu"
            || state.StateType == "loading")
        {
            return false;
        }
        if (state.LegalActions.Count == 0)
        {
            return false;
        }
        return IsBridgeStepStateSettled(state);
    }

    private static bool HasBridgeCombatInputSnapshotReady(GameState state)
    {
        if (!IsBridgeCombatLike(state.StateType))
        {
            return true;
        }
        if (state.Battle == null)
        {
            return false;
        }
        if (state.Battle.RoundNumber <= 0)
        {
            return false;
        }
        bool hasPlayCardAction = state.LegalActions.Any(action =>
            string.Equals(action.Action, "play_card", StringComparison.Ordinal));
        if (hasPlayCardAction && state.Battle.Hand.Count == 0)
        {
            return false;
        }
        return true;
    }

    private static bool IsBridgeEventDialogueOnly(GameState state)
    {
        if (!string.Equals(state.StateType, "event", StringComparison.Ordinal))
        {
            return false;
        }
        if (state.Event == null || !state.Event.InDialogue)
        {
            return false;
        }
        return state.LegalActions.Count == 1
            && string.Equals(state.LegalActions[0].Action, "advance_dialogue", StringComparison.Ordinal);
    }

    private static bool HasTransientChosenBridgeEventOption(GameState state)
    {
        if (!string.Equals(state.StateType, "event", StringComparison.Ordinal) || state.Event == null)
        {
            return false;
        }
        return state.Event.Options.Any(static option => option.IsChosen && !option.IsProceed);
    }

    private static bool IsBridgeCombatLike(string? stateType)
    {
        return stateType is "monster" or "elite" or "boss" or "combat" or "hand_select";
    }

    private static BridgeResponseEnvelope BuildBridgeError(BridgeMethod method, BridgeStatus status, string code, string message)
    {
        return new BridgeResponseEnvelope
        {
            Method = method,
            Status = status,
            Error = new BridgeError
            {
                ErrorCode = code,
                ErrorMessage = message,
            }
        };
    }

    private static BridgeStatus GetBridgeErrorStatus(Exception exception)
    {
        return exception is JsonException or InvalidProtocolBufferException or InvalidOperationException
            ? BridgeStatus.ProtocolError
            : BridgeStatus.SimulatorError;
    }

    private static Dictionary<string, JsonElement> BuildResetPayload(BridgeRequestEnvelope envelope, bool forceCombatReset)
    {
        var raw = new Dictionary<string, object?>();
        if (envelope.PayloadCase == BridgeRequestEnvelope.PayloadOneofCase.CombatReset)
        {
            CombatResetRequest reset = envelope.CombatReset;
            raw["character_id"] = reset.CharacterId;
            raw["encounter_id"] = reset.EncounterId;
            raw["ascension"] = reset.AscensionLevel;
            raw["seed"] = reset.Seed;
            raw["build"] = BuildSpecToJsonObject(reset.Build);
        }
        else
        {
            if (envelope.PayloadCase != BridgeRequestEnvelope.PayloadOneofCase.Reset)
            {
                throw new InvalidOperationException("reset request missing payload.");
            }
            BridgeResetPayload reset = envelope.Reset;
            raw["character_id"] = reset.CharacterId;
            raw["ascension"] = reset.AscensionLevel;
            raw["seed"] = reset.Seed;
            if (!string.IsNullOrWhiteSpace(reset.BuildJson))
            {
                using JsonDocument buildDoc = JsonDocument.Parse(reset.BuildJson);
                raw["build"] = buildDoc.RootElement.Clone();
            }
        }
        if (forceCombatReset && !raw.ContainsKey("encounter_id"))
        {
            raw["encounter_id"] = "";
        }
        raw["timeout_ms"] = 20000;
        raw["poll_delay_ms"] = 50;
        return ToJsonElementDictionary(raw);
    }

    private static Dictionary<string, JsonElement> BuildActionPayload(LegalAction action)
    {
        var raw = new Dictionary<string, object?>
        {
            ["action"] = action.Action ?? "",
            ["timeout_ms"] = 2000,
            ["poll_delay_ms"] = 25,
        };
        if (action.Index >= 0) raw["index"] = action.Index;
        if (action.CardIndex >= 0) raw["card_index"] = action.CardIndex;
        if (action.TargetId >= 0) raw["target_id"] = action.TargetId;
        if (action.Col >= 0) raw["col"] = action.Col;
        if (action.Row >= 0) raw["row"] = action.Row;
        if (action.Slot >= 0) raw["slot"] = action.Slot;
        if (!string.IsNullOrWhiteSpace(action.Label)) raw["value"] = action.Label;
        return ToJsonElementDictionary(raw);
    }

    private static Dictionary<string, JsonElement> ToJsonElementDictionary(Dictionary<string, object?> raw)
    {
        var result = new Dictionary<string, JsonElement>();
        foreach (KeyValuePair<string, object?> item in raw)
        {
            if (item.Value == null)
            {
                continue;
            }
            result[item.Key] = JsonSerializer.SerializeToElement(item.Value, _jsonOptions);
        }
        return NormalizeFullRunEnvPayload(result);
    }

    private static object? BuildSpecToJsonObject(BuildSpec? build)
    {
        if (build == null)
        {
            return null;
        }

        var raw = new Dictionary<string, object?>();
        if (build.Deck.Count > 0)
        {
            var deck = new List<Dictionary<string, object?>>();
            foreach (CardSpec card in build.Deck)
            {
                deck.Add(new Dictionary<string, object?>
                {
                    ["id"] = card.Id,
                    ["upgrade_level"] = card.UpgradeLevel,
                });
            }
            raw["deck"] = deck;
        }
        if (build.Relics.Count > 0)
        {
            var relics = new List<Dictionary<string, object?>>();
            foreach (RelicSpec relic in build.Relics)
            {
                relics.Add(new Dictionary<string, object?> { ["id"] = relic.Id });
            }
            raw["relics"] = relics;
        }
        if (build.Potions.Count > 0)
        {
            var potions = new List<Dictionary<string, object?>>();
            foreach (PotionSpec potion in build.Potions)
            {
                potions.Add(new Dictionary<string, object?>
                {
                    ["id"] = potion.Id,
                    ["slot"] = potion.Slot,
                });
            }
            raw["potions"] = potions;
        }
        if (build.HasCurrentHp) raw["current_hp"] = build.CurrentHp;
        if (build.HasMaxHp) raw["max_hp"] = build.MaxHp;
        if (build.HasMaxEnergy) raw["max_energy"] = build.MaxEnergy;
        if (build.HasGold) raw["gold"] = build.Gold;
        if (build.HasMaxPotionSlots) raw["max_potion_slots"] = build.MaxPotionSlots;
        return raw;
    }

    private static void SendBridgeJson(HttpListenerResponse response, BridgeResponseEnvelope envelope)
    {
        string json = JsonFormatter.Default.Format(envelope);
        byte[] buffer = Encoding.UTF8.GetBytes(json);
        response.ContentType = "application/json; charset=utf-8";
        response.ContentLength64 = buffer.Length;
        response.OutputStream.Write(buffer, 0, buffer.Length);
        response.OutputStream.Close();
    }
}
