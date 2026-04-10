using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text.Json;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Models;
using System.Threading;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Simulation;

namespace STS2_MCP;

public static partial class McpMod
{
    private static void HandleGetFullRunEnvState(HttpListenerResponse response)
    {
        try
        {
            var stateTask = RunOnMainThread(BuildFullRunEnvState);
            var state = stateTask.GetAwaiter().GetResult();
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
            var parsed = ParseFullRunEnvRequestObject(request, allowEmptyBody: true);

            if (IsFullRunSimulatorActive())
            {
                var resetReq = ParseSimResetRequest(parsed);
                // Chain ResetAsync + BuildFullRunEnvState in one main-thread call
                var outerTask = RunOnMainThreadAsync(() =>
                {
                    var resetTask = FullRunTrainingEnvService.Instance.ResetAsync(resetReq);
                    return resetTask.ContinueWith(_ => BuildFullRunEnvState(),
                        TaskContinuationOptions.ExecuteSynchronously);
                });
                var state = outerTask.GetAwaiter().GetResult();
                SendJson(response, state);
                return;
            }

            var startResultTask = RunOnMainThread(() => ExecuteStartRun(parsed));
            var startResult = startResultTask.GetAwaiter().GetResult();
            if (IsErrorResult(startResult, out var resetError))
                throw new InvalidOperationException(resetError ?? "Failed to start run.");

            var legacyState = WaitForFullRunEnvState(
                predicate: static state => GetStateType(state) != "menu" && IsSettledFullRunState(state) && IsActionableOrTerminalFullRunState(state),
                timeoutMs: GetOptionalInt(parsed, "timeout_ms", 20000),
                pollDelayMs: GetOptionalInt(parsed, "poll_delay_ms", 50));
            SendJson(response, legacyState);
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
            var parsed = ParseFullRunEnvRequestObject(request, allowEmptyBody: false);
            if (!parsed.TryGetValue("action", out var actionElem) || actionElem.ValueKind != JsonValueKind.String)
                throw new InvalidOperationException("Missing 'action' field.");

            if (IsFullRunSimulatorActive())
            {
                var actionReq = ParseSimActionRequest(parsed);
                // Chain StepAsync + BuildFullRunEnvState in a single main-thread call.
                // Using ContinueWith(ExecuteSynchronously) avoids a separate
                // RunOnMainThread enqueue for BuildFullRunEnvState, saving ~1 frame.
                // NOTE: Do NOT use an async lambda wrapper — that adds a SyncContext
                // hop and costs an extra frame (see CLAUDE.md §2.3).
                var outerTask = RunOnMainThreadAsync(() =>
                {
                    var stepTask = FullRunTrainingEnvService.Instance.StepAsync(actionReq);
                    return stepTask.ContinueWith(t =>
                    {
                        var simResult = t.Result;
                        var state = BuildFullRunEnvState();
                        return ShapeFullRunEnvStepResult(state, simResult.Accepted, simResult.Error, null);
                    }, TaskContinuationOptions.ExecuteSynchronously);
                });
                // Unwrap Task<Task<T>> → Task<T>
                var result = outerTask.GetAwaiter().GetResult();
                SendJson(response, result);
                return;
            }

            var beforeStateTask = RunOnMainThread(BuildFullRunEnvState);
            var beforeState = beforeStateTask.GetAwaiter().GetResult();

            var action = actionElem.GetString() ?? string.Empty;
            Dictionary<string, object?> state2;
            string? stepInfoCode = null;
            bool accepted;
            string? actionError;

            if (string.Equals(action, "wait", StringComparison.OrdinalIgnoreCase))
            {
                try
                {
                    state2 = WaitForChangedFullRunEnvState(
                        beforeState,
                        timeoutMs: GetOptionalInt(parsed, "timeout_ms", 2000),
                        pollDelayMs: GetOptionalInt(parsed, "poll_delay_ms", 25));
                }
                catch (TimeoutException)
                {
                    state2 = RunOnMainThread(BuildFullRunEnvState).GetAwaiter().GetResult();
                    stepInfoCode = "state_change_timeout";
                }
                accepted = true;
                actionError = null;
            }
            else
            {
                var resultTask = RunOnMainThread(() => ExecuteAction(action, parsed));
                var actionResult = resultTask.GetAwaiter().GetResult();

                accepted = !IsErrorResult(actionResult, out actionError);
                if (accepted)
                {
                    try
                    {
                        state2 = WaitForChangedFullRunEnvState(
                            beforeState,
                            timeoutMs: GetOptionalInt(parsed, "timeout_ms", 2000),
                            pollDelayMs: GetOptionalInt(parsed, "poll_delay_ms", 25));
                    }
                    catch (TimeoutException)
                    {
                        state2 = RunOnMainThread(BuildFullRunEnvState).GetAwaiter().GetResult();
                        stepInfoCode = "state_change_timeout";
                    }
                }
                else
                {
                    state2 = RunOnMainThread(BuildFullRunEnvState).GetAwaiter().GetResult();
                }
            }

            SendJson(response, ShapeFullRunEnvStepResult(state2, accepted, actionError, stepInfoCode));
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

    private static void HandlePostFullRunEnvBatchStep(HttpListenerRequest request, HttpListenerResponse response)
    {
        try
        {
            // Parse JSON body: { "actions": [ {action, index, ...}, ... ] }
            string body;
            using (var reader = new StreamReader(request.InputStream, request.ContentEncoding))
                body = reader.ReadToEnd();

            if (string.IsNullOrWhiteSpace(body))
                throw new InvalidOperationException("Empty request body.");

            using var doc = JsonDocument.Parse(body);
            var root = doc.RootElement;

            if (!root.TryGetProperty("actions", out var actionsElem) || actionsElem.ValueKind != JsonValueKind.Array)
                throw new InvalidOperationException("Missing 'actions' array field.");

            // Parse each action in the array
            var actionList = new List<FullRunSimulationActionRequest>();
            foreach (var actionJson in actionsElem.EnumerateArray())
            {
                if (actionJson.ValueKind != JsonValueKind.Object)
                    continue;
                // Convert JsonElement to Dictionary<string, JsonElement> for ParseSimActionRequest
                var dict = new Dictionary<string, JsonElement>();
                foreach (var prop in actionJson.EnumerateObject())
                    dict[prop.Name] = prop.Value;
                actionList.Add(ParseSimActionRequest(dict));
            }

            if (actionList.Count == 0)
                throw new InvalidOperationException("Actions array is empty.");

            // Execute all actions in a single main-thread call
            var outerTask = RunOnMainThreadAsync(() =>
            {
                var batchTask = FullRunTrainingEnvService.Instance.BatchStepAsync(actionList);
                return batchTask.ContinueWith(t =>
                {
                    var batchResult = t.Result;
                    var state = BuildFullRunEnvState();
                    var response = new Dictionary<string, object?>
                    {
                        ["accepted"] = batchResult.Accepted,
                        ["error"] = batchResult.Error,
                        ["steps_executed"] = batchResult.StepsExecuted,
                    };
                    // Merge full state into response
                    foreach (var kv in state)
                        response[kv.Key] = kv.Value;
                    return response;
                }, TaskContinuationOptions.ExecuteSynchronously);
            });

            var result = outerTask.GetAwaiter().GetResult();
            SendJson(response, result);
        }
        catch (Exception ex)
        {
            SendError(response, 500, $"Batch step failed: {ex.Message}");
        }
    }

    private static void HandlePostFullRunEnvSaveState(HttpListenerResponse response)
    {
        try
        {
            if (FullRunTrainingEnvService.Instance.IsPureSimulator)
            {
                string stateId = FullRunTrainingEnvService.Instance.SaveState();
                SendJson(response, new
                {
                    state_id = stateId,
                    cache_size = FullRunTrainingEnvService.Instance.StateCacheCount
                });
                return;
            }

            var task = RunOnMainThread(() =>
            {
                string stateId = FullRunTrainingEnvService.Instance.SaveState();
                return new
                {
                    state_id = stateId,
                    cache_size = FullRunTrainingEnvService.Instance.StateCacheCount
                };
            });
            SendJson(response, task.GetAwaiter().GetResult());
        }
        catch (Exception ex)
        {
            string? errorCode = GetStructuredErrorCode(ex);
            if (errorCode != null)
            {
                SendError(response, 400, ex.Message, errorCode);
                return;
            }
            SendError(response, 500, $"Full run env save_state failed: {ex.Message}");
        }
    }

    private static void HandlePostFullRunEnvLoadState(HttpListenerRequest request, HttpListenerResponse response)
    {
        try
        {
            var parsed = ParseFullRunEnvRequestObject(request, allowEmptyBody: false);
            if (!parsed.TryGetValue("state_id", out var stateIdElem) || stateIdElem.ValueKind != JsonValueKind.String)
                throw new InvalidOperationException("Missing 'state_id' field.");
            string stateId = stateIdElem.GetString() ?? string.Empty;

            if (FullRunTrainingEnvService.Instance.IsPureSimulator)
            {
                FullRunTrainingEnvService.Instance.LoadState(stateId).GetAwaiter().GetResult();
                SendJson(response, BuildFullRunEnvState());
                return;
            }

            var outerTask = RunOnMainThreadAsync(() =>
            {
                var loadTask = FullRunTrainingEnvService.Instance.LoadState(stateId);
                return loadTask.ContinueWith(_ => BuildFullRunEnvState(),
                    TaskContinuationOptions.ExecuteSynchronously);
            });
            SendJson(response, outerTask.GetAwaiter().GetResult());
        }
        catch (JsonException ex)
        {
            SendError(response, 400, $"Invalid JSON: {ex.Message}", "invalid_json");
        }
        catch (InvalidOperationException ex)
        {
            SendError(response, 400, ex.Message, "invalid_request");
        }
        catch (Exception ex)
        {
            string? errorCode = GetStructuredErrorCode(ex);
            if (errorCode != null)
            {
                SendError(response, 400, ex.Message, errorCode);
                return;
            }
            SendError(response, 500, $"Full run env load_state failed: {ex.Message}");
        }
    }

    private static void HandlePostFullRunEnvExportState(HttpListenerRequest request, HttpListenerResponse response)
    {
        try
        {
            var parsed = ParseFullRunEnvRequestObject(request, allowEmptyBody: false);
            if (!parsed.TryGetValue("path", out var pathElem) || pathElem.ValueKind != JsonValueKind.String)
                throw new InvalidOperationException("Missing 'path' field.");
            string path = pathElem.GetString() ?? string.Empty;
            string? stateId = parsed.TryGetValue("state_id", out var stateIdElem) && stateIdElem.ValueKind == JsonValueKind.String
                ? stateIdElem.GetString()
                : null;

            if (FullRunTrainingEnvService.Instance.IsPureSimulator)
            {
                string writtenPath = FullRunTrainingEnvService.Instance.ExportStateToFile(path, stateId);
                SendJson(response, new
                {
                    path = writtenPath,
                    state_id = stateId,
                    cache_size = FullRunTrainingEnvService.Instance.StateCacheCount
                });
                return;
            }

            var task = RunOnMainThread(() =>
            {
                string writtenPath = FullRunTrainingEnvService.Instance.ExportStateToFile(path, stateId);
                return new
                {
                    path = writtenPath,
                    state_id = stateId,
                    cache_size = FullRunTrainingEnvService.Instance.StateCacheCount
                };
            });
            SendJson(response, task.GetAwaiter().GetResult());
        }
        catch (JsonException ex)
        {
            SendError(response, 400, $"Invalid JSON: {ex.Message}", "invalid_json");
        }
        catch (InvalidOperationException ex)
        {
            SendError(response, 400, ex.Message, "invalid_request");
        }
        catch (Exception ex)
        {
            string? errorCode = GetStructuredErrorCode(ex);
            if (errorCode != null)
            {
                SendError(response, 400, ex.Message, errorCode);
                return;
            }
            SendError(response, 500, $"Full run env export_state failed: {ex.Message}");
        }
    }

    private static void HandlePostFullRunEnvImportState(HttpListenerRequest request, HttpListenerResponse response)
    {
        try
        {
            var parsed = ParseFullRunEnvRequestObject(request, allowEmptyBody: false);
            if (!parsed.TryGetValue("path", out var pathElem) || pathElem.ValueKind != JsonValueKind.String)
                throw new InvalidOperationException("Missing 'path' field.");
            string path = pathElem.GetString() ?? string.Empty;

            if (FullRunTrainingEnvService.Instance.IsPureSimulator)
            {
                FullRunTrainingEnvService.Instance.LoadStateFromFile(path).GetAwaiter().GetResult();
                SendJson(response, BuildFullRunEnvState());
                return;
            }

            var outerTask = RunOnMainThreadAsync(() =>
            {
                var loadTask = FullRunTrainingEnvService.Instance.LoadStateFromFile(path);
                return loadTask.ContinueWith(_ => BuildFullRunEnvState(),
                    TaskContinuationOptions.ExecuteSynchronously);
            });
            SendJson(response, outerTask.GetAwaiter().GetResult());
        }
        catch (JsonException ex)
        {
            SendError(response, 400, $"Invalid JSON: {ex.Message}", "invalid_json");
        }
        catch (InvalidOperationException ex)
        {
            SendError(response, 400, ex.Message, "invalid_request");
        }
        catch (Exception ex)
        {
            string? errorCode = GetStructuredErrorCode(ex);
            if (errorCode != null)
            {
                SendError(response, 400, ex.Message, errorCode);
                return;
            }
            SendError(response, 500, $"Full run env import_state failed: {ex.Message}");
        }
    }

    private static void HandlePostFullRunEnvDeleteState(HttpListenerRequest request, HttpListenerResponse response)
    {
        try
        {
            var parsed = ParseFullRunEnvRequestObject(request, allowEmptyBody: true);
            bool clearAll = parsed.TryGetValue("clear_all", out var clearElem) && clearElem.ValueKind == JsonValueKind.True;
            string? stateId = parsed.TryGetValue("state_id", out var stateIdElem) && stateIdElem.ValueKind == JsonValueKind.String
                ? stateIdElem.GetString()
                : null;

            if (FullRunTrainingEnvService.Instance.IsPureSimulator)
            {
                if (clearAll)
                {
                    FullRunTrainingEnvService.Instance.ClearStateCache();
                    SendJson(response, new { deleted = true, cache_size = 0 });
                    return;
                }

                bool deleted = FullRunTrainingEnvService.Instance.DeleteState(stateId ?? string.Empty);
                SendJson(response, new
                {
                    deleted,
                    cache_size = FullRunTrainingEnvService.Instance.StateCacheCount
                });
                return;
            }

            var task = RunOnMainThread(() =>
            {
                if (clearAll)
                {
                    FullRunTrainingEnvService.Instance.ClearStateCache();
                    return new { deleted = true, cache_size = 0 };
                }

                bool deleted = FullRunTrainingEnvService.Instance.DeleteState(stateId ?? string.Empty);
                return new
                {
                    deleted,
                    cache_size = FullRunTrainingEnvService.Instance.StateCacheCount
                };
            });
            SendJson(response, task.GetAwaiter().GetResult());
        }
        catch (JsonException ex)
        {
            SendError(response, 400, $"Invalid JSON: {ex.Message}", "invalid_json");
        }
        catch (Exception ex)
        {
            string? errorCode = GetStructuredErrorCode(ex);
            if (errorCode != null)
            {
                SendError(response, 400, ex.Message, errorCode);
                return;
            }
            SendError(response, 500, $"Full run env delete_state failed: {ex.Message}");
        }
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
        var deadline = DateTime.UtcNow.AddMilliseconds(Math.Max(100, timeoutMs));
        var delay = Math.Max(10, pollDelayMs);
        Dictionary<string, object?>? lastState = null;

        while (DateTime.UtcNow <= deadline)
        {
            var stateTask = RunOnMainThread(BuildFullRunEnvState);
            lastState = stateTask.GetAwaiter().GetResult();
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
        var previousSignature = GetFullRunStateSignature(previousState);
        var previousStateType = GetStateType(previousState);
        var deadline = DateTime.UtcNow.AddMilliseconds(Math.Max(100, timeoutMs));
        var delay = Math.Max(10, pollDelayMs);
        Dictionary<string, object?>? lastChangedState = null;
        string? lastChangedSignature = null;
        int stablePolls = 0;

        while (DateTime.UtcNow <= deadline)
        {
            var stateTask = RunOnMainThread(BuildFullRunEnvState);
            var state = stateTask.GetAwaiter().GetResult();
            var signature = GetFullRunStateSignature(state);
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

    private static Dictionary<string, object?> BuildFullRunEnvState()
    {
        if (IsFullRunSimulatorActive())
        {
            var simSnapshot = GetSimulatorSnapshot();
            if (simSnapshot != null)
            {
                // When the full-run simulator is active, build the HTTP v2 state
                // from the simulator snapshot itself instead of starting from the
                // legacy UI-driven BuildGameState().  The legacy path can expose a
                // thinner schema on reward/card-select screens, which breaks NN
                // parity against pure-sim even when runtime semantics are aligned.
                var richState = ConvertSimulatorApiStateToDictionary(BuildSimulatorApiState(simSnapshot));
                var richOutcome = ExtractFullRunOutcome(richState) ?? simSnapshot.RunOutcome;
                richState["run_outcome"] = richOutcome;
                richState["terminal"] = simSnapshot.IsTerminal || IsFullRunTerminalState(richState, richOutcome);
                return richState;
            }
            else
                FullRunSimulationTrace.Write("full_run_env.sim_snapshot.null");
        }

        var state = BuildGameState();
        var outcome = ExtractFullRunOutcome(state);
        state["legal_actions"] = BuildFullRunLegalActions(state);
        state["run_outcome"] = outcome;
        state["terminal"] = state.TryGetValue("terminal", out var legacyTerm) && legacyTerm is true
            ? true
            : IsFullRunTerminalState(state, outcome);
        return state;
    }

    private static Dictionary<string, object?> ConvertSimulatorApiStateToDictionary(FullRunApiState state)
    {
        JsonElement root = JsonSerializer.SerializeToElement(state, _jsonOptions);
        if (root.ValueKind != JsonValueKind.Object)
            return new Dictionary<string, object?>();
        return ConvertJsonObjectToDictionary(root);
    }

    private static Dictionary<string, object?> ConvertJsonObjectToDictionary(JsonElement element)
    {
        var dict = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var prop in element.EnumerateObject())
            dict[prop.Name] = ConvertJsonElementToObject(prop.Value);
        return dict;
    }

    private static object? ConvertJsonElementToObject(JsonElement element)
    {
        return element.ValueKind switch
        {
            JsonValueKind.Object => ConvertJsonObjectToDictionary(element),
            JsonValueKind.Array => element.EnumerateArray().Select(ConvertJsonElementToObject).ToList(),
            JsonValueKind.String => element.GetString(),
            JsonValueKind.Number when element.TryGetInt32(out var intValue) => intValue,
            JsonValueKind.Number when element.TryGetInt64(out var longValue) => longValue,
            JsonValueKind.Number => element.GetDouble(),
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            JsonValueKind.Null => null,
            JsonValueKind.Undefined => null,
            _ => element.ToString()
        };
    }

    /// <summary>
    /// Converts the simulator snapshot's LegalActions into the HTTP dict format.
    /// Must be called on the Godot main thread (already inside RunOnMainThread).
    /// </summary>
    private static List<Dictionary<string, object?>> BuildSimulatorLegalActions()
    {
        try
        {
            var snapshot = FullRunTrainingEnvService.Instance.GetState();
            var result = new List<Dictionary<string, object?>>(snapshot.LegalActions.Count);
            foreach (var a in snapshot.LegalActions)
            {
                var d = new Dictionary<string, object?> { ["action"] = a.Action };
                if (a.Index.HasValue)     d["index"]     = a.Index.Value;
                if (a.Col.HasValue)       d["col"]       = a.Col.Value;
                if (a.Row.HasValue)       d["row"]       = a.Row.Value;
                if (a.CardIndex.HasValue) d["card_index"] = a.CardIndex.Value;
                if (a.Slot.HasValue)      d["slot"]      = a.Slot.Value;
                if (a.TargetId.HasValue)  d["target_id"] = a.TargetId.Value;
                if (!string.IsNullOrEmpty(a.Target)) d["target"] = a.Target;
                if (!string.IsNullOrEmpty(a.Label))  d["label"]  = a.Label;
                if (!string.IsNullOrEmpty(a.CardId)) d["card_id"] = a.CardId;
                if (!string.IsNullOrEmpty(a.CardType)) d["card_type"] = a.CardType;
                if (!string.IsNullOrEmpty(a.CardRarity)) d["card_rarity"] = a.CardRarity;
                if (!string.IsNullOrEmpty(a.Cost)) d["cost"] = a.Cost;
                if (a.IsUpgraded.HasValue) d["is_upgraded"] = a.IsUpgraded.Value;
                if (!string.IsNullOrEmpty(a.RewardType)) d["reward_type"] = a.RewardType;
                if (!string.IsNullOrEmpty(a.RewardKey)) d["reward_key"] = a.RewardKey;
                if (!string.IsNullOrEmpty(a.RewardSource)) d["reward_source"] = a.RewardSource;
                if (a.Claimable.HasValue) d["claimable"] = a.Claimable.Value;
                if (!string.IsNullOrEmpty(a.ClaimBlockReason)) d["claim_block_reason"] = a.ClaimBlockReason;
                result.Add(d);
            }
            return result;
        }
        catch
        {
            return new List<Dictionary<string, object?>>();
        }
    }

    private static FullRunSimulationStateSnapshot? GetSimulatorSnapshot()
    {
        try { return FullRunTrainingEnvService.Instance.GetState(); }
        catch (Exception ex)
        {
            FullRunSimulationTrace.Write($"full_run_env.sim_snapshot.exception type={ex.GetType().Name} message={ex.Message}");
            return null;
        }
    }

    private static FullRunApiState BuildSimulatorApiState(FullRunSimulationStateSnapshot snapshot)
    {
        var runState = MegaCrit.Sts2.Core.Runs.RunManager.Instance.DebugOnlyGetState();
        return FullRunApiStateBuilder.Build(runState, snapshot);
    }

    private static List<Dictionary<string, object?>> BuildSimulatorLegalActionsFromSnapshot(FullRunSimulationStateSnapshot snapshot)
    {
        var result = new List<Dictionary<string, object?>>(snapshot.LegalActions.Count);
        foreach (var a in snapshot.LegalActions)
        {
            var dict = new Dictionary<string, object?>
            {
                ["action"] = a.Action,
            };
            if (a.Index.HasValue) dict["index"] = a.Index.Value;
            if (a.CardIndex.HasValue) dict["card_index"] = a.CardIndex.Value;
            if (a.TargetId.HasValue) dict["target_id"] = a.TargetId.Value;
            if (a.Slot.HasValue) dict["slot"] = a.Slot.Value;
            if (a.Col.HasValue) dict["col"] = a.Col.Value;
            if (a.Row.HasValue) dict["row"] = a.Row.Value;
            if (a.Label != null) dict["label"] = a.Label;
            if (a.Target != null) dict["target"] = a.Target;
            if (a.CardId != null) dict["card_id"] = a.CardId;
            if (a.CardType != null) dict["card_type"] = a.CardType;
            if (a.CardRarity != null) dict["card_rarity"] = a.CardRarity;
            if (a.Cost != null) dict["cost"] = a.Cost;
            if (a.IsUpgraded.HasValue) dict["is_upgraded"] = a.IsUpgraded.Value;
            if (a.RewardType != null) dict["reward_type"] = a.RewardType;
            if (a.RewardKey != null) dict["reward_key"] = a.RewardKey;
            if (a.RewardSource != null) dict["reward_source"] = a.RewardSource;
            if (a.Claimable.HasValue) dict["claimable"] = a.Claimable.Value;
            if (a.ClaimBlockReason != null) dict["claim_block_reason"] = a.ClaimBlockReason;
            if (a.Note != null) dict["note"] = a.Note;
            if (string.Equals(a.Action, "select_card_reward", StringComparison.OrdinalIgnoreCase)
                && string.IsNullOrEmpty(a.CardId))
            {
                FullRunSimulationTrace.Write(
                    $"full_run_env.card_reward_action.missing_card_metadata index={a.Index?.ToString() ?? "null"} " +
                    $"card_id={a.CardId ?? "null"} card_type={a.CardType ?? "null"} " +
                    $"card_rarity={a.CardRarity ?? "null"} cost={a.Cost ?? "null"} label={a.Label ?? "null"}");
            }
            result.Add(dict);
        }
        return result;
    }

    private static void MergeSimulatorScreenPayload(
        Dictionary<string, object?> state,
        string? stateType,
        FullRunApiState simApiState)
    {
        switch ((stateType ?? string.Empty).Trim().ToLowerInvariant())
        {
            case "combat_rewards":
                if (simApiState.rewards != null)
                    state["rewards"] = ShapeSimulatorRewardsState(simApiState.rewards);
                break;
            case "card_reward":
                if (simApiState.card_reward != null)
                    state["card_reward"] = ShapeSimulatorCardRewardState(simApiState.card_reward);
                else
                    FullRunSimulationTrace.Write("full_run_env.card_reward_payload.null");
                break;
        }
    }

    private static void EnrichCardRewardStateFromSnapshot(
        Dictionary<string, object?> state,
        FullRunSimulationStateSnapshot simSnapshot)
    {
        if (!string.Equals(simSnapshot.StateType, "card_reward", StringComparison.OrdinalIgnoreCase))
            return;

        FullRunPendingCardRewardSnapshot? rewardSelection = simSnapshot.CachedBridgeSnapshots?.CardRewardSelection;
        if (rewardSelection == null || rewardSelection.Options.Count == 0)
            return;

        if (state.TryGetValue("legal_actions", out var rawLegalActions) && rawLegalActions is List<Dictionary<string, object?>> legalActions)
            EnrichCardRewardActionsFromBridgeSnapshot(legalActions, rewardSelection);

        bool hasStructuredCardReward =
            TryGetDict(state, "card_reward", out var existingRewardState)
            && existingRewardState.TryGetValue("cards", out var rawCards)
            && rawCards is IEnumerable<object?> cardsEnumerable
            && cardsEnumerable.Any();
        if (!hasStructuredCardReward)
            state["card_reward"] = BuildCardRewardStateFromBridgeSnapshot(rewardSelection, state);
    }

    private static void EnrichCardRewardActionsFromBridgeSnapshot(
        List<Dictionary<string, object?>> legalActions,
        FullRunPendingCardRewardSnapshot rewardSelection)
    {
        foreach (Dictionary<string, object?> action in legalActions)
        {
            if (!string.Equals(action.TryGetValue("action", out var rawAction) ? rawAction as string : null, "select_card_reward", StringComparison.OrdinalIgnoreCase))
                continue;
            if (action.ContainsKey("card_id") && action["card_id"] != null)
                continue;

            int index = GetInt(action, "index", GetInt(action, "card_index", -1));
            if (index < 0 || index >= rewardSelection.Options.Count)
                continue;
            CardModel card = rewardSelection.Options[index].Card;
            action["card_id"] = card.Id.Entry;
            action["card_type"] = card.Type.ToString();
            action["card_rarity"] = card.Rarity.ToString();
            action["cost"] = card.EnergyCost.CostsX ? "X" : card.EnergyCost.GetAmountToSpend().ToString();
            action["is_upgraded"] = card.IsUpgraded;
        }
    }

    private static Dictionary<string, object?> BuildCardRewardStateFromBridgeSnapshot(
        FullRunPendingCardRewardSnapshot rewardSelection,
        Dictionary<string, object?> state)
    {
        Dictionary<string, object?> rewardState = new Dictionary<string, object?>
        {
            ["can_skip"] = rewardSelection.CanSkip,
            ["cards"] = rewardSelection.Options
                .Select((entry, index) => ShapeSimulatorCardOption(entry.Card, index))
                .Cast<object?>()
                .ToList()
        };
        if (state.TryGetValue("player", out var rawPlayer) && rawPlayer is Dictionary<string, object?> player)
            rewardState["player"] = player;
        return rewardState;
    }

    private static Dictionary<string, object?> ShapeSimulatorCardOption(CardModel card, int index)
    {
        return new Dictionary<string, object?>
        {
            ["index"] = index,
            ["id"] = card.Id.Entry,
            ["name"] = card.Title,
            ["type"] = card.Type.ToString(),
            ["rarity"] = card.Rarity.ToString(),
            ["cost"] = card.EnergyCost.CostsX ? "X" : card.EnergyCost.GetAmountToSpend().ToString(),
            ["is_upgraded"] = card.IsUpgraded,
            ["description"] = card.Description,
        };
    }

    private static Dictionary<string, object?> ShapeSimulatorRewardsState(FullRunApiRewardsState rewards)
    {
        return new Dictionary<string, object?>
        {
            ["player"] = ShapeSimulatorPlayerState(rewards.player),
            ["can_proceed"] = rewards.can_proceed,
            ["items"] = rewards.items.Select(static item => new Dictionary<string, object?>
            {
                ["index"] = item.index,
                ["type"] = item.type,
                ["label"] = item.label,
                ["reward_key"] = item.reward_key,
                ["reward_source"] = item.reward_source,
                ["claimable"] = item.claimable,
                ["claim_block_reason"] = item.claim_block_reason,
            }).Cast<object?>().ToList()
        };
    }

    private static Dictionary<string, object?> ShapeSimulatorCardRewardState(FullRunApiCardRewardState reward)
    {
        return new Dictionary<string, object?>
        {
            ["player"] = ShapeSimulatorPlayerState(reward.player),
            ["can_skip"] = reward.can_skip,
            ["cards"] = reward.cards.Select(ShapeSimulatorCardOption).Cast<object?>().ToList()
        };
    }

    private static Dictionary<string, object?> ShapeSimulatorPlayerState(FullRunApiPlayerState player)
    {
        return new Dictionary<string, object?>
        {
            ["character"] = player.character,
            ["hp"] = player.hp,
            ["current_hp"] = player.current_hp,
            ["max_hp"] = player.max_hp,
            ["block"] = player.block,
            ["gold"] = player.gold,
            ["energy"] = player.energy,
            ["max_energy"] = player.max_energy,
            ["draw_pile_count"] = player.draw_pile_count,
            ["discard_pile_count"] = player.discard_pile_count,
            ["exhaust_pile_count"] = player.exhaust_pile_count,
            ["open_potion_slots"] = player.open_potion_slots,
            ["status"] = player.status.Select(static power => new Dictionary<string, object?>
            {
                ["id"] = power.id,
                ["amount"] = power.amount,
            }).Cast<object?>().ToList(),
            ["hand"] = player.hand.Select(ShapeSimulatorCardOption).Cast<object?>().ToList(),
            ["deck"] = player.deck.Select(ShapeSimulatorCardOption).Cast<object?>().ToList(),
            ["relics"] = player.relics.Select(static relic => new Dictionary<string, object?>
            {
                ["index"] = relic.index,
                ["id"] = relic.id,
                ["name"] = relic.name,
                ["rarity"] = relic.rarity,
                ["description"] = relic.description,
            }).Cast<object?>().ToList(),
            ["potions"] = player.potions.Select(static potion => new Dictionary<string, object?>
            {
                ["slot"] = potion.slot,
                ["id"] = potion.id,
                ["name"] = potion.name,
                ["description"] = potion.description,
                ["target_type"] = potion.target_type,
                ["can_use_in_combat"] = potion.can_use_in_combat,
                ["keywords"] = potion.keywords.Cast<object?>().ToList(),
            }).Cast<object?>().ToList(),
        };
    }

    private static Dictionary<string, object?> ShapeSimulatorCardOption(FullRunApiCardOption card)
    {
        return new Dictionary<string, object?>
        {
            ["index"] = card.index,
            ["id"] = card.id,
            ["name"] = card.name,
            ["type"] = card.type,
            ["rarity"] = card.rarity,
            ["cost"] = card.cost,
            ["is_upgraded"] = card.is_upgraded,
            ["can_play"] = card.can_play,
            ["target_type"] = card.target_type,
            ["unplayable_reason"] = card.unplayable_reason,
            ["description"] = card.description,
            ["valid_target_ids"] = card.valid_target_ids.Cast<object?>().ToList(),
            ["keywords"] = card.keywords.Cast<object?>().ToList(),
        };
    }

    private static Dictionary<string, object?> ShapeFullRunEnvStepResult(
        Dictionary<string, object?> state,
        bool accepted,
        string? error,
        string? stepInfoCode = null)
    {
        var outcome = ExtractFullRunOutcome(state);
        var done = IsFullRunTerminalState(state, outcome);
        double reward = 0.0;
        if (done)
        {
            reward = outcome == "victory" || outcome == "win" ? 1.0 : -1.0;
        }

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
        var stateType = GetStateType(state);

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
                if (TryGetDict(state, "relic_select", out var relicSelectState))
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
        if (!TryGetDict(state, "menu", out var menu))
            return;

        foreach (var character in EnumerateDictionaries(menu.TryGetValue("available_characters", out var rawChars) ? rawChars : null))
        {
            if (GetBool(character, "is_locked"))
                continue;
            if (TryGetString(character, "id", out var characterId))
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
            var maxAscension = Math.Max(0, GetInt(menu, "max_ascension", 0));
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
            if (TryGetString(menu, "selected_character", out var selectedCharacter))
                startAction["character_id"] = selectedCharacter;
            actions.Add(startAction);
        }
    }

    private static void AppendCardRewardLegalActions(List<Dictionary<string, object?>> actions, Dictionary<string, object?> state)
    {
        if (!TryGetDict(state, "card_reward", out var rewardState))
            return;

        foreach (var card in EnumerateDictionaries(rewardState.TryGetValue("cards", out var rawCards) ? rawCards : null))
        {
            var action = new Dictionary<string, object?>
            {
                ["action"] = "select_card_reward",
                ["index"] = GetInt(card, "index", -1),
                ["card_index"] = GetInt(card, "index", -1)
            };
            if (TryGetString(card, "id", out var cardId))
                action["card_id"] = cardId;
            if (TryGetString(card, "type", out var cardType))
                action["card_type"] = cardType;
            if (TryGetString(card, "rarity", out var cardRarity))
                action["card_rarity"] = cardRarity;
            if (TryGetString(card, "cost", out var cardCost))
                action["cost"] = cardCost;
            if (card.TryGetValue("is_upgraded", out var isUpgraded) && isUpgraded is bool upgraded)
                action["is_upgraded"] = upgraded;
            if (TryGetString(card, "name", out var cardName))
                action["label"] = cardName;
            actions.Add(action);
        }

        AppendIfTrue(actions, rewardState, "can_skip", new Dictionary<string, object?> { ["action"] = "skip_card_reward" });
    }

    private static void AppendEventLegalActions(List<Dictionary<string, object?>> actions, Dictionary<string, object?> state)
    {
        if (!TryGetDict(state, "event", out var eventState))
            return;

        if (GetBool(eventState, "in_dialogue"))
        {
            actions.Add(new Dictionary<string, object?> { ["action"] = "advance_dialogue" });
            return;
        }

        foreach (var option in EnumerateDictionaries(eventState.TryGetValue("options", out var rawOptions) ? rawOptions : null))
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
        if (!TryGetDict(state, "shop", out var shopState))
            return;

        foreach (var item in EnumerateDictionaries(shopState.TryGetValue("items", out var rawItems) ? rawItems : null))
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

        // The player can always exit a shop, even with zero gold. The
        // `AppendProceedIfEnabled` helper gates on `shop.can_proceed`, but
        // BuildShopState reads that from NMerchantRoom.Instance.ProceedButton
        // which returns false when no items were just bought / the UI hasn't
        // wired the button yet. Emit `proceed` unconditionally here so the AI
        // doesn't get stuck on a broke shop with no affordable items and no
        // escape. 2026-04-09 fix.
        actions.Add(new Dictionary<string, object?> { ["action"] = "proceed" });
    }

    private static void AppendCardSelectLegalActions(List<Dictionary<string, object?>> actions, Dictionary<string, object?> state)
    {
        if (!TryGetDict(state, "card_select", out var selectState))
            return;

        bool previewShowing = GetBool(selectState, "preview_showing");
        int selectedCount = GetInt(selectState, "selected_count", 0);
        int maxSelect = GetInt(selectState, "max_select", -1);
        bool selectionQuotaReached = maxSelect > 0 && selectedCount >= maxSelect;

        if (!previewShowing && !selectionQuotaReached)
        {
            // Collect indices of cards already in the `selected_cards` list so
            // we don't re-emit them as legal actions. Without this filter the
            // AI gets stuck repeatedly selecting the same card on multi-pick
            // screens (e.g. "remove 2 cards") because the NN always picks the
            // first legal action and the first card stays at the top of the
            // list. 2026-04-09 fix.
            var selectedIndices = new HashSet<int>();
            foreach (var selCard in EnumerateDictionaries(selectState.TryGetValue("selected_cards", out var rawSel) ? rawSel : null))
            {
                int idx = GetInt(selCard, "index", -1);
                if (idx >= 0)
                    selectedIndices.Add(idx);
            }

            foreach (var card in EnumerateDictionaries(selectState.TryGetValue("cards", out var rawCards) ? rawCards : null))
            {
                int cardIndex = GetInt(card, "index", -1);
                if (cardIndex < 0 || selectedIndices.Contains(cardIndex))
                    continue;
                // ExecuteSelectCard reads the grid index from the "index" key
                // (see McpMod.Actions.cs). Earlier versions of this code
                // emitted "card_index" which made every select_card action
                // get rejected with "Missing 'index'". 2026-04-09 fix.
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
        if (!TryGetDict(state, "battle", out var battleState))
            return;
        // Sim-mode BuildSimulatorApiState nests the combat-phase player dict
        // inside `battle.player`, but the legacy UI-driven BuildGameState
        // leaves the player at the top level (`state.player`). Both schemas
        // are valid; prefer battle.player when present, otherwise fall
        // through to top-level player.
        if (!TryGetDict(battleState, "player", out var playerState)
            && !TryGetDict(state, "player", out playerState))
            return;

        string turn = GetString(battleState, "turn");
        bool isPlayPhase = GetBool(battleState, "is_play_phase", defaultValue: true);
        if (turn != "player" || !isPlayPhase)
            return;

        var enemies = EnumerateDictionaries(battleState.TryGetValue("enemies", out var rawEnemies) ? rawEnemies : null)
            .Where(enemy => GetBool(enemy, "is_alive", defaultValue: true))
            .ToList();

        foreach (var card in EnumerateDictionaries(playerState.TryGetValue("hand", out var rawHand) ? rawHand : null))
        {
            if (!GetBool(card, "can_play"))
                continue;

            var handIndex = GetInt(card, "index", -1);
            var targetType = GetString(card, "target_type");
            bool requiresTarget = targetType is "enemy" or "anyenemy" or "any_enemy";
            if (requiresTarget)
            {
                foreach (var enemy in enemies)
                {
                    string? targetId = null;
                    if (TryGetString(enemy, "entity_id", out var entityId))
                        targetId = entityId;
                    else if (TryGetString(enemy, "id", out var fallbackId))
                        targetId = fallbackId;

                    if (string.IsNullOrWhiteSpace(targetId))
                        continue;

                    actions.Add(new Dictionary<string, object?>
                    {
                        ["action"] = "play_card",
                        ["card_index"] = handIndex,
                        ["target"] = targetId
                    });
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

        actions.Add(new Dictionary<string, object?> { ["action"] = "end_turn" });
    }

    private static void AppendHandSelectLegalActions(List<Dictionary<string, object?>> actions, Dictionary<string, object?> state)
    {
        if (!TryGetDict(state, "hand_select", out var handSelectState))
            return;

        foreach (var card in EnumerateDictionaries(handSelectState.TryGetValue("cards", out var rawCards) ? rawCards : null))
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
        var containerKey = GetStateType(state) == "game_over" ? "game_over" : "overlay";
        if (!TryGetDict(state, containerKey, out var overlayState))
            return;

        foreach (var button in EnumerateDictionaries(overlayState.TryGetValue("buttons", out var rawButtons) ? rawButtons : null))
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
        if (!TryGetDict(state, containerKey, out var container))
            return;

        foreach (var item in EnumerateDictionaries(container.TryGetValue(collectionKey, out var rawItems) ? rawItems : null))
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
        if (!TryGetDict(state, containerKey, out var container))
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
        return state.TryGetValue("state_type", out var raw)
            ? (raw?.ToString() ?? string.Empty).Trim().ToLowerInvariant()
            : string.Empty;
    }

    private static int GetFullRunAct(Dictionary<string, object?> state)
    {
        return TryGetDict(state, "run", out var runState) ? GetInt(runState, "act", 0) : 0;
    }

    private static int GetFullRunFloor(Dictionary<string, object?> state)
    {
        return TryGetDict(state, "run", out var runState) ? GetInt(runState, "floor", 0) : 0;
    }

    private static bool IsSettledFullRunState(Dictionary<string, object?> state)
    {
        var stateType = GetStateType(state);
        return stateType is not "" and not "unknown" and not "menu";
    }

    private static bool ShouldReturnImmediatelyForChangedFullRunState(
        Dictionary<string, object?> previousState,
        string previousStateType,
        Dictionary<string, object?> state)
    {
        var stateType = GetStateType(state);
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

        if (!state.TryGetValue("legal_actions", out var rawActions) || rawActions is not IEnumerable enumerable || rawActions is string)
            return false;

        foreach (var item in enumerable)
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
        var stateType = GetStateType(state);
        return stateType == "game_over"
            || string.Equals(outcome, "victory", StringComparison.OrdinalIgnoreCase)
            || string.Equals(outcome, "death", StringComparison.OrdinalIgnoreCase)
            || string.Equals(outcome, "win", StringComparison.OrdinalIgnoreCase)
            || string.Equals(outcome, "loss", StringComparison.OrdinalIgnoreCase);
    }

    private static string? ExtractFullRunOutcome(Dictionary<string, object?> state)
    {
        if (TryGetString(state, "run_outcome", out var runOutcome))
            return runOutcome;

        if (TryGetDict(state, "game_over", out var gameOverState) && TryGetString(gameOverState, "outcome", out var outcome))
            return outcome;

        return null;
    }

    private static bool IsErrorResult(Dictionary<string, object?> result, out string? error)
    {
        error = null;
        if (result.TryGetValue("status", out var statusRaw)
            && string.Equals(statusRaw?.ToString(), "error", StringComparison.OrdinalIgnoreCase))
        {
            error = result.TryGetValue("error", out var errorRaw) ? errorRaw?.ToString() : "Action failed.";
            return true;
        }

        if (result.TryGetValue("error", out var directError) && directError is string directErrorText && !string.IsNullOrWhiteSpace(directErrorText))
        {
            error = directErrorText;
            return true;
        }

        return false;
    }

    private static bool TryGetDict(Dictionary<string, object?> state, string key, out Dictionary<string, object?> dict)
    {
        if (state.TryGetValue(key, out var raw) && raw is Dictionary<string, object?> typed)
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

        foreach (var item in enumerable)
        {
            if (item is Dictionary<string, object?> typed)
                yield return typed;
        }
    }

    private static bool TryGetString(Dictionary<string, object?> state, string key, out string text)
    {
        if (state.TryGetValue(key, out var raw) && raw != null)
        {
            text = raw.ToString() ?? string.Empty;
            return !string.IsNullOrWhiteSpace(text);
        }

        text = string.Empty;
        return false;
    }

    private static string GetString(Dictionary<string, object?> state, string key)
    {
        return TryGetString(state, key, out var text) ? text.Trim().ToLowerInvariant() : string.Empty;
    }

    private static int GetInt(Dictionary<string, object?> state, string key, int defaultValue)
    {
        if (!state.TryGetValue(key, out var raw) || raw == null)
            return defaultValue;
        return raw switch
        {
            int value => value,
            long value => (int)value,
            uint value => (int)value,
            JsonElement elem when elem.ValueKind == JsonValueKind.Number && elem.TryGetInt32(out var value) => value,
            _ when int.TryParse(raw.ToString(), out var parsed) => parsed,
            _ => defaultValue
        };
    }

    private static int GetOptionalInt(Dictionary<string, JsonElement> payload, string key, int defaultValue)
    {
        if (payload.TryGetValue(key, out var elem) && elem.ValueKind == JsonValueKind.Number)
            return elem.GetInt32();
        return defaultValue;
    }

    private static bool GetBool(Dictionary<string, object?> state, string key, bool defaultValue = false)
    {
        if (!state.TryGetValue(key, out var raw) || raw == null)
            return defaultValue;
        return raw switch
        {
            bool value => value,
            JsonElement elem when elem.ValueKind == JsonValueKind.True => true,
            JsonElement elem when elem.ValueKind == JsonValueKind.False => false,
            _ when bool.TryParse(raw.ToString(), out var parsed) => parsed,
            _ => defaultValue
        };
    }

    private static bool IsFullRunSimulatorActive()
    {
        try { return FullRunSimulationMode.IsServerActive || FullRunSimulationMode.IsSmokeActive; }
        catch { return false; }
    }

    private static FullRunSimulationResetRequest ParseSimResetRequest(Dictionary<string, JsonElement> parsed)
    {
        var req = new FullRunSimulationResetRequest();
        if (parsed.TryGetValue("character_id", out var charElem) && charElem.ValueKind == JsonValueKind.String)
            req.CharacterId = charElem.GetString();
        if (parsed.TryGetValue("seed", out var seedElem) && seedElem.ValueKind == JsonValueKind.String)
            req.Seed = seedElem.GetString();
        if (parsed.TryGetValue("ascension_level", out var ascElem) && ascElem.ValueKind == JsonValueKind.Number)
            req.AscensionLevel = ascElem.GetInt32();
        return req;
    }

    private static FullRunSimulationActionRequest ParseSimActionRequest(Dictionary<string, JsonElement> parsed)
    {
        var req = new FullRunSimulationActionRequest();
        if (parsed.TryGetValue("action", out var actionElem) && actionElem.ValueKind == JsonValueKind.String)
            req.Action = actionElem.GetString() ?? string.Empty;
        if (parsed.TryGetValue("index", out var indexElem) && indexElem.ValueKind == JsonValueKind.Number)
            req.Index = indexElem.GetInt32();
        if (parsed.TryGetValue("card_index", out var cardIndexElem) && cardIndexElem.ValueKind == JsonValueKind.Number)
            req.CardIndex = cardIndexElem.GetInt32();
        if (parsed.TryGetValue("hand_index", out var handIndexElem) && handIndexElem.ValueKind == JsonValueKind.Number)
            req.HandIndex = handIndexElem.GetInt32();
        if (parsed.TryGetValue("slot", out var slotElem) && slotElem.ValueKind == JsonValueKind.Number)
            req.Slot = slotElem.GetInt32();
        if (parsed.TryGetValue("col", out var colElem) && colElem.ValueKind == JsonValueKind.Number)
            req.Col = colElem.GetInt32();
        if (parsed.TryGetValue("row", out var rowElem) && rowElem.ValueKind == JsonValueKind.Number)
            req.Row = rowElem.GetInt32();
        if (parsed.TryGetValue("target", out var targetElem) && targetElem.ValueKind == JsonValueKind.String)
            req.Target = targetElem.GetString();
        if (parsed.TryGetValue("target_id", out var targetIdElem) && targetIdElem.ValueKind == JsonValueKind.Number)
            req.TargetId = targetIdElem.GetUInt32();
        if (parsed.TryGetValue("value", out var valueElem) && valueElem.ValueKind == JsonValueKind.String)
            req.Value = valueElem.GetString();
        return req;
    }
}
