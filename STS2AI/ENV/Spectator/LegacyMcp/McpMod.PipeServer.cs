using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.IO.Pipes;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Simulation;

namespace STS2_MCP;

/// <summary>
/// Named pipe server for high-speed IPC with Python MCTS clients.
/// Protocol: length-prefixed JSON messages (4-byte little-endian length + UTF-8 JSON).
/// Each pipe connection is handled on a thread pool thread, with game logic
/// dispatched to the Godot main thread via RunOnMainThread/RunOnMainThreadAsync.
///
/// ~50x faster than HTTP for small messages (no TCP handshake, no HTTP headers).
/// </summary>
public static partial class McpMod
{
    private static CancellationTokenSource? _pipeCts;
    private static readonly string PipeName = "sts2_mcts";
    private static readonly object _pipeSessionSync = new();
    private static long _nextPipeSessionId;
    private static long? _activePipeSessionId;
    private static DateTime _sessionAcquiredAt;
    private const int PipeSessionTimeoutSeconds = 90;

    /// <summary>
    /// Start the named pipe server. Call from Initialize() after HTTP server is up.
    /// Spawns a background listener that accepts connections and processes messages.
    /// </summary>
    internal static void StartPipeServer(int? portSuffix = null)
    {
        string pipeName = portSuffix.HasValue ? $"{PipeName}_{portSuffix}" : PipeName;
        _pipeCts = new CancellationTokenSource();
        var ct = _pipeCts.Token;

        Task.Run(async () =>
        {
            GD.Print($"[MCP-PIPE] Named pipe server starting on \\\\.\\pipe\\{pipeName}");
            while (!ct.IsCancellationRequested)
            {
                try
                {
                    var server = new NamedPipeServerStream(
                        pipeName,
                        PipeDirection.InOut,
                        NamedPipeServerStream.MaxAllowedServerInstances,
                        PipeTransmissionMode.Byte,
                        PipeOptions.Asynchronous);

                    await server.WaitForConnectionAsync(ct);
                    // Handle each connection on a separate thread
                    _ = Task.Run(() => HandlePipeConnection(server, ct), ct);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    GD.PrintErr($"[MCP-PIPE] Listener error: {ex.Message}");
                    await Task.Delay(100, ct);
                }
            }
            GD.Print("[MCP-PIPE] Named pipe server stopped.");
        }, ct);
    }

    internal static void StopPipeServer()
    {
        _pipeCts?.Cancel();
        _pipeCts?.Dispose();
        _pipeCts = null;
    }

    /// <summary>Per-client read timeout (seconds). If the client doesn't send a new
    /// request within this time, the connection is closed and session released.</summary>
    private const int PipeReadTimeoutSeconds = 60;

    /// <summary>Per-request processing timeout (seconds). If a simulator step takes
    /// longer than this, the request is cancelled and an error returned.</summary>
    private const int PipeRequestTimeoutSeconds = 45;

    private static async Task HandlePipeConnection(NamedPipeServerStream pipe, CancellationToken ct)
    {
        long sessionId = Interlocked.Increment(ref _nextPipeSessionId);
        if (!TryAcquirePipeSession(sessionId))
        {
            try
            {
                using (pipe)
                {
                    await WritePipeResponseAsync(
                        pipe,
                        SerializePipeError("simulator_busy", "The simulator runtime is already owned by another active pipe session."),
                        ct);
                }
            }
            catch
            {
                // Ignore best-effort busy notification failures.
            }
            return;
        }

        try
        {
            using (pipe)
            {
                await WritePipeResponseAsync(
                    pipe,
                    JsonSerializer.Serialize(new { ok = true }, _jsonOptions),
                    ct);

                while (pipe.IsConnected && !ct.IsCancellationRequested)
                {
                    // Read with per-message timeout so a dead client doesn't hold the session forever
                    using var readCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
                    readCts.CancelAfter(TimeSpan.FromSeconds(PipeReadTimeoutSeconds));

                    byte[] lenBuf = new byte[4];
                    int read;
                    try
                    {
                        read = await ReadExactAsync(pipe, lenBuf, readCts.Token);
                    }
                    catch (OperationCanceledException) when (!ct.IsCancellationRequested)
                    {
                        // Read timeout — client idle too long, close connection
                        GD.Print($"[MCP-PIPE] Client idle timeout ({PipeReadTimeoutSeconds}s), closing session {sessionId}");
                        break;
                    }
                    if (read < 4) break; // client disconnected

                    int msgLen = BitConverter.ToInt32(lenBuf, 0);
                    if (msgLen <= 0 || msgLen > 10_000_000)
                    {
                        GD.PrintErr($"[MCP-PIPE] Invalid message length: {msgLen}");
                        break;
                    }

                    byte[] msgBuf = new byte[msgLen];
                    try
                    {
                        read = await ReadExactAsync(pipe, msgBuf, readCts.Token);
                    }
                    catch (OperationCanceledException) when (!ct.IsCancellationRequested)
                    {
                        GD.Print("[MCP-PIPE] Read timeout during message body, closing session");
                        break;
                    }
                    if (read < msgLen) break;

                    string requestJson = Encoding.UTF8.GetString(msgBuf);

                    // Process request with timeout so a stuck step doesn't hold the session
                    string responseJson;
                    try
                    {
                        using var reqCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
                        reqCts.CancelAfter(TimeSpan.FromSeconds(PipeRequestTimeoutSeconds));
                        responseJson = await ProcessPipeRequest(requestJson, sessionId).WaitAsync(reqCts.Token);
                    }
                    catch (OperationCanceledException) when (!ct.IsCancellationRequested)
                    {
                        responseJson = SerializePipeError("request_timeout",
                            $"Request processing timed out after {PipeRequestTimeoutSeconds}s");
                        GD.PrintErr($"[MCP-PIPE] Request timeout ({PipeRequestTimeoutSeconds}s): {requestJson.Substring(0, Math.Min(100, requestJson.Length))}");
                    }
                    catch (Exception ex)
                    {
                        responseJson = SerializePipeError(GetPipeErrorCode(ex), ex.Message);
                    }

                    try
                    {
                        await WritePipeResponseAsync(pipe, responseJson, ct);
                    }
                    catch
                    {
                        break; // Write failed — client gone
                    }
                }
            }
        }
        catch (IOException)
        {
            // Client disconnected — normal
        }
        catch (OperationCanceledException)
        {
            // Server shutting down
        }
        catch (Exception ex)
        {
            GD.PrintErr($"[MCP-PIPE] Connection error: {ex.Message}");
        }
        finally
        {
            ReleasePipeSession(sessionId);
            GD.Print($"[MCP-PIPE] Session {sessionId} released");
        }
    }

    private static async Task<int> ReadExactAsync(Stream stream, byte[] buffer, CancellationToken ct)
    {
        int offset = 0;
        while (offset < buffer.Length)
        {
            int n = await stream.ReadAsync(buffer, offset, buffer.Length - offset, ct);
            if (n == 0) return offset; // EOF
            offset += n;
        }
        return offset;
    }

    /// <summary>Threshold in milliseconds. Only log requests slower than this to avoid spam.</summary>
    private const long PipeLogSlowThresholdMs = 100;

    private static async Task<string> ProcessPipeRequest(string requestJson, long sessionId)
    {
        EnsurePipeSessionOwner(sessionId);

        using var doc = JsonDocument.Parse(requestJson);
        var root = doc.RootElement;

        string method = root.GetProperty("method").GetString() ?? "";
        JsonElement paramsElem = root.TryGetProperty("params", out var p) ? p : default;

        var sw = Stopwatch.StartNew();
        string result;
        string detail = "";
        bool isError = false;

        try
        {
            // Build a short detail string for the log line
            if (method == "step" && paramsElem.ValueKind == JsonValueKind.Object)
            {
                string action = paramsElem.TryGetProperty("action", out var a) ? a.GetString() ?? "" : "";
                detail = action;
                if (paramsElem.TryGetProperty("card_index", out var ci))
                    detail += $" ci={ci}";
                else if (paramsElem.TryGetProperty("index", out var idx))
                    detail += $" index={idx}";
            }
            else if (method == "reset" && paramsElem.ValueKind == JsonValueKind.Object)
            {
                detail = paramsElem.TryGetProperty("character_id", out var c) ? c.GetString() ?? "" : "";
            }

            result = method switch
            {
                "reset" => await ProcessPipeReset(paramsElem),
                "step" => await ProcessPipeStep(paramsElem),
                "state" => await ProcessPipeGetState(),
                "legal_actions" => await ProcessPipeLegalActions(),
                "save_state" => await ProcessPipeSaveState(),
                "load_state" => await ProcessPipeLoadState(paramsElem),
                "delete_state" => await ProcessPipeDeleteState(paramsElem),
                "batch_step" => await ProcessPipeBatchStep(paramsElem),
                _ => SerializePipeError("unknown_method", $"Unknown method: {method}")
            };
        }
        catch (Exception)
        {
            isError = true;
            sw.Stop();
            GD.PrintErr($"[MCP-PIPE] {method} {detail} → {sw.ElapsedMilliseconds}ms ERROR");
            throw;
        }

        sw.Stop();
        long elapsedMs = sw.ElapsedMilliseconds;

        if (isError || elapsedMs >= PipeLogSlowThresholdMs)
        {
            GD.Print($"[MCP-PIPE] {method} {detail} → {elapsedMs}ms OK");
        }

        return result;
    }

    private static bool TryAcquirePipeSession(long sessionId)
    {
        lock (_pipeSessionSync)
        {
            if (_activePipeSessionId.HasValue)
            {
                // Auto-release stale sessions (crashed client left lock held)
                if ((DateTime.UtcNow - _sessionAcquiredAt).TotalSeconds > PipeSessionTimeoutSeconds)
                {
                    GD.PrintErr($"[MCP-PIPE] Force-releasing stale session {_activePipeSessionId} (held {(DateTime.UtcNow - _sessionAcquiredAt).TotalSeconds:F0}s)");
                    _activePipeSessionId = null;
                }
                else
                {
                    return false;
                }
            }
            _activePipeSessionId = sessionId;
            _sessionAcquiredAt = DateTime.UtcNow;
            return true;
        }
    }

    private static void ReleasePipeSession(long sessionId)
    {
        lock (_pipeSessionSync)
        {
            if (_activePipeSessionId == sessionId)
                _activePipeSessionId = null;
        }
    }

    private static void EnsurePipeSessionOwner(long sessionId)
    {
        lock (_pipeSessionSync)
        {
            if (_activePipeSessionId != sessionId)
            {
                throw new McpApiException(
                    "simulator_session_lost",
                    "This pipe session no longer owns the simulator runtime.");
            }
        }
    }

    private static string GetPipeErrorCode(Exception exception)
    {
        return GetStructuredErrorCode(exception) ?? "internal_error";
    }

    private static string SerializePipeError(string errorCode, string message)
    {
        return JsonSerializer.Serialize(new { error = message, error_code = errorCode }, _jsonOptions);
    }

    private static async Task WritePipeResponseAsync(Stream pipe, string responseJson, CancellationToken ct)
    {
        byte[] respBuf = Encoding.UTF8.GetBytes(responseJson);
        byte[] respLen = BitConverter.GetBytes(respBuf.Length);
        await pipe.WriteAsync(respLen, ct);
        await pipe.WriteAsync(respBuf, ct);
        await pipe.FlushAsync(ct);
    }

    /// <summary>
    /// Returns true when the full-run simulator is in pure-sim mode (ImmediateSimulatorClock).
    /// In this mode all game logic is synchronous — no Godot frame wait is needed, so we
    /// can call the simulator directly from the pipe-server thread pool thread instead of
    /// dispatching to the Godot main thread via RunOnMainThread.
    /// Safe because TestMode.IsOn bypasses all Godot-node access, and MCTS clients are
    /// strictly sequential (one request at a time per connection).
    /// </summary>
    private static bool IsPureSimActive()
        => MegaCrit.Sts2.Core.Simulation.CombatSimulationRuntime.IsPureCombatSimulator;

    private static async Task<string> ProcessPipeReset(JsonElement paramsElem)
    {
        var req = new FullRunSimulationResetRequest();
        if (paramsElem.ValueKind == JsonValueKind.Object)
        {
            if (paramsElem.TryGetProperty("character_id", out var c) && c.ValueKind == JsonValueKind.String)
                req.CharacterId = c.GetString();
            if (paramsElem.TryGetProperty("seed", out var s) && s.ValueKind == JsonValueKind.String)
                req.Seed = s.GetString();
            if (paramsElem.TryGetProperty("ascension_level", out var a) && a.ValueKind == JsonValueKind.Number)
                req.AscensionLevel = a.GetInt32();
        }

        // Pure-sim: call directly without frame dispatch (all logic is synchronous)
        if (IsPureSimActive() || FullRunTrainingEnvService.Instance.IsPureSimulator)
        {
            await FullRunTrainingEnvService.Instance.ResetAsync(req);
            var state = BuildFullRunEnvState();
            return JsonSerializer.Serialize(state, _jsonOptions);
        }

        var outerTask = RunOnMainThreadAsync(() =>
        {
            var resetTask = FullRunTrainingEnvService.Instance.ResetAsync(req);
            return resetTask.ContinueWith(_ => BuildFullRunEnvState(),
                TaskContinuationOptions.ExecuteSynchronously);
        });
        var stateViaMainThread = outerTask.GetAwaiter().GetResult();
        return JsonSerializer.Serialize(stateViaMainThread, _jsonOptions);
    }

    private static async Task<string> ProcessPipeStep(JsonElement paramsElem)
    {
        var actionReq = new FullRunSimulationActionRequest();
        if (paramsElem.ValueKind == JsonValueKind.Object)
        {
            if (paramsElem.TryGetProperty("action", out var a) && a.ValueKind == JsonValueKind.String)
                actionReq.Action = a.GetString() ?? "";
            if (paramsElem.TryGetProperty("index", out var i) && i.ValueKind == JsonValueKind.Number)
                actionReq.Index = i.GetInt32();
            if (paramsElem.TryGetProperty("card_index", out var ci) && ci.ValueKind == JsonValueKind.Number)
                actionReq.CardIndex = ci.GetInt32();
            if (paramsElem.TryGetProperty("hand_index", out var hi) && hi.ValueKind == JsonValueKind.Number)
                actionReq.HandIndex = hi.GetInt32();
            if (paramsElem.TryGetProperty("slot", out var sl) && sl.ValueKind == JsonValueKind.Number)
                actionReq.Slot = sl.GetInt32();
            if (paramsElem.TryGetProperty("target_id", out var ti) && ti.ValueKind == JsonValueKind.Number)
                actionReq.TargetId = ti.GetUInt32();
            if (paramsElem.TryGetProperty("target", out var t) && t.ValueKind == JsonValueKind.String)
                actionReq.Target = t.GetString();
        }

        // Pure-sim fast path: no Godot frame dispatch needed
        if (IsPureSimActive())
        {
            var simResult = await FullRunTrainingEnvService.Instance.StepAsync(actionReq);
            var state = BuildFullRunEnvState();
            var result = ShapeFullRunEnvStepResult(state, simResult.Accepted, simResult.Error, null);
            return JsonSerializer.Serialize(result, _jsonOptions);
        }

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
        var resultViaMainThread = outerTask.GetAwaiter().GetResult();
        return JsonSerializer.Serialize(resultViaMainThread, _jsonOptions);
    }

    private static async Task<string> ProcessPipeGetState()
    {
        if (IsPureSimActive())
            return JsonSerializer.Serialize(BuildFullRunEnvState(), _jsonOptions);

        var stateTask = RunOnMainThread(BuildFullRunEnvState);
        var state = stateTask.GetAwaiter().GetResult();
        return JsonSerializer.Serialize(state, _jsonOptions);
    }

    private static async Task<string> ProcessPipeLegalActions()
    {
        if (IsPureSimActive())
        {
            var snapshot = FullRunTrainingEnvService.Instance.GetState();
            var actions = BuildSimulatorLegalActionsFromSnapshot(snapshot);
            return JsonSerializer.Serialize(new { legal_actions = actions }, _jsonOptions);
        }

        var task = RunOnMainThread(() =>
        {
            var snapshot = FullRunTrainingEnvService.Instance.GetState();
            return BuildSimulatorLegalActionsFromSnapshot(snapshot);
        });
        var actionsViaMainThread = task.GetAwaiter().GetResult();
        return JsonSerializer.Serialize(new { legal_actions = actionsViaMainThread }, _jsonOptions);
    }

    private static async Task<string> ProcessPipeSaveState()
    {
        if (IsPureSimActive())
        {
            string stateId = FullRunTrainingEnvService.Instance.SaveState();
            return JsonSerializer.Serialize(new { state_id = stateId, cache_size = FullRunTrainingEnvService.Instance.StateCacheCount }, _jsonOptions);
        }

        var task = RunOnMainThread(() =>
        {
            string stateId = FullRunTrainingEnvService.Instance.SaveState();
            return new { state_id = stateId, cache_size = FullRunTrainingEnvService.Instance.StateCacheCount };
        });
        var result = task.GetAwaiter().GetResult();
        return JsonSerializer.Serialize(result, _jsonOptions);
    }

    private static async Task<string> ProcessPipeLoadState(JsonElement paramsElem)
    {
        string stateId = paramsElem.GetProperty("state_id").GetString() ?? "";

        if (IsPureSimActive())
        {
            await FullRunTrainingEnvService.Instance.LoadState(stateId);
            return JsonSerializer.Serialize(BuildFullRunEnvState(), _jsonOptions);
        }

        var outerTask = RunOnMainThreadAsync(() =>
        {
            var loadTask = FullRunTrainingEnvService.Instance.LoadState(stateId);
            return loadTask.ContinueWith(_ => BuildFullRunEnvState(),
                TaskContinuationOptions.ExecuteSynchronously);
        });
        var stateViaMainThread = outerTask.GetAwaiter().GetResult();
        return JsonSerializer.Serialize(stateViaMainThread, _jsonOptions);
    }

    private static async Task<string> ProcessPipeDeleteState(JsonElement paramsElem)
    {
        bool clearAll = paramsElem.TryGetProperty("clear_all", out var ca) && ca.ValueKind == JsonValueKind.True;
        string? stateId = paramsElem.TryGetProperty("state_id", out var si) ? si.GetString() : null;

        // Delete is always safe to call directly (pure memory operation)
        if (IsPureSimActive() || true)
        {
            if (clearAll)
            {
                FullRunTrainingEnvService.Instance.ClearStateCache();
                return JsonSerializer.Serialize(new { deleted = true, cache_size = 0 }, _jsonOptions);
            }
            bool deleted = FullRunTrainingEnvService.Instance.DeleteState(stateId ?? "");
            return JsonSerializer.Serialize(new { deleted, cache_size = FullRunTrainingEnvService.Instance.StateCacheCount }, _jsonOptions);
        }

        // Fallback (unreachable but kept for consistency)
        var task = RunOnMainThread(() =>
        {
            if (clearAll)
            {
                FullRunTrainingEnvService.Instance.ClearStateCache();
                return new { deleted = true, cache_size = 0 };
            }
            bool deleted = FullRunTrainingEnvService.Instance.DeleteState(stateId ?? "");
            return new { deleted, cache_size = FullRunTrainingEnvService.Instance.StateCacheCount };
        });
        var result = task.GetAwaiter().GetResult();
        return JsonSerializer.Serialize(result, _jsonOptions);
    }

    private static async Task<string> ProcessPipeBatchStep(JsonElement paramsElem)
    {
        if (!paramsElem.TryGetProperty("actions", out var actionsElem) || actionsElem.ValueKind != JsonValueKind.Array)
            return JsonSerializer.Serialize(new { error = "batch_step requires 'actions' array" });

        var actions = new System.Collections.Generic.List<FullRunSimulationActionRequest>();
        foreach (var a in actionsElem.EnumerateArray())
        {
            var req = new FullRunSimulationActionRequest();
            if (a.TryGetProperty("action", out var act) && act.ValueKind == JsonValueKind.String)
                req.Action = act.GetString() ?? "";
            if (a.TryGetProperty("index", out var idx) && idx.ValueKind == JsonValueKind.Number)
                req.Index = idx.GetInt32();
            if (a.TryGetProperty("card_index", out var ci) && ci.ValueKind == JsonValueKind.Number)
                req.CardIndex = ci.GetInt32();
            if (a.TryGetProperty("target_id", out var ti) && ti.ValueKind == JsonValueKind.Number)
                req.TargetId = ti.GetUInt32();
            if (a.TryGetProperty("slot", out var sl) && sl.ValueKind == JsonValueKind.Number)
                req.Slot = sl.GetInt32();
            actions.Add(req);
        }

        // Pure-sim fast path
        if (IsPureSimActive())
        {
            var batchResult = await FullRunTrainingEnvService.Instance.BatchStepAsync(actions);
            var state = BuildFullRunEnvState();
            var response = new System.Collections.Generic.Dictionary<string, object?>
            {
                ["accepted"] = batchResult.Accepted,
                ["error"] = batchResult.Error,
                ["steps_executed"] = batchResult.StepsExecuted,
            };
            foreach (var kv in state)
                response[kv.Key] = kv.Value;
            return JsonSerializer.Serialize(response, _jsonOptions);
        }

        var outerTask = RunOnMainThreadAsync(() =>
        {
            var batchTask = FullRunTrainingEnvService.Instance.BatchStepAsync(actions);
            return batchTask.ContinueWith(t =>
            {
                var batchResult = t.Result;
                var state = BuildFullRunEnvState();
                var response = new System.Collections.Generic.Dictionary<string, object?>
                {
                    ["accepted"] = batchResult.Accepted,
                    ["error"] = batchResult.Error,
                    ["steps_executed"] = batchResult.StepsExecuted,
                };
                foreach (var kv in state)
                    response[kv.Key] = kv.Value;
                return response;
            }, TaskContinuationOptions.ExecuteSynchronously);
        });
        var result = outerTask.GetAwaiter().GetResult();
        return JsonSerializer.Serialize(result, _jsonOptions);
    }
}
