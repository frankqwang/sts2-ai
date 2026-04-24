using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Nodes.Debug;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;

namespace STS2_MCP;

[ModInitializer("Initialize")]
public static partial class McpMod
{
    public const string Version = "0.1.0";

    private static HttpListener? _listener;
    private static Thread? _serverThread;
    private static readonly ConcurrentQueue<Action> _mainThreadQueue = new();
    private static NAiDecisionOverlay? _decisionOverlay;
    private static string? _decisionOverlayFile;
    private static bool _ftuesDisabled;
    private static bool _fastModeSet;
    internal static readonly JsonSerializerOptions _jsonOptions = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
    };

    public static void Initialize()
    {
        try
        {
            var tree = (SceneTree)Engine.GetMainLoop();
            tree.Connect(SceneTree.SignalName.ProcessFrame, Callable.From(ProcessMainThreadQueue));

            _decisionOverlayFile = CommandLineHelper.GetValue("mcp-decision-overlay-file");

            int port = 15526;
            string? mcpPortArg = CommandLineHelper.GetValue("mcp-port");
            if (!string.IsNullOrWhiteSpace(mcpPortArg) && int.TryParse(mcpPortArg, out int parsedPort))
            {
                port = Math.Clamp(parsedPort, 1024, 65535);
            }

            _listener = new HttpListener();
            _listener.Prefixes.Add($"http://localhost:{port}/");
            _listener.Prefixes.Add($"http://127.0.0.1:{port}/");
            _listener.Start();

            _serverThread = new Thread(ServerLoop)
            {
                IsBackground = true,
                Name = "STS2_MCP_Spectator_Server"
            };
            _serverThread.Start();

            GD.Print($"[STS2 MCP Spectator] v{Version} server started on http://localhost:{port}/");
            if (!string.IsNullOrWhiteSpace(_decisionOverlayFile))
            {
                GD.Print($"[STS2 MCP Spectator] decision overlay file: {_decisionOverlayFile}");
            }
        }
        catch (Exception ex)
        {
            GD.PrintErr($"[STS2 MCP Spectator] Failed to start: {ex}");
        }
    }

    private static void ProcessMainThreadQueue()
    {
        int processed = 0;
        while (_mainThreadQueue.TryDequeue(out Action? action) && processed < 50)
        {
            try
            {
                action();
            }
            catch (Exception ex)
            {
                GD.PrintErr($"[STS2 MCP Spectator] Main thread action error: {ex}");
            }

            processed++;
        }

        EnsureFtuesDisabled();
        EnsureFastModeInstant();
        EnsureDecisionOverlayAttached();

        // Manually drive overlay update since Godot won't call _Process on mod assemblies
        if (GodotObject.IsInstanceValid(_decisionOverlay))
        {
            _decisionOverlay.ManualProcess();
        }
    }

    private static void EnsureFastModeInstant()
    {
        if (_fastModeSet)
            return;
        try
        {
            var saveManager = SaveManager.Instance;
            if (saveManager == null)
                return;
            saveManager.PrefsSave.FastMode = FastModeType.Instant;
            _fastModeSet = true;
            GD.Print("[STS2 MCP Spectator] FastMode set to Instant for spectator mode");
        }
        catch
        {
            // Save not ready yet; retry next frame.
        }
    }

    private static void EnsureFtuesDisabled()
    {
        if (_ftuesDisabled)
            return;

        try
        {
            var saveManager = SaveManager.Instance;
            if (saveManager == null)
                return;

            _ = saveManager.CurrentProfileId;
            saveManager.SetFtuesEnabled(enabled: false);
            _ftuesDisabled = true;
            GD.Print("[STS2 MCP Spectator] FTUE disabled for spectator mode");
        }
        catch (InvalidOperationException)
        {
            // Save/profile boot is still in progress; retry next frame.
        }
    }

    internal static Task<T> RunOnMainThread<T>(Func<T> func)
    {
        var tcs = new TaskCompletionSource<T>();
        _mainThreadQueue.Enqueue(() =>
        {
            try
            {
                tcs.SetResult(func());
            }
            catch (Exception ex)
            {
                tcs.SetException(ex);
            }
        });
        return tcs.Task;
    }

    internal static Task RunOnMainThread(Action action)
    {
        var tcs = new TaskCompletionSource<bool>();
        _mainThreadQueue.Enqueue(() =>
        {
            try
            {
                action();
                tcs.SetResult(true);
            }
            catch (Exception ex)
            {
                tcs.SetException(ex);
            }
        });
        return tcs.Task;
    }

    private static void EnsureDecisionOverlayAttached()
    {
        if (string.IsNullOrWhiteSpace(_decisionOverlayFile))
        {
            return;
        }

        if (GodotObject.IsInstanceValid(_decisionOverlay))
        {
            if (_decisionOverlay!.OverlayFilePath != _decisionOverlayFile)
            {
                _decisionOverlay.OverlayFilePath = _decisionOverlayFile;
            }
            return;
        }

        if (Engine.GetMainLoop() is not SceneTree tree)
        {
            return;
        }

        Node parent = tree.Root;
        if (!GodotObject.IsInstanceValid(parent))
        {
            return;
        }

        _decisionOverlay = new NAiDecisionOverlay
        {
            Name = "STS2McpDecisionOverlay",
            OverlayFilePath = Path.GetFullPath(_decisionOverlayFile)
        };
        parent.AddChild(_decisionOverlay);
        _decisionOverlay.Initialize();
        GD.Print($"[STS2 MCP Spectator] decision overlay attached and initialized");
    }

    private static void ServerLoop()
    {
        while (_listener?.IsListening == true)
        {
            try
            {
                HttpListenerContext context = _listener.GetContext();
                ThreadPool.QueueUserWorkItem(_ => HandleRequest(context));
            }
            catch (HttpListenerException)
            {
                break;
            }
            catch (ObjectDisposedException)
            {
                break;
            }
        }
    }

    private static void HandleRequest(HttpListenerContext context)
    {
        try
        {
            HttpListenerRequest request = context.Request;
            HttpListenerResponse response = context.Response;
            response.Headers.Add("Access-Control-Allow-Origin", "*");
            response.Headers.Add("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            response.Headers.Add("Access-Control-Allow-Headers", "Content-Type");

            if (request.HttpMethod == "OPTIONS")
            {
                response.StatusCode = 204;
                response.Close();
                return;
            }

            string path = request.Url?.AbsolutePath ?? "/";
            if (path == "/")
            {
                SendJson(response, new
                {
                    message = $"Hello from STS2 MCP Spectator v{Version}",
                    status = "ok",
                    mode = "spectator_only"
                });
            }
            else if (path == "/api/game_bridge/rpc")
            {
                if (request.HttpMethod == "POST")
                {
                    HandleBridgeRpc(request, response);
                }
                else
                {
                    SendError(response, 405, "Method not allowed");
                }
            }
            else
            {
                SendError(response, 404, "Not found");
            }
        }
        catch (Exception ex)
        {
            try
            {
                SendError(context.Response, 500, $"Internal error: {ex.Message}");
            }
            catch
            {
                // Response may already be closed.
            }
        }
    }

    private static void HandleGetState(HttpListenerRequest request, HttpListenerResponse response)
    {
        string format = request.QueryString["format"] ?? "json";

        try
        {
            Dictionary<string, object?> state = RunOnMainThread(BuildGameState).GetAwaiter().GetResult();
            if (string.Equals(format, "markdown", StringComparison.OrdinalIgnoreCase))
            {
                SendText(response, FormatAsMarkdown(state), "text/markdown");
                return;
            }

            SendJson(response, state);
        }
        catch (Exception ex)
        {
            SendError(response, 500, $"Failed to read game state: {ex.Message}");
        }
    }

    private static void HandlePostAction(HttpListenerRequest request, HttpListenerResponse response)
    {
        string body;
        using (var reader = new StreamReader(request.InputStream, request.ContentEncoding))
        {
            body = reader.ReadToEnd();
        }

        Dictionary<string, JsonElement>? parsed;
        try
        {
            parsed = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(body);
        }
        catch
        {
            SendError(response, 400, "Invalid JSON");
            return;
        }

        if (parsed == null || !parsed.TryGetValue("action", out JsonElement actionElem))
        {
            SendError(response, 400, "Missing 'action' field");
            return;
        }

        string action = actionElem.GetString() ?? string.Empty;
        try
        {
            Dictionary<string, object?> result = RunOnMainThread(() => ExecuteAction(action, parsed)).GetAwaiter().GetResult();
            SendJson(response, result);
        }
        catch (Exception ex)
        {
            SendError(response, 500, $"Action failed: {ex.Message}");
        }
    }
}
