using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.IO.Pipes;
using System.Linq;
using System.Reflection;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Simulation;
using MegaCrit.Sts2.Core.TestSupport;
using MegaCrit.Sts2.Core.Training;

namespace HeadlessSim;

internal static class Program
{
	private sealed class RequestStateCache
	{
		public FullRunSimulationStateSnapshot? Snapshot { get; set; }

		public FullRunApiState? ApiState { get; set; }
	}

	private static readonly JsonSerializerOptions JsonOptions = new JsonSerializerOptions
	{
		PropertyNameCaseInsensitive = true,
		DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
	};

	public static async Task Main(string[] args)
	{
		// ThreadPool min threads: configurable via env var for A/B testing.
		// Default .NET min is often CPU core count, causing Task.Yield()
		// continuation delays when multiple HeadlessSim processes compete.
		string? minThreadsEnv = Environment.GetEnvironmentVariable("STS2_MIN_THREADS");
		int minThreads = 32; // default boost
		if (!string.IsNullOrEmpty(minThreadsEnv) && int.TryParse(minThreadsEnv, out int parsed) && parsed > 0)
			minThreads = parsed;
		System.Threading.ThreadPool.SetMinThreads(minThreads, minThreads);
		{
			int mw, mio;
			System.Threading.ThreadPool.GetMinThreads(out mw, out mio);
			Console.Error.WriteLine($"[THREADPOOL] MinThreads={mw} (requested={minThreads})");
		}

		HostOptions options = HostOptions.Parse(args);
		BootstrapStandaloneRuntime();
		if (options.ExportCardRuntimeTextsPath != null)
		{
			await ExportCardRuntimeTextsAsync(options.ExportCardRuntimeTextsPath, options.ExportLocales);
			return;
		}
		using IDisposable standaloneScope = FullRunTrainingEnvService.EnterStandaloneMode();
		FullRunTrainingEnvService service = FullRunTrainingEnvService.Instance;

		if (options.UseStdio)
		{
			Console.Error.WriteLine("HeadlessSim: stdio mode ready");
			await RunStdioAsync(service);
			return;
		}

		Console.Error.WriteLine($"HeadlessSim: pipe mode ready on \\\\.\\pipe\\{options.PipeName}");
		await RunPipeServerAsync(service, options);
	}

	private static void BootstrapStandaloneRuntime()
	{
		TestMode.IsOn = true;
		UserDataPathProvider.IsRunningModded = false;
		SaveManager saveManager = SaveManager.Instance;
		saveManager.InitSettingsDataForTest();
		ModelDb.Init();
		ModelIdSerializationCache.Init();
		ModelDb.InitIds();
		saveManager.InitProfileId(profileId: 1);
		saveManager.InitProgressData();
		saveManager.InitPrefsDataForTest();
	}

	private static async Task ExportCardRuntimeTextsAsync(string outputPath, IReadOnlyList<string> locales)
	{
		if (LocManager.Instance == null)
		{
			LocManager.Initialize();
		}

		List<CardRuntimeTextRecord> rows = new List<CardRuntimeTextRecord>();
		List<CardModel> cards = ModelDb.AllCards.OrderBy((CardModel c) => c.Id.Entry, StringComparer.OrdinalIgnoreCase).ToList();
		foreach (string locale in locales)
		{
			LocManager.Instance.SetLanguage(locale);
			foreach (CardModel card in cards)
			{
				rows.Add(new CardRuntimeTextRecord
				{
					Id = card.Id.Entry.ToLowerInvariant(),
					ClassName = card.GetType().Name,
					Locale = locale,
					Title = card.Title,
					DescriptionRuntime = card.GetDescriptionForPile(PileType.None),
					UpgradePreviewRuntime = card.GetDescriptionForUpgradePreview()
				});
			}
		}

		string? directory = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(directory))
		{
			Directory.CreateDirectory(directory);
		}

		JsonSerializerOptions exportOptions = new JsonSerializerOptions(JsonOptions)
		{
			WriteIndented = true
		};
		await File.WriteAllTextAsync(outputPath, JsonSerializer.Serialize(rows, exportOptions), Encoding.UTF8);
		Console.Error.WriteLine($"HeadlessSim: exported runtime card texts -> {outputPath} ({rows.Count} rows)");
	}

	private sealed class CardRuntimeTextRecord
	{
		public string Id { get; set; } = string.Empty;

		public string ClassName { get; set; } = string.Empty;

		public string Locale { get; set; } = string.Empty;

		public string Title { get; set; } = string.Empty;

		public string DescriptionRuntime { get; set; } = string.Empty;

		public string UpgradePreviewRuntime { get; set; } = string.Empty;
	}

	private static async Task RunStdioAsync(FullRunTrainingEnvService service)
	{
		string? line;
		while ((line = await Console.In.ReadLineAsync()) != null)
		{
			line = line.TrimStart('\uFEFF');
			if (string.IsNullOrWhiteSpace(line))
			{
				continue;
			}

			string responseJson;
			try
			{
				responseJson = await ProcessPipeRequestAsync(service, line);
			}
			catch (Exception ex)
			{
				responseJson = SerializePipeError(GetStructuredErrorCode(ex) ?? "internal_error", ex.ToString());
			}

			await Console.Out.WriteLineAsync(responseJson);
			await Console.Out.FlushAsync();
		}
	}

	private static async Task RunPipeServerAsync(FullRunTrainingEnvService service, HostOptions options)
	{
		using CancellationTokenSource cts = new CancellationTokenSource();
		Console.CancelKeyPress += (_, eventArgs) =>
		{
			eventArgs.Cancel = true;
			cts.Cancel();
		};

		PipeSessionManager sessions = new PipeSessionManager();
		while (!cts.IsCancellationRequested)
		{
			NamedPipeServerStream? server = null;
			try
			{
				server = new NamedPipeServerStream(
					options.PipeName,
					PipeDirection.InOut,
					NamedPipeServerStream.MaxAllowedServerInstances,
					PipeTransmissionMode.Byte,
					PipeOptions.Asynchronous);

				await server.WaitForConnectionAsync(cts.Token);
				NamedPipeServerStream connectedServer = server;
				_ = Task.Run(
					() => HandlePipeConnectionAsync(service, connectedServer, sessions, options, cts.Token),
					cts.Token);
				server = null;
			}
			catch (OperationCanceledException)
			{
				break;
			}
			catch (Exception ex)
			{
				Console.Error.WriteLine($"HeadlessSim: pipe listener error: {ex}");
				await Task.Delay(100, cts.Token);
			}
			finally
			{
				server?.Dispose();
			}
		}
	}

	private static async Task HandlePipeConnectionAsync(
		FullRunTrainingEnvService service,
		NamedPipeServerStream pipe,
		PipeSessionManager sessions,
		HostOptions options,
		CancellationToken cancellationToken)
	{
		long sessionId = sessions.TryAcquire();
		if (sessionId < 0)
		{
			using (pipe)
			{
				if (options.Protocol is HostProtocol.Binary or HostProtocol.Proto)
				{
					await WritePipeMessageAsync(
						pipe,
						BinaryProtocol.BuildErrorResponse(
							BinaryOpcode.Handshake,
							BinaryStatus.ProtocolError,
							"simulator_busy",
							"The simulator runtime is already owned by another active pipe session."),
						cancellationToken);
				}
				else
				{
					await WritePipeMessageAsync(
						pipe,
						SerializePipeError("simulator_busy", "The simulator runtime is already owned by another active pipe session."),
						cancellationToken);
				}
			}
			return;
		}

		try
		{
			using (pipe)
			{
				if (options.Protocol == HostProtocol.Binary)
				{
					await WritePipeMessageAsync(pipe, BinaryProtocol.BuildHandshakeResponse(), cancellationToken);
				}
				else if (options.Protocol == HostProtocol.Proto)
				{
					await WritePipeMessageAsync(pipe, ProtoStateBuilder.BuildHandshakeResponse(), cancellationToken);
				}
				else
				{
					await WritePipeMessageAsync(pipe, JsonSerializer.Serialize(new { ok = true }, JsonOptions), cancellationToken);
				}

				BinarySessionState? binarySession = options.Protocol == HostProtocol.Binary ? new BinarySessionState() : null;

				while (pipe.IsConnected && !cancellationToken.IsCancellationRequested)
				{
					byte[]? requestBytes = await ReadPipeMessageBytesAsync(pipe, options.ReadTimeout, cancellationToken);
					if (requestBytes == null)
					{
						break;
					}

					if (options.Protocol is HostProtocol.Binary or HostProtocol.Proto)
					{
						byte[] responseBytes;
						try
						{
							using CancellationTokenSource requestCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
							requestCts.CancelAfter(options.RequestTimeout);
							if (options.Protocol == HostProtocol.Proto)
							{
								responseBytes = await ProcessProtoRequestAsync(service, requestBytes).WaitAsync(requestCts.Token);
							}
							else
							{
								responseBytes = await ProcessBinaryRequestAsync(service, binarySession!, requestBytes).WaitAsync(requestCts.Token);
							}
						}
						catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
						{
							responseBytes = BinaryProtocol.BuildErrorResponse(
								BinaryProtocol.ParseOpcode(requestBytes),
								BinaryStatus.ProtocolError,
								"request_timeout",
								$"Request processing timed out after {options.RequestTimeout.TotalSeconds:F0}s");
						}
						catch (Exception ex)
						{
							Console.Error.WriteLine($"HeadlessSim: {options.Protocol} request error opcode={SafeParseOpcode(requestBytes)}: {ex}");
							responseBytes = BinaryProtocol.BuildErrorResponse(
								SafeParseOpcode(requestBytes),
								GetBinaryErrorStatus(ex),
								GetStructuredErrorCode(ex) ?? "internal_error",
								ex.Message);
						}

						await WritePipeMessageAsync(pipe, responseBytes, cancellationToken);
					}
					else
					{
						string requestJson = Encoding.UTF8.GetString(requestBytes);
						string responseJson;
						try
						{
							using CancellationTokenSource requestCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
							requestCts.CancelAfter(options.RequestTimeout);
							responseJson = await ProcessPipeRequestAsync(service, requestJson).WaitAsync(requestCts.Token);
						}
						catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
						{
							responseJson = SerializePipeError(
								"request_timeout",
								$"Request processing timed out after {options.RequestTimeout.TotalSeconds:F0}s");
						}
						catch (Exception ex)
						{
							Console.Error.WriteLine($"HeadlessSim: json request error request={requestJson}: {ex}");
							responseJson = SerializePipeError(GetStructuredErrorCode(ex) ?? "internal_error", ex.Message);
						}

						await WritePipeMessageAsync(pipe, responseJson, cancellationToken);
					}
				}
			}
		}
		catch (IOException)
		{
		}
		catch (OperationCanceledException)
		{
		}
		catch (Exception ex)
		{
			Console.Error.WriteLine($"HeadlessSim: pipe connection error: {ex}");
		}
		finally
		{
			sessions.Release(sessionId);
		}
	}

	private static async Task<string?> ReadPipeMessageAsync(Stream stream, TimeSpan readTimeout, CancellationToken cancellationToken)
	{
		byte[]? payload = await ReadPipeMessageBytesAsync(stream, readTimeout, cancellationToken);
		return payload == null ? null : Encoding.UTF8.GetString(payload);
	}

	private static async Task<byte[]?> ReadPipeMessageBytesAsync(Stream stream, TimeSpan readTimeout, CancellationToken cancellationToken)
	{
		using CancellationTokenSource readCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
		readCts.CancelAfter(readTimeout);

		byte[] lenBuffer = new byte[4];
		int lenRead;
		try
		{
			lenRead = await ReadExactAsync(stream, lenBuffer, readCts.Token);
		}
		catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
		{
			return null;
		}

		if (lenRead == 0)
		{
			return null;
		}

		if (lenRead < 4)
		{
			throw new EndOfStreamException("Incomplete pipe length prefix.");
		}

		int messageLength = BinaryPrimitives.ReadInt32LittleEndian(lenBuffer);
		if (messageLength <= 0 || messageLength > 10_000_000)
		{
			throw new InvalidOperationException($"Invalid pipe message length: {messageLength}");
		}

		byte[] messageBuffer = new byte[messageLength];
		int messageRead = await ReadExactAsync(stream, messageBuffer, readCts.Token);
		if (messageRead < messageLength)
		{
			throw new EndOfStreamException("Incomplete pipe payload.");
		}

		return messageBuffer;
	}

	private static async Task<int> ReadExactAsync(Stream stream, byte[] buffer, CancellationToken cancellationToken)
	{
		int offset = 0;
		while (offset < buffer.Length)
		{
			int read = await stream.ReadAsync(buffer.AsMemory(offset, buffer.Length - offset), cancellationToken);
			if (read == 0)
			{
				return offset;
			}

			offset += read;
		}

		return offset;
	}

	private static async Task WritePipeMessageAsync(Stream stream, string payload, CancellationToken cancellationToken)
	{
		byte[] body = Encoding.UTF8.GetBytes(payload);
		await WritePipeMessageAsync(stream, body, cancellationToken);
	}

	private static async Task WritePipeMessageAsync(Stream stream, byte[] body, CancellationToken cancellationToken)
	{
		byte[] prefix = new byte[4];
		BinaryPrimitives.WriteInt32LittleEndian(prefix, body.Length);
		await stream.WriteAsync(prefix, cancellationToken);
		await stream.WriteAsync(body, cancellationToken);
		await stream.FlushAsync(cancellationToken);
	}

	private static async Task<byte[]> ProcessBinaryRequestAsync(
		FullRunTrainingEnvService service,
		BinarySessionState session,
		byte[] requestBytes)
	{
		long requestStart = Stopwatch.GetTimestamp();
		BinaryOpcode opcode = BinaryProtocol.ParseOpcode(requestBytes);
		RequestStateCache cache = new RequestStateCache();
		try
		{
			return opcode switch
			{
				BinaryOpcode.Reset => await ProcessBinaryResetAsync(service, session, requestBytes, cache),
				BinaryOpcode.State => ProcessBinaryState(service, session, cache),
				BinaryOpcode.Step => await ProcessBinaryStepAsync(service, session, requestBytes, cache),
				BinaryOpcode.BatchStep => await ProcessBinaryBatchStepAsync(service, session, requestBytes, cache),
				BinaryOpcode.SaveState => ProcessBinarySaveState(service),
				BinaryOpcode.ExportState => ProcessBinaryExportState(service, requestBytes),
				BinaryOpcode.LoadState => await ProcessBinaryLoadStateAsync(service, session, requestBytes, cache),
				BinaryOpcode.ImportState => await ProcessBinaryImportStateAsync(service, session, requestBytes, cache),
				BinaryOpcode.DeleteState => ProcessBinaryDeleteState(service, requestBytes),
				BinaryOpcode.PerfStats => BinaryProtocol.BuildPerfStatsResponse(FullRunSimulationDiagnostics.Snapshot()),
				BinaryOpcode.ResetPerfStats => ProcessBinaryResetPerfStats(),
			BinaryOpcode.StepLocalPolicy => await ProcessBinaryStepLocalPolicyAsync(service, session, cache),
			BinaryOpcode.LoadOrtModel => ProcessBinaryLoadOrtModel(requestBytes),
			BinaryOpcode.RunCombatLocal => await ProcessBinaryRunCombatLocalAsync(service, session, requestBytes, cache),
			BinaryOpcode.SkipCombat => await ProcessBinarySkipCombatAsync(service, session, cache),
			BinaryOpcode.SearchCombatMcts => await ProcessBinarySearchCombatMctsAsync(service, requestBytes, cache),
				_ => BinaryProtocol.BuildErrorResponse(opcode, BinaryStatus.ProtocolError, "unknown_method", $"Unknown opcode: {(byte)opcode}")
			};
		}
		finally
		{
			double elapsedMs = (Stopwatch.GetTimestamp() - requestStart) * 1000.0 / Stopwatch.Frequency;
			FullRunSimulationDiagnostics.RecordTiming($"request.{opcode.ToString().ToLowerInvariant()}.total_ms", elapsedMs);
			FullRunSimulationDiagnostics.RecordTiming("request.binary_total_ms", elapsedMs);
			FullRunSimulationDiagnostics.Increment($"request.{opcode.ToString().ToLowerInvariant()}.count");
			FullRunSimulationDiagnostics.Increment("request.binary.count");
		}
	}

	// ================================================================
	// Proto protocol request router — reuses Binary request parsing,
	// returns proto-serialized state payloads.
	// ================================================================

	private static async Task<byte[]> ProcessProtoRequestAsync(
		FullRunTrainingEnvService service,
		byte[] requestBytes)
	{
		long requestStart = Stopwatch.GetTimestamp();
		BinaryOpcode opcode = BinaryProtocol.ParseOpcode(requestBytes);
		RequestStateCache cache = new RequestStateCache();
		try
		{
			return opcode switch
			{
				BinaryOpcode.Reset => await ProcessProtoResetAsync(service, requestBytes, cache),
				BinaryOpcode.State => ProcessProtoState(service, cache),
				BinaryOpcode.Step => await ProcessProtoStepAsync(service, requestBytes, cache),
				BinaryOpcode.BatchStep => await ProcessProtoBatchStepAsync(service, requestBytes, cache),
				// Non-state responses: identical format, delegate to Binary/Proto builders
				BinaryOpcode.SaveState => ProtoStateBuilder.BuildSaveStateResponse(
					service.SaveState(), service.StateCacheCount),
				BinaryOpcode.ExportState => ProcessProtoExportState(service, requestBytes),
				BinaryOpcode.LoadState => await ProcessProtoLoadStateAsync(service, requestBytes, cache),
				BinaryOpcode.ImportState => await ProcessProtoImportStateAsync(service, requestBytes, cache),
				BinaryOpcode.DeleteState => ProcessProtoDeleteState(service, requestBytes),
				BinaryOpcode.PerfStats => ProtoStateBuilder.BuildPerfStatsResponse(FullRunSimulationDiagnostics.Snapshot()),
				BinaryOpcode.ResetPerfStats => ProcessProtoResetPerfStats(),
				BinaryOpcode.StepLocalPolicy => await ProcessProtoStepLocalPolicyAsync(service, cache),
				BinaryOpcode.LoadOrtModel => ProcessBinaryLoadOrtModel(requestBytes),
				BinaryOpcode.RunCombatLocal => await ProcessProtoRunCombatLocalAsync(service, requestBytes, cache),
				BinaryOpcode.SkipCombat => await ProcessProtoSkipCombatAsync(service, cache),
				BinaryOpcode.SearchCombatMcts => await ProcessBinarySearchCombatMctsAsync(service, requestBytes, cache),
				_ => ProtoStateBuilder.BuildErrorResponse(opcode, BinaryStatus.ProtocolError, "unknown_method", $"Unknown opcode: {(byte)opcode}")
			};
		}
		finally
		{
			double elapsedMs = (Stopwatch.GetTimestamp() - requestStart) * 1000.0 / Stopwatch.Frequency;
			FullRunSimulationDiagnostics.RecordTiming($"request.{opcode.ToString().ToLowerInvariant()}.total_ms", elapsedMs);
			FullRunSimulationDiagnostics.RecordTiming("request.proto_total_ms", elapsedMs);
			FullRunSimulationDiagnostics.Increment($"request.{opcode.ToString().ToLowerInvariant()}.count");
			FullRunSimulationDiagnostics.Increment("request.proto.count");
		}
	}

	private static async Task<byte[]> ProcessProtoResetAsync(
		FullRunTrainingEnvService service, byte[] requestBytes, RequestStateCache cache)
	{
		FullRunSimulationResetRequest request = BinaryProtocol.ParseResetRequest(requestBytes);
		FullRunSimulationStateSnapshot snapshot;
		using (FullRunSimulationDiagnostics.Measure("request.reset.runtime_ms"))
		{
			snapshot = await service.ResetAsync(request);
		}
		using (FullRunSimulationDiagnostics.Measure("request.proto_encode_ms"))
		{
			return ProtoStateBuilder.BuildStateResponse(BinaryOpcode.Reset, snapshot);
		}
	}

	private static byte[] ProcessProtoState(FullRunTrainingEnvService service, RequestStateCache cache)
	{
		FullRunSimulationStateSnapshot snapshot;
		using (FullRunSimulationDiagnostics.Measure("request.get_state.runtime_ms"))
		{
			snapshot = GetSnapshot(service, cache);
		}
		using (FullRunSimulationDiagnostics.Measure("request.proto_encode_ms"))
		{
			return ProtoStateBuilder.BuildStateResponse(BinaryOpcode.State, snapshot);
		}
	}

	private static async Task<byte[]> ProcessProtoStepAsync(
		FullRunTrainingEnvService service, byte[] requestBytes, RequestStateCache cache)
	{
		FullRunSimulationActionRequest action = BinaryProtocol.ParseActionRequest(requestBytes);
		FullRunSimulationStepResult result;
		using (FullRunSimulationDiagnostics.Measure("request.step.runtime_ms"))
		{
			result = await service.StepAsync(action);
		}
		FullRunSimulationStateSnapshot snapshot = result.State ?? GetSnapshot(service, cache);
		// Auto-advance through non-decision states (same logic as binary)
		int autoAdvanceCount = 0;
		const int maxAutoAdvance = 50;
		while (!IsDecisionState(snapshot) && autoAdvanceCount < maxAutoAdvance)
		{
			autoAdvanceCount++;
			FullRunSimulationDiagnostics.Increment("step.auto_advance");
			if (snapshot.LegalActions.Count == 0)
			{
				var waitResult = await service.StepAsync(new FullRunSimulationActionRequest { Action = "wait" });
				snapshot = waitResult.State ?? GetSnapshot(service, cache);
			}
			else
			{
				var la = snapshot.LegalActions[0];
				var autoAction = new FullRunSimulationActionRequest
				{
					Action = la.Action ?? "",
					Index = la.Index,
					Col = la.Col,
					Row = la.Row,
					Slot = la.Slot,
					TargetId = la.TargetId,
					Target = la.Target,
					CardIndex = la.CardIndex,
					Value = la.Label,
				};
				var autoResult = await service.StepAsync(autoAction);
				snapshot = autoResult.State ?? GetSnapshot(service, cache);
			}
		}
		using (FullRunSimulationDiagnostics.Measure("request.proto_encode_ms"))
		{
			return ProtoStateBuilder.BuildStepResponse(result, snapshot);
		}
	}

	private static async Task<byte[]> ProcessProtoBatchStepAsync(
		FullRunTrainingEnvService service, byte[] requestBytes, RequestStateCache cache)
	{
		List<FullRunSimulationActionRequest> actions = BinaryProtocol.ParseBatchActionRequest(requestBytes);
		FullRunSimulationBatchStepResult result;
		using (FullRunSimulationDiagnostics.Measure("request.batch_step.runtime_ms"))
		{
			result = await service.BatchStepAsync(actions);
		}
		FullRunSimulationStateSnapshot snapshot = result.State ?? GetSnapshot(service, cache);
		using (FullRunSimulationDiagnostics.Measure("request.proto_encode_ms"))
		{
			return ProtoStateBuilder.BuildBatchStepResponse(result, snapshot);
		}
	}

	private static byte[] ProcessProtoExportState(FullRunTrainingEnvService service, byte[] requestBytes)
	{
		// TODO: ExportState not yet implemented on FullRunTrainingEnvService
		throw new NotImplementedException("ExportState not yet available in proto mode");
	}

	private static async Task<byte[]> ProcessProtoLoadStateAsync(
		FullRunTrainingEnvService service, byte[] requestBytes, RequestStateCache cache)
	{
		string stateId = BinaryProtocol.ParseStateIdRequest(BinaryOpcode.LoadState, requestBytes);
		using (FullRunSimulationDiagnostics.Measure("request.load_state.runtime_ms"))
		{
			service.LoadState(stateId);
		}
		FullRunSimulationStateSnapshot snapshot = GetSnapshot(service, cache);
		return ProtoStateBuilder.BuildStateResponse(BinaryOpcode.LoadState, snapshot);
	}

	private static async Task<byte[]> ProcessProtoImportStateAsync(
		FullRunTrainingEnvService service, byte[] requestBytes, RequestStateCache cache)
	{
		// TODO: ImportState not yet implemented on FullRunTrainingEnvService
		throw new NotImplementedException("ImportState not yet available in proto mode");
	}

	private static byte[] ProcessProtoDeleteState(FullRunTrainingEnvService service, byte[] requestBytes)
	{
		bool clearAll = BinaryProtocol.ParseDeleteClearAll(requestBytes, out string? stateId);
		bool deleted;
		if (clearAll) { service.ClearStateCache(); deleted = true; }
		else if (stateId != null) { deleted = service.DeleteState(stateId); }
		else { deleted = false; }
		return ProtoStateBuilder.BuildDeleteStateResponse(deleted, service.StateCacheCount);
	}

	private static byte[] ProcessProtoResetPerfStats()
	{
		FullRunSimulationDiagnostics.Reset();
		return ProtoStateBuilder.BuildResetPerfStatsResponse();
	}

	private static async Task<byte[]> ProcessProtoSkipCombatAsync(
		FullRunTrainingEnvService service, RequestStateCache cache)
	{
		FullRunSimulationStepResult skipResult = await service.StepAsync(
			new FullRunSimulationActionRequest { Action = "skip_combat" });
		FullRunSimulationStateSnapshot snapshot = skipResult.State ?? GetSnapshot(service, cache);
		return ProtoStateBuilder.BuildStepResponse(skipResult, snapshot);
	}

	private static async Task<byte[]> ProcessProtoStepLocalPolicyAsync(
		FullRunTrainingEnvService service, RequestStateCache cache)
	{
		// 委托给 binary 版本的 StepLocalPolicy 逻辑 — ORT 推理和 state 序列化无关
		// 但需要用 proto 编码 state 响应，所以不能直接转发
		if (_ortPolicy == null)
		{
			return ProtoStateBuilder.BuildErrorResponse(
				BinaryOpcode.StepLocalPolicy, BinaryStatus.SimulatorError,
				"no_ort_model", "No ORT model loaded. Call load_ort_model first.");
		}
		FullRunSimulationStateSnapshot snapshot = GetSnapshot(service, cache);
		// 使用 ORT 策略选择 action，然后 step
		int actionIndex = _ortPolicy.SelectAction(snapshot, null, _ortRng).Item1;
		if (actionIndex < 0 || actionIndex >= snapshot.LegalActions.Count)
		{
			return ProtoStateBuilder.BuildErrorResponse(
				BinaryOpcode.StepLocalPolicy, BinaryStatus.SimulatorError,
				"invalid_action_index", $"ORT policy returned invalid action index: {actionIndex}");
		}
		var la = snapshot.LegalActions[actionIndex];
		var action = new FullRunSimulationActionRequest
		{
			Action = la.Action ?? "",
			Index = la.Index,
			Col = la.Col,
			Row = la.Row,
			Slot = la.Slot,
			TargetId = la.TargetId,
			Target = la.Target,
			CardIndex = la.CardIndex,
			Value = la.Label,
		};
		FullRunSimulationStepResult result = await service.StepAsync(action);
		FullRunSimulationStateSnapshot nextSnapshot = result.State ?? GetSnapshot(service, cache);
		return ProtoStateBuilder.BuildStepResponse(result, nextSnapshot);
	}

	private static async Task<byte[]> ProcessProtoRunCombatLocalAsync(
		FullRunTrainingEnvService service, byte[] requestBytes, RequestStateCache cache)
	{
		// RunCombatLocal 的响应格式比较特殊（timing 等），直接复用 binary 实现
		// 因为 state payload 只在最终结果中，用 BinarySessionState 临时包装
		BinarySessionState tempSession = new BinarySessionState();
		return await ProcessBinaryRunCombatLocalAsync(service, tempSession, requestBytes, cache);
	}

	private static async Task<byte[]> ProcessBinaryResetAsync(
		FullRunTrainingEnvService service,
		BinarySessionState session,
		byte[] requestBytes,
		RequestStateCache cache)
	{
		FullRunSimulationResetRequest request = BinaryProtocol.ParseResetRequest(requestBytes);
		FullRunSimulationStateSnapshot snapshot;
		using (FullRunSimulationDiagnostics.Measure("request.reset.runtime_ms"))
		{
			snapshot = await service.ResetAsync(request);
		}

		using (FullRunSimulationDiagnostics.Measure("request.binary_encode_ms"))
		{
			return BinaryProtocol.BuildStateResponse(BinaryOpcode.Reset, session, snapshot);
		}
	}

	private static byte[] ProcessBinaryState(
		FullRunTrainingEnvService service,
		BinarySessionState session,
		RequestStateCache cache)
	{
		FullRunSimulationStateSnapshot snapshot;
		using (FullRunSimulationDiagnostics.Measure("request.get_state.runtime_ms"))
		{
			snapshot = GetSnapshot(service, cache);
		}

		using (FullRunSimulationDiagnostics.Measure("request.binary_encode_ms"))
		{
			return BinaryProtocol.BuildStateResponse(BinaryOpcode.State, session, snapshot);
		}
	}

	// State types that require agent decision (return to Python)
	private static readonly HashSet<string> DecisionStateTypes = new(StringComparer.OrdinalIgnoreCase)
	{
		"map", "combat_rewards", "card_reward", "card_select", "relic_select",
		"shop", "rest_site", "campfire", "event", "treasure",
		"monster", "elite", "boss", "combat", "hand_select",
		"game_over", "menu",
	};

	private static bool IsDecisionState(FullRunSimulationStateSnapshot snapshot)
	{
		if (snapshot.IsTerminal || snapshot.StateType == "game_over")
			return true;
		if (snapshot.LegalActions.Count == 0)
			return false; // pending/wait state — not a decision
		return DecisionStateTypes.Contains(snapshot.StateType);
	}

	private static async Task<byte[]> ProcessBinaryStepAsync(
		FullRunTrainingEnvService service,
		BinarySessionState session,
		byte[] requestBytes,
		RequestStateCache cache)
	{
		FullRunSimulationActionRequest action = BinaryProtocol.ParseActionRequest(requestBytes);
		FullRunSimulationStepResult result;
		using (FullRunSimulationDiagnostics.Measure("request.step.runtime_ms"))
		{
			result = await service.StepAsync(action);
		}

		FullRunSimulationStateSnapshot snapshot = result.State ?? GetSnapshot(service, cache);

		// Auto-advance through non-decision states (combat_rewards, card_select, etc.)
		// This reduces Python↔C# round-trips by ~96% (only return on real decisions).
		int autoAdvanceCount = 0;
		const int maxAutoAdvance = 50; // safety cap
		while (!IsDecisionState(snapshot) && autoAdvanceCount < maxAutoAdvance)
		{
			autoAdvanceCount++;
			FullRunSimulationDiagnostics.Increment("step.auto_advance");

			if (snapshot.LegalActions.Count == 0)
			{
				// No legal actions — send wait
				var waitResult = await service.StepAsync(new FullRunSimulationActionRequest { Action = "wait" });
				snapshot = waitResult.State ?? GetSnapshot(service, cache);
			}
			else
			{
				// Auto-pick first legal action (convert LegalAction → ActionRequest)
				var la = snapshot.LegalActions[0];
				var autoAction = new FullRunSimulationActionRequest
				{
					Action = la.Action ?? "",
					Index = la.Index,
					Col = la.Col,
					Row = la.Row,
					Slot = la.Slot,
					TargetId = la.TargetId,
					Target = la.Target,
					CardIndex = la.CardIndex,
					Value = la.Label,
				};
				var autoResult = await service.StepAsync(autoAction);
				snapshot = autoResult.State ?? GetSnapshot(service, cache);
			}
		}

		using (FullRunSimulationDiagnostics.Measure("request.binary_encode_ms"))
		{
			return BinaryProtocol.BuildStepResponse(session, result, snapshot);
		}
	}

	private static async Task<byte[]> ProcessBinaryBatchStepAsync(
		FullRunTrainingEnvService service,
		BinarySessionState session,
		byte[] requestBytes,
		RequestStateCache cache)
	{
		List<FullRunSimulationActionRequest> actions = BinaryProtocol.ParseBatchActionRequest(requestBytes);
		if (actions.Count == 0)
		{
			throw new InvalidOperationException("batch_step requires at least one action.");
		}

		FullRunSimulationBatchStepResult result;
		using (FullRunSimulationDiagnostics.Measure("request.batch_step.runtime_ms"))
		{
			result = await service.BatchStepAsync(actions);
		}

		FullRunSimulationStateSnapshot snapshot = result.State ?? GetSnapshot(service, cache);
		using (FullRunSimulationDiagnostics.Measure("request.binary_encode_ms"))
		{
			return BinaryProtocol.BuildBatchStepResponse(session, result, snapshot);
		}
	}

	private static byte[] ProcessBinarySaveState(FullRunTrainingEnvService service)
	{
		string stateId = service.SaveState();
		return BinaryProtocol.BuildSaveStateResponse(stateId, service.StateCacheCount);
	}

	private static byte[] ProcessBinaryExportState(FullRunTrainingEnvService service, byte[] requestBytes)
	{
		(string path, string? stateId) = BinaryProtocol.ParseExportStateRequest(requestBytes);
		string writtenPath = service.ExportStateToFile(path, stateId);
		return BinaryProtocol.BuildExportStateResponse(writtenPath, service.StateCacheCount);
	}

	private static async Task<byte[]> ProcessBinaryLoadStateAsync(
		FullRunTrainingEnvService service,
		BinarySessionState session,
		byte[] requestBytes,
		RequestStateCache cache)
	{
		string stateId = BinaryProtocol.ParseStateIdRequest(BinaryOpcode.LoadState, requestBytes);
		FullRunSimulationStateSnapshot snapshot;
		using (FullRunSimulationDiagnostics.Measure("request.load_state.runtime_ms"))
		{
			snapshot = await service.LoadState(stateId);
		}

		cache.Snapshot = snapshot;
		using (FullRunSimulationDiagnostics.Measure("request.binary_encode_ms"))
		{
			return BinaryProtocol.BuildStateResponse(BinaryOpcode.LoadState, session, snapshot);
		}
	}

	private static async Task<byte[]> ProcessBinaryImportStateAsync(
		FullRunTrainingEnvService service,
		BinarySessionState session,
		byte[] requestBytes,
		RequestStateCache cache)
	{
		string path = BinaryProtocol.ParsePathRequest(BinaryOpcode.ImportState, requestBytes);
		FullRunSimulationStateSnapshot snapshot;
		using (FullRunSimulationDiagnostics.Measure("request.import_state.runtime_ms"))
		{
			snapshot = await service.LoadStateFromFile(path);
		}

		cache.Snapshot = snapshot;
		using (FullRunSimulationDiagnostics.Measure("request.binary_encode_ms"))
		{
			return BinaryProtocol.BuildStateResponse(BinaryOpcode.ImportState, session, snapshot);
		}
	}

	private static byte[] ProcessBinaryDeleteState(FullRunTrainingEnvService service, byte[] requestBytes)
	{
		bool clearAll = BinaryProtocol.ParseDeleteClearAll(requestBytes, out string? stateId);
		bool deleted;
		if (clearAll)
		{
			service.ClearStateCache();
			deleted = true;
		}
		else
		{
			deleted = service.DeleteState(stateId!);
		}

		return BinaryProtocol.BuildDeleteStateResponse(deleted, service.StateCacheCount);
	}

	/// <summary>
	/// Run entire combat locally using ORT CPU actor. No per-step Python round-trips.
	/// Returns final state + combat step count + action history for PPO recomputation.
	/// </summary>
	private static async Task<byte[]> ProcessBinaryRunCombatLocalAsync(
		FullRunTrainingEnvService service,
		BinarySessionState session,
		byte[] requestBytes,
		RequestStateCache cache)
	{
		if (_ortPolicy == null)
		{
			return BinaryProtocol.BuildErrorResponse(
				BinaryOpcode.RunCombatLocal, BinaryStatus.SimulatorError,
				"ort_not_loaded", "ORT model not loaded. Send LoadOrtModel first.");
		}

		try
		{
			// Parse max steps from request (opcode + uint16 max_steps)
			int maxCombatSteps = 600;
			if (requestBytes.Length >= 3)
			{
				using var reqReader = new BinaryReader(new MemoryStream(requestBytes));
				reqReader.ReadByte(); // opcode
				maxCombatSteps = reqReader.ReadUInt16();
			}

			var snapshot = GetSnapshot(service, cache);
			bool isCombat = snapshot.StateType is "monster" or "elite" or "boss" or "combat";

			if (!isCombat)
			{
				// Not in combat — just return current state
				var fakeResult = new FullRunSimulationStepResult { Accepted = true, State = snapshot };
				return BinaryProtocol.BuildStepResponse(session, fakeResult, snapshot);
			}

			// Run combat loop internally (with 10s timeout to prevent straggler)
			int combatSteps = 0;
			int waitSteps = 0;
			var stopwatch = System.Diagnostics.Stopwatch.StartNew();
			const long COMBAT_TIMEOUT_MS = 10_000;

			// Per-step timing accumulators (ticks)
			long totalGetSnapshotTicks = 0;
			long totalOrtTicks = 0;
			long totalStepAsyncTicks = 0;
			long totalWaitAsyncTicks = 0;
			long maxStepAsyncTicks = 0;
			long maxWaitAsyncTicks = 0;

			for (int step = 0; step < maxCombatSteps; step++)
			{
				long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
				snapshot = GetSnapshot(service, cache);
				long t1 = System.Diagnostics.Stopwatch.GetTimestamp();
				totalGetSnapshotTicks += (t1 - t0);

				// Check timeout
				if (stopwatch.ElapsedMilliseconds > COMBAT_TIMEOUT_MS)
				{
					FullRunSimulationDiagnostics.Increment("request.run_combat_local.timeout");
					break;
				}

				// Check if combat ended
				if (snapshot.IsTerminal || snapshot.StateType == "game_over")
					break;

				bool stillCombat = snapshot.StateType is "monster" or "elite" or "boss" or "combat";
				if (!stillCombat)
					break;

				if (snapshot.LegalActions.Count == 0)
				{
					// No legal actions — auto-advance (wait)
					long tw0 = System.Diagnostics.Stopwatch.GetTimestamp();
					var waitResult = await service.StepAsync(
						new FullRunSimulationActionRequest { Action = "wait" });
					long tw1 = System.Diagnostics.Stopwatch.GetTimestamp();
					long waitTicks = tw1 - tw0;
					totalWaitAsyncTicks += waitTicks;
					if (waitTicks > maxWaitAsyncTicks) maxWaitAsyncTicks = waitTicks;
					waitSteps++;
					if (waitResult.State != null)
						cache.Snapshot = null; cache.ApiState = null;
					continue;
				}

				// ORT inference + action selection
				long to0 = System.Diagnostics.Stopwatch.GetTimestamp();
				var (actionIdx, logits) = _ortPolicy.SelectAction(snapshot, session, _ortRng);
				long to1 = System.Diagnostics.Stopwatch.GetTimestamp();
				totalOrtTicks += (to1 - to0);

				// Execute action
				var action = snapshot.LegalActions[actionIdx];
				var stepRequest = new FullRunSimulationActionRequest
				{
					Action = action.Action,
					Index = action.Index,
					CardIndex = action.CardIndex,
					Slot = action.Slot,
					Col = action.Col,
					Row = action.Row,
				};
				if (action.TargetId.HasValue)
					stepRequest.TargetId = action.TargetId;

				long ts0 = System.Diagnostics.Stopwatch.GetTimestamp();
				var result = await service.StepAsync(stepRequest);
				long ts1 = System.Diagnostics.Stopwatch.GetTimestamp();
				long stepTicks = ts1 - ts0;
				totalStepAsyncTicks += stepTicks;
				if (stepTicks > maxStepAsyncTicks) maxStepAsyncTicks = stepTicks;
				cache.Snapshot = null; cache.ApiState = null;
				combatSteps++;

				// Auto-advance non-decision states within combat
				const int maxAutoAdvance = 30;
				for (int i = 0; i < maxAutoAdvance; i++)
				{
					var advState = result.State ?? GetSnapshot(service, cache);
					if (advState.IsTerminal || advState.StateType == "game_over")
						break;
					if (advState.LegalActions.Count > 0)
						break;
					result = await service.StepAsync(
						new FullRunSimulationActionRequest { Action = "wait" });
					cache.Snapshot = null; cache.ApiState = null;
				}
			}

			stopwatch.Stop();
			FullRunSimulationDiagnostics.Increment("request.run_combat_local.calls");
			FullRunSimulationDiagnostics.Increment("request.run_combat_local.total_steps", combatSteps);

			// Timing breakdown (convert ticks to ms)
			double tickFreq = System.Diagnostics.Stopwatch.Frequency / 1000.0;
			float getSnapshotMs = (float)(totalGetSnapshotTicks / tickFreq);
			float ortMs = (float)(totalOrtTicks / tickFreq);
			float stepAsyncMs = (float)(totalStepAsyncTicks / tickFreq);
			float waitAsyncMs = (float)(totalWaitAsyncTicks / tickFreq);
			float maxStepMs = (float)(maxStepAsyncTicks / tickFreq);
			float maxWaitMs = (float)(maxWaitAsyncTicks / tickFreq);

			// Log long-tail diagnostics if max step > 100ms
			if (maxStepMs > 100 || maxWaitMs > 100)
			{
				int minWorker, minIO, maxWorker, maxIO;
				System.Threading.ThreadPool.GetMinThreads(out minWorker, out minIO);
				System.Threading.ThreadPool.GetMaxThreads(out maxWorker, out maxIO);
				int avail, availIO;
				System.Threading.ThreadPool.GetAvailableThreads(out avail, out availIO);
				Console.Error.WriteLine(
					$"[ORT LONGTAIL] steps={combatSteps} waits={waitSteps} " +
					$"maxStep={maxStepMs:F1}ms maxWait={maxWaitMs:F1}ms " +
					$"totalStep={stepAsyncMs:F0}ms totalWait={waitAsyncMs:F0}ms " +
					$"ThreadPool min={minWorker} max={maxWorker} avail={avail}");
			}

			// Get final state and return
			var finalSnapshot = GetSnapshot(service, cache);

			// Build response with timing breakdown
			using var ms = new MemoryStream();
			using var writer = new BinaryWriter(ms);
			writer.Write((byte)BinaryStatus.Ok);
			writer.Write((byte)BinaryOpcode.RunCombatLocal);
			session.WritePendingSymbolUpdates(writer);
			writer.Write((ushort)combatSteps);
			writer.Write((float)stopwatch.Elapsed.TotalMilliseconds);
			// Timing breakdown (6 floats)
			writer.Write(getSnapshotMs);
			writer.Write(ortMs);
			writer.Write(stepAsyncMs);
			writer.Write(waitAsyncMs);
			writer.Write(maxStepMs);
			writer.Write(maxWaitMs);
			// Write final state using standard state payload
			byte[] statePayload = BinaryProtocol.BuildStatePayload(session, finalSnapshot);
			writer.Write(statePayload);
			return ms.ToArray();
		}
		catch (Exception ex)
		{
			Console.Error.WriteLine($"[ORT] RunCombatLocal error: {ex.Message}");
			return BinaryProtocol.BuildErrorResponse(
				BinaryOpcode.RunCombatLocal, BinaryStatus.SimulatorError,
				"ort_combat_error", ex.Message);
		}
	}

	/// <summary>
	/// Build Mode: instantly win the current combat by killing all enemies.
	/// Only works when in a combat state. Returns the post-combat state
	/// (typically combat_rewards or map).
	/// </summary>
	private static async Task<byte[]> ProcessBinarySkipCombatAsync(
		FullRunTrainingEnvService service,
		BinarySessionState session,
		RequestStateCache cache)
	{
		try
		{
			var snapshot = GetSnapshot(service, cache);
			bool isCombat = snapshot.StateType is "monster" or "elite" or "boss" or "combat";

			if (!isCombat)
			{
				// Not in combat — just return current state
				var noopResult = new FullRunSimulationStepResult { Accepted = true, State = snapshot };
				return BinaryProtocol.BuildStepResponse(session, noopResult, snapshot);
			}

			// Enter pure-combat-simulator mode to skip all UI/presentation code
			// (replay writer, music, achievements, save, map screen, etc.)
			// This makes it safe to call EndCombatInternal from both headless sim
			// AND visible Godot client — same logic path.
			service.ResetCombatFollowupStateForExternalCombatResolution();
			using (CombatSimulationRuntime.EnterPureCombatSimulator())
			{
				await CombatManager.Instance.EndCombatInternal();
			}
			cache.Snapshot = null;
			cache.ApiState = null;

			// Auto-advance through any pending transitions (rewards screen, etc.)
			for (int i = 0; i < 30; i++)
			{
				var advSnapshot = GetSnapshot(service, cache);
				if (advSnapshot.IsTerminal || advSnapshot.StateType == "game_over")
					break;
				if (advSnapshot.LegalActions.Count > 0)
					break;
				await service.StepAsync(
					new FullRunSimulationActionRequest { Action = "wait" });
				cache.Snapshot = null;
				cache.ApiState = null;
			}

			FullRunSimulationDiagnostics.Increment("request.skip_combat.calls");

			// Get final state and return using standard step response format
			var finalSnapshot = GetSnapshot(service, cache);
			var finalResult = new FullRunSimulationStepResult { Accepted = true, State = finalSnapshot };
			return BinaryProtocol.BuildStepResponse(session, finalResult, finalSnapshot);
		}
		catch (Exception ex)
		{
			Console.Error.WriteLine($"[SkipCombat] error: {ex.Message}");
			return BinaryProtocol.BuildErrorResponse(
				BinaryOpcode.SkipCombat, BinaryStatus.SimulatorError,
				"skip_combat_error", ex.Message);
		}
	}

	private static async Task<byte[]> ProcessBinarySearchCombatMctsAsync(
		FullRunTrainingEnvService service,
		byte[] requestBytes,
		RequestStateCache cache)
	{
		if (_ortPolicy == null)
		{
			return BinaryProtocol.BuildErrorResponse(
				BinaryOpcode.SearchCombatMcts,
				BinaryStatus.SimulatorError,
				"ort_not_loaded",
				"ORT model not loaded. Send LoadOrtModel first.");
		}

		try
		{
			BinarySearchCombatMctsRequest request = BinaryProtocol.ParseSearchCombatMctsRequest(requestBytes);
			FullRunSimulationStateSnapshot snapshot = GetSnapshot(service, cache);
			bool isCombat = snapshot.StateType is "monster" or "elite" or "boss" or "combat" or "hand_select" or "card_select" or "combat_pending" or "combat_start_pending";
			if (!isCombat)
			{
				return BinaryProtocol.BuildErrorResponse(
					BinaryOpcode.SearchCombatMcts,
					BinaryStatus.ProtocolError,
					"not_in_combat",
					$"search_combat_mcts requires a combat state, got '{snapshot.StateType}'.");
			}

			CombatMctsSearchEngine engine = new CombatMctsSearchEngine(service, _ortPolicy, _mctsRng);
			CombatMctsResult result = await engine.SearchAsync(
				snapshot,
				new CombatMctsConfig
				{
					NumSimulations = Math.Max(1, request.NumSimulations),
					CPuct = request.CPuct,
					DirichletAlpha = request.DirichletAlpha,
					DirichletFraction = request.DirichletFraction,
					MaxStepBudget = Math.Max(1, request.MaxStepBudget),
				FinalActionMode = request.FinalActionMode,
				FinalActionTopK = Math.Max(1, request.FinalActionTopK),
				FinalActionQWeight = request.FinalActionQWeight,
				UseContinuationValue = request.UseContinuationValue,
				EnableDebugTrace = request.EnableDebugTrace,
			});
			cache.Snapshot = service.GetState();
			cache.ApiState = null;
			return BinaryProtocol.BuildSearchCombatMctsResponse(result);
		}
		catch (Exception ex)
		{
			Console.Error.WriteLine($"[MCTS] SearchCombatMcts error: {ex}");
			return BinaryProtocol.BuildErrorResponse(
				BinaryOpcode.SearchCombatMcts,
				BinaryStatus.SimulatorError,
				"combat_mcts_error",
				ex.Message);
		}
	}

	private static byte[] ProcessBinaryResetPerfStats()
	{
		FullRunSimulationDiagnostics.Reset();
		return BinaryProtocol.BuildResetPerfStatsResponse();
	}

	// --- Local ORT actor policy ---
	private static OrtActorPolicy? _ortPolicy;
	private static Random _ortRng = new Random(42);
	private static Random _mctsRng = new Random(1234);

	private static byte[] ProcessBinaryLoadOrtModel(byte[] requestBytes)
	{
		try
		{
			// Request: opcode(1) + path_length(2) + path_bytes
			using var reader = new BinaryReader(new MemoryStream(requestBytes));
			reader.ReadByte(); // skip opcode
			int pathLen = reader.ReadUInt16();
			string onnxPath = System.Text.Encoding.UTF8.GetString(reader.ReadBytes(pathLen));

			_ortPolicy?.Dispose();
			// Look for vocab_mapping.json next to the ONNX file
			string? vocabPath = Path.Combine(Path.GetDirectoryName(onnxPath) ?? "", "vocab_mapping.json");
			if (!File.Exists(vocabPath)) vocabPath = null;
			_ortPolicy = new OrtActorPolicy(onnxPath, argmax: false, vocabPath: vocabPath);
			Console.Error.WriteLine(
				$"[ORT] Loaded model from {onnxPath} (vocab={vocabPath != null}, provider={_ortPolicy.ExecutionProviderName}, requested={_ortPolicy.RequestedDevice}, fallback={_ortPolicy.FellBackToCpu})");

			// Use standard response format: status + opcode + payload
			using var ms = new MemoryStream();
			using var writer = new BinaryWriter(ms);
			writer.Write((byte)BinaryStatus.Ok);
			writer.Write((byte)BinaryOpcode.LoadOrtModel);
			writer.Write((ushort)0); // zero symbol updates
			writer.Write((byte)1); // loaded = true
			writer.Write((byte)(_ortPolicy.Metadata.HasValueOutput ? 1 : 0));
			writer.Write((byte)(_ortPolicy.Metadata.HasDeckInputs ? 1 : 0));
			writer.Write((byte)(_ortPolicy.Metadata.HasContinuationOutput ? 1 : 0));
			writer.Write((byte)(_ortPolicy.Metadata.HasExtraScalarsInput ? 1 : 0));
			BinaryProtocol.WriteString(writer, _ortPolicy.ExecutionProviderName);
			BinaryProtocol.WriteString(writer, _ortPolicy.RequestedDevice);
			writer.Write((byte)(_ortPolicy.FellBackToCpu ? 1 : 0));
			return ms.ToArray();
		}
		catch (Exception ex)
		{
			Console.Error.WriteLine($"[ORT] Load failed: {ex.Message}");
			return BinaryProtocol.BuildErrorResponse(
				BinaryOpcode.LoadOrtModel, BinaryStatus.SimulatorError,
				"ort_load_error", ex.Message);
		}
	}

	private static async Task<byte[]> ProcessBinaryStepLocalPolicyAsync(
		FullRunTrainingEnvService service,
		BinarySessionState session,
		RequestStateCache cache)
	{
		if (_ortPolicy == null)
		{
			return BinaryProtocol.BuildErrorResponse(
				BinaryOpcode.StepLocalPolicy, BinaryStatus.SimulatorError,
				"ort_not_loaded", "ORT model not loaded. Send LoadOrtModel first.");
		}

		try
		{
			var snapshot = GetSnapshot(service, cache);
			bool isCombat = snapshot.StateType is "monster" or "elite" or "boss" or "combat";

			if (!isCombat || snapshot.LegalActions.Count == 0 || snapshot.IsTerminal)
			{
				// Not a combat decision — return step response with accepted=true, current state
				// Python handles non-combat screens normally
				var fakeResult = new FullRunSimulationStepResult { Accepted = true, State = snapshot };
				return BinaryProtocol.BuildStepResponse(session, fakeResult, snapshot);
			}

			// Local ORT inference + action selection
			var (actionIdx, logits) = _ortPolicy.SelectAction(snapshot, session, _ortRng);

			// Execute the selected action
			var action = snapshot.LegalActions[actionIdx];
			var stepRequest = new FullRunSimulationActionRequest
			{
				Action = action.Action,
				Index = action.Index,
				CardIndex = action.CardIndex,
				Slot = action.Slot,
				Col = action.Col,
				Row = action.Row,
			};
			if (action.TargetId.HasValue)
				stepRequest.TargetId = action.TargetId;

			FullRunSimulationStepResult result;
			using (FullRunSimulationDiagnostics.Measure("request.step_local_policy.runtime_ms"))
			{
				result = await service.StepAsync(stepRequest);
			}

			// Auto-advance non-decision states
			const int maxAutoAdvance = 30;
			for (int i = 0; i < maxAutoAdvance && result.Accepted && result.State != null; i++)
			{
				if (result.State.IsTerminal || result.State.StateType == "game_over")
					break;
				if (result.State.LegalActions.Count > 0)
					break;
				result = await service.StepAsync(new FullRunSimulationActionRequest { Action = "wait" });
			}

			var nextSnapshot = result.State ?? GetSnapshot(service, cache);
			FullRunSimulationDiagnostics.Increment("request.step_local_policy.calls");

			// Return as standard step response (Python decodes normally)
			return BinaryProtocol.BuildStepResponse(session, result, nextSnapshot);
		}
		catch (Exception ex)
		{
			Console.Error.WriteLine($"[ORT] StepLocalPolicy error: {ex.Message}");
			return BinaryProtocol.BuildErrorResponse(
				BinaryOpcode.StepLocalPolicy, BinaryStatus.SimulatorError,
				"ort_step_error", ex.Message);
		}
	}

	private static BinaryOpcode SafeParseOpcode(byte[] requestBytes)
	{
		try
		{
			return BinaryProtocol.ParseOpcode(requestBytes);
		}
		catch
		{
			return BinaryOpcode.State;
		}
	}

	private static BinaryStatus GetBinaryErrorStatus(Exception exception)
	{
		return exception switch
		{
			InvalidOperationException => BinaryStatus.ProtocolError,
			JsonException => BinaryStatus.ProtocolError,
			EndOfStreamException => BinaryStatus.ProtocolError,
			TimeoutException => BinaryStatus.ProtocolError,
			_ => BinaryStatus.SimulatorError
		};
	}

	private static async Task<string> ProcessPipeRequestAsync(FullRunTrainingEnvService service, string requestJson)
	{
		long requestStart = Stopwatch.GetTimestamp();
		RequestStateCache cache = new RequestStateCache();
		using JsonDocument doc = JsonDocument.Parse(requestJson);
		JsonElement root = doc.RootElement;
		string method = root.TryGetProperty("method", out JsonElement methodElement)
			? methodElement.GetString() ?? string.Empty
			: string.Empty;
		JsonElement paramsElement = root.TryGetProperty("params", out JsonElement paramsValue)
			? paramsValue
			: default;

		if (string.IsNullOrWhiteSpace(method))
		{
			return SerializePipeError("invalid_request", "Request must include a method.");
		}

		object response = method switch
		{
			"state" or "get_state" => BuildApiState(service, cache),
			"legal_actions" => new Dictionary<string, object?>
			{
				["legal_actions"] = BuildApiState(service, cache).legal_actions
			},
			"reset" => BuildApiState(await ResetAsync(service, paramsElement), cache),
			"combat_state" => BuildCombatApiState(),
			"combat_reset" => BuildCombatApiState(await CombatResetAsync(paramsElement)),
			"combat_step" => await CombatStepAsync(paramsElement),
			"combat_catalog" => BuildCombatCatalog(),
			// game_catalog: 完整静态数据（cards/relics/monsters/potions/encounters/powers）
			// Python 侧 GAME_CATALOG.attach_sim(client) 调一次，所有特征工程查这里
			// 规范见 STS2AI/docs/design/SCHEMA_CONVENTION.md
			"game_catalog" => BuildGameCatalog(),
			"step" => await StepAsync(service, paramsElement, cache),
			"batch_step" => await BatchStepAsync(service, paramsElement, cache),
			"save_state" => new Dictionary<string, object?>
			{
				["state_id"] = service.SaveState(),
				["cache_size"] = service.StateCacheCount
			},
			"export_state" => ExportState(service, paramsElement),
			"load_state" => BuildApiState(await LoadStateAsync(service, paramsElement), cache),
			"import_state" => BuildApiState(await ImportStateAsync(service, paramsElement), cache),
			"delete_state" => DeleteState(service, paramsElement),
			"clear_state_cache" => ClearStateCache(service),
			"state_cache_count" => new Dictionary<string, object?> { ["count"] = service.StateCacheCount },
			"perf_stats" => FullRunSimulationDiagnostics.Snapshot(),
			"reset_perf_stats" => ResetPerfStats(),
			_ => BuildErrorPayload("unknown_method", $"Unknown method: {method}")
		};

		try
		{
			return JsonSerializer.Serialize(response, JsonOptions);
		}
		catch (Exception ex)
		{
			FullRunSimulationTrace.Write($"headless_pipe.serialize_exception method={method} exception={ex}");
			throw;
		}
		finally
		{
			double elapsedMs = (Stopwatch.GetTimestamp() - requestStart) * 1000.0 / Stopwatch.Frequency;
			FullRunSimulationDiagnostics.RecordTiming($"request.{method}.total_ms", elapsedMs);
			FullRunSimulationDiagnostics.Increment($"request.{method}.count");
		}
	}

	private static async Task<FullRunSimulationStateSnapshot> ResetAsync(FullRunTrainingEnvService service, JsonElement paramsElement)
	{
		FullRunSimulationResetRequest request = new FullRunSimulationResetRequest();
		if (paramsElement.ValueKind == JsonValueKind.Object)
		{
			if (paramsElement.TryGetProperty("character_id", out JsonElement characterId) && characterId.ValueKind == JsonValueKind.String)
			{
				request.CharacterId = characterId.GetString();
			}

			if (paramsElement.TryGetProperty("character", out JsonElement character) && character.ValueKind == JsonValueKind.String)
			{
				request.Character = character.GetString();
			}

			if (paramsElement.TryGetProperty("seed", out JsonElement seed) && seed.ValueKind == JsonValueKind.String)
			{
				request.Seed = seed.GetString();
			}

			if (paramsElement.TryGetProperty("ascension_level", out JsonElement ascensionLevel) && ascensionLevel.ValueKind == JsonValueKind.Number)
			{
				request.AscensionLevel = ascensionLevel.GetInt32();
			}
			else if (paramsElement.TryGetProperty("ascension", out JsonElement ascension) && ascension.ValueKind == JsonValueKind.Number)
			{
				request.Ascension = ascension.GetInt32();
			}

			if (paramsElement.TryGetProperty("build", out JsonElement build))
			{
				request.Build = SimulationBuildSupport.ParseJsonElement(build);
			}
		}

		using IDisposable _ = FullRunSimulationDiagnostics.Measure("request.reset.runtime_ms");
		return await service.ResetAsync(request);
	}

	private static async Task<CombatTrainingStateSnapshot> CombatResetAsync(JsonElement paramsElement)
	{
		CombatTrainingResetRequest request = new CombatTrainingResetRequest();
		if (paramsElement.ValueKind == JsonValueKind.Object)
		{
			if (paramsElement.TryGetProperty("character_id", out JsonElement characterId) && characterId.ValueKind == JsonValueKind.String)
			{
				request.CharacterId = characterId.GetString();
			}
			else if (paramsElement.TryGetProperty("character", out JsonElement character) && character.ValueKind == JsonValueKind.String)
			{
				request.CharacterId = character.GetString();
			}

			if (paramsElement.TryGetProperty("encounter_id", out JsonElement encounterId) && encounterId.ValueKind == JsonValueKind.String)
			{
				request.EncounterId = encounterId.GetString();
			}
			else if (paramsElement.TryGetProperty("encounter", out JsonElement encounter) && encounter.ValueKind == JsonValueKind.String)
			{
				request.EncounterId = encounter.GetString();
			}

			if (paramsElement.TryGetProperty("seed", out JsonElement seed) && seed.ValueKind == JsonValueKind.String)
			{
				request.Seed = seed.GetString();
			}

			if (paramsElement.TryGetProperty("ascension_level", out JsonElement ascensionLevel) && ascensionLevel.ValueKind == JsonValueKind.Number)
			{
				request.AscensionLevel = ascensionLevel.GetInt32();
			}
			else if (paramsElement.TryGetProperty("ascension", out JsonElement ascension) && ascension.ValueKind == JsonValueKind.Number)
			{
				request.AscensionLevel = ascension.GetInt32();
			}

			if (paramsElement.TryGetProperty("build", out JsonElement build))
			{
				request.Build = SimulationBuildSupport.ParseJsonElement(build);
			}
		}

		using IDisposable _ = FullRunSimulationDiagnostics.Measure("request.combat_reset.runtime_ms");
		return await CombatTrainingEnvService.Instance.ResetAsync(request);
	}

	private static async Task<FullRunSimulationStateSnapshot> LoadStateAsync(FullRunTrainingEnvService service, JsonElement paramsElement)
	{
		using IDisposable _ = FullRunSimulationDiagnostics.Measure("request.load_state.runtime_ms");
		return await service.LoadState(GetRequiredString(paramsElement, "state_id"));
	}

	private static object ExportState(FullRunTrainingEnvService service, JsonElement paramsElement)
	{
		using IDisposable _ = FullRunSimulationDiagnostics.Measure("request.export_state.runtime_ms");
		string path = GetRequiredString(paramsElement, "path");
		string? stateId = null;
		if (paramsElement.ValueKind == JsonValueKind.Object &&
			paramsElement.TryGetProperty("state_id", out JsonElement stateIdElement) &&
			stateIdElement.ValueKind == JsonValueKind.String)
		{
			stateId = stateIdElement.GetString();
		}
		return new Dictionary<string, object?>
		{
			["path"] = service.ExportStateToFile(path, stateId),
			["state_id"] = stateId,
			["cache_size"] = service.StateCacheCount
		};
	}

	private static async Task<FullRunSimulationStateSnapshot> ImportStateAsync(FullRunTrainingEnvService service, JsonElement paramsElement)
	{
		using IDisposable _ = FullRunSimulationDiagnostics.Measure("request.import_state.runtime_ms");
		return await service.LoadStateFromFile(GetRequiredString(paramsElement, "path"));
	}

	private static async Task<Dictionary<string, object?>> StepAsync(FullRunTrainingEnvService service, JsonElement paramsElement, RequestStateCache cache)
	{
		FullRunSimulationActionRequest action = ParseActionRequest(paramsElement);
		try
		{
			FullRunSimulationTrace.Write(
				$"headless_pipe.step.begin action={action.Action ?? action.Type ?? "null"} index={action.Index} col={action.Col} row={action.Row} target_id={action.TargetId}");
			FullRunSimulationStepResult result;
			using (FullRunSimulationDiagnostics.Measure("request.step.runtime_ms"))
			{
				result = await service.StepAsync(action);
			}

			// Advance until the agent needs to make a decision.
			// Eliminates Python round-trips for combat_pending / empty legal actions.
			const int maxAutoAdvance = 30;
			for (int autoIter = 0; autoIter < maxAutoAdvance && result.Accepted && result.State != null; autoIter++)
			{
				FullRunSimulationStateSnapshot advState = result.State;
				if (advState.IsTerminal || advState.StateType == "game_over")
					break;
				if (advState.LegalActions.Count > 0)
					break;
				// No legal actions — auto-advance with "wait"
				using (FullRunSimulationDiagnostics.Measure("request.step.auto_advance_ms"))
				{
					result = await service.StepAsync(new FullRunSimulationActionRequest { Action = "wait" });
				}
				FullRunSimulationDiagnostics.Increment("request.step.auto_advance_count");
			}

			FullRunApiState state = BuildApiState(result.State ?? GetSnapshot(service, cache), cache);
			FullRunSimulationTrace.Write(
				$"headless_pipe.step.done accepted={result.Accepted} error={result.Error ?? "null"} state_type={state.state_type} floor={state.run.floor} terminal={state.terminal}");
			return new Dictionary<string, object?>
			{
				["accepted"] = result.Accepted,
				["error"] = result.Error,
				["state"] = state,
				["reward"] = ComputeTerminalReward(state.run_outcome, state.terminal),
				["done"] = state.terminal,
				["info"] = new Dictionary<string, object?>
				{
					["state_type"] = state.state_type,
					["run_outcome"] = state.run_outcome
				}
			};
		}
		catch (Exception ex)
		{
			FullRunSimulationTrace.Write(
				$"headless_pipe.step.exception action={action.Action ?? action.Type ?? "null"} index={action.Index} col={action.Col} row={action.Row} exception={ex}");
			throw;
		}
	}

	private static async Task<Dictionary<string, object?>> CombatStepAsync(JsonElement paramsElement)
	{
		CombatTrainingActionRequest action = ParseCombatActionRequest(paramsElement);
		CombatTrainingStepResult result;
		using (FullRunSimulationDiagnostics.Measure("request.combat_step.runtime_ms"))
		{
			result = await CombatTrainingEnvService.Instance.StepAsync(action);
		}

		return new Dictionary<string, object?>
		{
			["accepted"] = result.Accepted,
			["error"] = result.Error,
			["state"] = BuildCombatApiState(result.State ?? CombatTrainingEnvService.Instance.GetState()),
		};
	}

	private static async Task<Dictionary<string, object?>> BatchStepAsync(FullRunTrainingEnvService service, JsonElement paramsElement, RequestStateCache cache)
	{
		if (!paramsElement.TryGetProperty("actions", out JsonElement actionsElement) || actionsElement.ValueKind != JsonValueKind.Array)
		{
			throw new InvalidOperationException("batch_step requires an 'actions' array.");
		}

		List<FullRunSimulationActionRequest> actions = new List<FullRunSimulationActionRequest>();
		foreach (JsonElement actionElement in actionsElement.EnumerateArray())
		{
			actions.Add(ParseActionRequest(actionElement));
		}

		if (actions.Count == 0)
		{
			throw new InvalidOperationException("batch_step requires at least one action.");
		}

		try
		{
			FullRunSimulationTrace.Write($"headless_pipe.batch_step.begin count={actions.Count}");
			FullRunSimulationBatchStepResult result;
			using (FullRunSimulationDiagnostics.Measure("request.batch_step.runtime_ms"))
			{
				result = await service.BatchStepAsync(actions);
			}
			FullRunApiState state = BuildApiState(result.State ?? GetSnapshot(service, cache), cache);
			FullRunSimulationTrace.Write(
				$"headless_pipe.batch_step.done accepted={result.Accepted} steps_executed={result.StepsExecuted} error={result.Error ?? "null"} state_type={state.state_type} floor={state.run.floor}");
			return new Dictionary<string, object?>
			{
				["accepted"] = result.Accepted,
				["error"] = result.Error,
				["steps_executed"] = result.StepsExecuted,
				["state"] = state
			};
		}
		catch (Exception ex)
		{
			FullRunSimulationTrace.Write($"headless_pipe.batch_step.exception count={actions.Count} exception={ex}");
			throw;
		}
	}

	private static CombatTrainingActionRequest ParseCombatActionRequest(JsonElement paramsElement)
	{
		CombatTrainingActionRequest request = new CombatTrainingActionRequest
		{
			Type = ParseCombatActionType(paramsElement)
		};
		if (paramsElement.ValueKind != JsonValueKind.Object)
		{
			return request;
		}
		if (paramsElement.TryGetProperty("hand_index", out JsonElement handIndex) && handIndex.ValueKind == JsonValueKind.Number)
		{
			request.HandIndex = handIndex.GetInt32();
		}
		else if (paramsElement.TryGetProperty("card_index", out JsonElement cardIndex) && cardIndex.ValueKind == JsonValueKind.Number)
		{
			request.HandIndex = cardIndex.GetInt32();
		}
		else if (paramsElement.TryGetProperty("index", out JsonElement index) && index.ValueKind == JsonValueKind.Number)
		{
			request.HandIndex = index.GetInt32();
		}

		if (paramsElement.TryGetProperty("choice_index", out JsonElement choiceIndex) && choiceIndex.ValueKind == JsonValueKind.Number)
		{
			request.ChoiceIndex = choiceIndex.GetInt32();
		}

		if (paramsElement.TryGetProperty("slot", out JsonElement slot) && slot.ValueKind == JsonValueKind.Number)
		{
			request.Slot = slot.GetInt32();
		}

		if (paramsElement.TryGetProperty("target_id", out JsonElement targetId))
		{
			if (targetId.ValueKind == JsonValueKind.Number)
			{
				request.TargetId = targetId.GetUInt32();
			}
			else if (targetId.ValueKind == JsonValueKind.String && uint.TryParse(targetId.GetString(), out uint parsedTargetId))
			{
				request.TargetId = parsedTargetId;
			}
		}

		return request;
	}

	private static CombatTrainingActionType ParseCombatActionType(JsonElement paramsElement)
	{
		string? raw = null;
		if (paramsElement.ValueKind == JsonValueKind.Object)
		{
			if (paramsElement.TryGetProperty("action", out JsonElement action) && action.ValueKind == JsonValueKind.String)
			{
				raw = action.GetString();
			}
			else if (paramsElement.TryGetProperty("type", out JsonElement type) && type.ValueKind == JsonValueKind.String)
			{
				raw = type.GetString();
			}
		}

		return (raw ?? string.Empty).Trim().ToLowerInvariant() switch
		{
			"play_card" => CombatTrainingActionType.PlayCard,
			"end_turn" => CombatTrainingActionType.EndTurn,
			"select_hand_card" => CombatTrainingActionType.SelectHandCard,
			"select_card_option" => CombatTrainingActionType.SelectCardChoice,
			"confirm_selection" => CombatTrainingActionType.ConfirmSelection,
			"cancel_selection" => CombatTrainingActionType.CancelSelection,
			"use_potion" => CombatTrainingActionType.UsePotion,
			_ => throw new InvalidOperationException($"Unsupported combat action type: {raw}")
		};
	}

	private static object BuildCombatCatalog()
	{
		List<Dictionary<string, object?>> encounters = ModelDb.AllEncounters
			.Where(static encounter => encounter.RoomType is RoomType.Monster or RoomType.Elite or RoomType.Boss)
			.OrderBy(static encounter => encounter.RoomType)
			.ThenBy(static encounter => encounter.Id.Entry, StringComparer.Ordinal)
			.Select(static encounter => new Dictionary<string, object?>
			{
				["encounter_id"] = encounter.Id.Entry,
				["room_type"] = encounter.RoomType.ToString().ToLowerInvariant(),
			})
			.ToList();
		return new Dictionary<string, object?>
		{
			["encounters"] = encounters,
		};
	}

	// ==========================================================================
	// game_catalog: 完整游戏静态数据（cards/relics/monsters/potions/encounters/powers）
	// --------------------------------------------------------------------------
	// Python 侧 `GAME_CATALOG.attach_sim(client)` 启动时调一次，后续特征工程全走
	// 缓存结果。避免开发者手写卡名/怪名/power class name（STS1 残留或拼写错）。
	//
	// C# 侧也缓存：ModelDb 是启动后静态数据，不变；重复调用只返回同一对象。
	// 多 sim 进程共启动时各自 build 一次。
	//
	// 规范：STS2AI/docs/design/SCHEMA_CONVENTION.md
	// TODO（另一 AI / Spectator）：把这段逻辑提到 shared lib，Spectator 侧也暴露
	// 相同 endpoint，以便 Spectator 也能用统一特征。
	// ==========================================================================
	private static object? _gameCatalogCache;
	private static readonly object _gameCatalogLock = new();

	private static object BuildGameCatalog()
	{
		// 静态数据，只 build 一次
		if (_gameCatalogCache != null)
		{
			return _gameCatalogCache;
		}
		lock (_gameCatalogLock)
		{
			if (_gameCatalogCache != null)
			{
				return _gameCatalogCache;
			}
			_gameCatalogCache = BuildGameCatalogOnce();
			return _gameCatalogCache;
		}
	}

	private static object BuildGameCatalogOnce()
	{
		// encounters：id / room_type / monster_ids / act_index
		// act_index 来自 ModelDb.Acts 枚举（0 = act1, 1 = act2, ...）
		Dictionary<string, int> encounterAct = new();
		try
		{
			int actIdx = 0;
			foreach (ActModel act in ModelDb.Acts)
			{
				foreach (EncounterModel e in act.AllEncounters)
				{
					if (!encounterAct.ContainsKey(e.Id.Entry))
					{
						encounterAct[e.Id.Entry] = actIdx;
					}
				}
				actIdx++;
			}
		}
		catch { }

		List<Dictionary<string, object?>> encounters = ModelDb.AllEncounters
			.OrderBy(static e => e.RoomType)
			.ThenBy(static e => e.Id.Entry, StringComparer.Ordinal)
			.Select(e => new Dictionary<string, object?>
			{
				["encounter_id"] = e.Id.Entry,
				["room_type"] = e.RoomType.ToString().ToLowerInvariant(),
				["monster_ids"] = BuildEncounterMonsterIds(e),
				["act_index"] = encounterAct.TryGetValue(e.Id.Entry, out int a) ? a : -1,
			})
			.ToList();

		// monsters：id / class_name / initial powers / hp
		List<Dictionary<string, object?>> monsters = ModelDb.Monsters
			.OrderBy(static m => m.Id.Entry, StringComparer.Ordinal)
			.Select(static m => new Dictionary<string, object?>
			{
				["monster_id"] = m.Id.Entry,
				["class_name"] = m.GetType().Name,
				["powers"] = BuildMonsterPowerClassNames(m),
			})
			.ToList();

		// cards：id / type / cost / rarity / target_type / tags / keywords / gains_block
		List<Dictionary<string, object?>> cards = ModelDb.AllCards
			.OrderBy(static c => c.Id.Entry, StringComparer.Ordinal)
			.Select(static c => new Dictionary<string, object?>
			{
				["card_id"] = c.Id.Entry,
				["class_name"] = c.GetType().Name,
				["card_type"] = c.Type.ToString().ToLowerInvariant(),
				["rarity"] = c.Rarity.ToString().ToLowerInvariant(),
				["target_type"] = c.TargetType.ToString().ToLowerInvariant(),
				// BaseEnergyCost 通过 EnergyCost.BaseCost（可能为 null，X-cost）
				["base_cost"] = SafeBaseCost(c),
				["is_x_cost"] = c.EnergyCost?.GetType().Name.Contains("XCost") ?? false,
				["gains_block"] = c.GainsBlock,
				["tags"] = c.Tags.Select(t => t.ToString()).OrderBy(s => s, StringComparer.Ordinal).ToList(),
				["keywords"] = c.CanonicalKeywords.Select(k => k.ToString()).OrderBy(s => s, StringComparer.Ordinal).ToList(),
			})
			.ToList();

		// relics：id / class_name / rarity / tags
		List<Dictionary<string, object?>> relics = ModelDb.AllRelics
			.OrderBy(static r => r.Id.Entry, StringComparer.Ordinal)
			.Select(static r => new Dictionary<string, object?>
			{
				["relic_id"] = r.Id.Entry,
				["class_name"] = r.GetType().Name,
				["rarity"] = SafeRelicRarity(r),
				["tags"] = SafeRelicTags(r),
			})
			.ToList();

		// potions：id / class_name / rarity
		List<Dictionary<string, object?>> potions = ModelDb.AllPotions
			.OrderBy(static p => p.Id.Entry, StringComparer.Ordinal)
			.Select(static p => new Dictionary<string, object?>
			{
				["potion_id"] = p.Id.Entry,
				["class_name"] = p.GetType().Name,
				["rarity"] = SafePotionRarity(p),
			})
			.ToList();

		// powers：class_name + base class chain + 类别（buff/debuff/等，基于 class 名 heuristic）
		// 加 base_classes 让 Python 侧能识别 power 继承类型（如 "xxx → TimedPower → PowerModel" 表示临时 power）
		List<Dictionary<string, object?>> powers = ModelDb.AllPowers
			.OrderBy(static p => p.GetType().Name, StringComparer.Ordinal)
			.Select(static p => new Dictionary<string, object?>
			{
				["class_name"] = p.GetType().Name,
				["base_classes"] = GetBaseClassChain(p.GetType(), "PowerModel"),
				// Heuristic：类名含 "Debuff" / "Buff" 或者根据已知 debuff list
				["is_debuff_hint"] = IsDebuffByName(p.GetType().Name),
			})
			.ToList();

		return new Dictionary<string, object?>
		{
			["encounters"] = encounters,
			["monsters"] = monsters,
			["cards"] = cards,
			["relics"] = relics,
			["potions"] = potions,
			["powers"] = powers,
		};
	}

	private static List<string> BuildEncounterMonsterIds(EncounterModel encounter)
	{
		// EncounterModel.AllPossibleMonsters -> IEnumerable<MonsterModel>
		List<string> ids = new();
		try
		{
			foreach (MonsterModel m in encounter.AllPossibleMonsters)
			{
				if (m != null)
				{
					ids.Add(m.GetType().Name);
				}
			}
		}
		catch { }
		return ids;
	}

	private static List<string> BuildMonsterPowerClassNames(MonsterModel monster)
	{
		// Monster 初始 power 是 runtime 阶段 AddPower 才创建的，ModelDb 没有静态列表。
		// 暂时返回空；initial powers 仍由 source_knowledge.sqlite 提供（build 脚本扫源码）。
		// TODO：扫源码注释 / MoveModel 的 PowerFactory 类型注解可提取，需另一 AI 实现。
		return new List<string>();
	}

	// ---- game_catalog 辅助 ----
	private static int SafeBaseCost(CardModel c)
	{
		try
		{
			// CardEnergyCost.Canonical (int) = base cost (未升级/未 modifier 前)
			return c.EnergyCost?.Canonical ?? 0;
		}
		catch { return 0; }
	}

	private static string SafeRelicRarity(RelicModel r)
	{
		try
		{
			var prop = r.GetType().GetProperty("Rarity");
			var v = prop?.GetValue(r);
			return v?.ToString()?.ToLowerInvariant() ?? "";
		}
		catch { return ""; }
	}

	private static List<string> SafeRelicTags(RelicModel r)
	{
		try
		{
			var prop = r.GetType().GetProperty("Tags");
			var v = prop?.GetValue(r);
			if (v is IEnumerable<object> list)
			{
				return list.Where(x => x != null).Select(x => x!.ToString()!).OrderBy(s => s).ToList();
			}
		}
		catch { }
		return new List<string>();
	}

	private static string SafePotionRarity(PotionModel p)
	{
		try
		{
			var prop = p.GetType().GetProperty("Rarity");
			var v = prop?.GetValue(p);
			return v?.ToString()?.ToLowerInvariant() ?? "";
		}
		catch { return ""; }
	}

	private static List<string> GetBaseClassChain(Type t, string stopAt)
	{
		List<string> chain = new();
		Type? cur = t.BaseType;
		while (cur != null && cur != typeof(object))
		{
			chain.Add(cur.Name);
			if (cur.Name == stopAt) break;
			cur = cur.BaseType;
		}
		return chain;
	}

	private static bool IsDebuffByName(string className)
	{
		// Known STS2 debuff powers by class name suffix/stem
		string[] debuffHints = {
			"WeakPower", "VulnerablePower", "FrailPower", "PoisonPower",
			"ShacklesPower", "StranglePower", "ConfusedPower", "NoDrawPower",
			"EntangledPower", "HexPower", "LockOnPower",
		};
		foreach (var h in debuffHints)
		{
			if (className == h) return true;
		}
		return className.EndsWith("DebuffPower", StringComparison.Ordinal);
	}

	private static object BuildCombatApiState()
	{
		return BuildCombatApiState(CombatTrainingEnvService.Instance.GetState());
	}

	private static object BuildCombatApiState(CombatTrainingStateSnapshot snapshot)
	{
		return new Dictionary<string, object?>
		{
			["trainer_active"] = snapshot.IsTrainerActive,
			["pure_simulator"] = snapshot.IsPureSimulator,
			["choice_adapter_kind"] = snapshot.ChoiceAdapterKind,
			["combat_active"] = snapshot.IsCombatActive,
			["episode_done"] = snapshot.IsEpisodeDone,
			["victory"] = snapshot.Victory,
			["episode_number"] = snapshot.EpisodeNumber,
			["seed"] = snapshot.Seed,
			["character_id"] = snapshot.CharacterId,
			["encounter_id"] = snapshot.EncounterId,
			["ascension_level"] = snapshot.AscensionLevel,
			["round_number"] = snapshot.RoundNumber,
			["current_side"] = snapshot.CurrentSide.ToString().ToLowerInvariant(),
			["is_play_phase"] = snapshot.IsPlayPhase,
			["player_actions_disabled"] = snapshot.PlayerActionsDisabled,
			["is_action_queue_running"] = snapshot.IsActionQueueRunning,
			["is_hand_selection_active"] = snapshot.IsHandSelectionActive,
			["is_card_selection_active"] = snapshot.IsCardSelectionActive,
			["can_end_turn"] = snapshot.CanEndTurn,
			["player"] = snapshot.Player,
			["enemies"] = snapshot.Enemies,
			["hand"] = snapshot.Hand,
			["piles"] = snapshot.Piles,
			["hand_selection"] = snapshot.HandSelection,
			["card_selection"] = snapshot.CardSelection,
		};
	}

	private static object DeleteState(FullRunTrainingEnvService service, JsonElement paramsElement)
	{
		bool clearAll = paramsElement.ValueKind == JsonValueKind.Object
			&& paramsElement.TryGetProperty("clear_all", out JsonElement clearAllElement)
			&& clearAllElement.ValueKind == JsonValueKind.True;

		if (clearAll)
		{
			service.ClearStateCache();
			return new Dictionary<string, object?>
			{
				["deleted"] = true,
				["cache_size"] = 0
			};
		}

		string stateId = GetRequiredString(paramsElement, "state_id");
		bool deleted = service.DeleteState(stateId);
		return new Dictionary<string, object?>
		{
			["deleted"] = deleted,
			["cache_size"] = service.StateCacheCount
		};
	}

	private static object ClearStateCache(FullRunTrainingEnvService service)
	{
		service.ClearStateCache();
		return new Dictionary<string, object?>
		{
			["deleted"] = true,
			["cache_size"] = 0
		};
	}

	private static FullRunSimulationActionRequest ParseActionRequest(JsonElement paramsElement)
	{
		if (paramsElement.ValueKind != JsonValueKind.Object)
		{
			throw new InvalidOperationException("step requires an action payload.");
		}

		FullRunSimulationActionRequest request = new FullRunSimulationActionRequest();
		if (paramsElement.TryGetProperty("action", out JsonElement action) && action.ValueKind == JsonValueKind.String)
		{
			request.Action = action.GetString() ?? string.Empty;
		}

		if (paramsElement.TryGetProperty("type", out JsonElement type) && type.ValueKind == JsonValueKind.String)
		{
			request.Type = type.GetString();
		}

		if (paramsElement.TryGetProperty("value", out JsonElement value) && value.ValueKind == JsonValueKind.String)
		{
			request.Value = value.GetString();
		}

		if (paramsElement.TryGetProperty("target", out JsonElement target) && target.ValueKind == JsonValueKind.String)
		{
			request.Target = target.GetString();
		}

		if (paramsElement.TryGetProperty("index", out JsonElement index) && index.ValueKind == JsonValueKind.Number)
		{
			request.Index = index.GetInt32();
		}

		if (paramsElement.TryGetProperty("card_index", out JsonElement cardIndex) && cardIndex.ValueKind == JsonValueKind.Number)
		{
			request.CardIndex = cardIndex.GetInt32();
		}

		if (paramsElement.TryGetProperty("hand_index", out JsonElement handIndex) && handIndex.ValueKind == JsonValueKind.Number)
		{
			request.HandIndex = handIndex.GetInt32();
		}

		if (paramsElement.TryGetProperty("slot", out JsonElement slot) && slot.ValueKind == JsonValueKind.Number)
		{
			request.Slot = slot.GetInt32();
		}

		if (paramsElement.TryGetProperty("col", out JsonElement col) && col.ValueKind == JsonValueKind.Number)
		{
			request.Col = col.GetInt32();
		}

		if (paramsElement.TryGetProperty("row", out JsonElement row) && row.ValueKind == JsonValueKind.Number)
		{
			request.Row = row.GetInt32();
		}

		if (paramsElement.TryGetProperty("target_id", out JsonElement targetId) && targetId.ValueKind == JsonValueKind.Number)
		{
			request.TargetId = targetId.GetUInt32();
		}

		return request;
	}

	private static FullRunSimulationStateSnapshot GetSnapshot(FullRunTrainingEnvService service, RequestStateCache cache)
	{
		if (cache.Snapshot == null)
		{
			using IDisposable _ = FullRunSimulationDiagnostics.Measure("request.get_state.runtime_ms");
			cache.Snapshot = service.GetState();
		}
		return cache.Snapshot;
	}

	private static FullRunApiState BuildApiState(FullRunTrainingEnvService service, RequestStateCache cache)
	{
		return BuildApiState(GetSnapshot(service, cache), cache);
	}

	private static FullRunApiState BuildApiState(FullRunSimulationStateSnapshot snapshot, RequestStateCache cache)
	{
		if (cache.ApiState != null && ReferenceEquals(cache.Snapshot, snapshot))
		{
			return cache.ApiState;
		}

		cache.Snapshot = snapshot;
		RunState? runState = RunManager.Instance.DebugOnlyGetState();
		try
		{
			FullRunSimulationTrace.Write(
				$"headless_pipe.build_api_state.begin state_type={snapshot.StateType} floor={snapshot.TotalFloor} terminal={snapshot.IsTerminal} " +
				$"run_state_null={runState == null} current_room={runState?.CurrentRoom?.GetType().Name ?? "null"} players={(runState?.Players?.Count ?? 0)}");
			FullRunApiState state;
			using (FullRunSimulationDiagnostics.Measure("request.api_build_ms"))
			{
				state = FullRunApiStateBuilder.Build(runState, snapshot);
			}
			FullRunSimulationTrace.Write(
				$"headless_pipe.build_api_state.done state_type={state.state_type} legal_actions={state.legal_actions.Count} " +
				$"run_floor={state.run?.floor}");
			cache.ApiState = state;
			return state;
		}
		catch (Exception ex)
		{
			string playerSummary = "none";
			try
			{
				Player? player = runState?.Players?.FirstOrDefault();
				if (player != null)
				{
					playerSummary =
						$"character={player.Character?.Id.Entry ?? "null"} hp={player.Creature?.CurrentHp} max_hp={player.Creature?.MaxHp} " +
						$"gold={player.Gold} deck={(player.Deck?.Cards?.Count ?? -1)} relics={(player.Relics?.Count ?? -1)}";
				}
			}
			catch
			{
				playerSummary = "player_summary_failed";
			}

			FullRunSimulationTrace.Write(
				$"headless_pipe.build_api_state.exception state_type={snapshot.StateType} floor={snapshot.TotalFloor} terminal={snapshot.IsTerminal} " +
				$"player={playerSummary} exception={ex}");
			throw;
		}
	}

	private static Dictionary<string, object?> ResetPerfStats()
	{
		FullRunSimulationDiagnostics.Reset();
		return new Dictionary<string, object?>
		{
			["reset"] = true
		};
	}

	private static double ComputeTerminalReward(string? runOutcome, bool terminal)
	{
		if (!terminal)
		{
			return 0.0;
		}

		string outcome = (runOutcome ?? string.Empty).Trim().ToLowerInvariant();
		return outcome switch
		{
			"victory" or "win" => 1.0,
			"defeat" or "loss" or "death" => -1.0,
			_ => 0.0
		};
	}

	private static Dictionary<string, object?> BuildErrorPayload(string errorCode, string error)
	{
		return new Dictionary<string, object?>
		{
			["error"] = error,
			["error_code"] = errorCode
		};
	}

	private static string SerializePipeError(string errorCode, string error)
	{
		return JsonSerializer.Serialize(BuildErrorPayload(errorCode, error), JsonOptions);
	}

	private static string GetRequiredString(JsonElement element, string propertyName)
	{
		if (element.ValueKind == JsonValueKind.Object
			&& element.TryGetProperty(propertyName, out JsonElement property)
			&& property.ValueKind == JsonValueKind.String)
		{
			string? value = property.GetString();
			if (!string.IsNullOrWhiteSpace(value))
			{
				return value;
			}
		}

		throw new InvalidOperationException($"Request requires a non-empty '{propertyName}' string.");
	}

	private static string? GetStructuredErrorCode(Exception exception)
	{
		if (exception is JsonException)
		{
			return "invalid_json";
		}

		PropertyInfo? errorCodeProperty = exception.GetType().GetProperty(
			"ErrorCode",
			BindingFlags.Public | BindingFlags.Instance);
		if (errorCodeProperty?.PropertyType == typeof(string))
		{
			return errorCodeProperty.GetValue(exception) as string;
		}

		return exception switch
		{
			InvalidOperationException => "invalid_request",
			TimeoutException => "request_timeout",
			_ => null
		};
	}

	private sealed class PipeSessionManager
	{
		private readonly object _sync = new object();
		private long _nextSessionId;
		private long? _activeSessionId;

		public long TryAcquire()
		{
			lock (_sync)
			{
				if (_activeSessionId.HasValue)
				{
					return -1;
				}

				_nextSessionId++;
				_activeSessionId = _nextSessionId;
				return _nextSessionId;
			}
		}

		public void Release(long sessionId)
		{
			lock (_sync)
			{
				if (_activeSessionId == sessionId)
				{
					_activeSessionId = null;
				}
			}
		}
	}

	private sealed class HostOptions
	{
		public int Port { get; private set; } = 15527;

		public bool UseStdio { get; private set; }

		public HostProtocol Protocol { get; private set; } = HostProtocol.Json;

		public TimeSpan ReadTimeout { get; private set; } = TimeSpan.FromSeconds(60);

		public TimeSpan RequestTimeout { get; private set; } = TimeSpan.FromSeconds(45);

		public string? ExportCardRuntimeTextsPath { get; private set; }

		public IReadOnlyList<string> ExportLocales { get; private set; } = new[] { "eng", "zhs" };

		public string PipeName => BinaryProtocol.PipeName(Port, Protocol);

		public static HostOptions Parse(IEnumerable<string> args)
		{
			HostOptions options = new HostOptions();
			string[] values = args.ToArray();
			for (int i = 0; i < values.Length; i++)
			{
				switch (values[i])
				{
					case "--stdio":
						options.UseStdio = true;
						break;
					case "--port" when i + 1 < values.Length && int.TryParse(values[i + 1], out int port):
						options.Port = port;
						i++;
						break;
					case "--read-timeout-seconds" when i + 1 < values.Length && double.TryParse(values[i + 1], out double readSeconds):
						options.ReadTimeout = TimeSpan.FromSeconds(Math.Max(1, readSeconds));
						i++;
						break;
					case "--request-timeout-seconds" when i + 1 < values.Length && double.TryParse(values[i + 1], out double requestSeconds):
						options.RequestTimeout = TimeSpan.FromSeconds(Math.Max(1, requestSeconds));
						i++;
						break;
					case "--protocol" when i + 1 < values.Length:
						string protocol = values[i + 1].Trim().ToLowerInvariant();
						options.Protocol = protocol switch
						{
							"json" => HostProtocol.Json,
							"bin" or "binary" => HostProtocol.Binary,
							"proto" or "protobuf" => HostProtocol.Proto,
							_ => throw new InvalidOperationException($"Unknown protocol '{values[i + 1]}'. Expected 'json', 'bin', or 'proto'.")
						};
						i++;
						break;
					case "--export-card-runtime-texts" when i + 1 < values.Length:
						options.ExportCardRuntimeTextsPath = values[i + 1];
						i++;
						break;
					case "--export-locales" when i + 1 < values.Length:
						options.ExportLocales = values[i + 1]
							.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
							.Select(static s => s.ToLowerInvariant())
							.ToArray();
						i++;
						break;
				}
			}

			return options;
		}
	}
}
