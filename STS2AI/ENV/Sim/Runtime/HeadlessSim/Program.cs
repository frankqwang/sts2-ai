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
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Simulation;
using MegaCrit.Sts2.Core.TestSupport;

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
				if (options.Protocol == HostProtocol.Binary)
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

					if (options.Protocol == HostProtocol.Binary)
					{
						byte[] responseBytes;
						try
						{
							using CancellationTokenSource requestCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
							requestCts.CancelAfter(options.RequestTimeout);
							responseBytes = await ProcessBinaryRequestAsync(service, binarySession!, requestBytes).WaitAsync(requestCts.Token);
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
							Console.Error.WriteLine($"HeadlessSim: binary request error opcode={SafeParseOpcode(requestBytes)}: {ex}");
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
		"map", "card_reward", "shop", "rest_site", "campfire", "event",
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

	private static byte[] ProcessBinaryResetPerfStats()
	{
		FullRunSimulationDiagnostics.Reset();
		return BinaryProtocol.BuildResetPerfStatsResponse();
	}

	// --- Local ORT actor policy ---
	private static OrtActorPolicy? _ortPolicy;
	private static Random _ortRng = new Random(42);

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
			Console.Error.WriteLine($"[ORT] Loaded model from {onnxPath} (vocab={vocabPath != null})");

			// Use standard response format: status + opcode + payload
			using var ms = new MemoryStream();
			using var writer = new BinaryWriter(ms);
			writer.Write((byte)BinaryStatus.Ok);
			writer.Write((byte)BinaryOpcode.LoadOrtModel);
			writer.Write((ushort)0); // zero symbol updates
			writer.Write((byte)1); // loaded = true
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
		}

		using IDisposable _ = FullRunSimulationDiagnostics.Measure("request.reset.runtime_ms");
		return await service.ResetAsync(request);
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
							_ => throw new InvalidOperationException($"Unknown protocol '{values[i + 1]}'. Expected 'json' or 'bin'.")
						};
						i++;
						break;
				}
			}

			return options;
		}
	}
}
