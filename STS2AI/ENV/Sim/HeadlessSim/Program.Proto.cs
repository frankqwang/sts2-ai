using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Google.Protobuf;
using MegaCrit.Sts2.Core.Simulation;
using MegaCrit.Sts2.Core.Training;
using STS2AI.Bridge;

namespace HeadlessSim;

// Proto pipe protocol: request router + all ProcessProto* handlers.
// All messages go through PipeRequestEnvelope / PipeResponseEnvelope.
// sim populates GameState.legal_actions directly; Python never infers them.
internal static partial class Program
{
	private static async Task<byte[]> ProcessProtoRequestAsync(
		FullRunTrainingEnvService service,
		byte[] requestBytes)
	{
		long requestStart = Stopwatch.GetTimestamp();
		PipeRequestEnvelope request = PipeRequestEnvelope.Parser.ParseFrom(requestBytes);
		if (request.Method == PipeMethod.Handshake)
		{
			throw new InvalidOperationException("Handshake is server-initiated and must not be sent as a request.");
		}
		PipeMethod method = request.Method;
		RequestStateCache cache = new RequestStateCache();
		try
		{
			return method switch
			{
				PipeMethod.Reset => await ProcessProtoResetAsync(service, request, cache),
				PipeMethod.State => ProcessProtoState(service, cache),
				PipeMethod.Step => await ProcessProtoStepAsync(service, request, cache),
				PipeMethod.BatchStep => await ProcessProtoBatchStepAsync(service, request, cache),
				PipeMethod.SaveState => ProtoStateBuilder.BuildSaveStateResponse(
					PipeMethod.SaveState, service.SaveState(), service.StateCacheCount),
				PipeMethod.SaveSearchState => ProtoStateBuilder.BuildSaveStateResponse(
					PipeMethod.SaveSearchState, service.SaveSearchState(), service.StateCacheCount),
				PipeMethod.ExportState => ProcessProtoExportState(service, request),
				PipeMethod.LoadState => await ProcessProtoLoadStateAsync(service, request, cache),
				PipeMethod.ImportState => await ProcessProtoImportStateAsync(service, request, cache),
				PipeMethod.DeleteState => ProcessProtoDeleteState(service, request),
				PipeMethod.PerfStats => ProtoStateBuilder.BuildPerfStatsResponse(FullRunSimulationDiagnostics.Snapshot()),
				PipeMethod.ResetPerfStats => ProcessProtoResetPerfStats(),
				PipeMethod.StepLocalPolicy => await ProcessProtoStepLocalPolicyAsync(service, cache),
				PipeMethod.LoadOrtModel => BuildLoadOrtModelResponse(request),
				PipeMethod.RunCombatLocal => await ProcessProtoRunCombatLocalAsync(service, request, cache),
				PipeMethod.SkipCombat => await ProcessProtoSkipCombatAsync(service, cache),
				PipeMethod.SearchCombatMcts => await ProcessSearchCombatMctsAsync(service, request, cache),
				PipeMethod.CombatReset => await ProcessProtoCombatResetAsync(request),
				PipeMethod.CombatStep => await ProcessProtoCombatStepAsync(request),
				PipeMethod.CombatState => ProcessProtoCombatState(),
				_ => ProtoStateBuilder.BuildErrorResponse(method, PipeStatus.ProtocolError, "unknown_method", $"Unknown method: {method}")
			};
		}
		finally
		{
			double elapsedMs = (Stopwatch.GetTimestamp() - requestStart) * 1000.0 / Stopwatch.Frequency;
			FullRunSimulationDiagnostics.RecordTiming($"request.{method.ToString().ToLowerInvariant()}.total_ms", elapsedMs);
			FullRunSimulationDiagnostics.RecordTiming("request.proto_total_ms", elapsedMs);
			FullRunSimulationDiagnostics.Increment($"request.{method.ToString().ToLowerInvariant()}.count");
			FullRunSimulationDiagnostics.Increment("request.proto.count");
		}
	}

	private static async Task<byte[]> ProcessProtoResetAsync(
		FullRunTrainingEnvService service, PipeRequestEnvelope requestEnvelope, RequestStateCache cache)
	{
		if (requestEnvelope.PayloadCase != PipeRequestEnvelope.PayloadOneofCase.Reset)
		{
			throw new InvalidOperationException("reset request missing payload.");
		}
		FullRunSimulationResetRequest request = BuildProtoResetRequest(requestEnvelope.Reset);
		FullRunSimulationStateSnapshot snapshot;
		using (FullRunSimulationDiagnostics.Measure("request.reset.runtime_ms"))
		{
			snapshot = await service.ResetAsync(request);
		}
		using (FullRunSimulationDiagnostics.Measure("request.proto_encode_ms"))
		{
			return ProtoStateBuilder.BuildStateResponse(PipeMethod.Reset, snapshot);
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
			return ProtoStateBuilder.BuildStateResponse(PipeMethod.State, snapshot);
		}
	}

	private static async Task<byte[]> ProcessProtoStepAsync(
		FullRunTrainingEnvService service, PipeRequestEnvelope requestEnvelope, RequestStateCache cache)
	{
		if (requestEnvelope.PayloadCase != PipeRequestEnvelope.PayloadOneofCase.Step)
		{
			throw new InvalidOperationException("step request missing payload.");
		}
		FullRunSimulationActionRequest action = BuildFullRunSimulationActionRequest(requestEnvelope.Step.Action);
		(FullRunSimulationStepResult result, FullRunSimulationStateSnapshot snapshot) =
			await ExecuteFullRunStepAsync(service, cache, action, autoAdvanceToDecisionState: true);
		using (FullRunSimulationDiagnostics.Measure("request.proto_encode_ms"))
		{
			return ProtoStateBuilder.BuildStepResponse(PipeMethod.Step, result, snapshot);
		}
	}

	private static async Task<byte[]> ProcessProtoBatchStepAsync(
		FullRunTrainingEnvService service, PipeRequestEnvelope requestEnvelope, RequestStateCache cache)
	{
		if (requestEnvelope.PayloadCase != PipeRequestEnvelope.PayloadOneofCase.BatchStep)
		{
			throw new InvalidOperationException("batch_step request missing payload.");
		}
		List<FullRunSimulationActionRequest> actions = requestEnvelope.BatchStep.Actions
			.Select(BuildFullRunSimulationActionRequest)
			.ToList();
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

	private static byte[] ProcessProtoExportState(FullRunTrainingEnvService service, PipeRequestEnvelope requestEnvelope)
	{
		if (requestEnvelope.PayloadCase != PipeRequestEnvelope.PayloadOneofCase.ExportState)
		{
			throw new InvalidOperationException("export_state request missing payload.");
		}
		string writtenPath = service.ExportStateToFile(
			requestEnvelope.ExportState.Path,
			string.IsNullOrWhiteSpace(requestEnvelope.ExportState.StateId) ? null : requestEnvelope.ExportState.StateId);
		return ProtoStateBuilder.BuildExportStateResponse(writtenPath, service.StateCacheCount);
	}

	private static async Task<byte[]> ProcessProtoLoadStateAsync(
		FullRunTrainingEnvService service, PipeRequestEnvelope requestEnvelope, RequestStateCache cache)
	{
		if (requestEnvelope.PayloadCase != PipeRequestEnvelope.PayloadOneofCase.LoadState)
		{
			throw new InvalidOperationException("load_state request missing payload.");
		}
		string stateId = requestEnvelope.LoadState.StateId;
		FullRunSimulationStateSnapshot snapshot;
		using (FullRunSimulationDiagnostics.Measure("request.load_state.runtime_ms"))
		{
			snapshot = await service.LoadState(stateId);
		}
		cache.Snapshot = snapshot;
		if (IsCombatLikeLoadSnapshot(snapshot))
		{
			CombatTrainingStateSnapshot? combatSnapshot = CombatTrainingEnvService.BuildStateSnapshot();
			if (combatSnapshot != null && combatSnapshot.IsCombatActive)
			{
				return ProtoStateBuilder.BuildCombatStateResponse(PipeMethod.LoadState, combatSnapshot);
			}
		}
		return ProtoStateBuilder.BuildStateResponse(PipeMethod.LoadState, snapshot);
	}

	private static bool IsCombatLikeLoadSnapshot(FullRunSimulationStateSnapshot snapshot)
	{
		return snapshot.StateType == "monster"
			|| snapshot.StateType == "elite"
			|| snapshot.StateType == "boss"
			|| snapshot.StateType == "hand_select";
	}

	private static async Task<byte[]> ProcessProtoImportStateAsync(
		FullRunTrainingEnvService service, PipeRequestEnvelope requestEnvelope, RequestStateCache cache)
	{
		if (requestEnvelope.PayloadCase != PipeRequestEnvelope.PayloadOneofCase.ImportState)
		{
			throw new InvalidOperationException("import_state request missing payload.");
		}
		string path = requestEnvelope.ImportState.Path;
		FullRunSimulationStateSnapshot snapshot;
		using (FullRunSimulationDiagnostics.Measure("request.import_state.runtime_ms"))
		{
			snapshot = await service.LoadStateFromFile(path);
		}
		cache.Snapshot = snapshot;
		return ProtoStateBuilder.BuildStateResponse(PipeMethod.ImportState, snapshot);
	}

	private static byte[] ProcessProtoDeleteState(FullRunTrainingEnvService service, PipeRequestEnvelope requestEnvelope)
	{
		if (requestEnvelope.PayloadCase != PipeRequestEnvelope.PayloadOneofCase.DeleteState)
		{
			throw new InvalidOperationException("delete_state request missing payload.");
		}
		bool clearAll = requestEnvelope.DeleteState.ClearAll;
		string? stateId = clearAll ? null : requestEnvelope.DeleteState.StateId;
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
		return ProtoStateBuilder.BuildStepResponse(PipeMethod.SkipCombat, skipResult, snapshot);
	}

	// ================================================================
	// Proto combat-only opcodes (2026-04-18)
	//
	// 请求/响应全部走 PipeRequestEnvelope / PipeResponseEnvelope。
	//
	// sim 直接 populate GameState.legal_actions,Python 端不再自己推断。
	// ================================================================

	private static async Task<byte[]> ProcessProtoCombatResetAsync(PipeRequestEnvelope requestEnvelope)
	{
		if (requestEnvelope.PayloadCase != PipeRequestEnvelope.PayloadOneofCase.CombatReset)
		{
			throw new InvalidOperationException("combat_reset request missing payload.");
		}
		STS2AI.Bridge.CombatResetRequest req = requestEnvelope.CombatReset;
		CombatTrainingResetRequest request = BuildCombatTrainingResetRequest(req);
		CombatTrainingStateSnapshot snapshot;
		try
		{
			using IDisposable _ = FullRunSimulationDiagnostics.Measure("request.combat_reset.runtime_ms");
			snapshot = await CombatTrainingEnvService.Instance.ResetAsync(request);
		}
		catch (Exception exc)
		{
			return ProtoStateBuilder.BuildErrorResponse(
				PipeMethod.CombatReset, PipeStatus.SimulatorError,
				"combat_reset_error", exc.Message);
		}
		return ProtoStateBuilder.BuildCombatStateResponse(PipeMethod.CombatReset, snapshot);
	}

	private static async Task<byte[]> ProcessProtoCombatStepAsync(PipeRequestEnvelope requestEnvelope)
	{
		if (requestEnvelope.PayloadCase != PipeRequestEnvelope.PayloadOneofCase.CombatStep)
		{
			throw new InvalidOperationException("combat_step request missing payload.");
		}
		STS2AI.Bridge.CombatStepRequest req = requestEnvelope.CombatStep;
		CombatTrainingActionRequest action;
		try
		{
			action = BuildCombatTrainingActionRequest(req.Action);
		}
		catch (Exception exc)
		{
			return ProtoStateBuilder.BuildErrorResponse(
				PipeMethod.CombatStep, PipeStatus.ProtocolError,
				"action_decode_error", exc.Message);
		}
		CombatTrainingStepResult result;
		using (FullRunSimulationDiagnostics.Measure("request.combat_step.runtime_ms"))
		{
			result = await CombatTrainingEnvService.Instance.StepAsync(action);
		}
		CombatTrainingStateSnapshot snapshot = result.State ?? CombatTrainingEnvService.Instance.GetState();
		return ProtoStateBuilder.BuildCombatStepResponse(result, snapshot);
	}

	private static byte[] ProcessProtoCombatState()
	{
		CombatTrainingStateSnapshot snapshot = CombatTrainingEnvService.Instance.GetState();
		return ProtoStateBuilder.BuildCombatStateResponse(PipeMethod.CombatState, snapshot);
	}

	private static FullRunSimulationResetRequest BuildProtoResetRequest(PipeResetPayload payload)
	{
		return new FullRunSimulationResetRequest
		{
			CharacterId = string.IsNullOrWhiteSpace(payload.CharacterId) ? null : payload.CharacterId,
			Seed = string.IsNullOrWhiteSpace(payload.Seed) ? null : payload.Seed,
			AscensionLevel = payload.AscensionLevel,
			Build = SimulationBuildSupport.ParseJson(
				string.IsNullOrWhiteSpace(payload.BuildJson) ? null : payload.BuildJson),
		};
	}

	private static FullRunSimulationActionRequest BuildFullRunSimulationActionRequest(
		STS2AI.Bridge.LegalAction? action)
	{
		if (action == null)
		{
			throw new InvalidOperationException("step request action is missing.");
		}

		return new FullRunSimulationActionRequest
		{
			Action = string.IsNullOrWhiteSpace(action.Action) ? "other" : action.Action,
			Type = string.IsNullOrWhiteSpace(action.Action) ? "other" : action.Action,
			Index = action.Index >= 0 ? action.Index : null,
			CardIndex = action.CardIndex >= 0 ? action.CardIndex : null,
			TargetId = action.TargetId >= 0 ? (uint?)action.TargetId : null,
			Col = action.Col >= 0 ? action.Col : null,
			Row = action.Row >= 0 ? action.Row : null,
			Slot = action.Slot >= 0 ? action.Slot : null,
			Value = string.IsNullOrWhiteSpace(action.Label) ? null : action.Label,
		};
	}

	private static CombatTrainingResetRequest BuildCombatTrainingResetRequest(
		STS2AI.Bridge.CombatResetRequest req)
	{
		CombatTrainingResetRequest request = new CombatTrainingResetRequest
		{
			CharacterId = string.IsNullOrWhiteSpace(req.CharacterId) ? null : req.CharacterId,
			EncounterId = string.IsNullOrWhiteSpace(req.EncounterId) ? null : req.EncounterId,
			Seed = string.IsNullOrWhiteSpace(req.Seed) ? null : req.Seed,
			AscensionLevel = req.AscensionLevel,
		};
		if (req.Build != null)
		{
			SimulationBuildSpec build = new SimulationBuildSpec();
			if (req.Build.HasCurrentHp)
			{
				build.CurrentHp = req.Build.CurrentHp;
			}
			if (req.Build.HasMaxHp)
			{
				build.MaxHp = req.Build.MaxHp;
			}
			if (req.Build.HasMaxEnergy)
			{
				build.MaxEnergy = req.Build.MaxEnergy;
			}
			if (req.Build.HasGold)
			{
				build.Gold = req.Build.Gold;
			}
			if (req.Build.Deck.Count > 0)
			{
				build.Deck = new List<SimulationBuildCardSpec>();
				foreach (var card in req.Build.Deck)
				{
					if (string.IsNullOrWhiteSpace(card.Id)) continue;
					build.Deck.Add(new SimulationBuildCardSpec
					{
						Id = card.Id,
						UpgradeLevel = card.UpgradeLevel,
					});
				}
			}
			if (req.Build.Relics.Count > 0)
			{
				build.Relics = new List<SimulationBuildRelicSpec>();
				foreach (var relic in req.Build.Relics)
				{
					if (string.IsNullOrWhiteSpace(relic.Id)) continue;
					build.Relics.Add(new SimulationBuildRelicSpec { Id = relic.Id });
				}
			}
			if (req.Build.Potions.Count > 0)
			{
				build.Potions = new List<SimulationBuildPotionSpec>();
				foreach (var potion in req.Build.Potions)
				{
					if (string.IsNullOrWhiteSpace(potion.Id)) continue;
					build.Potions.Add(new SimulationBuildPotionSpec
					{
						Id = potion.Id,
						Slot = potion.Slot
					});
				}
			}
			if (req.Build.HasMaxPotionSlots)
			{
				build.MaxPotionSlots = req.Build.MaxPotionSlots;
			}
			request.Build = build;
		}
		return request;
	}

	private static CombatTrainingActionRequest BuildCombatTrainingActionRequest(
		STS2AI.Bridge.LegalAction? action)
	{
		if (action == null)
		{
			throw new InvalidOperationException("CombatStepRequest.action is missing.");
		}
		string raw = (action.Action ?? string.Empty).Trim().ToLowerInvariant();
		CombatTrainingActionType type = ParseCombatActionType(raw);
		CombatTrainingActionRequest req = new CombatTrainingActionRequest { Type = type };
		int chosenIdx = action.CardIndex >= 0 ? action.CardIndex : action.Index;
		if (chosenIdx >= 0)
		{
			// select_card_option 走 ChoiceIndex;其他 (play_card/select_hand_card)走 HandIndex
			if (type == CombatTrainingActionType.SelectCardChoice)
			{
				req.ChoiceIndex = chosenIdx;
			}
			else
			{
				req.HandIndex = chosenIdx;
			}
		}
		if (action.TargetId > 0)
		{
			req.TargetId = (uint)action.TargetId;
		}
		if (action.Slot != 0)
		{
			req.Slot = action.Slot;
		}
		return req;
	}

	private static async Task<byte[]> ProcessProtoStepLocalPolicyAsync(
		FullRunTrainingEnvService service, RequestStateCache cache)
	{
		// ORT 推理走统一逻辑，但响应要直接回 proto step payload。
		if (_ortPolicy == null)
		{
			return ProtoStateBuilder.BuildErrorResponse(
				PipeMethod.StepLocalPolicy, PipeStatus.SimulatorError,
				"no_ort_model", "No ORT model loaded. Call load_ort_model first.");
		}
		FullRunSimulationStateSnapshot snapshot = GetSnapshot(service, cache);
		// 使用 ORT 策略选择 action，然后 step
		int actionIndex = _ortPolicy.SelectAction(snapshot, _ortRng).Item1;
		if (actionIndex < 0 || actionIndex >= snapshot.LegalActions.Count)
		{
			return ProtoStateBuilder.BuildErrorResponse(
				PipeMethod.StepLocalPolicy, PipeStatus.SimulatorError,
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
		return ProtoStateBuilder.BuildStepResponse(PipeMethod.StepLocalPolicy, result, nextSnapshot);
	}

	private static async Task<byte[]> ProcessProtoRunCombatLocalAsync(
		FullRunTrainingEnvService service, PipeRequestEnvelope requestEnvelope, RequestStateCache cache)
	{
		if (_ortPolicy == null)
		{
			return ProtoStateBuilder.BuildErrorResponse(
				PipeMethod.RunCombatLocal,
				PipeStatus.SimulatorError,
				"ort_not_loaded",
				"ORT model not loaded. Call load_ort_model first.");
		}

		try
		{
			int maxCombatSteps = 600;
			if (requestEnvelope.PayloadCase == PipeRequestEnvelope.PayloadOneofCase.RunCombatLocal)
			{
				maxCombatSteps = Math.Max(1, requestEnvelope.RunCombatLocal.MaxSteps);
			}

			FullRunSimulationStateSnapshot snapshot = GetSnapshot(service, cache);
			bool isCombat = snapshot.StateType is "monster" or "elite" or "boss" or "combat";
			if (!isCombat)
			{
			return ProtoStateBuilder.BuildRunCombatLocalResponse(
				0, 0f, 0f, 0f, 0f, 0f, 0f, 0f, snapshot);
			}

			int combatSteps = 0;
			int waitSteps = 0;
			Stopwatch stopwatch = Stopwatch.StartNew();
			const long combatTimeoutMs = 10_000;

			long totalGetSnapshotTicks = 0;
			long totalOrtTicks = 0;
			long totalStepAsyncTicks = 0;
			long totalWaitAsyncTicks = 0;
			long maxStepAsyncTicks = 0;
			long maxWaitAsyncTicks = 0;

			for (int step = 0; step < maxCombatSteps; step++)
			{
				long t0 = Stopwatch.GetTimestamp();
				snapshot = GetSnapshot(service, cache);
				long t1 = Stopwatch.GetTimestamp();
				totalGetSnapshotTicks += t1 - t0;

				if (stopwatch.ElapsedMilliseconds > combatTimeoutMs)
				{
					FullRunSimulationDiagnostics.Increment("request.run_combat_local.timeout");
					break;
				}

				if (snapshot.IsTerminal || snapshot.StateType == "game_over")
				{
					break;
				}

				bool stillCombat = snapshot.StateType is "monster" or "elite" or "boss" or "combat";
				if (!stillCombat)
				{
					break;
				}

				if (snapshot.LegalActions.Count == 0)
				{
					long waitStart = Stopwatch.GetTimestamp();
					FullRunSimulationStepResult waitResult = await service.StepAsync(
						new FullRunSimulationActionRequest { Action = "wait" });
					long waitEnd = Stopwatch.GetTimestamp();
					long waitTicks = waitEnd - waitStart;
					totalWaitAsyncTicks += waitTicks;
					if (waitTicks > maxWaitAsyncTicks)
					{
						maxWaitAsyncTicks = waitTicks;
					}
					waitSteps++;
					if (waitResult.State != null)
					{
						cache.Snapshot = null;
						cache.ApiState = null;
					}
					continue;
				}

				long ortStart = Stopwatch.GetTimestamp();
				var (actionIdx, _) = _ortPolicy.SelectAction(snapshot, _ortRng);
				long ortEnd = Stopwatch.GetTimestamp();
				totalOrtTicks += ortEnd - ortStart;

				FullRunSimulationLegalAction action = snapshot.LegalActions[actionIdx];
				FullRunSimulationActionRequest stepRequest = new FullRunSimulationActionRequest
				{
					Action = action.Action,
					Index = action.Index,
					CardIndex = action.CardIndex,
					Slot = action.Slot,
					Col = action.Col,
					Row = action.Row,
				};
				if (action.TargetId.HasValue)
				{
					stepRequest.TargetId = action.TargetId;
				}

				long stepStart = Stopwatch.GetTimestamp();
				FullRunSimulationStepResult result = await service.StepAsync(stepRequest);
				long stepEnd = Stopwatch.GetTimestamp();
				long stepTicks = stepEnd - stepStart;
				totalStepAsyncTicks += stepTicks;
				if (stepTicks > maxStepAsyncTicks)
				{
					maxStepAsyncTicks = stepTicks;
				}
				cache.Snapshot = null;
				cache.ApiState = null;
				combatSteps++;

				const int maxAutoAdvance = 30;
				for (int i = 0; i < maxAutoAdvance; i++)
				{
					FullRunSimulationStateSnapshot advState = result.State ?? GetSnapshot(service, cache);
					if (advState.IsTerminal || advState.StateType == "game_over")
					{
						break;
					}
					if (advState.LegalActions.Count > 0)
					{
						break;
					}
					result = await service.StepAsync(new FullRunSimulationActionRequest { Action = "wait" });
					cache.Snapshot = null;
					cache.ApiState = null;
				}
			}

			stopwatch.Stop();
			FullRunSimulationDiagnostics.Increment("request.run_combat_local.calls");
			FullRunSimulationDiagnostics.Increment("request.run_combat_local.total_steps", combatSteps);

			double tickFreq = Stopwatch.Frequency / 1000.0;
			float getSnapshotMs = (float)(totalGetSnapshotTicks / tickFreq);
			float ortMs = (float)(totalOrtTicks / tickFreq);
			float stepAsyncMs = (float)(totalStepAsyncTicks / tickFreq);
			float waitAsyncMs = (float)(totalWaitAsyncTicks / tickFreq);
			float maxStepMs = (float)(maxStepAsyncTicks / tickFreq);
			float maxWaitMs = (float)(maxWaitAsyncTicks / tickFreq);

			if (maxStepMs > 100 || maxWaitMs > 100)
			{
				ThreadPool.GetMinThreads(out int minWorker, out int minIo);
				ThreadPool.GetMaxThreads(out int maxWorker, out int maxIo);
				ThreadPool.GetAvailableThreads(out int availWorker, out int availIo);
				Console.Error.WriteLine(
					$"[ORT LONGTAIL] steps={combatSteps} waits={waitSteps} " +
					$"maxStep={maxStepMs:F1}ms maxWait={maxWaitMs:F1}ms " +
					$"totalStep={stepAsyncMs:F0}ms totalWait={waitAsyncMs:F0}ms " +
					$"ThreadPool min={minWorker}/{minIo} max={maxWorker}/{maxIo} avail={availWorker}/{availIo}");
			}

			FullRunSimulationStateSnapshot finalSnapshot = GetSnapshot(service, cache);
			return ProtoStateBuilder.BuildRunCombatLocalResponse(
				combatSteps,
				(float)stopwatch.Elapsed.TotalMilliseconds,
				getSnapshotMs,
				ortMs,
				stepAsyncMs,
				waitAsyncMs,
				maxStepMs,
				maxWaitMs,
				finalSnapshot);
		}
		catch (Exception ex)
		{
			Console.Error.WriteLine($"[ORT] RunCombatLocal error: {ex.Message}");
			return ProtoStateBuilder.BuildErrorResponse(
				PipeMethod.RunCombatLocal,
				PipeStatus.SimulatorError,
				"ort_combat_error",
				ex.Message);
		}
	}

	// --- Local ORT actor policy ---
	private static OrtActorPolicy? _ortPolicy;
	private static readonly Random _ortRng = new(42);
	private static readonly Random _mctsRng = new(1234);

	private static byte[] BuildLoadOrtModelResponse(PipeRequestEnvelope requestEnvelope)
	{
		try
		{
			if (requestEnvelope.PayloadCase != PipeRequestEnvelope.PayloadOneofCase.LoadOrtModel)
			{
				throw new InvalidOperationException("load_ort_model request missing payload.");
			}
			string onnxPath = requestEnvelope.LoadOrtModel.Path;

			_ortPolicy?.Dispose();
			string? vocabPath = Path.Combine(Path.GetDirectoryName(onnxPath) ?? string.Empty, "vocab_mapping.json");
			if (!File.Exists(vocabPath))
			{
				vocabPath = null;
			}

			_ortPolicy = new OrtActorPolicy(onnxPath, argmax: false, vocabPath: vocabPath);
			Console.Error.WriteLine(
				$"[ORT] Loaded model from {onnxPath} (vocab={vocabPath != null}, provider={_ortPolicy.ExecutionProviderName}, requested={_ortPolicy.RequestedDevice}, fallback={_ortPolicy.FellBackToCpu})");
			return ProtoStateBuilder.BuildLoadOrtModelResponse(
				loaded: true,
				hasValueOutput: _ortPolicy.Metadata.HasValueOutput,
				hasDeckInputs: _ortPolicy.Metadata.HasDeckInputs,
				hasContinuationOutput: _ortPolicy.Metadata.HasContinuationOutput,
				hasExtraScalarsInput: _ortPolicy.Metadata.HasExtraScalarsInput,
				executionProviderName: _ortPolicy.ExecutionProviderName,
				requestedDevice: _ortPolicy.RequestedDevice,
				fellBackToCpu: _ortPolicy.FellBackToCpu);
		}
		catch (Exception ex)
		{
			Console.Error.WriteLine($"[ORT] Load failed: {ex.Message}");
			return ProtoStateBuilder.BuildErrorResponse(
				PipeMethod.LoadOrtModel,
				PipeStatus.SimulatorError,
				"ort_load_error",
				ex.Message);
		}
	}

	private static async Task<byte[]> ProcessSearchCombatMctsAsync(
		FullRunTrainingEnvService service,
		PipeRequestEnvelope requestEnvelope,
		RequestStateCache cache)
	{
		if (_ortPolicy == null)
		{
			return ProtoStateBuilder.BuildErrorResponse(
				PipeMethod.SearchCombatMcts,
				PipeStatus.SimulatorError,
				"ort_not_loaded",
				"ORT model not loaded. Call load_ort_model first.");
		}

		try
		{
			if (requestEnvelope.PayloadCase != PipeRequestEnvelope.PayloadOneofCase.SearchCombatMcts)
			{
				throw new InvalidOperationException("search_combat_mcts request missing payload.");
			}
			PipeSearchCombatMctsRequest request = requestEnvelope.SearchCombatMcts;
			FullRunSimulationStateSnapshot snapshot = GetSnapshot(service, cache);
			bool isCombat = snapshot.StateType is "monster" or "elite" or "boss" or "combat" or "hand_select" or "card_select" or "combat_pending" or "combat_start_pending";
			if (!isCombat)
			{
				return ProtoStateBuilder.BuildErrorResponse(
					PipeMethod.SearchCombatMcts,
					PipeStatus.ProtocolError,
					"not_in_combat",
					$"search_combat_mcts requires a combat state, got '{snapshot.StateType}'.");
			}

			CombatMctsSearchEngine engine = new(service, _ortPolicy, _mctsRng);
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
			return ProtoStateBuilder.BuildSearchCombatMctsResponse(result);
		}
		catch (Exception ex)
		{
			Console.Error.WriteLine($"[MCTS] SearchCombatMcts error: {ex}");
			return ProtoStateBuilder.BuildErrorResponse(
				PipeMethod.SearchCombatMcts,
				PipeStatus.SimulatorError,
				"combat_mcts_error",
				ex.Message);
		}
	}

	private static PipeMethod SafeParseMethod(byte[] requestBytes)
	{
		try
		{
			return PipeRequestEnvelope.Parser.ParseFrom(requestBytes).Method;
		}
		catch
		{
			return PipeMethod.State;
		}
	}

	private static PipeStatus GetProtoPipeErrorStatus(Exception exception)
	{
		return exception switch
		{
			InvalidOperationException => PipeStatus.ProtocolError,
			JsonException => PipeStatus.ProtocolError,
			EndOfStreamException => PipeStatus.ProtocolError,
			InvalidProtocolBufferException => PipeStatus.ProtocolError,
			TimeoutException => PipeStatus.ProtocolError,
			_ => PipeStatus.SimulatorError
		};
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
}
