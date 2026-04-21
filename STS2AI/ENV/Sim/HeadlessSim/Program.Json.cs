using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Simulation;
using MegaCrit.Sts2.Core.Training;

namespace HeadlessSim;

// JSON pipe protocol: request router + per-method handlers + JSON-only helpers
// (catalog builders, api-state builders, action parsers).
internal static partial class Program
{
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
			"save_search_state" => new Dictionary<string, object?>
			{
				["state_id"] = service.SaveSearchState(GetOptionalBoolean(paramsElement, "include_full_fallback")),
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

		return ParseCombatActionType((raw ?? string.Empty).Trim().ToLowerInvariant());
	}

	private static CombatTrainingActionType ParseCombatActionType(string raw)
	{
		return raw switch
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

	private static bool GetOptionalBoolean(JsonElement element, string propertyName, bool defaultValue = false)
	{
		if (element.ValueKind == JsonValueKind.Object
			&& element.TryGetProperty(propertyName, out JsonElement property)
			&& (property.ValueKind == JsonValueKind.True || property.ValueKind == JsonValueKind.False))
		{
			return property.GetBoolean();
		}

		return defaultValue;
	}
}
