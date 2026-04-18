using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Simulation;

public sealed class FullRunSimulatorSmokeRunner
{
	private const int MaxSmokeSteps = 8;

	private static readonly JsonSerializerOptions JsonOptions = new JsonSerializerOptions
	{
		WriteIndented = true
	};

	public async Task<int> StartAsync(NGame game)
	{
		if (game == null)
		{
			throw new ArgumentNullException(nameof(game));
		}
		try
		{
			FullRunSimulationTrace.Reset();
			FullRunSimulationTrace.Write("full_run_smoke.start");
			Log.Info("[FullRunSimSmoke] Starting pure full-run simulator smoke run");
			FullRunSimulationStateSnapshot state = await FullRunTrainingEnvService.Instance.ResetAsync(new FullRunSimulationResetRequest
			{
				CharacterId = FullRunSimulationMode.CharacterId,
				Seed = FullRunSimulationMode.Seed,
				AscensionLevel = FullRunSimulationMode.AscensionLevel
			});
			FullRunSimulationTrace.Write($"full_run_smoke.after_reset state_type={state.StateType} room_type={state.RoomType} floor={state.TotalFloor} seed={state.Seed}");
			FullRunSimulationTrace.Write($"full_run_smoke.policy_version=v2 max_steps={MaxSmokeSteps}");
			Log.Info($"[FullRunSimSmoke] Reset complete. state_type={state.StateType} room_type={state.RoomType} floor={state.TotalFloor} seed={state.Seed} pure={state.IsPureSimulator}");
			if (!state.IsRunActive || state.StateType == "menu")
			{
				FullRunSimulationTrace.Write("full_run_smoke.invalid_state");
				return 2;
			}
			var initialApiState = FullRunApiStateBuilder.Build(RunManager.Instance.DebugOnlyGetState(), state);
			var stepRecords = new System.Collections.Generic.List<object>();
			bool anyRejected = false;
			FullRunSimulationStateSnapshot currentState = state;
			FullRunApiState currentApiState = initialApiState;
			for (int stepIndex = 0; stepIndex < MaxSmokeSteps; stepIndex++)
			{
				FullRunSimulationActionRequest? nextAction = BuildNextAction(currentState, currentApiState);
				if (nextAction == null)
				{
					FullRunSimulationTrace.Write($"full_run_smoke.no_action step_index={stepIndex} state_type={currentState.StateType}");
					break;
				}

				FullRunSimulationStepResult stepResult = await FullRunTrainingEnvService.Instance.StepAsync(nextAction);
				FullRunApiState? nextApiState = stepResult.State == null
					? null
					: FullRunApiStateBuilder.Build(RunManager.Instance.DebugOnlyGetState(), stepResult.State);
				stepRecords.Add(new
				{
					index = stepIndex,
					action = nextAction,
					result = stepResult,
					api_state = nextApiState
				});
				FullRunSimulationTrace.Write($"full_run_smoke.after_step index={stepIndex} action={nextAction.Type} accepted={stepResult.Accepted} state_type={stepResult.State?.StateType} room_type={stepResult.State?.RoomType} floor={stepResult.State?.TotalFloor}");
				if (!stepResult.Accepted || stepResult.State == null)
				{
					if (!stepResult.Accepted)
					{
						anyRejected = true;
					}
					break;
				}

				currentState = stepResult.State;
				currentApiState = nextApiState ?? currentApiState;
			}
			string artifactDir = Path.Combine("artifacts", "full_run_sim_smoke");
			Directory.CreateDirectory(artifactDir);
			File.WriteAllText(Path.Combine(artifactDir, "latest.json"), JsonSerializer.Serialize(new
			{
				completed = true,
				initial_state = state,
				initial_api_state = initialApiState,
				final_state = currentState,
				final_api_state = FullRunApiStateBuilder.Build(RunManager.Instance.DebugOnlyGetState(), currentState),
				steps = stepRecords
			}, JsonOptions));
			FullRunSimulationTrace.Write("full_run_smoke.done");
			return anyRejected ? 3 : 0;
		}
		catch (Exception ex)
		{
			FullRunSimulationTrace.Write($"full_run_smoke.exception={ex.GetType().Name}:{ex.Message}");
			Log.Error($"[FullRunSimSmoke] Failed: {ex}");
			return 1;
		}
	}

	private static FullRunSimulationActionRequest? BuildNextAction(FullRunSimulationStateSnapshot state, FullRunApiState? apiState)
	{
		switch (state.StateType)
		{
			case "event":
				return BuildEventAction(state);
			case "combat_rewards":
				return BuildCombatRewardsAction(state, apiState);
			case "card_reward":
				return BuildCardRewardAction(state);
			case "card_select":
				return BuildCardSelectAction(state, apiState);
			case "relic_select":
				return BuildRelicSelectAction(state);
			case "rest_site":
				return BuildRestSiteAction(state, apiState);
			case "shop":
				return BuildShopAction(state, apiState);
			case "treasure":
				return BuildTreasureAction(state, apiState);
			case "map":
				return BuildMapAction(state);
		}

		FullRunSimulationLegalAction? action = state.LegalActions.FirstOrDefault(static item => item.IsSupported);
		if (action == null)
		{
			return null;
		}

		return new FullRunSimulationActionRequest
		{
			Type = action.Action,
			Index = action.Index,
			Col = action.Col,
			Row = action.Row
		};
	}

	private static FullRunSimulationActionRequest? BuildEventAction(FullRunSimulationStateSnapshot state)
	{
		FullRunSimulationLegalAction? proceed = state.LegalActions.FirstOrDefault(static action => action.IsSupported && action.Action == "proceed");
		if (proceed != null)
		{
			return ToRequest(proceed);
		}

		FullRunSimulationLegalAction? choose = state.LegalActions
			.Where(static action => action.IsSupported && action.Action == "choose_event_option")
			.OrderBy(static action => action.Index ?? int.MaxValue)
			.FirstOrDefault();
		if (choose != null)
		{
			return ToRequest(choose);
		}

		FullRunSimulationLegalAction? advance = state.LegalActions.FirstOrDefault(static action => action.IsSupported && action.Action == "advance_dialogue");
		return advance == null ? null : ToRequest(advance);
	}

	private static FullRunSimulationActionRequest? BuildCombatRewardsAction(FullRunSimulationStateSnapshot state, FullRunApiState? apiState)
	{
		if (apiState?.rewards?.items != null)
		{
			foreach (FullRunApiRewardItem reward in apiState.rewards.items
				.OrderBy(static item => RewardPriority(item.type))
				.ThenBy(static item => item.index))
			{
				FullRunSimulationLegalAction? claim = state.LegalActions.FirstOrDefault(action =>
					action.IsSupported &&
					action.Action == "claim_reward" &&
					action.Index == reward.index);
				if (claim != null)
				{
					return ToRequest(claim);
				}
			}
		}

		FullRunSimulationLegalAction? proceed = state.LegalActions.FirstOrDefault(static action => action.IsSupported && action.Action == "proceed");
		return proceed == null ? null : ToRequest(proceed);
	}

	private static FullRunSimulationActionRequest? BuildCardRewardAction(FullRunSimulationStateSnapshot state)
	{
		FullRunSimulationLegalAction? select = state.LegalActions
			.Where(static action => action.IsSupported && action.Action == "select_card_reward")
			.OrderBy(static action => action.Index ?? int.MaxValue)
			.FirstOrDefault();
		if (select != null)
		{
			return ToRequest(select);
		}

		FullRunSimulationLegalAction? skip = state.LegalActions.FirstOrDefault(static action => action.IsSupported && action.Action == "skip_card_reward");
		return skip == null ? null : ToRequest(skip);
	}

	private static FullRunSimulationActionRequest? BuildCardSelectAction(FullRunSimulationStateSnapshot state, FullRunApiState? apiState)
	{
		if (apiState?.card_select?.can_confirm == true)
		{
			FullRunSimulationLegalAction? confirm = state.LegalActions.FirstOrDefault(static action => action.IsSupported && action.Action == "confirm_selection");
			if (confirm != null)
			{
				return ToRequest(confirm);
			}
		}

		HashSet<int> selectedIndexes = apiState?.card_select?.selected_cards?
			.Select(static card => card.index)
			.ToHashSet() ?? new HashSet<int>();
		FullRunSimulationLegalAction? choose = state.LegalActions
			.Where(static action => action.IsSupported && action.Action == "select_card")
			.OrderBy(action => selectedIndexes.Contains(action.Index ?? -1))
			.ThenBy(static action => action.Index ?? int.MaxValue)
			.FirstOrDefault();
		if (choose != null)
		{
			return ToRequest(choose);
		}

		FullRunSimulationLegalAction? cancel = state.LegalActions.FirstOrDefault(static action => action.IsSupported && action.Action == "cancel_selection");
		return cancel == null ? null : ToRequest(cancel);
	}

	private static FullRunSimulationActionRequest? BuildMapAction(FullRunSimulationStateSnapshot state)
	{
		FullRunSimulationLegalAction? choose = state.LegalActions
			.Where(static action => action.IsSupported && action.Action == "choose_map_node")
			.OrderBy(static action => MapPointPriority(action.Label))
			.ThenBy(static action => action.Row ?? int.MaxValue)
			.ThenBy(static action => action.Col ?? int.MaxValue)
			.FirstOrDefault();
		return choose == null ? null : ToRequest(choose);
	}

	private static FullRunSimulationActionRequest? BuildRelicSelectAction(FullRunSimulationStateSnapshot state)
	{
		FullRunSimulationLegalAction? choose = state.LegalActions
			.Where(static action => action.IsSupported && action.Action == "select_relic")
			.OrderBy(static action => action.Index ?? int.MaxValue)
			.FirstOrDefault();
		if (choose != null)
		{
			return ToRequest(choose);
		}

		FullRunSimulationLegalAction? skip = state.LegalActions.FirstOrDefault(static action => action.IsSupported && action.Action == "skip_relic_selection");
		return skip == null ? null : ToRequest(skip);
	}

	private static FullRunSimulationActionRequest? BuildRestSiteAction(FullRunSimulationStateSnapshot state, FullRunApiState? apiState)
	{
		if (apiState?.rest_site?.options != null)
		{
			FullRunApiRestSiteOption? smith = apiState.rest_site.options.FirstOrDefault(static option =>
				option.is_enabled && string.Equals(option.id, "smith", StringComparison.OrdinalIgnoreCase));
			if (smith != null)
			{
				FullRunSimulationLegalAction? chooseSmith = state.LegalActions.FirstOrDefault(action =>
					action.IsSupported &&
					action.Action == "choose_rest_option" &&
					action.Index == smith.index);
				if (chooseSmith != null)
				{
					return ToRequest(chooseSmith);
				}
			}

			FullRunApiRestSiteOption? firstEnabled = apiState.rest_site.options.FirstOrDefault(static option => option.is_enabled);
			if (firstEnabled != null)
			{
				FullRunSimulationLegalAction? choose = state.LegalActions.FirstOrDefault(action =>
					action.IsSupported &&
					action.Action == "choose_rest_option" &&
					action.Index == firstEnabled.index);
				if (choose != null)
				{
					return ToRequest(choose);
				}
			}
		}

		FullRunSimulationLegalAction? proceed = state.LegalActions.FirstOrDefault(static action => action.IsSupported && action.Action == "proceed");
		return proceed == null ? null : ToRequest(proceed);
	}

	private static FullRunSimulationActionRequest? BuildShopAction(FullRunSimulationStateSnapshot state, FullRunApiState? apiState)
	{
		if (apiState?.shop?.items != null)
		{
			FullRunApiShopItem? best = apiState.shop.items
				.Where(static item => item.is_stocked && item.can_afford)
				.OrderBy(static item => ShopCategoryPriority(item.category))
				.ThenBy(static item => item.index)
				.FirstOrDefault();
			if (best != null)
			{
				FullRunSimulationLegalAction? buy = state.LegalActions.FirstOrDefault(action =>
					action.IsSupported &&
					action.Action == "shop_purchase" &&
					action.Index == best.index);
				if (buy != null)
				{
					return ToRequest(buy);
				}
			}
		}

		FullRunSimulationLegalAction? proceed = state.LegalActions.FirstOrDefault(static action => action.IsSupported && action.Action == "proceed");
		return proceed == null ? null : ToRequest(proceed);
	}

	private static FullRunSimulationActionRequest? BuildTreasureAction(FullRunSimulationStateSnapshot state, FullRunApiState? apiState)
	{
		if (apiState?.treasure?.relics != null)
		{
			FullRunApiRelicOption? best = apiState.treasure.relics
				.OrderBy(static relic => RelicRarityPriority(relic.rarity))
				.ThenBy(static relic => relic.index)
				.FirstOrDefault();
			if (best != null)
			{
				FullRunSimulationLegalAction? claim = state.LegalActions.FirstOrDefault(action =>
					action.IsSupported &&
					action.Action == "claim_treasure_relic" &&
					action.Index == best.index);
				if (claim != null)
				{
					return ToRequest(claim);
				}
			}
		}

		FullRunSimulationLegalAction? proceed = state.LegalActions.FirstOrDefault(static action => action.IsSupported && action.Action == "proceed");
		return proceed == null ? null : ToRequest(proceed);
	}

	private static int RewardPriority(string? rewardType)
	{
		return rewardType?.ToLowerInvariant() switch
		{
			"card" => 0,
			"relic" => 1,
			"gold" => 2,
			"linked" => 3,
			"potion" => 4,
			_ => 5
		};
	}

	private static int ShopCategoryPriority(string? category)
	{
		return category?.ToLowerInvariant() switch
		{
			"relic" => 0,
			"card_removal" => 1,
			"card" => 2,
			"potion" => 3,
			_ => 4
		};
	}

	private static int RelicRarityPriority(string? rarity)
	{
		return rarity?.ToLowerInvariant() switch
		{
			"rare" => 0,
			"uncommon" => 1,
			"common" => 2,
			_ => 3
		};
	}

	private static int MapPointPriority(string? label)
	{
		string normalized = (label ?? string.Empty).ToLowerInvariant();
		if (normalized.Contains("treasure"))
		{
			return 0;
		}
		if (normalized.Contains("shop"))
		{
			return 1;
		}
		if (normalized.Contains("restsite") || normalized.Contains("rest"))
		{
			return 2;
		}
		if (normalized.Contains("unknown") || normalized.Contains("event") || normalized.Contains("ancient"))
		{
			return 3;
		}
		if (normalized.Contains("monster"))
		{
			return 4;
		}
		if (normalized.Contains("elite"))
		{
			return 5;
		}
		if (normalized.Contains("boss"))
		{
			return 6;
		}
		return 7;
	}

	private static FullRunSimulationActionRequest ToRequest(FullRunSimulationLegalAction action)
	{
		return new FullRunSimulationActionRequest
		{
			Type = action.Action,
			Index = action.Index,
			Col = action.Col,
			Row = action.Row
		};
	}
}
