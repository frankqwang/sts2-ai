using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Simulation;

namespace STS2_MCP;

/// <summary>
/// Converts the existing Dictionary-based game state (from McpMod.BuildGameState)
/// into strongly-typed <see cref="FullRunApiState"/> DTOs shared with the Sim backend.
/// This ensures both backends produce identical JSON when serialized.
/// </summary>
public static class SpectatorApiStateBuilder
{
	/// <summary>
	/// Build a FullRunApiState by calling the existing BuildGameState/BuildFullRunLegalActions
	/// and converting the dict output to the shared DTO types.
	/// </summary>
	public static FullRunApiState Build()
	{
		// Reuse existing state building logic (proven correct)
		Dictionary<string, object?> dict = McpMod.BuildGameStateForApi();
		string stateType = GetString(dict, "state_type") ?? "unknown";

		// Build legal actions from the dict state (reuses existing logic)
		var dictLegalActions = McpMod.BuildFullRunLegalActionsForApi(dict);

		var state = new FullRunApiState
		{
			state_type = stateType,
			backend_kind = "spectator",
			coverage_tier = "visible",
			is_pure_simulator = false,
			run = ConvertRun(dict),
			legal_actions = dictLegalActions.Select(ConvertAction).ToList()
		};

		// Convert screen-specific state
		switch (stateType)
		{
			case "map":
				state.map = ConvertMapState(GetDict(dict, "map"));
				break;
			case "monster":
			case "elite":
			case "boss":
				state.battle = ConvertBattleState(GetDict(dict, "battle"), dict);
				break;
			case "hand_select":
				state.battle = ConvertBattleState(GetDict(dict, "battle"), dict);
				state.hand_select = ConvertHandSelectState(GetDict(dict, "hand_select"));
				break;
			case "event":
				state.@event = ConvertEventState(GetDict(dict, "event"));
				break;
			case "shop":
				state.shop = ConvertShopState(GetDict(dict, "shop"));
				break;
			case "rest_site":
				state.rest_site = ConvertRestSiteState(GetDict(dict, "rest_site"));
				break;
			case "treasure":
				state.treasure = ConvertTreasureState(GetDict(dict, "treasure"));
				break;
			case "combat_rewards":
				state.rewards = ConvertRewardsState(GetDict(dict, "rewards"));
				break;
			case "card_reward":
				state.card_reward = ConvertCardRewardState(GetDict(dict, "card_reward"));
				break;
			case "card_select":
				state.card_select = ConvertCardSelectState(GetDict(dict, "card_select"));
				break;
			case "relic_select":
				state.relic_select = ConvertRelicSelectState(GetDict(dict, "relic_select"));
				break;
			case "game_over":
				var goDict = GetDict(dict, "game_over");
				string? outcome = GetString(goDict, "outcome") ?? GetString(dict, "run_outcome");
				state.run_outcome = outcome;
				state.terminal = true;
				state.game_over = new FullRunApiGameOverState
				{
					run_outcome = outcome
				};
				break;
		}

		// Terminal / outcome
		if (state.run_outcome == null)
		{
			if (stateType == "game_over")
				state.run_outcome = GetString(dict, "run_outcome");
		}
		state.terminal = state.terminal || IsTerminal(stateType, state.run_outcome);

		return state;
	}

	/// <summary>Lightweight signature for change-detection polling.</summary>
	public static string Signature(FullRunApiState state)
	{
		var sb = new System.Text.StringBuilder(128);
		sb.Append(state.state_type);
		sb.Append('|'); sb.Append(state.terminal ? '1' : '0');
		sb.Append('|'); sb.Append(state.run?.floor ?? 0);
		sb.Append('|'); sb.Append(state.run_outcome ?? "");
		sb.Append('|'); sb.Append(state.legal_actions?.Count ?? 0);
		if (state.legal_actions != null)
		{
			foreach (var a in state.legal_actions)
			{
				sb.Append('|'); sb.Append(a.action);
				sb.Append(':'); sb.Append(a.index ?? -1);
				sb.Append('/'); sb.Append(a.card_index ?? -1);
				sb.Append('/'); sb.Append(a.slot ?? -1);
				sb.Append('/'); sb.Append(a.target_id ?? 0);
				sb.Append(','); sb.Append(a.col ?? -1);
				sb.Append(','); sb.Append(a.row ?? -1);
				sb.Append(','); sb.Append(a.reward_type ?? "");
				sb.Append(','); sb.Append(a.reward_key ?? "");
				sb.Append(','); sb.Append(a.label ?? "");
			}
		}
		return sb.ToString();
	}

	// ── Run info ──────────────────────────────────────────────

	private static FullRunApiRun ConvertRun(Dictionary<string, object?> state)
	{
		var runDict = GetDict(state, "run");
		var playerDict = GetDict(state, "player");
		return new FullRunApiRun
		{
			character_id = GetString(playerDict, "character") ?? GetString(runDict, "character_id"),
			seed = GetString(runDict, "seed"),
			ascension_level = GetInt(runDict, "ascension") ?? GetInt(runDict, "ascension_level") ?? 0,
			act = GetInt(runDict, "act") ?? 0,
			floor = GetInt(runDict, "floor") ?? 0,
			room_type = GetString(runDict, "room_type") ?? GetString(state, "room_type"),
			room_model_id = GetString(runDict, "room_model_id")
		};
	}

	// ── Player state ──────────────────────────────────────────

	private static FullRunApiPlayerState ConvertPlayerState(Dictionary<string, object?>? dict)
	{
		if (dict == null) return new FullRunApiPlayerState();
		var state = new FullRunApiPlayerState
		{
			character = GetString(dict, "character"),
			hp = GetInt(dict, "hp") ?? GetInt(dict, "current_hp") ?? 0,
			max_hp = GetInt(dict, "max_hp") ?? 1,
			block = GetInt(dict, "block") ?? 0,
			gold = GetInt(dict, "gold") ?? 0,
			energy = GetInt(dict, "energy") ?? 0,
			max_energy = GetInt(dict, "max_energy") ?? 0,
			draw_pile_count = GetInt(dict, "draw_pile_count") ?? 0,
			discard_pile_count = GetInt(dict, "discard_pile_count") ?? 0,
			exhaust_pile_count = GetInt(dict, "exhaust_pile_count") ?? 0,
			open_potion_slots = GetInt(dict, "open_potion_slots") ?? GetInt(dict, "potion_slots") ?? 0,
			stars = GetInt(dict, "stars"),
			orb_slots = GetInt(dict, "orb_slots"),
			orb_empty_slots = GetInt(dict, "orb_empty_slots"),
			status = GetListOfDicts(dict, "status").Select(ConvertPower).ToList(),
			hand = GetListOfDicts(dict, "hand").Select((d, i) => ConvertCard(d, i)).ToList(),
			deck = GetListOfDicts(dict, "deck").Select((d, i) => ConvertCard(d, i)).ToList(),
			relics = GetListOfDicts(dict, "relics").Select((d, i) => ConvertRelic(d, i)).ToList(),
			potions = GetListOfDicts(dict, "potions").Select(ConvertPotion).ToList()
		};

		// Orbs
		var orbsList = GetListOfDicts(dict, "orbs");
		if (orbsList.Count > 0)
		{
			state.orbs = orbsList.Select(o => new FullRunApiOrbState
			{
				id = GetString(o, "id"),
				name = GetString(o, "name"),
				passive_val = GetInt(o, "passive_val") ?? 0,
				evoke_val = GetInt(o, "evoke_val") ?? 0
			}).ToList();
		}

		// Pile contents
		var drawPile = GetListOfDicts(dict, "draw_pile");
		if (drawPile.Count > 0)
			state.draw_pile = drawPile.Select((d, i) => ConvertCard(d, i)).ToList();
		var discardPile = GetListOfDicts(dict, "discard_pile");
		if (discardPile.Count > 0)
			state.discard_pile = discardPile.Select((d, i) => ConvertCard(d, i)).ToList();
		var exhaustPile = GetListOfDicts(dict, "exhaust_pile");
		if (exhaustPile.Count > 0)
			state.exhaust_pile = exhaustPile.Select((d, i) => ConvertCard(d, i)).ToList();

		return state;
	}

	// ── Battle state ──────────────────────────────────────────

	private static FullRunApiBattleState? ConvertBattleState(Dictionary<string, object?>? dict, Dictionary<string, object?> rootState)
	{
		if (dict == null) return null;
		var battle = new FullRunApiBattleState
		{
			round = GetInt(dict, "round") ?? 0,
			turn = GetString(dict, "turn") ?? "player",
			is_play_phase = GetBool(dict, "is_play_phase"),
			player = ConvertPlayerState(GetDict(rootState, "player")),
			enemies = GetListOfDicts(dict, "enemies").Select(ConvertEnemy).ToList()
		};
		return battle;
	}

	private static FullRunApiBattleEnemy ConvertEnemy(Dictionary<string, object?> dict)
	{
		return new FullRunApiBattleEnemy
		{
			entity_id = GetString(dict, "entity_id"),
			combat_id = GetUInt(dict, "combat_id"),
			name = GetString(dict, "name"),
			hp = GetInt(dict, "hp") ?? GetInt(dict, "current_hp") ?? 0,
			max_hp = GetInt(dict, "max_hp") ?? 0,
			block = GetInt(dict, "block") ?? 0,
			is_alive = GetBool(dict, "is_alive", defaultValue: true),
			is_hittable = GetBool(dict, "is_hittable", defaultValue: true),
			next_move_id = GetString(dict, "next_move_id"),
			intends_to_attack = GetBool(dict, "intends_to_attack"),
			status = GetListOfDicts(dict, "status").Select(ConvertPower).ToList(),
			intents = GetListOfDicts(dict, "intents").Select(ConvertIntent).ToList()
		};
	}

	private static FullRunApiIntent ConvertIntent(Dictionary<string, object?> dict)
	{
		return new FullRunApiIntent
		{
			type = GetString(dict, "type")?.ToLowerInvariant(),
			label = GetString(dict, "label"),
			title = GetString(dict, "title"),
			description = GetString(dict, "description"),
			total_damage = GetInt(dict, "total_damage"),
			damage = GetInt(dict, "damage"),
			repeats = GetInt(dict, "repeats")
		};
	}

	// ── Map state ─────────────────────────────────────────────

	private static FullRunApiMapState? ConvertMapState(Dictionary<string, object?>? dict)
	{
		if (dict == null) return null;
		return new FullRunApiMapState
		{
			player = ConvertPlayerState(GetDict(dict, "player")),
			next_options = GetListOfDicts(dict, "next_options").Select((d, i) => new FullRunApiMapOption
			{
				index = GetInt(d, "index") ?? i,
				col = GetInt(d, "col") ?? 0,
				row = GetInt(d, "row") ?? 0,
				point_type = (GetString(d, "point_type") ?? GetString(d, "type"))?.ToLowerInvariant()
			}).ToList()
		};
	}

	// ── Event state ───────────────────────────────────────────

	private static FullRunApiEventState? ConvertEventState(Dictionary<string, object?>? dict)
	{
		if (dict == null) return null;
		return new FullRunApiEventState
		{
			event_id = GetString(dict, "event_id"),
			player = ConvertPlayerState(GetDict(dict, "player")),
			in_dialogue = GetBool(dict, "in_dialogue"),
			is_finished = GetBool(dict, "is_finished"),
			options = GetListOfDicts(dict, "options").Select((d, i) => new FullRunApiEventOption
			{
				index = GetInt(d, "index") ?? i,
				text = GetString(d, "text") ?? GetString(d, "title") ?? "",
				is_locked = GetBool(d, "is_locked"),
				is_chosen = GetBool(d, "is_chosen") || GetBool(d, "was_chosen"),
				is_proceed = GetBool(d, "is_proceed")
			}).ToList()
		};
	}

	// ── Shop state ────────────────────────────────────────────

	private static FullRunApiShopState? ConvertShopState(Dictionary<string, object?>? dict)
	{
		if (dict == null) return null;
		return new FullRunApiShopState
		{
			player = ConvertPlayerState(GetDict(dict, "player")),
			is_open = GetBool(dict, "is_open", defaultValue: true),
			can_proceed = GetBool(dict, "can_proceed"),
			items = GetListOfDicts(dict, "items").Select((d, i) => new FullRunApiShopItem
			{
				index = GetInt(d, "index") ?? i,
				category = GetString(d, "category"),
				cost = GetInt(d, "cost") ?? 0,
				can_afford = GetBool(d, "can_afford"),
				is_stocked = GetBool(d, "is_stocked", defaultValue: true),
				on_sale = GetBool(d, "on_sale"),
				name = GetString(d, "name"),
				description = GetString(d, "description"),
				card_id = GetString(d, "card_id"),
				card_name = GetString(d, "card_name"),
				card_type = GetString(d, "card_type"),
				card_rarity = GetString(d, "card_rarity"),
				card_description = GetString(d, "card_description"),
				relic_id = GetString(d, "relic_id"),
				relic_name = GetString(d, "relic_name"),
				relic_description = GetString(d, "relic_description"),
				potion_id = GetString(d, "potion_id"),
				potion_name = GetString(d, "potion_name"),
				potion_description = GetString(d, "potion_description"),
				keywords = GetStringList(d, "keywords")
			}).ToList()
		};
	}

	// ── Rest site state ───────────────────────────────────────

	private static FullRunApiRestSiteState? ConvertRestSiteState(Dictionary<string, object?>? dict)
	{
		if (dict == null) return null;
		return new FullRunApiRestSiteState
		{
			player = ConvertPlayerState(GetDict(dict, "player")),
			can_proceed = GetBool(dict, "can_proceed"),
			options = GetListOfDicts(dict, "options").Select((d, i) => new FullRunApiRestSiteOption
			{
				index = GetInt(d, "index") ?? i,
				id = GetString(d, "id"),
				name = GetString(d, "name"),
				description = GetString(d, "description"),
				is_enabled = GetBool(d, "is_enabled", defaultValue: true)
			}).ToList()
		};
	}

	// ── Treasure state ────────────────────────────────────────

	private static FullRunApiTreasureState? ConvertTreasureState(Dictionary<string, object?>? dict)
	{
		if (dict == null) return null;
		return new FullRunApiTreasureState
		{
			player = ConvertPlayerState(GetDict(dict, "player")),
			can_proceed = GetBool(dict, "can_proceed"),
			relics = GetListOfDicts(dict, "relics").Select((d, i) => ConvertRelic(d, i)).ToList()
		};
	}

	// ── Rewards state ─────────────────────────────────────────

	private static FullRunApiRewardsState? ConvertRewardsState(Dictionary<string, object?>? dict)
	{
		if (dict == null) return null;
		return new FullRunApiRewardsState
		{
			player = ConvertPlayerState(GetDict(dict, "player")),
			can_proceed = GetBool(dict, "can_proceed"),
			items = GetListOfDicts(dict, "items").Select((d, i) => new FullRunApiRewardItem
			{
				index = GetInt(d, "index") ?? i,
				type = GetString(d, "type"),
				label = GetString(d, "label") ?? GetString(d, "description"),
				reward_key = GetString(d, "reward_key"),
				reward_source = GetString(d, "reward_source"),
				claimable = GetBool(d, "claimable", defaultValue: true),
				claim_block_reason = GetString(d, "claim_block_reason")
			}).ToList()
		};
	}

	// ── Card reward state ─────────────────────────────────────

	private static FullRunApiCardRewardState? ConvertCardRewardState(Dictionary<string, object?>? dict)
	{
		if (dict == null) return null;
		return new FullRunApiCardRewardState
		{
			player = ConvertPlayerState(GetDict(dict, "player")),
			can_skip = GetBool(dict, "can_skip"),
			cards = GetListOfDicts(dict, "cards").Select((d, i) => ConvertCard(d, i)).ToList()
		};
	}

	// ── Card select state ─────────────────────────────────────

	private static FullRunApiCardSelectState? ConvertCardSelectState(Dictionary<string, object?>? dict)
	{
		if (dict == null) return null;
		return new FullRunApiCardSelectState
		{
			player = ConvertPlayerState(GetDict(dict, "player")),
			screen_type = GetString(dict, "screen_type"),
			prompt = GetString(dict, "prompt"),
			min_select = GetInt(dict, "min_select") ?? 0,
			max_select = GetInt(dict, "max_select") ?? 0,
			selected_count = GetInt(dict, "selected_count") ?? 0,
			remaining_picks = GetInt(dict, "remaining_picks") ?? 0,
			can_confirm = GetBool(dict, "can_confirm"),
			can_cancel = GetBool(dict, "can_cancel"),
			preview_showing = GetBool(dict, "preview_showing"),
			requires_manual_confirmation = GetBool(dict, "requires_manual_confirmation"),
			cards = GetListOfDicts(dict, "cards").Select((d, i) => ConvertCard(d, i)).ToList(),
			selected_cards = GetListOfDicts(dict, "selected_cards").Select((d, i) => ConvertCard(d, i)).ToList()
		};
	}

	// ── Relic select state ────────────────────────────────────

	private static FullRunApiRelicSelectState? ConvertRelicSelectState(Dictionary<string, object?>? dict)
	{
		if (dict == null) return null;
		return new FullRunApiRelicSelectState
		{
			player = ConvertPlayerState(GetDict(dict, "player")),
			can_skip = GetBool(dict, "can_skip"),
			relics = GetListOfDicts(dict, "relics").Select((d, i) => ConvertRelic(d, i)).ToList()
		};
	}

	// ── Hand select state ─────────────────────────────────────

	private static FullRunApiHandSelectState? ConvertHandSelectState(Dictionary<string, object?>? dict)
	{
		if (dict == null) return null;
		return new FullRunApiHandSelectState
		{
			prompt = GetString(dict, "prompt"),
			min_select = GetInt(dict, "min_select") ?? 0,
			max_select = GetInt(dict, "max_select") ?? 0,
			can_confirm = GetBool(dict, "can_confirm"),
			cards = GetListOfDicts(dict, "cards").Select((d, i) => ConvertCard(d, i)).ToList(),
			selected_cards = GetListOfDicts(dict, "selected_cards").Select((d, i) => ConvertCard(d, i)).ToList()
		};
	}

	// ── Shared converters ─────────────────────────────────────

	private static FullRunApiPower ConvertPower(Dictionary<string, object?> dict)
	{
		return new FullRunApiPower
		{
			id = GetString(dict, "id") ?? GetString(dict, "name"),
			amount = GetInt(dict, "amount") ?? 0
		};
	}

	private static FullRunApiCardOption ConvertCard(Dictionary<string, object?> dict, int fallbackIndex)
	{
		return new FullRunApiCardOption
		{
			index = GetInt(dict, "index") ?? fallbackIndex,
			id = GetString(dict, "id"),
			name = GetString(dict, "name"),
			type = GetString(dict, "type"),
			rarity = GetString(dict, "rarity"),
			cost = GetInt(dict, "cost"),
			is_upgraded = GetBool(dict, "is_upgraded"),
			can_play = GetNullableBool(dict, "can_play"),
			target_type = GetString(dict, "target_type"),
			unplayable_reason = GetString(dict, "unplayable_reason"),
			description = GetString(dict, "description"),
			valid_target_ids = GetUIntList(dict, "valid_target_ids"),
			keywords = GetStringList(dict, "keywords")
		};
	}

	private static FullRunApiRelicOption ConvertRelic(Dictionary<string, object?> dict, int fallbackIndex)
	{
		return new FullRunApiRelicOption
		{
			index = GetInt(dict, "index") ?? fallbackIndex,
			id = GetString(dict, "id"),
			name = GetString(dict, "name"),
			rarity = GetString(dict, "rarity"),
			description = GetString(dict, "description"),
			counter = GetInt(dict, "counter")
		};
	}

	private static FullRunApiPotionState ConvertPotion(Dictionary<string, object?> dict)
	{
		return new FullRunApiPotionState
		{
			slot = GetInt(dict, "slot") ?? 0,
			id = GetString(dict, "id"),
			name = GetString(dict, "name"),
			description = GetString(dict, "description"),
			target_type = GetString(dict, "target_type"),
			can_use_in_combat = GetBool(dict, "can_use_in_combat"),
			keywords = GetStringList(dict, "keywords")
		};
	}

	private static FullRunApiAction ConvertAction(Dictionary<string, object?> dict)
	{
		return new FullRunApiAction
		{
			action = GetString(dict, "action") ?? "",
			index = GetInt(dict, "index"),
			col = GetInt(dict, "col"),
			row = GetInt(dict, "row"),
			card_index = GetInt(dict, "card_index"),
			slot = GetInt(dict, "slot"),
			target_id = GetUInt(dict, "target_id"),
			target = GetString(dict, "target"),
			card_id = GetString(dict, "card_id"),
			card_type = GetString(dict, "card_type"),
			card_rarity = GetString(dict, "card_rarity"),
			cost = GetString(dict, "cost"),
			is_upgraded = GetNullableBool(dict, "is_upgraded"),
			reward_type = GetString(dict, "reward_type"),
			reward_key = GetString(dict, "reward_key"),
			reward_source = GetString(dict, "reward_source"),
			claimable = GetNullableBool(dict, "claimable"),
			claim_block_reason = GetString(dict, "claim_block_reason"),
			label = GetString(dict, "label"),
			is_enabled = GetNullableBool(dict, "is_enabled") ?? true,
			note = GetString(dict, "note")
		};
	}

	// ── Helpers ────────────────────────────────────────────────

	private static bool IsTerminal(string stateType, string? outcome)
	{
		if (stateType == "game_over") return true;
		if (outcome == null) return false;
		return outcome.Equals("victory", StringComparison.OrdinalIgnoreCase)
			|| outcome.Equals("win", StringComparison.OrdinalIgnoreCase)
			|| outcome.Equals("death", StringComparison.OrdinalIgnoreCase)
			|| outcome.Equals("loss", StringComparison.OrdinalIgnoreCase);
	}

	private static string? GetString(Dictionary<string, object?>? dict, string key)
	{
		if (dict == null) return null;
		return dict.TryGetValue(key, out var val) ? val?.ToString() : null;
	}

	private static int? GetInt(Dictionary<string, object?>? dict, string key)
	{
		if (dict == null) return null;
		if (!dict.TryGetValue(key, out var val) || val == null) return null;
		if (val is int i) return i;
		if (val is long l) return (int)l;
		if (val is double d) return (int)d;
		if (val is float f) return (int)f;
		if (val is decimal m) return (int)m;
		if (int.TryParse(val.ToString(), out int parsed)) return parsed;
		return null;
	}

	private static uint? GetUInt(Dictionary<string, object?>? dict, string key)
	{
		if (dict == null) return null;
		if (!dict.TryGetValue(key, out var val) || val == null) return null;
		if (val is uint u) return u;
		if (val is int i) return (uint)i;
		if (val is long l) return (uint)l;
		if (uint.TryParse(val.ToString(), out uint parsed)) return parsed;
		return null;
	}

	private static bool GetBool(Dictionary<string, object?>? dict, string key, bool defaultValue = false)
	{
		if (dict == null) return defaultValue;
		if (!dict.TryGetValue(key, out var val) || val == null) return defaultValue;
		if (val is bool b) return b;
		return defaultValue;
	}

	private static bool? GetNullableBool(Dictionary<string, object?>? dict, string key)
	{
		if (dict == null) return null;
		if (!dict.TryGetValue(key, out var val) || val == null) return null;
		if (val is bool b) return b;
		return null;
	}

	private static Dictionary<string, object?>? GetDict(Dictionary<string, object?>? dict, string key)
	{
		if (dict == null) return null;
		if (!dict.TryGetValue(key, out var val)) return null;
		return val as Dictionary<string, object?>;
	}

	private static List<Dictionary<string, object?>> GetListOfDicts(Dictionary<string, object?>? dict, string key)
	{
		if (dict == null) return new List<Dictionary<string, object?>>();
		if (!dict.TryGetValue(key, out var val) || val == null) return new List<Dictionary<string, object?>>();
		if (val is IEnumerable<object?> enumerable)
			return enumerable.OfType<Dictionary<string, object?>>().ToList();
		if (val is System.Collections.IEnumerable ie)
		{
			var result = new List<Dictionary<string, object?>>();
			foreach (var item in ie)
			{
				if (item is Dictionary<string, object?> d)
					result.Add(d);
			}
			return result;
		}
		return new List<Dictionary<string, object?>>();
	}

	private static List<string> GetStringList(Dictionary<string, object?>? dict, string key)
	{
		if (dict == null) return new List<string>();
		if (!dict.TryGetValue(key, out var val) || val == null) return new List<string>();
		if (val is IEnumerable<object?> enumerable)
			return enumerable.Where(x => x != null).Select(x => x!.ToString()!).ToList();
		if (val is IEnumerable<string> strings)
			return strings.ToList();
		return new List<string>();
	}

	private static List<uint> GetUIntList(Dictionary<string, object?>? dict, string key)
	{
		if (dict == null) return new List<uint>();
		if (!dict.TryGetValue(key, out var val) || val == null) return new List<uint>();
		if (val is IEnumerable<uint> uints)
			return uints.ToList();
		if (val is IEnumerable<object?> enumerable)
		{
			var result = new List<uint>();
			foreach (var item in enumerable)
			{
				if (item is uint u) result.Add(u);
				else if (item is int i) result.Add((uint)i);
				else if (item is long l) result.Add((uint)l);
			}
			return result;
		}
		return new List<uint>();
	}
}
