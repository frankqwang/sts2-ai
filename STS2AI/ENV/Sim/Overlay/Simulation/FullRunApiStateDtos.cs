using System.Collections.Generic;

namespace MegaCrit.Sts2.Core.Simulation;

public sealed class FullRunApiAction
{
	public string action { get; set; } = string.Empty;

	public int? index { get; set; }

	public int? col { get; set; }

	public int? row { get; set; }

	public int? card_index { get; set; }

	public int? slot { get; set; }

	public uint? target_id { get; set; }

	public string? target { get; set; }

	public string? card_id { get; set; }

	public string? card_type { get; set; }

	public string? card_rarity { get; set; }

	public string? cost { get; set; }

	public bool? is_upgraded { get; set; }

	public string? reward_type { get; set; }

	public string? reward_key { get; set; }

	public string? reward_source { get; set; }

	public bool? claimable { get; set; }

	public string? claim_block_reason { get; set; }

	public string? label { get; set; }

	public bool? is_enabled { get; set; }

	public string? note { get; set; }
}

public sealed class FullRunApiRun
{
	public string? character_id { get; set; }

	public string? seed { get; set; }

	public int ascension_level { get; set; }

	public int ascension => ascension_level;

	public int act { get; set; }

	public int floor { get; set; }

	public string? room_type { get; set; }

	public string? room_model_id { get; set; }
}

public sealed class FullRunApiPower
{
	public string? id { get; set; }

	public int amount { get; set; }
}

public sealed class FullRunApiIntent
{
	public string? type { get; set; }

	public string? label { get; set; }

	public string? title { get; set; }

	public string? description { get; set; }

	public int? total_damage { get; set; }

	// 2026-04-08: forward per-hit damage and repeat count so the
	// Python trainer can build accurate per-attack features for multi-hit intents.
	// Field renamed `base_damage` -> `damage` 2026-04-08 PM to align with the binary
	// pipe wire format and the encoder's `intent.get("damage", ...)` read order.
	public int? damage { get; set; }

	public int? repeats { get; set; }
}

public sealed class FullRunApiCardOption
{
	public int index { get; set; }

	public string? id { get; set; }

	public string? name { get; set; }

	public string? type { get; set; }

	public string? rarity { get; set; }

	public int? cost { get; set; }

	public bool is_upgraded { get; set; }

	public bool? can_play { get; set; }

	public string? target_type { get; set; }

	public string? unplayable_reason { get; set; }

	public string? description { get; set; }

	public List<uint> valid_target_ids { get; set; } = new List<uint>();

	public List<string> keywords { get; set; } = new List<string>();
}

public sealed class FullRunApiPotionState
{
	public int slot { get; set; }

	public string? id { get; set; }

	public string? name { get; set; }

	public string? description { get; set; }

	public string? target_type { get; set; }

	public bool can_use_in_combat { get; set; }

	public List<string> keywords { get; set; } = new List<string>();
}

public sealed class FullRunApiRelicOption
{
	public int index { get; set; }

	public string? id { get; set; }

	public string? name { get; set; }

	public string? rarity { get; set; }

	public string? description { get; set; }
}

public sealed class FullRunApiPlayerState
{
	public string? character { get; set; }

	public int hp { get; set; }

	public int current_hp => hp;

	public int max_hp { get; set; }

	public int block { get; set; }

	public int gold { get; set; }

	public int energy { get; set; }

	public int max_energy { get; set; }

	public int draw_pile_count { get; set; }

	public int discard_pile_count { get; set; }

	public int exhaust_pile_count { get; set; }

	public int open_potion_slots { get; set; }

	public List<FullRunApiPower> status { get; set; } = new List<FullRunApiPower>();

	public List<FullRunApiCardOption> hand { get; set; } = new List<FullRunApiCardOption>();

	public List<FullRunApiCardOption> deck { get; set; } = new List<FullRunApiCardOption>();

	public List<FullRunApiRelicOption> relics { get; set; } = new List<FullRunApiRelicOption>();

	public List<FullRunApiPotionState> potions { get; set; } = new List<FullRunApiPotionState>();
}

public sealed class FullRunApiBattleEnemy
{
	public string? entity_id { get; set; }

	public uint? combat_id { get; set; }

	public string? name { get; set; }

	public int hp { get; set; }

	public int max_hp { get; set; }

	public int block { get; set; }

	public bool is_alive { get; set; }

	// True iff this creature is targetable by single-target player cards. Boss
	// summons or invulnerable shields can be alive but not hittable; without
	// this flag the AI wastes actions trying to attack them.
	public bool is_hittable { get; set; }

	// Boss move/phase id from CombatTrainingCreatureSnapshot.NextMoveId.
	// Critical for multi-phase bosses (Vantom: InkBlot/InkyLance/Dismember/Prepare,
	// Slime Boss: split phases). Was missing pre-2026-04-08.
	public string? next_move_id { get; set; }

	public bool intends_to_attack { get; set; }

	public List<FullRunApiPower> status { get; set; } = new List<FullRunApiPower>();

	public List<FullRunApiIntent> intents { get; set; } = new List<FullRunApiIntent>();
}

public sealed class FullRunApiBattleState
{
	public int round { get; set; }

	public string? turn { get; set; }

	public bool is_play_phase { get; set; }

	public FullRunApiPlayerState player { get; set; } = new FullRunApiPlayerState();

	public List<FullRunApiBattleEnemy> enemies { get; set; } = new List<FullRunApiBattleEnemy>();

	public FullRunApiCombatCardSelectionState? card_selection { get; set; }
}

public sealed class FullRunApiHandSelectState
{
	public string? prompt { get; set; }

	public int min_select { get; set; }

	public int max_select { get; set; }

	public bool can_confirm { get; set; }

	public List<FullRunApiCardOption> cards { get; set; } = new List<FullRunApiCardOption>();

	public List<FullRunApiCardOption> selected_cards { get; set; } = new List<FullRunApiCardOption>();
}

public sealed class FullRunApiCombatCardSelectionState
{
	public string? prompt { get; set; }

	public string? mode { get; set; }

	public int min_select { get; set; }

	public int max_select { get; set; }

	public bool can_confirm { get; set; }

	public bool can_cancel { get; set; }

	public List<FullRunApiCardOption> selectable_cards { get; set; } = new List<FullRunApiCardOption>();

	public List<FullRunApiCardOption> selected_cards { get; set; } = new List<FullRunApiCardOption>();
}

public sealed class FullRunApiMapOption
{
	public int index { get; set; }

	public int col { get; set; }

	public int row { get; set; }

	public string? point_type { get; set; }
}

public sealed class FullRunApiMapState
{
	public FullRunApiPlayerState player { get; set; } = new FullRunApiPlayerState();

	public List<FullRunApiMapOption> next_options { get; set; } = new List<FullRunApiMapOption>();
}

public sealed class FullRunApiEventOption
{
	public int index { get; set; }

	public string? text { get; set; }

	public bool is_locked { get; set; }

	public bool is_chosen { get; set; }

	public bool is_proceed { get; set; }
}

public sealed class FullRunApiEventState
{
	public string? event_id { get; set; }

	public FullRunApiPlayerState player { get; set; } = new FullRunApiPlayerState();

	public bool in_dialogue { get; set; }

	public bool is_finished { get; set; }

	public List<FullRunApiEventOption> options { get; set; } = new List<FullRunApiEventOption>();
}

public sealed class FullRunApiRewardItem
{
	public int index { get; set; }

	public string? type { get; set; }

	public string? label { get; set; }

	public string? reward_key { get; set; }

	public string? reward_source { get; set; }

	public bool claimable { get; set; } = true;

	public string? claim_block_reason { get; set; }
}

public sealed class FullRunApiRewardsState
{
	public FullRunApiPlayerState player { get; set; } = new FullRunApiPlayerState();

	public bool can_proceed { get; set; }

	public List<FullRunApiRewardItem> items { get; set; } = new List<FullRunApiRewardItem>();
}

public sealed class FullRunApiCardRewardState
{
	public FullRunApiPlayerState player { get; set; } = new FullRunApiPlayerState();

	public bool can_skip { get; set; }

	public List<FullRunApiCardOption> cards { get; set; } = new List<FullRunApiCardOption>();
}

public sealed class FullRunApiRelicSelectState
{
	public FullRunApiPlayerState player { get; set; } = new FullRunApiPlayerState();

	public bool can_skip { get; set; }

	public List<FullRunApiRelicOption> relics { get; set; } = new List<FullRunApiRelicOption>();
}

public sealed class FullRunApiRestSiteOption
{
	public int index { get; set; }

	public string? id { get; set; }

	public string? name { get; set; }

	public string? description { get; set; }

	public bool is_enabled { get; set; }
}

public sealed class FullRunApiRestSiteState
{
	public FullRunApiPlayerState player { get; set; } = new FullRunApiPlayerState();

	public bool can_proceed { get; set; }

	public List<FullRunApiRestSiteOption> options { get; set; } = new List<FullRunApiRestSiteOption>();
}

public sealed class FullRunApiShopItem
{
	public int index { get; set; }

	public string? category { get; set; }

	public int cost { get; set; }

	public bool can_afford { get; set; }

	public bool is_stocked { get; set; }

	public bool on_sale { get; set; }

	public string? name { get; set; }

	public string? description { get; set; }

	public string? card_id { get; set; }

	public string? card_name { get; set; }

	public string? card_type { get; set; }

	public string? card_rarity { get; set; }

	public string? card_description { get; set; }

	public string? relic_id { get; set; }

	public string? relic_name { get; set; }

	public string? relic_description { get; set; }

	public string? potion_id { get; set; }

	public string? potion_name { get; set; }

	public string? potion_description { get; set; }

	public List<string> keywords { get; set; } = new List<string>();
}

public sealed class FullRunApiShopState
{
	public FullRunApiPlayerState player { get; set; } = new FullRunApiPlayerState();

	public bool is_open { get; set; }

	public bool can_proceed { get; set; }

	public List<FullRunApiShopItem> items { get; set; } = new List<FullRunApiShopItem>();
}

public sealed class FullRunApiTreasureState
{
	public FullRunApiPlayerState player { get; set; } = new FullRunApiPlayerState();

	public bool can_proceed { get; set; }

	public List<FullRunApiRelicOption> relics { get; set; } = new List<FullRunApiRelicOption>();
}

public sealed class FullRunApiCardSelectState
{
	public FullRunApiPlayerState player { get; set; } = new FullRunApiPlayerState();

	public string? screen_type { get; set; }

	public string? prompt { get; set; }

	public int min_select { get; set; }

	public int max_select { get; set; }

	public int selected_count { get; set; }

	public int remaining_picks { get; set; }

	public bool can_confirm { get; set; }

	public bool can_cancel { get; set; }

	public bool preview_showing { get; set; }

	public bool requires_manual_confirmation { get; set; }

	public List<FullRunApiCardOption> cards { get; set; } = new List<FullRunApiCardOption>();

	public List<FullRunApiCardOption> selected_cards { get; set; } = new List<FullRunApiCardOption>();
}

public sealed class FullRunApiMenuState
{
	public List<FullRunApiAction> actions { get; set; } = new List<FullRunApiAction>();
}

public sealed class FullRunApiGameOverState
{
	public string? run_outcome { get; set; }

	public List<FullRunApiAction> available_actions { get; set; } = new List<FullRunApiAction>();
}

public sealed class FullRunApiState
{
	public string state_type { get; set; } = "menu";

	public bool terminal { get; set; }

	public bool is_pure_simulator { get; set; }

	public string? backend_kind { get; set; }

	public string? coverage_tier { get; set; }

	public string? run_outcome { get; set; }

	public FullRunApiRun run { get; set; } = new FullRunApiRun();

	public List<FullRunApiAction> legal_actions { get; set; } = new List<FullRunApiAction>();

	public FullRunApiBattleState? battle { get; set; }

	public FullRunApiHandSelectState? hand_select { get; set; }

	public FullRunApiCombatCardSelectionState? card_selection { get; set; }

	public FullRunApiMenuState? menu { get; set; }

	public FullRunApiMapState? map { get; set; }

	public FullRunApiEventState? @event { get; set; }

	public FullRunApiRestSiteState? rest_site { get; set; }

	public FullRunApiShopState? shop { get; set; }

	public FullRunApiTreasureState? treasure { get; set; }

	public FullRunApiRewardsState? rewards { get; set; }

	public FullRunApiCardRewardState? card_reward { get; set; }

	public FullRunApiCardSelectState? card_select { get; set; }

	public FullRunApiRelicSelectState? relic_select { get; set; }

	public FullRunApiGameOverState? game_over { get; set; }
}
