using System.Collections.Generic;
using System.Text.Json.Serialization;
using MegaCrit.Sts2.Core.Saves.Runs;

namespace MegaCrit.Sts2.Core.Simulation;

public sealed class SimulationBuildSpec
{
	[JsonPropertyName("deck")]
	public List<SimulationBuildCardSpec>? Deck { get; set; }

	[JsonPropertyName("relics")]
	public List<SimulationBuildRelicSpec>? Relics { get; set; }

	[JsonPropertyName("potions")]
	public List<SimulationBuildPotionSpec>? Potions { get; set; }

	[JsonPropertyName("current_hp")]
	public int? CurrentHp { get; set; }

	[JsonPropertyName("max_hp")]
	public int? MaxHp { get; set; }

	[JsonPropertyName("max_energy")]
	public int? MaxEnergy { get; set; }

	[JsonPropertyName("max_potion_slots")]
	public int? MaxPotionSlots { get; set; }

	[JsonPropertyName("gold")]
	public int? Gold { get; set; }
}

public sealed class SimulationBuildCardSpec
{
	[JsonPropertyName("id")]
	public string? Id { get; set; }

	[JsonPropertyName("upgrade_level")]
	public int? UpgradeLevel { get; set; }

	[JsonPropertyName("floor_added_to_deck")]
	public int? FloorAddedToDeck { get; set; }

	[JsonPropertyName("props")]
	public SavedProperties? Props { get; set; }
}

public sealed class SimulationBuildRelicSpec
{
	[JsonPropertyName("id")]
	public string? Id { get; set; }

	[JsonPropertyName("floor_added_to_deck")]
	public int? FloorAddedToDeck { get; set; }
}

public sealed class SimulationBuildPotionSpec
{
	[JsonPropertyName("id")]
	public string? Id { get; set; }

	[JsonPropertyName("slot")]
	public int? Slot { get; set; }

	[JsonPropertyName("slot_index")]
	public int? SlotIndex { get; set; }
}
