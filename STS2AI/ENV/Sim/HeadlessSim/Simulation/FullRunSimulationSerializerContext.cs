using System.Text.Json.Serialization;

namespace MegaCrit.Sts2.Core.Simulation;

[JsonSourceGenerationOptions(
	WriteIndented = true,
	IncludeFields = true,
	UseStringEnumConverter = true,
	PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
[JsonSerializable(typeof(FullRunExportedRunSnapshot))]
[JsonSerializable(typeof(FullRunPendingSelectionRestoreSnapshot))]
[JsonSerializable(typeof(FullRunPendingRewardSelectionRestoreSnapshot))]
[JsonSerializable(typeof(FullRunPendingCardRewardRestoreSnapshot))]
[JsonSerializable(typeof(FullRunPendingCombatCardSelectionRestoreSnapshot))]
[JsonSerializable(typeof(FullRunPendingRewardRestoreEntrySnapshot))]
[JsonSerializable(typeof(FullRunSimulatorRuntimeFacade.SavedCombatSnapshot))]
[JsonSerializable(typeof(FullRunSimulatorRuntimeFacade.SavedCombatEncounterMonsterSnapshot))]
[JsonSerializable(typeof(FullRunSimulatorRuntimeFacade.SavedCombatPlayerSnapshot))]
[JsonSerializable(typeof(FullRunSimulatorRuntimeFacade.SavedCombatMonsterMoveSnapshot))]
[JsonSerializable(typeof(FullRunSimulatorRuntimeFacade.SavedShopSnapshot))]
[JsonSerializable(typeof(FullRunSimulatorRuntimeFacade.SavedShopCardEntry))]
[JsonSerializable(typeof(FullRunSimulatorRuntimeFacade.SavedShopRelicEntry))]
[JsonSerializable(typeof(FullRunSimulatorRuntimeFacade.SavedShopPotionEntry))]
[JsonSerializable(typeof(FullRunSimulatorRuntimeFacade.SavedShopCardRemovalEntry))]
[JsonSerializable(typeof(FullRunSimulatorRuntimeFacade.SavedTreasureSnapshot))]
internal partial class FullRunSimulationSerializerContext : JsonSerializerContext
{
}
