using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Simulation;

public sealed class FullRunExportedRunSnapshot
{
	public string ExportedAtUtc { get; set; } = string.Empty;

	public string SourceStateId { get; set; } = string.Empty;

	public SerializableRun RunSnapshot { get; set; } = null!;

	public string ExactSignature { get; set; } = string.Empty;

	public string StateType { get; set; } = string.Empty;

	public FullRunPendingSelectionRestoreSnapshot? PendingSelection { get; set; }

	public FullRunSimulatorRuntimeFacade.SavedCombatSnapshot? CombatSnapshot { get; set; }

	public FullRunSimulatorRuntimeFacade.SavedShopSnapshot? ShopSnapshot { get; set; }

	public FullRunSimulatorRuntimeFacade.SavedTreasureSnapshot? TreasureSnapshot { get; set; }
}
