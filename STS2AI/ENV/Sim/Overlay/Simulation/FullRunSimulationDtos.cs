using System.Collections.Generic;
using MegaCrit.Sts2.Core.Training;

namespace MegaCrit.Sts2.Core.Simulation;

public sealed class FullRunSimulationResetRequest
{
	public string? CharacterId { get; set; }

	public string? Character { get; set; }

	public string? Seed { get; set; }

	public int? AscensionLevel { get; set; }

	public int? Ascension { get; set; }

	public int? TimeoutMs { get; set; }
}

public sealed class FullRunSimulationActionRequest
{
	public string Type { get; set; } = string.Empty;

	public string Action { get; set; } = string.Empty;

	public int? Index { get; set; }

	public int? Col { get; set; }

	public int? Row { get; set; }

	public string? Value { get; set; }

	public int? CardIndex { get; set; }

	public int? HandIndex { get; set; }

	public int? Slot { get; set; }

	public uint? TargetId { get; set; }

	public string? Target { get; set; }
}

public sealed class FullRunSimulationLegalAction
{
	public string Action { get; set; } = string.Empty;

	public int? Index { get; set; }

	public int? Col { get; set; }

	public int? Row { get; set; }

	public string? Label { get; set; }

	public int? CardIndex { get; set; }

	public int? Slot { get; set; }

	public uint? TargetId { get; set; }

	public string? Target { get; set; }

	public string? CardId { get; set; }

	public string? CardType { get; set; }

	public string? CardRarity { get; set; }

	public string? Cost { get; set; }

	public bool? IsUpgraded { get; set; }

	public string? RewardType { get; set; }

	public string? RewardKey { get; set; }

	public string? RewardSource { get; set; }

	public bool? Claimable { get; set; }

	public string? ClaimBlockReason { get; set; }

	public bool IsSupported { get; set; }

	public string? Note { get; set; }
}

public sealed class FullRunSimulationMapOption
{
	public int Index { get; set; }

	public int Col { get; set; }

	public int Row { get; set; }

	public string? PointType { get; set; }
}

public sealed class FullRunSimulationMapNode
{
	public int Col { get; set; }

	public int Row { get; set; }

	public string PointType { get; set; } = "unknown";

	public List<(int Col, int Row)> Children { get; set; } = new();
}

public sealed class FullRunSimulationStateSnapshot
{
	public bool IsRunActive { get; set; }

	public bool IsPureSimulator { get; set; }

	public bool IsTerminal { get; set; }

	public bool IsActionable { get; set; }

	public string StateType { get; set; } = "menu";

	public string? RoomType { get; set; }

	public string? RoomModelId { get; set; }

	public string BackendKind { get; set; } = "full_run_runtime_seam";

	public string CoverageTier { get; set; } = "skeleton";

	public string? CharacterId { get; set; }

	public string? Seed { get; set; }

	public int AscensionLevel { get; set; }

	public int CurrentActIndex { get; set; }

	public int ActFloor { get; set; }

	public int TotalFloor { get; set; }

	public string? RunOutcome { get; set; }

	public List<FullRunSimulationLegalAction> LegalActions { get; set; } = new List<FullRunSimulationLegalAction>();

	public List<FullRunSimulationMapOption> MapOptions { get; set; } = new List<FullRunSimulationMapOption>();

	public List<FullRunSimulationMapNode> MapNodes { get; set; } = new List<FullRunSimulationMapNode>();

	public int BossCol { get; set; } = -1;

	public int BossRow { get; set; } = -1;

	internal CombatTrainingStateSnapshot? CachedCombatState { get; set; }

	internal FullRunChoiceBridgeSnapshotCache? CachedBridgeSnapshots { get; set; }
}

public sealed class FullRunSimulationStepResult
{
	public bool Accepted { get; set; }

	public string? Error { get; set; }

	public string? FailureCode { get; set; }

	public FullRunSimulationStateSnapshot? State { get; set; }
}

public sealed class FullRunSimulationBatchStepResult
{
	public bool Accepted { get; set; }

	public string? Error { get; set; }

	public string? FailureCode { get; set; }

	public int StepsExecuted { get; set; }

	public FullRunSimulationStateSnapshot? State { get; set; }
}
