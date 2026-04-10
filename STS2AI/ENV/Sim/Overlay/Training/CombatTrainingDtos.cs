using System.Collections.Generic;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Models;

namespace MegaCrit.Sts2.Core.Training;

public sealed class CombatTrainingStepResult
{
	public bool Accepted { get; set; }

	public string? Error { get; set; }

	public CombatTrainingStateSnapshot? State { get; set; }
}

public sealed class CombatTrainingStateSnapshot
{
	public bool IsTrainerActive { get; set; }

	public bool IsPureSimulator { get; set; }

	public string ChoiceAdapterKind { get; set; } = "";

	public bool IsCombatActive { get; set; }

	public bool IsEpisodeDone { get; set; }

	public bool? Victory { get; set; }

	public int EpisodeNumber { get; set; }

	public string? Seed { get; set; }

	public string? CharacterId { get; set; }

	public string? EncounterId { get; set; }

	public int AscensionLevel { get; set; }

	public int RoundNumber { get; set; }

	public CombatSide CurrentSide { get; set; }

	public bool IsPlayPhase { get; set; }

	public bool PlayerActionsDisabled { get; set; }

	public bool IsActionQueueRunning { get; set; }

	public bool IsHandSelectionActive { get; set; }

	public bool IsCardSelectionActive { get; set; }

	public bool CanEndTurn { get; set; }

	public CombatTrainingPlayerSnapshot? Player { get; set; }

	public List<CombatTrainingCreatureSnapshot> Enemies { get; set; } = new List<CombatTrainingCreatureSnapshot>();

	public List<CombatTrainingHandCardSnapshot> Hand { get; set; } = new List<CombatTrainingHandCardSnapshot>();

	public CombatTrainingPileSnapshot Piles { get; set; } = new CombatTrainingPileSnapshot();

	public CombatTrainingHandSelectionSnapshot? HandSelection { get; set; }

	public CombatTrainingCardSelectionSnapshot? CardSelection { get; set; }
}

public sealed class CombatTrainingPlayerSnapshot
{
	public ulong NetId { get; set; }

	public uint? CombatId { get; set; }

	public int CurrentHp { get; set; }

	public int MaxHp { get; set; }

	public int Block { get; set; }

	public int Energy { get; set; }

	public int MaxEnergy { get; set; }

	public int Stars { get; set; }

	public List<CombatTrainingPowerSnapshot> Powers { get; set; } = new List<CombatTrainingPowerSnapshot>();
}

public sealed class CombatTrainingCreatureSnapshot
{
	public uint? CombatId { get; set; }

	public string? Id { get; set; }

	public string Name { get; set; } = "";

	public int CurrentHp { get; set; }

	public int MaxHp { get; set; }

	public int Block { get; set; }

	public bool IsAlive { get; set; }

	public bool IsHittable { get; set; }

	public string? NextMoveId { get; set; }

	public bool IntendsToAttack { get; set; }

	public List<CombatTrainingIntentSnapshot> Intents { get; set; } = new List<CombatTrainingIntentSnapshot>();

	public List<CombatTrainingPowerSnapshot> Powers { get; set; } = new List<CombatTrainingPowerSnapshot>();
}

public sealed class CombatTrainingIntentSnapshot
{
	public string IntentType { get; set; } = "";

	public int Repeats { get; set; }

	public int? Damage { get; set; }

	public int? TotalDamage { get; set; }
}

public sealed class CombatTrainingPowerSnapshot
{
	public string Id { get; set; } = "";

	public int Amount { get; set; }
}

public sealed class CombatTrainingHandCardSnapshot
{
	public int HandIndex { get; set; }

	public uint CombatCardIndex { get; set; }

	public string Id { get; set; } = "";

	public string Title { get; set; } = "";

	public int EnergyCost { get; set; }

	public bool CostsX { get; set; }

	public int StarCost { get; set; }

	public TargetType TargetType { get; set; }

	public bool CanPlay { get; set; }

	public bool RequiresTarget { get; set; }

	public List<uint> ValidTargetIds { get; set; } = new List<uint>();

	public string CardType { get; set; } = "";

	public string? Description { get; set; }

	public bool IsUpgraded { get; set; }

	public bool GainsBlock { get; set; }

	public List<string> Keywords { get; set; } = new List<string>();
}

public sealed class CombatTrainingPileSnapshot
{
	public int Draw { get; set; }

	public int Discard { get; set; }

	public int Exhaust { get; set; }

	public int Play { get; set; }

	/// <summary>Card IDs in draw pile (order is hidden from player but contents are known).</summary>
	public List<string> DrawCardIds { get; set; } = new();

	/// <summary>Card IDs in discard pile (publicly visible).</summary>
	public List<string> DiscardCardIds { get; set; } = new();

	/// <summary>Card IDs in exhaust pile (publicly visible).</summary>
	public List<string> ExhaustCardIds { get; set; } = new();
}

public sealed class CombatTrainingHandSelectionSnapshot
{
	public string ChoiceAdapterKind { get; set; } = "";

	public bool IsBackendAvailable { get; set; } = true;

	public string Mode { get; set; } = "";

	public string PromptText { get; set; } = "";

	public int MinSelect { get; set; }

	public int MaxSelect { get; set; }

	public bool CanConfirm { get; set; }

	public bool Cancelable { get; set; }

	public List<CombatTrainingHandCardSnapshot> SelectableCards { get; set; } = new List<CombatTrainingHandCardSnapshot>();

	public List<CombatTrainingHandCardSnapshot> SelectedCards { get; set; } = new List<CombatTrainingHandCardSnapshot>();
}

public sealed class CombatTrainingCardSelectionSnapshot
{
	public string ChoiceAdapterKind { get; set; } = "";

	public bool IsBackendAvailable { get; set; } = true;

	public string Mode { get; set; } = "";

	public string PromptText { get; set; } = "";

	public int MinSelect { get; set; }

	public int MaxSelect { get; set; }

	public bool CanConfirm { get; set; }

	public bool Cancelable { get; set; }

	public List<CombatTrainingSelectableCardSnapshot> SelectableCards { get; set; } = new List<CombatTrainingSelectableCardSnapshot>();

	public List<CombatTrainingSelectableCardSnapshot> SelectedCards { get; set; } = new List<CombatTrainingSelectableCardSnapshot>();
}

public sealed class CombatTrainingSelectableCardSnapshot
{
	public int ChoiceIndex { get; set; }

	public string Id { get; set; } = "";

	public string Title { get; set; } = "";

	public CardType Type { get; set; }

	public int EnergyCost { get; set; }

	public bool CostsX { get; set; }

	public int StarCost { get; set; }

	public TargetType TargetType { get; set; }

	public string SourcePile { get; set; } = "";

	public bool IsUpgraded { get; set; }

	public bool IsUpgradable { get; set; }
}
