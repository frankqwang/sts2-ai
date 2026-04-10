using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Simulation;

namespace MegaCrit.Sts2.Core.Training;

public sealed class CombatTrainingEnvService
{
	public static CombatTrainingEnvService Instance { get; } = new CombatTrainingEnvService();

	private static readonly ICombatRuntimeFacade SimulatorRuntime = new CombatSimulatorRuntimeFacade();

	// Pre-allocated empty lists to avoid per-call allocations in hot path.
	private static readonly List<uint> EmptyUintList = new List<uint>();
	private static readonly List<string> EmptyStringList = new List<string>();
	private static readonly List<CombatTrainingIntentSnapshot> EmptyIntentList = new List<CombatTrainingIntentSnapshot>();

	private static bool UseTrainerBackend => CombatTrainingMode.IsActive && !CombatSimulationMode.IsServerActive;

	private CombatTrainingEnvService()
	{
	}

	public async Task<CombatTrainingStateSnapshot> ResetAsync(CombatTrainingResetRequest? request = null)
	{
		EnsureEnvIsAvailable();
		if (UseTrainerBackend)
		{
			CombatTrainingSession combatTrainingSession = GetSession();
			await combatTrainingSession.ResetAsync(request);
			return await WaitForSettledAndSnapshotAsync();
		}
		return await SimulatorRuntime.ResetAsync(request);
	}

	public CombatTrainingStateSnapshot GetState()
	{
		EnsureEnvIsAvailable();
		if (UseTrainerBackend)
		{
			return BuildStateSnapshot();
		}
		return SimulatorRuntime.GetState();
	}

	public async Task<CombatTrainingStepResult> StepAsync(CombatTrainingActionRequest action)
	{
		EnsureEnvIsAvailable();
		if (!UseTrainerBackend)
		{
			return await SimulatorRuntime.StepAsync(action);
		}
		return await StepAgainstActiveCombatAsync(action);
	}

	internal static async Task<CombatTrainingStepResult> StepAgainstActiveCombatAsync(CombatTrainingActionRequest action)
	{
		if (action == null)
		{
			return BuildRejectedResult("Action request is required.");
		}
		CombatState? combatState = CombatManager.Instance.DebugOnlyGetState();
		if (combatState == null)
		{
			return BuildRejectedResult("Combat state is not initialized.");
		}
		if (!CombatManager.Instance.IsInProgress)
		{
			return BuildRejectedResult("Combat is already over.");
		}
		ICombatChoiceAdapter choiceAdapter = GetChoiceAdapter();
		Player player = LocalContext.GetMe(combatState);
		switch (action.Type)
		{
		case CombatTrainingActionType.PlayCard:
			{
				if (choiceAdapter.IsSelectionActive)
				{
					return BuildRejectedResult("Cannot play cards while a hand selection prompt is active.");
				}
				CombatTrainingStepResult? validationFailure = TryValidatePlayCardRequest(player, combatState, action, out CardModel card, out Creature? target);
				if (validationFailure != null)
				{
					return validationFailure;
				}
				RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new PlayCardAction(card, target));
				break;
			}
		case CombatTrainingActionType.EndTurn:
			{
				if (choiceAdapter.IsSelectionActive)
				{
					return BuildRejectedResult("Cannot end turn while a hand selection prompt is active.");
				}
				if (!CombatManager.Instance.IsPlayPhase)
				{
					return BuildRejectedResult("Cannot end turn outside of the player play phase.");
				}
				if (CombatManager.Instance.IsPlayerReadyToEndTurn(player))
				{
					return BuildRejectedResult("End turn has already been requested for this turn.");
				}
				RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new EndPlayerTurnAction(player, combatState.RoundNumber));
				break;
			}
		case CombatTrainingActionType.UsePotion:
			{
				if (choiceAdapter.IsSelectionActive)
				{
					return BuildRejectedResult("Cannot use potions while a card selection prompt is active.");
				}
				CombatTrainingStepResult? validationFailure = TryValidateUsePotionRequest(player, combatState, action, out int slot, out Creature? target);
				if (validationFailure != null)
				{
					return validationFailure;
				}
				RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new UsePotionAction(player, (uint)slot, target?.CombatId, target?.Player?.NetId, CombatManager.Instance.IsInProgress));
				break;
			}
		case CombatTrainingActionType.SelectHandCard:
			{
				CombatTrainingStepResult? validationFailure = TryValidateSelectionRequest(action, choiceAdapter, requireHandSelection: true);
				if (validationFailure != null)
				{
					return validationFailure;
				}
				if (!action.HandIndex.HasValue)
				{
					return BuildRejectedResult("SelectHandCard requires hand_index.");
				}
				if (!choiceAdapter.TrySelectHandCard(action.HandIndex.Value, out string error))
				{
					return BuildRejectedResult(error);
				}
				break;
			}
		case CombatTrainingActionType.SelectCardChoice:
			{
				CombatTrainingStepResult? validationFailure = TryValidateSelectionRequest(action, choiceAdapter, requireHandSelection: false);
				if (validationFailure != null)
				{
					return validationFailure;
				}
				if (!action.ChoiceIndex.HasValue)
				{
					return BuildRejectedResult("SelectCardChoice requires choice_index.");
				}
				if (!choiceAdapter.TrySelectCardChoice(action.ChoiceIndex.Value, out string error))
				{
					return BuildRejectedResult(error);
				}
				break;
			}
		case CombatTrainingActionType.ConfirmSelection:
			{
				CombatTrainingStepResult? validationFailure = TryValidateSelectionRequest(action, choiceAdapter, requireHandSelection: null);
				if (validationFailure != null)
				{
					return validationFailure;
				}
				if (!choiceAdapter.TryConfirmSelection(out string error))
				{
					return BuildRejectedResult(error);
				}
				break;
			}
		case CombatTrainingActionType.CancelSelection:
			{
				CombatTrainingStepResult? validationFailure = TryValidateSelectionRequest(action, choiceAdapter, requireHandSelection: null);
				if (validationFailure != null)
				{
					return validationFailure;
				}
				if (!choiceAdapter.TryCancelSelection(out string error))
				{
					return BuildRejectedResult(error);
				}
				break;
			}
		default:
			return BuildRejectedResult($"Unsupported action type: {action.Type}.");
		}
		return new CombatTrainingStepResult
		{
			Accepted = true,
			State = await WaitForSettledAndSnapshotAsync()
		};
	}

	internal static CombatTrainingStepResult? TryValidatePlayCardRequest(Player player, CombatState combatState, CombatTrainingActionRequest action, out CardModel card, out Creature? target)
	{
		card = null;
		target = null;
		if (!CombatManager.Instance.IsPlayPhase)
		{
			return BuildRejectedResult("Cannot play cards outside of the player play phase.");
		}
		if (!action.HandIndex.HasValue)
		{
			return BuildRejectedResult("PlayCard requires hand_index.");
		}
		int handIndex = action.HandIndex.Value;
		IReadOnlyList<CardModel> handCards = player.PlayerCombatState.Hand.Cards;
		if (handIndex < 0 || handIndex >= handCards.Count)
		{
			return BuildRejectedResult($"Hand index {handIndex} is out of range.");
		}
		card = handCards[handIndex];
		if (action.TargetId.HasValue)
		{
			target = combatState.GetCreature(action.TargetId.Value);
			if (target == null)
			{
				return BuildRejectedResult($"Target {action.TargetId.Value} was not found.");
			}
		}
		bool requiresTarget = CardRequiresTarget(card);
		if (requiresTarget && target == null)
		{
			return BuildRejectedResult($"Card '{card.Id.Entry}' requires a target.");
		}
		if (!requiresTarget && target != null)
		{
			return BuildRejectedResult($"Card '{card.Id.Entry}' does not accept a target.");
		}
		if (!card.CanPlay(out UnplayableReason reason, out _))
		{
			return BuildRejectedResult($"Card '{card.Id.Entry}' is not playable: {reason}.");
		}
		if (!card.IsValidTarget(target))
		{
			return BuildRejectedResult($"Target is not valid for card '{card.Id.Entry}'.");
		}
		return null;
	}

	internal static CombatTrainingStepResult? TryValidateUsePotionRequest(Player player, CombatState combatState, CombatTrainingActionRequest action, out int slot, out Creature? target)
	{
		slot = -1;
		target = null;
		if (!action.Slot.HasValue)
		{
			return BuildRejectedResult("UsePotion requires slot.");
		}
		slot = action.Slot.Value;
		PotionModel? potion = player.GetPotionAtSlotIndex(slot);
		if (potion == null)
		{
			return BuildRejectedResult($"Potion slot {slot} is empty.");
		}
		if (potion.IsQueued)
		{
			return BuildRejectedResult($"Potion '{potion.Id.Entry}' is already queued for use.");
		}
		if (potion.Usage != PotionUsage.CombatOnly && potion.Usage != PotionUsage.AnyTime)
		{
			return BuildRejectedResult($"Potion '{potion.Id.Entry}' cannot be used manually in combat.");
		}
		if (action.TargetId.HasValue)
		{
			target = combatState.GetCreature(action.TargetId.Value);
			if (target == null)
			{
				return BuildRejectedResult($"Target {action.TargetId.Value} was not found.");
			}
		}
		if (potion.TargetType == TargetType.Self || potion.TargetType == TargetType.AnyPlayer || potion.TargetType == TargetType.AnyAlly)
		{
			target ??= player.Creature;
		}
		bool requiresTarget = potion.TargetType.IsSingleTarget();
		if (requiresTarget && target == null)
		{
			return BuildRejectedResult($"Potion '{potion.Id.Entry}' requires a target.");
		}
		if (!requiresTarget && target != null)
		{
			return BuildRejectedResult($"Potion '{potion.Id.Entry}' does not accept a target.");
		}
		if (requiresTarget && !IsPotionTargetValid(potion, target))
		{
			return BuildRejectedResult($"Target is not valid for potion '{potion.Id.Entry}'.");
		}
		return null;
	}

	internal static async Task<CombatTrainingStateSnapshot> WaitForSettledAndSnapshotAsync()
	{
		if (CombatSimulationRuntime.IsPureCombatSimulator)
		{
			await CombatSimulationRuntime.Clock.YieldAsync();
		}
		else
		{
			await Task.Yield();
		}
		using (FullRunSimulationDiagnostics.Measure("combat_step.action_executor_ms"))
		{
			await RunManager.Instance.ActionExecutor.FinishedExecutingActions();
		}
		ICombatChoiceAdapter choiceAdapter = GetChoiceAdapter();
		if (choiceAdapter.RequiresFrameSync && Engine.GetMainLoop() != null)
		{
			await Engine.GetMainLoop().ToSignal(Engine.GetMainLoop(), SceneTree.SignalName.ProcessFrame);
		}
		CombatTrainingStateSnapshot snapshot;
		using (FullRunSimulationDiagnostics.Measure("combat_step.internal_snapshot_ms"))
		{
			snapshot = BuildStateSnapshot();
		}
		return snapshot;
	}

	internal static CombatTrainingStateSnapshot BuildStateSnapshot()
	{
		CombatTrainingSession? session = CombatTrainingSession.Instance;
		CombatState? combatState = CombatManager.Instance.DebugOnlyGetState();
		ICombatChoiceAdapter choiceAdapter = GetChoiceAdapter();
		CombatTrainingStateSnapshot snapshot = new CombatTrainingStateSnapshot
		{
			IsTrainerActive = UseTrainerBackend,
			IsPureSimulator = !UseTrainerBackend,
			ChoiceAdapterKind = choiceAdapter.BackendKind,
			IsCombatActive = CombatManager.Instance.IsInProgress,
			IsEpisodeDone = session?.LastCombatWasVictory.HasValue ?? SimulatorRuntime.LastCombatWasVictory.HasValue,
			Victory = session?.LastCombatWasVictory ?? SimulatorRuntime.LastCombatWasVictory,
			EpisodeNumber = session?.CurrentEpisodeNumber ?? SimulatorRuntime.EpisodeNumber,
			Seed = session?.CurrentSeed ?? SimulatorRuntime.CurrentSeed,
			CharacterId = session?.CurrentCharacterId ?? SimulatorRuntime.CurrentCharacterId,
			EncounterId = session?.CurrentEncounterId ?? SimulatorRuntime.CurrentEncounterId,
			AscensionLevel = session?.CurrentAscensionLevel ?? SimulatorRuntime.CurrentAscensionLevel,
			RoundNumber = combatState?.RoundNumber ?? 0,
			CurrentSide = combatState?.CurrentSide ?? CombatSide.Player,
			IsPlayPhase = CombatManager.Instance.IsPlayPhase,
			PlayerActionsDisabled = CombatManager.Instance.PlayerActionsDisabled,
			IsActionQueueRunning = RunManager.Instance.IsInProgress && RunManager.Instance.ActionExecutor.IsRunning,
			IsHandSelectionActive = false,
			IsCardSelectionActive = false
		};
		if (combatState == null)
		{
			return snapshot;
		}
		Player? player = null;
		try
		{
			player = LocalContext.GetMe(combatState);
		}
		catch (InvalidOperationException) when (!UseTrainerBackend)
		{
			player = combatState.Players.FirstOrDefault();
		}
		if (player == null && !UseTrainerBackend)
		{
			player = combatState.Players.FirstOrDefault();
		}
		if (player != null && !UseTrainerBackend)
		{
			// Pure-sim full-run is effectively single-player. If the local context drifts,
			// fall back to the only combat player so long boss fights don't degrade into
			// empty-action snapshots.
			LocalContext.NetId = player.NetId;
		}
		if (player == null)
		{
			snapshot.IsPlayPhase = false;
			snapshot.PlayerActionsDisabled = true;
			snapshot.IsActionQueueRunning = true;
			return snapshot;
		}
		if (player.PlayerCombatState == null)
		{
			snapshot.IsPlayPhase = false;
			snapshot.PlayerActionsDisabled = true;
			snapshot.IsActionQueueRunning = true;
			return snapshot;
		}
		snapshot.Player = BuildPlayerSnapshot(player);
		snapshot.Enemies = BuildEnemySnapshots(combatState);
		snapshot.Hand = BuildHandSnapshot(player, combatState);
		snapshot.Piles = new CombatTrainingPileSnapshot
		{
			Draw = player.PlayerCombatState.DrawPile.Cards.Count,
			Discard = player.PlayerCombatState.DiscardPile.Cards.Count,
			Exhaust = player.PlayerCombatState.ExhaustPile.Cards.Count,
			Play = player.PlayerCombatState.PlayPile.Cards.Count,
			DrawCardIds = player.PlayerCombatState.DrawPile.Cards
				.Select(static c => c.Id.Entry).ToList(),
			DiscardCardIds = player.PlayerCombatState.DiscardPile.Cards
				.Select(static c => c.Id.Entry).ToList(),
			ExhaustCardIds = player.PlayerCombatState.ExhaustPile.Cards
				.Select(static c => c.Id.Entry).ToList(),
		};
		snapshot.HandSelection = choiceAdapter.BuildHandSelectionSnapshot(combatState);
		snapshot.CardSelection = choiceAdapter.BuildCardSelectionSnapshot(combatState);
		snapshot.IsHandSelectionActive = snapshot.HandSelection != null;
		snapshot.IsCardSelectionActive = snapshot.CardSelection != null;
		snapshot.CanEndTurn = CombatManager.Instance.IsInProgress && CombatManager.Instance.IsPlayPhase && !choiceAdapter.IsSelectionActive && !CombatManager.Instance.IsPlayerReadyToEndTurn(player);
		return snapshot;
	}

	private static CombatTrainingPlayerSnapshot BuildPlayerSnapshot(Player player)
	{
		return new CombatTrainingPlayerSnapshot
		{
			NetId = player.NetId,
			CombatId = player.Creature.CombatId,
			CurrentHp = player.Creature.CurrentHp,
			MaxHp = player.Creature.MaxHp,
			Block = player.Creature.Block,
			Energy = player.PlayerCombatState.Energy,
			MaxEnergy = player.PlayerCombatState.MaxEnergy,
			Stars = player.PlayerCombatState.Stars,
			Powers = BuildPowerSnapshot(player.Creature)
		};
	}

	private static CombatTrainingCreatureSnapshot BuildCreatureSnapshot(Creature creature)
	{
		MoveState? nextMove = creature.Monster?.NextMove;
		Creature[] intentTargets = creature.CombatState?.RunState.Players.Select(static player => player.Creature).ToArray() ?? Array.Empty<Creature>();
		return new CombatTrainingCreatureSnapshot
		{
			CombatId = creature.CombatId,
			Id = creature.Monster?.Id.Entry,
			Name = SafeCreatureName(creature),
			CurrentHp = creature.CurrentHp,
			MaxHp = creature.MaxHp,
			Block = creature.Block,
			IsAlive = creature.IsAlive,
			IsHittable = creature.IsHittable,
			NextMoveId = nextMove?.Id,
			IntendsToAttack = creature.Monster?.IntendsToAttack ?? false,
			Intents = BuildIntentSnapshot(nextMove, creature, intentTargets),
			Powers = BuildPowerSnapshot(creature)
		};
	}

	private static List<CombatTrainingCreatureSnapshot> BuildEnemySnapshots(CombatState combatState)
	{
		List<CombatTrainingCreatureSnapshot> enemies = new List<CombatTrainingCreatureSnapshot>();
		foreach (Creature enemy in combatState.Enemies)
		{
			if (enemy == null || !enemy.IsAlive)
			{
				continue;
			}
			try
			{
				enemies.Add(BuildCreatureSnapshot(enemy));
			}
			catch (Exception ex)
			{
				FullRunSimulationTrace.Write($"headless_combat_snapshot.enemy_exception exception={ex}");
			}
		}
		return enemies;
	}

	private static string SafeCreatureName(Creature creature)
	{
		try
		{
			return creature.Name;
		}
		catch
		{
			return creature.Monster?.Id.Entry ?? creature.GetType().Name;
		}
	}

	private static List<CombatTrainingIntentSnapshot> BuildIntentSnapshot(MoveState? nextMove, Creature owner, IReadOnlyList<Creature> targets)
	{
		if (nextMove == null)
		{
			return EmptyIntentList;
		}
		List<CombatTrainingIntentSnapshot> result = new List<CombatTrainingIntentSnapshot>(nextMove.Intents.Count);
		foreach (AbstractIntent intent in nextMove.Intents)
		{
			result.Add(BuildIntentSnapshot(intent, owner, targets));
		}
		return result;
	}

	private static CombatTrainingIntentSnapshot BuildIntentSnapshot(AbstractIntent intent, Creature owner, IReadOnlyList<Creature> targets)
	{
		CombatTrainingIntentSnapshot snapshot = new CombatTrainingIntentSnapshot
		{
			IntentType = intent.IntentType.ToString(),
			Repeats = 0
		};
		if (intent is AttackIntent attackIntent)
		{
			snapshot.Repeats = attackIntent.Repeats;
			snapshot.Damage = attackIntent.GetSingleDamage(targets, owner);
			snapshot.TotalDamage = attackIntent.GetTotalDamage(targets, owner);
		}
		return snapshot;
	}

	private static List<CombatTrainingPowerSnapshot> BuildPowerSnapshot(Creature creature)
	{
		var powers = creature.Powers;
		if (powers.Count == 0) return new List<CombatTrainingPowerSnapshot>();
		List<CombatTrainingPowerSnapshot> result = new List<CombatTrainingPowerSnapshot>(powers.Count);
		foreach (var power in powers)
		{
			result.Add(new CombatTrainingPowerSnapshot
			{
				Id = power.Id.Entry,
				Amount = power.Amount
			});
		}
		return result;
	}

	private static List<CombatTrainingHandCardSnapshot> BuildHandSnapshot(Player player, CombatState combatState)
	{
		List<CombatTrainingHandCardSnapshot> cards = new List<CombatTrainingHandCardSnapshot>();
		for (int i = 0; i < player.PlayerCombatState.Hand.Cards.Count; i++)
		{
			try
			{
				cards.Add(BuildHandCardSnapshot(player.PlayerCombatState.Hand.Cards[i], combatState, i));
			}
			catch (Exception ex)
			{
				FullRunSimulationTrace.Write($"headless_combat_snapshot.hand_card_exception index={i} exception={ex}");
			}
		}
		return cards;
	}

	private static CombatTrainingHandCardSnapshot BuildHandCardSnapshot(CardModel card, CombatState combatState, int? explicitHandIndex = null)
	{
		int handIndex = explicitHandIndex ?? GetHandIndex(card);
		List<uint> validTargetIds = GetValidTargetIds(card, combatState);
		return new CombatTrainingHandCardSnapshot
		{
			HandIndex = handIndex,
			CombatCardIndex = NetCombatCard.FromModel(card).CombatCardIndex,
			Id = card.Id.Entry,
			Title = SafeCardTitle(card),
			EnergyCost = NormalizeApiCardCost(card.EnergyCost.GetWithModifiers(CostModifiers.All), card.EnergyCost.CostsX),
			IsUpgraded = card.IsUpgraded,
			CostsX = card.EnergyCost.CostsX,
			StarCost = card.GetStarCostWithModifiers(),
			TargetType = card.TargetType,
			CanPlay = card.CanPlay(),
			RequiresTarget = CardRequiresTarget(card),
			ValidTargetIds = validTargetIds,
			CardType = card.Type.ToString(),
			Description = SafeCardDescription(card),
			Keywords = SafeCardKeywords(card),
			GainsBlock = card.GainsBlock
		};
	}

	private static string SafeCardTitle(CardModel card)
	{
		try
		{
			return card.Title;
		}
		catch
		{
			return card.Id.Entry;
		}
	}

	private static string SafeCardDescription(CardModel card)
	{
		try
		{
			return card.GetDescriptionForPile(card.Pile?.Type ?? PileType.Hand);
		}
		catch
		{
			try
			{
				return card.Description.GetRawText();
			}
			catch
			{
				return string.Empty;
			}
		}
	}

	private static List<string> SafeCardKeywords(CardModel card)
	{
		try
		{
			var keywords = card.Keywords;
			if (keywords.Count == 0) return EmptyStringList;
			List<string> result = new List<string>(keywords.Count);
			foreach (var keyword in keywords)
			{
				result.Add(keyword.ToString());
			}
			return result;
		}
		catch
		{
			return EmptyStringList;
		}
	}

	private static int NormalizeApiCardCost(int energyCost, bool costsX)
	{
		if (costsX)
		{
			return energyCost;
		}
		return energyCost < 0 ? 0 : energyCost;
	}

	private static List<uint> GetValidTargetIds(CardModel card, CombatState combatState)
	{
		if (!CardRequiresTarget(card))
		{
			return EmptyUintList;
		}
		List<uint>? result = null;
		foreach (Creature creature in combatState.Creatures)
		{
			if (creature.CombatId.HasValue && card.IsValidTarget(creature))
			{
				result ??= new List<uint>();
				result.Add(creature.CombatId.Value);
			}
		}
		return result ?? EmptyUintList;
	}

	private static bool CardRequiresTarget(CardModel card)
	{
		return card.TargetType == TargetType.AnyEnemy || card.TargetType == TargetType.AnyAlly;
	}

	private static bool IsPotionTargetValid(PotionModel potion, Creature? target)
	{
		if (target == null)
		{
			return false;
		}
		return potion.TargetType switch
		{
			TargetType.Self => target == potion.Owner.Creature,
			TargetType.AnyEnemy => target.Side == CombatSide.Enemy && target.IsHittable,
			TargetType.AnyAlly => target.Side == CombatSide.Player && target.IsHittable,
			TargetType.AnyPlayer => target.IsPlayer && target.IsHittable,
			_ => true
		};
	}

	private static int GetHandIndex(CardModel card)
	{
		IReadOnlyList<CardModel> cards = PileType.Hand.GetPile(card.Owner).Cards;
		for (int i = 0; i < cards.Count; i++)
		{
			if (cards[i] == card)
			{
				return i;
			}
		}
		return -1;
	}


	internal static CombatTrainingStepResult? TryValidateSelectionRequest(CombatTrainingActionRequest action, ICombatChoiceAdapter choiceAdapter, bool? requireHandSelection)
	{
		if (!choiceAdapter.IsSelectionActive)
		{
			return BuildRejectedResult("Card selection is not active.");
		}
		if (!CombatManager.Instance.IsInProgress)
		{
			return BuildRejectedResult("Combat is already over.");
		}
		bool isHandSelectionActive = choiceAdapter.BuildHandSelectionSnapshot(CombatManager.Instance.DebugOnlyGetState()) != null;
		bool isCardSelectionActive = choiceAdapter.BuildCardSelectionSnapshot(CombatManager.Instance.DebugOnlyGetState()) != null;
		if (requireHandSelection == true && !isHandSelectionActive)
		{
			return BuildRejectedResult("Hand card selection is not active.");
		}
		if (requireHandSelection == false && !isCardSelectionActive)
		{
			return BuildRejectedResult("Card selection is not active.");
		}
		if (action.Type == CombatTrainingActionType.SelectHandCard && !action.HandIndex.HasValue)
		{
			return BuildRejectedResult("SelectHandCard requires hand_index.");
		}
		if (action.Type == CombatTrainingActionType.SelectCardChoice && !action.ChoiceIndex.HasValue)
		{
			return BuildRejectedResult("SelectCardChoice requires choice_index.");
		}
		return null;
	}

	internal static ICombatChoiceAdapter GetChoiceAdapter()
	{
		return CombatTrainingChoiceAdapterResolver.Resolve();
	}

	private static CombatTrainingSession GetSession()
	{
		return CombatTrainingSession.Instance ?? throw new InvalidOperationException("Combat training session is not active.");
	}

	private static void EnsureEnvIsAvailable()
	{
		if (!UseTrainerBackend && !SimulatorRuntime.IsPureSimulator)
		{
			throw new InvalidOperationException("CombatTrainingEnvService requires either --combat-trainer or the pure simulator runtime.");
		}
		if ((UseTrainerBackend || CombatSimulationMode.IsServerActive) && !Nodes.NGame.IsMainThread())
		{
			throw new InvalidOperationException("CombatTrainingEnvService must be called from the main game thread.");
		}
	}

	internal static CombatTrainingStateSnapshot CaptureStateSnapshotForRecorder()
	{
		return BuildStateSnapshot();
	}

	internal static CombatTrainingStepResult BuildRejectedResult(string error)
	{
		return new CombatTrainingStepResult
		{
			Accepted = false,
			Error = error,
			State = BuildStateSnapshot()
		};
	}
}
