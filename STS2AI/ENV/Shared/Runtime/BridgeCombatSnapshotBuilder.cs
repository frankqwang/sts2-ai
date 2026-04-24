using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Simulation;
using MegaCrit.Sts2.Core.Training;

namespace STS2AI.Bridge.Runtime;

public static class BridgeCombatSnapshotBuilder
{
	private static readonly List<CombatTrainingIntentSnapshot> EmptyIntentList = new();
	private static readonly List<CombatTrainingPowerSnapshot> EmptyPowerList = new();
	private static readonly List<CombatTrainingHandCardSnapshot> EmptyHandList = new();
	private static readonly List<string> EmptyStringList = new();
	private static readonly List<uint> EmptyUintList = new();

	public static CombatTrainingStateSnapshot BuildStateSnapshot()
	{
		CombatState? combatState = CombatManager.Instance.DebugOnlyGetState();
		ICombatChoiceAdapter choiceAdapter = CombatTrainingChoiceAdapterResolver.Resolve();
		RunState? runState = RunManager.Instance.DebugOnlyGetState();
		Player? runPlayer = TryResolveRunPlayer(runState);
		CombatTrainingStateSnapshot snapshot = new CombatTrainingStateSnapshot
		{
			IsTrainerActive = false,
			IsPureSimulator = false,
			ChoiceAdapterKind = choiceAdapter.BackendKind,
			IsCombatActive = CombatManager.Instance.IsInProgress,
			IsEpisodeDone = combatState == null && !CombatManager.Instance.IsInProgress,
			Victory = null,
			EpisodeNumber = 0,
			Seed = runState?.Rng.StringSeed,
			CharacterId = runPlayer?.Character.Id.Entry,
			EncounterId = combatState?.Encounter.Id.Entry,
			AscensionLevel = runState?.AscensionLevel ?? 0,
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

		Player? player = TryResolveCombatPlayer(combatState);
		if (player == null || player.PlayerCombatState == null)
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
			DrawCardIds = player.PlayerCombatState.DrawPile.Cards.Select(static c => c.Id.Entry).ToList(),
			DiscardCardIds = player.PlayerCombatState.DiscardPile.Cards.Select(static c => c.Id.Entry).ToList(),
			ExhaustCardIds = player.PlayerCombatState.ExhaustPile.Cards.Select(static c => c.Id.Entry).ToList()
		};
		snapshot.HandSelection = choiceAdapter.BuildHandSelectionSnapshot(combatState);
		snapshot.CardSelection = choiceAdapter.BuildCardSelectionSnapshot(combatState);
		snapshot.IsHandSelectionActive = snapshot.HandSelection != null;
		snapshot.IsCardSelectionActive = snapshot.CardSelection != null;
		snapshot.CanEndTurn = CombatManager.Instance.IsInProgress
			&& CombatManager.Instance.IsPlayPhase
			&& !choiceAdapter.IsSelectionActive
			&& !CombatManager.Instance.IsPlayerReadyToEndTurn(player);
		return snapshot;
	}

	private static Player? TryResolveRunPlayer(RunState? runState)
	{
		if (runState == null) return null;
		try
		{
			Player? player = LocalContext.GetMe(runState.Players);
			if (player != null) return player;
		}
		catch
		{
		}
		Player? fallback = runState.Players.FirstOrDefault();
		if (fallback != null) LocalContext.NetId ??= fallback.NetId;
		return fallback;
	}

	private static Player? TryResolveCombatPlayer(CombatState combatState)
	{
		try
		{
			Player? player = LocalContext.GetMe(combatState);
			if (player != null) return player;
		}
		catch
		{
		}
		Player? fallback = combatState.Players.FirstOrDefault();
		if (fallback != null) LocalContext.NetId ??= fallback.NetId;
		return fallback;
	}

	private static CombatTrainingPlayerSnapshot BuildPlayerSnapshot(Player player) => new CombatTrainingPlayerSnapshot
	{
		NetId = player.NetId,
		CombatId = player.Creature.CombatId,
		CurrentHp = player.Creature.CurrentHp,
		MaxHp = player.Creature.MaxHp,
		Block = player.Creature.Block,
		Energy = player.PlayerCombatState?.Energy ?? 0,
		MaxEnergy = player.PlayerCombatState?.MaxEnergy ?? 0,
		Stars = player.PlayerCombatState?.Stars ?? 0,
		Powers = BuildPowerSnapshot(player.Creature)
	};

	private static List<CombatTrainingCreatureSnapshot> BuildEnemySnapshots(CombatState combatState)
	{
		List<CombatTrainingCreatureSnapshot> enemies = new();
		foreach (Creature enemy in combatState.Enemies)
		{
			if (enemy == null || !enemy.IsAlive) continue;
			try
			{
				enemies.Add(BuildCreatureSnapshot(enemy));
			}
			catch (Exception ex)
			{
				FullRunSimulationTrace.Write($"bridge_combat_snapshot.enemy_exception exception={ex}");
			}
		}
		return enemies;
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
		if (nextMove == null) return EmptyIntentList;
		List<CombatTrainingIntentSnapshot> result = new(nextMove.Intents.Count);
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
		if (powers.Count == 0) return EmptyPowerList;
		List<CombatTrainingPowerSnapshot> result = new(powers.Count);
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
		if (player.PlayerCombatState == null) return EmptyHandList;
		List<CombatTrainingHandCardSnapshot> cards = new();
		for (int i = 0; i < player.PlayerCombatState.Hand.Cards.Count; i++)
		{
			try
			{
				cards.Add(BuildHandCardSnapshot(player.PlayerCombatState.Hand.Cards[i], combatState, i));
			}
			catch (Exception ex)
			{
				FullRunSimulationTrace.Write($"bridge_combat_snapshot.hand_card_exception index={i} exception={ex}");
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
			CanPlay = SafeBool(() => card.CanPlay()),
			RequiresTarget = CardRequiresTarget(card),
			ValidTargetIds = validTargetIds,
			CardType = card.Type.ToString(),
			Description = CombatTrainingCardDescription.GetDescription(card, card.Pile?.Type ?? PileType.Hand),
			Keywords = SafeCardKeywords(card),
			GainsBlock = card.GainsBlock,
			PreviewDamagePerTarget = CombatTrainingCardDescription.BuildPreviewDamagePerTarget(card, combatState, validTargetIds),
			PreviewBlock = CombatTrainingCardDescription.GetPreviewBlock(card)
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

	private static List<string> SafeCardKeywords(CardModel card)
	{
		try
		{
			var keywords = card.Keywords;
			if (keywords.Count == 0) return EmptyStringList;
			return keywords.Select(static keyword => keyword.ToString()).ToList();
		}
		catch
		{
			return EmptyStringList;
		}
	}

	private static List<uint> GetValidTargetIds(CardModel card, CombatState combatState)
	{
		if (!CardRequiresTarget(card)) return EmptyUintList;
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
		=> card.TargetType is TargetType.AnyEnemy or TargetType.AnyAlly;

	private static int NormalizeApiCardCost(int energyCost, bool costsX)
		=> costsX ? energyCost : Math.Max(0, energyCost);

	private static int GetHandIndex(CardModel card)
	{
		IReadOnlyList<CardModel> cards = PileType.Hand.GetPile(card.Owner).Cards;
		for (int i = 0; i < cards.Count; i++)
		{
			if (cards[i] == card) return i;
		}
		return -1;
	}

	private static bool SafeBool(Func<bool> read, bool fallback = false)
	{
		try
		{
			return read();
		}
		catch
		{
			return fallback;
		}
	}
}
