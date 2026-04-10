using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Runs;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Rooms;

public class CombatRoom : AbstractRoom, ICombatRoomVisuals
{
	private bool _isPreFinished;

	private Dictionary<ulong, SerializablePlayerCombatState>? _savedPlayerCombatStates;
	private List<SerializableCreatureState>? _savedCreatureStates;

	/// <summary>
	/// True after RestoreSavedCombatPileState() — tells CombatManager to skip
	/// the initial StartTurn (energy reset + draw) since piles are already correct.
	/// Consumed (set false) after one check.
	/// </summary>
	internal bool SkipInitialStartTurn { get; set; }

	private readonly Dictionary<Player, List<Reward>> _extraRewards = new Dictionary<Player, List<Reward>>();

	public override RoomType RoomType => Encounter.RoomType;

	public override ModelId ModelId => Encounter.Id;

	public EncounterModel Encounter => CombatState.Encounter;

	public CombatState CombatState { get; }

	public IEnumerable<Creature> Allies => CombatState.Allies;

	public IEnumerable<Creature> Enemies => CombatState.Enemies;

	public ActModel Act => CombatState.RunState.Act;

	public override bool IsPreFinished => _isPreFinished;

	public float GoldProportion { get; private set; } = 1f;

	public IReadOnlyDictionary<Player, List<Reward>> ExtraRewards => _extraRewards;

	public bool ShouldCreateCombat { get; init; } = true;

	public bool ShouldResumeParentEventAfterCombat { get; init; } = true;

	public ModelId? ParentEventId { get; init; }

	public CombatRoom(EncounterModel encounter, IRunState? runState)
	{
		encounter.AssertMutable();
		CombatState = new CombatState(encounter, runState, runState?.Modifiers, runState?.MultiplayerScalingModel);
	}

	public CombatRoom(CombatState combatState)
	{
		CombatState = combatState;
	}

	public new static CombatRoom FromSerializable(SerializableRoom serializableRoom, IRunState? runState)
	{
		if (serializableRoom.ExtraRewards.Count > 0 && runState == null)
		{
			throw new InvalidOperationException("Cannot load extra rewards without a run state.");
		}
		EncounterModel encounterModel = SaveUtil.EncounterOrDeprecated(serializableRoom.EncounterId).ToMutable();
		encounterModel.LoadCustomState(serializableRoom.EncounterState);
		CombatRoom combatRoom = new CombatRoom(encounterModel, runState)
		{
			GoldProportion = serializableRoom.GoldProportion,
			_isPreFinished = serializableRoom.IsPreFinished,
			ShouldResumeParentEventAfterCombat = serializableRoom.ShouldResumeParentEvent,
			ParentEventId = serializableRoom.ParentEventId
		};
		foreach (KeyValuePair<ulong, List<SerializableReward>> extraReward in serializableRoom.ExtraRewards)
		{
			extraReward.Deconstruct(out var key, out var value);
			ulong netId = key;
			List<SerializableReward> source = value;
			Player player = runState.GetPlayer(netId);
			List<Reward> value2 = source.Select((SerializableReward sr) => Reward.FromSerializable(sr, player)).ToList();
			combatRoom._extraRewards.Add(player, value2);
		}
		if (serializableRoom.PlayerCombatStates is { Count: > 0 })
		{
			combatRoom._savedPlayerCombatStates = serializableRoom.PlayerCombatStates;
		}
		if (serializableRoom.CreatureStates is { Count: > 0 })
		{
			combatRoom._savedCreatureStates = serializableRoom.CreatureStates;
		}
		if (serializableRoom.IsPreFinished)
		{
			combatRoom.MarkPreFinished();
		}
		return combatRoom;
	}

	public override async Task Enter(IRunState? runState, bool isRestoringRoomStackBase)
	{
		if (isRestoringRoomStackBase)
		{
			throw new InvalidOperationException("CombatRoom does not support room stack reconstruction.");
		}
		if (CombatState.Players.Count == 0)
		{
			foreach (Player item in runState?.Players ?? Array.Empty<Player>())
			{
				CombatState.AddPlayer(item);
			}
		}
		if (IsPreFinished)
		{
			await StartPreFinishedCombat();
		}
		else
		{
			await StartCombat(runState);
		}
	}

	public override Task Exit(IRunState? runState)
	{
		CombatManager.Instance.Reset();
		if (IsPreFinished)
		{
			foreach (Creature item in CombatState.PlayerCreatures.ToList())
			{
				CombatState.RemoveCreature(item);
			}
		}
		return Task.CompletedTask;
	}

	public override Task Resume(AbstractRoom _, IRunState? runState)
	{
		throw new NotImplementedException();
	}

	public override SerializableRoom ToSerializable()
	{
		if (ParentEventId != null && !IsPreFinished)
		{
			throw new InvalidOperationException("Cannot serialize a CombatRoom with a ParentEventId that is not pre-finished.");
		}
		SerializableRoom serializableRoom = base.ToSerializable();
		serializableRoom.EncounterId = Encounter.Id;
		serializableRoom.IsPreFinished = IsPreFinished;
		serializableRoom.GoldProportion = GoldProportion;
		serializableRoom.ParentEventId = ParentEventId;
		serializableRoom.ShouldResumeParentEvent = ShouldResumeParentEventAfterCombat;
		serializableRoom.EncounterState = Encounter.SaveCustomState();
		foreach (var (player2, source) in ExtraRewards)
		{
			serializableRoom.ExtraRewards[player2.NetId] = source.Select((Reward r) => r.ToSerializable()).ToList();
		}
		// Capture mid-combat pile ordering for save/load fidelity (MCTS tree search).
		// Without this, LoadState() would reshuffle cards via PopulateCombatState().
		SerializeCombatPileState(serializableRoom);
		return serializableRoom;
	}

	private void SerializeCombatPileState(SerializableRoom serializableRoom)
	{
		foreach (Player player in CombatState.Players)
		{
			var pcs = player.PlayerCombatState;
			if (pcs == null) continue;
			serializableRoom.PlayerCombatStates[player.NetId] = new SerializablePlayerCombatState
			{
				Hand = pcs.Hand.Cards.Select(c => c.ToSerializable()).ToList(),
				DrawPile = pcs.DrawPile.Cards.Select(c => c.ToSerializable()).ToList(),
				DiscardPile = pcs.DiscardPile.Cards.Select(c => c.ToSerializable()).ToList(),
				ExhaustPile = pcs.ExhaustPile.Cards.Select(c => c.ToSerializable()).ToList(),
				Energy = pcs.Energy,
				Block = player.Creature.Block,
				Powers = player.Creature.Powers.Select(p => new Saves.Runs.SerializablePower
				{
					Id = p.Id.Entry,
					Amount = p.Amount
				}).ToList(),
			};
		}

		// Serialize all creature states (enemies + player creatures)
		serializableRoom.CreatureStates = new List<Saves.Runs.SerializableCreatureState>();
		foreach (Creature creature in CombatState.Enemies)
		{
			serializableRoom.CreatureStates.Add(new Saves.Runs.SerializableCreatureState
			{
				Id = creature.ModelId.Entry,
				CombatId = creature.CombatId ?? 0,
				Hp = creature.CurrentHp,
				MaxHp = creature.MaxHp,
				Block = creature.Block,
				Powers = creature.Powers.Select(p => new Saves.Runs.SerializablePower
				{
					Id = p.Id.Entry,
					Amount = p.Amount
				}).ToList(),
			});
		}

		// TurnNumber not currently tracked by CombatManager; could add later.
	}

	public void MarkPreFinished()
	{
		_isPreFinished = true;
	}

	public void AddExtraReward(Player player, Reward reward)
	{
		if (!ExtraRewards.ContainsKey(player))
		{
			_extraRewards.Add(player, new List<Reward>());
		}
		ExtraRewards[player].Add(reward);
	}

	internal Dictionary<Player, List<Reward>> CloneExtraRewardsForSimulationSave()
	{
		return _extraRewards.ToDictionary(static entry => entry.Key, static entry => entry.Value.ToList());
	}

	internal void ReplaceExtraRewardsForSimulationSave(Dictionary<Player, List<Reward>> rewards)
	{
		_extraRewards.Clear();
		foreach (KeyValuePair<Player, List<Reward>> rewardSet in rewards)
		{
			_extraRewards[rewardSet.Key] = rewardSet.Value.ToList();
		}
	}

	private async Task StartCombat(IRunState? runState)
	{
		if (!Encounter.HaveMonstersBeenGenerated)
		{
			Encounter.GenerateMonstersWithSlots(CombatState.RunState);
		}
		if (ShouldCreateCombat)
		{
			await PreloadManager.LoadRoomCombatAssets(Encounter, runState ?? NullRunState.Instance);
		}
		foreach (var (monsterModel, slot) in Encounter.MonstersWithSlots)
		{
			monsterModel.AssertMutable();
			if (ShouldCreateCombat)
			{
				Creature creature = CombatState.CreateCreature(monsterModel, CombatSide.Enemy, slot);
				CombatState.AddCreature(creature);
			}
			CombatState.RunState.CurrentMapPointHistoryEntry.Rooms.Last().MonsterIds.Add(monsterModel.Id);
		}
		if (ShouldCreateCombat)
		{
			NRun.Instance?.SetCurrentRoom(NCombatRoom.Create(this, CombatRoomMode.ActiveCombat));
		}
		else
		{
			NCombatRoom.Instance?.TransitionToActiveCombat(this);
		}
		CombatManager.Instance.SetUpCombat(CombatState);
		RestoreSavedCombatPileState();
		if (runState != null)
		{
			await Hook.AfterRoomEntered(runState, this);
		}
		CombatManager.Instance.AfterCombatRoomLoaded();
	}

	/// <summary>
	/// After SetUpCombat() places all cloned deck cards into the draw pile (shuffled),
	/// this method redistributes them into hand/draw/discard/exhaust to match the
	/// saved mid-combat state. Cards are matched by serialized identity (Id + upgrade + enchantment).
	/// </summary>
	private void RestoreSavedCombatPileState()
	{
		if (_savedPlayerCombatStates == null || _savedPlayerCombatStates.Count == 0)
			return;

		foreach (Player player in CombatState.Players)
		{
			if (!_savedPlayerCombatStates.TryGetValue(player.NetId, out var saved))
				continue;

			var pcs = player.PlayerCombatState;
			if (pcs == null) continue;

			// Collect all combat cards currently in draw pile (put there by PopulateCombatState).
			// We'll redistribute them into the correct piles.
			var availableCards = new List<CardModel>(pcs.DrawPile.Cards);

			// Clear all piles silently (cards were just created, no UI to update).
			pcs.DrawPile.Clear(silent: true);

			// Helper: find and remove a matching card from the available pool
			CardModel? TakeMatchingCard(SerializableCard sc)
			{
				for (int i = 0; i < availableCards.Count; i++)
				{
					var card = availableCards[i];
					if (card.Id == sc.Id && card.CurrentUpgradeLevel == sc.CurrentUpgradeLevel)
					{
						availableCards.RemoveAt(i);
						return card;
					}
				}
				// Card not found in pool — might have been created mid-combat.
				// Create a fresh combat card from the serializable data.
				CardModel freshCard = CardModel.FromSerializable(sc);
				CombatState.AddCard(freshCard, player);
				return freshCard;
			}

			// Restore each pile in saved order
			foreach (var sc in saved.Hand)
			{
				CardModel? card = TakeMatchingCard(sc);
				if (card != null) pcs.Hand.AddInternal(card, silent: true);
			}
			foreach (var sc in saved.DrawPile)
			{
				CardModel? card = TakeMatchingCard(sc);
				if (card != null) pcs.DrawPile.AddInternal(card, silent: true);
			}
			foreach (var sc in saved.DiscardPile)
			{
				CardModel? card = TakeMatchingCard(sc);
				if (card != null) pcs.DiscardPile.AddInternal(card, silent: true);
			}
			foreach (var sc in saved.ExhaustPile)
			{
				CardModel? card = TakeMatchingCard(sc);
				if (card != null) pcs.ExhaustPile.AddInternal(card, silent: true);
			}

			// Any remaining cards in availableCards were not in the saved state
			// (e.g., cards removed mid-combat). Remove them from CombatState tracking.
			foreach (var leftover in availableCards)
			{
				CombatState.RemoveCard(leftover);
			}

			// Restore energy and block
			pcs.Energy = saved.Energy;
			// Block setter is private; use internal methods to set the correct value.
			if (player.Creature.Block > saved.Block)
				player.Creature.LoseBlockInternal(player.Creature.Block - saved.Block);
			else if (player.Creature.Block < saved.Block)
				player.Creature.GainBlockInternal(saved.Block - player.Creature.Block);
		}

		// Restore creature (enemy) states: HP, block, powers
		if (_savedCreatureStates != null && _savedCreatureStates.Count > 0)
		{
			foreach (var savedCreature in _savedCreatureStates)
			{
				// Match by CombatId or ModelId
				Creature? creature = null;
				foreach (Creature enemy in CombatState.Enemies)
				{
					if ((enemy.CombatId ?? 0) == savedCreature.CombatId
						|| enemy.ModelId.Entry == savedCreature.Id)
					{
						creature = enemy;
						break;
					}
				}
				if (creature == null) continue;

				// Restore HP
				creature.SetCurrentHpInternal(savedCreature.Hp);

				// Restore block
				if (creature.Block > savedCreature.Block)
					creature.LoseBlockInternal(creature.Block - savedCreature.Block);
				else if (creature.Block < savedCreature.Block)
					creature.GainBlockInternal(savedCreature.Block - creature.Block);

				// Restore powers
				foreach (var existing in creature.Powers.ToList())
				{
					existing.RemoveInternal();
				}
				foreach (var savedPower in savedCreature.Powers)
				{
					try
					{
						var powerModel = ModelDb.GetByIdOrNull<PowerModel>(
							new ModelId(ModelDb.GetCategory(typeof(PowerModel)), savedPower.Id));
						if (powerModel != null)
						{
							var mutable = powerModel.ToMutable();
							mutable.ApplyInternal(creature, savedPower.Amount, silent: true);
						}
					}
					catch { }
				}
			}
		}

		// Restore player powers from combat state
		foreach (Player player in CombatState.Players)
		{
			if (!_savedPlayerCombatStates!.TryGetValue(player.NetId, out var saved))
				continue;
			if (saved.Powers == null || saved.Powers.Count == 0)
				continue;

			foreach (var existing in player.Creature.Powers.ToList())
			{
				existing.RemoveInternal();
			}
			foreach (var savedPower in saved.Powers)
			{
				try
				{
					var powerModel = ModelDb.GetByIdOrNull<PowerModel>(
						new ModelId(ModelDb.GetCategory(typeof(PowerModel)), savedPower.Id));
					if (powerModel != null)
					{
						var mutable = powerModel.ToMutable();
						mutable.ApplyInternal(player.Creature, savedPower.Amount, silent: true);
					}
				}
				catch { }
			}
		}

		// Clear saved state — it's been consumed
		SkipInitialStartTurn = true;
		_savedPlayerCombatStates = null;
		_savedCreatureStates = null;
	}

	public void OnCombatEnded()
	{
		GoldProportion = 1f - (float)CombatState.EscapedCreatures.Count / (float)Encounter.MonstersWithSlots.Count;
	}

	private async Task StartPreFinishedCombat()
	{
		if (TestMode.IsOn)
		{
			return;
		}
		Encounter.GenerateMonstersWithSlots(CombatState.RunState);
		await PreloadManager.LoadRoomCombatAssets(Encounter, CombatState.RunState);
		NCombatRoom nCombatRoom = NCombatRoom.Create(this, CombatRoomMode.FinishedCombat);
		NRun.Instance?.SetCurrentRoom(nCombatRoom);
		nCombatRoom?.SetUpBackground(CombatState.RunState);
		NMapScreen.Instance.SetTravelEnabled(enabled: true);
		foreach (Player player in CombatState.RunState.Players)
		{
			player.ResetCombatState();
		}
		RunManager.Instance.ActionExecutor.Unpause();
		Player me = LocalContext.GetMe(CombatState);
		TaskHelper.RunSafely(RewardsCmd.OfferForRoomEnd(me, this));
	}
}
