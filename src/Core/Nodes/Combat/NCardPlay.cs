using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Ftue;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public abstract partial class NCardPlay : Node
{
	[Signal]
	public delegate void FinishedEventHandler(bool success);

	private static int _totalCardsPlayedForFtue;

	private const int _numCardPlayedUntilDisableFtue = 8;

	protected Viewport _viewport;

	private bool _isTryingToPlayCard;

	public NHandCardHolder Holder { get; protected set; }

	protected NCard? CardNode => Holder.CardNode;

	protected CardModel? Card => CardNode?.Model;

	protected NCreature? CardOwnerNode => NCombatRoom.Instance.GetCreatureNode(Card?.Owner.Creature);

	public Player Player { get; protected set; }

	public override void _Ready()
	{
		_viewport = GetViewport();
	}

	public abstract void Start();

	protected void TryPlayCard(Creature? target)
	{
		if (Card == null)
		{
			return;
		}
		if (Card.TargetType == TargetType.AnyEnemy && target == null)
		{
			CancelPlayCard();
			return;
		}
		if (Card.TargetType == TargetType.AnyAlly && target == null)
		{
			CancelPlayCard();
			return;
		}
		if (!Holder.CardModel.CanPlayTargeting(target))
		{
			CannotPlayThisCardFtueCheck(Holder.CardModel);
			CancelPlayCard();
			return;
		}
		CardModel card = Card;
		_isTryingToPlayCard = true;
		TargetType targetType = card.TargetType;
		bool flag = ((targetType == TargetType.AnyEnemy || targetType == TargetType.AnyAlly) ? true : false);
		bool flag2 = ((!flag) ? card.TryManualPlay(null) : card.TryManualPlay(target));
		_isTryingToPlayCard = false;
		if (flag2)
		{
			AutoDisableCannotPlayCardFtueCheck();
			targetType = card.TargetType;
			flag = ((targetType == TargetType.AnyEnemy || targetType == TargetType.AnyAlly) ? true : false);
			if (flag && Holder.IsInsideTree())
			{
				Vector2 size = GetViewport().GetVisibleRect().Size;
				Holder.SetTargetPosition(new Vector2(size.X / 2f, size.Y - Holder.Size.Y));
			}
			Cleanup();
			EmitSignal(SignalName.Finished, true);
			NCombatRoom.Instance?.Ui.Hand.TryGrabFocus();
		}
		else
		{
			CancelPlayCard();
		}
	}

	public void CancelPlayCard()
	{
		if (!_isTryingToPlayCard)
		{
			ClearTarget();
			Cleanup();
			EmitSignal(SignalName.Finished, false);
			OnCancelPlayCard();
		}
	}

	protected virtual void OnCancelPlayCard()
	{
	}

	protected virtual void Cleanup()
	{
		HideTargetingVisuals();
		HideEvokingOrbs();
		this.QueueFreeSafely();
	}

	protected void OnCreatureHover(NCreature creature)
	{
		CardNode?.SetPreviewTarget(creature.Entity);
	}

	protected void OnCreatureUnhover(NCreature _)
	{
		ClearTarget();
	}

	protected void CenterCard()
	{
		Vector2 size = _viewport.GetVisibleRect().Size;
		Holder.SetTargetPosition(new Vector2(size.X / 2f, size.Y - Holder.Hitbox.Size.Y * 0.75f / 2f));
		Holder.SetTargetScale(Vector2.One * 0.75f);
	}

	protected void CannotPlayThisCardFtueCheck(CardModel card)
	{
		if (!SaveManager.Instance.SeenFtue("cannot_play_card_ftue") && !card.CanPlay(out UnplayableReason reason, out AbstractModel _) && reason == UnplayableReason.EnergyCostTooHigh)
		{
			NModalContainer.Instance.Add(NCannotPlayCardFtue.Create());
			SaveManager.Instance.MarkFtueAsComplete("cannot_play_card_ftue");
		}
	}

	protected void HideTargetingVisuals()
	{
		foreach (NCreature creatureNode in NCombatRoom.Instance.CreatureNodes)
		{
			creatureNode.HideMultiselectReticle();
		}
		CardNode?.SetPreviewTarget(null);
		CardNode?.UpdateVisuals((Card?.Pile?.Type).GetValueOrDefault(), CardPreviewMode.Normal);
	}

	protected void ShowMultiCreatureTargetingVisuals()
	{
		if (CardNode == null || Card?.CombatState == null)
		{
			return;
		}
		TargetType targetType = Card.TargetType;
		if ((uint)(targetType - 3) <= 1u)
		{
			IReadOnlyList<Creature> hittableEnemies = Card.CombatState.HittableEnemies;
			if (hittableEnemies.Count == 1)
			{
				CardNode.SetPreviewTarget(hittableEnemies[0]);
			}
			CardNode.UpdateVisuals((CardNode.Model?.Pile?.Type).GetValueOrDefault(), CardPreviewMode.MultiCreatureTargeting);
			foreach (Creature item in hittableEnemies)
			{
				NCombatRoom.Instance.GetCreatureNode(item)?.ShowMultiselectReticle();
			}
		}
		if (Card.TargetType == TargetType.AllAllies)
		{
			IEnumerable<Creature> enumerable = Card.CombatState.PlayerCreatures.Where((Creature c) => c.IsAlive);
			{
				foreach (Creature item2 in enumerable)
				{
					NCombatRoom.Instance.GetCreatureNode(item2)?.ShowMultiselectReticle();
				}
				return;
			}
		}
		if (Card.TargetType == TargetType.Self)
		{
			NCombatRoom.Instance.GetCreatureNode(Card.Owner.Creature)?.ShowMultiselectReticle();
		}
		else if (Card.TargetType == TargetType.Osty)
		{
			NCombatRoom.Instance.GetCreatureNode(Card.Owner.Osty)?.ShowMultiselectReticle();
		}
	}

	private void AutoDisableCannotPlayCardFtueCheck()
	{
		_totalCardsPlayedForFtue++;
		if (_totalCardsPlayedForFtue == 8 && !SaveManager.Instance.SeenFtue("cannot_play_card_ftue"))
		{
			Log.Info("Cannot play cards FTUE was disabled, the player never saw it!!");
			SaveManager.Instance.MarkFtueAsComplete("cannot_play_card_ftue");
		}
	}

	protected void TryShowEvokingOrbs()
	{
		if (Card != null)
		{
			CardOwnerNode?.OrbManager?.UpdateVisuals(Card.OrbEvokeType);
		}
	}

	private void HideEvokingOrbs()
	{
		CardOwnerNode?.OrbManager?.UpdateVisuals(OrbEvokeType.None);
	}

	private void ClearTarget()
	{
		CardNode?.SetPreviewTarget(null);
	}
}
