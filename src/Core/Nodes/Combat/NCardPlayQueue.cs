using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Actions;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Cards;
using MegaCrit.Sts2.Core.Models.Exceptions;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Multiplayer;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NCardPlayQueue : Control
{
	private class QueueItem
	{
		public required NCard card;

		public required GameAction action;

		public Tween? currentTween;
	}

	private List<QueueItem> _playQueue = new List<QueueItem>();

	public static NCardPlayQueue? Instance => NCombatRoom.Instance?.Ui.PlayQueue;

	public override void _Ready()
	{
		RunManager.Instance.ActionQueueSet.ActionEnqueued += OnActionEnqueued;
	}

	public override void _ExitTree()
	{
		RunManager.Instance.ActionQueueSet.ActionEnqueued -= OnActionEnqueued;
		_playQueue.Clear();
	}

	public void OnLocalCardPlayed(PlayCardAction action, NCardHolder? holder, CardModel card)
	{
		NCard nCard = holder?.CardNode ?? NCard.Create(card);
		CardModel? model = nCard.Model;
		if (model != null && model.Pile?.Type == PileType.Hand)
		{
			QueueItem item = new QueueItem
			{
				card = nCard,
				action = action
			};
			if (nCard.IsInsideTree())
			{
				nCard.Reparent(this);
			}
			else
			{
				this.AddChildSafely(nCard);
			}
			MoveChild(nCard, 0);
			if (holder != null && holder.IsValid())
			{
				NPlayerHand.Instance.RemoveCardHolder(holder);
			}
			_playQueue.Add(item);
			TweenCardToQueuePosition(item, _playQueue.Count - 1);
		}
	}

	private void OnActionEnqueued(GameAction action)
	{
		if (!(action is PlayCardAction { NetCombatCard: var netCombatCard } playCardAction))
		{
			return;
		}
		CardModel cardModel = netCombatCard.ToCardModelOrNull();
		if (cardModel == null)
		{
			try
			{
				cardModel = ModelDb.GetById<CardModel>(playCardAction.CardModelId);
			}
			catch (ModelNotFoundException)
			{
				cardModel = ModelDb.Card<DeprecatedCard>();
			}
		}
		if (LocalContext.IsMe(playCardAction.Player))
		{
			NCardHolder cardHolder = NPlayerHand.Instance.GetCardHolder(cardModel);
			OnLocalCardPlayed(playCardAction, cardHolder, cardModel);
			return;
		}
		NCreature creatureNode = NCombatRoom.Instance.GetCreatureNode(playCardAction.Player.Creature);
		NMultiplayerPlayerIntentHandler playerIntentHandler = creatureNode.PlayerIntentHandler;
		NCard nCard = NCard.Create(cardModel);
		Vector2 globalPosition = playerIntentHandler.CardIntent.GlobalPosition + playerIntentHandler.CardIntent.Size * 0.5f;
		nCard.GlobalPosition = globalPosition;
		nCard.Scale = Vector2.One * 0.25f;
		this.AddChildSafely(nCard);
		MoveChild(nCard, 0);
		QueueItem item = new QueueItem
		{
			card = nCard,
			action = playCardAction
		};
		_playQueue.Add(item);
		UpdateCardVisuals(item);
		TweenCardToQueuePosition(item, _playQueue.Count - 1);
	}

	public void ReAddCardAfterPlayerChoice(NCard card, GameAction action)
	{
		if (action.State == GameActionState.Executing)
		{
			card.Reparent(NCombatRoom.Instance.Ui.PlayContainer);
			card.AnimCardToPlayPile();
			return;
		}
		QueueItem item = new QueueItem
		{
			card = card,
			action = action
		};
		card.Reparent(this);
		MoveChild(card, 0);
		_playQueue.Add(item);
		TweenCardToQueuePosition(item, _playQueue.Count - 1);
		action.BeforeResumedAfterPlayerChoice += BeforeRemoteCardPlayResumedAfterPlayerChoice;
	}

	private void BeforeRemoteCardPlayResumedAfterPlayerChoice(GameAction action)
	{
		action.BeforeResumedAfterPlayerChoice -= BeforeRemoteCardPlayResumedAfterPlayerChoice;
		int num = _playQueue.FindIndex((QueueItem i) => i.action == action);
		if (num >= 0)
		{
			QueueItem queueItem = _playQueue[num];
			RemoveCardFromQueue(num);
			queueItem.card.Reparent(NCombatRoom.Instance.Ui.PlayContainer);
			queueItem.card.AnimCardToPlayPile();
		}
	}

	public void RemoveCardFromQueueForCancellation(PlayCardAction action)
	{
		int num = _playQueue.FindIndex((QueueItem i) => i.action == action);
		if (num >= 0)
		{
			RemoveCardFromQueueForCancellation(num);
		}
	}

	public void RemoveCardFromQueueForCancellation(NCard card, bool forceReturnToHand = false)
	{
		int num = _playQueue.FindIndex((QueueItem i) => i.card == card);
		if (num >= 0)
		{
			RemoveCardFromQueueForCancellation(num, forceReturnToHand);
		}
	}

	private void RemoveCardFromQueueForCancellation(int index, bool forceReturnToHand = false)
	{
		QueueItem queueItem = _playQueue[index];
		RemoveCardFromQueue(index);
		if (queueItem.action.OwnerId == LocalContext.NetId)
		{
			CardModel? model = queueItem.card.Model;
			if ((model != null && model.Pile?.Type == PileType.Hand) || forceReturnToHand)
			{
				NPlayerHand.Instance.Add(queueItem.card);
			}
			else
			{
				TweenCardForCancellation(queueItem);
			}
		}
		else
		{
			TweenCardForCancellation(queueItem);
		}
	}

	public void UpdateCardBeforeExecution(PlayCardAction playCardAction)
	{
		int num = _playQueue.FindIndex((QueueItem i) => i.action == playCardAction);
		if (num < 0)
		{
			return;
		}
		QueueItem queueItem = _playQueue[num];
		queueItem.card.Model = playCardAction.NetCombatCard.ToCardModel();
		UpdateCardVisuals(queueItem);
		if (LocalContext.IsMe(queueItem.card.Model.Owner))
		{
			NCardHolder nCardHolder = NPlayerHand.Instance?.GetCardHolder(queueItem.card.Model);
			if (nCardHolder != null)
			{
				NPlayerHand.Instance?.Remove(queueItem.card.Model);
			}
		}
	}

	public void RemoveCardFromQueueForExecution(CardModel card)
	{
		int num = _playQueue.FindIndex((QueueItem i) => i.card.Model == card);
		if (num < 0)
		{
			throw new InvalidOperationException();
		}
		RemoveCardFromQueue(num);
	}

	private void UpdateCardVisuals(QueueItem item)
	{
		if (item.action is PlayCardAction playCardAction)
		{
			item.card.SetPreviewTarget(playCardAction.Target);
		}
		item.card.UpdateVisuals(item.card.Model.Pile?.Type ?? PileType.None, CardPreviewMode.Normal);
	}

	private void RemoveCardFromQueue(NCard card)
	{
		int num = _playQueue.FindIndex((QueueItem i) => i.card == card);
		if (num >= 0)
		{
			RemoveCardFromQueue(num);
		}
	}

	private void RemoveCardFromQueue(int index)
	{
		QueueItem queueItem = _playQueue[index];
		queueItem.currentTween?.Kill();
		_playQueue.RemoveAt(index);
		TweenAllToQueuePosition();
	}

	private void TweenAllToQueuePosition()
	{
		for (int i = 0; i < _playQueue.Count; i++)
		{
			TweenCardToQueuePosition(_playQueue[i], i);
		}
	}

	public NCard? GetCardNode(CardModel card)
	{
		return _playQueue.FirstOrDefault((QueueItem i) => i.card.Model == card)?.card;
	}

	public void AnimOut()
	{
		foreach (QueueItem item in _playQueue)
		{
			item.currentTween?.Kill();
			if (item.action.OwnerId == LocalContext.NetId)
			{
				CardModel? model = item.card.Model;
				if (model != null && model.Pile?.Type == PileType.Hand)
				{
					NPlayerHand.Instance.Add(item.card);
					continue;
				}
			}
			TweenCardForCancellation(item);
		}
		_playQueue.Clear();
	}

	private void TweenCardForCancellation(QueueItem item)
	{
		item.currentTween?.Kill();
		item.currentTween = CreateTween().SetParallel();
		item.currentTween.TweenProperty(item.card, "position:y", 30f, 0.5).AsRelative().SetEase(Tween.EaseType.Out)
			.SetTrans(Tween.TransitionType.Cubic);
		item.currentTween.TweenProperty(item.card, "modulate:a", 0f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		item.currentTween.Chain().TweenCallback(Callable.From(item.card.QueueFreeSafely));
		item.currentTween.Play();
	}

	private void TweenCardToQueuePosition(QueueItem item, int queueIndex)
	{
		item.currentTween?.Kill();
		item.currentTween = CreateTween().SetParallel();
		item.currentTween.TweenProperty(item.card, "position", GetPositionForQueueIndex(item.card, queueIndex), 0.3499999940395355).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		item.currentTween.TweenProperty(item.card, "scale", GetScaleForQueueIndex(queueIndex), 0.3499999940395355).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		item.currentTween.TweenProperty(item.card, "modulate:a", 1f, 0.3499999940395355).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		item.currentTween.Play();
	}

	private Vector2 GetScaleForQueueIndex(int index)
	{
		index++;
		float num = 1f - (float)index / (float)(index + 1);
		return num * Vector2.One * 0.8f;
	}

	private Vector2 GetPositionForQueueIndex(NCard card, int index)
	{
		index++;
		float num = (float)index / (float)(index + 2);
		return PileType.Play.GetTargetPosition(card) + Vector2.Left * 300f * num;
	}
}
