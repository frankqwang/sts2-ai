using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.UI;
using MegaCrit.Sts2.Core.Extensions;
using MegaCrit.Sts2.Core.Factories;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;

namespace MegaCrit.Sts2.Core.Nodes.Cards;

public partial class NTransformPreview : Control
{
	private Control _before;

	private Control _after;

	private Control _arrows;

	private CancellationTokenSource? _cancelTokenSource;

	public Vector2 SelectedCardPosition => _before.GlobalPosition;

	public override void _Ready()
	{
		_before = GetNode<Control>("%Before");
		_after = GetNode<Control>("%After");
		_arrows = GetNode<Control>("Arrows");
	}

	public override void _ExitTree()
	{
		_cancelTokenSource?.Cancel();
	}

	public void Initialize(IEnumerable<CardTransformation> cardTransformations)
	{
		RemoveExistingCards();
		_cancelTokenSource?.Cancel();
		List<CardTransformation> list = cardTransformations.ToList();
		float num = _before.GlobalPosition.X - 100f;
		float num2 = Math.Min(num / ((float)list.Count * 300f + (float)(list.Count - 1) * 30f), 1f);
		for (int i = 0; i < list.Count; i++)
		{
			CardTransformation cardTransformation = list[i];
			NPlayerHand nPlayerHand = NCombatRoom.Instance?.Ui.Hand;
			NPreviewCardHolder nPreviewCardHolder = NPreviewCardHolder.Create(NCard.Create(cardTransformation.Original), nPlayerHand == null, nPlayerHand != null);
			_before.AddChildSafely(nPreviewCardHolder);
			nPreviewCardHolder.FocusMode = FocusModeEnum.All;
			nPreviewCardHolder.CardNode.UpdateVisuals(cardTransformation.Original.Pile.Type, CardPreviewMode.Normal);
			nPreviewCardHolder.SetCardScale(Vector2.One * num2);
			int num3 = list.Count - i;
			nPreviewCardHolder.Position = new Vector2((0f - ((float)num3 - 0.5f)) * 300f * num2 - (float)(num3 - 1) * 30f, 0f);
			NCard card = ((cardTransformation.Replacement == null) ? NCard.Create(cardTransformation.Original) : NCard.Create(cardTransformation.Replacement));
			NPreviewCardHolder nPreviewCardHolder2 = NPreviewCardHolder.Create(card, showHoverTips: true, scaleOnHover: false);
			nPreviewCardHolder2.FocusMode = FocusModeEnum.None;
			_after.AddChildSafely(nPreviewCardHolder2);
			nPreviewCardHolder2.CardNode.UpdateVisuals(cardTransformation.Original.Pile.Type, CardPreviewMode.Normal);
			nPreviewCardHolder2.Scale = Vector2.One * num2;
			nPreviewCardHolder2.Position = new Vector2(((float)i + 0.5f) * 300f * num2 + (float)i * 30f, 0f);
			if (cardTransformation.Replacement == null)
			{
				nPreviewCardHolder2.Hitbox.MouseFilter = MouseFilterEnum.Ignore;
				TaskHelper.RunSafely(CycleThroughCards(possibleTransformations: (cardTransformation.ReplacementOptions == null) ? CardFactory.GetDefaultTransformationOptions(cardTransformation.Original, cardTransformation.IsInCombat) : cardTransformation.ReplacementOptions, holder: nPreviewCardHolder2, cardPile: cardTransformation.Original.Pile));
			}
		}
	}

	public void Uninitialize()
	{
		_cancelTokenSource?.Cancel();
	}

	private async Task CycleThroughCards(NPreviewCardHolder holder, CardPile cardPile, IEnumerable<CardModel> possibleTransformations)
	{
		_cancelTokenSource = new CancellationTokenSource();
		List<CardModel> cards = possibleTransformations.ToList();
		cards.UnstableShuffle(Rng.Chaotic);
		int cardIndex = 0;
		while (!_cancelTokenSource.IsCancellationRequested)
		{
			holder.ReassignToCard(cards[cardIndex], cardPile.Type, null, ModelVisibility.Visible);
			cardIndex++;
			if (cardIndex >= cards.Count)
			{
				cards.UnstableShuffle(Rng.Chaotic);
				cardIndex = 0;
			}
			if (SaveManager.Instance.PrefsSave.FastMode == FastModeType.Instant)
			{
				await Task.Delay(200);
			}
			else
			{
				await Cmd.Wait(0.2f, ignoreCombatEnd: true);
			}
		}
	}

	private void RemoveExistingCards()
	{
		foreach (Node child in _before.GetChildren())
		{
			child.QueueFreeSafely();
		}
		foreach (Node child2 in _after.GetChildren())
		{
			child2.QueueFreeSafely();
		}
	}
}
