using Godot;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;

namespace MegaCrit.Sts2.Core.Nodes.Cards;

public partial class NUpgradePreview : Control
{
	private Control _before;

	private Control _after;

	private Control _arrows;

	private CardModel? _card;

	public Vector2 SelectedCardPosition => _before.GlobalPosition;

	public CardModel? Card
	{
		get
		{
			return _card;
		}
		set
		{
			_card = value;
			Reload();
		}
	}

	public override void _Ready()
	{
		_before = GetNode<Control>("%Before");
		_after = GetNode<Control>("%After");
		_arrows = GetNode<Control>("Arrows");
	}

	private void Reload()
	{
		RemoveExistingCards();
		_arrows.Visible = Card != null;
		if (Card != null)
		{
			NPlayerHand nPlayerHand = NCombatRoom.Instance?.Ui.Hand;
			NPreviewCardHolder nPreviewCardHolder = NPreviewCardHolder.Create(NCard.Create(Card), nPlayerHand == null, nPlayerHand != null);
			_before.AddChildSafely(nPreviewCardHolder);
			nPreviewCardHolder.FocusMode = FocusModeEnum.All;
			nPreviewCardHolder.CardNode.UpdateVisuals(Card.Pile.Type, CardPreviewMode.Normal);
			if (nPlayerHand != null)
			{
				nPreviewCardHolder.Connect(NCardHolder.SignalName.Pressed, Callable.From<NCardHolder>(ReturnCard));
			}
			CardModel cardModel = Card.CardScope.CloneCard(Card);
			cardModel.UpgradeInternal();
			cardModel.UpgradePreviewType = ((!Card.Pile.IsCombatPile) ? CardUpgradePreviewType.Deck : CardUpgradePreviewType.Combat);
			NPreviewCardHolder nPreviewCardHolder2 = NPreviewCardHolder.Create(NCard.Create(cardModel), showHoverTips: true, scaleOnHover: false);
			nPreviewCardHolder2.FocusMode = FocusModeEnum.None;
			_after.AddChildSafely(nPreviewCardHolder2);
			nPreviewCardHolder2.CardNode.ShowUpgradePreview();
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

	private void ReturnCard(NCardHolder holder)
	{
		holder.Pressed -= ReturnCard;
		NCombatRoom.Instance?.Ui.Hand.DeselectCard(holder.CardNode);
		Card = null;
	}
}
