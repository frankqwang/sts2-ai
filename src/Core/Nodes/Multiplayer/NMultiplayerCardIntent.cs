using Godot;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;

namespace MegaCrit.Sts2.Core.Nodes.Multiplayer;

public partial class NMultiplayerCardIntent : Control
{
	private NCard _cardNode;

	private CardModel? _card;

	public CardModel? Card
	{
		get
		{
			return _card;
		}
		set
		{
			_cardNode.Model = value;
			_card = value;
			if (_card != null && IsNodeReady())
			{
				_cardNode.UpdateVisuals(_card.Pile?.Type ?? PileType.None, CardPreviewMode.Normal);
			}
		}
	}

	public override void _Ready()
	{
		_cardNode = GetNode<NCard>("%Card");
		if (_card != null)
		{
			_cardNode.UpdateVisuals(_card.Pile?.Type ?? PileType.None, CardPreviewMode.Normal);
		}
	}
}
