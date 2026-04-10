using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Cards;

namespace MegaCrit.Sts2.Core.Nodes.HoverTips;

public partial class NHoverTipCardContainer : Control
{
	private const string _cardHoverTipScenePath = "res://scenes/ui/card_hover_tip.tscn";

	private const float _padding = 4f;

	private IEnumerable<Control> Tips => GetChildren().OfType<Control>();

	public void Add(CardHoverTip cardTip)
	{
		Control control = PreloadManager.Cache.GetScene("res://scenes/ui/card_hover_tip.tscn").Instantiate<Control>(PackedScene.GenEditState.Disabled);
		this.AddChildSafely(control);
		NCard node = control.GetNode<NCard>("%Card");
		node.Model = cardTip.Card;
		node.UpdateVisuals(PileType.Deck, CardPreviewMode.Normal);
	}

	public void LayoutResizeAndReposition(Vector2 globalStartLocation, HoverTipAlignment alignment)
	{
		Vector2 size = NGame.Instance.GetViewportRect().Size;
		Vector2 size2 = Vector2.Zero;
		Vector2 zero = Vector2.Zero;
		float b = 0f;
		foreach (Control tip in Tips)
		{
			tip.Position = zero;
			size2 = new Vector2(Mathf.Max(zero.X + tip.Size.X, size2.X), Mathf.Max(zero.Y + tip.Size.Y, size2.Y));
			zero += Vector2.Down * (tip.Size.Y + 4f);
			b = Mathf.Max(tip.Size.X, b);
		}
		switch (alignment)
		{
		case HoverTipAlignment.Right:
			base.GlobalPosition = globalStartLocation;
			break;
		case HoverTipAlignment.Left:
			base.GlobalPosition = globalStartLocation + Vector2.Left * size2.X;
			break;
		}
		base.Size = size2;
	}
}
