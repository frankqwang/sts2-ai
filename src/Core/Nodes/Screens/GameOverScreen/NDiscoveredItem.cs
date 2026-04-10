using Godot;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;

namespace MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;

public partial class NDiscoveredItem : NButton
{
	private HoverTip _hoverTip;

	private static readonly Vector2 _hoverTipOffset = new Vector2(-76f, -200f);

	public override void _Ready()
	{
		ConnectSignals();
	}

	public void SetHoverTip(HoverTip hoverTip)
	{
		_hoverTip = hoverTip;
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		NHoverTipSet.CreateAndShow(this, _hoverTip).GlobalPosition = base.GlobalPosition + _hoverTipOffset;
	}

	protected override void OnUnfocus()
	{
		NHoverTipSet.Remove(this);
	}

	protected override void OnPress()
	{
	}
}
