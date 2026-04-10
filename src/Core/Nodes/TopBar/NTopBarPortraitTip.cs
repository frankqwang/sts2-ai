using Godot;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.sts2.Core.Nodes.TopBar;

public partial class NTopBarPortraitTip : TextureRect
{
	private IHoverTip _hoverTip;

	private bool _showTip;

	public override void _Ready()
	{
		Connect(Control.SignalName.MouseEntered, Callable.From(OnHovered));
		Connect(Control.SignalName.MouseExited, Callable.From(OnUnhovered));
	}

	public void Initialize(IRunState runState)
	{
		int ascensionLevel = runState.AscensionLevel;
		_showTip = ascensionLevel > 0;
		if (_showTip)
		{
			_hoverTip = AscensionHelper.GetHoverTip(LocalContext.GetMe(runState).Character, ascensionLevel);
		}
	}

	private void OnHovered()
	{
		if (_showTip)
		{
			NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, _hoverTip);
			nHoverTipSet.GlobalPosition = base.GlobalPosition + new Vector2(0f, base.Size.Y + 20f);
		}
	}

	private void OnUnhovered()
	{
		NHoverTipSet.Remove(this);
	}
}
