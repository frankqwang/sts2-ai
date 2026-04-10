using Godot;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.HoverTips;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NFullscreenHoverTip : Control
{
	private IHoverTip _hoverTip;

	public override void _Ready()
	{
		Connect(Control.SignalName.MouseEntered, Callable.From(OnHovered));
		Connect(Control.SignalName.MouseExited, Callable.From(OnUnhovered));
		_hoverTip = new HoverTip(new LocString("settings_ui", "FULLSCREEN_HEADER"), new LocString("settings_ui", "FULLSCREEN_DESCRIPTION"));
	}

	private void OnHovered()
	{
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, _hoverTip);
		nHoverTipSet.GlobalPosition = base.GlobalPosition + NSettingsScreen.settingTipsOffset;
	}

	private void OnUnhovered()
	{
		NHoverTipSet.Remove(this);
	}
}
