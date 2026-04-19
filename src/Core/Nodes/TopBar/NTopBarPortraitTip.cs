using Godot;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.sts2.Core.Nodes.TopBar;

public partial class NTopBarPortraitTip : NClickableControl
{
	private IHoverTip _hoverTip;

	public bool ShowTip { get; private set; }

	public override void _Ready()
	{
		ConnectSignals();
	}

	public void Initialize(IRunState runState)
	{
		int ascensionLevel = runState.AscensionLevel;
		bool flag = runState.GameMode.AreAchievementsAndEpochsLocked();
		ShowTip = ascensionLevel > 0 || flag;
		if (ShowTip)
		{
			_hoverTip = AscensionHelper.GetHoverTip(LocalContext.GetMe(runState).Character, ascensionLevel, flag);
		}
		base.FocusMode = (FocusModeEnum)(ShowTip ? 2 : 0);
	}

	protected override void OnFocus()
	{
		if (ShowTip)
		{
			NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, _hoverTip);
			nHoverTipSet.GlobalPosition = base.GlobalPosition + new Vector2(0f, base.Size.Y + 20f);
		}
	}

	protected override void OnUnfocus()
	{
		NHoverTipSet.Remove(this);
	}
}
