using Godot;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;

namespace MegaCrit.Sts2.Core.Nodes.Screens.DailyRun;

public partial class NDailyRunScoreWarning : NClickableControl
{
	private static readonly LocString _hoverTipTitle = new LocString("main_menu_ui", "DAILY_RUN_MENU.NO_UPLOAD_HOVERTIP.title");

	private static readonly LocString _hoverTipDescription = new LocString("main_menu_ui", "DAILY_RUN_MENU.NO_UPLOAD_HOVERTIP.description");

	private static readonly HoverTip _hoverTip = new HoverTip(_hoverTipTitle, _hoverTipDescription);

	public override void _Ready()
	{
		ConnectSignals();
	}

	protected override void OnFocus()
	{
		NHoverTipSet.CreateAndShow(this, new global::_003C_003Ez__ReadOnlySingleElementList<IHoverTip>(_hoverTip), HoverTipAlignment.Left);
		base.Scale = Vector2.One * 1.1f;
	}

	protected override void OnUnfocus()
	{
		NHoverTipSet.Remove(this);
		base.Scale = Vector2.One;
	}
}
