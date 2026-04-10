using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens;

public partial class NShareButton : NButton
{
	public override void _Ready()
	{
		ConnectSignals();
		Disable();
		GetNode<MegaLabel>("%ShareLabel").SetTextAutoSize(new LocString("main_menu_ui", "RUN_HISTORY.SHARE.title").GetFormattedText());
	}
}
