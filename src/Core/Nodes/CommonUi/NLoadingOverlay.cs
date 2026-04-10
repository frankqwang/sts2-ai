using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Localization;

namespace MegaCrit.Sts2.Core.Nodes.CommonUi;

public partial class NLoadingOverlay : Control
{
	public override void _Ready()
	{
		GetNode<MegaLabel>("%Label").SetTextAutoSize(new LocString("main_menu_ui", "LOADING_OVERLAY.label").GetFormattedText());
	}
}
