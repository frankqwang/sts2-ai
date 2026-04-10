using Godot;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Nodes.CommonUi;

namespace MegaCrit.Sts2.Core.Nodes.Screens.InspectScreens;

public partial class NUpgradePreviewTickbox : NTickbox
{
	protected override string[] Hotkeys => new string[1] { MegaInput.accept };

	public override void _Ready()
	{
		ConnectSignals();
	}
}
