using Godot;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Potions;

public partial class NPotionShortcutButton : NButton
{
	protected override string[] Hotkeys => new string[1] { MegaInput.topPanel };

	public override void _Ready()
	{
		ConnectSignals();
	}
}
