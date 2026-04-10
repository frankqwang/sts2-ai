using Godot;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;

public partial class NPatchNotesButton : NButton
{
	private static readonly StringName _v = new StringName("v");

	private ShaderMaterial _hsv;

	private Control _icon;

	protected override string[] Hotkeys => new string[1] { MegaInput.accept };

	public override void _Ready()
	{
		ConnectSignals();
		_icon = GetNode<TextureRect>("Icon");
		_hsv = (ShaderMaterial)_icon.Material;
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_hsv.SetShaderParameter(_v, 1.2f);
		_icon.RotationDegrees = 5f;
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_hsv.SetShaderParameter(_v, 1f);
		_icon.RotationDegrees = 0f;
	}
}
