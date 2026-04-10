using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Localization.Fonts;
using MegaCrit.Sts2.Core.Platform;

namespace MegaCrit.Sts2.Core.Nodes.GodotExtensions;

public partial class NMegaLineEdit : LineEdit
{
	public override void _Ready()
	{
		this.ApplyLocaleFontSubstitution(FontType.Regular, ThemeConstants.LineEdit.font);
	}

	public override void _GuiInput(InputEvent inputEvent)
	{
		base._GuiInput(inputEvent);
		if (inputEvent is InputEventMouseButton inputEventMouseButton && inputEventMouseButton.ButtonIndex == MouseButton.Left && inputEventMouseButton.IsPressed())
		{
			OpenKeyboard();
		}
		if (inputEvent.IsActionPressed(MegaInput.select))
		{
			OpenKeyboard();
		}
		if (inputEvent.IsActionPressed(MegaInput.cancel) && IsEditing())
		{
			Unedit();
			GetViewport()?.SetInputAsHandled();
			PlatformUtil.CloseVirtualKeyboard();
		}
	}

	private void OpenKeyboard()
	{
		Edit();
		PlatformUtil.OpenVirtualKeyboard();
	}
}
