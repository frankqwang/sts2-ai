using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Localization;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NResetToDefaultControlsButton : NSettingsButton
{
	private Control _image;

	private MegaRichTextLabel _label;

	public override void _Ready()
	{
		ConnectSignals();
		_image = GetNode<Control>("Image");
		_label = GetNode<MegaRichTextLabel>("Label");
		_label.SetTextAutoSize(new LocString("settings_ui", "INPUT_SETTINGS.RESET_TO_DEFAULT").GetFormattedText());
	}
}
