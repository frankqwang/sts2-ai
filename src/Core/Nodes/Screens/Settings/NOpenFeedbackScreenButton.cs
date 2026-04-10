using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Localization;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NOpenFeedbackScreenButton : NSettingsButton
{
	private TextureRect _image;

	private MegaLabel _label;

	public override void _Ready()
	{
		ConnectSignals();
		_image = GetNode<TextureRect>("Image");
		_label = GetNode<MegaLabel>("Label");
		_label.SetTextAutoSize(new LocString("settings_ui", "SEND_FEEDBACK_BUTTON_LABEL").GetFormattedText());
	}
}
