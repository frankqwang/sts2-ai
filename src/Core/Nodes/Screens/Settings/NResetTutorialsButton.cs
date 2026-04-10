using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NResetTutorialsButton : NSettingsButton
{
	private readonly LocString _header = new LocString("settings_ui", "TUTORIAL_RESET_POPUP_HEADER");

	private readonly LocString _description = new LocString("settings_ui", "TUTORIAL_RESET_POPUP_DESCRIPTION");

	private TextureRect _image;

	private MegaLabel _label;

	public override void _Ready()
	{
		if (!NGame.IsReleaseGame())
		{
			GetNode<Control>("%ResetTutorialDivider").Visible = true;
			GetNode<Control>("%ResetTutorials").Visible = true;
		}
		ConnectSignals();
		_image = GetNode<TextureRect>("Image");
		_label = GetNode<MegaLabel>("Label");
		_label.SetTextAutoSize(new LocString("settings_ui", "TUTORIAL_RESET_BUTTON_LABEL").GetFormattedText());
		Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OpenPopup));
	}

	private void OpenPopup(NButton _)
	{
		NModalContainer.Instance.Add(NSettingsScreenPopup.Create(_header, _description));
	}
}
