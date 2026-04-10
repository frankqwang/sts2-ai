using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Screens.Credits;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NCreditsButton : NSettingsButton
{
	private TextureRect _image;

	private MegaLabel _label;

	public override void _Ready()
	{
		ConnectSignals();
		_image = GetNode<TextureRect>("Image");
		_label = GetNode<MegaLabel>("Label");
		_label.SetTextAutoSize(new LocString("settings_ui", "CREDITS_BUTTON_LABEL").GetFormattedText());
	}

	protected override void OnRelease()
	{
		base.OnRelease();
		NCreditsScreen nCreditsScreen = NCreditsScreen.Create();
		if (nCreditsScreen != null)
		{
			NModalContainer.Instance.Add(nCreditsScreen, showBackstop: false);
		}
	}
}
