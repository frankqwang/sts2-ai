using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Localization;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NOpenModdingScreenButton : NSettingsButton
{
	private TextureRect _image;

	private MegaLabel _label;

	public override void _Ready()
	{
		ConnectSignals();
		base.PivotOffset = base.Size * 0.5f;
		_image = GetNode<TextureRect>("Image");
		_label = GetNode<MegaLabel>("Label");
		_label.SetTextAutoSize(new LocString("settings_ui", "MODDING_SCREEN_BUTTON_LABEL").GetFormattedText());
	}

	protected override void OnEnable()
	{
		base.OnEnable();
		_image.Modulate = Colors.White;
	}

	protected override void OnDisable()
	{
		base.OnDisable();
		_image.Modulate = Colors.Gray;
	}
}
