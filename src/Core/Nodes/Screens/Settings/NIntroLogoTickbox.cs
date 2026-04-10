using Godot;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NIntroLogoTickbox : NSettingsTickbox, IResettableSettingNode
{
	private NSettingsScreen _settingsScreen;

	public override void _Ready()
	{
		ConnectSignals();
		_settingsScreen = this.GetAncestorOfType<NSettingsScreen>();
	}

	public void SetFromSettings()
	{
		base.IsTicked = SaveManager.Instance.SettingsSave.SkipIntroLogo;
	}

	protected override void OnTick()
	{
		_settingsScreen.ShowToast(new LocString("settings_ui", "TOAST_SKIP_INTRO_LOGO_ON"));
		SaveManager.Instance.SettingsSave.SkipIntroLogo = true;
	}

	protected override void OnUntick()
	{
		_settingsScreen.ShowToast(new LocString("settings_ui", "TOAST_SKIP_INTRO_LOGO_OFF"));
		SaveManager.Instance.SettingsSave.SkipIntroLogo = false;
	}
}
