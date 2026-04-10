using Godot;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NBackgroundModeTickbox : NSettingsTickbox, IResettableSettingNode
{
	private NSettingsScreen _settingsScreen;

	public override void _Ready()
	{
		ConnectSignals();
		_settingsScreen = this.GetAncestorOfType<NSettingsScreen>();
		SetFromSettings();
	}

	public void SetFromSettings()
	{
		base.IsTicked = SaveManager.Instance.SettingsSave.LimitFpsInBackground;
	}

	protected override void OnTick()
	{
		_settingsScreen.ShowToast(new LocString("settings_ui", "TOAST_LIMIT_FPS_IN_BACKGROUND_ON"));
		SaveManager.Instance.SettingsSave.LimitFpsInBackground = true;
	}

	protected override void OnUntick()
	{
		_settingsScreen.ShowToast(new LocString("settings_ui", "TOAST_LIMIT_FPS_IN_BACKGROUND_OFF"));
		SaveManager.Instance.SettingsSave.LimitFpsInBackground = false;
	}
}
