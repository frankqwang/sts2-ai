using Godot;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NMuteInBackgroundTickbox : NSettingsTickbox
{
	private NSettingsScreen _settingsScreen;

	public override void _Ready()
	{
		ConnectSignals();
		_settingsScreen = this.GetAncestorOfType<NSettingsScreen>();
		base.IsTicked = SaveManager.Instance.PrefsSave.MuteInBackground;
	}

	protected override void OnTick()
	{
		_settingsScreen.ShowToast(new LocString("settings_ui", "TOAST_MUTE_IN_BACKGROUND_ON"));
		SaveManager.Instance.PrefsSave.MuteInBackground = true;
	}

	protected override void OnUntick()
	{
		_settingsScreen.ShowToast(new LocString("settings_ui", "TOAST_MUTE_IN_BACKGROUND_OFF"));
		SaveManager.Instance.PrefsSave.MuteInBackground = false;
	}
}
