using Godot;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NRunTimerTickbox : NSettingsTickbox, IResettableSettingNode
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
		base.IsTicked = SaveManager.Instance.PrefsSave.ShowRunTimer;
	}

	protected override void OnTick()
	{
		_settingsScreen.ShowToast(new LocString("settings_ui", "TOAST_RUN_TIMER_ON"));
		SaveManager.Instance.PrefsSave.ShowRunTimer = true;
		TryRefreshRunTimer();
	}

	protected override void OnUntick()
	{
		_settingsScreen.ShowToast(new LocString("settings_ui", "TOAST_RUN_TIMER_OFF"));
		SaveManager.Instance.PrefsSave.ShowRunTimer = false;
		TryRefreshRunTimer();
	}

	private static void TryRefreshRunTimer()
	{
		if (NRun.Instance != null)
		{
			NRun.Instance.GlobalUi.TopBar.Timer.RefreshVisibility();
		}
	}
}
