using Godot;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NShowMpMapDrawingsTickbox : NSettingsTickbox, IResettableSettingNode
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
		base.IsTicked = SaveManager.Instance.PrefsSave.ShowMultiplayerDrawings;
	}

	protected override void OnTick()
	{
		_settingsScreen.ShowToast(new LocString("settings_ui", "TOAST_SHOW_MP_DRAWINGS_ON"));
		SaveManager.Instance.PrefsSave.ShowMultiplayerDrawings = true;
		TryRefreshMapDrawings();
	}

	protected override void OnUntick()
	{
		_settingsScreen.ShowToast(new LocString("settings_ui", "TOAST_SHOW_MP_DRAWINGS_OFF"));
		SaveManager.Instance.PrefsSave.ShowMultiplayerDrawings = false;
		TryRefreshMapDrawings();
	}

	private static void TryRefreshMapDrawings()
	{
		if (NMapScreen.Instance != null)
		{
			NMapScreen.Instance.Drawings.UpdateVisibilityFromSettings();
		}
	}
}
