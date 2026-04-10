using Godot;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NHandCardCountTickbox : NSettingsTickbox, IResettableSettingNode
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
		base.IsTicked = SaveManager.Instance.PrefsSave.ShowCardIndices;
	}

	protected override void OnTick()
	{
		_settingsScreen.ShowToast(new LocString("settings_ui", "TOAST_HAND_CARD_COUNT_ON"));
		SaveManager.Instance.PrefsSave.ShowCardIndices = true;
		TryRefreshHandIndices();
	}

	protected override void OnUntick()
	{
		_settingsScreen.ShowToast(new LocString("settings_ui", "TOAST_HAND_CARD_COUNT_OFF"));
		SaveManager.Instance.PrefsSave.ShowCardIndices = false;
		TryRefreshHandIndices();
	}

	private static void TryRefreshHandIndices()
	{
		if (NPlayerHand.Instance != null)
		{
			NPlayerHand.Instance.ForceRefreshCardIndices();
		}
	}
}
