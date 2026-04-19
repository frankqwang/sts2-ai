using Godot;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NPhobiaModeTickbox : NSettingsTickbox, IResettableSettingNode
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
		base.IsTicked = SaveManager.Instance.PrefsSave.PhobiaMode;
	}

	protected override void OnTick()
	{
		_settingsScreen.ShowToast(new LocString("settings_ui", "TOAST_PHOBIA_MODE_ON"));
		SaveManager.Instance.PrefsSave.PhobiaMode = true;
		NGame.Instance?.EmitSignal(NGame.SignalName.PhobiaModeToggled);
	}

	protected override void OnUntick()
	{
		_settingsScreen.ShowToast(new LocString("settings_ui", "TOAST_PHOBIA_MODE_OFF"));
		SaveManager.Instance.PrefsSave.PhobiaMode = false;
		NGame.Instance?.EmitSignal(NGame.SignalName.PhobiaModeToggled);
	}
}
