using Godot;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NLongPressConfirmationTickbox : NSettingsTickbox, IResettableSettingNode
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
		base.IsTicked = SaveManager.Instance.PrefsSave.IsLongPressEnabled;
	}

	protected override void OnTick()
	{
		SaveManager.Instance.PrefsSave.IsLongPressEnabled = true;
	}

	protected override void OnUntick()
	{
		SaveManager.Instance.PrefsSave.IsLongPressEnabled = false;
	}
}
