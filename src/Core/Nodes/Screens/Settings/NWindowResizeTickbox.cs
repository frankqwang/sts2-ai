using Godot;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NWindowResizeTickbox : NSettingsTickbox, IResettableSettingNode
{
	private NSettingsScreen _settingsScreen;

	public override void _Ready()
	{
		ConnectSignals();
		_settingsScreen = this.GetAncestorOfType<NSettingsScreen>();
		NGame.Instance.Connect(NGame.SignalName.WindowChange, Callable.From<bool>(OnWindowChange));
		SetFromSettings();
		RefreshEnabled();
	}

	protected override void OnTick()
	{
		_settingsScreen.ShowToast(new LocString("settings_ui", "TOAST_WINDOW_RESIZING_ON"));
		SaveManager.Instance.SettingsSave.ResizeWindows = true;
		NGame.Instance.ApplyDisplaySettings();
	}

	protected override void OnUntick()
	{
		_settingsScreen.ShowToast(new LocString("settings_ui", "TOAST_WINDOW_RESIZING_OFF"));
		SaveManager.Instance.SettingsSave.ResizeWindows = false;
		NGame.Instance.ApplyDisplaySettings();
	}

	public void SetFromSettings()
	{
		base.IsTicked = SaveManager.Instance.SettingsSave.ResizeWindows;
	}

	private void OnWindowChange(bool isAutoAspectRatio)
	{
		RefreshEnabled();
	}

	private void RefreshEnabled()
	{
		if (SaveManager.Instance.SettingsSave.Fullscreen || PlatformUtil.GetSupportedWindowMode().ShouldForceFullscreen())
		{
			Disable();
		}
		else
		{
			Enable();
		}
	}
}
