using Godot;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NFullscreenTickbox : NSettingsTickbox
{
	public override void _Ready()
	{
		ConnectSignals();
		NGame.Instance.Connect(NGame.SignalName.WindowChange, Callable.From<bool>(OnWindowChange));
		OnWindowChange(SaveManager.Instance.SettingsSave.AspectRatioSetting == AspectRatioSetting.Auto);
		if (PlatformUtil.GetSupportedWindowMode().ShouldForceFullscreen())
		{
			Disable();
		}
	}

	protected override void OnTick()
	{
		SetFullscreen(fullscreen: true);
	}

	protected override void OnUntick()
	{
		SetFullscreen(fullscreen: false);
	}

	private void OnWindowChange(bool _)
	{
		base.IsTicked = SaveManager.Instance.SettingsSave.Fullscreen;
	}

	public static void SetFullscreen(bool fullscreen)
	{
		if (PlatformUtil.GetSupportedWindowMode().ShouldForceFullscreen() && !fullscreen)
		{
			Log.Warn($"Tried to go to windowed mode, but the current platform doesn't support it ({PlatformUtil.GetSupportedWindowMode()})");
			return;
		}
		int num = DisplayServer.WindowGetCurrentScreen();
		SettingsSave settingsSave = SaveManager.Instance.SettingsSave;
		if (fullscreen)
		{
			Log.Info($"Setting FULLSCREEN for display [{num}]");
			settingsSave.TargetDisplay = num;
			settingsSave.Fullscreen = true;
			settingsSave.WindowSize = DisplayServer.WindowGetSize();
			settingsSave.WindowPosition = new Vector2I(-1, -1);
		}
		else
		{
			Log.Info($"Exiting FULLSCREEN for display [{num}]");
			if (settingsSave.WindowSize >= DisplayServer.ScreenGetSize(num))
			{
				settingsSave.WindowSize = DisplayServer.ScreenGetSize(num) - new Vector2I(8, 48);
				settingsSave.WindowPosition = new Vector2I(4, 44);
			}
			settingsSave.Fullscreen = false;
		}
		NGame.Instance.ApplyDisplaySettings();
	}
}
