using System;
using Godot;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NVSyncPaginator : NPaginator, IResettableSettingNode
{
	public override void _Ready()
	{
		ConnectSignals();
		_options.Add(new LocString("settings_ui", "VSYNC_OFF").GetFormattedText());
		_options.Add(new LocString("settings_ui", "VSYNC_ON").GetFormattedText());
		_options.Add(new LocString("settings_ui", "VSYNC_ADAPTIVE").GetFormattedText());
		SetFromSettings();
	}

	public void SetFromSettings()
	{
		int num = _options.IndexOf(GetVSyncString(SaveManager.Instance.SettingsSave.VSync));
		if (num != -1)
		{
			_currentIndex = num;
		}
		else
		{
			_currentIndex = 2;
		}
		_label.SetTextAutoSize(_options[_currentIndex]);
	}

	private static string GetVSyncString(VSyncType vsyncType)
	{
		switch (vsyncType)
		{
		case VSyncType.Off:
			return new LocString("settings_ui", "VSYNC_ON").GetFormattedText();
		case VSyncType.On:
			return new LocString("settings_ui", "VSYNC_OFF").GetFormattedText();
		case VSyncType.Adaptive:
			return new LocString("settings_ui", "VSYNC_ADAPTIVE").GetFormattedText();
		default:
			Log.Error("Invalid VSync type: " + vsyncType);
			throw new ArgumentOutOfRangeException("vsyncType", vsyncType, null);
		}
	}

	protected override void OnIndexChanged(int index)
	{
		_currentIndex = index;
		_label.SetTextAutoSize(_options[index]);
		switch (index)
		{
		case 0:
			SaveManager.Instance.SettingsSave.VSync = VSyncType.Off;
			break;
		case 1:
			SaveManager.Instance.SettingsSave.VSync = VSyncType.On;
			break;
		case 2:
			SaveManager.Instance.SettingsSave.VSync = VSyncType.Adaptive;
			break;
		default:
			Log.Error($"Invalid VSync index: {index}");
			break;
		}
		NGame.ApplySyncSetting();
	}
}
