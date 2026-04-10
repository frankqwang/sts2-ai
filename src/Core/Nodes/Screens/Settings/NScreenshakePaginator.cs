using Godot;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NScreenshakePaginator : NPaginator, IResettableSettingNode
{
	public override void _Ready()
	{
		ConnectSignals();
		_options.Add(new LocString("settings_ui", "SCREENSHAKE_NONE").GetFormattedText());
		_options.Add(new LocString("settings_ui", "SCREENSHAKE_SOME").GetFormattedText());
		_options.Add(new LocString("settings_ui", "SCREENSHAKE_NORMAL").GetFormattedText());
		_options.Add(new LocString("settings_ui", "SCREENSHAKE_LOTS").GetFormattedText());
		_options.Add(new LocString("settings_ui", "SCREENSHAKE_CAAAW").GetFormattedText());
		SetFromSettings();
	}

	public void SetFromSettings()
	{
		_currentIndex = SaveManager.Instance.PrefsSave.ScreenShakeOptionIndex;
		if (_currentIndex >= _options.Count)
		{
			_label.SetTextAutoSize(">:P");
		}
		else
		{
			_label.SetTextAutoSize(_options[_currentIndex]);
		}
		NGame.Instance.SetScreenshakeMultiplier(GetShakeMultiplier(_currentIndex));
	}

	protected override void OnIndexChanged(int index)
	{
		if (_currentIndex >= _options.Count)
		{
			_currentIndex = 2;
		}
		_currentIndex = index;
		_label.SetTextAutoSize(_options[index]);
		SaveManager.Instance.PrefsSave.ScreenShakeOptionIndex = _currentIndex;
		Log.Info($"Screenshake set to: {_currentIndex}");
		NGame.Instance.SetScreenshakeMultiplier(GetShakeMultiplier(_currentIndex));
		NGame.Instance.ScreenShakeTrauma(ShakeStrength.Medium);
	}

	public static float GetShakeMultiplier(int index)
	{
		return index switch
		{
			0 => 0f, 
			1 => 0.5f, 
			2 => 1f, 
			3 => 2f, 
			4 => 4f, 
			_ => index, 
		};
	}
}
