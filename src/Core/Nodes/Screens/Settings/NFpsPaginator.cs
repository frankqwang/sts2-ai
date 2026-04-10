using Godot;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NFpsPaginator : NPaginator, IResettableSettingNode
{
	public override void _Ready()
	{
		ConnectSignals();
		_options.Add("24");
		_options.Add("30");
		_options.Add("59");
		_options.Add("60");
		_options.Add("75");
		_options.Add("90");
		_options.Add("120");
		_options.Add("144");
		_options.Add("165");
		_options.Add("240");
		_options.Add("360");
		_options.Add("500");
		SetFromSettings();
	}

	public void SetFromSettings()
	{
		int num = _options.IndexOf(SaveManager.Instance.SettingsSave.FpsLimit.ToString());
		_currentIndex = ((num != -1) ? num : 3);
		_label.SetTextAutoSize(_options[_currentIndex]);
	}

	protected override void OnIndexChanged(int index)
	{
		_currentIndex = index;
		_label.SetTextAutoSize(_options[index]);
		SaveManager.Instance.SettingsSave.FpsLimit = int.Parse(_options[index]);
		Log.Info($"FPS Limit: {SaveManager.Instance.SettingsSave.FpsLimit}");
		Engine.MaxFps = SaveManager.Instance.SettingsSave.FpsLimit;
	}
}
