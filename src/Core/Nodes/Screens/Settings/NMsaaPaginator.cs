using Godot;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NMsaaPaginator : NPaginator, IResettableSettingNode
{
	public override void _Ready()
	{
		ConnectSignals();
		_options.Add("0");
		_options.Add("2");
		_options.Add("4");
		_options.Add("8");
		SetFromSettings();
	}

	public void SetFromSettings()
	{
		int num = _options.IndexOf(SaveManager.Instance.SettingsSave.Msaa.ToString());
		_currentIndex = ((num != -1) ? num : 3);
		_label.SetTextAutoSize(GetMsaaLabel(int.Parse(_options[_currentIndex])));
	}

	protected override void OnIndexChanged(int index)
	{
		_currentIndex = index;
		_label.SetTextAutoSize(GetMsaaLabel(int.Parse(_options[index])));
		SaveManager.Instance.SettingsSave.Msaa = int.Parse(_options[index]);
		Log.Info("MSAA: " + _label.Text);
		RenderingServer.ViewportSetMsaa2D(GetViewport().GetViewportRid(), GetMsaa(SaveManager.Instance.SettingsSave.Msaa));
	}

	private string GetMsaaLabel(int msaaAmount)
	{
		return msaaAmount switch
		{
			2 => "2x", 
			4 => "4x", 
			8 => "8x", 
			_ => new LocString("settings_ui", "MSAA_NONE").GetFormattedText(), 
		};
	}

	private RenderingServer.ViewportMsaa GetMsaa(int index)
	{
		return index switch
		{
			2 => RenderingServer.ViewportMsaa.Msaa2X, 
			4 => RenderingServer.ViewportMsaa.Msaa4X, 
			8 => RenderingServer.ViewportMsaa.Msaa8X, 
			_ => RenderingServer.ViewportMsaa.Disabled, 
		};
	}
}
