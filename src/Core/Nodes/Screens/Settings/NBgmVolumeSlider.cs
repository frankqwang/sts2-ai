using Godot;
using MegaCrit.Sts2.Core.Nodes.Audio;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NBgmVolumeSlider : NSettingsSlider
{
	public override void _Ready()
	{
		ConnectSignals();
		_slider.SetValueWithoutAnimation(SaveManager.Instance.SettingsSave.VolumeBgm * 100f);
		_slider.Connect(Range.SignalName.ValueChanged, Callable.From<double>(OnValueChanged));
	}

	private void OnValueChanged(double value)
	{
		float num = (float)(value / 100.0);
		SaveManager.Instance.SettingsSave.VolumeBgm = num;
		NAudioManager.Instance?.SetBgmVol(num);
	}
}
