using Godot;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.Nodes.Audio;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NMasterVolumeSlider : NSettingsSlider
{
	public override void _Ready()
	{
		ConnectSignals();
		_slider.SetValueWithoutAnimation(SaveManager.Instance.SettingsSave.VolumeMaster * 100f);
		_slider.Connect(Range.SignalName.ValueChanged, Callable.From<double>(OnValueChanged));
	}

	private void OnValueChanged(double value)
	{
		float num = (float)value * 0.01f;
		SaveManager.Instance.SettingsSave.VolumeMaster = num;
		NAudioManager.Instance?.SetMasterVol(num);
		NDebugAudioManager.Instance?.SetMasterAudioVolume(num);
	}
}
