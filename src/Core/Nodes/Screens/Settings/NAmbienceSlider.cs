using Godot;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.Nodes.Audio;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NAmbienceSlider : NSettingsSlider
{
	public override void _Ready()
	{
		ConnectSignals();
		_slider.SetValueWithoutAnimation(SaveManager.Instance.SettingsSave.VolumeAmbience * 100f);
		_slider.Connect(Range.SignalName.ValueChanged, Callable.From<double>(OnValueChanged));
	}

	private static void OnValueChanged(double value)
	{
		float num = (float)value / 100f;
		SaveManager.Instance.SettingsSave.VolumeAmbience = num;
		NAudioManager.Instance?.SetAmbienceVol(num);
		NDebugAudioManager.Instance?.SetSfxAudioVolume(num);
	}
}
