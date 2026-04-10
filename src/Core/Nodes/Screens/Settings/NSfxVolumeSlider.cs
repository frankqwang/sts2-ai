using Godot;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.Nodes.Audio;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NSfxVolumeSlider : NSettingsSlider
{
	public override void _Ready()
	{
		ConnectSignals();
		_slider.SetValueWithoutAnimation(SaveManager.Instance.SettingsSave.VolumeSfx * 100f);
		_slider.Connect(Range.SignalName.ValueChanged, Callable.From<double>(OnValueChanged));
		_slider.Connect(NSlider.SignalName.MouseReleased, Callable.From<bool>(OnDragEnded));
	}

	private static void OnValueChanged(double value)
	{
		float num = (float)value / 100f;
		SaveManager.Instance.SettingsSave.VolumeSfx = num;
		NAudioManager.Instance?.SetSfxVol(num);
		NDebugAudioManager.Instance?.SetSfxAudioVolume(num);
	}

	private static void OnDragEnded(bool valueChanged)
	{
		if (valueChanged)
		{
			NDebugAudioManager.Instance?.Play("dagger_throw.mp3");
		}
	}
}
