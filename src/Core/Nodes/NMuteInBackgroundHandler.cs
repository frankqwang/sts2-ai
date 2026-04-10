using Godot;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes;

public partial class NMuteInBackgroundHandler : Node
{
	private Tween? _tween;

	public override void _Notification(int what)
	{
		if ((long)what == 1005)
		{
			Mute();
		}
		else if ((long)what == 1004)
		{
			Unmute();
		}
	}

	private void Mute()
	{
		PrefsSave prefsSave = SaveManager.Instance.PrefsSave;
		SettingsSave settingsSave = SaveManager.Instance.SettingsSave;
		if (prefsSave != null && settingsSave != null && prefsSave.MuteInBackground)
		{
			_tween = CreateTween();
			_tween.TweenMethod(Callable.From<float>(SetMasterVolume), settingsSave.VolumeMaster, 0f, 1.0);
		}
	}

	private void Unmute()
	{
		if (_tween != null)
		{
			_tween?.Kill();
			_tween = null;
			SetMasterVolume(SaveManager.Instance.SettingsSave.VolumeMaster);
		}
	}

	private static void SetMasterVolume(float volume)
	{
		NGame.Instance.AudioManager.SetMasterVol(volume);
		NGame.Instance.DebugAudio.SetMasterAudioVolume(volume);
	}
}
