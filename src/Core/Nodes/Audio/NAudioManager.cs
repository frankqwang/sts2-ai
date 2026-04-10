using System.Collections.Generic;
using Godot;
using Godot.Collections;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Audio;

public partial class NAudioManager : Node
{
	private static readonly StringName _setBgmVolume = new StringName("set_bgm_volume");

	private static readonly StringName _setAmbienceVolume = new StringName("set_ambience_volume");

	private static readonly StringName _setSfxVolume = new StringName("set_sfx_volume");

	private static readonly StringName _setMasterVolume = new StringName("set_master_volume");

	private static readonly StringName _stopMusic = new StringName("stop_music");

	private static readonly StringName _playMusic = new StringName("play_music");

	private static readonly StringName _playOneShot = new StringName("play_one_shot");

	private static readonly StringName _stopAllLoops = new StringName("stop_all_loops");

	private static readonly StringName _setParam = new StringName("set_param");

	private static readonly StringName _stopLoop = new StringName("stop_loop");

	private static readonly StringName _playLoop = new StringName("play_loop");

	private static readonly StringName _updateMusicParameterCallback = new StringName("update_music_parameter");

	private Node _audioNode;

	public static NAudioManager? Instance => NGame.Instance?.AudioManager;

	public override void _EnterTree()
	{
		_audioNode = GetNode<Node>("Proxy");
	}

	public void PlayLoop(string path, bool usesLoopParam)
	{
		if (!TestMode.IsOn)
		{
			_audioNode.Call(_playLoop, path, usesLoopParam);
		}
	}

	public void StopLoop(string path)
	{
		if (!TestMode.IsOn)
		{
			_audioNode.Call(_stopLoop, path);
		}
	}

	public void SetParam(string path, string param, float value)
	{
		if (!TestMode.IsOn)
		{
			_audioNode.Call(_setParam, path, param, value);
		}
	}

	public void StopAllLoops()
	{
		if (!TestMode.IsOn)
		{
			_audioNode.Call(_stopAllLoops);
		}
	}

	public void PlayOneShot(string path, System.Collections.Generic.Dictionary<string, float> parameters, float volume = 1f)
	{
		if (TestMode.IsOn)
		{
			return;
		}
		Dictionary dictionary = new Dictionary();
		foreach (KeyValuePair<string, float> parameter in parameters)
		{
			dictionary.Add(parameter.Key, parameter.Value);
		}
		_audioNode.Call(_playOneShot, path, dictionary, volume);
	}

	public void PlayOneShot(string path, float volume = 1f)
	{
		if (!TestMode.IsOn)
		{
			PlayOneShot(path, new System.Collections.Generic.Dictionary<string, float>(), volume);
		}
	}

	public void PlayMusic(string music)
	{
		if (!TestMode.IsOn)
		{
			_audioNode.Call(_playMusic, music);
		}
	}

	public void UpdateMusicParameter(string parameter, string value)
	{
		if (!NonInteractiveMode.IsActive)
		{
			_audioNode.Call(_updateMusicParameterCallback, parameter, value);
		}
	}

	public void StopMusic()
	{
		if (!TestMode.IsOn)
		{
			_audioNode.Call(_stopMusic);
		}
	}

	public void SetMasterVol(float volume)
	{
		if (!TestMode.IsOn)
		{
			_audioNode.Call(_setMasterVolume, Mathf.Pow(volume, 2f));
		}
	}

	public void SetSfxVol(float volume)
	{
		if (!TestMode.IsOn)
		{
			_audioNode.Call(_setSfxVolume, Mathf.Pow(volume, 2f));
		}
	}

	public void SetAmbienceVol(float volume)
	{
		if (!TestMode.IsOn)
		{
			_audioNode.Call(_setAmbienceVolume, Mathf.Pow(volume, 2f));
		}
	}

	public void SetBgmVol(float volume)
	{
		if (!TestMode.IsOn)
		{
			_audioNode.Call(_setBgmVolume, Mathf.Pow(volume, 2f));
		}
	}
}
