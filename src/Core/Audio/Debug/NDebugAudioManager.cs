using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Random;

namespace MegaCrit.Sts2.Core.Audio.Debug;

public partial class NDebugAudioManager : Node
{
	private struct PlayingSound
	{
		public int id;

		public AudioStreamPlayer player;

		public Callable callable;
	}

	private static readonly StringName _sfx = new StringName("SFX");

	private static readonly StringName _master = new StringName("Master");

	private List<AudioStreamPlayer> _freeAudioPlayers = new List<AudioStreamPlayer>();

	private readonly List<PlayingSound> _playingSounds = new List<PlayingSound>();

	private int _nextId;

	public static NDebugAudioManager? Instance => NGame.Instance?.DebugAudio;

	public override void _Ready()
	{
		_freeAudioPlayers.AddRange(GetChildren().OfType<AudioStreamPlayer>());
	}

	public int Play(string streamName, float volume = 1f, PitchVariance variance = PitchVariance.None)
	{
		AudioStreamPlayer streamPlayer;
		if (_freeAudioPlayers.Count > 0)
		{
			List<AudioStreamPlayer> freeAudioPlayers = _freeAudioPlayers;
			streamPlayer = freeAudioPlayers[freeAudioPlayers.Count - 1];
			_freeAudioPlayers.RemoveAt(_freeAudioPlayers.Count - 1);
		}
		else
		{
			streamPlayer = new AudioStreamPlayer();
			this.AddChildSafely(streamPlayer);
		}
		AudioStream asset = PreloadManager.Cache.GetAsset<AudioStream>(TmpSfx.GetPath(streamName));
		streamPlayer.Name = "StreamPlayer-" + streamName;
		streamPlayer.Stream = asset;
		streamPlayer.VolumeLinear = volume;
		streamPlayer.PitchScale = GetRandomPitchScale(variance);
		streamPlayer.Bus = _sfx;
		Callable callable = Callable.From(delegate
		{
			PlayerFinished(streamPlayer);
		});
		streamPlayer.Connect(AudioStreamPlayer.SignalName.Finished, callable);
		streamPlayer.Play();
		PlayingSound item = new PlayingSound
		{
			id = _nextId,
			player = streamPlayer,
			callable = callable
		};
		_playingSounds.Add(item);
		_nextId++;
		return item.id;
	}

	public void Stop(int id, float fadeTime = 0.5f)
	{
		int i;
		for (i = 0; i < _playingSounds.Count; i++)
		{
			PlayingSound playingSound = _playingSounds[i];
			if (playingSound.id != id)
			{
				continue;
			}
			if (fadeTime > 0f)
			{
				Tween tween = CreateTween();
				tween.TweenProperty(playingSound.player, "volume_linear", 0f, fadeTime);
				tween.TweenCallback(Callable.From(delegate
				{
					StopInternalById(i);
				}));
			}
			else
			{
				StopInternal(i);
			}
			return;
		}
		Log.Warn($"Tried to stop sound with ID {id} but no sound with that ID was found!");
	}

	private void StopInternalById(int id)
	{
		for (int i = 0; i < _playingSounds.Count; i++)
		{
			if (_playingSounds[i].id == id)
			{
				StopInternal(i);
				break;
			}
		}
	}

	private void StopInternal(int soundIndex)
	{
		PlayingSound playingSound = _playingSounds[soundIndex];
		if (playingSound.player.IsPlaying())
		{
			playingSound.player.Stop();
		}
		playingSound.player.Disconnect(AudioStreamPlayer.SignalName.Finished, playingSound.callable);
		_playingSounds.RemoveAt(soundIndex);
		_freeAudioPlayers.Add(playingSound.player);
	}

	public void SetMasterAudioVolume(float linearVolume)
	{
		AudioServer.Singleton.SetBusVolumeDb(AudioServer.Singleton.GetBusIndex(_master), Mathf.LinearToDb(Mathf.Pow(linearVolume, 2f)));
	}

	public void SetSfxAudioVolume(float linearVolume)
	{
		AudioServer.Singleton.SetBusVolumeDb(AudioServer.Singleton.GetBusIndex(_sfx), Mathf.LinearToDb(Mathf.Pow(linearVolume, 2f)));
	}

	private void PlayerFinished(AudioStreamPlayer player)
	{
		for (int i = 0; i < _playingSounds.Count; i++)
		{
			if (_playingSounds[i].player == player)
			{
				StopInternal(i);
				break;
			}
		}
	}

	private float GetRandomPitchScale(PitchVariance variance)
	{
		float num = variance switch
		{
			PitchVariance.None => 0f, 
			PitchVariance.Small => 0.02f, 
			PitchVariance.Medium => 0.05f, 
			PitchVariance.Large => 0.1f, 
			PitchVariance.TooMuch => 0.2f, 
			_ => throw new ArgumentOutOfRangeException("variance", variance, null), 
		};
		if (num == 0f)
		{
			return 1f;
		}
		return 1f + Rng.Chaotic.NextFloat(0f - num, num);
	}
}
