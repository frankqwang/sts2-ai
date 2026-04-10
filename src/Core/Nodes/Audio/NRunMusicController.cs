using System;
using Godot;
using Godot.Collections;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Audio;

public partial class NRunMusicController : Node
{
	private enum MusicProgressTrack
	{
		Init,
		Enemy,
		Merchant,
		Rest,
		Unknown,
		Treasure,
		Elite,
		CombatEnd,
		Elite2,
		MerchantEnd
	}

	private enum CampfireState
	{
		On,
		Off
	}

	private static readonly StringName _stopAmbience = new StringName("stop_ambience");

	private static readonly StringName _stopMusic = new StringName("stop_music");

	private const string _musicProgressParameter = "Progress";

	private const string _updateGlobalParameterCallback = "update_global_parameter";

	private const string _updateMusicParameterCallback = "update_music_parameter";

	private const string _updateMusicCallback = "update_music";

	private const string _updateAmbienceCallback = "update_ambience";

	private const string _updateCampfireAmbienceCallback = "update_campfire_ambience";

	private const string _updateCustomTrack = "update_custom_track";

	private const string _loadActBanksCallback = "load_act_banks";

	private const string _unloadActBanksCallback = "unload_act_banks";

	private IRunState _runState = NullRunState.Instance;

	private Node _proxy;

	private string _currentTrack;

	private string _currentAmbience;

	public static NRunMusicController? Instance => NRun.Instance?.RunMusicController;

	private MusicProgressTrack GetTrack(RoomType roomType)
	{
		if (roomType.IsCombatRoom() && !CombatManager.Instance.IsInProgress)
		{
			return MusicProgressTrack.CombatEnd;
		}
		switch (roomType)
		{
		case RoomType.Shop:
			return MusicProgressTrack.Merchant;
		case RoomType.RestSite:
			return MusicProgressTrack.Rest;
		case RoomType.Treasure:
			return MusicProgressTrack.Treasure;
		case RoomType.Monster:
			return MusicProgressTrack.Enemy;
		case RoomType.Event:
			if (_runState.CurrentRoom is EventRoom eventRoom && eventRoom.CanonicalEvent is AncientEventModel)
			{
				return MusicProgressTrack.Init;
			}
			return MusicProgressTrack.Unknown;
		case RoomType.Elite:
			return MusicProgressTrack.Elite;
		case RoomType.Boss:
			return MusicProgressTrack.Elite;
		default:
			return MusicProgressTrack.Init;
		}
	}

	public override void _Ready()
	{
		_proxy = GetNode<Node>("Proxy");
	}

	public override void _ExitTree()
	{
		StopMusic();
	}

	public void SetRunState(IRunState runState)
	{
		_runState = runState;
	}

	public void UpdateMusic()
	{
		if (!NonInteractiveMode.IsActive)
		{
			string[] bgMusicOptions = _runState.Act.BgMusicOptions;
			string[] musicBankPaths = _runState.Act.MusicBankPaths;
			int num = new Rng(_runState.Rng.Seed).NextInt(0, bgMusicOptions.Length);
			LoadActBank(musicBankPaths[num]);
			_currentTrack = bgMusicOptions[num];
			_proxy.Call("update_music", _currentTrack);
			_proxy.Call("update_global_parameter", "Progress", 0);
			UpdateAmbience();
		}
	}

	public void PlayCustomMusic(string customMusic)
	{
		if (!NonInteractiveMode.IsActive)
		{
			_proxy.Call(_stopMusic);
			_proxy.Call("update_music", customMusic);
		}
	}

	public void UpdateCustomTrack(string customTrack, float label)
	{
		if (!NonInteractiveMode.IsActive && RunManager.Instance.IsInProgress)
		{
			_proxy.Call("update_custom_track", customTrack, label);
		}
	}

	public void StopCustomMusic()
	{
		if (!NonInteractiveMode.IsActive)
		{
			_proxy.Call(_stopMusic);
			_proxy.Call("update_music", _currentTrack);
			_proxy.Call("update_global_parameter", "Progress", 7);
		}
	}

	public void UpdateAmbience()
	{
		if (!NonInteractiveMode.IsActive)
		{
			string ambientSfx = _runState.Act.AmbientSfx;
			EncounterModel encounterModel = (_runState.CurrentRoom as CombatRoom)?.Encounter;
			if (encounterModel != null && encounterModel.HasAmbientSfx)
			{
				ambientSfx = encounterModel.AmbientSfx;
			}
			if (_currentAmbience != ambientSfx)
			{
				_currentAmbience = ambientSfx;
				_proxy.Call("update_ambience", _currentAmbience);
			}
		}
	}

	public void UpdateTrack()
	{
		if (!NonInteractiveMode.IsActive)
		{
			MusicProgressTrack track = GetTrack(_runState.CurrentRoom.RoomType);
			UpdateTrack("Progress", (float)track);
			if (_runState.CurrentRoom is RestSiteRoom)
			{
				_proxy.Call("update_campfire_ambience", 0);
			}
		}
	}

	private void UpdateTrack(string label, float trackIndex)
	{
		_proxy.Call("update_global_parameter", label, trackIndex);
	}

	public void UpdateMusicParameter(string label, float trackIndex)
	{
		if (!NonInteractiveMode.IsActive)
		{
			_proxy.Call("update_music_parameter", label, trackIndex);
		}
	}

	public void ToggleMerchantTrack()
	{
		if (!NonInteractiveMode.IsActive && _runState.CurrentRoom != null)
		{
			if (_runState.CurrentRoom.RoomType != RoomType.Shop)
			{
				throw new InvalidOperationException("You can only trigger the merchant transition in a merchant room");
			}
			NMapScreen? instance = NMapScreen.Instance;
			MusicProgressTrack musicProgressTrack = ((instance != null && instance.IsVisible()) ? MusicProgressTrack.MerchantEnd : MusicProgressTrack.Merchant);
			_proxy.Call("update_global_parameter", "Progress", (int)musicProgressTrack);
		}
	}

	public void TriggerEliteSecondPhase()
	{
		if (!NonInteractiveMode.IsActive)
		{
			if (_runState.CurrentRoom.RoomType != RoomType.Elite)
			{
				throw new InvalidOperationException("You can only trigger the elite transition in an elite room");
			}
			_proxy.Call("update_global_parameter", "Progress", 8);
		}
	}

	public void TriggerCampfireGoingOut()
	{
		if (!NonInteractiveMode.IsActive)
		{
			if (_runState.CurrentRoom.RoomType != RoomType.RestSite)
			{
				throw new InvalidOperationException("You can only trigger the rest site transition in a rest site room");
			}
			_proxy.Call("update_campfire_ambience", 1);
		}
	}

	public void StopMusic()
	{
		if (!NonInteractiveMode.IsActive)
		{
			_proxy.Call(_stopMusic);
			_proxy.Call(_stopAmbience);
			UnloadActBanks();
		}
	}

	private void LoadActBank(string bankPath)
	{
		Godot.Collections.Array array = new Godot.Collections.Array { bankPath };
		_proxy.Call("load_act_banks", array);
	}

	private void UnloadActBanks()
	{
		_proxy.Call("unload_act_banks");
	}
}
