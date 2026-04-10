using System;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;

namespace MegaCrit.Sts2.Core.Training;

public partial class CombatTrainingSession : Node
{
	public static CombatTrainingSession? Instance { get; private set; }

	private int _episodeIndex;

	private bool _isStartingEpisode;

	private bool _isHandlingCombatEnd;

	public string? CurrentSeed { get; private set; }

	public string? CurrentCharacterId { get; private set; }

	public string? CurrentEncounterId { get; private set; }

	public int CurrentAscensionLevel { get; private set; }

	public bool? LastCombatWasVictory { get; private set; }

	public int CurrentEpisodeNumber => _episodeIndex;

	public override void _EnterTree()
	{
		Instance = this;
		CombatManager.Instance.CombatEnded += OnCombatEnded;
		CombatManager.Instance.CombatWon += OnCombatWon;
	}

	public override void _ExitTree()
	{
		if (Instance == this)
		{
			Instance = null;
		}
		CombatManager.Instance.CombatEnded -= OnCombatEnded;
		CombatManager.Instance.CombatWon -= OnCombatWon;
	}

	public async Task StartAsync(Nodes.NGame game)
	{
		SaveManager.Instance.PrefsSave.FastMode = FastModeType.Instant;
		SaveManager.Instance.SetFtuesEnabled(enabled: false);
		Engine.MaxFps = 0;
		await StartEpisodeAsync(game, null);
	}

	public async Task ResetAsync(CombatTrainingResetRequest? request = null)
	{
		if (Nodes.NGame.Instance == null)
		{
			throw new InvalidOperationException("Combat training session requires an active NGame instance.");
		}
		await StartEpisodeAsync(Nodes.NGame.Instance, request);
	}

	private void OnCombatWon(CombatRoom room)
	{
		LastCombatWasVictory = true;
	}

	private void OnCombatEnded(CombatRoom room)
	{
		LastCombatWasVictory ??= false;
		if (_isHandlingCombatEnd)
		{
			return;
		}
		TaskHelper.RunSafely(HandleCombatEndedAsync());
	}

	private async Task HandleCombatEndedAsync()
	{
		if (_isHandlingCombatEnd)
		{
			return;
		}
		_isHandlingCombatEnd = true;
		try
		{
			if (Nodes.NGame.Instance == null)
			{
				return;
			}
			await Nodes.NGame.Instance.ToSignal(Nodes.NGame.Instance.GetTree(), SceneTree.SignalName.ProcessFrame);
			if (CombatTrainingMode.ShouldLoop)
			{
				await StartEpisodeAsync(Nodes.NGame.Instance, null);
			}
			else if (CombatTrainingMode.ShouldQuitOnCombatEnd)
			{
				Nodes.NGame.Instance.Quit();
			}
		}
		finally
		{
			_isHandlingCombatEnd = false;
		}
	}

	private async Task StartEpisodeAsync(Nodes.NGame game, CombatTrainingResetRequest? request)
	{
		if (_isStartingEpisode)
		{
			return;
		}
		_isStartingEpisode = true;
		try
		{
			if (RunManager.Instance.IsInProgress)
			{
				RunManager.Instance.CleanUp();
			}
			int episodeNumber = _episodeIndex + 1;
			CharacterModel character = CombatTrainingMode.ResolveCharacter(request?.CharacterId);
			EncounterModel encounter = CombatTrainingMode.ResolveEncounter(request?.EncounterId).ToMutable();
			string seed = CombatTrainingMode.ResolveEpisodeSeed(request?.Seed);
			int ascensionLevel = CombatTrainingMode.ResolveAscensionLevel(request?.AscensionLevel);
			LastCombatWasVictory = null;
			CurrentSeed = seed;
			CurrentCharacterId = character.Id.Entry;
			CurrentEncounterId = encounter.Id.Entry;
			CurrentAscensionLevel = ascensionLevel;
			Log.Info($"[CombatTrainer] Starting episode {episodeNumber} with character={character.Id.Entry}, encounter={encounter.Id.Entry}, seed={seed}, ascension={ascensionLevel}");
			await game.StartNewSingleplayerRun(character, shouldSave: false, ActModel.GetDefaultList(), Array.Empty<ModifierModel>(), seed, ascensionLevel);
			await RunManager.Instance.EnterRoomDebug(encounter.RoomType, MapPointType.Unassigned, encounter, showTransition: false);
			_episodeIndex++;
		}
		finally
		{
			_isStartingEpisode = false;
		}
	}
}
