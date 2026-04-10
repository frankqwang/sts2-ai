using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;
using MegaCrit.Sts2.Core.Training;
using MegaCrit.Sts2.Core.Unlocks;

namespace MegaCrit.Sts2.Core.Simulation;

public sealed class CombatSimulatorRuntimeFacade : ICombatRuntimeFacade, IDisposable
{
	private IDisposable? _runtimeScope;

	private IDisposable? _selectorScope;

	private bool _subscribedToCombatEvents;

	public bool IsActive => _runtimeScope != null;

	public bool IsPureSimulator => true;

	public bool? LastCombatWasVictory { get; private set; }

	public int EpisodeNumber { get; private set; }

	public string? CurrentSeed { get; private set; }

	public string? CurrentCharacterId { get; private set; }

	public string? CurrentEncounterId { get; private set; }

	public int CurrentAscensionLevel { get; private set; }

	public CombatSimulatorRuntimeFacade()
	{
		SubscribeCombatEvents();
	}

	public async Task<CombatTrainingStateSnapshot> ResetAsync(CombatTrainingResetRequest? request = null)
	{
		CombatSimulationTrace.Write("runtime.reset.begin");
		CleanUpPreviousEpisode();
		CombatSimulationTrace.Write("runtime.reset.cleaned_previous");
		_runtimeScope = new SimulationRuntimeScope();
		CombatSimulationTrace.Write("runtime.reset.entered_scope");
		CombatTrainingSimulatorChoiceBridge.Instance.Reset();
		_selectorScope = CardSelectCmd.UseSelector(CombatTrainingSimulatorChoiceBridge.Instance);
		CombatSimulationTrace.Write("runtime.reset.selector_ready");
		SaveManager.Instance?.SetFtuesEnabled(enabled: false);
		if (SaveManager.Instance != null)
		{
			SaveManager.Instance.PrefsSave.FastMode = FastModeType.Instant;
		}
		LocalContext.NetId = NetSingleplayerGameService.defaultNetId;
		CharacterModel character = CombatTrainingMode.ResolveCharacter(request?.CharacterId);
		EncounterModel encounter = CombatTrainingMode.ResolveEncounter(request?.EncounterId).ToMutable();
		string seed = CombatTrainingMode.ResolveEpisodeSeed(request?.Seed);
		int ascensionLevel = CombatTrainingMode.ResolveAscensionLevel(request?.AscensionLevel);
		List<ActModel> acts = ActModel.GetDefaultList().Select(static act => act.ToMutable()).ToList();
		Player player = Player.CreateForNewRun(character, UnlockState.all, NetSingleplayerGameService.defaultNetId);
		RunState runState = RunState.CreateForNewRun(new List<Player> { player }, acts, Array.Empty<ModifierModel>(), ascensionLevel, seed);
		RunManager.Instance.SetUpTest(runState, new NetSingleplayerGameService(), shouldSave: false);
		CombatSimulationTrace.Write("runtime.reset.run_setup_test");
		CombatState combatState = new CombatState(encounter, runState, runState.Modifiers, runState.MultiplayerScalingModel);
		foreach (Player runPlayer in runState.Players)
		{
			combatState.AddPlayer(runPlayer);
		}
		if (!encounter.HaveMonstersBeenGenerated)
		{
			encounter.GenerateMonstersWithSlots(runState);
		}
		foreach (var (monsterModel, slot) in encounter.MonstersWithSlots)
		{
			monsterModel.AssertMutable();
			Creature creature = combatState.CreateCreature(monsterModel, CombatSide.Enemy, slot);
			combatState.AddCreature(creature);
		}
		runState.PushRoom(new CombatRoom(combatState));
		CombatSimulationTrace.Write("runtime.reset.room_pushed");
		LastCombatWasVictory = null;
		CurrentSeed = seed;
		CurrentCharacterId = character.Id.Entry;
		CurrentEncounterId = encounter.Id.Entry;
		CurrentAscensionLevel = ascensionLevel;
		EpisodeNumber++;
		CombatManager.Instance.SetUpCombat(combatState);
		if (RunManager.Instance.CombatReplayWriter.IsEnabled)
		{
			RunManager.Instance.CombatReplayWriter.RecordInitialState(RunManager.Instance.ToSave(null));
		}
		CombatSimulationTrace.Write("runtime.reset.combat_setup");
		await CombatManager.Instance.StartCombatInternal();
		CombatSimulationTrace.Write("runtime.reset.combat_started");
		return await CombatTrainingEnvService.WaitForSettledAndSnapshotAsync();
	}

	public CombatTrainingStateSnapshot GetState()
	{
		return CombatTrainingEnvService.BuildStateSnapshot();
	}

	public Task<CombatTrainingStepResult> StepAsync(CombatTrainingActionRequest action)
	{
		return CombatTrainingEnvService.StepAgainstActiveCombatAsync(action);
	}

	public void Dispose()
	{
		CleanUpPreviousEpisode();
		if (_subscribedToCombatEvents)
		{
			CombatManager.Instance.CombatWon -= OnCombatWon;
			CombatManager.Instance.CombatEnded -= OnCombatEnded;
			_subscribedToCombatEvents = false;
		}
	}

	private void SubscribeCombatEvents()
	{
		if (_subscribedToCombatEvents)
		{
			return;
		}
		CombatManager.Instance.CombatWon += OnCombatWon;
		CombatManager.Instance.CombatEnded += OnCombatEnded;
		_subscribedToCombatEvents = true;
	}

	private void CleanUpPreviousEpisode()
	{
		CombatTrainingSimulatorChoiceBridge.Instance.Reset();
		if (RunManager.Instance.IsInProgress)
		{
			RunManager.Instance.CleanUp();
		}
		else
		{
			CombatManager.Instance.Reset();
		}
		_selectorScope?.Dispose();
		_selectorScope = null;
		_runtimeScope?.Dispose();
		_runtimeScope = null;
	}

	private void OnCombatWon(CombatRoom room)
	{
		if (IsActive)
		{
			LastCombatWasVictory = true;
		}
	}

	private void OnCombatEnded(CombatRoom room)
	{
		if (IsActive)
		{
			LastCombatWasVictory ??= false;
		}
	}
}
