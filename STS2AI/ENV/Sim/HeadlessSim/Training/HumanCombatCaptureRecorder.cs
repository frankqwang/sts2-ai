using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Relics;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Training;

public sealed partial class HumanCombatCaptureRecorder : Node
{
	private readonly object _fileLock = new object();

	private readonly JsonSerializerOptions _jsonOptions = new JsonSerializerOptions
	{
		WriteIndented = false
	};

	private int _actionIndex;

	private string _sessionId = "";

	private string _outputRoot = "";

	private string _recordsPath = "";

	private string _sessionManifestPath = "";

	public static HumanCombatCaptureRecorder? Instance { get; private set; }

	private static int FindCardIndex(IReadOnlyList<CardModel> cards, CardModel? target)
	{
		if (target == null)
		{
			return -1;
		}
		for (int i = 0; i < cards.Count; i++)
		{
			if (ReferenceEquals(cards[i], target))
			{
				return i;
			}
		}
		return -1;
	}

	public static bool IsEnabled => CommandLineHelper.HasArg("human-full-run-capture") || CommandLineHelper.HasArg("human-combat-capture");

	public override void _EnterTree()
	{
		if (Instance != null && Instance != this)
		{
			Log.Warn("HumanCombatCaptureRecorder already exists. Dropping duplicate recorder node.");
			this.QueueFreeSafely();
			return;
		}
		Instance = this;
		InitializeOutput();
		WriteSessionManifest();
		Log.Info($"[HumanFullRunCapture] Recording manual full-run demos to '{_recordsPath}'.");
	}

	public override void _ExitTree()
	{
		if (Instance == this)
		{
			Instance = null;
		}
	}

	public static void RecordQueuedAction(GameAction action)
	{
		Instance?.TryRecordQueuedAction(action);
	}

	public static void RecordSelectionAction(string actionType, CardModel? card = null)
	{
		Instance?.TryRecordCombatSelectionAction(actionType, card);
	}

	public static void RecordRestSiteChoice(RunState runState, IReadOnlyList<RestSiteOption> options, int optionIndex)
	{
		if (options == null || optionIndex < 0 || optionIndex >= options.Count)
		{
			return;
		}
		Dictionary<string, object?> rawState = HumanFullRunCaptureStateBuilder.BuildRestSiteState(runState, options);
		List<Dictionary<string, object?>> candidateActions = HumanFullRunCaptureStateBuilder.BuildRestSiteCandidateActions(options);
		Dictionary<string, object?> chosenAction = new Dictionary<string, object?>
		{
			["action"] = "choose_rest_option",
			["index"] = optionIndex,
			["option_id"] = options[optionIndex].OptionId.ToLowerInvariant()
		};
		Instance?.TryRecordManualStep("rest_site", rawState, candidateActions, chosenAction);
	}

	public static void RecordEventChoice(RunState runState, EventModel eventModel, int optionIndex)
	{
		if (eventModel == null || optionIndex < 0 || optionIndex >= eventModel.CurrentOptions.Count)
		{
			return;
		}
		Dictionary<string, object?> rawState = HumanFullRunCaptureStateBuilder.BuildEventState(runState, eventModel);
		List<Dictionary<string, object?>> candidateActions = HumanFullRunCaptureStateBuilder.BuildEventCandidateActions(eventModel);
		EventOption option = eventModel.CurrentOptions[optionIndex];
		Dictionary<string, object?> chosenAction = new Dictionary<string, object?>
		{
			["action"] = "choose_event_option",
			["index"] = optionIndex,
			["option_id"] = option.TextKey.ToLowerInvariant()
		};
		Instance?.TryRecordManualStep("event", rawState, candidateActions, chosenAction);
	}

	public static void RecordShopPurchase(MerchantInventory inventory, MerchantEntry entry)
	{
		if (inventory == null || entry == null || !entry.IsStocked || !entry.EnoughGold)
		{
			return;
		}
		Dictionary<string, object?> rawState = HumanFullRunCaptureStateBuilder.BuildShopState(inventory);
		List<Dictionary<string, object?>> candidateActions = HumanFullRunCaptureStateBuilder.BuildShopCandidateActions(inventory);
		Dictionary<string, object?> chosenAction = HumanFullRunCaptureStateBuilder.NormalizeShopPurchaseAction(inventory, entry);
		Instance?.TryRecordManualStep("shop", rawState, candidateActions, chosenAction);
	}

	public static void RecordShopProceed(MerchantInventory inventory)
	{
		if (inventory == null)
		{
			return;
		}
		Dictionary<string, object?> rawState = HumanFullRunCaptureStateBuilder.BuildShopState(inventory);
		List<Dictionary<string, object?>> candidateActions = HumanFullRunCaptureStateBuilder.BuildShopCandidateActions(inventory);
		Dictionary<string, object?> chosenAction = new Dictionary<string, object?>
		{
			["action"] = "proceed"
		};
		Instance?.TryRecordManualStep("shop", rawState, candidateActions, chosenAction);
	}

	public static void RecordCardRewardChoice(
		Player player,
		IReadOnlyList<CardCreationResult> options,
		IReadOnlyList<CardRewardAlternative> extraOptions,
		int? cardIndex,
		bool skipped)
	{
		if (player == null)
		{
			return;
		}
		Dictionary<string, object?> rawState = HumanFullRunCaptureStateBuilder.BuildCardRewardState(player, options, extraOptions);
		List<Dictionary<string, object?>> candidateActions = HumanFullRunCaptureStateBuilder.BuildCardRewardCandidateActions(options, extraOptions);
		Dictionary<string, object?> chosenAction = skipped
			? new Dictionary<string, object?> { ["action"] = "skip_card_reward" }
			: new Dictionary<string, object?>
			{
				["action"] = "select_card_reward",
				["card_index"] = cardIndex ?? -1,
				["card_id"] = cardIndex.HasValue && cardIndex.Value >= 0 && cardIndex.Value < options.Count
					? options[cardIndex.Value].Card.Id.Entry.ToLowerInvariant()
					: ""
			};
		Instance?.TryRecordManualStep("card_reward", rawState, candidateActions, chosenAction);
	}

	public static void RecordTreasureChoice(RunState runState, IReadOnlyList<RelicModel> relics, int relicIndex)
	{
		if (relics == null || relicIndex < 0 || relicIndex >= relics.Count)
		{
			return;
		}
		Dictionary<string, object?> rawState = HumanFullRunCaptureStateBuilder.BuildTreasureState(runState, relics);
		List<Dictionary<string, object?>> candidateActions = HumanFullRunCaptureStateBuilder.BuildTreasureCandidateActions(relics);
		Dictionary<string, object?> chosenAction = new Dictionary<string, object?>
		{
			["action"] = "claim_treasure_relic",
			["index"] = relicIndex,
			["relic_id"] = relics[relicIndex].Id.Entry.ToLowerInvariant()
		};
		Instance?.TryRecordManualStep("treasure", rawState, candidateActions, chosenAction);
	}

	public static void RecordCardSelectChoice(
		Player player,
		IReadOnlyList<CardModel> cards,
		CardSelectorPrefs prefs,
		IReadOnlyList<CardModel> selectedCards,
		bool canSkip)
	{
		if (player == null)
		{
			return;
		}
		Dictionary<string, object?> rawState = HumanFullRunCaptureStateBuilder.BuildCardSelectState(player, cards, prefs, Array.Empty<CardModel>(), canSkip);
		List<Dictionary<string, object?>> candidateActions = HumanFullRunCaptureStateBuilder.BuildCardSelectCandidateActions(cards, prefs, Array.Empty<CardModel>(), canSkip);
		Dictionary<string, object?> chosenAction;
		if (selectedCards == null || selectedCards.Count == 0)
		{
			chosenAction = canSkip
				? new Dictionary<string, object?> { ["action"] = "cancel_selection", ["is_skip"] = true }
				: new Dictionary<string, object?> { ["action"] = "cancel_selection" };
		}
		else if (selectedCards.Count == 1 && !prefs.RequireManualConfirmation)
		{
			int index = FindCardIndex(cards, selectedCards[0]);
			chosenAction = new Dictionary<string, object?>
			{
				["action"] = "select_card",
				["index"] = index,
				["card_id"] = selectedCards[0].Id.Entry.ToLowerInvariant()
			};
		}
		else
		{
			chosenAction = new Dictionary<string, object?>
			{
				["action"] = "confirm_selection",
				["selected_indexes"] = selectedCards.Select(card => FindCardIndex(cards, card)).ToList(),
				["selected_card_ids"] = selectedCards.Select(static card => card.Id.Entry.ToLowerInvariant()).ToList()
			};
		}
		Instance?.TryRecordManualStep("card_select", rawState, candidateActions, chosenAction);
	}

	private void InitializeOutput()
	{
		string timestamp = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss");
		_sessionId = $"human_fullrun_{timestamp}";
		string? configuredRoot = CommandLineHelper.GetValue("human-full-run-capture-output");
		if (string.IsNullOrWhiteSpace(configuredRoot))
		{
			configuredRoot = CommandLineHelper.GetValue("human-combat-capture-output");
		}
		if (string.IsNullOrWhiteSpace(configuredRoot))
		{
			_outputRoot = ProjectSettings.GlobalizePath($"user://human_full_run_capture/{_sessionId}");
		}
		else if (Path.IsPathRooted(configuredRoot))
		{
			_outputRoot = Path.GetFullPath(configuredRoot);
		}
		else
		{
			_outputRoot = Path.GetFullPath(Path.Combine(System.Environment.CurrentDirectory, configuredRoot));
		}
		Directory.CreateDirectory(_outputRoot);
		_recordsPath = Path.Combine(_outputRoot, "human_full_run_capture.jsonl");
		_sessionManifestPath = Path.Combine(_outputRoot, "session_manifest.json");
	}

	private void WriteSessionManifest()
	{
		Dictionary<string, object?> manifest = new Dictionary<string, object?>
		{
			["schema_version"] = "human_full_run_capture_session.v1",
			["session_id"] = _sessionId,
			["created_utc"] = DateTime.UtcNow.ToString("O"),
			["records_path"] = "human_full_run_capture.jsonl",
			["output_root"] = _outputRoot
		};
		File.WriteAllText(_sessionManifestPath, JsonSerializer.Serialize(manifest, _jsonOptions));
	}

	private void TryRecordQueuedAction(GameAction action)
	{
		if (!ShouldCapture() || action == null)
		{
			return;
		}

		if (CombatManager.Instance.IsInProgress)
		{
			CombatTrainingStateSnapshot beforeState = CombatTrainingEnvService.CaptureStateSnapshotForRecorder();
			Dictionary<string, object?>? normalizedCombatAction = NormalizeCombatQueuedAction(action, beforeState);
			if (normalizedCombatAction == null)
			{
				return;
			}
			Dictionary<string, object?> rawState = BuildCombatRawState(beforeState);
			List<Dictionary<string, object?>> candidateActions = BuildCombatCandidateActions(beforeState);
			TryRecordManualStep("combat", rawState, candidateActions, normalizedCombatAction);
			return;
		}

		RunState? runState = RunManager.Instance.DebugOnlyGetState();
		if (runState == null)
		{
			return;
		}
		switch (action)
		{
		case MoveToMapCoordAction mapAction:
			TryRecordManualStep(
				"map",
				HumanFullRunCaptureStateBuilder.BuildMapState(runState),
				HumanFullRunCaptureStateBuilder.BuildMapCandidateActions(runState),
				HumanFullRunCaptureStateBuilder.NormalizeMapAction(runState, mapAction));
			break;
		case PickRelicAction pickRelicAction:
			List<RelicModel> relics = RunManager.Instance.TreasureRoomRelicSynchronizer.CurrentRelics?.ToList() ?? new List<RelicModel>();
			if (pickRelicAction.RelicIndex >= 0 && pickRelicAction.RelicIndex < relics.Count)
			{
				RecordTreasureChoice(runState, relics, pickRelicAction.RelicIndex);
			}
			break;
		}
	}

	private void TryRecordCombatSelectionAction(string actionType, CardModel? card)
	{
		if (!ShouldCapture() || !CombatManager.Instance.IsInProgress)
		{
			return;
		}
		CombatTrainingStateSnapshot beforeState = CombatTrainingEnvService.CaptureStateSnapshotForRecorder();
		Dictionary<string, object?>? normalizedAction = NormalizeSelectionAction(actionType, card, beforeState);
		if (normalizedAction == null)
		{
			return;
		}
		Dictionary<string, object?> rawState = BuildCombatRawState(beforeState);
		List<Dictionary<string, object?>> candidateActions = BuildCombatCandidateActions(beforeState);
		TryRecordManualStep("combat", rawState, candidateActions, normalizedAction);
	}

	private void TryRecordManualStep(
		string stateType,
		Dictionary<string, object?> rawState,
		List<Dictionary<string, object?>> candidateActions,
		Dictionary<string, object?> chosenAction)
	{
		if (!ShouldCapture())
		{
			return;
		}

		RunState? runState = RunManager.Instance.DebugOnlyGetState();
		Player? localPlayer = runState != null ? LocalContext.GetMe(runState) : null;
		int stepIndex;
		lock (_fileLock)
		{
			_actionIndex++;
			stepIndex = _actionIndex;
		}

		Dictionary<string, object?> record = new Dictionary<string, object?>
		{
			["schema_version"] = "human_full_run_capture.v1",
			["run_id"] = _sessionId,
			["session_id"] = _sessionId,
			["step_index"] = stepIndex,
			["timestamp_utc"] = DateTime.UtcNow.ToString("O"),
			["seed"] = runState?.Rng?.StringSeed,
			["act"] = runState?.CurrentActIndex + 1 ?? 0,
			["floor"] = runState?.ActFloor ?? 0,
			["character_id"] = localPlayer?.Character?.Id.Entry,
			["ascension_level"] = runState?.AscensionLevel ?? 0,
			["state_type"] = stateType,
			["raw_state"] = rawState,
			["candidate_actions"] = candidateActions,
			["chosen_action"] = chosenAction,
			["action_source"] = "human_manual",
			["source"] = "human_manual",
			["next_state"] = null,
			["terminal"] = false
		};
		TaskHelper.RunSafely(CompleteAndPersistRecordAsync(record));
	}

	private async Task CompleteAndPersistRecordAsync(Dictionary<string, object?> record)
	{
		await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		record["next_state"] = TryCaptureKnownNextState();
		record["terminal"] = RunManager.Instance.DebugOnlyGetState() == null;
		string json = JsonSerializer.Serialize(record, _jsonOptions);
		lock (_fileLock)
		{
			File.AppendAllText(_recordsPath, json + System.Environment.NewLine);
		}
	}

	private Dictionary<string, object?>? TryCaptureKnownNextState()
	{
		RunState? runState = RunManager.Instance.DebugOnlyGetState();
		if (runState == null)
		{
			return null;
		}
		if (CombatManager.Instance.IsInProgress)
		{
			return BuildCombatRawState(CombatTrainingEnvService.CaptureStateSnapshotForRecorder());
		}
		return HumanFullRunCaptureStateBuilder.BuildRunContext(runState);
	}

	private bool ShouldCapture()
	{
		if (!IsEnabled || CombatTrainingMode.IsActive)
		{
			return false;
		}
		return RunManager.Instance.IsSinglePlayerOrFakeMultiplayer && RunManager.Instance.DebugOnlyGetState() != null;
	}

	private static Dictionary<string, object?> BuildCombatRawState(CombatTrainingStateSnapshot snapshot)
	{
		RunState? runState = RunManager.Instance.DebugOnlyGetState();
		return new Dictionary<string, object?>
		{
			["state_type"] = snapshot.IsHandSelectionActive ? "hand_select" : ResolveCombatStateType(runState),
			["run"] = HumanFullRunCaptureStateBuilder.BuildRunContext(runState),
			["combat"] = snapshot,
			["player"] = snapshot.Player,
			["enemies"] = snapshot.Enemies,
			["hand_cards"] = snapshot.Hand,
			["hand_selection"] = snapshot.HandSelection
		};
	}

	private static List<Dictionary<string, object?>> BuildCombatCandidateActions(CombatTrainingStateSnapshot snapshot)
	{
		List<Dictionary<string, object?>> actions = new List<Dictionary<string, object?>>();
		if (snapshot.IsHandSelectionActive && snapshot.HandSelection != null)
		{
			foreach (CombatTrainingHandCardSnapshot card in snapshot.HandSelection.SelectableCards)
			{
				actions.Add(new Dictionary<string, object?>
				{
					["action"] = "select_hand_card",
					["hand_index"] = card.HandIndex
				});
			}
			if (snapshot.HandSelection.CanConfirm)
			{
				actions.Add(new Dictionary<string, object?> { ["action"] = "confirm_selection" });
			}
			return actions;
		}

		foreach (CombatTrainingHandCardSnapshot card2 in snapshot.Hand)
		{
			if (!card2.CanPlay)
			{
				continue;
			}
			if (card2.RequiresTarget)
			{
				foreach (CombatTrainingCreatureSnapshot enemy in snapshot.Enemies.Where(static enemy => enemy.IsAlive))
				{
					actions.Add(new Dictionary<string, object?>
					{
						["action"] = "play_card",
						["hand_index"] = card2.HandIndex,
						["target_id"] = enemy.CombatId
					});
				}
			}
			else
			{
				actions.Add(new Dictionary<string, object?>
				{
					["action"] = "play_card",
					["hand_index"] = card2.HandIndex
				});
			}
		}
		if (snapshot.CanEndTurn)
		{
			actions.Add(new Dictionary<string, object?> { ["action"] = "end_turn" });
		}
		return actions;
	}

	private static string ResolveCombatStateType(RunState? runState)
	{
		string roomType = runState?.CurrentRoom?.RoomType.ToString() ?? "";
		return roomType.ToLowerInvariant() switch
		{
			"elite" => "elite",
			"boss" => "boss",
			_ => "monster"
		};
	}

	private static Dictionary<string, object?>? NormalizeCombatQueuedAction(GameAction action, CombatTrainingStateSnapshot snapshot)
	{
		switch (action)
		{
		case PlayCardAction playCardAction:
			if (!LocalContext.IsMe(playCardAction.Player))
			{
				return null;
			}
			CombatTrainingHandCardSnapshot? playedCard = snapshot.Hand.FirstOrDefault((CombatTrainingHandCardSnapshot card) => card.CombatCardIndex == playCardAction.NetCombatCard.CombatCardIndex);
			if (playedCard == null)
			{
				return null;
			}
			Dictionary<string, object?> playAction = new Dictionary<string, object?>
			{
				["action"] = "play_card",
				["hand_index"] = playedCard.HandIndex
			};
			if (playCardAction.TargetId.HasValue)
			{
				playAction["target_id"] = (int)playCardAction.TargetId.Value;
			}
			return playAction;
		case EndPlayerTurnAction:
			return new Dictionary<string, object?>
			{
				["action"] = "end_turn"
			};
		default:
			return null;
		}
	}

	private static Dictionary<string, object?>? NormalizeSelectionAction(string actionType, CardModel? card, CombatTrainingStateSnapshot snapshot)
	{
		string normalized = (actionType ?? "").Trim().ToLowerInvariant();
		if (normalized == "confirm_selection")
		{
			return new Dictionary<string, object?>
			{
				["action"] = "confirm_selection"
			};
		}
		if (normalized == "cancel_selection")
		{
			return new Dictionary<string, object?>
			{
				["action"] = "cancel_selection"
			};
		}
		if (normalized != "select_hand_card" || card == null)
		{
			return null;
		}
		uint combatCardIndex = NetCombatCard.FromModel(card).CombatCardIndex;
		CombatTrainingHandCardSnapshot? selectedCard = snapshot.HandSelection?.SelectableCards?.FirstOrDefault((CombatTrainingHandCardSnapshot entry) => entry.CombatCardIndex == combatCardIndex);
		selectedCard ??= snapshot.Hand.FirstOrDefault((CombatTrainingHandCardSnapshot entry) => entry.CombatCardIndex == combatCardIndex);
		if (selectedCard == null)
		{
			return null;
		}
		return new Dictionary<string, object?>
		{
			["action"] = "select_hand_card",
			["hand_index"] = selectedCard.HandIndex
		};
	}
}
