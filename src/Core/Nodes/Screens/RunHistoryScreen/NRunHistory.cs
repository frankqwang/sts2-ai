using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Models.Exceptions;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Debug;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Potions;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Runs.History;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;
using MegaCrit.Sts2.Core.Unlocks;

namespace MegaCrit.Sts2.Core.Nodes.Screens.RunHistoryScreen;

public partial class NRunHistory : NSubmenu
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/run_history_screen/run_history");

	public const string locTable = "run_history";

	private static readonly LocString _leftQuote = new LocString("game_over_screen", "ENCOUNTER_QUOTE_LEFT");

	private static readonly LocString _rightQuote = new LocString("game_over_screen", "ENCOUNTER_QUOTE_RIGHT");

	private Control _screenContents;

	private Control _playerIconContainer;

	private MegaLabel _hpLabel;

	private MegaLabel _goldLabel;

	private Control _potionHolder;

	private MegaLabel _floorLabel;

	private MegaLabel _timeLabel;

	private MegaRichTextLabel _dateLabel;

	private MegaRichTextLabel _seedLabel;

	private MegaRichTextLabel _gameModeLabel;

	private MegaRichTextLabel _buildLabel;

	private MegaRichTextLabel _deathQuoteLabel;

	private NMapPointHistory _mapPointHistory;

	private NRelicHistory _relicHistory;

	private NDeckHistory _deckHistory;

	private Control _outOfDateVisual;

	private readonly List<string> _runNames = new List<string>();

	private int _index;

	private RunHistory _history;

	private NRunHistoryArrowButton _prevButton;

	private NRunHistoryArrowButton _nextButton;

	private NRunHistoryPlayerIcon? _selectedPlayerIcon;

	private Tween? _screenTween;

	protected override Control? InitialFocusedControl => null;

	public static string[] AssetPaths => new string[1] { _scenePath };

	public static NRunHistory? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NRunHistory>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		ConnectSignals();
		_screenContents = GetNode<Control>("ScreenContents");
		_playerIconContainer = GetNode<Control>("%PlayerIconContainer");
		_hpLabel = GetNode<MegaLabel>("%HpLabel");
		_goldLabel = GetNode<MegaLabel>("%GoldLabel");
		_potionHolder = GetNode<Control>("%PotionHolders");
		_floorLabel = GetNode<MegaLabel>("%FloorNumLabel");
		_timeLabel = GetNode<MegaLabel>("%RunTimeLabel");
		_dateLabel = GetNode<MegaRichTextLabel>("%DateLabel");
		_seedLabel = GetNode<MegaRichTextLabel>("%SeedLabel");
		_gameModeLabel = GetNode<MegaRichTextLabel>("%GameModeLabel");
		_buildLabel = GetNode<MegaRichTextLabel>("%BuildLabel");
		_deathQuoteLabel = GetNode<MegaRichTextLabel>("%DeathQuoteLabel");
		_mapPointHistory = GetNode<NMapPointHistory>("%MapPointHistory");
		_relicHistory = GetNode<NRelicHistory>("%RelicHistory");
		_deckHistory = GetNode<NDeckHistory>("%DeckHistory");
		_outOfDateVisual = GetNode<Control>("%OutOfDateVisual");
		_prevButton = GetNode<NRunHistoryArrowButton>("LeftArrow");
		_prevButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnLeftButtonButtonReleased));
		_nextButton = GetNode<NRunHistoryArrowButton>("RightArrow");
		_nextButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnRightButtonButtonReleased));
		_prevButton.IsLeft = true;
		_mapPointHistory.SetDeckHistory(_deckHistory);
		_mapPointHistory.SetRelicHistory(_relicHistory);
	}

	private void OnLeftButtonButtonReleased(NButton _)
	{
		TaskHelper.RunSafely(RefreshAndSelectRun(_index + 1));
		_screenTween?.Kill();
		_screenTween = CreateTween().SetParallel();
		_screenTween.TweenProperty(_screenContents, "position", Vector2.Zero, 0.5).SetTrans(Tween.TransitionType.Expo).SetEase(Tween.EaseType.Out)
			.From(Vector2.Zero + new Vector2(-1000f, 0f));
		_screenTween.TweenProperty(_screenContents, "modulate:a", 1f, 0.4).SetTrans(Tween.TransitionType.Linear).From(0f);
	}

	private void OnRightButtonButtonReleased(NButton _)
	{
		TaskHelper.RunSafely(RefreshAndSelectRun(_index - 1));
		_screenTween?.Kill();
		_screenTween = CreateTween().SetParallel();
		_screenTween.TweenProperty(_screenContents, "position", Vector2.Zero, 0.5).SetTrans(Tween.TransitionType.Expo).SetEase(Tween.EaseType.Out)
			.From(Vector2.Zero + new Vector2(1000f, 0f));
		_screenTween.TweenProperty(_screenContents, "modulate:a", 1f, 0.4).SetTrans(Tween.TransitionType.Linear).From(0f);
	}

	public static bool CanBeShown()
	{
		return SaveManager.Instance.GetRunHistoryCount() > 0;
	}

	public override void OnSubmenuOpened()
	{
		_runNames.Clear();
		_runNames.AddRange(SaveManager.Instance.GetAllRunHistoryNames());
		_runNames.Reverse();
		TaskHelper.RunSafely(RefreshAndSelectRun(0));
	}

	protected override void OnSubmenuShown()
	{
		if (!CanBeShown())
		{
			throw new InvalidOperationException("Tried to show run history screen with no runs!");
		}
		_screenTween?.Kill();
		_screenTween = CreateTween();
		_screenTween.TweenProperty(_screenContents, "modulate:a", 1f, 0.4).From(0f);
	}

	protected override void OnSubmenuHidden()
	{
		_screenTween?.Kill();
	}

	private Task RefreshAndSelectRun(int index)
	{
		if (index < 0 || index >= _runNames.Count)
		{
			Log.Error($"Invalid run index {index}, valid range is 0-{_runNames.Count - 1}");
			return Task.CompletedTask;
		}
		_prevButton.Disable();
		_nextButton.Disable();
		_outOfDateVisual.Visible = false;
		try
		{
			ReadSaveResult<RunHistory> readSaveResult = SaveManager.Instance.LoadRunHistory(_runNames[index]);
			if (readSaveResult.Success)
			{
				DisplayRun(readSaveResult.SaveData);
			}
			else
			{
				Log.Error($"Could not load run {_runNames[index]} at index {index}: {readSaveResult.ErrorMessage} ({readSaveResult.Status})");
				_outOfDateVisual.Visible = true;
			}
		}
		catch (Exception value)
		{
			Log.Error($"Exception {value} while loading run at index {index}");
			_outOfDateVisual.Visible = true;
			throw;
		}
		finally
		{
			_index = index;
			if (index < _runNames.Count - 1)
			{
				_prevButton.Enable();
			}
			if (index > 0)
			{
				_nextButton.Enable();
			}
			_prevButton.Visible = index < _runNames.Count - 1;
			_nextButton.Visible = index > 0;
		}
		return Task.CompletedTask;
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (!IsVisibleInTree() || NDevConsole.Instance.Visible || !NControllerManager.Instance.IsUsingController)
		{
			return;
		}
		Control control = GetViewport().GuiGetFocusOwner();
		bool flag = ((control is TextEdit || control is LineEdit) ? true : false);
		if (!flag && ActiveScreenContext.Instance.IsCurrent(this))
		{
			Control control2 = GetViewport().GuiGetFocusOwner();
			if ((control2 == null || !IsAncestorOf(control2)) && (inputEvent.IsActionPressed(MegaInput.left) || inputEvent.IsActionPressed(MegaInput.right) || inputEvent.IsActionPressed(MegaInput.up) || inputEvent.IsActionPressed(MegaInput.down) || inputEvent.IsActionPressed(MegaInput.select)))
			{
				GetViewport()?.SetInputAsHandled();
				_mapPointHistory.DefaultFocusedControl?.TryGrabFocus();
			}
		}
	}

	private void DisplayRun(RunHistory history)
	{
		_selectedPlayerIcon?.Deselect();
		_selectedPlayerIcon = null;
		foreach (NRunHistoryPlayerIcon item in _playerIconContainer.GetChildren().OfType<NRunHistoryPlayerIcon>())
		{
			item.QueueFreeSafely();
		}
		_history = history;
		ulong localPlayerId = PlatformUtil.GetLocalPlayerId(history.PlatformType);
		LoadPlayerFloor(history);
		LoadGameModeDetails(history);
		LoadTimeDetails(history);
		_mapPointHistory.LoadHistory(history);
		bool flag = false;
		NRunHistoryPlayerIcon nRunHistoryPlayerIcon = null;
		foreach (RunHistoryPlayer player in history.Players)
		{
			NRunHistoryPlayerIcon playerIcon = PreloadManager.Cache.GetScene(NRunHistoryPlayerIcon.scenePath).Instantiate<NRunHistoryPlayerIcon>(PackedScene.GenEditState.Disabled);
			if (nRunHistoryPlayerIcon == null)
			{
				nRunHistoryPlayerIcon = playerIcon;
			}
			_playerIconContainer.AddChildSafely(playerIcon);
			playerIcon.LoadRun(player, history);
			playerIcon.Connect(NClickableControl.SignalName.MouseReleased, Callable.From<InputEvent>(delegate
			{
				SelectPlayer(playerIcon);
			}));
			if (player.Id == localPlayerId)
			{
				flag = true;
				SelectPlayer(playerIcon);
			}
		}
		if (!flag)
		{
			if (history.Players.Count > 1)
			{
				Log.Warn($"Local player with ID {localPlayerId} not found in multiplayer run history file! Defaulting to first player");
			}
			SelectPlayer(nRunHistoryPlayerIcon);
		}
	}

	private void SelectPlayer(NRunHistoryPlayerIcon playerIcon)
	{
		_selectedPlayerIcon?.Deselect();
		_selectedPlayerIcon = playerIcon;
		playerIcon.Select();
		if (_history.Players.Count == 1)
		{
			CharacterModel byId = ModelDb.GetById<CharacterModel>(playerIcon.Player.Character);
			Color nameColor = byId.NameColor;
		}
		else
		{
			LocString locString = new LocString("run_history", "PLAYER_NAME");
			locString.Add("PlayerName", PlatformUtil.GetPlayerName(_history.PlatformType, playerIcon.Player.Id));
		}
		UnlockState unlockState = SaveManager.Instance.GenerateUnlockStateFromProgress();
		Player player = Player.CreateForNewRun(ModelDb.GetById<CharacterModel>(playerIcon.Player.Character), unlockState, playerIcon.Player.Id);
		LoadGoldHpAndPotionInfo(playerIcon);
		LoadDeathQuote(_history, playerIcon.Player.Character);
		_mapPointHistory.SetPlayer(playerIcon.Player);
		_relicHistory.LoadRelics(player, playerIcon.Player.Relics);
		_deckHistory.LoadDeck(player, playerIcon.Player.Deck);
	}

	private void LoadGoldHpAndPotionInfo(NRunHistoryPlayerIcon icon)
	{
		if (!_history.MapPointHistory.Any())
		{
			CharacterModel byId = ModelDb.GetById<CharacterModel>(icon.Player.Character);
			_hpLabel.SetTextAutoSize($"{byId.StartingHp}/{byId.StartingHp}");
			_goldLabel.SetTextAutoSize($"{byId.StartingGold}");
		}
		else
		{
			MapPointHistoryEntry mapPointHistoryEntry = _history.MapPointHistory.Last().Last();
			PlayerMapPointHistoryEntry playerMapPointHistoryEntry = mapPointHistoryEntry.PlayerStats.First((PlayerMapPointHistoryEntry stat) => stat.PlayerId == icon.Player.Id);
			_hpLabel.SetTextAutoSize($"{playerMapPointHistoryEntry.CurrentHp}/{playerMapPointHistoryEntry.MaxHp}");
			_goldLabel.SetTextAutoSize($"{playerMapPointHistoryEntry.CurrentGold}");
		}
		_potionHolder.FreeChildren();
		RunHistoryPlayer runHistoryPlayer = _history.Players.First((RunHistoryPlayer player) => player.Id == icon.Player.Id);
		List<PotionModel> list = runHistoryPlayer.Potions.Select(PotionModel.FromSerializable).ToList();
		List<NPotionHolder> list2 = new List<NPotionHolder>();
		for (int num = 0; num < runHistoryPlayer.MaxPotionSlotCount; num++)
		{
			NPotionHolder nPotionHolder = NPotionHolder.Create(isUsable: false);
			_potionHolder.AddChildSafely(nPotionHolder);
			list2.Add(nPotionHolder);
		}
		UnlockState unlockState = SaveManager.Instance.GenerateUnlockStateFromProgress();
		Player owner = Player.CreateForNewRun(ModelDb.GetById<CharacterModel>(icon.Player.Character), unlockState, icon.Player.Id);
		for (int num2 = 0; num2 < list.Count && num2 < runHistoryPlayer.MaxPotionSlotCount; num2++)
		{
			NPotion nPotion = NPotion.Create(list[num2]);
			nPotion.Model.Owner = owner;
			list2[num2].AddPotion(nPotion);
			nPotion.Position = Vector2.Zero;
		}
	}

	private void LoadPlayerFloor(RunHistory history)
	{
		int value = history.MapPointHistory.Sum((List<MapPointHistoryEntry> rooms) => rooms.Count);
		_floorLabel.SetTextAutoSize($"{value}");
	}

	private void LoadGameModeDetails(RunHistory history)
	{
		LocString locString = new LocString("run_history", "GAME_MODE.title");
		if (history.Players.Count > 1)
		{
			locString.Add("PlayerCount", new LocString("run_history", "PLAYER_COUNT.multiplayer"));
		}
		else
		{
			locString.Add("PlayerCount", new LocString("run_history", "PLAYER_COUNT.singleplayer"));
		}
		switch (history.GameMode)
		{
		case GameMode.Custom:
			locString.Add("GameMode", new LocString("run_history", "GAME_MODE.custom"));
			break;
		case GameMode.Daily:
			locString.Add("GameMode", new LocString("run_history", "GAME_MODE.daily"));
			break;
		case GameMode.Standard:
			locString.Add("GameMode", new LocString("run_history", "GAME_MODE.standard"));
			break;
		default:
			locString.Add("GameMode", new LocString("run_history", "GAME_MODE.unknown"));
			break;
		}
		_gameModeLabel.Text = "[right]" + locString.GetFormattedText() + "[/right]";
	}

	public static GameOverType GetGameOverType(RunHistory history)
	{
		if (history.Win)
		{
			return GameOverType.FalseVictory;
		}
		if (history.WasAbandoned)
		{
			return GameOverType.AbandonedRun;
		}
		if (history.KilledByEncounter != ModelId.none)
		{
			return GameOverType.CombatDeath;
		}
		if (history.KilledByEvent != ModelId.none)
		{
			return GameOverType.EventDeath;
		}
		Log.Warn("How did the game end??");
		return GameOverType.None;
	}

	public static string GetDeathQuote(RunHistory history, ModelId characterId, GameOverType gameOverType)
	{
		CharacterModel byId = ModelDb.GetById<CharacterModel>(characterId);
		StringBuilder stringBuilder = new StringBuilder();
		stringBuilder.Append(_leftQuote.GetRawText());
		Rng rng = new Rng((uint)StringHelper.GetDeterministicHashCode(history.Seed));
		switch (gameOverType)
		{
		case GameOverType.AbandonedRun:
		{
			LocString randomWithPrefix2 = LocString.GetRandomWithPrefix("run_history", "MAP_POINT_HISTORY.abandon", rng);
			byId.AddDetailsTo(randomWithPrefix2);
			stringBuilder.Append(randomWithPrefix2.GetFormattedText());
			break;
		}
		case GameOverType.EventDeath:
		{
			EventModel eventModel;
			try
			{
				eventModel = ModelDb.GetById<EventModel>(history.KilledByEvent);
			}
			catch (ModelNotFoundException)
			{
				eventModel = ModelDb.Event<DeprecatedEvent>();
			}
			LocString locString2 = new LocString(eventModel.LocTable, eventModel.Id.Entry + ".loss");
			byId.AddDetailsTo(locString2);
			locString2.Add("event", eventModel.Title);
			stringBuilder.Append(locString2.GetFormattedText());
			break;
		}
		case GameOverType.CombatDeath:
		{
			EncounterModel encounterModel = SaveUtil.EncounterOrDeprecated(history.KilledByEncounter);
			LocString lossMessageFor = encounterModel.GetLossMessageFor(byId);
			stringBuilder.Append(lossMessageFor.GetFormattedText());
			break;
		}
		case GameOverType.FalseVictory:
		{
			LocString randomWithPrefix = LocString.GetRandomWithPrefix("run_history", "MAP_POINT_HISTORY.falseVictory", rng);
			byId.AddDetailsTo(randomWithPrefix);
			stringBuilder.Append(randomWithPrefix.GetFormattedText());
			break;
		}
		case GameOverType.None:
		case GameOverType.TrueVictory:
		{
			LocString locString = new LocString("run_history", "MAP_POINT_HISTORY.debug");
			byId.AddDetailsTo(locString);
			stringBuilder.Append(locString.GetFormattedText());
			break;
		}
		default:
			Log.Error("Unimplemented GameOverType: " + gameOverType);
			throw new ArgumentOutOfRangeException("gameOverType", gameOverType, null);
		}
		stringBuilder.Append(_rightQuote.GetRawText());
		return stringBuilder.ToString();
	}

	private void LoadDeathQuote(RunHistory history, ModelId characterId)
	{
		CharacterModel byId = ModelDb.GetById<CharacterModel>(characterId);
		StringBuilder stringBuilder = new StringBuilder();
		stringBuilder.Append(_leftQuote.GetRawText());
		Rng rng = new Rng((uint)StringHelper.GetDeterministicHashCode(history.Seed));
		if (history.Win)
		{
			_deathQuoteLabel.AddThemeColorOverride(ThemeConstants.RichTextLabel.defaultColor, StsColors.green);
			LocString randomWithPrefix = LocString.GetRandomWithPrefix("run_history", "MAP_POINT_HISTORY.falseVictory", rng);
			byId.AddDetailsTo(randomWithPrefix);
			StringBuilder stringBuilder2 = stringBuilder;
			StringBuilder stringBuilder3 = stringBuilder2;
			StringBuilder.AppendInterpolatedStringHandler handler = new StringBuilder.AppendInterpolatedStringHandler(0, 1, stringBuilder2);
			handler.AppendFormatted(randomWithPrefix.GetFormattedText());
			stringBuilder3.Append(ref handler);
		}
		else if (history.WasAbandoned)
		{
			_deathQuoteLabel.AddThemeColorOverride(ThemeConstants.RichTextLabel.defaultColor, StsColors.red);
			LocString randomWithPrefix2 = LocString.GetRandomWithPrefix("run_history", "MAP_POINT_HISTORY.abandon", rng);
			byId.AddDetailsTo(randomWithPrefix2);
			stringBuilder.Append(randomWithPrefix2.GetFormattedText());
		}
		else if (history.KilledByEncounter != ModelId.none)
		{
			_deathQuoteLabel.AddThemeColorOverride(ThemeConstants.RichTextLabel.defaultColor, StsColors.red);
			EncounterModel encounterModel = SaveUtil.EncounterOrDeprecated(history.KilledByEncounter);
			LocString lossMessageFor = encounterModel.GetLossMessageFor(byId);
			StringBuilder stringBuilder2 = stringBuilder;
			StringBuilder stringBuilder4 = stringBuilder2;
			StringBuilder.AppendInterpolatedStringHandler handler = new StringBuilder.AppendInterpolatedStringHandler(0, 1, stringBuilder2);
			handler.AppendFormatted(lossMessageFor.GetFormattedText());
			stringBuilder4.Append(ref handler);
		}
		else if (history.KilledByEvent != ModelId.none)
		{
			_deathQuoteLabel.AddThemeColorOverride(ThemeConstants.RichTextLabel.defaultColor, StsColors.red);
			EventModel eventModel;
			try
			{
				eventModel = ModelDb.GetById<EventModel>(history.KilledByEvent);
			}
			catch (ModelNotFoundException)
			{
				eventModel = ModelDb.Event<DeprecatedEvent>();
			}
			string text = eventModel.Id.Entry + ".loss";
			LocString locString = ((!LocString.Exists("events", text)) ? new LocString("run_history", "DEFAULT_EVENT_LOSS_MESSAGE") : new LocString("events", text));
			byId.AddDetailsTo(locString);
			locString.Add("event", eventModel.Title);
			StringBuilder stringBuilder2 = stringBuilder;
			StringBuilder stringBuilder5 = stringBuilder2;
			StringBuilder.AppendInterpolatedStringHandler handler = new StringBuilder.AppendInterpolatedStringHandler(0, 1, stringBuilder2);
			handler.AppendFormatted(locString.GetFormattedText());
			stringBuilder5.Append(ref handler);
		}
		stringBuilder.Append(_rightQuote.GetRawText());
		_deathQuoteLabel.Text = stringBuilder.ToString();
	}

	private void LoadTimeDetails(RunHistory history)
	{
		DateTimeFormatInfo dateTimeFormat = LocManager.Instance.CultureInfo.DateTimeFormat;
		DateTime dateTime = TimeZoneInfo.ConvertTimeFromUtc(DateTimeOffset.FromUnixTimeSeconds(history.StartTime).UtcDateTime, TimeZoneInfo.Local);
		string value = dateTime.ToString("MMMM d, yyyy", dateTimeFormat);
		string value2 = dateTime.ToString("h:mm tt", dateTimeFormat);
		_dateLabel.Text = $"[right][gold]{value}[/gold], [blue]{value2}[/blue][/right]";
		_seedLabel.Text = "[right][gold]Seed[/gold]: " + history.Seed + "[/right]";
		_buildLabel.Text = "[right]" + history.BuildId + "[/right]";
		_timeLabel.SetTextAutoSize(TimeFormatting.Format(history.RunTime));
	}
}
