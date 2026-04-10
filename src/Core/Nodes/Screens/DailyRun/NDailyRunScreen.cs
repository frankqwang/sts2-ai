using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Daily;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Extensions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Modifiers;
using MegaCrit.Sts2.Core.Multiplayer;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Multiplayer.Game.Lobby;
using MegaCrit.Sts2.Core.Multiplayer.Messages.Lobby;
using MegaCrit.Sts2.Core.Nodes.Audio;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Multiplayer;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;
using MegaCrit.Sts2.Core.Unlocks;

namespace MegaCrit.Sts2.Core.Nodes.Screens.DailyRun;

public partial class NDailyRunScreen : NSubmenu, IStartRunLobbyListener
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/daily_run/daily_run_screen");

	private static readonly LocString _timeLeftLoc = new LocString("main_menu_ui", "DAILY_RUN_MENU.TIME_LEFT");

	public static readonly string dateFormat = LocManager.Instance.GetTable("main_menu_ui").GetRawText("DAILY_RUN_MENU.DATE_FORMAT");

	private static readonly string _timeLeftFormat = LocManager.Instance.GetTable("main_menu_ui").GetRawText("DAILY_RUN_MENU.TIME_FORMAT");

	private MegaLabel _titleLabel;

	private MegaRichTextLabel _dateLabel;

	private MegaRichTextLabel _timeLeftLabel;

	private NDailyRunCharacterContainer _characterContainer;

	private NConfirmButton _embarkButton;

	private NBackButton _backButton;

	private NBackButton _unreadyButton;

	private NDailyRunLeaderboard _leaderboard;

	private MegaLabel _modifiersTitleLabel;

	private Control _modifiersContainer;

	private readonly List<NDailyRunScreenModifier> _modifierContainers = new List<NDailyRunScreenModifier>();

	private NRemoteLobbyPlayerContainer _remotePlayerContainer;

	private Control _readyAndWaitingContainer;

	private DateTimeOffset _endOfDay;

	private INetGameService _netService;

	private StartRunLobby? _lobby;

	private int? _lastSetTimeLeftSecond;

	public static string[] AssetPaths => new string[1] { _scenePath };

	protected override Control? InitialFocusedControl => null;

	public static NDailyRunScreen? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NDailyRunScreen>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		ConnectSignals();
		_titleLabel = GetNode<MegaLabel>("%Title");
		_dateLabel = GetNode<MegaRichTextLabel>("%Date");
		_embarkButton = GetNode<NConfirmButton>("%ConfirmButton");
		_backButton = GetNode<NBackButton>("%BackButton");
		_unreadyButton = GetNode<NBackButton>("%UnreadyButton");
		_timeLeftLabel = GetNode<MegaRichTextLabel>("%TimeLeft");
		_leaderboard = GetNode<NDailyRunLeaderboard>("%Leaderboards");
		_modifiersTitleLabel = GetNode<MegaLabel>("%ModifiersLabel");
		_modifiersContainer = GetNode<Control>("%ModifiersContainer");
		_characterContainer = GetNode<NDailyRunCharacterContainer>("ChallengeContainer/CenterContainer/HBoxContainer/CharacterContainer");
		_remotePlayerContainer = GetNode<NRemoteLobbyPlayerContainer>("%RemotePlayerContainer");
		_readyAndWaitingContainer = GetNode<Control>("%ReadyAndWaitingPanel");
		_titleLabel.SetTextAutoSize(new LocString("main_menu_ui", "DAILY_RUN_MENU.DAILY_TITLE").GetFormattedText());
		_modifiersTitleLabel.SetTextAutoSize(new LocString("main_menu_ui", "DAILY_RUN_MENU.MODIFIERS").GetFormattedText());
		_dateLabel.SetTextAutoSize(new LocString("main_menu_ui", "DAILY_RUN_MENU.FETCHING_TIME").GetFormattedText());
		foreach (NDailyRunScreenModifier item in _modifiersContainer.GetChildren().OfType<NDailyRunScreenModifier>())
		{
			_modifierContainers.Add(item);
		}
		_embarkButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnEmbarkPressed));
		_embarkButton.Disable();
		_unreadyButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnUnreadyPressed));
		_unreadyButton.Disable();
		_remotePlayerContainer.Visible = false;
		_readyAndWaitingContainer.Visible = false;
		_leaderboard.Cleanup();
	}

	public void InitializeMultiplayerAsHost(INetGameService gameService)
	{
		if (gameService.Type != NetGameType.Host)
		{
			throw new InvalidOperationException($"Initialized character select screen with GameService of type {gameService.Type} when hosting!");
		}
		_netService = gameService;
	}

	public void InitializeMultiplayerAsClient(INetGameService gameService, ClientLobbyJoinResponseMessage message)
	{
		if (gameService.Type != NetGameType.Client)
		{
			throw new InvalidOperationException($"Initialized character select screen with GameService of type {gameService.Type} when joining!");
		}
		_netService = gameService;
		_lobby = new StartRunLobby(GameMode.Daily, gameService, this, message.dailyTime.Value, -1);
		_lobby.InitializeFromMessage(message);
		SetupLobbyParams(_lobby);
		AfterLobbyInitialized();
	}

	public void InitializeSingleplayer()
	{
		_netService = new NetSingleplayerGameService();
	}

	public override void OnSubmenuOpened()
	{
		base.OnSubmenuOpened();
		NetGameType type = _netService.Type;
		if ((uint)(type - 1) <= 1u)
		{
			TaskHelper.RunSafely(SetupLobbyForHostOrSingleplayer());
		}
		else
		{
			SetIsLoading(isLoading: false);
		}
	}

	public override void OnSubmenuClosed()
	{
		_embarkButton.Disable();
		_remotePlayerContainer.Cleanup();
		_leaderboard.Cleanup();
		StartRunLobby? lobby = _lobby;
		if (lobby != null && lobby.NetService.Type.IsMultiplayer())
		{
			PlatformUtil.SetRichPresence("MAIN_MENU", null, null);
		}
		CleanUpLobby(disconnectSession: true);
	}

	private void InitializeLeaderboard()
	{
		_leaderboard.Initialize(_lobby.DailyTime.Value.serverTime, _lobby.Players.Select((LobbyPlayer p) => p.id), allowPagination: false);
	}

	private async Task SetupLobbyForHostOrSingleplayer()
	{
		if (_netService.Type != NetGameType.Host && _netService.Type != NetGameType.Singleplayer)
		{
			throw new InvalidOperationException("Should only be called as host or singleplayer!");
		}
		SetIsLoading(isLoading: true);
		TimeServerResult timeServerResult = await GetTimeServerTime();
		if (GodotObject.IsInstanceValid(this))
		{
			_lobby = new StartRunLobby(GameMode.Daily, _netService, this, timeServerResult, 4);
			_lobby.AddLocalHostPlayer(new UnlockState(SaveManager.Instance.Progress), SaveManager.Instance.Progress.MaxMultiplayerAscension);
			SetupLobbyParams(_lobby);
			AfterLobbyInitialized();
			SetIsLoading(isLoading: false);
			Log.Info($"Daily initialized with seed: {_lobby.Seed} time: {GetServerRelativeTime()}");
		}
	}

	private async Task<TimeServerResult> GetTimeServerTime()
	{
		TimeServerResult? result = null;
		if (TimeServer.RequestTimeTask?.IsCompleted ?? false)
		{
			if (!TimeServer.RequestTimeTask.IsFaulted)
			{
				result = await TimeServer.RequestTimeTask;
			}
			if (!result.HasValue)
			{
				try
				{
					result = await TimeServer.FetchDailyTime();
				}
				catch (HttpRequestException ex)
				{
					Log.Error(ex.ToString());
				}
			}
		}
		else
		{
			try
			{
				result = await TimeServer.FetchDailyTime();
			}
			catch (HttpRequestException ex2)
			{
				Log.Error(ex2.ToString());
			}
		}
		if (!result.HasValue)
		{
			Log.Info("Couldn't retrieve time from time server, using local time");
			result = new TimeServerResult
			{
				serverTime = DateTimeOffset.UtcNow,
				localReceivedTime = DateTimeOffset.UtcNow
			};
		}
		return result.Value;
	}

	private DateTimeOffset GetServerRelativeTime()
	{
		return _lobby.DailyTime.Value.serverTime + (DateTimeOffset.UtcNow - _lobby.DailyTime.Value.localReceivedTime);
	}

	private void SetupLobbyParams(StartRunLobby lobby)
	{
		DateTimeOffset serverRelativeTime = GetServerRelativeTime();
		string str = SeedHelper.CanonicalizeSeed(serverRelativeTime.ToString("dd_MM_yyyy"));
		string text = SeedHelper.CanonicalizeSeed(serverRelativeTime.ToString($"dd_MM_yyyy_{lobby.Players.Count}p"));
		Rng rng = new Rng((uint)StringHelper.GetDeterministicHashCode(str));
		Rng rng2 = new Rng(rng.NextUnsignedInt());
		Rng rng3 = new Rng(rng.NextUnsignedInt());
		Rng rng4 = new Rng(rng.NextUnsignedInt());
		CharacterModel characterModel = null;
		foreach (LobbyPlayer player in lobby.Players)
		{
			CharacterModel characterModel2 = rng2.NextItem(ModelDb.AllCharacters);
			if (player.id == lobby.LocalPlayer.id)
			{
				characterModel = characterModel2;
			}
		}
		int num = rng3.NextInt(0, 11);
		List<ModifierModel> list = RollModifiers(rng4);
		NetGameType type = lobby.NetService.Type;
		if ((uint)(type - 1) <= 1u)
		{
			if (lobby.Seed != text)
			{
				lobby.SetSeed(text);
			}
			if (lobby.Ascension != num)
			{
				lobby.SyncAscensionChange(num);
			}
			if (list.Any((ModifierModel m) => lobby.Modifiers.FirstOrDefault(m.IsEquivalent) == null))
			{
				lobby.SetModifiers(list);
			}
		}
		if (lobby.LocalPlayer.character != characterModel)
		{
			lobby.SetLocalCharacter(characterModel);
		}
		InitializeDisplay();
	}

	private void InitializeDisplay()
	{
		if (_lobby == null)
		{
			throw new InvalidOperationException("Tried to initialize daily run display before lobby was initialized!");
		}
		DateTimeOffset serverRelativeTime = GetServerRelativeTime();
		_endOfDay = new DateTimeOffset(serverRelativeTime.Year, serverRelativeTime.Month, serverRelativeTime.Day, 0, 0, 0, TimeSpan.Zero) + TimeSpan.FromDays(1);
		_remotePlayerContainer.Visible = _lobby.NetService.Type.IsMultiplayer();
		CharacterModel character = _lobby.LocalPlayer.character;
		_characterContainer.Fill(character, _lobby.LocalPlayer.id, _lobby.Ascension, _lobby.NetService);
		_dateLabel.Modulate = StsColors.blue;
		_dateLabel.Text = serverRelativeTime.ToString(dateFormat);
		for (int i = 0; i < _lobby.Modifiers.Count; i++)
		{
			_modifierContainers[i].Fill(_lobby.Modifiers[i]);
		}
	}

	private List<ModifierModel> RollModifiers(Rng rng)
	{
		List<ModifierModel> list = new List<ModifierModel>();
		List<ModifierModel> list2 = ModelDb.GoodModifiers.ToList().StableShuffle(rng);
		for (int i = 0; i < 2; i++)
		{
			ModifierModel canonicalModifier = rng.NextItem(list2);
			if (canonicalModifier == null)
			{
				throw new InvalidOperationException("There were not enough good modifiers to fill the daily!");
			}
			ModifierModel modifierModel = canonicalModifier.ToMutable();
			if (modifierModel is CharacterCards characterCards)
			{
				IEnumerable<CharacterModel> second = _lobby.Players.Select((LobbyPlayer p) => p.character);
				characterCards.CharacterModel = rng.NextItem(ModelDb.AllCharacters.Except(second)).Id;
			}
			list.Add(modifierModel);
			list2.Remove(canonicalModifier);
			IReadOnlySet<ModifierModel> readOnlySet = ModelDb.MutuallyExclusiveModifiers.FirstOrDefault((IReadOnlySet<ModifierModel> s) => s.Contains(canonicalModifier));
			if (readOnlySet == null)
			{
				continue;
			}
			foreach (ModifierModel item in readOnlySet)
			{
				list2.Remove(item);
			}
		}
		list.Add(rng.NextItem(ModelDb.BadModifiers).ToMutable());
		return list;
	}

	private void SetIsLoading(bool isLoading)
	{
		if (isLoading)
		{
			_remotePlayerContainer.Visible = false;
			_readyAndWaitingContainer.Visible = false;
		}
		_timeLeftLabel.Visible = !isLoading;
		_characterContainer.Visible = !isLoading;
		_modifiersTitleLabel.Visible = !isLoading;
		_modifiersContainer.Visible = !isLoading;
		if (isLoading)
		{
			_embarkButton.Disable();
		}
		else
		{
			_embarkButton.Enable();
		}
	}

	public override void _Process(double delta)
	{
		if (_lobby != null)
		{
			DateTimeOffset serverRelativeTime = GetServerRelativeTime();
			if (serverRelativeTime > _endOfDay)
			{
				SetupLobbyParams(_lobby);
			}
			TimeSpan timeSpan = _endOfDay - serverRelativeTime;
			if (_lastSetTimeLeftSecond != timeSpan.Seconds)
			{
				string variable = timeSpan.ToString(_timeLeftFormat);
				_timeLeftLoc.Add("time", variable);
				_timeLeftLabel.Text = _timeLeftLoc.GetFormattedText();
				_lastSetTimeLeftSecond = timeSpan.Seconds;
			}
			if (_lobby.NetService.IsConnected)
			{
				_lobby.NetService.Update();
			}
		}
	}

	public void PlayerConnected(LobbyPlayer player)
	{
		_remotePlayerContainer.OnPlayerConnected(player);
		SetupLobbyParams(_lobby);
		InitializeLeaderboard();
		UpdateRichPresence();
	}

	public void PlayerChanged(LobbyPlayer player)
	{
		_remotePlayerContainer.OnPlayerChanged(player);
		if (player.id == _netService.NetId && _netService.Type.IsMultiplayer())
		{
			_characterContainer.SetIsReady(player.isReady);
		}
	}

	public void MaxAscensionChanged()
	{
	}

	public void AscensionChanged()
	{
		InitializeDisplay();
	}

	public void SeedChanged()
	{
	}

	public void ModifiersChanged()
	{
		InitializeDisplay();
	}

	public void RemotePlayerDisconnected(LobbyPlayer player)
	{
		_remotePlayerContainer.OnPlayerDisconnected(player);
		SetupLobbyParams(_lobby);
		InitializeLeaderboard();
		UpdateRichPresence();
	}

	public void BeginRun(string seed, List<ActModel> acts, IReadOnlyList<ModifierModel> modifiers)
	{
		NAudioManager.Instance?.StopMusic();
		if (_lobby.NetService.Type == NetGameType.Singleplayer)
		{
			TaskHelper.RunSafely(StartNewSingleplayerRun(seed, acts, modifiers));
		}
		else
		{
			TaskHelper.RunSafely(StartNewMultiplayerRun(seed, acts, modifiers));
		}
	}

	public void LocalPlayerDisconnected(NetErrorInfo info)
	{
		if (info.SelfInitiated && info.GetReason() == NetError.Quit)
		{
			return;
		}
		_stack.Pop();
		if (TestMode.IsOff)
		{
			NErrorPopup nErrorPopup = NErrorPopup.Create(info);
			if (nErrorPopup != null)
			{
				NModalContainer.Instance.Add(nErrorPopup);
			}
		}
	}

	private void OnEmbarkPressed(NButton _)
	{
		_embarkButton.Disable();
		_backButton.Disable();
		_lobby.SetReady(ready: true);
		if (_lobby.NetService.Type != NetGameType.Singleplayer && !_lobby.IsAboutToBeginGame())
		{
			_readyAndWaitingContainer.Visible = true;
			_unreadyButton.Enable();
		}
	}

	private void OnUnreadyPressed(NButton _)
	{
		_lobby.SetReady(ready: false);
		_readyAndWaitingContainer.Visible = false;
		_embarkButton.Enable();
		_backButton.Enable();
		_unreadyButton.Disable();
	}

	private void UpdateRichPresence()
	{
		StartRunLobby? lobby = _lobby;
		if (lobby != null && lobby.NetService.Type.IsMultiplayer())
		{
			PlatformUtil.SetRichPresence("DAILY_MP_LOBBY", _lobby.NetService.GetRawLobbyIdentifier(), _lobby.Players.Count);
		}
	}

	public async Task StartNewSingleplayerRun(string seed, List<ActModel> acts, IReadOnlyList<ModifierModel> modifiers)
	{
		Log.Info($"Embarking on a DAILY {_lobby.LocalPlayer.character.Id.Entry} run with {_lobby.Players.Count} players. Ascension: {_lobby.Ascension} Seed: {seed}");
		SfxCmd.Play(_lobby.LocalPlayer.character.CharacterTransitionSfx);
		await NGame.Instance.Transition.FadeOut(0.8f, _lobby.LocalPlayer.character.CharacterSelectTransitionPath);
		await NGame.Instance.StartNewSingleplayerRun(_lobby.LocalPlayer.character, shouldSave: true, acts, modifiers, seed, _lobby.Ascension, _lobby.DailyTime.Value.serverTime);
		CleanUpLobby(disconnectSession: false);
	}

	public async Task StartNewMultiplayerRun(string seed, List<ActModel> acts, IReadOnlyList<ModifierModel> modifiers)
	{
		Log.Info($"Embarking on a DAILY multiplayer run. Players: {string.Join(",", _lobby.Players)}. Ascension: {_lobby.Ascension} Seed: {seed}");
		SfxCmd.Play(_lobby.LocalPlayer.character.CharacterTransitionSfx);
		await NGame.Instance.Transition.FadeOut(0.8f, _lobby.LocalPlayer.character.CharacterSelectTransitionPath);
		await NGame.Instance.StartNewMultiplayerRun(_lobby, shouldSave: true, acts, modifiers, seed, _lobby.Ascension, _lobby.DailyTime.Value.serverTime);
		CleanUpLobby(disconnectSession: false);
	}

	private void CleanUpLobby(bool disconnectSession)
	{
		_lobby?.CleanUp(disconnectSession);
		_lobby = null;
	}

	private void AfterLobbyInitialized()
	{
		NGame.Instance.RemoteCursorContainer.Initialize(_lobby.InputSynchronizer, _lobby.Players.Select((LobbyPlayer p) => p.id));
		NGame.Instance.ReactionContainer.InitializeNetworking(_lobby.NetService);
		NGame.Instance.TimeoutOverlay.Initialize(_lobby.NetService, isGameLevel: true);
		_remotePlayerContainer.Initialize(_lobby, displayLocalPlayer: false);
		UpdateRichPresence();
		MegaCrit.Sts2.Core.Logging.Logger.logLevelTypeMap[LogType.Network] = ((_lobby.NetService.Type == NetGameType.Singleplayer) ? LogLevel.Info : LogLevel.Debug);
		MegaCrit.Sts2.Core.Logging.Logger.logLevelTypeMap[LogType.Actions] = ((_lobby.NetService.Type == NetGameType.Singleplayer) ? LogLevel.Info : LogLevel.VeryDebug);
		MegaCrit.Sts2.Core.Logging.Logger.logLevelTypeMap[LogType.GameSync] = ((_lobby.NetService.Type == NetGameType.Singleplayer) ? LogLevel.Info : LogLevel.VeryDebug);
		NGame.Instance.DebugSeedOverride = null;
		_embarkButton.Enable();
	}
}
