using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Multiplayer.Connection;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;
using MegaCrit.Sts2.Core.Nodes.Screens.CustomRun;
using MegaCrit.Sts2.Core.Nodes.Screens.DailyRun;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.Platform.Steam;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;

public partial class NJoinFriendScreen : NSubmenu
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/join_friend_submenu");

	private Control _buttonContainer;

	private Control _loadingOverlay;

	private Control _loadingFriendsIndicator;

	private MegaLabel _noFriendsLabel;

	private NJoinFriendRefreshButton _refreshButton;

	private Task? _refreshTask;

	private JoinFlow? _currentJoinFlow;

	protected override Control? InitialFocusedControl
	{
		get
		{
			if (_buttonContainer.GetChildCount() <= 0)
			{
				return _refreshButton;
			}
			return _buttonContainer.GetChild<Control>(0);
		}
	}

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlyArray<string>(new string[2]
	{
		_scenePath,
		NJoinFriendButton.scenePath
	});

	public bool DebugFriendsButtons => false;

	public static NJoinFriendScreen? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NJoinFriendScreen>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		ConnectSignals();
		_buttonContainer = GetNode<Control>("%ButtonContainer");
		_loadingOverlay = GetNode<Control>("%LoadingOverlay");
		_loadingFriendsIndicator = GetNode<Control>("%LoadingIndicator");
		_noFriendsLabel = GetNode<MegaLabel>("%NoFriendsText");
		_refreshButton = GetNode<NJoinFriendRefreshButton>("%RefreshButton");
		GetNode<MegaLabel>("TitleLabel").SetTextAutoSize(new LocString("main_menu_ui", "JOIN_FRIENDS_MENU.title").GetFormattedText());
		_noFriendsLabel.SetTextAutoSize(new LocString("main_menu_ui", "JOIN_FRIENDS_MENU.noFriends").GetFormattedText());
		_refreshButton.Connect(NClickableControl.SignalName.Released, Callable.From<NClickableControl>(delegate
		{
			RefreshButtonClicked();
		}));
		_loadingFriendsIndicator.Visible = false;
		_noFriendsLabel.Visible = false;
	}

	public override void OnSubmenuOpened()
	{
		base.OnSubmenuOpened();
		_loadingOverlay.Visible = false;
		if ((!SteamInitializer.Initialized || CommandLineHelper.HasArg("fastmp")) && !DebugFriendsButtons)
		{
			TaskHelper.RunSafely(FastMpJoin());
		}
		else
		{
			_refreshTask = TaskHelper.RunSafely(ShowFriends());
		}
	}

	public override void OnSubmenuClosed()
	{
		_currentJoinFlow?.CancelToken.Cancel();
	}

	private async Task FastMpJoin()
	{
		ulong netId = 1000uL;
		if (CommandLineHelper.TryGetValue("clientId", out string value))
		{
			netId = ulong.Parse(value);
		}
		DisplayServer.WindowSetTitle("Slay The Spire 2 (Client)");
		await JoinGameAsync(new ENetClientConnectionInitializer(netId, "127.0.0.1", 33771));
	}

	private void RefreshButtonClicked()
	{
		Task refreshTask = _refreshTask;
		if (refreshTask == null || refreshTask.IsCompleted)
		{
			_refreshTask = TaskHelper.RunSafely(RefreshButtonClickedAsync());
		}
	}

	private async Task RefreshButtonClickedAsync()
	{
		_noFriendsLabel.Visible = false;
		if (SteamInitializer.Initialized)
		{
			_loadingFriendsIndicator.Visible = true;
			await Cmd.Wait(0.5f);
			_loadingFriendsIndicator.Visible = false;
		}
		await ShowFriends();
		InitialFocusedControl?.TryGrabFocus();
	}

	private async Task ShowFriends()
	{
		_loadingFriendsIndicator.Visible = true;
		foreach (Node child2 in _buttonContainer.GetChildren())
		{
			child2.QueueFreeSafely();
		}
		if (SteamInitializer.Initialized)
		{
			IEnumerable<ulong> enumerable = await PlatformUtil.GetFriendsWithOpenLobbies(PlatformType.Steam);
			NButton nButton = null;
			foreach (ulong item in enumerable)
			{
				NJoinFriendButton nJoinFriendButton = NJoinFriendButton.Create(item);
				_buttonContainer.AddChildSafely(nJoinFriendButton);
				SteamClientConnectionInitializer connInitializer = SteamClientConnectionInitializer.FromPlayer(item);
				nJoinFriendButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(delegate
				{
					JoinGame(connInitializer);
				}));
				if (nButton == null)
				{
					nButton = nJoinFriendButton;
				}
			}
		}
		if (DebugFriendsButtons)
		{
			for (int num = 0; num < Rng.Chaotic.NextInt(5, 20); num++)
			{
				ulong playerId = (ulong)(int)((num == 0) ? 1u : ((uint)(num * 1000)));
				NJoinFriendButton child = NJoinFriendButton.Create(playerId);
				_buttonContainer.AddChildSafely(child);
			}
		}
		ActiveScreenContext.Instance.Update();
		_loadingFriendsIndicator.Visible = false;
		_noFriendsLabel.Visible = _buttonContainer.GetChildCount() == 0;
	}

	private void JoinGame(IClientConnectionInitializer connInitializer)
	{
		TaskHelper.RunSafely(JoinGameAsync(connInitializer));
	}

	public async Task JoinGameAsync(IClientConnectionInitializer connInitializer)
	{
		if (_currentJoinFlow?.NetService?.IsConnected == true)
		{
			Log.Warn($"Tried to join game with connection {connInitializer} while we were already joining a game! Ignoring this attempt");
			return;
		}
		_loadingOverlay.Visible = true;
		_currentJoinFlow = new JoinFlow();
		try
		{
			Log.Info($"Attempting to join game with connection initializer {connInitializer}");
			JoinResult joinResult = await _currentJoinFlow.Begin(connInitializer, GetTree());
			if (joinResult.sessionState == RunSessionState.InLobby)
			{
				if (joinResult.gameMode == GameMode.Standard)
				{
					NCharacterSelectScreen submenuType = _stack.GetSubmenuType<NCharacterSelectScreen>();
					submenuType.InitializeMultiplayerAsClient(_currentJoinFlow.NetService, joinResult.joinResponse.Value);
					_stack.Push(submenuType);
					return;
				}
				if (joinResult.gameMode == GameMode.Daily)
				{
					NDailyRunScreen submenuType2 = _stack.GetSubmenuType<NDailyRunScreen>();
					submenuType2.InitializeMultiplayerAsClient(_currentJoinFlow.NetService, joinResult.joinResponse.Value);
					_stack.Push(submenuType2);
					return;
				}
				if (joinResult.gameMode != GameMode.Custom)
				{
					throw new ArgumentOutOfRangeException("gameMode", joinResult.gameMode, "Invalid game mode!");
				}
				NCustomRunScreen submenuType3 = _stack.GetSubmenuType<NCustomRunScreen>();
				submenuType3.InitializeMultiplayerAsClient(_currentJoinFlow.NetService, joinResult.joinResponse.Value);
				_stack.Push(submenuType3);
			}
			else if (joinResult.sessionState == RunSessionState.InLoadedLobby)
			{
				if (joinResult.gameMode == GameMode.Standard)
				{
					NMultiplayerLoadGameScreen submenuType4 = _stack.GetSubmenuType<NMultiplayerLoadGameScreen>();
					submenuType4.InitializeAsClient(_currentJoinFlow.NetService, joinResult.loadJoinResponse.Value);
					_stack.Push(submenuType4);
				}
				else if (joinResult.gameMode == GameMode.Daily)
				{
					NDailyRunLoadScreen submenuType5 = _stack.GetSubmenuType<NDailyRunLoadScreen>();
					submenuType5.InitializeAsClient(_currentJoinFlow.NetService, joinResult.loadJoinResponse.Value);
					_stack.Push(submenuType5);
				}
				else if (joinResult.gameMode == GameMode.Custom)
				{
					NCustomRunLoadScreen submenuType6 = _stack.GetSubmenuType<NCustomRunLoadScreen>();
					submenuType6.InitializeAsClient(_currentJoinFlow.NetService, joinResult.loadJoinResponse.Value);
					_stack.Push(submenuType6);
				}
			}
			else if (joinResult.sessionState == RunSessionState.Running)
			{
				throw new NotImplementedException("Rejoining a game is not yet implemented");
			}
		}
		catch (ClientConnectionFailedException ex)
		{
			Log.Error($"Received connection failed exception while joining game: {ex}");
			NErrorPopup nErrorPopup = NErrorPopup.Create(ex.info);
			if (nErrorPopup != null)
			{
				NModalContainer.Instance.Add(nErrorPopup);
			}
			_currentJoinFlow.NetService?.Disconnect(ex.info.GetReason());
		}
		catch (OperationCanceledException)
		{
			Log.Warn("Joining was canceled by user");
		}
		catch
		{
			Log.Error("Received unexpected exception while joining game! Disconnecting with InternalError");
			NErrorPopup nErrorPopup2 = NErrorPopup.Create(new NetErrorInfo(NetError.InternalError, selfInitiated: false));
			if (nErrorPopup2 != null)
			{
				NModalContainer.Instance.Add(nErrorPopup2);
			}
			_currentJoinFlow.NetService?.Disconnect(NetError.InternalError);
			throw;
		}
		finally
		{
			if (GodotObject.IsInstanceValid(this))
			{
				_loadingOverlay.Visible = false;
			}
		}
	}
}
