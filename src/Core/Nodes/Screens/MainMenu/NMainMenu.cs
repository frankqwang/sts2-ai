using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Daily;
using MegaCrit.Sts2.Core.Debug;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Multiplayer;
using MegaCrit.Sts2.Core.Multiplayer.Connection;
using MegaCrit.Sts2.Core.Nodes.Audio;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Multiplayer;
using MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;
using MegaCrit.Sts2.Core.Nodes.Screens.ModdingScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.ProfileScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Nodes.Screens.Settings;
using MegaCrit.Sts2.Core.Nodes.Screens.Timeline;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.Platform.Steam;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;
using MegaCrit.Sts2.Core.Timeline.Epochs;

namespace MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;

public partial class NMainMenu : Control, IScreenContext
{
	private static readonly StringName _lod = new StringName("lod");

	private static readonly StringName _mixPercentage = new StringName("mix_percentage");

	private const string _scenePath = "res://scenes/screens/main_menu.tscn";

	private const string _menuMusicParam = "menu_progress";

	private Window _window;

	private NMainMenuTextButton _continueButton;

	private NMainMenuTextButton _abandonRunButton;

	private NMainMenuTextButton _singleplayerButton;

	private NMainMenuTextButton _compendiumButton;

	private NMainMenuTextButton _timelineButton;

	private NMainMenuTextButton _settingsButton;

	private NMainMenuTextButton _quitButton;

	private NMainMenuTextButton _multiplayerButton;

	private const float _reticleYOffset = 5f;

	private const float _reticlePadding = 28f;

	private Control _buttonReticleLeft;

	private Control _buttonReticleRight;

	private Tween? _reticleTween;

	private NPatchNotesButton _patchNotesButtonNode;

	private NOpenProfileScreenButton _openProfileScreenButton;

	private NMainMenuTextButton? _lastHitButton;

	private NContinueRunInfo _runInfo;

	private Control _timelineNotificationDot;

	private Tween? _backstopTween;

	private NMainMenuBg _bg;

	private ShaderMaterial _blur;

	private bool _openTimeline;

	private ReadSaveResult<SerializableRun>? _readRunSaveResult;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>("res://scenes/screens/main_menu.tscn");

	public NPatchNotesScreen PatchNotesScreen { get; private set; }

	public Control BlurBackstop { get; private set; }

	private NButton[] MainMenuButtons => new NButton[8] { _continueButton, _abandonRunButton, _singleplayerButton, _multiplayerButton, _timelineButton, _settingsButton, _compendiumButton, _quitButton };

	public NMainMenuSubmenuStack SubmenuStack { get; private set; }

	public NContinueRunInfo ContinueRunInfo => _runInfo;

	public Control DefaultFocusedControl
	{
		get
		{
			if (_lastHitButton == null || !_lastHitButton.IsVisible())
			{
				return MainMenuButtons.First((NButton b) => b.IsEnabled && b.IsVisible());
			}
			return _lastHitButton;
		}
	}

	public static NMainMenu Create(bool openTimeline)
	{
		NMainMenu nMainMenu = PreloadManager.Cache.GetScene("res://scenes/screens/main_menu.tscn").Instantiate<NMainMenu>(PackedScene.GenEditState.Disabled);
		nMainMenu._openTimeline = openTimeline;
		return nMainMenu;
	}

	public override void _Ready()
	{
		Log.Info($"[Startup] Time to main menu (Godot ticks): {Time.GetTicksMsec()}ms");
		_window = GetTree().Root;
		NGame.Instance.Connect(NGame.SignalName.WindowChange, Callable.From<bool>(OnWindowChange));
		if (SaveManager.Instance.SettingsSave.AspectRatioSetting == AspectRatioSetting.Auto)
		{
			OnWindowChange(isAspectRatioAuto: true);
		}
		_continueButton = GetNode<NMainMenuTextButton>("MainMenuTextButtons/ContinueButton");
		_continueButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnContinueButtonPressed));
		_continueButton.SetLocalization("CONTINUE");
		_abandonRunButton = GetNode<NMainMenuTextButton>("MainMenuTextButtons/AbandonRunButton");
		_abandonRunButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnAbandonRunButtonPressed));
		_abandonRunButton.SetLocalization("ABANDON_RUN");
		_singleplayerButton = GetNode<NMainMenuTextButton>("MainMenuTextButtons/SingleplayerButton");
		_singleplayerButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(SingleplayerButtonPressed));
		_singleplayerButton.SetLocalization("SINGLE_PLAYER");
		_multiplayerButton = GetNode<NMainMenuTextButton>("MainMenuTextButtons/MultiplayerButton");
		_multiplayerButton.Connect(NClickableControl.SignalName.Released, Callable.From((Action<NButton>)OpenMultiplayerSubmenu));
		_multiplayerButton.SetLocalization("MULTIPLAYER");
		_compendiumButton = GetNode<NMainMenuTextButton>("MainMenuTextButtons/CompendiumButton");
		_compendiumButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OpenCompendiumSubmenu));
		_compendiumButton.SetLocalization("COMPENDIUM");
		_timelineButton = GetNode<NMainMenuTextButton>("MainMenuTextButtons/TimelineButton");
		_timelineButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OpenTimelineScreen));
		_timelineButton.SetLocalization("TIMELINE");
		_timelineNotificationDot = GetNode<Control>("%TimelineNotificationDot");
		_timelineNotificationDot.Visible = SaveManager.Instance.GetDiscoveredEpochCount() > 0;
		_settingsButton = GetNode<NMainMenuTextButton>("MainMenuTextButtons/SettingsButton");
		_settingsButton.Connect(NClickableControl.SignalName.Released, Callable.From((Action<NButton>)OpenSettingsMenu));
		_settingsButton.SetLocalization("SETTINGS");
		_quitButton = GetNode<NMainMenuTextButton>("MainMenuTextButtons/QuitButton");
		_quitButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(Quit));
		_quitButton.SetLocalization("QUIT");
		_buttonReticleLeft = GetNode<Control>("%ButtonReticleLeft");
		_buttonReticleRight = GetNode<Control>("%ButtonReticleRight");
		ConnectMainMenuTextButtonFocusLogic();
		PatchNotesScreen = GetNode<NPatchNotesScreen>("%PatchNotesScreen");
		SubmenuStack = GetNode<NMainMenuSubmenuStack>("%Submenus");
		_runInfo = GetNode<NContinueRunInfo>("%ContinueRunInfo");
		_patchNotesButtonNode = GetNode<NPatchNotesButton>("%PatchNotesButton");
		_patchNotesButtonNode.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OpenPatchNotes));
		_openProfileScreenButton = GetNode<NOpenProfileScreenButton>("%ChangeProfileButton");
		_bg = GetNode<NMainMenuBg>("%MainMenuBg");
		BlurBackstop = GetNode<Control>("BlurBackstop");
		_blur = (ShaderMaterial)BlurBackstop.Material;
		_timelineButton.Visible = SaveManager.Instance.Progress.Epochs.Count > 0;
		NGame.Instance.SetScreenShakeTarget(this);
		NAudioManager.Instance?.PlayMusic("event:/music/menu_update");
		SubmenuStack.InitializeForMainMenu(this);
		SubmenuStack.Connect(NSubmenuStack.SignalName.StackModified, Callable.From(OnSubmenuStackChanged));
		OnSubmenuStackChanged();
		ActiveScreenContext.Instance.Update();
		RefreshButtons();
		CheckCommandLineArgs();
		if (SaveManager.Instance.SettingsSave.ModSettings == null && ModManager.AllMods.Count > 0)
		{
			NModalContainer.Instance.Add(NConfirmModLoadingPopup.Create());
		}
		TaskHelper.RunSafely(NGame.Instance.Transition.FadeIn(3f));
		PlatformUtil.SetRichPresence("MAIN_MENU", null, null);
		if (_openTimeline)
		{
			TaskHelper.RunSafely(OpenTimelineFromGameOverScreen());
		}
		if (NGame.IsReleaseGame() && !SaveManager.Instance.SettingsSave.SeenEaDisclaimer)
		{
			NModalContainer.Instance.Add(NEarlyAccessDisclaimer.Create());
		}
	}

	private void ConnectMainMenuTextButtonFocusLogic()
	{
		foreach (NMainMenuTextButton item in GetNode<Control>("%MainMenuTextButtons").GetChildren().OfType<NMainMenuTextButton>())
		{
			item.Connect(NClickableControl.SignalName.Unfocused, Callable.From<NMainMenuTextButton>(MainMenuButtonUnfocused));
			item.Connect(NClickableControl.SignalName.Focused, Callable.From(delegate(NMainMenuTextButton b)
			{
				Callable.From(delegate
				{
					MainMenuButtonFocused(b);
				}).CallDeferred();
			}));
		}
	}

	private void MainMenuButtonFocused(NMainMenuTextButton button)
	{
		_reticleTween?.Kill();
		_reticleTween = CreateTween().SetParallel();
		_buttonReticleLeft.GlobalPosition = new Vector2(0f, button.GlobalPosition.Y + 5f);
		_buttonReticleRight.GlobalPosition = new Vector2(0f, button.GlobalPosition.Y + 5f);
		float num = button.label?.GlobalPosition.X ?? 0f;
		float num2 = button.label?.Size.X ?? 0f;
		float num3 = num - 20f - 6f;
		float num4 = num + num2 - 20f + 6f;
		_reticleTween.TweenProperty(_buttonReticleLeft, "global_position:x", num3 - 28f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
			.From(num3);
		_reticleTween.TweenProperty(_buttonReticleLeft, "modulate", StsColors.gold, 0.05).From(StsColors.transparentWhite);
		_reticleTween.TweenProperty(_buttonReticleRight, "global_position:x", num4 + 28f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
			.From(num4);
		_reticleTween.TweenProperty(_buttonReticleRight, "modulate", StsColors.gold, 0.05).From(StsColors.transparentWhite);
	}

	private void MainMenuButtonUnfocused(NMainMenuTextButton obj)
	{
		_reticleTween?.Kill();
		_reticleTween = CreateTween().SetParallel();
		_reticleTween.TweenProperty(_buttonReticleLeft, "modulate", StsColors.transparentWhite, 0.25);
		_reticleTween.TweenProperty(_buttonReticleRight, "modulate", StsColors.transparentWhite, 0.25);
	}

	private async Task OpenTimelineFromGameOverScreen()
	{
		if (SaveManager.Instance.PrefsSave.FastMode != FastModeType.Instant)
		{
			await Task.Delay(500);
		}
		SubmenuStack.PushSubmenuType<NTimelineScreen>();
	}

	public void EnableBackstop()
	{
		_bg.HideLogo();
		BlurBackstop.MouseFilter = MouseFilterEnum.Stop;
		_backstopTween?.Kill();
		_backstopTween = CreateTween().SetParallel();
		_backstopTween.TweenMethod(Callable.From<float>(UpdateShaderLod), _blur.GetShaderParameter(_lod), 3f, 0.25);
		_backstopTween.TweenMethod(Callable.From<float>(UpdateShaderMix), _blur.GetShaderParameter(_mixPercentage), 0.7f, 0.25);
	}

	public void DisableBackstop()
	{
		_bg.ShowLogo();
		BlurBackstop.MouseFilter = MouseFilterEnum.Ignore;
		_backstopTween?.Kill();
		_backstopTween = CreateTween().SetParallel();
		_backstopTween.TweenMethod(Callable.From<float>(UpdateShaderLod), _blur.GetShaderParameter(_lod), 0f, 0.15);
		_backstopTween.TweenMethod(Callable.From<float>(UpdateShaderMix), _blur.GetShaderParameter(_mixPercentage), 0f, 0.15);
	}

	public void DisableBackstopInstantly()
	{
		_backstopTween?.Kill();
		_blur.SetShaderParameter(_lod, 0f);
		_blur.SetShaderParameter(_mixPercentage, 0f);
	}

	public void EnableBackstopInstantly()
	{
		_backstopTween?.Kill();
		_blur.SetShaderParameter(_lod, 3f);
		_blur.SetShaderParameter(_mixPercentage, 0.7f);
	}

	private void UpdateShaderMix(float obj)
	{
		_blur.SetShaderParameter(_mixPercentage, obj);
	}

	private void UpdateShaderLod(float obj)
	{
		_blur.SetShaderParameter(_lod, obj);
	}

	public void RefreshButtons()
	{
		if (SaveManager.Instance.HasRunSave)
		{
			_readRunSaveResult = SaveManager.Instance.LoadRunSave();
			_singleplayerButton.Visible = false;
			_abandonRunButton.Visible = true;
			_continueButton.Visible = true;
			_continueButton.SetEnabled(enabled: true);
			_runInfo.SetResult(_readRunSaveResult);
		}
		else
		{
			_readRunSaveResult = null;
			_singleplayerButton.Visible = true;
			_abandonRunButton.Visible = false;
			_continueButton.Visible = false;
			_continueButton.SetEnabled(enabled: false);
			_runInfo.SetResult(null);
		}
		UpdateTimelineButtonBehavior();
		_compendiumButton.Visible = SaveManager.Instance.IsCompendiumAvailable();
		ActiveScreenContext.Instance.Update();
	}

	private void UpdateTimelineButtonBehavior()
	{
		if (!DebugSettings.DevSkip && SaveManager.Instance.GetDiscoveredEpochCount() > 0 && !SaveManager.Instance.HasRunSave)
		{
			_timelineButton.Enable();
			_singleplayerButton.Disable();
			_multiplayerButton.Disable();
			_compendiumButton.Disable();
			_timelineButton.Visible = true;
			_timelineNotificationDot.Visible = true;
		}
		else if (SaveManager.Instance.Progress.Epochs.Count > 1 && SaveManager.Instance.IsEpochRevealed<NeowEpoch>())
		{
			_timelineButton.Visible = true;
			if (SaveManager.Instance.GetDiscoveredEpochCount() == 0)
			{
				_timelineButton.Enable();
			}
			else
			{
				_timelineButton.Disable();
			}
			_timelineNotificationDot.Visible = false;
			_singleplayerButton.Enable();
			_multiplayerButton.Enable();
			_compendiumButton.Enable();
		}
		else if (SaveManager.Instance.Progress.Epochs.Count > 1)
		{
			_timelineButton.Disable();
		}
		else
		{
			_timelineButton.Visible = false;
		}
	}

	private void OnSubmenuStackChanged()
	{
		GetNode<Control>("MainMenuTextButtons").Visible = !SubmenuStack.SubmenusOpen;
		_patchNotesButtonNode.Visible = !SubmenuStack.SubmenusOpen;
		if (SubmenuStack.SubmenusOpen)
		{
			_openProfileScreenButton.Visible = false;
			_patchNotesButtonNode.Visible = false;
			_openProfileScreenButton.Disable();
			_patchNotesButtonNode.Disable();
		}
		else
		{
			_openProfileScreenButton.Visible = true;
			_patchNotesButtonNode.Visible = true;
			_openProfileScreenButton.Enable();
			_patchNotesButtonNode.Enable();
			NAudioManager.Instance?.UpdateMusicParameter("menu_progress", "main");
		}
	}

	private void OnContinueButtonPressed(NButton _)
	{
		if (_readRunSaveResult == null || !_readRunSaveResult.Success || _readRunSaveResult.SaveData == null)
		{
			DisplayLoadSaveError();
		}
		else
		{
			TaskHelper.RunSafely(OnContinueButtonPressedAsync());
		}
	}

	private async Task OnContinueButtonPressedAsync()
	{
		_ = 2;
		try
		{
			_continueButton.Disable();
			NAudioManager.Instance?.StopMusic();
			SerializableRun serializableRun = _readRunSaveResult.SaveData;
			RunState runState = RunState.FromSerializable(serializableRun);
			RunManager.Instance.SetUpSavedSinglePlayer(runState, serializableRun);
			Log.Info($"Continuing run with character: {serializableRun.Players[0].CharacterId}");
			SfxCmd.Play(runState.Players[0].Character.CharacterTransitionSfx);
			await NGame.Instance.Transition.FadeOut(0.8f, runState.Players[0].Character.CharacterSelectTransitionPath);
			NGame.Instance.ReactionContainer.InitializeNetworking(new NetSingleplayerGameService());
			await NGame.Instance.LoadRun(runState, serializableRun.PreFinishedRoom);
			await NGame.Instance.Transition.FadeIn();
		}
		catch (Exception)
		{
			DisplayLoadSaveError();
			throw;
		}
	}

	private void DisplayLoadSaveError()
	{
		NErrorPopup modalToCreate = NErrorPopup.Create(new LocString("main_menu_ui", "INVALID_SAVE_POPUP.title"), new LocString("main_menu_ui", "INVALID_SAVE_POPUP.description_run"), new LocString("main_menu_ui", "INVALID_SAVE_POPUP.dismiss"), showReportBugButton: true);
		NModalContainer.Instance.Add(modalToCreate);
		NModalContainer.Instance.ShowBackstop();
		_continueButton.Disable();
	}

	private void OnAbandonRunButtonPressed(NButton _)
	{
		NModalContainer.Instance.Add(NAbandonRunConfirmPopup.Create(this));
		_lastHitButton = _abandonRunButton;
	}

	public void AbandonRun()
	{
		if (_readRunSaveResult == null)
		{
			return;
		}
		if (_readRunSaveResult.Success && _readRunSaveResult.SaveData != null)
		{
			try
			{
				Log.Info("Abandoning run from main menu");
				SerializableRun saveData = _readRunSaveResult.SaveData;
				SaveManager.Instance.UpdateProgressWithRunData(saveData, victory: false);
				RunHistoryUtilities.CreateRunHistoryEntry(saveData, victory: false, isAbandoned: true, saveData.PlatformType);
				if (saveData.DailyTime.HasValue)
				{
					int score = ScoreUtility.CalculateScore(saveData, won: false);
					TaskHelper.RunSafely(DailyRunUtility.UploadScore(saveData.DailyTime.Value, score, saveData.Players));
				}
			}
			catch (Exception value)
			{
				Log.Error($"ERROR: Failed to upload run history/metrics: {value}");
			}
		}
		else
		{
			Log.Info($"Abandoning run with invalid save (status={_readRunSaveResult.Status})");
		}
		SaveManager.Instance.DeleteCurrentRun();
		RefreshButtons();
		GC.Collect();
	}

	private void SingleplayerButtonPressed(NButton _)
	{
		if (SaveManager.Instance.Progress.NumberOfRuns > 0)
		{
			OpenSingleplayerSubmenu();
			return;
		}
		NCharacterSelectScreen submenuType = SubmenuStack.GetSubmenuType<NCharacterSelectScreen>();
		submenuType.InitializeSingleplayer();
		SubmenuStack.Push(submenuType);
		_lastHitButton = _singleplayerButton;
	}

	public NSingleplayerSubmenu OpenSingleplayerSubmenu()
	{
		_lastHitButton = _singleplayerButton;
		return SubmenuStack.PushSubmenuType<NSingleplayerSubmenu>();
	}

	public void OpenMultiplayerSubmenu(NButton _)
	{
		OpenMultiplayerSubmenu();
	}

	public NMultiplayerSubmenu OpenMultiplayerSubmenu()
	{
		_lastHitButton = _multiplayerButton;
		return SubmenuStack.PushSubmenuType<NMultiplayerSubmenu>();
	}

	private void OpenCompendiumSubmenu(NButton _)
	{
		_lastHitButton = _compendiumButton;
		SubmenuStack.PushSubmenuType<NCompendiumSubmenu>();
	}

	private void OpenTimelineScreen(NButton obj)
	{
		_lastHitButton = _timelineButton;
		NAudioManager.Instance?.UpdateMusicParameter("menu_progress", "timeline");
		SubmenuStack.PushSubmenuType<NTimelineScreen>();
	}

	private void OpenSettingsMenu(NButton _)
	{
		OpenSettingsMenu();
	}

	public void OpenProfileScreen()
	{
		SubmenuStack.PushSubmenuType<NProfileScreen>();
	}

	public void OpenSettingsMenu()
	{
		_lastHitButton = _settingsButton;
		SubmenuStack.PushSubmenuType<NSettingsScreen>();
	}

	private void OpenPatchNotes(NButton _)
	{
		PatchNotesScreen.Open();
	}

	public async Task JoinGame(IClientConnectionInitializer connInitializer)
	{
		NMultiplayerSubmenu nMultiplayerSubmenu = OpenMultiplayerSubmenu();
		NJoinFriendScreen nJoinFriendScreen = nMultiplayerSubmenu.OnJoinFriendsPressed();
		await nJoinFriendScreen.JoinGameAsync(connInitializer);
	}

	private static void Quit(NButton _)
	{
		Log.Info("Quit button pressed");
		TaskHelper.RunSafely(ConfirmAndQuit());
	}

	private static async Task ConfirmAndQuit()
	{
		NGenericPopup nGenericPopup = NGenericPopup.Create();
		NModalContainer.Instance.Add(nGenericPopup);
		if (await nGenericPopup.WaitForConfirmation(new LocString("main_menu_ui", "QUIT_CONFIRM_POPUP.body"), new LocString("main_menu_ui", "QUIT_CONFIRM_POPUP.header"), new LocString("main_menu_ui", "GENERIC_POPUP.cancel"), new LocString("main_menu_ui", "GENERIC_POPUP.confirm")))
		{
			Log.Info("Quit confirmed");
			NGame.Instance?.Quit();
		}
		else
		{
			Log.Info("Quit cancelled");
		}
	}

	private void OnWindowChange(bool isAspectRatioAuto)
	{
		if (isAspectRatioAuto)
		{
			float num = (float)_window.Size.X / (float)_window.Size.Y;
			if (num > 2.3888888f)
			{
				_window.ContentScaleAspect = Window.ContentScaleAspectEnum.KeepWidth;
				_window.ContentScaleSize = new Vector2I(2580, 1080);
			}
			else if (num < 1.3333334f)
			{
				_window.ContentScaleAspect = Window.ContentScaleAspectEnum.KeepHeight;
				_window.ContentScaleSize = new Vector2I(1680, 1260);
			}
			else
			{
				_window.ContentScaleAspect = Window.ContentScaleAspectEnum.Expand;
				_window.ContentScaleSize = new Vector2I(1680, 1080);
			}
		}
	}

	private void CheckCommandLineArgs()
	{
		if (!CommandLineHelper.TryGetValue("fastmp", out string value))
		{
			return;
		}
		NMultiplayerSubmenu nMultiplayerSubmenu = OpenMultiplayerSubmenu();
		switch (value)
		{
		case "host":
		case "host_standard":
		case "host_daily":
		case "host_custom":
		{
			GameMode gameMode = value switch
			{
				"host_standard" => GameMode.Standard, 
				"host_daily" => GameMode.Daily, 
				"host_custom" => GameMode.Custom, 
				_ => GameMode.None, 
			};
			if (gameMode != GameMode.None)
			{
				nMultiplayerSubmenu.FastHost(gameMode);
			}
			break;
		}
		case "load":
		{
			PlatformType platformType = (SteamInitializer.Initialized ? PlatformType.Steam : PlatformType.None);
			ulong localPlayerId = PlatformUtil.GetLocalPlayerId(platformType);
			ReadSaveResult<SerializableRun> readSaveResult = SaveManager.Instance.LoadAndCanonicalizeMultiplayerRunSave(localPlayerId);
			if (readSaveResult.SaveData != null)
			{
				nMultiplayerSubmenu.StartHost(readSaveResult.SaveData);
			}
			else
			{
				Log.Error("Failed to load multiplayer save");
			}
			break;
		}
		case "join":
			nMultiplayerSubmenu.OnJoinFriendsPressed();
			break;
		default:
			Log.Error("fastmp command line argument passed with invalid value: " + value + ". Expected host, load, or join");
			break;
		}
	}
}
