using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.AutoSlay;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Debug;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Extensions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Leaderboard;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Multiplayer.Game.Lobby;
using MegaCrit.Sts2.Core.Nodes.Audio;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Debug;
using MegaCrit.Sts2.Core.Nodes.Multiplayer;
using MegaCrit.Sts2.Core.Nodes.Reaction;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.FeedbackScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.InspectScreens;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Nodes.Screens.Settings;
using MegaCrit.Sts2.Core.Nodes.Vfx;
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.Platform.Steam;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Runs;
using MegaCrit.Sts2.Core.Settings;
using MegaCrit.Sts2.Core.Unlocks;

namespace MegaCrit.Sts2.Core.Nodes;

public partial class NGame : Control
{
	[Signal]
	public delegate void WindowChangeEventHandler();

	public static readonly Vector2 devResolution = new Vector2(1920f, 1080f);

	private Control _inspectionContainer;

	private NScreenShake _screenShake;

	private static int? _mainThreadId;

	private static Window _window = null;

	private CancellationTokenSource? _logoCancelToken;

	private SteamJoinCallbackHandler? _joinCallbackHandler;

	public static NGame? Instance { get; private set; }

	public NSceneContainer RootSceneContainer { get; private set; }

	public Node? HoverTipsContainer { get; private set; }

	public NMainMenu? MainMenu => RootSceneContainer.CurrentScene as NMainMenu;

	public NRun? CurrentRunNode => RootSceneContainer.CurrentScene as NRun;

	public NLogoAnimation? LogoAnimation => RootSceneContainer.CurrentScene as NLogoAnimation;

	public NTransition Transition { get; private set; }

	public NMultiplayerTimeoutOverlay TimeoutOverlay { get; private set; }

	public NAudioManager AudioManager { get; private set; }

	public NRemoteMouseCursorContainer RemoteCursorContainer { get; private set; }

	public NInputManager InputManager { get; private set; }

	public NHotkeyManager HotkeyManager { get; private set; }

	public NReactionWheel ReactionWheel { get; private set; }

	public NReactionContainer ReactionContainer { get; private set; }

	public NCursorManager CursorManager { get; private set; }

	public NDebugAudioManager DebugAudio { get; private set; }

	public string? DebugSeedOverride { get; set; }

	public bool StartOnMainMenu { get; set; } = true;

	public static bool IsTrailerMode { get; private set; }

	public static bool IsDebugHidingHoverTips { get; private set; }

	public static bool IsDebugHidingProceedButton { get; private set; }

	public NInspectRelicScreen? InspectRelicScreen { get; set; }

	public NInspectCardScreen? InspectCardScreen { get; set; }

	public NSendFeedbackScreen FeedbackScreen { get; set; }

	private WorldEnvironment WorldEnvironment { get; set; }

	private NHitStop HitStop { get; set; }

	public event Action? DebugToggleProceedButton;

	public override void _EnterTree()
	{
		if (Instance != null)
		{
			Log.Error("NGame already exists.");
			this.QueueFreeSafely();
			return;
		}
		Instance = this;
		SentryService.Initialize();
		RootSceneContainer = GetNode<NSceneContainer>("%RootSceneContainer");
		HoverTipsContainer = GetNode<Node>("%HoverTipsContainer");
		DebugAudio = GetNode<NDebugAudioManager>("%DebugAudioManager");
		AudioManager = GetNode<NAudioManager>("%AudioManager");
		RemoteCursorContainer = GetNode<NRemoteMouseCursorContainer>("%RemoteCursorContainer");
		InputManager = GetNode<NInputManager>("%InputManager");
		CursorManager = GetNode<NCursorManager>("%CursorManager");
		ReactionWheel = GetNode<NReactionWheel>("%ReactionWheel");
		ReactionContainer = GetNode<NReactionContainer>("%ReactionContainer");
		TimeoutOverlay = GetNode<NMultiplayerTimeoutOverlay>("%MultiplayerTimeoutOverlay");
		WorldEnvironment = GetNode<WorldEnvironment>("%WorldEnvironment");
		HotkeyManager = GetNode<NHotkeyManager>("%HotkeyManager");
		_inspectionContainer = GetNode<Control>("%InspectionContainer");
		_screenShake = GetNode<NScreenShake>("ScreenShake");
		HitStop = GetNode<NHitStop>("HitStop");
		Transition = GetNode<NTransition>("%GameTransitionRect");
		FeedbackScreen = GetNode<NSendFeedbackScreen>("%FeedbackScreen");
		_mainThreadId = System.Environment.CurrentManagedThreadId;
		TaskHelper.RunSafely(GameStartupWrapper());
	}

	public override void _Ready()
	{
		_window = GetTree().Root;
		_window.Connect(Viewport.SignalName.SizeChanged, Callable.From(OnWindowChange));
		this.RemoveChildSafely(WorldEnvironment);
	}

	private async Task GameStartupWrapper()
	{
		if (!(await InitializePlatform()))
		{
			return;
		}
		TaskHelper.RunSafely(OsDebugInfo.LogSystemInfo());
		TaskHelper.RunSafely(GitHelper.Initialize());
		try
		{
			await GameStartup();
		}
		catch
		{
			TaskHelper.RunSafely(GameStartupError());
			throw;
		}
	}

	private async Task TryErrorInit()
	{
		try
		{
			if (SaveManager.Instance.SettingsSave == null)
			{
				SaveManager.Instance.InitSettingsData();
			}
			if (LocManager.Instance == null)
			{
				LocManager.Initialize();
			}
		}
		catch (Exception value)
		{
			Log.Error($"Failed to show error dialog! Exception: {value}");
			GetTree().Quit();
			throw;
		}
		if (!IsNodeReady())
		{
			await ToSignal(this, Node.SignalName.Ready);
		}
		Transition.Visible = false;
	}

	private async Task GameStartupError()
	{
		Log.Error("Encountered error on game startup! Attempting to show error dialog");
		await TryErrorInit();
		NGenericPopup nGenericPopup = NGenericPopup.Create();
		NModalContainer.Instance.Add(nGenericPopup);
		await nGenericPopup.WaitForConfirmation(new LocString("main_menu_ui", "STARTUP_ERROR.description"), new LocString("main_menu_ui", "STARTUP_ERROR.title"), null, new LocString("main_menu_ui", "QUIT"));
		GetTree().Quit();
	}

	private async Task GameStartup()
	{
		AccountScopeUserDataMigrator.MigrateToUserScopedDirectories();
		AccountScopeUserDataMigrator.ArchiveLegacyData();
		ProfileAccountScopeMigrator.MigrateToProfileScopedDirectories();
		ProfileAccountScopeMigrator.ArchiveLegacyData();
		bool flag = await SaveManager.Instance.TryFirstTimeCloudSync();
		Task cloudSavesTask = null;
		if (!flag)
		{
			cloudSavesTask = Task.Run((Func<Task?>)SaveManager.Instance.SyncCloudToLocal);
		}
		InitPools();
		OneTimeInitialization.ExecuteEssential();
		if (!IsNodeReady())
		{
			await ToSignal(this, Node.SignalName.Ready);
		}
		Callable.From(InitializeGraphicsPreferences).CallDeferred();
		AudioManager.SetMasterVol(SaveManager.Instance.SettingsSave.VolumeMaster);
		AudioManager.SetSfxVol(SaveManager.Instance.SettingsSave.VolumeSfx);
		AudioManager.SetAmbienceVol(SaveManager.Instance.SettingsSave.VolumeAmbience);
		AudioManager.SetBgmVol(SaveManager.Instance.SettingsSave.VolumeBgm);
		DebugAudio.SetMasterAudioVolume(SaveManager.Instance.SettingsSave.VolumeMaster);
		DebugAudio.SetSfxAudioVolume(SaveManager.Instance.SettingsSave.VolumeSfx);
		LeaderboardManager.Initialize();
		SteamStatsManager.Initialize();
		if (cloudSavesTask != null)
		{
			await cloudSavesTask;
		}
		SaveManager.Instance.InitProfileId();
		ReadSaveResult<SerializableProgress> progressReadResult = SaveManager.Instance.InitProgressData();
		ReadSaveResult<PrefsSave> prefsReadResult = SaveManager.Instance.InitPrefsData();
		SentryService.SetUserContext(SaveManager.Instance.Progress.UniqueId);
		SentryService.SetPlatformBranch(PlatformUtil.GetPlatformBranch());
		_screenShake.SetMultiplier(NScreenshakePaginator.GetShakeMultiplier(SaveManager.Instance.PrefsSave.ScreenShakeOptionIndex));
		if (!OS.HasFeature("editor") && SaveManager.Instance.PrefsSave.FastMode == FastModeType.Instant)
		{
			SaveManager.Instance.PrefsSave.FastMode = FastModeType.Fast;
		}
		if (!IsReleaseGame() && CommandLineHelper.HasArg("autoslay"))
		{
			await LaunchMainMenu(skipLogo: true);
			string seed = CommandLineHelper.GetValue("seed") ?? SeedHelper.GetRandomSeed();
			string value = CommandLineHelper.GetValue("log-file");
			AutoSlayer autoSlayer = new AutoSlayer();
			autoSlayer.Start(seed, value);
		}
		else if (CommandLineHelper.HasArg("bootstrap"))
		{
			NSceneBootstrapper child = SceneHelper.Instantiate<NSceneBootstrapper>("debug/scene_bootstrapper");
			this.AddChildSafely(child);
		}
		else if (StartOnMainMenu)
		{
			bool skipLogo = DebugSettings.DevSkip || SaveManager.Instance.SettingsSave.SkipIntroLogo || CommandLineHelper.HasArg("fastmp");
			await LaunchMainMenu(skipLogo);
			CheckShowSaveFileError(progressReadResult, prefsReadResult, OneTimeInitialization.SettingsReadResult);
			CheckShowLocalizationOverrideErrors();
		}
		ModManager.OnModDetected += OnNewModDetected;
	}

	private void OnWindowChange()
	{
		Log.Info($"Window changed! New size: {DisplayServer.WindowGetSize()}");
		EmitSignal(SignalName.WindowChange, SaveManager.Instance.SettingsSave.AspectRatioSetting == AspectRatioSetting.Auto);
	}

	public static bool IsMainThread()
	{
		if (!_mainThreadId.HasValue)
		{
			_mainThreadId = System.Environment.CurrentManagedThreadId;
			return true;
		}
		return _mainThreadId == System.Environment.CurrentManagedThreadId;
	}

	public override void _ExitTree()
	{
		ModManager.OnModDetected -= OnNewModDetected;
		ModManager.Dispose();
		_joinCallbackHandler?.Dispose();
		SteamInitializer.Uninitialize();
		SentryService.Shutdown();
	}

	public static bool IsReleaseGame()
	{
		return true;
	}

	private void InitializeGraphicsPreferences()
	{
		if (!DisplayServer.GetName().Equals("headless", StringComparison.OrdinalIgnoreCase))
		{
			ApplyDisplaySettings();
			ApplySyncSetting();
		}
		Engine.MaxFps = SaveManager.Instance.SettingsSave.FpsLimit;
	}

	public void ApplyDisplaySettings()
	{
		bool flag = false;
		SettingsSave settingsSave = SaveManager.Instance.SettingsSave;
		if (settingsSave.TargetDisplay == -1)
		{
			Log.Info("First time setup for display settings...");
			settingsSave.TargetDisplay = DisplayServer.GetPrimaryScreen();
		}
		bool flag2 = settingsSave.Fullscreen;
		if (PlatformUtil.GetSupportedWindowMode().ShouldForceFullscreen())
		{
			if (!flag2)
			{
				Log.Warn($"Settings has fullscreen set to false, but we're forcing fullscreen because the platform reports our supported window mode as {PlatformUtil.GetSupportedWindowMode()}");
			}
			flag2 = true;
		}
		Log.Info($"Applying display settings...\n  FULLSCREEN: {flag2}\n  ASPECT_RATIO: ({settingsSave.AspectRatioSetting})\n  TARGET_DISPLAY: ({settingsSave.TargetDisplay})\n  WINDOW_SIZE: {settingsSave.WindowSize}\n  POSITION: {settingsSave.WindowPosition}");
		Log.Info($"[Display] Min size: {DisplayServer.WindowGetMinSize()} Max size: {DisplayServer.WindowGetMaxSize()}");
		if (settingsSave.AspectRatioSetting != AspectRatioSetting.Auto)
		{
			_window.ContentScaleAspect = Window.ContentScaleAspectEnum.Keep;
		}
		switch (settingsSave.AspectRatioSetting)
		{
		case AspectRatioSetting.Auto:
			flag = true;
			break;
		case AspectRatioSetting.FourByThree:
			_window.ContentScaleSize = new Vector2I(1680, 1260);
			break;
		case AspectRatioSetting.SixteenByTen:
			_window.ContentScaleSize = new Vector2I(1920, 1200);
			break;
		case AspectRatioSetting.SixteenByNine:
			_window.ContentScaleSize = new Vector2I(1920, 1080);
			break;
		case AspectRatioSetting.TwentyOneByNine:
			_window.ContentScaleSize = new Vector2I(2580, 1080);
			break;
		default:
			throw new ArgumentOutOfRangeException($"Invalid Aspect Ratio: {settingsSave.AspectRatioSetting}");
		}
		int num = System.Environment.GetCommandLineArgs().IndexOf("-wpos");
		if (flag2 && num < 0)
		{
			if (_window.Unresizable)
			{
				_window.Unresizable = false;
			}
			Log.Info($"[Display] Setting FULLSCREEN on Display: {settingsSave.TargetDisplay + 1} of {DisplayServer.GetScreenCount()}");
			if (settingsSave.TargetDisplay >= DisplayServer.GetScreenCount())
			{
				Log.Warn($"[Display] FAILED: Display {settingsSave.TargetDisplay} is missing. Fallback to primary.");
				DisplayServer.WindowSetCurrentScreen(DisplayServer.GetPrimaryScreen());
				settingsSave.TargetDisplay = DisplayServer.GetPrimaryScreen();
			}
			else
			{
				DisplayServer.WindowSetCurrentScreen(settingsSave.TargetDisplay);
			}
			DisplayServer.WindowSetMode(DisplayServer.WindowMode.Fullscreen);
		}
		else
		{
			Log.Info($"[Display] Attempting WINDOWED mode on Display {settingsSave.TargetDisplay + 1} of {DisplayServer.GetScreenCount()} at position {settingsSave.WindowPosition}");
			DisplayServer.WindowSetMode(DisplayServer.WindowMode.Windowed);
			if (_window.Unresizable != !settingsSave.ResizeWindows)
			{
				_window.Unresizable = !settingsSave.ResizeWindows;
			}
			if (num >= 0)
			{
				Log.Info("[Display] -wpos called. Applying special logic.");
				settingsSave.Fullscreen = false;
				Vector2I vector2I = new Vector2I(int.Parse(System.Environment.GetCommandLineArgs()[num + 1]), int.Parse(System.Environment.GetCommandLineArgs()[num + 2]));
				Vector2I vector2I2 = DisplayServer.ScreenGetPosition(DisplayServer.WindowGetCurrentScreen());
				Log.Info($"Applying window position from command line arg: {vector2I} ({string.Join(",", System.Environment.GetCommandLineArgs())} {vector2I2})");
				DisplayServer.WindowSetPosition(vector2I2 + vector2I);
				DisplayServer.WindowSetSize(settingsSave.WindowSize);
			}
			else
			{
				Vector2I vector2I3 = DisplayServer.ScreenGetSize(settingsSave.TargetDisplay);
				Vector2I windowSize = settingsSave.WindowSize;
				if (settingsSave.WindowPosition == new Vector2I(-1, -1))
				{
					Log.Info($"[Display] Going from fullscreen to windowed. Attempting to center window on screen {settingsSave.TargetDisplay}");
					settingsSave.WindowPosition = vector2I3 / 2 - windowSize / 2;
				}
				Vector2I vector2I4 = settingsSave.WindowPosition;
				if (vector2I4.X < 0 || vector2I4.Y < 0 || vector2I4.X > vector2I3.X || vector2I4.Y > vector2I3.Y)
				{
					Log.Warn("[Display] WARN: Game Window was offscreen. Resetting to top left corner.");
					vector2I4 = new Vector2I(8, 48);
				}
				if (settingsSave.TargetDisplay >= DisplayServer.GetScreenCount())
				{
					Log.Info($"[Display] FAILED: Display {settingsSave.TargetDisplay + 1} is missing. Fallback to primary.");
					settingsSave.WindowPosition = new Vector2I(8, 48);
					DisplayServer.WindowSetSize(DisplayServer.ScreenGetSize(DisplayServer.GetPrimaryScreen()) - new Vector2I(8, 48));
					DisplayServer.WindowSetPosition(DisplayServer.ScreenGetPosition(DisplayServer.GetPrimaryScreen()) + settingsSave.WindowPosition);
				}
				else
				{
					Vector2I vector2I5 = DisplayServer.ScreenGetPosition(settingsSave.TargetDisplay);
					if (windowSize.X > vector2I3.X)
					{
						windowSize.X = vector2I3.X;
					}
					if (windowSize.Y > vector2I3.Y)
					{
						windowSize.Y = vector2I3.Y;
					}
					Log.Info($"[Display] SUCCESS: {windowSize} Windowed mode in Display {settingsSave.TargetDisplay}: Position {vector2I5 + vector2I4} ({vector2I4})");
					DisplayServer.WindowSetSize(windowSize);
					DisplayServer.WindowSetPosition(vector2I5 + vector2I4);
					Log.Info($"[Display] New size: {DisplayServer.WindowGetSize()} position: {DisplayServer.WindowGetPosition()}");
				}
			}
		}
		if (flag)
		{
			Log.Info("Manual window change signal because of auto scaling");
			EmitSignal(SignalName.WindowChange, settingsSave.AspectRatioSetting == AspectRatioSetting.Auto);
		}
	}

	public NInspectRelicScreen GetInspectRelicScreen()
	{
		if (InspectRelicScreen == null)
		{
			InspectRelicScreen = NInspectRelicScreen.Create();
			_inspectionContainer.AddChildSafely(InspectRelicScreen);
		}
		return InspectRelicScreen;
	}

	public NInspectCardScreen GetInspectCardScreen()
	{
		if (InspectCardScreen == null)
		{
			InspectCardScreen = NInspectCardScreen.Create();
			_inspectionContainer.AddChildSafely(InspectCardScreen);
		}
		return InspectCardScreen;
	}

	public static void ApplySyncSetting()
	{
		switch (SaveManager.Instance.SettingsSave.VSync)
		{
		case VSyncType.Off:
			Log.Info("VSync: Off");
			DisplayServer.WindowSetVsyncMode(DisplayServer.VSyncMode.Disabled);
			break;
		case VSyncType.On:
			Log.Info("VSync: On");
			DisplayServer.WindowSetVsyncMode(DisplayServer.VSyncMode.Enabled);
			break;
		case VSyncType.Adaptive:
			Log.Info("VSync: Adaptive");
			DisplayServer.WindowSetVsyncMode(DisplayServer.VSyncMode.Adaptive);
			break;
		default:
			Log.Error($"Invalid VSync type: {SaveManager.Instance.SettingsSave.VSync}");
			break;
		}
	}

	public static void Reset()
	{
		Instance?.QueueFreeSafely();
		Instance = null;
	}

	public override void _Notification(int what)
	{
		if ((long)what == 1006)
		{
			Quit();
		}
	}

	public void Quit()
	{
		Log.Info("NGame.Quit called");
		if (!PlatformUtil.GetSupportedWindowMode().ShouldForceFullscreen() && !SaveManager.Instance.SettingsSave.Fullscreen)
		{
			SettingsSave settingsSave = SaveManager.Instance.SettingsSave;
			settingsSave.WindowSize = DisplayServer.WindowGetSize();
			settingsSave.TargetDisplay = DisplayServer.WindowGetCurrentScreen();
			settingsSave.WindowPosition = DisplayServer.WindowGetPosition() - DisplayServer.ScreenGetPosition(SaveManager.Instance.SettingsSave.TargetDisplay);
			Log.Info($"[Display] On exit, saving window size: {settingsSave.WindowSize} display: {settingsSave.TargetDisplay} position: {settingsSave.WindowPosition}");
		}
		SaveManager.Instance.SaveSettings();
		SaveManager.Instance.SavePrefsFile();
		SaveManager.Instance.SaveProgressFile();
		SaveManager.Instance.SaveProfile();
		GetTree().Quit();
	}

	private async Task LaunchMainMenu(bool skipLogo)
	{
		NLogoAnimation logoAnimation = null;
		if (skipLogo)
		{
			await PreloadManager.LoadMainMenuEssentials();
		}
		else
		{
			await PreloadManager.LoadLogoAnimation();
			logoAnimation = NLogoAnimation.Create();
			RootSceneContainer.SetCurrentScene(logoAnimation);
			await PreloadManager.LoadMainMenuEssentials();
		}
		if (logoAnimation != null)
		{
			_logoCancelToken = new CancellationTokenSource();
			await Transition.FadeIn(0.8f, "res://materials/transitions/fade_transition_mat.tres", _logoCancelToken.Token);
			await logoAnimation.PlayAnimation(_logoCancelToken.Token);
			await Transition.FadeOut();
		}
		await LoadMainMenu();
		Log.Info($"[Startup] Time to main menu: {Time.GetTicksMsec():N0}ms");
		LogResourceStats("main menu loaded (essential)");
		TaskHelper.RunSafely(LoadDeferredStartupAssetsAsync());
		_joinCallbackHandler?.CheckForCommandLineJoin();
	}

	private async Task LoadDeferredStartupAssetsAsync()
	{
		OneTimeInitialization.ExecuteDeferred();
		await PreloadManager.LoadCommonAndMainMenuAssets();
		LogResourceStats("main menu loaded (complete)");
	}

	public async Task GoToTimelineAfterRun()
	{
		await GoToTimeline();
	}

	public async Task ReturnToMainMenuAfterRun()
	{
		await ReturnToMainMenu();
	}

	public async Task GoToTimeline()
	{
		await Transition.FadeOut();
		await PreloadManager.LoadCommonAndMainMenuAssets();
		RunManager.Instance.CleanUp();
		await LoadMainMenu(openTimeline: true);
	}

	public async Task ReturnToMainMenu()
	{
		await Transition.FadeOut();
		await PreloadManager.LoadCommonAndMainMenuAssets();
		RunManager.Instance.CleanUp();
		await LoadMainMenu();
	}

	public void Relocalize()
	{
		ReloadMainMenu();
		FeedbackScreen.Relocalize();
		TimeoutOverlay.Relocalize();
	}

	public void ReloadMainMenu()
	{
		if (MainMenu == null)
		{
			throw new InvalidOperationException("Tried to reload main menu when not already on the main menu!");
		}
		TaskHelper.RunSafely(LoadMainMenu());
	}

	private async Task LoadMainMenu(bool openTimeline = false)
	{
		Task currentRunSaveTask = SaveManager.Instance.CurrentRunSaveTask;
		if (currentRunSaveTask != null)
		{
			Log.Info("Saving in progress, waiting for it to be finished before loading the main menu");
			try
			{
				await currentRunSaveTask;
			}
			catch (Exception value)
			{
				Log.Error($"Save task failed while waiting to load main menu: {value}");
			}
		}
		NMainMenu currentScene = NMainMenu.Create(openTimeline);
		RootSceneContainer.SetCurrentScene(currentScene);
	}

	public async Task<RunState> StartNewSingleplayerRun(CharacterModel character, bool shouldSave, IReadOnlyList<ActModel> acts, IReadOnlyList<ModifierModel> modifiers, string seed, int ascensionLevel = 0, DateTimeOffset? dailyTime = null)
	{
		UnlockState unlockState = SaveManager.Instance.GenerateUnlockStateFromProgress();
		RunState runState = RunState.CreateForNewRun(new global::_003C_003Ez__ReadOnlySingleElementList<Player>(Player.CreateForNewRun(character, unlockState, 1uL)), acts.Select((ActModel a) => a.ToMutable()).ToList(), modifiers, ascensionLevel, seed);
		RunManager.Instance.SetUpNewSinglePlayer(runState, shouldSave, dailyTime);
		await StartRun(runState);
		return runState;
	}

	public async Task<RunState> StartNewMultiplayerRun(StartRunLobby lobby, bool shouldSave, IReadOnlyList<ActModel> acts, IReadOnlyList<ModifierModel> modifiers, string seed, int ascensionLevel, DateTimeOffset? dailyTime = null)
	{
		RunState runState = RunState.CreateForNewRun(lobby.Players.Select((LobbyPlayer p) => Player.CreateForNewRun(p.character, UnlockState.FromSerializable(p.unlockState), p.id)).ToList(), acts.Select((ActModel a) => a.ToMutable()).ToList(), modifiers, ascensionLevel, seed);
		RunManager.Instance.SetUpNewMultiPlayer(runState, lobby, shouldSave, dailyTime);
		await StartRun(runState);
		return runState;
	}

	public async Task LoadRun(RunState runState, SerializableRoom? preFinishedRoom)
	{
		await PreloadManager.LoadRunAssets(runState.Players.Select((Player p) => p.Character));
		await PreloadManager.LoadActAssets(runState.Act);
		RunManager.Instance.Launch();
		RootSceneContainer.SetCurrentScene(NRun.Create(runState));
		await RunManager.Instance.GenerateMap();
		await RunManager.Instance.LoadIntoLatestMapCoord(AbstractRoom.FromSerializable(preFinishedRoom, runState));
		if (RunManager.Instance.MapDrawingsToLoad != null)
		{
			NRun.Instance.GlobalUi.MapScreen.Drawings.LoadDrawings(RunManager.Instance.MapDrawingsToLoad);
			RunManager.Instance.MapDrawingsToLoad = null;
		}
	}

	private async Task StartRun(RunState runState)
	{
		using (new NetLoadingHandle(RunManager.Instance.NetService))
		{
			await PreloadManager.LoadRunAssets(runState.Players.Select((Player p) => p.Character));
			await PreloadManager.LoadActAssets(runState.Acts[0]);
			await RunManager.Instance.FinalizeStartingRelics();
			RunManager.Instance.Launch();
			RootSceneContainer.SetCurrentScene(NRun.Create(runState));
			await RunManager.Instance.EnterAct(0, doTransition: false);
		}
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (inputEvent.IsActionReleased(DebugHotkey.speedUp))
		{
			DebugModifyTimescale(0.1);
		}
		else if (inputEvent.IsActionReleased(DebugHotkey.speedDown))
		{
			DebugModifyTimescale(-0.1);
		}
		else if (inputEvent.IsActionReleased(DebugHotkey.hideProceedButton))
		{
			IsDebugHidingProceedButton = !IsDebugHidingProceedButton;
			Instance.AddChildSafely(NFullscreenTextVfx.Create(IsDebugHidingProceedButton ? "Hide Proceed Button" : "Show Proceed Button"));
			this.DebugToggleProceedButton?.Invoke();
		}
		else if (inputEvent.IsActionReleased(DebugHotkey.hideHoverTips))
		{
			IsDebugHidingHoverTips = !IsDebugHidingHoverTips;
			Instance.AddChildSafely(NFullscreenTextVfx.Create(IsDebugHidingHoverTips ? "Hide HoverTips" : "Show HoverTips"));
		}
		if (inputEvent is InputEventMouseButton { Pressed: not false } || inputEvent.IsActionPressed(MegaInput.select) || inputEvent.IsActionPressed(MegaInput.cancel))
		{
			_logoCancelToken?.Cancel();
		}
		if (!(inputEvent is InputEventKey inputEventKey))
		{
			return;
		}
		if (OS.GetName().Contains("Windows"))
		{
			if (inputEventKey.Pressed && inputEventKey.AltPressed && inputEventKey.Keycode == Key.Enter)
			{
				ToggleFullscreen();
			}
		}
		else if (OS.GetName().Contains("macOS") && inputEventKey.Pressed && inputEventKey.CtrlPressed && inputEventKey.MetaPressed && inputEventKey.Keycode == Key.F)
		{
			ToggleFullscreen();
		}
	}

	private void ToggleFullscreen()
	{
		Log.Info("Used FULLSCREEN shortcut");
		NFullscreenTickbox.SetFullscreen(!SaveManager.Instance.SettingsSave.Fullscreen);
	}

	private void DebugModifyTimescale(double offset)
	{
		double value = Math.Round(Engine.TimeScale + offset, 1);
		value = Math.Clamp(value, 0.1, 4.0);
		Engine.TimeScale = value;
		this.AddChildSafely(NFullscreenTextVfx.Create($"TimeScale:{Engine.TimeScale}"));
	}

	public WorldEnvironment ActivateWorldEnvironment()
	{
		this.AddChildSafely(WorldEnvironment);
		return WorldEnvironment;
	}

	public void DeactivateWorldEnvironment()
	{
		this.RemoveChildSafely(WorldEnvironment);
	}

	public void SetScreenShakeTarget(Control target)
	{
		_screenShake.SetTarget(target);
	}

	public void ClearScreenShakeTarget()
	{
		_screenShake.ClearTarget();
	}

	public void ScreenShake(ShakeStrength strength, ShakeDuration duration, float degAngle = -1f)
	{
		if (degAngle < 0f)
		{
			degAngle = Rng.Chaotic.NextFloat(360f);
		}
		_screenShake.Shake(strength, duration, degAngle);
	}

	public void ScreenRumble(ShakeStrength strength, ShakeDuration duration, RumbleStyle style)
	{
		_screenShake.Rumble(strength, duration, style);
	}

	public void ScreenShakeTrauma(ShakeStrength strength)
	{
		_screenShake.AddTrauma(strength);
	}

	public void DoHitStop(ShakeStrength strength, ShakeDuration duration)
	{
		HitStop.DoHitStop(strength, duration);
	}

	public static void ToggleTrailerMode()
	{
		IsTrailerMode = !IsTrailerMode;
	}

	public void SetScreenshakeMultiplier(float multiplier)
	{
		_screenShake.SetMultiplier(multiplier);
	}

	private void InitPools()
	{
		NCard.InitPool();
		NGridCardHolder.InitPool();
	}

	private void OnNewModDetected(Mod mod)
	{
		if (!NModalContainer.Instance.GetChildren().OfType<NErrorPopup>().Any())
		{
			NErrorPopup modalToCreate = NErrorPopup.Create(new LocString("main_menu_ui", "MOD_NOT_LOADED_POPUP.title"), new LocString("main_menu_ui", "MOD_NOT_LOADED_POPUP.description"), null, showReportBugButton: false);
			NModalContainer.Instance.Add(modalToCreate);
		}
	}

	public void CheckShowSaveFileError(ReadSaveResult<SerializableProgress> progressReadResult, ReadSaveResult<PrefsSave> prefsReadResult, ReadSaveResult<SettingsSave>? settingsReadResult)
	{
		LocString locString = null;
		if (!progressReadResult.Success && progressReadResult.Status != ReadSaveStatus.FileNotFound)
		{
			locString = new LocString("main_menu_ui", "INVALID_SAVE_POPUP.description_progress");
		}
		else if (settingsReadResult != null && !settingsReadResult.Success && settingsReadResult.Status != ReadSaveStatus.FileNotFound)
		{
			locString = new LocString("main_menu_ui", "INVALID_SAVE_POPUP.description_settings");
		}
		else if (!prefsReadResult.Success && prefsReadResult.Status != ReadSaveStatus.FileNotFound)
		{
			locString = new LocString("main_menu_ui", "INVALID_SAVE_POPUP.description_settings");
		}
		if (locString != null)
		{
			NErrorPopup modalToCreate = NErrorPopup.Create(new LocString("main_menu_ui", "INVALID_SAVE_POPUP.title"), locString, new LocString("main_menu_ui", "INVALID_SAVE_POPUP.dismiss"), showReportBugButton: true);
			NModalContainer.Instance.Add(modalToCreate);
		}
	}

	private void CheckShowLocalizationOverrideErrors()
	{
		if (LocManager.Instance.ValidationErrors.Count != 0)
		{
			List<IGrouping<string, LocValidationError>> list = (from e in LocManager.Instance.ValidationErrors
				group e by e.FilePath).ToList();
			string text = string.Join("\n", from g in list.Take(5)
				select $"{Path.GetFileName(g.Key)} ({g.Count()} errors)");
			if (list.Count > 5)
			{
				text += $"\n... and {list.Count - 5} more files";
			}
			string body = "Errors found in the following localization override files:\n\n" + text + "\n\n[gold]Check the console logs for detailed error messages.[/gold]\n\nTo fix: Remove or correct invalid override files in your localization_override folder.";
			NErrorPopup modalToCreate = NErrorPopup.Create("Localization Override Errors", body, showReportBugButton: false);
			NModalContainer.Instance.Add(modalToCreate);
		}
	}

	private async Task<bool> InitializePlatform()
	{
		bool flag = CommandLineHelper.HasArg("force-steam");
		string text = CommandLineHelper.GetValue("force-steam") ?? "";
		if (!text.Equals("on", StringComparison.OrdinalIgnoreCase) && (!flag || !(text == string.Empty)) && (text.Equals("off", StringComparison.OrdinalIgnoreCase) || OS.HasFeature("editor")))
		{
			Log.Info("Steam initialization skipped (editor mode). Use --force-steam to enable.");
			return true;
		}
		bool steamInitialized = SteamInitializer.Initialize(this);
		if (!steamInitialized)
		{
			Log.Error("Failed to initialize Steam! Attempting to show error popup");
			await TryErrorInit();
			NGenericPopup nGenericPopup = NGenericPopup.Create();
			NModalContainer.Instance.Add(nGenericPopup);
			LocString locString = new LocString("main_menu_ui", "STEAM_INIT_ERROR.description");
			locString.Add("details", $"{SteamInitializer.InitResult}: {SteamInitializer.InitErrorMessage}");
			await nGenericPopup.WaitForConfirmation(locString, new LocString("main_menu_ui", "STEAM_INIT_ERROR.title"), null, new LocString("main_menu_ui", "QUIT"));
			GetTree().Quit();
		}
		else
		{
			_joinCallbackHandler = new SteamJoinCallbackHandler();
		}
		return steamInitialized;
	}

	public static void LogResourceStats(string context)
	{
		ulong staticMemoryUsage = OS.GetStaticMemoryUsage();
		ulong renderingInfo = RenderingServer.GetRenderingInfo(RenderingServer.RenderingInfo.VideoMemUsed);
		int value = (int)Performance.GetMonitor(Performance.Monitor.ObjectCount);
		int value2 = (int)Performance.GetMonitor(Performance.Monitor.ObjectResourceCount);
		int value3 = (int)Performance.GetMonitor(Performance.Monitor.ObjectNodeCount);
		int value4 = PreloadManager.Cache.GetCacheKeys().Count();
		Log.Info($"[Startup] Resource stats ({context}): StaticMem={FormatBytes(staticMemoryUsage)}, VRAM={FormatBytes(renderingInfo)}, Objects={value:N0}, Resources={value2:N0}, Nodes={value3:N0}, CachedAssets={value4:N0}");
	}

	private static string FormatBytes(ulong bytes)
	{
		string[] array = new string[4] { "B", "KB", "MB", "GB" };
		int num = 0;
		double num2 = bytes;
		while (num2 >= 1024.0 && num < array.Length - 1)
		{
			num2 /= 1024.0;
			num++;
		}
		return $"{num2:0.#}{array[num]}";
	}
}
