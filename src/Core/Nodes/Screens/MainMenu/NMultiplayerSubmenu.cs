using System;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Daily;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Multiplayer;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Multiplayer;
using MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;
using MegaCrit.Sts2.Core.Nodes.Screens.CustomRun;
using MegaCrit.Sts2.Core.Nodes.Screens.DailyRun;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.Platform.Steam;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;

public partial class NMultiplayerSubmenu : NSubmenu
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/multiplayer_submenu");

	private NSubmenuButton _hostButton;

	private NSubmenuButton _loadButton;

	private NSubmenuButton _abandonButton;

	private NSubmenuButton _joinButton;

	private const string _keyHost = "HOST";

	private const string _keyLoad = "MP_LOAD";

	private const string _keyJoin = "JOIN";

	private const string _keyAbandon = "MP_ABANDON";

	private Control _loadingOverlay;

	protected override Control InitialFocusedControl
	{
		get
		{
			if (SaveManager.Instance.HasMultiplayerRunSave)
			{
				return _loadButton;
			}
			return _hostButton;
		}
	}

	public static NMultiplayerSubmenu? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NMultiplayerSubmenu>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		ConnectSignals();
		_loadingOverlay = GetNode<Control>("%LoadingOverlay");
		_hostButton = GetNode<NSubmenuButton>("ButtonContainer/HostButton");
		_hostButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnHostPressed));
		_hostButton.SetIconAndLocalization("HOST");
		_loadButton = GetNode<NSubmenuButton>("ButtonContainer/LoadButton");
		_loadButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(StartLoad));
		_loadButton.SetIconAndLocalization("MP_LOAD");
		_joinButton = GetNode<NSubmenuButton>("ButtonContainer/JoinButton");
		_joinButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OpenJoinFriendsScreen));
		_joinButton.SetIconAndLocalization("JOIN");
		_abandonButton = GetNode<NSubmenuButton>("ButtonContainer/AbandonButton");
		_abandonButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(AbandonRun));
		_abandonButton.SetIconAndLocalization("MP_ABANDON");
		UpdateButtons();
	}

	private void UpdateButtons()
	{
		_hostButton.Visible = !SaveManager.Instance.HasMultiplayerRunSave;
		_loadButton.Visible = SaveManager.Instance.HasMultiplayerRunSave;
		_abandonButton.Visible = SaveManager.Instance.HasMultiplayerRunSave;
	}

	private void AbandonRun(NButton _)
	{
		TaskHelper.RunSafely(TryAbandonMultiplayerRun());
	}

	private async Task TryAbandonMultiplayerRun()
	{
		LocString header = new LocString("main_menu_ui", "ABANDON_RUN_CONFIRMATION.header");
		LocString body = new LocString("main_menu_ui", "ABANDON_RUN_CONFIRMATION.body");
		LocString yesButton = new LocString("main_menu_ui", "GENERIC_POPUP.confirm");
		LocString noButton = new LocString("main_menu_ui", "GENERIC_POPUP.cancel");
		NGenericPopup nGenericPopup = NGenericPopup.Create();
		NModalContainer.Instance.Add(nGenericPopup);
		if (!(await nGenericPopup.WaitForConfirmation(body, header, noButton, yesButton)))
		{
			return;
		}
		ReadSaveResult<SerializableRun> readSaveResult = SaveManager.Instance.LoadAndCanonicalizeMultiplayerRunSave(PlatformUtil.GetLocalPlayerId(PlatformUtil.PrimaryPlatform));
		if (readSaveResult.Success && readSaveResult.SaveData != null)
		{
			try
			{
				SerializableRun saveData = readSaveResult.SaveData;
				SaveManager.Instance.UpdateProgressWithRunData(saveData, victory: false);
				RunHistoryUtilities.CreateRunHistoryEntry(saveData, victory: false, isAbandoned: true, saveData.PlatformType);
				if (saveData.DailyTime.HasValue)
				{
					PlatformUtil.GetLocalPlayerId(saveData.PlatformType);
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
			Log.Error($"ERROR: Failed to load multiplayer run save: status={readSaveResult.Status}. Deleting current run...");
		}
		SaveManager.Instance.DeleteCurrentMultiplayerRun();
		UpdateButtons();
	}

	private void StartLoad(NButton _)
	{
		PlatformType platformType = ((SteamInitializer.Initialized && !CommandLineHelper.HasArg("fastmp")) ? PlatformType.Steam : PlatformType.None);
		ReadSaveResult<SerializableRun> readSaveResult = SaveManager.Instance.LoadAndCanonicalizeMultiplayerRunSave(PlatformUtil.GetLocalPlayerId(platformType));
		if (!readSaveResult.Success || readSaveResult.SaveData == null)
		{
			Log.Warn("Broken multiplayer run save detected, disabling button");
			_loadButton.Disable();
			NErrorPopup modalToCreate = NErrorPopup.Create(new LocString("main_menu_ui", "INVALID_SAVE_POPUP.title"), new LocString("main_menu_ui", "INVALID_SAVE_POPUP.description_run"), new LocString("main_menu_ui", "INVALID_SAVE_POPUP.dismiss"), showReportBugButton: true);
			NModalContainer.Instance.Add(modalToCreate);
			NModalContainer.Instance.ShowBackstop();
		}
		else
		{
			StartHost(readSaveResult.SaveData);
		}
	}

	private void OnHostPressed(NButton _)
	{
		if (SaveManager.Instance.Progress.NumberOfRuns > 0)
		{
			_stack.PushSubmenuType<NMultiplayerHostSubmenu>();
		}
		else
		{
			TaskHelper.RunSafely(NMultiplayerHostSubmenu.StartHostAsync(GameMode.Standard, _loadingOverlay, _stack));
		}
	}

	public void FastHost(GameMode gameMode)
	{
		NMultiplayerHostSubmenu nMultiplayerHostSubmenu = _stack.PushSubmenuType<NMultiplayerHostSubmenu>();
		nMultiplayerHostSubmenu.StartHost(gameMode);
	}

	public void StartHost(SerializableRun run)
	{
		TaskHelper.RunSafely(StartHostAsync(run));
	}

	private async Task StartHostAsync(SerializableRun run)
	{
		PlatformType platformType = ((SteamInitializer.Initialized && !CommandLineHelper.HasArg("fastmp")) ? PlatformType.Steam : PlatformType.None);
		_loadingOverlay.Visible = true;
		try
		{
			NetHostGameService netService = new NetHostGameService();
			NetErrorInfo? netErrorInfo = null;
			if (platformType == PlatformType.Steam)
			{
				netErrorInfo = await netService.StartSteamHost(4);
			}
			else
			{
				netService.StartENetHost(33771, 4);
			}
			if (!netErrorInfo.HasValue)
			{
				if (run.Modifiers.Count > 0)
				{
					if (run.DailyTime.HasValue)
					{
						NDailyRunLoadScreen submenuType = _stack.GetSubmenuType<NDailyRunLoadScreen>();
						submenuType.InitializeAsHost(netService, run);
						_stack.Push(submenuType);
					}
					else
					{
						NCustomRunLoadScreen submenuType2 = _stack.GetSubmenuType<NCustomRunLoadScreen>();
						submenuType2.InitializeAsHost(netService, run);
						_stack.Push(submenuType2);
					}
				}
				else
				{
					NMultiplayerLoadGameScreen submenuType3 = _stack.GetSubmenuType<NMultiplayerLoadGameScreen>();
					submenuType3.InitializeAsHost(netService, run);
					_stack.Push(submenuType3);
				}
			}
			else
			{
				NErrorPopup nErrorPopup = NErrorPopup.Create(netErrorInfo.Value);
				if (nErrorPopup != null)
				{
					NModalContainer.Instance.Add(nErrorPopup);
				}
			}
		}
		finally
		{
			_loadingOverlay.Visible = false;
		}
	}

	private void OpenJoinFriendsScreen(NButton _)
	{
		OnJoinFriendsPressed();
	}

	public NJoinFriendScreen OnJoinFriendsPressed()
	{
		return _stack.PushSubmenuType<NJoinFriendScreen>();
	}

	protected override void OnSubmenuShown()
	{
		base.OnSubmenuShown();
		if (!SaveManager.Instance.SeenFtue("multiplayer_warning") && SaveManager.Instance.Progress.NumberOfRuns == 0 && !CommandLineHelper.HasArg("fastmp"))
		{
			NMultiplayerWarningPopup modalToCreate = NMultiplayerWarningPopup.Create();
			NModalContainer.Instance.Add(modalToCreate);
		}
	}
}
