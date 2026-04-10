using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Multiplayer;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;
using MegaCrit.Sts2.Core.Nodes.Screens.CustomRun;
using MegaCrit.Sts2.Core.Nodes.Screens.DailyRun;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.Platform.Steam;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;
using MegaCrit.Sts2.Core.Timeline.Epochs;

namespace MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;

public partial class NMultiplayerHostSubmenu : NSubmenu
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/multiplayer_host_submenu");

	private NSubmenuButton _standardButton;

	private NSubmenuButton _dailyButton;

	private NSubmenuButton _customButton;

	private const string _keyStandard = "STANDARD_MP";

	private const string _keyDaily = "DAILY_MP";

	private const string _keyCustom = "CUSTOM_MP";

	private Control _loadingOverlay;

	protected override Control InitialFocusedControl => _standardButton;

	public static NMultiplayerHostSubmenu? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NMultiplayerHostSubmenu>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		ConnectSignals();
		_loadingOverlay = GetNode<Control>("%LoadingOverlay");
		_standardButton = GetNode<NSubmenuButton>("StandardButton");
		_standardButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnStandardPressed));
		_standardButton.SetIconAndLocalization("STANDARD_MP");
		_dailyButton = GetNode<NSubmenuButton>("DailyButton");
		_dailyButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnDailyPressed));
		_dailyButton.SetIconAndLocalization("DAILY_MP");
		_customButton = GetNode<NSubmenuButton>("CustomRunButton");
		_customButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnCustomPressed));
		_customButton.SetIconAndLocalization("CUSTOM_MP");
	}

	private void RefreshButtons()
	{
		_dailyButton.SetEnabled(SaveManager.Instance.IsEpochRevealed<DailyRunEpoch>());
		_customButton.SetEnabled(SaveManager.Instance.IsEpochRevealed<CustomAndSeedsEpoch>());
	}

	public override void OnSubmenuOpened()
	{
		RefreshButtons();
	}

	private void OnStandardPressed(NButton _)
	{
		StartHost(GameMode.Standard);
	}

	private void OnDailyPressed(NButton _)
	{
		StartHost(GameMode.Daily);
	}

	private void OnCustomPressed(NButton _)
	{
		StartHost(GameMode.Custom);
	}

	public void StartHost(GameMode gameMode)
	{
		TaskHelper.RunSafely(StartHostAsync(gameMode, _loadingOverlay, _stack));
	}

	public static async Task StartHostAsync(GameMode gameMode, Control loadingOverlay, NSubmenuStack stack)
	{
		PlatformType platformType = ((SteamInitializer.Initialized && !CommandLineHelper.HasArg("fastmp")) ? PlatformType.Steam : PlatformType.None);
		loadingOverlay.Visible = true;
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
				switch (gameMode)
				{
				case GameMode.Standard:
				{
					NCharacterSelectScreen submenuType3 = stack.GetSubmenuType<NCharacterSelectScreen>();
					submenuType3.InitializeMultiplayerAsHost(netService, 4);
					stack.Push(submenuType3);
					break;
				}
				case GameMode.Daily:
				{
					NDailyRunScreen submenuType2 = stack.GetSubmenuType<NDailyRunScreen>();
					submenuType2.InitializeMultiplayerAsHost(netService);
					stack.Push(submenuType2);
					break;
				}
				default:
				{
					NCustomRunScreen submenuType = stack.GetSubmenuType<NCustomRunScreen>();
					submenuType.InitializeMultiplayerAsHost(netService, 4);
					stack.Push(submenuType);
					break;
				}
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
		catch
		{
			NErrorPopup nErrorPopup2 = NErrorPopup.Create(new NetErrorInfo(NetError.InternalError, selfInitiated: false));
			if (nErrorPopup2 != null)
			{
				NModalContainer.Instance.Add(nErrorPopup2);
			}
			throw;
		}
		finally
		{
			loadingOverlay.Visible = false;
		}
	}
}
