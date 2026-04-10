using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.FeedbackScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.CommonUi;

public partial class NErrorPopup : NVerticalPopup, IScreenContext
{
	private NVerticalPopup _verticalPopup;

	private string _title;

	private string _body;

	private LocString? _cancel;

	private bool _showReportBugButton;

	private static readonly string _scenePath = SceneHelper.GetScenePath("ui/error_popup");

	public Control? DefaultFocusedControl => null;

	public new static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public override void _Ready()
	{
		_verticalPopup = GetNode<NVerticalPopup>("VerticalPopup");
		_verticalPopup.SetText(_title, _body);
		if (_showReportBugButton)
		{
			_verticalPopup.InitYesButton(new LocString("main_menu_ui", "NETWORK_ERROR.report_bug"), OnReportBugButtonPressed);
		}
		else
		{
			_verticalPopup.InitYesButton(new LocString("main_menu_ui", "GENERIC_POPUP.ok"), OnOkButtonPressed);
		}
		if (_cancel != null)
		{
			_verticalPopup.InitNoButton(_cancel, OnCancelButtonPressed);
		}
		else
		{
			_verticalPopup.HideNoButton();
		}
	}

	public static NErrorPopup? Create(NetErrorInfo info)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		if (info.SelfInitiated && info.GetReason() == NetError.Quit)
		{
			return null;
		}
		bool showReportBugButton;
		return Create(new LocString("main_menu_ui", "NETWORK_ERROR.header"), LocStringFromNetError(info, out showReportBugButton), null, showReportBugButton);
	}

	public static NErrorPopup? Create(LocString title, LocString body, LocString? cancel, bool showReportBugButton)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NErrorPopup nErrorPopup = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NErrorPopup>(PackedScene.GenEditState.Disabled);
		nErrorPopup._title = title.GetFormattedText();
		nErrorPopup._body = body.GetFormattedText();
		nErrorPopup._showReportBugButton = showReportBugButton;
		nErrorPopup._cancel = cancel;
		return nErrorPopup;
	}

	public static NErrorPopup? Create(string title, string body, bool showReportBugButton)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NErrorPopup nErrorPopup = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NErrorPopup>(PackedScene.GenEditState.Disabled);
		nErrorPopup._title = title;
		nErrorPopup._body = body;
		nErrorPopup._showReportBugButton = showReportBugButton;
		return nErrorPopup;
	}

	private static LocString LocStringFromNetError(NetErrorInfo info, out bool showReportBugButton)
	{
		NetError reason = info.GetReason();
		string text = default(string);
		switch (reason)
		{
		case NetError.None:
			text = null;
			break;
		case NetError.QuitGameOver:
			text = null;
			break;
		case NetError.CancelledJoin:
			text = null;
			break;
		case NetError.LobbyFull:
			text = "NETWORK_ERROR.LOBBY_FULL.body";
			break;
		case NetError.Quit:
			text = "NETWORK_ERROR.QUIT.body";
			break;
		case NetError.HostAbandoned:
			text = "NETWORK_ERROR.HOST_ABANDONED.body";
			break;
		case NetError.Kicked:
			text = "NETWORK_ERROR.KICKED.body";
			break;
		case NetError.InvalidJoin:
			text = "NETWORK_ERROR.INVALID_JOIN.body";
			break;
		case NetError.RunInProgress:
			text = "NETWORK_ERROR.RUN_IN_PROGRESS.body";
			break;
		case NetError.StateDivergence:
			text = "NETWORK_ERROR.STATE_DIVERGENCE.body";
			break;
		case NetError.ModMismatch:
			text = "NETWORK_ERROR.MOD_MISMATCH.body";
			break;
		case NetError.JoinBlockedByUser:
			text = "NETWORK_ERROR.JOIN_BLOCKED_BY_USER.body";
			break;
		case NetError.NoInternet:
			text = "NETWORK_ERROR.NO_INTERNET.body";
			break;
		case NetError.Timeout:
			text = "NETWORK_ERROR.TIMEOUT.body";
			break;
		case NetError.HandshakeTimeout:
			text = "NETWORK_ERROR.TIMEOUT.body";
			break;
		case NetError.InternalError:
			text = "NETWORK_ERROR.INTERNAL_ERROR.body";
			break;
		case NetError.UnknownNetworkError:
			text = "NETWORK_ERROR.UNKNOWN_ERROR.body";
			break;
		case NetError.TryAgainLater:
			text = "NETWORK_ERROR.TRY_AGAIN_LATER.body";
			break;
		case NetError.FailedToHost:
			text = "NETWORK_ERROR.FAILED_TO_HOST.body";
			break;
		case NetError.NotInSaveGame:
			text = "NETWORK_ERROR.NOT_IN_SAVE_GAME.body";
			break;
		case NetError.VersionMismatch:
			text = "NETWORK_ERROR.VERSION_MISMATCH.body";
			break;
		default:
			throw new System.Runtime.CompilerServices.SwitchExpressionException(reason);
			break;
		}
		string text2 = text;
		bool flag = ((reason == NetError.None || reason == NetError.StateDivergence || (uint)(reason - 17) <= 1u) ? true : false);
		showReportBugButton = flag;
		if (text2 == null)
		{
			Log.Error($"Invalid net error passed to NNetworkErrorPopup: {info}!");
			text2 = "NETWORK_ERROR.INTERNAL_ERROR.body";
			showReportBugButton = true;
		}
		LocString locString = new LocString("main_menu_ui", text2);
		locString.Add("info", info.GetErrorString());
		return locString;
	}

	private void OnOkButtonPressed(NButton _)
	{
		this.QueueFreeSafely();
	}

	private void OnCancelButtonPressed(NButton _)
	{
		this.QueueFreeSafely();
	}

	private void OnReportBugButtonPressed(NButton _)
	{
		TaskHelper.RunSafely(OpenFeedbackScreen());
	}

	private async Task OpenFeedbackScreen()
	{
		SceneTree sceneTree = GetTree();
		this.QueueFreeSafely();
		await sceneTree.ToSignal(sceneTree, SceneTree.SignalName.ProcessFrame);
		await sceneTree.ToSignal(sceneTree, SceneTree.SignalName.ProcessFrame);
		await NFeedbackScreenOpener.Instance.OpenFeedbackScreen();
	}
}
