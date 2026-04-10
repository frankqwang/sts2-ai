using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Multiplayer;

public partial class NGenericPopup : Control, IScreenContext
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("ui/generic_popup");

	private NVerticalPopup _verticalPopup;

	private TaskCompletionSource<bool> _confirmationCompletionSource;

	private ulong _steamId;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public Control? DefaultFocusedControl => null;

	public static NGenericPopup? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NGenericPopup>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		_verticalPopup = GetNode<NVerticalPopup>("VerticalPopup");
		_confirmationCompletionSource = new TaskCompletionSource<bool>();
	}

	public Task<bool> WaitForConfirmation(LocString body, LocString header, LocString? noButton, LocString yesButton)
	{
		_verticalPopup.SetText(header, body);
		_verticalPopup.InitYesButton(yesButton, OnYesButtonPressed);
		if (noButton != null)
		{
			_verticalPopup.InitNoButton(noButton, OnNoButtonPressed);
		}
		else
		{
			_verticalPopup.HideNoButton();
		}
		return _confirmationCompletionSource.Task;
	}

	private void OnYesButtonPressed(NButton _)
	{
		_confirmationCompletionSource.SetResult(result: true);
		this.QueueFreeSafely();
	}

	private void OnNoButtonPressed(NButton _)
	{
		_confirmationCompletionSource.SetResult(result: false);
		this.QueueFreeSafely();
	}
}
