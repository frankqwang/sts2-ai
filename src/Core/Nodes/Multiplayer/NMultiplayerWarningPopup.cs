using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Multiplayer;

public partial class NMultiplayerWarningPopup : NVerticalPopup, IScreenContext
{
	public const string ftueId = "multiplayer_warning";

	private NVerticalPopup _verticalPopup;

	private static readonly string _scenePath = SceneHelper.GetScenePath("ui/multiplayer_warning_popup");

	public new static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public Control? DefaultFocusedControl => null;

	public static NMultiplayerWarningPopup? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NMultiplayerWarningPopup>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		_verticalPopup = GetNode<NVerticalPopup>("VerticalPopup");
		_verticalPopup.SetText(new LocString("main_menu_ui", "MULTIPLAYER_WARNING_POPUP.title"), new LocString("main_menu_ui", "MULTIPLAYER_WARNING_POPUP.body"));
		_verticalPopup.InitYesButton(new LocString("main_menu_ui", "MULTIPLAYER_WARNING_POPUP.back"), OnBackButtonPressed);
		_verticalPopup.InitNoButton(new LocString("main_menu_ui", "MULTIPLAYER_WARNING_POPUP.ignore"), OnIgnoreButtonPressed);
	}

	private void OnBackButtonPressed(NButton _)
	{
		this.QueueFreeSafely();
		NGame.Instance.MainMenu.SubmenuStack.Pop();
		NModalContainer.Instance.HideBackstop();
	}

	private void OnIgnoreButtonPressed(NButton _)
	{
		SaveManager.Instance.MarkFtueAsComplete("multiplayer_warning");
		this.QueueFreeSafely();
		NModalContainer.Instance.HideBackstop();
	}
}
