using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.CommonUi;

public partial class NDisconnectConfirmPopup : Control, IScreenContext
{
	private MegaLabel _header;

	private MegaRichTextLabel _description;

	private NButton _noButton;

	private NButton _yesButton;

	private NMainMenu? _mainMenuNode;

	private static readonly string _scenePath = SceneHelper.GetScenePath("ui/disconnect_confirm_popup");

	private NVerticalPopup _verticalPopup;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public Control? DefaultFocusedControl => null;

	public override void _Ready()
	{
		_verticalPopup = GetNode<NVerticalPopup>("VerticalPopup");
		_verticalPopup.SetText(new LocString("settings_ui", "DISCONNECT_CONFIRMATION.header"), new LocString("settings_ui", "DISCONNECT_CONFIRMATION.body"));
		_verticalPopup.InitNoButton(new LocString("main_menu_ui", "GENERIC_POPUP.cancel"), OnNoButtonPressed);
		_verticalPopup.InitYesButton(new LocString("main_menu_ui", "GENERIC_POPUP.confirm"), OnYesButtonPressed);
	}

	public static NDisconnectConfirmPopup? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NDisconnectConfirmPopup>(PackedScene.GenEditState.Disabled);
	}

	private void OnYesButtonPressed(NButton _)
	{
		RunManager.Instance.NetService.Disconnect(NetError.Quit);
		this.QueueFreeSafely();
	}

	private void OnNoButtonPressed(NButton _)
	{
		this.QueueFreeSafely();
	}
}
