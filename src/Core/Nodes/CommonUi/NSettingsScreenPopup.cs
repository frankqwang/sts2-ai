using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.CommonUi;

public partial class NSettingsScreenPopup : Control, IScreenContext
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("ui/settings_screen_popup");

	private ColorRect _backstop;

	private LocString _header;

	private LocString _description;

	public NVerticalPopup VerticalPopup { get; private set; }

	public Control? DefaultFocusedControl => null;

	public static NSettingsScreenPopup? Create(LocString header, LocString description)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NSettingsScreenPopup nSettingsScreenPopup = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NSettingsScreenPopup>(PackedScene.GenEditState.Disabled);
		nSettingsScreenPopup._header = header;
		nSettingsScreenPopup._description = description;
		return nSettingsScreenPopup;
	}

	public override void _Ready()
	{
		VerticalPopup = GetNode<NVerticalPopup>("VerticalPopup");
		VerticalPopup.InitNoButton(new LocString("main_menu_ui", "GENERIC_POPUP.cancel"), delegate
		{
		});
		VerticalPopup.InitYesButton(new LocString("main_menu_ui", "GENERIC_POPUP.confirm"), OnYesButtonPressed);
		VerticalPopup.SetText(_header, _description);
	}

	private void OnYesButtonPressed(NButton _)
	{
		Log.Info("FTUEs have been reset!");
		SaveManager.Instance.ResetFtues();
	}
}
