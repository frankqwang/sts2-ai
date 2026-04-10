using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens.ModdingScreen;

public partial class NConfirmModLoadingPopup : Control, IScreenContext
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("ui/confirm_mod_loading_popup");

	private NVerticalPopup _verticalPopup;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public Control DefaultFocusedControl => _verticalPopup.NoButton;

	public static NConfirmModLoadingPopup? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NConfirmModLoadingPopup>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		_verticalPopup = GetNode<NVerticalPopup>("VerticalPopup");
		_verticalPopup.SetText(new LocString("main_menu_ui", "MODDING_POPUP.title"), new LocString("main_menu_ui", "MODDING_POPUP.description"));
		_verticalPopup.InitYesButton(new LocString("main_menu_ui", "MODDING_POPUP.load_mods"), OnYesButtonPressed);
		_verticalPopup.InitNoButton(new LocString("main_menu_ui", "MODDING_POPUP.cancel"), OnNoButtonPressed);
	}

	private void OnYesButtonPressed(NButton _)
	{
		Log.Info("Player chose to load mods");
		SettingsSave settingsSave = SaveManager.Instance.SettingsSave;
		if (settingsSave.ModSettings == null)
		{
			ModSettings modSettings = (settingsSave.ModSettings = new ModSettings());
		}
		SaveManager.Instance.SettingsSave.ModSettings.PlayerAgreedToModLoading = true;
		SaveManager.Instance.SaveSettings();
		NGame.Instance.Quit();
		this.QueueFreeSafely();
	}

	private void OnNoButtonPressed(NButton _)
	{
		Log.Info("Player chose not to load mods");
		SettingsSave settingsSave = SaveManager.Instance.SettingsSave;
		if (settingsSave.ModSettings == null)
		{
			ModSettings modSettings = (settingsSave.ModSettings = new ModSettings());
		}
		SaveManager.Instance.SettingsSave.ModSettings.PlayerAgreedToModLoading = false;
		SaveManager.Instance.SaveSettings();
		if (NGame.Instance?.MainMenu?.SubmenuStack.Peek() is NModdingScreen)
		{
			NGame.Instance.MainMenu.SubmenuStack.Pop();
		}
		this.QueueFreeSafely();
	}
}
