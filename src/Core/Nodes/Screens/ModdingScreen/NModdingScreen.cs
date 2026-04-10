using System.Linq;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens.ModdingScreen;

public partial class NModdingScreen : NSubmenu
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/modding/modding_screen");

	private NModInfoContainer _modInfoContainer;

	private Control _modRowContainer;

	private Control _pendingChangesWarning;

	protected override Control? InitialFocusedControl => null;

	public static string[] AssetPaths => new string[1] { _scenePath };

	public static NModdingScreen? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NModdingScreen>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		_modInfoContainer = GetNode<NModInfoContainer>("%ModInfoContainer");
		_modRowContainer = GetNode<Control>("%ModsScrollContainer/Mask/Content");
		_pendingChangesWarning = GetNode<Control>("%PendingChangesLabel");
		NButton node = GetNode<NButton>("%GetModsButton");
		NButton node2 = GetNode<NButton>("%MakeModsButton");
		foreach (Node child in _modRowContainer.GetChildren())
		{
			child.QueueFreeSafely();
		}
		foreach (Mod allMod in ModManager.AllMods)
		{
			OnNewModDetected(allMod);
		}
		node.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnGetModsPressed));
		node2.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnMakeModsPressed));
		node.GetNode<MegaLabel>("Visuals/Label").SetTextAutoSize(new LocString("settings_ui", "MODDING_SCREEN.GET_MODS_BUTTON").GetFormattedText());
		node2.GetNode<MegaLabel>("Visuals/Label").SetTextAutoSize(new LocString("settings_ui", "MODDING_SCREEN.MAKE_MODS_BUTTON").GetFormattedText());
		GetNode<MegaRichTextLabel>("%InstalledModsTitle").SetTextAutoSize(new LocString("settings_ui", "MODDING_SCREEN.INSTALLED_MODS_TITLE").GetFormattedText());
		GetNode<MegaRichTextLabel>("%PendingChangesLabel").SetTextAutoSize(new LocString("settings_ui", "MODDING_SCREEN.PENDING_CHANGES_WARNING").GetFormattedText());
		node2.Visible = false;
		node.Visible = false;
		_pendingChangesWarning.Visible = false;
		ModManager.OnModDetected += OnNewModDetected;
		ConnectSignals();
	}

	public override void OnSubmenuOpened()
	{
		if (!ModManager.PlayerAgreedToModLoading && ModManager.AllMods.Count > 0)
		{
			NModalContainer.Instance.Add(NConfirmModLoadingPopup.Create());
		}
	}

	private void OnGetModsPressed(NButton _)
	{
		PlatformUtil.OpenUrl("https://steamcommunity.com/app/2868840/workshop/");
	}

	private void OnMakeModsPressed(NButton _)
	{
		PlatformUtil.OpenUrl("https://gitlab.com/megacrit/sts2/example-mod/-/wikis/home");
	}

	public void OnRowSelected(NModMenuRow row)
	{
		row.SetSelected(isSelected: true);
		_modInfoContainer.Fill(row.Mod);
		foreach (NModMenuRow item in _modRowContainer.GetChildren().OfType<NModMenuRow>())
		{
			if (item != row)
			{
				item.SetSelected(isSelected: false);
			}
		}
	}

	private void OnNewModDetected(Mod mod)
	{
		NModMenuRow child = NModMenuRow.Create(this, mod);
		_modRowContainer.AddChildSafely(child);
		OnModEnabledOrDisabled();
	}

	public void OnModEnabledOrDisabled()
	{
		foreach (Mod allMod in ModManager.AllMods)
		{
			bool flag = SaveManager.Instance.SettingsSave.ModSettings?.IsModDisabled(allMod) ?? false;
			if (allMod.wasLoaded == flag)
			{
				_pendingChangesWarning.Visible = true;
				return;
			}
		}
		_pendingChangesWarning.Visible = false;
	}

	public override void _ExitTree()
	{
		ModManager.OnModDetected -= OnNewModDetected;
	}
}
