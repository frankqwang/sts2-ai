using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens.ProfileScreen;

public partial class NProfileScreen : NSubmenu
{
	public static int? forceShowProfileAsDeleted;

	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/profiles/profile_screen");

	private Control _loadingOverlay;

	private readonly List<NProfileButton> _profileButtons = new List<NProfileButton>();

	private readonly List<NDeleteProfileButton> _deleteButtons = new List<NDeleteProfileButton>();

	public static string[] AssetPaths
	{
		get
		{
			List<string> list = new List<string>();
			list.Add(_scenePath);
			list.AddRange(NProfileButton.AssetPaths);
			return list.ToArray();
		}
	}

	protected override Control InitialFocusedControl => _profileButtons[SaveManager.Instance.CurrentProfileId - 1];

	public static NProfileScreen? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NProfileScreen>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		ConnectSignals();
		_loadingOverlay = GetNode<Control>("%LoadingOverlay");
		GetNode<MegaLabel>("%ChooseProfileMessage").SetTextAutoSize(new LocString("main_menu_ui", "PROFILE_SCREEN.BUTTON.chooseProfileMessage").GetFormattedText());
		_profileButtons.Add(GetNode<NProfileButton>("%ProfileButton1"));
		_profileButtons.Add(GetNode<NProfileButton>("%ProfileButton2"));
		_profileButtons.Add(GetNode<NProfileButton>("%ProfileButton3"));
		_deleteButtons.Add(GetNode<NDeleteProfileButton>("%DeleteProfileButton1"));
		_deleteButtons.Add(GetNode<NDeleteProfileButton>("%DeleteProfileButton2"));
		_deleteButtons.Add(GetNode<NDeleteProfileButton>("%DeleteProfileButton3"));
		if (_profileButtons.Count != 3)
		{
			Log.Error($"There are {_profileButtons.Count} profile buttons, but max profile count in ProfileSaveManager is {3}! This might result in subtle bugs");
		}
		for (int i = 0; i < _profileButtons.Count; i++)
		{
			_profileButtons[i].FocusNeighborTop = _profileButtons[i].GetPath();
			_profileButtons[i].FocusNeighborBottom = _deleteButtons[i].GetPath();
			NProfileButton nProfileButton = _profileButtons[i];
			NodePath path;
			if (i <= 0)
			{
				List<NProfileButton> profileButtons = _profileButtons;
				path = profileButtons[profileButtons.Count - 1].GetPath();
			}
			else
			{
				path = _profileButtons[i - 1].GetPath();
			}
			nProfileButton.FocusNeighborLeft = path;
			_profileButtons[i].FocusNeighborRight = ((i < _profileButtons.Count - 1) ? _profileButtons[i + 1].GetPath() : _profileButtons[0].GetPath());
			_deleteButtons[i].FocusNeighborTop = _profileButtons[i].GetPath();
			_deleteButtons[i].FocusNeighborBottom = _deleteButtons[i].GetPath();
			NDeleteProfileButton nDeleteProfileButton = _deleteButtons[i];
			NodePath path2;
			if (i <= 0)
			{
				List<NDeleteProfileButton> deleteButtons = _deleteButtons;
				path2 = deleteButtons[deleteButtons.Count - 1].GetPath();
			}
			else
			{
				path2 = _deleteButtons[i - 1].GetPath();
			}
			nDeleteProfileButton.FocusNeighborLeft = path2;
			_deleteButtons[i].FocusNeighborRight = ((i < _deleteButtons.Count - 1) ? _deleteButtons[i + 1].GetPath() : _deleteButtons[0].GetPath());
		}
	}

	public override void OnSubmenuOpened()
	{
		Refresh();
	}

	public void ShowLoading()
	{
		_loadingOverlay.Visible = true;
	}

	public void Refresh()
	{
		for (int i = 0; i < _profileButtons.Count; i++)
		{
			_profileButtons[i].Initialize(this, i + 1);
		}
		for (int j = 0; j < _deleteButtons.Count; j++)
		{
			_deleteButtons[j].Initialize(this, j + 1);
		}
		forceShowProfileAsDeleted = null;
		ActiveScreenContext.Instance.Update();
	}
}
