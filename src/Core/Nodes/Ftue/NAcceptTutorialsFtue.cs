using System;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Ftue;

public partial class NAcceptTutorialsFtue : NFtue
{
	public const string id = "accept_tutorials_ftue";

	private static readonly string _scenePath = SceneHelper.GetScenePath("ftue/accept_tutorials_ftue");

	private NCharacterSelectScreen _charSelectScreen;

	private NVerticalPopup _verticalPopup;

	private Action _onFinished;

	public override void _Ready()
	{
		_verticalPopup = GetNode<NVerticalPopup>("VerticalPopup");
		_verticalPopup.SetText(new LocString("main_menu_ui", "ENABLE_TUTORIALS.title"), new LocString("main_menu_ui", "ENABLE_TUTORIALS.description"));
		_verticalPopup.InitYesButton(new LocString("main_menu_ui", "GENERIC_POPUP.confirm"), YesTutorials);
		_verticalPopup.InitNoButton(new LocString("main_menu_ui", "GENERIC_POPUP.cancel"), NoTutorials);
	}

	public static NAcceptTutorialsFtue? Create(NCharacterSelectScreen charSelectScreen, Action onFinished)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NAcceptTutorialsFtue nAcceptTutorialsFtue = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NAcceptTutorialsFtue>(PackedScene.GenEditState.Disabled);
		nAcceptTutorialsFtue._charSelectScreen = charSelectScreen;
		nAcceptTutorialsFtue._onFinished = onFinished;
		return nAcceptTutorialsFtue;
	}

	private void NoTutorials(NButton _)
	{
		SaveManager.Instance.MarkFtueAsComplete("accept_tutorials_ftue");
		SaveManager.Instance.SetFtuesEnabled(enabled: false);
		_onFinished();
		CloseFtue();
	}

	private void YesTutorials(NButton _)
	{
		SaveManager.Instance.MarkFtueAsComplete("accept_tutorials_ftue");
		_onFinished();
		CloseFtue();
	}
}
