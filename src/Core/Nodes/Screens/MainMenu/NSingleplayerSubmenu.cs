using System;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;
using MegaCrit.Sts2.Core.Nodes.Screens.CustomRun;
using MegaCrit.Sts2.Core.Nodes.Screens.DailyRun;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;
using MegaCrit.Sts2.Core.Timeline.Epochs;

namespace MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;

public partial class NSingleplayerSubmenu : NSubmenu
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/singleplayer_submenu");

	private NSubmenuButton _standardButton;

	private NSubmenuButton _dailyButton;

	private NSubmenuButton _customButton;

	private const string _keyStandard = "STANDARD";

	private const string _keyDaily = "DAILY";

	private const string _keyCustom = "CUSTOM";

	protected override Control InitialFocusedControl => _standardButton;

	public static NSingleplayerSubmenu? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NSingleplayerSubmenu>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		ConnectSignals();
		_standardButton = GetNode<NSubmenuButton>("StandardButton");
		_standardButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OpenCharacterSelect));
		_standardButton.SetIconAndLocalization("STANDARD");
		_dailyButton = GetNode<NSubmenuButton>("DailyButton");
		_dailyButton.Connect(NClickableControl.SignalName.Released, Callable.From((Action<NButton>)OpenDailyScreen));
		_dailyButton.SetIconAndLocalization("DAILY");
		_customButton = GetNode<NSubmenuButton>("CustomRunButton");
		_customButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OpenCustomScreen));
		_customButton.SetIconAndLocalization("CUSTOM");
	}

	private void RefreshButtons()
	{
		_dailyButton.SetEnabled(SaveManager.Instance.IsEpochRevealed<DailyRunEpoch>());
		_customButton.SetEnabled(SaveManager.Instance.IsEpochRevealed<CustomAndSeedsEpoch>());
		_dailyButton.RefreshLabels();
		_customButton.RefreshLabels();
	}

	public override void OnSubmenuOpened()
	{
		RefreshButtons();
	}

	private void OpenCharacterSelect(NButton _)
	{
		NCharacterSelectScreen submenuType = _stack.GetSubmenuType<NCharacterSelectScreen>();
		submenuType.InitializeSingleplayer();
		_stack.Push(submenuType);
	}

	private void OpenDailyScreen(NButton _)
	{
		OpenDailyScreen();
	}

	private void OpenDailyScreen()
	{
		NDailyRunScreen submenuType = _stack.GetSubmenuType<NDailyRunScreen>();
		submenuType.InitializeSingleplayer();
		_stack.Push(submenuType);
	}

	private void OpenCustomScreen(NButton _)
	{
		NCustomRunScreen submenuType = _stack.GetSubmenuType<NCustomRunScreen>();
		submenuType.InitializeSingleplayer();
		_stack.Push(submenuType);
	}
}
