using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.CommonUi;

public partial class NAbandonRunConfirmPopup : Control, IScreenContext
{
	private NVerticalPopup _verticalPopup;

	private NMainMenu? _mainMenuNode;

	private Tween? _tween;

	private static readonly string _scenePath = SceneHelper.GetScenePath("ui/abandon_run_confirm_popup");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public Control? DefaultFocusedControl => null;

	public override void _Ready()
	{
		_verticalPopup = GetNode<NVerticalPopup>("VerticalPopup");
		_verticalPopup.SetText(new LocString("main_menu_ui", "ABANDON_RUN_CONFIRMATION.header"), new LocString("main_menu_ui", "ABANDON_RUN_CONFIRMATION.body"));
		_verticalPopup.InitYesButton(new LocString("main_menu_ui", "GENERIC_POPUP.confirm"), OnYesButtonPressed);
		_verticalPopup.InitNoButton(new LocString("main_menu_ui", "GENERIC_POPUP.cancel"), OnNoButtonPressed);
	}

	public override void _EnterTree()
	{
		base.Modulate = StsColors.transparentBlack;
		float y = base.Position.Y;
		base.Position += new Vector2(0f, 100f);
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "modulate", Colors.White, 0.1);
		_tween.TweenProperty(this, "position:y", y, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
	}

	public override void _ExitTree()
	{
		_verticalPopup.DisconnectSignals();
		_verticalPopup.DisconnectHotkeys();
	}

	public static NAbandonRunConfirmPopup? Create(NMainMenu? mainMenu)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NAbandonRunConfirmPopup nAbandonRunConfirmPopup = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NAbandonRunConfirmPopup>(PackedScene.GenEditState.Disabled);
		nAbandonRunConfirmPopup._mainMenuNode = mainMenu;
		return nAbandonRunConfirmPopup;
	}

	private void OnYesButtonPressed(NButton _)
	{
		if (_mainMenuNode == null)
		{
			RunManager.Instance.Abandon();
		}
		else
		{
			_mainMenuNode.AbandonRun();
		}
		this.QueueFreeSafely();
	}

	private void OnNoButtonPressed(NButton _)
	{
		this.QueueFreeSafely();
	}
}
