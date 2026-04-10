using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.ProfileScreen;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;

public partial class NOpenProfileScreenButton : NButton
{
	private readonly LocString _titleLoc = new LocString("main_menu_ui", "OPEN_PROFILE_SCREEN.title");

	private static readonly LocString _descriptionLoc = new LocString("main_menu_ui", "OPEN_PROFILE_SCREEN.description");

	private NProfileIcon _profileIcon;

	private MegaLabel _title;

	private MegaLabel _description;

	private Tween? _tween;

	protected override string[] Hotkeys => new string[1] { MegaInput.pauseAndBack };

	public override void _Ready()
	{
		ConnectSignals();
		_profileIcon = GetNode<NProfileIcon>("ProfileIcon");
		_title = GetNode<MegaLabel>("Title");
		_description = GetNode<MegaLabel>("Description");
		RefreshLabels();
		_profileIcon.SetProfileId(SaveManager.Instance.CurrentProfileId);
		UpdateDescription();
	}

	public override void _EnterTree()
	{
		base._EnterTree();
		if (NControllerManager.Instance != null)
		{
			NControllerManager.Instance.Connect(NControllerManager.SignalName.MouseDetected, Callable.From(UpdateDescription));
			NControllerManager.Instance.Connect(NControllerManager.SignalName.ControllerDetected, Callable.From(UpdateDescription));
		}
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		if (NControllerManager.Instance != null)
		{
			NControllerManager.Instance.Disconnect(NControllerManager.SignalName.ControllerDetected, Callable.From(UpdateDescription));
			NControllerManager.Instance.Disconnect(NControllerManager.SignalName.MouseDetected, Callable.From(UpdateDescription));
		}
	}

	public override void _Notification(int what)
	{
		if ((long)what == 2010 && IsNodeReady())
		{
			RefreshLabels();
		}
	}

	private void RefreshLabels()
	{
		_titleLoc.Add("Id", SaveManager.Instance.CurrentProfileId);
		_title.SetTextAutoSize(_titleLoc.GetFormattedText());
		_description.SetTextAutoSize(_descriptionLoc.GetFormattedText());
	}

	protected override void OnRelease()
	{
		NGame.Instance.MainMenu.OpenProfileScreen();
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_tween?.Kill();
		base.Scale = Vector2.One * 1.02f;
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_tween?.Kill();
		_tween = CreateTween();
		_tween.TweenProperty(this, "scale", Vector2.One * 1f, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	private void UpdateDescription()
	{
		if (NControllerManager.Instance != null)
		{
			_description.SetVisible(!NControllerManager.Instance.IsUsingController);
		}
	}
}
