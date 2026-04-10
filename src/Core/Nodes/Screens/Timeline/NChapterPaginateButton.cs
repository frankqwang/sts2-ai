using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Timeline;

public partial class NChapterPaginateButton : NButton
{
	private Tween? _tween;

	private TextureRect _icon;

	private bool _isFacingRight;

	protected override string ClickedSfx => "event:/sfx/ui/timeline/ui_timeline_click";

	public override void _Ready()
	{
		ConnectSignals();
		_icon = GetNode<TextureRect>("Arrow");
		SetPivotOffset(base.Size * 0.5f);
	}

	public void SetDirection(bool isFacingRight)
	{
		_isFacingRight = isFacingRight;
		_icon.FlipH = !isFacingRight;
	}

	protected override void OnEnable()
	{
		base.OnEnable();
		base.Modulate = Colors.White;
		base.MouseFilter = MouseFilterEnum.Stop;
	}

	protected override void OnDisable()
	{
		base.OnDisable();
		base.Modulate = StsColors.transparentWhite;
		base.MouseFilter = MouseFilterEnum.Ignore;
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "scale", Vector2.One * 1.1f, 0.05);
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	protected override void OnPress()
	{
		base.OnPress();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "scale", Vector2.One * 0.95f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	protected override void OnRelease()
	{
		base.OnRelease();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "scale", Vector2.One, 0.05);
		Disable();
		base.MouseFilter = MouseFilterEnum.Ignore;
	}
}
