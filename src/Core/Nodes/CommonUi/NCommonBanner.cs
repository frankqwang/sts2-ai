using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.Nodes.CommonUi;

public partial class NCommonBanner : Control
{
	public MegaLabel label;

	private Tween? _tween;

	private Vector2 _showPos;

	private Vector2 _hidePos;

	private static readonly Vector2 _hideOffset = new Vector2(0f, 50f);

	private Vector2 _imgOffset;

	public override void _Ready()
	{
		label = GetNode<MegaLabel>("MegaLabel");
		base.Modulate = StsColors.transparentWhite;
		_imgOffset = new Vector2(GetViewportRect().Size.X * 0.5f, GetViewportRect().Size.Y * 0.5f) - base.GlobalPosition;
		GetTree().Root.Connect(Viewport.SignalName.SizeChanged, Callable.From(OnWindowChange));
		OnWindowChange();
	}

	private void OnWindowChange()
	{
		_showPos = new Vector2(GetViewportRect().Size.X * 0.5f, GetViewportRect().Size.Y * 0.5f) - _imgOffset;
		_hidePos = _showPos + _hideOffset;
		base.Position = _showPos;
	}

	public void AnimateIn()
	{
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "modulate:a", 1f, 0.4).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(this, "global_position", _showPos, 0.4).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
			.From(_hidePos);
	}

	public void AnimateOut()
	{
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "modulate:a", 0f, 0.4).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}
}
