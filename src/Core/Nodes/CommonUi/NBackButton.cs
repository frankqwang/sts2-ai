using Godot;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.CommonUi;

public partial class NBackButton : NButton
{
	private Control _outline;

	private Control _buttonImage;

	private Color _defaultOutlineColor = StsColors.cream;

	private Color _hoveredOutlineColor = StsColors.gold;

	private Color _downColor = Colors.Gray;

	private Color _outlineColor = new Color("F0B400");

	private Color _outlineTransparentColor = new Color("FF000000");

	private static readonly Vector2 _hoverScale = Vector2.One * 1.05f;

	private static readonly Vector2 _downScale = Vector2.One;

	private const double _animInOutDur = 0.35;

	private Vector2 _posOffset;

	private Vector2 _showPos;

	private Vector2 _hidePos;

	private static readonly Vector2 _hideOffset = new Vector2(-180f, 0f);

	private Tween? _hoverTween;

	private Tween? _moveTween;

	protected override string ClickedSfx => "event:/sfx/ui/clicks/ui_back";

	protected override string[] Hotkeys => new string[3]
	{
		MegaInput.cancel,
		MegaInput.pauseAndBack,
		MegaInput.back
	};

	protected override string ControllerIconHotkey => MegaInput.cancel;

	public override void _Ready()
	{
		ConnectSignals();
		_outline = GetNode<Control>("Outline");
		_buttonImage = GetNode<Control>("Image");
		_isEnabled = false;
		_posOffset = new Vector2(base.OffsetLeft + 80f, 0f - base.OffsetBottom + 110f);
		GetTree().Root.Connect(Viewport.SignalName.SizeChanged, Callable.From(OnWindowChange));
		OnWindowChange();
		OnDisable();
	}

	private void OnWindowChange()
	{
		_showPos = new Vector2(0f, GetWindow().ContentScaleSize.Y) - _posOffset;
		_hidePos = _showPos + _hideOffset;
		base.Position = (_isEnabled ? _showPos : _hidePos);
	}

	public void MoveToHidePosition()
	{
		base.GlobalPosition = _hidePos;
	}

	protected override void OnEnable()
	{
		base.OnEnable();
		_isEnabled = true;
		_outline.Modulate = Colors.Transparent;
		_buttonImage.Modulate = Colors.White;
		base.Scale = Vector2.One;
		_moveTween?.Kill();
		_moveTween = CreateTween();
		_moveTween.TweenProperty(this, "global_position", _showPos, 0.35).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
	}

	protected override void OnDisable()
	{
		base.OnDisable();
		_isEnabled = false;
		_moveTween?.Kill();
		_moveTween = CreateTween();
		_moveTween.TweenProperty(this, "global_position", _hidePos, 0.35).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_hoverTween?.Kill();
		_hoverTween = CreateTween().SetParallel();
		_hoverTween.TweenProperty(this, "scale", _hoverScale, 0.05);
		_hoverTween.TweenProperty(_outline, "modulate", _outlineColor, 0.05);
	}

	protected override void OnUnfocus()
	{
		_hoverTween?.Kill();
		_hoverTween = CreateTween().SetParallel();
		_hoverTween.TweenProperty(this, "scale", _hoverScale, 0.5).SetTrans(Tween.TransitionType.Expo).SetEase(Tween.EaseType.Out);
		_hoverTween.TweenProperty(_outline, "modulate", _outlineTransparentColor, 0.5);
	}

	protected override void OnPress()
	{
		base.OnPress();
		_hoverTween?.Kill();
		_hoverTween = CreateTween().SetParallel();
		_hoverTween.TweenProperty(this, "scale", _downScale, 0.25).SetTrans(Tween.TransitionType.Expo).SetEase(Tween.EaseType.Out);
		_hoverTween.TweenProperty(_buttonImage, "modulate", _downColor, 0.25).SetTrans(Tween.TransitionType.Cubic).SetEase(Tween.EaseType.Out);
		_hoverTween.TweenProperty(_outline, "modulate", _outlineTransparentColor, 0.25).SetTrans(Tween.TransitionType.Cubic).SetEase(Tween.EaseType.Out);
	}
}
