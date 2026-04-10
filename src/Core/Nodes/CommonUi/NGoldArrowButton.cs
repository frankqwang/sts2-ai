using Godot;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.CommonUi;

public partial class NGoldArrowButton : NButton
{
	private static readonly StringName _v = new StringName("v");

	protected TextureRect _icon;

	private ShaderMaterial _hsv;

	private Tween? _animTween;

	private float _valueDefault = 0.9f;

	private float _valueHovered = 1.2f;

	private Vector2 _hoverScale = Vector2.One * 1.1f;

	public override void _Ready()
	{
		ConnectSignals();
		_icon = GetNode<TextureRect>("TextureRect");
		_hsv = (ShaderMaterial)_icon.Material;
		_hsv.SetShaderParameter(_v, _valueDefault);
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_animTween?.Kill();
		_hsv.SetShaderParameter(_v, _valueHovered);
		_icon.Scale = _hoverScale;
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_animTween?.Kill();
		_animTween = CreateTween().SetParallel();
		_animTween.TweenMethod(Callable.From<float>(UpdateShaderParam), _valueHovered, _valueDefault, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_animTween.TweenProperty(_icon, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	protected override void OnPress()
	{
		base.OnPress();
		_animTween?.Kill();
		_animTween = CreateTween();
		_hsv.SetShaderParameter(_v, 0.7f);
		_animTween.TweenProperty(_icon, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	protected override void OnRelease()
	{
		if (base.IsFocused)
		{
			_animTween?.Kill();
			_hsv.SetShaderParameter(_v, _valueHovered);
			_icon.Scale = _hoverScale;
		}
	}

	private void UpdateShaderParam(float newV)
	{
		_hsv.SetShaderParameter(_v, newV);
	}
}
