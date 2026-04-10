using Godot;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;

namespace MegaCrit.Sts2.Core.Nodes.Screens.CardLibrary;

public partial class NCardTypeTickbox : NButton
{
	[Signal]
	public delegate void ToggledEventHandler(NCardTypeTickbox tickbox);

	private static readonly StringName _v = new StringName("v");

	private static readonly StringName _s = new StringName("s");

	private bool _isTicked = true;

	private float _baseS = 1f;

	private float _baseV = 1.2f;

	private Control _outline;

	private Control _image;

	private ShaderMaterial _hsv;

	private Tween? _tween;

	private Vector2 _baseScale;

	private float _hoverScale = 1.2f;

	private float _pressDownScale = 0.9f;

	public bool IsTicked
	{
		get
		{
			return _isTicked;
		}
		set
		{
			_isTicked = value;
			_outline.Visible = _isTicked;
			OnToggle();
		}
	}

	public LocString Loc { get; set; }

	public override void _Ready()
	{
		ConnectSignals();
		_baseScale = base.Scale;
		_image = GetNode<Control>("%Image");
		_outline = GetNode<Control>("%Outline");
		_hsv = (ShaderMaterial)_image.Material;
	}

	private void OnToggle()
	{
		_baseS = (_isTicked ? 1f : 0.65f);
		_baseV = (_isTicked ? 1.2f : 0.7f);
		UpdateShaderS(_baseS);
		UpdateShaderV(_baseV);
	}

	protected override void OnRelease()
	{
		base.OnRelease();
		IsTicked = !IsTicked;
		EmitSignal(SignalName.Toggled, this);
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_image, "scale", _baseScale * _hoverScale, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_s), _baseS, 0.05);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderS), _hsv.GetShaderParameter(_v), _baseV, 0.05);
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_image, "scale", _baseScale * _hoverScale, 0.05);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderS), _hsv.GetShaderParameter(_s), _isTicked ? 1.4f : 1f, 0.05);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), _isTicked ? 1.4f : 1f, 0.05);
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, new HoverTip(Loc));
		nHoverTipSet.GlobalPosition = new Vector2(310f, base.GlobalPosition.Y);
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_image, "scale", _baseScale, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderS), _hsv.GetShaderParameter(_s), _baseS, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), _baseV, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		NHoverTipSet.Remove(this);
	}

	protected override void OnPress()
	{
		base.OnPress();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_image, "scale", _baseScale * _pressDownScale, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderS), _hsv.GetShaderParameter(_s), _baseS, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), _baseV, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	private void UpdateShaderS(float value)
	{
		_hsv.SetShaderParameter(_s, value);
	}

	private void UpdateShaderV(float value)
	{
		_hsv.SetShaderParameter(_v, value);
	}
}
