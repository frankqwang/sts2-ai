using System;
using Godot;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.CommonUi;

public partial class NTickbox : NButton
{
	[Signal]
	public delegate void ToggledEventHandler(NTickbox tickbox);

	private static readonly StringName _v = new StringName("v");

	private bool _isTicked = true;

	private Control _imageContainer;

	private Control _tickedImage;

	private Control _notTickedImage;

	private ShaderMaterial _hsv;

	private Tween? _tween;

	private Vector2 _baseScale;

	private float _hoverScale = 1.05f;

	private float _pressDownScale = 0.95f;

	private float _hoverV = 1.2f;

	public bool IsTicked
	{
		get
		{
			return _isTicked;
		}
		set
		{
			_isTicked = value;
			_tickedImage.Visible = _isTicked;
			_notTickedImage.Visible = !_isTicked;
		}
	}

	public override void _Ready()
	{
		if (GetType() != typeof(NTickbox))
		{
			Log.Error($"{GetType()}");
			throw new InvalidOperationException("Don't call base._Ready()! Call ConnectSignals() instead.");
		}
		ConnectSignals();
	}

	protected override void ConnectSignals()
	{
		base.ConnectSignals();
		_imageContainer = GetNode<Control>("%TickboxVisuals");
		_baseScale = _imageContainer.Scale;
		_hsv = (ShaderMaterial)_imageContainer.Material;
		_tickedImage = GetNode<Control>("%TickboxVisuals/Ticked");
		_notTickedImage = GetNode<Control>("%TickboxVisuals/NotTicked");
	}

	protected override void OnRelease()
	{
		base.OnRelease();
		IsTicked = !IsTicked;
		if (IsTicked)
		{
			SfxCmd.Play("event:/sfx/ui/clicks/ui_checkbox_on");
			OnTick();
		}
		else
		{
			SfxCmd.Play("event:/sfx/ui/clicks/ui_checkbox_off");
			OnUntick();
		}
		EmitSignal(SignalName.Toggled, this);
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_imageContainer, "scale", _baseScale * _hoverScale, 0.05);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), _hoverV, 0.05);
	}

	protected virtual void OnUntick()
	{
	}

	protected virtual void OnTick()
	{
	}

	protected override void OnFocus()
	{
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_imageContainer, "scale", _baseScale * _hoverScale, 0.05);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), _hoverV, 0.05);
	}

	protected override void OnUnfocus()
	{
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_imageContainer, "scale", _baseScale, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	protected override void OnPress()
	{
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_imageContainer, "scale", _baseScale * _pressDownScale, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 0.8f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	protected override void OnDisable()
	{
		base.OnDisable();
		base.Modulate = StsColors.gray;
	}

	protected override void OnEnable()
	{
		base.OnEnable();
		base.Modulate = Colors.White;
	}

	private void UpdateShaderV(float value)
	{
		_hsv.SetShaderParameter(_v, value);
	}
}
