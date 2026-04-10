using Godot;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;

namespace MegaCrit.Sts2.Core.Nodes.Screens.CardLibrary;

public partial class NCardPoolFilter : NButton
{
	[Signal]
	public delegate void ToggledEventHandler(NCardPoolFilter filter);

	private static readonly StringName _v = new StringName("v");

	private static readonly StringName _s = new StringName("s");

	private bool _isSelected;

	private Control _image;

	private ShaderMaterial _hsv;

	private NSelectionReticle _controllerSelectionReticle;

	private Tween? _tween;

	private const float _focusedMultiplier = 1.2f;

	private const float _pressDownMultiplier = 0.8f;

	private static readonly Vector2 _enabledScale = Vector2.One * 1.1f;

	private static readonly Vector2 _disabledScale = Vector2.One * 0.95f;

	public LocString? Loc { get; set; }

	public bool IsSelected
	{
		get
		{
			return _isSelected;
		}
		set
		{
			_isSelected = value;
			OnToggle();
		}
	}

	public override void _Ready()
	{
		ConnectSignals();
		_image = GetNode<Control>("Image");
		_controllerSelectionReticle = GetNode<NSelectionReticle>("%SelectionReticle");
		_hsv = (ShaderMaterial)_image.GetMaterial();
	}

	private void OnToggle()
	{
		_tween?.Kill();
		_hsv.SetShaderParameter(_s, _isSelected ? 1f : 0.3f);
		_hsv.SetShaderParameter(_v, _isSelected ? 1f : 0.55f);
		if (!_isSelected)
		{
			_tween = CreateTween().SetParallel();
			_tween.TweenProperty(_image, "scale", _disabledScale, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		}
		else
		{
			_tween = CreateTween().SetParallel();
			_tween.TweenProperty(_image, "scale", _enabledScale, 0.2).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
		}
	}

	protected override void OnRelease()
	{
		base.OnRelease();
		IsSelected = !IsSelected;
		EmitSignal(SignalName.Toggled, this);
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_image, "scale", (_isSelected ? _enabledScale : _disabledScale) * 1.2f, 0.05);
		if (NControllerManager.Instance.IsUsingController)
		{
			_controllerSelectionReticle.OnSelect();
		}
		if (Loc != null)
		{
			NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, new HoverTip(Loc));
			nHoverTipSet.GlobalPosition = new Vector2(310f, base.GlobalPosition.Y);
		}
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_image, "scale", _isSelected ? _enabledScale : _disabledScale, 0.3);
		_controllerSelectionReticle.OnDeselect();
		if (Loc != null)
		{
			NHoverTipSet.Remove(this);
		}
	}

	protected override void OnPress()
	{
		if (!_isSelected)
		{
			base.OnPress();
			_tween?.Kill();
			_tween = CreateTween().SetParallel();
			_tween.TweenProperty(_image, "scale", (_isSelected ? _enabledScale : _disabledScale) * 0.8f, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		}
	}
}
