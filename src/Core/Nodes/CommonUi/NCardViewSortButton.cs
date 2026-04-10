using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.CommonUi;

public partial class NCardViewSortButton : NButton
{
	private static readonly StringName _v = new StringName("v");

	private static readonly StringName _s = new StringName("s");

	private static readonly StringName _h = new StringName("h");

	private bool _isDescending;

	private Control _button;

	private ShaderMaterial _hsv;

	private MegaLabel _label;

	private TextureRect _sortIcon;

	private NSelectionReticle _selectionReticle;

	private Tween? _tween;

	public bool IsDescending
	{
		get
		{
			return _isDescending;
		}
		set
		{
			_isDescending = value;
			OnToggle();
		}
	}

	public override void _Ready()
	{
		ConnectSignals();
		_button = GetNode<Control>("%ButtonImage");
		_hsv = (ShaderMaterial)_button.GetMaterial();
		_label = GetNode<MegaLabel>("%Label");
		_sortIcon = GetNode<TextureRect>("%Image");
		_selectionReticle = GetNode<NSelectionReticle>("SelectionReticle");
	}

	public void SetHue(ShaderMaterial mat)
	{
		_hsv.SetShaderParameter(_h, mat.GetShaderParameter(_h));
	}

	private void OnToggle()
	{
		_sortIcon.SetFlipV(!_isDescending);
	}

	protected override void OnRelease()
	{
		base.OnRelease();
		IsDescending = !IsDescending;
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_button, "scale", new Vector2(1.05f, 1.05f), 0.05);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderS), _hsv.GetShaderParameter(_s), 1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_button, "scale", new Vector2(1.05f, 1.05f), 0.05);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderS), _hsv.GetShaderParameter(_s), 1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		if (NControllerManager.Instance.IsUsingController)
		{
			_selectionReticle.OnSelect();
		}
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_button, "scale", new Vector2(1.05f, 1f), 0.5);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderS), _hsv.GetShaderParameter(_s), 0.8f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 0.8f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		if (NControllerManager.Instance.IsUsingController)
		{
			_selectionReticle.OnDeselect();
		}
	}

	protected override void OnPress()
	{
		base.OnPress();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_button, "scale", new Vector2(1f, 0.95f), 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderS), _hsv.GetShaderParameter(_s), 0.8f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 0.8f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	public void SetLabel(string text)
	{
		_label.SetTextAutoSize(text);
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
