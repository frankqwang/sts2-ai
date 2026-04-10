using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.PauseMenu;

public partial class NPauseMenuButton : NButton
{
	private static readonly StringName _v = new StringName("v");

	private static readonly StringName _s = new StringName("s");

	private NBackButton _backButton;

	private MegaLabel _label;

	private TextureRect _image;

	private ShaderMaterial _hsv;

	private Tween? _tween;

	private const float _hoverS = 1.1f;

	private const float _hoverV = 1.1f;

	private const float _unhoverS = 0.8f;

	private const float _unhoverV = 0.9f;

	private const float _pressDownYOffset = 6f;

	public bool UseSharedBackstop => true;

	public override void _Ready()
	{
		ConnectSignals();
		_label = GetNode<MegaLabel>("Label");
		_image = GetNode<TextureRect>("ButtonImage");
		_hsv = (ShaderMaterial)_image.Material;
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_hsv.SetShaderParameter(_s, 1.1f);
		_hsv.SetShaderParameter(_v, 1.1f);
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_image, "scale", Vector2.One * 0.95f, 0.05);
		_tween.TweenProperty(_label, "theme_override_colors/font_color", StsColors.gold, 0.05000000074505806);
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenMethod(Callable.From<float>(UpdateShaderS), _hsv.GetShaderParameter(_s), 0.8f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 0.9f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(_image, "scale", Vector2.One * 0.9f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(_image, "position:y", 0f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(_label, "position:y", 0f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(_label, "theme_override_colors/font_color", StsColors.cream, 0.05000000074505806);
	}

	protected override void OnDisable()
	{
		base.OnDisable();
		base.Modulate = StsColors.gray;
	}

	protected override void OnEnable()
	{
		base.OnDisable();
		base.Modulate = Colors.White;
	}

	protected override void OnPress()
	{
		base.OnPress();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenMethod(Callable.From<float>(UpdateShaderS), _hsv.GetShaderParameter(_s), 0.8f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 0.9f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(_image, "scale", Vector2.One * 0.85f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(_image, "position:y", 6f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(_label, "position:y", 6f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
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
