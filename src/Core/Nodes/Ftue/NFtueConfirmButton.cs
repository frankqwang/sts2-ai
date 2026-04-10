using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Ftue;

public partial class NFtueConfirmButton : NButton
{
	private static readonly StringName _v = new StringName("v");

	private TextureRect _outline;

	private Tween? _outlineTween;

	private Tween? _scaleTween;

	private ShaderMaterial _hsv;

	private MegaLabel _label;

	protected override string[] Hotkeys => new string[1] { MegaInput.select };

	public override void _Ready()
	{
		ConnectSignals();
		_outline = GetNode<TextureRect>("Outline");
		_hsv = (ShaderMaterial)base.Material;
		_label = GetNode<MegaLabel>("%Label");
		_label.SetTextAutoSize(new LocString("ftues", "CONFIRM_BUTTON").GetRawText());
		_outlineTween = CreateTween().SetLoops();
		_outlineTween.TweenProperty(_outline, "modulate:a", 1f, 0.6);
		_outlineTween.TweenProperty(_outline, "modulate:a", 0.25f, 0.6);
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_outlineTween?.Kill();
		_outline.Modulate = StsColors.gold;
		_scaleTween?.Kill();
		_scaleTween = CreateTween();
		_scaleTween.TweenProperty(this, "scale", Vector2.One * 1.05f, 0.05).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_scaleTween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 1.4f, 0.05).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_scaleTween?.Kill();
		_scaleTween = CreateTween();
		_scaleTween.TweenProperty(this, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_scaleTween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_outlineTween?.Kill();
		_outlineTween = CreateTween().SetLoops();
		_outlineTween.TweenProperty(_outline, "modulate:a", 0.25f, 0.6);
		_outlineTween.TweenProperty(_outline, "modulate:a", 1f, 0.6).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
	}

	private void UpdateShaderV(float value)
	{
		_hsv.SetShaderParameter(_v, value);
	}
}
