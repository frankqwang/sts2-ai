using System.Linq;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Events;

public partial class NAncientDialogueHitbox : NButton
{
	private MegaLabel _label;

	private TextureRect _arrow;

	private Tween? _loopTween;

	private Tween? _tween;

	private bool _isAnimating;

	protected override string[] Hotkeys => new string[1] { MegaInput.accept };

	public string? GetHotkey()
	{
		return Hotkeys.FirstOrDefault();
	}

	public override void _Ready()
	{
		ConnectSignals();
		_label = GetNode<MegaLabel>("%Label");
		_label.SelfModulate = StsColors.transparentWhite;
		_label.Text = string.Empty;
		_arrow = GetNode<TextureRect>("%Arrow");
		_arrow.SelfModulate = StsColors.transparentWhite;
		_loopTween = CreateTween().SetParallel().SetLoops();
		_loopTween.TweenProperty(_arrow, "position:x", _arrow.Position.X + 4f, 0.4).SetEase(Tween.EaseType.InOut).SetTrans(Tween.TransitionType.Sine);
		_loopTween.Chain().TweenProperty(_arrow, "position:x", _arrow.Position.X - 4f, 0.6).SetEase(Tween.EaseType.InOut)
			.SetTrans(Tween.TransitionType.Sine);
		Tween tween = CreateTween().SetParallel();
		tween.TweenProperty(_label, "self_modulate:a", 1f, 1.0).SetDelay(0.5);
		tween.TweenProperty(_arrow, "self_modulate:a", 1f, 1.0).SetDelay(0.5);
	}

	protected override void OnRelease()
	{
		base.OnRelease();
		_loopTween?.Play();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_label, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(_arrow, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	protected override void OnPress()
	{
		base.OnPress();
		_loopTween?.Pause();
		_label.PivotOffset = _label.Size * 0.5f;
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_label, "scale", Vector2.One * 0.9f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(_arrow, "scale", Vector2.One * 0.9f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_isAnimating = false;
	}

	protected override void OnFocus()
	{
	}
}
