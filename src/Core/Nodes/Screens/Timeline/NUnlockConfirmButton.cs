using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Timeline;

public partial class NUnlockConfirmButton : NButton
{
	private Tween? _tween;

	private Tween? _hoverTween;

	protected override string[] Hotkeys => new string[2]
	{
		MegaInput.select,
		MegaInput.pauseAndBack
	};

	public override void _Ready()
	{
		ConnectSignals();
		GetNode<MegaLabel>("Label").SetTextAutoSize(new LocString("timeline", "UNLOCK_CONFIRM").GetFormattedText());
		_tween = CreateTween();
		float y = base.Position.Y;
		base.Position = new Vector2(base.Position.X, y + 180f);
		_tween.TweenProperty(this, "position:y", y, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
			.SetDelay(0.1);
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_hoverTween?.Kill();
		_hoverTween = CreateTween();
		_hoverTween.TweenProperty(this, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_hoverTween?.Kill();
		_hoverTween = CreateTween();
		_hoverTween.TweenProperty(this, "scale", Vector2.One * 1.1f, 0.05);
	}

	public override void _Process(double delta)
	{
	}

	protected override void OnPress()
	{
		base.OnPress();
		_hoverTween?.Kill();
		_hoverTween = CreateTween().SetParallel();
		_hoverTween.TweenProperty(this, "scale", Vector2.One * 0.95f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	protected override void OnRelease()
	{
		Disable();
		_hoverTween?.Kill();
		_hoverTween = CreateTween().SetParallel();
		_hoverTween.TweenProperty(this, "scale", new Vector2(3f, 0.1f), 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_hoverTween.TweenProperty(this, "modulate:a", 0f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
	}
}
