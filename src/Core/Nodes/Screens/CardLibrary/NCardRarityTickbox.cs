using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.HoverTips;

namespace MegaCrit.Sts2.Core.Nodes.Screens.CardLibrary;

public partial class NCardRarityTickbox : NTickbox
{
	private MegaLabel _label;

	private Tween? _labelTween;

	public LocString Loc { get; set; }

	public override void _Ready()
	{
		ConnectSignals();
		_label = GetNode<MegaLabel>("Label");
	}

	protected override void OnRelease()
	{
		base.OnRelease();
		_labelTween?.Kill();
		_labelTween = CreateTween().SetParallel();
		_labelTween.TweenProperty(_label, "scale", Vector2.One * 1.1f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Bounce);
		_labelTween.TweenProperty(_label, "self_modulate", Colors.White, 0.05);
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_labelTween?.Kill();
		_labelTween = CreateTween().SetParallel();
		_labelTween.TweenProperty(_label, "scale", Vector2.One * 1.1f, 0.05);
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, new HoverTip(Loc));
		nHoverTipSet.GlobalPosition = new Vector2(310f, base.GlobalPosition.Y);
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_labelTween?.Kill();
		_labelTween = CreateTween().SetParallel();
		_labelTween.TweenProperty(_label, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		NHoverTipSet.Remove(this);
	}

	protected override void OnPress()
	{
		base.OnPress();
		_labelTween?.Kill();
		_labelTween = CreateTween().SetParallel();
		_labelTween.TweenProperty(_label, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_labelTween.TweenProperty(_label, "self_modulate", StsColors.lightGray, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	protected override void OnDisable()
	{
		base.Modulate = StsColors.gray;
	}

	protected override void OnEnable()
	{
		base.Modulate = Colors.White;
	}

	public void SetLabel(string text)
	{
		_label.SetTextAutoSize(text);
	}
}
