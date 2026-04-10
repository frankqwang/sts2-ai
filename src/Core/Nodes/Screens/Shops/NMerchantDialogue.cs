using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Random;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Shops;

public partial class NMerchantDialogue : Node2D
{
	private static readonly StringName _v = new StringName("v");

	private static readonly StringName _s = new StringName("s");

	private static readonly StringName _h = new StringName("h");

	private static readonly Vector2 _xRange = new Vector2(450f, 1450f);

	private MegaRichTextLabel _label;

	private Node2D _dialogueBox;

	private Tween? _tween;

	private Sprite2D _bubble;

	private ShaderMaterial _hsv;

	private MerchantDialogueSet _dialogueSet;

	public override void _Ready()
	{
		_label = GetNode<MegaRichTextLabel>("%Text");
		_dialogueBox = GetNode<Node2D>("%DialogueBox");
		_bubble = GetNode<Sprite2D>("%Bubble");
		base.Modulate = Colors.Transparent;
		_hsv = (ShaderMaterial)_bubble.Material;
		_hsv.SetShaderParameter(_h, 1f);
		_hsv.SetShaderParameter(_s, 1.2f);
		_hsv.SetShaderParameter(_v, 0.4f);
	}

	public void Initialize(MerchantDialogueSet dialogueSet)
	{
		_dialogueSet = dialogueSet;
	}

	public void ShowOnInventoryOpen()
	{
		ShowRandom(_dialogueSet.OpenInventoryLines);
	}

	public void ShowForPurchaseAttempt(PurchaseStatus status)
	{
		ShowRandom(_dialogueSet.GetPurchaseSuccessLines(status));
	}

	private void ShowRandom(IEnumerable<LocString> lines)
	{
		LocString locString = Rng.Chaotic.NextItem(lines);
		if (locString != null)
		{
			_label.Text = "[fly_in]" + locString.GetFormattedText() + "[/fly_in]";
			base.Modulate = StsColors.transparentWhite;
			base.Position = new Vector2(Rng.Chaotic.NextFloat(_xRange.X, _xRange.Y), base.Position.Y);
			_tween?.Kill();
			_tween = CreateTween().SetParallel();
			_tween.TweenProperty(this, "modulate:a", 1f, 0.25);
			_tween.TweenProperty(this, "scale", Vector2.One, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
			_tween.TweenProperty(_label, "visible_ratio", 1f, 0.4).From(0f);
			_tween.TweenProperty(_bubble, "scale", new Vector2(0.75f, 0.75f), 0.5).From(new Vector2(0.25f, 0.25f)).SetEase(Tween.EaseType.Out)
				.SetTrans(Tween.TransitionType.Expo);
			_tween.TweenProperty(_dialogueBox, "position:y", 0f, 0.5).From(-80f).SetEase(Tween.EaseType.Out)
				.SetTrans(Tween.TransitionType.Back);
			_tween.Chain();
			_tween.TweenInterval(1.0);
			_tween.Chain();
			_tween.TweenProperty(this, "modulate:a", 0f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Sine);
		}
	}
}
