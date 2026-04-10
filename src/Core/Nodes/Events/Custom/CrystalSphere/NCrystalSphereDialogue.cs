using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Random;

namespace MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;

public partial class NCrystalSphereDialogue : Node2D
{
	private static readonly StringName _v = new StringName("v");

	private static readonly StringName _s = new StringName("s");

	private static readonly StringName _h = new StringName("h");

	private static readonly Vector2 _xRange = new Vector2(1100f, 1200f);

	private static readonly LocString[] _startLines = new LocString[2]
	{
		new LocString("events", "CRYSTAL_SPHERE.banter.START.1"),
		new LocString("events", "CRYSTAL_SPHERE.banter.START.2")
	};

	private static readonly LocString[] _revealBadLines = new LocString[3]
	{
		new LocString("events", "CRYSTAL_SPHERE.banter.REVEAL_BAD.1"),
		new LocString("events", "CRYSTAL_SPHERE.banter.REVEAL_BAD.2"),
		new LocString("events", "CRYSTAL_SPHERE.banter.REVEAL_BAD.3")
	};

	private static readonly LocString[] _revealGoodLines = new LocString[5]
	{
		new LocString("events", "CRYSTAL_SPHERE.banter.REVEAL_GOOD.1"),
		new LocString("events", "CRYSTAL_SPHERE.banter.REVEAL_GOOD.2"),
		new LocString("events", "CRYSTAL_SPHERE.banter.REVEAL_GOOD.3"),
		new LocString("events", "CRYSTAL_SPHERE.banter.REVEAL_GOOD.4"),
		new LocString("events", "CRYSTAL_SPHERE.banter.REVEAL_GOOD.5")
	};

	private static readonly LocString[] _endLines = new LocString[2]
	{
		new LocString("events", "CRYSTAL_SPHERE.banter.END.1"),
		new LocString("events", "CRYSTAL_SPHERE.banter.END.2")
	};

	private MegaRichTextLabel _label;

	private Node2D _dialogueBox;

	private Tween? _tween;

	private Sprite2D _bubble;

	private ShaderMaterial _hsv;

	public override void _Ready()
	{
		_label = GetNode<MegaRichTextLabel>("%Text");
		_dialogueBox = GetNode<Node2D>("%DialogueBox");
		_bubble = GetNode<Sprite2D>("%Bubble");
		base.Modulate = Colors.Transparent;
		_hsv = (ShaderMaterial)_bubble.Material;
		_hsv.SetShaderParameter(_h, 0.25f);
		_hsv.SetShaderParameter(_s, 0.75f);
		_hsv.SetShaderParameter(_v, 1f);
	}

	public void PlayStart()
	{
		Play(Rng.Chaotic.NextItem(_startLines));
	}

	public void PlayBad()
	{
		Play(Rng.Chaotic.NextItem(_revealBadLines));
	}

	public void PlayGood()
	{
		Play(Rng.Chaotic.NextItem(_revealGoodLines));
	}

	public void PlayEnd()
	{
		Play(Rng.Chaotic.NextItem(_endLines));
	}

	private void Play(LocString locString)
	{
		_label.Text = "[fly_in]" + locString.GetFormattedText() + "[/fly_in]";
		Log.Info(_label.Text ?? "");
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
		_tween.TweenInterval(1.5);
		_tween.Chain();
		_tween.TweenProperty(this, "modulate:a", 0f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Sine);
	}
}
