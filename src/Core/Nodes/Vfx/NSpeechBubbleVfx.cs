using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NSpeechBubbleVfx : Control
{
	private static readonly StringName _v = new StringName("v");

	private static readonly StringName _s = new StringName("s");

	private static readonly StringName _h = new StringName("h");

	private Control _container;

	private MegaRichTextLabel _label;

	private Node2D _contents;

	private Sprite2D _bubble;

	private Sprite2D _shadow;

	private ShaderMaterial _hsv;

	private const string _path = "res://scenes/vfx/vfx_speech_bubble.tscn";

	private Tween? _tween;

	private const float _spawnProportionToTopOfHitbox = 0.75f;

	private const float _spawnProportionToEdgeOfHitbox = 0.75f;

	private Vector2 _startPos;

	private VfxColor _vfxColor;

	private DialogueStyle _style;

	private DialogueSide _side;

	private string _text;

	private float _elapsedTime = 3.14f;

	private const float _waveFrequency = 4.5f;

	private const float _waveAmplitude = 2f;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>("res://scenes/vfx/vfx_speech_bubble.tscn");

	public double SecondsToDisplay { get; private set; }

	public static NSpeechBubbleVfx? Create(string text, Creature speaker, double secondsToDisplay, VfxColor vfxColor = VfxColor.White)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NSpeechBubbleVfx nSpeechBubbleVfx = CreateInternal(text, speaker.Side switch
		{
			CombatSide.Player => DialogueSide.Left, 
			CombatSide.Enemy => DialogueSide.Right, 
			_ => throw new ArgumentOutOfRangeException(), 
		}, secondsToDisplay);
		nSpeechBubbleVfx._startPos = GetCreatureSpeechPosition(speaker);
		nSpeechBubbleVfx._vfxColor = vfxColor;
		return nSpeechBubbleVfx;
	}

	public static NSpeechBubbleVfx? Create(string text, DialogueSide side, Vector2 globalPosition, double secondsToDisplay, VfxColor vfxColor = VfxColor.White)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NSpeechBubbleVfx nSpeechBubbleVfx = CreateInternal(text, side, secondsToDisplay);
		nSpeechBubbleVfx._startPos = globalPosition;
		nSpeechBubbleVfx._vfxColor = vfxColor;
		return nSpeechBubbleVfx;
	}

	public override void _Ready()
	{
		_contents = GetNode<Node2D>("%Contents");
		_bubble = GetNode<Sprite2D>("%Bubble");
		_shadow = GetNode<Sprite2D>("%Shadow");
		_container = GetNode<Control>("%Container");
		_label = GetNode<MegaRichTextLabel>("%Text");
		_hsv = (ShaderMaterial)_bubble.Material;
		SetSpeechBubbleColor();
		base.GlobalPosition = _startPos;
		if (_side == DialogueSide.Right)
		{
			_container.Position = new Vector2(0f - _container.Size.X - _container.Position.X, _container.Position.Y);
			_bubble.FlipH = true;
			_shadow.FlipH = true;
		}
		TaskHelper.RunSafely(AnimateSpeechBubble());
	}

	private void SetSpeechBubbleColor()
	{
		switch (_vfxColor)
		{
		case VfxColor.Blue:
			_hsv.SetShaderParameter(_h, 0.05f);
			_hsv.SetShaderParameter(_s, 1.3f);
			_hsv.SetShaderParameter(_v, 0.55f);
			break;
		case VfxColor.Green:
			_hsv.SetShaderParameter(_h, 0.8f);
			_hsv.SetShaderParameter(_s, 1.5f);
			_hsv.SetShaderParameter(_v, 0.6f);
			break;
		case VfxColor.Purple:
			_hsv.SetShaderParameter(_h, 0.3f);
			_hsv.SetShaderParameter(_s, 0.6f);
			_hsv.SetShaderParameter(_v, 0.5f);
			break;
		case VfxColor.Red:
			_hsv.SetShaderParameter(_h, 0.48f);
			_hsv.SetShaderParameter(_s, 2f);
			_hsv.SetShaderParameter(_v, 0.5f);
			break;
		case VfxColor.Black:
			_hsv.SetShaderParameter(_h, 1f);
			_hsv.SetShaderParameter(_s, 0.25f);
			_hsv.SetShaderParameter(_v, 0.3f);
			break;
		default:
			_hsv.SetShaderParameter(_h, 1f);
			_hsv.SetShaderParameter(_s, 0.9f);
			_hsv.SetShaderParameter(_v, 0.5f);
			break;
		}
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
	}

	private async Task AnimateSpeechBubble()
	{
		_tween = CreateTween().SetParallel();
		float value = 60f * ((_side == DialogueSide.Left) ? (-1f) : 1f);
		_label.Text = $"[center][fly_in offset_x={value} offset_y=40]{_text}[/fly_in][/center]";
		_tween.TweenProperty(this, "modulate:a", 1f, 0.4).From(0f);
		_tween.TweenProperty(_label, "visible_ratio", 1f, 0.4).From(0f);
		_tween.TweenProperty(_bubble, "scale", new Vector2(0.75f, 0.75f), 0.5).From(new Vector2(0.25f, 0.25f)).SetEase(Tween.EaseType.Out)
			.SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(this, "rotation_degrees", 0f, 0.3).From(7f).SetEase(Tween.EaseType.Out)
			.SetTrans(Tween.TransitionType.Sine);
		_tween.Chain();
		double time = Math.Max(SecondsToDisplay - 1.0, 1.0);
		_tween.TweenInterval(time);
		_tween.Chain();
		await AnimOutInternal();
	}

	public async Task AnimOut()
	{
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		await AnimOutInternal();
	}

	private async Task AnimOutInternal()
	{
		_tween.TweenProperty(this, "modulate", StsColors.transparentBlack, 0.4).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Sine);
		await ToSignal(_tween, Tween.SignalName.Finished);
		this.QueueFreeSafely();
	}

	private static NSpeechBubbleVfx CreateInternal(string text, DialogueSide side, double secondsToDisplay)
	{
		NSpeechBubbleVfx nSpeechBubbleVfx = PreloadManager.Cache.GetScene("res://scenes/vfx/vfx_speech_bubble.tscn").Instantiate<NSpeechBubbleVfx>(PackedScene.GenEditState.Disabled);
		nSpeechBubbleVfx._side = side;
		nSpeechBubbleVfx._text = text;
		nSpeechBubbleVfx.SecondsToDisplay = secondsToDisplay;
		return nSpeechBubbleVfx;
	}

	public override void _Process(double delta)
	{
		_elapsedTime += (float)delta * 4.5f;
		_contents.Position = new Vector2(0f, Mathf.Sin(_elapsedTime) * 2f);
	}

	private static Vector2 GetCreatureSpeechPosition(Creature speaker)
	{
		NCreature creatureNode = NCombatRoom.Instance.GetCreatureNode(speaker);
		if (creatureNode.Visuals.TalkPosition != null)
		{
			return creatureNode.Visuals.TalkPosition.GlobalPosition;
		}
		Vector2 result = creatureNode.VfxSpawnPosition + new Vector2(0f, (0f - creatureNode.Hitbox.Size.Y) * 0.5f * 0.75f);
		if (speaker.Side == CombatSide.Player)
		{
			result.X += creatureNode.Hitbox.Size.X * 0.75f;
		}
		else
		{
			result.X -= creatureNode.Hitbox.Size.X * 0.75f;
		}
		return result;
	}
}
