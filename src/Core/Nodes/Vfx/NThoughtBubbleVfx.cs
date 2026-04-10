using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NThoughtBubbleVfx : Control
{
	private Control _container;

	private MegaRichTextLabel _label;

	private TextureRect _textureRect;

	private Node2D _contents;

	private Node2D _tail;

	private const string _path = "res://scenes/vfx/vfx_thought_bubble.tscn";

	private const float _spawnProportionToTopOfHitbox = 0.75f;

	private const float _spawnProportionToEdgeOfHitbox = 0.75f;

	private Vector2? _startPos;

	private DialogueStyle _style;

	private DialogueSide _side;

	private string? _text;

	private Texture2D? _texture;

	private double? _secondsToDisplay;

	private Tween? _tween;

	private CancellationTokenSource? _cts;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>("res://scenes/vfx/vfx_thought_bubble.tscn");

	public static NThoughtBubbleVfx? Create(string text, Creature speaker, double? secondsToDisplay)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NThoughtBubbleVfx nThoughtBubbleVfx = CreateInternal(text, null, speaker.Side switch
		{
			CombatSide.Player => DialogueSide.Left, 
			CombatSide.Enemy => DialogueSide.Right, 
			_ => throw new ArgumentOutOfRangeException(), 
		}, secondsToDisplay);
		nThoughtBubbleVfx._startPos = GetCreatureSpeechPosition(speaker);
		return nThoughtBubbleVfx;
	}

	public static NThoughtBubbleVfx? Create(string text, DialogueSide side, double? secondsToDisplay)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return CreateInternal(text, null, side, secondsToDisplay);
	}

	public static NThoughtBubbleVfx? Create(Texture2D texture, DialogueSide side, double? secondsToDisplay)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return CreateInternal(null, texture, side, secondsToDisplay);
	}

	public override void _Ready()
	{
		_contents = GetNode<Node2D>("%Contents");
		_container = GetNode<Control>("%Container");
		_label = GetNode<MegaRichTextLabel>("%Text");
		_textureRect = GetNode<TextureRect>("%Image");
		_tail = GetNode<Node2D>("%Tail");
		if (_startPos.HasValue)
		{
			base.GlobalPosition = _startPos.Value;
		}
		if (_side == DialogueSide.Right)
		{
			_container.Position = new Vector2(0f - _container.Size.X - _container.Position.X, _container.Position.Y);
			_tail.Scale = new Vector2(-1f, 1f);
		}
		TaskHelper.RunSafely(AnimateThoughtBubble());
	}

	private async Task AnimateThoughtBubble()
	{
		_tween = CreateTween().SetParallel();
		float value = 30f * ((_side == DialogueSide.Left) ? (-1f) : 1f);
		if (_text != null)
		{
			_label.Text = $"[center][fly_in offset_x={value} offset_y=40]{_text}[/fly_in][/center]";
		}
		_label.Visible = _text != null;
		if (_texture != null)
		{
			_textureRect.Texture = _texture;
		}
		_textureRect.Visible = _texture != null;
		base.Scale = Vector2.One * 0.75f;
		base.Modulate = StsColors.transparentWhite;
		_tween.TweenProperty(_label, "visible_ratio", 1f, 0.4).From(0f);
		_tween.TweenProperty(_textureRect, "modulate:a", 1f, 0.4).From(0f);
		_tween.TweenProperty(this, "modulate:a", 1f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_tween.TweenProperty(this, "scale", Vector2.One, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
		if (_secondsToDisplay.HasValue)
		{
			double num = Math.Max(_secondsToDisplay.Value, 1.0);
			_cts = new CancellationTokenSource();
			await Cmd.Wait((float)num, _cts.Token);
			await GoAway();
		}
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
		_cts?.Cancel();
		_cts?.Dispose();
	}

	public async Task GoAway()
	{
		if (GodotObject.IsInstanceValid(this))
		{
			_tween?.Kill();
			_tween = CreateTween();
			_tween.TweenProperty(this, "modulate:a", 0f, 0.4).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Sine);
			await ToSignal(_tween, Tween.SignalName.Finished);
			this.QueueFreeSafely();
		}
	}

	private static NThoughtBubbleVfx CreateInternal(string? text, Texture2D? texture, DialogueSide side, double? secondsToDisplay)
	{
		NThoughtBubbleVfx nThoughtBubbleVfx = PreloadManager.Cache.GetScene("res://scenes/vfx/vfx_thought_bubble.tscn").Instantiate<NThoughtBubbleVfx>(PackedScene.GenEditState.Disabled);
		nThoughtBubbleVfx._side = side;
		nThoughtBubbleVfx._text = text;
		nThoughtBubbleVfx._texture = texture;
		nThoughtBubbleVfx._secondsToDisplay = secondsToDisplay;
		return nThoughtBubbleVfx;
	}

	public static Vector2 GetCreatureSpeechPosition(Creature speaker)
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

	public void SetTexture(Texture2D texture)
	{
		if (_texture == null)
		{
			throw new NotImplementedException("Can't set texture unless thought bubble was initialized with a texture");
		}
		_texture = texture;
		_textureRect.Texture = texture;
	}
}
