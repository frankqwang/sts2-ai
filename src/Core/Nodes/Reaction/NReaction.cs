using System;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Random;

namespace MegaCrit.Sts2.Core.Nodes.Reaction;

public partial class NReaction : TextureRect
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("ui/reaction");

	private const string _exclamationPath = "res://images/ui/emote/exclaim.png";

	private const string _skullPath = "res://images/ui/emote/skull.png";

	private const string _thumbDownPath = "res://images/ui/emote/thumb_down.png";

	private const string _sadSlimePath = "res://images/ui/emote/slime_sad.png";

	private const string _questionMarkPath = "res://images/ui/emote/question.png";

	private const string _heartPath = "res://images/ui/emote/heart.png";

	private const string _thumbUpPath = "res://images/ui/emote/thumb_up.png";

	private const string _happyCultistPath = "res://images/ui/emote/happy_cultist.png";

	public ReactionType Type => TextureToType(base.Texture);

	public static NReaction Create(Texture2D reactionTexture)
	{
		NReaction nReaction = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NReaction>(PackedScene.GenEditState.Disabled);
		nReaction.Texture = reactionTexture;
		return nReaction;
	}

	public static NReaction Create(ReactionType type)
	{
		return Create(TypeToTexture(type));
	}

	public void BeginAnim()
	{
		TaskHelper.RunSafely(DoAnim());
	}

	private async Task DoAnim()
	{
		NReaction nReaction = this;
		Color modulate = base.Modulate;
		modulate.A = 0f;
		nReaction.Modulate = modulate;
		float num = Rng.Chaotic.NextFloat(40f, 60f);
		float deg = Rng.Chaotic.NextFloat(-30f, 30f);
		Vector2 vector = base.Position + Vector2.Up.Rotated(Mathf.DegToRad(deg)) * num;
		Tween tween = CreateTween();
		tween.SetParallel();
		tween.TweenProperty(this, "position", vector, 0.30000001192092896).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		tween.TweenProperty(this, "modulate:a", 1f, 0.30000001192092896).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		tween.SetParallel(parallel: false);
		tween.TweenProperty(this, "modulate:a", 0f, 0.20000000298023224).SetDelay(0.6000000238418579).SetEase(Tween.EaseType.In)
			.SetTrans(Tween.TransitionType.Expo);
		await ToSignal(tween, Tween.SignalName.Finished);
		this.QueueFreeSafely();
	}

	private static Texture2D TypeToTexture(ReactionType type)
	{
		AssetCache cache = PreloadManager.Cache;
		return cache.GetTexture2D(type switch
		{
			ReactionType.Exclamation => "res://images/ui/emote/exclaim.png", 
			ReactionType.Skull => "res://images/ui/emote/skull.png", 
			ReactionType.ThumbDown => "res://images/ui/emote/thumb_down.png", 
			ReactionType.SadSlime => "res://images/ui/emote/slime_sad.png", 
			ReactionType.QuestionMark => "res://images/ui/emote/question.png", 
			ReactionType.Heart => "res://images/ui/emote/heart.png", 
			ReactionType.ThumbUp => "res://images/ui/emote/thumb_up.png", 
			ReactionType.HappyCultist => "res://images/ui/emote/happy_cultist.png", 
			_ => throw new ArgumentOutOfRangeException("type", type, null), 
		});
	}

	private static ReactionType TextureToType(Texture2D texture)
	{
		return texture.ResourcePath switch
		{
			"res://images/ui/emote/exclaim.png" => ReactionType.Exclamation, 
			"res://images/ui/emote/skull.png" => ReactionType.Skull, 
			"res://images/ui/emote/thumb_down.png" => ReactionType.ThumbDown, 
			"res://images/ui/emote/slime_sad.png" => ReactionType.SadSlime, 
			"res://images/ui/emote/question.png" => ReactionType.QuestionMark, 
			"res://images/ui/emote/heart.png" => ReactionType.Heart, 
			"res://images/ui/emote/thumb_up.png" => ReactionType.ThumbUp, 
			"res://images/ui/emote/happy_cultist.png" => ReactionType.HappyCultist, 
			_ => throw new ArgumentOutOfRangeException("texture", texture, null), 
		};
	}
}
