using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Random;

namespace MegaCrit.Sts2.Core.Nodes.Vfx.Cards;

public partial class NHellraiserAttackVfx : Node2D
{
	public override void _Ready()
	{
		TextureRect node = GetNode<TextureRect>("%Sword");
		node.FlipH = Rng.Chaotic.NextBool();
		Tween tween = CreateTween().SetParallel();
		tween.TweenProperty(node, "position:y", 90, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		tween.TweenProperty(this, "modulate", Colors.White, 0.30000001192092896).From(Colors.Red);
		tween.Chain().TweenInterval(0.10000000149011612);
		tween.Chain();
		tween.TweenProperty(node, "position:y", 300, 0.5).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Expo);
		tween.TweenProperty(this, "modulate:a", 0f, 0.30000001192092896);
		tween.Chain().TweenCallback(Callable.From(OnTweenFinished));
	}

	private void OnTweenFinished()
	{
		this.QueueFreeSafely();
	}
}
