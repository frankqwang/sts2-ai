using System;
using Godot;
using Godot.Collections;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Animation;

[GlobalClass]
public partial class NSpineAutoPlayer : Node
{
	public override void _Ready()
	{
		MegaSprite megaSprite = new MegaSprite(GetParent());
		Array<GodotObject> animations = megaSprite.GetSkeleton().GetData().GetAnimations();
		if (animations.Count != 1)
		{
			throw new InvalidOperationException($"{"NSpineAutoPlayer"}'s parent's skeleton data must have exactly 1 animation. This has {animations.Count}.");
		}
		megaSprite.GetAnimationState().SetAnimation(new MegaAnimation(animations[0]).GetName());
	}
}
