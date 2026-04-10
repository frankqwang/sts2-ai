using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;
using MegaCrit.Sts2.Core.Random;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Shops;

public partial class NMerchantCharacter : Node2D
{
	public override void _Ready()
	{
		PlayAnimation("relaxed_loop", loop: true);
	}

	public void PlayAnimation(string anim, bool loop = false)
	{
		MegaTrackEntry megaTrackEntry = new MegaSprite(GetChild(0)).GetAnimationState().SetAnimation(anim, loop);
		if (loop)
		{
			megaTrackEntry?.SetTrackTime(megaTrackEntry.GetAnimationEnd() * Rng.Chaotic.NextFloat());
		}
	}
}
