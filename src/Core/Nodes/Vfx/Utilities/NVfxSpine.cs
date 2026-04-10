using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;

public partial class NVfxSpine : Node2D
{
	[Export(PropertyHint.None, "")]
	private string _animation;

	public override void _Ready()
	{
		MegaSprite megaSprite = new MegaSprite(GetNode("SpineSprite"));
		megaSprite.ConnectAnimationCompleted(Callable.From<GodotObject, GodotObject, GodotObject>(AnimationEnded));
		megaSprite.GetAnimationState().SetAnimation(_animation, loop: false);
	}

	private void AnimationEnded(GodotObject _, GodotObject __, GodotObject ___)
	{
		this.QueueFreeSafely();
	}
}
