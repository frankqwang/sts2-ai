using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NNecrobinderFlameVfx : Node
{
	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent.GetParent());
		_animController.ConnectAnimationStarted(Callable.From<GodotObject, GodotObject, GodotObject>(UpdateFlameVisibility));
	}

	private void UpdateFlameVisibility(GodotObject spineSprite, GodotObject animationState, GodotObject trackEntry)
	{
		_parent.Visible = new MegaAnimationState(animationState).GetCurrent(0).GetAnimation().GetName() != "die";
	}
}
