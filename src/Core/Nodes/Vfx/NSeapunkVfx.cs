using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NSeapunkVfx : Node
{
	private GpuParticles2D _bubbleParticles;

	private GpuParticles2D _weedParticles;

	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_bubbleParticles = _parent.GetNode<GpuParticles2D>("BubbleParticles");
		_weedParticles = _parent.GetNode<GpuParticles2D>("WeedParticles");
		_bubbleParticles.OneShot = true;
		_bubbleParticles.Emitting = false;
		_weedParticles.OneShot = true;
		_weedParticles.Emitting = false;
		_animController.GetAnimationState().SetAnimation("cast");
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		string eventName = new MegaEvent(spineEvent).GetData().GetEventName();
		if (!(eventName == "bubbles_start"))
		{
			if (eventName == "weeds_start")
			{
				StartWeeds();
			}
		}
		else
		{
			StartBubbles();
		}
	}

	private void StartBubbles()
	{
		_bubbleParticles.Restart();
	}

	private void StartWeeds()
	{
		_weedParticles.Restart();
	}
}
