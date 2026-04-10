using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NLostAndForgottenVfx : Node
{
	private GpuParticles2D _dustParticles;

	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_dustParticles = _parent.GetNode<GpuParticles2D>("GranuleEmitterBone/DustParticles");
		_dustParticles.Emitting = false;
		_animController.GetAnimationState().SetAnimation("die");
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		string eventName = new MegaEvent(spineEvent).GetData().GetEventName();
		if (!(eventName == "start_dust"))
		{
			if (eventName == "stop_dust")
			{
				OnDustStop();
			}
		}
		else
		{
			OnDustStart();
		}
	}

	private void OnDustStart()
	{
		_dustParticles.Emitting = true;
	}

	private void OnDustStop()
	{
		_dustParticles.Emitting = false;
	}
}
