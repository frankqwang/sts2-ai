using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NFossilStalkerVfx : Node
{
	private GpuParticles2D _debuffParticles;

	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_debuffParticles = GetNode<GpuParticles2D>("../DebuffBone/DebuffParticles");
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_debuffParticles.Emitting = false;
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		string eventName = new MegaEvent(spineEvent).GetData().GetEventName();
		if (!(eventName == "debuff_spray_start"))
		{
			if (eventName == "debuff_spray_stop")
			{
				StopDebuff();
			}
		}
		else
		{
			StartDebuff();
		}
	}

	private void StartDebuff()
	{
		_debuffParticles.Emitting = true;
	}

	private void StopDebuff()
	{
		_debuffParticles.Emitting = false;
	}
}
