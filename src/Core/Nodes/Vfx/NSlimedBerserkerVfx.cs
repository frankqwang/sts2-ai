using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NSlimedBerserkerVfx : Node
{
	private GpuParticles2D _gooParticlesR;

	private GpuParticles2D _gooParticlesL;

	private GpuParticles2D _gooParticlesVomit;

	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_gooParticlesR = _parent.GetNode<GpuParticles2D>("ParticleSlotNodeR/GooParticles");
		_gooParticlesL = _parent.GetNode<GpuParticles2D>("ParticleSlotNodeL/GooParticles");
		_gooParticlesVomit = _parent.GetNode<GpuParticles2D>("ParticleSlotNodeVomit/GooParticles");
		StopGooParticles();
		StopVomitParticles();
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		switch (new MegaEvent(spineEvent).GetData().GetEventName())
		{
		case "goo_start":
			StartGooParticles();
			break;
		case "goo_stop":
			StopGooParticles();
			break;
		case "vomit_start":
			StartVomitParticles();
			break;
		case "vomit_stop":
			StopVomitParticles();
			break;
		}
	}

	private void StartGooParticles()
	{
		_gooParticlesR.Emitting = true;
		_gooParticlesL.Emitting = true;
	}

	private void StopGooParticles()
	{
		_gooParticlesR.Emitting = false;
		_gooParticlesL.Emitting = false;
	}

	private void StopVomitParticles()
	{
		_gooParticlesVomit.Emitting = false;
	}

	private void StartVomitParticles()
	{
		_gooParticlesVomit.Emitting = true;
	}
}
