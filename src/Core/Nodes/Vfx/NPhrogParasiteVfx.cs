using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NPhrogParasiteVfx : Node
{
	private GpuParticles2D _bubbleParticlesA;

	private GpuParticles2D _bubbleParticlesB;

	private GpuParticles2D _bubbleParticlesC;

	private GpuParticles2D _gooParticlesDeath;

	private GpuParticles2D _wormParticlesDeath;

	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_bubbleParticlesA = _parent.GetNode<GpuParticles2D>("BubbleABoneNode/WormParticlesA");
		_bubbleParticlesB = _parent.GetNode<GpuParticles2D>("BubbleBSlotNode/WormParticlesB");
		_bubbleParticlesC = _parent.GetNode<GpuParticles2D>("BubbleCBoneNode/WormParticlesC");
		_gooParticlesDeath = _parent.GetNode<GpuParticles2D>("DeathParticles");
		_wormParticlesDeath = _parent.GetNode<GpuParticles2D>("DeathWormParticles");
		_bubbleParticlesA.Emitting = false;
		_bubbleParticlesB.Emitting = false;
		_bubbleParticlesC.Emitting = false;
		_gooParticlesDeath.Emitting = false;
		_wormParticlesDeath.Emitting = false;
		_gooParticlesDeath.OneShot = true;
		_wormParticlesDeath.OneShot = true;
		_animController.GetAnimationState().SetAnimation("die");
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		switch (new MegaEvent(spineEvent).GetData().GetEventName())
		{
		case "infect":
			TurnOnInfect();
			break;
		case "stop_infect":
			TurnOffInfect();
			break;
		case "explode":
			StartExplode();
			break;
		}
	}

	private void TurnOnInfect()
	{
		_bubbleParticlesA.Emitting = true;
		_bubbleParticlesB.Emitting = true;
		_bubbleParticlesC.Emitting = true;
	}

	private void TurnOffInfect()
	{
		_bubbleParticlesA.Emitting = false;
		_bubbleParticlesB.Emitting = false;
		_bubbleParticlesC.Emitting = false;
	}

	private void StartExplode()
	{
		_gooParticlesDeath.Restart();
		_wormParticlesDeath.Restart();
	}
}
