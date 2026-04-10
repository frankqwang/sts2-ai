using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NKinPriestVfx : Node
{
	private GpuParticles2D _sparkParticles;

	private NKinPriestBeamVfx _beamVfx;

	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_sparkParticles = _parent.GetNode<GpuParticles2D>("TorchFireBone/SparkParticles");
		_sparkParticles.Emitting = false;
		_beamVfx = _parent.GetNode<NKinPriestBeamVfx>("Beam");
		_animController.GetAnimationState().SetAnimation("attack_laser");
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		switch (new MegaEvent(spineEvent).GetData().GetEventName())
		{
		case "sparks_start":
			StartSparks();
			break;
		case "sparks_end":
			EndSparks();
			break;
		case "laser_fire":
			FireLaser();
			break;
		}
	}

	private void StartSparks()
	{
		_sparkParticles.Emitting = true;
	}

	private void EndSparks()
	{
		_sparkParticles.Emitting = false;
	}

	private void FireLaser()
	{
		_beamVfx.Fire();
	}
}
