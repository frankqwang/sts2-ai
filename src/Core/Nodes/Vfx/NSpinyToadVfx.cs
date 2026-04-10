using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NSpinyToadVfx : Node
{
	private GpuParticles2D _spineParticles;

	private GpuParticles2D _spineSubParticles;

	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_spineParticles = _parent.GetNode<GpuParticles2D>("BurstParticles");
		_spineSubParticles = _parent.GetNode<GpuParticles2D>("BurstParticlesSub");
		_spineParticles.Emitting = false;
		_spineParticles.OneShot = true;
		_spineSubParticles.Emitting = false;
		_spineSubParticles.OneShot = true;
		_animController.GetAnimationState().SetAnimation("explode");
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		string eventName = new MegaEvent(spineEvent).GetData().GetEventName();
		if (eventName == "explode")
		{
			StartExplosion();
		}
	}

	private void StartExplosion()
	{
		_spineParticles.Restart();
	}
}
