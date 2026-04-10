using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NSpectralKnightVfx : Node
{
	private GpuParticles2D _flameParticlesFlat;

	private GpuParticles2D _flameParticlesAdd;

	private GpuParticles2D _cinderParticles;

	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_animController.ConnectAnimationStarted(Callable.From<GodotObject, GodotObject, GodotObject>(OnAnimationStart));
		_flameParticlesAdd = _parent.GetNode<GpuParticles2D>("FlameParticlesAdd");
		_flameParticlesFlat = _parent.GetNode<GpuParticles2D>("FlameParticlesFlat");
		_cinderParticles = _parent.GetNode<GpuParticles2D>("CinderParticles");
		TurnOffFire();
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		string eventName = new MegaEvent(spineEvent).GetData().GetEventName();
		if (!(eventName == "flame_start"))
		{
			if (eventName == "flame_end")
			{
				TurnOffFire();
			}
		}
		else
		{
			TurnOnFire();
		}
	}

	private void OnAnimationStart(GodotObject spineSprite, GodotObject animationState, GodotObject trackEntry)
	{
		if (new MegaAnimationState(animationState).GetCurrent(0).GetAnimation().GetName() != "attack")
		{
			TurnOffFire();
		}
	}

	private void TurnOnFire()
	{
		_flameParticlesAdd.Restart();
		_flameParticlesFlat.Restart();
		_cinderParticles.Restart();
	}

	private void TurnOffFire()
	{
		_flameParticlesAdd.Emitting = false;
		_flameParticlesFlat.Emitting = false;
		_cinderParticles.Emitting = false;
	}
}
