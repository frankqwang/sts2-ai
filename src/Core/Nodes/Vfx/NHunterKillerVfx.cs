using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NHunterKillerVfx : Node
{
	private MegaSprite _megaSprite;

	private GpuParticles2D _particles;

	public override void _Ready()
	{
		_particles = GetNode<GpuParticles2D>("../MouthBone/Particles");
		_particles.Emitting = false;
		_megaSprite = new MegaSprite(GetParent<Node2D>());
		_megaSprite.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_megaSprite.ConnectAnimationStarted(Callable.From<GodotObject, GodotObject, GodotObject>(OnAnimationStart));
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		string eventName = new MegaEvent(spineEvent).GetData().GetEventName();
		if (!(eventName == "spit_start"))
		{
			if (eventName == "spit_end")
			{
				OnParticlesEnd();
			}
		}
		else
		{
			OnParticlesStart();
		}
	}

	private void OnAnimationStart(GodotObject spineSprite, GodotObject animationState, GodotObject trackEntry)
	{
		if (new MegaAnimationState(animationState).GetCurrent(0).GetAnimation().GetName() != "cast")
		{
			OnParticlesEnd();
		}
	}

	private void OnParticlesStart()
	{
		_particles.Emitting = true;
	}

	private void OnParticlesEnd()
	{
		_particles.Emitting = false;
	}
}
