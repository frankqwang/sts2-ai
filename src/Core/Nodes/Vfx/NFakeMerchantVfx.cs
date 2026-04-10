using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NFakeMerchantVfx : Node
{
	private MegaSprite _megaSprite;

	private GpuParticles2D _particles;

	public override void _Ready()
	{
		_particles = GetNode<GpuParticles2D>("../ParticlesSlot/Particles");
		_particles.Emitting = false;
		_megaSprite = new MegaSprite(GetParent<Node2D>());
		_megaSprite.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		string eventName = new MegaEvent(spineEvent).GetData().GetEventName();
		if (!(eventName == "particles_start"))
		{
			if (eventName == "particles_end")
			{
				OnParticlesEnd();
			}
		}
		else
		{
			OnParticlesStart();
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
