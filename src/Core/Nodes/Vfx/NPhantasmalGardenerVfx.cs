using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NPhantasmalGardenerVfx : Node
{
	private MegaSprite _megaSprite;

	private GpuParticles2D _spewParticles;

	public override void _Ready()
	{
		_spewParticles = GetNode<GpuParticles2D>("../SpewSlotNode/SpewParticles");
		_spewParticles.Emitting = false;
		_megaSprite = new MegaSprite(GetParent<Node2D>());
		_megaSprite.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		string eventName = new MegaEvent(spineEvent).GetData().GetEventName();
		if (!(eventName == "spew_start"))
		{
			if (eventName == "spew_end")
			{
				OnSpewEnd();
			}
		}
		else
		{
			OnSpewStart();
		}
	}

	private void OnSpewStart()
	{
		_spewParticles.Emitting = true;
	}

	private void OnSpewEnd()
	{
		_spewParticles.Emitting = false;
	}
}
