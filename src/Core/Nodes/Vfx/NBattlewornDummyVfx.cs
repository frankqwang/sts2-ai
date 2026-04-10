using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NBattlewornDummyVfx : Node
{
	private MegaSprite _megaSprite;

	private GpuParticles2D _damageParticles;

	public override void _Ready()
	{
		_megaSprite = new MegaSprite(GetParent<Node2D>());
		_megaSprite.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_damageParticles = GetParent<Node2D>().GetNode<GpuParticles2D>("ParticlesSlot/DamageParticles");
		_damageParticles.Emitting = false;
		_damageParticles.OneShot = true;
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		string eventName = new MegaEvent(spineEvent).GetData().GetEventName();
		if (eventName == "fire_chips")
		{
			OnDamageChips();
		}
	}

	private void OnDamageChips()
	{
		_damageParticles.Restart();
	}
}
