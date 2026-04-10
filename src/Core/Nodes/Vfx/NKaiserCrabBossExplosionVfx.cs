using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NKaiserCrabBossExplosionVfx : Node
{
	private MegaSprite _megaSprite;

	private Node2D _parent;

	private GpuParticles2D _explosionParticles;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_explosionParticles = _parent.GetNode<GpuParticles2D>("ExplosionSlot/ExplosionParticles");
		_megaSprite = new MegaSprite(GetParent<Node2D>());
		_megaSprite.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_explosionParticles.Emitting = false;
		_explosionParticles.OneShot = true;
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		string eventName = new MegaEvent(spineEvent).GetData().GetEventName();
		if (eventName == "left_embers_start")
		{
			OnLeftEmbersStart();
		}
	}

	private void OnLeftEmbersStart()
	{
		_explosionParticles.Restart();
	}
}
