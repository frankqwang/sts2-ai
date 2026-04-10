using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NGasBombVfx : Node
{
	private MegaSprite _megaSprite;

	private GpuParticles2D _explodePuffParticles;

	private GpuParticles2D _puffParticles;

	private GpuParticles2D _dotParticles;

	private GpuParticles2D _bitParticles;

	public override void _Ready()
	{
		_explodePuffParticles = GetNode<GpuParticles2D>("../SmokeBallSlot/ExplodePuffParticles");
		_puffParticles = GetNode<GpuParticles2D>("../SmokeBallSlot/PuffParticles");
		_dotParticles = GetNode<GpuParticles2D>("../SmokeBallSlot/DotParticles");
		_bitParticles = GetNode<GpuParticles2D>("../SmokeBallSlot/BitParticles");
		_megaSprite = new MegaSprite(GetParent<Node2D>());
		_megaSprite.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_explodePuffParticles.Emitting = false;
		_explodePuffParticles.OneShot = true;
		_bitParticles.Emitting = false;
		_bitParticles.OneShot = true;
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		switch (new MegaEvent(spineEvent).GetData().GetEventName())
		{
		case "burst":
			OnBurst();
			break;
		case "idle_particles":
			OnIdleParticles();
			break;
		case "dissipate":
			OnDissipate();
			break;
		}
	}

	private void OnBurst()
	{
		_dotParticles.Emitting = false;
		_puffParticles.Emitting = false;
		_puffParticles.SetVisible(visible: false);
		_explodePuffParticles.Restart();
		_bitParticles.Restart();
	}

	private void OnIdleParticles()
	{
		_dotParticles.Emitting = true;
		_puffParticles.Emitting = true;
		_puffParticles.SetVisible(visible: true);
	}

	private void OnDissipate()
	{
		_dotParticles.Emitting = false;
		_puffParticles.Emitting = false;
	}
}
