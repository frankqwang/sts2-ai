using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NMechaKnightVfx : Node
{
	private GpuParticles2D _flameThrowerParticlesDark;

	private GpuParticles2D _flameThrowerParticlesLight;

	private GpuParticles2D _cinderParticles;

	private GpuParticles2D _glowParticles;

	private GpuParticles2D _engineParticles;

	private GpuParticles2D _engineParticlesDark;

	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_animController.ConnectAnimationStarted(Callable.From<GodotObject, GodotObject, GodotObject>(OnAnimationStart));
		_flameThrowerParticlesDark = _parent.GetNode<GpuParticles2D>("FlameParticlesBone/FlameParticlesDark");
		_flameThrowerParticlesLight = _parent.GetNode<GpuParticles2D>("FlameParticlesBone/FlameParticlesLight");
		_cinderParticles = _parent.GetNode<GpuParticles2D>("FlameParticlesBone/CinderParticles");
		_glowParticles = _parent.GetNode<GpuParticles2D>("FlameParticlesBone/GlowParticles");
		_engineParticles = _parent.GetNode<GpuParticles2D>("EngineSlot/EngineBone/EngineParticles");
		_engineParticlesDark = _parent.GetNode<GpuParticles2D>("EngineSlot/EngineBone/EngineParticlesDark");
		TurnOffFlameThrower();
		TurnOffEngine();
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		switch (new MegaEvent(spineEvent).GetData().GetEventName())
		{
		case "flame_start":
			TurnOnFlameThrower();
			break;
		case "flame_end":
			TurnOffFlameThrower();
			break;
		case "engine_start":
			TurnOnEngine();
			break;
		case "engine_stop":
			TurnOffEngine();
			break;
		}
	}

	private void OnAnimationStart(GodotObject spineSprite, GodotObject animationState, GodotObject trackEntry)
	{
		if (new MegaAnimationState(animationState).GetCurrent(0).GetAnimation().GetName() != "attack")
		{
			TurnOffFlameThrower();
		}
		bool flag = new MegaAnimationState(animationState).GetCurrent(0).GetAnimation().GetName() == "hurt";
		_flameThrowerParticlesDark.Visible = !flag;
		_flameThrowerParticlesLight.Visible = !flag;
		_cinderParticles.Visible = !flag;
		_glowParticles.Visible = !flag;
	}

	private void TurnOnFlameThrower()
	{
		_flameThrowerParticlesDark.Restart();
		_flameThrowerParticlesLight.Restart();
		_cinderParticles.Restart();
		_glowParticles.Restart();
	}

	private void TurnOffFlameThrower()
	{
		_flameThrowerParticlesDark.Emitting = false;
		_flameThrowerParticlesLight.Emitting = false;
		_cinderParticles.Emitting = false;
		_glowParticles.Emitting = false;
	}

	private void TurnOnEngine()
	{
		_engineParticles.Restart();
		_engineParticlesDark.Restart();
	}

	private void TurnOffEngine()
	{
		_engineParticles.Emitting = false;
		_engineParticlesDark.Emitting = false;
	}
}
